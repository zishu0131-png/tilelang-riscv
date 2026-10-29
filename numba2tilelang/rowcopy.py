# -*- coding: utf-8 -*-
"""G2-B 窄通道 `ROWCOPY`（v2：绑定身份/调用来源/ABI/卫生修复）。

严格形态（按绑定身份，且名字互异、不与任何形参冲突）：

    idx = cuda.grid(1)                    # 来自 numba.cuda 的 grid，无 keywords，维度必须是 int 1
    if idx < S:                           # S、L 必须是 int64 标量形参
        t = index[idx]                    # 一次 index 读取（恒等下标）
        for j in range(L):                # 内建 range，无 keywords，单参
            out[idx * L + j] = src[t * L + j]

拒绝：绑定名重名/遮蔽形参、cuda/range 来源不可证（模块或局部遮蔽、自定义 range）、
调用带 keywords、grid 维度是 bool/非 int、签名 rank 非一维、未使用/多余形参、
保留名（T/tilelang/block/range/cuda/_tl_*）冲突。
"""
import ast
import textwrap

from semantic import Unsupported

_ROWCOPY_TENSOR_DTYPES = {'float32', 'int32'}
_RESERVED_NAMES = {'T', 'tilelang', 'block', 'range', 'cuda',
                    'type', 'int', 'float', 'bool', 'ValueError', 'Exception',
                    'print', 'len'}
_RESERVED_PREFIX = '_tl_'


class RowCopy:
    def __init__(self, name, params, tensor_types, scalar_types, src, index, out,
                 s_name, l_name, idx_name, row_name, loop_name):
        self.name = name
        self.params = list(params)
        self.tensor_types = dict(tensor_types)
        self.scalar_types = dict(scalar_types)
        self.src = src
        self.index = index
        self.out = out
        self.s_name = s_name
        self.l_name = l_name
        self.idx_name = idx_name
        self.row_name = row_name
        self.loop_name = loop_name


def _fail(message):
    raise Unsupported('ROWCOPY: ' + message)


def _is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _call_target(node):
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id + '.' + func.attr
    return None


_ALLOWED_PLAIN_IMPORTS = {'math', 'numpy'}


def _parse_module(src):
    """封闭模块白名单：docstring + 受认可的绝对 import/from-import + 唯一目标 kernel。"""
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError as exc:
        raise Unsupported('Python 语法错误: ' + str(exc)) from exc
    from source_contract import is_cuda_jit, is_device_jit
    kernel = None
    cuda_imports = 0
    for index, stmt in enumerate(tree.body):
        if (index == 0 and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)):
            continue  # 模块 docstring
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name not in _ALLOWED_PLAIN_IMPORTS:
                    _fail('模块顶层只允许受认可的绝对 import（math/numpy）: ' + alias.name)
                bound = alias.asname or alias.name.split('.')[0]
                if bound in _RESERVED_NAMES or bound.startswith(_RESERVED_PREFIX):
                    _fail('导入别名与生成器保留名冲突: ' + bound)
            continue
        if isinstance(stmt, ast.ImportFrom):
            if stmt.level != 0:
                _fail('只允许绝对 import（不允许相对导入）')
            if stmt.module != 'numba':
                _fail('from-import 只允许 `from numba import cuda`（其它模块不在封闭白名单）')
            if len(stmt.names) != 1 or stmt.names[0].name == '*':
                _fail('from numba 只允许恰好导入 cuda（不允许星号）')
            alias = stmt.names[0]
            if alias.name != 'cuda':
                _fail('from numba 只允许导入 cuda')
            if alias.asname not in (None, 'cuda'):
                _fail('cuda 的导入别名必须是 cuda')
            cuda_imports += 1
            continue
        if isinstance(stmt, ast.FunctionDef):
            if not any(is_cuda_jit(d) and not is_device_jit(d) for d in stmt.decorator_list):
                _fail('模块顶层只允许唯一目标 kernel（不允许 helper/其它函数）')
            if kernel is not None:
                _fail('模块顶层只允许唯一目标 kernel')
            kernel = stmt
            continue
        _fail('模块顶层只允许 docstring、受认可的绝对 import 与唯一目标 kernel'
              '（赋值/表达式/复合语句/类/其它函数一律拒绝）')
    if kernel is None:
        _fail('需要一个 @cuda.jit 目标 kernel')
    if cuda_imports != 1:
        _fail('cuda 必须恰有一次绝对 `from numba import cuda`')
    return tree, kernel


def _declared_types(tree, fn):
    from source_contract import normalize_decorator
    return normalize_decorator(tree, fn)


def _signature_text(fn):
    """若装饰器带静态签名，返回其字符串；否则 None。"""
    decorator = fn.decorator_list[0]
    if isinstance(decorator, ast.Call) and decorator.args:
        arg = decorator.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def _check_signature_rank(signature, params, tensor_names):
    """签名 rank 与一维用法一致性：张量项必须恰是 `dtype[:]`，且数量与张量形参一致。"""
    if signature is None:
        return
    if not (signature.startswith('void(') and signature.endswith(')')):
        _fail('仅支持 "void(...)" 形式的签名')
    parts = [item.strip() for item in signature[len('void('):-1].split(',')]
    if len(parts) != len(params):
        _fail('签名参数数量与函数形参不一致')
    tensor_entries = [item for item in parts if '[' in item]
    for item in tensor_entries:
        if item.count(':') != 1 or item.count('[') != 1 or item.count(']') != 1:
            _fail('签名里的张量必须是一维（只允许 dtype[:]）: ' + item)
    if len(tensor_entries) != len(tensor_names):
        _fail('签名里的张量数量与源码张量形参数量不一致')


def _module_scope_bindings(tree):
    """模块作用域绑定事件：递归进入 if/for/while/with/try 的模块级复合语句，
    但不进入函数/类/lambda 的函数体；函数/类名本身记为绑定。"""
    bound = {}

    def add(name):
        bound[name] = bound.get(name, 0) + 1

    def scan_statements(statements):
        for stmt in statements:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                add(stmt.name)
                continue
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    add(alias.asname or alias.name.split('.')[0])
                continue
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    add('*' if alias.name == '*' else (alias.asname or alias.name))
                continue
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    for node in ast.walk(target):
                        if isinstance(node, ast.Name):
                            add(node.id)
                continue
            if isinstance(stmt, ast.AnnAssign):
                for node in ast.walk(stmt.target):
                    if isinstance(node, ast.Name):
                        add(node.id)
                continue
            if isinstance(stmt, ast.AugAssign):
                for node in ast.walk(stmt.target):
                    if isinstance(node, ast.Name):
                        add(node.id)
                continue
            if isinstance(stmt, ast.Delete):
                for target in stmt.targets:
                    for node in ast.walk(target):
                        if isinstance(node, ast.Name):
                            add('del:' + node.id)
                continue
            if isinstance(stmt, ast.If):
                scan_statements(stmt.body)
                scan_statements(stmt.orelse)
                continue
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                for node in ast.walk(stmt.target):
                    if isinstance(node, ast.Name):
                        add(node.id)
                scan_statements(stmt.body)
                scan_statements(stmt.orelse)
                continue
            if isinstance(stmt, ast.While):
                scan_statements(stmt.body)
                scan_statements(stmt.orelse)
                continue
            if isinstance(stmt, ast.With):
                for item in stmt.items:
                    if item.optional_vars is not None:
                        for node in ast.walk(item.optional_vars):
                            if isinstance(node, ast.Name):
                                add(node.id)
                scan_statements(stmt.body)
                continue
            if isinstance(stmt, ast.Try):
                scan_statements(stmt.body)
                for handler in stmt.handlers:
                    if handler.name:
                        add(handler.name)
                    scan_statements(handler.body)
                scan_statements(stmt.orelse)
                scan_statements(stmt.finalbody)
                continue

    scan_statements(tree.body)
    return bound


def _check_provenance(tree, fn):
    """封闭白名单已固定 cuda 来源；此处只核函数局部不绑定 cuda/range。"""
    params = [arg.arg for arg in fn.args.args]
    local_bound = {node.id for node in ast.walk(fn)
                   if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    if 'cuda' in local_bound or 'cuda' in params:
        _fail('cuda 不能在函数内被绑定或作为形参')
    if 'range' in local_bound or 'range' in params:
        _fail('range 不能被子句/形参绑定')
    return params


def _check_signature_duplicates(params):
    if len(set(params)) != len(params):
        _fail('函数形参不能重名')


def _validate_metadata(tensor_types, scalar_types, declared_tensors, declared_scalars,
                       all_params, tensor_names):
    if type(tensor_types) is not dict or not tensor_types:
        raise ValueError('tensor_types 必须是非空 dict（ROWCOPY 通道要求显式张量类型）')
    for key, value in tensor_types.items():
        if type(key) is not str or type(value) is not str:
            raise ValueError('tensor_types keys and values must be strings')
        if key not in all_params:
            raise ValueError('tensor_types 里有未知形参: ' + key)
        if key in scalar_types:
            raise ValueError('tensor_types 与 scalar_types 不能声明同一个名字: ' + key)
        if value not in _ROWCOPY_TENSOR_DTYPES:
            raise ValueError("tensor_types 本批只支持 'float32'/'int32'（%s=%s）" % (key, value))
        if key in declared_tensors and declared_tensors[key] != value:
            raise ValueError('tensor_types 与静态签名的张量类型冲突: %s（%s vs %s）'
                             % (key, declared_tensors[key], value))
        if key in declared_scalars:
            raise ValueError('tensor_types 不能声明标量形参: ' + key)
    declared_names = set(tensor_types)
    if declared_names != set(tensor_names):
        missing = sorted(set(tensor_names) - declared_names)
        extra = sorted(declared_names - set(tensor_names))
        if missing:
            raise ValueError('ROWCOPY 要求所有张量形参都显式声明 tensor_types，缺少: '
                             + ', '.join(missing))
        raise ValueError('tensor_types 声明了非张量形参: ' + ', '.join(extra))
    for key, value in scalar_types.items():
        if key not in all_params:
            raise ValueError('scalar_types 里有未知形参: ' + key)
        if key in declared_names:
            raise ValueError('scalar_types 不能声明张量形参: ' + key)
        if value != 'int64':
            raise ValueError("scalar_types 目前只支持 'int64'（%s=%s）" % (key, value))
        if key in declared_scalars and declared_scalars[key] != value:
            raise ValueError('scalar_types 与静态签名的标量类型冲突: %s（%s vs %s）'
                             % (key, declared_scalars[key], value))


def capture(src, tensor_types, scalar_types=None):
    tree, fn = _parse_module(src)
    if len(fn.decorator_list) != 1:
        _fail('不支持额外装饰器或返回类型声明')
    if fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.defaults:
        _fail('只支持无默认值的普通形参')
    params = [arg.arg for arg in fn.args.args]
    _check_signature_duplicates(params)
    declared_scalars, declared_tensors = _declared_types(tree, fn)
    scalar_types = dict(scalar_types or {})
    # 元数据错误优先以 ValueError 报出（不依赖形态识别是否走到角色检查）。
    from source_contract import tensor_used_names
    used_tensors = tensor_used_names(fn)
    _validate_metadata(tensor_types, scalar_types, declared_tensors, declared_scalars,
                       params, used_tensors)

    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body.pop(0)
    _check_provenance(tree, fn)
    if len(body) != 2:
        _fail('函数体必须是 cuda.grid 赋值 + 单个 guard 块')
    grid, guard = body

    if not (isinstance(grid, ast.Assign) and len(grid.targets) == 1
            and isinstance(grid.targets[0], ast.Name) and isinstance(grid.value, ast.Call)):
        _fail('首条语句必须是 idx = cuda.grid(1)')
    idx_name = grid.targets[0].id
    grid_call = grid.value
    if _call_target(grid_call) != 'cuda.grid' or grid_call.keywords:
        _fail('grid 必须是无 keywords 的 cuda.grid(...) 调用')
    if len(grid_call.args) != 1:
        _fail('cuda.grid 必须恰有一个位置参数')
    dimension = grid_call.args[0]
    if not (isinstance(dimension, ast.Constant) and type(dimension.value) is int
            and dimension.value == 1):
        _fail('cuda.grid 维度必须是 int 常量 1（bool/True 不算）')

    if not isinstance(guard, ast.If) or guard.orelse:
        _fail('第二条语句必须是 if idx < S:（不允许 else）')
    condition = guard.test
    if not (isinstance(condition, ast.Compare) and len(condition.ops) == 1
            and isinstance(condition.ops[0], ast.Lt) and _is_name(condition.left, idx_name)
            and isinstance(condition.comparators[0], ast.Name)):
        _fail('guard 必须是 idx < S')
    s_name = condition.comparators[0].id
    if s_name not in scalar_types or scalar_types[s_name] != 'int64':
        _fail('guard 上界必须是 scalar_types 中声明为 int64 的标量')

    if len(guard.body) != 2:
        _fail('guard 内必须恰为“一次 index 读取绑定 + 一个循环”')
    binding, loop = guard.body
    if not (isinstance(binding, ast.Assign) and len(binding.targets) == 1
            and isinstance(binding.targets[0], ast.Name)
            and isinstance(binding.value, ast.Subscript)):
        _fail('guard 内第一条必须是 t = index[idx]')
    row_name = binding.targets[0].id
    index_sub = binding.value
    if not (isinstance(index_sub.value, ast.Name) and _is_name(index_sub.slice, idx_name)):
        _fail('index 必须以恒等下标 idx 读取')
    index_name = index_sub.value.id

    if not (isinstance(loop, ast.For) and isinstance(loop.target, ast.Name) and not loop.orelse):
        _fail('guard 内第二条必须是 for j in range(L):（不允许 else）')
    loop_name = loop.target.id
    if not (isinstance(loop.iter, ast.Call) and _call_target(loop.iter) == 'range'
            and not loop.iter.keywords and len(loop.iter.args) == 1
            and isinstance(loop.iter.args[0], ast.Name)):
        _fail('循环必须是无 keywords 的 range(L) 单参形式')
    l_name = loop.iter.args[0].id
    if l_name not in scalar_types or scalar_types[l_name] != 'int64':
        _fail('range 上界必须是 scalar_types 中声明为 int64 的标量')
    if len(loop.body) != 1 or not isinstance(loop.body[0], ast.Assign) \
            or len(loop.body[0].targets) != 1:
        _fail('循环体必须恰好一条张量写回')
    store = loop.body[0]
    if not (isinstance(store.targets[0], ast.Subscript)
            and isinstance(store.targets[0].value, ast.Name)):
        _fail('写回目标必须是张量下标')
    out_name = store.targets[0].value.id
    if not _add_pattern(store.targets[0].slice, idx_name, l_name, loop_name):
        _fail('写回下标必须精确是 out[idx * L + j]')
    if not (isinstance(store.value, ast.Subscript) and isinstance(store.value.value, ast.Name)):
        _fail('读端必须是单个张量下标')
    src_name = store.value.value.id
    if not _add_pattern(store.value.slice, row_name, l_name, loop_name):
        _fail('读下标必须精确是 src[t * L + j]')

    # 绑定身份：idx/t/j 互异、且不与任何形参冲突；S/L 与张量角色严格分离
    if len({idx_name, row_name, loop_name}) != 3:
        _fail('idx/t/j 三个绑定名必须互异')
    for name in (idx_name, row_name, loop_name):
        if name in params:
            _fail('grid/index/循环绑定名不能与形参同名: ' + name)
    if len({src_name, index_name, out_name}) != 3:
        _fail('src/index/out 必须是三个不同的张量形参')
    tensor_names = [src_name, index_name, out_name]
    for name in tensor_names:
        if name not in params:
            _fail('张量名必须是函数形参: ' + name)
    if s_name == l_name:
        _fail('S 与 L 不能是同一标量')
    if {s_name, l_name} & set(tensor_names):
        _fail('S/L 不能与张量角色同名')
    if set(params) != set(tensor_names) | {s_name, l_name}:
        extra = sorted(set(params) - set(tensor_names) - {s_name, l_name})
        if extra:
            _fail('存在未使用/多余形参（本通道要求形参恰为 3 张量 + S/L）: ' + ', '.join(extra))
    if set(scalar_types) != {s_name, l_name}:
        raise ValueError('scalar_types 必须恰好声明 guard 与 range 的两个上界（当前: %s）'
                         % ', '.join(sorted(scalar_types)))
    if tensor_types.get(index_name) != 'int32':
        _fail('本批 index 必须是 int32')
    if tensor_types.get(src_name) != 'float32' or tensor_types.get(out_name) != 'float32':
        _fail('本批 src/out 必须是 float32')

    _check_signature_rank(_signature_text(fn), params, tensor_names)

    # 保留名/内部前缀：统一盘点生成器实际引用的未限定名（含函数名本身）
    for name in set(params) | {idx_name, row_name, loop_name} | {fn.name}:
        if name in _RESERVED_NAMES or name.startswith(_RESERVED_PREFIX):
            _fail('名字与生成器保留名冲突: ' + name)

    # 不允许其它绑定（set 只作辅助；唯一性/身份已在结构上逐项断言）
    allowed = {idx_name, row_name, loop_name}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) \
                and node.id not in allowed:
            _fail('不允许额外绑定: ' + node.id)
    return RowCopy(fn.name, params, dict(tensor_types), dict(scalar_types), src_name,
                   index_name, out_name, s_name, l_name, idx_name, row_name, loop_name)


def _add_pattern(node, left, l, loop):
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
            and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Mult)
            and _is_name(node.left.left, left) and _is_name(node.left.right, l)
            and _is_name(node.right, loop))


def emit(rowcopy):
    """按**源码形参顺序**发射真正的 @tilelang.jit(out_idx=[]) factory；block 严格 128。"""
    params = rowcopy.params
    dynamic = {name: '_tl_dyn_%d' % index for index, name in enumerate(params)
               if rowcopy.tensor_types.get(name) in ('float32', 'int32')}
    imports = ['import tilelang', 'import tilelang.language as T']
    lines = [
        '# ROWCOPY（G2-B 窄通道）：完整 selection_gather_kernel 形态；caller-output。',
        '# 调用方契约（运行期，不校验）：out.numel() == S*L；population.numel() == pop_size*L；',
        '# len(index) >= S；0 <= t < pop_size；三个 buffer 连续且互不别名；S>=1、L>=1。',
        '# 动态长度由适配器从 tensor shape 提取（int32 实参）；S/L 为运行期 int64 标量；block 固定 128。',
    ] + imports + [
        '',
        '@tilelang.jit(out_idx=[], pass_configs={"tl.disable_warp_specialized": True},',
        '              compile_flags=["--fmad=false"])',
        'def %s(%s, block=128):' % (rowcopy.name, ', '.join(params)),
        '    if type(block) is not int or block != 128:',
        '        raise ValueError("block must be exactly 128 in this batch")',
    ]
    for name, var in dynamic.items():
        lines.append('    %s = T.dynamic("%s_len")' % (var, name))
    lines.append('    @T.prim_func')
    lines.append('    def _tl_rowcopy(%s):' % ', '.join(
        '%s: T.Tensor((%s,), "%s")' % (name, dynamic[name], rowcopy.tensor_types[name])
        if name in dynamic else '%s: T.int64' % name for name in params))
    lines += [
        '        with T.Kernel(T.ceildiv(%s, 128), threads=128) as _tl_bx:' % rowcopy.s_name,
        '            for _tl_lane in T.Parallel(128):',
        '                _tl_p = T.cast(_tl_bx, "int64") * T.cast(128, "int64") + T.cast(_tl_lane, "int64")',
        '                if _tl_p < %s:' % rowcopy.s_name,
        '                    _tl_t = T.cast(%s[_tl_p], "int64")' % rowcopy.index,
        '                    for %s in T.serial(%s):' % (rowcopy.loop_name, rowcopy.l_name),
        '                        %s[_tl_p * %s + %s] = %s[_tl_t * %s + %s]' % (
            rowcopy.out, rowcopy.l_name, rowcopy.loop_name,
            rowcopy.src, rowcopy.l_name, rowcopy.loop_name),
        '    return _tl_rowcopy',
        '',
    ]
    return '\n'.join(lines)


def convert_rowcopy(src, tensor_types, scalar_types=None):
    return emit(capture(src, tensor_types, scalar_types))
