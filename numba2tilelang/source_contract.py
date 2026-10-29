# -*- coding: utf-8 -*-
"""显式签名与线程坐标规范化：只接受可证明符合当前契约的源码。"""
import ast
from semantic import Unsupported
from thread_ir import DTYPES, DTYPE_ALIASES

# cuda.jit 选项分两类：改变语义的必须显式关闭，只影响编译产物/诊断/调度的可以忽略。
# - MUST_BE_FALSE：fastmath 改变浮点语义，device 会把 kernel 变成 device 函数，
#   debug/boundscheck 插入诊断与越界检查（本链路对"完整覆盖输出域"的假设不能靠它）。
# - NO_EFFECT_OPTIONS：lineinfo/cache 只涉及诊断与编译缓存，opt/launch_bounds
#   只影响优化级别与线程调度，都不改变某个输入的计算结果。
MUST_BE_FALSE = ('device', 'fastmath', 'debug', 'boundscheck')
NO_EFFECT_OPTIONS = ('lineinfo', 'cache', 'opt', 'launch_bounds')


def is_device_jit(decorator):
    """`@cuda.jit(device=True)` 修饰的是 helper，不是入口 kernel。"""
    return isinstance(decorator, ast.Call) and any(
        keyword.arg == 'device' and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True for keyword in decorator.keywords)


def is_cuda_jit(node):
    called = node.func if isinstance(node, ast.Call) else node
    return isinstance(called, ast.Attribute) and called.attr == 'jit' \
        and isinstance(called.value, ast.Name) and called.value.id == 'cuda'


def tensor_ranks(fn, params):
    ranks = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in params:
            rank = len(node.slice.elts) if isinstance(node.slice, ast.Tuple) else 1
            if ranks.setdefault(node.value.id, rank) != rank:
                raise Unsupported('同一张量的访问rank不一致: ' + node.value.id)
    return ranks


def tensor_used_names(fn):
    """以**张量方式**使用的名字集合：`name[...]`、`len(name)`、`name.shape/size/ndim`。

    注意：这只说明"看起来像张量"，**不能**反过来当作"未被这样使用的名字一定是标量"的证明确认；
    调用方显式声明类型时用它做冲突检查。
    """
    used = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            used.add(node.value.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name)                 and node.func.id == 'len' and node.args                 and isinstance(node.args[0], ast.Name):
            used.add(node.args[0].id)
        elif isinstance(node, ast.Attribute) and node.attr in ('shape', 'size', 'ndim')                 and isinstance(node.value, ast.Name):
            used.add(node.value.id)
    return used


def infer_scalars(fn, params):
    """无静态签名时推断标量形参。

    可证明的部分：函数体内从未以 `name[...]`、`len(name)`、`name.shape`、`name.size`
    出现的形参**不可能是张量**。不可证明的是 dtype——源码没写，numba 从调用点推断，
    这里一律按 Python float 的默认 `float64`，并由 `spec.inferred_scalars` 记录，
    生成物必须把这层推断写进头注释。

    从未出现过的形参**不**被推断（那是未使用参数，仍应拒绝）。
    """
    tensor_used, loaded = set(), set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            tensor_used.add(node.value.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == 'len' and node.args \
                and isinstance(node.args[0], ast.Name):
            tensor_used.add(node.args[0].id)
        elif isinstance(node, ast.Attribute) and node.attr in ('shape', 'size', 'ndim') \
                and isinstance(node.value, ast.Name):
            tensor_used.add(node.value.id)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
    return [name for name in params if name in loaded and name not in tensor_used]


def integer_scalars(fn, params):
    """无签名时，出现在位运算操作数里的标量形参只能是整数。

    位运算在 Python/numba 里对 float 直接报错，因此这个推断没有歧义；
    典型来源是噪声/RNG 的 seed（`(x * 374761393 + seed) & 0xFFFFFFFF`）。
    """
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
            for operand in (node.left, node.right):
                if isinstance(operand, ast.Name) and operand.id in params:
                    names.add(operand.id)
        # `range(n)` 的参数只能是整数，同样没有歧义。
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name)                 and node.func.id == 'range':
            for argument in node.args:
                if isinstance(argument, ast.Name) and argument.id in params:
                    names.add(argument.id)
    # 与 grid 坐标出现在同一个比较式里的标量按整数处理（`if tid < n_bodies:`）。
    # 线程坐标是整数，这类标量在源码里就是域上界；当成 float64 会让 guard 无法发射。
    grid_names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Attribute) \
                and node.value.func.attr == 'grid':
            for target in node.targets:
                for item in (target.elts if isinstance(target, ast.Tuple) else [target]):
                    if isinstance(item, ast.Name):
                        grid_names.add(item.id)
    if grid_names:
        for node in ast.walk(fn):
            if isinstance(node, ast.Compare) \
                    and any(isinstance(item, ast.Name) and item.id in grid_names
                            for item in ast.walk(node)):
                for item in ast.walk(node):
                    if isinstance(item, ast.Name) and item.id in params:
                        names.add(item.id)
    return names


def normalize_decorator(tree, fn):
    """校验并消去签名装饰器；返回 (标量形参 -> dtype, 张量形参 -> dtype)。"""
    scalars, tensor_dtypes = {}, {}
    # `-> None` 是无语义的返回注解（numba 里等价于无返回），允许。
    returns_none = (fn.returns is None
                    or (isinstance(fn.returns, ast.Constant) and fn.returns.value is None))
    if len(fn.decorator_list) != 1 or not returns_none:
        raise Unsupported('不支持额外装饰器或返回类型声明')
    decorator = fn.decorator_list[0]
    if not isinstance(decorator, ast.Call):
        return scalars, tensor_dtypes
    if len(decorator.args) > 1:
        raise Unsupported('cuda.jit只支持一个静态签名')
    seen = set()
    for keyword in decorator.keywords:
        if keyword.arg in seen:
            raise Unsupported('装饰器选项重复: ' + str(keyword.arg))
        seen.add(keyword.arg)
        if keyword.arg in NO_EFFECT_OPTIONS:
            # 只影响编译产物/诊断/调度的选项，不改变计算语义。取值不参与判定：
            # 这些选项不会让 kernel 的计算结果因取值而不同。
            continue
        if keyword.arg == 'fastmath' and isinstance(keyword.value, ast.Constant) \
                and keyword.value.value is True:
            # 源码允许放宽浮点精度（重排/近似）；生成物用严格语义，结果只会更准。
            # 这是显式选择：接受它的写法，但不复制 fastmath 的近似行为。
            continue
        if keyword.arg not in MUST_BE_FALSE or not isinstance(keyword.value, ast.Constant) \
                or keyword.value.value is not False:
            raise Unsupported('装饰器选项仅支持显式device/fastmath/debug/boundscheck=False，'
                              '以及无语义影响的lineinfo/cache/opt/launch_bounds')
    if decorator.args:
        signature = decorator.args[0]
        string_signature = isinstance(signature, ast.Constant) and isinstance(signature.value, str)
        if string_signature:
            try:
                signature = ast.parse(signature.value, mode='eval').body
            except SyntaxError as exc:
                raise Unsupported('无法解析静态签名') from exc
        else:
            # 不eval签名；类型标识符必须真实绑定到numba对应类型，任何覆盖都拒绝。
            needed = {node.id for node in ast.walk(signature) if isinstance(node, ast.Name)}
            bindings = {name: [] for name in needed}
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        name = alias.asname or alias.name.split('.')[0]
                        if name in bindings:
                            # `numba.types` 是类型名的规范出处（float16 只在那里导出，
                            # 顶层 `from numba import float16` 根本导入不到）。
                            # 20261025-149：`numba.cuda` 也再导出同一组标量类型与 `void`
                            # （`from numba.cuda import float32, int32, void`），按等价接受；
                            # 名字本身仍必须在 IR dtype 白名单内，别名覆盖规则不变。
                            valid = isinstance(node, ast.ImportFrom) and node.level == 0 \
                                and node.module in ('numba', 'numba.types', 'numba.cuda') \
                                and alias.name == name \
                                and (name == 'void' or DTYPE_ALIASES.get(name, name) in DTYPES)
                            bindings[name].append(valid)
            if any(not values or not all(values) for values in bindings.values()):
                raise Unsupported('签名类型名必须有明确的 numba 导入，且不能被别名覆盖')
        if isinstance(signature, ast.Call) and isinstance(signature.func, ast.Name) \
                and signature.func.id == 'void' and not signature.keywords:
            arguments = signature.args
        elif isinstance(signature, ast.Tuple):
            arguments = signature.elts
        else:
            raise Unsupported('签名只支持 void(...) 或参数类型元组，类型名须在 IR dtype 白名单内')
        params = [arg.arg for arg in fn.args.args]
        if len(arguments) != len(params):
            raise Unsupported('签名参数数量与kernel形参不一致')
        ranks = tensor_ranks(fn, params)
        scalars = {}
        for name, argument in zip(params, arguments):
            if isinstance(argument, ast.Name):
                # 标量形参（如 `n`、`alpha`）：类型名须在白名单内，且不参与 shape/rank 推断。
                dtype = DTYPE_ALIASES.get(argument.id, argument.id)
                if dtype not in DTYPES:
                    raise Unsupported('签名标量类型必须是 IR 白名单 dtype')
                # 归一后再写进标量表：生成模板按 `T.<dtype>` 发射，不能带 numba 别名（如 double）。
                scalars[name] = dtype
                continue
            if not isinstance(argument, ast.Subscript) or not isinstance(argument.value, ast.Name) \
                    or DTYPE_ALIASES.get(argument.value.id, argument.value.id) not in DTYPES:
                raise Unsupported('签名张量 dtype 必须是 IR 白名单类型之一: ' + '/'.join(DTYPES))
            tensor_dtypes[name] = DTYPE_ALIASES.get(argument.value.id, argument.value.id)
            dimensions = argument.slice.elts if isinstance(argument.slice, ast.Tuple) else [argument.slice]
            if len(dimensions) != ranks.get(name):
                raise Unsupported('签名rank与张量直接访问rank不一致: ' + name)
            for index, dimension in enumerate(dimensions):
                if not isinstance(dimension, ast.Slice) or dimension.lower is not None \
                        or dimension.upper is not None:
                    raise Unsupported('签名维度只支持 : 或末维 ::1')
                if dimension.step is not None and (index != len(dimensions)-1 \
                        or not isinstance(dimension.step, ast.Constant) \
                        or type(dimension.step.value) is not int or dimension.step.value != 1):
                    raise Unsupported('签名布局只支持任意布局或C连续；运行输入仍要求C连续')
    # 验证后才消去装饰器形式；不能让未实现的导入类型进入计算表达式。
    fn.decorator_list = [decorator.func]
    return scalars, tensor_dtypes


def global_axis(node):
    """精确匹配 t + b*d，支持加法/乘法两侧交换，不做一般整数代数重写。"""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return None
    for thread, product in ((node.left, node.right), (node.right, node.left)):
        if not isinstance(product, ast.BinOp) or not isinstance(product.op, ast.Mult):
            continue
        for axis in 'xyz':
            if ast.unparse(thread) != 'cuda.threadIdx.'+axis:
                continue
            factors = {ast.unparse(product.left), ast.unparse(product.right)}
            if factors == {'cuda.blockIdx.'+axis, 'cuda.blockDim.'+axis}:
                return axis
    return None


def _sum_terms(node):
    """把加法链拆成项；只拆 Add，不做一般代数重写。"""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sum_terms(node.left) + _sum_terms(node.right)
    return [node]


def _is_col_component(node):
    """`cuda.threadIdx.x` 或 `cuda.blockIdx.x * cuda.blockDim.x`（乘序可交换）。"""
    text = ast.unparse(node)
    if text == 'cuda.threadIdx.x':
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        factors = {ast.unparse(node.left), ast.unparse(node.right)}
        return factors == {'cuda.blockIdx.x', 'cuda.blockDim.x'}
    return False


def _row_term(node):
    """`cuda.blockIdx.y * S` 中的 S（乘序可交换）；不是行项返回 None。"""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
        return None
    for block, other in ((node.left, node.right), (node.right, node.left)):
        if ast.unparse(block) != 'cuda.blockIdx.y':
            continue
        if any(attribute in ast.unparse(other) for attribute in ('threadIdx', 'blockIdx', 'blockDim')):
            return None
        return other
    return None


def flat_form(node, columns):
    """识别展平线性坐标 `by*S + (bx*blockDim.x + tx)`。

    返回 (行跨距表达式, 列变量名或 None)；不是该形态返回 None。
    只匹配"行项 + 完整列分量"的三项和/两步和，不做一般整数代数。
    """
    terms = _sum_terms(node)
    row, column_terms = None, []
    for term in terms:
        stride = _row_term(term)
        if stride is not None:
            if row is not None:
                return None
            row = stride
            continue
        if isinstance(term, ast.Name) and term.id in columns:
            column_terms.append(term)
            continue
        if _is_col_component(term):
            column_terms.append(term)
            continue
        return None
    if row is None:
        return None
    if len(column_terms) == 1 and isinstance(column_terms[0], ast.Name):
        return row, column_terms[0].id
    if len(column_terms) == 2 and all(_is_col_component(term) for term in column_terms):
        return row, None
    return None


def normalize_flat_grid(body):
    """把展平的二维坐标规范化成 `cuda.grid(1)`，并说明用到了哪个行跨距。

    形态（线程别名已展开后）：
        col = bx*blockDim.x + tx          # 可省略：列分量内联在下一行
        p   = by*S + col
    前提（与 `cuda.grid(n)` 路径同一类前置条件，必须由调用方声明）：
        源 launch 的 (by, col) 稠密且唯一覆盖输出域，行跨距 S 是输出行长度。
    返回 (新函数体, 行跨距表达式或 None)。识别失败时原样返回，不做部分改写。
    """
    if not body:
        return body, None
    columns, consumed, flat = {}, 0, None
    for statement in body:
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            break
        form = flat_form(statement.value, columns)
        if form is not None:
            flat = (statement, form)
            consumed += 1
            break
        terms = _sum_terms(statement.value)
        if terms and all(_is_col_component(term) or
                         (isinstance(term, ast.Name) and term.id in columns) for term in terms):
            columns[statement.targets[0].id] = True
            consumed += 1
            continue
        break
    if flat is None:
        return body, None
    statement, (stride, column_var) = flat
    rest = body[consumed:]
    # 被折叠掉的列变量/线程属性不能在后面继续使用，否则改写会静默改变语义。
    used = {node.id for node in ast.walk(ast.Module(body=rest, type_ignores=[]))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    dangling = set(columns) & used
    if dangling:
        raise Unsupported('展平坐标的列分量在后续语句中被使用，暂不支持: ' + ', '.join(sorted(dangling)))
    attributes = {ast.unparse(node) for item in rest for node in ast.walk(item)
                  if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                  and isinstance(node.value.value, ast.Name) and node.value.value.id == 'cuda'
                  and node.value.attr in _THREAD_ATTRS}
    if attributes:
        raise Unsupported('展平坐标之后仍直接使用线程属性，暂不支持: ' + ', '.join(sorted(attributes)))
    grid = ast.Assign(targets=[statement.targets[0]], value=ast.Call(
        func=ast.Attribute(value=ast.Name(id='cuda', ctx=ast.Load()), attr='grid', ctx=ast.Load()),
        args=[ast.Constant(value=1)], keywords=[]))
    return [ast.fix_missing_locations(ast.copy_location(grid, statement))] + rest, stride


_THREAD_ATTRS = ('threadIdx', 'blockIdx', 'blockDim')


def expand_thread_aliases(body):
    """展开 `tx = cuda.threadIdx.x` 这类别名，让 normalize_grid 能匹配手写全局坐标。

    只展开函数体开头连续的定义；定义的属性种类限于 threadIdx/blockIdx/blockDim 的 x/y/z。
    """
    aliases = {}
    consumed = 0
    for statement in body:
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            break
        value = statement.value
        if not (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Attribute)
                and isinstance(value.value.value, ast.Name) and value.value.value.id == 'cuda'
                and value.value.attr in _THREAD_ATTRS and value.attr in 'xyz'):
            break
        aliases[statement.targets[0].id] = value
        consumed += 1
    if not consumed:
        return body

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in aliases:
                replacement = ast.parse(ast.unparse(aliases[node.id]), mode='eval').body
                return ast.copy_location(replacement, node)
            return node

    rest = [Substitute().visit(statement) for statement in body[consumed:]]
    return [ast.fix_missing_locations(statement) for statement in rest]


def _cuda_grid_shape(node, attribute):
    """`cuda.grid(n)` / `cuda.gridsize(n)` → n；其它形态返回 None。"""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'cuda' \
            and node.func.attr == attribute and not node.keywords and len(node.args) == 1:
        value = node.args[0]
        if isinstance(value, ast.Constant) and type(value.value) is int:
            return value.value
    return None


def expand_tuple_assign(body):
    """`a, b = x, y` → 两条独立赋值。

    N-body 类代码常态（`ax, ay, az = 0.0, 0.0, 0.0`、
    `px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]`）；展开后就是普通绑定。
    只处理"两侧都是元组、元素全是 Name/表达式"的情形，其余原样交回。
    """
    result = []
    for node in body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1                 and isinstance(node.targets[0], ast.Tuple)                 and isinstance(node.value, ast.Tuple)                 and len(node.targets[0].elts) == len(node.value.elts)                 and all(isinstance(item, ast.Name) for item in node.targets[0].elts):
            for target, value in zip(node.targets[0].elts, node.value.elts):
                result.append(ast.copy_location(
                    ast.Assign(targets=[ast.Name(id=target.id, ctx=ast.Store())], value=value),
                    node))
            continue
        if isinstance(node, ast.For) and not node.orelse:
            node.body = expand_tuple_assign(node.body)
        elif isinstance(node, ast.If):
            node.body = expand_tuple_assign(node.body)
            node.orelse = expand_tuple_assign(node.orelse)
        result.append(node)
    return result


def normalize_grid_stride(body):
    """把网格步长循环规范化成一次覆盖。

    形态一（`for`）：
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(start, STOP, stride):
            <body>

    形态二（`while`，20261026-163 S1）：
        i = cuda.grid(1)
        stride = cuda.gridsize(1)
        while i < STOP:
            <body>
            i += stride

    tile 级没有"每线程多轮"的对应物：所有线程合起来正好覆盖 `[0, STOP)`，
    因此改写成 `i = cuda.grid(1)` + `if i < STOP:`，交给逐元素/展平路径。
    等价前提（2026-10-26 更正，原文“网格总量不足 STOP 时源码不完整”是错的）：
    线程 `t` 的迭代集合是 `{t + kG}`（`G = gridDim·blockDim > 0`），每个 `n ≥ 0` 唯一写成
    `n = qG + r`（`0 ≤ r < G`），故 `∪_{t<G}{t+kG} = {0,1,2,…}` 且无重复——**任意 `G > 0`**
    都恰好覆盖 `[0, STOP)` 每元素一次，不要求 `G ≥ STOP`。需要的前提是：
    源 launch 唯一枚举 `t ∈ [0, G)`；写入位置与值只依赖 `i` 与该元素自身（跨元素依赖由下游
    投影/依赖门拒绝，本通道不另证）；`i + kG` 在 int64 内不溢出。
    `while` 形态额外要求：循环是函数体最后一条语句且**无 else**、前缀恰为 grid/gridsize 两条绑定、
    `STOP` 与 `i`/`stride` 无关、尾部恰一条 `i += stride`、循环体不重绑定 `i`/`stride`、
    不含 `break/continue/return` 与内层 `while`、体内读 `i` 且不读 `stride`。
    最后一条只用于排除“每轮写同一地址”（写次数与目标不同），**不作为等价证明**。
    任何一条不满足都原样交回旧门拒绝。
    `for...else`：非空 `else` 在循环正常结束时执行；一次性覆盖改写没有对应物，因此**不采纳**、原样交回旧路径明确拒绝（20261026-166 起，此前会静默丢弃 else）。
    """
    if not body:
        return body
    aliases = {}
    for position, statement in enumerate(body):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                and isinstance(statement.targets[0], ast.Name):
            aliases[statement.targets[0].id] = statement.value
            continue
        if isinstance(statement, ast.While):
            rewritten = _normalize_while_grid_stride(body, position, aliases)
            return body if rewritten is None else rewritten
        if not isinstance(statement, ast.For):
            return body
        call = statement.iter
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == 'range' and not call.keywords and len(call.args) == 3):
            return body
        start, stop, step = call.args
        # `for...else`：非空 else 在循环正常结束时执行；一次性覆盖改写没有对应物，
        # 丢弃它就是静默改语义（165 及以前会直接丢）。不采纳，交回旧路径明确拒绝。
        if statement.orelse:
            return body
        names = set()
        resolved = start
        if isinstance(start, ast.Name) and start.id in aliases:
            names.add(start.id)
            resolved = aliases[start.id]
        start_dim = _cuda_grid_shape(resolved, 'grid')
        resolved = step
        if isinstance(step, ast.Name) and step.id in aliases:
            names.add(step.id)
            resolved = aliases[step.id]
        step_dim = _cuda_grid_shape(resolved, 'gridsize')
        if start_dim is None or step_dim is None or start_dim != step_dim:
            return body
        axis = statement.target.id
        assign = ast.Assign(
            targets=[ast.Name(id=axis, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id='cuda', ctx=ast.Load()),
                                   attr='grid', ctx=ast.Load()),
                args=[ast.Constant(value=start_dim)], keywords=[]))
        guard = ast.If(
            test=ast.Compare(left=ast.Name(id=axis, ctx=ast.Load()),
                             ops=[ast.Lt()], comparators=[stop]),
            body=statement.body, orelse=[])
        kept = body[:position] + [assign, guard] + body[position + 1:]
        # 起点/步长变量只服务于这个循环；别处仍在使用时保留，避免改写改变语义。
        used = {node.id for item in kept for node in ast.walk(item)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        result = [item for item in kept
                  if not (isinstance(item, ast.Assign) and len(item.targets) == 1
                          and isinstance(item.targets[0], ast.Name)
                          and item.targets[0].id in names
                          and item.targets[0].id not in used)]
        return [ast.fix_missing_locations(item) for item in result]
    return body


def _bound_names(nodes):
    """这些语句（含嵌套）里被绑定的名字集合（Assign/For/AugAssign/AnnAssign 目标）。"""
    return {node.id for item in nodes for node in ast.walk(item)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}


def _loaded_names(nodes):
    return {node.id for item in nodes for node in ast.walk(item)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def _normalize_while_grid_stride(body, position, aliases):
    """`while i < STOP: <body>; i += stride` → `i = cuda.grid(k)` + `if i < STOP: <body>`。

    与 `for` 形态产出同样的两个节点；资格不满足时返回 `None`，由旧路径拒绝
    （不是"接受但零校验"）。资格见 `normalize_grid_stride` 文档串。
    """
    statement = body[position]
    # 循环后仍有语句：源码里 `i` 的残余值（首个 >= STOP 的值）会被使用，一次性改写不等价。
    if position + 1 != len(body):
        return None
    # `while...else`：非空 else 在循环正常结束时执行一次；一次性改写没有对应物，明确拒绝。
    if statement.orelse:
        return None
    test = statement.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Lt) and isinstance(test.left, ast.Name)):
        return None
    axis = test.left.id
    stop = test.comparators[0]
    start_dim = _cuda_grid_shape(aliases.get(axis), 'grid')
    if start_dim is None:
        return None
    loop_body = statement.body
    if not loop_body:
        return None
    tail = loop_body[-1]
    if not (isinstance(tail, ast.AugAssign) and isinstance(tail.target, ast.Name)
            and tail.target.id == axis and isinstance(tail.op, ast.Add)
            and isinstance(tail.value, ast.Name)):
        return None
    stride = tail.value.id
    if _cuda_grid_shape(aliases.get(stride), 'gridsize') != start_dim:
        return None
    inner = loop_body[:-1]
    bound = _bound_names(inner)
    if axis in bound or stride in bound:
        return None
    for item in inner:
        for node in ast.walk(item):
            if isinstance(node, (ast.Return, ast.Break, ast.Continue, ast.While)):
                return None
    loaded = _loaded_names(inner)
    # `i` 必须真的被使用（否则每轮写同一地址，"一次性覆盖"不等价）；`stride` 不得参与计算。
    if axis not in loaded or stride in loaded:
        return None
    if _loaded_names([stop]) & {axis, stride}:
        return None
    grid = ast.Assign(
        targets=[ast.Name(id=axis, ctx=ast.Store())],
        value=ast.Call(
            func=ast.Attribute(value=ast.Name(id='cuda', ctx=ast.Load()),
                               attr='grid', ctx=ast.Load()),
            args=[ast.Constant(value=start_dim)], keywords=[]))
    guard = ast.If(
        test=ast.Compare(left=ast.Name(id=axis, ctx=ast.Load()),
                         ops=[ast.Lt()], comparators=[stop]),
        body=inner, orelse=[])
    # 前缀只允许"本循环的两个绑定"：`i = cuda.grid(k)`（必须是首条）与
    # `stride = cuda.gridsize(k)`，别无其它语句。前缀里出现原子操作/未知调用/别名等
    # 一律 **不改写**，原样交回旧路径（删除无人读取的赋值会丢副作用，不能做）。
    prefix = body[:position]
    if len(prefix) != 2:
        return None
    head, step_bind = prefix
    if not (isinstance(head, ast.Assign) and len(head.targets) == 1
            and isinstance(head.targets[0], ast.Name) and head.targets[0].id == axis
            and _cuda_grid_shape(head.value, 'grid') == start_dim):
        return None
    if not (isinstance(step_bind, ast.Assign) and len(step_bind.targets) == 1
            and isinstance(step_bind.targets[0], ast.Name) and step_bind.targets[0].id == stride
            and _cuda_grid_shape(step_bind.value, 'gridsize') == start_dim):
        return None
    return [ast.fix_missing_locations(item) for item in (grid, guard)]


def normalize_grid(body):
    if not body:
        return body
    assignments, coordinates, consumed = [], [], 0
    for statement in body[:3]:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            break
        target = statement.targets[0]
        targets = target.elts if isinstance(target, ast.Tuple) else [target]
        values = statement.value.elts if isinstance(statement.value, ast.Tuple) else [statement.value]
        if len(targets) != len(values) or not all(isinstance(item, ast.Name) for item in targets):
            break
        axes = [global_axis(value) for value in values]
        if any(axis is None for axis in axes):
            break
        assignments.extend(targets)
        coordinates.extend(axes)
        consumed += 1
    if not consumed:
        return body
    if coordinates != list('xyz'[:len(coordinates)]) or not 1 <= len(coordinates) <= 3:
        raise Unsupported('手写全局坐标必须依次完整对应x/y/z，不能混轴或重复轴')
    target = assignments[0] if len(assignments) == 1 else ast.Tuple(elts=assignments, ctx=ast.Store())
    grid = ast.Assign(targets=[target], value=ast.Call(
        func=ast.Attribute(value=ast.Name(id='cuda', ctx=ast.Load()), attr='grid', ctx=ast.Load()),
        args=[ast.Constant(value=len(assignments))], keywords=[]))
    return [ast.fix_missing_locations(ast.copy_location(grid, body[0]))] + body[consumed:]
