# -*- coding: utf-8 -*-
"""源码形式规范化；不执行导入，不推测未知张量rank。"""
import ast
import math
import operator
from semantic import MATH_FUNCTIONS, Unsupported
from thread_ir import DTYPES


PROTECTED_NAMES = {'cuda', 'range', 'math', 'numpy', 'np', 'abs', 'max', 'min', 'len'}
# 与 thread_capture.CUDA_MEMORY_PRIMITIVES 一致：这里只为"把真因报在导入名之前"，
# 判定本身不新增或放宽任何能力。
CUDA_MEMORY_PRIMITIVES = {'shared', 'local'}


def is_cuda_memory_call(node):
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == 'cuda'
            and node.func.value.attr in CUDA_MEMORY_PRIMITIVES)
ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
              ast.Mod: operator.mod, ast.Pow: operator.pow}
INT64 = (-(1 << 63), (1 << 63))


def literal_arithmetic(node, values):
    """只折叠字面量算术；任何调用、下标、容器或未知名都返回None表示求不出。"""
    if isinstance(node, ast.Constant):
        if type(node.value) is float and not math.isfinite(node.value):
            return None                     # 1e400 等源码常量在 AST 里已是 inf
        if type(node.value) is int and not INT64[0] <= node.value < INT64[1]:
            return None
        return node.value if type(node.value) in (bool, int, float) else None
    if isinstance(node, ast.Name):
        value = values.get(node.id)
        # 序列值不能当标量参与折叠：`col * 2` 在 numpy 里是逐元素，不是元组重复。
        return None if isinstance(value, (tuple, list)) else value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = literal_arithmetic(node.operand, values)
        if value is None or isinstance(value, bool) or type(value) not in (int, float):
            return None
        result = value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(result, int) and not INT64[0] <= result < INT64[1]:
            return None
        return result
    if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC:
        left = literal_arithmetic(node.left, values)
        right = literal_arithmetic(node.right, values)
        if left is None or right is None or isinstance(left, bool) or isinstance(right, bool):
            return None
        try:
            result = ARITHMETIC[type(node.op)](left, right)
        except (ZeroDivisionError, OverflowError, ValueError, TypeError):
            return None
        # 正面限定标量结果：`(-8)**0.5` 会得到 complex，不能当常量使用。
        if type(result) not in (int, float):
            return None
        if isinstance(result, float) and not math.isfinite(result):
            return None
        if isinstance(result, int) and not INT64[0] <= result < INT64[1]:
            return None
        return result
    return None


def literal_sequence(node, values):
    """模块级常量序列：元组/列表字面量，或 `np.array([...])` 字面量。

    只接受逐元素可静态求值的标量；任何未知名或调用都返回 None（不猜值）。
    """
    is_array = False
    if isinstance(node, (ast.Tuple, ast.List)):
        items = node.elts
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and isinstance(node.func.value, ast.Name) and node.func.value.id in ('np', 'numpy') \
            and node.func.attr == 'array' and len(node.args) == 1 and not node.keywords \
            and isinstance(node.args[0], (ast.List, ast.Tuple)):
        items = node.args[0].elts
        is_array = True
    else:
        return None
    result = []
    for item in items:
        value = literal_arithmetic(item, values)
        if value is None or type(value) not in (int, float):
            return None
        if isinstance(value, int) and not INT64[0] <= value < INT64[1]:
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        result.append(value)
    if is_array and len({type(value) for value in result}) > 1:
        # numpy 会把混合 int/float 整体提升到 float64（大整数会丢精度）；
        # 逐元素保留原类型会改变语义，因此这类字面量不折叠。
        return None
    return tuple(result)


def _root_name(target):
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _module_write_counts(tree):
    """模块级（tree.body 顶层）每个名字被写入的次数。

    函数体内的同名普通赋值是局部遮蔽，不算模块修改（helper 里的 `factor = 7`
    不能使模块常量 `factor = 3` 失效）。
    """
    counts = {}

    def bump(target):
        name = _root_name(target)
        if name:
            counts[name] = counts.get(name, 0) + 1

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue          # 函数体内的同名赋值是局部遮蔽
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    bump(target)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                bump(node.target)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                bump(node.target)
    return counts


def _all_bindings(tree):
    """整棵树（含函数体）里每个名字被绑定的次数——严格版，用于常量上界 guard 窄门。

    与 `_module_write_counts` 不同：这里连函数体内的同名赋值/形参也算一次绑定，
    所以"模块常量名在别处被遮蔽"也会被排除。宁严勿宽：本窄门只放行
    明确"恰好绑定一次"的模块级字面量。
    """
    counts = {}

    def bump(name):
        counts[name] = counts.get(name, 0) + 1

    def bind_target(target):
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind_target(element)
            return
        if isinstance(target, ast.Starred):
            bind_target(target.value)
            return
        if isinstance(target, ast.Name):
            bump(target.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind_target(target)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            bind_target(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bind_target(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bind_target(item.optional_vars)
        elif isinstance(node, ast.comprehension):
            bind_target(node.target)
        elif isinstance(node, ast.NamedExpr):
            bind_target(node.target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bump(node.name)
            arguments = node.args
            for argument in (list(arguments.posonlyargs) + list(arguments.args)
                             + list(arguments.kwonlyargs)):
                bump(argument.arg)
            if arguments.vararg is not None:
                bump(arguments.vararg.arg)
            if arguments.kwarg is not None:
                bump(arguments.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            arguments = node.args
            for argument in (list(arguments.posonlyargs) + list(arguments.args)
                             + list(arguments.kwonlyargs)):
                bump(argument.arg)
            if arguments.vararg is not None:
                bump(arguments.vararg.arg)
            if arguments.kwarg is not None:
                bump(arguments.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            bump(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bump(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bump(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bump(alias.asname or alias.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bump(node.name)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                bind_target(target)
    return counts


def strict_int_constants(tree):
    """模块级 `int` **字面量**常量，且整个模块内**恰好绑定一次**。

    只用于"常量上界 guard"窄门（`if x < NX`）：域是常量，输出形状按等值校验。
    显式支持元组解包（`nx, ny = 420, 180`）。以下一律**不算**：
    表达式（`n = 2*420`）、float、解包出的非 int、循环变量/形参、import 别名，
    以及任何第二次绑定（含函数体内同名赋值遮蔽）——宁严勿宽。
    """
    counts = _all_bindings(tree)
    values = {}

    def literal_ints(nodes):
        items = []
        for node in nodes:
            if not (isinstance(node, ast.Constant) and type(node.value) is int):
                return None
            items.append(node.value)
        return items

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name):
            names, literals = [target.id], literal_ints([value])
        elif isinstance(target, (ast.Tuple, ast.List)) \
                and isinstance(value, (ast.Tuple, ast.List)) \
                and len(target.elts) == len(value.elts):
            names = [element.id if isinstance(element, ast.Name) else None
                     for element in target.elts]
            literals = literal_ints(value.elts)
        else:
            continue
        if literals is None or any(name is None for name in names):
            continue
        for name, literal in zip(names, literals):
            if name in PROTECTED_NAMES or name.startswith('_tl_'):
                continue
            if counts.get(name, 0) != 1:
                continue
            values[name] = literal
    return values


def _conditional_writes(tree):
    """模块级控制流（if/for/while/with/try）内写入的名字：执行与否不可静态确定。"""
    names = set()

    def collect(target):
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                collect(element)
            return
        name = _root_name(target)
        if name:
            names.add(name)

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While,
                             ast.With, ast.AsyncWith, ast.Try)):
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        collect(target)
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    collect(node.target)
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    collect(node.target)          # `for factor in ...:` 也会重绑定
                elif isinstance(node, (ast.With, ast.AsyncWith)):
                    for item in node.items:
                        if item.optional_vars is not None:
                            collect(item.optional_vars)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    names.add(node.name)
    return names


def module_scalars(tree, fn):
    """按源码顺序折叠模块级字面量赋值；只读AST，绝不执行模块级代码。

    依赖前面的绑定（如 omega = 1/(3*nulb+0.5)）；求不出的名字不进结果，
    kernel 引用它时仍按“未知名称”拒绝，而不是猜一个值。
    顺序语义：可求值的重新赋值覆盖旧值，不可求值的赋值删除旧值；
    模块级控制流内的赋值执行与否不确定，一律删除旧值（保守）。
    """
    values = {}
    conditional = _conditional_writes(tree)
    references = {}

    def sequence_refs(node):
        """RHS 引用的常量序列：既认 values 里的序列，也认 references 里的别名节点。"""
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)
                and (isinstance(values.get(n.id), tuple) or n.id in references)}
    for node in tree.body:
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While,
                             ast.With, ast.AsyncWith, ast.Try)):
            for name in conditional:
                values.pop(name, None)
            continue
        if isinstance(node, ast.Delete):
            # `del factor` 后名字不再有值。
            for target in node.targets:
                name = _root_name(target)
                if name:
                    values.pop(name, None)
            continue
        if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            # 增量赋值依赖运行期旧值；注解赋值按普通绑定处理（无 value 时忽略）。
            name = _root_name(node.target)
            if name:
                values.pop(name, None)
            if not (isinstance(node, ast.AnnAssign) and node.value is not None):
                continue
            node = ast.Assign(targets=[node.target], value=node.value)
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            # 链式赋值（`factor = other = 9`）：所有目标绑定同一值。
            value = literal_arithmetic(node.value, values)
            if value is None:
                value = literal_sequence(node.value, values)
            assigned = []
            rhs_refs = sequence_refs(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.append(target.id)
                    if value is None:
                        values.pop(target.id, None)
                    else:
                        values[target.id] = value
                else:
                    name = _root_name(target)
                    if name:
                        values.pop(name, None)
            if len(assigned) > 1 or rhs_refs:
                # 链式赋值共享同一个可变对象：任一目标的 mutation 影响全部目标。
                # RHS 是已有引用时（`other = third = seq`）同样要建边。
                for name in assigned:
                    references[name] = (set(references.get(name, ()))
                                        | (set(assigned) - {name}) | rhs_refs)
            continue
        targets = node.targets[0]
        names = targets.elts if isinstance(targets, ast.Tuple) else [targets]
        # 只有目标也是元组时才是解包；`shape = (4, 8)` 是单个名字绑定常量元组。
        items = node.value.elts if isinstance(targets, ast.Tuple) \
            and isinstance(node.value, ast.Tuple) else [node.value]
        if len(names) != len(items) or not all(isinstance(n, ast.Name) for n in names):
            continue
        # 多目标赋值（`x, y = y, x`）必须先统一求值再写回，否则会读到本次写入的新值。
        pending = {}
        refs = sequence_refs(node.value)
        for name, item in zip(names, items):
            if name.id in PROTECTED_NAMES or name.id.startswith('_tl_'):
                continue
            value = literal_arithmetic(item, values)
            if value is None:
                value = literal_sequence(item, values)
            if value is None:
                # 不可静态求值的赋值：旧值作废；若它引用常量序列，记录用于失效追踪。
                values.pop(name.id, None)
                if refs:
                    references[name.id] = refs
            pending[name.id] = value
        for name, value in pending.items():
            if value is not None:
                values[name] = value
    # 下标/属性写与任意方法调用都可能修改同一个模块对象：
    # 使该名字引用的常量序列失效（保守拒绝，不追踪方法语义）。
    def pollute(name):
        if not name:
            return
        # 20261025-146：未知调用只能使**容器/序列**常量失效。int/float/bool 在 Python 里
        # 不可变，作为实参传进未知调用不可能被改写；旧逻辑把它们一并删除，导致
        # `from math import sin` 这类裸名调用所在表达式里的模块标量（如 `ny`）被误判失效。
        # 只按**绑定值类型**判定；`safe_calls` 白名单与未知调用策略都不动。
        stack = [name]
        while stack:
            current = stack.pop()
            if current in polluted:
                continue
            value = values.get(current)
            if value is not None and not isinstance(value, tuple):
                continue          # 标量常量：不受未知调用影响
            polluted.add(current)
            stack.extend(references.get(current, ()))

    polluted = set()
    from thread_ir import DTYPES as _DTYPES
    safe_calls = {'int', 'float', 'abs', 'min', 'max', 'len', 'range', 'sum'} | set(_DTYPES)
    # 只有确切的构造/纯函数才免于失效；不能按整个 np 模块豁免
    # （np.copyto / np.put / 带 out= 的 ufunc 都会修改传入容器）。
    np_constructors = {'array', 'zeros', 'ones', 'empty', 'arange', 'linspace',
                       'int32', 'int64', 'uint8', 'uint32', 'float32', 'float64', 'bool_'}
    np_unary = {'sqrt', 'exp', 'log', 'log2', 'log10', 'fabs', 'abs',
                'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                'sinh', 'cosh', 'tanh', 'floor', 'ceil', 'round', 'trunc'}
    np_binary = {'power', 'arctan2', 'max', 'min', 'fmax', 'fmin'}

    def numpy_pure(func_name, call):
        """按签名判断：ufunc 的第三个位置实参是 out，会修改传入容器。"""
        if call.keywords:
            return False
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            return False          # `np.power(*(seq,2,seq))` 展开后仍可能命中 out
        count = len(call.args)
        if func_name in np_constructors:
            return True
        if func_name in np_unary:
            return count == 1
        if func_name in np_binary:
            return count <= 2
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, ast.Store):
            pollute(_root_name(node))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                root = func
                while isinstance(root, ast.Attribute):
                    root = root.value
                module = root.id if isinstance(root, ast.Name) else None
                if module == 'math':
                    continue                  # math 只有纯函数/常量
                if module == 'cuda':
                    continue                  # 分配/索引/原子不修改 Python 容器
                if module in ('np', 'numpy'):
                    direct = isinstance(func.value, ast.Name) and func.value.id in ('np', 'numpy')
                    if direct and numpy_pure(func.attr, node):
                        continue              # 直接成员 np.x 且为纯构造/纯 ufunc（无 out=/where=）
                    # np.put / np.copyto / 带 out 位置的 ufunc：按未知调用处理。
                    for argument in list(node.args) + [kw.value for kw in node.keywords]:
                        for name in ast.walk(argument):
                            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load):
                                pollute(name.id)
                    continue
                if isinstance(func.value, ast.Name):
                    pollute(func.value.id)    # 接收者可能被修改，但实参仍要检查
            if isinstance(func, ast.Name) and func.id in safe_calls:
                continue
            # 未知调用可能修改作为实参（含关键字实参）传入的容器：保守失效。
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                for name in ast.walk(arg):
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load):
                        pollute(name.id)
    for name in polluted:
        values.pop(name, None)
    return values


def validate_imports(tree, fn, allow_module_bindings=False, extra_math=()):
    modules, unused, functions, bindings = set(), set(), {}, {}

    def bind(name, origin):
        # 同一来源的重复导入是幂等绑定（包括sqrt as sqrt），不是冲突。
        if name in bindings and bindings[name] != origin:
            raise Unsupported('导入名称存在冲突绑定: ' + name)
        bindings[name] = origin

    for node in tree.body:
        if node is fn or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                          and isinstance(node.value.value, str)):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                if (alias.name, alias.asname) in {
                        ('math', None), ('numpy', None), ('numpy', 'np'), ('numba', None)}:
                    bind(name, (alias.name, None))
                    modules.add(name)
                    if name == 'numba':
                        unused.add(name)
                elif allow_module_bindings:
                    # 整文件输入：白名单外的导入只作AST存在；kernel真正用到才报错。
                    bind(name, (alias.name, None))
                    unused.add(name)
                else:
                    raise Unsupported('只支持明确的 math/numpy/numba 模块导入')
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in ('numba', 'math'):
            for alias in node.names:
                if node.module == 'numba' and alias.name == 'cuda' and alias.asname is None:
                    bind('cuda', ('numba', 'cuda'))
                    continue
                name = alias.asname or alias.name
                if alias.name == 'cuda' or alias.name == '*' or name in PROTECTED_NAMES:
                    raise Unsupported('导入不能覆盖已识别模块/内置函数，且不支持星号导入')
                bind(name, (node.module, alias.name))
                if node.module == 'math' and (alias.name in MATH_FUNCTIONS or alias.name in extra_math):
                    functions[name] = alias.name
                else:
                    unused.add(name)
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 0 and allow_module_bindings:
            for alias in node.names:
                if alias.name == '*':
                    raise Unsupported('不支持星号导入')
                name = alias.asname or alias.name
                bind(name, (node.module, alias.name))
                unused.add(name)
            continue
        if allow_module_bindings:
            # 整文件输入：其余模块级语句只作AST存在，不解析也不执行；
            # kernel 真正引用的名字仍须由自动折叠或显式constants给出。
            continue
        raise Unsupported('模块只支持kernel、文档字符串及明确的numba/math/numpy导入')
    # 未实现的导入只允许保留在源码外围；不能误认其调用或全局常量。
    # dtype 名（`float64` 等）是**参数/分配的类型文本**，不是"使用了一个未支持的函数"：
    # 它们可以出现在 `cuda.shared.array(dtype=float64)`、`float64(x)` 等位置。
    parents = {child: node for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}
    used = set()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        parent = parents.get(node)
        # `numba.float64` / `np.int32` 是类型文本，不是"使用了未支持的模块成员"。
        if isinstance(parent, ast.Attribute) and parent.value is node \
                and parent.attr in DTYPES and node.id in ('numba', 'numpy', 'np'):
            continue
        used.add(node.id)
    unresolved = (used & unused) - set(DTYPES)
    if unresolved:
        raise Unsupported('尚不支持使用这些导入名称: ' + ', '.join(sorted(unresolved)))
    return modules, functions


def normalize_math_calls(fn, functions):
    """导入来源已证明、局部遮蔽已拒绝；仅规范直接调用，不支持函数值传递。"""
    class Calls(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in functions:
                node.func = ast.copy_location(ast.Attribute(
                    value=ast.Name(id='math', ctx=ast.Load()),
                    attr=functions[node.func.id], ctx=ast.Load()), node.func)
            return self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in functions:
                raise Unsupported('导入数学函数仅支持直接调用，不能作为值使用: ' + node.id)
            return node

    return ast.fix_missing_locations(Calls().visit(fn))


def normalize_sizes(fn, params):
    from source_contract import tensor_ranks
    ranks = tensor_ranks(fn, params)

    class Sizes(ast.NodeTransformer):
        def shape(self, node, name, size=False):
            rank = ranks.get(name)
            if name not in params or rank is None:
                raise Unsupported('尺寸引用需要张量形参与直接访问rank证据: ' + name)
            if size and rank != 1:
                raise Unsupported('仅支持一维张量.size；多维size不是某一维长度: ' + name)
            return ast.copy_location(ast.Subscript(
                value=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()),
                                    attr='shape', ctx=ast.Load()),
                slice=ast.Constant(value=0), ctx=ast.Load()), node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'len':
                if node.keywords or len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                    raise Unsupported('len只支持直接张量形参: ' + ast.unparse(node))
                return self.shape(node, node.args[0].id)
            return self.generic_visit(node)

        def visit_Attribute(self, node):
            if node.attr == 'size' and isinstance(node.value, ast.Name):
                return self.shape(node, node.value.id, size=True)
            return self.generic_visit(node)

    return ast.fix_missing_locations(Sizes().visit(fn))
