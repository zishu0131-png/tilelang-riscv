# -*- coding: utf-8 -*-
"""
numba2tilelang —— 把 Numba CUDA 的 kernel 转成 TileLang 的 T.xxx

设计思路（四层）：
    ① 解析层    Python AST → KernelSpec（结构化的 kernel 描述）
    ② 模式层    语句序列 → 计算模式（点积/逐元素）
    ③ 提升层    线程级模式 → tile 级结构（引入 shared / tiling / pipeline）
    ④ 生成层    输出 T.xxx 源码

支持范围（刻意限定，宁可报错不瞎猜）：
    - @cuda.jit 的 kernel，参数是张量
    - cuda.grid(n) 取线程索引
    - 【二维乘加归约】→ tile 级 GEMM（四种转置、交换项、输出轴换序）
    - 【逐元素算术表达式】→ T.Parallel（轴投影广播/转置读取）
    - 【单轴 sum/max/min 表达式】→ 分块 T.reduce_sum/max/min
    - 【平面多阶段与独立多累加器】→ 顺序归约及后处理
    - 【作用域嵌套归约】→ 外层串行、叶子块协作归约
    - 公共语义基础与分类型捕获/生成并存；各模块只接受已验证子集
    - 完整捕获受限函数体；其余计算模式明确拒绝
"""
from __future__ import annotations
import ast, inspect, math, textwrap
from dataclasses import dataclass, field
from semantic import (Unsupported, Compute, Reduction, capture_expr, capture_pointwise,
                      reads, validate_map, validate_sum)


# ══════════════════════════ ① 解析层 ══════════════════════════

@dataclass
class LoopSpec:
    var: str                      # 循环变量，如 'k'
    extent: str                   # 上界表达式，如 'K'
    acc_var: str                  # 被累加的变量，如 'tmp'
    acc_terms: list               # 累加项，如 [('A', ('i','k')), ('B', ('k','j'))]
    op: str = '*'                 # 乘法
    start: str = '0'
    step: str = '1'
    init_value: float = 0.0
    update_op: str = '+'
    update_expr: str = ''          # 完整右值；不允许只留下部分叶子


@dataclass
class KernelSpec:
    name: str = ''
    params: list = field(default_factory=list)   # 参数名，如 ['A','B','C']
    grid_dims: int = 0                            # cuda.grid(2) → 2
    idx_vars: list = field(default_factory=list)  # 线程索引变量，如 ['i','j']
    loops: list = field(default_factory=list)     # LoopSpec 列表
    store_target: str = ''                        # 输出缓冲名，如 'C'
    store_index: tuple = ()                       # 输出下标，如 ('i','j')
    guard: str = ''                               # 边界条件原文
    pattern: str = ''                             # 'gemm' | 'elementwise'
    lhs: str = ''
    rhs: str = ''
    axis_roles: dict = field(default_factory=dict)
    transpose_A: bool = False
    transpose_B: bool = False
    guard_kind: str = 'none'
    k_constant: int | None = None
    extent_expression: str | None = None          # 归约长度是运行期标量时的源码表达式
    dtype: str = 'float32'                        # 目标契约，不是从源码推断的类型
    compute: Compute | None = None              # 与具体原语无关的计算描述
    read_axes: dict = field(default_factory=dict)  # 输入名 -> 物理下标顺序的计算轴名元组
    program: object = None
    scalar_params: dict = field(default_factory=dict)  # 签名标量形参 -> dtype
    layout: object = None
    flat_stride: str = ''                         # 展平坐标的行跨距原文（非展平为空）
    divisible_extent: tuple = ()                  # A 片：展平基址改写时的 (张量, 维, L)，构造期校验整除
    flat_base_runtime: tuple = ()                 # C2：运行期 L 的 (张量, 维, 标量名)，按前缀截断
    tensor_dtypes: dict = field(default_factory=dict)  # 签名张量形参 -> dtype（无签名为空）


def strip_trailing_sync(body):
    """去掉结尾的无参 `cuda.syncthreads()`。

    逐元素映射里每个线程只碰自己的输出，屏障没有可观察作用；
    展平映射的 tile 级产物不允许出现线程级同步原语，所以只允许它出现在末尾。
    """
    while body and isinstance(body[-1], ast.Expr) \
            and isinstance(body[-1].value, ast.Call) \
            and ast.unparse(body[-1].value.func) == 'cuda.syncthreads' \
            and not body[-1].value.args and not body[-1].value.keywords:
        body = body[:-1]
    return body


def _call_name(node):
    """取出调用名，如 cuda.grid → 'cuda.grid'"""
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    if isinstance(node, ast.Name):
        return node.id
    return ''


def _tensor_access(node):
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        raise Unsupported('只支持直接张量访问，如 A[i, k]')
    indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    if not all(isinstance(x, ast.Name) for x in indices):
        raise Unsupported('张量下标必须是独立变量，不支持偏移、切片或常量下标')
    return node.value.id, tuple(x.id for x in indices)


def _lift_output_loop(body, spec):
    """140 窄门：把"输出列循环"提升为第二并行轴（规则循环映射）。

    只认：body 恰为单个 `for j in range(T.shape[1])`，T 为形参且在循环内以两维读出现；
    循环体仅单目标纯局部赋值 + 一次 `OUT[row, j] = ...` 写回；无分支/嵌套/自增/原地。
    界必须是 `shape(dim=1)`（常量界无法证明完整写域，拒绝）。
    """
    if len(body) != 1 or not isinstance(body[0], ast.For):
        return body
    loop = body[0]
    if loop.orelse or not isinstance(loop.target, ast.Name):
        return body
    j = loop.target.id
    it = loop.iter
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'range'
            and len(it.args) == 1 and not it.keywords):
        return body
    bound = it.args[0]
    if not (isinstance(bound, ast.Subscript) and isinstance(bound.value, ast.Attribute)
            and bound.value.attr == 'shape' and isinstance(bound.value.value, ast.Name)
            and bound.value.value.id in spec.params
            and isinstance(bound.slice, ast.Constant) and bound.slice.value == 1):
        return body
    bound_tensor = bound.value.value.id
    if bound_tensor not in spec.params or not spec.idx_vars or j in spec.idx_vars:
        return body
    row = spec.idx_vars[0]

    # bound 张量必须在循环内以两维下标出现（证明它确实是二维数据）。
    two_dim = False
    for stmt in loop.body:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id == bound_tensor and isinstance(node.slice, ast.Tuple)
                    and len(node.slice.elts) == 2):
                two_dim = True
    if not two_dim:
        return body

    # 第一遍：收集循环内定义的局部名。
    assigned = set()
    for stmt in loop.body:
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.Return, ast.Break, ast.Continue,
                             ast.AugAssign)):
            return body
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return body
        tgt = stmt.targets[0]
        if isinstance(tgt, ast.Name):
            if tgt.id in spec.params or tgt.id in (row, j) or tgt.id in assigned:
                return body
            assigned.add(tgt.id)

    local_names = set()
    store = None
    for stmt in loop.body:
        tgt = stmt.targets[0]
        loads = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)
                 and isinstance(n.ctx, ast.Load)}
        if isinstance(tgt, ast.Name):
            if (loads & (assigned - local_names - {tgt.id})) or tgt.id in loads:
                return body  # 前向引用/自引用（循环携带）
            local_names.add(tgt.id)
        else:
            if not (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
                    and isinstance(tgt.slice, ast.Tuple) and len(tgt.slice.elts) == 2
                    and isinstance(tgt.slice.elts[0], ast.Name)
                    and tgt.slice.elts[0].id == row
                    and isinstance(tgt.slice.elts[1], ast.Name)
                    and tgt.slice.elts[1].id == j):
                return body
            if store is not None:
                return body
            store = tgt.value.id
    if store is None:
        return body
    # 141 收口：输出张量不得在循环体任何位置被读（含局部赋值/表达式），不只 store RHS。
    for stmt in loop.body:
        for node in ast.walk(stmt.value):
            if (isinstance(node, ast.Name) and node.id == store
                    and isinstance(node.ctx, ast.Load)):
                return body

    compare = ast.Compare(left=ast.Name(id=j, ctx=ast.Load()), ops=[ast.Lt()],
                          comparators=[bound])
    spec.guard = ('(%s) and (%s)' % (spec.guard, ast.unparse(compare))
                  if spec.guard else ast.unparse(compare))
    spec.idx_vars = list(spec.idx_vars) + [j]
    spec.grid_dims = len(spec.idx_vars)
    return list(loop.body)


def _rewrite_softmax_roundtrip(body, spec):
    """143 窄门：只有**可证明等价**的 softmax 三段式才改写，其余原样交回旧拒绝。

    被改写/删除的信息逐项校验：三个 loop 的 target/iter/orelse、三处坐标、init 常量、
    exp 参数整棵树、变量角色与重定义、scores/result 的 f32 契约、额外读写与别名。
    """
    import copy

    def is_name(node, value):
        return isinstance(node, ast.Name) and node.id == value

    def index2(node, row, ivar):
        return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2
                and is_name(node.slice.elts[0], row) and is_name(node.slice.elts[1], ivar))

    if len(body) != 5 or not spec.idx_vars or len(spec.idx_vars) != 1:
        return body
    row = spec.idx_vars[0]
    max_init, loop1, sum_init, loop2, loop3 = body

    # max_init 必须是 max_var = -float('inf')
    if not (isinstance(max_init, ast.Assign) and len(max_init.targets) == 1
            and isinstance(max_init.targets[0], ast.Name)):
        return body
    max_var = max_init.targets[0].id
    value = max_init.value
    if not (isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub)
            and isinstance(value.operand, ast.Call)
            and isinstance(value.operand.func, ast.Name) and value.operand.func.id == 'float'
            and len(value.operand.args) == 1 and not value.operand.keywords
            and isinstance(value.operand.args[0], ast.Constant)
            and value.operand.args[0].value == 'inf'):
        return body

    # 三个 loop 必须同 target、同 iter、无 else
    loops = (loop1, loop2, loop3)
    if not all(isinstance(loop, ast.For) and not loop.orelse
               and isinstance(loop.target, ast.Name) and loop.target.id != row
               for loop in loops):
        return body
    ivar = loop1.target.id
    if any(loop.target.id != ivar for loop in loops):
        return body
    if any(ast.unparse(loop.iter) != ast.unparse(loop1.iter) for loop in loops):
        return body
    if not (isinstance(loop1.iter, ast.Call) and isinstance(loop1.iter.func, ast.Name)
            and loop1.iter.func.id == 'range' and len(loop1.iter.args) == 1
            and not loop1.iter.keywords):
        return body

    # loop1：max(max_var, scores[row, ivar])
    if len(loop1.body) != 1:
        return body
    stmt = loop1.body[0]
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and is_name(stmt.targets[0], max_var)
            and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == 'max' and len(stmt.value.args) == 2):
        return body
    first, second = stmt.value.args
    if not (is_name(first, max_var) and index2(second, row, ivar)):
        return body
    scores_name = second.value.id
    if scores_name not in spec.params:
        return body
    declared = dict(getattr(spec, 'tensor_dtypes', {}) or {})
    if declared.get(scores_name, 'float32') != 'float32':
        return body

    # sum_init：sum_var = 0.0，且角色名互异
    if not (isinstance(sum_init, ast.Assign) and len(sum_init.targets) == 1
            and isinstance(sum_init.targets[0], ast.Name)
            and isinstance(sum_init.value, ast.Constant)
            and sum_init.value.value == 0.0):
        return body
    sum_var = sum_init.targets[0].id
    if sum_var in (max_var, ivar, row, scores_name):
        return body

    # loop2：val = exp(scores[row,ivar] - max_var); result[row,ivar] = val; sum_var += val
    if len(loop2.body) != 3:
        return body
    a1, a2, a3 = loop2.body
    if not (isinstance(a1, ast.Assign) and len(a1.targets) == 1
            and isinstance(a1.targets[0], ast.Name)
            and isinstance(a1.value, ast.Call) and isinstance(a1.value.func, ast.Attribute)
            and isinstance(a1.value.func.value, ast.Name) and a1.value.func.value.id == 'math'
            and a1.value.func.attr == 'exp' and len(a1.value.args) == 1
            and not a1.value.keywords):
        return body
    val_name = a1.targets[0].id
    if val_name in (max_var, sum_var, ivar, row, scores_name):
        return body
    arg = a1.value.args[0]
    if not (isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Sub)
            and index2(arg.left, row, ivar) and arg.left.value.id == scores_name
            and is_name(arg.right, max_var)):
        return body
    if not (isinstance(a2, ast.Assign) and len(a2.targets) == 1
            and index2(a2.targets[0], row, ivar) and is_name(a2.value, val_name)):
        return body
    result_name = a2.targets[0].value.id
    if result_name not in spec.params or result_name == scores_name:
        return body
    if declared.get(result_name, 'float32') != 'float32':
        return body
    # 循环体内不得读 result（含局部表达式）；存储目标的基名本身不算读。
    # 也要核 loop1 与 loop3 的 iter，避免 result 通过其它段漏入。
    for root in (loop1.body[0], loop1.iter, loop3.iter, a1, a2.value, a3, loop2.iter):
        for node in ast.walk(root):
            if (isinstance(node, ast.Name) and node.id == result_name
                    and isinstance(node.ctx, ast.Load)):
                return body
    if not (isinstance(a3, ast.AugAssign) and is_name(a3.target, sum_var)
            and isinstance(a3.op, ast.Add) and is_name(a3.value, val_name)):
        return body

    # loop3：result[row,ivar] /= sum_var
    if len(loop3.body) != 1:
        return body
    b1 = loop3.body[0]
    if not (isinstance(b1, ast.AugAssign) and index2(b1.target, row, ivar)
            and isinstance(b1.op, ast.Div) and is_name(b1.value, sum_var)):
        return body
    if b1.target.value.id != result_name:
        return body

    exp_expr = copy.deepcopy(a1.value)
    new_loop2 = ast.For(target=loop2.target, iter=loop2.iter, body=[
        ast.AugAssign(target=ast.Name(id=sum_var, ctx=ast.Store()), op=ast.Add(),
                      value=copy.deepcopy(exp_expr))], orelse=[])
    cast_inner = ast.Call(func=ast.Name(id='float32', ctx=ast.Load()),
                          args=[copy.deepcopy(exp_expr)], keywords=[])
    divide = ast.BinOp(left=cast_inner, op=ast.Div(),
                       right=ast.Name(id=sum_var, ctx=ast.Load()))
    cast_outer = ast.Call(func=ast.Name(id='float32', ctx=ast.Load()),
                          args=[divide], keywords=[])
    new_loop3 = ast.For(target=loop3.target, iter=loop3.iter, body=[
        ast.Assign(targets=[copy.deepcopy(b1.target)], value=cast_outer)], orelse=[])
    ast.copy_location(new_loop2, loop2)
    ast.fix_missing_locations(new_loop2)
    ast.copy_location(new_loop3, loop3)
    ast.fix_missing_locations(new_loop3)
    spec.softmax_roundtrip = True
    return [max_init, loop1, sum_init, new_loop2, new_loop3]


def parse_source(src: str, scalar_types=None) -> KernelSpec:
    """按执行顺序捕获受限 kernel；无法表达的语法明确拒绝，不跳过。"""
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError as exc:
        raise Unsupported('Python 语法错误: ' + str(exc)) from exc
    from source_contract import (is_cuda_jit, is_device_jit, normalize_decorator, normalize_grid,
                                 normalize_grid_stride, normalize_flat_grid,
                                 expand_thread_aliases)
    # device=True 的是 helper（内联进调用点），不是入口 kernel。
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and any(is_cuda_jit(d) and not is_device_jit(d) for d in node.decorator_list)]
    if len(functions) != 1:
        raise Unsupported('需要唯一一个带 @cuda.jit 的函数')
    fn = functions[0]
    from frontend import validate_imports, normalize_math_calls, normalize_sizes, PROTECTED_NAMES
    # 先内联 linear helper：签名 rank 校验需要看到展开后的直接下标。
    from helper_inline import inline_linear_helpers
    inline_linear_helpers(tree, fn)
    scalar_params, tensor_dtypes = normalize_decorator(tree, fn)
    from semantic import MATH_FUNCTIONS
    modules, math_functions = validate_imports(
        tree, fn, allow_module_bindings=True,
        extra_math=tuple(MATH_FUNCTIONS) + ('sin', 'cos', 'tan', 'atan2', 'fabs',
                                            'isinf', 'pow', 'pi', 'e', 'tau'))
    # `-> None` 是无语义的返回注解（numba 里等价于无返回），允许。
    returns_none = (fn.returns is None
                    or (isinstance(fn.returns, ast.Constant) and fn.returns.value is None))
    if len(fn.decorator_list) != 1 or not returns_none:
        raise Unsupported('不支持额外装饰器或返回类型声明')
    args = fn.args
    # 参数类型注解（`boids: DeviceNDArray`）不影响语义：dtype 仍由静态签名或
    # 使用方式推断，注解只当注释忽略。默认值/*args/**kwargs 改变调用约定，仍拒绝。
    if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults:
        raise Unsupported('只支持无默认值的普通张量形参')
    params = [a.arg for a in args.args]
    # 调用方声明的标量类型特化契约（`scalar_types`）：只影响类型，不改变源码语义。
    # 这是**声明**，不是从源码证明原调用类型，也不做运行期 Python 类型校验。
    declared = dict(scalar_types or {})
    if declared:
        from source_contract import tensor_used_names
        tensor_used = tensor_used_names(fn)
        for name, dtype in declared.items():
            if name not in params:
                raise Unsupported('scalar_types 里有未知形参: ' + name)
            if name in tensor_dtypes:
                raise Unsupported('scalar_types 不能声明张量形参: ' + name)
            if name in tensor_used:
                raise Unsupported('scalar_types 声明的形参在源码里以张量方式使用: ' + name)
            if name in scalar_params and scalar_params[name] != dtype:
                raise Unsupported('scalar_types 与静态签名的标量类型冲突: %s（%s vs %s）'
                                  % (name, scalar_params[name], dtype))
        # **只对声明项**补"未使用"检查：旧签名/推断路径的未使用参数规则不变
        used = {node.id for node in ast.walk(fn)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        unused_declared = [name for name in declared if name not in used]
        if unused_declared:
            raise Unsupported('scalar_types 声明的形参在源码中未被使用: '
                              + ', '.join(unused_declared))
    inferred = []
    if not scalar_params and not tensor_dtypes:
        # 无静态签名：按"从未以张量方式使用"推断标量，dtype 记 float64 并在生成物头注释声明。
        from source_contract import infer_scalars, integer_scalars
        inferred = infer_scalars(fn, params)
        integers = integer_scalars(fn, params)
        scalar_params = {name: ('int64' if name in integers else 'float64')
                         for name in inferred}
    # 声明优先于无签名默认；已声明的项不再算"推断"
    for name, dtype in declared.items():
        scalar_params[name] = dtype
    inferred = [name for name in inferred if name not in declared]
    spec = KernelSpec(name=fn.name, params=params,
                      scalar_params=dict(scalar_params),
                      tensor_dtypes=dict(tensor_dtypes))
    spec.inferred_scalars = tuple(inferred)
    spec.declared_scalar_types = tuple(sorted(declared))
    from semantic import Expr
    scalar_bindings = {name: Expr('scalar', value=name, dtype=dtype)
                       for name, dtype in spec.scalar_params.items()}
    # 数学常量（pi/e/tau）按字面量折叠；其余导入名才作为可调用函数规范化。
    math_constant_values = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
    constant_bindings = {name: value for name, original in math_functions.items()
                         if original in math_constant_values
                         for value in (math_constant_values[original],)}
    callable_functions = {name: original for name, original in math_functions.items()
                          if name not in constant_bindings}

    # 模块级标量常量：与 threads 路径同一折叠规则，作为 const 绑定注入表达式捕获。
    from frontend import module_scalars
    bound_norm = {fn.name, *spec.params} | {
        n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for name, value in constant_bindings.items():
        scalar_bindings[name] = Expr('const', value=value, dtype='float64')
    for name, value in module_scalars(tree, fn).items():
        if type(value) not in (bool, int, float) or name in bound_norm or name in spec.scalar_params:
            continue
        scalar_bindings[name] = Expr('const', value=value,
                                     dtype='int64' if type(value) is int else 'float64')
    # Python 的局部绑定作用于整个函数，而不只是赋值之后；捕获前统一拒绝遮蔽。
    # D1：`float`/`int` 作为内置转换名，禁止被形参/局部/函数名遮蔽（optimized 路径；
    # threads 路径不扩大）。整函数生效——调用在前、赋值在后也会被这里挡住。
    known_names = PROTECTED_NAMES | set(math_functions) | {'float', 'int'}
    bound_names = {fn.name, *spec.params} | {
        node.id for node in ast.walk(fn)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    if bound_names & known_names:
        raise Unsupported('函数、形参及局部绑定不能遮蔽已识别模块或内置函数的保留名称')
    # D1：模块级同名绑定会让 call 指向用户对象，整份源码拒绝。
    # 必须按**绑定来源**判定：Assign（含元组/列表解包）、AnnAssign、AugAssign、模块 for 目标、
    # 模块级 def/class 名、import 别名；不能只看顶层 `Name` 赋值。
    if _module_bound_names(tree.body) & {'float', 'int'}:
        shadowed = sorted(_module_bound_names(tree.body) & {'float', 'int'})
        raise Unsupported('模块级绑定不能遮蔽内置转换名: ' + ', '.join(shadowed))
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in {'math', 'numpy', 'np'} and node.value.id not in modules:
            raise Unsupported('数学模块必须有明确导入绑定: ' + node.value.id)

    # 先检查源码原有模块绑定，再规范已证明来源的直接数学调用。
    fn = normalize_math_calls(fn, callable_functions)
    fn = normalize_sizes(fn, spec.params)
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body.pop(0)  # 文档字符串没有计算副作用
    if not body:
        raise Unsupported('函数体为空')
    body = expand_thread_aliases(body)
    # 元组解包（`ax, ay, az = 0.0, 0.0, 0.0`）先展开成独立绑定。
    from source_contract import expand_tuple_assign
    body = expand_tuple_assign(body)
    # 展平坐标（by*S + bx*blockDim.x + tx）先规范成 cuda.grid(1)；失败时保持原样交回下面的门。
    body, flat_stride = normalize_flat_grid(body)
    # 网格步长循环（`range(cuda.grid(n), STOP, cuda.gridsize(n))` 与等价的 `while` 形态）
    # 先折成一次覆盖。
    body = normalize_grid_stride(body)
    body = normalize_grid(body)
    grid = body.pop(0)
    if not isinstance(grid, ast.Assign) or len(grid.targets) != 1 \
            or not isinstance(grid.value, ast.Call) or _call_name(grid.value.func) != 'cuda.grid' \
            or grid.value.keywords or len(grid.value.args) != 1:
        raise Unsupported('首条计算语句必须是 cuda.grid(n) 赋值')
    dim = grid.value.args[0]
    if not isinstance(dim, ast.Constant) or type(dim.value) is not int:
        raise Unsupported('cuda.grid 维数必须是整数常量')
    target = grid.targets[0]
    indices = target.elts if isinstance(target, ast.Tuple) else [target]
    if not all(isinstance(x, ast.Name) for x in indices):
        raise Unsupported('grid 必须赋值给独立索引变量')
    spec.grid_dims = dim.value
    spec.idx_vars = [x.id for x in indices]
    if flat_stride is not None:
        spec.flat_stride = ast.unparse(flat_stride)

    # 规范化等价的完整边界控制流；轴与shape对应关系仍由语义层验证。
    from guards import extract_guard
    # 末尾的 cuda.syncthreads() 只是屏障；先去掉，guard 才能被完整包围识别。
    spec.guard, body, spec.strict_early_return_guard = extract_guard(
        strip_trailing_sync(body), spec.params, spec.idx_vars)
    # 元素级掩码（`if solid_mask[i, j]:`）：条件由本元素自己的输入决定，符合范围判据
    # （每元素独立、不与其他线程交换数据），从边界 guard 里拆出来单独发射。
    spec.guard, spec.element_masks = split_element_masks(spec.guard, spec.params)
    # 20261025-156 批次：两个“部分域条件写”结构的独立资格门（失败即不改写，交回旧路径）。
    from frontend import module_scalars
    _consts = {name: value for name, value in module_scalars(tree, fn).items()
               if type(value) in (int, float)}
    from partial_write import (bind_axis_equality, const_dim_bounds, hoist_predicate_mask,
                               substitute_axis)
    # S1：体内唯一的 `if <轴算术谓词>:` → 元素掩码（掩码通道强制 caller-output）。
    # S2：guard 里的 `轴 == 常量` → 该轴绑定为常量下标。
    # 20261025-158（CHK-481 ②）：两模式同时命中 ⇒ 两处都不采纳、交回旧路径，
    # 避免 body 改写与 partial_const_dims 互相覆盖。
    _hoisted = hoist_predicate_mask(body, spec, _consts)
    _axis_bind = bind_axis_equality(spec, _consts)
    if _hoisted is not None and _axis_bind is not None:
        _hoisted = _axis_bind = None
    if _hoisted is not None:
        _inner, _cond, _const_dims = _hoisted
        spec.element_masks = tuple(spec.element_masks) + (_cond,)
        body = _inner
        spec.partial_predicate = True
        spec.partial_const_dims = tuple(_const_dims)
    if _axis_bind is not None:
        _guard_text, _axis, _value = _axis_bind
        _body = substitute_axis(body, _axis, _value)
        # 20261025-158（CHK-481 ①）：`const_dim_bounds` 返回 None 是安全门拒绝，
        # 与 S1 同口径整结构拒绝；不得折成 `()`（那会无下界校验地接受越界写）。
        # 20261025-159（CHK-484 选路 B）：界在**替换前**的 body 上推，此时被消去轴仍是轴名
        # （该维是真实轴、不推界）；循环外的常量下标等不可推定形态一律拒绝。
        _dims = None if _body is None else const_dim_bounds(body, _consts, spec.idx_vars)
        if _dims is not None:
            spec.guard = _guard_text
            body = _body
            spec.idx_vars = [name for name in spec.idx_vars if name != _axis]
            spec.grid_dims = len(spec.idx_vars)
            spec.partial_axis = True
            spec.partial_const_dims = tuple(_dims)
    # 20261025-153：rank-1 张量的标准行主序展平寻址（独立新模式，旧路径不受影响）。
    if not spec.element_masks:
        from flat_linear import recognize as _recognize_flat_linear
        _flat = _recognize_flat_linear(body, spec)
        if _flat is not None:
            spec.flat_linear = _flat
            spec.pattern = 'flat_linear'
            spec.store_target = _flat['output']
            spec.store_index = ()
            return spec
    # 基址 + 常量循环写不同输出元素：改写为展平逐元素（A 片，路线 b）。
    body = rewrite_flat_base(body, spec)
    # 常量范围的"按方向/通道写回"循环先展开，复用多写回与常量下标能力。
    body = unroll_constant_loops(body)
    body = _lift_output_loop(body, spec)
    body = _rewrite_softmax_roundtrip(body, spec)
    # 语义层（guard 校验）需要看到模块级标量常量：`if i < POPULATION` 的域是常量。
    spec.const_bindings = dict(scalar_bindings)
    # 20261025-144：常量上界 guard 窄门只认"恰好绑定一次的模块级 int 字面量"，
    # 与上面的宽松 const_bindings（允许表达式/重绑定）分开，互不影响既有路径。
    from frontend import strict_int_constants
    spec.guard_int_constants = strict_int_constants(tree)
    constant_guard = False
    if spec.guard and spec.guard_int_constants:
        guard_node = ast.parse(spec.guard, mode='eval').body
        guard_terms = (guard_node.values
                       if isinstance(guard_node, ast.BoolOp) and isinstance(guard_node.op, ast.And)
                       else [guard_node])
        for term in guard_terms:
            if isinstance(term, ast.Compare) and len(term.ops) == 1 \
                    and isinstance(term.ops[0], ast.Lt) and isinstance(term.left, ast.Name) \
                    and isinstance(term.comparators[0], ast.Name) \
                    and term.comparators[0].id in spec.guard_int_constants:
                constant_guard = True
    spec.constant_guard_used = constant_guard
    if not any(isinstance(statement, ast.For) for statement in body):
        spec.compute = capture_pointwise(body, spec.params, spec.idx_vars, scalar_bindings,
                                         spec.tensor_dtypes)
        spec.store_target = spec.compute.output
        spec.store_index = spec.compute.output_axes
        if spec.flat_stride:
            from semantic import validate_flat_map
            validate_flat_map(spec)
        return spec
    nested = any(isinstance(child, ast.For) for statement in body if isinstance(statement, ast.For)
                 for child in ast.walk(statement) if child is not statement)
    if nested:
        from scoped_capture import capture_scoped
        spec.program = capture_scoped(body, spec.params, spec.idx_vars,
                                      scalar_bindings, spec.tensor_dtypes)
        spec.store_target, spec.store_index = spec.program.output, spec.program.output_axes
        return spec
    from capture import capture_program
    from multi_capture import capture_multi, is_update
    def _update_count(statement):
        items = statement.body
        # 单个 `if cond:` 包裹的循环体（`if tid != j:`）也要看内部。
        if len(items) == 1 and isinstance(items[0], ast.If) and not items[0].orelse:
            items = items[0].body
        return sum(is_update(item) for item in items)

    multiple = any(_update_count(statement) > 1
                   for statement in body if isinstance(statement, ast.For))
    if multiple:
        program = capture_multi(body, spec.params, spec.idx_vars, scalar_bindings,
                                spec.tensor_dtypes)
    else:
        from multi_capture import unmask_loop_conditions
        program = capture_program(unmask_loop_conditions(body), spec.params, spec.idx_vars,
                                  scalar_bindings, spec.tensor_dtypes)
    spec.program = program
    spec.store_target, spec.store_index = program.output, program.output_axes
    for stage in program.stages:
        r = stage.reduction
        spec.loops.append(LoopSpec(r.axis, r.stop, r.accumulator, reads(stage.expression),
                                  start=r.start, step=r.step, init_value=r.init.value))
    # 旧单归约路径是新计算描述的特例，保留既有接口及GEMM调度。
    # `S[i,j] = acc * scale`：归约结果乘一个标量值（attention 的 QK^T 标准形态）。
    # 只认"结果 × 标量"，且 scale 记进 spec，由各 lowering 在写回阶段乘上去。
    tail, post_scale, post_bias, post_act = program.expression, None, None, None
    # 最外层激活：`max(acc [+ bias], 0)`——ReLU 的标准写法。只认与**常量**的比较，
    # 这样激活阈值与张量无关，可以在 tile 内用 `T.max` 逐元素套用。
    if tail.op in ('max', 'min') and len(tail.args) == 2:
        for core, bound in ((tail.args[0], tail.args[1]), (tail.args[1], tail.args[0])):
            if bound.op == 'const' and core.op in ('result', 'add', 'mul'):
                post_act, tail = (tail.op, bound), core
                break
    # 后处理项：从外层往里剥 `核心 ± 项`，项可以是张量读取（一维/二维）或常量。
    # 得到的是"从外到内"的顺序，应用时要反过来（内层先算）。
    post_terms = []
    while tail.op in ('add', 'sub') and len(tail.args) == 2:
        if tail.op == 'add':
            found = False
            for core, operand in ((tail.args[0], tail.args[1]), (tail.args[1], tail.args[0])):
                if operand.op in ('read', 'const') and core.op != 'read':
                    post_terms.append(('+', operand)); tail, found = core, True
                    break
            if not found:
                break
        else:
            if tail.args[1].op in ('read', 'const') and tail.args[0].op != 'read':
                post_terms.append(('-', tail.args[1])); tail = tail.args[0]
            else:
                break
    if post_bias is None and tail.op == 'mul' and len(tail.args) == 2:
        for left, right in ((tail.args[0], tail.args[1]), (tail.args[1], tail.args[0])):
            if left.op == 'result' and right.op in ('scalar', 'const'):
                tail, post_scale = left, right
                break
    if len(program.stages) == 1 and tail.op == 'result' and tail.value == 0:
        spec.gemm_scale = post_scale
        spec.gemm_post = tuple(reversed(post_terms))
        spec.gemm_activation = post_act
        stage = program.stages[0]
        from program import walk
        needs_shape = any(node.op == 'shape' for node in walk(stage.expression))
        if not needs_shape:
            spec.compute = Compute(stage.expression, program.output, program.output_axes,
                                   program.bindings + stage.bindings, stage.reduction)
        loop = next(node for node in body if isinstance(node, ast.For))
        update = loop.body[-1]
        # 条件形式的归约更新（`if A > acc: acc = A`）没有 .value，记整条语句。
        spec.loops[0].update_expr = ast.unparse(
            update.value if isinstance(update, (ast.Assign, ast.AugAssign)) else update)
    return spec


def _split_product(node):
    """把 A[i,k] * B[k,j] 拆成 [('A', ('i','k')), ('B', ('k','j'))]"""
    terms = []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        for side in (node.left, node.right):
            terms.extend(_split_product(side))
    elif isinstance(node, ast.Subscript):
        terms.append(_tensor_access(node))
    else:
        raise Unsupported('乘积中包含未支持的表达式，不能丢弃: ' + ast.unparse(node))
    return terms


# ══════════════════════════ ② 模式层 ══════════════════════════

def _shape_axis(node):
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
            and node.value.attr == 'shape' and isinstance(node.value.value, ast.Name) \
            and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int \
            and node.slice.value in (0, 1):
        return node.value.value.id, node.slice.value
    return None


def detect_pattern(spec: KernelSpec) -> KernelSpec:
    """以输出自由轴及归约轴识别乘加，再校验当前 TileLang 模板的契约。"""
    if spec.pattern in ('flat_elementwise', 'flat_linear'):
        return spec
    spec.pattern = ''
    spec.k_constant = None
    spec.extent_expression = None
    if spec.program is not None:
        from program import validate_program
        validate_program(spec)
        if spec.compute is None or spec.compute.reduction is None:
            return spec
        from selection import supports_single_reduction
        if not supports_single_reduction(spec):
            return spec
    if spec.compute is not None:
        if spec.compute.reduction is None:
            return validate_map(spec)
        # 先验证通用归约的域、绑定与访问，再选择优化；不捕获异常后盲目 fallback。
        validate_sum(spec)
        expression = spec.compute.expression
        reduction = spec.compute.reduction
        # `S[i,j] = acc * scale`：attention 的 QK^T 几乎总是带缩放。
        # 只接受"归约结果 × 一个标量值（常量或签名标量）"；GEMM 之后逐元素乘。
        # 这与把 scale 折进每个乘加项在浮点舍入上不同，属显式记录的语义选择。
        scale = None
        core = expression
        if reduction.op == 'sum' and len(spec.compute.output_axes) == 2 \
                and expression.op == 'mul' and len(expression.args) == 2:
            for left, right in ((expression.args[0], expression.args[1]),
                                (expression.args[1], expression.args[0])):
                if left.op == 'result' and right.op in ('scalar', 'const'):
                    core, scale = left, right
                    break
        expression = core
        if scale is not None:
            # parse_source 已经从 program.expression 里认出缩放；这里只在本地又找到时覆盖，
            # 否则会把已记录的值清空（曾导致 qk_score 生成物丢掉 * scale）。
            spec.gemm_scale = scale
        if reduction.op != 'sum' or len(spec.compute.output_axes) != 2 \
                or expression.op != 'mul' or len(expression.args) != 2 \
                or any(arg.op != 'read' for arg in expression.args):
            return spec
        terms = reads(expression)
        k = reduction.axis
        free_axes = []
        for name, indices in terms:
            if len(indices) != 2 or indices.count(k) != 1:
                return spec
            free_axes.append(indices[1 - indices.index(k)])
        if set(free_axes) != set(spec.compute.output_axes) \
                or len({name for name, _ in terms}) != 2:
            return spec
    if spec.grid_dims != 2 or len(spec.idx_vars) != 2 or len(set(spec.idx_vars)) != 2:
        raise Unsupported('GEMM 需要 cuda.grid(2) 的两个独立自由轴')
    if len(spec.loops) != 1 or len(spec.store_index) != 2:
        raise Unsupported('GEMM 需要单个归约循环和二维写回')
    lp = spec.loops[0]
    m, n = spec.store_index
    k = lp.var
    if len({m, n, k, lp.acc_var}) != 4 or set((m, n)) != set(spec.idx_vars):
        raise Unsupported('输出两轴、归约轴、累加器必须互异；输出轴必须来自 grid')
    if len(lp.acc_terms) != 2 or lp.op != '*' or lp.update_op != '+' or lp.init_value != 0:
        raise Unsupported('当前 GEMM 只支持零初始化的两项乘加归约')
    # GEMM是共享表达式树上的特化：不能将sum任意表达式都当成矩阵乘。
    expr = spec.compute.expression if spec.compute is not None else None
    if expr is None or expr.op != 'mul' or any(arg.op != 'read' for arg in expr.args) \
            or reads(expr) != lp.acc_terms:
        raise Unsupported('GEMM特化只支持两个纯张量读取相乘，其他表达式不套模板')
    operands = {}
    for name, idx in lp.acc_terms:
        if len(idx) != 2 or idx.count(k) != 1:
            raise Unsupported('每个输入必须是二维读取，归约轴恰出现一次')
        free = idx[1 - idx.index(k)]
        if free not in (m, n) or free in operands:
            raise Unsupported('两个输入必须分别连接输出行轴与列轴，不能共享同一个自由轴')
        operands[free] = (name, idx)
    a, ai = operands[m]
    b, bi = operands[n]
    # 签名标量（含无签名时推断的）不参与"三张量"判定：`qk_score(Q,K,S,scale)` 仍是 GEMM。
    scalars = set(getattr(spec, 'scalar_params', {}) or {})
    tensor_params = [name for name in spec.params if name not in scalars]
    # `nn.Linear` 形态：GEMM 写回后加一维偏置 `acc + bias[j]`。偏置只能是输出列轴上的向量。
    # 后处理项：每项可以是 (m,)/n 轴一维张量、(m,n) 二维张量或常量。
    post = tuple(getattr(spec, 'gemm_post', ()) or ())
    expected = {a, b, spec.store_target}
    resolved = []
    for sign, operand in post:
        if operand.op == 'const':
            resolved.append((sign, None, 'const', operand))
            continue
        name, indices = operand.value
        shape_kind = tuple(indices)
        if name in expected:
            raise Unsupported('GEMM 后处理张量不能与输入或输出同名: ' + name)
        if shape_kind == (n,):
            resolved.append((sign, name, 'n', operand))
        elif shape_kind == (m,):
            resolved.append((sign, name, 'm', operand))
        elif shape_kind == (m, n):
            resolved.append((sign, name, 'mn', operand))
        else:
            raise Unsupported('GEMM 后处理张量必须是 [j]、[i] 或 [i, j] 形态: ' + name)
        expected = expected | {name}
    spec.gemm_post_resolved = tuple(resolved)
    if len(tensor_params) != len(expected) or len({a, b, spec.store_target}) != 3             or set(tensor_params) != expected:
        raise Unsupported('GEMM 形参必须是：两个输入、一个输出，外加后处理张量')
    if set(spec.params) & {m, n, k, lp.acc_var, 'cuda', 'range'} \
            or {m, n, k, lp.acc_var} & {'cuda', 'range'}:
        raise Unsupported('张量参数不能被索引/累加器覆盖，也不能遮蔽 cuda 或 range')
    reserved = {'T', 'tilelang', '_PASS_CFG', 'block_M', 'block_N',
                'block_K', 'num_stages', 'threads', 'main', 'bx', 'by', 'ko',
                'A_shared', 'B_shared', 'C_local', 'any', 'type', 'int', 'ValueError'}
    if set(spec.params) & reserved or spec.name in reserved | {'cuda', 'range'}:
        raise Unsupported('名称与 TileLang 模板保留标识符冲突，请重命名')
    if lp.start != '0' or lp.step != '1':
        raise Unsupported('当前只支持从 0 开始、步长为 1 的完整归约')
    stop = ast.parse(lp.extent, mode='eval').body
    if _shape_axis(stop) not in ((a, ai.index(k)), (b, bi.index(k))):
        if isinstance(stop, ast.Constant) and type(stop.value) is int and stop.value > 0:
            spec.k_constant = stop.value
        else:
            raise Unsupported('归约上界必须是输入的 K 轴 shape 或正整数常量；未知符号不能猜测')

    # shape -> 语义轴；只有完整边界 guard 才可由 tile 拷贝的越界谓词替代。
    shapes = {(name, pos): axis for name, idx in lp.acc_terms
              for pos, axis in enumerate(idx)}
    shapes.update({(spec.store_target, 0): m, (spec.store_target, 1): n})
    spec.guard_kind = 'none'
    if spec.guard:
        guard = ast.parse(spec.guard, mode='eval').body
        if not isinstance(guard, ast.BoolOp) or not isinstance(guard.op, ast.And) \
                or len(guard.values) != 2:
            raise Unsupported('只支持两个输出轴的完整边界 guard，不支持算法掩码')
        covered = []
        for test in guard.values:
            if not isinstance(test, ast.Compare) or len(test.ops) != 1 \
                    or not isinstance(test.ops[0], ast.Lt) or not isinstance(test.left, ast.Name) \
                    or shapes.get(_shape_axis(test.comparators[0])) != test.left.id:
                raise Unsupported('guard 必须是自由轴 < 对应轴的完整 shape')
            covered.append(test.left.id)
        if set(covered) != {m, n}:
            raise Unsupported('guard 必须恰好覆盖两个输出自由轴')
        spec.guard_kind = 'bounds'
    spec.lhs, spec.rhs = a, b
    spec.transpose_A, spec.transpose_B = ai.index(k) == 0, bi.index(k) == 1
    spec.axis_roles = {m: 'M', n: 'N', k: 'K'}
    spec.pattern = 'gemm'
    return spec


# ══════════════════════════ ③ 提升层 ══════════════════════════

TILE = dict(bM=128, bN=128, bK=32, stages=3, threads=128)


def lift_gemm(spec: KernelSpec) -> str:
    """点积模式 → tile 级 GEMM

    这是【语义提升】：原代码是"每线程算一个元素"，要补上原代码没有的三样东西：
      · shared memory（原来的直接从 global 读）
      · tile 分块（原来是按元素划分线程）
      · 流水线（原来没有这个层次）
    """
    detect_pattern(spec)
    A_name, B_name, C_name = spec.lhs, spec.rhs, spec.store_target
    ta, tb = spec.transpose_A, spec.transpose_B
    a_shape = '(_tl_K, _tl_M)' if ta else '(_tl_M, _tl_K)'
    b_shape = '(_tl_N, _tl_K)' if tb else '(_tl_K, _tl_N)'
    a_tile = '(block_K, block_M)' if ta else '(block_M, block_K)'
    b_tile = '(block_N, block_K)' if tb else '(block_K, block_N)'
    a_offset = 'ko * block_K, by * block_M' if ta else 'by * block_M, ko * block_K'
    b_offset = 'bx * block_N, ko * block_K' if tb else 'ko * block_K, bx * block_N'
    post = tuple(getattr(spec, 'gemm_post_resolved', ()) or ())
    post_shapes = {'n': '(_tl_N,)', 'm': '(_tl_M,)', 'mn': '(_tl_M, _tl_N)'}
    shapes = {A_name: a_shape, B_name: b_shape, C_name: '(_tl_M, _tl_N)'}
    for _sign, _name, _kind, _operand in post:
        if _kind != 'const':
            shapes[_name] = post_shapes[_kind]
    # 签名标量（如 attention 的 scale）按标量参数发射，不占 shape 表。
    scalars = dict(getattr(spec, 'scalar_params', {}) or {})
    parameters = ',\n             '.join(
        f'{name}: T.{scalars[name]}' if name in scalars
        else f'{name}: T.Tensor({shapes[name]}, "float32")'
        for name in spec.params)
    t = TILE
    out_pos = spec.params.index(C_name)  # main 保留源形参顺序，输出索引不再错位
    k_check = '' if spec.k_constant is None else (
        f'    if _tl_K != {spec.k_constant}:\n'
        f'        raise ValueError("source reduction requires K == {spec.k_constant}")\n')
    # `S = (A@B) * scale`：归约结果乘标量。按"GEMM 之后逐元素乘"实现，
    # 与把 scale 折进每个乘加项的浮点舍入顺序不同——这是显式选择，不是等价改写。
    scale = getattr(spec, 'gemm_scale', None)
    scale_stmt = ''
    if scale is not None:
        rendered = (f'T.cast({literal(scale.value)}, "float32")' if scale.op == 'const'
                    else f'T.cast({scale.value}, "float32")')
        scale_stmt = (f'            for _tl_si, _tl_sj in T.Parallel(block_M, block_N):\n'
                      f'                C_local[_tl_si, _tl_sj] = C_local[_tl_si, _tl_sj] * {rendered}\n')
    if post:
        # 后处理项在同一个 T.Parallel 内按解析顺序逐项套用，全部是 tile 级逐元素。
        indexes = {'n': 'bx * block_N + _tl_sj', 'm': 'by * block_M + _tl_si',
                   'mn': 'by * block_M + _tl_si, bx * block_N + _tl_sj'}
        lines = []
        for sign, name, kind, operand in post:
            if kind == 'const':
                from lowering import literal as _lit2
                term = f'T.cast({_lit2(operand.value)}, "float32")'
            else:
                term = f'{name}[{indexes[kind]}]'
            lines.append('                C_local[_tl_si, _tl_sj] = C_local[_tl_si, _tl_sj] '
                         + sign + ' ' + term + '\n')
        scale_stmt += ('            for _tl_si, _tl_sj in T.Parallel(block_M, block_N):\n'
                       + ''.join(lines))
    activation = getattr(spec, 'gemm_activation', None)
    if activation is not None:
        # 激活在 tile 内逐元素套用：`C_local = T.max(C_local, 阈值)`，仍是 T.Parallel。
        kind, bound = activation
        from lowering import literal as _lit
        bound_text = f'T.cast({_lit(bound.value)}, "float32")'
        scale_stmt += (f'            for _tl_si, _tl_sj in T.Parallel(block_M, block_N):\n'
                       f'                C_local[_tl_si, _tl_sj] = T.{kind}'
                       f'(C_local[_tl_si, _tl_sj], {bound_text})\n')
    return f'''# 由 numba2tilelang 自动生成 —— 点积模式已提升为 tile 级 GEMM
# 原代码：每线程算一个元素（线程级）
# 生成后：tile 级 GEMM（shared + tiling + pipeline），tile 尺寸由默认值给出
# 契约：连续二维 float32 张量；源 launch 完整且唯一覆盖输出；允许 TF32 近似。
# 调用 kernel 时，输入按源形参顺序传入（去掉自动分配的输出）。
import tilelang
import tilelang.language as T


# 关掉 warp specialization：它依赖 Hopper(sm_90a) 的寄存器重配置特性，
# 目标是 Blackwell(sm_120) 等新架构时会编译失败
_PASS_CFG = {{"tl.disable_warp_specialized": True}}


@tilelang.jit(out_idx=[{out_pos}], pass_configs=_PASS_CFG)
def {spec.name}(_tl_M, _tl_N, _tl_K,
              block_M={t['bM']}, block_N={t['bN']}, block_K={t['bK']},
              num_stages={t['stages']}, threads={t['threads']}):
    if any(type(d) is not int or d <= 0 for d in (_tl_M, _tl_N, _tl_K)):
        raise ValueError("M, N, K must be positive integers")
{k_check}    @T.prim_func
    def main({parameters}):
        with T.Kernel(T.ceildiv(_tl_N, block_N), T.ceildiv(_tl_M, block_M),
                      threads=threads) as (bx, by):
            A_shared = T.alloc_shared({a_tile}, "float32")
            B_shared = T.alloc_shared({b_tile}, "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(_tl_K, block_K), num_stages=num_stages):
                T.copy({A_name}[{a_offset}], A_shared)
                T.copy({B_name}[{b_offset}], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_A={ta}, transpose_B={tb})

{scale_stmt}            T.copy(C_local, {C_name}[by * block_M, bx * block_N])

    return main
'''


# ══════════════════════════ ④ 生成层（入口） ══════════════════════════

def _module_bound_names(body):
    """模块顶层语句引入的绑定名（不解包函数体，避免把局部赋值误当模块绑定）。"""
    import ast as _ast
    names = set()

    def collect(target):
        if isinstance(target, _ast.Name):
            names.add(target.id)
        elif isinstance(target, (_ast.Tuple, _ast.List)):
            for element in target.elts:
                collect(element)

    def walk(statements):
        for statement in statements:
            if isinstance(statement, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                names.add(statement.name)          # 函数体内局部赋值不算模块绑定
                continue
            if isinstance(statement, _ast.Assign):
                for target in statement.targets:
                    collect(target)
            elif isinstance(statement, (_ast.AnnAssign, _ast.AugAssign)):
                collect(statement.target)
            elif isinstance(statement, (_ast.For, _ast.AsyncFor)):
                collect(statement.target)
            elif isinstance(statement, (_ast.Import, _ast.ImportFrom)):
                for alias in statement.names:
                    names.add((alias.asname or alias.name).split('.')[0])
            # 模块级复合语句（if/for/while/with/try）里的赋值同样产生模块绑定。
            for field in ('body', 'orelse', 'finalbody', 'handlers'):
                nested = getattr(statement, field, None)
                if not nested:
                    continue
                if field == 'handlers':
                    for handler in nested:
                        walk(handler.body)
                else:
                    walk(nested)

    walk(body)
    return names


def unroll_constant_loops(body, limit=32):
    """把 `for k in range(<常量>): <写回>` 展开成常量条写回。

    这类循环（LBM 的"按方向逐个写回"）没有归约，只是把同一个并行域的
    9 条写回写成循环。展开后直接复用"多写回 + 常量下标"的既有能力，
    不需要新的发射模板。`limit` 防止代码量爆炸。
    """
    import ast as _ast
    import copy as _copy

    def constant_extent(node):
        if not (isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
                and node.func.id == 'range' and len(node.args) == 1):
            return None
        argument = node.args[0]
        if isinstance(argument, _ast.Constant) and type(argument.value) is int:
            return argument.value
        return None

    class Substitute(_ast.NodeTransformer):
        def __init__(self, name, value):
            self.name, self.value = name, value

        def visit_Name(self, node):
            if node.id == self.name and isinstance(node.ctx, _ast.Load):
                return _ast.copy_location(_ast.Constant(value=self.value), node)
            return node

    out = []
    for statement in body:
        extent = constant_extent(statement.iter) if isinstance(statement, _ast.For) else None
        if extent is None or extent > limit or not isinstance(statement.target, _ast.Name):
            out.append(statement)
            continue
        # 只展开"纯写回"循环：含累加（归约）或不是以张量写回结束的保持原样，
        # 否则会把 `for k in range(4): acc += ...` 这类归约循环拆坏。
        if any(isinstance(item, _ast.AugAssign) for item in statement.body):
            out.append(statement)
            continue
        if not (statement.body and isinstance(statement.body[-1], _ast.Assign)
                and statement.body[-1].targets
                and isinstance(statement.body[-1].targets[0], _ast.Subscript)):
            out.append(statement)
            continue
        name = statement.target.id
        # 循环体里不能重新赋值循环变量，否则展开不成立。
        if any(isinstance(item, _ast.Name) and item.id == name
               and isinstance(item.ctx, _ast.Store)
               for item in _ast.walk(statement) if item is not statement.target):
            out.append(statement)
            continue
        for value in range(extent):
            for item in statement.body:
                out.append(_ast.fix_missing_locations(
                    Substitute(name, value).visit(_copy.deepcopy(item))))
    return out


def rewrite_flat_base(body, spec):
    """把"基址 + 常量循环写不同输出元素"改写成**展平逐元素**写回（A 片，路线 b）。

    接受形态（全部满足才改写）：
        base = <grid 轴> * L            # L 正整数常量，单次赋值，只作下标基址
        for j in range(L):              # 常量长度
            dst[base + j] = <expr，其中所有张量访问都是 X[base + j]>
        guard: <grid 轴> < T.shape[d] // L
    改写后：删除 base 与循环，所有 `X[base + j]` → `X[<grid 轴>]`，
    guard → `<grid 轴> < T.shape[d]`（域 = 整个缓冲区）。
    覆盖证明：p = i*L + j（j∈[0,L)、i∈[0,N)、N = shape[d]//L）；源码 guard 使
    i < N，故 p 恰枚举 [0, N*L)。本片额外要求 **shape[d] % L == 0**（构造期检查），
    此时 N*L == _tl_size，flat 枚举 p < _tl_size 与源码写入集合逐一对应。
    """
    import ast as _ast
    import copy as _copy

    axes = set(spec.idx_vars)
    for position, statement in enumerate(body):
        if not isinstance(statement, _ast.For):
            continue
        if not (isinstance(statement.target, _ast.Name) and not statement.orelse
                and isinstance(statement.iter, _ast.Call)
                and isinstance(statement.iter.func, _ast.Name)
                and statement.iter.func.id == 'range'
                and len(statement.iter.args) == 1
                and isinstance(statement.iter.args[0], (_ast.Constant, _ast.Name))):
            continue
        extent_node = statement.iter.args[0]
        if isinstance(extent_node, _ast.Constant):
            if type(extent_node.value) is not int or extent_node.value <= 0:
                continue
            length, length_name, inferred_length = extent_node.value, None, False
        else:
            # C2：运行期 L 只接受签名 int64 标量。
            name = extent_node.id
            if (spec.scalar_params or {}).get(name) != 'int64':
                continue
            # R4：运行期 L 必须来自静态签名，推断类型不算契约。
            # 注意：这里**只记录**，不能立刻拒绝——本函数在 pattern 判定之前运行，
            # 普通归约循环（`for j in range(n)`）也会经过这里；只有在确认是基址形态后才拒绝。
            inferred_length = name in (getattr(spec, 'inferred_scalars', ()) or ())
            length, length_name = name, name
        loop_name = statement.target.id
        # 前面必须是若干纯标量绑定，其中恰好一条 `base = 轴 * L`。
        base_name = None
        base_axis = None
        prefix = body[:position]
        for item in prefix:
            if not (isinstance(item, _ast.Assign) and len(item.targets) == 1
                    and isinstance(item.targets[0], _ast.Name)):
                return body          # 形态不符：交回原路径，由既有判据拒绝
            value = item.value
            if (isinstance(value, _ast.BinOp) and isinstance(value.op, _ast.Mult)
                    and isinstance(value.left, _ast.Name) and value.left.id in axes
                    and ((isinstance(value.right, _ast.Constant)
                          and type(value.right.value) is int and value.right.value == length)
                         or (isinstance(value.right, _ast.Name)
                             and value.right.id == length))):
                if base_name is not None:
                    return body
                base_name, base_axis = item.targets[0].id, value.left.id
        if base_name is None:
            continue
        if inferred_length:
            raise Unsupported('索引基址的运行期 L 需要显式 int64 类型来源（静态签名或 scalar_types 声明；推断不算契约）：%s'
                              % length_name)
        from program import RUNTIME_RESERVED_NAMES
        if length_name in RUNTIME_RESERVED_NAMES:
            raise Unsupported('运行期 L 与模板局部名冲突: ' + length_name)
        # guard 必须是 `<轴> < T.shape[d] // L`。
        guard = None
        if spec.guard:
            node = _ast.parse(spec.guard, mode='eval').body
            if (isinstance(node, _ast.Compare) and len(node.ops) == 1
                    and isinstance(node.ops[0], _ast.Lt)
                    and isinstance(node.left, _ast.Name) and node.left.id in axes
                    and isinstance(node.comparators[0], _ast.BinOp)
                    and isinstance(node.comparators[0].op, _ast.FloorDiv)
                    and isinstance(node.comparators[0].left, _ast.Subscript)
                    and isinstance(node.comparators[0].left.value, _ast.Attribute)
                    and node.comparators[0].left.value.attr == 'shape'
                    and isinstance(node.comparators[0].left.slice, _ast.Constant)
                    and type(node.comparators[0].left.slice.value) is int
                    and ((isinstance(node.comparators[0].right, _ast.Constant)
                          and node.comparators[0].right.value == length)
                         or (isinstance(node.comparators[0].right, _ast.Name)
                             and node.comparators[0].right.id == length))):
                shape_node = node.comparators[0].left
                guard = (shape_node.value.value.id, shape_node.slice.value, node.left.id)
        if guard is None:
            continue
        tensor_name, dim, axis = guard
        # 防御性不变量：base 的粗轴必须与 guard 比对的轴是同一个。
        if base_axis != axis:
            raise Unsupported('索引基址改写的 base 轴与 guard 轴不一致')
        # —— 从这一行起视为“本形态已认领”：任何不支持的使用都必须**明确拒绝**，
        # 不能 `continue` 回落到其它路径再接受（R1）。
        if getattr(spec, 'element_masks', ()) or getattr(spec, 'element_mask_expressions', ()):
            raise Unsupported('索引基址改写暂不支持元素掩码组合')

        def pure_scalar(item):
            if not (isinstance(item, _ast.Assign) and len(item.targets) == 1
                    and isinstance(item.targets[0], _ast.Name)):
                return False
            return not any(isinstance(node, _ast.Subscript) for node in _ast.walk(item.value))                 and not _mentions(item.value, {axis, base_name, loop_name})

        for item in prefix:
            if item.targets[0].id == base_name:
                continue
            if not pure_scalar(item):
                raise Unsupported('索引基址改写的前缀只允许与轴/基址无关的纯标量绑定')
        for item in body[position + 1:]:
            if not pure_scalar(item):
                raise Unsupported('索引基址改写后不允许与轴/基址有关的后续语句或张量写回')

        def is_base_index(node):
            return (isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Add)
                    and isinstance(node.left, _ast.Name) and node.left.id == base_name
                    and isinstance(node.right, _ast.Name) and node.right.id == loop_name)

        class Rewrite(_ast.NodeTransformer):
            def visit_Subscript(self, node):
                if isinstance(node.value, _ast.Name) and is_base_index(node.slice):
                    return _ast.copy_location(
                        _ast.Subscript(value=node.value,
                                       slice=_ast.Name(id=axis, ctx=_ast.Load()),
                                       ctx=node.ctx), node)
                node = self.generic_visit(node)
                if _mentions(node, {base_name, loop_name}):
                    raise Unsupported('A 片只支持 `张量[base + 循环轴]` 一种访问形态')
                return node

            def visit_Name(self, node):
                if node.id in (base_name, loop_name, axis):
                    raise Unsupported('索引基址改写：轴/基址/循环轴只能出现在 `X[base + 循环轴]` 里')
                return node

        def rewrite(node, inside_loop):
            return Rewrite().visit(node)

        # guard 证明的是**哪个张量**的长度，就必须是循环体唯一写回目标：
        # `if i < src.shape[0]//L: dst[...]` 证到的是 src，不能拿来覆盖 dst 的写域。
        written = {item.targets[0].value.id for item in statement.body
                   if isinstance(item, _ast.Assign) and len(item.targets) == 1
                   and isinstance(item.targets[0], _ast.Subscript)
                   and isinstance(item.targets[0].value, _ast.Name)}
        if written != {tensor_name}:
            raise Unsupported('索引基址改写要求 guard 张量就是唯一的写回目标')
        rewritten_body = []
        for item in statement.body:   # 认领后不再吞掉 Unsupported（R1）
            if not (isinstance(item, _ast.Assign) and len(item.targets) == 1
                    and isinstance(item.targets[0], _ast.Subscript)):
                raise Unsupported('索引基址改写的循环体只允许直接张量写回')
            item = _copy.deepcopy(item)
            item.targets[0] = rewrite(item.targets[0], True)
            item.value = rewrite(item.value, True)
            rewritten_body.append(_ast.fix_missing_locations(item))
        # 前缀里 base 之外的绑定保留；删除 base 赋值。
        kept = [item for item in prefix if item.targets[0].id != base_name]
        spec.guard = '%s < %s.shape[%d]' % (axis, tensor_name, dim)
        if length_name is not None:
            # 运行期 L：不能构造期校验整除；写回条件按前缀 (shape//L)*L 截断。
            spec.flat_base_runtime = (tensor_name, dim, length_name)
        else:
            spec.divisible_extent = (tensor_name, dim, length)
        return kept + rewritten_body + body[position + 1:]
    return body


def _mentions(node, names):
    return any(isinstance(item, ast.Name) and item.id in names
               for item in ast.walk(node))


def split_element_masks(guard_text, params):
    """把 guard 里的**元素级张量条件**拆出来，返回 (边界条件文本, [掩码表达式文本])。

    `if i < nx and j < ny and solid_mask[i, j]:` → 前两项留作边界 guard，
    第三项是元素掩码：它只读本元素的输入，不与其他线程交换数据，属于范围之内。
    判据：条件里有 `张量[下标]` 形式的读取（`X.shape[k]` 是 Attribute，不算）。
    """
    import ast as _ast
    if not guard_text:
        return guard_text, ()
    node = _ast.parse(guard_text, mode='eval').body
    terms = node.values if isinstance(node, _ast.BoolOp) and isinstance(node.op, _ast.And) else [node]
    bounds, masks = [], []
    for term in terms:
        reads_tensor = any(
            isinstance(item, _ast.Subscript) and isinstance(item.value, _ast.Name)
            and item.value.id in set(params)
            for item in _ast.walk(term))
        (masks if reads_tensor else bounds).append(_ast.unparse(term))
    if len(bounds) == len(terms):
        return guard_text, ()
    return (' and '.join(bounds) if bounds else ''), tuple(masks)


def _apply_autotune(code, enabled):
    """给 GEMM 产物插入 tilelang 的 autotune 装饰器。

    tile 参数对 GEMM 影响约 30%（1024³ 默认 0.0910 ms、最优 0.0702 ms）。
    `configs` 必须是 **dict 列表**（写成 `dict(name=[...])` 会报
    `'str' object has no attribute 'get'`）。首次调用会跑完整搜索（约 1 分钟），
    所以默认关闭，由调用方显式开启。
    """
    if not enabled or '_tl_M' not in code:
        return code
    configs = '[' + ', '.join(
        'dict(block_M=%d, block_N=%d, block_K=%d, num_stages=%d)' % (bm, bn, bk, st)
        for bm in (32, 64, 128) for bn in (32, 64, 128)
        for bk in (16, 32) for st in (1, 2)) + ']'
    return code.replace(
        '@tilelang.jit(out_idx=',
        '@tilelang.autotune(configs=%s, warmup=10, rep=20)' % configs
        + chr(10) + '@tilelang.jit(out_idx=', 1)


def _finish(code, spec, precision, autotune):
    """统一收尾：自动调优包装 + fast 未实施路径的说明头。

    `fast` 不再做全文 dtype 替换（那会改掉 Tensor ABI 与调用方声明的语义）。
    唯一的 fast 收缩在 `lower_sum`：单轴浮点 sum 的累加器降为 float32，由
    `acc_dtype` 显式控制；其余路径保持 strict，并在生成物头部注明未实施。
    """
    code = _apply_autotune(code, autotune)
    if precision == 'fast' and not getattr(spec, 'precision_applied', False):
        if getattr(spec, 'precision_barrier', False):
            code = ("# 注意：precision='fast' 因源码含显式 float64 转换（精度屏障），"
                    "本生成物保留 float64 累加。\n") + code
        else:
            code = ("# 注意：precision='fast' 在 %s 路径尚未实施，本生成物按 strict 精度发射。\n"
                    "# 目前 fast 只收缩单轴浮点 sum 的累加器精度（acc_dtype=float32）。\n"
                    % spec.pattern) + code
    return code


# ══════════ 混合 dtype 窄门：整数标签谓词的浮点计数归约（20261025-138） ══════════
# 只开放"单个 sum 阶段 + 单自动浮点输出 + 只读整数输入 + 归约贡献为谓词 0/1 计数"。
# 资格遍历实际 IR 依赖（含 read offsets/scales、init、最终写回），不按源码拼写判断。

_MIXED_INT_DTYPES = ('int32', 'int64')
_MIXED_FLOAT_DTYPES = ('float32', 'float64')


def _mixed_offsets_ok(expr):
    for item in tuple(getattr(expr, 'offsets', ()) or ()) + tuple(getattr(expr, 'scales', ()) or ()):
        if isinstance(item, bool) or not isinstance(item, int):
            return False
    return True


def _mixed_int_term_ok(expr, int_names):
    if not hasattr(expr, 'op'):
        return False
    if expr.op == 'read':
        tensor = expr.value[0] if isinstance(expr.value, tuple) else None
        return tensor in int_names and not expr.args and _mixed_offsets_ok(expr)
    if expr.op in ('index', 'shape'):
        return True
    if expr.op == 'const':
        return isinstance(expr.value, int) and not isinstance(expr.value, bool)
    return False


def _mixed_predicate_ok(expr, int_names):
    op = expr.op
    if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
        return len(expr.args) == 2 and all(
            _mixed_int_term_ok(arg, int_names) for arg in expr.args)
    if op in ('and', 'or'):
        return bool(expr.args) and all(
            _mixed_predicate_ok(arg, int_names) for arg in expr.args)
    if op == 'not':
        return len(expr.args) == 1 and _mixed_predicate_ok(expr.args[0], int_names)
    return False


def _mixed_collect_reads(expr, out):
    if not hasattr(expr, 'op'):
        return
    if expr.op == 'read':
        out.append(expr)
    for arg in tuple(getattr(expr, 'args', ()) or ()):
        _mixed_collect_reads(arg, out)


def _mixed_is_const(expr, value):
    # 139 收口：只接受**浮点**常量（Python float 且 IR dtype 为 float32/float64）。
    # 整数 1/0 分支（IR 可为 int64）不属于"浮点 0/1 谓词计数"，一律拒绝。
    return (hasattr(expr, 'op') and expr.op == 'const'
            and isinstance(expr.value, float)
            and expr.dtype in _MIXED_FLOAT_DTYPES
            and float(expr.value) == float(value))


def _mixed_predicate_count_eligible(spec):
    """精确 IR 资格：只认"整数标签参与比较谓词的 0/1 浮点计数归约"。"""
    if spec.pattern != 'pipeline':
        return False
    program = getattr(spec, 'program', None)
    if program is None or len(getattr(program, 'stages', ()) or ()) != 1:
        return False
    if tuple(getattr(program, 'extra_stores', ()) or ()):
        return False
    stage = program.stages[0]
    reduction = getattr(stage, 'reduction', None)
    if reduction is None or reduction.op != 'sum':
        return False
    if not _mixed_is_const(reduction.init, 0.0):
        return False
    # 20261025-150：原先读 `reduction.dtype`（capture 里硬编码 'float64'，恒真无判别力）；
    # 改为按源语义重算：初值 + 贡献表达式的 Numba 提升结果。
    from semantic import source_accum_dtype
    if source_accum_dtype(stage) not in _MIXED_FLOAT_DTYPES:
        return False
    if not (hasattr(program.expression, 'op') and program.expression.op == 'result'
            and program.expression.value == 0):
        return False

    # 参数角色与 dtype：一个自动浮点输出 + 只读整数输入；不允许其它张量。
    abi = tuple(getattr(spec, 'param_abi', ()) or ())
    tensors = [item for item in abi if item.get('kind') == 'tensor']
    outputs = [item for item in tensors if item.get('role') == 'output']
    inputs = [item for item in tensors if item.get('role') == 'input']
    if len(outputs) != 1 or not inputs:
        return False
    if outputs[0].get('name') != program.output:
        return False
    if outputs[0].get('dtype') not in _MIXED_FLOAT_DTYPES:
        return False
    int_names = set()
    for item in inputs:
        if item.get('dtype') not in _MIXED_INT_DTYPES:
            return False
        int_names.add(item.get('name'))
    if outputs[0].get('name') in int_names:
        return False
    if set(item.get('name') for item in tensors) != set(int_names) | {program.output}:
        return False

    # 归约贡献：select(谓词, 1.0, 0.0)；整数只在谓词内被消费。
    expr = stage.expression
    if not (hasattr(expr, 'op') and expr.op == 'select' and len(expr.args) == 3):
        return False
    condition, on_true, on_false = expr.args
    if not (_mixed_is_const(on_true, 1.0) and _mixed_is_const(on_false, 0.0)):
        return False
    if not _mixed_predicate_ok(condition, int_names):
        return False
    all_reads, cond_reads = [], []
    _mixed_collect_reads(expr, all_reads)
    _mixed_collect_reads(condition, cond_reads)
    if {id(node) for node in all_reads} != {id(node) for node in cond_reads}:
        return False
    return True


def convert(src: str, *, gemm_policy='tf32', scoped_output_tile=1,
            scoped_output_interchange=False, scoped_leaf_policy='serial',
            execution='optimized', parameters=None, constants=None, entry=None,
            precision='strict', autotune=False, scalar_types=None,
            tensor_types=None) -> str:
    """主入口；threads显式保留线程和原地参数，不自动回退或使用GEMM策略。

    `precision`：`strict`（默认）按源码 dtype 发射；`fast` 只把单轴浮点 sum 的累加器
    换成 float32（输入表达式与输出仍按源码 dtype），其余路径等同 strict 并在生成物注明。
    """
    if execution not in ('optimized', 'threads'):
        raise ValueError('execution must be optimized or threads')
    # `scalar_types`：调用方声明的**标量类型特化契约**（不是从源码证明的推断）。
    # 本片只支持 optimized 逐元素路径上的 int64 标量声明；其余显式拒绝，不静默忽略。
    if scalar_types is not None:
        if type(scalar_types) is not dict:
            raise ValueError('scalar_types must be a dict of name -> dtype')
        for key, value in scalar_types.items():
            if type(key) is not str or type(value) is not str:
                raise ValueError('scalar_types keys and values must be strings')
            if value != 'int64':
                raise ValueError("scalar_types 目前只支持 'int64'（声明值: %s=%s）"
                                 % (key, value))
        if scalar_types and execution == 'threads':
            raise ValueError('threads 路径不接收 scalar_types')
        scalar_types = dict(scalar_types)
    # `tensor_types`：调用方对**无签名张量形参**的 dtype 声明（G2-B ROWCOPY 窄通道入口）。
    # 非空声明只服务 ROWCOPY；其它 optimized 路径会被分派拒绝，不做静默忽略。
    if tensor_types is not None:
        if type(tensor_types) is not dict:
            raise ValueError('tensor_types must be a dict of name -> dtype')
        for key, value in tensor_types.items():
            if type(key) is not str or type(value) is not str:
                raise ValueError('tensor_types keys and values must be strings')
        if tensor_types and execution == 'threads':
            raise ValueError('threads 路径不接收 tensor_types')
        if tensor_types and scalar_types:
            overlap = set(tensor_types) & set(scalar_types)
            if overlap:
                raise ValueError('tensor_types 与 scalar_types 不能声明同一个名字: '
                                 + ', '.join(sorted(overlap)))
        tensor_types = dict(tensor_types)
    if precision not in ('strict', 'fast'):
        raise ValueError("precision must be 'strict' or 'fast'")
    if type(autotune) is not bool:
        raise ValueError('autotune must be a bool')
    if execution == 'threads':
        if gemm_policy != 'tf32' or type(scoped_output_tile) is not int or scoped_output_tile != 1 \
                or scoped_output_interchange is not False or scoped_leaf_policy != 'serial':
            raise ValueError('threads does not accept non-default optimization options')
        from thread_capture import capture_thread
        from thread_lowering import lower_thread
        return lower_thread(capture_thread(src, parameters, constants, entry))
    if parameters is not None:
        raise ValueError('parameters is only supported with execution=threads')
    if constants is not None:
        raise ValueError('constants is only supported with execution=threads')
    if entry is not None:
        raise ValueError('entry is only supported with execution=threads')
    if gemm_policy not in ('tf32', 'scalar'):
        raise ValueError('gemm_policy must be tf32 or scalar')
    if type(scoped_output_tile) is not int or scoped_output_tile <= 0:
        raise ValueError('scoped_output_tile must be a positive integer')
    if type(scoped_output_interchange) is not bool:
        raise ValueError('scoped_output_interchange must be bool')
    if scoped_output_interchange and scoped_output_tile == 1:
        raise ValueError('output interchange requires scoped_output_tile > 1')
    policies = scoped_leaf_policy if type(scoped_leaf_policy) is tuple else (scoped_leaf_policy,)
    if any(type(policy) is not str or policy not in ('serial', 'block', 'tiled') for policy in policies):
        raise ValueError('scoped_leaf_policy must be serial, block, tiled, or a tuple of these policies')
    if (type(scoped_leaf_policy) is tuple or scoped_leaf_policy in ('block', 'tiled')) and not scoped_output_interchange:
        raise ValueError('cooperative or per-leaf policy requires output interchange')
    if 'tiled' in policies and scoped_output_tile not in (2, 4, 8):
        raise ValueError('tiled leaf policy requires scoped_output_tile in (2, 4, 8)')
    if tensor_types:
        # G2-B：非空 tensor_types 只进入 ROWCOPY 窄通道（独立识别/发射；不扩通用门）。
        import types as _types
        from rowcopy import convert_rowcopy
        rowcopy_code = convert_rowcopy(src, tensor_types, scalar_types)
        return _finish(rowcopy_code, _types.SimpleNamespace(pattern='rowcopy'),
                       precision, autotune)
    spec = parse_source(src, scalar_types=scalar_types)
    detect_pattern(spec)
    # B1：参数 ABI 的唯一来源（name/kind/dtype/role），lowerer 只从这里取 dtype。
    from abi import ensure_param_abi
    ensure_param_abi(spec)
    if getattr(spec, 'softmax_roundtrip', False) and precision != 'strict':
        raise Unsupported('softmax 三段式窄门仅支持 strict（fast 不开放）')
    if getattr(spec, 'constant_guard', False) and precision != 'strict':
        raise Unsupported('模块级常量上界 guard 窄门仅支持 strict（fast 不开放）')
    # 20261025-145：常量 guard 窄门的**累加器精度**标志（与 139 mixed_predicate_count、
    # 142 softmax_roundtrip 同型，只对本窄门新接纳的源生效）。
    # 依据：`temp = 0`（int 字面量）+ float32 贡献在 Numba/CUDA 下 `int64 + f32 → float64`，
    # 即源码是"f64 累加、f32 写回"（cuda_type_evidence_int_acc.json 的 typing 与 PTX 实测）。
    # 既有全 f32 张量的 pipeline 累加器策略**不变**——本标志只改常量 guard 窄门。
    spec.constant_guard_accum_f64 = False
    if getattr(spec, 'constant_guard', False) and precision == 'strict':
        stages = tuple(getattr(getattr(spec, 'program', None), 'stages', ()) or ())
        # 20261025-150：`stage.reduction.dtype` 是硬编码 'float64'（恒真），改用源语义重算，
        # 条件收紧为"初值+贡献按 Numba 规则提升为 f64，且贡献表达式本身是 f32"。
        from semantic import source_accum_dtype
        spec.constant_guard_accum_f64 = bool(stages) and all(
            stage.reduction.op == 'sum'
            and source_accum_dtype(stage) == 'float64'
            and stage.expression.dtype == 'float32'
            for stage in stages)
    if scalar_types and spec.pattern != 'elementwise':
        raise Unsupported('scalar_types 目前只支持逐元素路径（当前模式: %s）' % spec.pattern)
    # B2 门禁（按 param_abi 统一判断）：pipeline / nested 允许 float32 或 float64 张量，
    # 但本片只支持**所有张量同 dtype**；混合 dtype 明确拒绝（以后需要再单独放开）。
    tensor_dtypes_by_name = dict(getattr(spec, 'tensor_dtypes', {}) or {})
    spec.mixed_predicate_count = False
    if spec.pattern in ('pipeline', 'nested'):
        kinds = set(tensor_dtypes_by_name.values())
        if len(kinds) > 1:
            eligible = (spec.pattern == 'pipeline' and precision == 'strict'
                        and _mixed_predicate_count_eligible(spec))
            if not eligible:
                raise Unsupported(
                    '%s 路径目前只支持所有张量同 dtype（%s）；混合 dtype 未支持'
                    % (spec.pattern, ', '.join('%s=%s' % item
                                               for item in sorted(tensor_dtypes_by_name.items()))))
            spec.mixed_predicate_count = True
        if not spec.mixed_predicate_count:
            unsupported = {name: dtype for name, dtype in tensor_dtypes_by_name.items()
                           if dtype not in ('float32', 'float64')}
            if unsupported:
                raise Unsupported(
                    '%s 路径暂只支持 float32/float64 张量参数（%s）'
                    % (spec.pattern, ', '.join('%s=%s' % item for item in sorted(unsupported.items()))))
    if scoped_output_tile != 1 and spec.pattern != 'nested':
        raise Unsupported('输出分块复用当前只支持嵌套归约')
    spec.precision = precision
    if spec.pattern == 'gemm' and gemm_policy == 'scalar':
        from lowering import lower_sum
        return _finish(lower_sum(spec), spec, precision, autotune)
    if spec.pattern == 'nested':
        from scoped_lowering import lower_scoped
        return _finish(lower_scoped(
            spec, output_tile=scoped_output_tile,
            output_interchange=scoped_output_interchange,
            leaf_policy=scoped_leaf_policy), spec, precision, autotune)
    if spec.pattern == 'pipeline':
        from pipeline_lowering import lower_pipeline
        return _finish(lower_pipeline(spec), spec, precision, autotune)
    if spec.pattern == 'gemm':
        # GEMM 模板（T.gemm/tf32）当前只有 float32 实现：张量声明、shared 与 fragment 全是 float32。
        # 声明了其它 dtype 的源码走这个模板会被**静默改型**，因此在这里显式拒绝并报出参数名与 dtype。
        # 注意：`gemm_policy='scalar'` 走的是 lower_sum，另一条路径，不在此门禁范围。
        unsupported = {name: dtype
                       for name, dtype in (getattr(spec, 'tensor_dtypes', {}) or {}).items()
                       if dtype != 'float32'}
        if unsupported:
            raise Unsupported(
                'GEMM 模板目前只支持 float32 张量参数（%s）；'
                'float64/fp16 等需改用其它策略或拒绝'
                % ', '.join('%s=%s' % item for item in sorted(unsupported.items())))
        return _finish(lift_gemm(spec), spec, precision, autotune)
    if spec.pattern == 'flat_elementwise':
        from lowering import lower_flat_map
        return _finish(lower_flat_map(spec), spec, precision, autotune)
    if spec.pattern == 'flat_linear':
        from flat_linear import lower as lower_flat_linear
        return _finish(lower_flat_linear(spec), spec, precision, autotune)
    if spec.pattern == 'elementwise':
        if spec.compute is None:
            # 程序体是"输出域循环"但没有归约阶段（`for k in range(9): f[k,j,i] = ...`）：
            # 循环轴不参与并行化，逐元素模板只有单一写回表达式，接不了。
            raise Unsupported('输出域循环没有归约阶段（按方向/通道逐个写回的循环）：'
                              '逐元素模板只支持单次写回')
        from lowering import lower_map
        return _finish(lower_map(spec), spec, precision, autotune)
    if spec.pattern in ('reduce_sum', 'reduce_max', 'reduce_min'):
        from lowering import lower_sum
        return _finish(lower_sum(spec), spec, precision, autotune)
    raise Unsupported(f'未实现的模式: {spec.pattern}')


def convert_func(func, *, gemm_policy='tf32', scoped_output_tile=1,
            scoped_output_interchange=False, scoped_leaf_policy='serial',
            execution='optimized', parameters=None, constants=None, entry=None,
            scalar_types=None, tensor_types=None) -> str:
    """从函数对象转换（方便直接用），数值及调度策略与convert一致。"""
    return convert(inspect.getsource(func), gemm_policy=gemm_policy,
                   scoped_output_tile=scoped_output_tile,
                   scoped_output_interchange=scoped_output_interchange,
                   scoped_leaf_policy=scoped_leaf_policy,
                   execution=execution, parameters=parameters, constants=constants,
                   entry=entry, scalar_types=scalar_types, tensor_types=tensor_types)


if __name__ == '__main__':
    demo = '''
@cuda.jit
def matmul(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp
'''
    print(convert(demo))
