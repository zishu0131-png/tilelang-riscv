# -*- coding: utf-8 -*-
"""受限纯计算描述。节点表达计算，不包含 TileLang 调度或算子名称。"""
import ast
import math
from dataclasses import dataclass


class Unsupported(Exception):
    """语义无法忠实捕获或不满足当前后端约束。"""


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()
    value: object = None
    dtype: str = 'float32'
    # 仅 read 使用：各维常量偏移与系数（与 indices 等长）；stencil 形如 A[i-1]、A[2*i]。
    offsets: tuple = ()
    scales: tuple = ()
    # 仅 read 使用：下标由**张量读**决定（gather 一期形态），发射为 src[idx_expr]。
    indirect: bool = False


@dataclass(frozen=True)
class Reduction:
    axis: str
    start: str
    stop: str
    step: str
    accumulator: str
    init: Expr
    dtype: str
    op: str = 'sum'
    # 归约域的计算形式。start/stop 字符串是它们的历史投影，供既有比较逻辑使用；
    # 语义校验与发射都改走这里，字符串不再承担求值。
    start_expr: Expr | None = None
    stop_expr: Expr | None = None


def plain_text(expr):
    """把已捕获的整型表达式渲染成源码风格的字符串（不带 TIR cast）。"""
    if expr.op == 'const':
        return str(expr.value)
    if expr.op == 'shape':
        return f'{expr.value[0]}.shape[{expr.value[1]}]'
    if expr.op in ('index', 'scalar'):
        return expr.value
    if expr.op in ('add', 'sub', 'mul'):
        operator = {'add': '+', 'sub': '-', 'mul': '*'}[expr.op]
        return f'({plain_text(expr.args[0])} {operator} {plain_text(expr.args[1])})'
    raise Unsupported('循环域只支持常量/shape引用/索引变量/标量形参的整型组合')


def polynomial(expr):
    """整型表达式的稀疏多项式：{有序因子元组: 整数系数}；不支持时返回 None。

    用途是把 `range(b*win, (b+1)*win)` 的域长度化简成 `win`：
    两项相减后轴项抵消，剩下的表达式与归约轴无关，才是可用的归约长度。
    """
    if expr.op == 'const' and type(expr.value) is int:
        return {(): expr.value} if expr.value else {}
    if expr.op in ('index', 'scalar'):
        return {(expr.value,): 1}
    if expr.op == 'shape':
        # 运行期张量维度：长度不再是"纯标量表达式"，但仍是可合成的符号项。
        return {(f'{expr.value[0]}.shape[{expr.value[1]}]',): 1}
    if expr.op in ('add', 'sub'):
        left, right = polynomial(expr.args[0]), polynomial(expr.args[1])
        if left is None or right is None:
            return None
        sign = 1 if expr.op == 'add' else -1
        result = dict(left)
        for key, value in right.items():
            merged = result.get(key, 0) + sign * value
            if merged:
                result[key] = merged
            else:
                result.pop(key, None)
        return result
    if expr.op == 'mul':
        left, right = polynomial(expr.args[0]), polynomial(expr.args[1])
        if left is None or right is None:
            return None
        result = {}
        for left_key, left_value in left.items():
            for right_key, right_value in right.items():
                key = tuple(sorted(left_key + right_key))
                result[key] = result.get(key, 0) + left_value * right_value
        return {key: value for key, value in result.items() if value}
    return None


def polynomial_text(poly):
    """把多项式渲染成可发射的算术字符串；空多项式是 0。"""
    if not poly:
        return '0'
    parts = []
    for key, coeff in sorted(poly.items(), key=lambda item: (len(item[0]), item[0])):
        body = ' * '.join(key)
        if not body:
            parts.append(('-' if coeff < 0 else '+', str(abs(coeff))))
        elif abs(coeff) == 1:
            parts.append(('-' if coeff < 0 else '+', body))
        else:
            parts.append(('-' if coeff < 0 else '+', f'{abs(coeff)} * {body}'))
    text = ('-' if parts[0][0] == '-' else '') + parts[0][1]
    for sign, body in parts[1:]:
        text += f' {sign} {body}'
    return text


# 每个归约操作的单位元：屏蔽越界 lane 与初始化累加器都用它，不能用近似哨兵。
REDUCTION_IDENTITY = {'sum': 0.0, 'max': float('-inf'), 'min': float('inf')}


@dataclass(frozen=True)
class Compute:
    expression: Expr
    output: str
    output_axes: tuple
    bindings: tuple = ()
    reduction: Reduction | None = None
    # 附加输出：(张量, 下标, 表达式)。同一并行域内一次写多个张量（`x[i]=i; y[i]=i`）。
    extra_stores: tuple = ()
    # 主输出的完整写回下标：可能含常量维（`O[n,oy,ox,0]`），那些维不进并行域。
    store_indices: tuple = ()
    store_constants: tuple = ()   # 与 store_indices 等长，常量维的固定下标（其余为 None）
    # 写回分组：有数据依赖时拆成多个并行域，每组按顺序发射一个 T.Parallel。
    # 组内是该写回在 [主输出] + extra_stores 里的位置，例如 ((0,), (1,))。
    # 空元组表示"全部写回同一组"（无依赖的普通多写回）。
    store_groups: tuple = ()
    # 需要物化的局部绑定：(名字, 表达式)。它们保存写回前的原值，发射层会存进
    # per-lane 缓冲，所有写回都从缓冲读（bounce-back 的整体交换）。
    materialized_bindings: tuple = ()
    # 按组物化：与 `store_groups` 等长，第 g 项是该组**组首**需要保存的绑定
    # （名字, 表达式）。空元组表示"全部在第 0 组组首保存"（旧行为）。
    # 用于"写回 → 局部值 → 再写回"：局部值必须在它定义之后、下一次覆盖之前取值。
    group_bindings: tuple = ()


def output_names(spec):
    """本规格写回的全部张量：主输出 + 附加输出（逐元素与归约两条路径都看）。"""
    names = [spec.store_target] if getattr(spec, 'store_target', None) else []
    for holder in (getattr(spec, 'compute', None), getattr(spec, 'program', None)):
        if holder is not None:
            names += [item[0] for item in (getattr(holder, 'extra_stores', ()) or ())]
    return list(dict.fromkeys(names))


def _shift(offset, delta):
    """偏移可能是整数或形状引用 `(符号, 张量, 维)`；只允许叠加常量。"""
    if isinstance(offset, tuple):
        if delta:
            raise Unsupported('形状偏移不能再叠加常量偏移')
        return offset
    return offset + delta


# 常量下标维的占位轴名：不参与轴匹配，长度由构造期参数单独给出。
AXIS_CONSTANT = '#'


def _axis_affine(node, scalars=()):
    """把下标解析成 `(轴名, 系数, 偏移)`，覆盖 `i` / `i±c` / `a*i` / `a*i±c`。

    系数只接受正整数常量：`X[2*oy]` 是"第 oy 个输出读第 2*oy 个输入"（stride 采样），
    而 `X[i//2]` 这类除法不做代数重写，直接拒绝——它把多个输出映射到同一输入，
    与逐元素并行模型冲突。

    系数也可以是**运行期标量形参**（`F[b*win]` 的窗口起点），记为 `('scalar', 名字)`；
    该形态只对归约初值合法，逐元素映射会显式拒绝。
    """
    if isinstance(node, ast.Constant) and type(node.value) is int:
        # 常量下标读（`X[n,y,x,0]` 的通道位、`F[0,i]` 的首元素）：
        # 该维不参与轴匹配，占位符 AXIS_CONSTANT 表示"这一维取固定下标"。
        return AXIS_CONSTANT, 1, node.value
    if isinstance(node, ast.Name):
        return node.id, 1, 0
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        sign = 1 if isinstance(node.op, ast.Add) else -1
        head, tail = _axis_affine(node.left, scalars), node.right
        if head is not None and isinstance(tail, ast.Constant) and type(tail.value) is int:
            return head[0], head[1], _shift(head[2], sign * tail.value)
        # `i - X.shape[k]`：偏移是一个**运行期**张量维度（分段搬运，如两段拼接）。
        # 该维度会被提成生成物的构造期参数，因此这里只记录引用，不试图求值。
        ref = shape_axis(tail) if head is not None else None
        if ref is not None and not isinstance(head[2], tuple):
            return head[0], head[1], (sign, ref[0], ref[1])
        if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Constant)                 and type(node.left.value) is int:
            got = _axis_affine(node.right, scalars)
            if got is not None:
                return got[0], got[1], got[2] + node.left.value
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        for constant, inner in ((node.left, node.right), (node.right, node.left)):
            if isinstance(constant, ast.Constant) and type(constant.value) is int                     and constant.value > 0:
                got = _axis_affine(inner, scalars)
                if got is not None and got[1] == 1:
                    if isinstance(got[2], tuple):
                        raise Unsupported('形状偏移上不能再乘系数')
                    return got[0], constant.value, constant.value * got[2]
        # 运行期标量系数：`F[b*win]`（窗口起点）。只对归约初值合法，调用方负责限定用途。
        for factor, inner in ((node.left, node.right), (node.right, node.left)):
            if isinstance(factor, ast.Name) and factor.id in scalars:
                got = _axis_affine(inner, scalars)
                if got is not None and got[1] == 1:
                    if isinstance(got[2], tuple):
                        raise Unsupported('形状偏移上不能再乘系数')
                    return got[0], ('scalar', factor.id), got[2]
        return None
    return None


def _axis_with_offset(node):
    """兼容包装：只接受系数为 1 的形态，返回 (轴名, 偏移)。"""
    parsed = _axis_affine(node)
    return None if parsed is None or parsed[1] != 1 else (parsed[0], parsed[2])


def tensor_access(node, with_offsets=False, scalars=(), fallback=None):
    """解析张量下标。

    `fallback` 提供 `(bindings, params, dtypes, axes)` 时，解析不出仿射式的下标
    （`Qin[i, d + half]`、`Cos[i, pair]`）按**运行期地址表达式**捕获：该维不参与轴匹配，
    坐标由表达式在 lane 上下文里求值。写回下标不允许这种形态（store 路径不传 fallback）。
    """
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        raise Unsupported('只支持直接张量访问，如 A[i, k]')
    indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    parsed = []
    axis_names = set(fallback[3]) if fallback is not None else None
    for item in indices:
        form = _axis_affine(item, scalars)
        # 名字不是计算轴，说明它是局部绑定（`pair = d // 2`），按运行期地址处理。
        if form is not None and axis_names is not None \
                and form[0] != AXIS_CONSTANT and form[0] not in axis_names:
            form = None
        if form is None and fallback is not None:
            bindings, params, dtypes, axes = fallback
            form = (AXIS_CONSTANT, 1,
                    capture_expr(item, bindings, params, dtypes, axes))
        parsed.append(form)
    if any(p is None for p in parsed):
        raise Unsupported('张量下标必须是"轴变量"、"轴变量±常量"或"正整数×轴变量±常量"')
    if with_offsets:
        return (node.value.id, tuple(p[0] for p in parsed),
                tuple(p[2] for p in parsed), tuple(p[1] for p in parsed))
    # 写回下标允许常量维（`O[n,oy,ox,0]`：把一个邻域展开到通道维），但不允许系数或轴偏移。
    if any(p[1] != 1 or (p[0] != AXIS_CONSTANT and p[2] != 0) for p in parsed):
        raise Unsupported('该位置不支持带系数或偏移的下标（仅逐元素/展平映射支持）')
    return node.value.id, tuple(p[0] for p in parsed)


# 受限逐元素数学函数；数值容差验证不等于两侧逐位一致证明。
MATH_FUNCTIONS = ('exp', 'exp2', 'log', 'log2', 'log10', 'sqrt', 'rsqrt', 'tanh',
                  'sin', 'cos', 'tan', 'fabs', 'floor', 'ceil', 'erf')
# 表达式里的类型转换调用（`int32(math.floor(sy))`）：与装饰器里的类型名同一套别名。
CAST_TYPES = ('bool', 'boolean', 'int8', 'int16', 'int32', 'int64',
              'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64',
              'double')
# math 模块常量按字面量折叠（与 threads 路径一致）。
MATH_CONSTANTS = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
# float32 的最小有限值；小于它的负数对 float32 数据等价于 -inf。
FLOAT32_MIN = float.fromhex('-0x1.fffffep+127')


def numeric_dtype(*operands):
    """Numba 规则：全 float32 保持 float32；全 int64 按 int64（与 numba 的 64 位环绕一致）；
    混合类型按 float64 提升。int32/uint 的混合运算会先提升到 int64，避免静默截断。

    float16 是窄类型：两个 float16 **逐运算按 float16 舍入**（实测 numba 不做 float32
    中间精度，链式运算与 fp32 中间量差 1082/4096 个元素）；只要混进 float32/float64
    （含 Python 浮点字面量，它在 numba 里是 float64）就整体提升。
    """
    if all(operand.dtype == 'float16' for operand in operands):
        return 'float16'
    if all(operand.dtype in ('float16', 'float32') for operand in operands)         and any(operand.dtype == 'float32' for operand in operands):
        return 'float32'
    if all(operand.dtype == 'float32' for operand in operands):
        return 'float32'
    if all(operand.dtype == 'int64' for operand in operands):
        return 'int64'
    if any(operand.dtype in ('int32', 'uint8', 'uint32') for operand in operands)             and all(operand.dtype in ('int32', 'int64') for operand in operands):
        return 'int64'
    return 'float64'


def source_accum_dtype(stage):
    """源语义的**归约累加 dtype**：`numeric_dtype(初值, 贡献表达式)`。

    `Reduction.dtype` 在 `capture.py` 里是硬编码 'float64'（历史占位），不能当类型证据；
    本函数按 Numba 提升规则重算：`0`/`0.0` 字面量初值 + f32 贡献 → f64；
    f32 读初值 + f32 贡献 → f32。门禁与发射都应使用它，而不是 `reduction.dtype`。
    """
    return numeric_dtype(stage.reduction.init, stage.expression)


def _capture_call(node, bindings, params, dtypes=None, axes=()):
    if node.keywords:
        raise Unsupported('调用不支持关键字参数')
    called = node.func
    # D1：内置转换名被局部绑定/形参/模块绑定遮蔽时**先拒绝**（在 inf 特例之前）。
    if isinstance(called, ast.Name) and called.id in ('float', 'int')             and (called.id in bindings or called.id in params):
        raise Unsupported('%s 被局部绑定或形参遮蔽，不能当作内置转换' % called.id)
    # `float('inf')` / `float('-inf')`：Python 里构造无穷的常见写法（softmax 的初值）。
    if isinstance(called, ast.Name) and called.id == 'float' and len(node.args) == 1             and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        text = node.args[0].value.strip().lower()
        values = {'inf': float('inf'), '+inf': float('inf'), 'infinity': float('inf'),
                  '-inf': float('-inf'), '-infinity': float('-inf')}
        if text in values:
            return Expr('const', value=values[text], dtype='float64')
        raise Unsupported('float() 只支持 inf/infinity 这类常量: ' + node.args[0].value)
    called = node.func
    module = called.value.id if isinstance(called, ast.Attribute) \
        and isinstance(called.value, ast.Name) else None
    name = called.attr if module else (called.id if isinstance(called, ast.Name) else None)
    arguments = [capture_expr(argument, bindings, params, dtypes, axes) for argument in node.args]
    if module in ('math', 'np', 'numpy'):
        if name == 'atan2' and len(arguments) == 2:
            if any(argument.dtype == 'int64' for argument in arguments):
                raise Unsupported('数学函数的参数必须来自浮点表达式')
            return Expr('atan2', tuple(arguments), dtype=numeric_dtype(*arguments))
        if len(arguments) != 1 or name not in MATH_FUNCTIONS:
            raise Unsupported('只支持 math/np 的: ' + ', '.join(MATH_FUNCTIONS))
        if arguments[0].dtype == 'int64':
            raise Unsupported('数学函数的参数必须来自浮点表达式')
        return Expr(name, tuple(arguments), dtype=arguments[0].dtype)
    if name == 'round' and module is None and len(arguments) == 1:
        # 内置 `round(x)`：numba 里等价于就近取整（banker's rounding 与 T.round 一致）。
        if arguments[0].dtype == 'int64':
            raise Unsupported('round 的参数必须来自浮点表达式')
        return Expr('round', tuple(arguments), dtype=arguments[0].dtype)
    if module is not None:
        raise Unsupported('只支持 math/np 的数学函数，以及内置 abs / max / min / round')
    # D1：内置 float()/int() 的 **CUDA typing** 结果（probe_numba_casts.json）：
    # float: float16/32/64 保型，整型→float64；int: 整型→同宽，bool→int64。
    # int(float*) 需要运行期值域约束，归 D2，本片明确拒绝。
    if module is None and name in ('float', 'int') and len(arguments) == 1:
        operand = arguments[0]
        if name == 'float':
            if operand.dtype in ('float16', 'float32', 'float64'):
                return Expr('cast', (operand,), dtype=operand.dtype)
            if operand.dtype in INT_DTYPES:
                return Expr('cast', (operand,), dtype='float64')
            raise Unsupported('float() 的实参类型暂不支持: ' + operand.dtype)
        if operand.dtype in INT_DTYPES:
            return Expr('cast', (operand,), dtype=operand.dtype)
        if operand.dtype == 'bool':
            return Expr('cast', (operand,), dtype='int64')
        if operand.dtype in ('float16', 'float32', 'float64'):
            raise Unsupported('int(浮点) 暂不支持（需要运行期值域约束，归 D2）')
        raise Unsupported('int() 的实参类型暂不支持: ' + operand.dtype)
    if name in CAST_TYPES and len(arguments) == 1:
        # 显式类型转换（`int32(x)`、`float32(y)`）：按目标类型做一次 cast。
        target = {'boolean': 'bool', 'double': 'float64'}.get(name, name)
        return Expr('cast', (arguments[0],), dtype=target)
    if name == 'abs' and len(arguments) == 1:
        if arguments[0].dtype == 'int64':
            raise Unsupported('abs 的参数必须来自浮点表达式')
        return Expr('abs', tuple(arguments), dtype=arguments[0].dtype)
    if name in ('max', 'min') and len(arguments) == 2:
        return Expr(name, tuple(arguments), dtype=numeric_dtype(*arguments))
    raise Unsupported('不支持的调用: ' + ast.unparse(node))


def capture_expr(node, bindings, params, dtypes=None, axes=()):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if isinstance(node.value, float):
            # 浮点常量只要求有限；2^53 精确性约束只针对整数（如归约初值 -1e300 是合法的）。
            if not math.isfinite(node.value):
                raise Unsupported('浮点常量必须有限')
        elif abs(node.value) > 2 ** 53:
            raise Unsupported('整数常量必须位于可精确表示范围内')
        return Expr('const', value=node.value,
                    dtype='int64' if type(node.value) is int else 'float64')
    if isinstance(node, ast.Name) and node.id in bindings:
        return bindings[node.id]
    if isinstance(node, ast.Name) and node.id in axes:
        # 索引变量作为“值”参与表达式（`t < n`、`i if c else j`）：每个 lane 的坐标本身就是标量。
        return Expr('index', value=node.id, dtype='int64')
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id == 'math' and node.attr == 'inf':
        return Expr('const', value=float('inf'), dtype='float64')
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id == 'math' and node.attr in MATH_CONSTANTS:
        return Expr('const', value=MATH_CONSTANTS[node.attr], dtype='float64')
    if shape_axis(node) is not None:
        name, dim = shape_axis(node)
        if name not in params:
            raise Unsupported('shape必须来自张量形参: ' + name)
        return Expr('shape', value=(name, dim), dtype='int64')
    if isinstance(node, ast.Subscript):
        # F4：语法层先拦多级 gather（`src[a[b[i]]]`），避免内层先被 dtype 门拦下、
        # 使多级判定依赖 dtype 组合。
        if _subscript_nested_in_address(node):
            raise Unsupported('gather 一期不支持多级 gather（下标里还有张量读的下标）')
        # 签名标量可以作为下标系数（`F[b*win]`）；它们只来自 scalar_bindings。
        scalar_names = {key for key, value in (bindings or {}).items()
                        if getattr(value, 'op', '') == 'scalar'}
        name, indices, offsets, scales = tensor_access(
            node, with_offsets=True, scalars=scalar_names,
            fallback=(bindings, params, dtypes, set(axes)))
        if name not in params:
            raise Unsupported('读取的张量必须是形参: ' + name)
        # 张量 dtype 来自静态签名；没有签名时维持历史约定 float32。
        dtype = (dtypes or {}).get(name, 'float32')
        if _has_read_in_address(offsets, scales):
            return _capture_gather(name, dtype, indices, offsets, scales, params, dtypes)
        return Expr('read', value=(name, indices), offsets=offsets, scales=scales,
                    dtype=dtype)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = capture_expr(node.operand, bindings, params, dtypes, axes)
        # 常量上的正负号直接折叠：归约初值 `acc = -1e300` 必须保持为 const 才能被模式识别。
        if operand.op == 'const' and type(operand.value) in (int, float):
            value = -operand.value if isinstance(node.op, ast.USub) else operand.value
            return Expr('const', value=value, dtype=operand.dtype)
        return Expr('pos' if isinstance(node.op, ast.UAdd) else 'neg', (operand,), dtype=operand.dtype)
    if isinstance(node, ast.Call):
        return _capture_call(node, bindings, params, dtypes, axes)
    if isinstance(node, ast.Compare) and len(node.ops) == 1             and isinstance(node.ops[0], (ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq)):
        left = capture_expr(node.left, bindings, params, dtypes, axes)
        right = capture_expr(node.comparators[0], bindings, params, dtypes, axes)
        operation = {ast.Lt: 'lt', ast.Gt: 'gt', ast.LtE: 'le',
                     ast.GtE: 'ge', ast.Eq: 'eq', ast.NotEq: 'ne'}[type(node.ops[0])]
        return Expr(operation, (left, right), dtype='bool')
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        # 逻辑组合（`i > 0 and i < n-1`）：只允许逐项是布尔表达式。
        # 不做短路求值——这些项都是纯比较，没有副作用，与 Python 语义一致。
        items = [capture_expr(value, bindings, params, dtypes, axes) for value in node.values]
        if any(item.dtype != 'bool' for item in items):
            raise Unsupported('逻辑运算的每一项必须是布尔表达式')
        operation = 'and' if isinstance(node.op, ast.And) else 'or'
        result = items[0]
        for item in items[1:]:
            result = Expr(operation, (result, item), dtype='bool')
        return result
    if isinstance(node, ast.IfExp):
        condition = capture_expr(node.test, bindings, params, dtypes, axes)
        if condition.dtype != 'bool':
            raise Unsupported('条件表达式必须产生 bool')
        yes = capture_expr(node.body, bindings, params, dtypes, axes)
        no = capture_expr(node.orelse, bindings, params, dtypes, axes)
        return Expr('select', (condition, yes, no), dtype=numeric_dtype(yes, no))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        left = capture_expr(node.left, bindings, params, dtypes, axes)
        right = capture_expr(node.right, bindings, params, dtypes, axes)
        return Expr('pow', (left, right), dtype=numeric_dtype(left, right))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor,
                                                             ast.LShift, ast.RShift)):
        # 整数哈希 / 噪声生成里的位运算；`& 0xFFFFFFFF` 也是模拟 uint32 的常用写法。
        left = capture_expr(node.left, bindings, params, dtypes, axes)
        right = capture_expr(node.right, bindings, params, dtypes, axes)
        if left.dtype not in INT_DTYPES or right.dtype not in INT_DTYPES:
            raise Unsupported('位运算的操作数必须是整型表达式')
        operation = {ast.BitAnd: 'bitand', ast.BitOr: 'bitor', ast.BitXor: 'bitxor',
                     ast.LShift: 'lshift', ast.RShift: 'rshift'}[type(node.op)]
        return Expr(operation, (left, right), dtype='int64')
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                                             ast.FloorDiv, ast.Mod)):
        left = capture_expr(node.left, bindings, params, dtypes, axes)
        right = capture_expr(node.right, bindings, params, dtypes, axes)
        operation = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul', ast.Div: 'div',
                     ast.FloorDiv: 'floordiv', ast.Mod: 'mod'}[type(node.op)]
        if operation in ('floordiv', 'mod'):
            # 整除/取余只对整数成立：Python 语义向下取整，并按除数符号修正。
            # 声明为 int64，避免 float 参与后 TIR 的整除语义与源码不一致。
            if left.dtype not in INT_DTYPES or right.dtype not in INT_DTYPES:
                raise Unsupported('整除/取余的操作数必须是整型表达式')
            if right.op == 'const' and type(right.value) is int and right.value == 0:
                raise Unsupported('除数不能为零常量')
            return Expr(operation, (left, right), dtype='int64')
        if (operation == 'div' and left.dtype in INT_DTYPES + ('bool',)
                and right.dtype in INT_DTYPES + ('bool',)):
            # 真除：两侧都是整型/布尔时结果是 float64（Numba CUDA 前端的定型，
            # 见 evidence/diagnosis/div_semantics_20261024）。其余组合沿用 numeric_dtype。
            return Expr(operation, (left, right), dtype='float64')
        return Expr(operation, (left, right), dtype=numeric_dtype(left, right))
    raise Unsupported('不支持的纯表达式: ' + ast.unparse(node))


def _offset_reads(expression):
    """read 节点 offsets/scales 里嵌套的读（gather 的 index 读就在这里）。"""
    found = []
    if expression.op != 'read':
        return found
    for item in tuple(expression.offsets or ()) + tuple(expression.scales or ()):
        if isinstance(item, Expr):
            found.extend(reads(item))
    return found


def reads(expression):
    """保留访问的出现次数；A*A不是只出现一个A。

    20261026-173：read 的 offsets/scales 里若嵌套了张量读（gather 的 index），
    一并登记，避免“直接式漏记输入”。
    """
    if expression.op == 'read':
        return [expression.value] + _offset_reads(expression)
    return [read for arg in expression.args for read in reads(arg)]


def referenced_axes(expression):
    """表达式里出现的计算轴名：读取下标里的轴，以及作为值参与的索引变量。"""
    found = set()
    stack = [expression]
    while stack:
        node = stack.pop()
        if node.op == 'read':
            found.update(node.value[1])
        elif node.op == 'index':
            found.add(node.value)
        stack.extend(node.args)
    return found


AUG_OPS = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul', ast.Div: 'div'}


def expand_augassign(store):
    """`x[i] += v` → `x[i] = x[i] + v`。

    增量写回（`arr[i] += 1`、`arr[i] /= val`）在逐元素映射里与读改写等价：
    每个 lane 只碰自己的元素，tile 级一次性完整覆盖输出域。
    """
    if not isinstance(store, ast.AugAssign):
        return store
    if type(store.op) not in AUG_OPS:
        raise Unsupported('不支持的增量运算符: ' + type(store.op).__name__)
    target = store.target
    if not isinstance(target, ast.Subscript):
        raise Unsupported('增量写回只支持张量下标')
    load = ast.parse(ast.unparse(target), mode='eval').body   # 同一目标，Load 上下文
    value = ast.BinOp(left=load, op=store.op, right=store.value)
    return ast.copy_location(ast.Assign(targets=[target], value=value), store)


def merge_branch_store(store):
    """`if c: O[i] = a` / `else: O[i] = b` → `O[i] = a if c else b`。

    只合并**同一输出、同一下标**的两分支——这是逐元素 select，每个 lane 独立选值。
    分支写不同张量/不同下标时保持原样，交给下游按原规则拒绝。
    """
    if not isinstance(store, ast.If) or not store.orelse:
        return store
    if len(store.body) != 1 or len(store.orelse) != 1:
        return store
    yes, no = store.body[0], store.orelse[0]
    if not (isinstance(yes, ast.Assign) and len(yes.targets) == 1
            and isinstance(no, ast.Assign) and len(no.targets) == 1):
        return store
    if ast.dump(yes.targets[0]) != ast.dump(no.targets[0]):
        return store
    value = ast.IfExp(test=store.test, body=yes.value, orelse=no.value)
    return ast.copy_location(ast.Assign(targets=[yes.targets[0]], value=value), store)


def read_accesses(expression):
    """**本阶段实际发生的内存读取**：bound 节点只读 per-lane 缓冲，不再展开其来源。

    形状/输入/in-place 这类契约读集合用 `reads()`（它会穿透 bound 的来源），
    两者不能混用：把历史读取挪到当前阶段会触发错误的跨 lane 判定或重复读全局内存。
    """
    if expression.op == 'bound':
        return []
    if expression.op == 'read':
        nested = []
        for item in tuple(expression.offsets or ()) + tuple(expression.scales or ()):
            if isinstance(item, Expr):
                nested.extend(read_accesses(item))
        return [expression] + nested
    return [node for arg in expression.args for node in read_accesses(arg)]


def _access(node):
    """(张量, 下标, 常量偏移, 系数)：判断"是不是同一个元素"必须四项一起比。"""
    return node.value[0], node.value[1], node.offsets, node.scales


def _same_element(left, right):
    """两次访问是否指向同一元素（同一 lane 在前后两个并行域里碰的是同一格）。

    常量维下标的运行期偏移按结构相等比较（绑定已内联，结构相同即同一地址）；
    系数不同的仿射下标（`A[i]` 与 `A[2*i]`）是不同元素。
    """
    return tuple(left[0]) == tuple(right[0]) \
        and tuple(left[1]) == tuple(right[1]) \
        and tuple(left[2]) == tuple(right[2])


# 运行期是 int64 环绕算术：常量差必须按模 2**64 比较。
# 例：H = 2**62 时，`x` 与 `x+H+H+H+H` 在 int64 下**是同一个地址**（4H ≡ 0 mod 2**64），
# 用 Python 无界整数相减会错误地判成不相交。
INT64_WRAP = 1 << 64


def _int_offset_key(offset):
    """把**int64 域**的偏移归一成 `(base_key, const)`；无法证明时返回 None。

    白名单：int 常量、**int64** scalar 形参、**int64** shape 引用、以及它们的 add/sub 常量链。
    任何 float 参与、任何 cast、非白名单算子、**非 int64 的整型（位宽不同）** → None。
    """
    if isinstance(offset, int):
        return ('const',), offset
    if not isinstance(offset, Expr) or offset.dtype != 'int64':
        return None
    if offset.op == 'scalar':
        return ('scalar', str(offset.value)), 0
    if offset.op == 'shape':
        return ('shape', tuple(offset.value)), 0
    if offset.op in ('add', 'sub') and len(offset.args) == 2:
        left, right = offset.args

        def const_of(node):
            # 常量项也必须是 int64 域（位宽不同就不在同一个环绕算术域里，不证明）
            if isinstance(node, Expr) and node.op == 'const' \
                    and type(node.value) is int and node.dtype == 'int64':
                return node.value
            return None

        right_const = const_of(right)
        if right_const is not None:
            head = _int_offset_key(left) if isinstance(left, Expr) else _int_offset_key(left)
            if head is None:
                return None
            delta = right_const if offset.op == 'add' else -right_const
            return head[0], head[1] + delta
        if offset.op == 'add':
            left_const = const_of(left)
            if left_const is not None:
                head = _int_offset_key(right)
                if head is None:
                    return None
                return head[0], head[1] + left_const
    return None


def _provably_distinct(left, right):
    """能否证明两次访问覆盖的元素集合不相交。

    两条**充分**判据（任一成立即可）：
    1. 某一维在两边都是**整型常量**且不同（`vel[tid,0]` 与 `vel[tid,1]`）——原有能力，保留；
    2. 某一维**只以常量下标出现**（`AXIS_CONSTANT`），且两边偏移是
       "**同一整型基底 + 不同常量项**"（`fin[.., nx-1, ..]` 与 `fin[.., nx-2, ..]`）。
       该维与并行轴无关，所以对每个 lane 都成立 → **整域**不相交。

    轴偏移不同**不能**当作不相交：`A[i]` 与 `A[i-1]` 整片重叠，只是错开一格。
    浮点参与、cast、非白名单算子、非 int64 位宽 → 无法证明，返回 False。

    **环绕安全**：运行期是 int64 环绕算术，常量差按**模 2**64** 比较。
    `x` 与 `x+H+H+H+H`（H=2**62）在 int64 下是同一地址，Python 无界整数相减会误判，
    这里显式取模。
    """
    if len(left[0]) != len(right[0]):
        return False
    for position, (left_axis, right_axis) in enumerate(zip(left[0], right[0])):
        if left_axis != AXIS_CONSTANT or right_axis != AXIS_CONSTANT:
            continue
        left_offset, right_offset = left[1][position], right[1][position]
        if isinstance(left_offset, int) and isinstance(right_offset, int) \
                and (left_offset - right_offset) % INT64_WRAP != 0:
            # 字面量分支同样按 int64 环绕比较：`0` 与 `2**64` 在运行期是同一地址。
            return True
        left_key = _int_offset_key(left_offset)
        right_key = _int_offset_key(right_offset)
        if left_key is not None and right_key is not None \
                and left_key[0] == right_key[0] \
                and (left_key[1] - right_key[1]) % INT64_WRAP != 0:
            return True
    return False


def _substitute(expr, slot_of_expr):
    """把引用了已物化绑定的子树替换成 `bound` 节点（发射时读 per-lane 缓冲）。"""
    if not isinstance(expr, Expr):
        return expr
    slot = slot_of_expr.get(id(expr))
    if slot is not None:
        return Expr('bound', (expr,), slot, expr.dtype)
    if not expr.args:
        return expr
    new_args = tuple(_substitute(arg, slot_of_expr) for arg in expr.args)
    if all(new is old for new, old in zip(new_args, expr.args)):
        return expr
    return Expr(expr.op, new_args, expr.value, expr.dtype, expr.offsets, expr.scales)



def _require_zero_store_offsets(indices, offsets, label):
    """写回下标的**非常量维**不允许带偏移（`out[i+1]`）。

    发射路径只按轴名渲染写回地址，轴偏移在捕获后就丢掉——165 及以前生成物会静默变成
    `out[i]`。在丢信息之前明确拒绝；常量维（`out[idx, 5]`）与读侧偏移（`x[i-1]`）不受影响，
    threads 路径（保真发射 `i+1`）与 program/归约路径（`capture.store_access` 已拒绝）不走本门。
    """
    for axis, offset in zip(indices, offsets):
        if axis != AXIS_CONSTANT and offset != 0:
            raise Unsupported('写回下标不支持轴偏移（发射时会丢失信息）: %s' % label)


def _expr_contains_read(node):
    """表达式子树里是否含张量读（递归 args / offsets / scales）。"""
    if not isinstance(node, Expr):
        return False
    if node.op == 'read':
        return True
    if any(_expr_contains_read(arg) for arg in node.args):
        return True
    return any(_expr_contains_read(item)
               for item in tuple(node.offsets or ()) + tuple(node.scales or ()))


def _subscript_nested_in_address(node):
    """AST 层：下标表达式里是否还有“下标里再套下标”（多级 gather 的语法形态）。

    例：`src[a[b[i]]]` 的地址是 `a[b[i]]`（Subscript 的 slice 里又含 Subscript）→ True；
    `src[index[i]]` 的地址是 `index[i]`（slice 里没有 Subscript）→ False。
    """
    if not isinstance(node, ast.Subscript):
        return False
    items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    for item in items:
        for sub in ast.walk(item):
            if isinstance(sub, ast.Subscript):
                inner = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
                if any(isinstance(x, ast.Subscript) for x in inner):
                    return True
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Subscript):
                return True
    return False


def _has_read_in_address(offsets, scales):
    """read 的 offsets/scales 里是否嵌套了张量读（gather 判定用，含复合表达式）。"""
    return any(_expr_contains_read(item)
               for item in tuple(offsets or ()) + tuple(scales or ()))


# K 上限按用户裁定保守收窄到 32767（SPEC-GATHER-G2A-v3 附录 D）；C 仍为 0..2^31-1。
# 这是保守支持域选择，不声称 1..32767 已被穷尽证明；K=32768 由本门显式拒绝。
GATHER_K_MAX = 32767
GATHER_C_MAX = 2 ** 31 - 1


def _gather_int_literal(expr):
    """gather 地址里的整数字面量：必须恰为 Python int（排除 bool）；否则 None。"""
    if not isinstance(expr, Expr) or expr.op != 'const':
        return None
    if type(expr.value) is not int:
        return None
    return expr.value


def _match_affine_gather_offset(offset):
    """把 `index[i]*K + C` / `index[i]*K` 解析成 (read, K, C)；其它形态返回 None。

    只接受这一种书写顺序（读在左、常量在右；加在外、乘在内），不做代数化简：
    `K * index[i]`、`C + index[i]*K`、`index[i]*K - C` 均不匹配，交由调用方按不支持拒绝。
    """
    if not isinstance(offset, Expr):
        return None
    if offset.op == 'mul':
        left, right = offset.args
        factor = _gather_int_literal(right)
        if factor is None or not (isinstance(left, Expr) and left.op == 'read'):
            return None
        return left, factor, 0
    if offset.op == 'add' and len(offset.args) == 2:
        product, constant_expr = offset.args
        constant = _gather_int_literal(constant_expr)
        if constant is None or not (isinstance(product, Expr) and product.op == 'mul'):
            return None
        left, right = product.args
        factor = _gather_int_literal(right)
        if factor is None or not (isinstance(left, Expr) and left.op == 'read'):
            return None
        return left, factor, constant
    return None


def _capture_gather(name, dtype, indices, offsets, scales, params, dtypes):
    """gather 一期（20261026-173）：`src[index[i]]` 的窄资格判定。

    只接受单维、`offset` 恰为一个 int32/int64 张量读；src 仅 float32；
    src 与 index 不得同名；index 必须以**恒等下标**读取（禁 `index[i±c]`/系数）；
    多级 gather（index 的下标里含张量读）显式拒绝。其余形态（复合运算、常量位置读、多维）同样拒绝。

    注意（F2/F3 口径）：本门只约束 **src/index**；`out` 的 dtype 沿用既有逐元素转换规则；
    gather 读可出现在普通表达式与 guard 中，每个读各自过门。
    """
    if len(indices) != 1 or indices[0] != AXIS_CONSTANT:
        raise Unsupported('gather 一期只支持一维 src[index[i]]')
    if len(offsets) != 1 or not isinstance(offsets[0], Expr):
        raise Unsupported('gather 只支持下标恰为一个张量读或 index[i]*K + C（含经局部别名的写法）')
    if offsets[0].op == 'read':
        index_read = offsets[0]
        factor, constant = 1, 0
    else:
        matched = _match_affine_gather_offset(offsets[0])
        if matched is None:
            raise Unsupported('gather 二期只支持 index[i]*K + C 的常量仿射地址（K 正整数、C 非负整数）')
        index_read, factor, constant = matched
        if not (1 <= factor <= GATHER_K_MAX):
            raise Unsupported('gather 的步长 K 必须是 1..32767 的整数字面量（bool 不算；保守支持域，见 SPEC 附录 D）')
        if not (0 <= constant <= GATHER_C_MAX):
            raise Unsupported('gather 的偏移 C 必须是 0..2^31-1 的整数字面量（bool 不算）')
        if factor == 1 and constant == 0:
            # `index[i]*1` 与裸 `index[i]` 是同一地址形态：归一为 P1a 结构，产物应与裸式一致。
            offsets = (index_read,)
    # F4：多级 gather（`src[a[b[i]]]`）——index 读的下标里又含张量读 → 显式拒绝，
    # 不依赖 dtype 组合“碰巧”被拒。
    if _has_read_in_address(index_read.offsets, index_read.scales):
        raise Unsupported('gather 一期不支持多级 gather（index 的下标里含张量读）')
    # F1：index 自身的下标必须是恒等（无轴偏移、无系数），否则 index 张量会越界读。
    for position, axis in enumerate(index_read.value[1]):
        offset = index_read.offsets[position] if position < len(index_read.offsets) else 0
        scale = index_read.scales[position] if position < len(index_read.scales) else 1
        if isinstance(offset, int) and offset == 0 and not isinstance(scale, tuple) and scale == 1:
            continue
        raise Unsupported('gather 一期要求 index 以恒等下标读取（不允许 index[i±c] 或系数）')
    index_name, index_indices = index_read.value
    if index_name not in params:
        raise Unsupported('gather 的 index 必须是形参: ' + index_name)
    if index_name == name:
        raise Unsupported('gather 的 src 与 index 不能是同一形参')
    if any(axis == AXIS_CONSTANT for axis in index_indices):
        raise Unsupported('gather 一期只支持 index 以计算轴下标读取')
    index_dtype = (dtypes or {}).get(index_name, 'float32')
    if index_dtype not in ('int32', 'int64'):
        raise Unsupported('gather 一期要求 index 为 int32/int64 整数张量（无签名默认 float32，拒绝）')
    if dtype != 'float32':
        raise Unsupported('gather 一期 src/out 仅支持 float32')
    return Expr('read', value=(name, indices), offsets=offsets, scales=scales,
                dtype=dtype, indirect=True)


def _normalize_store_indices(subscript, indices, offsets, axes, params, dtypes,
                             scalar_env=None):
    """写回下标里，若某一维解析出的"轴名"**不是并行轴**，说明它其实是运行期地址
    （`fin[.., nx-1, ..]`）：改记为常量下标维 + 捕获后的表达式。
    否则这个假轴名会流进 layout/依赖检查，既推不出形状也判不了不相交。"""
    items = subscript.slice.elts if isinstance(subscript.slice, ast.Tuple)         else [subscript.slice]
    if len(items) != len(indices):
        return indices, offsets
    new_indices, new_offsets = [], []
    for item, axis, offset in zip(items, indices, offsets):
        if axis != AXIS_CONSTANT and axis not in set(axes):
            new_indices.append(AXIS_CONSTANT)
            new_offsets.append(capture_expr(item, dict(scalar_env or {}), params,
                                            dtypes, axes))
        else:
            new_indices.append(axis)
            new_offsets.append(offset)
    return tuple(new_indices), tuple(new_offsets)


def _fold_branch_chain(statement, bindings, params, dtypes, axes):
    """`if/elif/else` 纯标量绑定 → 逐名右折叠 `select`（**旧环境**语义）。

    源语义：后序分支只在前序条件全假时才求值，此时目标变量仍是最初值。因此
    **所有条件与分支值都用 If 语句之前的 bindings 捕获**，绝不能看到前面分支的赋值。
    例：`if a < 0: x = 2.0 elif x > 1.0: x = 3.0`（旧 x=0、a<0）源语义 x=2.0；
    若先折叠第一支再用新 x 求值第二支会得到 3.0——本实现按旧环境，得 2.0。

    收窄（证明不出来就拒绝）：
      - 每个分支体（含 else）**恰好一条** `Name = 纯表达式`；
      - 所有分支写**同一个名字**，且该名字在 If 之前已初始化；
      - 不允许元组解包、AugAssign、嵌套 if/循环、张量写。
    """
    chain, node, final = [], statement, None
    while True:
        chain.append((node.test, node.body))
        orelse = node.orelse
        if len(orelse) == 1 and isinstance(orelse[0], ast.If):
            node = orelse[0]
            continue
        final = orelse
        break
    old = dict(bindings)
    name = None
    parts = []
    bodies = [(test, body) for test, body in chain]
    if final:
        bodies.append((None, final))
    for test, body in bodies:
        where = '条件分支内只允许纯标量绑定' if test is not None else 'else 分支内只允许纯标量绑定'
        if len(body) != 1:
            raise Unsupported('elif/else 折叠要求每个分支恰好一条标量绑定')
        inner = body[0]
        if not (isinstance(inner, ast.Assign) and len(inner.targets) == 1
                and isinstance(inner.targets[0], ast.Name)):
            raise Unsupported(where)
        target = inner.targets[0].id
        if name is None:
            name = target
            if name not in old:
                raise Unsupported('条件绑定的变量必须有循环前的初值: ' + name)
            if name in set(params) | set(axes):
                raise Unsupported('条件绑定不能遮蔽张量或索引: ' + name)
        elif target != name:
            raise Unsupported('elif/else 折叠要求所有分支写同一个变量: %s / %s'
                              % (name, target))
        value = capture_expr(inner.value, old, params, dtypes, axes)
        if test is None:
            parts.append((None, value))
            continue
        condition = capture_expr(test, old, params, dtypes, axes)
        if condition.dtype != 'bool':
            raise Unsupported('条件绑定的条件必须产生 bool')
        parts.append((condition, value))
    if name is None:
        raise Unsupported('条件绑定需要至少一个分支')
    folded = old[name]
    rest = parts if parts[-1][0] is not None else parts[:-1]
    if parts[-1][0] is None:
        folded = parts[-1][1]
    for condition, value in reversed(rest):
        folded = Expr('select', (condition, value, folded),
                      dtype=numeric_dtype(value, folded))
    return name, folded


def capture_pointwise(body, params, axes, initial=None, dtypes=None):
    if not body:
        raise Unsupported('缺少输出写回')
    # 有序写回：允许"写回 → 局部值 → 再写回"交错（inflow 的 u→t2/t3→rho——
    # 后一次写回要读前一次刚写的值）。第一个写回是主输出，其余按源码顺序进入
    # extra_stores；**依赖分组与物化判据完全沿用下面的既有逻辑**：
    # 同一并行域内 lane→元素映射相同，同元素写后读靠拆组保证顺序，跨元素一律拒绝。
    stores = []                     # (赋值语句, **当时冻结的** 值表达式)
    bindings = dict(initial or {})
    # 按定义顺序保留（`h = f(h)` 这类链式更新会重复定义同一个名字）；
    # 捕获时用 dict 查最新值，发射时按序列逐条产出，等价于 SSA 重命名。
    ordered = []
    # **显式绑定节点**：nodes[i] = (name, value_expr, defined_after)。
    # 名字与 Expr 地址都不能单独当定义身份：
    #   - 别名 `s = t` 复用同一个 Expr 对象 → **共享** node（不新增定义、不改定义点）；
    #   - 同名重绑定 `t = A[i]` 两次 → capture_expr 产生新对象 → **新建 node**。
    nodes = []
    node_of_expr = {}
    name_node = {}
    store_count = 0

    def define(name, value, alias_of=None):
        """登记一次绑定定义，返回 node_id。

        别名判定看**源码 RHS 是否引用已有局部值版本**（`s = t`），不看 Expr 地址：
        直接重读 `A[i]` 是新的求值事件，即使将来 Expr 被驻留也不能共享旧定义点。
        """
        nonlocal store_count
        if alias_of is not None:
            node_id = alias_of                      # 别名：共享版本与定义时刻
        else:
            node_id = len(nodes)
            nodes.append((name, value, store_count))
            node_of_expr.setdefault(id(value), node_id)
        name_node[name] = node_id
        return node_id
    for statement in body:
        # 增量写回与两分支写回先归一（`x[i] += v` → `x[i] = x[i] + v`）
        statement = merge_branch_store(expand_augassign(statement))
        if isinstance(statement, ast.If):
            # 条件绑定（Sobel 的 `if 内部: gx = ...`）：无 else 时按 `x = a if cond else x` 折叠，
            # 要求 x 在本语句之前已定义，否则语义上可能未初始化。
            if statement.orelse:
                # 20261025-148：`if/elif/else` 纯标量绑定的右折叠（见 `_fold_branch_chain`）。
                name, folded = _fold_branch_chain(statement, bindings, params, dtypes, axes)
                bindings[name] = folded
                ordered.append((name, folded))
                define(name, folded)
                continue
            condition = capture_expr(statement.test, bindings, params, dtypes, axes)
            if condition.dtype != 'bool':
                raise Unsupported('条件绑定的条件必须产生 bool')
            for inner in statement.body:
                if not (isinstance(inner, ast.Assign) and len(inner.targets) == 1
                        and isinstance(inner.targets[0], ast.Name)):
                    raise Unsupported('条件分支内只允许纯标量绑定')
                name = inner.targets[0].id
                if name not in bindings:
                    raise Unsupported('条件绑定的变量必须有循环前的初值: ' + name)
                if name in set(params) | set(axes):
                    raise Unsupported('条件绑定不能遮蔽张量或索引: ' + name)
                value = capture_expr(inner.value, bindings, params, dtypes, axes)
                merged = Expr('select', (condition, value, bindings[name]),
                              dtype=numeric_dtype(value, bindings[name]))
                bindings[name] = merged
                ordered.append((name, merged))
                define(name, merged)
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                and isinstance(statement.targets[0], ast.Subscript):
            # 写回：**立即**用当时的 bindings 冻结表达式。
            # 这是有序语义的关键——后面再绑定的变量不会回溯影响这次写回。
            stores.append((statement, capture_expr(statement.value, bindings, params,
                                                   dtypes, axes)))
            store_count += 1
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 \
                or not isinstance(statement.targets[0], ast.Name):
            raise Unsupported('写回之间只允许纯标量绑定与张量写回')
        name = statement.targets[0].id
        if name in set(params) | set(axes) | {'cuda', 'range'}:
            raise Unsupported('局部绑定不能遮蔽张量或索引: ' + name)
        value = capture_expr(statement.value, bindings, params, dtypes, axes)
        alias_of = None
        if isinstance(statement.value, ast.Name) and statement.value.id in name_node:
            alias_of = name_node[statement.value.id]   # `s = t`：共享 t 的版本
        bindings[name] = value
        ordered.append((name, value))
        define(name, value, alias_of)
    if not stores:
        raise Unsupported('需要唯一的直接张量写回')
    output, full_indices, offsets, scales = tensor_access(stores[0][0].targets[0],
                                                          with_offsets=True)
    full_indices, offsets = _normalize_store_indices(
        stores[0][0].targets[0], full_indices, offsets, axes, params, dtypes,
        scalar_env=initial)
    if any(scale != 1 for scale in scales):
        raise Unsupported('写回下标不能带系数')
    _require_zero_store_offsets(full_indices, offsets,
                                ast.unparse(stores[0][0].targets[0]))
    indices = tuple(axis for axis in full_indices if axis != AXIS_CONSTANT)
    constants = tuple(offset if axis == AXIS_CONSTANT else None
                      for axis, offset in zip(full_indices, offsets))
    # 写回表达式在扫描时已冻结，这里**不再重新捕获**（否则会用终态 bindings 覆盖早期语义）。
    expression = stores[0][1]
    extras = []
    extra_accesses = []
    for store_node, extra_value in stores[1:]:
        name, used, extra_offsets, extra_scales = tensor_access(store_node.targets[0],
                                                                with_offsets=True)
        used, extra_offsets = _normalize_store_indices(
            store_node.targets[0], used, extra_offsets, axes, params, dtypes,
            scalar_env=initial)
        if any(scale != 1 for scale in extra_scales):
            raise Unsupported('写回下标不能带系数')
        _require_zero_store_offsets(used, extra_offsets,
                                    ast.unparse(store_node.targets[0]))
        consts = tuple(offset if axis == AXIS_CONSTANT else None
                       for axis, offset in zip(used, extra_offsets))
        extras.append((name, used, consts, extra_value))
        extra_accesses.append((name, used, extra_offsets, extra_scales, extra_value))
    # 写回之间的数据依赖（leapfrog 的 `pos += vel*dt`、LBM 的碰撞后迁移）：
    # 后面的写回要读前面刚写的值，拆成**多个顺序并行域**，每组发射一个 T.Parallel。
    # 只有"读写同一元素"才是安全的：两个 T.Parallel 的 lane→元素映射相同，
    # 同一元素落在同一线程里，程序序保证后者看到前者的写。读写不同元素
    # （LBM 迁移读邻居）需要跨线程同步，tile 级并行域不提供，明确拒绝。
    # 局部绑定（`f0 = f[0,j,i]`）保存的是**写回前的原值**：绑定在发射时被内联成
    # 直接读取，一旦拆分并行域，后面的组会读到新值（bounce-back 的交换对全错）。
    # 记下这些 Expr 的对象身份，依赖检查时据此区分"读绑定"与"直接读"。
    binding_ids = {id(value) for _name, value in ordered}
    write_targets = {output} | {item[0] for item in extras}
    all_stores = [(output, full_indices, offsets, scales, expression)] + extra_accesses
    keys = [ast.dump(stores[0][0].targets[0])] + [
        ast.dump(node.targets[0]) for node, _value in stores[1:]]

    # ---- 需要物化的绑定：它读的张量也在写回集合里 → 值必须在**定义点**冻结 ----
    needing = []
    for node_id, (bname, bvalue, defined_after) in enumerate(nodes):
        # 20261025-152：定长元组的各分量由 `helper_inline` 在**原赋值位置**物化成
        # `_tl_tp_*` 绑定；它们必须在定义点冻结（元组在赋值处求值），不能因只读张量
        # 而被重新内联到调用处。只对我们自己合成的名字生效。
        if bname.startswith('_tl_tp_')                 or any(read_name in write_targets for read_name, _ in reads(bvalue)):
            needing.append((node_id, bname, bvalue, defined_after))
    # 物化**不能**绕过跨 lane 合法性：按它定义点之前已发生的写回逐条核对。
    for _index, bname, bvalue, stage in needing:
        for node in read_accesses(bvalue):
            read_name, read_idx, read_offsets, read_scales = _access(node)
            for ordinal in range(min(stage, len(all_stores))):
                name, store_idx, store_offsets, store_scales, _v = all_stores[ordinal]
                if read_name != name:
                    continue
                if _same_element((read_idx, read_offsets, read_scales),
                                 (store_idx, store_offsets, store_scales)):
                    continue
                if _provably_distinct((read_idx, read_offsets, read_scales),
                                      (store_idx, store_offsets, store_scales)):
                    continue
                raise Unsupported(
                    '局部值 %s 在写回之后读取了同一张量的其它元素（%s）：'
                    '需要跨线程同步，tile 并行域不支持' % (bname, read_name))
    # 在"绑定定义点"建立分组切点：所有在该点之前的写回先成一组，
    # 物化在该组之后立刻取值——这正是"写回 → 局部值 → 再写回"的顺序。
    cut_positions = {stage for _i, _n, _v, stage in needing
                     if 0 < stage < len(all_stores)}
    materialized = [(bname, bvalue) for _i, bname, bvalue, _s in needing]
    # id(绑定表达式) → 缓冲槽位；所有引用（包括派生绑定与写回）都替换成 `bound`
    slot_of_expr = {id(bvalue): slot
                    for slot, (_i, _n, bvalue, _s) in enumerate(needing)}
    # 绑定自身的根表达式不能替换成自己的 `bound`（否则自引用）
    # 按**节点**重写（ordered 与 nodes 不是一一对应：别名共享节点）
    rewritten_node_values = [
        _substitute(value, {key: slot for key, slot in slot_of_expr.items()
                            if key != id(value)})
        for _name, value, _stage in nodes]
    # 兼容字段 Compute.bindings：按定义顺序给名字→值（别名给同一个值）
    rewritten_values = [_substitute(value, slot_of_expr) for _name, value in ordered]
    # **所有** store（主输出 + 附加输出）统一回填重写后的表达式，避免两套表示
    extras = [(name, idx, consts, _substitute(value, slot_of_expr))
              for name, idx, consts, value in extras]
    all_stores = [(name, idx, offs, scales_, _substitute(value, slot_of_expr))
                  for name, idx, offs, scales_, value in all_stores]
    expression = all_stores[0][4]

    groups, current, group_keys = [], [], set()
    # 每个 store 序号 → 它所在组的组号（组首序号也记下来，供绑定归组）
    group_of = {}
    group_starts = []
    # 已写位置：(地址, 所属组号)。跨组同元素读合法，跨组跨元素读必须拒绝。
    written_by = {}
    for position, (name, store_idx, store_offsets, store_scales, value) in enumerate(all_stores):
        # 绑定定义点强制切组：保证物化值在"之前的写回之后、之后的写回之前"取值
        if position in cut_positions and current:
            groups.append(tuple(current))
            current, group_keys = [], set()
        if not current:
            group_starts.append(position)
        value_accesses = read_accesses(value)
        split = False
        for node in value_accesses:
            read_name, read_idx, read_offsets, read_scales = _access(node)
            # 同一次写回里读写同一张量的**不同元素**（`f[i] = f[i-1]`）：lane 之间
            # 没有先后关系，tile 并行域也不是双缓冲，生成出来就是数据竞争。
            if read_name == name and not _same_element(
                    (read_idx, read_offsets, read_scales),
                    (store_idx, store_offsets, store_scales)) \
                    and not _provably_distinct(
                        (read_idx, read_offsets, read_scales),
                        (store_idx, store_offsets, store_scales)):
                # 拒绝原因要能区分"确定重叠"与"无法证明"（比如偏移的整型语义未知）
                unknown_dtype = any(
                    isinstance(offset, Expr) and offset.dtype not in INT_DTYPES
                    for offset, axis in ((read_offsets[position], read_idx[position])
                                         for position in range(len(read_idx))
                                         if position < len(read_offsets)))
                if unknown_dtype:
                    raise Unsupported(
                        '无法证明写回与读取不相交（%s）：地址偏移里含有非整型或类型未知的基底，'
                        '不猜测整型语义；tile 并行域需要可证明的域不相交' % name)
                raise Unsupported(
                    '写回读取了同一张量的其它元素（%s）：lane 之间无先后关系，'
                    '需要显式分块与同步，tile 并行域不支持' % name)
            for written, written_group in written_by.get(read_name, ()):
                if _same_element((read_idx, read_offsets, read_scales), written):
                    # 同地址：同组内要按顺序看到新值 → 拆组；跨组由线程内程序序保证。
                    if written_group == len(groups):
                        split = True
                    continue
                if _provably_distinct((read_idx, read_offsets, read_scales), written):
                    continue          # 另一维常量不同，写的是别的元素，与本次读无关
                raise Unsupported(
                    '写回之间存在跨元素数据依赖（%s 的读下标与前面的写回不同）：'
                    '需要跨线程同步，tile 并行域不提供' % read_name)
        if split:
            groups.append(tuple(current))
            current, group_keys = [], set()
        if keys[position] in group_keys:
            raise Unsupported('同一位置不能在一次并行域内被写两次: ' + name)
        group_keys.add(keys[position])
        current.append(position)
        group_of[position] = len(groups)
        # 本 store 的写回落在当前组（= 已闭合的组数）。
        written_by.setdefault(name, []).append(
            ((store_idx, store_offsets, store_scales), len(groups)))
    groups.append(tuple(current))

    # ---- 物化绑定归组：按定义点所在组，**定义顺序**发射 ----
    per_group = [[] for _ in groups]
    for slot, (node_id, bname, _bvalue, stage) in enumerate(needing):
        group = 0 if stage == 0 else group_of.get(stage, len(groups) - 1)
        per_group[group].append((slot, bname, rewritten_node_values[node_id]))
    group_bindings = tuple(tuple(items) for items in per_group)
    # 签名标量不是"局部绑定"，不进入 bindings 元组（避免与形参遮蔽检查冲突）。
    scalar_names = set(initial or {})
    return Compute(expression, output, indices,
                   tuple((name, rewritten_values[index])
                         for index, (name, _value) in enumerate(ordered)
                         if name not in scalar_names),
                   extra_stores=tuple(extras),
                   store_indices=full_indices, store_constants=constants,
                   store_groups=tuple(groups),
                   materialized_bindings=tuple(materialized),
                   group_bindings=group_bindings)


def project(indices, allowed, what):
    """校验读取下标是 allowed 轴集合中的互异变量；重复轴是按对角线寻址，拒绝。

    常量下标维（AXIS_CONSTANT）不参与匹配：它取固定偏移，与并行域无关。
    """
    axes_used = [axis for axis in indices if axis != AXIS_CONSTANT]
    if len(set(axes_used)) != len(axes_used) or any(axis not in allowed for axis in axes_used):
        raise Unsupported(f'{what}的下标必须是互异且属于计算轴的变量')
    return indices


def _contains_indirect_read(expressions):
    """表达式集合里是否含 gather（indirect read）节点。"""
    for expression in expressions:
        stack = [expression]
        while stack:
            node = stack.pop()
            if not isinstance(node, Expr):
                continue
            if node.op == 'read' and getattr(node, 'indirect', False):
                return True
            stack.extend(arg for arg in node.args if isinstance(arg, Expr))
            for item in tuple(node.offsets or ()) + tuple(node.scales or ()):
                if isinstance(item, Expr):
                    stack.append(item)
    return False


def _address_reads(node):
    """gather 读的 offsets/scales 子树里所有张量读（递归，覆盖 `index[i]*K + C` 这类复合地址）。"""
    found = []
    stack = [item for item in tuple(node.offsets or ()) + tuple(node.scales or ())
             if isinstance(item, Expr)]
    while stack:
        item = stack.pop()
        if item.op == 'read':
            found.append(item)
        stack.extend(arg for arg in item.args if isinstance(arg, Expr))
        stack.extend(extra for extra in tuple(item.offsets or ()) + tuple(item.scales or ())
                     if isinstance(extra, Expr))
    return found


def _reject_gather_on_outputs(spec, expressions):
    """gather 一期：src/index 不得是输出张量（不许写输入、不许用输出当索引）。"""
    outputs = set(output_names(spec)) or {getattr(spec, 'store_target', None)}
    outputs.discard(None)
    for expression in expressions:
        stack = [expression]
        while stack:
            node = stack.pop()
            if node is None or not isinstance(node, Expr):
                continue
            if node.op == 'read' and getattr(node, 'indirect', False):
                if node.value[0] in outputs:
                    raise Unsupported('gather 一期不允许 src 是输出张量: ' + str(node.value[0]))
                for item in _address_reads(node):
                    if item.value[0] in outputs:
                        raise Unsupported('gather 一期不允许 index 是输出张量: ' + str(item.value[0]))
            stack.extend(arg for arg in node.args if isinstance(arg, Expr))
            for item in tuple(node.offsets or ()) + tuple(node.scales or ()):
                if isinstance(item, Expr):
                    stack.append(item)


def validate_map(spec):
    compute = spec.compute
    axes = compute.output_axes
    if spec.grid_dims not in (1, 2, 3) or len(axes) != spec.grid_dims \
            or len(set(axes)) != len(axes) or set(axes) != set(spec.idx_vars) \
            or len(spec.idx_vars) != spec.grid_dims:
        raise Unsupported('逐元素输出需要一到三个独立grid轴')
    # 附加输出（`x[i]=i; y[i]=i`）与主输出共享同一并行域，下标必须完全一致。
    extra = tuple(compute.extra_stores or ())
    for name, indices, _consts, _ in extra:
        if tuple(axis for axis in indices if axis != AXIS_CONSTANT) != tuple(axes):
            raise Unsupported('附加输出的写回下标必须与输出域一致: ' + name)
    expressions = [compute.expression] + [expression for _, _, _, expression in extra]
    # 单投影接口无法描述多投影读取；明确选择程序级约束分析，不吞异常回退。
    accesses = []
    for expression in expressions:
        accesses += reads(expression)
    for _, bound in compute.bindings:
        accesses += reads(bound)
    projections = {}
    for name, indices in accesses:
        projections.setdefault(name, set()).add(indices)
    if any(len(values) > 1 for values in projections.values()):
        from capture import Program
        from program import validate_program
        # 附加写回的读取也必须进入程序级分析：否则同一物理维上的运行期偏移
        # （`QKV[i, half + d]`）看不见，layout 会把该维错误地绑到输出轴。
        spec.program = Program((), compute.output, axes, compute.expression, compute.bindings,
                               extra_stores=tuple(compute.extra_stores or ()),
                               store_indices=tuple(getattr(compute, 'store_indices', ()) or ()),
                               store_constants=tuple(getattr(compute, 'store_constants', ()) or ()))
        validate_program(spec)
        spec.axis_roles = {axis: f'D{position}' for position, axis in enumerate(axes)}
        return spec
    # 低秩读取按轴名投影到输出轴（广播），形状由此推导而不是照抄输出形状。
    read_axes = {}
    for expression in expressions:
        for name, indices in reads(expression):
            if read_axes.setdefault(name, project(indices, axes, '逐元素读取')) != indices:
                raise Unsupported('同一输入不能以不同轴映射重复读取')
    for _, bound in compute.bindings:
        for name, indices in reads(bound):
            if read_axes.setdefault(name, project(indices, axes, '局部绑定读取')) != indices:
                raise Unsupported('同一输入不能以不同轴映射重复读取')
    # 元素级掩码（`if solid_mask[i, j]:`）：条件只读本元素的输入，属于范围之内。
    # 捕获成表达式后并入读取集合，rank/形状校验与普通读取一致。
    # 注意：这是**逐元素（grid）入口**，掩码由 lower_map 正常发射；
    # R3 的“展平入口丢弃掩码”门只加在 validate_flat_map，不能加在这里。
    mask_expressions = []
    for text in getattr(spec, 'element_masks', ()) or ():
        mask_node = ast.parse(text, mode='eval').body
        mask_expr = capture_expr(mask_node, {}, spec.params, getattr(spec, 'tensor_dtypes', None), axes)
        if mask_expr.dtype != 'bool':
            # `if solid_mask[i, j]:` 对数值张量是"非零为真"（与源码的 Python 真值一致）。
            mask_expr = Expr('ne', (mask_expr, Expr('const', value=0, dtype=mask_expr.dtype)),
                             dtype='bool')
        mask_expressions.append(mask_expr)
        for name, indices in reads(mask_expr):
            project(indices, axes, '元素掩码读取')
            if read_axes.setdefault(name, indices) != indices:
                raise Unsupported('元素掩码的张量必须以同一轴映射读取: ' + name)
    spec.element_mask_expressions = tuple(mask_expressions)
    # 纯常量写回（如 `x[i] = 3.14` 的初始化 kernel）：tile 级同样是完整并行写，
    # 不读取任何输入不构成语义缺口。形状由输出与guard提供，读不到任何输入时
    # 也就没有别的张量形参能通过下面的"形参必须恰好对应"检查。
    # S-C：guard（element mask）里也可能出现 gather；角色检查与附加输出限制必须覆盖同一集合，
    # 否则 guard 中的 `out[...]` 读写冲突/附加输出会绕过门禁（175 起即存在的漏门）。
    # 只复用已捕获的掩码表达式，不重新解析，也不改变普通非 gather 部分写的语义。
    gather_expressions = ([compute.expression] + [value for _, value in compute.bindings]
                          + list(getattr(spec, 'element_mask_expressions', ()) or ()))
    _reject_gather_on_outputs(spec, gather_expressions)
    if (compute.extra_stores or ()) and _contains_indirect_read(gather_expressions):
        raise Unsupported('gather 一期不支持附加输出/多写回')
    validate_names(spec, set(read_axes), set(axes) | {name for name, _ in compute.bindings})
    from program import walk
    for expression in [compute.expression] + [value for _, value in compute.bindings]:
        for node in walk(expression):
            if node.op == 'shape':
                name, dim = node.value
                rank = len(axes) if name == compute.output else len(read_axes.get(name, ()))
                if dim >= rank:
                    raise Unsupported('shape引用没有对应张量维度')
    # 输出张量的形状映射必须按**完整写回下标**算：`f[k,j,i]` 的 k 是常量维，
    # 后面两维才是并行轴（`f.shape[1]` 对应 j、`f.shape[2]` 对应 i）。
    store_indices = getattr(compute, 'store_indices', ()) or tuple(axes)
    validate_guard(spec, {(name, dim): axis for name, axes_used in read_axes.items()
                         for dim, axis in enumerate(axes_used)}
                        | {(compute.output, dim): axis
                           for dim, axis in enumerate(store_indices)
                           if axis != AXIS_CONSTANT})
    spec.read_axes = read_axes
    spec.pattern = 'elementwise'
    spec.axis_roles = {axis: f'D{position}' for position, axis in enumerate(axes)}
    return spec


def validate_sum(spec):
    compute = spec.compute
    reduction = compute.reduction
    axes = tuple(compute.output_axes)
    if spec.grid_dims not in (1, 2, 3) or len(axes) != spec.grid_dims \
            or len(set(axes)) != len(axes) or set(axes) != set(spec.idx_vars) \
            or len(spec.idx_vars) != spec.grid_dims:
        raise Unsupported('普通归约需要一到三个独立grid轴')
    axis = reduction.axis
    if len({*axes, axis, reduction.accumulator}) != len(axes) + 2:
        raise Unsupported('归约轴、输出轴及累加器必须互异')
    # 归约体是一个表达式；每个输入可取计算轴（输出轴 + 归约轴）的任意子集（广播）。
    combined = axes + (axis,)
    read_axes = {}
    for name, indices in reads(compute.expression):
        if read_axes.setdefault(name, project(indices, combined, '归约读取')) != indices:
            raise Unsupported('同一输入不能以不同轴映射重复读取')
    if not read_axes:
        raise Unsupported('归约表达式必须至少读取一个输入')
    # 旧单归约/GEMM接口只表达每个输入的一种轴投影；未使用绑定也不能引入被丢弃的约束。
    for _, bound in compute.bindings:
        for name, indices in reads(bound):
            project(indices, combined, '局部绑定读取')
            if read_axes.get(name) != indices:
                raise Unsupported('单归约特化的局部绑定不能引入额外输入或不同轴映射')
    # GEMM 的一维偏置（`acc + bias[j]`）只出现在写回后处理里，不参与归约域；
    # 但它是合法形参，必须让"形参恰好对应"的检查看见它。
    consumed = set(read_axes)
    for _sign, operand in (getattr(spec, 'gemm_post', ()) or ()):
        if operand.op == 'read':
            consumed.add(operand.value[0])
    validate_names(spec, consumed, {*axes, axis, reduction.accumulator})
    if reduction.op not in REDUCTION_IDENTITY:
        raise Unsupported('未支持的归约操作: ' + reduction.op)
    identity = REDUCTION_IDENTITY[reduction.op]
    if reduction.init.op != 'const':
        # 动态初值（`m = F[b*win, i]` 再从窗口第二个元素起取 max）：它必须只依赖输出坐标，
        # 否则每个 lane 的归约起点不同，块内定长归约不成立。
        if axis in referenced_axes(reduction.init):
            raise Unsupported('归约初值不能依赖归约轴')
    elif reduction.init.value != identity:
        # float32 数据不可能小于 -3.4028235e38：`-1e300` 对 max 等价于 -inf 单位元。
        value = reduction.init.value
        dtypes = dict(getattr(spec, 'tensor_dtypes', {}) or {})
        float32_only = all(dtypes.get(name, 'float32') == 'float32' for name in read_axes)
        equivalent = (type(value) is float and float32_only and reduction.op == 'max'
                      and value <= -3.4e38) or                      (type(value) is float and float32_only and reduction.op == 'min'
                      and value >= 3.4e38)
        if not equivalent:
            raise Unsupported(f'{reduction.op} 归约的累加器初值必须是它的单位元 {identity}')
    if reduction.step != '1':
        raise Unsupported('只支持步长1的归约')
    start_expr = getattr(reduction, 'start_expr', None)
    start_poly = None if start_expr is None else polynomial(start_expr)
    if start_expr is not None and start_poly is None:
        raise Unsupported('归约起点必须是整型的加减乘组合')
    if start_poly:
        # 非零起点（窗口归约 `range(b*win, (b+1)*win)`）：并行域长度必须与归约轴无关，
        # 否则每个输出元素的归约长度不同，块内定长归约就不成立。
        if any(axis in key for key in start_poly):
            raise Unsupported('归约起点不能依赖归约轴本身')
        stop_expr = getattr(reduction, 'stop_expr', None)
        stop_poly = None if stop_expr is None else polynomial(stop_expr)
        if stop_poly is None:
            raise Unsupported('非零起点归约的终点必须是整型加减乘组合')
        length = dict(stop_poly)
        for key, value in start_poly.items():
            merged = length.get(key, 0) - value
            if merged:
                length[key] = merged
            else:
                length.pop(key, None)
        if any(axis in key for key in length):
            raise Unsupported('归约域长度必须与归约轴无关，否则无法在构造期确定并行域')
        if all(not key for key in length):
            value = length.get((), 0)
            if value <= 0:
                raise Unsupported('归约域长度必须是正整数')
            spec.k_constant = value
        else:
            # 长度是运行期标量表达式：只能要求调用方传入相符的 reduction_extent。
            spec.extent_expression = polynomial_text(length)
    else:
        stop = ast.parse(reduction.stop, mode='eval').body
        carriers = {(name, indices.index(axis)) for name, indices in read_axes.items() if axis in indices}
        if shape_axis(stop) not in carriers:
            if isinstance(stop, ast.Constant) and type(stop.value) is int and stop.value > 0:
                spec.k_constant = stop.value
            elif isinstance(stop, ast.Name) and stop.id in (spec.scalar_params or {})                     and spec.scalar_params[stop.id] in INT_DTYPES:
                # `for j in range(n)`：n 是签名整数标量（N-body 类项目的常见写法）。
                spec.reduction_extent_scalar = stop.id
            else:
                raise Unsupported('归约终点必须是依赖归约轴的输入shape、正常量或签名整数标量')
    shapes = {(name, dim): used for name, used in read_axes.items()
              for dim, used in enumerate(used)}
    shapes.update({(spec.store_target, dim): used for dim, used in enumerate(axes)})
    validate_guard(spec, shapes)
    spec.read_axes = read_axes
    spec.axis_roles = {used: f'D{dim}' for dim, used in enumerate(axes)} | {axis: 'R0'}
    spec.pattern = 'reduce_' + reduction.op
    return spec


def validate_names(spec, inputs, locals_):
    scalars = set(getattr(spec, 'scalar_params', {}))
    # 原地更新（输出张量同时被读取，如 Y[i] = A[i]*X[i] + Y[i]）在逐元素映射里是安全的：
    # 每个 lane 只读写自己的元素。因此允许 output 出现在 inputs 中。
    outputs = set(output_names(spec)) or {spec.store_target}
    if set(spec.params) - scalars != inputs | outputs \
            or len(set(spec.params)) != len(spec.params):
        # 只出现在 guard 里的形参（`if i < n:`）几乎都是漏了静态签名的标量；
        # 没有签名就无法把它和张量区分开，这里给出准确归因而不是笼统"未使用"。
        if spec.guard:
            mentioned = {node.id for node in ast.walk(ast.parse(spec.guard, mode='eval'))
                         if isinstance(node, ast.Name)}
            hinted = (set(spec.params) - scalars - inputs - outputs) & mentioned
            if hinted:
                raise Unsupported('形参 %s 只出现在 guard 中：需要静态签名声明它是标量'
                                  % ','.join(sorted(hinted)))
        raise Unsupported('形参必须恰好对应读取输入与独立输出（签名标量除外），不支持未使用参数')
    if set(spec.params) & locals_ or (set(spec.params) | locals_) & {'cuda', 'range'}:
        raise Unsupported('参数、索引及局部变量不能相互遮蔽')
    # 新生成器以 _tl_ 前缀隔离模板局部名，避免继续扩张硬编码名字表。
    reserved = {'T', 'tilelang', 'range', 'cuda', 'any', 'type', 'int', 'len', 'tuple', 'float', 'ValueError'}
    if any(name.startswith('_tl_') for name in spec.params + [spec.name]) \
            or spec.name in reserved or set(spec.params) & reserved:
        raise Unsupported('名称与生成器保留标识符冲突')


def shape_axis(node):
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
            and node.value.attr == 'shape' and isinstance(node.value.value, ast.Name) \
            and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int \
            and node.slice.value >= 0:
        return node.value.value.id, node.slice.value
    return None


def validate_guard(spec, shapes):
    spec.guard_kind = 'none'
    spec.guard_lower = 0
    spec.guard_upper_offset = 0
    spec.guard_extent_scalar = None
    if not spec.guard:
        return
    guard = ast.parse(spec.guard, mode='eval').body
    terms = guard.values if isinstance(guard, ast.BoolOp) and isinstance(guard.op, ast.And) else [guard]
    covered = []
    single = len(spec.store_index) == 1
    # 每个输出轴的上界：`_tl_shape[k]`（张量形状）或签名标量名（`i < nx`）。
    # 多轴标量 guard（`if i < nx and j < ny`）在真实代码里很常见。
    lengths = [None] * len(spec.store_index)
    for term in terms:
        if isinstance(term, ast.Compare) and len(term.ops) == 1 and isinstance(term.left, ast.Name):
            left = term.left.id
            right = term.comparators[0]
            if isinstance(term.ops[0], ast.Lt):
                axis = shape_axis(right)
                if axis is not None and shapes.get(axis) == left:
                    covered.append(left)
                    lengths[spec.store_index.index(left)] = (
                        '_tl_shape[%d]' % spec.store_index.index(left))
                    continue
                # `arr.size` 只在输出张量本身、且输出是一维时等于 shape[0]；
                # 其它情形（多维张量的总元素数）语义不同，继续拒绝。
                if single and isinstance(right, ast.Attribute) and right.attr == 'size' \
                        and isinstance(right.value, ast.Name) \
                        and right.value.id == spec.store_target \
                        and shapes.get((spec.store_target, 0)) == left:
                    covered.append(left)
                    continue
                # `if i < POPULATION`，POPULATION 是模块级整数常量：域是常量。
                if isinstance(right, ast.Name):
                    constant = (getattr(spec, 'const_bindings', {}) or {}).get(right.id)
                    if constant is not None and constant.op == 'const'                             and type(constant.value) is int and constant.value > 0:
                        covered.append(left)
                        lengths[spec.store_index.index(left)] = str(constant.value)
                        continue
                # `if i < n:`，n 是签名整数标量：域是 [0, n)，与输出 shape 无关。
                # 多轴情形（`i < nx and j < ny`）给每个轴各记一个上界。
                # 调用方必须保证上界 <= 输出长度，否则越界——这条前提写进生成物头注释。
                if isinstance(right, ast.Name) \
                        and right.id in set(spec.scalar_params or {}) \
                        and spec.scalar_params[right.id] in INT_DTYPES:
                    covered.append(left)
                    lengths[spec.store_index.index(left)] = right.id
                    if single:
                        spec.guard_extent_scalar = right.id
                    continue
                # stencil 形态：`i < shape - c`
                if single and isinstance(right, ast.BinOp) and isinstance(right.op, ast.Sub)                         and shape_axis(right.left) is not None                         and shapes.get(shape_axis(right.left)) == left                         and isinstance(right.right, ast.Constant)                         and type(right.right.value) is int and right.right.value > 0:
                    covered.append(left)
                    spec.guard_upper_offset = right.right.value
                    continue
            # 下界：`i > c` / `i >= c`（仅单轴映射）
            if single and isinstance(term.ops[0], (ast.Gt, ast.GtE))                     and isinstance(right, ast.Constant) and type(right.value) is int:
                spec.guard_lower = right.value + (1 if isinstance(term.ops[0], ast.Gt) else 0)
                continue
        raise Unsupported('guard 只支持 <shape 上界、<shape-c 上界与常量下界')
    if len(covered) != len(spec.store_index) or set(covered) != set(spec.store_index):
        if not _supplement_early_return_bounds(spec, shapes, covered, lengths):
            raise Unsupported('guard必须恰好覆盖每个输出轴一次')
    # 每个轴都拿到了上界（形状或标量）→ 域按这些长度做行主序分解。
    if all(length is not None for length in lengths):
        spec.guard_lengths = tuple(lengths)
    spec.guard_kind = 'bounds'


def _supplement_early_return_bounds(spec, shapes, covered, lengths):
    """早退边界守卫（20261026-163 S2）：用写回张量自身 shape 补齐未覆盖的输出轴。

    形态：`idx, c_idx = cuda.grid(2); if idx >= forces.shape[0]: return; forces[idx, c_idx] = 0.0`
    —— 源码只守卫 dim0，其余写回轴由**源 launch 覆盖**。tile 级没有 launch 配置，
    因此未覆盖轴取写回张量自己的 `shape[d]`，把"launch 覆盖完整"写成显式前提
    （SPEC-163 §2.4）：源对未守卫轴越界写属未定义行为，目标选择不写，对该区域不作等价承诺。

    只在以下条件同时成立时补齐，否则返回 False 交回原拒绝（**不改 162 既有行为**）：
    - guard 属于严格早退（`spec.strict_early_return_guard`：紧邻 grid、全函数唯一 return、
      取反后恰一项比较、无 shape 别名前缀、后续无 return）；手写 guard / 多 return / `or`
      多条件即便覆盖不全也不补齐；
    - 单一写回张量；
    - guard 只有一项、且是 `轴 < 该写回张量.shape[d]`；
    - 已覆盖轴**没有重复**（重复覆盖不得借补齐通过）；
    - 每个未覆盖轴就是该张量对应维的网格轴。
    """
    if not getattr(spec, 'strict_early_return_guard', False):
        return False
    target = getattr(spec, 'store_target', None)
    if not target or output_names(spec) != [target]:
        return False
    guard = ast.parse(spec.guard, mode='eval').body
    terms = guard.values if isinstance(guard, ast.BoolOp) and isinstance(guard.op, ast.And) \
        else [guard]
    for term in terms:
        # 本批只补"轴与写回张量自身 shape 比较"的早退守卫；标量上界（`轴 >= n`）等形态
        # 仍按原规则拒绝，不借补齐通道顺带扩能力。
        if not (isinstance(term, ast.Compare) and len(term.ops) == 1
                and isinstance(term.ops[0], ast.Lt)):
            return False
        shape = shape_axis(term.comparators[0])
        if shape is None or shape[0] != target:
            return False
    if len(terms) != 1:
        return False
    if len(covered) != len(set(covered)):
        # 同一轴被重复覆盖：数量可能恰好相等，但语义上没有补齐资格。
        return False
    missing = [axis for axis in spec.store_index if axis not in set(covered)]
    if not missing:
        return False
    for axis in missing:
        if axis == AXIS_CONSTANT or axis not in (spec.idx_vars or ()):
            return False
        dim = spec.store_index.index(axis)
        if shapes.get((target, dim)) != axis:
            return False
        lengths[dim] = '_tl_shape[%d]' % dim
        covered.append(axis)
    return True

# ── 展平/仿射索引：单维域 + `<坐标> < 上界` 的程序模型 ──────────────────────
# 展平坐标 p = by*S + bx*blockDim.x + tx 是该 kernel 的唯一线程坐标；
# 域上界直接取源码 guard 里的上界表达式（签名整数标量、张量 shape 或正常量），
# 因此不需要假设 "xDim == gridDim.x*blockDim.x" 或 "totalNodes == 网格总量"。
# 仍然需要的前提（与 cuda.grid(n) 路径同级）：源 launch 对 [0, 上界) 内每个 p
# 至多枚举一次；重复枚举会与源码自身的写竞争等价，本项目一律不覆盖竞争程序。

INT_DTYPES = ('uint8', 'uint32', 'int32', 'int64')


def flat_extent(spec, axis):
    """展平域上界：只接受 `<展平坐标> < 上界`，上界必须是可证来源。"""
    if not spec.guard:
        raise Unsupported('展平映射需要上界guard：<展平坐标> < 域上界')
    node = ast.parse(spec.guard, mode='eval').body
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Lt) and isinstance(node.left, ast.Name)
            and node.left.id == axis):
        raise Unsupported('展平映射只支持单个 <展平坐标> < 域上界 的guard')
    bound = node.comparators[0]
    if isinstance(bound, ast.Name) and bound.id in spec.scalar_params:
        dtype = spec.scalar_params[bound.id]
        if dtype not in INT_DTYPES:
            raise Unsupported('展平域上界必须是整数标量: ' + bound.id)
        return ('scalar', bound.id, dtype)
    shape = shape_axis(bound)
    if shape is not None:
        if shape[0] not in spec.params:
            raise Unsupported('展平域上界必须来自张量形参: ' + shape[0])
        return ('shape', shape[0], shape[1])
    # 网格步长循环的常见写法：`range(cuda.grid(1), arr.size, cuda.gridsize(1))`。
    if isinstance(bound, ast.Attribute) and bound.attr == 'size' \
            and isinstance(bound.value, ast.Name):
        if bound.value.id not in spec.params:
            raise Unsupported('展平域上界必须来自张量形参: ' + bound.value.id)
        return ('size', bound.value.id)
    if isinstance(bound, ast.Constant) and type(bound.value) is int and bound.value > 0:
        return ('const', bound.value)
    raise Unsupported('展平域上界只支持签名整数标量、张量shape或正常量')


def validate_flat_map(spec):
    """展平逐元素映射：单线程坐标、必带域上界、每个张量都按该坐标直接寻址。"""
    # R3：展平发射模板不消费 `element_mask_expressions`，掩码条件会被静默丢弃。
    # 入口处明确拒绝（不实现掩码、也不留静默路径）。
    if getattr(spec, 'element_masks', ()):
        raise Unsupported('展平入口暂不支持元素掩码（会丢弃掩码条件）；请改用逐元素入口')
    compute = spec.compute
    if spec.grid_dims != 1 or len(spec.idx_vars) != 1:
        raise Unsupported('展平映射需要单一线程坐标')
    axis = spec.idx_vars[0]
    if tuple(compute.output_axes) != (axis,):
        raise Unsupported('展平映射的输出下标必须就是展平坐标: ' + axis)
    accesses = reads(compute.expression)
    for _, bound in compute.bindings:
        accesses += reads(bound)
    if not accesses:
        raise Unsupported('逐元素表达式必须至少读取一个输入')
    inputs = set()
    for name, indices in accesses:
        if tuple(indices) != (axis,):
            raise Unsupported('展平映射的读取只支持直接以展平坐标寻址: ' + name)
        if name not in spec.params:
            raise Unsupported('读取的张量必须是形参: ' + name)
        inputs.add(name)
    if compute.output not in spec.params:
        raise Unsupported('写回的张量必须是形参: ' + compute.output)
    # 展平后行跨距等标量可能不再被使用；它们仍按源形参顺序保留在生成签名里。
    scalars = set(spec.scalar_params)
    if set(spec.params) - scalars != inputs | {compute.output}:
        raise Unsupported('形参必须恰好对应读取输入与独立输出（签名标量除外）')
    if len(inputs | {compute.output}) != len(spec.params) - len(scalars):
        raise Unsupported('形参不能重复')
    spec.flat_extent = flat_extent(spec, axis)
    spec.read_axes = {name: (axis,) for name in inputs}
    spec.axis_roles = {axis: 'D0'}
    spec.guard_kind = 'flat_bound'
    spec.pattern = 'flat_elementwise'
    return spec
