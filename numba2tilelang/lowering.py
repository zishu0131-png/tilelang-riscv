# -*- coding: utf-8 -*-
"""已验证纯计算的 TileLang 发射；不从源码猜测算子。"""
import math
import re
from semantic import (AXIS_CONSTANT, REDUCTION_IDENTITY, Unsupported, output_names,
                      polynomial, reads)


def walk(expr):
    """深度优先遍历表达式树（与 program.walk 同构，独立实现避免循环导入）。"""
    yield expr
    for arg in getattr(expr, 'args', ()):
        yield from walk(arg)

# 每个归约操作对应：TileLang原语、段间合并方式。
REDUCTION_PRIMITIVE = {'sum': 'T.reduce_sum', 'max': 'T.reduce_max', 'min': 'T.reduce_min'}
# 归约长度不超过它时，改用"每 lane 一个输出 + 串行归约"的调度（不做 block 归约）。
SMALL_REDUCTION_LIMIT = 32
REDUCTION_COMBINE = {'sum': '{} + {}', 'max': 'T.max({}, {})', 'min': 'T.min({}, {})'}


def in_place_output(expressions, target):
    """写回目标是否同时被读取（`Y[i] = a * X[i] + Y[i]`）。

    这类产物的输出缓冲必须由调用方提供：交给 kernel 分配只会拿到一块零初值内存，
    读改写就退化成"读零"，是**静默错误**，不报编译或运行时错误。
    """
    return any(name == target for expression in expressions for name, _ in reads(expression))


def out_idx_text(in_place, position):
    return '[]' if in_place else '[%d]' % position


def literal(value):
    """无穷必须写成 float('inf')：repr 会给成裸 inf，那是个未定义名字。"""
    if isinstance(value, float) and math.isinf(value):
        return f"float('{'-' if value < 0 else ''}inf')"
    return repr(value)


def read_coordinates(expr, axes):
    """把一个 read 节点渲染成 `(张量名, 各维坐标串)`。

    坐标串同时承担两个作用：生成下标、以及作为"同址读取"的身份键。
    因此系数与偏移都必须并进去——`X[2*ox]` 与 `X[2*ox+1]` 不是同一个地址。
    """
    name, indices = expr.value
    offsets = tuple(getattr(expr, 'offsets', ()) or ())
    scales = tuple(getattr(expr, 'scales', ()) or ())
    parts = []
    for position, axis in enumerate(indices):
        if axis == AXIS_CONSTANT:
            # 常量下标维：取固定元素（`X[n,y,x,0]`）或运行期地址表达式（`Cos[i, d//2]`）。
            value = offsets[position] if position < len(offsets) else 0
            parts.append(str(value) if isinstance(value, int) else emit_expr(value, axes))
            continue
        coord = axes[axis]
        scale = scales[position] if position < len(scales) else 1
        offset = offsets[position] if position < len(offsets) else 0
        if isinstance(scale, tuple):
            # 运行期标量系数（`F[b*win]`）：形参名直接参与下标，由 TIR 做整型提升。
            coord = f'({scale[1]} * {coord})'
        elif scale != 1:
            coord = f'({scale} * {coord})'
        if isinstance(offset, tuple):
            # 形状偏移：运行期张量维度，已提成构造期参数 `_tl_dim_<张量>_<维>`。
            sign, dim_name, dim_index = offset
            term = f'_tl_dim_{dim_name}_{dim_index}'
            coord = f'{coord} - {term}' if sign < 0 else f'{coord} + {term}'
        elif offset:
            coord = f'{coord} - {abs(offset)}' if offset < 0 else f'{coord} + {offset}'
        parts.append(coord)
    return name, tuple(parts)


def emit_expr(expr, axes):
    # 值级复用：只由 lower_map 的单例窄门登记；其它路径没有这个键，行为不变。
    reuse = axes.get(('reuse', id(expr)))
    if reuse is not None:
        return reuse
    if expr.op == 'result':
        return axes.get(('result', expr.value), f'_tl_result_{expr.value}[0]')
    if expr.op == 'shape':
        return axes[('shape', *expr.value)]
    if expr.op == 'const':
        return f'T.cast({literal(expr.value)}, "{expr.dtype}")'
    if expr.op == 'scalar':
        # 签名标量形参：prim_func 的参数，直接按名字引用。
        return f'T.cast({expr.value}, "{expr.dtype}")'
    if expr.op == 'index':
        return axes[expr.value]
    if expr.op == 'bound':
        # 已物化的局部绑定：读 per-lane 缓冲（在它定义的那个并行域组首写入）。
        return axes[('bound', expr.value)]
    if expr.op == 'read':
        # 物化绑定：读 per-lane 缓冲里的**原值**，不再重新读张量。
        materialized = axes.get(('materialized', id(expr)))
        if materialized is not None:
            return materialized
        name, coords = read_coordinates(expr, axes)
        # 只复用当前发射作用域内已物化的同地址读取，不重写表达式树。
        return axes.get(('read', name, coords), f'{name}[{", ".join(coords)}]')
    if expr.op in ('neg', 'pos'):
        sign = '-' if expr.op == 'neg' else '+'
        return f'({sign}{emit_expr(expr.args[0], axes)})'
    if expr.op == 'cast':
        return f'T.cast({emit_expr(expr.args[0], axes)}, "{expr.dtype}")'
    if expr.op == 'pow':
        operands = [f'T.cast({emit_expr(arg, axes)}, "{expr.dtype}")' for arg in expr.args]
        return f'T.pow({operands[0]}, {operands[1]})'
    if expr.op in ('exp', 'exp2', 'log', 'log2', 'log10', 'sqrt', 'rsqrt', 'tanh', 'abs',
                   'sin', 'cos', 'tan', 'fabs', 'floor', 'ceil', 'erf', 'round'):
        argument = emit_expr(expr.args[0], axes)
        primitive = 'abs' if expr.op == 'fabs' else expr.op
        return f'T.cast(T.{primitive}(T.cast({argument}, "{expr.dtype}")), "{expr.dtype}")'
    if expr.op == 'atan2':
        operands = [f'T.cast({emit_expr(arg, axes)}, "{expr.dtype}")' for arg in expr.args]
        return f'T.atan2({operands[0]}, {operands[1]})'
    if expr.op in ('max', 'min'):
        operands = [f'T.cast({emit_expr(arg, axes)}, "{expr.dtype}")' for arg in expr.args]
        return f'T.{expr.op}({operands[0]}, {operands[1]})'
    if expr.op in ('and', 'or'):
        operands = [emit_expr(arg, axes) for arg in expr.args]
        primitive = 'And' if expr.op == 'and' else 'Or'
        return f'T.{primitive}({operands[0]}, {operands[1]})'
    if expr.op in ('lt', 'gt', 'le', 'ge', 'eq', 'ne'):
        operator = {'lt': '<', 'gt': '>', 'le': '<=', 'ge': '>=', 'eq': '==', 'ne': '!='}[expr.op]
        operands = [emit_expr(arg, axes) for arg in expr.args]
        return f'({operands[0]} {operator} {operands[1]})'
    if expr.op == 'select':
        condition, yes, no = (emit_expr(arg, axes) for arg in expr.args)
        return (f'T.if_then_else({condition}, '
                f'T.cast({yes}, "{expr.dtype}"), T.cast({no}, "{expr.dtype}"))')
    if expr.op in ('bitand', 'bitor', 'bitxor', 'lshift', 'rshift'):
        operands = [f'T.cast({emit_expr(arg, axes)}, "int64")' for arg in expr.args]
        primitive = {'bitand': 'T.bitwise_and', 'bitor': 'T.bitwise_or',
                     'bitxor': 'T.bitwise_xor', 'lshift': 'T.shift_left',
                     'rshift': 'T.shift_right'}[expr.op]
        return f'{primitive}({operands[0]}, {operands[1]})'
    if expr.op in ('floordiv', 'mod'):
        operands = [f'T.cast({emit_expr(arg, axes)}, "int64")' for arg in expr.args]
        primitive = 'T.floordiv' if expr.op == 'floordiv' else 'T.floormod'
        return f'{primitive}({operands[0]}, {operands[1]})'
    if expr.op in ('add', 'sub', 'mul', 'div'):
        operator = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[expr.op]
        operands = [f'T.cast({emit_expr(arg, axes)}, "{expr.dtype}")' for arg in expr.args]
        return f'({operands[0]} {operator} {operands[1]})'
    raise Unsupported('没有已验证的表达式发射规则: ' + expr.op)


def _walk_stage_reads(expression):
    """本阶段**实际**发生的内存读取遍历：`bound` 是 per-lane 缓冲读，
    不再展开它的来源（否则会把历史读取当成当前阶段重新读全局内存）。"""
    # 与 `walk` 同序（前序、参数按声明顺序），只跳过 `bound` 的来源
    yield expression
    if expression.op == 'bound':
        return
    for arg in expression.args:
        yield from _walk_stage_reads(arg)


def emit_shared_reads(expressions, prefix):
    """同一有效坐标作用域内的重复读取；调用者将定义放在边界判断内。"""
    counts = {}
    for expression, axes in expressions:
        for node in _walk_stage_reads(expression):
            if node.op != 'read':
                continue
            name, coords = read_coordinates(node, axes)
            key = ('read', name, coords)
            counts[key] = counts.get(key, 0) + 1
    shared = {key: f'{prefix}_{number}'
              for number, key in enumerate(key for key, count in counts.items() if count > 1)}
    definitions = [f'{local} = {name}[{", ".join(coordinates)}]'
                   for (_, name, coordinates), local in shared.items()]
    return definitions, shared


def _expr_has_memory_dependency(value):
    """地址（offsets/scales）里是否含内存读或绑定依赖（递归，含嵌套 args/offsets/scales）。"""
    if value is None or not hasattr(value, 'op'):
        return False
    if value.op in ('read', 'bound'):
        return True
    for arg in value.args:
        if _expr_has_memory_dependency(arg):
            return True
    for offset in getattr(value, 'offsets', ()) or ():
        if _expr_has_memory_dependency(offset):
            return True
    for scale in getattr(value, 'scales', ()) or ():
        if _expr_has_memory_dependency(scale):
            return True
    return False


def _shared_scan(expression, axes, strict_counts, lazy_keys, dtypes, dependent_keys):
    """严格求值脊上的重复读计数 + 惰性分支里的读集合。

    `select` 只严格遍历条件，两个分支是惰性的；`and`/`or` 只有第一个操作数必然求值。
    惰性分支里出现的读一律不进优化候选（不能提前求值）。
    `dtypes` 收集每个键的**全部** dtype（调用方据此排除冲突）；
    `dependent_keys` 记录地址里含内存读/绑定依赖的键（不做优化）。
    """
    node = expression
    if node.op == 'read':
        key = ('read',) + read_coordinates(node, axes)
        strict_counts[key] = strict_counts.get(key, 0) + 1
        dtypes.setdefault(key, set()).add(node.dtype)
        for value in list(getattr(node, 'offsets', ()) or ())                 + list(getattr(node, 'scales', ()) or ()):
            if _expr_has_memory_dependency(value):
                dependent_keys.add(key)
                break
        return
    if node.op == 'bound':
        return
    if node.op == 'select':
        # 值表达式里的 select：条件与两个分支都不作为物化候选（保守；条件读也不提前）。
        for branch in node.args:
            _lazy_scan(branch, axes, lazy_keys)
        return
    if node.op in ('and', 'or'):
        _shared_scan(node.args[0], axes, strict_counts, lazy_keys, dtypes, dependent_keys)
        for branch in node.args[1:]:
            _lazy_scan(branch, axes, lazy_keys)
        return
    for arg in node.args:
        _shared_scan(arg, axes, strict_counts, lazy_keys, dtypes, dependent_keys)


def _lazy_scan(expression, axes, lazy_keys):
    if expression.op == 'read':
        lazy_keys.add(('read',) + read_coordinates(expression, axes))
        return
    if expression.op == 'bound':
        return
    for arg in expression.args:
        _lazy_scan(arg, axes, lazy_keys)


REUSE_DTYPES = ('float16', 'float32', 'float64')


def _reuse_subtree_has(node, wanted_ops):
    """递归检查 args 与地址字段（offsets/scales）里的实际 Expr 依赖。

    只把带 `op` 的条目当作 Expr；int/tuple 等元数据跳过。
    不用“凡 read 一律算依赖”的规则——正常 read 仍需允许。
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if current.op in wanted_ops:
            return True
        stack.extend(current.args)
        for extra in list(getattr(current, 'offsets', ()) or ()) \
                + list(getattr(current, 'scales', ()) or ()):
            if hasattr(extra, 'op'):
                stack.append(extra)
    return False


def _reuse_contains(root, target_id):
    """root 的严格后代里是否有 target（不把 root 自己算作包含）。"""
    stack = list(root.args)
    while stack:
        current = stack.pop()
        if id(current) == target_id:
            return True
        stack.extend(current.args)
    return False


def _reuse_strict_scan(node, counts, refs):
    """只有严格求值位置上的 select 才计数；每个出现路径各计一次（无 visited 去重）。"""
    if node.op == 'select':
        counts[id(node)] = counts.get(id(node), 0) + 1
        refs[id(node)] = node
        # 只有条件向 strict 传播；两个分支是惰性的。
        _reuse_strict_scan(node.args[0], counts, refs)
        return
    if node.op in ('and', 'or'):
        _reuse_strict_scan(node.args[0], counts, refs)
        return
    if node.op in ('read', 'bound'):
        return
    for arg in node.args:
        _reuse_strict_scan(arg, counts, refs)


def reuse_plan(expression):
    """候选 B 的窄门：严格位置重复 ≥2 的完整 select。

    返回 [(root, dtype), ...]（按首次出现顺序）；资格不足返回 []。
    单元测试依赖这个函数；不要在其它 lowerer 调用它。
    """
    counts, refs = {}, {}
    _reuse_strict_scan(expression, counts, refs)
    if not counts:
        return []
    candidates = []
    for node_id, count in counts.items():
        node = refs[node_id]
        if count < 2:
            continue
        if node.dtype not in REUSE_DTYPES:
            continue
        # 条件里含内存读/绑定：不物化（惰性/顺序语义复杂）。
        if _reuse_subtree_has(node.args[0], ('read', 'bound')):
            continue
        # 候选子树任意位置有 bound 依赖：不物化。
        if _reuse_subtree_has(node, ('bound',)):
            continue
        candidates.append(node)
    if not candidates:
        return []
    # 若候选之间仍互相包含/共享候选，保守放弃该 store 的优化（不扩成 DAG 调度）。
    selected = []
    for node in candidates:
        if any(_reuse_contains(other, id(node)) or _reuse_contains(node, id(other))
               for other in candidates if other is not node):
            return []
        selected.append(node)
    seen = set()
    ordered = []
    for node in selected:
        if id(node) in seen:
            continue
        seen.add(id(node))
        ordered.append((node, node.dtype))
    return ordered


def flat_indices(dim_exprs, flat='_tl_p'):
    """把展平下标还原成各维下标（行主序）。1/2 维保持已验证的简单形式。"""
    count = len(dim_exprs)
    if count == 1:
        return [flat]
    if count == 2:
        return [f'({flat} // {dim_exprs[1]})', f'({flat} % {dim_exprs[1]})']
    indices = []
    for dim in range(count):
        stride = ' * '.join(dim_exprs[dim + 1:])
        if dim == 0:
            indices.append(f'({flat} // ({stride}))')
        elif dim == count - 1:
            indices.append(f'({flat} % {dim_exprs[-1]})')
        else:
            indices.append(f'(({flat} // ({stride})) % {dim_exprs[dim]})')
    return indices


def split_schedule_axes(spec):
    """归约依赖的输出坐标决定并行网格；其余输出坐标在块内遍历。

    返回 (grid_axes, remaining_axes)，均按输出顺序。归约只依赖 grid 坐标，
    因此在块内遍历 remaining 不会改变结果，只是不再按每个输出元素重算一遍。
    调用者在 grid 为空或等于全部输出坐标时保持原有整输出网格。
    """
    output_axes = spec.program.output_axes
    used = set()
    for axes in spec.layout['stage_axes']:
        used.update(axes)
    return (tuple(axis for axis in output_axes if axis in used),
            tuple(axis for axis in output_axes if axis not in used))


def emit_layout_checks(layout):
    """所有layout路径共享工厂约束，必须放在构造prim_func之前。"""
    lines = []
    for length, value in layout['checks']:
        lines += [f'    if {length} != {value}:',
                  '        raise ValueError("source extent mismatch")']
    for left, right in layout['equalities']:
        lines += [f'    if {left} != {right}:',
                  '        raise ValueError("source requires equal dimension lengths")']
    return lines


def inferred_note(spec):
    """无静态签名时推断出的标量，必须在生成物头注释里显式声明，不能当成已确认契约。"""
    names = getattr(spec, 'inferred_scalars', ()) or ()
    if not names:
        return ''
    return ('# ⚠️ 源码无静态签名：%s 按"未以张量方式使用"推断为标量，dtype 记为 float64。\n'
            '#    numba 的 dtype 来自调用点；实际传入 int/float32 时调用方须自行保证类型。\n'
            % ', '.join(names))


def lower_sum(spec):
    reduction = spec.compute.reduction
    axes = tuple(spec.store_index)
    axis = reduction.axis
    # 计算轴的物理长度：输出轴取输出shape，归约轴取外部给定的reduction_extent。
    lengths = [f'_tl_shape[{dim}]' for dim in range(len(axes))] + ['_tl_extent']
    combined = axes + (axis,)
    scalar_params = dict(getattr(spec, 'scalar_params', {}) or {})
    tensor_dtypes = dict(getattr(spec, 'tensor_dtypes', {}) or {})
    tensor_params = [name for name in spec.params if name not in scalar_params]
    positions = {name: tuple(combined.index(used) for used in spec.read_axes.get(name, ()))
                 for name in tensor_params}
    shapes = {name: '_tl_shape' if name == spec.store_target
              else '(' + ', '.join(lengths[position] for position in positions[name]) + ',)'
              for name in tensor_params}
    from abi import dtype_of, is_scalar
    parameters = ', '.join(
        f'{name}: T.Tensor({shapes[name]}, "{dtype_of(spec, name)}")'
        if not is_scalar(spec, name) else f'{name}: T.{dtype_of(spec, name)}'
        for name in spec.params)
    # 展平输出域: 每个输出元素对应一个 block，块内线程分k段归约。
    extent = ' * '.join(f'_tl_shape[{dim}]' for dim in range(len(axes)))
    index = dict(zip(axes, flat_indices([f'_tl_shape[{dim}]' for dim in range(len(axes))])))
    # 非零起点（窗口归约）：归约轴坐标是"起点 + 块内位置"，起点按输出元素逐块变化。
    start_expr = getattr(reduction, 'start_expr', None)
    if start_expr is not None and polynomial(start_expr):
        index[axis] = f'({emit_expr(start_expr, index)} + _tl_k)'
    else:
        index[axis] = '_tl_k'
    definitions, shared = emit_shared_reads([(spec.compute.expression, index)], '_tl_read')
    read_setup = ''.join(' ' * 24 + line + '\n' for line in definitions)
    expression = emit_expr(spec.compute.expression, index | shared)
    output_index = ', '.join(index[used] for used in axes)
    # 越界 lane 填该操作的单位元；累加器也从单位元起算，两者都必须精确，不能用哨兵值。
    identity = literal(REDUCTION_IDENTITY[reduction.op])
    primitive = REDUCTION_PRIMITIVE[reduction.op]
    combine = REDUCTION_COMBINE[reduction.op]
    # 初值若已通过"float32 等价于单位元"校验（如 max 的 -1e300），直接发射单位元：
    # 既保持语义（float32 数据不可能更小），也避免超出 float32 的常量被 TIR 拒绝。
    init_text = (literal(REDUCTION_IDENTITY[reduction.op]) if reduction.init.op == 'const'
                 else emit_expr(reduction.init, index))
    # `precision='fast'` 的安全收缩：**只**对单轴浮点 sum 降低累加器精度。
    # 输入表达式仍按其语义 dtype 完整计算，再转换到 acc_dtype；输出按 storage dtype 写回。
    # max/min、整数/布尔以及其它 lowerer 一律保持 strict（不"假装"有 fast 收益）。
    precision = getattr(spec, 'precision', 'strict')
    float_sum = reduction.op == 'sum' and spec.compute.expression.dtype in ('float32', 'float64')
    # 显式精度屏障：源码里写了 `float64(...)`（求和项或初始化）就是调用方**显式要求**的精度，
    # 不因 `fast` 而收缩累加器。只检查这两棵语义树上的 `Expr('cast')`（发射后的通用
    # `T.cast` 不算——那是本转换器自己加的）。
    explicit_sum = any(node.op == 'cast' for node in walk(spec.compute.expression))
    explicit_init = reduction.init is not None and any(
        node.op == 'cast' for node in walk(reduction.init))
    barrier = explicit_sum or explicit_init
    fast_applied = precision == 'fast' and float_sum and not barrier
    spec.precision_applied = fast_applied
    spec.precision_barrier = precision == 'fast' and float_sum and barrier
    # 20261025-151：单轴模板的累加槽固定 float64；源语义若不是 f32/f64（int64/float16），
    # 这里会**静默改型**——按同一门禁拒绝，不静默映射。
    from semantic import source_accum_dtype as _source_accum_dtype
    _source_acc = _source_accum_dtype(spec.compute)
    if _source_acc not in ('float32', 'float64'):
        raise Unsupported('归约累加 dtype 只支持 float32/float64（源语义为 %s）' % _source_acc)
    acc_dtype = 'float32' if fast_applied else 'float64'
    total = f'T.cast({init_text}, "{acc_dtype}")'
    from abi import store_dtype as _store_dtype
    store_dtype = _store_dtype(spec)
    merge = combine.format('_tl_total[_tl_one]', '_tl_partial[_tl_one]')
    constant_check = '' if spec.k_constant is None else (
        f'    if _tl_extent != {spec.k_constant}:\n'
        f'        raise ValueError("source requires reduction_extent == {spec.k_constant}")\n')
    # 20261025-144：模块级常量上界 guard → 输出形状是生成期常量，各维按**等值**校验
    # （不是 `<=`）。输入张量的同根维度按 _tl_shape 声明，因此一并被等值约束。
    guard_check = ''.join(
        f'    if _tl_shape[{dim}] != {value}:\n'
        f'        raise ValueError("source extent mismatch")\n'
        for dim, value in sorted(getattr(spec, 'constant_guard_dims', {}).items()))
    # 归约长度：`range(n)` 的 n 是 prim_func 参数时直接用名字，否则用构造期参数。
    extent_var = getattr(spec, 'reduction_extent_scalar', None) or '_tl_extent'
    in_place = in_place_output([spec.compute.expression], spec.store_target)
    out_idx = out_idx_text(in_place, spec.params.index(spec.store_target))
    call_note = (f'# ⚠️ 原地更新：{spec.store_target} 同时被读取，必须由调用方传入（含初值）。\n'
                 if in_place else '')
    # 归约长度小的时候，"每个输出元素一个 block + 块内 block 归约"会把 128 个线程
    # 摊到 8 个元素上，还要做一次全块归约：实测 m3_expert_combine（extent=8，
    # 26 万个输出）慢 100 倍。构造期已知归约长度，直接改选"每 lane 一个输出、
    # 串行扫归约轴"的调度，不做 block 归约。
    small_body = f'''    @T.prim_func
    def _tl_main_small({parameters}):
        with T.Kernel(T.ceildiv(_tl_size, block), threads=128) as _tl_bid:
            _tl_acc = T.alloc_fragment((block,), "{acc_dtype}")
            for _tl_lane in T.Parallel(block):
                _tl_p = _tl_bid * block + _tl_lane
                if _tl_p < _tl_size:
                    _tl_acc[_tl_lane] = {total}
            for _tl_k in T.serial({extent_var}):
                for _tl_lane in T.Parallel(block):
                    _tl_p = _tl_bid * block + _tl_lane
                    if _tl_p < _tl_size:
{read_setup}                        _tl_acc[_tl_lane] = {combine.format('_tl_acc[_tl_lane]', f'T.cast({expression}, "{acc_dtype}")')}
            for _tl_lane in T.Parallel(block):
                _tl_p = _tl_bid * block + _tl_lane
                if _tl_p < _tl_size:
                    {spec.store_target}[{output_index}] = T.cast(_tl_acc[_tl_lane], "{store_dtype}")
'''
    if fast_applied:
        acc_note = '累加器float32（precision=fast：只降累加器精度，输入表达式与输出仍按源码dtype）'
    elif spec.precision_barrier:
        acc_note = '累加器float64（precision=fast，但源码含显式float64转换：精度屏障，保留原精度）'
    else:
        acc_note = '累加器float64'
    return f'''# 单轴{reduction.op}表达式；输入float32，表达式按受限Numba规则提升，{acc_note}。
# 重排加法归约允许舍入差异；输出独立、源launch完整覆盖输出。
{call_note}import tilelang
import tilelang.language as T

@tilelang.jit(out_idx={out_idx},
              pass_configs={{"tl.disable_warp_specialized": True}}, compile_flags=["--fmad=false"])
def {spec.name}(shape, reduction_extent, block=128):
    _tl_shape = tuple(shape)
    _tl_extent = reduction_extent
    if len(_tl_shape) != {len(axes)} or any(type(d) is not int or d <= 0 for d in _tl_shape):
        raise ValueError("shape must contain {len(axes)} positive extents")
    if type(_tl_extent) is not int or _tl_extent <= 0:
        raise ValueError("reduction_extent must be positive")
    if type(block) is not int or block <= 0 or block % 32:
        raise ValueError("block must be a positive multiple of 32")
    _tl_size = {extent}
{guard_check}{constant_check}    @T.prim_func
    def _tl_main({parameters}):
        with T.Kernel(_tl_size, threads=128) as _tl_p:
            _tl_values = T.alloc_fragment((1, block), "{acc_dtype}")
            _tl_partial = T.alloc_fragment((1,), "{acc_dtype}")
            _tl_total = T.alloc_fragment((1,), "{acc_dtype}")
            for _tl_one in T.Parallel(1):
                _tl_total[_tl_one] = {total}
            # 每个 lane 先把自己负责的分块累加到槽位，**循环外只归约一次**。
            # 原先每个 chunk 归约一次（extent=1024、block=128 就是 8 次 block 归约），
            # 实测 rowsum 比 torch 慢 3 倍；改成累加式后与 torch 同档。
            for _tl_lane in T.Parallel(block):
                _tl_values[0, _tl_lane] = T.cast({identity}, "{acc_dtype}")
            for _tl_chunk in T.serial(T.ceildiv({extent_var}, block)):
                for _tl_lane in T.Parallel(block):
                    _tl_k = _tl_chunk * block + _tl_lane
                    if _tl_k < {extent_var}:
{read_setup}                        _tl_values[0, _tl_lane] = {combine.format('_tl_values[0, _tl_lane]', f'T.cast({expression}, "{acc_dtype}")')}
            {primitive}(_tl_values, _tl_partial, dim=1, clear=True)
            for _tl_one in T.Parallel(1):
                _tl_total[_tl_one] = {merge}
            for _tl_one in T.Parallel(1):
                {spec.store_target}[{output_index}] = T.cast(_tl_total[_tl_one], "{store_dtype}")
{small_body}    return _tl_main_small if _tl_extent <= {SMALL_REDUCTION_LIMIT} else _tl_main
'''


def tensor_dim_forms(spec):
    """收集每个 `(张量, 维)` 的仿射读取形态 `(轴, 系数, 偏移)`。

    同一维允许出现多个偏移（`X[2*ox]` 与 `X[2*ox+1]` 是同一组采样），但**系数必须一致**——
    同一维既按 `2*oy` 又按 `oy` 读，说明该轴的语义不唯一，直接拒绝。
    """
    forms = {}
    expressions = [spec.compute.expression] + [value for _, value in spec.compute.bindings]
    for expression in expressions:
        for node in walk(expression):
            if node.op != 'read':
                continue
            name, indices = node.value
            scales = tuple(getattr(node, 'scales', ()) or ()) or (1,) * len(indices)
            offsets = tuple(getattr(node, 'offsets', ()) or ()) or (0,) * len(indices)
            for dim, axis in enumerate(indices):
                if axis == AXIS_CONSTANT:
                    # 常量/运行期地址维：长度不参与"同维同形态"的一致性检查。
                    continue
                scale = scales[dim] if dim < len(scales) else 1
                offset = offsets[dim] if dim < len(offsets) else 0
                if isinstance(scale, tuple):
                    # 运行期标量系数只在归约初值里有意义；逐元素映射无法在构造期确定形状。
                    raise Unsupported('逐元素映射不支持运行期标量下标系数: %s[%d]' % (name, dim))
                form = forms.setdefault((name, dim), (axis, scale, set()))
                if form[0] != axis or form[1] != scale:
                    raise Unsupported('同一张量同一维不能混用不同的轴或系数: %s[%d]' % (name, dim))
                form[2].add(offset)
    return forms


def shape_refs(spec):
    """收集所有"运行期张量维度"引用，按出现顺序返回 `[(tensor, dim), ...]`。

    这些维度无法从输出形状推导（例如两段拼接里的 `B[t - A.shape[0]]`），
    必须在构造期由调用方给出，生成成工厂参数 `_tl_dim_<张量>_<维>`。
    """
    found = []
    expressions = [spec.compute.expression] + [value for _, value in spec.compute.bindings]
    for expression in expressions:
        for node in walk(expression):
            if node.op != 'read':
                continue
            for offset in (tuple(getattr(node, 'offsets', ()) or ())):
                if isinstance(offset, tuple):
                    key = (offset[1], offset[2])
                    if key not in found:
                        found.append(key)
    return found


def affine_extent(scale, offsets, length):
    """该维的物理长度：`scale > 1` 时按完整块（`scale*长度`），否则按最大读取位置。

    `X[2*ox]` / `X[2*ox+1]` 同时出现时，长度都是 `2*L`：取整块，不取 `2*L-1`——
    声明比调用方实际传入的小会让 TileLang 报 shape mismatch。
    """
    if scale == 1:
        refs = [o for o in offsets if isinstance(o, tuple)]
        if refs:
            # 形状偏移：长度 = 输出维 - 被减去的运行期维度（如 `B[t - A.shape[0]]`）。
            sign, dim_name, dim_index = refs[0]
            term = f'_tl_dim_{dim_name}_{dim_index}'
            return f'{length} - {term}' if sign < 0 else f'{length} + {term}'
        # 常量偏移（stencil `A[i±c]`）不改形状——越界由 guard（`i > c` / `i < n-c`）保证。
        return length
    if any(offset < 0 or offset >= scale for offset in offsets):
        raise Unsupported('系数下标的偏移必须落在 [0, 系数) 内')
    return '%d * %s' % (scale, length)


def lower_map(spec):
    dimensions = len(spec.store_index)
    extent = ' * '.join(f'_tl_shape[{i}]' for i in range(dimensions))
    # guard 上界是签名标量时（`if i < n`），并行域是 [0, n) 而不是输出全长；
    # 调用方须保证 n <= 输出元素数，这条前提写进生成物头注释。
    extent_scalar = getattr(spec, 'guard_extent_scalar', None)
    guard_lengths = getattr(spec, 'guard_lengths', None)
    # 20261025-147：常量上界 guard（`i < N`，N 是模块级 int 字面量）在**元素级**通道
    # 界定的是整个输出域；自动分配输出时必须按**等值**校验（不是 `<=`）。
    # 调用方传入输出（原地/掩码/运行期 L）仍保留“写入域 = 输出前缀子集”的既有语义。
    # 这里只收集常量维；标量上界与 shape 上界行为不变。
    guard_const_dims = tuple(
        (dim, int(length)) for dim, length in enumerate(guard_lengths or ())
        if isinstance(length, str) and length.isdigit())
    if guard_lengths and any(not length.startswith('_tl_shape[') for length in guard_lengths):
        # 多轴标量 guard（`i < nx and j < ny`）：域 = 各轴上界之积，
        # 坐标按这些长度行主序分解；标量是 prim_func 参数，_tl_size 放在函数体内。
        extent = ' * '.join(guard_lengths)
        axes = dict(zip(spec.store_index, flat_indices(list(guard_lengths))))
        extent_scalar = True
    elif extent_scalar:
        extent = extent_scalar
        axes = dict(zip(spec.store_index,
                        flat_indices([f'_tl_shape[{i}]' for i in range(dimensions)])))
    else:
        axes = dict(zip(spec.store_index,
                        flat_indices([f'_tl_shape[{i}]' for i in range(dimensions)])))
    scalar_params = dict(getattr(spec, 'scalar_params', {}) or {})
    tensor_params = [name for name in spec.params if name not in scalar_params]
    positions = {name: tuple(-1 if axis == AXIS_CONSTANT else spec.store_index.index(axis)
                             for axis in spec.read_axes.get(name, ()))
                 for name in tensor_params}
    # 运行期张量维度引用（如 `B[t - A.shape[0]]`）提成构造期参数，由调用方给出。
    # 常量下标维（`X[n,y,x,0]` 的通道维）长度同样推不出来，一并走 `_tl_extra`。
    extra_dims = shape_refs(spec)
    for name in tensor_params:
        for dim, position in enumerate(positions[name]):
            if position < 0 and (name, dim) not in extra_dims:
                extra_dims.append((name, dim))
    # 主输出的常量维（`O[n,oy,ox,0..3]` 的通道位）：长度同样推不出来。
    store_shape = tuple(getattr(spec.compute, 'store_indices', ()) or spec.store_index)
    store_consts = tuple(getattr(spec.compute, 'store_constants', ()) or ())
    if any(axis == AXIS_CONSTANT for axis in store_shape):
        for dim, axis in enumerate(store_shape):
            if axis == AXIS_CONSTANT and (spec.store_target, dim) not in extra_dims:
                extra_dims.append((spec.store_target, dim))
    forms = tensor_dim_forms(spec)
    shapes = {}
    for name in tensor_params:
        if name == spec.store_target:
            if any(axis == AXIS_CONSTANT for axis in store_shape):
                shapes[name] = '(' + ', '.join(
                    f'_tl_dim_{name}_{dim}' if axis == AXIS_CONSTANT
                    else f'_tl_shape[{spec.store_index.index(axis)}]'
                    for dim, axis in enumerate(store_shape)) + ',)'
            else:
                shapes[name] = '_tl_shape'
            continue
        dims = []
        for dim, position in enumerate(positions[name]):
            if position < 0:
                dims.append(f'_tl_dim_{name}_{dim}')
                continue
            axis, scale, offsets = forms.get((name, dim), (spec.store_index[position], 1, {0}))
            if (name, dim) in extra_dims:
                # 该维在源码里被别的张量当作偏移引用过，长度由构造期参数给出，
                # 不能按输出轴的形状推导（如两段拼接里的 A 比 O 短）。
                dims.append(f'_tl_dim_{name}_{dim}')
                continue
            length = affine_extent(scale, offsets, f'_tl_shape[{position}]')
            if length is None:
                raise Unsupported('该位置不支持带偏移的读取形状: ' + name)
            dims.append(length)
        shapes[name] = '(' + ', '.join(dims) + ',)' if dims else '_tl_shape'
    outputs = set(output_names(spec))
    for name in tensor_params:
        # program 路径的形状来自 layout，不靠这里的轴投影；positions 为空是正常的。
        if name in outputs or spec.program is not None:
            continue
        if len(positions[name]) == 0:
            raise Unsupported('不能读取一个没有轴投影的张量: ' + name)
    equality_checks = ''
    if spec.program is not None:
        # 坐标仍来自各个Expr.read，物理长度来自独立的约束分析。
        # layout 里没有并入任何已知长度的物理维写成 `_tl_extents[i]`；本模板没有
        # reduction_extents 形参，改由 `_tl_extra` 传入，与其它运行期维度同一机制。
        slot_names = {}
        for name, dims in spec.layout['dims'].items():
            for dim, length in enumerate(dims):
                if length.startswith('_tl_extents['):
                    slot_names[length] = f'_tl_dim_{name}_{dim}'
                    if (name, dim) not in extra_dims:
                        extra_dims.append((name, dim))
        # 常量下标维（`f[0,j,i]` 的通道位）：length 已经是 `_tl_dim_*`，同样走 `_tl_extra`。
        for name, dim in spec.layout.get('dim_extra', ()):
            if (name, dim) not in extra_dims:
                extra_dims.append((name, dim))
        shapes = {}
        for name, dims in spec.layout['dims'].items():
            rendered = [slot_names.get(length, length) for length in dims]
            shapes[name] = '(' + ', '.join(rendered) + ',)'
            for dim, length in enumerate(rendered):
                axes[('shape', name, dim)] = length
        # 输出张量一律按写回下标声明：layout 只包含主输出（`Program` 不带附加输出），
        # 且只按并行轴给 rank，直接照搬会漏掉附加输出、也会丢掉常量维。
        for name, indices in [(spec.store_target, store_shape)] + [
                (extra_name, extra_indices)
                for extra_name, extra_indices, _consts, _value
                in (spec.compute.extra_stores or ())]:
            if not indices:
                continue
            shapes[name] = '(' + ', '.join(
                f'_tl_dim_{name}_{dim}' if axis == AXIS_CONSTANT
                else f'_tl_shape[{spec.store_index.index(axis)}]'
                for dim, axis in enumerate(indices)) + ',)'
            for dim, axis in enumerate(indices):
                if axis == AXIS_CONSTANT and (name, dim) not in extra_dims:
                    extra_dims.append((name, dim))
        checks = emit_layout_checks(spec.layout)
        equality_checks = '\n'.join(checks) + ('\n' if checks else '')
    else:
        # 输出张量一律按**写回下标**声明：附加输出可能从未被读取（inflow 的 rho），
        # 读投影推导不到它；常量维走 `_tl_dim_*`，其余按并行轴序号取 `_tl_shape`。
        parallel_axes = [axis for axis in spec.store_index if axis != AXIS_CONSTANT]
        output_indices = {spec.store_target: store_shape}
        for extra_name, extra_indices, _consts, _value in (spec.compute.extra_stores or ()):
            output_indices.setdefault(extra_name, extra_indices)
        for name, indices in output_indices.items():
            if not indices:
                continue
            dims = []
            for dim, axis in enumerate(indices):
                if axis == AXIS_CONSTANT:
                    dims.append(f'_tl_dim_{name}_{dim}')
                    if (name, dim) not in extra_dims:
                        extra_dims.append((name, dim))
                else:
                    dims.append(f'_tl_shape[{parallel_axes.index(axis)}]')
            shapes[name] = '(' + ', '.join(dims) + ',)'
        for name in tensor_params:
            # 非输出张量仍按读取投影寻址。
            used_axes = spec.store_index if name in outputs else spec.read_axes[name]
            for dim, axis in enumerate(used_axes):
                if axis == AXIS_CONSTANT:
                    axes[('shape', name, dim)] = f'_tl_dim_{name}_{dim}'
                    continue
                axes[('shape', name, dim)] = f'_tl_shape[{parallel_axes.index(axis)}]'
    for name, dim in extra_dims:
        axes[('shape', name, dim)] = f'_tl_dim_{name}_{dim}'
    from abi import dtype_of, is_scalar
    parameters = ', '.join(
        f'{name}: T.Tensor({shapes[name]}, "{dtype_of(spec, name)}")'
        if not is_scalar(spec, name) else f'{name}: T.{dtype_of(spec, name)}'
        for name in spec.params)
    extra_stores = tuple(spec.compute.extra_stores or ())
    # 《主输出》+ 附加输出按源码顺序排成一张表；写回分组给的是这张表的位置。
    # 组内同一 T.Parallel 一次发射，组间顺序执行——后面的组读到前面组写下的值。
    all_stores = [(spec.store_target, store_shape, store_consts, spec.compute.expression)] \
        + list(extra_stores)
    groups = tuple(tuple(group) for group in (spec.compute.store_groups or ())) \
        or (tuple(range(len(all_stores))),)
    output_index = ', '.join(
        str(store_consts[dim]) if axis == AXIS_CONSTANT else axes[axis]
        for dim, axis in enumerate(store_shape))
    from abi import store_dtype as _store_dtype
    store_dtype = _store_dtype(spec)
    lower = getattr(spec, 'guard_lower', 0) or 0
    upper_offset = getattr(spec, 'guard_upper_offset', 0) or 0
    conditions = []
    if lower:
        conditions.append(f'_tl_p >= {lower}')
    conditions.append('_tl_p < _tl_size' + (f' - {upper_offset}' if upper_offset else ''))
    # 元素级掩码（`if solid_mask[i, j]:`）与边界 guard 合成同一个条件。
    for mask in getattr(spec, 'element_mask_expressions', ()) or ():
        conditions.append(emit_expr(mask, axes))
    # C2：运行期 L —— 写回条件截断到前缀 (shape[d]//L)*L；尾部空转不写。
    runtime_base = tuple(getattr(spec, 'flat_base_runtime', ()) or ())
    runtime_assert = ''
    if runtime_base:
        _name, _dim, _length = runtime_base
        conditions = [c for c in conditions
                      if not c.startswith('_tl_p < _tl_size')]
        conditions.append('(T.cast(_tl_p, "int64") < '
                          '(T.cast(_tl_shape[%d], "int64") // %s) * %s)'
                          % (_dim, _length, _length))
        runtime_assert = '            T.Assert(%s > 0, "%s must be > 0")\n' % (_length, _length)
    guard_condition = ' and '.join(conditions)
    # 运行期 L 只写前缀：自动分配的输出无法保证“尾部未写”的语义，
    # 因此强制由调用方传入输出缓冲（out_idx=[]），尾部保留调用方的原值。
    grid = 'T.max(T.ceildiv(_tl_size, block), 1)' if extent_scalar \
        else 'T.ceildiv(_tl_size, block)'
    # 标量上界是 prim_func 的形参，只能在函数体内绑定；shape 推导的 _tl_size 仍在工厂作用域。
    outer_size = '' if extent_scalar else f'    _tl_size = {extent}\n'
    inner_size = f'        _tl_size = {extent}\n' if extent_scalar else ''
    # 注释必须写**真实的域表达式**：多轴标量 guard 时 extent_scalar 只是布尔标记，
    # 直接插值会把 `True` 当名字印进产物（对新 API 用户是错误调用说明）。
    declared_scalars = set(getattr(spec, 'declared_scalar_types', ()) or ())
    # 运行期张量维度（形状偏移）按出现顺序绑定到构造期参数 `_tl_extra`。
    extra_bind = ''.join('    _tl_dim_%s_%d = _tl_extra[%d]\n' % (n, d, i)
                         for i, (n, d) in enumerate(extra_dims))
    # 20261025-156（S1/S2）：常量展开维的界来自源码显式常量（循环界 / 置换读的最大下标 +1），
    # 按**下界**校验（维度更大也安全）；不依赖运行尺寸反推。
    for _name, _dim, _bound in tuple(getattr(spec, 'partial_const_dims', ()) or ()):
        if (_name, _dim) not in extra_dims:
            # 20261025-159（CHK-484）：有界却没有运行期维度参数 = 校验会被静默跳过，
            # 属“接受但零校验”；直接拒绝而不是跳过。
            raise Unsupported('常量维界缺少运行期维度参数: %s dim %d' % (_name, _dim))
        extra_bind += ('    if _tl_dim_%s_%d < %d:\n'
                       '        raise ValueError("constant extent too small: %s dim %d")\n'
                       % (_name, _dim, _bound, _name, _dim))
    if tuple(getattr(spec, 'partial_const_dims', ()) or ()):
        extra_bind = ('# ⚠️ 常量展开维只做**下界**校验：传入的首维更大时，多出的切片不会被写，\n'
                      '#    由调用方预置的值保持不变；小于源码常量界则直接 raise。\n' + extra_bind)
    extra_note = ('# ⚠️ `_tl_extra` 传入源码里被引用的张量维度（按 shape_refs 顺序），\n'
                  '#    例如 `B[t - A.shape[0]]` 需要 `_tl_extra=(A.shape[0],)`。\n'
                  if extra_dims else '')
    # A 片：基址改写后域是整段 shape[d]，源码只覆盖 (shape[d]//L)*L；要求整除，否则拒绝。
    divisible_check = ''
    _divisible = tuple(getattr(spec, 'divisible_extent', ()) or ())
    if _divisible:
        _name, _dim, _length = _divisible
        divisible_check = (
            '    if _tl_shape[%d] %% %d:\n'
            '        raise ValueError("source requires %s.shape[%d] divisible by %d '
            '(flat base rewrite)")\n' % (_dim, _length, _name, _dim, _length))
    all_outputs = output_names(spec)
    all_expressions = [spec.compute.expression] + [expression for _, _, _, expression in extra_stores]
    # 运行期 L 只写前缀、元素掩码只写命中位置：两者都让“自动分配的输出”没有定义语义，
    # 因此统一强制调用方传入输出（保留 runtime_base 既有条件，与 mask 合并而不是覆盖）。
    reads_outputs = [name for name in all_outputs if in_place_output(all_expressions, name)]
    masks = tuple(getattr(spec, 'element_mask_expressions', ()) or ())
    caller_output = bool(runtime_base) or bool(masks)
    in_place = bool(reads_outputs) or caller_output
    # out_idx=[] 时调用方按**源码形参原序**传参（不重排成 inputs..., outputs...）。
    out_idx = ('[]' if in_place else
               '[' + ', '.join(str(spec.params.index(name)) for name in all_outputs) + ']')
    # 20261025-147：常量上界 guard 的两种契约。
    #   · 自动分配输出（out_idx 非空）：常量界定整个输出域 ⇒ 各维按**等值**校验；
    #   · 调用方传入输出（原地 / 掩码 / 运行期 L）：写入域是输出**前缀子集**，保留 “<=”。
    const_shape_check = ''
    if extent_scalar:
        label = extent if isinstance(extent_scalar, bool) else extent_scalar
        if guard_const_dims:
            if in_place:
                extent_note = (
                    f'# ⚠️ 常量上界 guard：写入域是输出**前缀子集**（仅 `{label}` 个元素），\n'
                    '#    输出由调用方传入，未覆盖的尾部保留传入值；不保证完整覆盖。\n')
            else:
                const_shape_check = ''.join(
                    f'    if _tl_shape[{dim}] != {value}:\n'
                    '        raise ValueError("source extent mismatch")\n'
                    for dim, value in guard_const_dims)
                extent_note = (
                    f'# ⚠️ 常量上界 guard：并行域固定为 `{label}`，输出形状必须与常量**完全相等**\n'
                    '#    （不等即 raise）；本产物只覆盖该常量域。\n')
        else:
            # 域表达式里出现的名字：任一是"调用方声明"的就按声明口径写
            names_in_extent = set(re.findall(r'[A-Za-z_]\w*', str(label)))
            origin = ('调用方声明的标量' if names_in_extent & declared_scalars
                      else '签名标量上界')
            extent_note = (f'# ⚠️ 并行域来自{origin} `{label}`：调用方必须保证 {label} <= 输出元素数，\n'
                           f'#    否则越界；{label} == 0 时整个 kernel 不写任何元素。\n')
    else:
        extent_note = ''
    if masks:
        call_note = ('# ⚠️ 条件写：带元素掩码时只写命中元素，所有输出必须由调用方传入'
                     '（按源码形参顺序）；未命中元素保留传入值，不会自动清零。\n')
        if reads_outputs:
            call_note += (f'# ⚠️ 原地更新：{", ".join(reads_outputs)} 同时被读取，'
                          '调用方必须传入（含初值）。\n')
    elif caller_output and not reads_outputs:
        call_note = ('# ⚠️ 运行期 L：只写前缀 (shape//L)*L，必须由调用方传入输出缓冲；'
                     '未覆盖的尾部保留传入值。\n')
    elif reads_outputs:
        call_note = (f'# ⚠️ 原地更新：{", ".join(reads_outputs)}'
                     ' 同时被读取，必须由调用方传入（含初值）；\n'
                     '#    产物不自动分配这些缓冲。\n')
    else:
        call_note = '# 输入按源形参顺序传入，自动分配输出；源launch必须完整且唯一覆盖输出。\n'
    def render_store(name, indices, consts):
        # 常量下标维的取值可能是 int，也可能是**运行期地址表达式**（`fin[.., nx-1, ..]`）
        parts = []
        for dim, axis in enumerate(indices):
            if axis != AXIS_CONSTANT:
                parts.append(axes[axis])
                continue
            value = consts[dim]
            parts.append(str(value) if isinstance(value, int) else emit_expr(value, axes))
        return ', '.join(parts)

    materialized = tuple(getattr(spec.compute, 'materialized_bindings', ()) or ())
    for index, (_bname, bvalue) in enumerate(materialized):
        axes[('materialized', id(bvalue))] = f'_tl_bind_{index}[_tl_lane]'
        axes[('bound', index)] = f'_tl_bind_{index}[_tl_lane]'
    bind_alloc = ''.join(
        f'            _tl_bind_{index} = T.alloc_fragment((block,), "{bvalue.dtype}")' + chr(10)
        for index, (_bname, bvalue) in enumerate(materialized))
    # 物化表达式的读取必须是直接读取（不能用 replacement，否则自引用）。
    bind_axes = {key: value for key, value in axes.items() if key[0] != 'materialized'}
    # 槽位 → 定义顺序里的位置（group_bindings 只给名字和值，按名字找到槽位）
    slot_of_name = {}
    for index, (bname, _bvalue) in enumerate(materialized):
        slot_of_name.setdefault(bname, index)
    group_materialized = tuple(getattr(spec.compute, 'group_bindings', ()) or ())

    def bind_writes_for(group_id):
        """该组组首要写的物化绑定（定义顺序）。旧 Compute 没有 group_bindings 时退回第 0 组。"""
        if group_materialized:
            items = group_materialized[group_id] if group_id < len(group_materialized) else ()
        else:
            items = tuple(materialized) if group_id == 0 else ()
        lines = []
        for entry in items:
            # 新格式：(slot, name, value)；旧行为兼容 (name, value)
            if len(entry) == 3:
                slot, _bname, bvalue = entry
            else:
                _bname, bvalue = entry
                slot = slot_of_name.get(_bname)
                if slot is None:
                    continue
            lines.append(
                f'                    _tl_bind_{slot}[_tl_lane] = T.cast('
                f'{emit_expr(bvalue, bind_axes)}, "{materialized[slot][1].dtype}")')
        return ''.join(line + chr(10) for line in lines)

    parallel_blocks = ''
    # 候选优化资格（本片范围）：仅 lower_map、自动分配全部输出（in_place=False）、
    # 单 group、无 materialized/group 绑定。其它路径保持原文本。
    opt_eligible = (not in_place) and len(groups) == 1 and not materialized \
        and not any(group_materialized)
    rd_alloc = ''
    reuse_alloc = ''
    reuse_slots = []
    for group_id, group in enumerate(groups):
        # 每组重新算共享读取：跨组的读取必须在新组里重新取值，
        # 复用一个更早算出的局部变量就会读到旧值（静默错误）。
        definitions, shared = emit_shared_reads(
            [(all_stores[position][3], axes) for position in group], f'_tl_read_{group_id}')
        materialize_keys = {}
        if opt_eligible:
            strict_counts, lazy_keys, key_dtypes, dependent_keys = {}, set(), {}, set()
            for position in group:
                _shared_scan(all_stores[position][3], axes, strict_counts,
                             lazy_keys, key_dtypes, dependent_keys)
            for key, count in strict_counts.items():
                if count <= 1 or key in lazy_keys or key in dependent_keys:
                    continue
                name = key[1]
                dtypes_for_key = key_dtypes.get(key) or set()
                if len(dtypes_for_key) != 1:
                    continue
                dtype = next(iter(dtypes_for_key))
                if name in all_outputs or dtype not in ('float16', 'float32', 'float64'):
                    continue
                materialize_keys[key] = ('_tl_rd_%d' % len(materialize_keys), dtype)
            if materialize_keys:
                rd_alloc = ''.join(
                    f'            {frag} = T.alloc_fragment((block,), "{dtype}")' + chr(10)
                    for frag, dtype in materialize_keys.values())
                new_defs, new_shared = [], {}
                for key, local in shared.items():
                    name, coords = key[1], key[2]
                    if key in materialize_keys:
                        frag = materialize_keys[key][0]
                        new_defs.append(
                            f'{frag}[_tl_lane] = {name}[{", ".join(coords)}]')
                        new_shared[key] = f'{frag}[_tl_lane]'
                    else:
                        new_defs.append(f'{local} = {name}[{", ".join(coords)}]')
                        new_shared[key] = local
                definitions, shared = new_defs, new_shared
        group_setup = ''.join(' ' * 20 + line + '\n' for line in definitions)
        group_writes = ''
        store_texts = []
        for position in group:
            name, indices, consts, store_expression = all_stores[position]
            reuse_map = {}
            if opt_eligible:
                # 每个 store 独立计划、独立命名，不跨 store 共享槽。
                for slot_index, (root, dtype) in enumerate(reuse_plan(store_expression)):
                    slot = '_tl_val_%d_%d' % (position, slot_index)
                    reuse_map[('reuse', id(root))] = f'{slot}[_tl_lane]'
                    # 定义期不使用 reuse 替换（避免自引用/引用另一根）。
                    group_writes += (' ' * 20 + f'{slot}[_tl_lane] = '
                                     f'{emit_expr(root, axes | shared)}' + '\n')
                    reuse_slots.append((slot, dtype))
            store_texts.append(
                '                    %s[%s] = T.cast(%s, "%s")\n' % (
                    name, render_store(name, indices, consts),
                    emit_expr(store_expression, axes | shared | reuse_map),
                    dict(getattr(spec, 'tensor_dtypes', {}) or {}).get(name, 'float32')))
        group_stores = ''.join(store_texts)
        parallel_blocks += (
            '            for _tl_lane in T.Parallel(block):\n'
            '                _tl_p = _tl_bid * block + _tl_lane\n'
            f'                if {guard_condition}:\n'
            + bind_writes_for(group_id)
            + f'{group_setup}{group_writes}{group_stores}')
    reuse_alloc = ''.join(
        f'            {slot} = T.alloc_fragment((block,), "{dtype}")' + chr(10)
        for slot, dtype in reuse_slots)
    return f'''# 逐元素计算；连续张量，输入形状由轴投影推导，内部保留float32/float64提升。
{call_note}{extent_note}{extra_note}{inferred_note(spec)}import tilelang
import tilelang.language as T

@tilelang.jit(out_idx={out_idx},
              pass_configs={{"tl.disable_warp_specialized": True}}, compile_flags=["--fmad=false"])
def {spec.name}(shape, _tl_extra=(), block=128):
    _tl_shape = tuple(shape)
    _tl_extra = tuple(_tl_extra)
    if len(_tl_extra) != {len(extra_dims)}:
        raise ValueError("shape extras must have {len(extra_dims)} entries")
{extra_bind}    if len(_tl_shape) != {dimensions} or any(type(d) is not int or d <= 0 for d in _tl_shape):
        raise ValueError("shape must contain {dimensions} positive integers")
    if type(block) is not int or block <= 0 or block % 32:
        raise ValueError("block must be a positive multiple of 32")
{divisible_check}{equality_checks}{const_shape_check}{outer_size}    @T.prim_func
    def _tl_main({parameters}):
{inner_size}        with T.Kernel({grid}, threads=128) as _tl_bid:
{runtime_assert}{bind_alloc}{rd_alloc}{reuse_alloc}{parallel_blocks}    return _tl_main
'''


def lower_flat_map(spec):
    """展平/仿射索引的逐元素映射 → T.Kernel + T.Parallel。

    域：源码 guard 给出的上界 `_tl_size`（签名整数标量 / 张量 shape / 正常量），
    不是从一维 grid 总量反推，也不假设 `xDim == gridDim.x*blockDim.x`。
    每个张量都按展平坐标直接寻址；签名标量即使展平后不再使用也按源顺序保留。
    """
    axis = spec.idx_vars[0]
    kind = spec.flat_extent[0]
    from abi import dtype_of, is_scalar, store_dtype
    tensors = [name for name in spec.params if not is_scalar(spec, name)]
    out_dtype = store_dtype(spec)
    parameters = ', '.join(
        f'{name}: T.{dtype_of(spec, name)}' if is_scalar(spec, name)
        else f'{name}: T.Tensor(_tl_shape, "{dtype_of(spec, name)}")' for name in spec.params)
    axes = {axis: '_tl_p'}
    for name in tensors:
        axes[('shape', name, 0)] = '_tl_size'
    lines = [f'{name} = {emit_expr(value, axes)}'
             for name, value in spec.compute.bindings]
    expression = emit_expr(spec.compute.expression, axes)
    body = ''.join(f'                {line}\n' for line in lines)
    # 域上界是运行期值（签名标量不能在构造期比较），生成签名要求调用方传 shape=(上界,)；
    # 这条要求写进模块头，不能替代运行期检查。
    if kind == 'scalar':
        bound = f'源 guard 上界 `{spec.flat_extent[1]}`'
    elif kind == 'shape':
        bound = f'`{spec.flat_extent[1]}.shape[{spec.flat_extent[2]}]`'
    elif kind == 'size':
        bound = f'`{spec.flat_extent[1]}.size`'
    else:
        bound = f'常量 {spec.flat_extent[1]}'
    extent_check = ''
    # A 片（基址改写）：源码只覆盖 [0, (shape[d]//L)*L)；展平域是整段 shape[d]。
    # 二者相等的前提是 shape[d] % L == 0，构造期显式校验，不做静默截断/扩展。
    divisible = tuple(getattr(spec, 'divisible_extent', ()) or ())
    if divisible:
        name, dim, length = divisible
        extent_check = (
            f'    if _tl_shape[{dim}] % {length}:\n'
            f'        raise ValueError("source requires {name}.shape[{dim}] divisible by {length} '
            f'(flat base rewrite)")\n')
    # C2：运行期 L —— 枚举仍按整段静态网格，写回条件截断到前缀 (shape[d]//L)*L；
    # 尾部线程空转不写；**自动分配的输出尾部内容未定义（不是零）**，
    # 因此运行期 L 强制调用方传入输出（out_idx=[]），尾部保留调用方原值。
    runtime_base = tuple(getattr(spec, 'flat_base_runtime', ()) or ())
    runtime_assert = ''
    if runtime_base:
        name, dim, length = runtime_base
        guard_condition = ('(T.cast(_tl_p, "int64") < '
                           '(T.cast(_tl_shape[%d], "int64") // %s) * %s)' % (dim, length, length))
        runtime_assert = '            T.Assert(%s > 0, "%s must be > 0")\n' % (length, length)
    else:
        # 展平映射的上界来自 flat_extent；目前没有下界/偏移形态。
        guard_condition = '_tl_p < _tl_size'
    # R2：flat 入口与 grid 入口统一——运行期 L 一律由调用方传入输出（out_idx=[]）。
    reads_output = in_place_output([spec.compute.expression], spec.store_target)
    caller_output = bool(runtime_base)
    in_place = reads_output or caller_output
    out_idx = out_idx_text(in_place, spec.params.index(spec.store_target))
    if caller_output and not reads_output:
        call_note = (f'# ⚠️ 运行期 L：只写前缀 (shape//L)*L，必须由调用方传入 {spec.store_target}；'
                     '未覆盖的尾部保留传入值。\n')
    elif reads_output:
        call_note = (f'# ⚠️ 原地更新：{spec.store_target} 同时被读取，必须由调用方传入（含初值）。\n')
    else:
        call_note = ''
    return f'''# 展平/仿射索引逐元素映射；域上界 = {bound}，调用方必须传 shape=(该值,)。
# 前提：源 launch 对 [0, 上界) 内每个索引至多枚举一次（本项目不覆盖写竞争程序）。
{call_note}import tilelang
import tilelang.language as T

@tilelang.jit(out_idx={out_idx},
              pass_configs={{"tl.disable_warp_specialized": True}}, compile_flags=["--fmad=false"])
def {spec.name}(shape, block=128):
    _tl_shape = tuple(shape)
    if len(_tl_shape) != 1 or any(type(d) is not int or d <= 0 for d in _tl_shape):
        raise ValueError("shape must contain 1 positive integer")
    if type(block) is not int or block <= 0 or block % 32:
        raise ValueError("block must be a positive multiple of 32")
    _tl_size = _tl_shape[0]
{extent_check}    @T.prim_func
    def _tl_main({parameters}):
        with T.Kernel(T.ceildiv(_tl_size, block), threads=128) as _tl_bid:
{runtime_assert}            for _tl_lane in T.Parallel(block):
                _tl_p = _tl_bid * block + _tl_lane
                if {guard_condition}:
{body}                    {spec.store_target}[_tl_p] = T.cast({expression}, "{out_dtype}")
    return _tl_main
'''
