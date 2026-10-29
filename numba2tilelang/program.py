# -*- coding: utf-8 -*-
"""程序级访问约束：物理维度、输出坐标和各归约坐标分别表示。"""
import ast
from semantic import (AXIS_CONSTANT, INT_DTYPES, Expr, Unsupported, REDUCTION_IDENTITY, polynomial,
                      reads, referenced_axes, shape_axis, validate_names)


def walk(expr):
    yield expr
    for arg in expr.args:
        yield from walk(arg)


def _references_own_shape(expression, name):
    """偏移表达式里是否含 `name.shape[k]`（该张量自己的物理维长度）。"""
    if not isinstance(expression, Expr):
        return False
    stack = [expression]
    while stack:
        node = stack.pop()
        if node.op == 'shape' and node.value[0] == name:
            return True
        stack.extend(node.args)
    return False


RUNTIME_RESERVED_NAMES = {'block', 'shape', 'reduction_extents', 'reduction_extent',
                        '_tl_shape', '_tl_extents', '_tl_extra', '_tl_size',
                        '_tl_p', '_tl_bid', '_tl_lane', '_tl_k'}


def validate_program(spec):
    p = spec.program
    axes = p.output_axes
    # 输出轴可以多于网格轴：归约之后对每个输出元素写一次的循环会引入额外输出维，
    # 那些维由 free_axes 在写回循环里遍历（层归一化、softmax）。
    if spec.grid_dims not in (1, 2, 3) or len(axes) < spec.grid_dims \
            or len(set(axes)) != len(axes) or not set(spec.idx_vars) <= set(axes) \
            or len(spec.idx_vars) != spec.grid_dims:
        raise Unsupported('输出需完整对应一到三个独立grid轴')
    parent, constants = {}, {}

    def root(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = root(parent[x])
        return parent[x]

    def merge(a, b):
        a, b = root(a), root(b)
        if a == b:
            return
        # 只合并长度约束；Expr中的输出/归约坐标仍各自保留。
        # 输出尺寸等式由layout记录，生成模块必须完整实现这些约束。
        parent[b] = a

    # 非零起点归约的归约轴（窗口归约）：该轴索引的物理维是整个被扫描轴的长度，
    # 与归约长度不再相等，不能参与长度合并，只能作为独立构造期参数传入。
    window_axes = {stage.reduction.axis for stage in p.stages
                   if getattr(stage.reduction, 'start_expr', None) is not None
                   and polynomial(stage.reduction.start_expr)}
    output_dims = {axis: ('out', dim) for dim, axis in enumerate(axes)}
    for token in output_dims.values():
        root(token)
    ranks, accesses, shape_refs = {}, [], []
    contexts = []
    scoped = hasattr(p, 'loops')
    if scoped:
        from scoped_validation import validate_scopes
        validate_scopes(p, spec.idx_vars)
        axis_tokens = {loop.axis: ('loop', lid) for lid, loop in enumerate(p.loops)}
    else:
        axis_tokens = {stage.reduction.axis: ('reduce', sid) for sid, stage in enumerate(p.stages)}
    for sid, stage in enumerate(p.stages):
        r = stage.reduction
        # 归约轴允许与输出轴同名：两阶段的写回循环会复用同一个迭代变量名，
        # 但两处处于不同上下文（stage 的 context 里该名字指向 reduce token）。
        if r.accumulator in axes or r.axis == r.accumulator:
            raise Unsupported('归约轴、累加器、输出轴必须互异')
        if r.op not in REDUCTION_IDENTITY:
            raise Unsupported('未支持的归约操作: ' + r.op)
        if r.init.op == 'const':
            if r.init.value != REDUCTION_IDENTITY[r.op] \
                    and not _equivalent_zero_element(spec, r):
                raise Unsupported('归约初值必须是对应单位元')
        elif r.axis in referenced_axes(r.init):
            raise Unsupported('归约初值不能依赖归约轴')
        if r.step != '1':
            raise Unsupported('只支持步长1的归约')
        context = dict(output_dims)
        if scoped:
            loop = p.loops[stage.loop]
            context.update({axis: axis_tokens[axis] for axis in loop.context if axis not in output_dims})
            merge(('reduce', sid), axis_tokens[r.axis])
            bindings = loop.bindings if sid == loop.results[0] else ()
        else:
            context[r.axis] = ('reduce', sid)
            bindings = stage.bindings
        root(context[r.axis])
        contexts.append(context)
        # 动态初值（`m = F[b*win, i]`）也是本阶段的读取，rank/维度必须一并登记。
        expressions = [stage.expression, stage.reduction.init] + [value for _, value in bindings]
        for expression in expressions:
            accesses.append((expression, context, sid))
    for _, expression in p.bindings:
        accesses.append((expression, output_dims, len(p.stages)))
    accesses.append((p.expression, output_dims, len(p.stages)))
    # 附加写回的读取与主写回同处输出坐标域，必须一并进入布局分析。
    extra_stores = tuple(getattr(p, 'extra_stores', ()) or ())
    for _extra_name, _extra_indices, _extra_consts, extra_expression in extra_stores:
        accesses.append((extra_expression, output_dims, len(p.stages)))
    # 带运行期标量系数的读取（`F[b*win, i]`）不是"第 b 行"，该物理维同样不能绑定输出维。
    # 运行期**偏移**（`QKV[i, half + d]`，half = QKV.shape[1] // 3）同理：同一个物理维
    # 上既有轴访问又有运行期地址，说明该维长度与输出轴无关；绑定上去会静默生成越界下标。
    scaled_dims = set()
    for expression, _, _ in accesses:
        for node in walk(expression):
            if node.op == 'read':
                name, indices = node.value
                for dim, scale in enumerate(getattr(node, 'scales', ()) or ()):
                    if isinstance(scale, tuple):
                        scaled_dims.add((name, dim))
                for dim, offset in enumerate(getattr(node, 'offsets', ()) or ()):
                    # 只有偏移里引用了**该张量自身**的 shape 才说明长度与输出轴无关
                    # （`QKV[i, QKV.shape[1]//3 + d]`：一维切成三段）。
                    # `Qin[i, d + Qout.shape[1]//2]`（RoPE）由输出长度决定，仍然等长。
                    if _references_own_shape(offset, name):
                        scaled_dims.add((name, dim))
    for expression, context, sid in accesses:
        for node in walk(expression):
            if node.op == 'result':
                if not 0 <= node.value < sid:
                    raise Unsupported('归约结果引用必须指向已完成阶段')
                if scoped:
                    producer = p.stages[node.value]
                    required = set(p.loops[producer.loop].context) - {producer.reduction.axis}
                    if not required <= set(context):
                        raise Unsupported('子归约结果不能逃逸其外层坐标作用域')
            if node.op == 'shape':
                shape_refs.append(node.value)
        for name, indices in reads(expression):
            # 常量下标维不参与轴匹配（`F[0, i]` 的首元素初值）。
            axes_used = [axis for axis in indices if axis != AXIS_CONSTANT]
            if len(set(axes_used)) != len(axes_used) \
                    or any(axis not in context for axis in axes_used):
                raise Unsupported('读取只支持当前阶段的互异计算轴')
            if name in ranks and ranks[name] != len(indices):
                raise Unsupported('同一张量的读取rank不一致')
            ranks[name] = len(indices)
            for dim, axis in enumerate(indices):
                if axis == AXIS_CONSTANT:
                    continue
                if axis in window_axes or (name, dim) in scaled_dims:
                    continue
                merge(('buffer', name, dim), context[axis])
    validate_names(spec, set(ranks), set(axes) |
                   {stage.reduction.accumulator for stage in p.stages} |
                   {name for name, _ in p.bindings})
    # 输出按**完整写回下标**登记：常量下标维不进并行域，但它仍是张量的物理维，
    # rank 与长度都必须保留（`positions[tid, 0]` 的 rank 是 2，不是 1）。
    output_stores = {p.output: (tuple(getattr(p, 'store_indices', ()) or ()) or axes,
                                tuple(getattr(p, 'store_constants', ()) or ()))}
    for name, indices, consts, _expression in extra_stores:
        output_stores.setdefault(name, (indices, consts))
    out_dim = {axis: dim for dim, axis in enumerate(axes)}
    for name, (indices, _consts) in output_stores.items():
        for dim, axis in enumerate(indices):
            if axis == AXIS_CONSTANT:
                continue
            if axis not in out_dim:
                raise Unsupported('写回下标必须使用输出轴: ' + name)
            merge(('buffer', name, dim), ('out', out_dim[axis]))
    all_ranks = dict(ranks)
    for name, (indices, _consts) in output_stores.items():
        if name in all_ranks and all_ranks[name] != len(indices):
            raise Unsupported('同一张量的写回rank与读取rank不一致: ' + name)
        all_ranks[name] = len(indices)
    for name, dim in shape_refs:
        if name not in all_ranks or dim >= all_ranks[name]:
            raise Unsupported('shape引用没有对应的合法张量维度')
    def loop_reads(lid):
        loop = p.loops[lid]
        found = []
        for _, expression in loop.bindings:
            found += reads(expression)
        for result in loop.results:
            found += reads(p.stages[result].expression)
        for child in loop.children:
            found += loop_reads(child)
        return found

    runtime_extents = {}
    for sid, stage in enumerate(p.stages):
        if scoped and sid != p.loops[stage.loop].results[0]:
            continue  # 所有成员已绑定同一loop域，证据只按实际循环体检查一次。
        # 嵌套归约的发射模板没有窗口起点语义（只用 ceildiv(stop, block) 从 0 开始）。
        # 非零起点若被放过会**静默丢掉 start**（实测 range(1,n) 被当成 range(n)），
        # 因此这里一律拒绝，而不是忽略。常量/运行期终点都一样。
        if scoped and str(stage.reduction.start).strip() != '0':
            raise Unsupported('嵌套归约暂不支持非零起点（range 的 start 必须为 0）')
        start_expr = getattr(stage.reduction, 'start_expr', None)
        start_poly = None if start_expr is None else polynomial(start_expr)
        if start_poly:
            # 非零起点（窗口归约）：长度是 stop-start，与任何物理维都不绑定。
            if len(p.stages) != 1:
                raise Unsupported('非零起点归约暂不支持多累加器')
            stop_expr = getattr(stage.reduction, 'stop_expr', None)
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
            if any(stage.reduction.axis in key for key in length):
                raise Unsupported('归约域长度必须与归约轴无关')
            if all(not key for key in length):
                if length.get((), 0) <= 0:
                    raise Unsupported('归约域长度必须是正整数')
                constants[('reduce', sid)] = length[()]
            # 运行期标量长度不绑定物理维，留待下方 extra 分配独立 extent 槽位。
            continue
        node = ast.parse(stage.reduction.stop, mode='eval').body
        dimension = shape_axis(node)
        if dimension:
            name, dim = dimension
            # 终点必须由当前循环对该物理维的访问提供证据，不能把M错当K。
            if scoped:
                candidates = loop_reads(stage.loop)
            else:
                candidates = reads(stage.expression)
                for _, expression in stage.bindings:
                    candidates += reads(expression)
            if not any(n == name and dim < len(idx) and idx[dim] == stage.reduction.axis
                       for n, idx in candidates):
                raise Unsupported('归约终点必须指向本阶段读取的归约维')
            merge(('reduce', sid), ('buffer', name, dim))
        elif isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0:
            constants[('reduce', sid)] = node.value
        elif isinstance(node, ast.Name) and node.id in (spec.scalar_params or {})                 and spec.scalar_params[node.id] in INT_DTYPES:
            # C1：scoped 的运行期归约长度只接受**签名 int64** 标量，且直接用 `n` 作循环上界，
            # 不再占 `_tl_extents` 槽位；pipeline 既有 `_tl_extents` 路径保持不变。
            if not scoped:
                continue
            if spec.scalar_params[node.id] != 'int64':
                raise Unsupported('运行期归约长度只支持签名 int64 标量（%s 声明为 %s）'
                                  % (node.id, spec.scalar_params[node.id]))
            # R4：`scalar_params` 含无签名推断项；运行期长度必须来自**静态签名**，
            # 推断出来的 float64/int64 不是调用方契约。
            if node.id in (getattr(spec, 'inferred_scalars', ()) or ()):
                raise Unsupported('运行期归约长度需要显式 int64 类型来源（静态签名或 scalar_types 声明；推断不算契约）：%s'
                                  % node.id)
            # R5：运行期标量名不能与模板局部名冲突（否则 prim_func 形参遮蔽 builder 局部）。
            if node.id in RUNTIME_RESERVED_NAMES:
                raise Unsupported('运行期标量名与模板局部名冲突: ' + node.id)
            candidates = loop_reads(stage.loop)
            target = None
            for name, indices in candidates:
                if stage.reduction.axis in indices:
                    target = (name, indices.index(stage.reduction.axis))
                    break
            if target is None:
                raise Unsupported('运行期归约长度必须对应一次读取该归约轴的张量维')
            runtime_extents[sid] = (node.id, target[0], target[1])
            continue
        else:
            raise Unsupported('归约域不是shape或正常量')
    # guard直接关联输出轴，不能因为尺寸相等就把坐标视为可交换。
    guard_constants = {}
    if spec.guard:
        guard = ast.parse(spec.guard, mode='eval').body
        terms = guard.values if isinstance(guard, ast.BoolOp) and isinstance(guard.op, ast.And) else [guard]
        covered = []
        for term in terms:
            if not isinstance(term, ast.Compare) or len(term.ops) != 1 \
                    or not isinstance(term.ops[0], ast.Lt) or not isinstance(term.left, ast.Name):
                raise Unsupported('只支持完整输出边界guard')
            dim = shape_axis(term.comparators[0])
            if dim is None:
                # 标量上界 guard（`if e < n_expert`）：上界是签名整数标量。
                bound = term.comparators[0]
                scalars = dict(getattr(spec, 'scalar_params', {}) or {})
                if isinstance(bound, ast.Name) and bound.id in scalars \
                        and scalars[bound.id] in ('int32', 'int64') \
                        and term.left.id in output_dims:
                    spec.program_extent_scalar = bound.id
                    covered.append(term.left.id)
                    continue
                # 模块级 int 字面量常量上界（`if x < NX`，20261025-144）：域是常量，
                # 因此输出形状变成**生成期常量**——所有由该常量界定的张量维度都按等值校验
                # （见下方 checks），不能只核输出、也不能退化成 `<=`。
                consts = dict(getattr(spec, 'guard_int_constants', {}) or {})
                if isinstance(bound, ast.Name) and bound.id in consts \
                        and term.left.id in output_dims:
                    value = consts[bound.id]
                    if type(value) is not int or value <= 0:
                        raise Unsupported('常量上界 guard 的常量必须是正整数: ' + bound.id)
                    guard_constants[term.left.id] = (value, bound.id)
                    covered.append(term.left.id)
                    continue
                raise Unsupported('guard轴与张量维度不对应')
            if dim[0] not in all_ranks or dim[1] >= all_ranks[dim[0]] \
                    or term.left.id not in output_dims \
                    or root(('buffer', *dim)) != root(output_dims[term.left.id]):
                raise Unsupported('guard轴与张量维度不对应')
            covered.append(term.left.id)
        # guard 只需覆盖网格轴：额外的输出维由写回循环遍历，天然在域内。
        if len(covered) != len(spec.idx_vars) or set(covered) != set(spec.idx_vars):
            raise Unsupported('guard必须完整覆盖输出轴')
        # 常量上界登记为输出维的**物理长度**：并查集把同一根上的所有张量维绑在一起，
        # 随后统一发射 `_tl_shape[d] != 常量 → raise`，输出的每个轴、以及由同一根
        # 关联的输入维都被等值校验（不是 `<=`，也不能只核输出）。
        if guard_constants:
            dims = {}
            for axis, (value, name) in guard_constants.items():
                dim = output_dims[axis][1]
                if ('out', dim) in constants and constants[('out', dim)] != value:
                    raise Unsupported('常量上界 guard 与其它常量界冲突: ' + name)
                constants[('out', dim)] = value
                dims[dim] = value
            spec.constant_guard = True
            # 单轴 sum 模板（lower_sum）不读 layout['checks']，必须单独拿到同一组等值校验。
            spec.constant_guard_dims = dims
    lengths, equalities = {}, []
    for dim in range(len(axes)):
        key, length = root(('out', dim)), f'_tl_shape[{dim}]'
        if key in lengths:
            equalities.append((length, lengths[key]))
        else:
            lengths[key] = length
    checks = []
    constant_values = {}
    for token, value in constants.items():
        key = root(token)
        if key in constant_values and constant_values[key] != value:
            raise Unsupported('同一物理归约维具有冲突的常量长度')
        constant_values[key] = value
        if key in lengths:
            checks.append((lengths[key], value))
        else:
            lengths[key] = str(value)
    extra = []
    for sid in range(len(p.stages)):
        key = root(('reduce', sid))
        if sid in runtime_extents:
            # 运行期 stage：上界直接用签名标量，不占 builder 槽位；
            # **不写 lengths**——物理维长度必须由 dim_extra 的 `_tl_dim_*` 给出，
            # 否则会把循环上界 n 误当成张量的物理长度（Assert 会退化成 n<=n）。
            continue
        if key not in lengths:
            lengths[key] = f'_tl_extents[{len(extra)}]'
            extra.append(key)
    # 窗口归约扫描的物理维：与归约长度无关，按 (张量名, 维) 的稳定顺序占后续槽位。
    # 窗口归约扫描的物理维由窗口机制单独占槽（长度 != 归约长度）。
    # 其余"从未与输出轴/归约轴绑定"的维是**常量下标维**（`positions[tid, 0]`）：
    # pipeline 模板没有 `_tl_extra` 通道，塞进 reduction_extents 会把张量声明成
    # 错误形状（实测 `(n, 2)` 被声明成 `(n, n)`，运行时 ABI 拒绝）。一律拒绝。
    window_dims = set()
    for stage in p.stages:
        start_expr = getattr(stage.reduction, 'start_expr', None)
        if start_expr is None or not polynomial(start_expr):
            continue
        for expression, _context, _sid in accesses:
            for node in walk(expression):
                if node.op == 'read' and stage.reduction.axis in node.value[1]:
                    name, indices = node.value
                    for dim, axis in enumerate(indices):
                        if axis == stage.reduction.axis:
                            window_dims.add((name, dim))
    dim_extra = []
    for name in sorted(all_ranks):
        for dim in range(all_ranks[name]):
            key = root(('buffer', name, dim))
            if key not in lengths:
                if (name, dim) in window_dims:
                    lengths[key] = f'_tl_extents[{len(extra)}]'
                    extra.append(key)
                    continue
                # 从未绑定任何计算轴的维是**常量下标维**（`positions[tid, 0]` 的分量位、
                # `f[0, j, i]` 的通道位）：长度推不出来，按 `_tl_extra` 参数由调用方给出。
                # 写回按每个 store 自己的完整下标渲染，常量维不会再被 free 轴覆盖。
                dim_extra.append((name, dim))
                lengths[key] = f'_tl_dim_{name}_{dim}'
    dims = {name: tuple(lengths[root(('buffer', name, d))] for d in range(rank))
            for name, rank in all_ranks.items()}
    extents = tuple(runtime_extents[sid][0] if sid in runtime_extents
                    else lengths[root(('reduce', sid))] for sid in range(len(p.stages)))
    # 值依赖由坐标和result边推导，不能用上面的长度并查集代替。
    stage_axes, stage_dependencies = [], []

    def dependencies(expression):
        used, results = set(), set()
        for node in walk(expression):
            if node.op == 'read':
                used.update(set(node.value[1]) & set(axes))
            elif node.op == 'result':
                results.add(node.value)
                used.update(stage_axes[node.value])
        return tuple(axis for axis in axes if axis in used), tuple(sorted(results))

    def domain_axes(reduction):
        """归约域里的索引变量：窗口归约 `range(b*win, (b+1)*win)` 的结果依赖 b。"""
        used = set()
        for expression in (getattr(reduction, 'start_expr', None),
                           getattr(reduction, 'stop_expr', None)):
            for node in (() if expression is None else walk(expression)):
                if node.op == 'index':
                    used.add(node.value)
        return used

    for stage in p.stages:
        used, results = dependencies(stage.expression)
        # 两阶段里归约轴也可能同时是输出轴（`Y[i,d]` 的 d 被 `for d` 归约过）。
        # 该轴对阶段结果不是自由坐标，必须排除，否则网格会覆盖整个输出域。
        used = tuple(axis for axis in used if axis != stage.reduction.axis)
        domain_used = domain_axes(stage.reduction)
        if domain_used:
            used = tuple(axis for axis in axes if axis in set(used) | domain_used)
        stage_axes.append(used)
        stage_dependencies.append(results)
    output_axes, output_dependencies = dependencies(p.expression)
    spec.layout = {'dims': dims, 'extents': extents, 'extra': len(extra),
                   'runtime_extents': tuple(sorted(runtime_extents.items())),
                   'dim_extra': tuple(dim_extra), 'checks': checks,
                   'equalities': tuple(equalities),
                   'stage_axes': tuple(stage_axes), 'stage_dependencies': tuple(stage_dependencies),
                   'output_axes': output_axes, 'output_dependencies': output_dependencies}
    if not scoped:
        # 归约不依赖的输出坐标无需进入网格；嵌套路径结构不同，暂不使用该划分。
        from lowering import split_schedule_axes
        spec.layout['grid_axes'], spec.layout['remaining_axes'] = split_schedule_axes(spec)
    if scoped:
        coordinate_dependencies = []
        for stage in p.stages:
            used = set()
            for node in walk(stage.expression):
                if node.op == 'read':
                    used.update(node.value[1])
                elif node.op == 'result':
                    used.update(coordinate_dependencies[node.value])
            used.discard(stage.reduction.axis)
            coordinate_dependencies.append(tuple(axis for axis in p.loops[stage.loop].context if axis in used))
        spec.layout['stage_coordinate_dependencies'] = tuple(coordinate_dependencies)
        spec.layout['stage_contexts'] = tuple(p.loops[stage.loop].context for stage in p.stages)
        # roots现在是循环ID；单结果迁移保持旧编号，显式记录结果所有权。
        spec.layout['roots'] = p.roots
        spec.layout['stage_loops'] = tuple(stage.loop for stage in p.stages)
        spec.layout['loop_results'] = tuple(loop.results for loop in p.loops)
    spec.pattern = 'nested' if scoped else ('pipeline' if p.stages else 'elementwise')
    return spec


# float32 的最小有限值（约 -3.4028235e38）：小于它的负数对 float32 数据等价于 -inf。
_FLOAT32_MIN = float.fromhex('-0x1.fffffep+127')


def _equivalent_zero_element(spec, reduction, ranks=None):
    """`max` 用 `-1e300`、`min` 用 `1e300` 这类初值，对 float32 张量等价于 ±inf 单位元。

    只对 float32 成立：float64 数据可以小于 -1e300，此时初值会成为结果本身。
    """
    if reduction.op not in ('max', 'min'):
        return False
    value = reduction.init.value
    if type(value) is not float:
        return False
    dtypes = dict(getattr(spec, 'tensor_dtypes', {}) or {})
    program = getattr(spec, 'program', None)
    sources = set()
    if program is not None:
        for stage in getattr(program, 'stages', ()):
            sources |= {name for name, _ in reads(stage.expression)}
            for _, bound in getattr(stage, 'bindings', ()):
                sources |= {name for name, _ in reads(bound)}
        for _, bound in getattr(program, 'bindings', ()):
            sources |= {name for name, _ in reads(bound)}
    if not sources:
        return False
    if any(dtypes.get(name, 'float32') != 'float32' for name in sources):
        return False
    if reduction.op == 'max':
        # 常用哨兵是 `-3.402823466e+38`（比 float32 最小值略大一位有效数字），
        # 直接和 `-3.4028235e38` 比会漏掉；这里按量级判定「足够小于任何 float32 数据」。
        return value <= -3.4e38
    return value >= 3.4e38
