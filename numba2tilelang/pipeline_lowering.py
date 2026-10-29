# -*- coding: utf-8 -*-
"""按已验证的阶段依赖发射，不按算子名字选择模板。"""
from lowering import (emit_expr, emit_shared_reads, emit_layout_checks, flat_indices, literal,
                      split_schedule_axes, in_place_output, REDUCTION_PRIMITIVE,
                      REDUCTION_COMBINE)
from semantic import AXIS_CONSTANT, REDUCTION_IDENTITY, Unsupported, polynomial, reads
from selection import group_reductions


def lower_pipeline(spec):
    p, layout = spec.program, spec.layout
    axes = p.output_axes
    rank = len(axes)
    # 同一归约域内的附加写回（层归一化的 Mean / Var）。
    extra_stores = tuple(getattr(p, 'extra_stores', ()) or ())
    # 同一张量的多个常量下标写回只算一个输出参数（`acc[tid,0]`、`acc[tid,1]`）。
    outputs = list(dict.fromkeys([p.output] + [name for name, _, _, _ in extra_stores]))
    dimension = {axis: dim for dim, axis in enumerate(axes)}
    # 归约只依赖部分输出坐标时，网格只需覆盖这些坐标；其余在写回循环里遍历，
    # 否则同一行统计量会按每个输出元素重算一遍。
    grid_axes, free_axes = split_schedule_axes(spec)
    if not grid_axes or not free_axes:
        grid_axes, free_axes = axes, ()
    grid_dims = [f'_tl_shape[{dimension[axis]}]' for axis in grid_axes]
    # 标量上界 guard（`if e < n_expert`）：并行域是 [0, n)，由调用方保证不越界。
    extent_scalar = getattr(spec, 'program_extent_scalar', None)
    if extent_scalar and len(grid_dims) == 1:
        grid_dims = [extent_scalar]
    free_dims = [f'_tl_shape[{dimension[axis]}]' for axis in free_axes]
    coords = dict(zip(grid_axes, flat_indices(grid_dims)))
    for name, dims in layout['dims'].items():
        for dim, length in enumerate(dims):
            coords[('shape', name, dim)] = length
    free = set(free_axes)
    for stage in p.stages:
        for _, indices in reads(stage.expression):
            for axis in indices:
                # 归约轴本身可以在 free 里：它在块内由 _tl_k 遍历，不是输出自由坐标。
                if axis in free and axis != stage.reduction.axis:
                    raise Unsupported('归约阶段不能依赖未纳入网格的输出坐标: ' + axis)
    # 签名标量（含无签名时推断的标量）在主 kernel 里以 `T.<dtype>` 标量参数出现，
    # 没有 shape 也没有 layout 维度；不能按张量展开。
    from abi import dtype_of, is_scalar
    parameters = ', '.join(
        f'{name}: T.{dtype_of(spec, name)}' if is_scalar(spec, name)
        else f'{name}: T.Tensor(({", ".join(layout["dims"][name])},), '
             f'"{dtype_of(spec, name)}")'
        for name in spec.params)
    code = ['import tilelang', 'import tilelang.language as T', '',
            '# 自动转换：顺序归约及后处理；float32输入，保留表达式类型提升。',
            '# 浮点归约允许重排误差；特殊值严格等价尚未保证；源launch完整覆盖输出。',
            # 原地更新（输出同时被读）时必须由调用方传缓冲：kernel 分配只会拿到零初值。
            '@tilelang.jit(out_idx=' + (
                '[]' if any(in_place_output(
                    [p.expression] + [expression for _, _, _, expression in extra_stores]
                    + [stage.expression for stage in p.stages], name)
                    for name in outputs)
                else '[' + ', '.join(str(spec.params.index(name)) for name in outputs) + ']') + ',',
            '              pass_configs={"tl.disable_warp_specialized": True}, compile_flags=["--fmad=false"])',
            f'def {spec.name}(shape, reduction_extents=(), block=128, _tl_extra=()):',
            '    _tl_shape = tuple(shape)', '    _tl_extents = tuple(reduction_extents)',
            '    _tl_extra = tuple(_tl_extra)',
            f'    if len(_tl_extra) != {len(layout.get("dim_extra", ()))}:',
            f'        raise ValueError("shape extras must have {len(layout.get("dim_extra", ()))} entries")']
    for index, (name, dim) in enumerate(layout.get('dim_extra', ())):
        code += [f'    _tl_dim_{name}_{dim} = _tl_extra[{index}]']
    code += [
            f'    if len(_tl_shape) != {rank} or any(type(d) is not int or d <= 0 for d in _tl_shape):',
            '        raise ValueError("invalid output shape")',
            f'    if len(_tl_extents) != {layout["extra"]} or any(type(d) is not int or d <= 0 for d in _tl_extents):',
            '        raise ValueError("invalid independent reduction extents")',
            '    if type(block) is not int or block <= 0 or block % 32:',
            '        raise ValueError("block must be a positive multiple of 32")']
    code += emit_layout_checks(layout)
    groups = group_reductions(spec)
    # 20261025-150（契约统一）：strict 下累加器按**源语义**取 dtype：
    #   acc = numeric_dtype(该组初值, 该组贡献表达式) —— `0`/`0.0` 初值 + f32 贡献 → f64；
    #   f32 读初值 + f32 贡献 → f32。此前按"张量 ABI dtype"固定 float32，与源码语义不符
    #   （`Reduction.dtype` 是硬编码 'float64' 的占位字段，不能当证据）。
    # 规则放在最前，随后三个既有标志仍按各自窄门强制 f64——**不改变它们的最终结果**。
    from semantic import source_accum_dtype
    from abi import accumulator_dtype
    source_dtypes = [source_accum_dtype(stage) for stage in p.stages]
    odd = sorted({dtype for dtype in source_dtypes if dtype not in ('float32', 'float64')})
    if odd:
        # 20261025-151：int64 / float16 等源累加 dtype 不能被静默映射成 float32；
        # 本片不支持，直接拒绝（将来要支持需先给出这些 dtype 的累加等价证明）。
        raise Unsupported('归约累加 dtype 只支持 float32/float64（源语义为 %s）'
                          % ', '.join(odd))
    if len(set(source_dtypes)) > 1:
        # 发射器只有一个累加槽 dtype；各阶段源语义不一致时无法逐阶段发射，拒绝而不是取最宽。
        raise Unsupported('各阶段归约累加 dtype 不一致（%s），当前模板只支持单一累加槽'
                          % ', '.join(source_dtypes))
    acc_dtype = source_dtypes[0] if source_dtypes else accumulator_dtype(spec, 'float32')
    # 20261025-138：仅对"混合 dtype 谓词计数"窄门，累加器跟随归约贡献 dtype，
    # 与源码语义一致；uniform 路径完全不变（spec.mixed_predicate_count 默认 False）。
    if getattr(spec, 'mixed_predicate_count', False):
        contrib_dtype = p.stages[0].reduction.dtype
        if contrib_dtype in ('float32', 'float64'):
            acc_dtype = contrib_dtype
    # 20261025-142：softmax 三段式窄门：分母按源码 f64 累加（val 为 f64、未存储值）。
    # 保留显式 f64 的理由：该形态已由三段式窄门逐项核对（初值/贡献全 f64），
    # 不依赖 `Reduction.dtype` 的硬编码值。
    if getattr(spec, 'softmax_roundtrip', False):
        acc_dtype = 'float64'
    # 20261025-145：常量 guard 窄门里 `temp = 0`（int 初值）+ float32 贡献的 sum，
    # 源码按 Numba 规则是 f64 累加（CUDA typing/PTX 实测）；既有 uniform 路径不变。
    if getattr(spec, 'constant_guard_accum_f64', False):
        acc_dtype = 'float64'
    code += ['    @T.prim_func', f'    def _tl_main({parameters}):',
             '        with T.Kernel(' + ' * '.join(grid_dims) + ', threads=128) as _tl_p:']
    # 槽位跨组复用，阶段结果仍独立保存；不按阶段总数扩大values缓冲。
    for slot in range(max(len(group) for group in groups)):
        code += [f'            _tl_values_{slot} = T.alloc_fragment((1, block), "{acc_dtype}")']
    # 段结果逐个产生并立即消费，不同归约无需同时保留partial。
    code += [f'            _tl_partial = T.alloc_fragment((1,), "{acc_dtype}")']
    for sid, stage in enumerate(p.stages):
        code += [f'            _tl_result_{sid} = T.alloc_fragment((1,), "{acc_dtype}")']
    for group in groups:
        extent = layout['extents'][group[0]]
        for sid in group:
            reduction = p.stages[sid].reduction
            # 常量初值已由 validate_program 校验为等价单位元，直接发射单位元；
            # 动态初值（`m = F[b*win, i]`）按输出坐标读取，再并入归约。
            init = (literal(REDUCTION_IDENTITY[reduction.op]) if reduction.init.op == 'const'
                    else emit_expr(reduction.init, coords))
            code += ['            for _tl_one in T.Parallel(1):',
                     f'                _tl_result_{sid}[_tl_one] = T.cast({init}, "{acc_dtype}")']
        # 每个 lane 先把跨 chunk 的贡献累加到自己的槽位，循环外只做一次 block 归约。
        # 此前在 chunk 循环内做归约：每轮一次 AllReduce + __syncthreads，实测慢两个数量级。
        for slot, sid in enumerate(group):
            identity = literal(REDUCTION_IDENTITY[p.stages[sid].reduction.op])
            code += ['            for _tl_lane in T.Parallel(block):',
                     f'                _tl_values_{slot}[0, _tl_lane] = T.cast({identity}, "{acc_dtype}")']
        code += [f'            for _tl_chunk in T.serial(T.ceildiv({extent}, block)):',
                 '                for _tl_lane in T.Parallel(block):',
                 '                    _tl_k = _tl_chunk * block + _tl_lane']
        indices = {}
        for sid in group:
            index = dict(coords)
            reduction = p.stages[sid].reduction
            start_expr = getattr(reduction, 'start_expr', None)
            # 非零起点（窗口归约）：归约轴坐标 = 起点 + 块内位置，起点逐输出元素变化。
            if start_expr is not None and polynomial(start_expr):
                index[reduction.axis] = f'({emit_expr(start_expr, index)} + _tl_k)'
            else:
                index[reduction.axis] = '_tl_k'
            indices[sid] = index
        definitions, shared = emit_shared_reads(
            [(p.stages[sid].expression, indices[sid]) for sid in group], f'_tl_read_{group[0]}')
        code += [f'                    if _tl_k < {extent}:']
        code += [' ' * 24 + line for line in definitions]
        for slot, sid in enumerate(group):
            index = indices[sid] | shared
            body = emit_expr(p.stages[sid].expression, index)
            operation = p.stages[sid].reduction.op
            accumulate = REDUCTION_COMBINE[operation].format(
                f'_tl_values_{slot}[0, _tl_lane]', f'T.cast({body}, "{acc_dtype}")')
            code += [f'                        _tl_values_{slot}[0, _tl_lane] = {accumulate}']
        # 循环外：每组只归约一次。
        for slot, sid in enumerate(group):
            operation = p.stages[sid].reduction.op
            merge = REDUCTION_COMBINE[operation].format(
                f'_tl_result_{sid}[_tl_one]', '_tl_partial[_tl_one]')
            code += [f'            {REDUCTION_PRIMITIVE[operation]}(_tl_values_{slot}, _tl_partial, dim=1, clear=True)',
                     '            for _tl_one in T.Parallel(1):',
                     f'                _tl_result_{sid}[_tl_one] = {merge}']
    written = dict(coords)
    if free_axes:
        written.update(zip(free_axes, flat_indices(free_dims, '_tl_out')))
    # 写回按**每个 store 自己的完整下标**渲染：常量下标维写固定下标
    # （`positions[tid, 0]`、`positions[tid, 1]`），不能一律用输出轴坐标，
    # 否则同一归约的多个分量写回会落到同一个地址上互相覆盖。
    main_indices = tuple(getattr(p, 'store_indices', ()) or ()) or axes
    main_consts = tuple(getattr(p, 'store_constants', ()) or ()) \
        or (None,) * len(main_indices)
    stores = [(p.output, main_indices, main_consts, p.expression)] + [
        (name, indices, tuple(consts or ()) or (None,) * len(indices), expression)
        for name, indices, consts, expression in extra_stores]
    for name, indices, _consts, _value in stores:
        for axis in indices:
            if axis != AXIS_CONSTANT and axis not in written:
                raise Unsupported('写回下标必须使用输出轴: ' + name)

    def render_store(indices, consts):
        return ', '.join(str(consts[dim]) if axis == AXIS_CONSTANT else written[axis]
                         for dim, axis in enumerate(indices))

    # 写回 cast 的目标 = **输出张量的 ABI dtype**（不是表达式/累加器 dtype）。
    from abi import store_dtype
    targets = [(f'{name}[{render_store(indices, consts)}]',
                emit_expr(expression, written), store_dtype(spec, name))
               for name, indices, consts, expression in stores]
    if free_axes:
        size = ' * '.join(free_dims)
        code += [f'            for _tl_ochunk in T.serial(T.ceildiv({size}, block)):',
                 '                for _tl_olane in T.Parallel(block):',
                 '                    _tl_out = _tl_ochunk * block + _tl_olane',
                 f'                    if _tl_out < {size}:']
        code += [f'                        {target} = T.cast({value}, "{dtype}")'
                 for target, value, dtype in targets]
    else:
        code += ['            for _tl_one in T.Parallel(1):']
        code += [f'                {target} = T.cast({value}, "{dtype}")'
                 for target, value, dtype in targets]
    code += ['    return _tl_main', '']
    return '\n'.join(code)
