# -*- coding: utf-8 -*-
"""从作用域树生成；向量叶支持串行、逐列block或二维分块，默认不变。"""
from lowering import emit_expr, emit_layout_checks, flat_indices, literal, REDUCTION_PRIMITIVE, REDUCTION_COMBINE
from semantic import REDUCTION_IDENTITY, Unsupported


def lower_scoped(spec, output_tile=1, output_interchange=False, leaf_policy='serial'):
    p, layout = spec.program, spec.layout
    runtime_extents = dict(layout.get('runtime_extents', ()))
    if runtime_extents:
        # C1：只支持串行（默认）调度；分块/交换/叶子策略与运行期长度组合明确拒绝，
        # 避免 n=0 预填、前缀/后缀发射等路径对不上。
        if output_tile != 1 or output_interchange or leaf_policy != 'serial':
            raise Unsupported('运行期归约长度暂只支持默认串行调度（不支持输出分块/交换/叶子策略）')
    rank = len(p.output_axes)
    dimensions = [f'_tl_shape[{d}]' for d in range(rank)]
    grid = ' * '.join(dimensions)
    coords = dict(zip(p.output_axes, flat_indices(dimensions)))
    schedule = None
    if output_tile != 1:
        from scoped_schedule import plan_output_tiles
        schedule = plan_output_tiles(p, allow_empty_prefix=output_interchange)
        tiles = f'T.ceildiv({dimensions[-1]}, {output_tile})'
        grid = ' * '.join(dimensions[:-1] + [tiles])
        coords = dict(zip(p.output_axes[:-1],
                          flat_indices(dimensions[:-1], f'(_tl_p // {tiles})')))
    exchanged = None
    vector_results = ()
    if output_interchange:
        from scoped_schedule import plan_interchange
        exchanged = plan_interchange(p, schedule)
        vector_results = exchanged.vector_results
    cooperative_leaves = {}
    tiled_leaves = set()
    if exchanged is not None:
        from program import walk
        from scoped_schedule import plan_leaf_policies, validate_tiled_leaf
        for lid, policy in plan_leaf_policies(p, exchanged, leaf_policy):
            loop = p.loops[lid]
            if policy == 'tiled':
                validate_tiled_leaf(p,lid,schedule.tile_axis)
                tiled_leaves.add(lid)
            if policy in ('block','tiled'):
                used = {node.value for node in walk(p.stages[loop.results[0]].expression)
                        if node.op == 'result'}
                cooperative_leaves[lid] = tuple(sid for sid in vector_results if sid in used)
    bridge_output = max((len(prior) for prior in cooperative_leaves.values()), default=0)
    for name, dims in layout['dims'].items():
        for dim, length in enumerate(dims):
            coords[('shape',name,dim)] = length
    # 参数声明按源码原序：张量用已声明的 dtype，显式标量（签名里是 int64/float32）用现有映射，
    # 不在这里另做类型推断；非张量、非声明标量的参数仍会被现有的非 float32 门禁/校验拒绝。
    from abi import dtype_of, is_scalar, store_dtype
    output_dtype = store_dtype(spec, p.output)
    parameters = ', '.join(
        f'{name}: T.{dtype_of(spec, name)}' if is_scalar(spec, name)
        else f'{name}: T.Tensor(({", ".join(layout["dims"][name])},), "{dtype_of(spec, name)}")'
        for name in spec.params)
    lines = ['import tilelang', 'import tilelang.language as T',
             '# 自动捕获的嵌套归约；外层串行，叶层并行。不保证特殊值严格等价。',
             f'@tilelang.jit(out_idx=[{spec.params.index(p.output)}], pass_configs={{"tl.disable_warp_specialized": True}}, compile_flags=["--fmad=false"])',
             f'def {spec.name}(shape, reduction_extents=(), block=128'
             + (', _tl_extra=()' if layout.get('dim_extra') else '') + '):',
             '    _tl_shape = tuple(shape)', '    _tl_extents = tuple(reduction_extents)']
    # 只有存在常量下标维时才加 `_tl_extra` 通道，既有产物逐字节不变。
    if layout.get('dim_extra'):
        lines += ['    _tl_extra = tuple(_tl_extra)',
                  f'    if len(_tl_extra) != {len(layout["dim_extra"])}:',
                  '        raise ValueError("shape extras must have %d entries")'
                  % len(layout['dim_extra'])]
    lines += [
             f'    if len(_tl_shape) != {rank} or any(type(d) is not int or d <= 0 for d in _tl_shape):',
             '        raise ValueError("invalid output shape")',
             f'    if len(_tl_extents) != {layout["extra"]} or any(type(d) is not int or d <= 0 for d in _tl_extents):',
             '        raise ValueError("invalid reduction extents")',
             '    if type(block) is not int or block <= 0 or block % 32:',
             '        raise ValueError("invalid block")']
    if layout.get('dim_extra'):
        lines += ['    _tl_dim_%s_%d = _tl_extra[%d]' % (name, dim, index)
                  for index, (name, dim) in enumerate(layout['dim_extra'])]
    lines += emit_layout_checks(layout)
    lines += ['    @T.prim_func',f'    def _tl_main({parameters}):',
              '        with T.Kernel('+grid+', threads=128) as _tl_p:']
    for sid, (scalar, name, dim) in sorted(runtime_extents.items()):
        bound = layout['dims'][name][dim]
        lines += [f'            T.Assert({scalar} >= 0, "{scalar} must be >= 0")',
                  f'            T.Assert({scalar} <= {bound}, "{scalar} must be <= physical extent")']
    for sid in range(len(p.stages)):
        width = output_tile if sid in vector_results else 1
        lines.append(f'            _tl_result_{sid} = T.alloc_fragment(({width},), "float64")')
    # 叶归约顺序执行并立即消费partial；跨阶段存活的只有各自result。
    if any(not loop.children for loop in p.loops):
        lines += ['            _tl_values = T.alloc_fragment((1, block), "float64")',
                  '            _tl_partial = T.alloc_fragment((1,), "float64")']

    if cooperative_leaves:
        lines += [f'            _tl_leaf_bridge = T.alloc_shared(({bridge_output + 1}, {output_tile}), "float64")',
                  '            _tl_leaf_acc = T.alloc_fragment((1,), "float64")']
    if tiled_leaves:
        lines += [f'            _tl_tiled_values = T.alloc_fragment((block, {output_tile}), "float64")',
                  f'            _tl_tiled_partial = T.alloc_fragment(({output_tile},), "float64")',
                  f'            _tl_tiled_acc = T.alloc_fragment(({output_tile},), "float64")']

    def emit_chunks(stage, sid, index, target, put):
        extent = layout['extents'][sid]
        identity = f'T.cast({literal(REDUCTION_IDENTITY[stage.reduction.op])}, "float64")'
        body = emit_expr(stage.expression,index)
        if sid in runtime_extents:
            # n=0 时 chunk 循环一次都不执行：先把槽位填成单位元，避免读未初始化 fragment。
            put(f'for _tl_lane_pre_{sid} in T.Parallel(block):')
            put(f'_tl_values[0, _tl_lane_pre_{sid}] = {identity}', 4)
        put(f'for _tl_chunk_{sid} in T.serial(T.ceildiv({extent}, block)):')
        put(f'for _tl_lane_{sid} in T.Parallel(block):',4)
        put(f'_tl_k_{sid} = _tl_chunk_{sid} * block + _tl_lane_{sid}',8)
        put(f'if _tl_k_{sid} < {extent}:',8)
        put(f'_tl_values[0, _tl_lane_{sid}] = T.cast({body}, "float64")',12)
        put('else:',8)
        put(f'_tl_values[0, _tl_lane_{sid}] = {identity}',12)
        put(f'{REDUCTION_PRIMITIVE[stage.reduction.op]}(_tl_values, _tl_partial, dim=1, clear=True)',4)
        put('for _tl_one in T.Parallel(1):',4)
        merged = REDUCTION_COMBINE[stage.reduction.op].format(target,'_tl_partial[_tl_one]')
        put(f'{target} = {merged}',8)

    def emit_tiled(lid, sid, identity):
        stage = p.stages[sid]
        extent = layout['extents'][sid]
        index = dict(coords)
        index[p.loops[lid].axis] = f'_tl_k_{sid}'
        index[schedule.tile_axis] = '_tl_output_index'
        for slot, prior in enumerate(cooperative_leaves[lid]):
            index[('result',prior)] = f'_tl_leaf_bridge[{slot}, _tl_output_lane]'
        body = emit_expr(stage.expression,index)
        merged = REDUCTION_COMBINE[stage.reduction.op].format(
            '_tl_tiled_acc[_tl_output_lane]','_tl_tiled_partial[_tl_output_lane]')
        return ['            T.sync_threads()',
                f'            for _tl_output_lane in T.Parallel({output_tile}):',
                f'                _tl_tiled_acc[_tl_output_lane] = T.cast({identity}, "float64")',
                f'            for _tl_chunk_{sid} in T.serial(T.ceildiv({extent}, block)):',
                f'                for _tl_lane_{sid}, _tl_output_lane in T.Parallel(block, {output_tile}):',
                f'                    _tl_k_{sid} = _tl_chunk_{sid} * block + _tl_lane_{sid}',
                f'                    _tl_output_index = (_tl_p % {tiles}) * {output_tile} + _tl_output_lane',
                f'                    if _tl_k_{sid} < {extent} and _tl_output_index < {dimensions[-1]}:',
                f'                        _tl_tiled_values[_tl_lane_{sid}, _tl_output_lane] = T.cast({body}, "float64")',
                '                    else:',
                f'                        _tl_tiled_values[_tl_lane_{sid}, _tl_output_lane] = T.cast({identity}, "float64")',
                f'                {REDUCTION_PRIMITIVE[stage.reduction.op]}(_tl_tiled_values, _tl_tiled_partial, dim=0, clear=True)',
                f'                for _tl_output_lane in T.Parallel({output_tile}):',
                f'                    _tl_tiled_acc[_tl_output_lane] = {merged}',
                f'            for _tl_output_lane in T.Parallel({output_tile}):',
                f'                _tl_leaf_bridge[{bridge_output}, _tl_output_lane] = _tl_tiled_acc[_tl_output_lane]']

    def emit(lid, axes, indent):
        loop = p.loops[lid]
        sid = loop.results[0]
        stage = p.stages[sid]
        r = stage.reduction
        extent = layout['extents'][sid]
        def put(text, depth=0):
            lines.append(' '*(indent+depth)+text)
        for result in loop.results:
            operation = p.stages[result].reduction.op
            initial = f'T.cast({literal(REDUCTION_IDENTITY[operation])}, "float64")'
            put('for _tl_one in T.Parallel(1):')
            put(f'_tl_result_{result}[_tl_one] = {initial}',4)
        index = dict(axes)
        index[r.axis] = f'_tl_k_{sid}'
        if loop.children:
            put(f'for _tl_k_{sid} in T.serial({extent}):')
            for child in loop.children:
                emit(child,index,indent+4)
            for result in loop.results:
                member = p.stages[result]
                body = emit_expr(member.expression,index)
                merged = REDUCTION_COMBINE[member.reduction.op].format(
                    f'_tl_result_{result}[_tl_one]',f'T.cast({body}, "float64")')
                put('for _tl_one in T.Parallel(1):',4)
                put(f'_tl_result_{result}[_tl_one] = {merged}',8)
        else:
            emit_chunks(stage,sid,index,f'_tl_result_{sid}[_tl_one]',put)
    if exchanged is not None:
        for lid in schedule.prefix_roots:
            emit(lid,coords,12)
        completed = {}
        for lid in exchanged.roots:
            if lid not in exchanged.vector_roots:
                emit(lid,coords,12)
                continue
            loop = p.loops[lid]
            sid = loop.results[0]
            for result in loop.results:
                identity = literal(REDUCTION_IDENTITY[p.stages[result].reduction.op])
                lines += [f'            for _tl_output_lane in T.Parallel({output_tile}):',
                          f'                _tl_result_{result}[_tl_output_lane] = T.cast({identity}, "float64")']
            if lid in cooperative_leaves:
                # fragment按列导出到shared，避免假定所有归约线程持有任意prior列。
                for slot, prior in enumerate(cooperative_leaves[lid]):
                    lines += [f'            for _tl_output_lane in T.Parallel({output_tile}):',
                              f'                _tl_leaf_bridge[{slot}, _tl_output_lane] = _tl_result_{prior}[_tl_output_lane]']
                if lid in tiled_leaves:
                    lines += emit_tiled(lid,sid,identity)
                else:
                    lines += ['            T.sync_threads()',
                              f'            for _tl_leaf_column in T.serial({output_tile}):',
                              f'                _tl_output_index = (_tl_p % {tiles}) * {output_tile} + _tl_leaf_column',
                              f'                if _tl_output_index < {dimensions[-1]}:',
                              '                    for _tl_one in T.Parallel(1):',
                              f'                        _tl_leaf_acc[_tl_one] = T.cast({identity}, "float64")']
                    index = dict(coords)
                    index[loop.axis] = f'_tl_k_{sid}'
                    index[schedule.tile_axis] = '_tl_output_index'
                    for slot, prior in enumerate(cooperative_leaves[lid]):
                        index[('result',prior)] = f'_tl_leaf_bridge[{slot}, _tl_leaf_column]'
                    def put_leaf(text, depth=0):
                        lines.append(' '*(20+depth)+text)
                    emit_chunks(p.stages[sid],sid,index,'_tl_leaf_acc[_tl_one]',put_leaf)
                    lines += ['                    for _tl_one in T.Parallel(1):',
                              f'                        _tl_leaf_bridge[{bridge_output}, _tl_leaf_column] = _tl_leaf_acc[_tl_one]']
                lines += ['            T.sync_threads()',
                          f'            for _tl_output_lane in T.Parallel({output_tile}):',
                          f'                _tl_output_index = (_tl_p % {tiles}) * {output_tile} + _tl_output_lane',
                          f'                if _tl_output_index < {dimensions[-1]}:',
                          f'                    _tl_result_{sid}[_tl_output_lane] = _tl_leaf_bridge[{bridge_output}, _tl_output_lane]',
                          '            T.sync_threads()']
                completed[('result',sid)] = f'_tl_result_{sid}[_tl_output_lane]'
                continue
            lines += [f'            for _tl_k_{sid} in T.serial({layout["extents"][sid]}):']
            index = dict(coords)
            index[loop.axis] = f'_tl_k_{sid}'
            for child in loop.children:
                emit(child,index,16)
            # 块级子树不接收向量槽位映射；只有逐列贡献可消费已完成的后缀结果。
            lines += [f'                for _tl_output_lane in T.Parallel({output_tile}):',
                      f'                    _tl_output_index = (_tl_p % {tiles}) * {output_tile} + _tl_output_lane',
                      f'                    if _tl_output_index < {dimensions[-1]}:']
            index[schedule.tile_axis] = '_tl_output_index'
            index.update(completed)
            for result in loop.results:
                member = p.stages[result]
                target = f'_tl_result_{result}[_tl_output_lane]'
                body = emit_expr(member.expression,index)
                merged = REDUCTION_COMBINE[member.reduction.op].format(target,f'T.cast({body}, "float64")')
                lines.append(f'                        {target} = {merged}')
            for result in loop.results:
                completed[('result',result)] = f'_tl_result_{result}[_tl_output_lane]'
        written = dict(coords)
        written[schedule.tile_axis] = '_tl_output_index'
        written.update(completed)
        lines += [f'            for _tl_output_lane in T.Parallel({output_tile}):',
                  f'                _tl_output_index = (_tl_p % {tiles}) * {output_tile} + _tl_output_lane',
                  f'                if _tl_output_index < {dimensions[-1]}:',
                  f'                    {p.output}[{", ".join(written[a] for a in p.output_axes)}] = T.cast({emit_expr(p.expression,written)}, "{output_dtype}")',
                  '    return _tl_main','']
        return '\n'.join(lines)

    indent = 12
    if schedule is None:
        for lid in p.roots:
            emit(lid,coords,indent)
    else:
        for lid in schedule.prefix_roots:
            emit(lid,coords,indent)
        lines += [f'            for _tl_output_offset in T.serial({output_tile}):',
                  f'                _tl_output_index = (_tl_p % {tiles}) * {output_tile} + _tl_output_offset',
                  f'                if _tl_output_index < {dimensions[-1]}:']
        coords[schedule.tile_axis] = '_tl_output_index'
        indent = 20
        # 尾块条件在整个block上一致；后缀每个输出重置result，前缀result保持存活。
        for lid in schedule.suffix_roots:
            emit(lid,coords,indent)
    lines += [' '*indent + 'for _tl_one in T.Parallel(1):',
              ' '*(indent+4) + f'{p.output}[{", ".join(coords[a] for a in p.output_axes)}] = T.cast({emit_expr(p.expression,coords)}, "{output_dtype}")',
              '    return _tl_main','']
    return '\n'.join(lines)
