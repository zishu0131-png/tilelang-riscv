# -*- coding: utf-8 -*-
"""生成模块的适用条件；调用者必须先完成程序级合法性验证。"""
from semantic import reads
from program import walk


def group_reductions(spec):
    """已验证平面阶段的执行分组；最多双归约以限制同时存活的fragment。"""
    groups = []
    for sid, stage in enumerate(spec.program.stages):
        expressions = [stage.expression] + [value for _, value in stage.bindings]
        dependencies = {node.value for expression in expressions
                        for node in walk(expression) if node.op == 'result'}
        if groups and len(groups[-1]) < 2 \
                and spec.layout['extents'][groups[-1][0]] == spec.layout['extents'][sid] \
                and not dependencies.intersection(groups[-1]):
            groups[-1].append(sid)
        else:
            groups.append([sid])
    return tuple(tuple(group) for group in groups)


def supports_single_reduction(spec):
    """旧接口只表达每输入一个轴投影；不能丢失Program中额外访问和维度约束。"""
    compute = spec.compute
    if compute is None or compute.reduction is None or len(spec.program.stages) != 1:
        return False
    # 标量上界 guard（`if i < n`）只能走 pipeline：单归约模板的并行域取自输出 shape。
    if getattr(spec, 'program_extent_scalar', None):
        return False
    if spec.layout['equalities']:
        return False
    if any(node.op == 'shape' for node in walk(compute.expression)):
        return False
    accesses = reads(compute.expression)
    if not accesses:
        return False
    projections = {}
    for name, indices in accesses:
        if projections.setdefault(name, indices) != indices:
            return False
    for _, value in compute.bindings:
        for name, indices in reads(value):
            if projections.get(name) != indices:
                return False
    # lower_sum的归约长度为独立参数；与输出维相等的约束必须交给layout路径。
    extent = spec.layout['extents'][0]
    if extent.startswith('_tl_shape'):
        return False
    lengths = {axis: f'_tl_shape[{dim}]' for dim, axis in enumerate(compute.output_axes)}
    lengths[compute.reduction.axis] = extent
    return all(tuple(lengths[axis] for axis in indices) == spec.layout['dims'][name]
               for name, indices in projections.items())
