# -*- coding: utf-8 -*-
"""已验证嵌套程序的完整子树依赖与保守前缀调度。"""
from dataclasses import dataclass
from program import walk
from semantic import Unsupported


@dataclass(frozen=True)
class ScopedSchedule:
    tile_axis: str
    prefix_roots: tuple
    suffix_roots: tuple
    root_dependencies: tuple
    loop_dependencies: tuple


def plan_output_tiles(program, *, allow_empty_prefix=False):
    """调用者须先验证Program；不重排根或拆分loop，空前缀仅供另验交换条件。"""
    axes = program.output_axes
    if len(axes) < 2:
        raise Unsupported('输出分块复用需要至少两个输出轴')
    results = {}
    loop_dependencies = [None] * len(program.loops)

    def dependencies(expression):
        used = set()
        for node in walk(expression):
            if node.op == 'read':
                used.update(set(node.value[1]) & set(axes))
            elif node.op == 'result':
                used.update(results[node.value])
        return used

    def visit(lid):
        loop = program.loops[lid]
        used = set()
        for kind, value in loop.events:
            if kind == 'loop':
                used.update(visit(value))
            else:
                used.update(dependencies(loop.bindings[value][1]))
        for sid in loop.results:
            used.update(dependencies(program.stages[sid].expression))
        summary = frozenset(used)
        loop_dependencies[lid] = tuple(axis for axis in axes if axis in used)
        for sid in loop.results:
            results[sid] = summary
        return summary

    roots = program.roots
    summaries = tuple(visit(lid) for lid in roots)
    cut = 0
    for used in summaries:
        if axes[-1] in used:
            break
        cut += 1
    if cut == len(roots) or (cut == 0 and not allow_empty_prefix):
        if allow_empty_prefix:
            raise Unsupported('输出循环交换需要非空逐输出后缀')
        raise Unsupported('输出分块复用需要无关根前缀及非空逐输出后缀')
    return ScopedSchedule(axes[-1], roots[:cut], roots[cut:],
                          tuple(tuple(axis for axis in axes if axis in used)
                                for used in summaries), tuple(loop_dependencies))


def interchange_root(program, schedule):
    """单后缀根兼容入口；若有子树，须满足列无关与块级存储约束。"""
    if len(schedule.suffix_roots) != 1:
        raise Unsupported('输出循环交换需要唯一后缀根循环')
    return interchange_roots(program, schedule)[0]


def interchange_roots(program, schedule):
    return plan_interchange(program, schedule).roots


@dataclass(frozen=True)
class InterchangeSchedule:
    roots: tuple
    vector_roots: tuple
    vector_results: tuple


def plan_leaf_policies(program, schedule, policy):
    """已验证交换计划中的向量叶根按源码顺序逐一绑定策略，不包含标量根。"""
    leaves = tuple(lid for lid in schedule.vector_roots if not program.loops[lid].children)
    policies = policy if type(policy) is tuple else (policy,) * len(leaves)
    if len(policies) != len(leaves):
        raise ValueError(f'expected {len(leaves)} leaf policies for vector leaf loops {leaves}, got {len(policies)}')
    return tuple(zip(leaves, policies))


def validate_tiled_leaf(program, lid, tile_axis):
    """首版二维叶要求实际二维读取，且列坐标沿连续张量的最后物理维。"""
    loop = program.loops[lid]
    joint = False
    for node in walk(program.stages[loop.results[0]].expression):
        if node.op != 'read':
            continue
        name, indices = node.value
        if tile_axis in indices:
            if indices[-1] != tile_axis:
                raise Unsupported(f'tiled leaf loop {lid}: {name}{indices} must place {tile_axis} last')
            joint |= loop.axis in indices
    if not joint:
        raise Unsupported(f'tiled leaf loop {lid}: requires a read containing {loop.axis} and {tile_axis}')


def plan_interchange(program, schedule):
    """保持根顺序；完整依赖决定存储宽度，块级读取须与标量存储一致。"""
    roots = schedule.suffix_roots
    if not roots:
        raise Unsupported('输出循环交换需要非空逐输出后缀')
    vector_roots = tuple(lid for lid in roots
                         if schedule.tile_axis in schedule.loop_dependencies[lid])
    vector_results = tuple(sid for lid in vector_roots for sid in program.loops[lid].results)

    def check_storage(lid):
        loop = program.loops[lid]
        expressions = [expression for _, expression in loop.bindings]
        expressions.extend(program.stages[sid].expression for sid in loop.results)
        if any(node.op == 'result' and node.value in vector_results
               for expression in expressions for node in walk(expression)):
            raise Unsupported('块级子归约不能读取逐列存储的后缀结果')
        for child in loop.children:
            check_storage(child)

    for lid in roots:
        loop = program.loops[lid]
        if lid not in vector_roots:
            check_storage(lid)
            continue
        for child in loop.children:
            if schedule.tile_axis in schedule.loop_dependencies[child]:
                raise Unsupported('输出循环交换的子归约不能依赖输出分块轴')
            check_storage(child)
    return InterchangeSchedule(roots, vector_roots, vector_results)
