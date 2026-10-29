# -*- coding: utf-8 -*-
"""按定义/循环事件验证结果完成时点；编号顺序不能代替作用域。"""
from semantic import Unsupported
from program import walk


def validate_scopes(program, axes):
    seen_loops, seen_results, seen_axes = set(), set(), set()

    def check(expression, available):
        for node in walk(expression):
            if node.op == 'result' and (type(node.value) is not int or node.value not in available):
                raise Unsupported('归约结果在当前定义位置尚未完成或不可见')

    def visit(events, bindings, context, inherited):
        available = set(inherited)
        seen_bindings = set()
        for kind, value in events:
            if type(value) is not int:
                raise Unsupported('作用域事件需要整数身份')
            if kind == 'binding':
                if not 0 <= value < len(bindings) or value in seen_bindings:
                    raise Unsupported('局部定义事件缺失或重复')
                seen_bindings.add(value)
                check(bindings[value][1], available)
            elif kind == 'loop':
                if not 0 <= value < len(program.loops) or value in seen_loops:
                    raise Unsupported('循环必须有唯一执行位置，不能重复或成环')
                seen_loops.add(value)
                loop = program.loops[value]
                if loop.axis in seen_axes or loop.axis in context \
                        or loop.context != context + (loop.axis,):
                    raise Unsupported('循环坐标必须唯一且符合祖先作用域')
                seen_axes.add(loop.axis)
                if not loop.results or (len(loop.results) > 1 and not loop.children):
                    raise Unsupported('多结果循环必须具有共享子循环')
                accumulators = set()
                for sid in loop.results:
                    if type(sid) is not int or not 0 <= sid < len(program.stages) \
                            or sid in seen_results:
                        raise Unsupported('归约结果必须有唯一所属循环')
                    stage = program.stages[sid]
                    r = stage.reduction
                    if r.accumulator in accumulators:
                        raise Unsupported('同循环归约结果不能重复使用累加器')
                    accumulators.add(r.accumulator)
                    if stage.loop != value or (r.axis, r.start, r.stop, r.step) != \
                            (loop.axis, loop.start, loop.stop, loop.step):
                        raise Unsupported('归约结果与所属循环的域不一致')
                    seen_results.add(sid)
                completed = visit(loop.events, loop.bindings, loop.context, available)
                for sid in loop.results:
                    check(program.stages[sid].expression, completed)
                # 子局部结果留在completed中，不随父循环退出向外发布。
                available.update(loop.results)
            else:
                raise Unsupported('未知作用域事件')
        if seen_bindings != set(range(len(bindings))):
            raise Unsupported('局部定义必须保留其定义位置')
        return available

    completed = visit(program.events, program.bindings, tuple(axes), set())
    check(program.expression, completed)
    if seen_loops != set(range(len(program.loops))) \
            or seen_results != set(range(len(program.stages))):
        raise Unsupported('存在未归属执行结构的循环或结果')
