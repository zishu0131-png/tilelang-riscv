# -*- coding: utf-8 -*-
"""递归归约捕获：唯一轴身份、局部结果可见性和每轮初始化。"""
import ast
import copy
from dataclasses import dataclass
from capture import capture_update, domain_text, fail
from semantic import Expr, Reduction, capture_expr, tensor_access


@dataclass(frozen=True)
class ScopedStage:
    reduction: Reduction
    expression: Expr
    line: int
    loop: int


@dataclass(frozen=True)
class ScopedLoop:
    axis: str
    start: str
    stop: str
    step: str
    context: tuple
    bindings: tuple
    events: tuple
    results: tuple

    @property
    def children(self):
        return tuple(value for kind, value in self.events if kind == 'loop')


@dataclass(frozen=True)
class ScopedProgram:
    stages: tuple
    output: str
    output_axes: tuple
    expression: Expr
    bindings: tuple
    loops: tuple
    events: tuple

    @property
    def roots(self):
        return tuple(value for kind, value in self.events if kind == 'loop')


def capture_scoped(body, params, axes, initial=None, dtypes=None):
    # 与 `capture_program` 同构：签名标量/模块常量进根 env，张量 dtype 表透传到所有值捕获。
    # 两者都由调用方（`parse_source`）从现有类型表传入，这里不做任何类型推断。
    stages, loops = [], []
    counter = 0
    names_in_source = {node.id for statement in body for node in ast.walk(statement) if isinstance(node, ast.Name)} | set(axes) | set(params)

    def expression(node, env, coordinates):
        class Rename(ast.NodeTransformer):
            def visit_Name(self, item):
                return ast.copy_location(ast.Name(id=coordinates.get(item.id, item.id), ctx=item.ctx), item)
        rewritten = Rename().visit(copy.deepcopy(node))
        # 嵌套归约的读下标只能是**当前活动坐标本身**：`A[i, j, k+1]` 在 k 的最后一轮越界，
        # 而 scoped 模板没有窗口/越界语义。以前这类写法靠"下标里的名字解不出来"被拒，
        # 现在下标能解析了，必须显式保留这条边界。
        active = set(coordinates.values())
        for item in ast.walk(rewritten):
            if not (isinstance(item, ast.Subscript) and isinstance(item.value, ast.Name)):
                continue
            indices = item.slice.elts if isinstance(item.slice, ast.Tuple) else [item.slice]
            for index in indices:
                if isinstance(index, ast.Name) and index.id in active:
                    continue
                # B3：允许**常量分量维**（`A[i, 0, k]`、模块常量 `C0` 等），长度交给
                # `layout['dim_extra']` 的 `_tl_extra` 通道；偏移/跨 lane/间接基址仍拒绝。
                if isinstance(index, ast.Constant) and type(index.value) is int                         and index.value >= 0:
                    continue
                bound = env.get(index.id) if isinstance(index, ast.Name) else None
                if getattr(bound, 'op', '') == 'const' and type(bound.value) is int                         and bound.value >= 0:
                    continue
                fail(item, '嵌套归约的读下标只能是当前活动坐标或非负常量分量（不支持偏移/间接）')
        # 下标里的轴必须作为 `index` 捕获：`-56` 起 `tensor_access` 会把下标交给
        # `capture_expr(..., axes)` 求值，而这里原先没传 axes，导致 `A[i, j, k]` 的
        # 每个 grid/循环轴都变成"未知名字"。传**活动坐标的规范化身份**（Rename 之后
        # 只剩 coordinates.values()）；不传全函数轴名全集，也不放开父轴依赖。
        return capture_expr(rewritten, env, params, dtypes, set(coordinates.values()))

    def scope(statements, inherited, coordinates, blocked, top=False):
        nonlocal counter
        env = dict(inherited)
        local_names, escaped, referenced = set(), set(), set()
        bindings, events = [], []
        store = None
        for statement in statements:
            if store is not None:
                fail(statement, '写回之后不能有语句')
            if isinstance(statement, ast.For):
                if not isinstance(statement.target, ast.Name) or statement.orelse:
                    fail(statement, '需要无else的单轴循环')
                axis_name = statement.target.id
                if axis_name in set(env) | set(coordinates) | set(params) | blocked:
                    fail(statement, '循环轴覆盖活动变量')
                call = statement.iter
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) \
                        or call.func.id != 'range' or call.keywords or not 1 <= len(call.args) <= 3:
                    fail(statement, '归约域需要range')
                args = [domain_text(a, env, params, dtypes=dtypes) for a in call.args]
                start, stop, step = ('0', args[0], '1') if len(args) == 1 else (args[0],args[1],args[2] if len(args)==3 else '1')
                updates = [n for n, item in enumerate(statement.body) if is_update(item)]
                if not updates:
                    fail(statement, '循环末尾需要归约更新')
                tail = statement.body[updates[0]:]
                # 更新之间可插入纯局部定义；更新之后仍不允许再出现子循环或写张量。
                for item in tail:
                    if is_update(item):
                        continue
                    if isinstance(item, ast.Assign) and len(item.targets) == 1 \
                            and isinstance(item.targets[0], ast.Name):
                        continue
                    fail(item, '父归约更新之后只允许独立更新和纯局部定义')
                parsed = [capture_update(item) for item in tail if is_update(item)]
                active = {acc for acc, _, _ in parsed}
                if len(active) != len(parsed):
                    fail(statement, '每个父累加器只能更新一次')
                for acc in active:
                    if acc not in local_names or acc in referenced or env[acc].op != 'const':
                        fail(statement, '累加器必须在当前作用域独立常量初始化，不能通过别名或外层继承')
                axis = f'_scope_axis_{counter}'
                counter += 1
                while axis in names_in_source:
                    axis = f'_scope_axis_{counter}'
                    counter += 1
                child_coordinates = dict(coordinates, **{axis_name: axis})
                child_env = {name: value for name, value in env.items() if name not in active}
                result_env, child_bindings, child_events, _, names = scope(
                    statement.body[:updates[0]], child_env, child_coordinates, blocked | active)
                child_bindings, child_events = list(child_bindings), list(child_events)
                # 按源码顺序扫描更新后缀；活动结果在全部贡献捕获完成前都不发布，
                # 因此插入的局部定义与任何贡献都不能读到本循环自己的活动累加器。
                contributions = []
                for item in tail:
                    if is_update(item):
                        acc, operation, value = capture_update(item)
                        contributions.append((acc, operation,
                                              expression(value, result_env, child_coordinates)))
                        continue
                    name = item.targets[0].id
                    if name in set(result_env) | set(child_coordinates) | set(params) \
                            | blocked | active | escaped | names:
                        fail(item, '局部定义覆盖或跨循环变量逃逸')
                    referenced.update(n.id for n in ast.walk(item.value) if isinstance(n, ast.Name))
                    child_events.append(('binding', len(child_bindings)))
                    result_env[name] = expression(item.value, result_env, child_coordinates)
                    child_bindings.append((name, result_env[name]))
                    names.add(name)
                lid, results = len(loops), []
                for acc, operation, body_expr in contributions:
                    reduction = Reduction(axis,start,stop,step,acc,env[acc],'float64',operation)
                    sid = len(stages)
                    stages.append(ScopedStage(reduction,body_expr,statement.lineno,lid))
                    results.append(sid)
                loops.append(ScopedLoop(axis,start,stop,step,tuple(child_coordinates.values()),
                                        tuple(child_bindings),tuple(child_events),tuple(results)))
                for (acc, _, _), sid in zip(contributions, results):
                    env[acc] = Expr('result',value=sid,dtype='float64')
                events.append(('loop',lid))
                escaped.update(names | {axis_name})
                continue
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                fail(statement, '作用域只允许纯定义、归约和顶层唯一写回')
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                if name in set(env) | set(coordinates) | set(params) | blocked | escaped:
                    fail(statement, '局部定义覆盖或跨循环变量逃逸')
                env[name] = expression(statement.value,env,coordinates)
                referenced.update(n.id for n in ast.walk(statement.value) if isinstance(n,ast.Name))
                events.append(('binding',len(bindings)))
                bindings.append((name,env[name]))
                local_names.add(name)
            elif top:
                output, indices = tensor_access(target)
                store = (output,indices,expression(statement.value,env,coordinates))
            else:
                fail(statement, '嵌套作用域不能写张量')
        return env,bindings,events,store,local_names | escaped

    from multi_capture import normalize_multi, is_update
    body = normalize_multi(body, preserve_parents=True)
    env, bindings, events, store, _ = scope(body, dict(initial or {}), {a:a for a in axes},set(),True)
    if store is None:
        fail(body[-1], '缺少唯一输出')
    return ScopedProgram(tuple(stages),*store,tuple(bindings),tuple(loops),tuple(events))
