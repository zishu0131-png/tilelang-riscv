# -*- coding: utf-8 -*-
"""独立多累加器模块：证明叶子循环可分离后，规范化为单归约阶段。"""
import ast
import copy
from capture import capture_program, capture_update, fail


def is_update(statement):
    if isinstance(statement, ast.AugAssign):
        return True
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
            and isinstance(statement.targets[0], ast.Name):
        name = statement.targets[0].id
        return any(isinstance(node, ast.Name) and node.id == name
                   for node in ast.walk(statement.value))
    return False


def capture_multi(body, params, axes, initial=None, dtypes=None):
    # 与 capture_program 同一入口约定：签名标量/常量绑定必须透传，否则标量形参不可见。
    return capture_program(normalize_multi(unmask_loop_conditions(body)), params, axes,
                           initial, dtypes)


def unmask_loop_conditions(body):
    """把 `for ...: if cond: <绑定 + 累加>` 展开成 `for ...: <绑定 + 条件化累加>`。

    N-body 的 `if tid != j:` 只影响本轮是否累加：把它改写成
    `acc += expr if cond else 0.0`，多归约/单归约路径都能直接处理。
    局部绑定提到循环层（即使条件为假也会算一次，但没有副作用，结果被丢弃）。
    """
    for statement in body:
        if not isinstance(statement, ast.For) or len(statement.body) != 1:
            continue
        inner = statement.body[0]
        if not (isinstance(inner, ast.If) and not inner.orelse and inner.body):
            continue
        # 只处理"条件包裹的累加"：条件里的张量写回（`if cond: Y[i]=...`）不能展开，
        # 否则条件会被静默丢弃（实测让 20 个范围语料用例变成错误结果）。
        if not any(is_update(item) for item in inner.body):
            continue
        if any(isinstance(item, ast.Assign) and item.targets
               and isinstance(item.targets[0], ast.Subscript) for item in inner.body):
            continue
        new_body = []
        for item in inner.body:
            if is_update(item):
                item.value = ast.fix_missing_locations(ast.IfExp(
                    test=copy.deepcopy(inner.test), body=item.value,
                    orelse=ast.Constant(value=0.0)))
            new_body.append(item)
        statement.body = new_body
    return body


def normalize_multi(body, preserve_parents=False):
    """拆分叶多归约；scoped可保留多更新父循环，交给作用域捕获验证。"""
    normalized = []
    for statement in body:
        if not isinstance(statement, ast.For):
            normalized.append(statement)
            continue
        updates = [item for item in statement.body if is_update(item)]
        has_children = any(isinstance(item, ast.For) for item in statement.body)
        if len(updates) < 2 or (preserve_parents and has_children):
            loop = copy.deepcopy(statement)
            loop.body = normalize_multi(statement.body, preserve_parents=preserve_parents)
            normalized.append(loop)
            continue
        parsed = [capture_update(item) for item in updates]
        accumulators = [name for name, _, _ in parsed]
        if len(set(accumulators)) != len(accumulators):
            fail(statement, '多归约循环中每个累加器只能更新一次')
        active = set(accumulators)
        definitions, seen = [], set()
        for item in statement.body:
            if is_update(item):
                _, _, value = capture_update(item)
            elif isinstance(item, ast.Assign) and len(item.targets) == 1 \
                    and isinstance(item.targets[0], ast.Name):
                name = item.targets[0].id
                if name in active or name in seen:
                    fail(item, '多归约局部定义不能覆盖累加器或重复定义')
                definitions.append(item)
                seen.add(name)
                value = item.value
            else:
                fail(item, '多归约循环只支持纯局部定义及独立累加器更新')
            if any(isinstance(node, ast.Name) and node.id in active for node in ast.walk(value)):
                fail(item, '独立多归约的贡献与局部定义不能读取任何活动累加器')
        # 不将后置定义移动到使用之前：每个贡献只能看到它之前的局部定义。
        # 未使用的后置定义也保留在该阶段中校验，不能静默丢弃非法读取。
        available = []
        for item in statement.body:
            if not is_update(item):
                available.append(item)
                continue
            _, _, value = capture_update(item)
            later = {definition.targets[0].id for definition in definitions[len(available):]}
            if any(isinstance(node, ast.Name) and node.id in later for node in ast.walk(value)):
                fail(item, '多归约贡献不能引用尚未定义的循环局部值')
            loop = copy.deepcopy(statement)
            loop.body = copy.deepcopy(definitions + [item])
            normalized.append(loop)
    return normalized
