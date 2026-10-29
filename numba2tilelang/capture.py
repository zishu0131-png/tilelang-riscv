# -*- coding: utf-8 -*-
"""顺序定义—使用捕获：不依赖算子名称，不执行用户源码。"""
import ast
from dataclasses import dataclass
from semantic import (AXIS_CONSTANT, Expr, INT_DTYPES, Reduction, Unsupported, capture_expr,
                      plain_text, tensor_access, reads)


@dataclass(frozen=True)
class Stage:
    reduction: Reduction
    expression: Expr
    bindings: tuple
    line: int


@dataclass(frozen=True)
class Program:
    stages: tuple
    output: str
    output_axes: tuple
    expression: Expr
    bindings: tuple
    # 附加写回：(张量, 下标, 常量维, 表达式)。层归一化在一个归约循环后同时写 Mean 和 Var，
    # LBM/N-body 的同一归约里按常量下标写多个分量（`positions[tid, 0..2]`）。
    extra_stores: tuple = ()
    # 主输出的**完整**写回下标（可能含常量维）与常量取值；output_axes 只是其中的并行轴。
    # 两者分开：常量维不代表并行域，但它是张量的物理维，写回时必须按固定下标渲染。
    store_indices: tuple = ()
    store_constants: tuple = ()


def fail(node, message):
    raise Unsupported(f'第{getattr(node, "lineno", "?")}行：{message}')


def domain_value(node, env, params, axes, dtypes=None):
    """循环域表达式：整数常量、shape 引用、索引变量或标量形参的加减乘组合。

    窗口归约的 `range(b*win, (b+1)*win)` 依赖这个：起点是输出轴的仿射函数，
    长度由 stop-start 的符号化简给出（见 semantic.validate_sum）。
    """
    value = capture_expr(node, env, params, dtypes, axes=set(axes))
    if value.op in ('shape', 'index', 'scalar', 'add', 'sub', 'mul'):
        if value.dtype not in INT_DTYPES:
            fail(node, '循环域必须是整型表达式')
        return value
    if value.op == 'const' and type(value.value) is int:
        return value
    fail(node, '循环域只支持整数常量、shape引用、索引变量或标量形参的加减乘')


def domain_text(node, env, params, axes=(), dtypes=None):
    """兼容包装：把循环域渲染成源码风格字符串（scoped 捕获仍走这里）。"""
    return plain_text(domain_value(node, env, params, axes, dtypes))


def capture_update(update):
    """平面与嵌套捕获共用唯一更新规范；保持操作数顺序。"""
    if isinstance(update, ast.AugAssign) and isinstance(update.target, ast.Name) \
            and isinstance(update.op, ast.Add):
        return update.target.id, 'sum', update.value
    if isinstance(update, ast.Assign) and len(update.targets) == 1 \
            and isinstance(update.targets[0], ast.Name):
        acc, rhs = update.targets[0].id, update.value
        if isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.Add) \
                and isinstance(rhs.left, ast.Name) and rhs.left.id == acc:
            return acc, 'sum', rhs.right
        if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) \
                and rhs.func.id in ('max', 'min') and len(rhs.args) == 2 and not rhs.keywords:
            if not isinstance(rhs.args[0], ast.Name) or rhs.args[0].id != acc:
                fail(update, 'max/min需以旧累加器为第一个参数')
            return acc, rhs.func.id, rhs.args[1]
    if isinstance(update, ast.If) and not update.orelse and len(update.body) == 1:
        # 掩码计数：`if cond: acc += v` 等价于 `acc += v if cond else 0`（直方图、专家计数）。
        inner = update.body[0]
        if isinstance(inner, ast.AugAssign) and isinstance(inner.op, ast.Add) \
                and isinstance(inner.target, ast.Name):
            guarded = ast.IfExp(test=update.test, body=inner.value,
                                orelse=ast.Constant(value=0))
            return inner.target.id, 'sum', ast.copy_location(guarded, update)
        # `if A > acc: acc = A` 是 max 归约的条件写法（max pooling）；
        # 只接受"写回值就是被比较的另一侧"，其余 if 形态没有可证明的等价归约。
        assignment, test = update.body[0], update.test
        # 条件写法只认 max/min：`if A > acc: acc = A`。掩码式递增（`if c: acc += 1`）
        # 与 top-k 这类带位置的选择都不是可证明的定长归约，统一在这里拒绝。
        message = '循环末尾需要唯一的独立归约更新（条件写法只支持 `if A > acc: acc = A`）'
        if not (isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
                and isinstance(assignment.targets[0], ast.Name)):
            fail(update, message)
        acc = assignment.targets[0].id
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], (ast.Gt, ast.Lt))):
            fail(update, message)
        left, right = test.left, test.comparators[0]
        if isinstance(left, ast.Name) and left.id == acc:
            other, keep_max = right, isinstance(test.ops[0], ast.Lt)
        elif isinstance(right, ast.Name) and right.id == acc:
            other, keep_max = left, isinstance(test.ops[0], ast.Gt)
        else:
            fail(update, '条件归约更新的比较必须涉及累加器')
        if ast.dump(other) != ast.dump(assignment.value):
            fail(update, '条件归约更新必须写回被比较的那个值')
        return acc, ('max' if keep_max else 'min'), other
    fail(update, '循环末尾需要唯一的独立归约更新')


def _is_update(statement):
    """该语句是否是归约更新（`acc += v`、`acc = max(acc, v)`、`if c: acc += v`）。"""
    try:
        capture_update(statement)
        return True
    except Unsupported:
        return False


def store_access(target):
    """写回下标 → (张量, 轴/常量占位, 常量维取值)。

    常量下标维（`positions[tid, 0]`）的固定下标必须保留：多个分量写回如果只按轴渲染
    会落到同一个地址上互相覆盖。轴偏移（`X[i+1]`）在 pipeline 写回里不支持。
    """
    name, indices, offsets, scales = tensor_access(target, with_offsets=True)
    if any(scale != 1 for scale in scales):
        raise Unsupported('写回下标不能带系数')
    consts = []
    for axis, offset in zip(indices, offsets):
        if axis == AXIS_CONSTANT:
            consts.append(offset)
        elif offset != 0:
            raise Unsupported('该位置不支持带偏移的写回下标（仅逐元素/展平映射支持）')
        else:
            consts.append(None)
    return name, indices, tuple(consts)


def capture_program(body, params, axes, initial=None, dtypes=None):
    # initial 是签名标量（及 math 常量/模块常量）的绑定；它们必须进入求值环境，
    # 否则 `alpha * tmp` 里的 alpha 会被当成未知名字。与 capture_pointwise 同构。
    env, stages, bindings = dict(initial or {}), [], []
    scalar_names = set(initial or {})
    loop_locals = set()
    referenced_before_loop = set()
    protected = set(params) | set(axes)
    stores = []

    def is_store(statement):
        return isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
            and isinstance(statement.targets[0], ast.Subscript)

    for statement in body:
        if stores and not is_store(statement):
            fail(statement, '写回之后只能跟同域的其它写回')
        if isinstance(statement, ast.For):
            if not isinstance(statement.target, ast.Name) or statement.orelse or not statement.body:
                fail(statement, '只支持无else的单轴归约循环')
            axis = statement.target.id
            # 输出域循环：归约之后"对每个输出元素写一次"的第二个循环（层归一化、softmax 的写回）。
            # 整个循环退化成一次带该轴的写回，轴成为额外输出维，由 free_axes 遍历。
            if all(not _is_update(item) for item in statement.body):
                inner_axes = set(axes) | {axis}
                last = statement.body[-1]
                if not (isinstance(last, ast.Assign) and len(last.targets) == 1
                        and isinstance(last.targets[0], ast.Subscript)):
                    fail(statement, '输出域循环必须以张量写回结束')
                local = dict(env)
                for item in statement.body[:-1]:
                    if not (isinstance(item, ast.Assign) and len(item.targets) == 1
                            and isinstance(item.targets[0], ast.Name)):
                        fail(item, '输出域循环内只允许纯局部定义与最后的张量写回')
                    name = item.targets[0].id
                    if name in protected | set(local) | {axis}:
                        fail(item, '输出域循环的局部值不能覆盖已有定义')
                    value = capture_expr(item.value, local, params,
                                         dtypes=dtypes, axes=inner_axes)
                    local[name] = value
                    bindings.append((name, value))
                output, indices, consts = store_access(last.targets[0])
                expression = capture_expr(last.value, local, params,
                                          dtypes=dtypes, axes=inner_axes)
                stores.append((output, indices, consts, expression))
                loop_locals.add(axis)
                continue
            if axis in protected or axis in env:
                fail(statement, '归约索引不能覆盖已有变量')
            call = statement.iter
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) \
                    or call.func.id != 'range' or call.keywords or not 1 <= len(call.args) <= 3:
                fail(statement, '归约域需要range')
            values = [domain_value(arg, env, params, axes) for arg in call.args]
            if len(values) == 1:
                start_value, stop_value, step_value = None, values[0], None
            elif len(values) == 2:
                start_value, stop_value, step_value = values[0], values[1], None
            else:
                start_value, stop_value, step_value = values
            start = '0' if start_value is None else plain_text(start_value)
            stop = plain_text(stop_value)
            step = '1' if step_value is None else plain_text(step_value)
            update = statement.body[-1]
            acc, operation, value_node = capture_update(update)
            if acc in referenced_before_loop:
                fail(update, '累加器初始化值在归约前被其他定义引用，暂不支持此别名关系')
            initial = env.get(acc)
            if initial is None or acc == axis:
                fail(update, '归约累加器必须在归约循环之前初始化')
            local = dict(env)
            del local[acc]  # 任何间接旧acc依赖都必须失败，而不能被替换为初值。
            local_bindings = []
            for definition in statement.body[:-1]:
                if not isinstance(definition, ast.Assign) or len(definition.targets) != 1 \
                        or not isinstance(definition.targets[0], ast.Name):
                    fail(definition, '归约更新前只允许纯局部定义')
                name = definition.targets[0].id
                if name in protected | set(env) | {axis, acc} or name in dict(local_bindings):
                    fail(definition, '循环局部值不能覆盖已有定义')
                value = capture_expr(definition.value, local, params, dtypes, axes=set(axes) | {axis})
                local[name] = value
                local_bindings.append((name, value))
            expression = capture_expr(value_node, local, params, dtypes, axes=set(axes) | {axis})
            reduction = Reduction(axis, start, stop, step, acc, initial, 'float64', operation,
                                  start_expr=start_value, stop_expr=stop_value)
            stages.append(Stage(reduction, expression, tuple(local_bindings), statement.lineno))
            env[acc] = Expr('result', value=len(stages) - 1, dtype='float64')
            loop_locals.update(name for name, _ in local_bindings)
            loop_locals.add(axis)
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            fail(statement, '只支持纯定义、归约循环及唯一张量写回')
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            name = target.id
            if name in protected | set(env) | loop_locals:
                fail(statement, '暂不支持变量重定义或循环局部值逃逸')
            value = capture_expr(statement.value, env, params, dtypes, axes=set(axes))
            referenced_before_loop.update(node.id for node in ast.walk(statement.value)
                                          if isinstance(node, ast.Name) and node.id in env)
            env[name] = value
            bindings.append((name, value))
        else:
            output, indices, consts = store_access(target)
            expression = capture_expr(statement.value, env, params, dtypes, axes=set(axes))
            stores.append((output, indices, consts, expression))
    if not stores:
        raise Unsupported('缺少唯一张量写回')
    first = stores[0]
    return Program(tuple(stages), first[0],
                   tuple(axis for axis in first[1] if axis != AXIS_CONSTANT),
                   first[3], tuple(bindings), extra_stores=tuple(stores[1:]),
                   store_indices=first[1], store_constants=first[2])
