"""单元素读取与同型浮点加法device函数的静态契约；不构造通用内联器。"""
import ast
import copy
from semantic import Unsupported
from source_contract import is_cuda_jit
from thread_atomic import validate_cuda_binding
from thread_bindings import Bindings
from thread_ir import ScalarArg, TensorArg


def fail(node, message):
    raise Unsupported(f'line {getattr(node, "lineno", 0)}: device只读子集: {message}')


class DeviceBindings(Bindings):
    def __init__(self, names, definitions):
        super().__init__(names)
        self.definitions = definitions

    def check(self, node, name):
        if name in self.names and node not in self.definitions:
            fail(node, '函数名称被遮蔽或重绑定: ' + name)


_MISSING = object()


class _DeviceTuple(tuple):
    """device 函数返回元组时的占位；只允许出现在元组赋值右侧。

    fix13 让 helper 可以 `return a, b`，但元组只在 `a, b = f(...)` 里有意义。
    若被当成标量使用（`out[i] = f(...)`、参与运算等），必须在这一层明确拒绝；
    否则元组会一路流到 lowering，以 AttributeError 的形式变成非预期异常。
    """
    def __new__(cls, items, node):
        obj = super().__new__(cls, items)
        obj.node = node
        return obj

    def __getattr__(self, item):
        node = self.__dict__.get('node')
        raise Unsupported(
            f'line {getattr(node, "lineno", 0)}: device 函数返回的元组只能用于元组赋值解包')


def early_return_ok(body):
    """判定"早退"形态：return 只出现在顶层语句或顶层 if 的**直接** body 里；
    if 的 else 分支、循环体内不得有 return。该形态可安全改写为 if/else 嵌套。"""
    for stmt in body:
        if isinstance(stmt, ast.Return):
            continue
        if isinstance(stmt, ast.If):
            # else 子句会被原样搬进改写后的 if/else，因此其中任何位置（含嵌套 if/循环）
            # 的 return 都改不了；不在这里拦下，就只能靠下游语句层报错。
            if any(isinstance(y, ast.Return)
                   for x in stmt.orelse for y in ast.walk(x)):
                return False
            for item in stmt.body:
                if isinstance(item, ast.Return):
                    continue
                if isinstance(item, (ast.If, ast.For, ast.While)) \
                        and any(isinstance(y, ast.Return) for y in ast.walk(item)):
                    return False
        elif isinstance(stmt, (ast.For, ast.While)):
            if any(isinstance(y, ast.Return) for y in ast.walk(stmt)):
                return False
    return True


def rewrite_early_returns(body, result_name):
    """把早退 return 改写为 if/else 嵌套 + 结果赋值；返回 (new_body, ok)。

    `if c: ... return e` 之后的所有语句在源语义里不可达，改写后放进 else 分支。
    """
    for index, stmt in enumerate(body):
        if isinstance(stmt, ast.Return):
            assign = ast.Assign(targets=[ast.Name(id=result_name, ctx=ast.Store())],
                                value=stmt.value if stmt.value is not None else ast.Constant(value=None))
            return list(body[:index]) + [ast.copy_location(assign, stmt)], True
        if isinstance(stmt, ast.If) and any(isinstance(x, ast.Return) for x in stmt.body):
            yes, ok_yes = rewrite_early_returns(stmt.body, result_name)
            if not ok_yes:
                return body, False
            rest, ok_rest = rewrite_early_returns(body[index + 1:], result_name)
            if not ok_rest:
                return body, False
            # 条件为假时：先执行原 else 分支（若有），再继续后续语句。
            # 原 else 不得丢弃——丢弃会生成语法合法但语义错误的代码。
            no = list(stmt.orelse) + rest
            branch = ast.If(test=stmt.test, body=yes, orelse=no)
            return list(body[:index]) + [ast.copy_location(branch, stmt)], True
    return body, False


def guard_void_returns(body):
    """void 过程的早退：`if c: return` + 后续语句 → `if not c: 后续语句`。

    只接受"该 if 的 body 全是 return 且无 else"的形态；返回改写后的语句列表，
    不支持时返回 None。提前返回不产生值，因此不需要结果槽。
    """
    for index, stmt in enumerate(body):
        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                return None
            return list(body[:index])          # 末尾 return：其后语句不可达
        if isinstance(stmt, ast.If) and any(isinstance(x, ast.Return) for x in stmt.body):
            if stmt.orelse or not all(isinstance(x, ast.Return) for x in stmt.body):
                return None
            if any(x.value is not None for x in stmt.body):
                return None
            rest = guard_void_returns(body[index + 1:])
            if rest is None:
                return None
            test = ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=stmt.test), stmt.test)
            guard = ast.If(test=test, body=rest, orelse=[])
            return list(body[:index]) + [ast.copy_location(guard, stmt)]
    return list(body)          # 无（更多）早退：原样返回，调用方继续正常展开


def _strip_docstring(body):
    body = list(body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body.pop(0)
    return body


def classify_device(fn):
    """把被 kernel 调用的模块级函数分成三类：legacy（有签名的旧两形态）/ inline / unsupported。

    这里只**判定**，不抛错——不可内联的形态要在**调用点**报出具体原因，
    而不是在收集阶段用旧的两形参错误把整例拦下（第一版就是这么退化的）。
    """
    args = fn.args
    if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults \
            or any(a.annotation is not None for a in args.args) or fn.returns is not None:
        return ('unsupported', 'device 函数仅支持无注解、无默认值的位置形参')
    if len(fn.decorator_list) > 1:
        return ('unsupported', '不支持额外的 device 函数装饰器')
    decorator = fn.decorator_list[0] if fn.decorator_list else None
    if decorator is not None:
        if not (isinstance(decorator, ast.Call) and is_cuda_jit(decorator)):
            return ('unsupported', 'device 函数装饰器必须是 cuda.jit')
        for kw in decorator.keywords:
            if kw.arg not in ('device', 'inline') or not isinstance(kw.value, ast.Constant) \
                    or type(kw.value.value) is not bool:
                return ('unsupported', '装饰器仅支持 device/inline 布尔常量')
        if decorator.args:
            return ('legacy', None)          # 显式签名：交给旧的标量加法/单元素读取校验
    body = _strip_docstring(fn.body)
    if not body:
        return ('unsupported', 'device 函数体为空')
    outer_returns = [i for i, s in enumerate(body) if isinstance(s, ast.Return)]
    nested_returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n not in body]
    if len(outer_returns) > 1:
        return ('unsupported', '内联 device 函数暂不支持多个顶层 return')
    if nested_returns:
        # 早退形态：return 只在顶层 if 的直接 body 里（无 else-return、无循环内 return），
        # 可安全改写为 if/else 嵌套后按普通内联展开。
        if not early_return_ok(body):
            return ('unsupported', '内联 device 函数暂不支持该分支内 return 形态')
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        if all(r.value is None for r in returns):
            # void 过程的提前返回：谓词化为 `if not c: 后续语句`。
            if guard_void_returns(body) is None:
                return ('unsupported', 'void 过程的早退形态无法安全改写')
            return ('inline-void-early', None)
        if any(r.value is None for r in returns):
            return ('unsupported', 'device 函数不能混用有值/无值 return')
        return ('inline-early', None)
    if outer_returns and outer_returns[0] != len(body) - 1:
        return ('unsupported', 'return 必须是函数体最后一条语句')
    # 循环体本身可以内联（逐条展开到调用者 IR）；循环里出现 return 已在上面按"分支内 return"拒绝。
    return ('inline', None)


def collect_device_reads(tree, kernel, signature_contract, aliases, reserved):
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    candidates = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decorated = any(is_cuda_jit(d) and isinstance(d, ast.Call) and any(
            k.arg == 'device' and isinstance(k.value, ast.Constant)
            and k.value.value is True for k in d.keywords) for d in node.decorator_list)
        # 无装饰器的模块级函数也纳入候选：kernel 内调用普通 Python 函数在 CUDA 语义下
        # 只能指向 device 函数（closure 整理时可能剥掉装饰器）。是否内联由 classify 决定。
        if decorated or not node.decorator_list:
            candidates.setdefault(node.name, []).append(node)
    # 取「kernel 直接调用 ∪ helper 体内再调用」的传递闭包：helper 链上的函数也必须收集，
    # 否则内联到第二层时调用点会落到"未实现的线程表达式"。
    used = set()
    frontier = {n.id for n in ast.walk(kernel) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)} & candidates.keys()
    while frontier:
        used |= frontier
        frontier = ({n.id for name in frontier for node in candidates[name]
                     for n in ast.walk(node) if isinstance(n, ast.Name)
                     and isinstance(n.ctx, ast.Load)} & candidates.keys()) - used
    if not used:
        return {}
    definitions = {}
    for name in used:
        matches = candidates[name]
        if len(matches) != 1 or parents[matches[0]] is not parents[kernel] \
                or not isinstance(parents[kernel], (ast.Module, ast.FunctionDef)):
            fail(kernel, '需要唯一同级device函数: ' + name)
        if name in reserved or name.startswith('_tl_'):
            fail(matches[0], '函数名称与保留名称冲突: ' + name)
        definitions[name] = matches[0]
    checker = DeviceBindings(used, set(definitions.values()))
    scope = kernel
    while True:
        if hasattr(scope, 'args'):
            checker.visit(scope.args)
        for node in scope.body:
            checker.visit(node)
        if scope is tree:
            break
        scope = parents[scope]
        while not isinstance(scope, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            scope = parents[scope]
    for node in ast.walk(kernel):
        if isinstance(node, ast.Name) and node.id in used:
            parent = parents[node]
            if not isinstance(parent, ast.Call) or parent.func is not node:
                fail(node, '函数仅支持名称直接调用')
    result = {}
    for name, fn in definitions.items():
        validate_cuda_binding(tree, fn, fn, parents)
        kind, reason = classify_device(fn)
        if kind in ('inline', 'inline-early', 'inline-void-early'):
            result[name] = (fn, None, None, tree, kind)
            continue
        if kind == 'unsupported':
            result[name] = (fn, None, None, tree, ('unsupported', reason))
            continue
        args = fn.args
        if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg \
                or args.defaults or len(args.args) != 2 \
                or any(a.annotation is not None for a in args.args) or fn.returns is not None:
            fail(fn, '需要两个无注解无默认值的普通形参')
        names = [a.arg for a in args.args]
        if len(set(names)) != 2:
            fail(fn, '形参名称必须不同')
        body = list(fn.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body.pop(0)
        value = body[0].value if len(body) == 1 and isinstance(body[0], ast.Return) else None
        scalar_add = (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
                      and isinstance(value.left, ast.Name) and value.left.id == names[0]
                      and isinstance(value.right, ast.Name) and value.right.id == names[1])
        if not scalar_add and (not isinstance(value, ast.Subscript)
                or not isinstance(value.value, ast.Name) or value.value.id != names[0]
                or not isinstance(value.slice, ast.Name) or value.slice.id != names[1]):
            fail(fn, '正文仅支持return 第一个形参[第二个形参]')
        if len(fn.decorator_list) != 1:
            fail(fn, '不支持额外装饰器')
        decorator = fn.decorator_list[0]
        seen = set()
        for kw in decorator.keywords:
            if kw.arg in seen or kw.arg not in ('device', 'inline') \
                    or not isinstance(kw.value, ast.Constant) or type(kw.value.value) is not bool:
                fail(kw, '装饰器仅支持device/inline布尔常量')
            seen.add(kw.arg)
        if len(decorator.args) != 1 or not isinstance(decorator.args[0], ast.Constant) \
                or type(decorator.args[0].value) is not str:
            fail(fn, '需要单一显式字符串返回签名')
        try:
            sig = ast.parse(decorator.args[0].value, mode='eval').body
        except SyntaxError:
            fail(fn, '无法解析返回签名')
        if not isinstance(sig, ast.Call) or not isinstance(sig.func, ast.Name) or sig.keywords:
            fail(fn, '需要标量返回签名')
        return_dtype = aliases.get(sig.func.id, sig.func.id)
        # 只在局部副本上将已核返回类型拆出，参数部分复用入口签名检查。
        checked = copy.deepcopy(fn)
        sig.func.id = 'void'
        checked.decorator_list[0].args = [ast.Constant(value=ast.unparse(sig))]
        checked.decorator_list[0].keywords = []
        result[name] = (checked, return_dtype, signature_contract, tree, scalar_add)
    return result


def capture_device_inline(capture, node, env, output, fn, early=False):
    """受限内联：形参在**调用者**作用域按源码顺序求值；函数体逐条展开；末尾 return 的值即结果。

    * 张量实参先在调用者作用域解析，再挂到本次独立的 `capture.tensor_aliases`。
      helper 只能看见自己的张量形参；调用后恢复调用者作用域。
    * void 过程（无 return）展开全部语句并返回 `None`；调用点若把它当值使用由语句层拒绝。
    * 局部变量只写临时作用域，不回写调用者。
    """
    params = [a.arg for a in fn.args.args]
    if node.keywords or len(node.args) != len(params):
        fail(node, '调用实参数与 device 函数形参不符')
    bindings = []
    array_bindings = {}
    # 实参全部在调用者（包括父 helper）的作用域求值，不能提前安装本次形参别名。
    for name, arg in zip(params, node.args):
        if isinstance(arg, ast.Name):
            if arg.id in capture.arrays:
                array_bindings[name] = capture.arrays[arg.id]
            elif arg.id not in env:
                actual = capture.tensor_name(arg.id)
                if isinstance(capture.parameters.get(actual), TensorArg):
                    capture.tensor(arg)  # 同时核对父 helper 的名称可见性。
                    bindings.append((name, actual, None))
                    continue
        bindings.append((name, None, capture.expr(arg, env, output)))
    assigned = {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    tensor_names = {name for name, actual, _ in bindings if actual is not None}
    if assigned & (tensor_names | set(array_bindings)):
        fail(fn, '暂不支持重绑定 device 张量或数组形参')
    saved_aliases = capture.tensor_aliases
    saved_arrays = capture.arrays
    saved_constants = capture.constants
    saved_tensors = capture.device_tensors
    try:
        capture.tensor_aliases = {}
        capture.arrays = dict(array_bindings)
        capture.device_tensors = tensor_names
        bound = set(params) | assigned
        capture.constants = {name: value for name, value in capture.module_values.items()
                             if name not in bound and name not in capture.reserved}
        local = capture.device_scope(fn, output)
        for name, actual, value in bindings:
            if actual is not None:
                capture.tensor_aliases[name] = actual
            else:
                local[name] = value
        body = _strip_docstring(fn.body)
        if early == 'void':
            # void 过程的早退：谓词化为 `if not c: 后续语句`，不产生结果值。
            guarded = guard_void_returns(body)
            if guarded is None:
                fail(fn, 'void 过程的早退形态无法安全改写')
            for stmt in guarded:
                statements, local = capture.block([stmt], local)
                output.extend(statements)
            return None
        if early:
            # 早退形态：改写成 if/else 嵌套 + 结果赋值，再按普通语句展开；
            # 结果在局部作用域里取值，不回写调用者。
            result_name = '_tl_early'
            while result_name in local:
                result_name += '_'
            body, ok = rewrite_early_returns(body, result_name)
            if not ok:
                fail(fn, '早退形态无法安全改写')
            if any(isinstance(x, ast.Return) for stmt in body for x in ast.walk(stmt)):
                # 改写必须覆盖所有 return；残留说明有语句被漏搬，宁可拒绝也不生成。
                fail(fn, '早退改写后仍有 return 残留，拒绝生成')
            for stmt in body:
                statements, local = capture.block([stmt], local)
                output.extend(statements)
            if result_name not in local:
                fail(fn, '早退改写未覆盖所有分支，函数可能不返回值')
            return local[result_name]
        tail = body[-1] if body and isinstance(body[-1], ast.Return) else None
        for stmt in (body[:-1] if tail else body):
            statements, local = capture.block([stmt], local)
            output.extend(statements)
        if tail is None:
            return None
        if isinstance(tail.value, ast.Tuple):
            # helper 返回元组：逐元素求值后包成 _DeviceTuple，供元组赋值拆包；
            # 被当标量使用时由包装类明确拒绝（见 _DeviceTuple）。
            return _DeviceTuple((capture.expr(item, local, output) for item in tail.value.elts), node)
        return capture.expr(tail.value, local, output)
    finally:
        capture.tensor_aliases = saved_aliases
        capture.arrays = saved_arrays
        capture.constants = saved_constants
        capture.device_tensors = saved_tensors


def capture_device_read(capture, node, env, output):
    checked, return_dtype, signature_contract, tree, scalar_add = capture.devices[node.func.id]
    if isinstance(scalar_add, tuple) and scalar_add[0] == 'unsupported':
        fail(node, scalar_add[1])
    if scalar_add in ('inline', 'inline-early', 'inline-void-early'):
        return capture_device_inline(capture, node, env, output, checked,
                                     early={'inline-early': 'value',
                                            'inline-void-early': 'void'}.get(scalar_add))
    if node.keywords or len(node.args) != 2:
        fail(node, '调用需要两个位置参数')
    if scalar_add:
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id in capture.devices for arg in node.args for n in ast.walk(arg)):
            fail(node, '标量加法不支持嵌套device调用')
        values = [capture.expr(arg, env, output) for arg in node.args]
        if return_dtype not in ('float32', 'float64') \
                or any(value.dtype != return_dtype for value in values):
            fail(node, '标量加法仅支持实参与返回签名同型的float32/float64')
        parameters = {arg.arg: ScalarArg(value.dtype)
                      for arg, value in zip(checked.args.args, values)}
        signature_contract(tree, copy.deepcopy(checked), parameters)
        return capture.calculate('+', tuple(values), node, output)
    tensor = capture.tensor(node.args[0])
    if len(tensor.shape) != 1:
        fail(node, '张量实参必须为rank1')
    index = capture.expr(node.args[1], env, output)
    if index.dtype not in ('int32', 'int64'):
        fail(node, '下标必须为int32或int64')
    if return_dtype != tensor.dtype:
        fail(node, '返回签名必须与元素dtype一致')
    parameters = {checked.args.args[0].arg: tensor,
                  checked.args.args[1].arg: ScalarArg(index.dtype)}
    signature_contract(tree, copy.deepcopy(checked), parameters)
    return capture.put(output, 'load', tensor.dtype, node, (index,), node.args[0].id)
