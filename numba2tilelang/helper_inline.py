# -*- coding: utf-8 -*-
"""optimized 路径的 AST 级 helper 内联。

把模块级 device 函数展开成**单个表达式**：
- 局部赋值按顺序折叠；
- `if c: return e` 的早退 → `c ? e : 后续表达式`；
- `if c: v += x` 这类无 return 的分支 → 合流为 `v = c ? v+x : v`。

只支持无循环的 helper（循环需要降成归约/并行域，属于另一层能力）。

20261026-167 两个正确性收口：
- S2：每次展开从**深拷贝**的 helper 体构造，展开不再原地改写共享定义
  （此前 `f(x,i,2.0)+f(x,i,3.0)` 两次都出 2.0）。
- S1：helper 的**自由名字**必须能证明为模块级绑定（import / 常量赋值 / 内建白名单），
  且不得与调用方形参/轴/局部同名（否则内联后会回落到调用方，静默误绑）；
  形参替换、合法局部同名不受影响。自由名字的顺序/分支判定见 `helper_free_names`。
"""
import ast
import copy

from semantic import Unsupported


def _strip_docstring(body):
    return [s for s in body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]


class _Substitute(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            replacement = self.mapping[node.id]
            return ast.copy_location(ast.parse(ast.unparse(replacement), mode='eval').body, node)
        return node


def _normalize_chained(node):
    """把 `a <= x < b` 规范化为 `a <= x and x < b`（guards 只认单比较）。"""
    class Chained(ast.NodeTransformer):
        def visit_Compare(self, node):
            self.generic_visit(node)
            if len(node.ops) <= 1:
                return node
            parts = []
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                parts.append(ast.Compare(left=left, ops=[op], comparators=[right]))
                left = right
            return ast.copy_location(ast.BoolOp(op=ast.And(), values=parts), node)

    return ast.fix_missing_locations(Chained().visit(node))


def _subst(node, env):
    return _normalize_chained(_Substitute(env).visit(node))


def _assign_expr(name, value, env):
    """把 `name = value` / `name += value` 折叠进 env，返回新表达式。"""
    return _subst(value, env)


def _eval_sequence(stmts, env):
    """把语句序列求值成表达式；无法表达或没有 return 时返回 None。"""
    index = 0
    while index < len(stmts):
        statement = stmts[index]
        if isinstance(statement, ast.Return):
            if statement.value is None:
                return None
            return _subst(statement.value, env)
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                and isinstance(statement.targets[0], ast.Name):
            env[statement.targets[0].id] = _subst(statement.value, env)
            index += 1
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                and isinstance(statement.targets[0], ast.Tuple):
            # 20261025-152：helper 首句“全量解包”——`a, b, c = args`。
            # 只接受**位置实参已在调用方元组赋值处物化**的元组（各分量都是 Name），
            # 保证求值发生在原赋值位置、从左到右且只求值一次；嵌套元组/星号/数量不符一律拒绝。
            target = statement.targets[0]
            value = _subst(statement.value, env)
            names = target.elts
            if any(isinstance(e, ast.Starred) for e in names) \
                    or not isinstance(value, ast.Tuple) \
                    or len(value.elts) != len(names) \
                    or not all(isinstance(e, ast.Name) for e in names) \
                    or not all(isinstance(e, ast.Name) for e in value.elts) \
                    or len({e.id for e in names}) != len(names):
                return None
            for element, component in zip(names, value.elts):
                env[element.id] = _subst(component, env)
            index += 1
            continue
        if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            name = statement.target.id
            env[name] = ast.BinOp(
                left=_subst(ast.Name(id=name, ctx=ast.Load()), env),
                op=statement.op,
                right=_subst(statement.value, env))
            index += 1
            continue
        if isinstance(statement, ast.If):
            body = [s for s in statement.body if not isinstance(s, ast.Pass)]
            if body and all(isinstance(s, ast.Return) for s in body):
                # 早退：c ? 返回值 : 后续语句的表达式
                returned = _subst(body[-1].value, env) if body[-1].value is not None else None
                if returned is None:
                    return None
                rest = _eval_sequence(stmts[index + 1:], dict(env))
                if rest is None:
                    return None
                return ast.IfExp(test=_subst(statement.test, env), body=returned, orelse=rest)
            if statement.orelse:
                return None          # 带 else 的分支合流尚未支持
            # 分支内只允许赋值：合流为 select
            branch_env = dict(env)
            for item in body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1 \
                        and isinstance(item.targets[0], ast.Name):
                    branch_env[item.targets[0].id] = _subst(item.value, branch_env)
                elif isinstance(item, ast.AugAssign) and isinstance(item.target, ast.Name):
                    name = item.target.id
                    branch_env[name] = ast.BinOp(
                        left=_subst(ast.Name(id=name, ctx=ast.Load()), branch_env),
                        op=item.op,
                        right=_subst(item.value, branch_env))
                else:
                    return None
            for name, value in branch_env.items():
                if env.get(name) is not value:
                    env[name] = ast.IfExp(test=_subst(statement.test, env),
                                          body=value, orelse=env.get(name, ast.Constant(value=0.0)))
            index += 1
            continue
        return None
    return None


def _collect_helpers(tree, kernel):
    helpers = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node is kernel:
            continue
        args = node.args
        if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults \
                or any(a.annotation is not None for a in args.args) or node.returns is not None:
            continue
        helpers[node.name] = node
    return helpers


def expand_call(node, helpers, visiting=(), scope=None):
    """把一次 helper 调用展开为表达式；不是可内联 helper 时返回 None。"""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) \
            or node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
        return None
    helper = helpers.get(node.func.id)
    if helper is None or helper.name in visiting:
        return None
    if scope is not None:
        _require_lexically_bound(helper, scope)
    params = [a.arg for a in helper.args.args]
    if len(params) != len(node.args):
        raise Unsupported('helper 实参数量不符: ' + helper.name)
    arguments = []
    for argument in node.args:
        if isinstance(argument, ast.Call) and isinstance(argument.func, ast.Name) \
                and argument.func.id in helpers:
            expanded = expand_call(argument, helpers, visiting + (helper.name,), scope=scope)
            arguments.append(expanded if expanded is not None else argument)
        else:
            arguments.append(argument)
    env = dict(zip(params, arguments))
    # S2（20261026-167）：每次展开都从**深拷贝**的 helper 体构造。
    # 直接对 helper.body 做替换会原地改写共享 AST：首个调用的实参被写回定义，
    # 后续调用从此读到被污染的 helper（f(x,i,2.0)+f(x,i,3.0) 两次都出 2.0）。
    body = _strip_docstring(copy.deepcopy(helper.body))
    result = _eval_sequence(body, env)
    if result is None:
        raise Unsupported('helper 含循环或无法表达的控制流，optimized 内联暂不支持: ' + helper.name)
    return ast.fix_missing_locations(ast.copy_location(result, node))


_BUILTIN_VISIBLE = frozenset((
    'abs', 'max', 'min', 'round', 'len', 'int', 'float', 'bool', 'tuple', 'type',
    'range', 'ValueError', 'isinstance', 'print',
))


def _target_names(target):
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names = set()
        for item in target.elts:
            names |= _target_names(item)
        return names
    return set()


def _function_locals(fn):
    """整个 helper 的**局部名集合**：函数体内任意位置（含分支/循环/增量赋值）被绑定的名字。

    Python 的局部性按整个函数决定，与出现顺序无关；`global` 声明的名字不算局部。
    """
    declared_global = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Global):
            declared_global.update(node.names)
    locals_ = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign,)):
            for target in node.targets:
                locals_ |= _target_names(target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor, ast.With)):
            target = getattr(node, 'target', None)
            if target is not None:
                locals_ |= _target_names(target)
            for item in getattr(node, 'optional_vars', []) or []:
                locals_ |= _target_names(item)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            locals_.add(node.name)
    return locals_ - {a.arg for a in fn.args.args} - declared_global


def helper_scope_report(fn):
    """把 helper 体内的名字访问分成三类（20261026-168 S1 双集合判定）。

    返回 `{'unbound_local': [...], 'free': [...]}`：
    - `unbound_local`：属于本函数局部（整个函数范围内有绑定），但在该读取处**未必已绑定**——
      源码会 UnboundLocalError，**禁止**回落模块/内建/调用方；
    - `free`：不属于局部的名字（真正的自由名字），只能按模块级静态可证绑定解析。

    顺序/分支规则（用于"必已绑定"集合）：赋值/元组解包/`for` 目标在语句之后才算绑定；
    `i += x` 的目标同时是一次读取；`if` 只在两个分支都绑定且 else 存在时，if 之后才算绑定。
    **不用"全体 Store 集合"冒充顺序定义证明**——它只用于判定局部身份。
    """
    params = {a.arg for a in fn.args.args}
    locals_ = _function_locals(fn)
    unbound_local, free = set(), set()

    def load(node, bound):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) \
                    and sub.id not in bound and sub.id not in params:
                if sub.id in locals_:
                    unbound_local.add(sub.id)
                else:
                    free.add(sub.id)

    def walk(stmts, bound):
        for statement in stmts:
            if isinstance(statement, ast.Return):
                if statement.value is not None:
                    load(statement.value, bound)
            elif isinstance(statement, ast.Assign):
                load(statement.value, bound)
                for target in statement.targets:
                    bound |= _target_names(target)
            elif isinstance(statement, ast.AnnAssign):
                if statement.value is not None:
                    load(statement.value, bound)
                bound |= _target_names(statement.target)
            elif isinstance(statement, ast.AugAssign):
                if isinstance(statement.target, ast.Name) and statement.target.id not in bound:
                    if statement.target.id in locals_:
                        unbound_local.add(statement.target.id)
                    elif statement.target.id not in params:
                        free.add(statement.target.id)
                load(statement.value, bound)
                bound |= _target_names(statement.target)
            elif isinstance(statement, ast.If):
                load(statement.test, bound)
                then_bound = set(bound)
                walk(statement.body, then_bound)
                if statement.orelse:
                    else_bound = set(bound)
                    walk(statement.orelse, else_bound)
                    bound |= (then_bound & else_bound)
                # 无 else：then 分支内的绑定在 if 之后不成立
            elif isinstance(statement, ast.For):
                load(statement.iter, bound)
                inner = set(bound) | _target_names(statement.target)
                walk(statement.body, inner)
            elif isinstance(statement, ast.While):
                load(statement.test, bound)
                walk(statement.body, set(bound))
            else:
                for sub in ast.walk(statement):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) \
                            and sub.id not in bound and sub.id not in params:
                        if sub.id in locals_:
                            unbound_local.add(sub.id)
                        else:
                            free.add(sub.id)

    walk(list(fn.body), set(params))
    return {'unbound_local': sorted(unbound_local), 'free': sorted(free)}


def helper_free_names(fn):
    """兼容接口：只返回真正自由的名字（不含"局部但此刻未绑定"的那一类）。"""
    return set(helper_scope_report(fn)['free'])


def _module_binding_events(tree):
    """模块级绑定事件表：`name -> [(kind, conditional), …]`（按源码顺序）。

    只扫模块顶层语句；**不进入** FunctionDef / ClassDef / Lambda 体（函数内部局部不是模块名）。
    模块级控制流（if/for/while/with/try）里的绑定记为 conditional=True。
    kind ∈ {'import', 'assign', 'augassign', 'delete', 'def', 'class', 'for', 'with', 'except'}。
    """
    events = {}

    def record(target, kind, conditional):
        for name in _target_names(target):
            events.setdefault(name, []).append((kind, conditional))

    def scan_block(stmts, conditional):
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                record(ast.Name(id=node.name, ctx=ast.Store()), 'def', conditional)
                continue                     # 不进入函数体
            if isinstance(node, ast.ClassDef):
                record(ast.Name(id=node.name, ctx=ast.Store()), 'class', conditional)
                continue                     # 不进入类体
            if isinstance(node, ast.Import):
                for alias in node.names:
                    record(ast.Name(id=alias.asname or alias.name.split('.')[0],
                                    ctx=ast.Store()), 'import', conditional)
                continue
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == '*':
                        continue
                    record(ast.Name(id=alias.asname or alias.name, ctx=ast.Store()),
                           'import', conditional)
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    record(target, 'assign', conditional)
                continue
            if isinstance(node, ast.AnnAssign):
                if node.value is not None:   # `x: int` 无值不绑定
                    record(node.target, 'assign', conditional)
                continue
            if isinstance(node, ast.AugAssign):
                record(node.target, 'augassign', conditional)
                continue
            if isinstance(node, ast.Delete):
                for target in node.targets:
                    record(target, 'delete', conditional)
                continue
            if isinstance(node, (ast.For, ast.AsyncFor)):
                record(node.target, 'for', conditional)
                scan_block(node.body, True)
                scan_block(node.orelse, True)
                continue
            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        record(item.optional_vars, 'with', conditional)
                scan_block(node.body, True)
                continue
            if isinstance(node, (ast.If, ast.While)):
                scan_block(node.body, True)
                scan_block(node.orelse, True)
                continue
            if isinstance(node, ast.Try):
                scan_block(node.body, True)
                for handler in node.handlers:
                    if handler.name:
                        record(ast.Name(id=handler.name, ctx=ast.Store()), 'except', True)
                    scan_block(handler.body, True)
                scan_block(node.orelse, True)
                scan_block(node.finalbody, True)
                continue
            # 其它顶层语句不产生模块绑定

    scan_block(list(tree.body), False)
    return events


def _fold_rhs(node, env):
    """在**受限事实环境**里折叠模块级赋值右值；只能在证明值可复现时返回数值。

    只支持：常量、已证 const 名字、单目 ±、四则与整除/取模/幂、常量元组。
    不执行源码、不调用任何函数/属性（`math.sqrt(4.0)` 一类一律不可证）。
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return True, node.value
        return False, None
    if isinstance(node, ast.Name):
        fact = env.get(node.id)
        if fact is not None and fact[0] == 'const':
            return True, fact[1]
        return False, None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        ok, value = _fold_rhs(node.operand, env)
        if not ok:
            return False, None
        if not isinstance(value, (int, float)):
            return False, None
        return True, (-value if isinstance(node.op, ast.USub) else value)
    if isinstance(node, ast.BinOp):
        ok_l, left = _fold_rhs(node.left, env)
        ok_r, right = _fold_rhs(node.right, env)
        if not (ok_l and ok_r) or not isinstance(left, (int, float)) \
                or not isinstance(right, (int, float)):
            return False, None
        try:
            if isinstance(node.op, ast.Add):
                return True, left + right
            if isinstance(node.op, ast.Sub):
                return True, left - right
            if isinstance(node.op, ast.Mult):
                return True, left * right
            if isinstance(node.op, ast.Div):
                return (True, left / right) if right != 0 else (False, None)
            if isinstance(node.op, ast.FloorDiv):
                return (True, left // right) if right != 0 else (False, None)
            if isinstance(node.op, ast.Mod):
                return (True, left % right) if right != 0 else (False, None)
            if isinstance(node.op, ast.Pow):
                if isinstance(left, int) and isinstance(right, int) and abs(right) <= 16:
                    return True, left ** right
                return False, None
        except Exception:  # noqa: BLE001
            return False, None
        return False, None
    if isinstance(node, ast.Tuple):
        values = []
        for element in node.elts:
            ok, value = _fold_rhs(element, env)
            if not ok:
                return False, None
            values.append(value)
        return True, tuple(values)
    return False, None


def _module_facts(tree):
    """按源码顺序、用**独立受限环境**计算模块级名字事实。

    事实：`('const', value)` / `('import',)` / `None`（不可证）。
    条件（if/for/while/with/try）内的任何绑定（含 import）→ 该名字事实立即失效且不重建（执行与否不可证）；
    import 覆盖旧 const（`from math import pi as X` 之后 X 不可证）；
    派生赋值只在 RHS 能由**此刻已证 const**折叠时建立 const 事实。
    原样保留源顺序，不做"最终状态倒推"。
    """
    env = {}

    def invalidate(target):
        for name in _target_names(target):
            env[name] = None

    def bind(target, fact):
        if isinstance(target, ast.Name):
            env[target.id] = fact
        elif isinstance(target, (ast.Tuple, ast.List)):
            if fact is None or fact[0] != 'const' or not isinstance(fact[1], tuple) \
                    or len(fact[1]) != len(target.elts):
                invalidate(target)
                return
            for element, value in zip(target.elts, fact[1]):
                if isinstance(element, ast.Name):
                    env[element.id] = ('const', value)
                else:
                    invalidate(element)

    def scan(stmts, conditional):
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                env[node.name] = None
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[0]
                    # 条件 import 不可证：不建立 import 事实，并使旧事实失效
                    env[name] = None if conditional else ('import',)
                continue
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == '*':
                        continue
                    name = alias.asname or alias.name
                    env[name] = None if conditional else ('import',)
                continue
            if isinstance(node, ast.Assign):
                ok, value = (False, None) if conditional else _fold_rhs(node.value, env)
                fact = ('const', value) if ok else None
                for target in node.targets:
                    bind(target, fact)
                continue
            if isinstance(node, ast.AnnAssign):
                if node.value is None:
                    continue                     # `x: float` 不绑定
                ok, value = (False, None) if conditional else _fold_rhs(node.value, env)
                bind(node.target, ('const', value) if ok else None)
                continue
            if isinstance(node, ast.AugAssign):
                invalidate(node.target)
                continue
            if isinstance(node, ast.Delete):
                for target in node.targets:
                    invalidate(target)
                continue
            if isinstance(node, (ast.For, ast.AsyncFor)):
                invalidate(node.target)
                scan(node.body, True)
                scan(node.orelse, True)
                continue
            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        invalidate(item.optional_vars)
                scan(node.body, True)
                continue
            if isinstance(node, (ast.If, ast.While)):
                scan(node.body, True)
                scan(node.orelse, True)
                continue
            if isinstance(node, ast.Try):
                scan(node.body, True)
                for handler in node.handlers:
                    if handler.name:
                        env[handler.name] = None
                    scan(handler.body, True)
                scan(node.orelse, True)
                scan(node.finalbody, True)
                continue

    scan(list(tree.body), False)
    return env


def _module_scope(tree, kernel=None):
    """模块作用域事实：`{'values': module_scalars, 'facts': 独立求值事实, 'bound': 有绑定的名字}`。

    两条来源**独立**：
    - `facts` 由 `_module_facts` 用受限环境按源码顺序算出（来源证明）；
    - `values` 是下游 `frontend.module_scalars` 实际会用的常量（使用方）。
    门里必须两边对账（类型 + 值），任何一方单独都不能当证明。
    """
    from frontend import module_scalars
    values = module_scalars(tree, kernel)
    facts = _module_facts(tree)
    return {'values': values, 'facts': facts, 'bound': set(facts)}


def _module_visible_names(tree, kernel=None):
    """兼容接口：可带入的名字（const 事实 ∪ import 事实）。"""
    scope = _module_scope(tree, kernel)
    return {name for name, fact in scope['facts'].items() if fact is not None}


def _require_lexically_bound(helper, scope):
    """S1（20261026-170）：双集合词法门 + **单一顺序事实**的模块名资格门。

    1. 属函数局部但此刻未必已绑定 → 拒绝（不回落模块/内建/调用方）。
    2. 自由名字：
       - 事实为 `None`（条件绑定 / 不可折叠 / 增强 / 删除 / def / class / for·with·except）→ 拒绝；
       - 事实 `('const', value)`：必须与下游 `module_scalars` 的值一致（防止旧事实越过失效）→ 否则拒绝；
       - 事实 `('import',)`：下游不得把它当常量（`K = 5.0; from math import pi as K` 这类
         旧标量资格必须失效）→ 否则拒绝；
       - 内建白名单只在模块里**没有任何同名绑定**时放行。
    3. 自由名字与调用方形参/轴/局部同名 → 拒绝（遮蔽不可保真）。
    4. 形参替换进调用方的 AST、合法局部同名（先绑定后使用）不受影响。
    """
    module_scope, caller_names = scope
    report = helper_scope_report(helper)
    if report['unbound_local']:
        raise Unsupported('helper 的局部名字在该读取处尚未必然绑定: %s（%s）'
                          % (', '.join(report['unbound_local']), helper.name))
    facts = module_scope['facts']
    values = module_scope['values']
    bound = module_scope['bound']
    unknown, inconsistent = [], []
    for name in report['free']:
        fact = facts.get(name)
        if fact is None and name in _BUILTIN_VISIBLE and name not in bound:
            continue                       # 真正的内建，且模块里没有同名绑定
        if fact is None:
            unknown.append(name)
            continue
        if fact[0] == 'const':
            # 类型与值层都对齐下游实际会用的常量；不是自比（facts 来自独立的受限求值）
            if name not in values or type(values[name]) is not type(fact[1]) \
                    or values[name] != fact[1]:
                inconsistent.append(name)
            continue
        # fact[0] == 'import'
        if name in values:
            # 下游会把 imports 之后被覆盖前的旧标量当常量用 → 事实不一致，拒绝
            inconsistent.append(name)
    if unknown:
        raise Unsupported('helper 的自由名字无法证明为模块级绑定: %s（%s）'
                          % (', '.join(sorted(unknown)), helper.name))
    if inconsistent:
        raise Unsupported('helper 的自由名字的模块末次绑定与下游常量不一致: %s（%s）'
                          % (', '.join(sorted(inconsistent)), helper.name))
    shadowed = sorted(name for name in report['free'] if name in caller_names)
    if shadowed:
        raise Unsupported('helper 的自由名字被调用方同名绑定遮蔽，无法保真: %s（%s）'
                          % (', '.join(shadowed), helper.name))


def _tuple_param_helpers(tree, kernel):
    """同文件、单形参、首句全量解包、随后单 return 的 helper（20261025-152 窄门）。"""
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node is kernel:
            continue
        args = node.args
        if len(args.args) != 1 or args.posonlyargs or args.kwonlyargs \
                or args.vararg or args.kwarg or args.defaults:
            continue
        body = _strip_docstring(node.body)
        if len(body) != 2 or not isinstance(body[0], ast.Assign) \
                or len(body[0].targets) != 1 or not isinstance(body[0].targets[0], ast.Tuple) \
                or not isinstance(body[0].value, ast.Name) \
                or body[0].value.id != args.args[0].arg \
                or not isinstance(body[1], ast.Return) or body[1].value is None:
            continue
        names = body[0].targets[0].elts
        if any(not isinstance(e, ast.Name) for e in names) \
                or len({e.id for e in names}) != len(names):
            continue
        found[node.name] = (len(names), {e.id for e in names})
    return found


def _all_names(tree):
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def materialize_fixed_tuple_helpers(tree, kernel):
    """把 `t = (e1, …, en)` + 紧邻唯一 `helper(t)` 就地物化成逐分量标量绑定。

    语义要点（20261025-152）：元组在**赋值处**求值，左到右、只求值一次；随后把调用
    实参换成这些**已绑定名**组成的元组，交给既有表达式内联。不移动/复制原始表达式，
    元组不得逃逸、别名、重绑定、动态索引、返回或多次使用。
    """
    helpers = _tuple_param_helpers(tree, kernel)
    if not helpers:
        return False
    used = _all_names(tree)
    counter = [0]

    def fresh(base):
        while True:
            counter[0] += 1
            name = '_tl_tp_%s_%d' % (base, counter[0])
            if name not in used:
                used.add(name)
                return name

    changed = [False]

    def rewrite(statements):
        index = 0
        while index < len(statements):
            statement = statements[index]
            if isinstance(statement, (ast.If, ast.For, ast.While, ast.With)):
                for field in ('body', 'orelse'):
                    block = getattr(statement, field, None)
                    if block:
                        rewrite(block)
                index += 1
                continue
            if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and isinstance(statement.value, ast.Tuple)):
                index += 1
                continue
            target = statement.targets[0]
            elements = statement.value.elts
            if any(isinstance(e, ast.Starred) or isinstance(e, ast.Tuple) for e in elements):
                index += 1
                continue
            # 目标名在整个 kernel 中必须恰好出现：一次元组赋值 + 一次调用实参。
            loads = [n for n in ast.walk(kernel)
                     if isinstance(n, ast.Name) and n.id == target.id
                     and isinstance(n.ctx, ast.Load)]
            stores = [n for n in ast.walk(kernel)
                      if isinstance(n, ast.Name) and n.id == target.id
                      and isinstance(n.ctx, (ast.Store, ast.Del))]
            if len(loads) != 1 or len(stores) != 1:
                index += 1
                continue
            if index + 1 >= len(statements):
                index += 1
                continue
            call_use = loads[0]
            following = statements[index + 1]
            if call_use not in list(ast.walk(following)):
                index += 1
                continue
            parent_call = None
            for node in ast.walk(kernel):
                if isinstance(node, ast.Call) and call_use in node.args:
                    parent_call = node
                    break
            if parent_call is None or parent_call.args != [call_use]:
                index += 1
                continue
            if not (isinstance(parent_call.func, ast.Name)
                    and parent_call.func.id in helpers):
                index += 1
                continue
            arity, _names = helpers[parent_call.func.id]
            if arity != len(elements):
                index += 1
                continue
            # 就地物化：逐分量标量绑定，保持原顺序；随后把调用实参换成这些已绑定名。
            components = [fresh(target.id) for _ in elements]
            new_statements = [ast.fix_missing_locations(ast.Assign(
                targets=[ast.Name(id=component, ctx=ast.Store())],
                value=element))
                for component, element in zip(components, elements)]
            statements[index:index + 1] = new_statements
            replacement = ast.Tuple(
                elts=[ast.Name(id=component, ctx=ast.Load()) for component in components],
                ctx=ast.Load())

            class Swap(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node is call_use:
                        return ast.copy_location(replacement, node)
                    return node

            next_index = index + len(new_statements)
            statements[next_index] = Swap().visit(statements[next_index])
            index = next_index + 1
            changed[0] = True

    rewrite(kernel.body)
    ast.fix_missing_locations(kernel)
    return changed[0]


def inline_linear_helpers(tree, kernel):
    """在 kernel 体内原地内联被调用的 helper；返回是否发生替换。"""
    materialize_fixed_tuple_helpers(tree, kernel)
    helpers = _collect_helpers(tree, kernel)
    if not helpers:
        return False
    module_scope = _module_scope(tree, kernel)
    caller_names = {arg.arg for arg in kernel.args.args}
    caller_names |= {node.id for node in ast.walk(kernel)
                     if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    scope = (module_scope, frozenset(caller_names))
    changed = [False]

    class Inliner(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in helpers:
                expanded = expand_call(node, helpers, scope=scope)
                if expanded is not None:
                    changed[0] = True
                    return expanded
            return node

    kernel.body = [Inliner().visit(statement) for statement in kernel.body]
    ast.fix_missing_locations(kernel)
    return changed[0]
