# -*- coding: utf-8 -*-
"""批次内两个“部分域条件写”结构的**独立资格门**（20261025-156）。

S1 圆域谓词（`partial_predicate`）：外层 guard 是常量界的完整轴界，体内只有一个
   `if <轴算术谓词>:` 包住的写回循环；把该谓词提升为**元素掩码**，体内语句就地展开。
   掩码通道会强制 caller-output（既有规则），未命中元素保持调用方传入值。
S2 轴等值 guard（`partial_axis`）：guard 里有一条 `轴 == 常量`，其余是 `<` 上界；
   把该轴**绑定为常量下标**（等价于“该轴恰为该值时才写”），并把它从 guard 中移除。

两者都只做**资格与 AST 归一**，发射完全复用既有 capture/lower 路径；
任何不满足形态的输入返回 False（不抛异常、不改写 body），交回旧路径。
"""
import ast


def _is_int_const(node, consts):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    return None


def _pure_predicate(node, axes, params, consts):
    """谓词只允许：轴名、整型/浮点常量、算术与比较；不得有调用/属性/下标/未知名。"""
    for item in ast.walk(node):
        if isinstance(item, (ast.Call, ast.Attribute, ast.Subscript)):
            return False
        if isinstance(item, ast.Name) and item.id not in axes and item.id not in consts:
            return False
    return True


def hoist_predicate_mask(body, spec, consts):
    """S1：把体内**最后一条** `if <轴算术谓词>:` 提升为元素掩码。

    允许 If 之前有若干“常量派生”的局部赋值（`cx = nx // 4`），它们先按字面量折叠进常量表；
    这些语句原样保留在 body 里（它们只是标量绑定）。成功返回 `(new_body, cond_text)`。
    """
    from frontend import literal_arithmetic, literal_sequence

    statements = [s for s in body
                  if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if not statements or not isinstance(statements[-1], ast.If):
        return None
    statement = statements[-1]
    if statement.orelse:
        return None
    local = dict(consts)
    for prefix in statements[:-1]:
        if not (isinstance(prefix, ast.Assign) and len(prefix.targets) == 1
                and isinstance(prefix.targets[0], ast.Name)):
            return None
        value = literal_arithmetic(prefix.value, local)
        if value is None:
            value = literal_sequence(prefix.value, local)
        if value is None:
            return None
        local[prefix.targets[0].id] = value
    if not _pure_predicate(statement.test, set(spec.idx_vars), set(spec.params), local):
        return None
    inner = [s for s in statement.body
             if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if not inner or any(isinstance(item, ast.If) for item in inner):
        return None
    # 常量派生的局部名（cx/cy/r）折叠成字面量后写进掩码文本；前缀赋值随之丢弃，
    # 掩码里只剩轴名与常量，交由既有掩码通道发射。
    class Fold(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in local and not isinstance(local[node.id], tuple):
                return ast.copy_location(ast.Constant(value=local[node.id]), node)
            return node

    folded = ast.fix_missing_locations(Fold().visit(statement.test))
    # 20261025-159：常量展开维的下界覆盖**全部非并行轴下标**（含循环外的常量下标），
    # 无法推定即整结构拒绝（CHK-484）。
    const_dims = const_dim_bounds(inner, local, set(spec.idx_vars))
    if const_dims is None:
        return None
    return inner, ast.unparse(folded), tuple(const_dims)


def const_dim_bounds(statements, consts=None, axes=()):
    """常量展开首维的下界；**作用域感知 + 默认拒绝、白名单分类**（CHK-487 / CHK-490）。

    覆盖**全部**下标形态：Tuple slice 取首元素，单值 slice（`x[i]` / `x[3]`）本身即第 0 维；
    张量名必须是 `Name`。首维判定（其余一律 `return None`）：
    - 循环变量只认**词法包围**该下标的 `for v in range(N)` 绑定（按作用域链由内向外取最近绑定，
      同名遮蔽按内层）；要求 `N >= 1`（空/负循环界拒绝）；`v` ⇒ 界 `N`；
      `m - v` 须证明完整区间 `[m-(N-1), m]` 两端非负 ⇒ 界 `max(N, m + 1)`；
    - 首维是并行轴名（`axes` 内、且未被当前作用域的循环变量遮蔽）⇒ 真实轴，不推界；
    - 其余（循环外常量下标、`v + c`、轴相关算术、循环界不可解析、下标引用未包围它的循环变量、
      嵌套/切片/非 Name 张量）⇒ 返回 None，整结构拒绝。

    每个张量的界取**所有相关下标推出的最大值**；`()` 只在确实没有常量展开首维时出现。
    第 ≥1 维的常量下标**不在本函数承诺内**（见 SPEC-161 §2）。
    界全部来自**源码显式常量**，不依赖运行尺寸；返回 `((张量, 0, 界), …)`。
    """
    from frontend import literal_arithmetic

    consts = dict(consts or {})
    axes = set(axes)

    def _const_value(node):
        value = literal_arithmetic(node, consts)
        if value is None and isinstance(node, ast.Constant) and type(node.value) is int:
            value = node.value
        return value if type(value) is int else None

    def _loop_target(node):
        target = node.target
        if isinstance(target, ast.Tuple) and len(target.elts) == 1:
            target = target.elts[0]
        return target.id if isinstance(target, ast.Name) else None

    def _range_bound(node):
        call = node.iter
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == 'range' and len(call.args) == 1):
            return _const_value(call.args[0])
        return None

    bounds, failure = {}, []

    def _index_extent(first, scope):
        if isinstance(first, ast.Name) and first.id in scope:
            bound = scope[first.id]         # None ⇒ 循环界不可解析 ⇒ 拒绝
            # 空/负循环界不产生合法下标区间（`range(0)` / `range(-3)` 一律拒绝）。
            return bound if bound is not None and bound >= 1 else None
        if (isinstance(first, ast.BinOp) and isinstance(first.op, ast.Sub)
                and isinstance(first.right, ast.Name) and first.right.id in scope):
            left = _const_value(first.left)
            bound = scope[first.right.id]
            if left is None or bound is None or bound < 1:
                return None
            # `m - v`（v ∈ 0..N-1）的完整区间是 [m-(N-1), m]：两端都必须非负，
            # 否则会发射负下标（源端尾部索引语义本通道不承诺）⇒ 整结构拒绝。
            if left - (bound - 1) < 0 or left < 0:
                return None
            return max(bound, left + 1)
        if isinstance(first, ast.Name) and first.id in axes:
            return 'axis'
        return None

    def _scan(node, scope):
        if failure:
            return
        if isinstance(node, ast.For):
            _scan(node.iter, scope)
            var = _loop_target(node)
            inner = dict(scope)
            if var is not None:
                inner[var] = _range_bound(node)
            for child in list(node.body) + list(node.orelse):
                _scan(child, inner)
            return
        if isinstance(node, ast.Subscript):
            # 封闭语义：张量必须是裸名；切片/属性/调用等形态一律拒绝（不“跳过即接受”）。
            if not isinstance(node.value, ast.Name):
                failure.append('tensor-not-name')
                return
            if isinstance(node.slice, ast.Tuple):
                if not node.slice.elts:
                    failure.append('empty-slice')
                    return
                first = node.slice.elts[0]
            else:
                # 单值 slice（rank-1 的 `x[i]` / `x[3]`）：该值就是第 0 维下标。
                first = node.slice
            extent = _index_extent(first, scope)
            if extent is None:
                failure.append('unresolved-index')
                return
            if extent != 'axis':
                key = (node.value.id, 0)
                bounds[key] = max(bounds.get(key, 0), extent)
        for child in ast.iter_child_nodes(node):
            _scan(child, scope)

    for statement in statements:
        _scan(statement, {})
    if failure:
        return None
    return tuple((name, dim, value) for (name, dim), value in sorted(bounds.items()))


def bind_axis_equality(spec, consts):
    """S2：把 guard 里的 `轴 == 常量` 变成常量下标绑定。

    返回 `(guard_text_without_term, axis_name, constant)` 或 None；
    调用方负责把 body 中该轴名替换成常量。
    """
    if not spec.guard:
        return None
    node = ast.parse(spec.guard, mode='eval').body
    terms = node.values if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) else [node]
    bound, remaining = None, []
    for term in terms:
        if isinstance(term, ast.Compare) and len(term.ops) == 1 \
                and isinstance(term.ops[0], ast.Eq) \
                and isinstance(term.left, ast.Name) and term.left.id in spec.idx_vars:
            value = _is_int_const(term.comparators[0], consts)
            if value is None or bound is not None:
                return None
            bound = (term.left.id, value)
            continue
        remaining.append(term)
    if bound is None:
        return None
    # 其余项必须都是 `轴 < 上界`，且不再引用被绑定的轴。
    for term in remaining:
        if not (isinstance(term, ast.Compare) and len(term.ops) == 1
                and isinstance(term.ops[0], ast.Lt)
                and isinstance(term.left, ast.Name)
                and isinstance(term.comparators[0], ast.Name)):
            return None
        if bound[0] in {n.id for n in ast.walk(term) if isinstance(n, ast.Name)}:
            return None
    guard_text = ' and '.join(ast.unparse(term) for term in remaining)
    return guard_text, bound[0], bound[1]


def substitute_axis(body, axis, value):
    """把 body 中的轴名整体替换成常量（该轴在本形态下恒等于该值）。"""
    class Subst(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == axis and isinstance(node.ctx, ast.Load):
                return ast.copy_location(ast.Constant(value=value), node)
            return node

    replaced = []
    for statement in body:
        replaced.append(Subst().visit(statement))
    # 被消去轴必须在**所有**使用处都已成为常量；任何残留（含 Store 重绑定）→ 整结构拒绝。
    for statement in replaced:
        if _mentions(statement, axis):
            return None
    return replaced


def _mentions(node, name):
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))
