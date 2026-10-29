# -*- coding: utf-8 -*-
"""边界控制流规范化；仅处理完整计算前的空return和无else包围。"""
import ast
import copy
from semantic import Unsupported, shape_axis


def boolean_terms(node, operator):
    """按源码顺序展开同类布尔连接；不分配、不交换、不消去条件。"""
    if isinstance(node, ast.BoolOp) and isinstance(node.op, operator):
        return [term for value in node.values for term in boolean_terms(value, operator)]
    return [node]



_NEGATE = {ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.LtE: ast.Gt,
           ast.Gt: ast.LtE, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}


def demorgan_not(node):
    """把 `not (A and B ...)` 展开为 `not A or not B ...`，并折叠单比较的取反。

    helper 内联后的 `if not in_bounds(...): return` 就是这种形态；
    guards 的后续规则只处理 `轴 >= shape` 的合取/析取项。
    """
    if not (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)):
        return node
    inner = node.operand
    if isinstance(inner, ast.BoolOp):
        values = [demorgan_not(ast.UnaryOp(op=ast.Not(), operand=value)) for value in inner.values]
        op = ast.Or() if isinstance(inner.op, ast.And) else ast.And()
        return ast.fix_missing_locations(ast.copy_location(ast.BoolOp(op=op, values=values), node))
    if isinstance(inner, ast.Compare) and len(inner.ops) == 1 and type(inner.ops[0]) in _NEGATE:
        negated = ast.Compare(left=inner.left, ops=[_NEGATE[type(inner.ops[0])]()],
                              comparators=[inner.comparators[0]])
        return ast.fix_missing_locations(ast.copy_location(negated, node))
    return node


def strict_early_on_source(body):
    """在**原始** body（grid 绑定已弹出、尚未剥壳/解析别名）上判定本批严格早退形态。

    只看原始 AST，不从后续“消化后的计数”推断邻接关系：
    - body 第一条是 `if <比较>: return`（空返回、无 else、body 恰一条 Return）；
    - 全函数**唯一**一个 Return（就是它）；
    - 取反（De Morgan）后是**单个**比较（不含 bool 连接）。
    """
    if not body:
        return False
    first = body[0]
    if not (isinstance(first, ast.If) and not first.orelse
            and len(first.body) == 1 and isinstance(first.body[0], ast.Return)
            and first.body[0].value is None):
        return False
    returns = [node for item in body for node in ast.walk(item)
               if isinstance(node, ast.Return)]
    if len(returns) != 1:
        return False
    test = demorgan_not(copy.deepcopy(first.test))
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return False
    return True


def grid_nonnegative_truth(node, axes):
    """仅证明grid轴与整数0的非负/负比较；其他条件返回未知。"""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    left, right, op = node.left, node.comparators[0], node.ops[0]
    if isinstance(left, ast.Constant) and type(left.value) is int and left.value == 0:
        left, right = right, left
        reverse = {ast.LtE: ast.GtE, ast.Gt: ast.Lt}
        if type(op) not in reverse:
            return None
        op = reverse[type(op)]()
    if not isinstance(left, ast.Name) or left.id not in axes \
            or not isinstance(right, ast.Constant) or type(right.value) is not int \
            or right.value != 0:
        return None
    if isinstance(op, ast.GtE):
        return True
    if isinstance(op, ast.Lt):
        return False
    return None


def extract_guard(body, params=(), axes=()):
    """返回 `(guard 文本, 剩余语句, 是否属于严格早退)`。

    第三个返回值只在以下条件同时成立时为 True：**原始 body** 的第一条就是
    `if <比较>: return`（空返回、无 else、body 恰一条 Return；不以剥壳后的计数代替邻接判断）、
    全函数只出现这一次、取反（De Morgan）后恰有一项比较、后续语句中不再有 return。
    它只用于开放"补齐未覆盖写回轴"这一**新增**资格（SPEC-164 §2）；手写 `if 轴 < 界:`、
    多 return、`or` 多条件等既有形态仍按 162 原逻辑处理，不因此被禁。
    """
    conditions, prefix, aliases = [], [], {}
    source_strict_early = strict_early_on_source(body)
    removed_lower_bound = False
    returns_consumed = 0
    prefix_used = False
    multi_term_return = False
    body = list(body)

    class Resolve(ast.NodeTransformer):
        def visit_Name(self, node):
            return copy.deepcopy(aliases[node.id]) if node.id in aliases else node

        def visit_Compare(self, node):
            node = self.generic_visit(node)
            if len(node.ops) == 2:
                lower, axis, upper = node.left, *node.comparators
                shape = shape_axis(upper)
                if isinstance(node.ops[0], ast.LtE) and isinstance(node.ops[1], ast.Lt) \
                        and isinstance(lower, ast.Constant) and type(lower.value) is int \
                        and lower.value == 0 and isinstance(axis, ast.Name) and axis.id in axes \
                        and shape is not None and shape[0] in params:
                    # grid坐标非负；只消去这一确切链式边界的冗余下界。
                    return ast.copy_location(ast.Compare(
                        left=axis, ops=[ast.Lt()], comparators=[upper]), node)
            if len(node.ops) != 1:
                return node
            shape = shape_axis(node.left)
            right = node.comparators[0]
            inverse = {ast.Gt: ast.Lt, ast.GtE: ast.LtE,
                       ast.Lt: ast.Gt, ast.LtE: ast.GtE}
            if shape is not None and shape[0] in params \
                    and isinstance(right, ast.Name) and right.id in axes \
                    and type(node.ops[0]) in inverse:
                # 仅交换无副作用的shape与grid轴；合法边界仍由后续层证明。
                return ast.copy_location(ast.Compare(
                    left=right, ops=[inverse[type(node.ops[0])]()],
                    comparators=[node.left]), node)
            return node

    while body:
        first = body[0]
        if isinstance(first, ast.Assign) and len(first.targets) == 1 \
                and isinstance(first.targets[0], ast.Name):
            value = Resolve().visit(copy.deepcopy(first.value))
            shape = shape_axis(value)
            if shape is not None and shape[0] in params:
                name = first.targets[0].id
                if name in set(params) | set(axes) | set(aliases):
                    raise Unsupported('边界尺寸绑定不能覆盖参数、索引或已有定义: ' + name)
                aliases[name] = value
                prefix_used = True
                prefix.append(first)  # 保留原绑定，由计算捕获层继续校验，不丢弃。
                body = body[1:]
                continue
        if not isinstance(first, ast.If) or first.orelse:
            break
        first = copy.deepcopy(first)
        first.test = demorgan_not(Resolve().visit(first.test))
        if len(first.body) == 1 and isinstance(first.body[0], ast.Return) \
                and first.body[0].value is None:
            # De Morgan: if i>=M or j>=N: return -> i<M and j<N。
            returns_consumed += 1
            tests = boolean_terms(first.test, ast.Or)
            if len(tests) != 1:
                multi_term_return = True
            for test in tests:
                if grid_nonnegative_truth(test, axes) is False:
                    removed_lower_bound = True
                    continue
                if not isinstance(test, ast.Compare) or len(test.ops) != 1 \
                        or not isinstance(test.ops[0], ast.GtE):
                    raise Unsupported('提前返回仅支持轴>=完整shape，多个条件以or连接')
                conditions.append(ast.Compare(left=test.left,ops=[ast.Lt()],comparators=test.comparators))
            body = body[1:]
        elif len(body) == 1:
            tests = boolean_terms(first.test, ast.And)
            for test in tests:
                if grid_nonnegative_truth(test, axes) is True:
                    removed_lower_bound = True
                else:
                    conditions.append(test)
            body = first.body
        else:
            break
    # 严格早退：本批（20261026-164）只在"grid 绑定后紧邻、全函数唯一、取反后恰一项比较、
    # 且此前没有 shape 别名前缀、后续不再有 return"时成立；这只是**新增补齐资格**的开关，
    # 不改变 162 既有完整 guard / 多 return / `or` 形态（它们不经过补齐分支）。
    trailing_return = any(isinstance(node, ast.Return)
                          for item in body for node in ast.walk(item))
    # 原始形态判定是权威来源；消化后的计数/前缀只作交叉校验（合取，不单独成立）。
    strict_early = (source_strict_early and returns_consumed == 1
                    and not prefix_used and not multi_term_return and not trailing_return)
    if not conditions:
        if removed_lower_bound:
            raise Unsupported('非负下界不能替代完整输出上界guard')
        return '', prefix + body, strict_early
    guard = conditions[0] if len(conditions) == 1 else ast.BoolOp(op=ast.And(),values=conditions)
    return ast.unparse(guard), prefix + body, strict_early
