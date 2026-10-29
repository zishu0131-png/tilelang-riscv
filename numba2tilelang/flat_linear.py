# -*- coding: utf-8 -*-
"""rank-1 张量的标准行主序展平寻址（20261025-153 引入；154 收口资格与 RHS 语义）。

识别的源形态（严格）：
    x[, y[, z]] = cuda.grid(2|3)
    if x < W and y < H [and z < D]:      # 全部严格 `<`，上界是标量形参
        idx = <标准行主序式>
        C[idx] = <受限表达式>

标准行主序式：`x + W*y`（2D）、`x + W*y + W*H*z`（3D）；系数即“该轴之前各轴的 extent”。

**154 收口点**（153 的缺口，均可在公共入口复现）：
1. 逐个 extent 正值 + 乘积**乘前**溢出界检查，均在任何坐标除模与 launch 之前；
2. RHS 不再由裸 AST 直接拼文本：改**带类型的受限捕获**——只允许同 idx 的 1D 张量读取、
   标量形参、模块常量、数值字面量、一元负号、`+ - * / // % **`、以及 `math./np.` 的**单参无关键字**
   白名单函数；每个操作数按既有 `numeric_dtype` 规则提升并显式 cast；
3. 输出张量自读（`C[idx] = C[idx] + …`）**拒绝**（本模式不实现 caller-output）；
4. RHS 里的 `idx` 与各轴名按**绑定含义**发射（idx→下标文本、轴→坐标文本，dtype int64）；
   `idx` 目标名与轴/extent/张量/标量形参同名时拒绝。

写域：坐标由 `_tl_p ∈ [0, W*H[*D])` 分解，映射为双射 ⇒ 完整写域、无重叠写。
输出物理 rank 仍为 1；长度由 `shape=(len,)` 给出，kernel 内断言 `len == W*H[*D]`，
同一 `_tl_shape` 供所有参与张量使用 ⇒ 断言覆盖输入与输出。
"""
import ast

from semantic import MATH_FUNCTIONS, Unsupported, numeric_dtype

INT64_MAX = (1 << 63) - 1


class _Mismatch(Exception):
    pass


class _Typed:
    """只带 dtype 的占位，用于复用既有 `numeric_dtype` 提升规则。"""

    def __init__(self, dtype):
        self.dtype = dtype


def _flatten_sum(node, parts):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        _flatten_sum(node.left, parts)
        _flatten_sum(node.right, parts)
    else:
        parts.append(node)


def _product_names(node, names):
    if isinstance(node, ast.Name):
        names.append(node.id)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        _product_names(node.left, names)
        _product_names(node.right, names)
        return
    raise _Mismatch()


def _parse_linear(expr, axes, lengths):
    """把下标表达式解析为“每轴系数名集合 == 该轴之前各轴 extent 集合”；否则抛 _Mismatch。"""
    parts = []
    _flatten_sum(expr, parts)
    strides = {}
    accumulated = []
    for axis in axes:
        strides[axis] = tuple(sorted(accumulated))
        accumulated.append(lengths[axis])
    found = {}
    for part in parts:
        if isinstance(part, ast.Name):
            axis, names = part.id, ()
        elif isinstance(part, ast.BinOp) and isinstance(part.op, ast.Mult):
            flat = []
            _product_names(part, flat)
            axis_candidates = [name for name in flat if name in axes]
            if len(axis_candidates) != 1:
                raise _Mismatch()
            axis = axis_candidates[0]
            names = tuple(sorted(name for name in flat if name != axis))
        else:
            raise _Mismatch()
        if axis not in axes or axis in found:
            raise _Mismatch()
        found[axis] = names
    if set(found) != set(axes):
        raise _Mismatch()
    for axis in axes:
        if found[axis] != strides[axis]:
            raise _Mismatch()
    return found


def recognize(body, spec):
    """识别展平寻址形态；成功返回描述字典，否则返回 None（交回旧路径，不抛异常）。"""
    try:
        return _recognize(body, spec)
    except _Mismatch:
        return None
    # 明确不支持的形态（输出自读等）向上抛，保留可读的拒绝原因，而不是回落成笼统错误。


def _recognize(body, spec):
    if len(spec.idx_vars) not in (2, 3):
        return None
    if not spec.guard:
        return None
    tree = ast.parse(spec.guard, mode='eval').body
    terms = tree.values if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And) else [tree]
    if len(terms) != len(spec.idx_vars):
        return None
    lengths = {}
    for term in terms:
        if not (isinstance(term, ast.Compare) and len(term.ops) == 1
                and isinstance(term.ops[0], ast.Lt)
                and isinstance(term.left, ast.Name)
                and isinstance(term.comparators[0], ast.Name)):
            return None
        left, right = term.left.id, term.comparators[0].id
        if left not in spec.idx_vars or right not in spec.scalar_params:
            return None
        lengths[left] = right
    if set(lengths) != set(spec.idx_vars):
        return None
    statements = [s for s in body
                  if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(statements) != 2:
        return None
    first, second = statements
    if not (isinstance(first, ast.Assign) and len(first.targets) == 1
            and isinstance(first.targets[0], ast.Name)
            and isinstance(second, ast.Assign) and len(second.targets) == 1
            and isinstance(second.targets[0], ast.Subscript)):
        return None
    idx_name = first.targets[0].id
    axes = tuple(spec.idx_vars)
    # idx 目标名不得与轴/extent/张量/标量形参同名（否则 RHS 里的同名引用含义不明）
    reserved = set(axes) | set(lengths.values()) | set(spec.params)
    if idx_name in reserved:
        return None
    _parse_linear(first.value, axes, lengths)
    store = second.targets[0]
    if not (isinstance(store.value, ast.Name) and isinstance(store.slice, ast.Name)
            and store.slice.id == idx_name):
        return None
    output = store.value.id
    # 20261025-155：RHS 收窄到两真源所需形态——**同一 idx 的 1D f32 读取的直接二元加法**。
    # 不在此处重建通用表达式类型系统：其它 RHS（字面量混合、除法、pow、math 调用、
    # idx/轴名参与、嵌套运算）一律拒绝，留给后续独立切片。
    expression = second.value
    if not (isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add)):
        raise Unsupported('展平寻址窄门：RHS 只支持 `T[idx] + U[idx]` 的直接二元加法')
    operands = []
    for side in (expression.left, expression.right):
        if not (isinstance(side, ast.Subscript)
                and isinstance(side.value, ast.Name)
                and isinstance(side.slice, ast.Name)
                and side.slice.id == idx_name):
            raise Unsupported('展平寻址窄门：RHS 两个操作数都必须是同一 idx 的 1D 张量读取')
        name = side.value.id
        if name == output:
            raise Unsupported('展平寻址窄门：输出张量自读（in-place）不支持，'
                              '需走其它路径或改写源')
        operands.append(name)
    dtypes = dict(getattr(spec, 'tensor_dtypes', {}) or {})
    for name in operands + [output]:
        if dtypes.get(name, 'float32') != 'float32':
            raise Unsupported('展平寻址窄门：本片只支持 float32 张量（%s=%s）'
                              % (name, dtypes.get(name, 'float32')))
    return {'output': output, 'index_name': idx_name, 'index': first.value,
            'axes': axes, 'lengths': lengths, 'tensors': sorted(set(operands)),
            'left': operands[0], 'right': operands[1], 'expression': expression}


def lower(spec):
    """发射 `T.Kernel + T.Parallel`；先逐个 extent 正值、再逐步受检乘积，最后才使用 `_tl_size`。"""
    from lowering import flat_indices
    info = spec.flat_linear
    axes, lengths = info['axes'], info['lengths']
    length_names = [lengths[axis] for axis in axes]
    coords = dict(zip(axes, flat_indices(length_names)))
    index_text = ast.unparse(info['index'])
    for axis in axes:
        index_text = _replace_name(index_text, axis, coords[axis])
    dtypes = dict(getattr(spec, 'tensor_dtypes', {}) or {})
    output = info['output']
    store_dtype = dtypes.get(output, 'float32')
    left, right = info['left'], info['right']
    # 窄切片：只有“同 idx 的 f32 读取直接相加”，两侧都显式 cast，结果再 cast 到存储 dtype。
    rhs_text = 'T.cast((T.cast(%s[%s], "%s") + T.cast(%s[%s], "%s")), "%s")' % (
        left, index_text, dtypes.get(left, 'float32'),
        right, index_text, dtypes.get(right, 'float32'), store_dtype)
    scalar_dtypes = dict(getattr(spec, 'scalar_params', {}) or {})
    params = ', '.join(
        '%s: T.%s' % (name, scalar_dtypes[name]) if name in scalar_dtypes else
        '%s: T.Tensor((_tl_shape[0],), "%s")' % (name, dtypes.get(name, 'float32'))
        for name in spec.params)
    # 断言顺序：正值 → 逐步乘前界（乘积只在通过后出现）→ 才赋值 `_tl_size` → 再比 shape。
    checks = ['        T.Assert(%s > 0, "extent must be positive")' % name
              for name in length_names]
    running = length_names[0]
    for position in range(1, len(length_names)):
        checks.append(
            '        T.Assert(%s <= %d // (%s), "flattened size overflows int64")'
            % (length_names[position], INT64_MAX, running))
        running = '(%s * %s)' % (running, length_names[position])
    size_expr = ' * '.join(length_names)
    return f'''# 展平寻址：rank-1 张量按标准行主序 `{index_text}` 访问；extent 正值/乘前界/长度断言均在 launch 前。
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[{spec.params.index(output)}],
              pass_configs={{"tl.disable_warp_specialized": True}}, compile_flags=["--fmad=false"])
def {spec.name}(shape, block=128):
    _tl_shape = tuple(shape)
    if len(_tl_shape) != 1 or any(type(d) is not int or d <= 0 for d in _tl_shape):
        raise ValueError("shape must contain 1 positive integer")
    if type(block) is not int or block <= 0 or block % 32:
        raise ValueError("block must be a positive multiple of 32")
    @T.prim_func
    def _tl_main({params}):
{chr(10).join(checks)}
        _tl_size = {size_expr}
        T.Assert(_tl_shape[0] == _tl_size,
                 "flattened length must equal the product of axis extents")
        with T.Kernel(T.max(T.ceildiv(_tl_size, block), 1), threads=128) as _tl_bid:
            for _tl_lane in T.Parallel(block):
                _tl_p = _tl_bid * block + _tl_lane
                if _tl_p < _tl_size:
                    {output}[{index_text}] = {rhs_text}
    return _tl_main
'''


def _replace_name(text, old, new):
    """按下标式文本替换轴名（整词匹配）。"""
    out, position = [], 0
    while True:
        found = text.find(old, position)
        if found < 0:
            out.append(text[position:])
            return ''.join(out)
        before = text[found - 1] if found else ''
        after = text[found + len(old)] if found + len(old) < len(text) else ''
        if (not before.isalnum() and before != '_') and (not after.isalnum() and after != '_'):
            out.append(text[position:found])
            out.append(new)
            position = found + len(old)
        else:
            out.append(text[position:found + len(old)])
            position = found + len(old)
