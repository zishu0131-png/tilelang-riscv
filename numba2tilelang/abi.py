# -*- coding: utf-8 -*-
"""B1：参数 ABI 的唯一来源。

三类 dtype 必须分开（审阅侧 2026-10-24 裁定）：
1) **参数 ABI dtype** —— 张量声明、标量形参 → `dtype_of` / `store_dtype`；
2) **表达式 dtype** —— 语义层已算好的 `Expr.dtype` → `result_dtype`，lowerer 不重新推导；
3) **累加器 dtype** —— 策略（pipeline 固定 float32、scoped 固定 float64、lower_sum 用 acc_dtype），
   **不属于本模块**，保持各路径现状。

`spec.param_abi` 由 `ensure_param_abi` 在捕获完成后一次性生成并缓存；
lowerer 只从这里取 ABI dtype，不再各自拼 “tensor_dtypes.get(name, 'float32')”。
"""


def ensure_param_abi(spec):
    cached = getattr(spec, 'param_abi', None)
    if cached:
        return cached
    scalars = dict(getattr(spec, 'scalar_params', {}) or {})
    tensors = dict(getattr(spec, 'tensor_dtypes', {}) or {})
    outputs = set()
    compute = getattr(spec, 'compute', None)
    if compute is not None:
        outputs.add(compute.output)
        outputs.update(row[0] for row in (compute.extra_stores or ()))
    program = getattr(spec, 'program', None)
    if program is not None:
        outputs.add(program.output)
    store_target = getattr(spec, 'store_target', None)
    rows = []
    for name in spec.params:
        kind = 'scalar' if name in scalars else 'tensor'
        if kind == 'scalar':
            dtype = scalars[name]
        else:
            dtype = tensors.get(name, 'float32')
        role = 'output' if name in outputs or name == store_target else 'input'
        rows.append({'name': name, 'kind': kind, 'dtype': dtype, 'role': role})
    spec.param_abi = tuple(rows)
    return spec.param_abi


def _row(spec, name):
    for row in ensure_param_abi(spec):
        if row['name'] == name:
            return row
    return None


def dtype_of(spec, name):
    """参数 ABI dtype：张量用声明 dtype，标量用签名/推断 dtype。"""
    row = _row(spec, name)
    if row is None:
        raise KeyError('unknown parameter: %s' % name)
    return row['dtype']


def is_scalar(spec, name):
    row = _row(spec, name)
    return bool(row and row['kind'] == 'scalar')


def param_dtype_text(spec, name):
    """prim_func 形参声明前的类型前缀：标量 `T.<dtype>`，张量由调用方拼 `T.Tensor`。"""
    return 'T.' + dtype_of(spec, name) if is_scalar(spec, name) else None


def store_dtype(spec, name=None):
    """写回 cast 的目标类型 = **输出张量的声明 dtype**（不是表达式/累加器 dtype）。"""
    return dtype_of(spec, name or spec.store_target)


def result_dtype(expression):
    """表达式 dtype 直接来自语义层，lowerer 不重新推导。"""
    return expression.dtype


def tensor_dtype_set(spec):
    """B2：参数 ABI 里张量的 dtype 集合（用于“全同 dtype”门禁与混合拒绝）。"""
    return {row['dtype'] for row in ensure_param_abi(spec) if row['kind'] == 'tensor'}


def expression_dtypes(spec):
    compute = getattr(spec, 'compute', None)
    expressions = []
    if compute is not None:
        expressions.append(compute.expression)
        expressions.extend(row[-1] for row in (compute.extra_stores or ()))
    program = getattr(spec, 'program', None)
    if program is not None:
        expressions.append(program.expression)
        expressions.extend(stage.expression for stage in program.stages)
    return [expr.dtype for expr in expressions if getattr(expr, 'dtype', None)]


def accumulator_dtype(spec, default):
    """B2：累加器 dtype 是**策略**，不是类型证据。

    strict 下：该路径涉及的任一**张量 ABI dtype** 为 float64 → float64；
    否则保持该路径现状（pipeline 传 'float32'、scoped 传 'float64'）。

    **不把表达式 dtype 纳入决策**：float32 张量 + Python float 常量会让表达式提升为
    float64，而既有 pipeline 的累加器策略是 float32（生成物注释：保留表达式类型提升）。
    若按表达式 dtype 升累加器，既有 float32 pipeline 产物会全部变化，违反本片
    “float32 路径逐字节不变”的验收条件；因此这里只按张量 ABI 决策，表达式 dtype
    仍按各路径原有方式参与计算（由 `emit_expr` 直接发射）。
    精度含义：float32 张量 + float64 表达式（Python float 常量提升）仍累加 float32，
    低于 Numba 的 float64 语义——这是既有策略，不是类型证据推出的结论。
    """
    # `fast` 与 strict 同规则：未实施 fast 的路径等同 strict（C2 规则），
    # 不能因 fast 把 float64 累加器静默降成 float32（那会与“按 strict 发射”的注释矛盾）。
    return 'float64' if 'float64' in tensor_dtype_set(spec) else default
