# -*- coding: utf-8 -*-
"""线程程序的参数契约和有序指令；不携带归约调度假设。"""
from dataclasses import dataclass, field


DTYPES = ('bool', 'uint8', 'uint32', 'int32', 'int64', 'float16', 'float32', 'float64')
# numba 的类型名到本契约的写法归一；只在构造时发生，不改调用方源码。
# `double` 是 numpy/numba 对 float64 的通用别名（真实语料大量使用 `void(double[:])`）；
# 只收录能确定性映射到 DTYPES 白名单内的别名，域外类型（int16/uint64/complex128 等）继续拒绝。
DTYPE_ALIASES = {'boolean': 'bool', 'double': 'float64'}
# 张量参数 rank 上限。放宽到 4 只因真实 CUDA 代码里 NHWC/四维缓冲常见；
# 它**只**约束参数契约，与 grid/block 维度无关（launch_dimensions 仍限 3 维，硬件亦然）。
# 4 维 Tensor 注解能否被 TileLang 编译**未经验证**——本项目所有生成物都未经编译验证，此处不例外。
MAX_TENSOR_RANK = 4


def check_value_range(dtype, bounds):
    """value_range 的形状/类型/范围校验；张量与标量共用同一契约。"""
    if dtype not in ('int32', 'int64') or type(bounds) is not tuple \
            or len(bounds) != 2 or any(type(x) is not int for x in bounds):
        raise ValueError('value_range requires an int32/int64 parameter and two integer endpoints')
    bits = 32 if dtype == 'int32' else 64
    if not -(1 << (bits-1)) <= bounds[0] <= bounds[1] < (1 << (bits-1)):
        raise ValueError('value_range must be ordered and fit the parameter dtype')


@dataclass(frozen=True)
class TensorArg:
    shape: tuple
    dtype: str
    value_range: tuple | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.dtype in DTYPE_ALIASES:
            object.__setattr__(self, 'dtype', DTYPE_ALIASES[self.dtype])
        if type(self.shape) is not tuple or not 1 <= len(self.shape) <= MAX_TENSOR_RANK \
                or any(type(d) is not int or d <= 0 for d in self.shape):
            raise ValueError('tensor shape must be a positive rank 1-%d tuple' % MAX_TENSOR_RANK)
        if self.dtype not in DTYPES:
            raise ValueError('unsupported tensor dtype: ' + str(self.dtype))
        if self.value_range is not None:
            check_value_range(self.dtype, self.value_range)


@dataclass(frozen=True)
class ScalarArg:
    """标量参数规格。`value_range` 与张量同义：调用时须满足该区间，供区间证明使用。"""

    dtype: str
    value_range: tuple | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.dtype in DTYPE_ALIASES:
            object.__setattr__(self, 'dtype', DTYPE_ALIASES[self.dtype])
        if self.dtype not in DTYPES:
            raise ValueError('unsupported scalar dtype: ' + str(self.dtype))
        if self.value_range is not None:
            check_value_range(self.dtype, self.value_range)


@dataclass(frozen=True)
class Value:
    name: str
    dtype: str


@dataclass(frozen=True)
class Instruction:
    op: str
    target: Value | None = None
    args: tuple = ()
    value: object = None
    line: int = 0


@dataclass(frozen=True)
class Branch:
    condition: Value
    yes: tuple
    no: tuple
    line: int = 0


@dataclass(frozen=True)
class SerialLoop:
    index: Value
    start: Value
    stop: Value
    body: tuple
    line: int = 0
    step: Value | None = None


@dataclass(frozen=True)
class WhileLoop:
    """有界计数while。tests是每次判断都要重新执行的指令，不是循环入口的快照。"""
    condition: Value
    tests: tuple
    body: tuple
    entry: tuple
    counter: str
    line: int = 0
    grid_stride: tuple = ()
    # halving while：计数器每轮整除 2（非负整数严格递减，终止性无需区间证明）。
    halving: bool = False


@dataclass(frozen=True)
class ThreadProgram:
    name: str
    parameters: tuple
    registers: tuple
    body: tuple
