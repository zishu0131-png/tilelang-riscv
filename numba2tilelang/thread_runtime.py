# -*- coding: utf-8 -*-
"""线程模式调用边界；不复制输入、不代替输出分配内存。"""
import math
import numbers
import struct
from thread_ir import TensorArg


SIZES = {'bool': 1, 'uint8': 1, 'uint32': 4, 'int32': 4, 'int64': 8, 'float32': 4, 'float64': 8}


def launch_dimensions(value, label):
    dims = (value,) if type(value) is int else value
    if type(dims) is not tuple or not 1 <= len(dims) <= 3 \
            or any(type(x) is not int or x <= 0 for x in dims):
        raise ValueError(label + ' must be a positive int or rank 1-3 tuple')
    return dims + (1,) * (3 - len(dims))


def validate_launch(grid, block):
    grid = launch_dimensions(grid, 'grid')
    block = launch_dimensions(block, 'block')
    # CUDA计算能力2.x以后的架构上限；设备线程总数限制在实际调用时再检查。
    if any(x > limit for x, limit in zip(grid, (2147483647, 65535, 65535))) \
            or any(x > limit for x, limit in zip(block, (1024, 1024, 64))) \
            or math.prod(block) > 1024:
        raise ValueError('launch exceeds CUDA dimension limits')
    return grid, block


def scalar_value(value, dtype):
    if dtype == 'bool':
        if type(value) is not bool:
            raise TypeError('bool scalar requires a Python bool')
        return value
    if dtype in ('int32', 'int64'):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError('integer scalar requires an integer, not bool')
        value = int(value)
        bits = SIZES[dtype] * 8
        if not -(1 << (bits-1)) <= value < (1 << (bits-1)):
            raise ValueError('scalar outside ' + dtype + ' range')
        return value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError('floating scalar requires a real number, not a tensor or bool')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('non-finite scalar arguments are not supported yet')
    if dtype == 'float32':
        try:
            result = struct.unpack('f', struct.pack('f', result))[0]
        except OverflowError as exc:
            raise ValueError('scalar outside float32 range') from exc
    if not math.isfinite(result):
        raise ValueError('scalar outside ' + dtype + ' range')
    return result


def validate_arguments(parameters, inputs, tensor_type):
    if len(parameters) != len(inputs):
        raise ValueError('argument count does not match source signature')
    converted, ranges, device = [], [], None
    for (name, spec), value in zip(parameters, inputs):
        if not isinstance(spec, TensorArg):
            scalar = scalar_value(value, spec.dtype)
            if spec.value_range is not None:
                lo, hi = spec.value_range
                if not lo <= scalar <= hi:
                    raise ValueError(f'scalar value_range violation: {name} outside {(lo, hi)}')
            converted.append(scalar)
            continue
        if not isinstance(value, tensor_type):
            raise TypeError('expected a CUDA tensor: ' + name)
        if not value.is_cuda or not value.is_contiguous():
            raise ValueError('expected a contiguous CUDA tensor: ' + name)
        if tuple(value.shape) != spec.shape or str(value.dtype) != 'torch.' + spec.dtype:
            raise ValueError('tensor shape/dtype mismatch: ' + name)
        if getattr(value, 'requires_grad', False) or value.is_conj() or value.is_neg():
            raise ValueError('autograd/conjugate/negative views are not supported: ' + name)
        if device is not None and device != value.device:
            raise ValueError('all tensors must be on the same CUDA device')
        device = value.device
        start = value.data_ptr()
        stop = start + math.prod(spec.shape) * SIZES[spec.dtype]
        if any(start < other_stop and other_start < stop for other_start, other_stop in ranges):
            raise ValueError('tensor arguments overlap: ' + name)
        ranges.append((start, stop))
        converted.append(value)
    if device is None:
        raise ValueError('first thread prototype requires at least one tensor argument')
    return tuple(converted), device


def validate_value_ranges(parameters, inputs, cuda):
    """调用期间至kernel完成前，调用者不得从外部修改受约束元数据。

    输入生产须与当前stream有序；每次扫描并同步，不能缓存检查结果。
    """
    bounded = [(name, spec.value_range, value)
               for (name, spec), value in zip(parameters, inputs)
               if isinstance(spec, TensorArg) and spec.value_range is not None]
    if not bounded:
        return
    if cuda.is_current_stream_capturing():
        raise ValueError('value_range validation does not support CUDA Graph capture')
    for name, (lo, hi), value in bounded:
        if not ((value >= lo) & (value <= hi)).all().item():
            raise ValueError(f'tensor value_range violation: {name} outside {(lo, hi)}')


class ThreadRunner:
    def __init__(self, factory, parameters, grid, block):
        self.grid, self.block = validate_launch(grid, block)
        self.parameters = parameters
        self.factory = factory
        self.kernel = None
        self.device = None

    def __call__(self, *inputs):
        import torch
        converted, device = validate_arguments(self.parameters, inputs, torch.Tensor)
        properties = torch.cuda.get_device_properties(device)
        if math.prod(self.block) > properties.max_threads_per_block:
            raise ValueError('block exceeds device thread limit')
        if self.device is not None and self.device != device:
            raise ValueError('runner is bound to its first CUDA device; create another runner')
        with torch.cuda.device(device):
            validate_value_ranges(self.parameters, converted, torch.cuda)
            if self.kernel is None:
                self.kernel = self.factory(self.grid, self.block)
                self.device = device
            self.kernel(*converted)
        return None
