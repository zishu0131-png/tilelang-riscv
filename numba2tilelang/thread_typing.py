# -*- coding: utf-8 -*-
"""只查询Numba白名单标量类型；不编译或执行源函数。"""
import operator
from functools import lru_cache
from semantic import Unsupported
from thread_ir import DTYPES


OPERATORS = {'+': operator.add, '-': operator.sub, '*': operator.mul,
             '/': operator.truediv, '//': operator.floordiv, '%': operator.mod,
             '**': operator.pow,
             '==': operator.eq, '!=': operator.ne,
             '<': operator.lt, '<=': operator.le, '>': operator.gt,
             '>=': operator.ge, 'neg': operator.neg, 'pos': operator.pos}


@lru_cache(maxsize=1)
def context():
    from numba.cuda.target import CUDATypingContext
    result = CUDATypingContext()
    result.refresh()
    return result


def nbtype(dtype):
    from numba import types
    return types.boolean if dtype == 'bool' else getattr(types, dtype)


def dtype(typ):
    from numba import types
    name = str(types.unliteral(typ))
    if name not in DTYPES:
        raise Unsupported('线程路径不能表达Numba类型: ' + name)
    return name


def literal_type(value):
    if type(value) not in (bool, int, float):
        raise Unsupported('只支持bool/int/float常量')
    if type(value) is int and not -(1 << 63) <= value < (1 << 63):
        raise Unsupported('整数常量超出int64范围')
    return dtype(context().resolve_value_type_prefer_literal(value))


@lru_cache(maxsize=256)
def operation(symbol, operands):
    signature = context().resolve_function_type(
        OPERATORS[symbol], tuple(nbtype(x) for x in operands), {})
    if signature is None:
        raise Unsupported('Numba不支持运算类型: ' + repr((symbol, operands)))
    return tuple(dtype(x) for x in signature.args), dtype(signature.return_type)


@lru_cache(maxsize=128)
def math_type(name, operands):
    """调用方已校验math白名单；返回Numba签名参数与结果类型。"""
    import math
    signature = context().resolve_function_type(
        getattr(math, name), tuple(nbtype(x) for x in operands), {})
    if signature is None:
        raise Unsupported('Numba不支持数学函数类型: ' + repr((name, operands)))
    return tuple(dtype(x) for x in signature.args), dtype(signature.return_type)


@lru_cache(maxsize=64)
def power_type(base, exponent):
    """常量整数指数的幂返回类型。

    用IntegerLiteral表达已知的常量指数；这不保证选中某个特定lowering。
    int32/bool底数的结果为int64，普通整数指数查询也会得到该返回类型。
    调用方即使处理**1也必须应用返回类型，不能直接返回未提升的底数。
    """
    from numba import types
    signature = context().resolve_function_type(
        OPERATORS['**'], (nbtype(base), types.IntegerLiteral(exponent)), {})
    if signature is None:
        raise Unsupported('Numba不支持幂类型: ' + repr((base, exponent)))
    return dtype(signature.return_type)


@lru_cache(maxsize=32)
def range_type(operands):
    signature = context().resolve_function_type(range, tuple(nbtype(x) for x in operands), {})
    if signature is None:
        raise Unsupported('Numba不支持range参数类型: ' + repr(operands))
    return dtype(signature.return_type.iterator_type.yield_type)


@lru_cache(maxsize=2)
def atomic_exchange_type(value_dtype):
    """查询完整CUDA调用签名；模板返回的目标类型仍须允许实参转换。"""
    from numba import cuda, types
    if value_dtype not in ('int32', 'int64'):
        raise Unsupported('原子写子集: value必须为int32或int64')
    ctx = context()
    args = (types.Array(types.int32, 1, 'C'), types.intp, nbtype(value_dtype))
    signature = ctx.resolve_function_type(cuda.atomic.exch, args, {})
    if signature is None or signature.args != (args[0], types.intp, types.int32) \
            or signature.return_type != types.int32 \
            or any(ctx.can_convert(a, b) is None for a, b in zip(args, signature.args)):
        raise Unsupported('原子写子集: Numba不支持该exch参数转换')
    return dtype(signature.args[2])


@lru_cache(maxsize=2)
def atomic_add_type(index_dtype):
    """受限float64原子计数：核实完整CUDA签名及实参可转换性。"""
    from numba import cuda, types
    if index_dtype not in ('int32', 'int64'):
        raise Unsupported('原子写子集: add下标必须为int32或int64')
    ctx = context()
    args = (types.Array(types.float64, 1, 'C'), nbtype(index_dtype), types.int64)
    signature = ctx.resolve_function_type(cuda.atomic.add, args, {})
    if signature is None or signature.args != (args[0], types.intp, types.float64) \
            or signature.return_type != types.float64 \
            or any(ctx.can_convert(a, b) is None for a, b in zip(args, signature.args)):
        raise Unsupported('原子写子集: Numba不支持该add参数转换')
    return dtype(signature.args[2])


def unify(left, right):
    result = context().unify_pairs(nbtype(left), nbtype(right))
    if result is None:
        raise Unsupported('局部变量类型不能合流: ' + left + '/' + right)
    return dtype(result)
