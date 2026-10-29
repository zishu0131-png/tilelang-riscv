# -*- coding: utf-8 -*-
"""按源语句顺序捕获线程程序，所有读取在发生位置物化。"""
import ast
import copy
import math
import textwrap
from frontend import validate_imports, PROTECTED_NAMES
from semantic import Unsupported
from source_contract import is_cuda_jit
from thread_ir import (DTYPES, TensorArg, ScalarArg, Value, Instruction,
                       Branch, SerialLoop, WhileLoop, ThreadProgram)
from thread_typing import (literal_type, operation, power_type, unify, range_type,
                           math_type, atomic_exchange_type, atomic_add_type)
from thread_atomic import is_exchange, is_atomic_add, validate_atomic
from thread_device import collect_device_reads, capture_device_read


BINARY = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
          ast.FloorDiv: '//', ast.Mod: '%'}
# Numba对同一dtype有多种拼写；只映射到DTYPES能表达的，其余仍拒绝。
# 这些名字必须真的从numba导入才会被采用（见signature_contract的导入绑定检查）。
# 线程路径认得但旧optimized路径不认的math函数；值是要生成的TileLang原语名。
# 返回类型按Numba CUDA签名确定；整数数学输入在原语调用前转为float64。
THREAD_MATH = {'sin': 'sin', 'cos': 'cos', 'exp': 'exp', 'sqrt': 'sqrt', 'log': 'log',
               'atan2': 'atan2', 'tanh': 'tanh', 'fabs': 'abs', 'isinf': 'isinf',
               'pow': 'pow'}
MATH_ARITY = {'sin': 1, 'cos': 1, 'exp': 1, 'sqrt': 1, 'log': 1, 'tanh': 1,
              'fabs': 1, 'isinf': 1, 'atan2': 2, 'pow': 2}
# 幂只展开0–32的常量整数指数；上限保留当前受限契约。
MAX_POWER = 32
# math模块的常量按字面量折叠，和模块级标量走同一条路径。
MATH_CONSTANTS = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
# 内建调用：min/max/abs走比较与绝对值，int/float是纯类型转换。
BUILTIN_ARITY = {'min': 2, 'max': 2, 'abs': 1, 'int': 1, 'float': 1}
SIGNATURE_ALIASES = {'bool_': 'bool', 'boolean': 'bool', 'double': 'float64',
                     'float_': 'float64', 'single': 'float32', 'int_': 'int64',
                     'long': 'int64', 'intc': 'int32'}
COMPARE = {ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=',
           ast.Gt: '>', ast.GtE: '>='}


def fail(node, message):
    raise Unsupported(f'line {getattr(node, "lineno", 0)}: {message}')


def is_infinity_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'float' and not node.keywords and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and type(node.args[0].value) is str and node.args[0].value == 'inf')


def validate_infinity_binding(tree, fn, constants):
    if not any(is_infinity_call(node) for node in ast.walk(fn)):
        return
    if 'float' in constants:
        fail(fn, '无穷常量需要未被遮蔽的内建float')
    from thread_bindings import Bindings

    class FloatBinding(Bindings):
        def check(self, node, name):
            if name == 'float':
                fail(node, '无穷常量需要未被遮蔽的内建float')

    checker = FloatBinding({'float'})
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    scope = fn
    while True:
        if isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(scope, 'args'):
                checker.visit(scope.args)
            for node in scope.body:
                checker.visit(node)
        if scope is tree:
            break
        scope = parents[scope]


def dtype_name(node):
    """dtype 字面：裸类型名（`float64`）或模块属性（`numba.float64`/`np.float64`）。"""
    if isinstance(node, ast.Name):
        return node.id if node.id in DTYPES else None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id in ('numba', 'numpy', 'np') and node.attr in DTYPES:
        return node.attr
    return None


def parse_cuda_array(node, constants=None):
    """识别 `cuda.shared.array(shape=..., dtype=...)` / `cuda.local.array(...)`。

    返回 (kind, shape_tuple, dtype) 或 None。shape 必须是可静态求值的正整数元组
    （CUDA 要求共享/局部数组形状为编译期常量）；也接受模块级整型常量（如 `MAX_TPB`）。
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    func = node.func
    if not (isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name)
            and func.value.value.id == 'cuda' and func.value.attr in ('shared', 'local')
            and func.attr == 'array'):
        return None
    shape = dtype = None
    for kw in node.keywords:
        if kw.arg == 'shape':
            shape = kw.value
        elif kw.arg == 'dtype':
            dtype = kw.value
    positional = [a for a in node.args]
    if positional and shape is None:
        shape = positional.pop(0)
    if positional and dtype is None:
        dtype = positional.pop(0)
    if positional or shape is None or dtype is None:
        return None
    name = dtype_name(dtype)
    if name is None:
        return None
    try:
        dims = ast.literal_eval(shape)
    except (ValueError, SyntaxError):
        if isinstance(shape, ast.Name) and constants and shape.id in constants:
            dims = constants[shape.id]        # 模块级整型常量形状（如 MAX_TPB = 1024）
        elif isinstance(shape, (ast.Tuple, ast.List)) and constants:
            # 元组元素可以是字面量或已知整型常量（如 `(BLOCK_DIM, BLOCK_DIM)`）。
            values = []
            for item in shape.elts:
                if isinstance(item, ast.Constant) and type(item.value) is int:
                    values.append(item.value)
                elif isinstance(item, ast.Name) and type(constants.get(item.id)) is int:
                    values.append(constants[item.id])
                else:
                    return None
            dims = tuple(values)
        else:
            return None
    if type(dims) is int:                 # numba 接受整数形状，等价于 1 维元组
        dims = (dims,)
    if not isinstance(dims, tuple) or not 1 <= len(dims) <= 4 \
            or any(type(d) is not int or d <= 0 for d in dims):
        return None
    return (func.value.attr, dims, name)


def is_numpy_int32_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'np'
            and node.func.attr == 'int32')


# 以下两类 cuda 原语**明确拒绝但指名**：笼统报"未实现的线程语句/表达式"会把真实原因
# （需要整块线程协同或共享内存语义）藏起来，妨碍失败聚类。判定与 frontend 的同名检查保持一致。
CUDA_SYNC_PRIMITIVES = {'syncthreads', 'syncthreads_count', 'syncthreads_and', 'syncthreads_or',
                        'syncwarp', 'threadfence', 'threadfence_block', 'threadfence_system'}
CUDA_MEMORY_PRIMITIVES = {'shared', 'local'}


def is_cuda_sync_call(node):
    return (isinstance(node, ast.Call) and not node.keywords
            and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'cuda' and node.func.attr in CUDA_SYNC_PRIMITIVES)


def is_cuda_memory_call(node):
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == 'cuda'
            and node.func.value.attr in CUDA_MEMORY_PRIMITIVES)


def validate_numpy_int32_binding(tree, fn, constants):
    if not any(is_numpy_int32_call(node) for node in ast.walk(fn)):
        return
    imports = [node for node in tree.body if isinstance(node, ast.Import)
               and any(alias.name == 'numpy' and alias.asname == 'np' for alias in node.names)]
    if not imports or 'np' in constants:
        fail(fn, 'np.int32需要未被遮蔽的模块顶层import numpy as np')
    from thread_bindings import Bindings

    class NumpyBinding(Bindings):
        def check(self, node, name):
            if name == 'np':
                fail(node, 'np.int32需要未被遮蔽的模块顶层import numpy as np')

    checker = NumpyBinding({'np'}, imports)
    for node in tree.body:
        checker.visit(node)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    scope = fn
    while scope is not tree:
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            checker = NumpyBinding({'np'})
            if hasattr(scope, 'args'):
                checker.visit(scope.args)
            for node in scope.body:
                checker.visit(node)
        scope = parents[scope]


def bare_return(body):
    """整个suite只由pass与无值return组成，且至少有一个return。"""
    return all(isinstance(x, (ast.Pass,)) or (isinstance(x, ast.Return) and x.value is None)
               for x in body) and any(isinstance(x, ast.Return) for x in body)


def conjuncts(test):
    """把顶层and递归展开成合取项；嵌套的or保持原样，不作为终止依据。"""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        result = []
        for value in test.values:
            result.extend(conjuncts(value))
        return result
    return [test]


def signature_contract(tree, fn, parameters):
    if len(fn.decorator_list) != 1 or fn.returns is not None:
        fail(fn, '不支持额外装饰器或返回注解')
    decorator = fn.decorator_list[0]
    if not isinstance(decorator, ast.Call):
        return
    if len(decorator.args) > 1:
        fail(decorator, '只支持单一静态签名')
    seen = set()
    for kw in decorator.keywords:
        if kw.arg in seen or kw.arg not in ('device', 'fastmath') \
                or not isinstance(kw.value, ast.Constant) or type(kw.value.value) is not bool:
            fail(kw, '装饰器选项仅支持device/fastmath的布尔常量')
        # device=True是辅助函数而不是kernel入口；fastmath只放宽精度承诺，
        # 生成侧仍用严格浮点（--fmad=false），所以接受它只会让结果更接近IEEE。
        if kw.arg == 'device' and kw.value.value is not False:
            fail(kw, 'device=True的函数是device辅助函数，不是kernel入口')
        seen.add(kw.arg)
    if decorator.args:
        sig = decorator.args[0]
        if isinstance(sig, ast.Constant) and type(sig.value) is str:
            try:
                sig = ast.parse(sig.value, mode='eval').body
            except SyntaxError:
                fail(sig, '无法解析静态签名')
        else:
            needed = {n.id for n in ast.walk(sig) if isinstance(n, ast.Name)}
            bindings = {name: [] for name in needed}
            for stmt in tree.body:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    for alias in stmt.names:
                        name = alias.asname or alias.name.split('.')[0]
                        if name in bindings:
                            bindings[name].append(isinstance(stmt, ast.ImportFrom)
                                and stmt.level == 0 and stmt.module == 'numba'
                                and alias.name == name)
            if any(not entries or not all(entries) for entries in bindings.values()):
                fail(sig, '签名类型必须明确导入numba且不能被覆盖')
        if isinstance(sig, ast.Call) and isinstance(sig.func, ast.Name) \
                and sig.func.id == 'void' and not sig.keywords:
            args = sig.args
        elif isinstance(sig, ast.Tuple):
            args = sig.elts
        else:
            fail(sig, '需要void签名或参数类型元组')
        if len(args) != len(parameters):
            fail(sig, '签名形参数量不匹配')
        for (name, parameter), argument in zip(parameters.items(), args):
            base = argument.value if isinstance(argument, ast.Subscript) else argument
            typename = base.id if isinstance(base, ast.Name) else None
            typename = SIGNATURE_ALIASES.get(typename, typename)
            if typename not in DTYPES or typename != parameter.dtype:
                fail(argument, '签名dtype与参数规格不符: ' + name)
            if isinstance(argument, ast.Subscript):
                dims = argument.slice.elts if isinstance(argument.slice, ast.Tuple) else [argument.slice]
                if not isinstance(parameter, TensorArg) or len(dims) != len(parameter.shape):
                    fail(argument, '签名rank与参数规格不符: ' + name)
                for i, dim in enumerate(dims):
                    if not isinstance(dim, ast.Slice) or dim.lower is not None or dim.upper is not None:
                        fail(dim, '签名维度需要 : 或末维 ::1')
                    if dim.step is not None and (i != len(dims)-1 or not isinstance(dim.step, ast.Constant)
                            or type(dim.step.value) is not int or dim.step.value != 1):
                        fail(dim, '第一版仅支持C连续输入')
            elif not isinstance(parameter, ScalarArg):
                fail(argument, '签名标量与张量规格不符: ' + name)
    fn.decorator_list = [decorator.func]


class Capture:
    def __init__(self, parameters, math_names=(), math_module=False, devices=None, constants=None):
        self.parameters = parameters
        self.constants = {} if constants is None else dict(constants)
        # kernel 内 `BLOCK_DIM = 32` 这类整数字面量绑定，供 shared/local 形状使用。
        self.local_constants = {}
        # device helper 的静态作用域来源：**未按 kernel 绑定过滤**的模块级常量。
        self.module_values = {}
        self.reserved = set()
        # 内联 helper 期间可见的调用者张量名；None 表示不在 helper 作用域内。
        self.device_tensors = None
        self.registers = []
        self.math = dict(math_names)   # kernel里可见的函数名 -> math原始函数名
        self.math_module = math_module
        self.devices = {} if devices is None else devices
        self.in_loop = 0
        # 内联 device 函数时：helper 形参名 -> 调用者实参名（张量只按名字引用，须改写成实参名）。
        self.tensor_aliases = {}
        # 共享/局部数组：AST 名 -> (dtype, shape)。它们是 kernel 内的分配，不是形参。
        self.arrays = {}
        # 分配编号是 Capture 级的：scope 切换/恢复 arrays 时也不能重号，
        # 否则 caller 与 helper 各自的 buffer 会共用同一个生成物名字。
        self.array_counter = 0

    def tensor_name(self, name):
        return self.tensor_aliases.get(name, name)

    def array_or_tensor(self, node, env):
        """返回 (dtype, shape, 生成物里的名字)：共享/局部数组用分配变量名，形参用（别名化的）张量名。"""
        if isinstance(node, ast.Name) and node.id in self.arrays:
            dtype, shape = self.arrays[node.id]
            return dtype, shape, env[node.id].name
        parameter = self.tensor(node)
        return parameter.dtype, parameter.shape, self.tensor_name(node.id)

    def array_constants(self):
        """共享/局部数组形状可用的常量：模块级常量 + kernel 内已绑定的整数字面量。"""
        return {**self.constants, **self.local_constants} if self.local_constants else self.constants

    def register(self, dtype):
        result = Value(f'_tl_v{len(self.registers)}', dtype)
        self.registers.append(result)
        return result

    def put(self, output, op, dtype, node, args=(), value=None):
        target = self.register(dtype)
        output.append(Instruction(op, target, tuple(args), value, node.lineno))
        return target

    def copy(self, output, target, value, node):
        output.append(Instruction('copy', target, (value,), line=node.lineno))

    def const(self, value, node, output, dtype=None):
        return self.put(output, 'const', dtype or literal_type(value), node, value=value)

    def calculate(self, symbol, operands, node, output):
        casts, result = operation(symbol, tuple(v.dtype for v in operands))
        if symbol in ('//', '%') and result not in ('int32', 'int64'):
            if result in ('float32', 'float64') \
                    and tuple(v.dtype for v in operands) == (result, result) \
                    and casts == (result, result):
                return self._float_divmod(symbol, operands, result, node, output)
            fail(node, f'浮点 {symbol} 仅支持同型float32/float64操作数与结果')
        if symbol == '/' and result in ('float32', 'float64'):
            # Python的int/int是真除法：整数操作数必须先转浮点，否则TVM拒绝歧义的整数/。
            casts = tuple(result for _ in casts)
        return self.put(output, 'op', result, node, operands, (symbol, casts))

    def _float_divmod(self, symbol, operands, dtype, node, output):
        """同型浮点商余展开；保留修正次序，余数路径不计算商。"""
        remainder = symbol == '%'
        x, y = operands
        result = self.register(dtype)
        zero = self.const(0.0, node, output, dtype)
        y_zero = self.calculate('==', (y, zero), node, output)
        zero_divisor, nonzero_divisor = [], []
        if remainder:
            zero_result = self.put(zero_divisor, 'op', dtype, node, (x, y),
                                   ('fmod', (dtype, dtype)))
        else:
            zero_result = self.calculate('/', (x, y), node, zero_divisor)
        self.copy(zero_divisor, result, zero_result, node)

        # 内部原语显式同型，不开放用户源码的math.fmod/math.floor。
        m = self.put(nonzero_divisor, 'op', dtype, node, (x, y),
                     ('fmod', (dtype, dtype)))
        if not remainder:
            difference = self.calculate('-', (x, m), node, nonzero_divisor)
            d = self.calculate('/', (difference, y), node, nonzero_divisor)
        m_zero = self.calculate('==', (m, zero), node, nonzero_divisor)
        zero_remainder, nonzero_remainder = [], []
        y_negative = self.calculate('<', (y, zero), node, zero_remainder)
        negative_zero, positive_zero = [], []
        signed_zero = self.const(-0.0, node, negative_zero, dtype)
        self.copy(negative_zero, m, signed_zero, node)
        self.copy(positive_zero, m, zero, node)
        zero_remainder.append(Branch(y_negative, tuple(negative_zero),
                                     tuple(positive_zero), node.lineno))

        # m==0的假臂包括NaN；符号不同比较的是两个bool。
        y_negative = self.calculate('<', (y, zero), node, nonzero_remainder)
        m_negative = self.calculate('<', (m, zero), node, nonzero_remainder)
        different_signs = self.calculate('!=', (y_negative, m_negative),
                                         node, nonzero_remainder)
        correction = []
        if not remainder:
            one = self.const(1.0, node, correction, dtype)
            corrected_d = self.calculate('-', (d, one), node, correction)
            self.copy(correction, d, corrected_d, node)
        corrected_m = self.calculate('+', (m, y), node, correction)
        self.copy(correction, m, corrected_m, node)
        nonzero_remainder.append(Branch(different_signs, tuple(correction), (), node.lineno))
        nonzero_divisor.append(Branch(m_zero, tuple(zero_remainder),
                                      tuple(nonzero_remainder), node.lineno))

        if remainder:
            self.copy(nonzero_divisor, result, m, node)
            output.append(Branch(y_zero, tuple(zero_divisor), tuple(nonzero_divisor), node.lineno))
            return result

        # 商路径在整个余数分支之后；先排NaN，再判非零。
        ordered = self.calculate('==', (d, d), node, nonzero_divisor)
        nonzero = self.register('bool')
        ordered_quotient, unordered_quotient = [], []
        is_nonzero = self.calculate('!=', (d, zero), node, ordered_quotient)
        self.copy(ordered_quotient, nonzero, is_nonzero, node)
        false = self.const(False, node, unordered_quotient, 'bool')
        self.copy(unordered_quotient, nonzero, false, node)
        nonzero_divisor.append(Branch(ordered, tuple(ordered_quotient),
                                      tuple(unordered_quotient), node.lineno))

        q = self.register(dtype)
        nonzero_quotient, zero_or_nan_quotient = [], []
        floored = self.put(nonzero_quotient, 'op', dtype, node, (d,), ('floor', (dtype,)))
        self.copy(nonzero_quotient, q, floored, node)
        fraction = self.calculate('-', (d, q), node, nonzero_quotient)
        half = self.const(0.5, node, nonzero_quotient, dtype)
        needs_snap = self.calculate('>', (fraction, half), node, nonzero_quotient)
        snap = []
        one = self.const(1.0, node, snap, dtype)
        snapped = self.calculate('+', (q, one), node, snap)
        self.copy(snap, q, snapped, node)
        nonzero_quotient.append(Branch(needs_snap, tuple(snap), (), node.lineno))
        # 不重结合乘除，保留零/NaN假臂的计算依赖。
        s = self.calculate('*', (d, d), node, zero_or_nan_quotient)
        product = self.calculate('*', (s, x), node, zero_or_nan_quotient)
        quotient = self.calculate('/', (product, y), node, zero_or_nan_quotient)
        self.copy(zero_or_nan_quotient, q, quotient, node)
        nonzero_divisor.append(Branch(nonzero, tuple(nonzero_quotient),
                                      tuple(zero_or_nan_quotient), node.lineno))
        self.copy(nonzero_divisor, result, q, node)
        output.append(Branch(y_zero, tuple(zero_divisor), tuple(nonzero_divisor), node.lineno))
        return result

    def power(self, node, env, output):
        """幂运算：非负整数常量平方-乘，浮点底数的有限非负float常量走pow。

        按Numba整数幂在当前非负指数域的平方-乘次序，先将底数cast到返回类型。
        x**4 的平方部分为 (x*x)*(x*x)，不能改成浮点左结合连乘。
        实际lowering路径与数值验证边界见handoff/checks.md；这里不保证任意
        浮点位型或后端优化都等价。负数及变量指数仍不支持。
        """
        exponent = node.right
        if isinstance(exponent, ast.Constant) and type(exponent.value) is float:
            if not math.isfinite(exponent.value) or exponent.value < 0:
                fail(node, '浮点幂只支持有限非负float常量指数')
            base = self.expr(node.left, env, output)
            if base.dtype not in ('float32', 'float64'):
                fail(node, '浮点常量幂底数必须为float32或float64')
            power = self.expr(exponent, env, output)
            casts, result = operation('**', (base.dtype, power.dtype))
            return self.put(output, 'op', result, node, (base, power), ('pow', casts))
        if not isinstance(exponent, ast.Constant) or type(exponent.value) is not int \
                or isinstance(exponent.value, bool) or not 0 <= exponent.value <= MAX_POWER:
            fail(node, f'只支持0–{MAX_POWER}的整数常量指数（浮点或变量指数走pow路径，尚未支持）')
        base = self.expr(node.left, env, output)
        result = power_type(base.dtype, exponent.value)
        # Numba先做 val = cast(底数, 返回类型)，连乘全在返回类型里进行。
        if base.dtype != result:
            converted = self.register(result)
            self.copy(output, converted, base, node)
            base = converted
        res = self.const(1, node, output, result)
        remaining = exponent.value
        while remaining:
            if remaining & 1:
                res = self.calculate('*', (res, base), node, output)
            remaining >>= 1
            if remaining:
                # 末轮平方的结果不再使用，因此只发射后续仍需使用的平方。
                base = self.calculate('*', (base, base), node, output)
        return res

    def math_call(self, name, node, env, output):
        if node.keywords or len(node.args) != MATH_ARITY[name] \
                or any(isinstance(arg, ast.Starred) for arg in node.args):
            fail(node, '数学函数需要固定数量的位置参数: ' + name)
        arguments = [self.expr(argument, env, output) for argument in node.args]
        if name == 'pow':
            # math.pow 是 C pow 语义：numba 返回 float64，与 `**` 的整数平方-乘不同。
            converted = []
            for value in arguments:
                if value.dtype == 'float64':
                    converted.append(value)
                else:
                    slot = self.register('float64')
                    self.copy(output, slot, value, node)
                    converted.append(slot)
            return self.put(output, 'op', 'float64', node, tuple(converted),
                            ('pow', ('float64', 'float64')))
        casts, result = math_type(name, tuple(arg.dtype for arg in arguments))
        # Numba先选择整数签名再lower到double；TileLang数学原语直接接浮点。
        casts = tuple('float64' if dtype in ('bool', 'int32', 'int64') else dtype
                      for dtype in casts)
        return self.put(output, 'op', result, node, tuple(arguments), (THREAD_MATH[name], casts))

    def builtin_call(self, name, node, env, output):
        if is_infinity_call(node):
            return self.put(output, 'infinity', 'float64', node)
        if name in ('int', 'float'):
            value = self.expr(node.args[0], env, output)
            target = 'int64' if name == 'int' else 'float64'
            converted = self.register(target)
            self.copy(output, converted, value, node)
            return converted
        operands = tuple(self.expr(argument, env, output) for argument in node.args)
        if name == 'abs':
            casts = (operands[0].dtype,)
            return self.put(output, 'op', operands[0].dtype, node, operands, ('abs', casts))
        if name in ('min', 'max'):
            # 多参形式按左结合逐个折叠：max(a,b,c) == max(max(a,b),c)，语义与 Python 一致。
            result = operands[0]
            for other in operands[1:]:
                unified = unify(result.dtype, other.dtype)
                result = self.put(output, 'op', unified, node, (result, other), (name, (unified, unified)))
            return result
        result = unify(operands[0].dtype, operands[1].dtype)
        casts = (result, result)
        return self.put(output, 'op', result, node, operands, (name, casts))

    def tensor(self, node):
        if not isinstance(node, ast.Name):
            fail(node, '需要直接张量形参')
        if self.device_tensors is not None and node.id not in self.device_tensors:
            # 在 device 函数内只允许通过**自己的形参名**引用张量；
            # 用源码名而不是别名判断，避免"形参与调用者同名张量"互相冒充。
            fail(node, 'device 函数不能引用调用者张量: ' + node.id)
        name = self.tensor_name(node.id)
        if not isinstance(self.parameters.get(name), TensorArg):
            fail(node, '需要直接张量形参')
        return self.parameters[name]

    def device_scope(self, fn, output):
        """device helper 的静态作用域：只含模块级常量，不含调用者任何局部名。

        只物化 helper **实际读取**、且未被自身绑定（形参与被赋值名）的标量常量；
        常量序列不物化，留给折叠逻辑。调用方应在内联期间把 `device_tensors`
        设为**本次 helper 的形参名集合**（源码名，不是实参名）。
        """
        bound = {a.arg for a in fn.args.args}
        bound |= {n.id for n in ast.walk(fn)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        used = {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        scope = {}
        for name in sorted(used - bound):
            value = self.module_values.get(name)
            if value is None or isinstance(value, tuple) or name in self.reserved:
                continue
            scope[name] = self.const(value, fn, output)
        return scope

    def partial_shape(self, node):
        """张量参数或其标量前缀下标（`A[i]`、`A[i, j]`）的剩余形状。

        只读元数据：前缀下标不求值、不生成 load，因此只接受可静态识别的标量名/整数字面量；
        切片与任意表达式都拒绝，避免把带副作用的求值当成合法前缀。
        """
        if isinstance(node, ast.Name):
            return self.tensor(node).shape
        if isinstance(node, ast.Subscript):
            base = self.partial_shape(node.value)
            items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if len(items) > len(base):
                fail(node, '张量下标rank不匹配')
            for item in items:
                if isinstance(item, ast.Slice):
                    fail(item, '切片不受支持')
                if isinstance(item, ast.Constant) and type(item.value) is int:
                    continue
                if isinstance(item, ast.Name) and item.id not in self.arrays \
                        and not isinstance(self.parameters.get(self.tensor_name(item.id)), TensorArg):
                    continue
                fail(node, 'shape属性仅支持张量形参与标量前缀下标')
            return base[len(items):]
        fail(node, '需要直接张量形参')

    def subscript_chain(self, node):
        """把 `A[i][j]` 展平为 (基名, 下标列表)；对逐维索引等价于 `A[i, j]`。"""
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if isinstance(node.value, ast.Subscript):
            base, prefix = self.subscript_chain(node.value)
            return base, prefix + items
        return node.value, items

    def subscript_target(self, node, env, output, *, allow_nested=False):
        """解析下标语句（含 `A[i][j]` 链）为 (下标元组, dtype, 生成物名字)。"""
        base, _ = self.subscript_chain(node)
        indices = self.indices(node, env, output, allow_nested=allow_nested)
        dtype, _, out_name = self.array_or_tensor(base, env)
        return indices, dtype, out_name

    def indices(self, node, env, output, *, allow_nested=False):
        base, indices = self.subscript_chain(node)
        _, shape, _ = self.array_or_tensor(base, env)
        if len(indices) != len(shape):
            fail(node, '张量下标rank不匹配')
        values = []
        for index in indices:
            if not allow_nested and any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                   and isinstance(self.parameters.get(n.value.id), TensorArg) for n in ast.walk(index)):
                fail(index, '间接张量索引尚未实现')
            value = self.expr(index, env, output)
            if value.dtype not in ('int32', 'int64'):
                fail(index, '张量下标必须是整数')
            values.append(value)
        return tuple(values)

    def condition(self, node, env, output):
        value = self.expr(node, env, output)
        if value.dtype != 'bool':
            if value.dtype in ('int32', 'int64'):
                # Numba 允许整数条件（非零为真）；生成显式比较，保持 TileLang 的 bool 条件契约。
                zero = self.const(0, node, output, value.dtype)
                return self.put(output, 'op', 'bool', node, (value, zero), ('!=', (value.dtype, value.dtype)))
            fail(node, '条件必须为bool或整数')
        return value

    def grid(self, node, output):
        kind = node.func.attr
        if node.keywords or len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) \
                or type(node.args[0].value) is not int or node.args[0].value not in (1, 2, 3):
            fail(node, 'cuda.' + kind + '需要1–3的整数常量')
        return tuple(self.put(output, kind, 'int64', node, value=i)
                     for i in range(node.args[0].value))

    def expr(self, node, env, output):
        if isinstance(node, ast.Constant):
            if type(node.value) is float and not math.isfinite(node.value):
                fail(node, '非有限源码常量尚未支持')
            return self.const(node.value, node, output)
        if isinstance(node, ast.Name):
            if node.id not in env:
                fail(node, '未知名称或并非所有路径已定义: ' + node.id)
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            return self.power(node, env, output)
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            a = self.expr(node.left, env, output)
            b = self.expr(node.right, env, output)
            return self.calculate(BINARY[type(node.op)], (a, b), node, output)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                value = self.condition(node.operand, env, output)
                return self.put(output, 'not', 'bool', node, (value,))
            if isinstance(node.op, (ast.UAdd, ast.USub)):
                value = self.expr(node.operand, env, output)
                return self.calculate('pos' if isinstance(node.op, ast.UAdd) else 'neg', (value,), node, output)
        if isinstance(node, ast.IfExp):
            condition = self.condition(node.test, env, output)
            yes, no = [], []
            a = self.expr(node.body, env, yes)
            b = self.expr(node.orelse, env, no)
            target = self.register(unify(a.dtype, b.dtype))
            self.copy(yes, target, a, node)
            self.copy(no, target, b, node)
            output.append(Branch(condition, tuple(yes), tuple(no), node.lineno))
            return target
        if isinstance(node, ast.BoolOp):
            first = self.condition(node.values[0], env, output)
            target = self.register('bool')
            self.copy(output, target, first, node)
            for item in node.values[1:]:
                branch = []
                value = self.condition(item, env, branch)
                self.copy(branch, target, value, node)
                yes, no = (tuple(branch), ()) if isinstance(node.op, ast.And) else ((), tuple(branch))
                output.append(Branch(target, yes, no, node.lineno))
            return target
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, env, output)
            target = self.register('bool')
            current = output
            # 链式比较的后续读取只发生在前一个比较成功的分支中。
            def chain(index, lhs, statements):
                op = COMPARE.get(type(node.ops[index]))
                if op is None:
                    fail(node, '不支持该比较运算')
                rhs = self.expr(node.comparators[index], env, statements)
                value = self.calculate(op, (lhs, rhs), node, statements)
                self.copy(statements, target, value, node)
                if index + 1 < len(node.ops):
                    rest = []
                    chain(index + 1, rhs, rest)
                    statements.append(Branch(target, tuple(rest), (), node.lineno))
            chain(0, left, current)
            return target
        if isinstance(node, ast.Attribute):
            if self.math_module and isinstance(node.value, ast.Name) and node.value.id == 'math':
                # `math.pi` 等模块属性常量与 `from math import pi` 同义，按字面量折叠；
                # `math.inf` 不能写成字面量，走既有的 infinity 原语路径。
                if node.attr == 'inf':
                    return self.put(output, 'infinity', 'float64', node)
                if node.attr in MATH_CONSTANTS:
                    return self.const(MATH_CONSTANTS[node.attr], node, output, 'float64')
            if node.attr == 'size' and isinstance(node.value, ast.Name):
                parameter = self.tensor(node.value)
                return self.const(math.prod(parameter.shape), node, output, 'int64')
            if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name) \
                    and node.value.value.id == 'cuda' and node.value.attr in (
                        'threadIdx', 'blockIdx', 'blockDim', 'gridDim') and node.attr in 'xyz':
                return self.put(output, 'dim', 'int32', node,
                                value=(node.value.attr, 'xyz'.index(node.attr)))
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and isinstance(self.constants.get(node.value.id), tuple):
                # 模块级常量序列的只读下标：值在源码里是字面量，直接折叠为常量。
                sequence = self.constants[node.value.id]
                if not isinstance(node.slice, ast.Constant) or type(node.slice.value) is not int:
                    fail(node, '常量序列下标必须为整数字面量')
                if not 0 <= node.slice.value < len(sequence):
                    fail(node, '常量序列下标越界')
                item = sequence[node.slice.value]
                return self.const(item, node, output, 'int64' if type(item) is int else 'float64')
            if isinstance(node.value, ast.Attribute) and node.value.attr == 'shape':
                shape = self.partial_shape(node.value.value)
                if not isinstance(node.slice, ast.Constant) or type(node.slice.value) is not int \
                        or not 0 <= node.slice.value < len(shape):
                    fail(node, 'shape下标必须为有效非负常量')
                return self.const(shape[node.slice.value], node, output, 'int64')
            indices, dtype, out_name = self.subscript_target(node, env, output, allow_nested=True)
            return self.put(output, 'load', dtype, node, indices, out_name)
        if isinstance(node, ast.Call):
            if is_numpy_int32_call(node):
                if node.keywords or len(node.args) != 1:
                    fail(node, 'np.int32仅支持一个位置浮点实参')
                value = self.expr(node.args[0], env, output)
                if value.dtype not in ('float32', 'float64'):
                    fail(node, 'np.int32仅支持float32/float64标量输入')
                return self.put(output, 'numpy_int32', 'int32', node, (value,))
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in self.devices:
                    return capture_device_read(self, node, env, output)
                if name in self.math:
                    return self.math_call(self.math[name], node, env, output)
                if not node.keywords and name in BUILTIN_ARITY and (
                        (name in ('min', 'max') and len(node.args) >= 2)
                        or len(node.args) == BUILTIN_ARITY[name]):
                    return self.builtin_call(name, node, env, output)
                if name in DTYPES and not node.keywords and len(node.args) == 1:
                    # numba 类型名当转换函数（`float32(x)` 等），与 `int()/float()` 同义，目标类型显式。
                    value = self.expr(node.args[0], env, output)
                    converted = self.register(name)
                    self.copy(output, converted, value, node)
                    return converted
            if self.math_module and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'math' \
                    and node.func.attr in THREAD_MATH:
                return self.math_call(node.func.attr, node, env, output)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'len' \
                    and not node.keywords and len(node.args) == 1:
                parameter = self.tensor(node.args[0])
                return self.const(parameter.shape[0], node, output, 'int64')
            if ast.unparse(node.func) in ('cuda.grid', 'cuda.gridsize'):
                values = self.grid(node, output)
                if len(values) != 1:
                    fail(node, '多维grid需要tuple拆包赋值')
                return values[0]
        if is_cuda_sync_call(node):
            fail(node, '同步原语尚未支持（需要整块线程协同）: ' + ast.unparse(node))
        if is_cuda_memory_call(node):
            fail(node, '共享/局部数组分配不受支持（需要静态正整数形状与受支持的 dtype，且须作为赋值右值）: '
                 + ast.unparse(node))
        fail(node, '未实现的线程表达式: ' + ast.unparse(node))

    def store(self, target, indices, value, node, output):
        output.append(Instruction('store', args=(*indices, value), value=target, line=node.lineno))

    def block(self, body, incoming, top=False):
        env, output = dict(incoming), []
        for position, node in enumerate(body):
            if isinstance(node, ast.Return):
                # return结束的是整个线程，不只是当前块；只有函数体顶层才能保证"其后无语句"，
                # 所以嵌套位置一律拒绝，避免把"退出线程"错当成"跳过本块剩余"。
                if not top:
                    fail(node, 'return 只支持出现在kernel函数体的顶层语句里')
                if node.value is not None:
                    fail(node, '仅支持无值return')
                if any(not isinstance(x, ast.Pass) for x in body[position + 1:]):
                    fail(node, '无值return只允许出现在末尾，或写成顶层guard "if 条件: return"')
                break
            if top and isinstance(node, ast.If) and not node.orelse and bare_return(node.body):
                condition = self.condition(node.test, env, output)
                rest = body[position + 1:]
                if not any(not isinstance(x, ast.Pass) for x in rest):
                    continue
                # 谓词化：return之后的所有语句都只在该条件不成立时执行。
                # return路径到不了块外，所以块外环境直接取另一支，不需要合流槽。
                # rest仍处在"函数体顶层"语境，所以串联多个guard也是安全的。
                branch, env = self.block(rest, env, top=True)
                output.append(Branch(condition, (), tuple(branch), node.lineno))
                break
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                continue      # 纯字符串语句无副作用（源码里的注释式说明），等价 no-op
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Call) \
                        and ast.unparse(node.value.func) in ('cuda.grid', 'cuda.gridsize'):
                    values = self.grid(node.value, output)
                    if len(values) != len(target.elts) or not all(isinstance(x, ast.Name) for x in target.elts):
                        fail(node, 'grid拆包数量或名称不合法')
                    for name, value in zip(target.elts, values):
                        env[name.id] = value
                elif isinstance(target, ast.Tuple):
                    items = node.value.elts if isinstance(node.value, (ast.Tuple, ast.List)) else None
                    if items is not None:
                        # 先把右值全部求值再逐个赋值，保证 a, b = b, a 这类交换正确。
                        values = [self.expr(item, env, output) for item in items]
                    else:
                        # 右侧是调用：device helper 可返回元组（逐元素已求值）。
                        produced = self.expr(node.value, env, output)
                        if not isinstance(produced, tuple):
                            fail(node, '元组赋值右侧必须是元组或返回元组的调用')
                        values = list(produced)
                    if len(values) != len(target.elts):
                        fail(node, '元组赋值两侧数量必须一致')
                    for name, value in zip(target.elts, values):
                        if isinstance(name, ast.Name):
                            snapshot = self.register(value.dtype)
                            self.copy(output, snapshot, value, node)
                            env[name.id] = snapshot
                        elif isinstance(name, ast.Subscript):
                            indices, _, out_name = self.subscript_target(name, env, output, allow_nested=True)
                            self.store(out_name, indices, value, node, output)
                        else:
                            fail(name, '不支持的元组赋值目标: ' + type(name).__name__)
                elif isinstance(target, ast.Name) and parse_cuda_array(node.value, self.array_constants()) is not None:
                    kind, dims, dtype = parse_cuda_array(node.value, self.array_constants())
                    alloc = Value(f'_tl_a{self.array_counter}', dtype)
                    self.array_counter += 1
                    self.arrays[target.id] = (dtype, dims)
                    env[target.id] = alloc
                    output.append(Instruction('alloc_' + kind, target=alloc, value=(dims, dtype),
                                              line=node.lineno))
                else:
                    value = self.expr(node.value, env, output)
                    if isinstance(target, ast.Name):
                        # 即使右侧只是循环携带变量，也要保存本次赋值发生时的值。
                        snapshot = self.register(value.dtype)
                        self.copy(output, snapshot, value, node)
                        env[target.id] = snapshot
                        if isinstance(node.value, ast.Constant) and type(node.value.value) is int:
                            self.local_constants[target.id] = node.value.value
                        else:
                            self.local_constants.pop(target.id, None)
                    elif isinstance(target, ast.Subscript):
                        indices, _, out_name = self.subscript_target(target, env, output, allow_nested=True)
                        self.store(out_name, indices, value, node, output)
                    else:
                        fail(target, '不支持该赋值目标')
            elif isinstance(node, ast.AugAssign) and type(node.op) in BINARY:
                target = node.target
                if isinstance(target, ast.Subscript):
                    indices, dtype, out_name = self.subscript_target(target, env, output, allow_nested=True)
                    left = self.put(output, 'load', dtype, node, indices, out_name)
                elif isinstance(target, ast.Name):
                    left = self.expr(target, env, output)
                else:
                    fail(target, '不支持该增量赋值目标')
                right = self.expr(node.value, env, output)
                value = self.calculate(BINARY[type(node.op)], (left, right), node, output)
                if isinstance(target, ast.Name):
                    env[target.id] = value
                else:
                    self.store(out_name, indices, value, node, output)
            elif isinstance(node, ast.If):
                condition = self.condition(node.test, env, output)
                yes, yes_env = self.block(node.body, env)
                no, no_env = self.block(node.orelse, env)
                merged = {}
                for name in sorted(yes_env.keys() & no_env.keys()):
                    a, b = yes_env[name], no_env[name]
                    if a == b:
                        merged[name] = a
                    else:
                        phi = self.register(unify(a.dtype, b.dtype))
                        self.copy(yes, phi, a, node)
                        self.copy(no, phi, b, node)
                        merged[name] = phi
                output.append(Branch(condition, tuple(yes), tuple(no), node.lineno))
                env = merged
            elif isinstance(node, ast.For):
                self.loop(node, env, output)
            elif isinstance(node, ast.While):
                self.while_loop(node, env, output)
            elif isinstance(node, ast.Expr) and is_atomic_add(node.value):
                call = node.value
                index = self.expr(call.args[1], env, output)
                value = self.expr(call.args[2], env, output)
                atomic_add_type(index.dtype)
                output.append(Instruction('atomic_add', args=(index, value),
                                          value=call.args[0].id, line=node.lineno))
            elif isinstance(node, ast.Expr) and is_exchange(node.value):
                call = node.value
                value = self.expr(call.args[2], env, output)
                atomic_exchange_type(value.dtype)
                output.append(Instruction('atomic_exch', args=(value,),
                                          value=(call.args[0].id, call.args[1].value),
                                          line=node.lineno))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                    and isinstance(node.value.func, ast.Name) \
                    and node.value.func.id in self.devices:
                capture_device_read(self, node.value, env, output)     # void device 过程
            elif isinstance(node, ast.Break):
                if not self.in_loop:
                    fail(node, 'break 只能出现在循环体内')
                output.append(Instruction('break', line=node.lineno))
            elif isinstance(node, ast.Expr) and is_cuda_sync_call(node.value):
                call = node.value
                # 只映射无参 __syncthreads()；带返回值的计数/谓词变体与 warp 级/内存栅栏不在本步范围。
                if call.func.attr != 'syncthreads' or call.args or call.keywords:
                    fail(call, '同步原语尚未支持（仅支持无参 cuda.syncthreads）: ' + ast.unparse(call))
                output.append(Instruction('sync', line=node.lineno))
            elif isinstance(node, ast.Pass):
                continue
            else:
                fail(node, '未实现的线程语句: ' + type(node).__name__)
        return output, env

    def probe(self, body, env, index_name, index, dtypes):
        """试探捕获循环体，只为读出回边类型；寄存器与指令全部回滚，不留副作用。"""
        saved = len(self.registers)
        try:
            inner = dict(env)
            for name, dtype in dtypes.items():
                inner[name] = Value(name, dtype)
            if index_name is not None:
                inner[index_name] = index
            _, final = self.block(body, inner)
        except Unsupported:
            return None
        finally:
            del self.registers[saved:]
        return {name: value.dtype for name, value in final.items()}

    def tensor_loop(self, node, env, output):
        parameter = self.tensor(node.iter)
        if len(parameter.shape) != 1:
            fail(node, '张量迭代仅支持rank1参数张量')
        if node.orelse or not isinstance(node.target, ast.Name):
            fail(node, '张量迭代需要单个名称目标，无for-else')
        name = node.target.id
        if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == name
               for statement in node.body for n in ast.walk(statement)):
            fail(node, '首版不支持循环体重绑循环变量')
        # 只改局部副本；隐藏下标与元素值分开，每轮先读取，再执行原body。
        index_name = '_tl_element_index_' + str(len(self.registers))
        expanded = copy.deepcopy(node)
        expanded.target = ast.Name(id=index_name, ctx=ast.Store())
        expanded.iter = ast.Call(func=ast.Name(id='range', ctx=ast.Load()),
                                 args=[ast.Attribute(value=copy.deepcopy(node.iter),
                                                     attr='size', ctx=ast.Load())], keywords=[])
        load = ast.Assign(targets=[copy.deepcopy(node.target)],
                          value=ast.Subscript(value=copy.deepcopy(node.iter),
                                              slice=ast.Name(id=index_name, ctx=ast.Load()),
                                              ctx=ast.Load()))
        expanded.body.insert(0, ast.copy_location(load, node))
        ast.fix_missing_locations(expanded)
        inner = dict(env)
        inner.pop(name, None)
        self.loop(expanded, inner, output)
        # 与range相同，不把循环目标暴露为循环后的旧值或未定义值。
        inner.pop(name, None)
        env.clear()
        env.update(inner)

    def constant_loop(self, node, env, output):
        """模块级常量序列的 for 迭代：逐元素展开，不生成运行期循环。

        只用于字面量可求的短序列；展开保持 Python 的逐次绑定语义（循环变量在体后仍存在，
        但本实现与 range 循环一致，循环结束后不再暴露该名字）。
        """
        if node.orelse:
            fail(node, '常量序列迭代不支持for-else')
        if not isinstance(node.target, ast.Name):
            fail(node, '常量序列迭代仅支持单一循环变量')
        sequence = self.constants[node.iter.id]
        if len(sequence) > 16:
            fail(node, '常量序列过长，需运行期常量内存支持')
        for statement in node.body:
            if any(isinstance(n, (ast.Break, ast.Continue, ast.Return)) for n in ast.walk(statement)):
                fail(node, '常量序列迭代不支持break/continue/return')
        names = {n.id for statement in node.body for n in ast.walk(statement)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if node.target.id in names:
            fail(node, '循环体不能重绑循环变量')
        for item in sequence:
            inner = dict(env)
            dtype = 'int64' if type(item) is int else 'float64'
            inner[node.target.id] = self.const(item, node, output, dtype)
            self.in_loop += 1
            try:
                statements, final = self.block(node.body, inner)
            finally:
                self.in_loop -= 1
            output.extend(statements)
            for name, value in final.items():
                if name != node.target.id:
                    env[name] = value
        env.pop(node.target.id, None)

    def loop(self, node, env, output):
        if isinstance(node.iter, ast.Name):
            if isinstance(self.constants.get(node.iter.id), tuple):
                return self.constant_loop(node, env, output)
            return self.tensor_loop(node, env, output)
        call = node.iter
        if node.orelse or not isinstance(node.target, ast.Name) or not isinstance(call, ast.Call) \
                or not isinstance(call.func, ast.Name) or call.func.id != 'range' \
                or call.keywords or len(call.args) not in (1, 2, 3):
            fail(node, '仅支持range的1–3个位置参数，无for-else')
        bounds = [self.expr(arg, env, output) for arg in call.args]
        if any(x.dtype not in ('int32', 'int64') for x in bounds):
            fail(node, 'range边界必须为整数')
        if len(bounds) == 1:
            bounds.insert(0, self.const(0, node, output, bounds[0].dtype))
        step = None
        if len(bounds) == 3:
            dtype = range_type(tuple(v.dtype for v in bounds))
            snapshots = []
            for value in bounds:
                snapshot = self.register(value.dtype)
                self.copy(output, snapshot, value, node)
                snapshots.append(snapshot)
            bounds = snapshots
            step = bounds[2]
        else:
            dtype = unify(bounds[0].dtype, bounds[1].dtype)
        index = self.register(dtype)
        names = {n.id for stmt in node.body for n in ast.walk(stmt)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if node.target.id in names:
            fail(node, '首版不支持循环体重绑循环变量')
        inner = dict(env)
        carried_names = sorted(names & env.keys())
        # 槽类型取不动点：先按进入时类型试捕获，回边若能unify就加宽并重试，直到稳定。
        # 试探捕获的寄存器与指令全部回滚，只有稳定的类型才真正发射。
        dtypes = {name: env[name].dtype for name in carried_names}
        for _ in range(4):
            probed = self.probe(node.body, inner, node.target.id, index, dtypes)
            if probed is None:
                break
            try:
                merged = {name: unify(dtypes[name], probed[name]) if name in probed else dtypes[name]
                          for name in carried_names}
            except Unsupported:
                break
            if merged == dtypes:
                break
            dtypes = merged
        carried = {}
        for name in carried_names:
            slot = self.register(dtypes[name])
            self.copy(output, slot, env[name], node)
            carried[name] = slot
            inner[name] = slot
        inner[node.target.id] = index
        self.in_loop += 1
        try:
            body, final = self.block(node.body, inner)
        finally:
            self.in_loop -= 1
        for name, slot in carried.items():
            if name not in final or final[name].dtype != slot.dtype:
                fail(node, '循环携带变量回边类型必须稳定: ' + name)
            self.copy(body, slot, final[name], node)
            env[name] = slot
        env.pop(node.target.id, None)
        output.append(SerialLoop(index, bounds[0], bounds[1], tuple(body), node.lineno, step))

    def while_counter(self, test):
        """取"计数器 < 整数标量上界"合取项的计数器名；不合契约时返回None。"""
        found = []
        for item in conjuncts(test):
            if isinstance(item, ast.Compare) and len(item.ops) == 1 and isinstance(item.ops[0], ast.Lt) \
                    and isinstance(item.left, ast.Name) and isinstance(item.comparators[0], ast.Name):
                spec = self.parameters.get(item.comparators[0].id)
                if isinstance(spec, ScalarArg) and spec.dtype in ('int32', 'int64'):
                    found.append(item.left.id)
        return found[0] if len(found) == 1 else None

    def grid_stride_counter(self, test):
        """精确的一维形状上界；grid来源在捕获后的IR上证明。"""
        if isinstance(test, ast.Compare) and len(test.ops) == 1 \
                and isinstance(test.ops[0], ast.Lt) and isinstance(test.left, ast.Name):
            bound = test.comparators[0]
            if isinstance(bound, ast.Attribute) and bound.attr == 'size' \
                    and isinstance(bound.value, ast.Name):
                spec = self.parameters.get(bound.value.id)
                if not isinstance(spec, TensorArg) or len(spec.shape) != 1:
                    fail(test, 'grid-stride while上界必须为rank1张量size')
                return test.left.id
        return None

    def halving_counter(self, test):
        """`v > 0` 形态：配合循环体末句 `v //= 2`，非负整数每轮严格递减，必然终止。"""
        if isinstance(test, ast.Compare) and len(test.ops) == 1                 and isinstance(test.ops[0], ast.Gt)                 and isinstance(test.left, ast.Name)                 and isinstance(test.comparators[0], ast.Constant)                 and type(test.comparators[0].value) is int                 and test.comparators[0].value == 0:
            return test.left.id
        return None

    def while_body_binding(self, node, counter, grid_stride=False, halving=False):
        """计数器每轮恰好自增1，且是循环体最后一个语句——循环后的值才是执行次数。"""
        for statement in node.body:
            if any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(statement)):
                fail(statement, 'while 循环体不支持嵌套 for/while')
            if any(isinstance(n, (ast.Break, ast.Continue, ast.Return)) for n in ast.walk(statement)):
                fail(statement, 'while 循环体不支持 break/continue/return')
        binds = [n for statement in node.body for n in ast.walk(statement)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == counter]
        if len(binds) != 1:
            fail(node, ('grid-stride while计数器每轮必须恰好更新一次: ' if grid_stride
                        else 'while 计数器每轮必须恰好 += 1: ') + counter)
        statements = [x for x in node.body if not isinstance(x, ast.Pass)]
        last = statements[-1] if statements else None
        if grid_stride:
            if not isinstance(last, ast.AugAssign) or not isinstance(last.op, ast.Add) \
                    or not isinstance(last.target, ast.Name) or last.target.id != counter \
                    or not isinstance(last.value, ast.Name):
                fail(node, 'grid-stride while末句必须为计数器 += 步长变量')
            step = last.value.id
            if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == step
                   for statement in node.body for n in ast.walk(statement)):
                fail(node, 'grid-stride while循环体不能重绑步长')
            return step
        if halving:
            aug = isinstance(last, ast.AugAssign) and isinstance(last.op, ast.FloorDiv) \
                and isinstance(last.target, ast.Name) and last.target.id == counter \
                and isinstance(last.value, ast.Constant) and last.value.value == 2
            assign = isinstance(last, ast.Assign) and len(last.targets) == 1 \
                and isinstance(last.targets[0], ast.Name) and last.targets[0].id == counter \
                and isinstance(last.value, ast.BinOp) and isinstance(last.value.op, ast.FloorDiv) \
                and isinstance(last.value.left, ast.Name) and last.value.left.id == counter \
                and isinstance(last.value.right, ast.Constant) and last.value.right.value == 2
            if not (aug or assign):
                fail(node, 'halving while 末句必须为计数器 //= 2: ' + counter)
            return None
        if not isinstance(last, ast.AugAssign) or not isinstance(last.op, ast.Add) \
                or not isinstance(last.target, ast.Name) or last.target.id != counter \
                or not isinstance(last.value, ast.Constant) or type(last.value.value) is not int \
                or last.value.value != 1:
            fail(node, 'while 计数器的 += 1 必须是循环体最后一个语句: ' + counter)

    def while_loop(self, node, env, output):
        """有界计数while：条件每轮重算，计数器可证明从0起逐步+1。

        步数上界由计数器与整数标量上界共同界定，因此不重写成固定次数的for：
        条件里的其它项（含张量读取）仍须按源码在每个判断点重新求值。
        """
        if self.in_loop:
            fail(node, 'while 不支持放在其它循环体内')
        if node.orelse:
            fail(node, 'while 不支持 else 子句')
        grid_counter = self.grid_stride_counter(node.test)
        counter = grid_counter if grid_counter is not None else self.while_counter(node.test)
        halving = False
        if counter is None:
            counter = self.halving_counter(node.test)
            halving = counter is not None
        if counter is None or counter in self.parameters:
            fail(node, 'while 条件无法证明有界: 需要且仅需要一个"计数器 < 整数标量上界"的合取条件')
        if counter not in env:
            fail(node, 'while 计数器必须在进入循环前已定义: ' + counter)
        step_name = self.while_body_binding(node, counter, grid_counter is not None, halving)
        grid_stride = ()
        if grid_counter is not None:
            if step_name not in env:
                fail(node, 'grid-stride while步长必须在进入循环前已定义')
            if env[counter].dtype != 'int64' or env[step_name].dtype != 'int64':
                fail(node, 'grid-stride while计数器与步长必须为int64')
            bound = self.expr(node.test.comparators[0], env, output)
            grid_stride = (env[step_name], bound)
        names = {n.id for statement in node.body for n in ast.walk(statement)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        inner = dict(env)
        carried_names = sorted(names & env.keys())
        dtypes = {name: env[name].dtype for name in carried_names}
        for _ in range(4):
            probed = self.probe(node.body, inner, None, None, dtypes)
            if probed is None:
                break
            try:
                merged = {name: unify(dtypes[name], probed[name]) if name in probed else dtypes[name]
                          for name in carried_names}
            except Unsupported:
                break
            if merged == dtypes:
                break
            dtypes = merged
        if dtypes.get(counter) != 'int64':
            fail(node, 'while 计数器必须是int64: ' + counter)
        entry = tuple((name, env[name]) for name in carried_names)
        for name in carried_names:
            slot = self.register(dtypes[name])
            self.copy(output, slot, env[name], node)
            inner[name] = slot
        tests = []
        condition = self.condition(node.test, inner, tests)
        self.in_loop += 1
        try:
            body, final = self.block(node.body, inner)
        finally:
            self.in_loop -= 1
        for name in carried_names:
            slot = inner[name]
            if name not in final or final[name].dtype != slot.dtype:
                fail(node, 'while 循环携带变量回边类型必须稳定: ' + name)
            self.copy(body, slot, final[name], node)
        for name in carried_names:
            env[name] = inner[name]
        output.append(WhileLoop(condition, tuple(tests), tuple(body), entry, counter,
                                node.lineno, grid_stride, halving))


def capture_thread(source, parameters, constants=None, entry=None):
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as exc:
        raise Unsupported('Python语法错误: ' + str(exc)) from exc
    # 真实代码常把kernel定义在函数或测试方法内部，所以按整棵树按行号查找而不是只看模块级。
    functions = sorted((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and any(is_cuda_jit(d) for d in n.decorator_list)),
                       key=lambda n: n.lineno)
    if entry is None:
        if len(functions) != 1:
            raise Unsupported('需要唯一一个带@cuda.jit的函数；多kernel文件请用entry指定')
        fn = functions[0]
    else:
        if type(entry) is not str:
            raise Unsupported('entry必须是kernel名的字符串')
        matched = [n for n in functions if n.name == entry]
        if len(matched) != 1:
            raise Unsupported('entry未匹配到唯一的kernel: ' + repr(entry))
        fn = matched[0]
    args = fn.args
    if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults \
            or any(a.annotation is not None for a in args.args):
        fail(fn, '仅支持无注解、无默认值的普通形参')
    names = [a.arg for a in args.args]
    if type(parameters) is not dict or set(parameters) != set(names) \
            or any(type(v) not in (TensorArg, ScalarArg) for v in parameters.values()):
        raise ValueError('parameters必须逐一对应源码形参及TensorArg/ScalarArg规格')
    parameters = {name: parameters[name] for name in names}
    constants = {} if constants is None else constants
    explicit_constants = dict(constants)   # 调用方显式给出的值优先于模块级折叠
    # 标量逐字接受；常量序列只接受字面量可求的标量元组（用于只读下标与短循环展开）。
    def scalar_or_sequence(value):
        if type(value) in (bool, int, float):
            return True
        return type(value) is tuple and all(type(v) in (int, float) for v in value)

    if type(constants) is not dict or any(
            type(name) is not str or not name.isidentifier()
            or not scalar_or_sequence(value) for name, value in constants.items()):
        raise ValueError('constants必须是以源码名为键的标量，或字面量可求的标量元组')
    validate_atomic(tree, fn, parameters)
    validate_infinity_binding(tree, fn, constants)
    validate_numpy_int32_binding(tree, fn, constants)
    signature_contract(tree, fn, parameters)
    modules, math_functions = validate_imports(
        tree, fn, allow_module_bindings=True,
        extra_math=tuple(THREAD_MATH) + tuple(MATH_CONSTANTS))
    from thread_bindings import validate_math_bindings
    validate_math_bindings(tree, fn, modules, math_functions)
    math_names = {name: original for name, original in math_functions.items()
                  if original in THREAD_MATH}
    math_values = {name: MATH_CONSTANTS[original] for name, original in math_functions.items()
                   if original in MATH_CONSTANTS}
    bound = {fn.name, *names} | {n.id for n in ast.walk(fn)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    reserved = PROTECTED_NAMES | set(math_functions)
    if bound & reserved or any(n.startswith('_tl_') for n in bound):
        fail(fn, '不能遮蔽已识别模块、内置函数或生成器保留名称')
    # 显式常量代表调用方意图，与函数名、形参或局部绑定同名会让"模块级"语义含糊，直接拒绝。
    if set(constants) & bound:
        fail(fn, '常量名与函数名、形参或局部绑定冲突: ' + ', '.join(sorted(set(constants) & bound)))
    if set(constants) & reserved or any(n.startswith('_tl_') for n in constants):
        fail(fn, '常量名不能遮蔽已识别模块、内置函数或生成器保留名称')
    # 整文件输入时自动折叠模块级标量；与被函数绑定或保留名同名的项属于局部/模块对象，
    # 丢弃而不是报错（源码里同名形参本就遮蔽模块级）。显式提供的值优先。
    from frontend import module_scalars
    module_values = module_scalars(tree, fn)
    folded = {name: value for name, value in module_values.items()
              if name not in bound and name not in reserved}
    folded.update(math_values)   # math常量按导入来源折叠，压过同名模块级赋值
    constants = {**folded, **constants}
    # 形参重绑定会令张量识别和参数身份分离，首版明确拒绝。
    assigned = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    if assigned & set(names):
        fail(fn, '首版不支持重绑定形参')
    devices = collect_device_reads(tree, fn, signature_contract, SIGNATURE_ALIASES, reserved)
    if set(devices) & set(constants):
        fail(fn, 'device函数名称与常量绑定冲突')
    capture = Capture(parameters, math_names, 'math' in modules, devices, constants)
    # 未过滤的模块级绑定：kernel 内同名局部赋值不应改变 device helper 看到的模块常量。
    capture.module_values = {**module_values, **math_values, **explicit_constants}
    capture.reserved = reserved
    env = {name: Value(name, p.dtype) for name, p in parameters.items() if isinstance(p, ScalarArg)}
    # 模块级标量在程序入口物化一次，保持单一定义供区间证明复用；未被引用的常量不发射。
    loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    prefix = []
    for name in sorted(set(constants) & loaded):
        if isinstance(constants[name], tuple):
            continue      # 常量序列不物化为运行期值，只在折叠下标/展开循环时使用
        env[name] = capture.const(constants[name], fn, prefix)
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body.pop(0)
    statements, _ = capture.block(body, env, top=True)
    statements = [*prefix, *statements]
    if not statements:
        fail(fn, '空线程程序')
    program = ThreadProgram(fn.name, tuple(parameters.items()), tuple(capture.registers), tuple(statements))
    from thread_ranges import validate_divisions, validate_ranges, validate_whiles
    validate_ranges(program)
    validate_divisions(program)
    validate_whiles(program)
    return program
