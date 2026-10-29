# -*- coding: utf-8 -*-
"""新三参数range的保守整数区间证明，不猜运行时数据或分支条件。"""
from semantic import Unsupported
from thread_ir import TensorArg, ScalarArg, Instruction, Branch, SerialLoop, WhileLoop


# 与thread_runtime.validate_launch的CUDA硬上限对应；各轴独立上界是保守包络。
GRID_LIMITS = (2147483647, 65535, 65535)
BLOCK_LIMITS = (1024, 1024, 64)
INTEGER_LIMITS = {'int32': (-(1 << 31), (1 << 31) - 1),
                  'int64': (-(1 << 63), (1 << 63) - 1)}


def walk(body):
    for statement in body:
        yield statement
        if isinstance(statement, Branch):
            yield from walk(statement.yes)
            yield from walk(statement.no)
        elif isinstance(statement, SerialLoop):
            yield from walk(statement.body)
        elif isinstance(statement, WhileLoop):
            yield from walk(statement.tests)
            yield from walk(statement.body)


def fits(interval, dtype):
    limits = INTEGER_LIMITS.get(dtype)
    return (interval if interval is not None and limits is not None
            and limits[0] <= interval[0] <= interval[1] <= limits[1] else None)


class Intervals:
    def __init__(self, body, parameters=()):
        self.parameters = dict(parameters)
        for statement in walk(body):
            if isinstance(statement, Instruction) and statement.op in ('store', 'atomic_exch'):
                name = statement.value if statement.op == 'store' else statement.value[0]
                spec = self.parameters.get(name)
                if isinstance(spec, TensorArg) and spec.value_range is not None:
                    raise Unsupported(f'line {statement.line}: 值域契约参数只读: {name}')
        self.definitions = {}
        self.cache = {}
        self.active = set()
        for statement in walk(body):
            if isinstance(statement, Instruction) and statement.target is not None:
                self.definitions.setdefault(statement.target, []).append(statement)
            elif isinstance(statement, SerialLoop):
                self.definitions.setdefault(statement.index, []).append(None)

    def get(self, value):
        if value in self.cache:
            return self.cache[value]
        definitions = self.definitions.get(value, ())
        if not definitions:
            # kernel 形参：标量参数可携带 value_range，与张量元素读取同义。
            spec = self.parameters.get(value.name)
            if isinstance(spec, ScalarArg) and spec.value_range is not None:
                return fits(spec.value_range, value.dtype)
            return None
        if value in self.active or len(definitions) != 1 or definitions[0] is None:
            return None
        self.active.add(value)
        interval = fits(self.evaluate(definitions[0]), value.dtype)
        self.active.remove(value)
        self.cache[value] = interval
        return interval

    def grid_origin(self, value, op):
        """只穿过同为int64的纯copy，不把类型转换或算术当作grid来源。"""
        seen = set()
        while value not in seen and value.dtype == 'int64':
            seen.add(value)
            definitions = self.definitions.get(value, ())
            if len(definitions) != 1 or definitions[0] is None:
                return False
            instruction = definitions[0]
            if instruction.op == op:
                return instruction.value == 0
            if instruction.op != 'copy':
                return False
            value = instruction.args[0]
        return False

    def evaluate(self, instruction):
        op, args, value = instruction.op, instruction.args, instruction.value
        if op == 'const':
            return (value, value) if type(value) is int else None
        if op == 'copy':
            return self.get(args[0])
        if op == 'load':
            spec = self.parameters.get(value)
            return spec.value_range if isinstance(spec, TensorArg) else None
        if op == 'gridsize':
            return (1, GRID_LIMITS[value] * BLOCK_LIMITS[value])
        if op == 'grid':
            return (0, GRID_LIMITS[value] * BLOCK_LIMITS[value] - 1)
        if op == 'dim':
            kind, axis = value
            return {'threadIdx': (0, BLOCK_LIMITS[axis] - 1),
                    'blockIdx': (0, GRID_LIMITS[axis] - 1),
                    'blockDim': (1, BLOCK_LIMITS[axis]),
                    'gridDim': (1, GRID_LIMITS[axis])}[kind]
        if op != 'op':
            return None
        symbol, casts = value
        intervals = [fits(self.get(arg), cast) for arg, cast in zip(args, casts)]
        if any(x is None for x in intervals):
            return None
        a = intervals[0]
        if symbol == 'pos':
            return a
        if symbol == 'neg':
            return (-a[1], -a[0])
        if symbol not in ('+', '-', '*', '//', '%'):
            return None
        b = intervals[1]
        if symbol == '+':
            return (a[0] + b[0], a[1] + b[1])
        if symbol == '-':
            return (a[0] - b[1], a[1] - b[0])
        if symbol == '*':
            products = [x * y for x in a for y in b]
            return (min(products), max(products))
        # floor除法对正除数在两端都单调，且Python的%对正除数落在[0, 除数-1]。
        if b[0] < 1:
            return None
        if symbol == '//':
            return (a[0] // b[1], a[1] // b[0])
        return (0, b[1] - 1)


def positive_trip_count(start, stop, step):
    """只用于已证明非负边界/正步长；也接受TIR整数表达式。调用方负责空域门。"""
    if type(start) is int and type(stop) is int and stop <= start:
        return 0
    return (stop - start - 1) // step + 1


def validate_whiles(program):
    """计数器必须在进入循环时可证明恰为0：由此界定步数，且计数更新不会溢出。"""
    intervals = Intervals(program.body, program.parameters)
    for statement in walk(program.body):
        if not isinstance(statement, WhileLoop):
            continue
        if statement.halving:
            # 非负整数的 `v //= 2` 每轮严格递减，必然终止；不依赖区间证明。
            continue
        entry = dict(statement.entry).get(statement.counter)
        if statement.grid_stride:
            step, stop = statement.grid_stride
            if entry is None or not intervals.grid_origin(entry, 'grid'):
                raise Unsupported(f'line {statement.line}: grid-stride while初值必须来自cuda.grid(1)')
            if not intervals.grid_origin(step, 'gridsize'):
                raise Unsupported(f'line {statement.line}: grid-stride while步长必须来自cuda.gridsize(1)')
            stop_range = intervals.get(stop)
            step_range = intervals.get(step)
            if stop_range is None or stop_range[0] < 1 or step_range is None \
                    or step_range[0] < 1 \
                    or stop_range[1] - 1 + step_range[1] > INTEGER_LIMITS['int64'][1]:
                raise Unsupported(f'line {statement.line}: grid-stride while上界或末次自增无法证明无溢出')
            continue
        if entry is None or intervals.get(entry) != (0, 0):
            raise Unsupported(f'line {statement.line}: while 计数器进入循环时必须为整数0: '
                              + statement.counter)


def validate_divisions(program):
    """整数//%要求除数不为零；证明为正还顺带排除负数取整与INT_MIN//-1。"""
    intervals = Intervals(program.body, program.parameters)
    for statement in walk(program.body):
        if not isinstance(statement, Instruction) or statement.op != 'op' \
                or statement.value[0] not in ('//', '%'):
            continue
        interval = intervals.get(statement.args[1])
        if interval is None or interval[0] < 1:
            raise Unsupported(f'line {statement.line}: 除数无法证明为严格正整数')


def range_direction(statement, intervals):
    """三参range仅在完整契约证明后返回方向，验证与生成共用。"""
    step = intervals.get(statement.step)
    if step is not None and step[1] < 0:
        if any(value.dtype != 'int64' for value in
               (statement.start, statement.stop, statement.step, statement.index)):
            raise Unsupported(f'line {statement.line}: 负步长range仅支持int64边界及索引')
        start = intervals.get(statement.start)
        stop = intervals.get(statement.stop)
        if start is None or start[0] < 0:
            raise Unsupported(f'line {statement.line}: 负步长range start 无法证明为非负整数且无溢出')
        if stop is None or stop[0] < -1:
            raise Unsupported(f'line {statement.line}: 负步长range stop 无法证明不小于-1且无溢出')
        minimum, maximum = INTEGER_LIMITS['int64']
        if step[0] <= minimum:
            raise Unsupported(f'line {statement.line}: 负步长range step 取负无法证明无溢出')
        if start[1] - stop[0] > maximum:
            raise Unsupported(f'line {statement.line}: 负步长range 跨度无法证明在int64范围内')
        return -1
    for label, value, minimum in (('start', statement.start, 0),
                                  ('stop', statement.stop, 0),
                                  ('step', statement.step, 1)):
        interval = intervals.get(value)
        if interval is None or interval[0] < minimum:
            raise Unsupported(f'line {statement.line}: range {label} 无法证明'
                              + ('为严格正整数且无溢出' if minimum else '为非负整数且无溢出'))
    return 1


def validate_ranges(program):
    intervals = Intervals(program.body, program.parameters)
    for statement in walk(program.body):
        if isinstance(statement, SerialLoop) and statement.step is not None:
            range_direction(statement, intervals)
