# -*- coding: utf-8 -*-
"""线程IR→TileLang；语句和读取顺序不交给表达式替换推测。"""
from thread_ir import TensorArg, Instruction, Branch, SerialLoop, WhileLoop
from thread_ranges import Intervals, range_direction, walk


def lower_thread(program):
    parameters = dict(program.parameters)
    # 共享/局部数组的 dtype：它们不是形参，store 的目标用分配时的声明类型。
    array_dtypes = {s.target.name: s.value[1] for s in walk(program.body)
                    if isinstance(s, Instruction) and s.op in ('alloc_shared', 'alloc_local')}
    specifications = []
    for name, spec in program.parameters:
        constructor = (f'_tl_TensorArg({spec.shape!r}, {spec.dtype!r})'
                       if isinstance(spec, TensorArg) else f'_tl_ScalarArg({spec.dtype!r})')
        if spec.value_range is not None:      # 张量与标量共用同一构造形态
            constructor = constructor[:-1] + f', value_range={spec.value_range!r})'
        specifications.append(f'({name!r}, {constructor})')
    parameter_code = '(' + ', '.join(specifications) + ',)'

    def read(value):
        # 逐线程局部槽由_tl_T.alloc_var提供，TileLang的rval本身会解引用成BufferLoad[0]。
        return value.name

    def cast(value, dtype):
        return f'_tl_T.cast({read(value)}, {dtype!r})'

    annotations = []
    for name, spec in program.parameters:
        annotation = f'_tl_T.Tensor({spec.shape!r}, {spec.dtype!r})' if isinstance(spec, TensorArg) else '_tl_T.' + spec.dtype
        annotations.append(name + ': ' + annotation)
    lines = ['import tilelang as _tl_tilelang', 'import tilelang.language as _tl_T',
             'from thread_ir import TensorArg as _tl_TensorArg, ScalarArg as _tl_ScalarArg',
             'from thread_runtime import ThreadRunner as _tl_ThreadRunner', '',
             '# 原始线程与写回语义；仅对访问有效且无数据竞态的源程序提供转换。',
             '@_tl_tilelang.jit(out_idx=[], pass_configs={"tl.disable_warp_specialized": True}, compile_flags=["--fmad=false"])',
             'def _tl_build(_tl_grid, _tl_block):',
             '    @_tl_T.prim_func',
             '    def _tl_main(' + ', '.join(annotations) + '):',
             '        with _tl_T.Kernel(_tl_grid[0], _tl_grid[1], _tl_grid[2], threads=_tl_block) as (_tl_bx, _tl_by, _tl_bz):']
    if any(isinstance(s, Instruction) and s.op == 'atomic_exch' for s in walk(program.body)):
        lines[5] = '# 受限原子写：目标无混合访问；调用期间禁止外部并发访问张量。'
        lines.insert(4, 'from tilelang.language.atomic import atomic_store as _tl_atomic_store')
    if any(isinstance(s, Instruction) and s.op == 'atomic_add' for s in walk(program.body)):
        lines[5] = '# 受限原子计数：目标无混合元素访问；调用期间禁止外部并发访问张量。'
    if any(isinstance(s, SerialLoop) and s.step is not None for s in walk(program.body)):
        intervals = Intervals(program.body, program.parameters)
        lines.insert(4, 'from thread_ranges import positive_trip_count as _tl_trip_count')
    if any(isinstance(s, Instruction) and s.op == 'infinity' for s in walk(program.body)):
        lines.insert(4, 'from tilelang.language.tir import infinity as _tl_infinity')
    if any(isinstance(s, Instruction) and s.op == 'numpy_int32' for s in walk(program.body)):
        lines.insert(4, '# np.int32调用方前置条件：输入有限且向零截断后在int32范围内；未自动检查。')
    for axis in range(3):
        lines.append(f'            _tl_t{axis} = _tl_T.get_thread_binding({axis})')
    for register in program.registers:
        lines.append(f'            {register.name} = _tl_T.alloc_var({register.dtype!r})')

    def emit_suite(statements, indent):
        before = len(lines)
        emit(statements, indent)
        if len(lines) == before:
            lines.append(' ' * indent + 'pass')

    def emit(statements, indent):
        def put(text):
            lines.append(' ' * indent + text)
        for statement in statements:
            if isinstance(statement, Branch):
                if not statement.yes and not statement.no:
                    continue
                if statement.yes:
                    put('if ' + read(statement.condition) + ':')
                    emit_suite(statement.yes, indent + 4)
                else:
                    put('if not ' + read(statement.condition) + ':')
                    emit_suite(statement.no, indent + 4)
                    continue
                if statement.no:
                    put('else:')
                    emit_suite(statement.no, indent + 4)
                continue
            if isinstance(statement, WhileLoop):
                # 判断suite只存一份：入口和回边各发射一次，因此每个判断点都真正重算条件，
                # 也不会把条件里的读取提前到循环外。
                emit(statement.tests, indent)
                put('while ' + read(statement.condition) + ':')
                emit_suite((*statement.body, *statement.tests), indent + 4)
                continue
            if isinstance(statement, SerialLoop):
                if not statement.body:
                    continue
                index = statement.index
                if statement.step is not None:
                    start = cast(statement.start, 'int64')
                    stop = cast(statement.stop, 'int64')
                    step = cast(statement.step, 'int64')
                    iterator = index.name + '_index'
                    if range_direction(statement, intervals) == -1:
                        put(f'if {start} > {stop}:')
                        count = f'_tl_trip_count({stop}, {start}, -{step})'
                        expression = f'{start} - {iterator} * (-{step})'
                    else:
                        put(f'if {stop} > {start}:')
                        count = f'_tl_trip_count({start}, {stop}, {step})'
                        expression = f'{start} + {iterator} * {step}'
                    lines.append(' ' * (indent + 4) + f'for {iterator} in _tl_T.serial(_tl_T.int64(0), {count}):')
                    lines.append(' ' * (indent + 8) + f'{index.name} = _tl_T.cast({expression}, {index.dtype!r})')
                    emit(statement.body, indent + 8)
                    continue
                start = cast(statement.start, index.dtype)
                stop = cast(statement.stop, index.dtype)
                # 显式非空门避免TIR负extent；range边界此前已经求值一次。
                put(f'if {stop} > {start}:')
                iterator = index.name + '_index'
                lines.append(' ' * (indent + 4) + f'for {iterator} in _tl_T.serial({start}, {stop}):')
                lines.append(' ' * (indent + 8) + f'{index.name} = {iterator}')
                emit(statement.body, indent + 8)
                continue
            target, args, value = statement.target, statement.args, statement.value
            if statement.op == 'sync':
                put('_tl_T.sync_threads()')
                continue
            if statement.op == 'break':
                put('break')
                continue
            if statement.op in ('alloc_shared', 'alloc_local'):
                dims, dtype = value
                put(f'{target.name} = _tl_T.{"alloc_shared" if statement.op == "alloc_shared" else "alloc_local"}'
                    f'({dims!r}, {dtype!r})')
                continue
            if statement.op == 'atomic_add':
                put(f'_tl_T.atomic_add({value}[{read(args[0])}], {cast(args[1], "float64")}, memory_order="relaxed", return_prev=False)')
                continue
            if statement.op == 'atomic_exch':
                name, index = value
                put(f'_tl_atomic_store({name}[{index}], {cast(args[0], "int32")}, memory_order="relaxed")')
                continue
            if statement.op == 'store':
                name = value
                indices = ', '.join(read(x) for x in args[:-1])
                spec = parameters.get(name)
                put(f'{name}[{indices}] = {cast(args[-1], spec.dtype if spec is not None else array_dtypes[name])}')
                continue
            if statement.op == 'infinity':
                expression = f'_tl_infinity({target.dtype!r})'
            elif statement.op == 'const':
                expression = f'_tl_T.cast({value!r}, {target.dtype!r})'
            elif statement.op in ('copy', 'numpy_int32'):
                expression = cast(args[0], target.dtype)
            elif statement.op == 'load':
                expression = value + '[' + ', '.join(read(x) for x in args) + ']'
            elif statement.op == 'not':
                expression = f'({read(args[0])} == _tl_T.bool(False))'
            elif statement.op == 'grid':
                block_index = ('_tl_bx', '_tl_by', '_tl_bz')[value]
                expression = f'(_tl_T.cast({block_index}, "int64") * _tl_T.cast(_tl_block[{value}], "int64") + _tl_T.cast(_tl_t{value}, "int64"))'
            elif statement.op == 'gridsize':
                expression = f'(_tl_T.cast(_tl_block[{value}], "int64") * _tl_T.cast(_tl_grid[{value}], "int64"))'
            elif statement.op == 'dim':
                kind, axis = value
                expression = {'threadIdx': f'_tl_t{axis}',
                              'blockIdx': ('_tl_bx', '_tl_by', '_tl_bz')[axis],
                              'blockDim': f'_tl_block[{axis}]', 'gridDim': f'_tl_grid[{axis}]'}[kind]
                expression = f'_tl_T.cast({expression}, "int32")'
            elif statement.op == 'op':
                symbol, dtypes = value
                operands = [cast(v, d) for v, d in zip(args, dtypes)]
                if symbol in ('pos', 'neg'):
                    expression = ('+' if symbol == 'pos' else '-') + operands[0]
                elif symbol in ('//', '%'):
                    # Python的//%是floor语义，必须用floordiv/floormod而不是trunc版。
                    primitive = 'floordiv' if symbol == '//' else 'floormod'
                    expression = f'_tl_T.{primitive}({operands[0]}, {operands[1]})'
                elif symbol in ('min', 'max'):
                    # 操作数都已物化成局部槽，所以重复引用不会重复求值。
                    # Python语义是"第二个严格更优才取第二个"（b if b<a else a），
                    # 顺序反了会让NaN与±0.0的取舍与Python/Numba不同。
                    compare = '<' if symbol == 'min' else '>'
                    expression = (f'_tl_T.if_then_else({operands[1]} {compare} {operands[0]}, '
                                  f'{operands[1]}, {operands[0]})')
                elif symbol in ('abs', 'sin', 'cos', 'exp', 'sqrt', 'log', 'atan2', 'tanh', 'isinf', 'pow',
                                'fmod', 'floor'):
                    expression = f'_tl_T.{symbol}({", ".join(operands)})'
                else:
                    expression = f'({operands[0]} {symbol} {operands[1]})'
            else:
                raise ValueError('unknown thread instruction: ' + statement.op)
            # 对已分配的local.var再赋值即buffer_store，不能用name[0]=。
            put(f'{target.name} = {expression}')
    emit(program.body, 12)
    lines += ['    return _tl_main', '', f'def {program.name}(*, grid, block):',
              f'    return _tl_ThreadRunner(_tl_build, {parameter_code}, grid, block)', '']
    return '\n'.join(lines)
