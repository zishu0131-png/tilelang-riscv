# -*- coding: utf-8 -*-
"""验收反例：错误程序不能被接受，合法归约不能被特化规则挡住。"""
import unittest
from convert import convert, detect_pattern, parse_source, Unsupported


def map_source(expression, header='', bindings=''):
    return (header + '\n@cuda.jit\ndef apply(A, C):\n'
            '    i, j = cuda.grid(2)\n'
            '    if i < C.shape[0] and j < C.shape[1]:\n' + bindings +
            f'        C[i, j] = {expression}\n')


def reduction_source(expression, operation='sum'):
    initial = {'sum': '0.0', 'max': '-math.inf', 'min': 'math.inf'}[operation]
    update = f'total += {expression}' if operation == 'sum' else f'total = {operation}(total, {expression})'
    return (f'import math\n@cuda.jit\ndef apply(A, C):\n'
            '    i, j = cuda.grid(2)\n'
            '    if i < C.shape[0] and j < C.shape[1]:\n'
            f'        total = {initial}\n'
            '        for k in range(A.shape[2]):\n'
            f'            {update}\n'
            '        C[i, j] = total\n')


class AcceptanceTests(unittest.TestCase):
    def test_direct_and_bound_mixed_maps_require_square_shape(self):
        from composition import Node, describe
        for source in (map_source('A[i,j] + A[j,i]'),
                       map_source('v + A[j,i]', bindings='        v = A[i,j]\n')):
            with self.subTest(source=source):
                spec = detect_pattern(parse_source(source))
                self.assertEqual(spec.pattern, 'elementwise')
                self.assertIs(detect_pattern(spec), spec)
                self.assertEqual(spec.layout['equalities'], (('_tl_shape[1]', '_tl_shape[0]'),))
                generated = convert(source)
                self.assertIn('source requires equal dimension lengths', generated)
                compile(generated, '<mixed-map>', 'exec')
                _, dims, _, _ = describe(Node(source, {}, (7,7)))
                self.assertEqual(dims, {'A': (7,7), 'C': (7,7)})
                with self.assertRaises(ValueError):
                    describe(Node(source, {}, (7,9)))
                with self.assertRaises(Unsupported):
                    convert(source.replace('A[j,i]', 'A[i,i]'))

    def test_calls_cannot_ignore_local_shadowing(self):
        for name, expression, header in (
                ('max', 'max(A[i,j], 0.0)', ''),
                ('min', 'min(A[i,j], 0.0)', ''),
                ('abs', 'abs(A[i,j])', ''),
                ('math', 'math.exp(A[i,j])', 'import math'),
                ('np', 'np.exp(A[i,j])', 'import numpy as np')):
            source = map_source(expression, header, f'        {name} = A[i,j]\n')
            with self.subTest(name=name), self.assertRaisesRegex(Unsupported, '绑定|遮蔽'):
                convert(source)

    def test_module_spelling_is_not_a_binding(self):
        for header, expression in (
                ('import numpy as cuda', 'A[i,j]'),
                ('import math as np', 'np.exp(A[i,j])'),
                ('', 'math.exp(A[i,j])'),
                ('import math', 'np.exp(A[i,j])')):
            with self.subTest(header=header, expression=expression), self.assertRaisesRegex(Unsupported, '导入|绑定'):
                convert(map_source(expression, header))

    def test_function_parameter_axis_and_accumulator_cannot_shadow_calls(self):
        source = map_source('max(A[i,j], 0.0)')
        cases = [source.replace('def apply(', 'def max('),
                 source.replace('A', 'max'),
                 source.replace('i, j', 'abs, j').replace('i <', 'abs <').replace('[i,j]', '[abs,j]'),
                 reduction_source('A[i,j,k]', 'max').replace('total', 'max')]
        for source in cases:
            with self.subTest(source=source), self.assertRaises(Unsupported):
                convert(source)

    def test_rank_three_product_reductions_are_not_gemm(self):
        for operation in ('sum', 'max', 'min'):
            source = reduction_source('A[i,j,k] * A[i,j,k]', operation)
            with self.subTest(operation=operation):
                spec = detect_pattern(parse_source(source))
                self.assertEqual(spec.pattern, 'reduce_' + operation)
                generated = convert(source)
                self.assertIn('T.reduce_' + operation, generated)
                self.assertNotIn('T.gemm', generated)
                compile(generated, '<acceptance>', 'exec')

    def test_valid_non_gemm_products_use_validated_reduction(self):
        from test_convert import SOURCE
        for old, new in (('B[k, j]', 'B[k, i]'),
                         ('A[i, k]', 'A[i, k, j]'), ('B[k, j]', 'B[k]')):
            with self.subTest(new=new):
                source = SOURCE.replace(old, new)
                self.assertEqual(detect_pattern(parse_source(source)).pattern, 'reduce_sum')
                self.assertIn('T.reduce_sum', convert(source))

    def test_canonical_imports_remain_supported(self):
        for header, expression in (('import math', 'math.exp(A[i,j])'),
                                   ('import numpy as np', 'np.sqrt(A[i,j])'),
                                   ('import numpy', 'numpy.tanh(A[i,j])')):
            with self.subTest(header=header):
                self.assertEqual(detect_pattern(parse_source(map_source(expression, header))).pattern,
                                 'elementwise')


if __name__ == '__main__':
    unittest.main()
