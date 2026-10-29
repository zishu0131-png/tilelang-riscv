# -*- coding: utf-8 -*-
"""公共接口回归测试：不导入 Numba、TileLang 或 GPU 运行时。"""
import ast
import unittest
from pathlib import Path

from convert import Unsupported, convert, detect_pattern, parse_source


SOURCE = '''@cuda.jit
def mm(A, B, C):
    i, j = cuda.grid(2)
    tmp = 0.0
    for k in range(A.shape[1]):
        tmp += A[i, k] * B[k, j]
    C[i, j] = tmp
'''


class CaptureTests(unittest.TestCase):
    def test_original_demo_converts_to_python(self):
        source = Path(__file__).with_name('demo_input.py').read_text(encoding='utf-8')
        spec = detect_pattern(parse_source(source))
        self.assertEqual(spec.pattern, 'gemm')
        compile(convert(source), '<generated>', 'exec')

    def test_extra_factor_is_preserved_instead_of_dropped(self):
        # 系数不再被静默丢弃：改由通用归约族正确表达（精度不套GEMM模板）。
        source = SOURCE.replace('A[i, k] * B[k, j]', '2.0 * A[i, k] * B[k, j]')
        spec = detect_pattern(parse_source(source))
        self.assertEqual(spec.pattern, 'reduce_sum')
        self.assertIn('2.0', convert(source))

    def test_range_domain_is_captured_without_losing_arguments(self):
        source = SOURCE.replace('range(A.shape[1])', 'range(0, A.shape[1], 1)')
        lp = parse_source(source).loops[0]
        self.assertEqual((lp.start, lp.extent, lp.step), ('0', 'A.shape[1]', '1'))
        self.assertEqual(lp.init_value, 0.0)
        self.assertEqual(lp.update_op, '+')
        self.assertEqual(lp.update_expr, 'A[i, k] * B[k, j]')

    def test_unsupported_statements_cannot_disappear(self):
        changes = [
            ('tmp = 0.0', 'tmp = 0.0\n    tmp = 3.0'),
            ('tmp += A[i, k] * B[k, j]', 'tmp += A[i, k] * B[k, j]\n        tmp += A[i, k] * B[k, j]'),
            ('C[i, j] = tmp', 'C[i, j] = tmp\n    A[i, j] = tmp'),
            ('for k in range(A.shape[1]):', 'for k in custom_range(A.shape[1]):'),
        ]
        for old, new in changes:
            with self.subTest(new=new), self.assertRaises(Unsupported):
                convert(SOURCE.replace(old, new))


class PatternTests(unittest.TestCase):
    def test_four_transposes_and_product_orders_have_same_axis_roles(self):
        cases = [
            ('A[i, k]', 'B[k, j]', 'A.shape[1]', False, False),
            ('A[i, k]', 'B[j, k]', 'A.shape[1]', False, True),
            ('A[k, i]', 'B[k, j]', 'A.shape[0]', True, False),
            ('A[k, i]', 'B[j, k]', 'A.shape[0]', True, True),
        ]
        for left, right, extent, ta, tb in cases:
            for expression in (f'{left} * {right}', f'{right} * {left}'):
                source = SOURCE.replace('A[i, k] * B[k, j]', expression).replace('A.shape[1]', extent)
                with self.subTest(expression=expression):
                    spec = detect_pattern(parse_source(source))
                    self.assertEqual((spec.lhs, spec.rhs), ('A', 'B'))
                    self.assertEqual(spec.axis_roles, {'i': 'M', 'j': 'N', 'k': 'K'})
                    self.assertEqual((spec.transpose_A, spec.transpose_B), (ta, tb))
                    self.assertEqual(spec.pattern, 'gemm')

    def test_invalid_axes_domains_and_parameters_are_rejected(self):
        changes = [
            ('A[i, k]', 'A[k, k]'),
            ('C[i, j]', 'C[i, i]'), ('C[i, j]', 'C[i, k]'),
            ('range(A.shape[1])', 'range(A.shape[1], 0, -1)'),
            ('range(A.shape[1])', 'range(1, A.shape[1])'),
            ('range(A.shape[1])', 'range(0, A.shape[1], 2)'),
            ('range(A.shape[1])', 'range(A.shape[0])'),
            ('range(A.shape[1])', 'range(K)'),
            ('def mm(A, B, C)', 'def mm(A, B, C, K)'),
            ('C[i, j] = tmp', 'A[i, j] = tmp'),
            ('i, j = cuda.grid(2)', 'i, i = cuda.grid(2)'),
            ('tmp', 'i'),
        ]
        for old, new in changes:
            with self.subTest(new=new), self.assertRaises(Unsupported):
                convert(SOURCE.replace(old, new))

    def test_guard_is_output_boundary_not_algorithmic_mask(self):
        body = SOURCE.split('    tmp =', 1)[1]
        template = SOURCE.split('    tmp =', 1)[0] + '    if GUARD:\n' + '\n'.join(
            '    ' + line for line in ('    tmp =' + body).splitlines()) + '\n'
        for guard in ('i < C.shape[0] and j < C.shape[1]',
                      'j < B.shape[1] and i < A.shape[0]'):
            self.assertEqual(detect_pattern(parse_source(template.replace('GUARD', guard))).guard_kind, 'bounds')
        for guard in ('i < 100 and j < 200', 'i < j',
                      'i < C.shape[1] and j < C.shape[0]',
                      'i < C.shape[0] or j < C.shape[1]'):
            with self.subTest(guard=guard), self.assertRaises(Unsupported):
                convert(template.replace('GUARD', guard))


class GenerationTests(unittest.TestCase):
    def test_transposed_buffers_and_gemm_flags_are_generated_together(self):
        source = SOURCE.replace('A[i, k] * B[k, j]', 'B[j, k] * A[k, i]').replace('A.shape[1]', 'A.shape[0]')
        out = convert(source)
        compile(out, '<generated>', 'exec')
        self.assertIn('A: T.Tensor((K, M)', out)
        self.assertIn('B: T.Tensor((N, K)', out)
        self.assertIn('T.alloc_shared((block_K, block_M)', out)
        self.assertIn('T.alloc_shared((block_N, block_K)', out)
        self.assertIn('T.copy(A[ko * block_K, by * block_M]', out)
        self.assertIn('T.copy(B[bx * block_N, ko * block_K]', out)
        self.assertIn('transpose_A=True, transpose_B=True', out)

    def test_generated_parameter_order_and_output_index_agree(self):
        out = convert(SOURCE.replace('def mm(A, B, C)', 'def mm(B, C, A)'))
        tree = ast.parse(out)
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        main = next(n for n in factory.body if isinstance(n, ast.FunctionDef))
        self.assertEqual([a.arg for a in main.args.args], ['B', 'C', 'A'])
        self.assertIn('out_idx=[1]', out)

    def test_constant_reduction_extent_becomes_factory_check(self):
        out = convert(SOURCE.replace('range(A.shape[1])', 'range(7)'))
        self.assertIn('K != 7', out)
        self.assertIn('raise ValueError', out)

    def test_output_axis_order_is_not_lost(self):
        source = SOURCE.replace('C[i, j]', 'C[j, i]')
        spec = detect_pattern(parse_source(source))
        self.assertEqual((spec.lhs, spec.rhs), ('B', 'A'))
        self.assertEqual((spec.transpose_A, spec.transpose_B), (True, True))
        compile(convert(source), '<generated>', 'exec')


class RejectionTests(unittest.TestCase):
    def test_unrepresented_syntax_is_rejected_with_domain_error(self):
        changes = [
            ('tmp = 0.0', 'tmp = False'), ('tmp = 0.0', 'tmp = 1.0'),
            ('tmp = 0.0', 'tmp = float(0)'),
            ('tmp +=', 'tmp -='), (' * B[k, j]', ' ** B[k, j]'),
            ('A[i, k]', 'A[i + 1, k]'), ('A[i, k]', 'A[i][k]'),
            ('A[i, k]', 'A[:, k]'), ('A[i, k]', 'A[0, k]'),
            ('C[i, j] = tmp', 'C = tmp'),
            ('range(A.shape[1])', 'range()'), ('range(A.shape[1])', 'range(stop=7)'),
            ('@cuda.jit', '@not_cuda.jit'), ('@cuda.jit', '@cuda.jit(device=True)'),
            ('def mm(A, B, C)', 'def mm(A, B, C=None)'),
            ('def mm(A, B, C)', 'def mm(A: float, B, C)'),
        ]
        for old, new in changes:
            with self.subTest(new=new), self.assertRaises(Unsupported):
                convert(SOURCE.replace(old, new))

    def test_source_bindings_cannot_silently_change_recognized_operations(self):
        cases = [
            'range = custom_range\n' + SOURCE,
            SOURCE.replace('tmp', 'range'),
            SOURCE.replace('def mm(', 'def T('),
            SOURCE.replace('A', 'A_shared'),
        ]
        for source in cases:
            with self.subTest(source=source), self.assertRaises(Unsupported):
                convert(source)

    def test_variable_renaming_and_explicit_unit_range_are_supported(self):
        source = SOURCE.replace('A', 'X').replace('B', 'Y').replace('C', 'Z')
        source = source.replace('range(X.shape[1])', 'range(0, Y.shape[0], 1)')
        spec = detect_pattern(parse_source(source))
        self.assertEqual((spec.lhs, spec.rhs, spec.store_target), ('X', 'Y', 'Z'))
        compile(convert(source), '<generated>', 'exec')

    def test_else_and_nested_guards_are_rejected(self):
        body = '\n'.join('    ' + line for line in SOURCE.splitlines()[3:])
        prefix = '\n'.join(SOURCE.splitlines()[:3]) + '\n'
        guarded = prefix + '    if i < C.shape[0] and j < C.shape[1]:\n' + body + '\n'
        for source in (guarded + '    else:\n        C[i, j] = 0.0\n',
                       guarded.replace('        tmp = 0.0', '        if i < j:\n            tmp = 0.0')):
            with self.subTest(source=source), self.assertRaises(Unsupported):
                convert(source)


if __name__ == '__main__':
    unittest.main()
