"""Integration tests for the bare-metal runner, independent of IR conversion."""
import unittest
from runner import RUNTIME, build_and_run


class RunnerTests(unittest.TestCase):
    def test_success_at_three_vlens(self):
        report = build_and_run('harness_smoke', RUNTIME +
                               'void kmain(void){print_text("HARNESS PASS");finish(0);}',
                               expected='HARNESS PASS')
        self.assertEqual([r['vlen'] for r in report['results']], [128, 256, 512])

    def test_failure_exit_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, r'Command failed \(1\)'):
            build_and_run('negative_exit', RUNTIME +
                          'void kmain(void){print_text("EXPECTED FAILURE");finish(1);}',
                          vlens=(128,))

    def test_trap_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, r'Command failed \(2\)'):
            build_and_run('negative_trap', RUNTIME +
                          'void kmain(void){__asm__ volatile(".word 0");finish(0);}',
                          vlens=(128,))


if __name__ == '__main__':
    unittest.main(verbosity=2)
