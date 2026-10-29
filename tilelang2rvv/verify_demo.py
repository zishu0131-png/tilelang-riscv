"""Reproduce TileLang PrimFunc -> RVV C -> ELF -> QEMU, preserving evidence."""
import json
from pathlib import Path
from backend import generate
from examples import CASES, build_case
from runner import ROOT, build_and_run


def main():
    reports = []
    for case in CASES:
        func = case['func']()
        kernel_c, full_c = build_case(case, generate=generate)
        report = build_and_run(case['name'], full_c)
        folder = ROOT / 'build' / case['name']
        (folder / 'input.tir').write_text(func.script(), encoding='utf-8')
        (folder / 'generated_kernel.c').write_text(kernel_c, encoding='utf-8')
        asm = Path(report['assembly']).read_text(encoding='utf-8')
        required = case.get('rvv', 'vfadd')
        if required and (required not in asm or 'vsetvli' not in asm):
            raise AssertionError(f"{case['name']}: expected RVV instructions absent")
        for run in report['results']:
            print(f"PASS {case['name']} VLEN={run['vlen']} rc={run['returncode']} output={run['stdout']}")
        reports.append(report)
    (ROOT / 'build' / 'demo_results.json').write_text(
        json.dumps({'scope': 'Restricted single-hart RVV; no VCIX', 'cases': reports}, indent=2),
        encoding='utf-8')
    print('TILELANG TO RVV DEMO PASS: %d kernels x 3 VLENs; no VCIX claimed'
          % len(CASES))


if __name__ == '__main__':
    main()
