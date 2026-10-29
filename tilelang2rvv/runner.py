"""Build real RV64GCV ELF files and execute them with bounded QEMU runs."""
from pathlib import Path
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parent
CLANG = Path(os.environ.get('RVV_CLANG', str(Path.home() / 'scoop/apps/llvm/current/bin/clang.exe')))
QEMU = Path(os.environ.get('RVV_QEMU', str(Path.home() / 'scoop/apps/qemu/current/qemu-system-riscv64.exe')))

RUNTIME = r'''
static void print_text(const char *s) {
    volatile unsigned char *uart = (volatile unsigned char *)0x10000000;
    while (*s) {
        while (!(uart[5] & 0x20)) {}
        uart[0] = (unsigned char)*s++;
    }
}
__attribute__((noreturn)) static void finish(int failed) {
    *(volatile unsigned int *)0x100000 = failed ? 0x13333 : 0x5555;
    for (;;) __asm__ volatile("wfi");
}
void *memset(void *p, int c, unsigned long n) {
    unsigned char *d = p;
    for (unsigned long i = 0; i < n; i++) d[i] = (unsigned char)c;
    return p;
}
'''

def command(args, timeout=60):
    result = subprocess.run([str(x) for x in args], cwd=ROOT,
                            capture_output=True, timeout=timeout)
    stdout = result.stdout.decode('utf-8', errors='replace')
    stderr = result.stderr.decode('utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {args}\n{stdout}\n{stderr}')
    return stdout, stderr


def build_and_run(name, source, vlens=(128, 256, 512), expected='PASS'):
    if not CLANG.is_file() or not QEMU.is_file():
        raise RuntimeError('Set RVV_CLANG and RVV_QEMU to installed tool paths')
    out = ROOT / 'build' / name
    out.mkdir(parents=True, exist_ok=True)
    cfile = out / 'kernel.c'
    elf = out / 'kernel.elf'
    asm = out / 'kernel.s'
    cfile.write_text(source, encoding='utf-8')
    flags = ['--target=riscv64-unknown-elf', '-march=rv64gcv', '-mabi=lp64d',
             '-mcmodel=medany', '-O2', '-ffreestanding', '-fno-builtin',
             '-fno-vectorize', '-fno-slp-vectorize', '-msmall-data-limit=0']
    command([CLANG, *flags, '-S', cfile, '-o', asm])
    command([CLANG, *flags, '-nostdlib', '-static', '-Wl,--no-relax',
             '-Wl,-T,' + str(ROOT / 'baremetal.ld'), ROOT / 'start.S', cfile, '-o', elf])
    results = []
    for vlen in vlens:
        stdout, stderr = command([QEMU, '-M', 'virt', '-m', '128M', '-smp', '1',
                                  '-cpu', f'rv64,v=true,vlen={vlen},elen=64',
                                  '-bios', 'none', '-kernel', elf, '-display', 'none',
                                  '-monitor', 'none', '-serial', 'stdio'], timeout=20)
        if expected not in stdout or 'TRAP' in stdout:
            raise AssertionError(f'VLEN={vlen}: missing success marker: {stdout}\n{stderr}')
        results.append({'vlen': vlen, 'returncode': 0, 'stdout': stdout.strip()})
    report = {'name': name, 'elf': str(elf), 'assembly': str(asm), 'results': results}
    (out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
