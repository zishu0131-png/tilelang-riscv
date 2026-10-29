"""test_backend.py -- tests for the TileLang -> RVV backend.

Three groups:

1. structural tests -- what ``generate()`` produces for the examples, checked
   as text (signatures, intrinsics, no ``main`` / ``_start``).
2. rejection tests -- the strictness contract.  Each one asserts *which*
   exception comes out and why, so a future loosening of the converter shows
   up as a failure here rather than as silently wrong code.
3. end-to-end tests -- the generated C is really compiled by clang for
   rv64gcv and really executed by QEMU at VLEN 128/256/512.  Plus mutation
   tests that rewrite the generated kernel to be wrong and assert the harness
   *fails*, so that a PASS is evidence and not just a marker the harness
   prints unconditionally.

Run:  python -m unittest -v test_backend      (from this directory)
"""

from __future__ import annotations

import os
import re
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import examples  # noqa: E402
from backend import (  # noqa: E402
    ConversionError,
    UnsupportedError,
    ValidationError,
    generate,
    supported_summary,
)

try:
    import runner  # noqa: E402
except Exception as exc:  # pragma: no cover
    runner = None
    _RUNNER_ERR = exc
else:
    _RUNNER_ERR = None

import tilelang  # noqa: E402,F401  (needed before tvm.tirx is importable)
import tilelang.language as T  # noqa: E402
from tilelang import tvm  # noqa: E402

tirx = tvm.tirx

BLOCK = examples.BLOCK

# a top-level C function definition
_FUNC_DEF = re.compile(r"^(?:static\s+)?(?:void|int|long|float|double)\s+(\w+)\s*\(")


def _defs(src):
    return {m.group(1) for m in (_FUNC_DEF.match(l) for l in src.splitlines()) if m}


def _toolchain_ok():
    return (
        runner is not None
        and runner.CLANG.is_file()
        and runner.QEMU.is_file()
    )


def _case(name):
    for c in examples.CASES:
        if c["name"] == name:
            return c
    raise KeyError(name)


# ==========================================================================
# 1. structure of the generated C
# ==========================================================================
class TestGeneratedStructure(unittest.TestCase):
    def setUp(self):
        self.kernels = {c["name"]: examples.build_case(c)[0] for c in examples.CASES}

    def test_signatures_and_parameter_order(self):
        """The caller needs these exactly: parameter order == prim_func order."""
        self.assertIn("void vadd_exact(const float *A, const float *B, float *C) {",
                      self.kernels["vadd_exact"])
        self.assertIn("void vadd_tail(int n, const float *A, const float *B, float *C) {",
                      self.kernels["vadd_tail"])
        self.assertIn("void gemm_tile(const float *A, const float *B, float *C) {",
                      self.kernels["gemm_tile"])
        self.assertIn("void gemm_multi_k(const float *A, const float *B, float *C) {",
                      self.kernels["gemm_multi_k"])

    def test_readonly_buffers_are_const_and_outputs_are_not(self):
        """Every pointer parameter except the last (the output) is const.

        Written position-wise rather than by name so it holds for the
        double-precision case and for kernels with a different arity.
        """
        for name, src in self.kernels.items():
            sig = [l for l in src.splitlines() if l.startswith("void ")][0]
            params = sig[sig.index("(") + 1:sig.rindex(")")]
            ptrs = [p.strip() for p in params.split(",") if "*" in p]
            self.assertTrue(ptrs, name)
            for p in ptrs[:-1]:
                self.assertTrue(p.startswith("const "),
                                "%s: input %r is not const" % (name, p))
            self.assertFalse(ptrs[-1].startswith("const "),
                             "%s: output is const" % name)

    def test_no_main_no_start_only_the_kernel(self):
        for name, src in self.kernels.items():
            defs = _defs(src)
            self.assertIn(name, defs, name)
            self.assertNotIn("main", defs, name)
            self.assertNotIn("_start", defs, name)
            self.assertTrue(defs <= {name, "tl_floordiv", "tl_floormod"},
                            "%s: unexpected definitions %s" % (name, defs))

    def test_exact_vadd_has_no_guard(self):
        src = self.kernels["vadd_exact"]
        self.assertIn("__riscv_vle32_v_f32m1", src)
        self.assertIn("__riscv_vfadd_vv_f32m1", src)
        self.assertIn("__riscv_vse32_v_f32m1", src)
        self.assertIn("__riscv_vsetvl_e32m1", src)
        # nothing to guard: no bound bookkeeping in the loop at all
        self.assertNotIn("_rem", src)
        self.assertNotIn("break;", src)

    def test_2d_indices_linearise_to_a_unit_stride(self):
        """A 2-D access written through a flattened lane must come out as one
        contiguous load, not a gather.

        The front end emits ``C[p // W, p % W]``; folding the strides gives
        ``(p // W) * W + p % W``, and only the simplifier turns that back into
        ``p``.  If it did not, the region would stop looking unit-stride and be
        refused -- so this also covers the per-lane ``Bind`` inlining.
        """
        src = self.kernels["vadd_2d"]
        self.assertIn("__riscv_vle32_v_f32m1(A + ", src)
        self.assertIn("__riscv_vle32_v_f32m1(B + ", src)
        self.assertIn("__riscv_vse32_v_f32m1(C + ", src)
        self.assertNotIn("vlse", src)

    def test_tail_vadd_clamps_vl_to_the_guard(self):
        src = self.kernels["vadd_tail"]
        self.assertIn("_rem0", src)
        self.assertIn("if (_rem0 <= 0) break;", src)
        self.assertIn("if (_rem0 < _cnt0) _cnt0 = _rem0;", src)
        self.assertIn("__riscv_vsetvl_e32m1((size_t)_cnt0)", src)
        # the kernel drives the grid off the runtime parameter
        self.assertIn("(int)tl_floordiv", src)

    def test_gemm_becomes_an_rvv_micro_kernel(self):
        src = self.kernels["gemm_tile"]
        self.assertIn("__riscv_vfmacc_vf_f32m1", src)
        self.assertIn("tl.tileop.gemm 32x32x64", src)
        self.assertNotIn("__riscv_vle32_v_f32m1_m", src)  # no masked forms needed
        # allocated tiles are core-local static arrays, not stack frames
        self.assertIn("static float A_s[2048];", src)
        self.assertIn("static float C_f[1024];", src)

    def test_multi_k_gemm_keeps_one_accumulator(self):
        """Two K steps share one C_f.

        The clear runs once *before* the ko loop and the gemm sits *inside* it,
        so step 2 accumulates onto step 1.  A converter that re-cleared the
        accumulator each step would produce a kernel that still contains a
        vfmacc and would pass a naive structure check -- the numeric harness is
        what actually pins this down (see the mutant test below).
        """
        src = self.kernels["gemm_multi_k"]
        ko = src.index("for (int ko = 0; ko < 2; ++ko)")
        self.assertLess(src.index("= 0;"), ko, "accumulator cleared inside the K loop")
        self.assertGreater(src.index("__riscv_vfmacc"), ko, "gemm not inside the K loop")
        self.assertEqual(src.count("__riscv_vfmacc"), 1)

    def test_ragged_gemm_clips_the_reads_and_the_writes(self):
        """A ragged shape needs a run-time boundary.

        A read past the end is predicated and padded with zero; a write past the
        end simply stops.  ``gemm_ragged_k`` pads *both* operands because its K
        direction is the ragged one.
        """
        rm = self.kernels["gemm_ragged_m"]
        self.assertIn("? A[", rm)          # predicated read ...
        self.assertIn(": 0;", rm)          # ... padded with zero
        rk = self.kernels["gemm_ragged_k"]
        self.assertIn("? A[", rk)
        self.assertIn("? B[", rk)

    def test_exact_shapes_generate_no_clipping_at_all(self):
        """The clip must cost nothing when the tiles divide.

        An always-on predicate would be a silent slowdown of every good kernel,
        so the exact cases have to stay byte-for-byte predicate-free.
        """
        for name in ("vadd_exact", "gemm_tile", "gemm_multi_k", "gemm_trans_a",
                     "gemm_trans_b", "gemm_trans_ab"):
            src = self.kernels[name]
            self.assertNotIn("_eff", src, name)
            self.assertNotIn("? A[", src, name)
            self.assertNotIn("? B[", src, name)

    def test_transpose_orientations_use_the_right_addressing(self):
        """Transposing an operand is a *layout* change and the emitted code has
        to follow it.

        ``gemm_trans_a`` stores A as (K, M), so the scalar A read steps down a
        K-row -- m is the inner index -- while B stays unit-stride.
        ``gemm_trans_b`` stores B as (N, K), so the micro-kernel's walk over n
        becomes a *strided* load.  The numbers themselves are pinned by the
        end-to-end harness and its mutant; this catches the code shape drifting.

        (The region/layout consistency check in ``_gemm`` is defence in depth:
        the TileLang front end already refuses a tile whose shape contradicts
        the M/N/K it derived, so a real PrimFunc cannot reach that branch.)
        """
        ta = self.kernels["gemm_trans_a"]
        tb = self.kernels["gemm_trans_b"]
        tab = self.kernels["gemm_trans_ab"]
        self.assertIn("A_s[((0 + _k8) * 32) + (0 + _m7)]", ta)   # k-major A read
        self.assertNotIn("vlse", ta)                              # B unit-stride
        self.assertIn("__riscv_vlse32_v_f32m1", tb)
        self.assertIn("__riscv_vlse32_v_f32m1", tab)
        self.assertIn("A_s[((0 + _k8) * 32) + (0 + _m7)]", tab)
        # the strided load steps one B row: BK=16 floats = 64 bytes
        self.assertIn(", 64, _vl10);", tb)

    def test_no_vcix_claim_anywhere(self):
        for name, src in self.kernels.items():
            low = src.lower()
            self.assertNotIn("vcix", low.replace("no vcix is used or claimed", ""), name)
            self.assertNotIn("vtcvt", low, name)
            self.assertNotIn("vsm3", low, name)

    def test_header_can_be_dropped(self):
        src = generate(examples.vadd_exact(), func_name="x", include_header=False)
        self.assertNotIn("#include", src)
        self.assertIn("void x(", src)

    def test_supported_summary_is_informative(self):
        s = supported_summary()
        self.assertIn("gemm", s)
        self.assertIn("guard", s)

    def test_harness_literals_agree_with_the_constants(self):
        """The harnesses inline their numbers; keep them honest."""
        self.assertEqual(examples.N_EXACT, 1024)
        self.assertEqual(examples.BLOCK, 128)
        self.assertEqual(examples.N_TAIL, 1000)
        self.assertNotEqual(examples.N_TAIL % examples.BLOCK, 0)
        self.assertEqual((examples.M, examples.N, examples.K), (64, 64, 64))
        for case in examples.CASES:
            h = case["harness"]()
            for want in examples.HARNESS_LITERALS[case["name"]]:
                self.assertIn(str(want), h, case["name"])

    def test_generated_file_is_a_translation_unit_tail_only(self):
        """No main/_start *definitions* and no other entry point."""
        for name, src in self.kernels.items():
            for line in src.splitlines():
                self.assertFalse(
                    re.match(r"^void\s+(main|_start)\s*\(", line),
                    "%s defines %s" % (name, line))


# ==========================================================================
# 2. rejections -- the strictness contract
# ==========================================================================
class TestRejections(unittest.TestCase):
    """Each case must be refused, with the documented exception type."""

    def test_error_hierarchy(self):
        self.assertTrue(issubclass(UnsupportedError, ConversionError))
        self.assertTrue(issubclass(ValidationError, ConversionError))

    def test_non_primfunc_is_a_type_error(self):
        with self.assertRaises(TypeError):
            generate(42)
        with self.assertRaises(TypeError):
            generate(None)

    def test_unsupported_elementwise_op(self):
        @T.prim_func
        def k(A: T.Tensor((1024,), "float32"), C: T.Tensor((1024,), "float32")):
            with T.Kernel(8, threads=BLOCK) as bx:
                for i in T.Parallel(BLOCK):
                    C[bx * BLOCK + i] = T.exp(A[bx * BLOCK + i])

        with self.assertRaises(UnsupportedError) as ctx:
            generate(k)
        # the region falls back to the scalar path, which still has no exp
        self.assertIn("T.exp", str(ctx.exception))

    def test_broadcast_store_is_not_an_elementwise_map(self):
        @T.prim_func
        def k(A: T.Tensor((1024,), "float32"), S: T.Tensor((8,), "float32")):
            with T.Kernel(8, threads=BLOCK) as bx:
                for i in T.Parallel(BLOCK):
                    S[bx] = A[bx * BLOCK + i]

        with self.assertRaises(UnsupportedError) as ctx:
            generate(k)
        self.assertIn("does not depend on the parallel loop variable",
                      str(ctx.exception))

    def test_stencil_dependence_is_refused(self):
        """A[i] = A[i-1] + 1 must not become a vector chunk."""

        @T.prim_func
        def k(A: T.Tensor((1024,), "float32")):
            with T.Kernel(8, threads=BLOCK) as bx:
                for i in T.Parallel(BLOCK):
                    A[bx * BLOCK + i] = A[bx * BLOCK + i - 1] + 1.0

        with self.assertRaises(UnsupportedError) as ctx:
            generate(k)
        self.assertIn("loop-carried dependence", str(ctx.exception))

    def test_unguarded_runtime_grid_is_refused(self):
        """A ceildiv grid with no guard cannot be proven in bounds."""

        @T.prim_func
        def k(n: T.int32, A: T.Tensor((1024,), "float32"),
              C: T.Tensor((1024,), "float32")):
            with T.Kernel(T.ceildiv(n, BLOCK), threads=BLOCK) as bx:
                for i in T.Parallel(BLOCK):
                    C[bx * BLOCK + i] = A[bx * BLOCK + i]

        with self.assertRaises(ValidationError) as ctx:
            generate(k)
        msg = str(ctx.exception)
        self.assertIn("cannot prove", msg)
        self.assertIn("guard", msg)

    def test_guard_that_overshoots_the_buffer_is_refused(self):
        """A constant bound past the end of the buffer is caught statically."""

        @T.prim_func
        def k(A: T.Tensor((1024,), "float32"), C: T.Tensor((1024,), "float32")):
            with T.Kernel(8, threads=BLOCK) as bx:
                for i in T.Parallel(BLOCK):
                    if bx * BLOCK + i < 2048:
                        C[bx * BLOCK + i] = A[bx * BLOCK + i]

        with self.assertRaises(ValidationError) as ctx:
            generate(k)
        self.assertIn("only has 1024 elements", str(ctx.exception))

    def test_gemm_without_initialisation_is_refused(self):
        @T.prim_func
        def k(A: T.Tensor((64, 64), "float32"), B: T.Tensor((64, 64), "float32"),
              C: T.Tensor((64, 64), "float32")):
            with T.Kernel(2, 2, threads=128) as (bx, by):
                A_s = T.alloc_shared((32, 64), "float32")
                B_s = T.alloc_shared((64, 32), "float32")
                C_f = T.alloc_fragment((32, 32), "float32")
                T.copy(A[by * 32, 0], A_s)
                T.copy(B[0, bx * 32], B_s)
                T.gemm(A_s, B_s, C_f)
                T.copy(C_f, C[by * 32, bx * 32])

        with self.assertRaises(ValidationError) as ctx:
            generate(k)
        self.assertIn("nothing initialises it", str(ctx.exception))

    def test_mixed_dtypes_are_refused(self):
        @T.prim_func
        def k(A: T.Tensor((1024,), "float32"), C: T.Tensor((1024,), "int32")):
            with T.Kernel(8, threads=BLOCK) as bx:
                for i in T.Parallel(BLOCK):
                    C[bx * BLOCK + i] = A[bx * BLOCK + i] + 1.0

        with self.assertRaises(UnsupportedError) as ctx:
            generate(k)
        self.assertIn("dtype", str(ctx.exception))

    def test_bounds_helpers_are_sound(self):
        """_upper/_lower guard a safety check: an under-estimate compiles OOB.

        These two must over/under-estimate respectively, or return None to say
        "cannot prove".  Brute force the real values over the loop's range and
        insist the bounds bracket them.
        """
        from backend import RvvBackend

        be = RvvBackend(examples.vadd_exact())
        bx = tirx.Var("bx", "int32")
        be.loop_max[bx] = 7
        be.loop_min[bx] = 0
        I = lambda v: tirx.IntImm("int32", v)  # noqa: E731
        cases = [
            (tirx.Add(tirx.Mul(bx, I(8)), I(3)), lambda v: v * 8 + 3),
            (tirx.Mul(bx, I(4)), lambda v: v * 4),
            (tirx.Div(bx, I(8)), lambda v: int(v / 8)),
            (tirx.FloorDiv(bx, I(8)), lambda v: v // 8),
            (tirx.Mod(bx, I(3)), lambda v: v - int(v / 3) * 3),
            (tirx.FloorMod(bx, I(3)), lambda v: v % 3),
            (tirx.Min(bx, I(5)), lambda v: min(v, 5)),
            (tirx.Max(bx, I(5)), lambda v: max(v, 5)),
            (tirx.Sub(bx, I(0)), lambda v: v),
        ]
        for expr, py in cases:
            ub, lb = be._upper(expr), be._lower(expr)
            with self.subTest(expr=str(expr)):
                self.assertIsNotNone(ub)
                self.assertIsNotNone(lb)
                for v in range(0, 8):
                    self.assertGreaterEqual(ub, py(v), "upper under-estimates")
                    self.assertLessEqual(lb, py(v), "lower over-estimates")

    def test_bounds_refuse_what_they_cannot_prove(self):
        from backend import RvvBackend

        be = RvvBackend(examples.vadd_exact())
        bx = tirx.Var("bx", "int32")
        other = tirx.Var("d", "int32")
        be.loop_max[bx] = 7
        be.loop_min[bx] = 0
        be.loop_max[other] = 5
        be.loop_min[other] = 1
        I = lambda v: tirx.IntImm("int32", v)  # noqa: E731
        # a variable divisor would need the *smallest* divisor: refuse
        self.assertIsNone(be._upper(tirx.Div(bx, other)))
        self.assertIsNone(be._lower(tirx.Div(bx, other)))
        self.assertIsNone(be._upper(tirx.FloorDiv(bx, other)))
        # subtraction of a non-zero constant can go negative: refuse
        self.assertIsNone(be._upper(tirx.Sub(bx, I(3))))
        self.assertIsNone(be._upper(I(-4)))
        self.assertEqual(be._lower(I(-4)), -4)
        self.assertFalse(be._non_negative(tirx.Sub(bx, I(1))))
        self.assertFalse(be._non_negative(I(-1)))
        self.assertTrue(be._non_negative(bx))

    def test_negative_offset_is_refused(self):
        """Walking off the front of a buffer is as bad as off the back."""

        @T.prim_func
        def k(A: T.Tensor((1024,), "float32"), C: T.Tensor((1024,), "float32")):
            with T.Kernel(8, threads=BLOCK) as bx:
                for i in T.Parallel(BLOCK):
                    C[bx * BLOCK + i - 4] = A[bx * BLOCK + i]

        with self.assertRaises(ValidationError) as ctx:
            generate(k)
        self.assertIn("cannot prove", str(ctx.exception))

    def test_region_entirely_past_the_end_is_refused(self):
        """A tile whose *smallest* start is already past the end cannot be
        clipped -- there is nothing left to clip to -- so it is a mistake in
        the addressing, not a boundary case, and must be refused."""
        @T.prim_func
        def k(A: T.Tensor((96, 48), "float32"), B: T.Tensor((48, 64), "float32"),
              C: T.Tensor((96, 64), "float32")):
            with T.Kernel(2, 2, threads=128) as (bx, by):
                A_s = T.alloc_shared((32, 16), "float32")
                B_s = T.alloc_shared((16, 32), "float32")
                C_f = T.alloc_fragment((32, 32), "float32")
                T.clear(C_f)
                T.copy(A[by * 32 + 96, 0], A_s)      # starts at 96 or 128, M is 96
                T.copy(B[0, bx * 32], B_s)
                T.gemm(A_s, B_s, C_f)
                T.copy(C_f, C[by * 32, bx * 32])

        with self.assertRaises(ValidationError) as ctx:
            generate(k)
        self.assertIn("starts at or past the end", str(ctx.exception))

    def test_run_time_grid_cannot_be_clipped_and_is_refused(self):
        """A run-time grid extent means the converter cannot bound where a tile
        starts, so it cannot clip it either -- refuse rather than guess."""
        @T.prim_func
        def k(n: T.int32, A: T.Tensor((96, 48), "float32"),
              B: T.Tensor((48, 64), "float32"), C: T.Tensor((96, 64), "float32")):
            with T.Kernel(2, T.ceildiv(n, 64), threads=128) as (bx, by):
                A_s = T.alloc_shared((64, 16), "float32")
                B_s = T.alloc_shared((16, 32), "float32")
                C_f = T.alloc_fragment((64, 32), "float32")
                T.clear(C_f)
                for ko in T.Pipelined(3, num_stages=1):
                    T.copy(A[by * 64, ko * 16], A_s)
                    T.copy(B[ko * 16, bx * 32], B_s)
                    T.gemm(A_s, B_s, C_f)
                T.copy(C_f, C[by * 64, bx * 32])

        with self.assertRaises(ValidationError) as ctx:
            generate(k)
        self.assertIn("cannot bound", str(ctx.exception))

    def test_thread_index_addressing_is_refused(self):
        """thdIdx-dependent indexing cannot be honoured on one hart."""

        @T.prim_func
        def k(A: T.Tensor((1024,), "float32"), C: T.Tensor((1024,), "float32")):
            with T.Kernel(8, threads=BLOCK) as bx:
                tx = T.get_thread_binding()
                C[tx] = A[tx]

        with self.assertRaises(UnsupportedError) as ctx:
            generate(k)
        self.assertIn("launch_thread_idx", str(ctx.exception))

    def test_sync_is_refused(self):
        @T.prim_func
        def k(A: T.Tensor((1024,), "float32"), C: T.Tensor((1024,), "float32")):
            with T.Kernel(8, threads=BLOCK) as bx:
                A_s = T.alloc_shared((BLOCK,), "float32")
                for i in T.Parallel(BLOCK):
                    A_s[i] = A[bx * BLOCK + i]
                T.sync_threads()
                for i in T.Parallel(BLOCK):
                    C[bx * BLOCK + i] = A_s[i]

        with self.assertRaises(UnsupportedError) as ctx:
            generate(k)
        self.assertIn("not in the supported subset", str(ctx.exception))

    def test_unknown_annotation_is_refused(self):
        """Unknown annotations are metadata the converter does not understand.

        A TileLang ``T.Parallel`` / ``T.Kernel`` body offers no way to attach a
        novel annotation from the front end, so this drives the guard directly:
        the point is that an annotation the converter has not been taught does
        not get silently ignored.
        """
        from backend import RvvBackend

        be = RvvBackend(examples.vadd_exact())
        # the annotations the converter does understand are accepted
        be._check_annotations({"tl.launch_threads": [128, 1, 1], "num_stages": 1},
                              "SBlock")
        for bad in ({"tl.launch_threads": [1, 1, 1], "mystery": 1},
                    {"made_up_annotation": 7}):
            with self.assertRaises(UnsupportedError) as ctx:
                be._check_annotations(bad, "SBlock")
            self.assertIn("unknown annotation", str(ctx.exception))

    def test_function_attributes_are_metadata_not_semantics(self):
        """global_symbol is read for the C name; other attrs do not mislead us."""
        fn = examples.vadd_exact().with_attr("global_symbol", "renamed")
        self.assertIn("void renamed(", generate(fn))
        fn2 = examples.vadd_exact().with_attr("made_up_annotation", 7)
        # a function-level attribute is not a statement: it is not interpreted,
        # and it cannot change what the function body means
        self.assertEqual(generate(fn2), generate(examples.vadd_exact()))

    def test_positive_controls_still_convert(self):
        """The rejections above must not have made the good cases fail."""
        for c in examples.CASES:
            src = generate(c["func"]())
            self.assertIn("void " + c["name"] + "(", src)

    def test_staging_and_same_lane_inplace_are_accepted(self):
        """The dependence rules must not be blunt: these two are safe."""

        @T.prim_func
        def inplace(A: T.Tensor((1024,), "float32"), C: T.Tensor((1024,), "float32")):
            with T.Kernel(8, threads=BLOCK) as bx:
                for i in T.Parallel(BLOCK):
                    C[bx * BLOCK + i] = C[bx * BLOCK + i] * 0.5 + A[bx * BLOCK + i]

        src = generate(inplace)
        self.assertIn("__riscv_vfmadd", src + "__riscv_vfmadd_ok")  # vfadd of (vmul, vle)
        self.assertIn("__riscv_vfadd_vv_f32m1", src)

        @T.prim_func
        def staged(A: T.Tensor((1024,), "float32"), C: T.Tensor((1024,), "float32")):
            with T.Kernel(8, threads=BLOCK) as bx:
                A_s = T.alloc_shared((BLOCK,), "float32")
                for i in T.Parallel(BLOCK):
                    A_s[i] = A[bx * BLOCK + i]
                for i in T.Parallel(BLOCK):
                    C[bx * BLOCK + i] = A_s[i]

        src = generate(staged)
        self.assertIn("static float A_s[128];", src)


# ==========================================================================
# 3. end to end: clang -> elf -> QEMU, at three VLENs
# ==========================================================================
@unittest.skipUnless(_toolchain_ok(), "RVV_CLANG / RVV_QEMU not installed")
class TestEndToEnd(unittest.TestCase):
    maxDiff = None

    def _run(self, name, source, expected="PASS"):
        return runner.build_and_run(name, source, vlens=(128, 256, 512),
                                   expected=expected)

    def _assert_run_fails(self, name, source, why):
        """A wrong kernel must fail the run *and* say FAIL.

        runner.py raises AssertionError when the success marker is missing and
        RuntimeError when QEMU exits non-zero (the harness's finish(1) path).
        Either proves the harness rejected the kernel; for the non-zero exit we
        insist the FAIL text really came out, so the negative control is not
        just a crash.
        """
        try:
            self._run(name, source)
        except AssertionError as exc:
            self.assertIn("missing success marker", str(exc), why)
            return
        except RuntimeError as exc:
            self.assertIn("FAIL", str(exc), why)
            return
        self.fail("%s: a wrong kernel ran to completion and reported PASS" % why)

    def test_examples_pass_on_a_real_rvv_hart(self):
        for case in examples.CASES:
            with self.subTest(case=case["name"]):
                kernel_c, full_c = examples.build_case(case)
                # the kernel on its own is a leaf function: no entry point
                self.assertNotIn("main(", kernel_c.replace("no main,", ""))
                report = self._run(case["name"], full_c)
                self.assertEqual(len(report["results"]), 3)
                for r in report["results"]:
                    self.assertIn("PASS", r["stdout"])
                    self.assertNotIn("TRAP", r["stdout"])

    def test_mutant_wrong_op_fails(self):
        """If the kernel computes the wrong thing the harness must notice."""
        case = _case("vadd_exact")
        kernel_c, _ = examples.build_case(case)
        self.assertIn("__riscv_vfadd_vv_f32m1", kernel_c)
        bad = kernel_c.replace("__riscv_vfadd_vv_f32m1", "__riscv_vfsub_vv_f32m1")
        self.assertNotEqual(bad, kernel_c)
        full = runner.RUNTIME + "\n" + bad + "\n" + case["harness"]()
        self._assert_run_fails("vadd_exact_mutant_op", full, "wrong operator")

    def test_mutant_tail_guard_removed_fails(self):
        """Removing the tail clamp writes past N and the harness catches it."""
        case = _case("vadd_tail")
        kernel_c, _ = examples.build_case(case)
        self.assertIn("if (_rem0 < _cnt0) _cnt0 = _rem0;", kernel_c)
        bad = kernel_c.replace("if (_rem0 < _cnt0) _cnt0 = _rem0;",
                               "/* mutant: tail clamp removed */")
        self.assertNotIn("if (_rem0 < _cnt0)", bad)
        full = runner.RUNTIME + "\n" + bad + "\n" + case["harness"]()
        self._assert_run_fails("vadd_tail_mutant_noguard", full, "guard removed")

    def test_mutant_multi_k_dropping_a_step_fails(self):
        """A kernel that runs one of the two K steps is short by a whole product
        term; the harness must report FAIL and not a smaller PASS."""
        case = _case("gemm_multi_k")
        kernel_c, _ = examples.build_case(case)
        self.assertIn("ko < 2", kernel_c)
        bad = kernel_c.replace("ko < 2", "ko < 1")
        self.assertNotEqual(bad, kernel_c)
        full = runner.RUNTIME + "\n" + bad + "\n" + case["harness"]()
        self._assert_run_fails("gemm_multi_k_mutant_onestep", full, "dropped K step")

    def test_mutant_trans_b_unit_stride_load_fails(self):
        """B is stored (N, K); reading it as if consecutive n were contiguous
        picks the wrong element.  If the strided load were not doing its job the
        number would be wrong, and the harness has to say so."""
        case = _case("gemm_trans_b")
        kernel_c, _ = examples.build_case(case)
        self.assertIn(", 64, _vl10);", kernel_c)
        bad = kernel_c.replace(", 64, _vl10);", ", 4, _vl10);")
        self.assertNotEqual(bad, kernel_c)
        full = runner.RUNTIME + "\n" + bad + "\n" + case["harness"]()
        self._assert_run_fails("gemm_trans_b_mutant_unit_stride", full, "unit-stride B")

    def test_mutant_ragged_m_unclipped_store_fails(self):
        """Without the store clip the last row-tile writes past the end of C;
        the harness's guard band is what catches it."""
        case = _case("gemm_ragged_m")
        kernel_c, _ = examples.build_case(case)
        clip = "for (int _c15 = 0; _c15 < _eff17; ++_c15) {"
        self.assertIn(clip, kernel_c)
        bad = kernel_c.replace(clip, "for (int _c15 = 0; _c15 < 64; ++_c15) {")
        full = runner.RUNTIME + "\n" + bad + "\n" + case["harness"]()
        self._assert_run_fails("gemm_ragged_m_mutant_noclip", full, "unclipped store")

    def test_mutant_ragged_k_nonzero_pad_fails(self):
        """The K-direction pad is summed into every real output element, so a
        non-zero pad corrupts the whole result -- not just an edge."""
        case = _case("gemm_ragged_k")
        kernel_c, _ = examples.build_case(case)
        self.assertIn(": 0;", kernel_c)
        bad = kernel_c.replace(": 0;", ": 1.0f;")
        self.assertNotEqual(bad, kernel_c)
        full = runner.RUNTIME + "\n" + bad + "\n" + case["harness"]()
        self._assert_run_fails("gemm_ragged_k_mutant_pad", full, "non-zero K pad")

    def test_same_elf_runs_at_every_vlen(self):
        """One binary, three VLENs: length-agnostic code, not baked-in widths."""
        case = _case("vadd_exact")
        _, full = examples.build_case(case)
        report = self._run("vadd_exact_vlen_sweep", full)
        vlens = [r["vlen"] for r in report["results"]]
        self.assertEqual(vlens, [128, 256, 512])
        self.assertTrue(all("PASS" in r["stdout"] for r in report["results"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
