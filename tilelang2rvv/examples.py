"""examples.py -- TileLang programs handed to backend.py, plus their harnesses.

Every example is a genuine ``@T.prim_func`` written with ``T.Kernel`` /
``T.Parallel`` / ``T.gemm``.  Nothing here is a regex over source text: the
examples exist so that the converter can be exercised against the IR that
TileLang really produces.

Each example comes with a ``harness_*()`` function returning the C text of a
``kmain`` that calls the generated kernel, checks the result against a
straightforward reference and prints ``PASS`` / ``FAIL``.  ``build_case()``
glues ``runner.RUNTIME`` + generated kernel + harness together; the QEMU
runner in runner.py compiles and executes that on a real RV64G**V** hart.

The harnesses write their constants literally so the C text is plain and
readable; ``test_backend.py`` asserts those literals still agree with the
constants below.

The generated kernel itself never contains ``main`` or ``_start`` -- the
harness supplies ``kmain`` and runner.py supplies ``_start``.
"""

from __future__ import annotations

import os as _os

# The release does not bypass duplicate OpenMP runtime checks.

import tilelang.language as T  # noqa: E402

__all__ = [
    "vadd_exact",
    "vadd_tail",
    "vadd_2d",
    "vadd_f64",
    "rowsum",
    "rowmax",
    "gemm_tile",
    "gemm_multi_k",
    "gemm_trans_a",
    "gemm_trans_b",
    "gemm_trans_ab",
    "gemm_ragged_m",
    "gemm_ragged_n",
    "gemm_ragged_k",
    "harness_vadd_exact",
    "harness_vadd_tail",
    "harness_vadd_2d",
    "harness_vadd_f64",
    "harness_rowsum",
    "harness_rowmax",
    "harness_gemm_tile",
    "harness_gemm_multi_k",
    "harness_gemm_trans_a",
    "harness_gemm_trans_b",
    "harness_gemm_trans_ab",
    "harness_gemm_ragged_m",
    "harness_gemm_ragged_n",
    "harness_gemm_ragged_k",
    "build_case",
    "CASES",
    "N_EXACT",
    "BLOCK",
    "N_TAIL",
    "M",
    "N",
    "K",
    "BM",
    "BN",
    "BK",
    "BK2",
    "TR_M",
    "TR_N",
    "TR_K",
    "TR_BM",
    "TR_BN",
    "TR_BK",
]

# ---------------------------------------------------------------------------
# sizes
# ---------------------------------------------------------------------------
N_EXACT = 1024        # exact: 1024 == 8 * BLOCK, the grid covers it exactly
BLOCK = 128           # T.Parallel extent == T.Kernel threads
N_TAIL = 1000         # deliberately NOT a multiple of BLOCK -> tail guard
M = 64
N = 64
K = 64
BM = 32
BN = 32
BK = 64
BK2 = 32             # K step for the multi-K example: K // BK2 == 2 steps


# ---------------------------------------------------------------------------
# 1. exact vector add -- no guard needed, the converter proves the grid fits
# ---------------------------------------------------------------------------
def vadd_exact():
    """C = A + B over the whole 1024-element buffer, no guard written."""

    @T.prim_func
    def vadd_exact(
        A: T.Tensor((N_EXACT,), "float32"),
        B: T.Tensor((N_EXACT,), "float32"),
        C: T.Tensor((N_EXACT,), "float32"),
    ):
        with T.Kernel(N_EXACT // BLOCK, threads=BLOCK) as bx:
            for i in T.Parallel(BLOCK):
                C[bx * BLOCK + i] = A[bx * BLOCK + i] + B[bx * BLOCK + i]

    return vadd_exact


# ---------------------------------------------------------------------------
# 2. tail -- N is a runtime parameter and NOT a multiple of BLOCK, so the
#    TileLang program writes an explicit guard.  The converter turns that
#    guard into a clamped vl (an exact tail) instead of a full-width run that
#    would walk past the end of the buffer.
# ---------------------------------------------------------------------------
def vadd_tail():
    """Same as vadd_exact but the TileLang source guards the last block."""

    @T.prim_func
    def vadd_tail(
        n: T.int32,
        A: T.Tensor((N_EXACT,), "float32"),
        B: T.Tensor((N_EXACT,), "float32"),
        C: T.Tensor((N_EXACT,), "float32"),
    ):
        with T.Kernel(T.ceildiv(n, BLOCK), threads=BLOCK) as bx:
            for i in T.Parallel(BLOCK):
                if bx * BLOCK + i < n:
                    C[bx * BLOCK + i] = A[bx * BLOCK + i] + B[bx * BLOCK + i]

    return vadd_tail


# ---------------------------------------------------------------------------
# 2b. 2-D elementwise over a flattened lane -- the shape the tile-level front
#     ends actually emit: a per-lane Bind and 2-D indices that only linearise
#     back to a unit stride if the offsets are folded and simplified.
# ---------------------------------------------------------------------------
V2D = 32


def vadd_2d():
    """C = A + B over a 32x32 tensor, addressed through a flattened lane."""

    @T.prim_func
    def vadd_2d(
        A: T.Tensor((V2D, V2D), "float32"),
        B: T.Tensor((V2D, V2D), "float32"),
        C: T.Tensor((V2D, V2D), "float32"),
    ):
        with T.Kernel(V2D * V2D // BLOCK, threads=BLOCK) as bx:
            for i in T.Parallel(BLOCK):
                p = bx * BLOCK + i
                if p < V2D * V2D:
                    C[p // V2D, p % V2D] = A[p // V2D, p % V2D] + B[p // V2D, p % V2D]

    return vadd_2d


# ---------------------------------------------------------------------------
# 2c. float64 elementwise -- the e64m1 vector path
# ---------------------------------------------------------------------------
def vadd_f64():
    """C = A + B over float64: exercises the e64m1 intrinsics."""

    @T.prim_func
    def vadd_f64(
        A: T.Tensor((256,), "float64"),
        B: T.Tensor((256,), "float64"),
        C: T.Tensor((256,), "float64"),
    ):
        with T.Kernel(2, threads=BLOCK) as bx:
            for i in T.Parallel(BLOCK):
                C[bx * BLOCK + i] = A[bx * BLOCK + i] + B[bx * BLOCK + i]

    return vadd_f64


# ---------------------------------------------------------------------------
# 3. tiled gemm -- T.gemm over fp32 shared tiles, emitted as an RVV vfmacc
#    micro-kernel.  The tile counts divide the shapes exactly, so every
#    T.copy is statically in bounds (the converter refuses anything else).
# ---------------------------------------------------------------------------
def gemm_tile():
    """C = A @ B, 64x64 @ 64x64, 32x32 tiles, one K step."""

    @T.prim_func
    def gemm_tile(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(N // BN, M // BM, threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), "float32")
            B_s = T.alloc_shared((BK, BN), "float32")
            C_f = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(K // BK, num_stages=1):
                T.copy(A[by * BM, ko * BK], A_s)
                T.copy(B[ko * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * BM, bx * BN])

    return gemm_tile


# ---------------------------------------------------------------------------
# 4. multi-K gemm -- same shapes as gemm_tile but the K axis is split into two
#    steps.  This is the case that checks the fp32 accumulator survives across
#    K steps: each T.gemm must *accumulate* into C_f, not overwrite it.  A
#    converter that cleared the accumulator per step would still pass the
#    one-step gemm_tile and fail here.
# ---------------------------------------------------------------------------
def gemm_multi_k():
    """C = A @ B, 64x64 @ 64x64, 32x32 tiles, two K steps of 32."""

    @T.prim_func
    def gemm_multi_k(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(N // BN, M // BM, threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK2), "float32")
            B_s = T.alloc_shared((BK2, BN), "float32")
            C_f = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(K // BK2, num_stages=1):
                T.copy(A[by * BM, ko * BK2], A_s)
                T.copy(B[ko * BK2, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * BM, bx * BN])

    return gemm_multi_k


# ---------------------------------------------------------------------------
# 5-7. transposed gemm.  Non-square shapes and non-square tiles so the stored
#      layout is actually observable: with M == K a transposed tile is square
#      and a converter that ignored the transpose could still "work".  Here the
#      reduction axis (K=48) differs from both M and N, so reading A or B down
#      the wrong axis reads a different number entirely.
# ---------------------------------------------------------------------------
TR_M, TR_N, TR_K = 96, 64, 48
TR_BM, TR_BN, TR_BK = 32, 32, 16


def gemm_trans_a():
    """C = A^T @ B with A stored as (K, M); A's tile is (BK, BM)."""

    @T.prim_func
    def gemm_trans_a(
        A: T.Tensor((TR_K, TR_M), "float32"),
        B: T.Tensor((TR_K, TR_N), "float32"),
        C: T.Tensor((TR_M, TR_N), "float32"),
    ):
        with T.Kernel(TR_N // TR_BN, TR_M // TR_BM, threads=128) as (bx, by):
            A_s = T.alloc_shared((TR_BK, TR_BM), "float32")
            B_s = T.alloc_shared((TR_BK, TR_BN), "float32")
            C_f = T.alloc_fragment((TR_BM, TR_BN), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(TR_K // TR_BK, num_stages=1):
                T.copy(A[ko * TR_BK, by * TR_BM], A_s)
                T.copy(B[ko * TR_BK, bx * TR_BN], B_s)
                T.gemm(A_s, B_s, C_f, transpose_A=True)
            T.copy(C_f, C[by * TR_BM, bx * TR_BN])

    return gemm_trans_a


def gemm_trans_b():
    """C = A @ B^T with B stored as (N, K); B's tile is (BN, BK).

    This is the orientation that forces a strided vector load: the micro-kernel
    walks n, and inside B's tile consecutive n are a whole row apart.
    """

    @T.prim_func
    def gemm_trans_b(
        A: T.Tensor((TR_M, TR_K), "float32"),
        B: T.Tensor((TR_N, TR_K), "float32"),
        C: T.Tensor((TR_M, TR_N), "float32"),
    ):
        with T.Kernel(TR_N // TR_BN, TR_M // TR_BM, threads=128) as (bx, by):
            A_s = T.alloc_shared((TR_BM, TR_BK), "float32")
            B_s = T.alloc_shared((TR_BN, TR_BK), "float32")
            C_f = T.alloc_fragment((TR_BM, TR_BN), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(TR_K // TR_BK, num_stages=1):
                T.copy(A[by * TR_BM, ko * TR_BK], A_s)
                T.copy(B[bx * TR_BN, ko * TR_BK], B_s)
                T.gemm(A_s, B_s, C_f, transpose_B=True)
            T.copy(C_f, C[by * TR_BM, bx * TR_BN])

    return gemm_trans_b


def gemm_trans_ab():
    """C = A^T @ B^T: both operands transposed at once."""

    @T.prim_func
    def gemm_trans_ab(
        A: T.Tensor((TR_K, TR_M), "float32"),
        B: T.Tensor((TR_N, TR_K), "float32"),
        C: T.Tensor((TR_M, TR_N), "float32"),
    ):
        with T.Kernel(TR_N // TR_BN, TR_M // TR_BM, threads=128) as (bx, by):
            A_s = T.alloc_shared((TR_BK, TR_BM), "float32")
            B_s = T.alloc_shared((TR_BN, TR_BK), "float32")
            C_f = T.alloc_fragment((TR_BM, TR_BN), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(TR_K // TR_BK, num_stages=1):
                T.copy(A[ko * TR_BK, by * TR_BM], A_s)
                T.copy(B[bx * TR_BN, ko * TR_BK], B_s)
                T.gemm(A_s, B_s, C_f, transpose_A=True, transpose_B=True)
            T.copy(C_f, C[by * TR_BM, bx * TR_BN])

    return gemm_trans_ab


# ---------------------------------------------------------------------------
# 7b. row sum -- float32 input accumulated through a float64 tile, then
#     T.reduce.  This is the shape the upstream front end emits for a Numba
#     `acc = 0.0` reduction: the mixed float widths force the scalar fallback
#     inside T.Parallel, and T.reduce carries the ordered fold.
# ---------------------------------------------------------------------------
RS_ROWS = 64
RS_COLS = 128


def rowsum():
    """B[p] = sum_k A[p, k], accumulated in float64."""

    @T.prim_func
    def rowsum(
        A: T.Tensor((RS_ROWS, RS_COLS), "float32"),
        B: T.Tensor((RS_ROWS,), "float32"),
    ):
        with T.Kernel(RS_ROWS, threads=BLOCK) as p:
            values = T.alloc_fragment((1, RS_COLS), "float64")
            partial = T.alloc_fragment((1,), "float64")
            total = T.alloc_fragment((1,), "float64")
            for one in T.Parallel(1):
                total[one] = T.cast(0.0, "float64")
            for lane in T.Parallel(RS_COLS):
                k = lane
                if k < RS_COLS:
                    values[0, lane] = T.cast(A[p, k], "float64")
                else:
                    values[0, lane] = T.cast(0.0, "float64")
            T.reduce_sum(values, partial, dim=1, clear=True)
            for one in T.Parallel(1):
                total[one] = total[one] + partial[one]
            for one in T.Parallel(1):
                B[p] = T.cast(total[one], "float32")

    return rowsum


def rowmax():
    """B[p] = max_k A[p, k] -- same shape as rowsum but a max fold."""

    @T.prim_func
    def rowmax(
        A: T.Tensor((RS_ROWS, RS_COLS), "float32"),
        B: T.Tensor((RS_ROWS,), "float32"),
    ):
        with T.Kernel(RS_ROWS, threads=BLOCK) as p:
            values = T.alloc_fragment((1, RS_COLS), "float32")
            partial = T.alloc_fragment((1,), "float32")
            for lane in T.Parallel(RS_COLS):
                values[0, lane] = A[p, lane]
            T.reduce_max(values, partial, dim=1, clear=True)
            for one in T.Parallel(1):
                B[p] = partial[one]

    return rowmax


# ---------------------------------------------------------------------------
# 8-10. ragged gemm: the shapes are NOT multiples of the tile sizes, so every
#      case needs a run-time boundary.  Each one is ragged in exactly one
#      direction (M, N, K) so a failure points straight at that direction.
#
#      The tile grid still covers whole tiles -- TileLang's own T.copy has no
#      out-of-range predicate in this IR -- so the converter has to clip.  A
#      read past the end is padded with zero; a write past the end is not
#      performed at all.
# ---------------------------------------------------------------------------
def gemm_ragged_m():
    """M=96 with BM=64: the last row-tile is only half valid."""

    @T.prim_func
    def gemm_ragged_m(
        A: T.Tensor((96, 48), "float32"),
        B: T.Tensor((48, 64), "float32"),
        C: T.Tensor((96, 64), "float32"),
    ):
        with T.Kernel(64 // 32, T.ceildiv(96, 64), threads=128) as (bx, by):
            A_s = T.alloc_shared((64, 16), "float32")
            B_s = T.alloc_shared((16, 32), "float32")
            C_f = T.alloc_fragment((64, 32), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(48 // 16, num_stages=1):
                T.copy(A[by * 64, ko * 16], A_s)
                T.copy(B[ko * 16, bx * 32], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * 64, bx * 32])

    return gemm_ragged_m


def gemm_ragged_n():
    """N=96 with BN=64: the last column-tile is only half valid."""

    @T.prim_func
    def gemm_ragged_n(
        A: T.Tensor((64, 48), "float32"),
        B: T.Tensor((48, 96), "float32"),
        C: T.Tensor((64, 96), "float32"),
    ):
        with T.Kernel(T.ceildiv(96, 64), 64 // 32, threads=128) as (bx, by):
            A_s = T.alloc_shared((32, 16), "float32")
            B_s = T.alloc_shared((16, 64), "float32")
            C_f = T.alloc_fragment((32, 64), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(48 // 16, num_stages=1):
                T.copy(A[by * 32, ko * 16], A_s)
                T.copy(B[ko * 16, bx * 64], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * 32, bx * 64])

    return gemm_ragged_n


def gemm_ragged_k():
    """K=50 with BK=16: the last K step has 2 valid columns of 16.

    This is the direction where the padding matters numerically: the fourteen
    out-of-range columns are summed into *every* real output element, so a
    wrong pad (stale data, garbage) corrupts the whole result rather than an
    edge.  `gemm_ragged_m`/`_n` only need the store to stop early.
    """

    @T.prim_func
    def gemm_ragged_k(
        A: T.Tensor((64, 50), "float32"),
        B: T.Tensor((50, 64), "float32"),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(2, 2, threads=128) as (bx, by):
            A_s = T.alloc_shared((32, 16), "float32")
            B_s = T.alloc_shared((16, 32), "float32")
            C_f = T.alloc_fragment((32, 32), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(T.ceildiv(50, 16), num_stages=1):
                T.copy(A[by * 32, ko * 16], A_s)
                T.copy(B[ko * 16, bx * 32], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * 32, bx * 32])

    return gemm_ragged_k


# ---------------------------------------------------------------------------
# harnesses.  These are appended after the generated kernel and compiled
# together with runner.py's start.S, so they run on the QEMU RVV hart.
# print_text / finish come from runner.RUNTIME, which build_case() prepends.
# ---------------------------------------------------------------------------

_VADD_FILL = """
static float _A[1024];
static float _B[1024];
static float _C[1024];

static void fill_inputs(void) {
    for (int i = 0; i < 1024; ++i) {
        _A[i] = (float)(i % 7) * 0.5f + 1.0f;
        _B[i] = (float)(i % 5) - 2.0f;
        _C[i] = 0.0f;
    }
}
"""


def harness_vadd_2d():
    """kmain for vadd_2d: same arithmetic, the buffer is 32x32 (1024 floats)."""
    return _VADD_FILL + """
void kmain(void) {
    fill_inputs();
    vadd_2d(_A, _B, _C);
    int bad = -1;
    for (int i = 0; i < 1024; ++i) {
        if (_C[i] != _A[i] + _B[i]) { bad = i; break; }
    }
    if (bad >= 0) { print_text("FAIL\\n"); finish(1); }
    print_text("PASS\\n");
    finish(0);
}
"""


def harness_vadd_f64():
    """kmain for vadd_f64: double-precision inputs and output."""
    return """
static double _A[256];
static double _B[256];
static double _C[256];

void kmain(void) {
    for (int i = 0; i < 256; ++i) {
        _A[i] = (double)(i % 7) * 0.5 + 1.0;
        _B[i] = (double)(i % 5) - 2.0;
        _C[i] = 0.0;
    }
    vadd_f64(_A, _B, _C);
    int bad = -1;
    for (int i = 0; i < 256; ++i) {
        if (_C[i] != _A[i] + _B[i]) { bad = i; break; }
    }
    if (bad >= 0) { print_text("FAIL\\n"); finish(1); }
    print_text("PASS\\n");
    finish(0);
}
"""


def harness_rowsum():
    """kmain for rowsum: the reference sums in float32; integer inputs keep it
    exact, so the float64 accumulator must agree bit for bit."""
    return """
static float _A[64 * 128];
static float _B[64];

void kmain(void) {
    for (int i = 0; i < 64 * 128; ++i) _A[i] = (float)((i % 3) - 1);
    for (int i = 0; i < 64; ++i) _B[i] = -999.0f;
    rowsum(_A, _B);
    int bad = -1;
    for (int i = 0; i < 64; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < 128; ++j) acc += _A[i * 128 + j];
        if (_B[i] != acc) { bad = i; break; }
    }
    if (bad >= 0) { print_text("FAIL\\n"); finish(1); }
    print_text("PASS\\n");
    finish(0);
}
"""


def harness_rowmax():
    """kmain for rowmax: values span negative and positive so the identity
    element of the fold is actually exercised."""
    return """
static float _A[64 * 128];
static float _B[64];

void kmain(void) {
    for (int i = 0; i < 64 * 128; ++i) _A[i] = (float)((i % 13) - 6) * 0.5f;
    for (int i = 0; i < 64; ++i) _B[i] = 999.0f;
    rowmax(_A, _B);
    int bad = -1;
    for (int i = 0; i < 64; ++i) {
        float best = _A[i * 128];
        for (int j = 1; j < 128; ++j) if (_A[i * 128 + j] > best) best = _A[i * 128 + j];
        if (_B[i] != best) { bad = i; break; }
    }
    if (bad >= 0) { print_text("FAIL\\n"); finish(1); }
    print_text("PASS\\n");
    finish(0);
}
"""


def harness_vadd_exact():
    """kmain for vadd_exact: every element, no guard involved."""
    return _VADD_FILL + """
void kmain(void) {
    fill_inputs();
    vadd_exact(_A, _B, _C);
    int bad = -1;
    for (int i = 0; i < 1024; ++i) {
        if (_C[i] != _A[i] + _B[i]) { bad = i; break; }
    }
    if (bad >= 0) { print_text("FAIL\\n"); finish(1); }
    print_text("PASS\\n");
    finish(0);
}
"""


def harness_vadd_tail():
    """kmain for vadd_tail: only the first 1000 of 1024 elements are valid.

    It also asserts that elements 1000..1023 were left alone, which is what
    catches a tail guard that only *looks* right.
    """
    return _VADD_FILL + """
void kmain(void) {
    int n = 1000;
    fill_inputs();
    vadd_tail(n, _A, _B, _C);
    int bad = -1;
    for (int i = 0; i < 1000; ++i) {
        if (_C[i] != _A[i] + _B[i]) { bad = i; break; }
    }
    for (int i = 1000; i < 1024; ++i) {
        if (_C[i] != 0.0f) { bad = i; break; }
    }
    if (bad >= 0) { print_text("FAIL\\n"); finish(1); }
    print_text("PASS\\n");
    finish(0);
}
"""


_GEMM_HARNESS = """
static float _A[64 * 64];
static float _B[64 * 64];
static float _C[64 * 64];
static float _R[64 * 64];

void kmain(void) {
    for (int i = 0; i < 64 * 64; ++i) _A[i] = (float)((i % 3) - 1);
    for (int i = 0; i < 64 * 64; ++i) _B[i] = (float)(((i / 7) % 3) - 1);
    for (int i = 0; i < 64 * 64; ++i) { _C[i] = 0.0f; _R[i] = 0.0f; }
    __KERNEL__(_A, _B, _C);
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < 64; ++kk) acc += _A[i * 64 + kk] * _B[kk * 64 + j];
            _R[i * 64 + j] = acc;
        }
    }
    int bad = -1;
    for (int i = 0; i < 64 * 64; ++i) {
        if (_C[i] != _R[i]) { bad = i; break; }
    }
    if (bad >= 0) { print_text("FAIL\\n"); finish(1); }
    print_text("PASS\\n");
    finish(0);
}
"""


def harness_gemm_tile():
    """kmain for gemm_tile (single K step).

    Inputs are integer-valued ({-1,0,1}) so every product and every partial
    sum is exactly representable in float32: the check is an exact ``!=``
    even though the kernel and the reference may sum in a different order.
    """
    return _GEMM_HARNESS.replace("__KERNEL__", "gemm_tile")


def harness_gemm_multi_k():
    """kmain for gemm_multi_k: identical check, the reference sums all K.

    The reference loop already walks the full K, so the same harness proves a
    multi-step kernel too -- and it would fail a converter that dropped the
    second K step (the partial sum would be short by one product term).
    """
    return _GEMM_HARNESS.replace("__KERNEL__", "gemm_multi_k")


# The transposed cases need their own reference: the operand the kernel is
# handed is stored transposed, so the reference reads it down the other axis.
# Integer-valued inputs keep every product and partial sum exact in fp32.
_TRANSPOSE_HARNESS = """
static float _A[__ASZ__];
static float _B[__BSZ__];
static float _C[__CSZ__];
static float _R[__CSZ__];

void kmain(void) {
    for (int i = 0; i < __ASZ__; ++i) _A[i] = (float)((i % 3) - 1);
    for (int i = 0; i < __BSZ__; ++i) _B[i] = (float)(((i / 7) % 3) - 1);
    for (int i = 0; i < __CSZ__; ++i) { _C[i] = 0.0f; _R[i] = 0.0f; }
    __KERNEL__(_A, _B, _C);
    for (int i = 0; i < __M__; ++i) {
        for (int j = 0; j < __N__; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < __K__; ++kk) acc += (__ATERM__) * (__BTERM__);
            _R[i * __N__ + j] = acc;
        }
    }
    int bad = -1;
    for (int i = 0; i < __CSZ__; ++i) {
        if (_C[i] != _R[i]) { bad = i; break; }
    }
    if (bad >= 0) { print_text("FAIL\\n"); finish(1); }
    print_text("PASS\\n");
    finish(0);
}
"""


def _transpose_harness(name, trans_a, trans_b):
    aterm = ("_A[kk * %d + i]" % TR_M) if trans_a else ("_A[i * %d + kk]" % TR_K)
    bterm = ("_B[j * %d + kk]" % TR_K) if trans_b else ("_B[kk * %d + j]" % TR_N)
    return (_TRANSPOSE_HARNESS
            .replace("__ASZ__", str(TR_M * TR_K))
            .replace("__BSZ__", str(TR_N * TR_K))
            .replace("__CSZ__", str(TR_M * TR_N))
            .replace("__M__", str(TR_M))
            .replace("__N__", str(TR_N))
            .replace("__K__", str(TR_K))
            .replace("__ATERM__", aterm)
            .replace("__BTERM__", bterm)
            .replace("__KERNEL__", name))


def harness_gemm_trans_a():
    """kmain for gemm_trans_a: reference reads A down its K-major layout."""
    return _transpose_harness("gemm_trans_a", True, False)


def harness_gemm_trans_b():
    """kmain for gemm_trans_b: reference reads B down its K-major layout."""
    return _transpose_harness("gemm_trans_b", False, True)


def harness_gemm_trans_ab():
    """kmain for gemm_trans_ab: both operands stored K-major."""
    return _transpose_harness("gemm_trans_ab", True, True)


# The ragged cases trick the harness into proving *two* things: that the valid
# part of C is numerically right, and that the store really stopped at the
# shape.  C lives inside a larger array with a sentinel-filled guard band right
# after it, so a store that ran past the end would trample the band and be
# caught.  (An over-run in the *column* direction stays inside C and is caught
# by the numeric comparison instead.)
_RAGGED_HARNESS = """
static float _A[__ASZ__];
static float _B[__BSZ__];
static float _CS[__CSZ__ + __GUARD__];
static float _R[__CSZ__];
#define _C _CS

void kmain(void) {
    for (int i = 0; i < __ASZ__; ++i) _A[i] = (float)((i % 3) - 1);
    for (int i = 0; i < __BSZ__; ++i) _B[i] = (float)(((i / 7) % 3) - 1);
    for (int i = 0; i < __CSZ__ + __GUARD__; ++i) _CS[i] = 777.0f;
    __KERNEL__(_A, _B, _C);
    for (int i = 0; i < __M__; ++i) {
        for (int j = 0; j < __N__; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < __K__; ++kk)
                acc += _A[i * __K__ + kk] * _B[kk * __N__ + j];
            _R[i * __N__ + j] = acc;
        }
    }
    int bad = -1;
    for (int i = 0; i < __CSZ__; ++i) {
        if (_C[i] != _R[i]) { bad = i; break; }
    }
    for (int i = 0; i < __GUARD__; ++i) {
        if (_CS[__CSZ__ + i] != 777.0f) { bad = __CSZ__ + i; break; }
    }
    if (bad >= 0) { print_text("FAIL\\n"); finish(1); }
    print_text("PASS\\n");
    finish(0);
}
"""


def _ragged_harness(name, m, n, k, guard=4096):
    return (_RAGGED_HARNESS
            .replace("__ASZ__", str(m * k))
            .replace("__BSZ__", str(k * n))
            .replace("__CSZ__", str(m * n))
            .replace("__GUARD__", str(guard))
            .replace("__M__", str(m))
            .replace("__N__", str(n))
            .replace("__K__", str(k))
            .replace("__KERNEL__", name))


def harness_gemm_ragged_m():
    """kmain for gemm_ragged_m, with the guard band after C."""
    return _ragged_harness("gemm_ragged_m", 96, 64, 48)


def harness_gemm_ragged_n():
    """kmain for gemm_ragged_n."""
    return _ragged_harness("gemm_ragged_n", 64, 96, 48)


def harness_gemm_ragged_k():
    """kmain for gemm_ragged_k."""
    return _ragged_harness("gemm_ragged_k", 64, 64, 50)


# ---------------------------------------------------------------------------
# case registry
# ---------------------------------------------------------------------------
# ``rvv`` names the instruction the kernel must actually contain: verify_demo
# asserts it against the disassembly, so a case that stopped vectorising is
# caught there and not only by the numeric check.
CASES = [
    {"name": "vadd_exact", "func": vadd_exact, "harness": harness_vadd_exact,
     "rvv": "vfadd"},
    {"name": "vadd_tail", "func": vadd_tail, "harness": harness_vadd_tail,
     "rvv": "vfadd"},
    {"name": "vadd_2d", "func": vadd_2d, "harness": harness_vadd_2d,
     "rvv": "vfadd"},
    {"name": "vadd_f64", "func": vadd_f64, "harness": harness_vadd_f64,
     "rvv": "vfadd"},
    {"name": "gemm_tile", "func": gemm_tile, "harness": harness_gemm_tile,
     "rvv": "vfmacc"},
    {"name": "gemm_multi_k", "func": gemm_multi_k, "harness": harness_gemm_multi_k,
     "rvv": "vfmacc"},
    {"name": "gemm_trans_a", "func": gemm_trans_a, "harness": harness_gemm_trans_a,
     "rvv": "vfmacc"},
    {"name": "gemm_trans_b", "func": gemm_trans_b, "harness": harness_gemm_trans_b,
     "rvv": "vfmacc"},
    {"name": "gemm_trans_ab", "func": gemm_trans_ab, "harness": harness_gemm_trans_ab,
     "rvv": "vfmacc"},
    {"name": "gemm_ragged_m", "func": gemm_ragged_m, "harness": harness_gemm_ragged_m,
     "rvv": "vfmacc"},
    {"name": "gemm_ragged_n", "func": gemm_ragged_n, "harness": harness_gemm_ragged_n,
     "rvv": "vfmacc"},
    {"name": "gemm_ragged_k", "func": gemm_ragged_k, "harness": harness_gemm_ragged_k,
     "rvv": "vfmacc"},
    # rowsum is deliberately scalar (mixed float widths), so it has no RVV
    # instruction to assert on
    {"name": "rowsum", "func": rowsum, "harness": harness_rowsum,
     "rvv": None},
    {"name": "rowmax", "func": rowmax, "harness": harness_rowmax,
     "rvv": "vle32"},
]

HARNESS_LITERALS = {
    "vadd_exact": (1024,),
    "vadd_tail": (1024, 1000),
    "vadd_2d": (1024,),
    "vadd_f64": (256,),
    "gemm_tile": (64,),
    "gemm_multi_k": (64,),
    "gemm_trans_a": (96, 64, 48),
    "gemm_trans_b": (96, 64, 48),
    "gemm_trans_ab": (96, 64, 48),
    "gemm_ragged_m": (96, 64, 48),
    "gemm_ragged_n": (64, 96, 48),
    "gemm_ragged_k": (64, 64, 50),
    "rowsum": (64, 128),
    "rowmax": (64, 128),
}


def build_case(case, generate=None):
    """Return ``(kernel_c, full_c)`` for one case.

    ``kernel_c`` is exactly what ``backend.generate()`` produced -- no main,
    no _start.  ``full_c`` is what ``runner.build_and_run()`` compiles: the
    runner's RUNTIME helpers, then the kernel, then the harness ``kmain``.
    """
    if generate is None:
        from backend import generate as generate
    from runner import RUNTIME

    func = case["func"]()
    kernel_c = generate(func, func_name=case["name"])
    full_c = RUNTIME + "\n" + kernel_c + "\n" + case["harness"]()
    return kernel_c, full_c
