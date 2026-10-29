"""backend.py -- TileLang (TVM ``tirx``) PrimFunc -> restricted RVV C.

WHAT THIS IS
------------
A *reader* for the real TileLang IR, not a source-pattern matcher.  The
converter walks the ``tirx`` PrimFunc that ``@T.prim_func`` actually produced
(the pre-lowering IR, the one that still carries ``T.Kernel`` / ``T.Parallel``
/ ``T.gemm`` tile ops) and translates the nodes it can prove correct.  Every
node it meets must be in the supported subset; anything else raises
``UnsupportedError`` naming the node.  No statement is silently dropped.

TARGET MODEL
------------
A single RV64 hart with the V extension (RVV 1.0).  Consequences:

* a ``blockIdx.*`` loop becomes a sequential ``for`` on this hart
* ``threadIdx.*`` (``tl.launch_thread_idx``) is 0; code whose addressing
  depends on it is *rejected* (executing all parallel iterations on one hart
  would otherwise double-count)
* ``alloc_shared`` / ``alloc_fragment`` become core-local ``static`` arrays
  (they are private to the one hart)
* pipelining annotations (``num_stages``) are optimisation hints and are
  ignored; the loops run serially, which is the semantic fallback on a hart
  with no async copy engine
* synchronisation / barrier / cross-hart ops are not in the subset -> rejected

SUPPORTED SUBSET  (everything else raises)
------------------------------------------
Statements: ``SeqStmt``, ``SBlock`` / ``SBlockRealize`` (predicate true,
no block iter vars), ``For`` (SERIAL / THREAD_BINDING=blockIdx.* / PARALLEL),
``IfThenElse``, ``BufferStore``, ``Bind``/``Let``, ``Evaluate`` holding
``tl.tileop.fill`` / ``tl.tileop.clear`` / ``tl.tileop.copy`` /
``tl.tileop.gemm``.

``T.Parallel`` regions are the vector part.  A region qualifies only if its
body is a sequence of ``BufferStore`` where *every* access sits at
``<lane-var> * 1 + <loop-invariant rest>`` with the *same* rest, so the region
is a unit-stride elementwise map.  The dtype is float32 (e32m1).  Two shapes:

* unguarded  -- the converter must *prove* statically that the largest index
  the grid can reach is inside the buffer, otherwise the IR is rejected and
  the TileLang program is told to write an explicit ``if idx < N`` guard.
* guarded    -- the guard must be ``idx < bound`` (or ``idx <= bound``) with
  ``idx`` the very same rest+lane expression the accesses use and ``bound``
  lane-invariant.  The guard becomes an exact tail: ``vl`` is clamped to the
  number of still-valid lanes and the loop breaks once the bound is passed.
  Non-prefix guards (``idx != 5``) are rejected.

``T.gemm`` is emitted as an RVV micro-kernel (``vfmacc`` over the N dimension)
for fp32 tiles.  All four orientations are handled by following the stored
layout: ``transpose_A`` puts the reduction axis first in A's tile (so A is read
scalar-wise down its rows), and ``transpose_B`` does the same for B, whose tile
then needs a *strided* load because consecutive ``n`` live a row apart.

Tile sizes need not divide the shapes.  TileLang's ``T.copy`` carries no
out-of-range predicate in this IR, so the converter supplies the boundary
itself: a *read* that can run past the end of a global buffer is predicated and
padded with zero -- the padding must be zero because it is summed into real
output elements -- and a *write* that can run past the end simply stops at the
shape.  Nothing is emitted when the tiles already divide, so an exact kernel
pays nothing for this.  A region whose *smallest* start is already past the end
is still refused: that is a mistake in the addressing, not a boundary case.

``generate(func)`` returns a complete C translation unit containing the kernel
function only -- no ``main``, no ``_start``.

NOT VCIX.  Nothing here targets a VCIX matrix unit; there is no chip spec in
this repository to check instruction names, encodings or register widths
against, so no VCIX claim is made anywhere.
"""

from __future__ import annotations

import os as _os
import re

# The release does not bypass duplicate OpenMP runtime checks.

try:  # pragma: no cover - import plumbing
    import tilelang  # noqa: F401  (registers the bundled ``tvm`` package)
    from tilelang import tvm
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "backend.py needs the tilelang package to read the IR "
        "(tilelang 0.1.15). Original error: %r" % (exc,)
    ) from exc

tirx = tvm.tirx

__all__ = [
    "ConversionError",
    "UnsupportedError",
    "ValidationError",
    "generate",
    "convert",
    "RvvBackend",
    "supported_summary",
]


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------
class ConversionError(Exception):
    """Base class for every refusal this backend makes."""


class UnsupportedError(ConversionError):
    """The IR contains a construct outside the supported subset."""


class ValidationError(ConversionError):
    """The IR is inside the subset but violates a correctness precondition."""


class _NeedScalar(Exception):
    """Internal: this T.Parallel region cannot be one vector expression.

    Raised before any code is emitted, so the caller can re-emit it as a
    scalar loop without leaving half a vector region behind.
    """


# --------------------------------------------------------------------------
# type & intrinsic tables
# --------------------------------------------------------------------------
_C_TYPES = {
    "float16": "_Float16",
    "float32": "float",
    "float64": "double",
    "int8": "signed char",
    "int16": "short",
    "int32": "int",
    "int64": "long long",
    "uint8": "unsigned char",
    "uint16": "unsigned short",
    "uint32": "unsigned int",
    "uint64": "unsigned long long",
    "bool": "int",
}

# Only float32 lanes are implemented.  Adding int32 is a matter of one more
# entry here plus a test; claiming a mapping we never compiled would be worse
# than refusing, so anything absent from this table is rejected.
_VEC = {
    "float32": {
        "suffix": "e32m1",
        "vtype": "vfloat32m1_t",
        "elem": "float",
        "elem_size": 4,
        "load": "__riscv_vle32_v_f32m1",
        "load_strided": "__riscv_vlse32_v_f32m1",
        "store": "__riscv_vse32_v_f32m1",
        "splat": "__riscv_vfmv_v_f_f32m1",
        "macc": "__riscv_vfmacc_vf_f32m1",
        "binary": {
            "Add": "__riscv_vfadd_vv_f32m1",
            "Sub": "__riscv_vfsub_vv_f32m1",
            "Mul": "__riscv_vfmul_vv_f32m1",
            "Div": "__riscv_vfdiv_vv_f32m1",
        },
        "unary_min": "__riscv_vfmin_vv_f32m1",
        "unary_max": "__riscv_vfmax_vv_f32m1",
    },
    "float64": {
        "suffix": "e64m1",
        "vtype": "vfloat64m1_t",
        "elem": "double",
        "elem_size": 8,
        "load": "__riscv_vle64_v_f64m1",
        "load_strided": "__riscv_vlse64_v_f64m1",
        "store": "__riscv_vse64_v_f64m1",
        "splat": "__riscv_vfmv_v_f_f64m1",
        "macc": "__riscv_vfmacc_vf_f64m1",
        "binary": {
            "Add": "__riscv_vfadd_vv_f64m1",
            "Sub": "__riscv_vfsub_vv_f64m1",
            "Mul": "__riscv_vfmul_vv_f64m1",
            "Div": "__riscv_vfdiv_vv_f64m1",
        },
        "unary_min": "__riscv_vfmin_vv_f64m1",
        "unary_max": "__riscv_vfmax_vv_f64m1",
    },
}

_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline",
    "int", "long", "register", "restrict", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned",
    "void", "volatile", "while", "bool", "size_t", "main", "kmain",
}

_BLOCK_TAGS = ("blockIdx.x", "blockIdx.y", "blockIdx.z")
_TILE_OPS = {
    "tl.tileop.fill",
    "tl.tileop.clear",
    "tl.tileop.copy",
    "tl.tileop.gemm",
    "tl.tileop.reduce",
}
_IGNORABLE_ANNOTATIONS = {"tl.launch_threads", "num_stages", "num_threads", "tl.threads"}


def supported_summary() -> str:
    """One-paragraph description of the accepted subset (used in tests/docs)."""
    return (
        "statements: SeqStmt/SBlock/SBlockRealize/For(serial,blockIdx.*,parallel)"
        "/IfThenElse/BufferStore/Bind/Let/Evaluate(tl.tileop.fill,clear,copy,gemm); "
        "vector regions: float32 unit-stride elementwise T.Parallel, optional "
        "prefix guard 'idx < bound'; gemm: fp32 tiles, optional "
        "transpose_A/transpose_B, tile sizes may be ragged (clipped at run time)"
    )


# --------------------------------------------------------------------------
# helpers over the IR
# --------------------------------------------------------------------------
def _c_ident(name) -> str:
    out = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
    if not out or out[0].isdigit():
        out = "_" + out
    if out in _KEYWORDS:
        out += "_v"
    return out


def _op_name(node) -> str:
    op = getattr(node, "op", None)
    name = getattr(op, "name", None)
    if name:
        return str(name)
    return str(op)


def _as_int(e):
    """Return e as a python int if it is an integer immediate, else None."""
    if isinstance(e, (tirx.IntImm,)):
        return int(e.value)
    return None


def _is_true(e) -> bool:
    v = _as_int(e)
    return v is not None and v != 0


def _children(e):
    """Every PrimExpr directly inside e.

    Deliberately field-name driven so a var hiding inside an unusual node is
    still found: missing one here could make an index look lane-invariant and
    produce a wrong mask, so we err towards reporting more children.
    """
    out = []
    for field in (
        "a", "b", "c", "value", "condition", "true_value", "false_value",
        "index", "base", "stride", "lanes", "min", "max", "var", "extent",
    ):
        if not hasattr(e, field):
            continue
        v = getattr(e, field)
        if isinstance(v, tirx.PrimExpr):
            out.append(v)
        elif isinstance(v, (list, tuple)) and v and all(
            isinstance(x, tirx.PrimExpr) for x in v
        ):
            out.extend(v)
    for field in ("indices", "args"):
        if hasattr(e, field):
            v = getattr(e, field)
            if isinstance(v, (list, tuple)) and v and all(
                isinstance(x, tirx.PrimExpr) for x in v
            ):
                out.extend(v)
    return out


def _contains_var(e, var) -> bool:
    if isinstance(e, tirx.Var):
        return bool(e.same_as(var))
    for child in _children(e):
        if _contains_var(child, var):
            return True
    return False


def _struct_eq(a, b) -> bool:
    try:
        return bool(tvm.ir.structural_equal(a, b))
    except Exception:  # pragma: no cover - very old/new tvm
        return str(a) == str(b)


# --------------------------------------------------------------------------
# buffer bookkeeping
# --------------------------------------------------------------------------
class _Buf:
    __slots__ = ("name", "dtype", "shape", "strides", "scope", "is_param", "written")

    def __init__(self, buffer, is_param, written=False):
        self.name = _c_ident(buffer.name)
        self.dtype = str(buffer.dtype)
        self.shape = [int(x) for x in buffer.shape]
        strides = list(buffer.strides) if buffer.strides else []
        if strides:
            self.strides = [int(s) for s in strides]
        else:  # row-major fallback
            acc, rev = 1, []
            for d in reversed(self.shape):
                rev.append(acc)
                acc *= d
            self.strides = list(reversed(rev))
        try:
            self.scope = str(buffer.scope())
        except Exception:
            self.scope = "global" if is_param else "local"
        self.is_param = is_param
        self.written = written

    @property
    def nelem(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


# --------------------------------------------------------------------------
# the converter
# --------------------------------------------------------------------------
class RvvBackend:
    """Translate one PrimFunc into a restricted RVV C kernel function."""

    def __init__(self, func, func_name=None, include_header=True):
        if not isinstance(func, tirx.PrimFunc):
            raise TypeError("generate() expects a tvm.tirx.PrimFunc, got %r" % (type(func),))
        self.func = func
        self.name = _c_ident(func_name or self._default_name())
        self.include_header = include_header

        self.buffers = {}          # buffer object -> _Buf
        self.param_buf = {}        # handle Var -> _Buf (buffer_map params)
        self._names = {}           # Var/obj -> C identifier
        self._taken = set()
        self.thread_idx_vars = {}  # Var -> C name (must stay unused)
        self.loop_max = {}         # Var -> max value, inside the current nest
        self.loop_min = {}         # Var -> min value, inside the current nest
        self.grid_axes = []        # [(tag, varname, extent-expr)] in order
        self.out = []              # emitted body lines
        self.ind = 1
        self._tmp = 0
        self._seq = 0
        self._need_floordiv = False
        self._need_floormod = False
        self._notes = []
        self._analyzer = tvm.arith.Analyzer()

    # -- names ------------------------------------------------------------
    def _default_name(self):
        attrs = getattr(self.func, "attrs", None)
        try:
            if attrs is not None and "global_symbol" in attrs:
                return str(attrs["global_symbol"])
        except Exception:
            pass
        return "tl_kernel"

    def _fresh(self, base):
        name = base
        i = 0
        while name in self._taken:
            i += 1
            name = "%s_%d" % (base, i)
        self._taken.add(name)
        return name

    def _loopname(self, prefix):
        """A unique name for a generated loop variable."""
        self._seq += 1
        return "_%s%d" % (prefix, self._seq)

    def _bind_name(self, var):
        if var in self._names:
            return self._names[var]
        name = self._fresh(_c_ident(getattr(var, "name", "v")))
        self._names[var] = name
        return name

    def _varname(self, var):
        if var in self.thread_idx_vars:
            raise UnsupportedError(
                "expression uses %s (tl.launch_thread_idx): addressing that "
                "depends on a thread index cannot be honoured on a single hart. "
                "Use a T.Parallel loop instead." % self.thread_idx_vars[var]
            )
        if var not in self._names:
            raise UnsupportedError("expression uses an unbound variable %r" % (var,))
        return self._names[var]

    # -- buffers ----------------------------------------------------------
    def _collect_buffers(self):
        bmap = self.func.buffer_map
        for param in self.func.params:
            is_buf = False
            try:
                is_buf = param in bmap
            except Exception:
                is_buf = False
            if not is_buf:
                continue
            buffer = bmap[param]
            info = _Buf(buffer, is_param=True)
            info.name = self._fresh(info.name)
            self.buffers[buffer] = info
            self.param_buf[param] = info
            self._names[param] = info.name

    def _local_buffer(self, buffer):
        if buffer in self.buffers:
            return self.buffers[buffer]
        info = _Buf(buffer, is_param=False)
        info.name = self._fresh(info.name)
        self.buffers[buffer] = info
        return info

    def _buf(self, buffer):
        if buffer not in self.buffers:
            raise UnsupportedError(
                "statement touches buffer %r which was never declared "
                "(no buffer_map entry and no alloc_buffer for it)" % (buffer.name,)
            )
        return self.buffers[buffer]

    def _linear_index(self, info, indices):
        """Flatten a multi-dimensional access to the buffer's linear offset.

        Tile-level programs address 2-D/3-D tensors directly; the vector path
        reasons about linear offsets, so fold the strides in and simplify.  The
        simplification is what recovers unit stride: a 2-D elementwise kernel
        written over a flattened ``p`` indexes ``C[p // W, p % W]``, whose
        linear form ``(p // W) * W + p % W`` simplifies back to ``p``.
        """
        if len(indices) == 1:
            return indices[0]
        if len(indices) != len(info.shape):
            raise UnsupportedError(
                "access to %s has %d indices but the buffer is %d-D"
                % (info.name, len(indices), len(info.shape)))
        total = None
        for dim, idx in enumerate(indices):
            stride = info.strides[dim]
            term = idx if stride == 1 else idx * stride
            total = term if total is None else total + term
        try:
            return self._analyzer.simplify(total)
        except Exception:  # pragma: no cover - simplifier is best-effort
            return total

    # -- line plumbing ----------------------------------------------------
    def _line(self, text):
        self.out.append("    " * self.ind + text)

    # -- expressions ------------------------------------------------------
    def _lit(self, v):
        if v > 0x7FFFFFFF:
            return "%dLL" % v
        if v < -0x80000000:
            return "%dLL" % v
        return "%d" % v

    def _expr(self, e):
        """Scalar C expression for a PrimExpr (strict: unknown node -> refuse)."""
        if isinstance(e, tirx.IntImm):
            return self._lit(int(e.value))
        if isinstance(e, tirx.FloatImm):
            txt = repr(float(e.value))
            if "e" not in txt and "E" not in txt and "." not in txt:
                txt += ".0"
            dt = str(e.dtype)
            if dt == "float32":
                txt += "f"
            elif dt == "float64":
                pass
            return txt
        if isinstance(e, tirx.Var):
            return self._varname(e)
        if isinstance(e, (tirx.Add, tirx.Sub, tirx.Mul)):
            sym = {tirx.Add: "+", tirx.Sub: "-", tirx.Mul: "*"}[type(e)]
            return "(%s %s %s)" % (self._expr(e.a), sym, self._expr(e.b))
        if isinstance(e, tirx.Div):
            return "(%s / %s)" % (self._expr(e.a), self._expr(e.b))
        if isinstance(e, tirx.FloorDiv):
            self._need_floordiv = True
            return "((%s)tl_floordiv((long)(%s), (long)(%s)))" % (
                self._ctype(e.dtype), self._expr(e.a), self._expr(e.b))
        if isinstance(e, tirx.Mod):
            return "(%s %% %s)" % (self._expr(e.a), self._expr(e.b))
        if isinstance(e, tirx.FloorMod):
            self._need_floormod = True
            return "((%s)tl_floormod((long)(%s), (long)(%s)))" % (
                self._ctype(e.dtype), self._expr(e.a), self._expr(e.b))
        if isinstance(e, (tirx.Min, tirx.Max)):
            op = "<" if isinstance(e, tirx.Min) else ">"
            return "((%s) %s (%s) ? (%s) : (%s))" % (
                self._expr(e.a), op, self._expr(e.b), self._expr(e.a), self._expr(e.b))
        if isinstance(e, tirx.Select):
            return "((%s) ? (%s) : (%s))" % (
                self._expr(e.condition), self._expr(e.true_value),
                self._expr(e.false_value))
        if isinstance(e, tirx.Cast):
            return "((%s)(%s))" % (self._ctype(e.dtype), self._expr(e.value))
        if isinstance(e, tirx.Not):
            return "(!(%s))" % self._expr(e.a)
        if isinstance(e, (tirx.And, tirx.Or)):
            sym = "&&" if isinstance(e, tirx.And) else "||"
            return "((%s) %s (%s))" % (self._expr(e.a), sym, self._expr(e.b))
        if isinstance(e, (tirx.EQ, tirx.NE, tirx.LT, tirx.LE, tirx.GT, tirx.GE)):
            sym = {tirx.EQ: "==", tirx.NE: "!=", tirx.LT: "<",
                   tirx.LE: "<=", tirx.GT: ">", tirx.GE: ">="}[type(e)]
            return "(%s %s %s)" % (self._expr(e.a), sym, self._expr(e.b))
        if isinstance(e, tirx.BufferLoad):
            info = self._buf(e.buffer)
            idx = self._linear_index(info, list(e.indices))
            return "%s[%s]" % (info.name, self._expr(idx))
        raise UnsupportedError(
            "unsupported expression node %s: %s" % (type(e).__name__, str(e)[:120]))

    def _ctype(self, dtype) -> str:
        dt = str(dtype)
        if dt not in _C_TYPES:
            raise UnsupportedError("unsupported dtype %r" % (dt,))
        if dt.startswith("uint"):
            return _C_TYPES[dt].replace("unsigned", "unsigned")
        return _C_TYPES[dt]

    # -- static bounds ----------------------------------------------------
    # _upper/_lower are only ever used to *prove* a region is in bounds.  They
    # must over/under-estimate respectively: returning None means "cannot
    # prove" and the caller refuses.  Never used to size anything.
    def _upper(self, e):
        """An int >= the true maximum of e, or None when undecidable."""
        v = _as_int(e)
        if v is not None:
            return v if v >= 0 else None
        if isinstance(e, tirx.Var):
            return self.loop_max.get(e)
        if isinstance(e, tirx.Add):
            a, b = self._upper(e.a), self._upper(e.b)
            return None if a is None or b is None else a + b
        if isinstance(e, tirx.Sub):
            a, b = self._upper(e.a), self._upper(e.b)
            return a if b == 0 else None
        if isinstance(e, tirx.Mul):
            a, b = self._upper(e.a), self._upper(e.b)
            return None if a is None or b is None else a * b
        if isinstance(e, (tirx.FloorDiv, tirx.Div)):
            # only sound with a constant positive divisor: with a variable one
            # the maximum of a/b comes from the *smallest* b, which we do not
            # track, so using the largest would under-estimate.
            a = self._upper(e.a)
            d = _as_int(e.b)
            if a is None or d is None or d <= 0:
                return None
            return a // d
        if isinstance(e, (tirx.FloorMod, tirx.Mod)):
            d = _as_int(e.b)
            if d is None or d <= 0:
                return None
            return d - 1
        if isinstance(e, tirx.Min):
            a, b = self._upper(e.a), self._upper(e.b)
            return None if a is None or b is None else min(a, b)
        if isinstance(e, tirx.Max):
            a, b = self._upper(e.a), self._upper(e.b)
            return None if a is None or b is None else max(a, b)
        if isinstance(e, tirx.Cast):
            return _as_int(e.value)   # narrowing casts can wrap; refuse the rest
        return None

    def _lower(self, e):
        """An int <= the true minimum of e, or None when undecidable."""
        v = _as_int(e)
        if v is not None:
            return v
        if isinstance(e, tirx.Var):
            return self.loop_min.get(e)
        if isinstance(e, tirx.Add):
            a, b = self._lower(e.a), self._lower(e.b)
            return None if a is None or b is None else a + b
        if isinstance(e, tirx.Sub):
            a, b = self._lower(e.a), self._lower(e.b)
            return a if b == 0 else None
        if isinstance(e, tirx.Mul):
            la, lb_ = self._lower(e.a), self._lower(e.b)
            ua, ub_ = self._upper(e.a), self._upper(e.b)
            if None in (la, lb_, ua, ub_):
                return None
            return min(la * lb_, la * ub_, ua * lb_, ua * ub_)
        if isinstance(e, (tirx.FloorDiv, tirx.Div)):
            a = self._lower(e.a)
            d = _as_int(e.b)
            if a is None or d is None or d <= 0 or a < 0:
                return None
            return a // d
        if isinstance(e, (tirx.FloorMod, tirx.Mod)):
            d = _as_int(e.b)
            if d is None or d <= 0:
                return None
            return 0
        if isinstance(e, tirx.Min):
            a, b = self._lower(e.a), self._lower(e.b)
            return None if a is None or b is None else min(a, b)
        if isinstance(e, tirx.Max):
            a, b = self._lower(e.a), self._lower(e.b)
            return None if a is None or b is None else max(a, b)
        if isinstance(e, tirx.Cast):
            return _as_int(e.value)
        return None

    def _non_negative(self, e):
        """True when e is provably >= 0."""
        lo = self._lower(e)
        return lo is not None and lo >= 0

    # -- lane split -------------------------------------------------------
    def _add_terms(self, e, sign=1):
        if isinstance(e, tirx.Add):
            yield from self._add_terms(e.a, sign)
            yield from self._add_terms(e.b, sign)
        elif isinstance(e, tirx.Sub):
            yield from self._add_terms(e.a, sign)
            yield from self._add_terms(e.b, -sign)
        else:
            yield (sign, e)

    def _split_lane(self, e, var, allow_constant=False):
        """Split e into (lane_coeff, rest_terms) with lane_coeff == 1 required.

        ``allow_constant`` admits a *constant* region (extent 1): there is only
        one lane, so an access that does not mention the lane variable is the
        whole expression, not a broadcast.
        """
        coeff = 0
        rest = []
        for sign, term in self._add_terms(e):
            if _contains_var(term, var):
                if term.same_as(var) and sign == 1:
                    coeff += 1
                else:
                    raise UnsupportedError(
                        "T.Parallel access %r is not a unit-stride lane "
                        "expression (lane var must appear exactly once, added, "
                        "with coefficient 1)" % (str(e)[:120],))
            else:
                rest.append((sign, term))
        if coeff == 0:
            if allow_constant:
                return [(1, e)]
            raise UnsupportedError(
                "T.Parallel region accesses %r which does not depend on the "
                "parallel loop variable -- it is a broadcast/reduction, not an "
                "elementwise map" % (str(e)[:120],))
        if coeff != 1:
            raise UnsupportedError(
                "T.Parallel access %r has lane coefficient %d, only 1 is "
                "supported" % (str(e)[:120], coeff))
        return rest

    def _rest_key(self, rest_terms):
        return " | ".join("%s%s" % ("+" if s > 0 else "-", str(t)) for s, t in rest_terms)

    def _rest_text(self, rest_terms):
        """C text of the lane-invariant part plus the strides already folded in."""
        parts = []
        for sign, term in rest_terms:
            txt = self._expr(term)
            parts.append(("%s" % txt) if sign > 0 else ("(-%s)" % txt))
        if not parts:
            return None
        return " + ".join(parts)

    # -- vector expressions ----------------------------------------------
    def _vec(self, e, vc):
        """Vector C expression for e in the context vc (see _parallel)."""
        if isinstance(e, (tirx.IntImm, tirx.FloatImm)):
            return "%s(%s, %s)" % (vc["t"]["splat"], self._expr(e), vc["vl"])
        if isinstance(e, (tirx.Add, tirx.Sub, tirx.Mul, tirx.Div)):
            key = {tirx.Add: "Add", tirx.Sub: "Sub",
                   tirx.Mul: "Mul", tirx.Div: "Div"}[type(e)]
            return "%s(%s, %s, %s)" % (
                vc["t"]["binary"][key], self._vec(e.a, vc), self._vec(e.b, vc), vc["vl"])
        if isinstance(e, (tirx.Min, tirx.Max)):
            key = "unary_min" if isinstance(e, tirx.Min) else "unary_max"
            return "%s(%s, %s, %s)" % (
                vc["t"][key], self._vec(e.a, vc), self._vec(e.b, vc), vc["vl"])
        if isinstance(e, tirx.Cast):
            if str(e.dtype) != vc["dtype"]:
                raise UnsupportedError(
                    "cast to %s inside a vector region: only casts that keep "
                    "the lane dtype %s are supported" % (e.dtype, vc["dtype"]))
            return self._vec(e.value, vc)
        if isinstance(e, tirx.BufferLoad):
            info = self._buf(e.buffer)
            if info.dtype != vc["dtype"]:
                raise UnsupportedError(
                    "T.Parallel region mixes dtypes: %s is %s but the region is %s"
                    % (info.name, info.dtype, vc["dtype"]))
            k = self._key_of(info, list(e.indices), vc["var"], vc["allow_const"])
            base = vc["base_of"].get(k)
            if base is None:
                raise UnsupportedError(
                    "internal: no lane base recorded for %s[%s]"
                    % (info.name, str(e)[:80]))
            return "%s(%s + %s, %s)" % (vc["t"]["load"], info.name, base, vc["vl"])
        raise UnsupportedError(
            "unsupported expression inside a T.Parallel region: %s (%s)"
            % (type(e).__name__, str(e)[:120]))

    # -- statements -------------------------------------------------------
    def _stmt(self, n):
        if isinstance(n, tirx.SeqStmt):
            for s in n.seq:
                self._stmt(s)
        elif isinstance(n, tirx.SBlockRealize):
            if not _is_true(n.predicate):
                raise UnsupportedError(
                    "SBlockRealize carries a predicate %s; conditional blocks are "
                    "not in the supported subset" % n.predicate)
            if len(n.iter_values):
                raise UnsupportedError("SBlockRealize with block iter values is not supported")
            self._stmt(n.block)
        elif isinstance(n, tirx.SBlock):
            self._sblock(n)
        elif isinstance(n, tirx.For):
            self._for(n)
        elif isinstance(n, tirx.IfThenElse):
            self._if(n)
        elif isinstance(n, tirx.BufferStore):
            self._store(n)
        elif isinstance(n, tirx.Evaluate):
            self._evaluate(n)
        elif isinstance(n, tirx.Bind):
            self._bind(n)
        elif isinstance(n, tirx.Let):
            self._let(n)
        else:
            raise UnsupportedError(
                "unsupported statement %s: %s" % (type(n).__name__, str(n)[:160]))

    def _push_loop_range(self, n):
        """Record [min, min+extent-1] for a loop var, when both ends are known."""
        extent = _as_int(n.extent)
        lo = _as_int(n.min)
        if extent is None or lo is None:
            return None
        saved = (self.loop_max.get(n.loop_var), self.loop_min.get(n.loop_var))
        self.loop_max[n.loop_var] = lo + extent - 1
        self.loop_min[n.loop_var] = lo
        return saved

    def _pop_loop_range(self, n, saved):
        if saved is None:
            return
        for table, value in zip((self.loop_max, self.loop_min), saved):
            if value is None:
                table.pop(n.loop_var, None)
            else:
                table[n.loop_var] = value

    def _check_annotations(self, annotations, where):
        for key in annotations or ():
            if str(key) not in _IGNORABLE_ANNOTATIONS:
                raise UnsupportedError(
                    "unknown annotation %r on %s" % (str(key), where))

    def _sblock(self, n):
        self._check_annotations(n.annotations, "SBlock")
        for buf in n.alloc_buffers or ():
            info = self._local_buffer(buf)
            dtype = self._ctype(info.dtype)
            self._line("/* %s.alloc_buffer %s, scope=%s */" % (
                n.name_hint, info.name, info.scope))
            self._line("static %s %s[%d];" % (dtype, info.name, info.nelem))
        self._stmt(n.body)

    def _for(self, n):
        kind = n.kind
        if kind == tirx.ForKind.PARALLEL:
            return self._parallel(n)
        if kind == tirx.ForKind.SERIAL:
            return self._serial(n)
        if kind == tirx.ForKind.THREAD_BINDING:
            return self._grid(n)
        raise UnsupportedError(
            "loop kind %s over %s is not supported (only serial, blockIdx.* and "
            "T.Parallel)" % (kind, n.loop_var))

    def _bound_text(self, n):
        """C text for the exclusive upper bound of a For (min is almost always 0)."""
        mn = _as_int(n.min)
        ext = _as_int(n.extent)
        if mn == 0:
            return "(%s)" % self._expr(n.extent) if ext is None else "%d" % ext
        return "(%s) + (%s)" % (self._expr(n.min), self._expr(n.extent))

    def _serial(self, n):
        var = self._bind_name(n.loop_var)
        lo = self._expr(n.min)
        hi = self._bound_text(n)
        self._check_annotations(n.annotations, "serial loop")
        saved = self._push_loop_range(n)
        self._line("for (int %s = %s; %s < %s; ++%s) {" % (var, lo, var, hi, var))
        self.ind += 1
        self._stmt(n.body)
        self.ind -= 1
        self._line("}")
        self._pop_loop_range(n, saved)

    def _grid(self, n):
        tb = n.thread_binding
        tag = getattr(tb, "thread_tag", None) if tb is not None else None
        if tag not in _BLOCK_TAGS:
            raise UnsupportedError(
                "thread binding %r is not supported: this backend targets one "
                "RV64 hart, so only blockIdx.* grid loops can be honoured (code "
                "addressing through threadIdx would be mis-compiled)" % (tag,))
        var = self._bind_name(n.loop_var)
        lo = self._expr(n.min)
        hi = self._bound_text(n)
        self.grid_axes.append((tag, var, n.extent))
        saved = self._push_loop_range(n)
        self._line("/* T.Kernel grid %s (extent %s) -- one hart -> sequential */"
                   % (tag, str(n.extent)))
        self._line("for (int %s = %s; %s < %s; ++%s) {" % (var, lo, var, hi, var))
        self.ind += 1
        self._stmt(n.body)
        self.ind -= 1
        self._line("}")
        self._pop_loop_range(n, saved)

    def _if(self, n):
        cond = self._expr(n.condition)
        self._line("if (%s) {" % cond)
        self.ind += 1
        self._stmt(n.then_case)
        self.ind -= 1
        if n.else_case is not None:
            self._line("} else {")
            self.ind += 1
            self._stmt(n.else_case)
            self.ind -= 1
        self._line("}")

    def _store(self, n):
        info = self._buf(n.buffer)
        info.written = True
        idx = self._linear_index(info, list(n.indices))
        self._line("%s[%s] = %s;" % (
            info.name, self._expr(idx), self._expr(n.value)))

    def _bind(self, n):
        if _op_name(n.value) == "tl.launch_thread_idx":
            idx = _as_int(n.value.args[0]) if n.value.args else None
            name = self._bind_name(n.var)
            self.thread_idx_vars[n.var] = name
            self._line("/* T.launch_thread_idx(%s): single hart -> 0 */" % (idx,))
            self._line("const int %s = 0;" % name)
            return
        raise UnsupportedError(
            "unsupported Bind of %s = %s" % (n.var, str(n.value)[:120]))

    def _let(self, n):
        name = self._bind_name(n.var)
        self._line("const %s %s = %s;" % (self._ctype(n.var.dtype), name, self._expr(n.value)))
        self._stmt(n.body)

    # -- tile ops ---------------------------------------------------------
    def _evaluate(self, n):
        call = n.value
        if not isinstance(call, tirx.Call):
            raise UnsupportedError(
                "Evaluate of a non-call expression is not supported: %s"
                % str(call)[:120])
        name = _op_name(call)
        if name in ("tl.tileop.fill", "tl.tileop.clear"):
            return self._fill(call, name)
        if name == "tl.tileop.copy":
            return self._copy(call)
        if name == "tl.tileop.gemm":
            return self._gemm(call)
        if name == "tl.tileop.reduce":
            return self._reduce(call)
        raise UnsupportedError(
            "tile op %r is not in the supported subset (supported: %s)"
            % (name, ", ".join(sorted(_TILE_OPS))))

    def _region(self, e):
        if _op_name(e) != "tl.region":
            raise UnsupportedError("expected a tl.region, got %s" % str(e)[:120])
        args = list(e.args)
        if len(args) < 3:
            raise UnsupportedError("malformed tl.region: %s" % str(e)[:120])
        load = args[0]
        if not isinstance(load, tirx.BufferLoad):
            raise UnsupportedError("tl.region does not start with a buffer load")
        access = _as_int(args[1])
        extents = [_as_int(a) for a in args[2:]]
        if access is None or any(x is None for x in extents):
            raise UnsupportedError(
                "tl.region with non-constant access/extent is not supported: %s"
                % str(e)[:120])
        return load.buffer, list(load.indices), extents, access

    def _region_plan(self, info, base, extents, what):
        """Per-dimension valid-length plan for a region over ``info``.

        Each entry is ``(const, text)``:

        * ``(n, None)``    -- the whole extent is statically in bounds, use ``n``
        * ``(None, expr)`` -- the tile can run past the end for some loop value,
          so only ``min(extent, size - base)`` elements are valid and the
          generated code must clip.

        A tile whose *smallest* start is already past the end is refused rather
        than clipped to nothing: that means the grid and the tile sizes
        disagree, and silently emitting an empty loop would hide the mistake.
        """
        if len(base) != len(info.shape):
            raise UnsupportedError("%s: rank mismatch on %s" % (what, info.name))
        plan = []
        for dim, extent in enumerate(extents):
            # the smallest value the start can take decides both whether the
            # region can be clipped at all and where clipping would begin
            lo = self._lower(base[dim])
            if lo is None:
                raise ValidationError(
                    "%s: cannot bound where %s starts along dim %d (base %s), "
                    "so the tile cannot be clipped safely"
                    % (what, info.name, dim, str(base[dim])[:60]))
            if lo < 0:
                raise ValidationError(
                    "%s: cannot prove %s[%s] stays at a non-negative index "
                    "(dim %d)" % (what, info.name, str(base[dim])[:60], dim))
            ub = self._upper(base[dim])
            if ub is not None and ub + extent <= info.shape[dim]:
                plan.append((extent, None))
                continue
            if lo >= info.shape[dim]:
                raise ValidationError(
                    "%s: every tile of %s along dim %d starts at or past the "
                    "end (smallest start %d >= %d): the grid and the tile sizes "
                    "disagree" % (what, info.name, dim, lo, info.shape[dim]))
            plan.append((None, None))
        return plan

    def _emit_clip(self, info, base, extents, plan):
        """Emit the run-time clip variables a plan needs and name them in it."""
        for dim, (const, text) in enumerate(plan):
            if const is not None:
                continue
            v = self._loopname("eff")
            plan[dim] = (None, v)
            self._line("long %s = (long)%d - (long)(%s);"
                       % (v, info.shape[dim], self._expr(base[dim])))
            self._line("if (%s > %d) %s = %d;" % (v, extents[dim], v, extents[dim]))

    def _fill(self, call, opname):
        args = list(call.args)
        if len(args) != 2:
            raise UnsupportedError("%s expects (region, value)" % opname)
        buffer, base, extents, _ = self._region(args[0])
        info = self._buf(buffer)
        value = self._expr(args[1])
        plan = self._region_plan(info, base, extents, opname)
        if info.is_param and any(c is None for c, _ in plan):
            self._emit_clip(info, base, extents, plan)
        info.written = True
        self._emit_region_loop(
            info, base, extents, "fill", plan,
            assign=lambda off: "%s[%s] = %s;" % (info.name, off, value))
        self._mark_cleared(info, base, extents)

    def _mark_cleared(self, info, base, extents):
        if id(info) not in self._cleared:
            self._cleared[id(info)] = False
        if all(_as_int(b) == 0 for b in base) and extents == info.shape:
            self._cleared[id(info)] = True

    def _emit_region_loop(self, info, base, extents, tag, plan, assign):
        if len(extents) != len(info.shape):
            raise UnsupportedError(
                "%s on %s: region rank %d != buffer rank %d"
                % (tag, info.name, len(extents), len(info.shape)))
        ivars = []
        for _ in extents:
            ivars.append(self._loopname(tag[0]))
        for dim, (name, extent) in enumerate(zip(ivars, extents)):
            bound = "%d" % extent
            if plan is not None and plan[dim][0] is None:
                bound = plan[dim][1]      # clip a store to the valid prefix
            self._line("for (int %s = 0; %s < %s; ++%s) {" % (name, name, bound, name))
            self.ind += 1
        off = self._offset_text(info, base, ivars)
        self._line(assign(off))
        for _ in ivars:
            self.ind -= 1
            self._line("}")

    def _offset_text(self, info, base, ivars):
        parts = []
        for dim, var in enumerate(ivars):
            stride = info.strides[dim]
            term = "(%s + %s)" % (self._expr(base[dim]), var)
            parts.append(term if stride == 1 else "(%s * %d)" % (term, stride))
        return " + ".join(parts)

    def _copy(self, call):
        args = list(call.args)
        if len(args) != 2:
            raise UnsupportedError("tl.tileop.copy expects (src, dst)")
        sbuf, sbase, sext, sacc = self._region(args[0])
        dbuf, dbase, dext, dacc = self._region(args[1])
        sinfo = self._buf(sbuf)
        dinfo = self._buf(dbuf)
        if sext != dext:
            raise UnsupportedError(
                "tl.tileop.copy with mismatched extents %s -> %s" % (sext, dext))
        if sacc not in (1, 3) or dacc not in (2, 3):
            raise UnsupportedError(
                "tl.tileop.copy with unexpected access kinds (src %s, dst %s)"
                % (sacc, dacc))
        if sinfo.dtype != dinfo.dtype:
            raise UnsupportedError(
                "tl.tileop.copy between different dtypes (%s -> %s)"
                % (sinfo.dtype, dinfo.dtype))
        s_plan = self._region_plan(sinfo, sbase, sext, "tl.tileop.copy source")
        d_plan = self._region_plan(dinfo, dbase, dext, "tl.tileop.copy destination")

        # A source tile that runs past the end of a global buffer is read under
        # a predicate and padded with zero: the out-of-range lanes never touch
        # memory and -- crucially -- the padding cannot contaminate the
        # accumulator those lanes feed (a non-zero or stale pad in the K
        # direction would silently corrupt every real output).
        # A *destination* that runs past the end is simply written as far as it
        # is valid; whatever would have gone beyond is tile padding that no
        # real output depends on.
        s_clip = sinfo.is_param and any(c is None for c, _ in s_plan)
        d_clip = dinfo.is_param and any(c is None for c, _ in d_plan)
        svars = [self._loopname("c") for _ in sext]
        if s_clip:
            self._emit_clip(sinfo, sbase, sext, s_plan)
        if d_clip:
            self._emit_clip(dinfo, dbase, dext, d_plan)
        dinfo.written = True
        for dim, name in enumerate(svars):
            bound = "%d" % sext[dim]
            if d_clip and d_plan[dim][0] is None:
                bound = d_plan[dim][1]        # stop the store at the valid end
            self._line("for (int %s = 0; %s < %s; ++%s) {" % (name, name, bound, name))
            self.ind += 1
        value = "%s[%s]" % (sinfo.name, self._offset_text(sinfo, sbase, svars))
        if s_clip:
            conds = ["(%s < %s)" % (v, s_plan[d][1])
                     for d, v in enumerate(svars) if s_plan[d][0] is None]
            if conds:
                value = "(%s) ? %s : 0" % (" && ".join(conds), value)
        self._line("%s[%s] = %s;" % (
            dinfo.name, self._offset_text(dinfo, dbase, svars), value))
        for _ in svars:
            self.ind -= 1
            self._line("}")

    def _gemm(self, call):
        args = list(call.args)
        if len(args) < 8:
            raise UnsupportedError("tl.tileop.gemm with only %d args" % len(args))
        abuf, abase, aext, aacc = self._region(args[0])
        bbuf, bbase, bext, bacc = self._region(args[1])
        cbuf, cbase, cext, cacc = self._region(args[2])
        ainfo, binfo, cinfo = self._buf(abuf), self._buf(bbuf), self._buf(cbuf)
        trans_a, trans_b = _is_true(args[3]), _is_true(args[4])
        m, n, k = _as_int(args[5]), _as_int(args[6]), _as_int(args[7])
        if m is None or n is None or k is None:
            raise UnsupportedError("tl.tileop.gemm with non-constant M/N/K is not supported")
        for extra in args[8:]:
            if _as_int(extra) is None:
                raise UnsupportedError(
                    "tl.tileop.gemm with a non-constant schedule argument (%s)"
                    % str(extra)[:60])
        policy = _as_int(args[8])
        clear_accum = _is_true(args[9]) if len(args) > 9 else False
        if policy not in (0, None):
            raise UnsupportedError("tl.tileop.gemm policy %s is not supported" % policy)
        # transpose_A/B mean the *stored* tile carries (K, M) / (N, K): the
        # region extents say which layout the front end produced, so check the
        # regions against the transposed expectation rather than assuming them.
        want_a = [k, m] if trans_a else [m, k]
        want_b = [n, k] if trans_b else [k, n]
        if aext != want_a or bext != want_b or cext != [m, n]:
            raise UnsupportedError(
                "tl.tileop.gemm regions do not match M/N/K (transpose_A=%s, "
                "transpose_B=%s): A%s B%s C%s vs M=%d N=%d K=%d"
                % (trans_a, trans_b, aext, bext, cext, m, n, k))
        for info, label in ((ainfo, "A"), (binfo, "B"), (cinfo, "C")):
            if len(info.shape) != 2:
                raise UnsupportedError("tl.tileop.gemm operand %s is not 2-D" % label)
            if info.dtype not in _VEC:
                raise UnsupportedError(
                    "tl.tileop.gemm operand %s has dtype %s; only %s tiles are "
                    "supported" % (label, info.dtype, "/".join(sorted(_VEC))))
        # every tile is addressed through its strides, so all three need a
        # row-major inner dimension whatever their orientation
        if ainfo.strides[1] != 1 or binfo.strides[1] != 1 or cinfo.strides[1] != 1:
            raise UnsupportedError(
                "tl.tileop.gemm needs row-major (unit inner stride) A, B and C tiles")
        if not self._cleared.get(id(cinfo), False) and not clear_accum:
            raise ValidationError(
                "tl.tileop.gemm accumulates into %s but nothing initialises it: "
                "insert T.clear(%s) before the gemm (or pass clear_accum)"
                % (cinfo.name, cinfo.name))

        t = _VEC[cinfo.dtype]
        gm, gk, gn = self._loopname("m"), self._loopname("k"), self._loopname("n")
        gvl, ga, gb, gc = (self._loopname(x) for x in ("vl", "a", "b", "c"))
        # index each tile down its *physical* axes: a transposed operand stores
        # its reduction axis first, so which loop variable is the contiguous
        # one changes, and the offsets must follow the stored layout
        aoff = self._offset_text(ainfo, abase, [gk, gm] if trans_a else [gm, gk])
        boff = self._offset_text(binfo, bbase, [gn, gk] if trans_b else [gk, gn])
        coff = self._offset_text(cinfo, cbase, [gm, gn])
        self._line("/* tl.tileop.gemm %dx%dx%d%s (fp32 -> RVV vfmacc) */"
                   % (m, n, k,
                      "" if not (trans_a or trans_b) else
                      " A%s B%s" % ("^T" if trans_a else "", "^T" if trans_b else "")))
        if clear_accum and not self._cleared.get(id(cinfo), False):
            self._line("for (int %s = 0; %s < %d; ++%s) {" % (gm, gm, m, gm))
            self.ind += 1
            self._line("for (int %s = 0; %s < %d; ++%s) %s[%s] = 0;"
                       % (gn, gn, n, gn, cinfo.name, coff))
            self.ind -= 1
            self._line("}")
        self._line("for (int %s = 0; %s < %d; ++%s) {" % (gm, gm, m, gm))
        self.ind += 1
        self._line("for (int %s = 0; %s < %d; ++%s) {" % (gk, gk, k, gk))
        self.ind += 1
        self._line("%s %s = %s[%s];" % (t["elem"], ga, ainfo.name, aoff))
        self._line("for (int %s = 0; %s < %d; ) {" % (gn, gn, n))
        self.ind += 1
        self._line("size_t %s = __riscv_vsetvl_%s((size_t)(%d - %s));"
                   % (gvl, t["suffix"], n, gn))
        if trans_b:
            # B_s is (N, K): consecutive n are a whole row apart, so ask for a
            # strided load instead of a unit-stride one
            bstride = binfo.strides[0] * t["elem_size"]
            self._line("%s %s = %s(%s + (%s), %d, %s);"
                       % (t["vtype"], gb, t["load_strided"], binfo.name, boff,
                          bstride, gvl))
        else:
            self._line("%s %s = %s(%s + (%s), %s);"
                       % (t["vtype"], gb, t["load"], binfo.name, boff, gvl))
        self._line("%s %s = %s(%s + (%s), %s);"
                   % (t["vtype"], gc, t["load"], cinfo.name, coff, gvl))
        self._line("%s = %s(%s, %s, %s, %s);" % (gc, t["macc"], gc, ga, gb, gvl))
        self._line("%s(%s + (%s), %s, %s);" % (t["store"], cinfo.name, coff, gc, gvl))
        self._line("%s += (int)%s;" % (gn, gvl))
        self.ind -= 1
        self._line("}")
        self.ind -= 1
        self._line("}")
        self.ind -= 1
        self._line("}")
        cinfo.written = True

    def _reduce(self, call):
        """``T.reduce(src_region, dst_region, "sum", dim, clear)``.

        Emitted as a scalar accumulation over the reduced axis: correct at any
        tile size, and reduction is not the hot inner loop of these kernels.
        Only ``sum`` is implemented; the other reductions would need an
        explicit ordered fold and are refused rather than guessed.
        """
        args = list(call.args)
        if len(args) < 4:
            raise UnsupportedError("tl.tileop.reduce with only %d args" % len(args))
        sbuf, sbase, sext, _ = self._region(args[0])
        dbuf, dbase, dext, _ = self._region(args[1])
        sinfo, dinfo = self._buf(sbuf), self._buf(dbuf)
        op = str(getattr(args[2], "value", args[2]))
        if op not in ("sum", "max", "min"):
            raise UnsupportedError(
                "tl.tileop.reduce with op %r is not supported (sum/max/min)"
                % (op,))
        dim = _as_int(args[3])
        clear = _is_true(args[4]) if len(args) > 4 else False
        if dim is None or not 0 <= dim < len(sext):
            raise UnsupportedError(
                "tl.tileop.reduce with dim %s is outside the source rank %d"
                % (args[3], len(sext)))
        out_dims = [d for d in range(len(sext)) if d != dim]
        if dext != [sext[d] for d in out_dims]:
            raise UnsupportedError(
                "tl.tileop.reduce destination %s does not match source %s "
                "with dim %d removed" % (dext, sext, dim))
        if sinfo.dtype not in _VEC or dinfo.dtype not in _VEC:
            raise UnsupportedError(
                "tl.tileop.reduce over %s -> %s: only %s are implemented"
                % (sinfo.dtype, dinfo.dtype, "/".join(sorted(_VEC))))

        ctype = self._ctype(sinfo.dtype)
        ivars = [self._loopname("r") for _ in out_dims]
        acc = self._loopname("acc")
        kvar = self._loopname("k")
        for name, d in zip(ivars, out_dims):
            self._line("for (int %s = 0; %s < %d; ++%s) {" % (name, name, sext[d], name))
            self.ind += 1
        self._line("/* tl.tileop.reduce %s over dim %d */" % (op, dim))
        if clear:
            # sum starts from 0; max/min start from the identity element so a
            # tile of all-negative (or all-positive) values still folds right
            init = {"sum": "0",
                    "max": "-__builtin_inf()",
                    "min": "__builtin_inf()"}[op]
            self._line("%s %s = %s;" % (ctype, acc, init))
        else:
            self._line("%s %s = %s[%s];" % (
                ctype, acc, dinfo.name, self._offset_text(dinfo, dbase, ivars)))
        src_vars = [kvar if d == dim else ivars[out_dims.index(d)]
                    for d in range(len(sext))]
        src_txt = "%s[%s]" % (
            sinfo.name, self._offset_text(sinfo, sbase, src_vars))
        self._line("for (int %s = 0; %s < %d; ++%s) {" % (kvar, kvar, sext[dim], kvar))
        self.ind += 1
        if op == "sum":
            self._line("%s += %s;" % (acc, src_txt))
        else:
            cmp = ">" if op == "max" else "<"
            v = self._loopname("v")
            self._line("%s %s = %s;" % (ctype, v, src_txt))
            self._line("if (%s %s %s) %s = %s;" % (v, cmp, acc, acc, v))
        self.ind -= 1
        self._line("}")
        self._line("%s[%s] = %s;" % (
            dinfo.name, self._offset_text(dinfo, dbase, ivars), acc))
        dinfo.written = True
        for _ in ivars:
            self.ind -= 1
            self._line("}")

    # -- T.Parallel -------------------------------------------------------
    def _key_of(self, info, indices, var, allow_constant=False):
        """Rest key for one access: identical strings mean identical offsets."""
        return self._rest_key(self._split_lane(
            self._linear_index(info, indices), var, allow_constant))

    def _parse_guard(self, cond, var):
        """Return (rest_terms, coeff, bound_text, bound_expr) for a tail guard.

        The condition has to be ``index < bound`` (or ``<=`` / reversed), with
        ``index`` a unit-stride lane expression and ``bound`` lane-invariant,
        so that the set of surviving lanes is a prefix.  Only then can the
        guard become a clamped ``vl``.
        """
        if isinstance(cond, (tirx.LT, tirx.LE)):
            left, right, inclusive = cond.a, cond.b, isinstance(cond, tirx.LE)
        elif isinstance(cond, (tirx.GT, tirx.GE)):
            left, right, inclusive = cond.b, cond.a, isinstance(cond, tirx.GE)
        else:
            raise UnsupportedError(
                "guard %s inside T.Parallel is not a comparison, so it cannot "
                "be turned into a lane bound" % str(cond)[:120])
        if _contains_var(right, var):
            raise UnsupportedError(
                "guard %s has the parallel loop variable on both sides; only "
                "'index < loop-invariant bound' is supported" % str(cond)[:120])
        coeff = 0
        rest = []
        for sign, term in self._add_terms(left):
            if _contains_var(term, var):
                if term.same_as(var) and sign == 1:
                    coeff += 1
                else:
                    raise UnsupportedError(
                        "guard %s is not a unit-stride lane expression"
                        % str(cond)[:120])
            else:
                rest.append((sign, term))
        if coeff != 1:
            raise UnsupportedError(
                "guard %s does not advance one lane at a time, so it does not "
                "select a prefix of the lanes" % str(cond)[:120])
        if inclusive:
            # x <= b  <=>  x < b+1 ; text only, so no new IR node is built
            return rest, coeff, ("(%s) + 1" % self._expr(right)), None
        return rest, coeff, self._expr(right), right

    def _provable(self, info, rest_terms, extent_i):
        """Can the converter prove rest+lane stays inside info for lane<E?

        Both ends matter: a rest of ``-4`` would walk off the *front* of the
        buffer and the upper bound alone would not notice.
        """
        if extent_i is None:
            return False
        total = 0
        for sign, term in rest_terms:
            if sign > 0:
                if not self._non_negative(term):
                    return False
                ub = self._upper(term)
                if ub is None:
                    return False
                total += ub
            else:
                ub = self._upper(term)
                if ub is None or ub != 0:
                    return False      # a subtracted non-zero term can go below 0
        return total + extent_i <= info.nelem

    def _inline(self, node, bindings):
        """Substitute per-lane intermediates until nothing changes.

        A ``Bind`` inside ``T.Parallel`` is how the front end names a per-lane
        temporary (``_p = bx * block + lane``).  Inlining it keeps the region an
        elementwise map of the lane variable alone, which is what the vector
        path can express.  Bindings can chain, so iterate to a fixed point.
        """
        if not bindings:
            return node
        for _ in range(len(bindings) + 1):
            new = tirx.stmt_functor.substitute(node, bindings)
            if new is node or _struct_eq(new, node):
                return new
            node = new
        return node

    def _mixed_dtype(self, e, dtype):
        """True when e crosses into a different float width than the region.

        RVV intrinsics are width-specific (``vle32``/``vle64``), so a region
        that reads float32 and accumulates in float64 cannot be one vector
        expression.  Detecting it up front lets the caller pick the scalar
        fallback instead of emitting half a vector loop and then refusing.
        """
        if isinstance(e, tirx.Cast):
            return str(e.dtype) != dtype or self._mixed_dtype(e.value, dtype)
        if isinstance(e, tirx.BufferLoad):
            if self._buf(e.buffer).dtype != dtype:
                return True
        return any(self._mixed_dtype(c, dtype) for c in _children(e))

    def _vec_ok(self, e, dtype):
        """Can the vector table express this value expression at ``dtype``?"""
        if isinstance(e, (tirx.IntImm, tirx.FloatImm)):
            return True
        if isinstance(e, (tirx.Add, tirx.Sub, tirx.Mul, tirx.Div,
                          tirx.Min, tirx.Max)):
            return self._vec_ok(e.a, dtype) and self._vec_ok(e.b, dtype)
        if isinstance(e, tirx.Cast):
            return str(e.dtype) == dtype and self._vec_ok(e.value, dtype)
        if isinstance(e, tirx.BufferLoad):
            return self._buf(e.buffer).dtype == dtype
        return False

    def _parallel_scalar_stmt(self, n, var):
        """Emit a T.Parallel region as a plain scalar loop.

        Used when the region cannot be one vector expression: mixed float
        widths (float32 read into a float64 accumulator), a per-lane if/else,
        or an operator the vector table lacks.  A scalar loop is exact; these
        regions are the exception, so the lost vectorisation is acceptable.
        """
        iv = self._loopname("i")
        saved = self._names.get(var)
        self._names[var] = iv
        ext = self._expr(n.extent)
        self._line("/* T.Parallel(%s) -> scalar loop */" % ext)
        self._line("for (int %s = 0; %s < %s; ++%s) {" % (iv, iv, ext, iv))
        self.ind += 1
        self._scalar_stmt(n.body)
        self.ind -= 1
        self._line("}")
        if saved is None:
            self._names.pop(var, None)
        else:
            self._names[var] = saved

    def _scalar_stmt(self, node):
        if isinstance(node, tirx.SeqStmt):
            for s in node.seq:
                self._scalar_stmt(s)
        elif isinstance(node, tirx.Bind):
            name = self._bind_name(node.var)
            self._line("const %s %s = %s;" % (
                self._ctype(node.var.dtype), name, self._expr(node.value)))
        elif isinstance(node, tirx.IfThenElse):
            self._line("if (%s) {" % self._expr(node.condition))
            self.ind += 1
            self._scalar_stmt(node.then_case)
            self.ind -= 1
            if node.else_case is not None:
                self._line("} else {")
                self.ind += 1
                self._scalar_stmt(node.else_case)
                self.ind -= 1
            self._line("}")
        elif isinstance(node, tirx.BufferStore):
            info = self._buf(node.buffer)
            info.written = True
            idx = self._linear_index(info, list(node.indices))
            self._line("%s[%s] = %s;" % (
                info.name, self._expr(idx), self._expr(node.value)))
        else:
            raise UnsupportedError(
                "T.Parallel scalar body contains %s: %s"
                % (type(node).__name__, str(node)[:120]))

    def _parallel(self, n):
        try:
            return self._parallel_vector(n)
        except _NeedScalar:
            return self._parallel_scalar_stmt(n, n.loop_var)

    def _parallel_vector(self, n):
        var = n.loop_var
        extent_i = _as_int(n.extent)
        allow_const = extent_i == 1
        bindings = {}
        guard_cond = None
        stores = []

        def gather(node):
            if isinstance(node, tirx.SeqStmt):
                for s in node.seq:
                    gather(s)
            elif isinstance(node, tirx.Bind):
                bindings[node.var] = node.value
            elif isinstance(node, tirx.IfThenElse):
                nonlocal guard_cond
                if guard_cond is not None:
                    raise UnsupportedError(
                        "T.Parallel body has more than one guard; only a single "
                        "prefix guard can become a clamped vl")
                if node.else_case is not None:
                    # a per-lane if/else is a select; keep it for the scalar path
                    raise _NeedScalar()
                guard_cond = node.condition
                gather(node.then_case)
            elif isinstance(node, tirx.BufferStore):
                stores.append(node)
            else:
                raise UnsupportedError(
                    "T.Parallel body contains %s, which cannot be vectorised "
                    "(%s)" % (type(node).__name__, str(node)[:120]))

        gather(n.body)
        if not stores:
            raise UnsupportedError("T.Parallel body has no buffer store")
        if bindings:
            stores = [self._inline(s, bindings) for s in stores]
        if guard_cond is not None:
            guard_cond = self._inline(guard_cond, bindings)
        guard = self._parse_guard(guard_cond, var) if guard_cond is not None else None

        # every access must be a unit-stride lane access; collect the distinct
        # offset patterns (one C base pointer each)
        pairs = {}     # key -> rest_terms
        accs = []      # (info, key)
        touched = {}   # buffer name -> {"info":, "r": set(), "w": set()}

        def note(buffer, indices, where, is_write):
            info = self._buf(buffer)
            rest = self._split_lane(self._linear_index(info, indices), var, allow_const)
            k = self._rest_key(rest)
            if k not in pairs:
                pairs[k] = rest
            accs.append((info, k))
            slot = touched.setdefault(info.name, {"info": info, "r": set(), "w": set()})
            slot["w" if is_write else "r"].add(k)

        for st in stores:
            note(st.buffer, list(st.indices), "store into", True)
        for st in stores:
            for load in self._loads(st.value):
                note(load.buffer, list(load.indices), "reads", False)

        # dependence rules.  A vector chunk reads every lane before it stores,
        # so a buffer touched at one lane offset is safe, but a buffer read at
        # one offset and written at another (A[i] = A[i-1] + 1) has a
        # loop-carried dependence that the chunk would silently break.
        for name, slot in touched.items():
            if len(slot["w"]) > 1:
                raise UnsupportedError(
                    "T.Parallel region writes %s at %d different lane offsets; "
                    "only one write offset per buffer is supported"
                    % (name, len(slot["w"])))
            if slot["w"] and slot["r"] and slot["r"] != slot["w"]:
                raise UnsupportedError(
                    "T.Parallel region both reads and writes %s at different "
                    "lane offsets (read %s, written %s): that is a loop-carried "
                    "dependence across lanes and must not be vectorised"
                    % (name, sorted(slot["r"]), sorted(slot["w"])))

        dtype = None
        mixed = False
        for info, _ in accs:
            if dtype is None:
                dtype = info.dtype
            elif info.dtype != dtype:
                mixed = True
        if not mixed:
            mixed = any(self._mixed_dtype(st.value, dtype) for st in stores)
        if mixed:
            # float32 read through a float64 accumulator: the widths differ,
            # so one vector expression cannot express it.  A scalar loop is
            # exact; refuse only if neither width is implemented.
            widths = {info.dtype for info, _ in accs}
            widths |= {str(c.dtype) for st in stores
                       for c in _children(st.value) if isinstance(c, tirx.Cast)}
            if not widths <= set(_VEC):
                raise UnsupportedError(
                    "T.Parallel region mixes dtypes %s and neither width is "
                    "implemented (implemented: %s)"
                    % (sorted(widths), "/".join(sorted(_VEC))))
            raise _NeedScalar()
        if dtype not in _VEC:
            raise UnsupportedError(
                "T.Parallel vector region over dtype %s is not supported "
                "(implemented: %s)" % (dtype, "/".join(sorted(_VEC))))
        t = _VEC[dtype]
        if not all(self._vec_ok(st.value, dtype) for st in stores):
            # e.g. integer floor-division inside the value: the vector table has
            # no intrinsic for it, the scalar path does
            raise _NeedScalar()

        gkey = self._rest_key(guard[0]) if guard is not None else None
        if guard is not None and _as_int(guard[3]) is not None:
            # a constant bound can be checked against the buffers right here
            bound = _as_int(guard[3])
            for info, k in accs:
                if k == gkey and bound > info.nelem:
                    raise ValidationError(
                        "T.Parallel guard allows indices up to %d but %s only "
                        "has %d elements" % (bound - 1, info.name, info.nelem))
        for info, k in accs:
            if guard is not None and k == gkey and info.is_param:
                continue      # the user's own guard is the bound, as in TileLang
            if not self._provable(info, pairs[k], extent_i):
                why = ("has a runtime extent and no guard"
                       if guard is None and extent_i is None else
                       "cannot prove the largest reachable index stays inside it")
                raise ValidationError(
                    "T.Parallel %s: the converter refuses to emit a vector loop "
                    "it cannot prove correct (access into %s at offset %r, "
                    "extent %s, %d elements). Either make the tiles divide the "
                    "shape exactly, or write an explicit guard 'if idx < N:' in "
                    "the TileLang program."
                    % (why, info.name, self._rest_text(pairs[k]) or "0",
                       str(n.extent), info.nelem))

        # ---- emit
        seq = self._tmp
        self._tmp += 1
        iv = "_i%d" % seq
        vl = "_vl%d" % seq
        base_of = {}
        for k in pairs:
            base_of[k] = "_g%d_%d" % (seq, len(base_of))
        if guard is not None and gkey not in base_of:
            base_of[gkey] = "_g%d_%d" % (seq, len(base_of))
        ext_c = self._expr(n.extent)
        vc = {"var": var, "vl": vl, "dtype": dtype, "t": t,
              "base_of": base_of, "allow_const": allow_const}
        self._line("/* T.Parallel(%s) -> RVV length-agnostic loop, %s lanes%s */"
                   % (ext_c, t["suffix"],
                      ", prefix guard clamps vl" if guard is not None else ""))
        self._line("for (int %s = 0; %s < (%s); ) {" % (iv, iv, ext_c))
        self.ind += 1
        for k, rest_terms in pairs.items():
            rest_text = self._rest_text(rest_terms)
            if rest_text is None:
                self._line("long %s = (long)%s;" % (base_of[k], iv))
            else:
                self._line("long %s = (long)(%s) + (long)%s;"
                           % (base_of[k], rest_text, iv))
        if guard is None:
            self._line("size_t %s = __riscv_vsetvl_%s((size_t)((long)(%s) - (long)%s));"
                       % (vl, t["suffix"], ext_c, iv))
        else:
            bound_text = guard[2]
            self._line("long _rem%d = (long)(%s) - %s;" % (seq, bound_text, base_of[gkey]))
            # no valid lane left anywhere in this block: stop walking the grid
            self._line("if (_rem%d <= 0) break;" % seq)
            self._line("long _cnt%d = (long)(%s) - (long)%s;" % (seq, ext_c, iv))
            self._line("if (_rem%d < _cnt%d) _cnt%d = _rem%d;" % (seq, seq, seq, seq))
            self._line("size_t %s = __riscv_vsetvl_%s((size_t)_cnt%d);"
                       % (vl, t["suffix"], seq))
        for st in stores:
            info = self._buf(st.buffer)
            info.written = True
            k = self._key_of(info, list(st.indices), var, allow_const)
            self._line("%s(%s + %s, %s, %s);" % (
                t["store"], info.name, base_of[k], self._vec(st.value, vc), vl))
        self._line("%s += (int)%s;" % (iv, vl))
        self.ind -= 1
        self._line("}")
        self._tmp += 1

    def _loads(self, e):
        out = []
        def walk(x):
            if isinstance(x, tirx.BufferLoad):
                out.append(x)
            for child in _children(x):
                walk(child)
        walk(e)
        return out

    # -- driver -----------------------------------------------------------
    def _signature(self):
        parts = []
        for param in self.func.params:
            if param in self.param_buf:
                info = self.param_buf[param]
                const = "" if info.written else "const "
                parts.append("%s%s *%s" % (const, self._ctype(info.dtype), info.name))
            else:
                parts.append("%s %s" % (self._ctype(param.dtype), self._bind_name(param)))
        return "void %s(%s)" % (self.name, ", ".join(parts) if parts else "void")

    def convert(self) -> str:
        self._cleared = {}
        self._collect_buffers()
        # allocate names for scalar params up front so they keep their order/name
        for param in self.func.params:
            if param not in self.param_buf:
                self._bind_name(param)
        try:
            self._stmt(self.func.body)
        except ConversionError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ConversionError(
                "internal failure while translating %s: %r" % (self.name, exc)) from exc

        head = []
        head.append("/*")
        head.append(" * Generated by tilelang2rvv backend.py from TileLang PrimFunc "
                    "'%s'." % self._name_hint())
        head.append(" * Target: single RV64 hart + RVV 1.0 (e32m1 lanes). "
                    "No VCIX is used or claimed.")
        head.append(" * Mapping: T.Kernel grid -> sequential for; "
                    "T.launch_thread_idx -> 0;")
        head.append(" *          alloc_shared/alloc_fragment -> core-local static "
                    "arrays;")
        head.append(" *          T.Parallel -> unit-stride RVV loop "
                    "(optional prefix guard clamps vl);")
        head.append(" *          T.gemm(fp32) -> RVV vfmacc micro-kernel.")
        if self.grid_axes:
            head.append(" * Grid: %s" % ", ".join(
                "%s=%s" % (t, str(e)) for t, _v, e in self.grid_axes))
        head.append(" * This translation unit defines the kernel function only: "
                    "no main, no _start.")
        head.append(" */")
        lines = list(head)
        if self.include_header:
            lines.append("#include <riscv_vector.h>")
        if self._need_floordiv or self._need_floormod:
            lines.append("")
        if self._need_floordiv:
            lines.append("static long tl_floordiv(long a, long b) {")
            lines.append("    long q = a / b;")
            lines.append("    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;")
            lines.append("    return q;")
            lines.append("}")
        if self._need_floormod:
            lines.append("static long tl_floormod(long a, long b) {")
            lines.append("    long r = a % b;")
            lines.append("    if (r != 0 && ((r < 0) != (b < 0))) r += b;")
            lines.append("    return r;")
            lines.append("}")
        lines.append("")
        lines.append(self._signature() + " {")
        lines.extend(self.out)
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _name_hint(self):
        return self._default_name()


def generate(func, func_name=None, include_header=True) -> str:
    """Translate ``func`` (a ``tvm.tirx.PrimFunc``) into RVV C.

    Returns the C translation unit text -- the kernel function plus, when
    needed, a couple of ``static`` integer-division helpers.  It contains no
    ``main`` and no ``_start``: the caller drives it (see runner.py).

    Raises ``UnsupportedError`` for IR outside the subset and
    ``ValidationError`` when the IR is in the subset but a correctness
    precondition cannot be proven.  Both derive from ``ConversionError``.
    """
    if func is None:
        raise TypeError("generate() expects a PrimFunc, got None")
    return RvvBackend(func, func_name=func_name, include_header=include_header).convert()


convert = generate
