#===========================================================================#
# Survey of the serial loops that could gain from `Polyester.@batch`
# (gpena/Bramble.jl#432, subplan S2.1 of the v3.22.0 plan).
#
#     julia --project=benchmark --threads=4 --startup-file=no benchmark/batch_survey.jl
#
# The nine candidates are the serial hot loops listed in the plan's notes (section F,
# items 1 to 9; item 10 was judged unsuitable and is not measured). For each one the script
# times the loop as `src/` runs it today and as a `@batch` prototype written here, on a
# small and a large non-uniform grid. It then measures the loop's share of three
# representative serial solves on the large grid, and prints one `SURVEY` row per
# candidate:
#
#     SURVEY k=<k> name=<id> serial_small=<s> batch_small=<s> serial_large=<s> batch_large=<s> share=<f> <ABOVE|BELOW>
#
# `ABOVE` iff `serial_large / batch_large >= 1.5` and `share >= 0.05`. The speedup alone is
# not enough: a memory-bound stream can gain 1.2x and still cost nothing in a solve.
#
# ## Grids
#
# Both grids are 2D, uniform points each jittered by up to ±0.3h along each axis (the
# multigrid tests' mesh): non-uniform everywhere, with bounded cell aspect ratio so that the
# multigrid-preconditioned CG converges in a realistic number of iterations. The point counts
# are 2^k + 1, so the multigrid hierarchy coarsens all the way down.
#
# ## Timings
#
# Each timing is the minimum of one `BenchmarkTools.@belapsed` on a warm call through a
# function barrier, the serial loop and its prototype interleaved five times (a background
# load or a move to an efficiency core then hits both, not one). The rows report the median
# of the five, and the verdict rests on those medians; the least and greatest per-repeat
# speedup are printed on a line of their own. Before anything is timed, each
# prototype's result is checked against the serial loop's (`_check`), so a wrong prototype
# cannot report a speedup.
#
# Arrays that a prototype hands to a Bramble function go into `@batch` through a `Ref`, not
# captured directly: `@batch` converts a captured array to a `PtrArray`, and Bramble's
# kernels ran about half as fast on one (candidate 1: no gain at all at two threads,
# against 2x through the `Ref`; the script prints both). Loops written out in the body
# index the captured arrays directly, which `@batch` handles at full speed.
#
# The prototypes live only in this script: `src/` is untouched until the winners are
# converted. Candidate 4 adds three methods to Bramble internals, in this process only (see
# there). Several call Bramble internals (`Bramble._name`), so a refactor in `src/` can
# break this script; it is a survey, not a maintained benchmark.
#
# ## Shares
#
# `share` is the fraction of a solve's serial wall time spent inside the candidate,
# measured with the sampling profiler, not estimated from call counts. Each solve runs
# serially (`CpuSerial`, one BLAS thread) under `Profile`. Every sample of the running task
# counts towards the solve (a sample inside BLAS may not unwind to the solve's frame, but
# it is still the solve's time), and towards a candidate if its stack holds one of the
# candidate's frames (a function name, or a source line identified by its text, so the
# match survives line-number drift in `src/`). For candidate 1 the script checks the
# profile against a call count. The row's share is the largest over the three solves. Every solve includes its setup, since candidates 4, 5 and 9 run only there.
# The three solves, all on the large grid with the same SPD form
# `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))` (mass plus diffusion, natural boundary):
#
#   gmg_cg   gmg_preconditioner (matrix-free levels, Chebyshev smoother) + PCG to 1e-8
#   explicit semidiscretize + semidiscretize_rhs + 1000 forward-Euler steps through
#            SemidiscretizeRHS (assembled CSC operator)
#   kron_cg  kronecker_operator + unpreconditioned CG, 1e-8 or 300 iterations
#
# A share of 0 means no sample landed in the candidate. For candidates 2, 6 and 7 that is
# structural (`UNREACHED` says why); for a setup-only candidate (4, 9) it is a share below
# the sampling resolution, about 1e-4 of a solve.
#
# The setup's weight in `explicit` depends on the number of steps (`EXPLICIT_STEPS`); the
# script prints the setup fraction so the reader can rescale.
#===========================================================================#

using Bramble
using Bramble: Δₕ!, semidiscretize_rhs, change_points!
using Polyester
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Random
using Printf
using Profile

set_zero_subnormals(true)
BLAS.set_num_threads(1)

# The explicit solve's step count: a run long enough to reach a meaningful time (100 steps
# of dt = 0.1 hmin² reach only t ≈ 1.7e-6 on the large grid). The share of setup-time loops
# shrinks as the run grows (candidate 5: 0.067 of the solve at 100 steps, 0.0086 at 1000).
const EXPLICIT_STEPS = 1000
const N_SMALL = 65
const N_LARGE = parse(Int, get(ENV, "BATCH_SURVEY_N_LARGE", "1025"))  # override for a quick trial only
const SEED = 4321
const BENCH_SECONDS = parse(Float64, get(ENV, "BATCH_SURVEY_SECONDS", "1.0"))
const NTHREADS = Threads.nthreads()

# --- Meshes and the form ------------------------------------------------------------ #

function jitter_mesh(n; seed = SEED)
    rng = Xoshiro(seed)
    Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n), (true, true))
    h = 1 / (n - 1)
    function pts()
        x = collect(range(0.0, 1.0; length = n)) .+ 0.3h .* (2 .* rand(rng, n) .- 1)
        x[1], x[end] = 0.0, 1.0
        return sort!(x)
    end
    change_points!(Ω, (pts(), pts()))
    return Ω
end

poisson_form(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
source_fn(x) = sin(π * x[1]) * cos(π * x[2]) + 1

# Prototype results must match the serial loop's. A relative floor alone fails on entries
# of order 1e-17 (bramble-verification §6), so an absolute floor scaled to the data is added.
function _agree(a, b; rtol = 1e-12)
    return isapprox(a, b; rtol = rtol, atol = rtol * max(norm(a, Inf), norm(b, Inf)))
end

function _check(name, a, b; rtol = 1e-12)
    _agree(a, b; rtol = rtol) && return nothing
    return error("batch_survey: the @batch prototype of $name disagrees with the serial loop")
end

# --- Candidate 1: `_kron_fused!`, the KroneckerLinearOperator product ---------------- #
#
# The serial method is `mul!(y, K, x)` as Bramble runs it. The prototype is the same line
# loop with `@batch` over the lines, and each line writes the disjoint slice
# `y[off+1 : off+m]`. It stays faithful because the per-line work is Bramble's own
# `_kron_line_init!`/`_kron_line_terms!`.

c1_serial!(y, K, x) = mul!(y, K, x)

@inline function _kron_one_line!(y, terms, cs, x, ss, lines, o::Int, m::Int)
    off = (o - 1) * m
    Bramble._kron_line_init!(y, false, off, m)
    Bramble._kron_line_terms!(y, terms, cs, x, Tuple(lines[o]), ss, off, m)
    return nothing
end

function _kron_batch_args(y, K::Bramble.KroneckerLinearOperator{T}, x) where {T}
    cs = map(t -> Bramble._kron_scalar(T, true * Bramble._kron_coeff(t.scales)), K.terms)
    dims = K.dims
    ss = Base.front(cumprod(dims))::Tuple{Vararg{Int}}
    return (y, K.terms, cs, x, ss, CartesianIndices(Base.tail(dims)), dims[1])
end

function c1_batch!(y, K, x)
    args = Ref(_kron_batch_args(y, K, x))
    @batch for o in 1:length(args[][6])
        a = args[]
        _kron_one_line!(a[1], a[2], a[3], a[4], a[5], a[6], o, a[7])
    end
    return y
end

# The same prototype with `y` and `x` captured by `@batch` directly, for the record: it
# converts them to `PtrArray`s, and the line kernels run about half as fast on those.
function c1_batch_captured!(y, K, x)
    _, terms, cs, _, ss, lines, m = _kron_batch_args(y, K, x)
    @batch for o in 1:length(lines)
        _kron_one_line!(y, terms, cs, x, ss, lines, o, m)
    end
    return y
end

function setup_c1(Ω)
    W = gridspace(Ω)
    K = kronecker_operator(poisson_form(W))
    x = randn(Xoshiro(1), npoints(Ω))
    return (similar(x), K, x), (similar(x), K, x)
end

# --- Candidate 2: `Δₕ!` -------------------------------------------------------------- #
#
# Serial: `Δₕ!(vₕ, uₕ)`. Prototype: bands of the last axis under `@batch`, each band zeroing
# its own slab and running the fused engine `_accumulate_laplacian!` restricted to it (the
# same `_band_slab` cut the other engines take). Writes stay at the point, so bands are
# disjoint. Faithful to the loop body; written for 2D only, which is all this survey runs.

@inline function _lap_band!(
        out, u, hb, hs, dims::NTuple{D, Int}, ::Val{DIM}, nbands, b
) where {D, DIM}
    li = LinearIndices(dims)
    step = Bramble._stencil_step(Val(DIM), Val(D))
    interior, _ = Bramble._stencil_ranges(axes(li), Val(DIM), Bramble.Forward())
    interior = Bramble._band_slab(interior, Bramble._band_range(axes(li, D), nbands, b))
    @inbounds @simd for I in CartesianIndices(interior)
        idx, fwd = li[I], li[I + step]
        i = I[DIM]
        forward_flux = (u[fwd] - u[idx]) / Bramble._get_h_val(hb, i + 1)
        backward_flux = i == 1 ? zero(forward_flux) :
                        (u[idx] - u[li[I - step]]) / Bramble._get_h_val(hb, i)
        out[idx] += (forward_flux - backward_flux) / Bramble._get_h_val(hs, i)
    end
    return nothing
end

c2_serial!(vₕ, uₕ) = Δₕ!(vₕ, uₕ)

function c2_batch!(vₕ, uₕ)
    Ωₕ = mesh(space(uₕ))
    dims = npoints(Ωₕ, Tuple)
    out, u = parent(vₕ), parent(uₕ)
    hb1, hs1 = Bramble.backward_spacings_for_derivative(Ωₕ(1)), Bramble.star_spacings(Ωₕ(1))
    hb2, hs2 = Bramble.backward_spacings_for_derivative(Ωₕ(2)), Bramble.star_spacings(Ωₕ(2))
    nbands = min(NTHREADS, dims[2])
    args = Ref((out, u, hb1, hs1, hb2, hs2, dims))
    @batch for b in 1:nbands
        _lap_band_all!(args[]..., nbands, b)
    end
    return vₕ
end

@inline function _lap_band_all!(out, u, hb1, hs1, hb2, hs2, dims, nbands, b)
    band = Bramble._band_range(1:dims[2], nbands, b)
    li = LinearIndices(dims)
    @inbounds for j in band, i in 1:dims[1]

        out[li[i, j]] = 0.0
    end
    _lap_band!(out, u, hb2, hs2, dims, Val(2), nbands, b)
    _lap_band!(out, u, hb1, hs1, dims, Val(1), nbands, b)
    return nothing
end

function setup_c2(Ω)
    W = gridspace(Ω)
    uₕ = Rₕ(W, source_fn)
    return (element(W), uₕ), (element(W), uₕ)
end

# --- Candidate 3: the assembled SpMV in SemidiscretizeRHS --------------------------- #
#
# `SemidiscretizeRHS` calls `mul!(du, A, u, -1, 1)` on the CSC matrix (SparseArrays'
# column scatter, which no task split keeps disjoint). The prototype is a row-parallel CSR
# product under `@batch`, the CSR layout taken from `sparse(transpose(A))`. So that the row
# measures `@batch` alone, its serial side is the same CSR loop without `@batch`, not the
# CSC product: the layout change is a second, separate gain. The script prints it on its own
# line, with the CSC product's time and the cost of building the CSR copy (`A` is symmetric
# here, so the copy has `A`'s values). Left out: keeping that copy in step with `A`, a
# backend or storage decision for S2.2. The CSR rows are summed in a different order than
# the CSC scatter, hence the tolerance in the layout check.

c3_csc!(du, A, u) = mul!(du, A, u, -1, 1)

function c3_serial!(du, At::SparseMatrixCSC, u)
    rows = rowvals(At)
    vals = nonzeros(At)
    colptr = At.colptr
    for i in 1:size(At, 2)
        acc = 0.0
        @inbounds for k in colptr[i]:(colptr[i + 1] - 1)
            acc += vals[k] * u[rows[k]]
        end
        @inbounds du[i] -= acc
    end
    return du
end

function c3_batch!(du, At::SparseMatrixCSC, u)
    rows = rowvals(At)
    vals = nonzeros(At)
    colptr = At.colptr
    @batch for i in 1:size(At, 2)
        acc = 0.0
        @inbounds for k in colptr[i]:(colptr[i + 1] - 1)
            acc += vals[k] * u[rows[k]]
        end
        @inbounds du[i] -= acc
    end
    return du
end

function setup_c3(Ω)
    A = assemble(poisson_form(gridspace(Ω)))
    u = randn(Xoshiro(3), size(A, 1))
    du = randn(Xoshiro(4), size(A, 1))
    At = sparse(transpose(A))
    return (copy(du), At, u), (copy(du), At, u)
end

# The CSC product, the serial CSR loop and the cost of building the CSR copy, on one grid.
function c3_layout(Ω)
    A = assemble(poisson_form(gridspace(Ω)))
    u = randn(Xoshiro(3), size(A, 1))
    du = randn(Xoshiro(4), size(A, 1))
    At = sparse(transpose(A))
    _check("assembled_spmv (CSC vs CSR)", c3_csc!(copy(du), A, u), c3_serial!(copy(du), At, u))
    tcsc = @belapsed c3_csc!($du, $A, $u) seconds = BENCH_SECONDS
    tcsr = @belapsed c3_serial!($du, $At, $u) seconds = BENCH_SECONDS
    tconv = @belapsed sparse(transpose($A)) seconds = BENCH_SECONDS
    return tcsc, tcsr, tconv
end

# --- Candidate 4: the Jacobi diagonal build ----------------------------------------- #
#
# The serial method is `_mf_apply!(CpuSerial(), DiagonalSink(d), form)`, line for line what
# `jacobi_preconditioner` runs. The prototype borrows the fused matrix-free sweep's row
# ownership and runs `@batch` over bands of the last axis. Each band walks its widened
# range. On the rim it keeps only the diagonal entries whose row it owns (`_OwnedAction`).
# That ownership exists in `src/` for action sinks alone, so the first two methods below
# extend it to `DiagonalSink`, in this process only. Every row then receives its entries in the
# serial order, which makes the result bitwise the serial one.

function Bramble._mf_owned(s::Bramble.DiagonalSink, lo::Int, hi::Int, ro::Int, ::Int)
    Bramble._OwnedAction(
        s, lo + ro, hi + ro, 0, -1)
end
@inline function Bramble._sink_entry!(
        o::Bramble._OwnedAction{<:Bramble.DiagonalSink}, row::Int, col::Int, weight, slot::Int
)
    o.lo <= row <= o.hi && Bramble._sink_entry!(o.s, row, col, weight, slot)
    return nothing
end
# The band visit gathers a unit's interior for action sinks only; the serial walk scatters
# every unit into a `DiagonalSink` (`visit_bilinear_stencil`), so the band scatters too.
@inline function Bramble._mf_visit_band!(
        p::Bramble._MFPass, s::Bramble.DiagonalSink, term::TERM, sp, ro::Int, co::Int
) where {TERM}
    dims = size(Bramble.indices(Bramble.mesh(sp)))
    len = last(dims)
    stride = prod(Base.front(dims))
    a, b = first(p.own), last(p.own)
    vlo, vhi = max(1, a - p.omax), min(len, b - p.omin)
    clo, chi = max(vlo, a - p.omin), min(vhi, b - p.omax)
    owned = Bramble._mf_owned(s, (a - 1) * stride + 1, b * stride, ro, co)
    lo, core, hi = clo > chi ? (vlo:vhi, 1:0, 1:0) : (vlo:(clo - 1), clo:chi, (chi + 1):vhi)
    Bramble._mf_visit_slab!(s, owned, term, sp, ro, co, lo, core, hi)
    return nothing
end

function c4_serial!(d, a)
    fill!(d, 0.0)
    Bramble._mf_apply!(Bramble.CpuSerial(), Bramble.DiagonalSink(d), a)
    return d
end

function c4_batch!(d, a, plan)
    fill!(d, 0.0)
    len = last(plan.dims)
    nbands = min(NTHREADS, len)
    args = Ref((d, a, plan.omin, plan.omax, len))
    @batch for b in 1:nbands
        _diag_band!(args[]..., nbands, b)
    end
    return d
end

@inline function _diag_band!(d, a, omin, omax, len, nbands, b)
    own = Bramble._band_range(1:len, nbands, b)
    pass = Bramble._MFPass(Bramble._MF_BAND, own, omin, omax, Bramble._MF_NO_COLLECT)
    Bramble._mf_apply!(pass, Bramble.DiagonalSink(d), a)
    return nothing
end

function setup_c4(Ω)
    a = poisson_form(gridspace(Ω))
    plan = Bramble._mf_plan(Bramble.CpuThreaded(), a)
    plan === nothing && error("batch_survey: no fused plan for the Poisson form")
    plan.interp && error("batch_survey: the Poisson form has a serial interpolation unit")
    d = zeros(npoints(Ω))
    return (d, a), (similar(d), a, plan)
end

# --- Candidate 5: the bilinear first fill / recording ------------------------------- #
#
# The serial method is `_form_coordinates` followed by `_coordinates_to_positions!`, as the
# first `assemble` runs them. The first is the coordinate walk, a count pass and then a
# fill pass over every (term, block) unit. The second is the CSC position search. The
# prototype runs both passes of the walk under `@batch` over the units, then the search
# under `@batch` over the coordinates.
#
# The walk is independent per unit, as notes F says. The count pass of a unit writes only
# that unit's point pointers and count, here into slot `u` instead of `push!`. Once the
# counts are known, a prefix sum fixes each unit's offset in `I` and `J`, which is the
# `base` the serial fill pass accumulates, so the fill pass of a unit writes only its own
# slice. Within a unit the walk stays serial: its sink advances one running counter, so
# splitting a unit would need the per-point offsets threaded through the walk. The gain is
# therefore capped by the number of units (3 for this form) and the largest unit's walk.
# The pattern construction (`sparse!`), the segment layout and the replay are left out, and
# all of them are sequential here. The row's share is that of the walk and the search
# together, and the script also prints the whole recording's share. Recording would matter
# in a one-shot assemble-and-solve, which none of the three solves is. Here it runs once,
# in the explicit solve's setup, so its share is small.

function c5_serial(W, A, ast)
    p = Bramble._form_coordinates(W, W, ast)
    Bramble._coordinates_to_positions!(A, p, ast)
    return p
end

@noinline _throw_missing_entry() = throw(ArgumentError("batch_survey: missing pattern entry"))

function _c5_search!(Iv::Vector{Int}, Jv::Vector{Int}, A::SparseMatrixCSC)
    @batch for k in 1:length(Iv)
        pos = Bramble._scatter_position(A, Iv[k], Jv[k])
        pos == 0 && _throw_missing_entry()
        Iv[k] = pos
    end
    return nothing
end

# The count pass of unit `u`, `_CoordPass`'s own branch with the `push!`es into slot `u`.
function _c5_count!(p, unit, u::Int)
    term, sp, ro, co, dr, dc, half = unit
    hp = Bramble.host_weights(sp)
    Bramble._validate_term_markers(term, Bramble.markers(mesh(sp)), p.context)
    npts = length(Bramble.indices(mesh(sp)))
    ptr = Vector{Int}(undef, npts + 1)
    sink = Bramble._CoordSink(ptr, p.I, p.J, 0, 0, dr, dc, half, false, 0)
    Bramble._coord_walk!(sink, term, hp, ro, co)
    @inbounds ptr[npts + 1] = sink.n + 1
    p.ptrs[u] = ptr
    p.counts[u] = sink.n
    p.margins[u] = Bramble._stencil_margin(term)
    p.halves[u] = half
    return nothing
end

# The fill pass of unit `u` at its offset `base`.
function _c5_fill!(p, unit, u::Int, base::Int)
    term, sp, ro, co, dr, dc, half = unit
    nd = Bramble._direct_count(p.counts[u], half)
    sink = Bramble._CoordSink(p.ptrs[u], p.I, p.J, base, base + nd, dr, dc, half, true, 0)
    Bramble._coord_walk!(sink, term, Bramble.host_weights(sp), ro, co)
    return nothing
end

function c5_batch(W, A, ast)
    units = Any[]
    Bramble._foreach_unit((args...) -> (push!(units, args); nothing), W, W, ast)
    nu = length(units)
    p = Bramble._CoordPass("the form's space")
    resize!(p.ptrs, nu)
    resize!(p.counts, nu)
    resize!(p.margins, nu)
    resize!(p.halves, nu)
    args = Ref((p, units))
    @batch for u in 1:nu
        a = args[]
        _c5_count!(a[1], a[2][u], u)
    end
    bases = zeros(Int, nu)
    total = 0
    for u in 1:nu
        bases[u] = total
        n, half = p.counts[u], p.halves[u]
        total += Bramble._direct_count(n, half) + Bramble._transposed_count(n, half)
    end
    resize!(p.I, total)
    resize!(p.J, total)
    p.fill = true
    fargs = Ref((p, units, bases))
    @batch for u in 1:nu
        a = fargs[]
        _c5_fill!(a[1], a[2][u], u, a[3][u])
    end
    p.unit, p.base = nu, total
    _c5_search!(p.I, p.J, A)
    return p
end

function setup_c5(Ω)
    W = gridspace(Ω)
    a = poisson_form(W)
    A = assemble(a)
    return (W, A, a.ast), (W, A, a.ast)
end

# The units and their entry counts, to show the load the unit split has to balance.
function c5_units(Ω)
    W = gridspace(Ω)
    a = poisson_form(W)
    p = Bramble._form_coordinates(W, W, a.ast)
    return p.counts
end

# --- Candidate 6: the sparse Dirichlet sweep `_dirichlet_bc_rows!` ------------------ #
#
# The serial method is `_dirichlet_bc_rows!(A, entries)` on the mesh's `:boundary` marker.
# The prototype runs `@batch` over columns, each column's stored values written only by its
# own task. It leaves out the rare `A[j, j] = one(T)` insertion of a missing diagonal,
# which changes the pattern and cannot run in parallel. The script checks that every
# constrained column already stores its diagonal (true for any assembled Poisson matrix),
# so nothing is skipped here. The sweep is idempotent, so repeated timing runs see the
# same work.

c6_serial!(A, entries) = Bramble._dirichlet_bc_rows!(A, entries)

function c6_batch!(A::SparseMatrixCSC, entries)
    rows = rowvals(A)
    vals = nonzeros(A)
    @batch for j in 1:size(A, 2)
        column_is_constrained = Bramble._row_marked(entries, j)
        @inbounds for k in nzrange(A, j)
            row = rows[k]
            if column_is_constrained && row == j
                vals[k] = 1.0
            elseif Bramble._row_marked(entries, row)
                vals[k] = 0.0
            end
        end
    end
    return A
end

function setup_c6(Ω)
    W = gridspace(Ω)
    A = assemble(poisson_form(W))
    entries = Bramble._leaf_entries(Bramble.leaf_spaces_offsets(W), (:boundary,), nothing)
    for j in axes(A, 2)
        Bramble._row_marked(entries, j) || continue
        any(k -> rowvals(A)[k] == j, nzrange(A, j)) ||
            error("batch_survey: a constrained column has no stored diagonal")
    end
    return (copy(A), entries), (copy(A), entries)
end

# --- Candidate 7: `snorm₁ₕ`, the squared seminorm along each direction -------------- #
#
# Serial: `Bramble._snorm₁ₕ_sq(uₕ)`, both directions of `_seminorm_sq_along`. Prototype:
# the same per-line body under `@batch reduction = (+, s)` over the lines. Faithful; the
# sum is reordered across tasks, hence the tolerance.

# One line's contribution, `_seminorm_sq_along`'s loop body verbatim (Float64 data). The
# stride is `sd`, not `stride`: `@batch` treats a local named like a global function (or
# `LinearAlgebra.I`) as that global, and then fails to build its closure.
@inline function _seminorm_line(
        J, factors, f₁, li, data, h, n₁::Int, sd::Int, ::Val{d}, ::Val{D}
) where {d, D}
    c = 1.0
    for k in 2:D
        c *= factors[k][J[k - 1]]
    end
    offset = li[CartesianIndex(1, Tuple(J)...)] - 1
    line_sum = 0.0
    if d == 1
        @inbounds @simd for i₁ in 2:n₁
            k = offset + i₁
            δ = (data[k] - data[k - 1]) / h[i₁]
            line_sum = muladd(f₁[i₁], δ * δ, line_sum)
        end
    else
        @inbounds @simd for i₁ in 1:n₁
            k = offset + i₁
            δ = data[k] - data[k - sd]
            line_sum = muladd(f₁[i₁], δ * δ, line_sum)
        end
        ih = inv(h[J[d - 1]])
        c *= ih * ih
    end
    return c * line_sum
end

function _seminorm_batch(data, space, Ωₕ, li, vd::Val{d}, vD::Val{D}) where {d, D}
    h = Bramble.backward_spacings_for_derivative(Ωₕ(d))
    w = Bramble.weights(space, Bramble.Innerplus(), d)
    dims = size(li)
    factors = w.factors
    f₁ = first(factors)
    n₁ = first(dims)
    sd = d == 1 ? 1 : prod(ntuple(k -> dims[k], Val(d - 1)))
    lines = CartesianIndices(ntuple(
        k -> k + 1 == d ? (2:dims[k + 1]) : (1:dims[k + 1]), Val(D - 1)))
    return _seminorm_reduce(lines, factors, f₁, li, data, h, n₁, sd, vd, vD)
end

function _seminorm_reduce(lines, factors, f₁, li, data, h, n₁, sd, vd, vD)
    args = Ref((lines, factors, f₁, li, data, h, n₁, sd, vd, vD))
    s = 0.0
    @batch reduction = ((+, s),) for o in 1:length(lines)
        a = args[]
        s += _seminorm_line(a[1][o], Base.tail(a)...)
    end
    return s
end

c7_serial(uₕ) = Bramble._snorm₁ₕ_sq(uₕ)

function c7_batch(uₕ)
    W = space(uₕ)
    Ωₕ = mesh(W)
    li = LinearIndices(npoints(Ωₕ, Tuple))
    data = parent(uₕ)
    return _seminorm_batch(data, W, Ωₕ, li, Val(1), Val(2)) +
           _seminorm_batch(data, W, Ωₕ, li, Val(2), Val(2))
end

function setup_c7(Ω)
    uₕ = Rₕ(gridspace(Ω), source_fn)
    return (uₕ,), (uₕ,)
end

# --- Candidate 8: GMG and smoother vector updates ----------------------------------- #
#
# Prototyped: the Chebyshev smoother's two broadcasts per degree step,
# `d .= c .* d .+ e .* dinv .* r` and `y .+= d` (`_chebyshev!`, the default smoother, which
# runs them on every level of every cycle), each as its own `@batch` loop rather than fused,
# so the prototype changes only who runs the loop. The other updates the notes list (Jacobi
# and red-black updates, `x .+= r`, `copyto!`, `fill!`, `ldiv!`, `mul!`'s `β` scaling) are
# the same O(ndofs) memory-bound shape and are not timed separately; the share below counts
# all of them.

function c8_serial!(y, d, dinv, r, c, e)
    d .= c .* d .+ e .* dinv .* r
    y .+= d
    return y
end

function c8_batch!(y, d, dinv, r, c, e)
    @batch for i in eachindex(d)
        @inbounds d[i] = c * d[i] + e * dinv[i] * r[i]
    end
    @batch for i in eachindex(y)
        @inbounds y[i] += d[i]
    end
    return y
end

function setup_c8(Ω)
    n = npoints(Ω)
    rng = Xoshiro(8)
    dinv, r = rand(rng, n), randn(rng, n)
    # c < 1 keeps `d` bounded over the many repeated evaluations of the timing loop.
    return (zeros(n), zeros(n), dinv, r, 0.5, 0.3), (zeros(n), zeros(n), dinv, r, 0.5, 0.3)
end

# --- Candidate 9: setup-time weight builders ---------------------------------------- #
#
# Prototyped: `_average_weights!(v, Ωₕ, Forward(), Val(1))`, the one builder that fills a
# full O(ndofs) vector on a 2D grid, under `@batch` over the last axis in the serial
# iteration order. Left out: the 1D `_innerh_weights!`/`_innerplus_weights!`, which on a 2D
# grid fill per-axis vectors of `n` entries only (1025 here, too short to split) since the
# separable weights replaced full-grid fills. The share counts all three builders.

c9_serial!(v, Ωₕ) = (Bramble._average_weights!(v, Ωₕ, Bramble.Forward(), Val(1)); v)

function c9_batch!(v, Ωₕ)
    dims = npoints(Ωₕ, Tuple)
    n₁ = dims[1]
    @batch for j in 1:dims[2]
        off = (j - 1) * n₁
        @inbounds @simd for i in 1:n₁
            v[off + i] = i == n₁ ? 0.0 : 1.0
        end
    end
    return v
end

function setup_c9(Ω)
    n = npoints(Ω)
    return (fill(NaN, n), Ω), (fill(NaN, n), Ω)
end

# --- The candidate table ------------------------------------------------------------- #

struct Candidate
    k::Int
    name::String
    setup::Function
    serial::Function
    batch::Function
    rtol::Float64
end

const CANDIDATES = [
    Candidate(1, "kron_fused", setup_c1, c1_serial!, c1_batch!, 1e-14),
    Candidate(2, "laplacian", setup_c2, c2_serial!, c2_batch!, 1e-14),
    Candidate(3, "assembled_spmv", setup_c3, c3_serial!, c3_batch!, 1e-12),
    Candidate(4, "jacobi_diagonal", setup_c4, c4_serial!, c4_batch!, 0.0),
    Candidate(5, "bilinear_recording", setup_c5, c5_serial, c5_batch, 0.0),
    Candidate(6, "dirichlet_sweep", setup_c6, c6_serial!, c6_batch!, 0.0),
    Candidate(7, "seminorm", setup_c7, c7_serial, c7_batch, 1e-12),
    Candidate(8, "gmg_vector_updates", setup_c8, c8_serial!, c8_batch!, 1e-14),
    Candidate(9, "weight_builders", setup_c9, c9_serial!, c9_batch!, 0.0)
]

_result(x::AbstractVector) = copy(x)
_result(x::Bramble.VectorElement) = copy(parent(x))
_result(x::SparseMatrixCSC) = copy(nonzeros(x))
_result(x::Number) = [x]
_result(p::Bramble._CoordPass) = vcat(p.I, p.J, p.counts, p.margins, p.halves, p.ptrs...)

const N_REPEATS = 5

# One fresh setup, one call of each, compared; then `N_REPEATS` interleaved pairs of
# timings (serial, then batch), each the minimum of one `@belapsed`. The row reports the
# medians, and the spread of the per-repeat speedup is printed beside it.
function time_candidate(c::Candidate, Ω)
    sargs, bargs = c.setup(Ω)
    rs = _result(c.serial(sargs...))
    rb = _result(c.batch(bargs...))
    c.rtol == 0 ? (rs == rb || _check(c.name, rs, rb; rtol = 0.0)) :
    _check(c.name, rs, rb; rtol = c.rtol)
    f, g = c.serial, c.batch
    ts, tb = zeros(N_REPEATS), zeros(N_REPEATS)
    for r in 1:N_REPEATS
        ts[r] = @belapsed $f($(sargs)...) seconds = BENCH_SECONDS / 2
        tb[r] = @belapsed $g($(bargs)...) seconds = BENCH_SECONDS / 2
    end
    return ts, tb
end

_median(v) = (w = sort(v); n = length(w); isodd(n) ? w[(n + 1) ÷ 2] : (w[n ÷ 2] + w[n ÷ 2 + 1]) / 2)

# --- The three representative solves ------------------------------------------------- #

# Allocation-free PCG (`P === nothing`: plain CG); the vector updates are the script's own.
function pcg!(x, A, b, P, r, z, p, q; tol = 1e-8, maxiter = 500)
    fill!(x, 0.0)
    copyto!(r, b)
    P === nothing ? copyto!(z, r) : ldiv!(z, P, r)
    copyto!(p, z)
    rz = dot(r, z)
    target = tol * norm(b)
    for k in 1:maxiter
        mul!(q, A, p)
        α = rz / dot(p, q)
        axpy!(α, p, x)
        axpy!(-α, q, r)
        norm(r) <= target && return k
        P === nothing ? copyto!(z, r) : ldiv!(z, P, r)
        rz₊ = dot(r, z)
        axpby!(true, z, rz₊ / rz, p)
        rz = rz₊
    end
    return maxiter
end

@noinline function solve_gmg_cg(Ω, b, work)
    P = gmg_preconditioner(poisson_form, Ω)
    op = last(P.ops)
    return pcg!(work[1], op, b, P, work[2], work[3], work[4], work[5]; maxiter = 500)
end

@noinline function solve_explicit(Ω, dt, nsteps)
    W = gridspace(Ω)
    fₕ = Rₕ(W, source_fn)
    sd = semidiscretize(poisson_form(W), form(W, v -> innerₕ(fₕ, v)))
    rhs = semidiscretize_rhs(sd)
    u = zeros(npoints(Ω))
    du = similar(u)
    t = 0.0
    for _ in 1:nsteps
        rhs(du, u, nothing, t)
        u .+= dt .* du
        t += dt
    end
    return u
end

@noinline function solve_kron_cg(Ω, b, work)
    K = kronecker_operator(poisson_form(gridspace(Ω)))
    return pcg!(work[1], K, b, nothing, work[2], work[3], work[4], work[5]; maxiter = 300)
end

# The setup alone of the explicit solve, to report its weight.
@noinline function setup_explicit(Ω)
    W = gridspace(Ω)
    fₕ = Rₕ(W, source_fn)
    sd = semidiscretize(poisson_form(W), form(W, v -> innerₕ(fₕ, v)))
    return semidiscretize_rhs(sd)
end

# --- Matching profile frames to candidates ------------------------------------------- #

const _SRC_LINES = Dict{String, Vector{String}}()
function _line_text(file::String, line::Int)
    lines = get!(_SRC_LINES, file) do
        isfile(file) ? readlines(file) : String[]
    end
    return 1 <= line <= length(lines) ? lines[line] : ""
end

const _UPDATE_RE = r"(\.\+=|\.-=|\.\*=|\.=|copyto!\(|fill!\()"

# Which candidates (1..9, and 10 for candidate 5's whole recording) a stack frame belongs to.
function frame_candidates(fr)
    file = String(fr.file)
    occursin("Bramble", file) || occursin("SparseArrays", file) || return ()
    base = basename(file)
    fn = fr.func
    insrc = occursin("/src/", file) && occursin("Bramble", file)
    out = Int[]
    if insrc
        base == "kronecker.jl" && fn === :_kron_fused! && push!(out, 1)
        base == "vector_calculus.jl" &&
            fn in (:Δₕ!, :_laplacian_direction!, :_accumulate_laplacian!) && push!(out, 2)
        if base == "semidiscrete_rhs.jl" || base == "semidiscrete.jl"
            occursin("mul!(du, A, u", _line_text(file, fr.line)) && push!(out, 3)
        end
        if base == "matrix_free_preconditioners.jl"
            txt = _line_text(file, fr.line)
            occursin("DiagonalSink(d)", txt) && occursin("_mf_apply!", txt) && push!(out, 4)
            fn === :ldiv! && push!(out, 8)
        end
        # 5 is the prototyped coordinate walk and position search; 10 (reported, not a
        # row) is the whole recording: walk, search, `sparse!`, segment layout, first fill.
        fn in (:_form_coordinates, :_coord_walk!, :_coordinates_to_positions!) &&
            push!(out, 5)
        (fn === :_record_bilinear_core! || base == "bilinear_pattern.jl") && push!(out, 10)
        fn === :_dirichlet_bc_rows! && push!(out, 6)
        fn in (:_seminorm_sq_along, :_snorm₁ₕ_sq) && push!(out, 7)
        if base in ("gmg_smoothers.jl", "chebyshev.jl", "gmg_cycles.jl")
            txt = _line_text(file, fr.line)
            if fn === :_rb_update! ||
               (occursin(_UPDATE_RE, txt) && !occursin("mul!(", txt) &&
                !occursin("zeros(", txt) && !occursin("_fmg", txt))
                push!(out, 8)
            end
        end
        if base == "matrix_free.jl" && fn === :mul!
            txt = _line_text(file, fr.line)
            (occursin("fill!(", txt) || occursin(".*=", txt)) && push!(out, 8)
        end
        fn in (:_innerh_weights!, :_innerplus_weights!, :_innerplus_mean_weights!,
            :_average_weights!) && push!(out, 9)
    end
    return Tuple(unique(out))
end

# Share of each candidate in the samples of the calling task while `f` runs: every such
# sample is the solve's time, whether or not its stack unwinds to the solve's frame (a
# sample inside BLAS or LAPACK may not), so the total is the task's sample count. The
# fraction that does unwind to `solve` is returned as a check on the profiler.
function profile_shares(solve::Symbol, f)
    Profile.clear()
    Profile.init(n = 2 * 10^7, delay = 0.0002)
    task = UInt(pointer_from_objref(current_task()))
    @profile f()
    data = Profile.fetch(include_meta = true)
    blocks = UnitRange{Int}[]
    start = 1
    for i in eachindex(data)
        if Profile.is_block_end(data, i)
            data[i - Profile.META_OFFSET_TASKID] == task &&
                push!(blocks, start:(i - Profile.nmeta - 2))
            start = i + 1
        end
    end
    ips = unique(reduce(vcat, (data[r] for r in blocks); init = UInt64[]))
    lidict = Profile.getdict(ips)
    framecache = Dict{UInt64, Tuple{Bool, Vector{Int}}}()
    for ip in ips
        frames = get(lidict, ip, Base.StackTraces.StackFrame[])
        ks = Int[]
        for fr in frames
            append!(ks, frame_candidates(fr))
        end
        framecache[ip] = (any(fr -> fr.func === solve, frames), ks)
    end
    hits = zeros(Int, 10)
    unwound = 0
    seen = falses(10)
    for r in blocks
        fill!(seen, false)
        insolve = false
        for i in r
            iss, ks = framecache[data[i]]
            insolve |= iss
            for k in ks
                seen[k] = true
            end
        end
        unwound += insolve
        hits .+= seen
    end
    total = length(blocks)
    return total, unwound / max(total, 1), hits ./ max(total, 1)
end

# Why a candidate can show no sample (printed with a share of 0).
const UNREACHED = Dict(
    1 => "unexpected: kron_cg applies the operator every iteration",
    2 => "none of the three solves calls Δₕ!: the forms are assembled or matrix-free",
    3 => "unexpected: explicit runs the product every step",
    4 => "runs once per GMG level in setup, below the sampling resolution",
    5 => "runs once, in semidiscretize, below the sampling resolution",
    6 => "none of the three solves has a Dirichlet row (semidiscretize_rhs and gmg forms forbid them)",
    7 => "none of the three solves computes an H1 norm or seminorm",
    8 => "unexpected: the GMG cycles run them on every level",
    9 => "the 2D builders fill per-axis vectors once per space, below the sampling resolution"
)

# --- Run ----------------------------------------------------------------------------- #

function main()
    Ωs, Ωl = jitter_mesh(N_SMALL), jitter_mesh(N_LARGE)
    nl = npoints(Ωl)
    println("batch_survey: Julia $(VERSION), Bramble $(pkgversion(Bramble)), Polyester $(pkgversion(Polyester))")
    println("threads: $NTHREADS (Julia), BLAS: $(BLAS.get_num_threads())")
    println("small grid: $(N_SMALL)x$(N_SMALL) = $(npoints(Ωs)) dofs; large grid: $(N_LARGE)x$(N_LARGE) = $nl dofs (2D, points jittered ±0.3h, seed $SEED)")
    println("form: innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) (mass plus diffusion, natural boundary)")
    println("solve gmg_cg:   gmg_preconditioner (matrix-free levels, Chebyshev smoother) + PCG to rel. residual 1e-8, setup included")
    println("solve explicit: semidiscretize + semidiscretize_rhs + $EXPLICIT_STEPS forward-Euler steps through SemidiscretizeRHS (CSC operator), setup included")
    println("solve kron_cg:  kronecker_operator + unpreconditioned CG to 1e-8 or 300 iterations, setup included")

    # The solves, serial, on the large grid.
    b = randn(Xoshiro(7), nl)
    work = ntuple(_ -> zeros(nl), 5)
    hmin = minimum(minimum(diff(collect(points(Ωl(d))))) for d in 1:2)
    dt = 0.1 * hmin^2
    solves = (
        (:solve_gmg_cg, "gmg_cg", () -> solve_gmg_cg(Ωl, b, work)),
        (:solve_explicit, "explicit", () -> solve_explicit(Ωl, dt, EXPLICIT_STEPS)),
        (:solve_kron_cg, "kron_cg", () -> solve_kron_cg(Ωl, b, work))
    )
    # The first profile in a process attributed no sample to setup-only frames (candidate 4)
    # that later ones did, so one is taken and discarded.
    first(solves)[3]()
    profile_shares(first(solves)[1], first(solves)[3])
    shares = zeros(10, length(solves))
    t_expl = NaN
    for (j, (sym, label, f)) in enumerate(solves)
        r = f()                      # warm
        t = minimum(@elapsed(f()) for _ in 1:2)
        label == "explicit" && (t_expl = t)
        extra = label == "explicit" ? "" : ", $(r) iterations"
        nsamp, unwound, sh = profile_shares(sym, () -> (f(); f(); f()))
        shares[:, j] = sh
        @printf("solve %-9s serial wall %.3f s%s, %d profile samples (%.0f%% unwind to the solve)\n",
            label, t, extra, nsamp, 100 * unwound)
    end
    setup_explicit(Ωl)
    ts_setup = minimum(@elapsed(setup_explicit(Ωl)) for _ in 1:2)
    @printf("explicit: setup is %.3f s of %.3f s (%.0f%%); the shares of candidates 3 and 5 move with the step count\n",
        ts_setup, t_expl, 100 * ts_setup / t_expl)

    println("shares per solve (gmg_cg, explicit, kron_cg):")
    for c in CANDIDATES
        @printf("  k=%d %-20s %.3f %.3f %.3f\n", c.k, c.name, shares[c.k, :]...)
    end
    @printf("  k=5 whole recording      %.3f %.3f %.3f (walk, search, sparse!, layout, first fill; the row uses walk and search)\n",
        shares[10, :]...)

    rows = String[]
    for c in CANDIDATES
        tss, tbs = time_candidate(c, Ωs)
        tsl, tbl = time_candidate(c, Ωl)
        ss, bs, sl, bl = _median(tss), _median(tbs), _median(tsl), _median(tbl)
        share = maximum(shares[c.k, :])
        rs, rl = tss ./ tbs, tsl ./ tbl
        @printf("  k=%d %-20s speedup over %d repeats: small min %.2fx max %.2fx, large min %.2fx max %.2fx\n",
            c.k, c.name, N_REPEATS, minimum(rs), maximum(rs), minimum(rl), maximum(rl))
        if c.k == 3
            for (label, Ω) in (("small", Ωs), ("large", Ωl))
                tcsc, tcsr, tconv = c3_layout(Ω)
                @printf("k=3 layout, %s grid: CSC mul! %.3e s, serial CSR %.3e s (CSC to CSR %.2fx); building the CSR copy %.3e s = %.1f CSC products\n",
                    label, tcsc, tcsr, tcsc / tcsr, tconv, tconv / tcsc)
            end
        end
        c.k == 5 && println("k=5 units: $(length(c5_units(Ωl))), entries per unit on the large grid $(c5_units(Ωl))")
        if c.k == 1
            # Cross-check of the profiler against a call count: CG runs one product per
            # iteration.
            iters = solve_kron_cg(Ωl, b, work)
            tk = minimum(@elapsed(solve_kron_cg(Ωl, b, work)) for _ in 1:2)
            @printf("cross-check k=1: %d products x %.3e s = %.3f of the kron_cg solve (profile: %.3f)\n",
                iters, sl, iters * sl / tk, share)
            (y, K, x), _ = setup_c1(Ωl)
            _check("kron_fused (captured)", c1_serial!(copy(y), K, x), c1_batch_captured!(y, K, x))
            tc = @belapsed c1_batch_captured!($y, $K, $x) seconds = BENCH_SECONDS
            @printf("k=1 with y, x captured by @batch (PtrArray): %.3e s on the large grid, %.2fx\n",
                tc, sl / tc)
        end
        share == 0 && println("note k=$(c.k) $(c.name): no profile sample in any solve, share 0 ($(UNREACHED[c.k]))")
        verdict = (sl / bl >= 1.5 && share >= 0.05) ? "ABOVE" : "BELOW"
        @printf("  k=%d %-20s median speedup small %.2fx, large %.2fx\n", c.k, c.name, ss / bs, sl / bl)
        push!(rows,
            @sprintf("SURVEY k=%d name=%s serial_small=%.6e batch_small=%.6e serial_large=%.6e batch_large=%.6e share=%.4f %s",
                c.k, c.name, ss, bs, sl, bl, share, verdict))
    end
    foreach(println, rows)
    return nothing
end

main()
