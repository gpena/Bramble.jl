# test/ext/polyester_ext.jl: the Polyester extension (ext/BramblePolyesterExt.jl).
#
# Gated like every other ext/*.jl file (test/runtests.jl only reaches this group under
# BRAMBLE_TEST_GROUP=ext or full). Standalone:
#
#   julia --project=test -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/polyester_ext.jl")'
module TestPolyesterExt

using Test
using Bramble
using Bramble: CpuPolyester, Serial, execution_policy, test_space, _normalize_dirichlet,
               apply_dirichlet_conditions!, allocate_system_matrix, assemble_parallel!,
               inner₊ₓ, D₋ₓ, S₊ₓ!, S₊ᵧ!, S₊₂!, S₋ₓ!, S₋ᵧ!, S₋₂!, GeometricMeshHierarchy,
               change_points!
using Polyester
using SparseArrays
using SparseArrays: getcolptr
using LinearAlgebra: issymmetric, mul!, ldiv!
using Random
using ..TestUtils: alloc_test, _BROADCAST_SIZES, _broadcast_space, _check_broadcast_equal, _check_stencils,
                   _fillnz!, _grid, _mg_jitter_mesh, _mg_spd, _mg_transfer_meshes, _reset_seen!, _sine_source,
                   _spy, _STENCIL_F, _STENCIL_G, _stencil_mesh_pair, _stencil_op!, _threads_seen,
                   _unit_cube

const ZERO_BC = :dir => (x -> 0.0)

# One matched CpuPolyester/Parallel/Serial triple -- same domain, same mesh size, one backend
# swapped for another -- mirroring test/ext/sparse_csr_ext.jl's own `_poisson_pair`.
function _poisson_pair(dim::Val{D}, n::Integer; source = _sine_source(dim)) where {D}
    Iᴰ = _unit_cube(dim)
    Ωd = domain(Iᴰ, :dir => boundary_symbols(Iᴰ))
    Ωp = _grid(dim, Ωd, n; backend = backend(policy = Parallel()))
    Ωb = _grid(dim, Ωd, n; backend = backend(policy = CpuPolyester()))
    Ωs = _grid(dim, Ωd, n; backend = backend(policy = Serial()))
    Wp, Wb, Ws = gridspace(Ωp), gridspace(Ωb), gridspace(Ωs)

    build = (W) -> begin
        fₕ = Rₕ(W, source)
        a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(W, v -> innerₕ(fₕ, v))
        return a, l
    end
    ap, lp = build(Wp)
    ab, lb = build(Wb)
    as, ls = build(Ws)
    return (; Wp = Wp, Wb = Wb, Ws = Ws, ap = ap, lp = lp, ab = ab, lb = lb, as = as, ls = ls)
end

# The paths of "allocation under CpuPolyester" below. The child process counting Threads
# entry points reads this block from the file, between the two marker lines, so that it runs
# exactly the same calls. Each entry is a name, whether the file asserts bitwise equality
# with `CpuSerial` for that operator, how many `Base.RefValue`s a warm `CpuPolyester` call
# allocates (measured at -O1), and a setup taking (grid points per axis, policy) on a
# non-uniform 2D grid and returning the call to measure, a function reading its result, and
# the number of multigrid levels (0 elsewhere).
# Every path allocates nothing at all under `CpuPolyester` (gpena/Bramble.jl#433,
# gpena/Bramble.jl#437): what its loop captures crosses `@batch` as plain arrays and isbits
# values, so Polyester's argument box stays on the stack. An `avgₕ!` source closure over an
# array is never split (a rebuilt closure would capture a `PtrArray` in place of its
# `Vector`): its kernel reaches the tasks whole through a typed slot, and allocates nothing
# either. A path that boxes would be listed here with its bound.
const _PA_BOX_CEILINGS = Dict{String, Int}()

# BEGIN _pa paths
using Bramble: CpuPolyester, CpuSerial, CpuThreaded, change_points!, semidiscretize_rhs,
               allocate_system_matrix, D₋ₓ, inner₊ₓ, S₊ₓ!, restrict_to
using LinearAlgebra: Diagonal, mul!, ldiv!
using Random: Xoshiro, randn
using SparseArrays: nonzeros, spdiagm

function _pa_jitter(n, policy; seed = 4321)
    rng = Xoshiro(seed)
    X = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ω = mesh(domain(X, :dir => boundary_symbols(X)), (n, n), (true, true);
        backend = backend(policy = policy))
    h = 1 / (n - 1)
    function pts()
        x = collect(range(0.0, 1.0; length = n)) .+ 0.3h .* (2 .* rand(rng, n) .- 1)
        x[1], x[end] = 0.0, 1.0
        return sort!(x)
    end
    change_points!(Ω, (pts(), pts()))
    return Ω
end
_pa_space(n, policy) = gridspace(_pa_jitter(n, policy))
_pa_leaf(n, policy) = gridspace(_pa_jitter(n, policy; seed = n))

_pa_g(x) = sin(3x[1] + 2x[2]) + x[1] * x[2]
_pa_h(x) = x[1] * x[2]
_pa_spd(W) = (κ = Rₕ(W, x -> 1 + sum(abs2, x));
    form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))))
_pa_poisson(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
_pa_copy(u) = copy(parent(u))
_pa_flat(e) = reduce(vcat, [_pa_copy(e[i][j]) for i in 1:2 for j in 1:2])
_pa_two_leaf(u, v) = innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(2)), D₋ₓ(v(2))) + innerₕ(u(2), v(2))
_pa_case(call, result; levels = 0) = (; call, result, levels)
_pa_comps(u) = reduce(vcat, [_pa_copy(c) for c in Bramble.components(u)])
_pa_pair(x) = (_pa_g(x), _pa_h(x))
_pa_times0d(a, c) = 2.0 * a * c + 1.0

# A hand-built two-term Kronecker operator on an `n × n` grid that `kronecker_operator`
# does not build yet: non-symmetric banded factors on both axes (a row gather on axis 1
# over the neighbour lines of axis 2), plus a mass-shaped term.
function _pa_kron_general(n, policy)
    rng = Xoshiro(6)
    band(lo, hi) = spdiagm((k => randn(rng, n - abs(k)) for k in (-lo):hi)...)
    dg() = Diagonal(rand(rng, n) .+ 0.5)
    terms = (
        Bramble._kron_term((), (band(1, 2), band(1, 1))), Bramble._kron_term((2.0,), (dg(), dg()))
    )
    return Bramble.KroneckerLinearOperator{Float64, 2, typeof(terms), typeof(policy)}(
        terms, (n, n), n^2, policy)
end

function _pa_paths()
    P = Tuple{String, Bool, Int, Function}[]
    push!(P, ("difference D₋ₓ!", true, 0, (n, p) -> begin
        W = _pa_space(n, p)
        u = Rₕ(W, _pa_g)
        w = similar(u)
        _pa_case(() -> Bramble.D₋ₓ!(w, u), () -> _pa_copy(w))
    end))
    push!(P, ("average Mₓ!", true, 0, (n, p) -> begin
        W = _pa_space(n, p)
        u = Rₕ(W, _pa_g)
        w = similar(u)
        _pa_case(() -> Bramble.Mₓ!(w, u), () -> _pa_copy(w))
    end))
    push!(P, ("avgₕ!", false, 0, (n, p) -> begin
        u = element(_pa_space(n, p))
        _pa_case(() -> avgₕ!(u, _pa_g), () -> _pa_copy(u))
    end))
    push!(P, ("shift S₊ₓ!", true, 0, (n, p) -> begin
        W = _pa_space(n, p)
        u = Rₕ(W, _pa_g)
        w = similar(u)
        _pa_case(() -> S₊ₓ!(w, u), () -> _pa_copy(w))
    end))
    push!(P, ("divₕ!", true, 0, (n, p) -> begin
        W = _pa_space(n, p)
        u = (Rₕ(W, _pa_g), Rₕ(W, _pa_h))
        v = similar(u[1])
        _pa_case(() -> Bramble.divₕ!(v, u), () -> _pa_copy(v))
    end))
    push!(P, ("curlₕ!", true, 0, (n, p) -> begin
        W = _pa_space(n, p)
        u = (Rₕ(W, _pa_g), Rₕ(W, _pa_h))
        v = similar(u[1])
        _pa_case(() -> Bramble.curlₕ!(v, u), () -> _pa_copy(v))
    end))
    push!(P,
        ("εₕ!",
            true,
            0,
            (n, p) -> begin
                W = _pa_space(n, p)
                u = (Rₕ(W, _pa_g), Rₕ(W, _pa_h))
                e = ntuple(_ -> ntuple(_ -> similar(u[1]), 2), 2)
                _pa_case(() -> Bramble.εₕ!(e, u), () -> _pa_flat(e))
            end))
    push!(P, ("broadcast", true, 0, (n, p) -> begin
        W = _pa_space(n, p)
        u = Rₕ(W, _pa_g)
        w = Rₕ(W, x -> x[1])
        v = similar(u)
        _pa_case(() -> (v .= 2.0 .* u .+ w), () -> _pa_copy(v))
    end))
    push!(P, ("innerₕ", false, 0, (n, p) -> begin
        W = _pa_space(n, p)
        u = Rₕ(W, _pa_g)
        w = Rₕ(W, x -> x[1])
        _pa_case(() -> innerₕ(u, w), () -> innerₕ(u, w))
    end))
    push!(P, ("inner₊ₓ", false, 0, (n, p) -> begin
        W = _pa_space(n, p)
        u = Rₕ(W, _pa_g)
        w = Rₕ(W, x -> x[1])
        _pa_case(() -> inner₊ₓ(u, w), () -> inner₊ₓ(u, w))
    end))
    push!(P,
        ("innerₕ masked", false, 0,
            (n, p) -> begin
                W = _pa_space(n, p)
                u = Rₕ(W, _pa_g)
                w = Rₕ(W, x -> x[1])
                m = (:dir,)
                _pa_case(() -> innerₕ(u, w; markers = m), () -> innerₕ(u, w; markers = m))
            end))
    push!(P, ("assemble! bilinear (replay)", false, 0, (n, p) -> begin
        a = _pa_poisson(_pa_space(n, p))
        A = allocate_system_matrix(a)
        _pa_case(() -> assemble!(A, a), () -> copy(A))
    end))
    push!(P, ("assemble! linear", false, 0, (n, p) -> begin
        W = _pa_space(n, p)
        f = Rₕ(W, _pa_g)
        l = form(W, v -> innerₕ(f, v))
        b = zeros(ndofs(W))
        _pa_case(() -> assemble!(b, l), () -> copy(b))
    end))
    push!(P, ("matrix-free fused", false, 0, (n, p) -> begin
        op = matrix_free_operator(_pa_poisson(_pa_space(n, p)))
        x = randn(Xoshiro(1), size(op, 2))
        y = similar(x)
        _pa_case(() -> mul!(y, op, x), () -> copy(y))
    end))
    push!(P,
        ("matrix-free per-unit",
            false,
            0,
            (n, p) -> begin
                V = Bramble.CompositeGridSpace((_pa_leaf(n, p), _pa_leaf(n ÷ 2 + 1, p)))
                a = form(V, V, _pa_two_leaf)
                op = matrix_free_operator(a)
                x = randn(Xoshiro(2), size(op, 2))
                y = similar(x)
                _pa_case(() -> mul!(y, op, x), () -> copy(y))
            end))
    push!(P, (
        "GMG V-cycle", false, 0, (n, p) -> begin
            Ω = _pa_jitter(n, p)
            Pc = gmg_preconditioner(_pa_spd, Ω; cycle = :V)
            b = randn(Xoshiro(3), npoints(Ω))
            y = similar(b)
            _pa_case(() -> ldiv!(y, Pc, b), () -> copy(y); levels = length(Pc.ops))
        end))
    push!(P, ("Kronecker mul!", false, 0, (n, p) -> begin
        K = kronecker_operator(_pa_poisson(_pa_space(n, p)))
        x = randn(Xoshiro(4), size(K, 2))
        y = similar(x)
        _pa_case(() -> mul!(y, K, x), () -> copy(y))
    end))
    push!(P, ("Kronecker mul! general", true, 0, (n, p) -> begin
        K = _pa_kron_general(n, p)
        x = randn(Xoshiro(7), size(K, 2))
        y = similar(x)
        _pa_case(() -> mul!(y, K, x, 0.5, 0.0), () -> copy(y))
    end))
    push!(P, ("explicit RHS", false, 0,
        (n, p) -> begin
            W = _pa_space(n, p)
            f = Rₕ(W, _pa_g)
            l = form(W, v -> innerₕ(f, v))
            r = semidiscretize_rhs(semidiscretize(_pa_poisson(W), l))
            u = randn(Xoshiro(5), length(r.inv_mass_diag))
            du = similar(u)
            _pa_case(() -> r(du, u, nothing, 0.0), () -> copy(du))
        end))
    push!(P, ("space weights", true, 0, (n, p) -> begin
        Ω = _pa_jitter(n, p)
        u = zeros(npoints(Ω))
        _pa_case(() -> Bramble._innerh_weights!(u, Ω), () -> copy(u))
    end))
    push!(P, (
        "avgₕ! masked", false, 0, (n, p) -> begin
            u = element(_pa_space(n, p))
            _pa_case(() -> avgₕ!(u, _pa_g; markers = (:dir,)), () -> _pa_copy(u))
        end))
    push!(P, ("project! composite avg", false, 0, (n, p) -> begin
        u = element(gridspace(_pa_jitter(n, p), Val(2)))
        _pa_case(() -> avgₕ!(u, _pa_pair), () -> _pa_comps(u))
    end))
    push!(P, ("csr spmv", false, 0,
        (n, p) -> begin
            A = assemble(_pa_poisson(_pa_space(n, p)))
            csr = Bramble._rhs_csr(p, Val(false), A)
            u = randn(Xoshiro(8), size(A, 2))
            du0 = randn(Xoshiro(9), size(A, 1))
            du = similar(du0)
            _pa_case(() -> (copyto!(du, du0); Bramble._rhs_spmv!(du, A, csr, u)), () -> copy(du))
        end))
    push!(P,
        ("assemble! restricted",
            false,
            0,
            (n, p) -> begin
                W = _pa_space(n, p)
                a = form(W, W,
                    (u, v) -> innerₕ(u, v; markers = (:dir,)) +
                              inner₊(∇ₕ(u), ∇ₕ(v); markers = (:interior,)))
                A = allocate_system_matrix(a)
                _pa_case(() -> assemble!(A, a), () -> copy(A))
            end))
    push!(P, ("assemble! Ref coefficient", false, 0,
        (n, p) -> begin
            W = _pa_space(n, p)
            θ = Ref(2.5)
            a = form(W, W, (u, v) -> θ * innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
            A = allocate_system_matrix(a)
            first = similar(nonzeros(A))
            call = () -> begin
                θ[] = 2.5
                assemble!(A, a)
                copyto!(first, nonzeros(A))
                θ[] = 4.0
                assemble!(A, a)
                return nothing
            end
            _pa_case(call, () -> vcat(first, nonzeros(A)))
        end))
    push!(P, ("assemble! linear interpolation", false, 0, (n, p) -> begin
        W = _pa_space(n, p)
        uc = Rₕ(_pa_leaf(n ÷ 2 + 1, p), _pa_g)
        l = form(W, v -> innerₕ(πₕ(uc), v))
        b = zeros(ndofs(W))
        _pa_case(() -> assemble!(b, l), () -> copy(b))
    end))
    push!(P,
        ("assemble! bilinear (searching)",
            false,
            0,
            (n, p) -> begin
                a = _pa_poisson(_pa_space(n, p))
                A = allocate_system_matrix(a)
                call = () -> begin
                    Bramble._zero_stored!(A)
                    Bramble._assemble_bilinear_parallel_core!(A, a.trial_space, a.test_space, a.ast)
                end
                _pa_case(call, () -> copy(A))
            end))
    push!(P, ("Rₕ!", true, 0, (n, p) -> begin
        u = element(_pa_space(n, p))
        _pa_case(() -> Rₕ!(u, _pa_g), () -> _pa_copy(u))
    end))
    push!(P, ("Rₕ! masked", true, 0, (n, p) -> begin
        u = element(_pa_space(n, p))
        _pa_case(() -> Rₕ!(u, _pa_g; markers = (:dir,)), () -> _pa_copy(u))
    end))
    push!(P, (
        "project! composite Rₕ", true, 0, (n, p) -> begin
            u = element(gridspace(_pa_jitter(n, p), Val(2)))
            rule = Bramble.PointValue(_pa_pair)
            _pa_case(() -> Bramble.project!(u, rule), () -> _pa_comps(u))
        end))
    push!(P, ("avgₕ! closure", false, 0, (n, p) -> begin
        u = element(_pa_space(n, p))
        c = [0.3, 0.7]
        f = x -> c[1] * sin(3x[1] + 2x[2]) + c[2] * x[1] * x[2]
        _pa_case(() -> avgₕ!(u, f), () -> _pa_copy(u))
    end))
    push!(P, ("broadcast 0-dim", true, 0, (n, p) -> begin
        W = _pa_space(n, p)
        u = Rₕ(W, _pa_g)
        v = similar(u)
        c = fill(1.5)
        _pa_case(() -> (v .= _pa_times0d.(u, c)), () -> _pa_copy(v))
    end))
    push!(P,
        ("matrix-free fused masked", false, 0,
            (n, p) -> begin
                op = matrix_free_operator(_pa_poisson(_pa_space(n, p)); dirichlet = :dir)
                x = randn(Xoshiro(10), size(op, 2))
                y0 = randn(Xoshiro(11), size(op, 1))
                y = similar(y0)
                _pa_case(() -> (copyto!(y, y0); mul!(y, op, x, 2.5, 0.7)), () -> copy(y))
            end))
    push!(P,
        ("matrix-free fused restricted",
            false,
            0,
            (n, p) -> begin
                W = _pa_space(n, p)
                op = matrix_free_operator(form(W, W, (u, v) -> innerₕ(u, v) +
                                                               inner₊(∇ₕ(u), ∇ₕ(v)) +
                                                               innerₕ(u, restrict_to(:dir, v))))
                x = randn(Xoshiro(12), size(op, 2))
                y = similar(x)
                _pa_case(() -> mul!(y, op, x), () -> copy(y))
            end))
    return P
end
# END _pa paths

# The block above as text, for the child process.
function _pa_paths_src()
    src = read(@__FILE__, String)
    start = last(findfirst("# BEGIN _pa paths\n", src)) + 1
    stop = first(findfirst("# END _pa paths", src)) - 1
    return src[start:stop]
end

# The two grids: the warm bytes of every path must be the same on both.
const _PA_SMALL = 17
const _PA_LARGE = 129

# Counts the Threads entry points each path reaches, in a child process, because counting
# means overriding the only two in `src` for the whole process: `_static_or_serial`, behind
# every `Threads.@threads :static`, and `_mf_run_bands!(::CpuThreaded, …)`, the only
# `Threads.@spawn`. Later testsets need `CpuThreaded` to really thread, so the override must
# not live in this process. Every call of a path counts, warm-up included, on both grids. The
# child also counts two `CpuThreaded` calls, the positive control that the counters see a
# Threads path. Returns the hits keyed by path name, the controls under "control <name>".
function _pa_threads_hits()
    code = """
    using Bramble, Polyester
    const HITS = Ref(0)
    @eval Bramble @noinline function _mf_run_bands!(::CpuThreaded, s, a, plan, nbands::Int)
        Main.HITS[] += 1
        for b in 1:nbands
            _mf_band_task!(s.y, s, a, plan, nbands, nbands, b)
        end
        return nothing
    end
    @eval Bramble @inline function _static_or_serial(threaded!::F, serial!::G,
            args::Vararg{Any, N}) where {F, G, N}
        Main.HITS[] += 1
        return serial!(args...)
    end
    $(_pa_paths_src())
    hits(case) = (HITS[] = 0; case.call(); case.call(); case.call(); HITS[])
    for (name, _, _, setup) in _pa_paths()
        if name in ("difference D₋ₓ!", "matrix-free fused")
            println("HITS\\tcontrol ", name, "\\t", hits(setup($(_PA_LARGE), CpuThreaded())))
        end
        h = hits(setup($(_PA_SMALL), CpuPolyester())) + hits(setup($(_PA_LARGE), CpuPolyester()))
        println("HITS\\t", name, "\\t", h)
    end
    """
    project = something(Base.active_project())
    cmd = `$(Base.julia_cmd()) --project=$project --startup-file=no --threads=$(Threads.nthreads()) -e $code`
    out = Dict{String, Int}()
    for line in eachline(cmd)
        startswith(line, "HITS\t") || continue
        _, name, h = split(line, '\t')
        out[name] = parse(Int, h)
    end
    return out
end

# Polyester boxes its argument tuple on every `@batch` call (`ManualMemory.Reference`); a
# `Base.RefValue` carrying the arguments is the other box Bramble passes across `@batch`, at
# most one per launch.
# Profile is loaded by package id because it reaches the test environment through
# SnoopCompile, not as a direct dependency.
const _PA_PROFILE = Base.require(Base.PkgId(
    Base.UUID("9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"), "Profile"))
const _PA_REFERENCE = Base.loaded_modules[Base.PkgId(
    Base.UUID("d125e4d3-2237-4719-b19c-fa641b8a4667"), "ManualMemory")].Reference
_pa_isbox(T) = T <: _PA_REFERENCE || T <: Base.RefValue

# Function barriers (bramble-verification §1): warmed twice, then measured.
@noinline function _pa_bytes(call::F) where {F}
    call()
    call()
    return @allocated call()
end

# Every allocation of one warm call, at sample_rate = 1: the largest argument box, the type
# and size of everything that is not one, and the number of `Base.RefValue`s.
function _pa_allocations(call::F) where {F}
    call()
    call()
    Allocs = _PA_PROFILE.Allocs
    Allocs.clear()
    Allocs.start(; sample_rate = 1)
    try
        call()
    finally
        Allocs.stop()
    end
    r = Allocs.fetch().allocs
    box = maximum((x.size for x in r if _pa_isbox(x.type)); init = 0)
    other = Any[(x.type, x.size) for x in r if !_pa_isbox(x.type)]
    # Each `@batch` launch makes exactly one `Reference`; a `RefValue` counts as a box only
    # alongside one, so stray `Ref`s are not mistaken for boxes.
    nref = count(x -> x.type <: _PA_REFERENCE, r)
    nval = count(x -> x.type <: Base.RefValue, r)
    nval <= nref || push!(other, (Base.RefValue, nval - nref))
    return box, other, nval
end

_pa_close(b, s) = isapprox(b, s; rtol = 1e-12, atol = 1e-12 * max(1.0, maximum(abs, s)))

@testset "Polyester extension (CpuPolyester)" begin
    # Needs this extension to be loaded (S7.1).
    @testset "CpuPolyester backend and grid space" begin
        be = backend(policy = CpuPolyester())
        @test execution_policy(be) === CpuPolyester()

        Ω = domain(box((0.0, 0.0), (1.0, 1.0)))
        Ωb = mesh(Ω, (8, 7), true; backend = be)
        # `space_weights` fills through the same policy-dispatched sweep as everything
        # else (S7.1's own finding), so this is the first CpuPolyester call any program makes
        # -- it raises naming Polyester without this extension loaded.
        Wb = gridspace(Ωb)
        @test ndofs(Wb) == ndofs(gridspace(mesh(Ω, (8, 7), true)))
    end

    # The `@noinline` CpuPolyester arm takes its kernel as `f::F ... where {F}`: an untyped
    # `f` only passed through is compiled on `::Function`, and `_late` then calls the hook
    # dynamically (gpena/Bramble.jl#460). The `Fix1` kernel `__innerplus_weights!` builds,
    # on non-uniform factors, gives the CpuSerial result and specialises the arm on `Fix1`.
    @testset "Fix1 kernel, CpuPolyester arm (#460)" begin
        for dims in ((37,), (23, 29), (7, 9, 11))
            diags = map(n -> sort!(rand(Xoshiro(n), n)), dims)
            vp, vs = zeros(dims), zeros(dims)
            Bramble._sweep_for!(CpuPolyester(), vp, CartesianIndices(vp),
                Base.Fix1(Bramble.__prod, diags))
            Bramble._sweep_for!(CpuSerial(), vs, CartesianIndices(vs),
                Base.Fix1(Bramble.__prod, diags))
            @test vp == vs
            @test vs[end] == prod(last, diags)
        end
        loc = typeof(Bramble.locality(Array{Float64, 2}))
        arm = which(Bramble._sweep_for!,
            (loc, CpuPolyester, Matrix{Float64}, CartesianIndices{2}, Function))
        kslot(mi) = Base.unwrap_unionall(mi.specTypes).parameters[end]
        mis = collect(Base.specializations(arm))
        @test !isempty(mis)
        @test any(mi -> kslot(mi) <: Base.Fix1, mis)
    end

    # Rₕ!/avgₕ! agree with Parallel() and Serial() in 1D, 2D and 3D.
    @testset "Rₕ!/avgₕ!: agree with Parallel()" begin
        for (D, n) in ((1, 21), (2, 11), (3, 6))
            p = _poisson_pair(Val(D), n)
            src = _sine_source(Val(D))

            up, ub, us = Rₕ(p.Wp, src), Rₕ(p.Wb, src), Rₕ(p.Ws, src)
            @test isapprox(parent(up), parent(ub); atol = 1.0e-12)
            @test isapprox(parent(us), parent(ub); atol = 1.0e-12)

            avgₕ!(up, src)
            avgₕ!(ub, src)
            avgₕ!(us, src)
            @test isapprox(parent(up), parent(ub); atol = 1.0e-12)
            @test isapprox(parent(us), parent(ub); atol = 1.0e-12)
        end
    end

    # Bilinear assemble, assemble! and assemble_parallel! agree with Parallel() in 1D, 2D and 3D.
    @testset "bilinear assembly agrees" begin
        for (D, n) in ((1, 21), (2, 9), (3, 5))
            p = _poisson_pair(Val(D), n)

            Ap = assemble(p.ap)
            Ab = assemble(p.ab)
            @test isapprox(Matrix(Ap), Matrix(Ab); atol = 1.0e-12)

            # `assemble!` into a matrix pre-filled with garbage: if the zeroing
            # `_assemble_bilinear!` does before dispatching to the sweep (`_zero_stored!(A)`)
            # were ever skipped for `CpuPolyester`, this would silently add the garbage into the
            # real entries instead of replacing them -- exactly the trap a
            # naive skip would fall into.
            Ab2 = allocate_system_matrix(p.ab)
            _fillnz!(Ab2, 999.0)
            assemble!(Ab2, p.ab)
            @test isapprox(Matrix(Ap), Matrix(Ab2); atol = 1.0e-12)

            Ab3 = allocate_system_matrix(p.ab)
            _fillnz!(Ab3, -777.0)
            assemble_parallel!(Ab3, p.ab)
            @test isapprox(Matrix(Ap), Matrix(Ab3); atol = 1.0e-12)

            Apd, Fp = assemble(p.ap, p.lp; dirichlet = ZERO_BC, symmetrize = true)

            # `assemble(a::BilinearForm, l::LinearForm; ...)` calls `assemble(l; ...)` for
            # the vector half, and `LinearForm`'s own `assemble`/`assemble!` refuse any
            # `CpuPolyester` policy unconditionally, Polyester loaded or not
            # (`src/assembly/linear.jl`'s own fail-fast, S7.1's design -- see the "integrator
            # item" testset below, and this subplan's final report). `BilinearForm`'s
            # `assemble`/`assemble!` carry no such guard, so the matrix half reaches this
            # extension's hooks the direct way; the vector half is built the way
            # `assemble_parallel!(b, ::LinearForm)` documents itself as reaching those same
            # hooks regardless of policy, then the same dirichlet/symmetrize steps
            # `assemble(a, l; ...)` itself performs are applied by hand.
            Abd = assemble(p.ab; dirichlet = ZERO_BC)
            Fb = similar(parent(element(p.Wb)))
            assemble_parallel!(Fb, p.lb)
            dirichlet_labels, dirichlet_conditions = _normalize_dirichlet(ZERO_BC)
            apply_dirichlet_conditions!(Fb, p.lb, dirichlet_conditions, dirichlet_labels, nothing)
            symmetrize!(Abd, Fb, test_space(p.ab), dirichlet_labels...; components = nothing)

            @test isapprox(Matrix(Apd), Matrix(Abd); atol = 1.0e-12)
            @test isapprox(Fp, Fb; atol = 1.0e-12)
            @test isapprox(Apd \ Fp, Abd \ Fb; atol = 1.0e-10)
        end
    end

    # Linear assemble_parallel! agrees with Parallel(); assemble and assemble! cover the integrator item.
    @testset "linear assembly agrees" begin
        # `assemble_parallel!(b, ::LinearForm)` forces `_assemble_linear_parallel_core!`
        # regardless of the space's own backend policy (`src/assembly/linear.jl`'s own
        # documented contract), and that core computes its *effective* policy the same way
        # the bilinear sweep does, so `CpuPolyester` reaches this extension's
        # `_batch_linear_colour_sweep!`/`_batch_linear_band_sweep!` exactly as `Parallel()`
        # reaches `Threads.@threads`.
        for (D, n) in ((1, 21), (2, 9), (3, 5))
            p = _poisson_pair(Val(D), n)

            bp = parent(element(p.Wp))
            bb = parent(element(p.Wb))
            assemble_parallel!(bp, p.lp)
            fill!(bb, 555.0)   # the same zeroing check as the matrix case above
            assemble_parallel!(bb, p.lb)
            @test isapprox(bp, bb; atol = 1.0e-12)
        end

        # `assemble`/`assemble!` on a `LinearForm` used to throw unconditionally under
        # `CpuPolyester`: `_assemble_linear!` fast-failed on the policy before it could reach
        # `_assemble_linear_parallel_core!`, so it fired even with Polyester loaded and every
        # hook implemented, and a linear form could never be assembled under this policy at
        # all. The integrator removed that branch on 2026-09-19; the
        # `else` branch dispatches on the effective policy and reaches this extension's
        # hooks, and without Polyester the hook itself still raises, naming the package, one
        # frame deeper. So these now assert agreement rather than the old failure.
        p2 = _poisson_pair(Val(2), 9)
        @test assemble(p2.lb) ≈ assemble(p2.lp)
        bb = similar(parent(element(p2.Wb)))
        bp = similar(parent(element(p2.Wp)))
        assemble!(bb, p2.lb)
        assemble!(bp, p2.lp)
        @test bb ≈ bp
    end

    # Includes the masked and multi-marker _dot paths.
    @testset "innerₕ/inner₊ₓ agree with Parallel()" begin
        for (D, n) in ((1, 21), (2, 11), (3, 6))
            p = _poisson_pair(Val(D), n)
            src = _sine_source(Val(D))
            up, ub = Rₕ(p.Wp, src), Rₕ(p.Wb, src)
            vp, vb = Rₕ(p.Wp, x -> 1.0), Rₕ(p.Wb, x -> 1.0)

            @test isapprox(innerₕ(up, vp), innerₕ(ub, vb); atol = 1.0e-12)
            @test isapprox(inner₊ₓ(up, vp), inner₊ₓ(ub, vb); atol = 1.0e-12)
        end

        # A single named region: `_dot_masked(u, v, w, mask::BitVector)`.
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωd = domain(S, :bottom => :bottom, :left => :left)
        Ωp = mesh(Ωd, (12, 12), (true, true); backend = backend(policy = Parallel()))
        Ωb = mesh(Ωd, (12, 12), (true, true); backend = backend(policy = CpuPolyester()))
        Wp, Wb = gridspace(Ωp), gridspace(Ωb)
        up, ub = Rₕ(Wp, x -> x[1] + x[2]), Rₕ(Wb, x -> x[1] + x[2])
        vp, vb = Rₕ(Wp, x -> 1.0), Rₕ(Wb, x -> 1.0)

        @test isapprox(
            innerₕ(up, vp; markers = (:bottom,)), innerₕ(ub, vb; markers = (:bottom,));
            atol = 1.0e-12
        )
        # Two markers: `_dot_masked(u, v, w, mask::MarkedIndicesUnion)`.
        @test isapprox(
            innerₕ(up, vp; markers = (:bottom, :left)),
            innerₕ(ub, vb; markers = (:bottom, :left));
            atol = 1.0e-12
        )
    end

    @testset "two-field form agrees" begin
        n1, n2 = 9, 7
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωd = domain(I2, :dir => boundary_symbols(I2))
        Ωp = mesh(Ωd, (n1, n2), (true, true); backend = backend(policy = Parallel()))
        Ωb = mesh(Ωd, (n1, n2), (true, true); backend = backend(policy = CpuPolyester()))
        Vp, Vb = gridspace(Ωp, Val(2)), gridspace(Ωb, Val(2))

        g = (V) -> form(
            V, V,
            (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2)) +
                      innerₕ(u(1), v(2))
        )

        Ap, Ab = assemble(g(Vp)), assemble(g(Vb))
        @test isapprox(Matrix(Ap), Matrix(Ab); atol = 1.0e-12)

        Apd, Abd = assemble(g(Vp); dirichlet = (:dir,)), assemble(g(Vb); dirichlet = (:dir,))
        @test isapprox(Matrix(Apd), Matrix(Abd); atol = 1.0e-12)
    end

    # An `Int32` target through the searching `@batch` sweep stores the same bits as an `Int`
    # one (gpena/Bramble.jl#469): each task searches `_ScatterCSC` around the `Int32` arrays,
    # whose positions must be `Int` to reach `nzval[pos]` rather than a missing method.
    @testset "Int32 target: searching sweep (#469)" begin
        Random.seed!(469)
        Ω = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 11), (false, false);
            backend = backend(policy = CpuPolyester())
        )
        scalar(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
        composite(u, v) = innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) +
                          innerₕ(D₋ₓ(u(1)), v(2))
        for (W, f) in ((gridspace(Ω), scalar), (gridspace(Ω, Val(2)), composite))
            a = form(W, W, f)
            A64 = allocate_system_matrix(a)
            @assert A64 isa SparseMatrixCSC{Float64, Int}
            A32 = SparseMatrixCSC{Float64, Int32}(A64)
            for A in (A64, A32)
                _fillnz!(A, 0.0)
                Bramble._assemble_bilinear_parallel_core!(A, a.trial_space, a.test_space, a.ast)
            end
            @test getcolptr(A32) == getcolptr(A64) && rowvals(A32) == rowvals(A64)
            @test isequal(nonzeros(A32), nonzeros(A64))
            @test !iszero(nonzeros(A64))
        end
    end

    # Per path, a warm CpuPolyester call allocates the same bytes on a small and a large
    # non-uniform grid (a multigrid cycle per coarsening level, since it runs every level's
    # loops), nothing at all (no `@batch` argument box either, except on the paths in
    # `_PA_BOX_CEILINGS`, each within its bound), reaches no Threads entry point, and gives
    # the CpuSerial result: bitwise for the operators asserted bitwise elsewhere in this
    # file, to rounding for reductions, assembly and solvers. Silent on a single thread,
    # where `@batch` runs serially.
    if Threads.nthreads() >= 2
        @testset "allocation under CpuPolyester" begin
            hits = _pa_threads_hits()
            @test get(hits, "control difference D₋ₓ!", 0) > 0
            @test get(hits, "control matrix-free fused", 0) > 0
            @testset "$name" for (name, bitwise, nrefs, setup) in _pa_paths()
                small = setup(_PA_SMALL, CpuPolyester())
                large = setup(_PA_LARGE, CpuPolyester())
                bs, bl = _pa_bytes(small.call), _pa_bytes(large.call)
                if small.levels > 0
                    @test bs % (small.levels - 1) == 0 && bl % (large.levels - 1) == 0
                    bs, bl = bs ÷ (small.levels - 1), bl ÷ (large.levels - 1)
                end
                @test bs == bl
                for case in (small, large)
                    box, other, nval = _pa_allocations(case.call)
                    @test box <= get(_PA_BOX_CEILINGS, name, 0)
                    @test isempty(other)
                    @test nval == nrefs
                end
                @test get(hits, name, -1) == 0
                for (n, case) in ((_PA_SMALL, small), (_PA_LARGE, large))
                    ref = setup(n, CpuSerial())
                    ref.call()
                    case.call()
                    s, b = ref.result(), case.result()
                    @test bitwise ? b == s : _pa_close(b, s)
                end
            end
        end
    end

    @testset "Determinism across repeated runs" begin
        p = _poisson_pair(Val(2), 15)
        A1 = assemble(p.ab)
        A2 = assemble(p.ab)
        A3 = assemble(p.ab)
        @test Matrix(A1) == Matrix(A2) == Matrix(A3)

        u, v = Rₕ(p.Wb, x -> sin(x[1])), Rₕ(p.Wb, x -> cos(x[2]))
        d1, d2, d3 = innerₕ(u, v), innerₕ(u, v), innerₕ(u, v)
        @test d1 == d2 == d3
    end

    # A warmed `CpuPolyester` refill replays the form's recorded `nzval` positions instead of
    # searching: `_threaded_replay_policy(::CpuPolyester)` and
    # `_batch_bilinear_band_replay!`/`_batch_bilinear_colour_replay!` above. Checked the same
    # way `test/form/threaded_replay.jl` checks `CpuThreaded` -- agreement against a serial
    # `assemble` of the same non-uniform mesh, never against another threaded fill -- since
    # this extension's own `CpuPolyester` vs `Parallel()` testsets above never re-fill an
    # already-assembled matrix and so would not tell a replay from a re-search.
    # A warmed CpuPolyester refill replays the recording.
    @testset "warmed refill replays (#338)" begin
        _replay_domains = (
            domain(interval(0.0, 1.0)),
            domain(interval(0.0, 1.0) × interval(0.0, 2.0)),
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0))
        )
        _replay_mesh(D, n, policy; seed) = begin
            Random.seed!(seed)
            mesh(
                _replay_domains[D], ntuple(_ -> n, D), ntuple(_ -> false, D);
                backend = backend(policy = policy)
            )
        end
        _scalar(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(u), v)
        _pair(u, v) = innerₕ(D₋ₓ(u), v) + innerₕ(u, D₋ₓ(v)) + innerₕ(u, v)
        _composite(u, v) = innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) +
                           innerₕ(D₋ₓ(u(1)), v(2))
        sizes = (41, 13, 7)

        @testset "$(D)D, $(nm)" for D in 1:3,
            (nm, f, comps) in (
                ("scalar", _scalar, 1), ("pair", _pair, 1), ("composite", _composite, 2)
            )

            n = sizes[D]
            Ωs = _replay_mesh(D, n, Serial(); seed = 338)
            Ωb = _replay_mesh(D, n, CpuPolyester(); seed = 338)
            @test points(Ωs) == points(Ωb)
            space(Ω) = comps == 1 ? gridspace(Ω) : gridspace(Ω, Val(comps))
            as = form(space(Ωs), space(Ωs), f)
            ab = form(space(Ωb), space(Ωb), f)
            R = assemble(as)

            B = assemble(ab)
            @test getcolptr(B) == getcolptr(R) && rowvals(B) == rowvals(R)
            @test isapprox(B, R; rtol = 1e-12)

            # A warmed refill (the recording already exists): replays, not re-searches.
            _fillnz!(B, NaN)
            assemble!(B, ab)
            @test getcolptr(B) == getcolptr(R) && rowvals(B) == rowvals(R)
            @test isapprox(B, R; rtol = 1e-12)
        end

        # Warmed refill allocation is independent of grid size.
        @testset "refill allocation: size-free" begin
            sizes2 = (200, 800)
            bytes = map(sizes2) do n
                Ω = _replay_mesh(1, n, CpuPolyester(); seed = 338)
                a = form(gridspace(Ω), gridspace(Ω), _scalar)
                A = assemble(a)
                alloc_test(assemble!, A, a)
            end
            @test bytes[1] == bytes[2]
        end
    end
end

# Under `CpuPolyester` every CPU stencil engine (the one-sided and centered difference
# engines and both average engines) runs banded along the grid's last axis, one band per
# `@batch` task (mirroring `test/space/threaded_stencils.jl`'s own
# `CpuThreaded` check, `Bramble._batch_difference_engine!`/`_batch_average_engine!`/
# `_batch_centered_average_engine!` in `ext/BramblePolyesterExt.jl`). Every point is still
# computed by the very loop body the serial engine runs, so the answer must equal `Serial()`
# exactly, not merely to a tolerance -- the meshes are non-uniform for the same reason: on a
# uniform mesh a band that picked up the wrong spacing index would still give the right
# number. The fixtures are test/TestUtils.jl's, shared with the `Parallel()` run of
# `test/space/threaded_stencils.jl`.

@testset "Stencil engines equal to Serial, $(D)D" for D in 1:3
    sizes = D == 1 ? ((1,), (2,), (3,), (5,), (1001,)) :
            D == 2 ? ((9, 1), (9, 2), (3, 3), (11, 7), (40, 37)) :
            ((5, 4, 1), (5, 4, 2), (4, 3, 5), (9, 8, 13))
    # Banded axes shorter than the thread count, down to a single point, leave some bands
    # empty; the operator must not notice.
    foreach(n -> _check_stencils(n, CpuPolyester(); full = true), sizes)
end

# Warmed in-place allocation of the stencil engines is independent of grid size.
@testset "stencil engines: allocation size-free" begin
    function _poly_min_bytes(n)
        _, Ωb = _stencil_mesh_pair((n, n), CpuPolyester())
        ub = Rₕ(gridspace(Ωb), _STENCIL_F[2])
        w = similar(ub)
        return map((:D₋, :D₊, :Dc, :D̃, :D̽, :M, :M₊, :Mc)) do fam
            minimum(alloc_test(_stencil_op!(fam, 2), w, ub) for _ in 1:5)
        end
    end
    @test _poly_min_bytes(16) == _poly_min_bytes(160)
end

# --- Divergence, curl and strain-average engines under CpuPolyester --------- #
#
# `_run_bands!`'s `CpuPolyester` arm (`_batch_run_bands!`, this extension) is what the
# accumulating engines behind `divₕ!`/`curlₕ!`/`εₕ!` (operators/vector_calculus.jl)
# reach; before S7.5 they had no `CpuPolyester` hook and ran serially regardless of the
# policy, so an equality check against `Serial()` alone would pass either way -- serial and
# `@batch` give the same numbers. The load-bearing assertion is the thread count, checked
# with the storage spy (`_spy`, test/TestUtils.jl) that
# `test/space/threaded_vector_calculus.jl` uses for `CpuThreaded`.
const _V356_GRADIENTS = (:∇̃ₕ!, :∇cₕ!, :∇̽ₕ!)
const _V356_DIVERGENCES = (:divₕ!, :div₊ₕ!, :divcₕ!, :diṽₕ!, :div̽ₕ!)
const _V356_CURLS = (:curlₕ!, :curl₊ₕ!, :curlcₕ!, :curl̃ₕ!, :curl̽ₕ!)
const _V356_STRAINS = (:εₕ!, :ε₊ₕ!, :εcₕ!, :ε̽ₕ!)
_v356_op(name) = getproperty(Bramble, name)

if Threads.nthreads() >= 2
    # Divergence, curl and strain-average engines run on several threads and equal Serial.
    @testset "div/curl/strain threaded, $(D)D" for D in 2:3
        n = D == 2 ? (64, 64) : (12, 12, 12)
        Ωs, Ωb = _stencil_mesh_pair(n, CpuPolyester())
        Ws, Wb = gridspace(Ωs), gridspace(Ωb)
        us, ub = Rₕ(Ws, _STENCIL_F[D]), Rₕ(Wb, _STENCIL_F[D])
        fs = ntuple(d -> (x -> _STENCIL_G[D](x) + d * sum(x)), D)
        tups_s = ntuple(d -> Rₕ(Ws, fs[d]), D)
        tups_b = ntuple(d -> Rₕ(Wb, fs[d]), D)
        spies = map(_spy, tups_b)

        for name in _V356_GRADIENTS
            dest_s, dest_b = ntuple(_ -> similar(us), D), ntuple(_ -> similar(ub), D)
            _v356_op(name)(dest_s, us)
            _reset_seen!()
            _v356_op(name)(dest_b, _spy(ub))
            @test _threads_seen() >= 2
            @test all(parent(a) == parent(b) for (a, b) in zip(dest_s, dest_b))
        end

        for name in _V356_DIVERGENCES
            vs, vb = similar(us), similar(ub)
            _v356_op(name)(vs, tups_s)
            _reset_seen!()
            _v356_op(name)(vb, spies)
            @test _threads_seen() >= 2
            @test parent(vs) == parent(vb)
        end

        for name in _V356_CURLS
            dest_s = D == 2 ? similar(us) : ntuple(_ -> similar(us), 3)
            dest_b = D == 2 ? similar(ub) : ntuple(_ -> similar(ub), 3)
            _v356_op(name)(dest_s, tups_s)
            _reset_seen!()
            _v356_op(name)(dest_b, spies)
            @test _threads_seen() >= 2
            ds = dest_s isa Tuple ? dest_s : (dest_s,)
            db = dest_b isa Tuple ? dest_b : (dest_b,)
            @test all(parent(a) == parent(b) for (a, b) in zip(ds, db))
        end

        for name in _V356_STRAINS
            dest_s = ntuple(_ -> ntuple(_ -> similar(us), D), D)
            dest_b = ntuple(_ -> ntuple(_ -> similar(ub), D), D)
            _v356_op(name)(dest_s, tups_s)
            _reset_seen!()
            _v356_op(name)(dest_b, spies)
            @test _threads_seen() >= 2
            @test all(parent(dest_s[i][j]) == parent(dest_b[i][j]) for i in 1:D for j in 1:D)
        end
    end
end

# --- Broadcast under CpuPolyester ------------------------------------------ #
#
# `_broadcast_copyto!`'s `CpuPolyester` arm (`_polyester_broadcast!`/`_batch_broadcast!`,
# ext/BramblePolyesterExt.jl) runs `dest .= expr` in the same bands `_threaded_broadcast!`
# runs under `CpuThreaded`, one per `Polyester.@batch` task instead of one per
# `Threads.@threads` thread -- mirroring test/space/threaded_broadcast.jl's own `CpuThreaded`
# check. Every point runs the very loop body the serial broadcast runs, so the answer must
# equal `Serial()` exactly, not merely to a tolerance; the meshes are non-uniform for the
# same reason those are. The fixtures are test/TestUtils.jl's, shared with that file.

# Broadcast under CpuPolyester equals Serial.
@testset "broadcast equals Serial, n=$n" for n in _BROADCAST_SIZES
    _check_broadcast_equal(n, CpuPolyester())
end

# A 0-dimensional array leaf stays inside its `Extruded` when `_batch_broadcast!` hands the
# tree to `@batch` (`_bc_host_raw`, src/space/vectorelement.jl): bare, it threw on the first
# call in a session, since `StrideArraysCore` cannot make a `PtrArray` of it. The function
# is fresh to this testset, so the `CpuPolyester` call below is that broadcast's first.
@testset "broadcast, 0-dim leaf, equals Serial" begin
    times0d(a, c) = 2.0 * a * c + 1.0
    res = map((Serial(), CpuPolyester())) do policy
        Wₕ = _broadcast_space((9, 9), policy)
        uₕ = Rₕ(Wₕ, x -> sin(3sum(x)) + prod(x))
        v = similar(uₕ)
        v .= times0d.(uₕ, fill(1.5))
        return copy(parent(v))
    end
    @test res[2] == res[1]
end

# Silent on a single thread: there is nothing to band across.
if Threads.nthreads() >= 2
    # Broadcast runs on several threads under CpuPolyester.
    @testset "broadcast is threaded, $(D)D" for D in 1:3
        n = D == 1 ? (200_000,) : D == 2 ? (400, 400) : (60, 60, 60)
        Wₕ = _broadcast_space(n, CpuPolyester())
        uₕ, wₕ = Rₕ(Wₕ, x -> sin(sum(x))), Rₕ(Wₕ, x -> prod(x))
        v = similar(uₕ)
        _reset_seen!()
        v .= 2.0 .* _spy(uₕ) .+ wₕ
        @test _threads_seen() >= 2
        @test parent(v) == 2.0 .* parent(uₕ) .+ parent(wₕ)
    end
end

# A matrix-free product under `CpuPolyester` sweeps the colour bands through the replay hooks
# (`_batch_bilinear_band_replay!`/`_batch_bilinear_colour_replay!`) so
# it must equal the serial product on the same non-uniform mesh on every repeat, and what it
# allocates is the `@batch` launch cost, whatever the grid size.
@testset "matrix-free mul! (#326)" begin
    _mf_space(D, n, policy) = begin
        Random.seed!(326)
        doms = (
            domain(interval(0.0, 1.0)),
            domain(interval(0.0, 1.0) × interval(0.0, 2.0)),
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0))
        )
        gridspace(mesh(doms[D], ntuple(_ -> n, D), ntuple(_ -> false, D); backend = backend(policy = policy)))
    end
    _mf_close(a, b) = isapprox(a, b; rtol = 1e-12, atol = 1e-12 * max(1.0, maximum(abs, b)))
    _diff(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
    _pair(u, v) = innerₕ(D₋ₓ(u), v) + 2.0 * innerₕ(u, D₋ₓ(v))
    _composite(u, v) = innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) + innerₕ(D₋ₓ(u(1)), v(2))
    sizes = (41, 13, 7)
    @testset "$(D)D, $(nm)" for D in 1:3,
        (nm, f, comps, dl) in (
            ("diffusion", _diff, 1, :boundary), ("pair", _pair, 1, nothing),
            ("composite", _composite, 2, :boundary)
        )

        space(W) = comps == 1 ? W : W × W
        Ws, Wb = space(_mf_space(D, sizes[D], Serial())), space(_mf_space(D, sizes[D], CpuPolyester()))
        kw = dl === nothing ? (;) : (; dirichlet = dl)
        ops = matrix_free_operator(form(Ws, Ws, f); kw...)
        opb = matrix_free_operator(form(Wb, Wb, f); kw...)
        x = randn(size(ops, 2))
        ref = ops * x
        y = similar(ref)
        @test all(1:20) do _
            mul!(y, opb, x)
            return _mf_close(y, ref)
        end
        y0 = randn(size(ops, 1))
        y .= y0
        mul!(y, opb, x, 0.5, 2.0)
        @test _mf_close(y, 0.5 * ref + 2.0 * y0)
    end
    @testset "mul! allocation: size-free" begin
        bytes = map((200, 800)) do n
            W = _mf_space(1, n, CpuPolyester())
            op = matrix_free_operator(form(W, W, _diff); dirichlet = :boundary)
            x = randn(size(op, 2))
            alloc_test(mul!, similar(x), op, x)
        end
        @test bytes[1] == bytes[2]
    end
end

# Leaves of different sizes share no band cut, so the product sweeps each unit in its own
# colours through the CpuPolyester hooks: the tall leaf in bands, the short one (three slices,
# too few to band) point by point. It must equal the same form on serial leaves over the same
# non-uniform meshes.
@testset "mul!: per-unit sweep, unequal leaves" begin
    _leaf(n, policy, seed) = (Random.seed!(seed);
        gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), n, (false, false);
            backend = backend(policy = policy))))
    f(u, v) = inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) + innerₕ(u(2), v(2))
    Wt, Wsh = _leaf((9, 33), CpuPolyester(), 1), _leaf((13, 3), CpuPolyester(), 2)
    Vb = Wt × Wsh
    Vs = _leaf((9, 33), Serial(), 1) × _leaf((13, 3), Serial(), 2)
    op = matrix_free_operator(form(Vb, Vb, f); dirichlet = :boundary)
    @test execution_policy(Wt) isa CpuPolyester && execution_policy(Wsh) isa CpuPolyester
    @test op.plan === nothing
    A = assemble(form(Vs, Vs, f); dirichlet = :boundary)
    x = randn(size(A, 2))
    @test !iszero(A * x)
    @test isapprox(op * x, A * x; rtol = 1e-12, atol = 1e-12 * maximum(abs, A * x))
    @test isapprox(op * x, assemble(form(Vb, Vb, f); dirichlet = :boundary) * x; rtol = 1e-12)
end

# A region written as an id (`restrict_to(1, u)`) is refused by the walk's bind on every
# route. The fused bands bind on the host and let only the regions they bound skip the
# in-task bind, so a user's id must not read marker column 1 there either.
@testset "integer region throws on every policy" begin
    X = interval(0.0, 1.0) × interval(0.0, 1.0)
    W = gridspace(mesh(domain(X, :dir => boundary_symbols(X)), (13, 11), (true, true)))
    a = form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(restrict_to(1, u), v))
    x = ones(ndofs(W))
    for policy in (CpuSerial(), CpuThreaded(), CpuPolyester())
        op = matrix_free_operator(a; policy)
        @test (op.plan === nothing) == (policy isa CpuSerial)
        @test_throws MethodError op * x
    end
end

# Mixed leaf policies, CpuThreaded beside CpuPolyester (the Threaded + Serial case
# lives in test/form/threaded_replay.jl): whether a unit replays
# is decided from the leaf its sweep walks, so a composite's leaves, or a cross-mesh form's two
# meshes, may carry different policies. Each refill records, then replays, and agrees with the
# same form on all-serial leaves over the same non-uniform meshes.
@testset "mixed policies: Threaded + Polyester" begin
    _rmesh(n, policy, seed) = (Random.seed!(seed);
        mesh(domain(interval(0.0, 1.0)), n, false; backend = backend(policy = policy)))
    leaf(policy, seed) = gridspace(_rmesh(33, policy, seed))
    comp(p1, p2) = Bramble.CompositeGridSpace((leaf(p1, 1), leaf(p2, 2)))
    f(u, v) = innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(2)), D₋ₓ(v(2))) +
              innerₕ(D₋ₓ(u(1)), v(2)) + innerₕ(u(2), D₋ₓ(v(1))) +   # pair across leaves
              innerₕ(D₋ₓ(u(2)), v(2)) + innerₕ(u(2), D₋ₓ(v(2)))     # pair on leaf 2
    g(u, v) = innerₕ(Bramble.πₕ(u), v)
    Wu(p) = gridspace(_rmesh(17, p, 3))
    Wv(p) = gridspace(_rmesh(33, p, 4))
    P, Ps = Bramble.Parallel(), Serial()
    cases = (
        (form(comp(P, CpuPolyester()), comp(P, CpuPolyester()), f),
            assemble(form(comp(Ps, Ps), comp(Ps, Ps), f))),
        (form(Wu(P), Wv(CpuPolyester()), g), assemble(form(Wu(Ps), Wv(Ps), g)))
    )
    for (a, R) in cases, refill! in (assemble!, assemble_parallel!)

        A = copy(R)
        for _ in 1:2   # record, then replay
            _fillnz!(A, NaN)
            refill!(A, a)
            @test getcolptr(A) == getcolptr(R) && rowvals(A) == rowvals(R)
            @test isapprox(A, R; rtol = 1e-12)
        end
    end
end

# The shift engines under CpuPolyester: every point is computed by the
# same loop body under every policy, so the answers equal the Serial ones exactly. The
# meshes are non-uniform, as in test/space/shift.jl.
function _shift_mesh(D; policy = Serial())
    Random.seed!(352)
    n = D == 1 ? 7 : ntuple(i -> 4 + i, D)
    unif = D == 1 ? false : ntuple(_ -> false, D)
    return mesh(_unit_cube(Val(D)), n, unif; backend = backend(policy = policy))
end
_shift_f(x) = 1 + sum(abs2, x) + prod(x)
_shift_g(x) = sin(3 * x[1]) - x[end]

@testset "Shift engines under CpuPolyester" begin
    policy = CpuPolyester()
    for D in 1:3
        us = Rₕ(gridspace(_shift_mesh(D)), _shift_f)
        up = Rₕ(gridspace(_shift_mesh(D; policy)), _shift_f)
        @test execution_policy(mesh(space(up))) == policy
        @test parent(us) == parent(up)
        vs = Rₕ(gridspace(_shift_mesh(D), Val(2)), (_shift_f, _shift_g))
        vp = Rₕ(gridspace(_shift_mesh(D; policy), Val(2)), (_shift_f, _shift_g))
        for d in 1:D, (op, op!) in ((S₊ₕ[d], (S₊ₓ!, S₊ᵧ!, S₊₂!)[d]), (S₋ₕ[d], (S₋ₓ!, S₋ᵧ!, S₋₂!)[d]))

            wp = similar(up)
            parent(wp) .= NaN               # every point must be written
            op!(wp, up)
            @test parent(wp) == parent(op(us))
            @test parent(op(up)) == parent(op(us))
            @test parent(op(vp)) == parent(op(vs))
        end
    end
end

# The GMG transfers and cycles under CpuPolyester, against Serial on
# the same non-uniform meshes as test/solvers/multigrid.jl. The transfers write every point
# once, so they agree bitwise on every repeat; the cycles agree to rounding.
@testset "Multigrid under CpuPolyester" begin
    policy = CpuPolyester()
    Random.seed!(3291)
    for ((Ωs, L), (Ωt, _)) in zip(_mg_transfer_meshes(), _mg_transfer_meshes(backend(; policy)))
        @test execution_policy(Ωt) == policy
        @test points(Ωt) == points(Ωs)
        Hs, Ht = GeometricMeshHierarchy(Ωs, L), GeometricMeshHierarchy(Ωt, L)
        for l in 2:L
            xc, yf = randn(npoints(Hs[l - 1])), randn(npoints(Hs[l]))
            xf, yc = prolongate!(zeros(length(yf)), Hs, l, xc), coarsen!(zeros(length(xc)), Hs, l, yf)
            xt, yt = similar(xf), similar(yc)
            @test all(1:20) do _
                prolongate!(xt, Ht, l, xc)
                coarsen!(yt, Ht, l, yf)
                return xt == xf && yt == yc
            end
        end
    end
    for (D, n) in ((2, 33), (3, 9))
        Ωs = _mg_jitter_mesh(D, n)
        Ωt = _mg_jitter_mesh(D, n; bk = backend(; policy))
        @test points(Ωt) == points(Ωs)
        b = randn(npoints(Ωs))
        for cyc in (:V, :W, :FMG)
            Ps = gmg_preconditioner(_mg_spd, Ωs; cycle = cyc)
            Pt = gmg_preconditioner(_mg_spd, Ωt; cycle = cyc)
            @test all(op -> op.policy == policy, Pt.ops)
            ys = Ps \ b
            yt = similar(ys)
            @test all(1:20) do _
                ldiv!(yt, Pt, b)
                return isapprox(yt, ys; rtol = 1e-12, atol = 1e-14)
            end
        end
        @test isapprox(parent(gmg_solve(_mg_spd, Ωt, b)), parent(gmg_solve(_mg_spd, Ωs, b)); rtol = 1e-10)
    end
end

# Data `_batch_split` cannot take apart (`BigFloat` sources, coefficients and grid functions)
# makes every split hook capture its parts whole, as before gpena/Bramble.jl#437, instead of
# throwing. Each hook must have run on `BigFloat` data, and every result equal the serial one
# on the same jittered mesh. `_BfWrapped` is a user's array type whose field names `Vector`:
# it splits, but cannot be rebuilt around the `PtrArray` a task receives.
struct _BfWrapped{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(w::_BfWrapped) = size(w.data)
Base.getindex(w::_BfWrapped, i::Int) = w.data[i]

@testset "unsplittable data falls back" begin
    _bf_space(n, policy; seed = 4370) = (Random.seed!(seed);
        gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), n, (false, false);
            backend = backend(policy = policy))))
    _bf_close(a, b) = !iszero(b) && isapprox(a, b; rtol = eltype(a) == BigFloat ? 1e-60 : 1e-14)
    hooks = (:_batch_bilinear_band_replay!, :_batch_bilinear_colour_replay!,
        :_batch_mf_bands!, :_batch_linear_colour_sweep!, :_batch_linear_band_sweep!,
        :_batch_bilinear_colour_sweep!, :_batch_bilinear_band_sweep!)
    ext = Base.get_extension(Bramble, :BramblePolyesterExt)
    # How many of the extension's instances of hook `h` take `BigFloat` data.
    bigruns(h) = sum(methods(getfield(Bramble, h)); init = 0) do m
        m.module === ext || return 0
        return count(mi -> mi !== nothing && occursin("BigFloat", string(mi.specTypes)),
            Base.specializations(m))
    end
    before = map(bigruns, hooks)
    results = map((CpuSerial(), CpuPolyester())) do policy
        W = _bf_space((9, 11), policy)
        g = Rₕ(W, x -> big(x[1] + 1))
        c = _BfWrapped(collect(range(1.0, 2.0; length = ndofs(W))))
        bilinear = (form(W, W, (u, v) -> big"2.0" * inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)),
            form(W, W, (u, v) -> inner₊(g * ∇ₕ(u), ∇ₕ(v))),
            form(W, W, (u, v) -> inner₊(c * ∇ₕ(u), ∇ₕ(v))))
        r = Any[]
        for a in bilinear
            A = assemble(a)
            for _ in 1:2
                fill!(nonzeros(A), 0)
                assemble!(A, a)
            end
            push!(r, A)
        end
        op = matrix_free_operator(bilinear[1])
        push!(r, op * big.(range(-1.0, 2.0; length = size(op, 2))))
        V = _bf_space((9, 33), policy; seed = 1) × _bf_space((13, 3), policy; seed = 2)
        opv = matrix_free_operator(form(V, V,
            (u, v) -> big"2.0" * inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2))))
        push!(r, opv * big.(range(-1.0, 2.0; length = size(opv, 2))))
        f = Rₕ(W, x -> big(x[1]))
        for l in (form(W, v -> innerₕ(x -> big(x[1] + x[2]), v)), form(W, v -> innerₕ(f, v)),
            form(W, v -> big"2.0" * innerₕ(x -> x[1], v)))
            push!(r, assemble(l))
        end
        # The searching sweep, entered as a form with no replaying leaf enters it: point
        # colouring on 9×11, bands on 9×33.
        for Ws in (W, _bf_space((9, 33), policy; seed = 1))
            a = form(Ws, Ws, (u, v) -> big"2.0" * inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
            A = assemble(a)
            fill!(nonzeros(A), 0)
            Bramble._assemble_bilinear_parallel_core!(A, a.trial_space, a.test_space, a.ast)
            push!(r, A)
        end
        return r, (op.plan !== nothing, opv.plan === nothing)
    end
    (serial, _), (batched, (fused, per_unit)) = results
    @test fused && per_unit
    @test map(eltype, batched) == [BigFloat, BigFloat, Float64, fill(BigFloat, 7)...]
    @test all(map(_bf_close, batched, serial))
    @testset "$h ran on BigFloat data" for (h, n) in zip(hooks, before)
        @test bigruns(h) > n
    end

    W = _bf_space((9, 11), CpuPolyester())
    unit(l) = (l.test_space, Bramble._bind_walk(l.ast, l.test_space)...)
    @test Bramble._batch_splittable(typeof(unit(form(W, v -> innerₕ(x -> x[1], v)))))
    big_unit = unit(form(W, v -> big"2.0" * innerₕ(x -> x[1], v)))
    @test !Bramble._batch_splittable(typeof(big_unit))
    @test_throws ArgumentError Bramble._batch_split(big_unit)
    @test !Bramble._batch_splittable(typeof(unit(form(W, v -> innerₕ(Rₕ(W, x -> big(x[1])), v)))))
    @test !Bramble._batch_splittable(_BfWrapped{Float64})
    @test Bramble._batch_splittable(Tuple{Vector{Float64}, Base.RefValue{Float64}})

    # `Rₕ!` and `avgₕ!` evaluate the user's function per point inside the task, so a function
    # capturing what the split cannot take (a `Dict`, a `_BfWrapped`) crosses whole, and so
    # does a `BigFloat` element, on `_batch_for!`, its masked form and `_batch_scatter_for!`.
    d, c = Dict(:a => 2.0), _BfWrapped([0.5])
    f = x -> d[:a] * x[1] + c[1] * sin(x[2])
    fb = x -> big(x[1]) * x[2] + 1
    fs = x -> (d[:a] * x[1], c[1] * x[2])
    W = _bf_space((9, 11), CpuPolyester())
    k = Bramble._RₕKernel(f, mesh(W), Bramble.indices(mesh(W)))
    @test !Bramble._batch_splittable(typeof(k))
    @test !Bramble._batch_splittable(typeof((k, ones(BigFloat, 3))))
    hits(h, s) = sum(methods(getfield(Bramble, h)); init = 0) do m
        m.module === ext || return 0
        return count(mi -> mi !== nothing && occursin(s, string(mi.specTypes)),
            Base.specializations(m))
    end
    before = (hits(:_batch_for!, "BigFloat"), hits(:_batch_for!, "Dict"),
        hits(:_batch_scatter_for!, "Dict"))
    rk = map((CpuSerial(), CpuPolyester())) do policy
        Ws = _bf_space((9, 11), policy)
        Wd = gridspace(mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 2.0),
                :top => x -> x[2] ≈ 2.0), (9, 11), (false, false);
            backend = backend(policy = policy)))
        u, um, ua = element(Ws), element(Wd), element(Ws)
        ub = element(Ws, BigFloat)
        uv = element(Ws × Ws)
        Rₕ!(u, f)
        Rₕ!(um, f; markers = (:top,))
        avgₕ!(ua, f)
        Rₕ!(ub, fb)
        Rₕ!(uv, fs)
        return map(e -> copy(parent(e)), (u, um, ua, ub, uv(1), uv(2)))
    end
    @test rk[2] == rk[1]
    @test eltype(rk[2][4]) == BigFloat && count(!iszero, rk[2][2]) == 9
    after = (hits(:_batch_for!, "BigFloat"), hits(:_batch_for!, "Dict"),
        hits(:_batch_scatter_for!, "Dict"))
    @test all(after .> before)
end

# A user's function that throws per point (`sqrt` of a negative number past `x₁ = 0.7`) inside
# the `@batch` tasks of `Rₕ!`, its masked form, the composite scatter and `avgₕ!` reaches the
# caller as the `DomainError` a serial sweep raises, with the same message, and never crashes
# the process; the threads the failed sweeps reserved are released. Repeated, since the crash
# a task's exception caused before (gpena/Bramble.jl#437) was intermittent.
@testset "throwing user function rethrows" begin
    _tf_space(policy) = (Random.seed!(433);
        gridspace(mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 2.0), :right => x -> x[1] ≈ 1.0),
            (17, 9), (false, false); backend = backend(policy = policy))))
    f = x -> x[1] > 0.7 ? sqrt(0.7 - x[1]) : x[1] * x[2]
    # The last two cross whole: a closure over a `Vector` is never split.
    c = [1.0]
    calls = (u -> Rₕ!(u, f), u -> Rₕ!(u, f; markers = (:right,)), u -> avgₕ!(u, f),
        u -> Rₕ!(u, x -> (f(x), x[2])), u -> avgₕ!(u, x -> c[1] * f(x)),
        u -> Rₕ!(u, x -> (c[1] * f(x), x[2])))
    _tf_error(call, W) =
        try
            call(element(W))
            nothing
        catch e
            e
        end
    _tf_elem(k, W) = k in (4, 6) ? W × W : W
    for (k, call) in enumerate(calls)
        serial = _tf_error(call, _tf_elem(k, _tf_space(CpuSerial())))
        @test serial isa DomainError
        msg = sprint(showerror, serial)
        W = _tf_elem(k, _tf_space(CpuPolyester()))
        for _ in 1:8
            e = _tf_error(call, W)
            @test e isa DomainError && sprint(showerror, e) == msg
        end
    end
    W, Ws = _tf_space(CpuPolyester()), _tf_space(CpuSerial())
    g = x -> x[1] * x[2]
    @test parent(Rₕ!(element(W), g)) == parent(Rₕ!(element(Ws), g))
end

# A user's closure over a `Vector` reaches `Rₕ!`/`avgₕ!` (plain, masked and composite)
# whole under `CpuPolyester`, as under `CpuSerial`: a split would rebuild it around a
# `PtrArray`, so a method typed on `Vector{Float64}` would throw a `MethodError` and an
# `isa Vector` branch would take the other arm. An index collection other than a unit range
# (a `Vector{Int}`, a stepped range) takes the per-index sweep.
_uc_typed(c::Vector{Float64}, x) = c[1] * x[1] + c[2] * x[2]^2
@testset "user closures over arrays cross whole" begin
    _uc_space(policy) = (Random.seed!(436);
        gridspace(mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 2.0), :right => x -> x[1] ≈ 1.0),
            (17, 9), (false, false); backend = backend(policy = policy))))
    c = [0.3, 1.7]
    typed = x -> _uc_typed(c, x)
    branch = x -> c isa Vector ? c[1] * x[1] : -1.0
    res = map((CpuSerial(), CpuPolyester())) do policy
        W = _uc_space(policy)
        r = Vector{Float64}[]
        for f in (typed, branch)
            u, um, ua = element(W), element(W), element(W)
            uv = element(W × W)
            Rₕ!(u, f)
            Rₕ!(um, f; markers = (:right,))
            avgₕ!(ua, f)
            Rₕ!(uv, x -> (f(x), x[2]))
            append!(r, map(e -> copy(parent(e)), (u, um, ua, uv(1), uv(2))))
        end
        return r
    end
    @test res[2] == res[1]
    @test all(v -> all(>=(0), v), res[2][6:10]) && any(!iszero, res[2][6])

    n = 50
    for idxs in ([3, 1, 7, 50, 22], 2:3:n)
        vp, vs = zeros(n), zeros(n)
        Bramble._sweep_for!(CpuPolyester(), vp, idxs, i -> c[1] * i + 1)
        Bramble._sweep_for!(CpuSerial(), vs, idxs, i -> c[1] * i + 1)
        @test vp == vs && count(!iszero, vp) == length(idxs)
    end

    # Destinations and index ranges with offset axes take the per-index sweep: the slabs
    # count positions from 1, so they would write other cells of the parent.
    O = Base.IdentityUnitRange
    lin(i) = i isa CartesianIndex ? i[1] + 7i[2] : i
    for (mk, idxs) in ((() -> view(zeros(6, 6), O(2:4), O(3:5)), CartesianIndices((O(2:4), O(3:5)))),
        (() -> view(zeros(6), O(2:4)), O(2:4)))
        for f in (i -> c[1] * lin(i), i -> 1.0 + lin(i))
            vp, vs = mk(), mk()
            Bramble._sweep_for!(CpuPolyester(), vp, idxs, f)
            Bramble._sweep_for!(CpuSerial(), vs, idxs, f)
            @test parent(vp) == parent(vs) && count(!iszero, parent(vp)) == length(idxs)
        end
    end

    # Such a closure's sweep allocates nothing warm: its kernel reaches the tasks through a
    # typed slot, not Polyester's argument box (gpena/Bramble.jl#476). Silent on a single
    # thread, where `@batch` runs serially.
    # Fresh names: `u` and `W` are assigned inside the `map` above, so reusing them here
    # would box them and measure the box.
    if Threads.nthreads() >= 2
        wb = _uc_space(CpuPolyester())
        eb, evb = element(wb), element(wb × wb)
        @test _pa_bytes(() -> Rₕ!(eb, typed)) == 0
        @test _pa_bytes(() -> Rₕ!(eb, typed; markers = (:right,))) == 0
        @test _pa_bytes(() -> Rₕ!(evb, x -> (typed(x), x[2]))) == 0
        @test _pa_bytes(() -> avgₕ!(eb, typed)) == 0
        @test _pa_bytes(() -> avgₕ!(eb, typed; markers = (:right,))) == 0
        @test _pa_bytes(() -> avgₕ!(evb, x -> (typed(x), x[2]))) == 0
    end
end

# A kernel crossing `@batch` whole reaches the tasks through a typed slot
# (gpena/Bramble.jl#476). The slot table's limits fall back to crossing whole, never to a
# wrong result: every slot of the kernel's type taken, a slot retired by an interrupted
# call (never handed out again, still holding its kernel), and a full type table. The
# covering, slab, per-index and scatter sweeps each equal `CpuSerial`'s in every case.
_se_kernel(c) = i -> c[1] * i + 1
@testset "slot exhaustion falls back" begin
    PE = Base.get_extension(Bramble, :BramblePolyesterExt)
    n = 200
    function sweeps_agree(f)
        ok = true
        for idxs in (1:n, 3:(n - 2), 2:3:n)
            vp, vs = zeros(n), zeros(n)
            Bramble._sweep_for!(CpuPolyester(), vp, idxs, f)
            Bramble._sweep_for!(CpuSerial(), vs, idxs, f)
            ok &= vp == vs && count(!iszero, vp) == length(idxs)
            sp, ss = zeros(n), zeros(n)
            Bramble._sweep_scatter_for!(CpuPolyester(), (sp,), idxs, i -> (f(i),))
            Bramble._sweep_scatter_for!(CpuSerial(), (ss,), idxs, i -> (f(i),))
            ok &= sp == ss && count(!iszero, sp) == length(idxs)
        end
        return ok
    end
    f, held = _se_kernel([0.5]), _se_kernel([9.0])
    @test sweeps_agree(f)
    id = PE._slots_id(typeof(f))
    @test id > 0 && PE._slots_quiescent()

    # Every slot of the type taken: the sweeps cross whole and box (on two or more
    # threads, where `@batch` boxes at all), and allocate nothing again once freed.
    refs = [PE._claim(held) for _ in 1:64]
    @test all(r -> r.id == id, refs)
    @test allunique(r -> r.key, refs)
    @test PE._claim(f).id == 0
    @test sweeps_agree(_se_kernel([0.25]))
    vb = zeros(n)
    if Threads.nthreads() >= 2
        @test _pa_bytes(() -> Bramble._sweep_for!(CpuPolyester(), vb, 1:n, f)) > 0
    end
    foreach(PE._release, refs)
    @test PE._slots_quiescent()
    if Threads.nthreads() >= 2
        @test _pa_bytes(() -> Bramble._sweep_for!(CpuPolyester(), vb, 1:n, f)) == 0
    end

    # A retired slot keeps its kernel and is never claimed again.
    gone = PE._claim(held)
    PE._retire(gone)
    @test PE._slots_quiescent() && PE._take(gone) === held
    rest = [PE._claim(held) for _ in 1:63]
    @test all(r -> r.id == id, rest)
    @test gone.key ∉ [r.key for r in rest]
    @test PE._claim(f).id == 0
    @test sweeps_agree(_se_kernel([0.75]))
    foreach(PE._release, rest)
    @test PE._slots_quiescent() && PE._take(gone) === held && sweeps_agree(f)

    # A full type table: a new kernel type gets no slots and crosses whole.
    ids = @atomic PE._SLOT_TABLE.ids
    full = copy(ids)
    k = 0
    while length(full) < PE._SLOT_TYPES
        k += 1
        full[Val{(:filler, k)}] = id
    end
    fresh = let c = [1.5]
        i -> c[1] * i - 2
    end
    try
        @atomic PE._SLOT_TABLE.ids = full
        @test PE._slots_id(typeof(fresh)) == 0
        @test sweeps_agree(fresh)
    finally
        @atomic PE._SLOT_TABLE.ids = ids
    end
    @test PE._slot_types() == length(ids) && PE._slots_quiescent()
end

# A stencil entry missing from the matrix's pattern throws inside the searching sweep's
# `@batch` task. Before the split hooks caught it there, the throw on the host's chunk ended
# Polyester's `GC.@preserve` while the workers still read the matrix, and the process
# segfaulted in most runs. Point colouring on 9×11, bands on 9×33; the matrix stores the
# diagonal only. Repeated, since the crash was intermittent.
@testset "missing pattern entry rethrows" begin
    _mp_space(n, policy) = (Random.seed!(437);
        gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), n, (false, false);
            backend = backend(policy = policy))))
    _mp_form(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
    _mp_diag(W) = spdiagm(ones(ndofs(W)))
    _mp_error(f) =
        try
            f()
            nothing
        catch e
            e
        end
    for n in ((9, 11), (9, 33))
        Ws = _mp_space(n, CpuSerial())
        serial = _mp_error(() -> assemble!(_mp_diag(Ws), _mp_form(Ws)))
        @test serial isa ArgumentError
        msg = sprint(showerror, serial)
        for _ in 1:8
            W = _mp_space(n, CpuPolyester())
            a = _mp_form(W)
            core = _mp_error(() -> Bramble._assemble_bilinear_parallel_core!(
                _mp_diag(W), a.trial_space, a.test_space, a.ast))
            whole = _mp_error(() -> assemble!(_mp_diag(W), a))
            @test core isa ArgumentError && sprint(showerror, core) == msg
            @test whole isa ArgumentError && sprint(showerror, whole) == msg
        end
        # The threads the failed sweeps reserved were released: a sweep still assembles.
        W = _mp_space(n, CpuPolyester())
        @test assemble(_mp_form(W)) ≈ assemble(_mp_form(Ws))
    end
end

end # module
