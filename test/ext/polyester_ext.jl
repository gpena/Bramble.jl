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
using ..TestUtils: alloc_test

const ZERO_BC = :dir => (x -> 0.0)

_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))
_sine_source(::Val{1}) = x -> sin(π * x)
_sine_source(::Val{D}) where {D} = x -> prod(sin(π * xᵢ) for xᵢ in x)

_grid(::Val{1}, Ωd, n; backend) = mesh(Ωd, n, true; backend = backend)
function _grid(::Val{D}, Ωd, n; backend) where {D}
    mesh(
        Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> true, Val(D)); backend = backend
    )
end

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

# `assemble` and `allocate_system_matrix` infer a union that includes a dense `Matrix`, which
# has no `nonzeros`; the matrices filled here are always `SparseMatrixCSC`, so the assertion
# narrows the type for JET.
function _fillnz!(A, v)
    @assert A isa SparseMatrixCSC
    return fill!(nonzeros(A), v)
end

# The paths of "allocation under CpuPolyester" below. The child process counting Threads
# entry points reads this block from the file, between the two marker lines, so that it runs
# exactly the same calls. Each entry is a name, whether the file asserts bitwise equality
# with `CpuSerial` for that operator, how many `Base.RefValue`s a warm `CpuPolyester` call
# allocates (measured at -O1), and a setup taking (grid points per axis, policy) on a
# non-uniform 2D grid and returning the call to measure, a function reading its result, and
# the number of multigrid levels (0 elsewhere).
# BEGIN _pa paths
using Bramble: CpuPolyester, CpuSerial, CpuThreaded, change_points!, semidiscretize_rhs,
               allocate_system_matrix, D₋ₓ, inner₊ₓ, S₊ₓ!
using LinearAlgebra: mul!, ldiv!
using Random: Xoshiro, randn

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
            2,
            (n, p) -> begin
                W = _pa_space(n, p)
                u = (Rₕ(W, _pa_g), Rₕ(W, _pa_h))
                e = ntuple(_ -> ntuple(_ -> similar(u[1]), 2), 2)
                _pa_case(() -> Bramble.εₕ!(e, u), () -> _pa_flat(e))
            end))
    push!(P, ("broadcast", true, 1, (n, p) -> begin
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
    push!(P, ("Kronecker mul!", false, 1, (n, p) -> begin
        K = kronecker_operator(_pa_poisson(_pa_space(n, p)))
        x = randn(Xoshiro(4), size(K, 2))
        y = similar(x)
        _pa_case(() -> mul!(y, K, x), () -> copy(y))
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

    # Per path, a warm CpuPolyester call allocates the same bytes on a small and a large
    # non-uniform grid (a multigrid cycle per coarsening level, since it runs every level's
    # loops), nothing but `@batch` argument boxes of at most 512 B each, reaches no Threads
    # entry point, and gives the CpuSerial result: bitwise for the operators asserted bitwise
    # elsewhere in this file, to rounding for reductions, assembly and solvers. Silent on a
    # single thread, where `@batch` runs serially.
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
                    @test box <= 512
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
            _alloc(f::F, args...) where {F} = (f(args...); @allocated f(args...))
            sizes2 = (200, 800)
            bytes = map(sizes2) do n
                Ω = _replay_mesh(1, n, CpuPolyester(); seed = 338)
                a = form(gridspace(Ω), gridspace(Ω), _scalar)
                A = assemble(a)
                _alloc(assemble!, A, a)
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
# number.

# Every family reaching `_apply_stencil!` or `_apply_averaged!`, spelled from the operator's
# base name so no Unicode is retyped here -- the same set `threaded_stencils.jl` names.
const _POLY_FAMILIES = (:D₋, :D₊, :diff₋, :diff₊, :jump, :Dc, :D̃, :D̽, :M, :M₊, :Mc)
const _POLY_CENTERED = (:Dc, :D̽, :Mc)   # need three points along their direction
const _POLY_SUFFIXES = ("ₓ", "ᵧ", "₂")

_poly_op(fam, d) = getproperty(Bramble, Symbol(fam, _POLY_SUFFIXES[d]))
_poly_op!(fam, d) = getproperty(Bramble, Symbol(fam, _POLY_SUFFIXES[d], :!))

function _poly_domain(D)
    D == 1 ? domain(interval(0.0, 1.0)) :
    D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
end

# The same non-uniform mesh twice, once per policy: the seed fixes the random points so
# `CpuPolyester` and `Serial` share every grid point.
function _poly_mesh_pair(n::NTuple{D, Int}; seed = 356) where {D}
    dom = _poly_domain(D)
    npts = D == 1 ? n[1] : n
    unif = D == 1 ? false : ntuple(_ -> false, D)
    Random.seed!(seed)
    Ωs = mesh(dom, npts, unif; backend = backend(policy = Serial()))
    Random.seed!(seed)
    Ωb = mesh(dom, npts, unif; backend = backend(policy = CpuPolyester()))
    return Ωs, Ωb
end

const _POLY_F = (x -> sin(3x) + x^2, x -> sin(3x[1] + 2x[2]) + x[1] * x[2],
    x -> sin(3x[1] + 2x[2] - x[3]) + x[1] * x[3])
const _POLY_G = (x -> cos(2x), x -> exp(x[1]) * x[2], x -> x[1] + x[2]^2 * x[3])

# Every applicable family and direction, in place and allocating, scalar and composite,
# `CpuPolyester` against `Serial`, exact `==`.
function _poly_check_all(n::NTuple{D, Int}) where {D}
    Ωs, Ωb = _poly_mesh_pair(n)
    Ws, Wb = gridspace(Ωs), gridspace(Ωb)
    Vs, Vb = gridspace(Ωs, Val(2)), gridspace(Ωb, Val(2))
    us, ub = Rₕ(Ws, _POLY_F[D]), Rₕ(Wb, _POLY_F[D])
    vs, vb = Rₕ(Vs, (_POLY_F[D], _POLY_G[D])), Rₕ(Vb, (_POLY_F[D], _POLY_G[D]))
    @test parent(us) == parent(ub)
    for d in 1:D, fam in _POLY_FAMILIES

        fam in _POLY_CENTERED && n[d] < 3 && continue
        f, f! = _poly_op(fam, d), _poly_op!(fam, d)
        @testset "$(fam)$(_POLY_SUFFIXES[d]) n=$n" begin
            ws, wb = similar(us), similar(ub)
            parent(wb) .= NaN               # every point must be written
            f!(ws, us)
            @test f!(wb, ub) === wb
            @test parent(wb) == parent(ws)
            @test parent(f(ub)) == parent(f(us))

            ws2, wb2 = similar(vs), similar(vb)
            f!(ws2, vs)
            f!(wb2, vb)
            @test parent(wb2) == parent(ws2)
            @test parent(f(vb)) == parent(f(vs))
        end
    end
end

@testset "Stencil engines equal to Serial, $(D)D" for D in 1:3
    sizes = D == 1 ? ((1,), (2,), (3,), (5,), (1001,)) :
            D == 2 ? ((9, 1), (9, 2), (3, 3), (11, 7), (40, 37)) :
            ((5, 4, 1), (5, 4, 2), (4, 3, 5), (9, 8, 13))
    # Banded axes shorter than the thread count, down to a single point, leave some bands
    # empty; the operator must not notice.
    foreach(_poly_check_all, sizes)
end

# Warmed in-place allocation of the stencil engines is independent of grid size.
@testset "stencil engines: allocation size-free" begin
    function _poly_min_bytes(n)
        _, Ωb = _poly_mesh_pair((n, n))
        ub = Rₕ(gridspace(Ωb), _POLY_F[2])
        w = similar(ub)
        return map((:D₋, :D₊, :Dc, :D̃, :D̽, :M, :M₊, :Mc)) do fam
            minimum(alloc_test(_poly_op!(fam, 2), w, ub) for _ in 1:5)
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
# with the same storage-spy trick `test/space/threaded_vector_calculus.jl` uses for
# `CpuThreaded`.
const _V356_SEEN = Threads.Atomic{UInt64}(0)
struct _V356Spy{T} <: AbstractVector{T}
    x::Vector{T}
end
Base.size(s::_V356Spy) = size(s.x)
Base.IndexStyle(::Type{<:_V356Spy}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(s::_V356Spy, i::Int)
    Threads.atomic_or!(_V356_SEEN, UInt64(1) << ((Threads.threadid() - 1) % 64))
    return s.x[i]
end
_v356_spy(u) = Bramble.VectorElement(_V356Spy(copy(parent(u))), space(u))

const _V356_GRADIENTS = (:∇̃ₕ!, :∇cₕ!, :∇̽ₕ!)
const _V356_DIVERGENCES = (:divₕ!, :div₊ₕ!, :divcₕ!, :diṽₕ!, :div̽ₕ!)
const _V356_CURLS = (:curlₕ!, :curl₊ₕ!, :curlcₕ!, :curl̃ₕ!, :curl̽ₕ!)
const _V356_STRAINS = (:εₕ!, :ε₊ₕ!, :εcₕ!, :ε̽ₕ!)
_v356_op(name) = getproperty(Bramble, name)

if Threads.nthreads() >= 2
    # Divergence, curl and strain-average engines run on several threads and equal Serial.
    @testset "div/curl/strain threaded, $(D)D" for D in 2:3
        n = D == 2 ? (64, 64) : (12, 12, 12)
        Ωs, Ωb = _poly_mesh_pair(n)
        Ws, Wb = gridspace(Ωs), gridspace(Ωb)
        us, ub = Rₕ(Ws, _POLY_F[D]), Rₕ(Wb, _POLY_F[D])
        fs = ntuple(d -> (x -> _POLY_G[D](x) + d * sum(x)), D)
        tups_s = ntuple(d -> Rₕ(Ws, fs[d]), D)
        tups_b = ntuple(d -> Rₕ(Wb, fs[d]), D)
        spies = map(_v356_spy, tups_b)

        for name in _V356_GRADIENTS
            dest_s, dest_b = ntuple(_ -> similar(us), D), ntuple(_ -> similar(ub), D)
            _v356_op(name)(dest_s, us)
            _V356_SEEN[] = 0
            _v356_op(name)(dest_b, _v356_spy(ub))
            @test count_ones(_V356_SEEN[]) >= 2
            @test all(parent(a) == parent(b) for (a, b) in zip(dest_s, dest_b))
        end

        for name in _V356_DIVERGENCES
            vs, vb = similar(us), similar(ub)
            _v356_op(name)(vs, tups_s)
            _V356_SEEN[] = 0
            _v356_op(name)(vb, spies)
            @test count_ones(_V356_SEEN[]) >= 2
            @test parent(vs) == parent(vb)
        end

        for name in _V356_CURLS
            dest_s = D == 2 ? similar(us) : ntuple(_ -> similar(us), 3)
            dest_b = D == 2 ? similar(ub) : ntuple(_ -> similar(ub), 3)
            _v356_op(name)(dest_s, tups_s)
            _V356_SEEN[] = 0
            _v356_op(name)(dest_b, spies)
            @test count_ones(_V356_SEEN[]) >= 2
            ds = dest_s isa Tuple ? dest_s : (dest_s,)
            db = dest_b isa Tuple ? dest_b : (dest_b,)
            @test all(parent(a) == parent(b) for (a, b) in zip(ds, db))
        end

        for name in _V356_STRAINS
            dest_s = ntuple(_ -> ntuple(_ -> similar(us), D), D)
            dest_b = ntuple(_ -> ntuple(_ -> similar(ub), D), D)
            _v356_op(name)(dest_s, tups_s)
            _V356_SEEN[] = 0
            _v356_op(name)(dest_b, spies)
            @test count_ones(_V356_SEEN[]) >= 2
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
# same reason those are.

function _bc357_domain(D)
    D == 1 ? domain(interval(0.0, 1.0)) :
    D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 2.0)) :
    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
end

# The same non-uniform mesh under the given policy: the seed fixes the random points.
function _bc357_space(n::NTuple{D, Int}, policy; seed = 357) where {D}
    Random.seed!(seed)
    npts = D == 1 ? n[1] : n
    unif = D == 1 ? false : ntuple(_ -> false, D)
    return gridspace(mesh(_bc357_domain(D), npts, unif; backend = backend(policy = policy)))
end

const _BC357_SIZES = ((1,), (2,), (7,), (1001,), (5, 3), (40, 37), (4, 3, 5), (13, 11, 9))

# A handful of broadcast shapes, each writing into a fresh `NaN` destination (or updating a
# copy in place, aliasing `dest` on its own right-hand side).
function _bc357_results(n, policy)
    Wₕ = _bc357_space(n, policy)
    uₕ = Rₕ(Wₕ, x -> sin(3sum(x)) + prod(x))
    wₕ = Rₕ(Wₕ, x -> exp(first(x)) * last(x))
    plain = [cos(0.3i) for i in eachindex(parent(uₕ))]
    r = Ref(0.25)
    α = 1.5
    fresh() = (v = similar(uₕ); parent(v) .= NaN; v)
    out = Dict{String, Vector{Float64}}()

    v = fresh()
    v .= 2.0 .* uₕ .+ wₕ
    out["axpy"] = copy(parent(v))
    v = fresh()
    v .= uₕ .* plain .- r[] .* wₕ .+ 1
    out["mixed"] = copy(parent(v))
    v = fresh()
    v .= α .* sin.(uₕ) ./ (1 .+ wₕ .^ 2)
    out["nested"] = copy(parent(v))
    v = fresh()
    v .= r
    out["fill"] = copy(parent(v))
    v = fresh()
    v .= uₕ
    out["copy"] = copy(parent(v))
    a = copy(uₕ)
    a .= a .+ 0.5 .* wₕ
    out["self"] = copy(parent(a))
    a = copy(uₕ)
    a .= wₕ .- a .* a
    out["self twice"] = copy(parent(a))
    a = copy(uₕ)
    a .*= α
    out["scale"] = copy(parent(a))
    return out
end

function _bc357_check_equal(n)
    s, p = _bc357_results(n, Serial()), _bc357_results(n, CpuPolyester())
    for key in keys(s)
        @test p[key] == s[key]
    end
end

# Records which threads read it, to see the bands spread -- its own spy type rather than
# reusing `_V356Spy` above, since that one is scoped to the divergence/curl/strain testset.
const _BC357_SEEN = Threads.Atomic{UInt64}(0)
struct _BC357Spy{T} <: AbstractVector{T}
    x::Vector{T}
end
Base.size(s::_BC357Spy) = size(s.x)
Base.IndexStyle(::Type{<:_BC357Spy}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(s::_BC357Spy, i::Int)
    Threads.atomic_or!(_BC357_SEEN, UInt64(1) << ((Threads.threadid() - 1) % 64))
    return s.x[i]
end

# Broadcast under CpuPolyester equals Serial.
@testset "broadcast equals Serial, n=$n" for n in _BC357_SIZES
    _bc357_check_equal(n)
end

# Silent on a single thread: there is nothing to band across.
if Threads.nthreads() >= 2
    # Broadcast runs on several threads under CpuPolyester.
    @testset "broadcast is threaded, $(D)D" for D in 1:3
        n = D == 1 ? (200_000,) : D == 2 ? (400, 400) : (60, 60, 60)
        Wₕ = _bc357_space(n, CpuPolyester())
        uₕ, wₕ = Rₕ(Wₕ, x -> sin(sum(x))), Rₕ(Wₕ, x -> prod(x))
        spy = Bramble.VectorElement(_BC357Spy(copy(parent(uₕ))), Wₕ)
        v = similar(uₕ)
        _BC357_SEEN[] = 0
        v .= 2.0 .* spy .+ wₕ
        @test count_ones(_BC357_SEEN[]) >= 2
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
        _alloc(y, op, x) = (mul!(y, op, x); @allocated mul!(y, op, x))
        bytes = map((200, 800)) do n
            W = _mf_space(1, n, CpuPolyester())
            op = matrix_free_operator(form(W, W, _diff); dirichlet = :boundary)
            x = randn(size(op, 2))
            _alloc(similar(x), op, x)
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
function _mg_transfer_meshes(bk = backend())
    Random.seed!(3291)
    I(a = 0.0, b = 1.0) = interval(a, b)
    return (
        (mesh(domain(I()), 33, false; backend = bk), 4),
        (mesh(domain(I() × I(0.0, 2.0)), (17, 9), false; backend = bk), 3),
        (mesh(domain(I() × I(-1.0, 1.0) × I(0.0, 2.0)), (9, 5, 9), false; backend = bk), 3),
        (mesh(domain(I() × I(0.5, 0.5)), (17, 4), false; backend = bk), 3),
        (mesh(domain(I() × I(0.5, 0.5) × I()), (9, 4, 5), false; backend = bk), 2)
    )
end

function _mg_jitter_mesh(D, n; bk = backend())
    rng = Random.Xoshiro(3291)
    Ω = mesh(domain(_unit_cube(Val(D))), ntuple(_ -> n, D), ntuple(_ -> true, D); backend = bk)
    h = 1 / (n - 1)
    function pts()
        x = collect(range(0.0, 1.0; length = n)) .+ 0.3h .* (2 .* rand(rng, n) .- 1)
        x[1], x[end] = 0.0, 1.0
        return sort!(x)
    end
    change_points!(Ω, ntuple(_ -> pts(), D))
    return Ω
end

_mg_spd(W) = (κ = Rₕ(W, x -> 1 + sum(abs2, x)); form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))))

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

end # module
