# test/ext/ghost_polyester.jl: ghost forms assemble on a `CpuPolyester` mesh, to the serial
# matrix and load, and every warm in-place call allocates nothing (gpena/Bramble.jl#785).
#
# Gated like every other ext/*.jl file (test/runtests.jl only reaches this group under
# BRAMBLE_TEST_GROUP=ext or full). Standalone:
#
#   julia --project=test -t 2 -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/ghost_polyester.jl")'
#
# A `@batch` sweep passes the padded space as `_batch_split` gives it, and each task
# rebuilds a `GhostMeshState` over its meshes' walk states (src/utils/batch_split.jl), so
# the ghost methods (the corner rule, the physical facets) still apply there. The mixed and
# facet forms below change value where those methods are lost. At one thread `@batch` runs
# one chunk inline, so the split bites at 2 or more.
module GhostPolyesterTests

using Test
using Bramble
using Bramble: CpuPolyester, CpuSerial, Dcₓ, Dcᵧ, D̃ₓ, D₋ₓ, D̃ᵧ, D₋ᵧ
using LinearAlgebra: norm
using SparseArrays: SparseMatrixCSC, nonzeros
using Polyester

using ..TestUtils: WITH_3D_TESTS

const POLICIES = (CpuSerial(), CpuPolyester())
const K = [2.0 0.5; 0.5 1.0]

# A seeded non-uniform mesh on [0, 1] × [0, 2] × [0, 3] (its first `D` axes) under `pol`.
function ghost_mesh(pol, ns::NTuple{D, Int}; T = Float64, seed = 11) where {D}
    Bramble._seed_mesh1d_rng!(seed)
    be = backend(vector_type = Vector{T}, matrix_type = SparseMatrixCSC{T, Int},
        policy = pol)
    Ωd = domain(foldl(×, ntuple(d -> interval(T(0), T(d)), D)))
    Ω = mesh(Ωd, D == 1 ? only(ns) : ns, D == 1 ? false : ntuple(_ -> false, D);
        backend = be)
    Bramble._unseed_mesh1d_rng!()
    return Ω
end

# `assemble(a, l)` on the space `space(Ω)` under each policy, serial first.
function pair_by_policy(ns, space, f, g; kwargs...)
    return map(POLICIES) do pol
        C = space(ghost_mesh(pol, ns))
        assemble(form(C, C, f), form(C, g); kwargs...)
    end
end

relerr(X, Y) = norm(X - Y, Inf) / norm(Y, Inf)

laplacian(u, v) = innerₕ(-Δₕ(u), v)
complex_laplacian(u, v) = (1.0 + 0.5im) * innerₕ(-Δₕ(u), v)
source(v) = innerₕ(x -> 1.0 + x[1], v)
# Forms and loads whose element types widen each other on a Float32 mesh.
complex32_laplacian(u, v) = (1.0f0 + 0.5f0im) * innerₕ(-Δₕ(u), v)
wide_laplacian(u, v) = innerₕ(-2.0 * Δₕ(u), v)
complex32_source(v) = innerₕ(x -> ComplexF32(1 + x[1], 1), v)

# The bytes of the third call of `f`, the first two compiling and warming it.
alloc(f::F) where {F} = (f(); f(); @allocated f())

# The warm in-place calls on `constrained(gridspace(Ω; ghosts), rel)` under `pol`: the pair
# refill (its load reads `A_pad d ≠ 0`), `lift!`, the matrix refill, and the values they
# leave after three calls.
function warm_calls(pol, ns, ghosts, rel; f = laplacian, g = source, T = Float64)
    C = constrained(gridspace(ghost_mesh(pol, ns; T); ghosts), rel)
    a, l = form(C, C, f), form(C, g)
    A, F = assemble(a, l)
    u = A \ F
    U = Bramble.lift(C, zero(u))
    A2 = copy(A)
    bytes = (pair = alloc(() -> assemble!(A, F, a, l)),
        lift = alloc(() -> Bramble.lift!(U, C, u)),
        matrix = alloc(() -> assemble!(A2, a)))
    return bytes, (A, F, A2, parent(U)), any(!iszero, last(Bramble.closure(C)))
end

# The warm calls on a homogeneous Neumann space, whose rows are the physical nodes.
function warm_rows(pol, ns)
    rel = u -> map(F -> F => ∂ₙ(u) => 0.0, (:xmin, :xmax, :ymin, :ymax))
    C = constrained(gridspace(ghost_mesh(pol, ns); ghosts = :boundary), rel)
    a, l = form(C, C, laplacian), form(C, source)
    A, F = assemble(a, l)
    g = x -> x[2]
    A2, F2 = copy(A), copy(F)
    return (matrix = alloc(() -> assemble!(A, a; dirichlet = :xmin)),
        load = alloc(() -> assemble!(F, l)),
        load_dirichlet = alloc(() -> assemble!(F, l; dirichlet = :xmin => g)),
        pair_symmetrize = alloc(() -> assemble!(A2, F2, a, l; dirichlet = :xmin => g,
            symmetrize = true)),
        dirichlet_bc = alloc(() -> dirichlet_bc!(A, C, :xmin)),
        symmetrize = alloc(() -> symmetrize!(A, F, C, :xmin)))
end

# A vector that marks, per thread, every task that reads or writes it: a band function that
# runs on more than one thread marks more than one, a path left serial marks one.
struct ThreadMarks{T, V <: AbstractVector{T}} <: AbstractVector{T}
    data::V
    seen::Vector{Bool}
end
ThreadMarks(v::AbstractVector) = ThreadMarks(v, fill(false, Threads.maxthreadid()))
Base.size(r::ThreadMarks) = size(r.data)
Base.IndexStyle(::Type{<:ThreadMarks}) = IndexLinear()
@inline function Base.getindex(r::ThreadMarks, i::Int)
    @inbounds r.seen[Threads.threadid()] = true
    return r.data[i]
end
@inline function Base.setindex!(r::ThreadMarks, x, i::Int)
    @inbounds r.seen[Threads.threadid()] = true
    return setindex!(r.data, x, i)
end
nthreads_seen(r::ThreadMarks) = count(r.seen)

# The form `a` with its map's arrays marked, so that each path's band function reads one.
function marked_form(a)
    marks = (k, v) -> v isa AbstractVector && k !== :A_pad ? ThreadMarks(v) : v
    fields = map(k -> marks(k, getfield(a, k)), fieldnames(typeof(a)))
    return Bramble.GhostBilinearForm(fields...)
end

@testset "Ghost forms on CpuPolyester (#785)" begin
    @test Threads.nthreads() >= 2

    @testset "Rebuilt mesh keeps the ghost type" begin
        W = gridspace(ghost_mesh(CpuPolyester(), (6, 7)); ghosts = (:xmin, :ymax))
        P = Bramble.padded_space(W)
        G = mesh(P)
        @test Bramble._batch_splittable(typeof(P))
        skeleton, arrays = Bramble._batch_split(P)
        @test isbits(skeleton)
        R = mesh(Bramble._batch_rebuild(skeleton, arrays))
        @test R isa Bramble.GhostMeshState
        @test (R.n, R.lo, R.active, R.offset) == (G.n, G.lo, G.active, G.offset)
        @test Bramble.nghosts(R) == Bramble.nghosts(G)
        @test Bramble.physical_indices(R) == Bramble.physical_indices(G)
        @test all(d -> collect(points(R(d))) == collect(points(G(d))), 1:2)
    end

    @testset "2D Laplacian, Dirichlet relation" begin
        rel = u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[2],
            :ymin => u => x -> x[1], :ymax => ∂ₙ(u) => 1.0)
        space = Ω -> constrained(gridspace(Ω; ghosts = (:xmin, :xmax, :ymax)), rel)
        (A0, F0), (A1, F1) = pair_by_policy((9, 11), space, laplacian, source)
        @test any(!iszero, last(Bramble.closure(space(ghost_mesh(CpuSerial(), (9, 11))))))
        @test relerr(A1, A0) <= 1e-13
        @test relerr(F1, F0) <= 1e-13
    end

    @testset "2D conormal corners" begin
        rel = u -> map(F -> F => ∂ₙ(u; K) => 1.0, (:xmin, :xmax, :ymin, :ymax))
        space = Ω -> constrained(gridspace(Ω; ghosts = :boundary), rel)
        mixed = (u, v) -> innerₕ(K[1, 1] * D̃ₓ(D₋ₓ(u)) + 2 * K[1, 2] * Dcₓ(Dcᵧ(u)) +
                                 K[2, 2] * D̃ᵧ(D₋ᵧ(u)), v)
        facet = (u, v) -> innerₕ(-Δₕ(u), v) + inner_Γ(∂ₙ(u), v; markers = :boundary)
        for f in (mixed, facet)
            (A0, F0), (A1, F1) = pair_by_policy((9, 11), space, f, source)
            @test relerr(A1, A0) <= 1e-13
            @test relerr(F1, F0) <= 1e-13
        end
    end

    @testset "Dirichlet by row replacement" begin
        rel = u -> map(F -> F => ∂ₙ(u) => 0.5, (:xmin, :xmax, :ymin, :ymax))
        space = Ω -> constrained(gridspace(Ω; ghosts = :boundary), rel)
        for symmetrize in (false, true)
            (A0, F0), (A1, F1) = pair_by_policy((9, 11), space, laplacian, source;
                dirichlet = :xmin => (x -> x[2]), symmetrize)
            @test relerr(A1, A0) <= 1e-13
            @test relerr(F1, F0) <= 1e-13
        end
    end

    @testset "assemble! refills" begin
        # Homogeneous data, so `d = 0` and the load refills on its own (`assemble!(b, l)`).
        rel = u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => u + ∂ₙ(u) => 0.0,
            :ymin => ∂ₙ(u) => 0.0, :ymax => 2.0 * u + ∂ₙ(u) => 0.0)
        out = map(POLICIES) do pol
            C = constrained(gridspace(ghost_mesh(pol, (9, 11)); ghosts = :boundary), rel)
            a, l = form(C, C, laplacian), form(C, source)
            A, F = assemble(a), assemble(l)
            A0, F0 = copy(A), copy(F)
            fill!(nonzeros(A), 0.0)
            fill!(F, 0.0)
            assemble!(A, a)
            assemble!(F, l)
            return A, F, A0, F0
        end
        (A0, F0, Af0, Ff0), (A1, F1, Af1, Ff1) = out
        @test A1 == Af1
        @test F1 == Ff1
        @test relerr(A1, A0) <= 1e-13
        @test relerr(F1, F0) <= 1e-13
        @test norm(F0, Inf) > 0
    end

    @testset "1D Robin and Dirichlet relations" begin
        rel = u -> (:xmin => u + ∂ₙ(u) => 2.0, :xmax => u => 1.0)
        space = Ω -> constrained(gridspace(Ω; ghosts = :xmin), rel)
        (A0, F0), (A1, F1) = pair_by_policy((17,), space, laplacian, source)
        @test relerr(A1, A0) <= 1e-13
        @test relerr(F1, F0) <= 1e-13
    end

    # The reduction, the load and `lift` run in `@batch` bands that each own whole output
    # entries, so they give the serial values bit for bit, and a repeated run the same bits.
    @testset "Bands give the serial values bitwise" begin
        Kc = [2.0 1.0; 1.0 2.0]
        cases = (
            ((9, 11), :boundary,
                u -> (:xmin => ∂ₙ(u; K = Kc) => 1.0, :xmax => u + ∂ₙ(u; K = Kc) => x -> x[2],
                    :ymin => ∂ₙ(u; K = Kc) => 0.0, :ymax => 2.0 * u + ∂ₙ(u; K = Kc) => 1.0)),
            ((9, 11), (:xmin, :xmax, :ymax),
                u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙₙ(u) => x -> x[1],
                    :ymin => u => x -> x[1], :ymax => ∂ₙ(u) => 1.0)),
            ((17,), :xmin, u -> (:xmin => u + ∂ₙ(u) => 2.0, :xmax => u => 1.0)))
        case3 = ((5, 6, 4), (:xmin, :xmax, :ymin, :ymax, :zmax),
            u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[3],
                :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0, :zmin => u => x -> x[1],
                :zmax => ∂ₙ(u) => 1.0))
        for (ns, ghosts, rel) in (WITH_3D_TESTS ? (cases..., case3) : cases)
            r0, r1, r2 = map((CpuSerial(), CpuPolyester(), CpuPolyester())) do pol
                C = constrained(gridspace(ghost_mesh(pol, ns); ghosts), rel)
                a, l = form(C, C, laplacian), form(C, source)
                A, F = assemble(a, l)
                A2 = copy(A)
                fill!(nonzeros(A2), 0.0)
                assemble!(A2, a)
                u = pde_solve(A, F)
                return (A, F, A2, u, parent(Bramble.lift(C, u)), C)
            end
            @test any(!iszero, last(Bramble.closure(r0[6])))
            for k in 1:5
                @test r1[k] == r0[k]
                @test r2[k] == r1[k]
            end
        end
    end

    # A path that ignores the mesh's policy runs on the calling thread alone.
    @testset "Each path runs on several threads" begin
        rel = u -> map(F -> F => ∂ₙ(u; K) => 1.0, (:xmin, :xmax, :ymin, :ymax))
        C = constrained(gridspace(ghost_mesh(CpuPolyester(), (9, 11)); ghosts = :boundary),
            rel)
        a, l = form(C, C, laplacian), form(C, source)
        A, F = assemble(a, l)
        @test Bramble._ghost_policy(C) isa CpuPolyester
        am = marked_form(a)
        Am = copy(A)
        fill!(nonzeros(Am), 0.0)
        Bramble._assemble_ghost!(Am, am)
        Fm = ThreadMarks(zeros(length(F)))
        Bramble._assemble_ghost_load!(Fm, am, l)
        u = A \ F
        um = ThreadMarks(copy(u))
        U = parent(Bramble.lift(C, um))
        @test Am == A
        @test Fm.data == F
        @test U == parent(Bramble.lift(C, u))
        @test nthreads_seen(am.coef) >= 2         # the reduction
        @test nthreads_seen(am.liftcol) >= 2      # `F_pad − A_pad d`
        @test nthreads_seen(Fm) >= 2              # `E_vᵀ w`
        @test nthreads_seen(um) >= 2              # `lift`
    end

    # Every warm in-place call allocates nothing under `@batch`, its bands taking arrays
    # only, and leaves the serial values bit for bit.
    @testset "Warm calls allocate nothing" begin
        robin = u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[2],
            :ymin => ∂ₙ(u) => 0.0, :ymax => 2.0 * u + ∂ₙ(u) => 1.0)
        rel1 = u -> (:xmin => u + ∂ₙ(u) => 2.0, :xmax => ∂ₙ(u) => x -> x[1])
        # The last two on a Float32 mesh, form and load types widening each other.
        cases = (((9, 11), :boundary, robin, laplacian, source, Float64),
            ((30, 33), :boundary, robin, laplacian, source, Float64),
            ((9, 11), :boundary, robin, complex_laplacian, source, Float64),
            ((17,), :boundary, rel1, laplacian, source, Float64),
            ((9, 11), :boundary, robin, complex32_laplacian, source, Float32),
            ((9, 11), :boundary, robin, wide_laplacian, complex32_source, Float32))
        for (ns, ghosts, rel, f, g, T) in cases
            (b0, v0, lifted), (b1, v1, _) = map(POLICIES) do pol
                warm_calls(pol, ns, ghosts, rel; f, g, T)
            end
            @test lifted                        # the load reads `A_pad d`
            @test b1 == (pair = 0, lift = 0, matrix = 0)
            @test b0 == b1
            @test v1 == v0
        end
        for ns in ((9, 11), (30, 33))
            for (k, n) in pairs(warm_rows(CpuPolyester(), ns))
                @test n == 0
            end
        end
    end

    WITH_3D_TESTS && @testset "3D Laplacian, Dirichlet relation" begin
        rel = u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[3],
            :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0, :zmin => u => x -> x[1],
            :zmax => ∂ₙ(u) => 1.0)
        ghosts = (:xmin, :xmax, :ymin, :ymax, :zmax)
        space = Ω -> constrained(gridspace(Ω; ghosts), rel)
        (A0, F0), (A1, F1) = pair_by_policy((5, 6, 4), space, laplacian, source)
        @test relerr(A1, A0) <= 1e-13
        @test relerr(F1, F0) <= 1e-13
    end

    # Relation data in time (gpena/Bramble.jl#776): `d(t)` and `lift(C, u, t)` equal the
    # serial ones bitwise, and the warm in-place calls allocate nothing. The data include a
    # closure over a number, which rides into each task inside its band function.
    @testset "Relation data in time" begin
        wave(ω) = (x, s) -> cos(ω * s) + x[2]
        rel = u -> (:xmin => ∂ₙ(u; K) => (x, s) -> sin(s) * x[2],
            :xmax => u + ∂ₙ(u; K) => wave(2.0),
            :ymin => ∂ₙ(u; K) => 0.5,
            :ymax => u => (x, s) -> s * x[1],
            :ymax => ∂ₙ(u; K) => (x, s) -> s - x[1])
        C0, C1 = map(pol -> constrained(gridspace(ghost_mesh(pol, (9, 11));
                    ghosts = :boundary), rel), POLICIES)
        @test C1.td isa Bramble._TimeData
        u = collect(range(0.0, 1.0; length = Bramble.ndofs(C0)))
        d0, d1 = similar(C0.d), similar(C1.d)
        for t in (0.0, 0.3, 1.7)
            @test Bramble._lift_affine!(d1, C1, t) == Bramble._lift_affine!(d0, C0, t)
            @test parent(Bramble.lift(C1, u, t)) == parent(Bramble.lift(C0, u, t))
        end
        d17 = copy(Bramble._lift_affine!(d0, C0, 1.7))
        @test norm(d17 - Bramble._lift_affine!(d0, C0, 0.0)) > 0.1   # the data move `d`
        let d1 = d1, C1 = C1, u = u, U = Bramble.lift(C1, u, 0.0)
            @test alloc(() -> Bramble._lift_affine!(d1, C1, 0.6)) == 0
            @test alloc(() -> Bramble.lift!(U, C1, u, 0.6)) == 0
        end
    end
end

end
