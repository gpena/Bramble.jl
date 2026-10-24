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
using Random
using Bramble
using Bramble: CpuPolyester, CpuSerial, Dcₓ, Dcᵧ, D̃ₓ, D₋ₓ, D̃ᵧ, D₋ᵧ,
               matrix_free_operator, polygon, polygon_spaces
using LinearAlgebra: mul!, norm
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

# Polygon and level-set spaces (#878): `polygon_spaces` on a mesh built with a backend of
# policy `pol`, the way `ghost_mesh` builds its meshes. Builders copied from
# test/ghost/polygon_support.jl, polygon_oracle.jl and levelset_spaces.jl.
const FIG1 = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0),
    (4.0, 5.0), (2.0, 5.0), (0.0, 3.0)]
const VNOTCH = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 3.0), (1.0, 2.0),
    (0.0, 1.0)]
const OCTAGON = [(1.0, 0.0), (3.0, 0.0), (4.0, 1.0), (4.0, 3.0), (3.0, 4.0), (1.0, 4.0),
    (0.0, 3.0), (0.0, 1.0)]

disk(p) = hypot(p[1], p[2]) - 1

policy_backend(pol) = backend(vector_type = Vector{Float64},
    matrix_type = SparseMatrixCSC{Float64, Int}, policy = pol)

box(x0, x1, y0, y1) = interval(1.0 * x0, 1.0 * x1) × interval(1.0 * y0, 1.0 * y1)

# The jittered mesh of `box` with the curve `:P => shape` (a polygon or a level set).
function curve_mesh(pol, shape, box, n, seed)
    Random.seed!(seed)
    return mesh(domain(box, :P => shape), n, (false, false); backend = policy_backend(pol))
end

# (name, conflict nodes expected, matrix-free form, mesh builder of the policy): the
# jittered Figure 1, the V-notch, whose exterior node (0, 2) is read by two interior nodes
# (a conflict node, which a matrix-free mixed form refuses), the convex octagon and a
# level-set disk.
const CURVES = (
    (name = "Figure 1", conflict = false, f_free = :mixed,
        build = pol -> curve_mesh(pol, polygon(FIG1), box(0, 6, 0, 5), (10, 9), 2)),
    (name = "V-notch", conflict = true, f_free = :laplacian,
        build = pol -> curve_mesh(pol, polygon(VNOTCH), box(0, 4, 0, 4), (6, 7), 7)),
    (name = "octagon", conflict = false, f_free = :mixed,
        build = pol -> curve_mesh(pol, polygon(OCTAGON), box(-1, 5, -1, 5), (19, 19), 2)),
    (name = "level-set disk", conflict = false, f_free = :mixed,
        build = pol -> curve_mesh(pol, levelset(disk), box(-1.25, 1.25, -1.25, 1.25),
            (20, 20), 2)))

mixed(w) = Dcₓ(Dcᵧ(w)) + Dcᵧ(Dcₓ(w))
laplacian_mixed(u, v) = innerₕ(-1.0 * Δₕ(u) - 0.4 * mixed(u), v)
boundary_data(x) = 1 + x[1] - 2x[2] + x[1] * x[2]

# Matrix, load, refilled matrix, refilled pair, solution, its lift and the matrix-free
# product of the form with `f_free` on one mesh, under `pol`.
function curve_by(build, pol, f_free)
    Ω = build(pol)
    U, W = polygon_spaces(Ω, :P, boundary_data)
    a, l = form(U, W, laplacian_mixed), form(W, source)
    A, F = assemble(a, l)
    A2 = copy(A)
    fill!(nonzeros(A2), 0.0)
    assemble!(A2, a)
    A3, F3 = copy(A), copy(F)
    fill!(nonzeros(A3), 0.0)
    fill!(F3, 0.0)
    assemble!(A3, F3, a, l)
    u = pde_solve(A, F)
    af = form(U, W, f_free === :mixed ? laplacian_mixed : laplacian)
    op = matrix_free_operator(af)
    x = collect(range(-1.0, 2.0; length = size(op, 2)))
    z = zeros(size(op, 1))
    mul!(z, op, x)
    bytes = (matrix = alloc(() -> assemble!(A2, a)),
        pair = alloc(() -> assemble!(A3, F3, a, l)), mul = alloc(() -> mul!(z, op, x)))
    return (A = A, F = F, A2 = A2, A3 = A3, F3 = F3, u = u, U = Bramble.lift(U, u).data,
        z = z, zA = assemble(af) * x, conflicts = count(U.rows.conflict), bytes = bytes,
        policy = Bramble.execution_policy(Ω))
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
        # The matrix-free product (#877), with and without Dirichlet rows, on a ghost box
        # and a polygon space: its values are the assembled matrix's.
        rel = u -> map(F -> F => ∂ₙ(u) => 0.0, (:xmin, :xmax, :ymin, :ymax))
        C = constrained(gridspace(ghost_mesh(CpuPolyester(), (12, 11)); ghosts = :boundary),
            rel)
        be = backend(vector_type = Vector{Float64},
            matrix_type = SparseMatrixCSC{Float64, Int}, policy = CpuPolyester())
        octagon = [(1.0, 0.0), (3.0, 0.0), (4.0, 1.0), (4.0, 3.0), (3.0, 4.0), (1.0, 4.0),
            (0.0, 3.0), (0.0, 1.0)]
        Ωp = mesh(domain(interval(-1.0, 5.0) × interval(-1.0, 5.0), :P => polygon(octagon)),
            (19, 19), (false, false); backend = be)
        U, W = polygon_spaces(Ωp, :P, x -> 1 + x[1] - 2x[2])
        for (a, kw) in ((form(C, C, laplacian), (;)),
            (form(C, C, laplacian), (; dirichlet = :xmin)), (form(U, W, laplacian), (;)))
            op = matrix_free_operator(a; kw...)
            x = collect(range(-1.0, 2.0; length = size(op, 2)))
            z = zeros(size(op, 1))
            @test alloc(() -> mul!(z, op, x)) == 0
            @test z ≈ assemble(a; kw...) * x
        end
    end

    # Polygon and level-set spaces (#878): every result equals the serial one bit for bit
    # (the matrix-free product to 1e-13), and the warm `assemble!` calls and `mul!` leave
    # no bytes.
    @testset "$(c.name): polygon_spaces under CpuPolyester" for c in CURVES
        r0 = curve_by(c.build, CpuSerial(), c.f_free)
        r1 = curve_by(c.build, CpuPolyester(), c.f_free)
        @test r0.policy isa CpuSerial
        @test r1.policy isa CpuPolyester
        @test (r0.conflicts > 0) == c.conflict
        @test norm(r0.F, Inf) > 0 && norm(r0.z, Inf) > 0
        for k in (:A, :F, :A2, :A3, :F3, :u, :U)
            @test getfield(r1, k) == getfield(r0, k)
        end
        @test r0.A2 == r0.A
        @test r0.A3 == r0.A
        @test r0.F3 == r0.F
        @test relerr(r1.z, r0.z) <= 1e-13
        @test relerr(r1.z, r1.zA) <= 1e-13
        @test r1.bytes == (matrix = 0, pair = 0, mul = 0)
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
