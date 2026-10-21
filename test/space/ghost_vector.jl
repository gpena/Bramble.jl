module GhostVectorTests

# `GhostVector` and `update_ghosts!` (gpena/Bramble.jl#776): a `GhostVector` holds the
# caller's vector as its physical part and the ghosts in a facet buffer, and
# `update_ghosts!(v, Cₕ[, t])` writes the ghosts the relations eliminate.
#
# The reference is `lift(Cₕ, u[, t])`, which multiplies the whole closure `E u + d` on the
# padded grid; the physical and ghost values are read from it through `boundary_node` and
# `padded_index`, not through the update's stored rows. `CpuPolyester` is in
# test/ext/ghost_vector_polyester.jl, where Polyester is loaded.

using Test
using Bramble
import Bramble
using Bramble: CpuSerial, CpuThreaded, D₋ₓ!
using LinearAlgebra: norm
using Random: MersenneTwister
using SparseArrays: SparseMatrixCSC

using ..TestUtils: WITH_3D_TESTS, WITH_SLOW_TESTS

# Under `unit`, the serial update keeps its Neumann and time-dependent cases, a zero-byte pin
# per relation kind on a small mesh, and the threaded update keeps one static case, the bands
# and the bound. The other sweep cases run under `WITH_SLOW_TESTS`, which the nightly and
# `all` groups set.

const K = [2.0 1.0; 1.0 2.0]
const K3 = [2.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 2.0]

# The bytes of the third call of `f`, the first two compiling and warming it.
alloc(f::F) where {F} = (f(); f(); @allocated f())

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

# The physical values and the ghost values of the padded grid function `U` on the mesh `G`.
function split_padded(G, U::AbstractVector)
    pl = LinearIndices(npoints(G, Tuple))
    x = vec([U[pl[J]] for J in Bramble.physical_indices(G)])
    gh = map(1:Bramble.nghosts(G)) do g
        I, d, side = Bramble.boundary_node(G, g)
        step = CartesianIndex(ntuple(e -> e == d ? (side === :min ? -1 : 1) : 0, ndims(pl)))
        U[pl[Bramble.padded_index(G, I) + step]]
    end
    return x, gh
end

# Whether each ghost of `G` lies beyond one of the facets named in `facets`.
function on_facets(G, facets)
    names = ((:xmin, :xmax), (:ymin, :ymax), (:zmin, :zmax))
    return map(1:Bramble.nghosts(G)) do g
        _, d, side = Bramble.boundary_node(G, g)
        names[d][side === :min ? 1 : 2] in facets
    end
end

# A `GhostVector` holding the physical values of `lift(C, u[, t])`, its kept ghosts the
# lift's, and a sentinel in every other ghost, before and after `update_ghosts!`, with the
# lift's ghosts as the reference.
function updated(C, u, t...; unrelated = ())
    G = mesh(C)
    x, ref = split_padded(G, parent(Bramble.lift(C, u, t...)))
    keep = on_facets(G, unrelated)
    v = GhostVector(C, x)
    Bramble.ghost_values(v) .= ifelse.(keep, ref, 1e30)
    x0 = copy(x)
    update_ghosts!(v, C, t...)
    return (; v, x, x0, ref, keep)
end

relerr(a, b) = norm(a - b, Inf) / max(norm(b, Inf), 1.0)

# Every ghost equals the lift's, the kept ones (beyond the facets `unrelated`) bitwise, and
# the physical part is unchanged.
function check_update(C, t...; tol = 1e-12, unrelated = ())
    u = rand(MersenneTwister(1234), eltype(C), Bramble.ndofs(C))
    r = updated(C, u, t...; unrelated)
    gv = Bramble.ghost_values(r.v)
    @test Bramble.physical(r.v) === r.x
    @test r.x == r.x0
    @test relerr(gv, r.ref) <= tol
    @test gv[r.keep] == r.ref[r.keep]
    @test all(i -> r.v[length(r.x) + i] == gv[i], eachindex(gv))
    v, C_ = r.v, C
    @test (@inferred update_ghosts!(v, C_, t...)) === v
    @test alloc(() -> update_ghosts!(v, C_, t...)) == 0
    return r
end

# The zero-byte pin of `check_update` alone: a warm `update_ghosts!` of `C` at `t` allocates
# nothing. Each relation kind compiles its own update kernel, so each keeps this pin in unit.
function pin_update(C, t...)
    v = GhostVector(C, rand(eltype(C), length(Bramble.physical_indices(mesh(C)))))
    @test alloc(() -> update_ghosts!(v, C, t...)) == 0
end

function conormal2(u)
    (:xmin => ∂ₙ(u; K) => 1.0, :xmax => u + ∂ₙ(u; K) => x -> x[2],
        :ymin => ∂ₙ(u; K) => 0.0, :ymax => 2.0 * u + ∂ₙ(u; K) => 1.0)
end
neumann2(u) = (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙ(u) => x -> x[2],
    :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => x -> x[1])
function robin2(u)
    (:xmin => u + ∂ₙ(u) => 1.0, :xmax => 3.0 * u + ∂ₙ(u) => x -> x[2],
        :ymin => 0.5 * u + ∂ₙ(u) => 0.0, :ymax => 2.0 * u + ∂ₙ(u) => 1.0)
end
function dirichlet2(u)
    (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙₙ(u) => x -> x[1],
        :ymin => u => x -> 1.0 + x[1], :ymax => ∂ₙ(u) => 1.0)
end
# `:ymin` has ghosts and no relation: its ghosts stay unknowns, and the conormal rows of
# `:xmin` and `:xmax` read them at the corners.
unrelated2(u) = (:xmin => ∂ₙ(u; K) => 1.0, :xmax => u + ∂ₙ(u; K) => x -> x[2],
    :ymax => ∂ₙ(u; K) => 1.0)
function time2(u)
    (:xmin => ∂ₙ(u; K) => (x, t) -> sin(t) + x[2] * cos(t),
        :xmax => u + ∂ₙ(u; K) => (x, t) -> t * x[2], :ymin => ∂ₙ(u; K) => (x, t) -> exp(-t) * x[1],
        :ymax => 2.0 * u + ∂ₙ(u; K) => 1.0)
end
# `:ymin` has no ghosts, its Dirichlet datum in time fixes its nodes.
function time_dirichlet2(u)
    (:xmin => ∂ₙ(u) => (x, t) -> sin(t) + x[2] * cos(t),
        :xmax => u + 0.5 * ∂ₙ(u) => (x, t) -> t * x[2], :ymin => u => (x, t) -> exp(-t) * x[1],
        :ymax => ∂ₙₙ(u) => 1.0)
end
const TIME_GHOSTS = (:xmin, :xmax, :ymax)
# Data in time, and `:xmax` ghosted with no relation: its ghosts stay unknowns.
function time_unrelated2(u)
    (:xmin => ∂ₙ(u; K) => (x, t) -> 1 + t * x[2], :ymin => ∂ₙ(u; K) => (x, t) -> t * x[1],
        :ymax => 2.0 * u + ∂ₙ(u; K) => 1.0)
end
function time_unrelated3(u)
    (:xmin => ∂ₙ(u; K = K3) => (x, t) -> 1 + t * x[3], :xmax => u + ∂ₙ(u; K = K3) => 0.0,
        :ymin => ∂ₙ(u; K = K3) => (x, t) -> t * x[1], :ymax => ∂ₙ(u; K = K3) => 0.0,
        :zmin => 2.0 * u + ∂ₙ(u; K = K3) => 1.0)
end
robin1(u) = (:xmin => u + ∂ₙ(u) => 2.0, :xmax => ∂ₙ(u) => x -> x[1])
function robin3(u)
    (:xmin => ∂ₙ(u; K = K3) => 1.0, :xmax => u + ∂ₙ(u; K = K3) => x -> x[3],
        :ymin => ∂ₙ(u; K = K3) => 0.0, :ymax => ∂ₙ(u; K = K3) => 0.0,
        :zmin => 2.0 * u + ∂ₙ(u; K = K3) => x -> x[1], :zmax => ∂ₙ(u; K = K3) => 1.0)
end

constrained_on(pol, ns, ghosts, rel; kwargs...) = constrained(
    gridspace(ghost_mesh(pol, ns; kwargs...); ghosts), rel)

@testset "GhostVector holds the caller's vector" begin
    Ω = ghost_mesh(CpuSerial(), (6, 7))
    W = gridspace(Ω; ghosts = :boundary)
    G = mesh(W)
    np, ng = length(Bramble.physical_indices(G)), Bramble.nghosts(G)
    x = rand(np)
    v = @inferred GhostVector(W, x)
    @test Bramble.physical(v) === x
    @test length(v) == np + ng == Bramble.ndofs(W)
    @test Bramble.ghost_values(v) == zeros(ng)
    x[3] = 7.0                                  # the caller's writes show through
    @test v[3] == 7.0
    v[np + 2] = 5.0
    @test Bramble.ghost_values(v)[2] == 5.0
    v[1] = -1.0
    @test x[1] == -1.0
    @test_throws BoundsError v[np + ng + 1]
    z = @inferred GhostVector(W)
    @test z == zeros(np + ng)
    @test eltype(Bramble.physical(z)) == Float64
    # the unconstrained space's degrees of freedom: `lift(W, w)` read back in `v`'s order
    w = rand(np + ng)
    xs, gs = split_padded(G, parent(Bramble.lift(W, w)))
    @test vcat(xs, gs) == w
    # no relations: nothing to write
    gv = copy(Bramble.ghost_values(v))
    @test update_ghosts!(v, W) === v
    @test Bramble.ghost_values(v) == gv
    @test alloc(() -> update_ghosts!(v, W)) == 0
    # Float32 space, Float32 parts
    W32 = gridspace(ghost_mesh(CpuSerial(), (6, 7); T = Float32); ghosts = :boundary)
    @test eltype(GhostVector(W32)) == Float32
    @test Bramble.ghost_values(GhostVector(W32, rand(Float32, np))) isa Vector{Float32}
end

@testset "update_ghosts! equals lift (#776)" begin
    pol = CpuSerial()
    @testset "2D Neumann" begin
        check_update(constrained_on(pol, (6, 7), :boundary, neumann2))
    end

    WITH_SLOW_TESTS && @testset "2D Robin" begin
        check_update(constrained_on(pol, (6, 7), :boundary, robin2))
    end

    WITH_SLOW_TESTS && @testset "2D conormal corner blocks" begin
        for ns in ((6, 7), (13, 9))
            check_update(constrained_on(pol, ns, :boundary, conormal2))
        end
    end

    WITH_SLOW_TESTS && @testset "2D named facets, ∂ₙₙ, Dirichlet u => g" begin
        C = constrained_on(pol, (6, 7), (:xmin, :xmax, :ymax), dirichlet2)
        r = check_update(C)
        @test any(!iszero, last(Bramble.closure(C)))     # an affine part reaches the ghosts
        # a point the Dirichlet relation fixes gives its datum, whatever `v` holds there
        G = mesh(C)
        fixed = vec([J[2] == 1 for J in CartesianIndices(npoints(Bramble.physical_mesh(G),
            Tuple))])
        gv = copy(Bramble.ghost_values(r.v))
        Bramble.physical(r.v)[fixed] .= 1e6
        update_ghosts!(r.v, C)
        @test Bramble.ghost_values(r.v) == gv
    end

    WITH_SLOW_TESTS && @testset "2D facet left without a relation" begin
        C = constrained_on(pol, (6, 7), :boundary, unrelated2)
        r = check_update(C; unrelated = (:ymin,))
        @test count(r.keep) == 6                     # the :ymin ghosts stay unknowns
        @test count(!, r.keep) == Bramble.nghosts(mesh(C)) - 6
        # the eliminated corner ghosts read the kept ones from the buffer
        before = copy(Bramble.ghost_values(r.v))
        Bramble.ghost_values(r.v)[r.keep] .+= 1.0
        update_ghosts!(r.v, C)
        changed = Bramble.ghost_values(r.v) .!= before
        @test any(changed .& .!r.keep)
    end

    @testset "2D time-dependent data, lift(Cₕ, u, t)" begin
        C = constrained_on(pol, (6, 7), :boundary, time2)
        for t in (0.0, 0.3, 1.7)
            check_update(C, t)
        end
        # a caller's buffer gives the same values, and touches no stored state
        u = rand(Bramble.ndofs(C))
        r = updated(C, u, 0.9)
        @test relerr(Bramble.ghost_values(r.v), r.ref) <= 1e-12
        v = r.v
        g = Bramble._affine_buffer(C)
        gv = copy(Bramble.ghost_values(v))
        update_ghosts!(v, C, 0.4)
        update_ghosts!(v, C, 0.9, g)
        @test Bramble.ghost_values(v) == gv
        @test alloc(() -> update_ghosts!(v, C, 0.9, g)) == 0
        @test_throws ArgumentError update_ghosts!(v, C)
        # a static space ignores the time
        S = constrained_on(pol, (6, 7), :boundary, conormal2)
        w = GhostVector(S, rand(42))
        update_ghosts!(w, S)
        gs = copy(Bramble.ghost_values(w))
        update_ghosts!(w, S, 2.5)
        @test Bramble.ghost_values(w) == gs
    end

    # The zero-byte pin of each moved relation kind, on the smallest mesh that takes it.
    @testset "0 B pin of each relation kind" begin
        pol = CpuSerial()
        pin_update(constrained_on(pol, (4, 5), :boundary, robin2))
        pin_update(constrained_on(pol, (4, 5), :boundary, conormal2))
        pin_update(constrained_on(pol, (4, 5), (:xmin, :xmax, :ymax), dirichlet2))
        pin_update(constrained_on(pol, (4, 5), :boundary, unrelated2))
        pin_update(constrained_on(pol, (4, 5), TIME_GHOSTS, time_dirichlet2), 0.3)
        pin_update(constrained_on(pol, (4, 5), :boundary, time_unrelated2), 0.3)
        pin_update(constrained_on(pol, (5,), :boundary, robin1))
        pin_update(constrained_on(pol, (5,), (:xmax,), u -> (:xmax => ∂ₙ(u) => 3.0,)))
        pin_update(constrained_on(pol, (4, 5), :boundary, conormal2; T = Float32))
    end

    WITH_SLOW_TESTS && @testset "2D Dirichlet in time, lift(Cₕ, u, t)" begin
        C = constrained_on(pol, (6, 7), TIME_GHOSTS, time_dirichlet2)
        for t in (0.0, 0.3, 1.7)
            check_update(C, t)
        end
    end

    WITH_SLOW_TESTS && @testset "2D data in time, a facet unrelated" begin
        C = constrained_on(pol, (6, 7), :boundary, time_unrelated2)
        for t in (0.0, 0.3, 1.7)
            r = check_update(C, t; unrelated = (:xmax,))
            @test count(r.keep) == 7                 # the :xmax ghosts stay unknowns
            @test length(C.gu.slot) == Bramble.nghosts(mesh(C)) - 7
        end
    end

    WITH_SLOW_TESTS && @testset "1D Robin" begin
        check_update(constrained_on(pol, (17,), :boundary, robin1))
        check_update(constrained_on(pol, (17,), (:xmax,), u -> (:xmax => ∂ₙ(u) => 3.0,)))
    end

    WITH_SLOW_TESTS && @testset "Float32" begin
        C = constrained_on(pol, (6, 7), :boundary, conormal2; T = Float32)
        r = check_update(C; tol = 1e-5)
        @test eltype(Bramble.ghost_values(r.v)) == Float32
    end

    # nightly only: 3D meshes are left out of the unit group (#650)
    WITH_3D_TESTS && @testset "3D conormal, ∂ₙₙ, Dirichlet" begin
        check_update(constrained_on(pol, (5, 6, 4), :boundary, robin3))
        check_update(constrained_on(pol, (5, 6, 4), (:xmin, :ymax, :zmax),
            u -> (:xmin => ∂ₙ(u) => 1.0, :ymax => u + ∂ₙ(u) => x -> x[3],
                :ymin => u => x -> x[1], :zmax => ∂ₙₙ(u) => 0.0)))
    end

    WITH_3D_TESTS && @testset "3D data in time, a facet unrelated" begin
        C = constrained_on(pol, (5, 6, 4), :boundary, time_unrelated3)
        for t in (0.0, 0.3, 1.7)
            r = check_update(C, t; unrelated = (:zmax,))
            @test count(r.keep) == 30                # the :zmax ghosts stay unknowns
        end
    end
end

@testset "GhostVector refusals (#776)" begin
    Ω = ghost_mesh(CpuSerial(), (6, 7))
    C = constrained(gridspace(Ω; ghosts = :boundary), conormal2)
    np = length(Bramble.physical_indices(mesh(C)))
    @test_throws ArgumentError GhostVector(C, rand(np + 1))
    @test_throws ArgumentError GhostVector(C, rand(Bramble.ndofs(gridspace(Ω; ghosts = :boundary))))
    @test_throws ArgumentError GhostVector(gridspace(Ω), rand(np))
    @test_throws ArgumentError GhostVector(gridspace(Ω))
    v = GhostVector(C, rand(np))
    # another mesh with the same sizes
    C2 = constrained(gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = :boundary),
        conormal2)
    @test_throws ArgumentError update_ghosts!(v, C2)
    # another mesh with other sizes
    C3 = constrained(gridspace(ghost_mesh(CpuSerial(), (7, 7)); ghosts = :boundary),
        conormal2)
    @test_throws ArgumentError update_ghosts!(v, C3)
    # a physical part resized after the vector was built
    w = GhostVector(C, rand(np))
    push!(Bramble.physical(w), 1.0)
    @test_throws ArgumentError update_ghosts!(w, C)
    # a Float32 vector cannot hold the Float64 ghosts of a Float64 space
    @test_throws ArgumentError update_ghosts!(GhostVector(C, rand(Float32, np)), C)
    @test_throws ArgumentError update_ghosts!(rand(3), C)
    @test_throws ArgumentError update_ghosts!(v, gridspace(Ω))
    # a mesh moved in place after the space was built
    Ωm = ghost_mesh(CpuSerial(), (6, 7))
    Cm = constrained(gridspace(Ωm; ghosts = :boundary), conormal2)
    vm = GhostVector(Cm, rand(np))
    x, y = collect(points(Ωm(1))), collect(points(Ωm(2)))
    Bramble.change_points!(Ωm, markers(domain(interval(0.0, 1.0) × interval(0.0, 2.0))),
        (x .^ 2 ./ x[end], y))
    @test_throws ArgumentError update_ghosts!(vm, Cm)
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

# The ghosts under `pol` of the physical values `x`, at `t` when given.
function ghosts_under(pol, ns, ghosts, rel, x, t...)
    C = constrained_on(pol, ns, ghosts, rel)
    v = GhostVector(C, copy(x))
    update_ghosts!(v, C, t...)
    return Bramble.ghost_values(v)
end

# The warm bytes of `update_ghosts!` and of the plain `D₋ₓ!` on a `CpuThreaded` mesh.
function threaded_figures(ns)
    pol = CpuThreaded()
    Ω = ghost_mesh(pol, ns)
    C = constrained(gridspace(Ω; ghosts = :boundary), conormal2)
    Ct = constrained(gridspace(Ω; ghosts = TIME_GHOSTS), time_dirichlet2)
    np = length(Bramble.physical_indices(mesh(C)))
    v, vt = GhostVector(C, rand(np)), GhostVector(Ct, rand(np))
    W = gridspace(Ω)
    uₕ, outₕ = element(W, rand(np)), element(W)
    return (update = alloc(() -> update_ghosts!(v, C)),
        update_t = alloc(() -> update_ghosts!(vt, Ct, 0.7)),
        plain = alloc(() -> D₋ₓ!(outₕ, uₕ)),
        groups = length(Ct.td.groups))
end

@testset "update_ghosts! under CpuThreaded (#776)" begin
    @test Threads.nthreads() >= 2
    x = rand(99)
    @test ghosts_under(CpuThreaded(), (9, 11), :boundary, conormal2, x) ==
          ghosts_under(CpuSerial(), (9, 11), :boundary, conormal2, x)

    WITH_SLOW_TESTS && @testset "the other relations, and data in time" begin
        cases = (((9, 11), (:xmin, :xmax, :ymax), dirichlet2),
            ((9, 11), :boundary, unrelated2), ((17,), :boundary, robin1))
        for (ns, ghosts, rel) in cases
            y = rand(prod(ns))
            @test ghosts_under(CpuThreaded(), ns, ghosts, rel, y) ==
                  ghosts_under(CpuSerial(), ns, ghosts, rel, y)
        end
        x = rand(99)
        for t in (0.0, 1.3)
            @test ghosts_under(CpuThreaded(), (9, 11), :boundary, time2, x, t) ==
                  ghosts_under(CpuSerial(), (9, 11), :boundary, time2, x, t)
            @test ghosts_under(CpuThreaded(), (9, 11), TIME_GHOSTS, time_dirichlet2, x, t) ==
                  ghosts_under(CpuSerial(), (9, 11), TIME_GHOSTS, time_dirichlet2, x, t)
        end
    end

    @testset "the update runs in bands" begin
        C = constrained_on(CpuThreaded(), (9, 11), :boundary, conormal2)
        xm = ThreadMarks(rand(99))
        update_ghosts!(GhostVector(C, xm), C)
        @test nthreads_seen(xm) >= 2
        Ct = constrained_on(CpuThreaded(), (9, 11), :boundary, time2)
        xt = ThreadMarks(rand(99))
        update_ghosts!(GhostVector(Ct, xt), Ct, 0.5)
        @test nthreads_seen(xt) >= 2
    end

    # `CpuThreaded` spawns its tasks on each band pass: the static update is one pass, the
    # update at `t` one per datum in time and one for the ghosts; nothing grows with the grid.
    @testset "CpuThreaded bound" begin
        f1 = threaded_figures((9, 11))
        f2 = threaded_figures((30, 33))
        @test f1 == f2
        # Under CPU contention a pass can settle, for the rest of the process, into
        # re-boxing its two band closures in every task: `nthreads × 2 × closure` bytes
        # more, the same at every size. The update's closure is 80 bytes; the update at
        # `t` has one of 64 per data group and a ghost pass of 128. At 4 threads, the
        # ubuntu nightly measured update 2224 against plain 1584 (4 × 2 × 80); the same
        # +640 hit plain `D₋ₓ!` locally under 8 spinners.
        slack = 2 * Threads.nthreads()
        @test f1.update <= f1.plain + slack * 80
        @test f1.update_t <= (f1.groups + 1) * f1.plain + slack * (f1.groups * 64 + 128)
    end
end

end
