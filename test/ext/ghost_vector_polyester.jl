# test/ext/ghost_vector_polyester.jl: `update_ghosts!` on a `CpuPolyester` mesh writes the
# serial ghosts, in bands, and allocates nothing warm, and so do the runtime operators on a
# `GhostVector` (gpena/Bramble.jl#776).
#
# Gated like every other ext/*.jl file (test/runtests.jl only reaches this group under
# BRAMBLE_TEST_GROUP=ext or full). Standalone:
#
#   julia --project=test -t 2 -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/ghost_vector_polyester.jl")'
#
# The update's band function takes arrays only, so a `@batch` task boxes nothing. At one
# thread `@batch` runs one chunk inline, so the bands are checked at 2 or more.
module GhostVectorPolyesterTests

using Test
using Bramble
using Bramble: CpuPolyester, CpuSerial, Δₕ!
using SparseArrays: SparseMatrixCSC
using Polyester

using ..TestUtils: WITH_3D_TESTS

const K = [2.0 1.0; 1.0 2.0]

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

function conormal2(u)
    (:xmin => ∂ₙ(u; K) => 1.0, :xmax => u + ∂ₙ(u; K) => x -> x[2],
        :ymin => ∂ₙ(u; K) => 0.0, :ymax => 2.0 * u + ∂ₙ(u; K) => 1.0)
end
function dirichlet2(u)
    (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙₙ(u) => x -> x[1],
        :ymin => u => x -> 1.0 + x[1], :ymax => ∂ₙ(u) => 1.0)
end
unrelated2(u) = (:xmin => ∂ₙ(u; K) => 1.0, :xmax => u + ∂ₙ(u; K) => x -> x[2],
    :ymax => ∂ₙ(u; K) => 1.0)
function time2(u)
    (:xmin => ∂ₙ(u; K) => (x, t) -> sin(t) + x[2] * cos(t),
        :xmax => u + ∂ₙ(u; K) => (x, t) -> t * x[2], :ymin => ∂ₙ(u; K) => (x, t) -> exp(-t) * x[1],
        :ymax => 2.0 * u + ∂ₙ(u; K) => 1.0)
end
function time_dirichlet2(u)
    (:xmin => ∂ₙ(u) => (x, t) -> sin(t) + x[2] * cos(t),
        :xmax => u + 0.5 * ∂ₙ(u) => (x, t) -> t * x[2], :ymin => u => (x, t) -> exp(-t) * x[1],
        :ymax => ∂ₙₙ(u) => 1.0)
end
robin1(u) = (:xmin => u + ∂ₙ(u) => 2.0, :xmax => ∂ₙ(u) => x -> x[1])
function robin3(u)
    (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[3],
        :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙₙ(u) => 0.0, :zmin => u => x -> x[1],
        :zmax => ∂ₙ(u) => 1.0)
end

# The ghosts of the physical values `x` under `pol` (the kept ghosts start at `1:ng`), and
# the warm bytes of the update.
function update_under(pol, ns, ghosts, rel, x, t...)
    C = constrained(gridspace(ghost_mesh(pol, ns); ghosts), rel)
    v = GhostVector(C, copy(x))
    Bramble.ghost_values(v) .= 1:length(Bramble.ghost_values(v))
    update_ghosts!(v, C, t...)
    return copy(Bramble.ghost_values(v)), alloc(() -> update_ghosts!(v, C, t...))
end

# Polyester's ghosts equal serial's bitwise (each ghost is summed by one band in a fixed
# order), and the warm update allocates nothing.
function same_as_serial(ns, ghosts, rel, t...)
    x = rand(prod(ns))
    g0, _ = update_under(CpuSerial(), ns, ghosts, rel, x, t...)
    g1, bytes = update_under(CpuPolyester(), ns, ghosts, rel, x, t...)
    @test g1 == g0
    @test any(!iszero, g0)
    @test bytes == 0
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

@testset "update_ghosts! under CpuPolyester (#776)" begin
    @test Threads.nthreads() >= 2

    @testset "2D conormal corner blocks" begin
        for ns in ((9, 11), (30, 33))
            same_as_serial(ns, :boundary, conormal2)
        end
    end

    @testset "2D named facets, ∂ₙₙ, Dirichlet u => g" begin
        same_as_serial((9, 11), (:xmin, :xmax, :ymax), dirichlet2)
    end

    @testset "2D facet left without a relation" begin
        same_as_serial((9, 11), :boundary, unrelated2)
    end

    @testset "2D time-dependent data" begin
        for t in (0.0, 1.3), ns in ((9, 11), (30, 33))

            same_as_serial(ns, :boundary, time2, t)
            same_as_serial(ns, (:xmin, :xmax, :ymax), time_dirichlet2, t)
        end
    end

    @testset "1D Robin" begin
        same_as_serial((17,), :boundary, robin1)
    end

    # nightly only: 3D meshes are left out of the unit group (#650)
    WITH_3D_TESTS && @testset "3D Robin, ∂ₙₙ, Dirichlet" begin
        same_as_serial((5, 6, 4), (:xmin, :xmax, :ymin, :ymax, :zmax), robin3)
    end

    @testset "the update runs in bands" begin
        C = constrained(gridspace(ghost_mesh(CpuPolyester(), (9, 11)); ghosts = :boundary),
            conormal2)
        xm = ThreadMarks(rand(99))
        update_ghosts!(GhostVector(C, xm), C)
        @test nthreads_seen(xm) >= 2
        Ct = constrained(gridspace(ghost_mesh(CpuPolyester(), (9, 11)); ghosts = :boundary),
            time2)
        xt = ThreadMarks(rand(99))
        update_ghosts!(GhostVector(Ct, xt), Ct, 0.5)
        @test nthreads_seen(xt) >= 2
    end
end

# The seven one-axis families on a `GhostVector`, in place: D₋, D₊, M, M₊, Dc, D̃, D̽.
const BANGS = (Bramble.backward_finite_difference!, Bramble.forward_finite_difference!,
    Bramble.backward_average!, Bramble.forward_average!, Bramble.centered_difference!,
    Bramble.forward_star_difference!, Bramble.cross_weighted_difference!)

# The values of `f!` on `v` along `dv`, and its warm bytes, behind a function barrier.
function applied(f!::F, v, dv::Val) where {F}
    out = zeros(eltype(v), length(Bramble.physical(v)))
    f!(out, v, dv)
    return out, alloc(() -> f!(out, v, dv))
end
function applied(f!::typeof(Δₕ!), v)
    out = zeros(eltype(v), length(Bramble.physical(v)))
    f!(out, v)
    return out, alloc(() -> f!(out, v))
end

# A `GhostVector` under `pol` of the physical values `x`, its ghosts updated.
function operand_under(pol, ns, ghosts, rel, x)
    C = constrained(gridspace(ghost_mesh(pol, ns); ghosts), rel)
    return update_ghosts!(GhostVector(C, copy(x)), C)
end

# Every operator under Polyester gives serial's values bitwise (each point is written by one
# band, in the serial order), with ghosts reaching them, and allocates nothing warm.
function operators_as_serial(ns, ghosts, rel)
    x = rand(prod(ns))
    vs = operand_under(CpuSerial(), ns, ghosts, rel, x)
    vp = operand_under(CpuPolyester(), ns, ghosts, rel, x)
    for d in 1:length(ns), f! in BANGS

        rs, _ = applied(f!, vs, Val(d))
        rp, bytes = applied(f!, vp, Val(d))
        @test rp == rs
        @test bytes == 0
    end
    rs, _ = applied(Δₕ!, vs)
    rp, bytes = applied(Δₕ!, vp)
    @test rp == rs
    @test rp != Δₕ(element(gridspace(Bramble.physical_mesh(vp.mesh)), x))
    @test bytes == 0
end

@testset "GhostVector operators, CpuPolyester" begin
    @test Threads.nthreads() >= 2

    @testset "2D conormal, named facets" begin
        for ns in ((9, 11), (30, 33))
            operators_as_serial(ns, :boundary, conormal2)
        end
        operators_as_serial((9, 11), (:xmin, :xmax, :ymax), dirichlet2)
    end

    @testset "1D Robin" begin
        operators_as_serial((17,), :boundary, robin1)
    end

    # nightly only: 3D meshes are left out of the unit group (#650)
    WITH_3D_TESTS && @testset "3D Robin, ∂ₙₙ, Dirichlet" begin
        operators_as_serial((5, 6, 4), (:xmin, :xmax, :ymin, :ymax, :zmax), robin3)
    end

    @testset "the operators run in bands" begin
        C = constrained(gridspace(ghost_mesh(CpuPolyester(), (9, 11)); ghosts = :boundary),
            conormal2)
        for f! in (BANGS[1], BANGS[6], BANGS[7])
            xm = ThreadMarks(rand(99))
            f!(zeros(99), GhostVector(C, xm), Val(1))
            @test nthreads_seen(xm) >= 2
        end
        # `Δₕ!` runs the plain serial engine, then its facet slabs in bands
        xm = ThreadMarks(rand(99))
        v = GhostVector(C, xm)
        Bramble.ghost_values(v) .= 1.0
        Δₕ!(zeros(99), v)
        @test nthreads_seen(xm) >= 2
    end
end

end
