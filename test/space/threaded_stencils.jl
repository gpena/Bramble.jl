module SpaceThreadedStencilsTests

using Test
using Bramble
using Bramble: VectorElement
using Random
using ..TestUtils: alloc_test

# Under a `CpuThreaded` (`Parallel()`) backend every CPU stencil engine -- the one-sided and
# centered difference engines and both average engines -- runs banded along the grid's
# last axis, one band per thread (gpena/Bramble.jl#356). Every point is still computed by
# the very loop body the serial engine runs, so the answer must equal the `Serial()` one
# exactly, not merely to a tolerance. The meshes are non-uniform: on a uniform mesh a
# band that picked up the wrong spacing index would still give the right number.

# Every family reaching `_apply_stencil!` or `_apply_averaged!`, spelled from the
# operator's base name so no Unicode is retyped here.
const _FAMILIES = (:D₋, :D₊, :diff₋, :diff₊, :jump, :Dc, :D̃, :D̽, :M, :M₊, :Mc)
const _CENTERED = (:Dc, :D̽, :Mc)            # need three points along their direction
const _SUFFIXES = ("ₓ", "ᵧ", "₂")

_op(fam, d) = getproperty(Bramble, Symbol(fam, _SUFFIXES[d]))
_op!(fam, d) = getproperty(Bramble, Symbol(fam, _SUFFIXES[d], :!))

function _domain(D)
    D == 1 ? domain(interval(0.0, 1.0)) :
    D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
end

# The same non-uniform mesh twice, once per policy: the seed fixes the random points.
function _mesh_pair(n::NTuple{D, Int}; seed = 356) where {D}
    dom = _domain(D)
    npts = D == 1 ? n[1] : n
    unif = D == 1 ? false : ntuple(_ -> false, D)
    Random.seed!(seed)
    Ωs = mesh(dom, npts, unif; backend = backend(policy = Serial()))
    Random.seed!(seed)
    Ωp = mesh(dom, npts, unif; backend = backend(policy = Parallel()))
    return Ωs, Ωp
end

const _F = (x -> sin(3x) + x^2, x -> sin(3x[1] + 2x[2]) + x[1] * x[2],
    x -> sin(3x[1] + 2x[2] - x[3]) + x[1] * x[3])
const _G = (x -> cos(2x), x -> exp(x[1]) * x[2], x -> x[1] + x[2]^2 * x[3])

# Compare every applicable family and direction, in place and allocating, scalar and
# composite, between the two policies.
function _check_all(n::NTuple{D, Int}) where {D}
    Ωs, Ωp = _mesh_pair(n)
    Ws, Wp = gridspace(Ωs), gridspace(Ωp)
    Vs, Vp = gridspace(Ωs, Val(2)), gridspace(Ωp, Val(2))
    us, up = Rₕ(Ws, _F[D]), Rₕ(Wp, _F[D])
    vs, vp = Rₕ(Vs, (_F[D], _G[D])), Rₕ(Vp, (_F[D], _G[D]))
    @test parent(us) == parent(up)
    for d in 1:D, fam in _FAMILIES

        fam in _CENTERED && n[d] < 3 && continue
        f, f! = _op(fam, d), _op!(fam, d)
        @testset "$(fam)$(_SUFFIXES[d]) n=$n" begin
            ws, wp = similar(us), similar(up)
            parent(wp) .= NaN               # every point must be written
            f!(ws, us)
            @test f!(wp, up) === wp
            @test parent(wp) == parent(ws)
            @test parent(f(up)) == parent(f(us))

            ws2, wp2 = similar(vs), similar(vp)
            f!(ws2, vs)
            f!(wp2, vp)
            @test parent(wp2) == parent(ws2)
            @test parent(f(vp)) == parent(f(vs))
        end
    end
end

# A storage vector recording which threads read it: proves the banded path ran, rather
# than trusting the dispatch.
const _SEEN = Threads.Atomic{UInt64}(0)
struct _Spy{T} <: AbstractVector{T}
    x::Vector{T}
end
Base.size(s::_Spy) = size(s.x)
Base.IndexStyle(::Type{<:_Spy}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(s::_Spy, i::Int)
    Threads.atomic_or!(_SEEN, UInt64(1) << ((Threads.threadid() - 1) % 64))
    return s.x[i]
end

@testset "Threaded stencil engines" begin
    @testset "Equal to Serial, $(D)D" for D in 1:3
        sizes = D == 1 ? ((1,), (2,), (3,), (5,), (1001,)) :
                D == 2 ? ((9, 1), (9, 2), (3, 3), (11, 7), (40, 37)) :
                ((5, 4, 1), (5, 4, 2), (4, 3, 5), (9, 8, 13))
        # Banded axes shorter than the thread count, down to a single point, leave some
        # bands empty; the operator must not notice.
        foreach(_check_all, sizes)
    end

    @testset "Runs on several threads" begin
        if Threads.nthreads() < 2
            @test_skip false
        else
            _, Ωp = _mesh_pair((64, 64))
            up = Rₕ(gridspace(Ωp), _F[2])
            for fam in (:D₋, :Dc, :M₊, :Mc), d in 1:2

                spy = VectorElement(_Spy(copy(parent(up))), space(up))
                w = similar(up)
                _SEEN[] = 0
                _op!(fam, d)(w, spy)
                @test count_ones(_SEEN[]) >= 2
                @test parent(w) == parent(_op(fam, d)(up))
            end
        end
    end

    @testset "In-place allocation independent of grid size" begin
        function min_bytes(n)
            _, Ωp = _mesh_pair((n, n))
            up = Rₕ(gridspace(Ωp), _F[2])
            w = similar(up)
            return map((:D₋, :D₊, :Dc, :D̃, :D̽, :M, :M₊, :Mc)) do fam
                minimum(alloc_test(_op!(fam, 2), w, up) for _ in 1:5)
            end
        end
        @test min_bytes(16) == min_bytes(160)
    end

    @testset "CpuPolyester stubs name Polyester" begin
        # Untyped arguments reach the `src/` stub even when `BramblePolyesterExt` is loaded.
        @test_throws ArgumentError Bramble._batch_difference_engine!(
            nothing, nothing, nothing, nothing, nothing, nothing
        )
        @test_throws ArgumentError Bramble._batch_average_engine!(
            nothing, nothing, nothing, nothing, nothing
        )
        @test_throws ArgumentError Bramble._batch_centered_average_engine!(
            nothing, nothing, nothing, nothing
        )
    end
end

end
