module SpaceThreadedBroadcastTests

using Test
using Bramble
using Bramble: VectorElement, CpuPolyester
using Random
using ..TestUtils: alloc_test

# Under a `CpuThreaded` (`Parallel()`) backend, `dest .= expr` into a `VectorElement` runs
# in static bands of the destination's storage, one per thread (gpena/Bramble.jl#357).
# Every point runs the very loop body the serial broadcast runs, so the answer must equal
# the `Serial()` one exactly, not merely to a tolerance, including when `dest` itself
# appears on the right-hand side. The meshes are non-uniform, so the operands differ from
# point to point in a way a uniform mesh would not show.

function _domain(D)
    D == 1 ? domain(interval(0.0, 1.0)) :
    D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 2.0)) :
    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
end

# The same non-uniform mesh under the given policy: the seed fixes the random points.
function _space(n::NTuple{D, Int}, policy; seed = 357) where {D}
    Random.seed!(seed)
    npts = D == 1 ? n[1] : n
    unif = D == 1 ? false : ntuple(_ -> false, D)
    return gridspace(mesh(_domain(D), npts, unif; backend = backend(policy = policy)))
end

const _SIZES = ((1,), (2,), (7,), (1001,), (5, 3), (40, 37), (4, 3, 5), (13, 11, 9))

# Every broadcast shape the issue names, each writing into a fresh `NaN` destination (or
# updating a copy in place): VectorElements only, a plain vector, literal scalars, a `Ref`
# and a runtime `Float64`, and `dest` on its own right-hand side.
function _results(n, policy)
    Wₕ = _space(n, policy)
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

function _check_equal(n, policy)
    s, p = _results(n, Serial()), _results(n, policy)
    for key in keys(s)
        @test p[key] == s[key]
    end
end

# `dest .= expr` from inside a user's own `Threads.@threads` loop: the banded broadcast
# must run serially there rather than throw Base's nesting error.
function _nested(n)
    Wₕ = _space(n, Parallel())
    uₕ, wₕ = Rₕ(Wₕ, x -> sin(sum(x))), Rₕ(Wₕ, x -> prod(x))
    dests = [similar(uₕ) for _ in 1:2]
    Threads.@threads :static for k in 1:2
        dests[k] .= k .* uₕ .+ wₕ
    end
    return map(v -> copy(parent(v)), dests), parent(uₕ), parent(wₕ)
end

# Records which threads read it, to see the bands spread.
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

_axpy!(v, u, w) = (v .= 2.0 .* u .+ w)

@testset "Threaded broadcast" begin
    @testset "CpuThreaded equal to Serial, n=$n" for n in _SIZES
        _check_equal(n, Parallel())
    end

    @testset "Nested in a user's threaded region, n=$n" for n in ((1001,), (40, 37), (13, 11, 9))
        dests, u, w = _nested(n)
        for k in 1:2
            @test dests[k] == k .* u .+ w
        end
    end

    @testset "Mismatched spaces still throw" begin
        Up, Vp = _space((6, 5), Parallel()), _space((6, 5), Parallel(); seed = 1)
        u, v = Rₕ(Up, x -> sum(x)), Rₕ(Vp, x -> sum(x))
        @test_throws ArgumentError (u .= u .+ v)
    end

    # Silent on a single thread: there is nothing to band across.
    if Threads.nthreads() >= 2
        @testset "Runs on several threads, $(D)D" for D in 1:3
            n = D == 1 ? (200_000,) : D == 2 ? (400, 400) : (60, 60, 60)
            Wₕ = _space(n, Parallel())
            uₕ, wₕ = Rₕ(Wₕ, x -> sin(sum(x))), Rₕ(Wₕ, x -> prod(x))
            spy = VectorElement(_Spy(copy(parent(uₕ))), Wₕ)
            v = similar(uₕ)
            _SEEN[] = 0
            v .= 2.0 .* spy .+ wₕ
            @test count_ones(_SEEN[]) >= 2
            @test parent(v) == 2.0 .* parent(uₕ) .+ parent(wₕ)
        end
    end

    # The serial path is unchanged: inferred, and no allocation.
    @testset "Serial path, n=$n" for n in ((1001,), (40, 37))
        Wₕ = _space(n, Serial())
        u, w, v = Rₕ(Wₕ, x -> sum(x)), Rₕ(Wₕ, x -> prod(x)), similar(Rₕ(Wₕ, x -> 0.0))
        @test (@inferred _axpy!(v, u, w)) === v
        @test minimum(alloc_test(_axpy!, v, u, w) for _ in 1:5) == 0
    end

    # The Polyester arm runs only when `BramblePolyesterExt` is loaded in this process;
    # the broadcast still runs serially there until it gets its own batched loop.
    if Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing
        @testset "CpuPolyester equal to Serial, n=$n" for n in _SIZES
            _check_equal(n, CpuPolyester())
        end
    end
end

end # module
