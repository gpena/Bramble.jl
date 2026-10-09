module SpaceThreadedBroadcastTests

using Test
using Bramble
using ..TestUtils: alloc_test, _BROADCAST_SIZES, _broadcast_space, _check_broadcast_equal, _reset_seen!, _spy,
                   _threads_seen

# Under a `CpuThreaded` (`Parallel()`) backend, `dest .= expr` into a `VectorElement` runs
# in static bands of the destination's storage, one per thread.
# Every point runs the very loop body the serial broadcast runs, so the answer must equal
# the `Serial()` one exactly, not merely to a tolerance, including when `dest` itself
# appears on the right-hand side. The meshes are non-uniform, so the operands differ from
# point to point in a way a uniform mesh would not show.

# `dest .= expr` from inside a user's own `Threads.@threads` loop: the banded broadcast
# must run serially there rather than throw Base's nesting error.
function _nested(n)
    Wₕ = _broadcast_space(n, Parallel())
    uₕ, wₕ = Rₕ(Wₕ, x -> sin(sum(x))), Rₕ(Wₕ, x -> prod(x))
    dests = [similar(uₕ) for _ in 1:2]
    Threads.@threads :static for k in 1:2
        dests[k] .= k .* uₕ .+ wₕ
    end
    return map(v -> copy(parent(v)), dests), parent(uₕ), parent(wₕ)
end

_axpy!(v, u, w) = (v .= 2.0 .* u .+ w)

@testset "Threaded broadcast" begin
    @testset "CpuThreaded equal to Serial, n=$n" for n in _BROADCAST_SIZES
        _check_broadcast_equal(n, Parallel())
    end

    # Nested inside a user's own threaded region.
    @testset "Nested threaded region, n=$n" for n in ((1001,), (40, 37), (13, 11, 9))
        dests, u, w = _nested(n)
        for k in 1:2
            @test dests[k] == k .* u .+ w
        end
    end

    @testset "Mismatched spaces still throw" begin
        Up, Vp = _broadcast_space((6, 5), Parallel()), _broadcast_space((6, 5), Parallel(); seed = 1)
        u, v = Rₕ(Up, x -> sum(x)), Rₕ(Vp, x -> sum(x))
        @test_throws ArgumentError (u .= u .+ v)
    end

    # Silent on a single thread: there is nothing to band across.
    if Threads.nthreads() >= 2
        @testset "Runs on several threads, $(D)D" for D in 1:3
            n = D == 1 ? (200_000,) : D == 2 ? (400, 400) : (60, 60, 60)
            Wₕ = _broadcast_space(n, Parallel())
            uₕ, wₕ = Rₕ(Wₕ, x -> sin(sum(x))), Rₕ(Wₕ, x -> prod(x))
            v = similar(uₕ)
            _reset_seen!()
            v .= 2.0 .* _spy(uₕ) .+ wₕ
            @test _threads_seen() >= 2
            @test parent(v) == 2.0 .* parent(uₕ) .+ parent(wₕ)
        end
    end

    # The serial path is unchanged: inferred, and no allocation.
    @testset "Serial path, n=$n" for n in ((1001,), (40, 37))
        Wₕ = _broadcast_space(n, Serial())
        u, w, v = Rₕ(Wₕ, x -> sum(x)), Rₕ(Wₕ, x -> prod(x)), similar(Rₕ(Wₕ, x -> 0.0))
        @test (@inferred _axpy!(v, u, w)) === v
        @test minimum(alloc_test(_axpy!, v, u, w) for _ in 1:5) == 0
    end
end

end # module
