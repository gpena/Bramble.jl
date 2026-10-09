module SpaceThreadedStencilsTests

using Test
using Bramble
using ..TestUtils: alloc_test, WITH_SLOW_TESTS, _check_stencils, _reset_seen!, _spy, _STENCIL_F,
                   _stencil_mesh_pair, _stencil_op, _stencil_op!, _threads_seen

# Under a `CpuThreaded` (`Parallel()`) backend every CPU stencil engine -- the one-sided and
# centered difference engines and both average engines -- runs banded along the grid's
# last axis, one band per thread. Every point is still computed by
# the very loop body the serial engine runs, so the answer must equal the `Serial()` one
# exactly, not merely to a tolerance. The meshes are non-uniform: on a uniform mesh a
# band that picked up the wrong spacing index would still give the right number.

@testset "Threaded stencil engines" begin
    @testset "Equal to Serial, $(D)D" for D in 1:3
        # `unit` keeps, per dimension, the degenerate last axis (2 points, empty bands) and
        # the smallest size the centered families accept; `slow` runs the full sweep.
        sizes = if WITH_SLOW_TESTS
            D == 1 ? ((1,), (2,), (3,), (5,), (1001,)) :
            D == 2 ? ((9, 1), (9, 2), (3, 3), (11, 7), (40, 37)) :
            ((5, 4, 1), (5, 4, 2), (4, 3, 5), (9, 8, 13))
        else
            D == 1 ? ((2,), (3,)) : D == 2 ? ((9, 2), (3, 3)) : ((5, 4, 2), (4, 3, 3))
        end
        # Banded axes shorter than the thread count, down to a single point, leave some
        # bands empty; the operator must not notice.
        foreach(n -> _check_stencils(n, Parallel()), sizes)
    end

    @testset "Runs on several threads" begin
        if Threads.nthreads() < 2
            @test_skip false
        else
            _, Ωp = _stencil_mesh_pair((64, 64), Parallel())
            up = Rₕ(gridspace(Ωp), _STENCIL_F[2])
            for fam in (:D₋, :Dc, :M₊, :Mc), d in 1:2

                w = similar(up)
                _reset_seen!()
                _stencil_op!(fam, d)(w, _spy(up))
                @test _threads_seen() >= 2
                @test parent(w) == parent(_stencil_op(fam, d)(up))
            end
        end
    end

    @testset "In-place allocation is size-independent" begin
        function min_bytes(n)
            _, Ωp = _stencil_mesh_pair((n, n), Parallel())
            up = Rₕ(gridspace(Ωp), _STENCIL_F[2])
            w = similar(up)
            return map((:D₋, :D₊, :Dc, :D̃, :D̽, :M, :M₊, :Mc)) do fam
                minimum(alloc_test(_stencil_op!(fam, 2), w, up) for _ in 1:5)
            end
        end
        @test min_bytes(16) == min_bytes(160)
    end

    @testset "Batch engine stubs name Polyester" begin
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
