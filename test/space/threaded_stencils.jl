module SpaceThreadedStencilsTests

using Test
using Bramble
using ..TestUtils: alloc_test, TEST_DIMS, WITH_SLOW_TESTS, _check_stencils, _reset_seen!, _spy, _STENCIL_F,
                   _stencil_mesh_pair, _stencil_op, _stencil_op!, _threads_seen
using Bramble: StarSpacings, Forward, Backward, Centered, CrossWeighted, _average_band!,
               _average_band_adjoint!, _centered_average_band!, _centered_average_band_adjoint!,
               _difference_band!, _difference_band_adjoint!, _difference_spacing_adjoint!,
               _difference_band_spacing_tangent!, _star_spacings_adjoint!
using ForwardDiff
using LinearAlgebra: dot
using Random: MersenneTwister

# Under a `CpuThreaded` (`Parallel()`) backend every CPU stencil engine -- the one-sided and
# centered difference engines and both average engines -- runs banded along the grid's
# last axis, one band per thread. Every point is still computed by
# the very loop body the serial engine runs, so the answer must equal the `Serial()` one
# exactly, not merely to a tolerance. The meshes are non-uniform: on a uniform mesh a
# band that picked up the wrong spacing index would still give the right number.

@testset "Threaded stencil engines" begin
    @testset "Equal to Serial, $(D)D" for D in TEST_DIMS
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

# The transposed and linearised band kernels (operators/difference.jl, operators/average.jl)
# are what the Enzyme rules of `CpuThreaded` and `CpuPolyester` call, but they are plain
# Julia: checked here against the band kernels they transpose, and against ForwardDiff in
# the spacings, so the unit suite covers them without Enzyme.
const STENCILS = ((Backward(), Val(:none)), (Backward(), Val(:vector)),
    (Forward(), Val(:none)), (Forward(), Val(:view)), (Forward(), Val(:star)),
    (Centered(), Val(:star)), (CrossWeighted(), Val(:vector)))

# The spacing a family passes the engine, built from the spacings `s` of one axis.
_spacing(::Val{:none}, s) = nothing
_spacing(::Val{:vector}, s) = s
_spacing(::Val{:view}, s) = view(s, 2:length(s))
_spacing(::Val{:star}, s) = StarSpacings(s)

# The cotangent of `s` from that of the spacing values `d_hv` (indexed as the spacing is).
_fold(::Val{:vector}, d_hv, s) = copy(d_hv)
_fold(::Val{:view}, d_hv, s) = [0.0; d_hv]
function _fold(::Val{:star}, d_hv, s)
    d_s = zeros(length(s))
    _star_spacings_adjoint!(d_s, d_hv, 1, 1)
    return d_s
end

function _apply(x, h, dims, dir, dim_val, nbands)
    out = zeros(promote_type(eltype(x), eltype(something(h, Float64[]))), prod(dims))
    for b in 1:nbands
        _difference_band!(out, x, h, dims, dir, dim_val, nbands, b)
    end
    return out
end

# `band!(out, b)` and `adjoint!(d_in, b)` over every band; `⟨w, A x⟩ = ⟨Aᵀ w, x⟩`.
function _transposes(band!, adjoint!, x, w, nbands)
    out, d_in = zeros(length(x)), zeros(length(x))
    for b in 1:nbands
        band!(out, b)
        adjoint!(d_in, b)
    end
    return dot(w, out) ≈ dot(d_in, x)
end

@testset "Transposed band kernels" begin
    rng = MersenneTwister(369)
    @testset "Difference bands" begin
        for dims in ((9,), (7, 6), (5, 4, 6)), D in 1:length(dims), (dir, kind) in STENCILS
            n, len = dims[D], prod(dims)
            s = 0.1 .+ rand(rng, n)
            h = _spacing(kind, s)
            x, w = randn(rng, len), randn(rng, len)
            # Two bands, and more bands than the last axis has points (empty bands).
            for nbands in (2, last(dims) + 2)
                d_in = zeros(len)
                for b in 1:nbands
                    _difference_band_adjoint!(d_in, w, h, dims, dir, Val(D), nbands, b)
                end
                @test dot(w, _apply(x, h, dims, dir, Val(D), nbands)) ≈ dot(d_in, x) rtol=1e-12
            end
            h === nothing && continue
            # The spacing cotangent and tangent against ForwardDiff in the spacings.
            g(t) = dot(w, _apply(x, _spacing(kind, t), dims, dir, Val(D), 1))
            d_hv = zeros(length(h))
            for b in 1:3
                _difference_spacing_adjoint!(d_hv, x, w, h, dims, dir, Val(D), 3, b)
            end
            @test _fold(kind, d_hv, s) ≈ ForwardDiff.gradient(g, s) rtol=1e-12
            δ = randn(rng, n)
            tangent = zeros(len)
            for b in 1:3
                _difference_band_spacing_tangent!(tangent, x, h, _spacing(kind, δ), dims, dir,
                    Val(D), 3, b)
            end
            ref = ForwardDiff.derivative(t -> _apply(x, _spacing(kind, s .+ t .* δ), dims, dir,
                    Val(D), 1), 0.0)
            @test tangent ≈ ref rtol=1e-12
        end
    end

    @testset "Average bands" begin
        for dims in ((9,), (7, 6), (5, 4, 6)), D in 1:length(dims), nbands in (2, last(dims) + 2)
            len = prod(dims)
            x, w = randn(rng, len), randn(rng, len)
            for dir in (Forward(), Backward())
                @test _transposes(
                    (out, b) -> _average_band!(out, x, dims, dir, Val(D), nbands, b),
                    (d_in, b) -> _average_band_adjoint!(d_in, w, dims, dir, Val(D), nbands, b),
                    x, w, nbands)
            end
            @test _transposes(
                (out, b) -> _centered_average_band!(out, x, dims, Val(D), nbands, b),
                (d_in, b) -> _centered_average_band_adjoint!(d_in, w, dims, Val(D), nbands, b),
                x, w, nbands)
        end
    end
end

end
