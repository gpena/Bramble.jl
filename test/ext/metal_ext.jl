using Test
using Bramble
using Metal
using LinearAlgebra: I
using Bramble: Backend, vector, matrix, _backend_eye, _backend_zeros

# BrambleMetalExt's backend allocation primitives. Nothing here builds a mesh/gridspace on
# a Metal-backed vector: `mesh` construction fills point coordinates with a scalar CPU
# loop (`_points!`), which GPUArrays refuses on a device array ("Scalar indexing is
# disallowed") -- confirmed directly, not assumed. So a Metal backend is currently only
# exercised at the allocation layer this extension actually implements
# (`vector`/`matrix`/`_backend_eye`/`_backend_zeros`/`metal_backend`), not through a full
# PDE assembly pipeline; building that pipeline on a GPU-resident mesh is a separate gap,
# outside the extension's own scope.
#
# `Metal.functional()` gates everything below: precompiling and loading `Metal` succeeds on
# any platform (it degrades gracefully rather than erroring, the same convention CUDA.jl
# uses), but only a real Apple Silicon Mac has a working device, so a CI runner without one
# skips rather than fails.

@testset "BrambleMetalExt" begin
    if !Metal.functional()
        @test_skip "Metal backend not exercised: Metal.functional() is false on this host"
    else
        @testset "metal_backend element types" begin
            @test metal_backend() isa Backend
            @test metal_backend(Float32) isa Backend
            @test metal_backend(Float16) isa Backend
            # Float64 is unsupported on Apple Silicon GPUs. The Metal-loaded method only
            # matches `T <: Union{Float16, Float32}`, so `Float64` falls through to the
            # generic "requires Metal.jl" stub in main `src/` by ordinary dispatch
            # specificity -- the same `ErrorException` as the package-not-loaded case, even
            # though Metal is loaded here; only the type is rejected.
            @test_throws ErrorException metal_backend(Float64)
        end

        @testset "vector/matrix allocation" begin
            b = metal_backend()
            v = vector(b, 6)
            @test v isa MtlVector{Float32}
            @test length(v) == 6

            M = matrix(b, 3, 4)
            @test M isa MtlMatrix{Float32}
            @test size(M) == (3, 4)

            b16 = metal_backend(Float16)
            v16 = vector(b16, 4)
            @test v16 isa MtlVector{Float16}
        end

        @testset "_backend_eye / _backend_zeros" begin
            n = 5
            E = _backend_eye(MtlMatrix{Float32}, n)
            @test E isa MtlMatrix{Float32}
            @test Array(E) == Matrix{Float32}(I, n, n)

            Z = _backend_zeros(MtlMatrix{Float32}, n)
            @test Z isa MtlMatrix{Float32}
            @test Array(Z) == zeros(Float32, n, n)
        end

        @testset "Round-trips through Array" begin
            b = metal_backend()
            data = Float32[1.0, 2.0, 3.0]
            v = vector(b, 3)
            copyto!(v, data)
            @test Array(v) == data
        end
    end
end
