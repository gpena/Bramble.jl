module ExtMetalExtTests

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
# `Metal.functional()` gates every testset here that touches an actual device array:
# precompiling and loading `Metal` succeeds on any platform (it degrades gracefully rather
# than erroring, the same convention CUDA.jl uses), but only a real Apple Silicon Mac has a
# working device, so a CI runner without one skips those rather than fails. The
# "rejects a CPU policy over device storage" testset below is the one exception: it checks a
# construction-time `ArgumentError` derived from type information alone, so it runs whenever
# Metal is loaded, functional or not.

@testset "BrambleMetalExt" begin
    # A device VT (MtlVector) under a host CpuPolicy is rejected at construction
    # (gpena/Bramble.jl#296, #298): `_metal_backend` only builds `Backend{MtlVector{T},
    # MtlMatrix{T}, typeof(policy)}()`, a type-level construction that never allocates a
    # device array, so the rejection fires from `MtlVector`/`policy` type information alone.
    # That means it needs `using Metal` to be loaded (for the `MtlVector` type and the
    # `_metal_backend` method to exist) but not a functional device, so it runs outside the
    # `Metal.functional()` gate below and is exercised on any host with Metal loaded.
    @testset "metal_backend rejects a CPU policy over device storage" begin
        for cpu_policy in (CpuSerial(), CpuThreaded())
            err = try
                metal_backend(; policy = cpu_policy)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = sprint(showerror, err)
            @test occursin("MtlVector", msg)
            @test occursin(string(typeof(cpu_policy)), msg)
        end
    end

    if !Metal.functional()
        @test_skip "Metal backend not exercised: Metal.functional() is false on this host"
    else
        @testset "metal_backend element types" begin
            @test metal_backend() isa Backend
            # a GPU is massively parallel and cannot execute serially, so the default says
            # so (gpena/Bramble.jl#191); it used to be Serial()
            @test execution_policy(metal_backend()) === GpuAsync()
            @test execution_policy(metal_backend(Float16)) === GpuAsync()
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

end # module ExtMetalExtTests
