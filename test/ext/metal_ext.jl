module ExtMetalExtTests

using Test
using Bramble
using Metal
using SparseArrays
using LinearAlgebra: I, mul!
using Bramble: Backend, vector, matrix, _backend_eye, _backend_zeros, metal_sparse_csr,
               metal_sparse_csc

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

# ---------------------------------------------------------------------------
# Sparse CSR/CSC: construction, conversion, and SpMV/SpMM accuracy (gpena/Bramble.jl#250)
# ---------------------------------------------------------------------------
#
# Gated on `Metal.functional()` like the testset above, but the skip path here `@warn`s
# instead of only `@test_skip`ing: a silent skip is issue #84's failure mode, and this
# milestone has already shipped one silent skip that had to be fixed later, so a host
# without a functional device is loud about what it did not check.
if !Metal.functional()
    @warn "Skipping Metal sparse CSR/CSC tests: Metal.functional() is false on this host"
    @test_skip "Metal sparse CSR/CSC tests not exercised: Metal.functional() is false"
else
    @testset "metal_sparse_csr / metal_sparse_csc: non-densifying, round-trips" begin
        A = sprand(Float32, 100, 60, 0.05)
        @test nnz(A) > 0

        Gr = metal_sparse_csr(A)
        @test nnz(Gr) == nnz(A)
        @test SparseMatrixCSC(Gr) == A

        Gc = metal_sparse_csc(A)
        @test nnz(Gc) == nnz(A)
        @test SparseMatrixCSC(Gc) == A
    end

    @testset "SpMV: mul!(y, A::CSR, x, α, β) matches CPU SparseMatrixCSC * Vector" begin
        m, n = 50, 90 # non-square
        A = sprand(Float32, m, n, 0.05)
        x = rand(Float32, n)
        G = metal_sparse_csr(A)

        # β = 0: an uninitialised destination is never read, only overwritten.
        y = MtlArray{Float32}(undef, m)
        mul!(y, G, mtl(x), 1.0f0, 0.0f0)
        @test isapprox(Array(y), A * x; atol = 1.0f-5)

        # β != 0 against a non-zero destination: `iszero(β)` is special-cased, so a test
        # that only ever passes β = 0 would not catch a destination wrongly left untouched
        # or wrongly zeroed (gpena/Bramble.jl#250, S3.2).
        y0 = rand(Float32, m)
        α, β = 2.0f0, 3.0f0
        y = mtl(copy(y0))
        mul!(y, G, mtl(x), α, β)
        @test isapprox(Array(y), α .* (A * x) .+ β .* y0; atol = 1.0f-5)
    end

    @testset "SpMM: mul!(C, A::CSR, B, α, β) for dense right-hand sides" begin
        m, n, k = 50, 90, 4 # non-square A
        A = sprand(Float32, m, n, 0.05)
        B = rand(Float32, n, k)
        G = metal_sparse_csr(A)

        C = MtlArray{Float32}(undef, m, k)
        mul!(C, G, mtl(B), 1.0f0, 0.0f0)
        @test isapprox(Array(C), A * B; atol = 1.0f-5)

        # β != 0 against a non-zero destination -- same reason as the SpMV case above.
        C0 = rand(Float32, m, k)
        α, β = 2.0f0, 3.0f0
        C = mtl(copy(C0))
        mul!(C, G, mtl(B), α, β)
        @test isapprox(Array(C), α .* (A * B) .+ β .* C0; atol = 1.0f-5)
    end

    @testset "Float16 SpMV" begin
        A = sprand(Float16, 20, 20, 0.1)
        x = rand(Float16, 20)
        G = metal_sparse_csr(A)
        y = MtlArray{Float16}(undef, 20)
        mul!(y, G, mtl(x), Float16(1.0), Float16(0.0))
        @test isapprox(Array(y), A * x; atol = Float16(1.0e-2))
    end

    @testset "CSC mul! raises ArgumentError naming the CSR conversion" begin
        m, n = 20, 20
        A = sprand(Float32, m, n, 0.1)
        Gc = metal_sparse_csc(A)

        # Split into vector and matrix `mul!` methods on purpose (gpena/Bramble.jl#250,
        # S3.2): a single `::AbstractVecOrMat` signature ties with LinearAlgebra's own
        # generic `mul!` and raises `MethodError: ... is ambiguous` instead of this
        # `ArgumentError` -- so assert the error type and message, not merely that
        # something throws.
        err = try
            mul!(MtlArray{Float32}(undef, m), Gc, mtl(rand(Float32, n)), 1.0f0, 0.0f0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("CSR", sprint(showerror, err))

        err = try
            mul!(MtlArray{Float32}(undef, m, 3), Gc, mtl(rand(Float32, n, 3)), 1.0f0, 0.0f0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("CSR", sprint(showerror, err))
    end

    # #250 also asks for an `ArgumentError` on `Float64`. That guard
    # (`_check_metal_sparse_eltype` inside `mul!`, ext/BrambleMetalExt.jl) is real but
    # unreachable through the normal path: `metal_sparse_csr` on a `Float64` matrix already
    # throws inside Metal.jl's own `mtl()`, because `MtlVector{Float64}` cannot be
    # constructed at all in this Metal.jl version -- confirmed directly rather than
    # contrived. So this tests the behaviour a user actually gets: refusal at construction,
    # with a message naming both `Float64` and `Float32`, rather than reaching for a way to
    # exercise the deeper, currently-unreachable guard.
    @testset "Float64 is refused before it ever reaches mul!" begin
        A64 = sprand(Float64, 10, 10, 0.3)
        err = try
            metal_sparse_csr(A64)
            nothing
        catch e
            e
        end
        @test !isnothing(err)
        msg = sprint(showerror, err)
        @test occursin("Float64", msg)
        @test occursin("Float32", msg)
    end
end

end # module ExtMetalExtTests
