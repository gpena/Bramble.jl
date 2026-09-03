using Test
using Bramble
using Bramble: Backend, backend, vector, matrix, vector_type, matrix_type, backend_types,
               backend_eye, backend_zeros, execution_policy, Serial, Parallel
using SparseArrays
using LinearAlgebra: diag, I

# A lightweight mock GPU array wrapper that simulates GPU vendor arrays (like MtlArray/CuArray)
struct MockGPUArray{T, N} <: DenseArray{T, N}
    data::Array{T, N}
end
function MockGPUArray{T, N}(::UndefInitializer, dims::Vararg{Integer, N}) where {T, N}
    MockGPUArray(Array{T, N}(undef, dims...))
end
function MockGPUArray{T, N}(::UndefInitializer, dims::NTuple{N, Integer}) where {T, N}
    MockGPUArray(Array{T, N}(undef, dims))
end
Base.size(A::MockGPUArray) = size(A.data)
Base.getindex(A::MockGPUArray, i::Int...) = getindex(A.data, i...)
Base.setindex!(A::MockGPUArray, v, i::Int...) = setindex!(A.data, v, i...)
Base.IndexStyle(::Type{<:MockGPUArray}) = IndexLinear()
Base.fill!(A::MockGPUArray{T}, v) where {T} = (fill!(A.data, v); A)

const MockGPUVector{T} = MockGPUArray{T, 1}
const MockGPUMatrix{T} = MockGPUArray{T, 2}

@testset "Backend Tests" begin
    @testset "Backend Constructor" begin
        # Test default Backend
        be_default = backend()
        @test vector_type(be_default) === Vector{Float64}
        @test matrix_type(be_default) === SparseMatrixCSC{Float64, Int}
        @test execution_policy(be_default) === Serial()
        @test execution_policy(typeof(be_default)) === Serial()
        @test be_default isa Backend{Vector{Float64}, SparseMatrixCSC{Float64, Int}}

        # Test single-argument positional constructor
        be_pos_f64 = backend(Float64)
        @test be_pos_f64 === be_default
        be_pos_f32 = backend(Float32)
        @test vector_type(be_pos_f32) === Vector{Float32}
        @test matrix_type(be_pos_f32) === SparseMatrixCSC{Float32, Int}
        @test execution_policy(be_pos_f32) === Serial()

        be_pos_par = backend(Float32; policy = Parallel())
        @test execution_policy(be_pos_par) === Parallel()
        @test execution_policy(typeof(be_pos_par)) === Parallel()

        # Test custom Float32 Backend (Dense-Sparse)
        be_f32_ds = backend(vector_type = Vector{Float32}, matrix_type = SparseMatrixCSC{
            Float32, Int})
        @test vector_type(be_f32_ds) === Vector{Float32}
        @test matrix_type(be_f32_ds) === SparseMatrixCSC{Float32, Int}
        @test be_f32_ds isa Backend{Vector{Float32}, SparseMatrixCSC{Float32, Int}}

        # Test custom Float64 Backend (Dense-Dense)
        be_f64_dd = backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})
        @test vector_type(be_f64_dd) === Vector{Float64}
        @test matrix_type(be_f64_dd) === Matrix{Float64}
        @test be_f64_dd isa Backend{Vector{Float64}, Matrix{Float64}}
        # Non-dense vector types (e.g. SparseVector) are rejected at the type level
        @test_throws TypeError backend(vector_type = SparseVector{Float64, Int})

        # Test custom Complex Backend (Dense-Dense)
        be_c64_dd = backend(vector_type = Vector{ComplexF64}, matrix_type = Matrix{ComplexF64})
        @test vector_type(be_c64_dd) === Vector{ComplexF64}
        @test matrix_type(be_c64_dd) === Matrix{ComplexF64}
        @test be_c64_dd isa Backend{Vector{ComplexF64}, Matrix{ComplexF64}}
    end

    @testset "vector" begin
        n = 15

        # Default Backend (Vector{Float64})
        be_default = backend()
        v_default = vector(be_default, n)
        @test v_default isa Vector{Float64}
        @test length(v_default) == n
        @test eltype(v_default) === Float64

        # Dense Float32 Backend
        be_f32 = backend(vector_type = Vector{Float32}, matrix_type = Matrix{Float32})
        v_f32 = vector(be_f32, n)
        @test v_f32 isa Vector{Float32}
        @test length(v_f32) == n
        @test eltype(v_f32) === Float32

        # Zero length vector
        v_zero = vector(be_default, 0)
        @test v_zero isa Vector{Float64}
        @test length(v_zero) == 0
    end

    @testset "matrix" begin
        m, n = 10, 20

        # Default Backend (SparseMatrixCSC{Float64, Int})
        be_default = backend()
        M_default = matrix(be_default, m, n)
        @test M_default isa SparseMatrixCSC{Float64, Int}
        @test size(M_default) == (m, n)
        @test eltype(M_default) === Float64
        @test nnz(M_default) == 0

        # Dense Backend (Matrix{Float64})
        be_dense = backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})
        M_dense = matrix(be_dense, m, n)
        @test M_dense isa Matrix{Float64}
        @test size(M_dense) == (m, n)
        @test eltype(M_dense) === Float64

        # Sparse Matrix Float32 Backend
        be_f32_sparse = backend(vector_type = Vector{Float32},
            matrix_type = SparseMatrixCSC{Float32, Int32})
        M_f32 = matrix(be_f32_sparse, m, n)
        @test M_f32 isa SparseMatrixCSC{Float32, Int32}
        @test size(M_f32) == (m, n)
        @test eltype(M_f32) === Float32

        # Zero dimensions
        M_zero_row = matrix(be_default, 0, n)
        @test M_zero_row isa SparseMatrixCSC{Float64, Int}
        @test size(M_zero_row) == (0, n)

        M_zero_col = matrix(be_default, m, 0)
        @test M_zero_col isa SparseMatrixCSC{Float64, Int}
        @test size(M_zero_col) == (m, 0)

        M_zero_all = matrix(be_default, 0, 0)
        @test M_zero_all isa SparseMatrixCSC{Float64, Int}
        @test size(M_zero_all) == (0, 0)

        M_zero_row_dense = matrix(be_dense, 0, n)
        @test M_zero_row_dense isa Matrix{Float64}
        @test size(M_zero_row_dense) == (0, n)

        M_zero_col_dense = matrix(be_dense, m, 0)
        @test M_zero_col_dense isa Matrix{Float64}
        @test size(M_zero_col_dense) == (m, 0)
    end

    @testset "GPU-Agnostic Backend (Mock GPU Array)" begin
        be_gpu = backend(vector_type = MockGPUVector{Float32}, matrix_type = MockGPUMatrix{Float32})
        @test vector_type(be_gpu) === MockGPUVector{Float32}
        @test matrix_type(be_gpu) === MockGPUMatrix{Float32}
        @test eltype(be_gpu) === Float32

        v_gpu = vector(be_gpu, 25)
        @test v_gpu isa MockGPUVector{Float32}
        @test length(v_gpu) == 25

        m_gpu = matrix(be_gpu, 10, 20)
        @test m_gpu isa MockGPUMatrix{Float32}
        @test size(m_gpu) == (10, 20)

        z_gpu = backend_zeros(be_gpu, 8)
        @test z_gpu isa MockGPUMatrix{Float32}
        @test size(z_gpu) == (8, 8)
        @test all(z_gpu .== 0.0f0)

        eye_gpu = backend_eye(be_gpu, 5)
        @test eye_gpu isa MockGPUMatrix{Float32}
        @test size(eye_gpu) == (5, 5)
        @test eye_gpu[1, 1] == 1.0f0
        @test eye_gpu[1, 2] == 0.0f0
        @test eye_gpu[3, 3] == 1.0f0
    end

    # Conditional test for Metal on macOS without requiring it as a project dependency
    if Sys.isapple()
        metal_pkg = Base.find_package("Metal")
        if metal_pkg !== nothing
            try
                @eval using Metal
                if isdefined(Main, :Metal) && Metal.functional()
                    @testset "Metal GPU Backend (macOS Apple Silicon)" begin
                        be_metal = backend(vector_type = MtlVector{Float32}, matrix_type = MtlMatrix{Float32})
                        @test vector_type(be_metal) === MtlVector{Float32}
                        @test matrix_type(be_metal) === MtlMatrix{Float32}

                        v = vector(be_metal, 10)
                        @test v isa MtlVector{Float32}
                        @test length(v) == 10

                        m = matrix(be_metal, 5, 5)
                        @test m isa MtlMatrix{Float32}
                        @test size(m) == (5, 5)

                        z = backend_zeros(be_metal, 4)
                        @test z isa MtlMatrix{Float32}
                        @test size(z) == (4, 4)
                    end
                end
            catch e
                @info "Metal is installed but initialization skipped in this environment" exception=e
            end
        end
    end

    @testset "backend_types" begin
        be_default = backend()
        T, VT, MT, BType = backend_types(be_default)
        @test T === Float64
        @test VT === Vector{Float64}
        @test MT === SparseMatrixCSC{Float64, Int}
        @test BType === Backend{Vector{Float64}, SparseMatrixCSC{Float64, Int}, Serial}

        # Test on type
        T2, VT2, MT2, BType2 = backend_types(typeof(be_default))
        @test T2 === Float64
        @test VT2 === Vector{Float64}
        @test MT2 === SparseMatrixCSC{Float64, Int}
        @test BType2 === Backend{Vector{Float64}, SparseMatrixCSC{Float64, Int}, Serial}

        # Test Float32 backend
        be_f32 = backend(vector_type = Vector{Float32}, matrix_type = Matrix{Float32})
        T_f32, VT_f32, MT_f32, _ = backend_types(be_f32)
        @test T_f32 === Float32
        @test VT_f32 === Vector{Float32}
        @test MT_f32 === Matrix{Float32}
    end

    @testset "eltype on Backend" begin
        be_default = backend()
        @test eltype(be_default) === Float64
        @test eltype(typeof(be_default)) === Float64

        be_f32 = backend(vector_type = Vector{Float32}, matrix_type = Matrix{Float32})
        @test eltype(be_f32) === Float32
        @test eltype(typeof(be_f32)) === Float32

        be_complex = backend(vector_type = Vector{ComplexF64}, matrix_type = Matrix{ComplexF64})
        @test eltype(be_complex) === ComplexF64
    end

    @testset "backend_eye and backend_zeros" begin
        n = 5
        be_default = backend()
        be_dense = backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})

        # Sparse backend_eye
        I_sparse = backend_eye(be_default, n)
        @test I_sparse isa SparseMatrixCSC{Float64, Int}
        @test size(I_sparse) == (n, n)
        @test diag(I_sparse) == ones(n)
        @test nnz(I_sparse) == n

        # Dense backend_eye
        I_dense = backend_eye(be_dense, n)
        @test I_dense isa Matrix{Float64}
        @test size(I_dense) == (n, n)
        @test I_dense == Matrix{Float64}(I, n, n)

        # Sparse backend_zeros
        Z_sparse = backend_zeros(be_default, n)
        @test Z_sparse isa SparseMatrixCSC{Float64, Int}
        @test size(Z_sparse) == (n, n)
        @test nnz(Z_sparse) == 0

        # Dense backend_zeros
        Z_dense = backend_zeros(be_dense, n)
        @test Z_dense isa Matrix{Float64}
        @test size(Z_dense) == (n, n)
        @test all(Z_dense .== 0.0)
    end

    @testset "Type Stability & Allocations" begin
        be = backend()
        @inferred backend()
        @inferred vector_type(be)
        @inferred matrix_type(be)
        @inferred execution_policy(be)
        @inferred eltype(be)
        @inferred backend_types(be)

        # Backend constructor is zero-allocation singleton
        @test_allocs backend()
        @test_allocs vector_type(be)
        @test_allocs matrix_type(be)
        @test_allocs execution_policy(be)
        @test_allocs eltype(be)
        @test_allocs backend_types(be)
    end

    @testset "Display / Show" begin
        be = backend()
        io = IOBuffer()
        show(io, be)
        str = String(take!(io))
        @test occursin("Backend", str)
        @test occursin("vector = Vector{Float64}", str)

        show(IOContext(io, :compact => true), be)
        @test occursin("Backend{Float64}", String(take!(io)))
    end
end
@testset "Backend: additional coverage" begin
    @testset "vector_type / matrix_type on Type (lines 19, 27)" begin
        BE = Backend{Vector{Float64}, SparseMatrixCSC{Float64, Int}, Serial}
        @test vector_type(BE) === Vector{Float64}
        @test matrix_type(BE) === SparseMatrixCSC{Float64, Int}
    end

    @testset "generic vector/matrix fallback (lines 81-91, 102-112)" begin
        struct SizeConstructibleVec{T} <: DenseVector{T}
            data::Vector{T}
        end
        SizeConstructibleVec{T}(n::Integer) where {T} = SizeConstructibleVec{T}(zeros(T, n))
        Base.size(v::SizeConstructibleVec) = size(v.data)
        Base.getindex(v::SizeConstructibleVec, i) = v.data[i]

        struct SizeConstructibleMat{T} <: AbstractMatrix{T}
            data::Matrix{T}
        end
        SizeConstructibleMat{T}(n::Integer, m::Integer) where {T} = SizeConstructibleMat{T}(zeros(T, n, m))
        Base.size(v::SizeConstructibleMat) = size(v.data)
        Base.getindex(v::SizeConstructibleMat, i, j) = v.data[i, j]

        be_custom = backend(vector_type = SizeConstructibleVec{Float64},
            matrix_type = SizeConstructibleMat{Float64})

        v = vector(be_custom, 5)
        @test v isa SizeConstructibleVec{Float64}
        @test length(v) == 5

        M = matrix(be_custom, 3, 4)
        @test M isa SizeConstructibleMat{Float64}
        @test size(M) == (3, 4)

        # Types that fail both undef and size constructor -> trigger _throw_vector_error & _throw_matrix_error
        struct UnconstructibleVec{T} <: DenseVector{T} end
        struct UnconstructibleMat{T} <: AbstractMatrix{T} end

        be_fail = backend(vector_type = UnconstructibleVec{Float64},
            matrix_type = UnconstructibleMat{Float64})
        @test_throws ErrorException vector(be_fail, 5)
        @test_throws ErrorException matrix(be_fail, 3, 4)
    end

    @testset "metal_backend stub error" begin
        if isdefined(Main, :Metal)
            @test metal_backend() isa Backend
            @test metal_backend(Float32) isa Backend
        else
            @test_throws ErrorException metal_backend()
            @test_throws ErrorException metal_backend(Float32)
        end
    end
end
