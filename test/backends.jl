using Bramble: Backend, vector, matrix, vector_type, matrix_type
using SparseArrays

@testset "Backand Tests" begin
	@testset "Backend Constructor" begin
		# Test default Backend
		be_default = Backend()
		@test vector_type(be_default) === Vector{Float64}
		@test matrix_type(be_default) === SparseMatrixCSC{Float64,Int}
		@test be_default isa Backend{Vector{Float64},SparseMatrixCSC{Float64,Int}}

		# Test custom Float32 Backend (Dense-Sparse)
		be_f32_ds = Backend(vector_type = Vector{Float32}, matrix_type = SparseMatrixCSC{Float32,Int})
		@test vector_type(be_f32_ds) === Vector{Float32}
		@test matrix_type(be_f32_ds) === SparseMatrixCSC{Float32,Int}
		@test be_f32_ds isa Backend{Vector{Float32},SparseMatrixCSC{Float32,Int}}

		# Test custom Float64 Backend (Dense-Dense)
		be_f64_dd = Backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})
		@test vector_type(be_f64_dd) === Vector{Float64}
		@test matrix_type(be_f64_dd) === Matrix{Float64}
		@test be_f64_dd isa Backend{Vector{Float64},Matrix{Float64}}

		# Test custom Float64 Backend (Sparse-Sparse)
		be_f64_ss = Backend(vector_type = SparseVector{Float64,Int}, matrix_type = SparseMatrixCSC{Float64,Int})
		@test vector_type(be_f64_ss) === SparseVector{Float64,Int}
		@test matrix_type(be_f64_ss) === SparseMatrixCSC{Float64,Int}
		@test be_f64_ss isa Backend{SparseVector{Float64,Int},SparseMatrixCSC{Float64,Int}}

		# Test custom Complex Backend (Dense-Dense)
		be_c64_dd = Backend(vector_type = Vector{ComplexF64}, matrix_type = Matrix{ComplexF64})
		@test vector_type(be_c64_dd) === Vector{ComplexF64}
		@test matrix_type(be_c64_dd) === Matrix{ComplexF64}
		@test be_c64_dd isa Backend{Vector{ComplexF64},Matrix{ComplexF64}}
	end

	@testset "vector" begin
		n = 15

		# Default Backend (Vector{Float64}) - uses T(undef, n)
		be_default = Backend()
		v_default = vector(be_default, n)
		@test v_default isa Vector{Float64}
		@test length(v_default) == n
		@test eltype(v_default) === Float64

		# Sparse Backend (SparseVector{Float64, Int}) - uses T(n)
		be_sparse = Backend(vector_type = SparseVector{Float64,Int}, matrix_type = SparseMatrixCSC{Float64,Int})
		v_sparse = vector(be_sparse, n)
		@test v_sparse isa SparseVector{Float64,Int}
		@test length(v_sparse) == n
		@test eltype(v_sparse) === Float64
		@test nnz(v_sparse) == 0 # SparseVector(n) creates an empty vector

		# Dense Float32 Backend
		be_f32 = Backend(vector_type = Vector{Float32}, matrix_type = Matrix{Float32})
		v_f32 = vector(be_f32, n)
		@test v_f32 isa Vector{Float32}
		@test length(v_f32) == n
		@test eltype(v_f32) === Float32

		# Zero length vector
		v_zero = vector(be_default, 0)
		@test v_zero isa Vector{Float64}
		@test length(v_zero) == 0

		v_zero_sparse = vector(be_sparse, 0)
		@test v_zero_sparse isa SparseVector{Float64,Int}
		@test length(v_zero_sparse) == 0
	end

	@testset "matrix" begin
		m, n = 10, 20

		# Default Backend (SparseMatrixCSC{Float64, Int}) - uses T(m, n)
		be_default = Backend()
		M_default = matrix(be_default, m, n)
		@test M_default isa SparseMatrixCSC{Float64,Int}
		@test size(M_default) == (m, n)
		@test eltype(M_default) === Float64
		@test nnz(M_default) == 0 # SparseMatrixCSC(m, n) creates an empty matrix

		# Dense Backend (Matrix{Float64}) - uses T(undef, m, n)
		be_dense = Backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})
		M_dense = matrix(be_dense, m, n)
		@test M_dense isa Matrix{Float64}
		@test size(M_dense) == (m, n)
		@test eltype(M_dense) === Float64

		# Sparse Float32 Backend
		be_f32_sparse = Backend(vector_type = SparseVector{Float32,Int32}, matrix_type = SparseMatrixCSC{Float32,Int32})
		M_f32 = matrix(be_f32_sparse, m, n)
		@test M_f32 isa SparseMatrixCSC{Float32,Int32}
		@test size(M_f32) == (m, n)
		@test eltype(M_f32) === Float32

		# Zero dimensions
		M_zero_row = matrix(be_default, 0, n)
		@test M_zero_row isa SparseMatrixCSC{Float64,Int}
		@test size(M_zero_row) == (0, n)

		M_zero_col = matrix(be_default, m, 0)
		@test M_zero_col isa SparseMatrixCSC{Float64,Int}
		@test size(M_zero_col) == (m, 0)

		M_zero_all = matrix(be_default, 0, 0)
		@test M_zero_all isa SparseMatrixCSC{Float64,Int}
		@test size(M_zero_all) == (0, 0)

		M_zero_row_dense = matrix(be_dense, 0, n)
		@test M_zero_row_dense isa Matrix{Float64}
		@test size(M_zero_row_dense) == (0, n)

		M_zero_col_dense = matrix(be_dense, m, 0)
		@test M_zero_col_dense isa Matrix{Float64}
		@test size(M_zero_col_dense) == (m, 0)
	end
end # Top level testset
