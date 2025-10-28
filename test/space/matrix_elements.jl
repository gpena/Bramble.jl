import Bramble: MatrixElement, elements, space, eltype
import SparseArrays: SparseMatrixCSC, issparse, sprand, sparse
import LinearAlgebra: Diagonal

@testset "MatrixElement Tests" begin
	# Setup a mock space
	dofs = 4

	W = gridspace(mesh(domain(box(0, 1)), dofs, true))
	T = Float64

	MockSpace = typeof(W)
	MockBackend = typeof(backend(W))
	MatrixType = MatrixElement{MockSpace,T,SparseMatrixCSC{T,Int64}}

	A = sprand(T, dofs, dofs, 0.5)
	B = sprand(T, dofs, dofs, 0.5)
	v = rand(T, dofs)

	@testset "Constructors" begin
		# Test identity constructor
		I_el = elements(W)
		@test I_el isa MatrixElement
		@test space(I_el) === W
		@test eltype(I_el) == T

		# Test constructor with a given sparse matrix
		A_el = elements(W, A)
		@test A_el isa MatrixElement
		@test A_el.data == A
		@test space(A_el) === W
		@test eltype(A_el) == T
	end

	@testset "elements" begin
		U = elements(W)
		@test typeof(U) == MatrixType
		@test size(U.data) == (dofs, dofs)
		@test issparse(U.data)

		U_from_A = elements(W, A)
		@test typeof(U_from_A) == MatrixType
		@test U_from_A == A
	end

	@testset "properties and basic ops" begin
		U = elements(W)
		@test eltype(U) == T
		@test eltype(typeof(U)) == T
		@test length(U) == dofs*dofs
		@test space(U) == W
		@test size(U) == (dofs, dofs)

		Vₕ = similar(U)
		@test typeof(Vₕ) == typeof(U)
		@test size(Vₕ) == size(U)

		A = sprand(dofs, dofs, 0.5)
		U = elements(W, A)
		Vₕ = elements(W, sprand(dofs, dofs, 0.5))
		copyto!(U, Vₕ)
		@test U.data == Vₕ.data

		@testset "indexing" begin
			U[1] = 10.0
			@test U[1] == 10.0
			U[1, 1] = 20.0
			@test U[1, 1] == 20.0
			U[(1, 1)] = 30.0
			@test U[(1, 1)] == 30.0
			@test firstindex(U) == 1
			@test lastindex(U) == dofs * dofs
			@test axes(U) == (Base.OneTo(dofs), Base.OneTo(dofs))
		end
	end

	@testset "Core Operations" begin
		A_el = elements(W, A)

		# Test similar
		S_el = similar(A_el)
		@test S_el isa MatrixElement
		@test size(S_el) == size(A_el)
		@test space(S_el) === space(A_el)
		@test eltype(S_el) == eltype(A_el)

		# Test copyto!
		B_el = elements(W, B)
		copyto!(B_el, A_el)
		@test B_el.data == A_el.data
		# Ensure it's a deep copy of data, not a view
		B_el.data[1, 1] = 999.0
		@test A_el.data[1, 1] != 999.0
	end

	@testset "Indexing and Iteration" begin
		A_el = elements(W, copy(A))

		# Test getindex
		@test A_el[1, 1] == A[1, 1]
		@test A_el[end] == A[end]
		@test A_el[2, 3] == A[2, 3]

		# Test setindex!
		A_el[1, 2] = 123.45
		@test A_el[1, 2] == 123.45
		A_el[dofs*dofs] = 543.21
		@test A_el[dofs, dofs] == 543.21

		# Test index properties
		@test firstindex(A_el) == firstindex(A)
		@test lastindex(A_el) == lastindex(A)
		@test axes(A_el) == axes(A)

		# Test iteration
		@test sum(A_el) ≈ sum(A_el.data)
		collected_vals = [x for x in A_el]
		@test collected_vals == collect(A_el.data)
	end

	@testset "Binary Operators (Element-wise)" begin
		A_el = elements(W, A)
		B_el = elements(W, B)

		# Test addition
		C_el = A_el + B_el
		@test C_el == A + B
		@test C_el isa MatrixElement
		# Test subtraction
		C_el = A_el - B_el
		@test C_el == A - B
		@test C_el isa MatrixElement

		# Test multiplication
		C_el = A_el * B_el
		@test C_el == A * B
		@test C_el isa MatrixElement
	end

	@testset "Scalar-Matrix Operators" begin
		A_el = elements(W, A)
		α = 2.5

		# Test addition
		C = similar(A)
		C .= A
		C .= α .+ A

		@test (α + A_el).data.nzval == C.nzval
		@test (A_el + α).data.nzval == C.nzval

		# Test subtraction
		C .= α .- A
		@test (α - A_el).data.nzval == C.nzval
		C .= A .- α
		@test (A_el - α).data.nzval == C.nzval

		# Test multiplication
		C .= A
		C .= α .* A
		@test (α * A_el).data.nzval == C.nzval
		@test (A_el * α).data.nzval == C.nzval

		# Test division
		C .= A
		C = A ./ α
		@test (A_el / α).data.nzval ≈ C.nzval
	end

	@testset "Power Operator" begin
		A_el = elements(W, A)

		# Integer power
		C_el = A_el ^ 2
		@test C_el.data.nzval == A.nzval .^ 2

		# Real power
		C_el = A_el ^ 2.5
		@test C_el.data.nzval == A.nzval .^ 2.5
	end

	@testset "Vector-Matrix Multiplication" begin
		A_el = elements(W, A)
		v_el = element(W, v)

		# Test u * V
		Z_el = v_el * A_el
		@test Z_el.data ≈ Diagonal(v) * A
		@test Z_el isa MatrixElement

		# Test U * v
		Z_el = A_el * v_el
		@test Z_el.data ≈ A * Diagonal(v)
		@test Z_el isa MatrixElement

		# Test tuple versions
		#=
		v_tup = (v_el,)
		A_tup = (A_el, A_el)

		@test (v_tup * A_el).data == (v_el * A_el).data
		@test (A_el * v_tup).data == (A_el * v_el).data

		Z_tup = v_el * A_tup
		@test Z_tup isa NTuple{2,MatrixElement}
		@test Z_tup[1].data ≈ Diagonal(v) * A
		@test Z_tup[2].data ≈ Diagonal(v) * A

		# Test dot product on tuples
		v_tup_2 = (v_el, v_el)
		A_tup_2 = (A_el, elements(W, B))
		dot_prod = v_tup_2 ⋅ A_tup_2

		@test dot_prod isa NTuple{2,MatrixElement}
		@test dot_prod[1].data ≈ Diagonal(v) * A
		@test dot_prod[2].data ≈ Diagonal(v) * B
		=#
	end
end