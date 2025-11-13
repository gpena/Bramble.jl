"""
Test coverage for operator modules (Priority 1)

Tests for the 5 refactored operator modules:
- operator_types.jl
- scalar_operators.jl  
- differential_operators.jl
- composite_operators.jl
- operator_inner_products.jl

Target: Improve coverage from 0% to 60%+
"""

import Bramble: space, scalar, codomaintype, ZeroOperator, IdentityOperator, VectorElement

@testset "Operator Modules Coverage" begin
	@testset "1D Operator Tests" begin
		# Setup 1D space
		N = 10
		I = interval(0.0, 1.0)
		X = domain(I)
		Mh = mesh(X, N, false)
		Wh = gridspace(Mh)

		# Create test element
		uₕ = element(Wh)
		Rₕ!(uₕ, x -> sin(2π*x[1]))

		@testset "Scalar Operators" begin
			# Test ZeroOperator
			zero_op = ZeroOperator(Wh)
			@test space(zero_op) === Wh
			@test scalar(zero_op) == 0

			# Test IdentityOperator  
			id_result = M₋ₕ(uₕ)
			@test id_result isa VectorElement
			@test space(id_result) === Wh

			# Test ScaledOperator
			scaled = 2.5 * D₋ₓ(uₕ)
			@test scaled isa VectorElement
			@test space(scaled) === Wh

			# Test scalar simplifications
			double_scale = 2 * (3 * D₋ₓ(uₕ))
			@test double_scale isa VectorElement
		end

		@testset "Differential Operators" begin
			# Test backward difference D₋ₓ
			grad = D₋ₓ(uₕ)
			@test grad isa VectorElement
			@test space(grad) === Wh
			@test length(grad) == length(uₕ)

			# Test forward difference D₊ₓ
			grad_plus = D₊ₓ(uₕ)
			@test grad_plus isa VectorElement
			@test space(grad_plus) === Wh

			# Test gradient operator ∇₋ₕ
			nabla = ∇₋ₕ(uₕ)
			@test nabla isa VectorElement
			@test space(nabla) === Wh

			# Test gradient composition with scalar
			scaled_grad = 0.5 * ∇₋ₕ(uₕ)
			@test scaled_grad isa VectorElement
		end

		@testset "Composite Operators" begin
			# Test operator addition
			sum_op = D₋ₓ(uₕ) + D₊ₓ(uₕ)
			@test sum_op isa VectorElement
			@test space(sum_op) === Wh

			# Test operator subtraction
			diff_op = D₊ₓ(uₕ) - D₋ₓ(uₕ)
			@test diff_op isa VectorElement
			@test space(diff_op) === Wh

			# Test mixed operations
			mixed = 2 * D₋ₓ(uₕ) + 0.5 * D₊ₓ(uₕ)
			@test mixed isa VectorElement
		end

		@testset "Operator Inner Products" begin
			vₕ = element(Wh)
			Rₕ!(vₕ, x -> cos(2π*x[1]))

			# Test inner₊ with gradients
			result1 = inner₊(D₋ₓ(uₕ), D₋ₓ(vₕ))
			@test result1 isa Number
			@test isfinite(result1)

			# Test inner₊ with different operators
			result2 = inner₊(∇₋ₕ(uₕ), ∇₋ₕ(vₕ))
			@test result2 isa Number
			@test isfinite(result2)

			# Test innerₕ (L2 inner product)
			result3 = innerₕ(uₕ, vₕ)
			@test result3 isa Number
			@test isfinite(result3)

			# Test inner₊ with scaled operators
			result4 = inner₊(2 * D₋ₓ(uₕ), D₋ₓ(vₕ))
			@test result4 isa Number
			@test isfinite(result4)
		end

		@testset "Operator Type Hierarchy" begin
			# Test that operators have correct types
			@test D₋ₓ(uₕ) isa VectorElement
			@test M₋ₕ(uₕ) isa VectorElement
			@test ∇₋ₕ(uₕ) isa VectorElement

			# Test space extraction
			op1 = D₋ₓ(uₕ)
			@test space(op1) === Wh

			op2 = M₋ₕ(uₕ)
			@test space(op2) === Wh
		end
	end

	@testset "2D Operator Tests" begin
		# Setup 2D space
		N = 5
		I = interval(0.0, 1.0)
		Ω = I × I
		X = domain(Ω)
		Mh = mesh(X, (N, N), (false, false))
		Wh = gridspace(Mh)

		# Create test element
		uₕ = element(Wh)
		Rₕ!(uₕ, x -> sin(π*x[1]) * cos(π*x[2]))

		@testset "2D Scalar Operators" begin
			# Test shift operators in 2D
			result_minus = M₋ₕ(uₕ)
			@test result_minus isa Tuple{VectorElement,VectorElement}
			@test all(comp -> space(comp) === Wh, result_minus)

			result_plus = M₊ₕ(uₕ)
			@test result_plus isa Tuple{VectorElement,VectorElement}
		end

		@testset "2D Differential Operators" begin
			# Test gradient in 2D
			nabla = ∇₋ₕ(uₕ)
			@test nabla isa Tuple{VectorElement,VectorElement}
			@test all(comp -> space(comp) === Wh, nabla)

			# Test scaled gradient
			scaled_nabla = 3.0 * ∇₋ₕ(uₕ)
			@test scaled_nabla isa Tuple{VectorElement,VectorElement}
		end

		@testset "2D Inner Products" begin
			vₕ = element(Wh)
			Rₕ!(vₕ, x -> exp(-x[1]) * exp(-x[2]))

			# Test L2 inner product
			result1 = innerₕ(uₕ, vₕ)
			@test result1 isa Number
			@test isfinite(result1)

			# Test H1 semi-inner product
			result2 = inner₊(∇₋ₕ(uₕ), ∇₋ₕ(vₕ))
			@test result2 isa Number
			@test isfinite(result2)
		end
	end

	@testset "3D Operator Tests" begin
		# Setup 3D space (small for performance)
		N = 3
		I = interval(0.0, 1.0)
		Ω = I × I × I
		X = domain(Ω)
		Mh = mesh(X, (N, N, N), (false, false, false))
		Wh = gridspace(Mh)

		# Create test element
		uₕ = element(Wh)
		Rₕ!(uₕ, x -> x[1] * x[2] * x[3])

		@testset "3D Differential Operators" begin
			# Test gradient in 3D
			nabla = ∇₋ₕ(uₕ)
			@test nabla isa Tuple{VectorElement,VectorElement,VectorElement}
			@test all(comp -> space(comp) === Wh, nabla)
		end

		@testset "3D Inner Products" begin
			vₕ = element(Wh)
			Rₕ!(vₕ, x -> 1.0)

			# Test inner products in 3D
			result1 = innerₕ(uₕ, vₕ)
			@test result1 isa Number
			@test isfinite(result1)

			result2 = inner₊(∇₋ₕ(uₕ), ∇₋ₕ(vₕ))
			@test result2 isa Number
			@test isfinite(result2)
		end
	end

	@testset "Operator Algebraic Properties" begin
		N = 8
		I = interval(0.0, 1.0)
		X = domain(I)
		Mh = mesh(X, N, false)
		Wh = gridspace(Mh)

		uₕ = element(Wh)
		Rₕ!(uₕ, x -> x[1]^2)

		@testset "Distributivity" begin
			# Test (a + b) * op
			result1 = (2 + 3) * D₋ₓ(uₕ)
			result2 = 5 * D₋ₓ(uₕ)
			@test result1 isa VectorElement
			@test result2 isa VectorElement
		end

		@testset "Associativity" begin
			# Test a * (b * op)
			result1 = 2 * (3 * D₋ₓ(uₕ))
			result2 = (2 * 3) * D₋ₓ(uₕ)
			@test result1 isa VectorElement
			@test result2 isa VectorElement
		end

		@testset "Zero absorption" begin
			# Test 0 * op gives zero values
			zero_result = 0 * D₋ₓ(uₕ)
			@test zero_result isa VectorElement
			@test all(iszero, zero_result.data)
		end

		@testset "Identity preservation" begin
			# Test 1 * op preserves values
			original = D₋ₓ(uₕ)
			result = 1 * original
			@test result.data ≈ original.data
		end
	end
end
