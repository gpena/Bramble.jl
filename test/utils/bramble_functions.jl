
using Bramble
using Bramble: BrambleFunction
using Bramble: interval, ×, cartesian_product, embed_function, argstype, codomaintype

begin
	# --- Test Setup ---
	# Use the actual interval() and × provided
	Ω1 = interval(0.0, 1.0) # CartesianProduct{1, Float64}
	Ω2 = interval(0.0, 1.0) × interval(10.0, 20.0) # CartesianProduct{2, Float64}
	# Use cartesian_product for Float32 if interval promotes to Float64
	Ω3_box = ((0.0f0, 1.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0))
	Ω3 = cartesian_product(Ω3_box) # CartesianProduct{3, Float32}

	I = interval(0.0, 2.0) # Time interval, CartesianProduct{1, Float64}

	# Test Functions (same as before)
	f1(x) = 2.0 * x
	f2(x) = x[1] + x[2]^2
	f3(x::NTuple{3,Float32}) = x[1] * x[2] - x[3]

	f1t(x, t) = (1.0 + t) * x
	f2t(x, t) = t * (x[1] - x[2])

	# --- Tests (mostly unchanged, but now using real CartesianProduct) ---
	@testset "BrambleFunction Tests with Real CartesianProduct" begin
		@testset "Embed Non-Time-Dependent" begin
			@testset "1D Domain" begin
				# Test CoType inference improvement
				bf1_func = embed_function(Ω1, f1) # Returns Float64

				@test bf1_func isa BrambleFunction{Float64,false,Float64,typeof(Ω1)}

				@test bf1_func(0.2) ≈ 0.4
				@test bf1_func(0.0f0) ≈ 0.0 # Test Float32 input conversion

				@test argstype(bf1_func.wrapped) == Float64
				@test codomaintype(bf1_func.wrapped) == Float64
			end

			@testset "2D Domain" begin
				bf2_func = embed_function(Ω2, f2) # Returns Float64

				@test bf2_func isa BrambleFunction{NTuple{2,Float64},false,Float64,typeof(Ω2)}

				# Test points (note Ω2 y-range is [10, 20])
				pt1 = (0.5, 10.5)
				pt2 = (0.1, 10.2)
				pt3 = (1, 12.0) # Mixed Int/Float

				@test bf2_func(pt1) ≈ 0.5 + 10.5^2
				@test bf2_func(pt2...) ≈ 0.1 + 10.2^2 # Varargs call
				@test bf2_func(1, 12) ≈ 1.0 + 12.0^2 # Test conversion in varargs call

				@test argstype(bf2_func.wrapped) == NTuple{2,Float64}
				@test codomaintype(bf2_func.wrapped) == Float64
			end

			@testset "3D Domain (Float32)" begin
				bf3_func = embed_function(Ω3, f3) # Returns Float32

				@test bf3_func isa BrambleFunction{NTuple{3,Float32},false,Float32,typeof(Ω3)}

				pt1_f32 = (0.5f0, 0.5f0, 0.1f0)
				pt2_f64 = (0.1, 0.2, 0.3) # Test conversion

				@test bf3_func(pt1_f32...) ≈ 0.5f0 * 0.5f0 - 0.1f0
				# Need isapprox for Float32 comparisons after conversion
				@test bf3_func(pt2_f64...) ≈ (0.1f0 * 0.2f0 - 0.3f0)

				@test argstype(bf3_func.wrapped) == NTuple{3,Float32}
				@test codomaintype(bf3_func.wrapped) == Float32
			end

			@testset "Function Call Syntax (Non-Time)" begin
				# Use Ω2 which is 2D, expecting NTuple{2, Float64}
				bf2 = embed_function(Ω2, x->x[1] + x[2])
				@test bf2((1.0, 12.0)) ≈ 13.0
				@test bf2(1.0, 12.0) ≈ 13.0
				@test bf2((1, 12)) ≈ 13.0 # Conversion tuple
				@test bf2(1, 12) ≈ 13.0   # Conversion varargs

				# Use Ω1 which is 1D, expecting Float64
				bf1 = embed_function(Ω1, x->2x)
				@test bf1(0.5) ≈ 1.0
				@test bf1(1) ≈ 2.0 # Conversion scalar
				# Test edge cases - depending on exact call method definition
				# @test_throws MethodError bf1((0.5,)) # Tuple might cause method error if only scalar method exists
			end
		end # End Embed Non-Time-Dependent

		@testset "Embed Time-Dependent" begin
			@testset "1D Space + Time" begin
				bf1t_func = embed_function(Ω1, I, f1t) # Use the 3-arg version

				ExpectedInnerCoType = BrambleFunction{Float64,false,Float64,typeof(Ω1)}
				ExpectedOuterType = BrambleFunction{Float64,true,ExpectedInnerCoType,typeof(I)}

				@test bf1t_func isa ExpectedOuterType

				t2 = 1.0 # Within I=[0,2]

				bf1_at_t2 = bf1t_func(t2)

				@test bf1_at_t2 isa ExpectedInnerCoType

				x_val = 0.5 # Within Ω1=[0,1]
				@test bf1_at_t2(x_val) ≈ (1.0 + t2) * x_val

				@test argstype(bf1t_func.wrapped) == Float64 # Time type
				@test codomaintype(bf1t_func.wrapped) == ExpectedInnerCoType
			end

			@testset "2D Space + Time" begin
				bf2t_func = embed_function(Ω2, I, f2t)

				ExpectedInnerCoType = BrambleFunction{NTuple{2,Float64},false,Float64,typeof(Ω2)}
				ExpectedOuterType = BrambleFunction{Float64,true,ExpectedInnerCoType,typeof(I)}

				@test bf2t_func isa ExpectedOuterType

				t_val = 1.5 # Within I=[0,2]

				bf2_at_t = bf2t_func(t_val)
				@test bf2_at_t isa ExpectedInnerCoType

				pt1 = (0.5, 11.0) # Within Ω2=[0,1]x[10,20]
				pt2 = (0, 20) # Test conversion, edge of domain

				@test bf2_at_t(pt1) ≈ t_val * (pt1[1] - pt1[2])
				@test bf2_at_t(pt1...) ≈ t_val * (pt1[1] - pt1[2]) # Varargs
				@test bf2_at_t(pt2) ≈ t_val * (0.0 - 20.0) # Tuple conversion
				@test bf2_at_t(pt2...) ≈ t_val * (0.0 - 20.0) # Varargs conversion

				@test argstype(bf2t_func.wrapped) == Float64 # Time type
				@test codomaintype(bf2t_func.wrapped) == ExpectedInnerCoType
			end
		end # End Embed Time-Dependent

		@testset "has_time" begin
			using Bramble: has_time

			bf_notime = embed_function(Ω1, f1)
			@test has_time(bf_notime) == false
			@test has_time(typeof(bf_notime)) == false

			bf_withtime = embed_function(Ω1, I, f1t)
			@test has_time(bf_withtime) == true
			@test has_time(typeof(bf_withtime)) == true
		end

		@testset "argstype and codomaintype" begin
			bf1 = embed_function(Ω1, f1)
			@test argstype(bf1.wrapped) == Float64
			@test codomaintype(bf1.wrapped) == Float64

			bf2 = embed_function(Ω2, f2)
			@test argstype(bf2.wrapped) == NTuple{2,Float64}
			@test codomaintype(bf2.wrapped) == Float64

			bf3 = embed_function(Ω3, f3)
			@test argstype(bf3.wrapped) == NTuple{3,Float32}
			@test codomaintype(bf3.wrapped) == Float32
		end

		@testset "Edge Cases and Type Conversions" begin
			# Test with SVector inputs
			using StaticArrays
			bf2 = embed_function(Ω2, f2)
			sv = SVector(0.5, 12.0)
			@test bf2(sv) ≈ 0.5 + 12.0^2

			# Test identity function
			identity_bf = embed_function(Ω1, identity)
			@test identity_bf(0.7) ≈ 0.7

			# Test constant function
			const_func = x -> 42.0
			const_bf = embed_function(Ω1, const_func)
			@test const_bf(0.1) ≈ 42.0
			@test const_bf(0.9) ≈ 42.0

			# Test zero function
			zero_func = x -> 0.0
			zero_bf = embed_function(Ω1, zero_func)
			@test zero_bf(0.5) ≈ 0.0

			# Test 2D constant
			const_2d = x -> 100.0
			const_2d_bf = embed_function(Ω2, const_2d)
			@test const_2d_bf((0.5, 15.0)) ≈ 100.0

			# Test with integer arithmetic conversion
			int_func = x -> Int(round(10 * x))
			float_bf = embed_function(Ω1, int_func)
			@test float_bf(0.3) == 3
		end

		@testset "embed_function with BrambleFunction input" begin
			# Test that embed_function returns same instance for BrambleFunction input
			bf1 = embed_function(Ω1, f1)
			bf1_again = embed_function(Ω1, bf1)
			@test bf1_again === bf1
		end

		@testset "Time-Dependent Edge Cases" begin
			# Test with constant time function
			const_time = (x, t) -> x + 5.0
			bf_const_time = embed_function(Ω1, I, const_time)
			bf_at_t0 = bf_const_time(0.0)
			bf_at_t1 = bf_const_time(1.0)
			@test bf_at_t0(0.5) ≈ 5.5
			@test bf_at_t1(0.5) ≈ 5.5  # Same result regardless of t

			# Test with time-only dependence
			time_only = (x, t) -> t^2
			bf_time_only = embed_function(Ω1, I, time_only)
			bf_at_t05 = bf_time_only(0.5)
			@test bf_at_t05(0.0) ≈ 0.25
			@test bf_at_t05(1.0) ≈ 0.25  # Same result regardless of x

			# Test separable function
			separable = (x, t) -> x * t
			bf_sep = embed_function(Ω1, I, separable)
			bf_sep_t2 = bf_sep(2.0)
			@test bf_sep_t2(0.5) ≈ 1.0
			@test bf_sep_t2(0.25) ≈ 0.5
		end

		@testset "Complex Functions" begin
			# Test with more complex mathematical operations
			complex_func_1d = x -> sin(π * x) + cos(2π * x)
			bf_complex = embed_function(Ω1, complex_func_1d)
			@test bf_complex(0.0) ≈ sin(0) + cos(0) ≈ 1.0
			@test bf_complex(0.5) ≈ sin(π * 0.5) + cos(π) ≈ 0.0

			# Test 2D complex function
			complex_func_2d = x -> sqrt(x[1]^2 + x[2]^2)  # Distance from origin
			bf_complex_2d = embed_function(Ω2, complex_func_2d)
			@test bf_complex_2d((0.0, 10.0)) ≈ 10.0
			@test bf_complex_2d((1.0, 10.0)) ≈ sqrt(1.0 + 100.0)
		end
	end # End Testset
end