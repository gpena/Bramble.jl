
using Bramble
using Bramble: @embed
using Bramble: BrambleFunction
using Bramble: interval, ×, cartesianproduct, embed_function, argstype, codomaintype

# --- Test Setup ---
# Use the actual interval() and × provided
Ω1 = interval(0.0, 1.0) # CartesianProduct{1, Float64}
Ω2 = interval(0.0, 1.0) × interval(10.0, 20.0) # CartesianProduct{2, Float64}
# Use cartesianproduct for Float32 if interval promotes to Float64
Ω3_box = ((0.0f0, 1.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0))
Ω3 = cartesianproduct(Ω3_box) # CartesianProduct{3, Float32}

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
			bf1_macro = Bramble.@embed(Ω1, x->2.0 * x[1] + 1) # Returns Float64
			bf1_func = embed_function(Ω1, f1) # Returns Float64

			@test bf1_macro isa BrambleFunction{Float64,false,Float64}
			@test bf1_func isa BrambleFunction{Float64,false,Float64}

			@test bf1_macro(0.5) ≈ 2.0
			@test bf1_func(0.2) ≈ 0.4
			@test bf1_macro(1) ≈ 3.0 # Test integer input conversion (2*1+1)
			@test bf1_func(0.0f0) ≈ 0.0 # Test Float32 input conversion

			@test argstype(bf1_macro.wrapped) == Float64
			@test codomaintype(bf1_macro.wrapped) == Float64
		end

		@testset "2D Domain" begin
			bf2_macro = @embed(Ω2, x->x[1] + x[2]^2) # Returns Float64
			bf2_func = embed_function(Ω2, f2) # Returns Float64

			@test bf2_macro isa BrambleFunction{NTuple{2,Float64},false,Float64}
			@test bf2_func isa BrambleFunction{NTuple{2,Float64},false,Float64}

			# Test points (note Ω2 y-range is [10, 20])
			pt1 = (0.5, 10.5)
			pt2 = (0.1, 10.2)
			pt3 = (1, 12.0) # Mixed Int/Float

			@test bf2_macro(pt1) ≈ 0.5 + 10.5^2
			@test bf2_func(pt1) ≈ 0.5 + 10.5^2
			@test bf2_macro(pt2...) ≈ 0.1 + 10.2^2 # Varargs call
			@test bf2_func(pt2...) ≈ 0.1 + 10.2^2 # Varargs call

			@test bf2_macro(pt3) ≈ 1.0 + 12.0^2 # Test conversion in tuple call
			@test bf2_func(1, 12) ≈ 1.0 + 12.0^2 # Test conversion in varargs call

			@test argstype(bf2_macro.wrapped) == NTuple{2,Float64}
			@test codomaintype(bf2_macro.wrapped) == Float64
		end

		@testset "3D Domain (Float32)" begin
			bf3_macro = @embed(Ω3, f3) # Function symbol, returns Float32
			bf3_func = embed_function(Ω3, f3) # Returns Float32

			@test bf3_macro isa BrambleFunction{NTuple{3,Float32},false,Float32}
			@test bf3_func isa BrambleFunction{NTuple{3,Float32},false,Float32}

			pt1_f32 = (0.5f0, 0.5f0, 0.1f0)
			pt2_f64 = (0.1, 0.2, 0.3) # Test conversion

			@test bf3_macro(pt1_f32) ≈ 0.5f0 * 0.5f0 - 0.1f0
			@test bf3_func(pt1_f32...) ≈ 0.5f0 * 0.5f0 - 0.1f0
			# Need isapprox for Float32 comparisons after conversion
			@test bf3_macro(pt2_f64) ≈ (0.1f0 * 0.2f0 - 0.3f0)
			@test bf3_func(pt2_f64...) ≈ (0.1f0 * 0.2f0 - 0.3f0)

			@test argstype(bf3_macro.wrapped) == NTuple{3,Float32}
			@test codomaintype(bf3_macro.wrapped) == Float32
		end

		@testset "Function Call Syntax (Non-Time)" begin
			# Use Ω2 which is 2D, expecting NTuple{2, Float64}
			bf2 = @embed(Ω2, x->x[1] + x[2])
			@test bf2((1.0, 12.0)) ≈ 13.0
			@test bf2(1.0, 12.0) ≈ 13.0
			@test bf2((1, 12)) ≈ 13.0 # Conversion tuple
			@test bf2(1, 12) ≈ 13.0   # Conversion varargs

			# Use Ω1 which is 1D, expecting Float64
			bf1 = @embed(Ω1, x->2x)
			@test bf1(0.5) ≈ 1.0
			@test bf1(1) ≈ 2.0 # Conversion scalar
			# Test edge cases - depending on exact call method definition
			# @test_throws MethodError bf1((0.5,)) # Tuple might cause method error if only scalar method exists
		end
	end # End Embed Non-Time-Dependent

	@testset "Embed Time-Dependent" begin
		@testset "1D Space + Time" begin
			bf1t_macro = @embed(Ω1×I, (x, t)->(1.0 + t) * x[1])
			bf1t_func = embed_function(Ω1, I, f1t) # Use the 3-arg version

			ExpectedInnerCoType = BrambleFunction{Float64,false,Float64}
			ExpectedOuterType = BrambleFunction{Float64,true,ExpectedInnerCoType}

			@test bf1t_macro isa ExpectedOuterType
			@test bf1t_func isa ExpectedOuterType

			t1 = 0.5 # Within I=[0,2]
			t2 = 1.0 # Within I=[0,2]

			bf1_at_t1 = bf1t_macro(t1)
			bf1_at_t2 = bf1t_func(t2)

			@test bf1_at_t1 isa ExpectedInnerCoType
			@test bf1_at_t2 isa ExpectedInnerCoType

			x_val = 0.5 # Within Ω1=[0,1]
			@test bf1_at_t1(x_val) ≈ (1.0 + t1) * x_val
			@test bf1_at_t2(x_val) ≈ (1.0 + t2) * x_val
			@test bf1_at_t1(1) ≈ (1.0 + t1) * 1.0 # Test conversion in inner call

			@test argstype(bf1t_macro.wrapped) == Float64 # Time type
			@test codomaintype(bf1t_macro.wrapped) == ExpectedInnerCoType
		end

		@testset "2D Space + Time" begin
			bf2t_macro = @embed(Ω2×I, f2t) # Function symbol
			bf2t_func = embed_function(Ω2, I, f2t)

			ExpectedInnerCoType = BrambleFunction{NTuple{2,Float64},false,Float64}
			ExpectedOuterType = BrambleFunction{Float64,true,ExpectedInnerCoType}

			@test bf2t_macro isa ExpectedOuterType
			@test bf2t_func isa ExpectedOuterType

			t_val = 1.5 # Within I=[0,2]

			bf2_at_t = bf2t_macro(t_val)
			@test bf2_at_t isa ExpectedInnerCoType

			pt1 = (0.5, 11.0) # Within Ω2=[0,1]x[10,20]
			pt2 = (0, 20) # Test conversion, edge of domain

			@test bf2_at_t(pt1) ≈ t_val * (pt1[1] - pt1[2])
			@test bf2_at_t(pt1...) ≈ t_val * (pt1[1] - pt1[2]) # Varargs
			@test bf2_at_t(pt2) ≈ t_val * (0.0 - 20.0) # Tuple conversion
			@test bf2_at_t(pt2...) ≈ t_val * (0.0 - 20.0) # Varargs conversion

			@test argstype(bf2t_macro.wrapped) == Float64 # Time type
			@test codomaintype(bf2t_macro.wrapped) == ExpectedInnerCoType
		end
	end # End Embed Time-Dependent
end # End Testset
