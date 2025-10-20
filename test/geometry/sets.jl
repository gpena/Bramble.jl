using Bramble: CartesianProduct
using Bramble: interval, cartesian_product, projection, dim, tails, ×

@testset "CartesianProduct Tests" begin
	@testset "Constructors" begin
		# interval constructor (Float64 default)
		I_f64 = interval(-3.0, 10.0)
		@test I_f64 isa CartesianProduct{1,Float64}
		@test I_f64.box === ((-3.0, 10.0),)
		@test all(isapprox.(I_f64.box[1], (-3.0, 10.0)))
		# interval constructor (Int -> Float64)
		I_int = interval(-3, 10)
		@test I_int isa CartesianProduct{1,Float64} # Converts to float
		@test I_int.box === ((-3.0, 10.0),)
		@test all(isapprox.(I_int.box[1], (-3.0, 10.0)))
		# interval edge case: zero width
		I_zero = interval(5.5, 5.5)
		@test I_zero isa CartesianProduct{1,Float64}
		@test all(isapprox.(I_zero.box[1], (5.5, 5.5)))
		# interval constructor assertion x <= y
		@test_throws AssertionError interval(10, 1)

		# interval from CartesianProduct{1}
		I_f64_again = interval(I_f64)
		@test I_f64_again isa CartesianProduct{1,Float64}
		@test all(isapprox.(I_f64_again.box[1], (-3.0, 10.0)))
		# cartesian_product(x, y) alias
		cp_f64 = cartesian_product(-3.0, 10.0)
		@test cp_f64 isa CartesianProduct{1,Float64}
		@test all(isapprox.(cp_f64.box[1], (-3.0, 10.0)))
		# cartesian_product(NTuple) - Int
		cp_int_2d = cartesian_product(((0, 1), (4, 5)))
		@test cp_int_2d isa CartesianProduct{2,Float64}
		@test cp_int_2d.box === ((0.0, 1.0), (4.0, 5.0))

		# cartesian_product(NTuple) - Float32
		cp_f32_3d = cartesian_product(((0.0f0, 1.0f0), (2.0f0, 3.0f0), (-1.0f0, 0.0f0)))
		@test cp_f32_3d isa CartesianProduct{3,Float32}
		@test cp_f32_3d.box === ((0.0f0, 1.0f0), (2.0f0, 3.0f0), (-1.0f0, 0.0f0))

		# cartesian_product(NTuple) assertion min <= max
		@test_throws AssertionError cartesian_product(((0, 1), (5, 4)))

		# cartesian_product(CartesianProduct) identity
		cp_id = cartesian_product(cp_int_2d)
		@test cp_id === cp_int_2d # Should be the same object
	end

	@testset "Accessors and Properties" begin
		I = interval(0.0, 1.0)
		R2 = cartesian_product(((0, 1), (2, 3)))
		R3 = I × interval(2.0, 3.0) × interval(4.0, 5.0)

		# eltype
		@test eltype(I) === Float64
		@test eltype(typeof(I)) === Float64
		@test eltype(R2) === Float64
		@test eltype(typeof(R2)) === Float64
		@test eltype(R3) === Float64
		@test eltype(typeof(R3)) === Float64

		# dim
		@test dim(I) === 1
		@test dim(typeof(I)) === 1
		@test dim(R2) === 2
		@test dim(typeof(R2)) === 2
		@test dim(R3) === 3
		@test dim(typeof(R3)) === 3

		# Call syntax (X(i))
		@test all(isapprox.(I(1), (0.0, 1.0)))
		@test all(isapprox.(R2(1), (0, 1)))
		@test all(isapprox.(R2(2), (2, 3)))
		@test all(isapprox.(R3(1), (0.0, 1.0)))
		@test all(isapprox.(R3(2), (2.0, 3.0)))
		@test all(isapprox.(R3(3), (4.0, 5.0)))
		@test_throws AssertionError I(2)
		@test_throws AssertionError R2(0)
		@test_throws AssertionError R3(4)

		# tails(X, i)
		@test all(isapprox.(tails(I, 1), (0.0, 1.0)))
		@test all(isapprox.(tails(R2, 1), (0, 1)))
		@test all(isapprox.(tails(R2, 2), (2, 3)))
		@test all(isapprox.(tails(R3, 3), (4.0, 5.0)))
		@test_throws AssertionError tails(I, 2)
		@test_throws AssertionError tails(R2, 0)
		@test_throws AssertionError tails(R3, 4)

		# tails(X)
		@test all(isapprox.(tails(I), (0.0, 1.0)))
		@test tails(R2) === ((0.0, 1.0), (2.0, 3.0))
		@test all(tails(R3) .== ((0.0, 1.0), (2.0, 3.0), (4.0, 5.0)))
		# first/last (only for D=1)
		@test all(isapprox.(first(I), 0.0))
		@test all(isapprox.(last(I), 1.0))
		@test_throws MethodError first(R2)
		@test_throws MethodError last(R3)
	end

	@testset "Operations" begin
		I1 = interval(0.0, 1.0)
		I2 = interval(2.0, 3.0)
		I3_int = interval(4, 5) # Creates Float64 product
		R2 = cartesian_product(((10, 11), (12, 13)))

		# × operator (Float64 x Float64)
		P1 = I1 × I2
		@test P1 isa CartesianProduct{2,Float64}
		@test dim(P1) == 2
		@test all(tails(P1) .== ((0.0, 1.0), (2.0, 3.0)))
		# × operator (Float64 x Float64 x Float64)
		P2 = I1 × I2 × I3_int
		@test P2 isa CartesianProduct{3,Float64}
		@test dim(P2) == 3
		@test all(tails(P2) .== ((0.0, 1.0), (2.0, 3.0), (4.0, 5.0)))

		# projection
		P_proj = I1 × I2 × I3_int # Dim 3, Float64
		proj1 = projection(P_proj, 1)
		proj2 = projection(P_proj, 2)
		proj3 = projection(P_proj, 3)

		@test proj1 isa CartesianProduct{1,Float64}
		@test dim(proj1) == 1
		@test all(isapprox.(tails(proj1), (0.0, 1.0)))
		@test all(isapprox.(first(proj1), 0.0))
		@test all(isapprox.(last(proj1), 1.0))
		@test proj2 isa CartesianProduct{1,Float64}
		@test dim(proj2) == 1
		@test all(isapprox.(tails(proj2), (2.0, 3.0)))
		@test proj3 isa CartesianProduct{1,Float64}
		@test dim(proj3) == 1
		@test all(isapprox.(tails(proj3), (4.0, 5.0)))
		# projection index out of bounds
		@test_throws AssertionError projection(P_proj, 4)
		@test_throws AssertionError projection(P_proj, 0)
	end

	# Include original tests for regression checking
	@testset "Original Tests" begin
		I = interval(-3.0, 10.0)

		@test all(isapprox.(I.box[1][1], -3.0))
		@test all(isapprox.(I.box[1][2], 10.0))
		@test dim(I) == 1

		I2 = interval(70.0, 100.0)

		# Using list comprehension for combinations was fine, but direct construction is clearer
		set_2d = I × I2
		Ω2_x = projection(set_2d, 1)
		Ω2_y = projection(set_2d, 2)

		# Note: projection returns CartesianProduct{1, Float64}
		@test all(isapprox.(Ω2_y.box[1][1], 70.0))
		@test all(isapprox.(Ω2_x.box[1][1], -3.0))
		@test all(isapprox.(Ω2_y.box[1][2], 100.0))
		@test all(isapprox.(Ω2_x.box[1][2], 10.0))
		I3 = interval(-15.0, -1.0)

		set_3d = I × I2 × I3
		Ω3_x = projection(set_3d, 1)
		Ω3_y = projection(set_3d, 2)
		Ω3_z = projection(set_3d, 3)

		@test all(isapprox.(Ω3_x.box[1][1], -3.0))
		@test all(isapprox.(Ω3_x.box[1][2], 10.0))
		@test all(isapprox.(Ω3_y.box[1][1], 70.0))
		@test all(isapprox.(Ω3_y.box[1][2], 100.0))
		@test all(isapprox.(Ω3_z.box[1][1], -15.0))
		@test all(isapprox.(Ω3_z.box[1][2], -1.0))
	end
end # End CartesianProduct Tests