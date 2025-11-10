using Bramble: CartesianProduct
using Bramble: interval, cartesian_product, projection, dim, tails, ×, point, set, eltype, topo_dim, center, is_collapsed, point_type, first, last
using StaticArrays

@testset "CartesianProduct Tests" begin
	@testset "Constructors" begin
		# interval constructor (Float64 default)
		I_f64 = interval(-3.0, 10.0)
		@test I_f64 isa CartesianProduct{1,Float64}
		@test I_f64.box isa SVector{1}  # Now SVector for D=1
		@test I_f64.box[1] == (-3.0, 10.0)
		@test all(isapprox.(I_f64.box[1], (-3.0, 10.0)))

		# Test collapsed flag
		@test I_f64.collapsed[1] == false

		# interval constructor (Int -> Float64)
		I_int = interval(-3, 10)
		@test I_int isa CartesianProduct{1,Float64}
		@test I_int.box[1] == (-3.0, 10.0)
		@test all(isapprox.(I_int.box[1], (-3.0, 10.0)))

		# interval edge case: zero width
		I_zero = interval(5.5, 5.5)
		@test I_zero isa CartesianProduct{1,Float64}
		@test all(isapprox.(I_zero.box[1], (5.5, 5.5)))
		@test I_zero.collapsed[1] == true  # Should be marked as collapsed

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
		@test cp_int_2d.box isa SVector{2}  # SVector for D=2
		@test cp_int_2d.box[1] == (0.0, 1.0)
		@test cp_int_2d.box[2] == (4.0, 5.0)

		# cartesian_product(NTuple) - Float32 
		cp_f32_3d = cartesian_product(((0.0f0, 1.0f0), (2.0f0, 3.0f0), (-1.0f0, 0.0f0)))
		@test cp_f32_3d isa CartesianProduct{3,Float32}
		@test cp_f32_3d.box isa SVector{3}  # SVector for D=3
		@test cp_f32_3d.box[1] == (0.0f0, 1.0f0)
		@test cp_f32_3d.box[2] == (2.0f0, 3.0f0)
		@test cp_f32_3d.box[3] == (-1.0f0, 0.0f0)

		# cartesian_product(CartesianProduct) identity
		cp_id = cartesian_product(cp_int_2d)
		@test cp_id === cp_int_2d

		# point constructor (collapsed CartesianProduct)
		P_f64 = point(3.5)
		@test P_f64 isa CartesianProduct{1,Float64}
		@test P_f64.box[1] == (3.5, 3.5)
		@test P_f64.collapsed[1] == true

		# box constructors
		B1d = box(1.0, 5.0)
		@test B1d isa CartesianProduct{1,Float64}
		@test B1d.box[1] == (1.0, 5.0)

		B2d = box((0.0, 2.0), (1.0, 3.0))
		@test B2d isa CartesianProduct{2,Float64}
		@test B2d.box[1] == (0.0, 1.0)  # min/max
		@test B2d.box[2] == (2.0, 3.0)

		# box with reversed points (should compute min/max correctly)
		B2d_rev = box((5.0, 10.0), (2.0, 8.0))
		@test B2d_rev.box[1] == (2.0, 5.0)
		@test B2d_rev.box[2] == (8.0, 10.0)
	end

	@testset "Accessors and Properties" begin
		I = interval(0.0, 1.0)
		R2 = cartesian_product(((0, 1), (2, 3)))
		R3 = I × interval(2.0, 3.0) × interval(4.0, 5.0)

		# set accessor
		@test set(I) === I
		@test set(R2) === R2

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

		# topo_dim (topological dimension)
		@test topo_dim(I) === 1
		@test topo_dim(R2) === 2
		@test topo_dim(R3) === 3

		# topo_dim with collapsed dimensions
		P_collapsed = point(1.0)
		@test topo_dim(P_collapsed) === 0  # Point has topological dim 0

		I_line = I × point(2.0)  # Line in 2D
		@test dim(I_line) === 2
		@test topo_dim(I_line) === 1  # Only 1 non-collapsed dimension

		# center
		@test center(I) ≈ SVector(0.5)
		@test center(R2) ≈ SVector(0.5, 2.5)
		@test center(R3) ≈ SVector(0.5, 2.5, 4.5)

		# is_collapsed
		@test is_collapsed(I) == false
		@test is_collapsed(P_collapsed) == true
		@test is_collapsed(1.0, 1.0) == true
		@test is_collapsed(0.0, 1.0) == false

		# point_type
		@test point_type(I) === Float64
		@test point_type(R2) === NTuple{2,Float64}
		@test point_type(R3) === NTuple{3,Float64}

		# Call syntax (X(i))
		@test all(isapprox.(I(1), (0.0, 1.0)))
		@test all(isapprox.(R2(1), (0.0, 1.0)))
		@test all(isapprox.(R2(2), (2.0, 3.0)))
		@test all(isapprox.(R3(1), (0.0, 1.0)))
		@test all(isapprox.(R3(2), (2.0, 3.0)))
		@test all(isapprox.(R3(3), (4.0, 5.0)))
		@test_throws AssertionError I(2)
		@test_throws AssertionError R2(0)
		@test_throws AssertionError R3(4)

		# tails(X, i)
		@test all(isapprox.(tails(I, 1), (0.0, 1.0)))
		@test all(isapprox.(tails(R2, 1), (0.0, 1.0)))
		@test all(isapprox.(tails(R2, 2), (2.0, 3.0)))
		@test all(isapprox.(tails(R3, 3), (4.0, 5.0)))
		@test_throws AssertionError tails(I, 2)
		@test_throws AssertionError tails(R2, 0)
		@test_throws AssertionError tails(R3, 4)

		# tails(X)
		@test all(isapprox.(tails(I), (0.0, 1.0)))
		@test tails(R2) == ((0.0, 1.0), (2.0, 3.0))
		@test tails(R3) == ((0.0, 1.0), (2.0, 3.0), (4.0, 5.0))

		# first/last (only for D=1)
		@test isapprox(first(I), 0.0)
		@test isapprox(last(I), 1.0)
		@test_throws MethodError first(R2)
		@test_throws MethodError last(R3)
	end

	@testset "Operations" begin
		I1 = interval(0.0, 1.0)
		I2 = interval(2.0, 3.0)
		I3_int = interval(4, 5)
		R2 = cartesian_product(((10, 11), (12, 13)))

		# × operator (Float64 x Float64) - 2D uses SVector
		P1 = I1 × I2
		@test P1 isa CartesianProduct{2,Float64}
		@test dim(P1) == 2
		@test P1.box isa SVector{2}
		@test tails(P1) == ((0.0, 1.0), (2.0, 3.0))

		# × operator (Float64 x Float64 x Float64) - 3D uses SVector
		P2 = I1 × I2 × I3_int
		@test P2 isa CartesianProduct{3,Float64}
		@test dim(P2) == 3
		@test P2.box isa SVector{3}
		@test tails(P2) == ((0.0, 1.0), (2.0, 3.0), (4.0, 5.0))

		# × operator creating 5D
		I4 = interval(6.0, 7.0)
		I5 = interval(8.0, 9.0)
		P5 = I1 × I2 × I3_int × I4 × I5
		@test P5 isa CartesianProduct{5,Float64}
		@test dim(P5) == 5
		@test P5.box isa SVector{5}

		# projection
		P_proj = I1 × I2 × I3_int
		proj1 = projection(P_proj, 1)
		proj2 = projection(P_proj, 2)
		proj3 = projection(P_proj, 3)

		@test proj1 isa CartesianProduct{1,Float64}
		@test dim(proj1) == 1
		@test all(isapprox.(tails(proj1), (0.0, 1.0)))
		@test isapprox(first(proj1), 0.0)
		@test isapprox(last(proj1), 1.0)

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

		@test isapprox(I.box[1][1], -3.0)
		@test isapprox(I.box[1][2], 10.0)
		@test dim(I) == 1

		I2 = interval(70.0, 100.0)

		set_2d = I × I2
		Ω2_x = projection(set_2d, 1)
		Ω2_y = projection(set_2d, 2)

		@test isapprox(Ω2_y.box[1][1], 70.0)
		@test isapprox(Ω2_x.box[1][1], -3.0)
		@test isapprox(Ω2_y.box[1][2], 100.0)
		@test isapprox(Ω2_x.box[1][2], 10.0)

		I3 = interval(-15.0, -1.0)

		set_3d = I × I2 × I3
		Ω3_x = projection(set_3d, 1)
		Ω3_y = projection(set_3d, 2)
		Ω3_z = projection(set_3d, 3)

		@test isapprox(Ω3_x.box[1][1], -3.0)
		@test isapprox(Ω3_x.box[1][2], 10.0)
		@test isapprox(Ω3_y.box[1][1], 70.0)
		@test isapprox(Ω3_y.box[1][2], 100.0)
		@test isapprox(Ω3_z.box[1][1], -15.0)
		@test isapprox(Ω3_z.box[1][2], -1.0)
	end
end
