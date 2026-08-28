using Test
using Bramble
using Bramble: CartesianProduct
using StaticArrays

@testset "CartesianProduct Tests" begin
	@testset "Constructors" begin
		# interval constructor (Float64 default)
		I_f64 = interval(-3.0, 10.0)
		@test I_f64 isa CartesianProduct{1,Float64}
		@test I_f64.box isa SVector{1}
		@test I_f64.box[1] == (-3.0, 10.0)
		@test all(isapprox.(I_f64.box[1], (-3.0, 10.0)))
		@test I_f64.collapsed[1] == false

		# interval constructor (Int -> Float64)
		I_int = interval(-3, 10)
		@test I_int isa CartesianProduct{1,Float64}
		@test I_int.box[1] == (-3.0, 10.0)
		@test all(isapprox.(I_int.box[1], (-3.0, 10.0)))

		# interval constructor (Float32)
		I_f32 = interval(0.0f0, 1.0f0)
		@test I_f32 isa CartesianProduct{1,Float32}
		@test eltype(I_f32) === Float32

		# interval edge case: zero width
		I_zero = interval(5.5, 5.5)
		@test I_zero isa CartesianProduct{1,Float64}
		@test all(isapprox.(I_zero.box[1], (5.5, 5.5)))
		@test I_zero.collapsed[1] == true

		# interval constructor assertion x <= y
		@test_throws ArgumentError interval(10, 1)

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
		@test cp_int_2d.box isa SVector{2}
		@test cp_int_2d.box[1] == (0.0, 1.0)
		@test cp_int_2d.box[2] == (4.0, 5.0)

		# cartesian_product(NTuple) - Float32
		cp_f32_3d = cartesian_product(((0.0f0, 1.0f0), (2.0f0, 3.0f0), (-1.0f0, 0.0f0)))
		@test cp_f32_3d isa CartesianProduct{3,Float32}
		@test cp_f32_3d.box isa SVector{3}
		@test cp_f32_3d.box[1] == (0.0f0, 1.0f0)
		@test cp_f32_3d.box[2] == (2.0f0, 3.0f0)
		@test cp_f32_3d.box[3] == (-1.0f0, 0.0f0)

		# cartesian_product invalid assertion
		@test_throws ArgumentError cartesian_product(((1.0, 0.0), (2.0, 3.0)))

		# cartesian_product(CartesianProduct) identity
		cp_id = cartesian_product(cp_int_2d)
		@test cp_id === cp_int_2d

		# point constructor (collapsed CartesianProduct)
		P_f64 = point(3.5)
		@test P_f64 isa CartesianProduct{1,Float64}
		@test P_f64.box[1] == (3.5, 3.5)
		@test P_f64.collapsed[1] == true

		P_f32 = point(2.0f0)
		@test P_f32 isa CartesianProduct{1,Float32}
		@test eltype(P_f32) === Float32

		# box constructors
		B1d = box(1.0, 5.0)
		@test B1d isa CartesianProduct{1,Float64}
		@test B1d.box[1] == (1.0, 5.0)

		B1d_rev = box(5.0, 1.0)
		@test B1d_rev isa CartesianProduct{1,Float64}
		@test B1d_rev.box[1] == (1.0, 5.0)

		B2d = box((0.0, 2.0), (1.0, 3.0))
		@test B2d isa CartesianProduct{2,Float64}
		@test B2d.box[1] == (0.0, 1.0)
		@test B2d.box[2] == (2.0, 3.0)

		# box with reversed points
		B2d_rev = box((5.0, 10.0), (2.0, 8.0))
		@test B2d_rev.box[1] == (2.0, 5.0)
		@test B2d_rev.box[2] == (8.0, 10.0)

		B3d = box((0.0, 1.0, 2.0), (3.0, -1.0, 5.0))
		@test B3d isa CartesianProduct{3,Float64}
		@test B3d.box[1] == (0.0, 3.0)
		@test B3d.box[2] == (-1.0, 1.0)
		@test B3d.box[3] == (2.0, 5.0)
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
		@test topo_dim(P_collapsed) === 0

		I_line = I × point(2.0)
		@test dim(I_line) === 2
		@test topo_dim(I_line) === 1

		# center
		@test center(I) ≈ SVector(0.5)
		@test center(R2) ≈ SVector(0.5, 2.5)
		@test center(R3) ≈ SVector(0.5, 2.5, 4.5)

		# is_collapsed
		@test is_collapsed(I) == false
		@test is_collapsed(P_collapsed) == true
		@test is_collapsed(1.0, 1.0) == true
		@test is_collapsed(1, 1.0) == true
		@test is_collapsed(0.0, 1.0) == false

		# point_type
		@test point_type(I) === Float64
		@test point_type(typeof(I)) === Float64
		@test point_type(R2) === NTuple{2,Float64}
		@test point_type(typeof(R2)) === NTuple{2,Float64}
		@test point_type(R3) === NTuple{3,Float64}
		@test point_type(typeof(R3)) === NTuple{3,Float64}

		# Call syntax (X(i))
		@test all(isapprox.(I(1), (0.0, 1.0)))
		@test all(isapprox.(R2(1), (0.0, 1.0)))
		@test all(isapprox.(R2(2), (2.0, 3.0)))
		@test all(isapprox.(R3(1), (0.0, 1.0)))
		@test all(isapprox.(R3(2), (2.0, 3.0)))
		@test all(isapprox.(R3(3), (4.0, 5.0)))
		@test_throws BoundsError I(2)
		@test_throws BoundsError R2(0)
		@test_throws BoundsError R3(4)

		# tails(X, i)
		@test all(isapprox.(tails(I, 1), (0.0, 1.0)))
		@test all(isapprox.(tails(R2, 1), (0.0, 1.0)))
		@test all(isapprox.(tails(R2, 2), (2.0, 3.0)))
		@test all(isapprox.(tails(R3, 3), (4.0, 5.0)))
		@test_throws BoundsError tails(I, 2)
		@test_throws BoundsError tails(R2, 0)
		@test_throws BoundsError tails(R3, 4)

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

	@testset "Operations and Mixed Promotions" begin
		I1 = interval(0.0, 1.0)
		I2 = interval(2.0, 3.0)
		I3_int = interval(4, 5)
		I_f32 = interval(0.0f0, 1.0f0)

		# × operator (Float64 x Float64)
		P1 = I1 × I2
		@test P1 isa CartesianProduct{2,Float64}
		@test dim(P1) == 2
		@test P1.box isa SVector{2}
		@test tails(P1) == ((0.0, 1.0), (2.0, 3.0))

		# × operator with mixed types (Float32 x Float64)
		P_mixed = I_f32 × I1
		@test P_mixed isa CartesianProduct{2,Float64}
		@test eltype(P_mixed) === Float64
		@test tails(P_mixed) == ((0.0, 1.0), (0.0, 1.0))

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

		@test_throws BoundsError projection(P_proj, 4)
		@test_throws BoundsError projection(P_proj, 0)
	end

	@testset "Zero-Allocation & Type Inference" begin
		I1 = interval(0.0, 1.0)
		I2 = interval(2.0, 3.0)
		cp2 = I1 × I2
		cp3 = I1 × I2 × interval(4.0, 5.0)

		# Type inference
		@inferred interval(0.0, 1.0)
		@inferred point(0.5)
		@inferred box(0.0, 1.0)
		@inferred box((0.0, 1.0), (2.0, 3.0))
		@inferred center(cp2)
		@inferred center(cp3)
		@inferred dim(cp3)
		@inferred eltype(cp3)
		@inferred topo_dim(cp3)
		@inferred projection(cp3, 2)
		@inferred tails(cp3, 1)
		@inferred tails(cp3)
		@inferred cp2(1)
		@inferred is_collapsed(I1)
		@inferred is_collapsed(point(1.0))
		@inferred I1 × I2

		# Allocations: Zero heap allocations for core operations
		@test_allocs interval(0.0, 1.0)
		@test_allocs point(0.5)
		@test_allocs box(0.0, 1.0)
		@test_allocs box((0.0, 1.0), (2.0, 3.0))
		@test_allocs center(cp2)
		@test_allocs topo_dim(cp3)
		@test_allocs projection(cp3, 2)
		@test_allocs tails(cp3)
		@test_allocs tails(cp3, 1)
		@test_allocs cp2(1)
		@test_allocs (I1 × I2)
	end

	@testset "Display / Show" begin
		I = interval(0.0, 1.0)
		P = point(2.5)
		R2 = I × interval(2.0, 3.0)
		R2_collapsed = I × point(3.0)

		# Compact mode
		io_compact = IOBuffer()
		show(IOContext(io_compact, :compact => true), I)
		@test occursin("[0.0, 1.0]", String(take!(io_compact)))

		show(IOContext(io_compact, :compact => true), P)
		@test occursin("Point(2.5)", String(take!(io_compact)))

		show(IOContext(io_compact, :compact => true), R2)
		@test occursin("[0.0, 1.0] × [2.0, 3.0]", String(take!(io_compact)))

		show(IOContext(io_compact, :compact => true), R2_collapsed)
		@test occursin("[0.0, 1.0] × 3.0", String(take!(io_compact)))

		# Detailed multiline mode
		io_det = IOBuffer()
		show(io_det, I)
		str_I = String(take!(io_det))
		@test occursin("CartesianProduct{1,Float64}", str_I)
		@test occursin("Interval", str_I)

		show(io_det, P)
		str_P = String(take!(io_det))
		@test occursin("Point", str_P)

		show(io_det, R2)
		str_R2 = String(take!(io_det))
		@test occursin("CartesianProduct{2,Float64}", str_R2)

		show(io_det, R2_collapsed)
		str_R2c = String(take!(io_det))
		@test occursin("topological dim 1", str_R2c)
	end

	@testset "Point containment — AbstractVector and fallback" begin
		I = interval(0.0, 1.0)
		R2 = interval(0.0, 2.0) × interval(-1.0, 1.0)

		# AbstractVector path (line 220)
		@test [0.5] ∈ I
		@test [1.5] ∉ I
		@test [0.5, 0.0] ∈ R2
		@test [2.5, 0.0] ∉ R2
		@test [0.5, 0.0, 0.0] ∉ R2  # wrong dimension → false (line 220 length check)

		# Fallback dispatch for non-numeric, non-tuple input (line 221)
		@test ("hello" ∈ I) == false
		@test (:sym ∈ R2) == false
	end

	@testset "Pretty-print helpers" begin
		using Bramble: PrettyPrinter, with_indent, print_indent, print_colored, println_colored,
					  print_header, print_section_header, print_subsection_header,
					  print_key_value, print_label, print_value, print_interval,
					  print_dimension_info, print_empty_message, print_marker_summary,
					  print_labels_list, get_dimension_label

		io = IOBuffer()
		pp0 = PrettyPrinter(io, false, 0)
		pp1 = with_indent(pp0, 1)
		pp2 = with_indent(pp0, 2)

		# print_indent: level 0 → no output
		print_indent(pp0)
		@test isempty(String(take!(io)))

		# print_indent: level 1 → two spaces
		print_indent(pp1)
		@test String(take!(io)) == "  "

		# print_colored: default color
		print_colored(pp0, "hello")
		@test occursin("hello", String(take!(io)))

		# print_colored: with color
		print_colored(pp0, "world"; color = :blue)
		@test occursin("world", String(take!(io)))

		# println_colored
		println_colored(pp0, "line"; color = :green)
		@test occursin("line", String(take!(io)))

		# print_header without type_info
		print_header(pp0, "Header")
		@test occursin("Header", String(take!(io)))

		# print_header with type_info
		print_header(pp0, "Title", "Float64")
		str = String(take!(io))
		@test occursin("Title", str) && occursin("Float64", str)

		# print_section_header
		print_section_header(pp0, "Section:")
		@test occursin("Section:", String(take!(io)))

		# print_subsection_header without count
		print_subsection_header(pp0, "Sub", 0)
		@test occursin("Sub", String(take!(io)))

		# print_subsection_header with count
		print_subsection_header(pp0, "Sub", 3)
		@test occursin("(3)", String(take!(io)))

		# print_key_value
		print_key_value(pp0, "key", "val")
		str = String(take!(io))
		@test occursin("key", str) && occursin("val", str)

		# print_label
		print_label(pp0, :boundary)
		@test occursin(":boundary", String(take!(io)))

		# print_value
		print_value(pp0, 3.14)
		@test occursin("3.14", String(take!(io)))

		# print_interval — normal
		print_interval(pp0, 0.0, 1.0)
		@test occursin("0.0, 1.0", String(take!(io)))

		# print_interval — collapsed
		print_interval(pp0, 0.5, 0.5; collapsed = true)
		str = String(take!(io))
		@test occursin("collapsed", str)

		# print_dimension_info
		print_dimension_info(pp0, "x", 0.0, 1.0, false)
		str = String(take!(io))
		@test occursin("x", str) && occursin("0.0", str)

		# print_empty_message
		print_empty_message(pp0)
		@test occursin("none", String(take!(io)))

		# print_marker_summary — all types
		print_marker_summary(pp0, 2, 1, 0)
		str = String(take!(io))
		@test occursin("2 symbols", str) && occursin("1 tuple", str)

		# print_marker_summary — single marker (singular)
		print_marker_summary(pp0, 1, 0, 0)
		@test occursin("1 marker", String(take!(io)))

		# print_marker_summary — condition only
		print_marker_summary(pp0, 0, 0, 2)
		@test occursin("2 functions", String(take!(io)))

		# print_labels_list
		print_labels_list(pp0, [:a, :b, :c])
		str = String(take!(io))
		@test occursin(":a", str) && occursin(":b", str) && occursin(":c", str)

		# get_dimension_label
		@test get_dimension_label(1) == "x"
		@test get_dimension_label(2) == "y"
		@test get_dimension_label(3) == "z"
		@test get_dimension_label(7) == "x7"  # beyond precomputed table
	end
end
