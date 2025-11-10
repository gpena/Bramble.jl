using Bramble: @forward

@testset "@forward Macro Tests" begin
	@testset "Basic Forwarding" begin
		# Define a simple wrapper type
		struct SimpleWrapper
			data::Vector{Float64}
		end

		# Forward length and size to data field
		@forward SimpleWrapper.data (Base.length, Base.size)

		sw = SimpleWrapper([1.0, 2.0, 3.0, 4.0])

		@test length(sw) == 4
		@test size(sw) == (4,)
		@test length(sw) == length(sw.data)
		@test size(sw) == size(sw.data)
	end

	@testset "Single Function Forwarding" begin
		struct NumberWrapper
			value::Int
		end

		# Forward single function (not in tuple)
		@forward NumberWrapper.value Base.abs

		nw_pos = NumberWrapper(5)
		nw_neg = NumberWrapper(-5)

		@test abs(nw_pos) == 5
		@test abs(nw_neg) == 5
	end

	@testset "Multiple Functions" begin
		struct ArrayWrapper{T}
			array::Array{T}
		end

		@forward ArrayWrapper.array (Base.length, Base.size, Base.eltype, Base.ndims)

		aw_1d = ArrayWrapper([1, 2, 3])
		aw_2d = ArrayWrapper([1 2; 3 4])

		@test length(aw_1d) == 3
		@test size(aw_1d) == (3,)
		@test eltype(aw_1d) == Int
		@test ndims(aw_1d) == 1

		@test length(aw_2d) == 4
		@test size(aw_2d) == (2, 2)
		@test eltype(aw_2d) == Int
		@test ndims(aw_2d) == 2
	end

	@testset "Forwarding with Arguments" begin
		struct VectorContainer
			vec::Vector{Float64}
		end

		@forward VectorContainer.vec (Base.getindex, Base.setindex!)

		vc = VectorContainer([10.0, 20.0, 30.0])

		@test vc[1] == 10.0
		@test vc[2] == 20.0
		@test vc[3] == 30.0

		vc[2] = 25.0
		@test vc[2] == 25.0
		@test vc.vec[2] == 25.0
	end

	@testset "Forwarding Iterator Methods" begin
		struct IterableWrapper
			items::Vector{String}
		end

		@forward IterableWrapper.items (Base.iterate, Base.length, Base.eltype)

		iw = IterableWrapper(["a", "b", "c"])

		@test length(iw) == 3
		@test eltype(iw) == String

		# Test iteration
		collected = collect(iw)
		@test collected == ["a", "b", "c"]

		# Test for loop
		items = String[]
		for item in iw
			push!(items, item)
		end
		@test items == ["a", "b", "c"]
	end

	@testset "Forwarding with Keyword Arguments" begin
		struct StringWrapper
			str::String
		end

		@forward StringWrapper.str Base.split

		sw = StringWrapper("hello world foo")

		# Test with different keyword arguments
		@test split(sw) == ["hello", "world", "foo"]
		@test split(sw, " ") == ["hello", "world", "foo"]
		@test split(sw, keepempty = false) == ["hello", "world", "foo"]
	end

	@testset "Type Stability" begin
		struct TypedWrapper{T}
			value::T
		end

		@forward TypedWrapper.value Base.zero

		tw_int = TypedWrapper(42)
		tw_float = TypedWrapper(3.14)

		z_int = zero(tw_int)
		z_float = zero(tw_float)

		@test z_int isa Int
		@test z_int == 0

		@test z_float isa Float64
		@test z_float == 0.0
	end

	@testset "Custom Functions" begin
		# Define custom functions
		custom_double(x) = 2x
		custom_square(x) = x^2

		struct NumHolder
			num::Int
		end

		# Forward custom functions (note: need to use full path or define them before @forward)
		@eval @forward NumHolder.num (custom_double, custom_square)

		nh = NumHolder(5)

		@test custom_double(nh) == 10
		@test custom_square(nh) == 25
	end

	@testset "Error Cases" begin
		# Test invalid syntax
		@test_throws LoadError @eval @forward InvalidType func
		@test_throws LoadError @eval @forward Type.field.nested func
	end
end
