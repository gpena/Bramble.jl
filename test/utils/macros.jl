using Test
using Bramble: @forward

# Helper wrapper types defined at top level for method forwarding verification
struct SimpleWrapper
    data::Vector{Float64}
end
@forward SimpleWrapper.data (Base.length, Base.size)

struct NumberWrapper
    value::Int
end
@forward NumberWrapper.value Base.abs

struct ArrayWrapper{T}
    array::Array{T}
end
@forward ArrayWrapper.array (Base.length, Base.size, Base.eltype, Base.ndims)

struct VectorContainer
    vec::Vector{Float64}
end
@forward VectorContainer.vec (Base.getindex, Base.setindex!)

struct IterableWrapper
    items::Vector{String}
end
@forward IterableWrapper.items (Base.iterate, Base.length, Base.eltype)

struct StringWrapper
    str::String
end
@forward StringWrapper.str Base.split

struct TypedWrapper{T}
    value::T
end
@forward TypedWrapper.value Base.zero

custom_double(x) = 2x
custom_square(x) = x^2

struct NumHolder
    num::Int
end
@forward NumHolder.num (custom_double, custom_square)

@testset "Macro method forwarding" begin
    # Invariants tested:
    # 1. Forwarded methods return identical results to direct field access.
    @testset "Basic forwarding" begin
        sw = SimpleWrapper([1.0, 2.0, 3.0, 4.0])
        @test length(sw) == 4
        @test size(sw) == (4,)
        @test length(sw) == length(sw.data)
        @test size(sw) == size(sw.data)
    end

    # Invariants tested:
    # 1. Single function forward syntax @forward T.field f.
    @testset "Single function" begin
        nw_pos = NumberWrapper(5)
        nw_neg = NumberWrapper(-5)
        @test abs(nw_pos) == 5
        @test abs(nw_neg) == 5
    end

    # Invariants tested:
    # 1. Tuple syntax @forward T.field (f, g, ...) forwarding multiple methods simultaneously.
    @testset "Multiple functions" begin
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

    # Invariants tested:
    # 1. Positional arguments are passed through to the wrapped field (indexing and mutation).
    @testset "Positional arguments" begin
        vc = VectorContainer([10.0, 20.0, 30.0])
        @test vc[1] == 10.0
        @test vc[2] == 20.0
        @test vc[3] == 30.0

        vc[2] = 25.0
        @test vc[2] == 25.0
        @test vc.vec[2] == 25.0
    end

    # Invariants tested:
    # 1. Forwarding iterate enables Julia's standard iteration protocol (for loops, collect).
    @testset "Iteration protocol" begin
        iw = IterableWrapper(["a", "b", "c"])
        @test length(iw) == 3
        @test eltype(iw) == String

        collected = collect(iw)
        @test collected == ["a", "b", "c"]

        items = String[]
        for item in iw
            push!(items, item)
        end
        @test items == ["a", "b", "c"]
    end

    # Invariants tested:
    # 1. Keyword arguments are forwarded transparently to underlying methods.
    @testset "Keyword arguments" begin
        sw = StringWrapper("hello world foo")
        @test split(sw) == ["hello", "world", "foo"]
        @test split(sw, " ") == ["hello", "world", "foo"]
        @test split(sw, keepempty = false) == ["hello", "world", "foo"]
    end

    # Invariants tested:
    # 1. Forwarded methods preserve type inference and return values.
    @testset "Type stability" begin
        tw_int = TypedWrapper(42)
        tw_float = TypedWrapper(3.14)

        z_int = zero(tw_int)
        z_float = zero(tw_float)

        @test z_int isa Int
        @test z_int == 0

        @test z_float isa Float64
        @test z_float == 0.0
    end

    # Invariants tested:
    # 1. Custom non-Base functions can be forwarded.
    @testset "Custom functions" begin
        nh = NumHolder(5)
        @test custom_double(nh) == 10
        @test custom_square(nh) == 25
    end

    # Failure modes tested:
    # 1. Macro invocation without property access expression (T.x) raises actionable syntax error.
    @testset "Malformed syntax" begin
        err = try
            @eval @forward NotAFieldAccess (Base.size,)
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("@forward T.x", sprint(showerror, err))
    end
end
