using Test
using Bramble
using Bramble: BrambleFunction, interval, ×, cartesian_product, domain, embed_function,
               argstype, codomaintype, has_time
using StaticArrays

@testset "BrambleFunction Tests with Real CartesianProduct" begin
    # --- Test Setup ---
    Ω1 = interval(0.0, 1.0) # CartesianProduct{1, Float64}
    Ω2 = interval(0.0, 1.0) × interval(10.0, 20.0) # CartesianProduct{2, Float64}
    Ω3_box = ((0.0f0, 1.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0))
    Ω3 = cartesian_product(Ω3_box) # CartesianProduct{3, Float32}

    I = interval(0.0, 2.0) # Time interval, CartesianProduct{1, Float64}

    # Test Functions
    f1(x) = 2.0 * x
    f2(x) = x[1] + x[2]^2
    f3(x::NTuple{3, Float32}) = x[1] * x[2] - x[3]

    f1t(x, t) = (1.0 + t) * x
    f2t(x, t) = t * (x[1] - x[2])

    @testset "Embed Non-Time-Dependent" begin
        @testset "1D Domain" begin
            bf1_func = embed_function(Ω1, f1)

            @test bf1_func isa BrambleFunction{Float64, false, Float64, typeof(Ω1)}
            @test bf1_func(0.2) ≈ 0.4
            @test bf1_func(0.0f0) ≈ 0.0 # Float32 input conversion
            @test bf1_func((0.5,)) ≈ 1.0 # 1-tuple input
            @test bf1_func(SVector(0.5)) ≈ 1.0 # 1D SVector input
            @test bf1_func([0.5]) ≈ 1.0 # 1D Vector input

            @test argstype(bf1_func.wrapped) == Float64
            @test codomaintype(bf1_func.wrapped) == Float64
        end

        @testset "2D Domain" begin
            bf2_func = embed_function(Ω2, f2)

            @test bf2_func isa
                  BrambleFunction{NTuple{2, Float64}, false, Float64, typeof(Ω2)}

            pt1 = (0.5, 10.5)
            pt2 = (0.1, 10.2)
            pt3 = (1, 12.0)

            @test bf2_func(pt1) ≈ 0.5 + 10.5^2
            @test bf2_func(pt2...) ≈ 0.1 + 10.2^2
            @test bf2_func(1, 12) ≈ 1.0 + 12.0^2
            @test bf2_func(SVector(0.5, 10.5)) ≈ 0.5 + 10.5^2
            @test bf2_func([0.5, 10.5]) ≈ 0.5 + 10.5^2

            @test argstype(bf2_func.wrapped) == NTuple{2, Float64}
            @test codomaintype(bf2_func.wrapped) == Float64
        end

        @testset "3D Domain (Float32)" begin
            bf3_func = embed_function(Ω3, f3)

            @test bf3_func isa
                  BrambleFunction{NTuple{3, Float32}, false, Float32, typeof(Ω3)}

            pt1_f32 = (0.5f0, 0.5f0, 0.1f0)
            pt2_f64 = (0.1, 0.2, 0.3)

            @test bf3_func(pt1_f32...) ≈ 0.5f0 * 0.5f0 - 0.1f0
            @test bf3_func(pt2_f64...) ≈ (0.1f0 * 0.2f0 - 0.3f0)

            @test argstype(bf3_func.wrapped) == NTuple{3, Float32}
            @test codomaintype(bf3_func.wrapped) == Float32
        end

        @testset "Function Call Syntax (Non-Time)" begin
            bf2 = embed_function(Ω2, x -> x[1] + x[2])
            @test bf2((1.0, 12.0)) ≈ 13.0
            @test bf2(1.0, 12.0) ≈ 13.0
            @test bf2((1, 12)) ≈ 13.0
            @test bf2(1, 12) ≈ 13.0

            bf1 = embed_function(Ω1, x -> 2x)
            @test bf1(0.5) ≈ 1.0
            @test bf1(1) ≈ 2.0
        end

        @testset "Embed on Domain Struct" begin
            dom = domain(Ω2)
            bf_dom = embed_function(dom, f2)
            @test bf_dom isa
                  BrambleFunction{NTuple{2, Float64}, false, Float64, typeof(dom)}
            @test bf_dom((0.5, 11.0)) ≈ 0.5 + 11.0^2
        end
    end

    @testset "Embed Time-Dependent" begin
        @testset "1D Space + Time" begin
            bf1t_func = embed_function(Ω1, I, f1t)

            ExpectedInnerCoType = BrambleFunction{Float64, false, Float64, typeof(Ω1)}
            ExpectedOuterType = BrambleFunction{
                Float64, true, ExpectedInnerCoType, typeof(I)}

            @test bf1t_func isa ExpectedOuterType

            t2 = 1.0
            bf1_at_t2 = bf1t_func(t2)
            @test bf1_at_t2 isa ExpectedInnerCoType

            x_val = 0.5
            @test bf1_at_t2(x_val) ≈ (1.0 + t2) * x_val
            # Test direct (x, t) call
            @test bf1t_func(x_val, t2) ≈ (1.0 + t2) * x_val

            @test argstype(bf1t_func.wrapped) == Float64
            @test codomaintype(bf1t_func.wrapped) == ExpectedInnerCoType
        end

        @testset "2D Space + Time" begin
            bf2t_func = embed_function(Ω2, I, f2t)

            ExpectedInnerCoType = BrambleFunction{
                NTuple{2, Float64}, false, Float64, typeof(Ω2)}
            ExpectedOuterType = BrambleFunction{
                Float64, true, ExpectedInnerCoType, typeof(I)}

            @test bf2t_func isa ExpectedOuterType

            t_val = 1.5
            bf2_at_t = bf2t_func(t_val)
            @test bf2_at_t isa ExpectedInnerCoType

            pt1 = (0.5, 11.0)
            pt2 = (0, 20)

            @test bf2_at_t(pt1) ≈ t_val * (pt1[1] - pt1[2])
            @test bf2_at_t(pt1...) ≈ t_val * (pt1[1] - pt1[2])
            @test bf2_at_t(pt2) ≈ t_val * (0.0 - 20.0)
            @test bf2_at_t(pt2...) ≈ t_val * (0.0 - 20.0)

            # Direct (x, t) call
            @test bf2t_func(pt1, t_val) ≈ t_val * (pt1[1] - pt1[2])

            @test argstype(bf2t_func.wrapped) == Float64
            @test codomaintype(bf2t_func.wrapped) == ExpectedInnerCoType
        end
    end

    @testset "has_time" begin
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
        @test argstype(bf2.wrapped) == NTuple{2, Float64}
        @test codomaintype(bf2.wrapped) == Float64

        bf3 = embed_function(Ω3, f3)
        @test argstype(bf3.wrapped) == NTuple{3, Float32}
        @test codomaintype(bf3.wrapped) == Float32
    end

    @testset "Edge Cases and Type Conversions" begin
        # Identity function
        identity_bf = embed_function(Ω1, identity)
        @test identity_bf(0.7) ≈ 0.7

        # Constant function
        const_func = x -> 42.0
        const_bf = embed_function(Ω1, const_func)
        @test const_bf(0.1) ≈ 42.0
        @test const_bf(0.9) ≈ 42.0

        # Zero function
        zero_func = x -> 0.0
        zero_bf = embed_function(Ω1, zero_func)
        @test zero_bf(0.5) ≈ 0.0

        # 2D constant
        const_2d = x -> 100.0
        const_2d_bf = embed_function(Ω2, const_2d)
        @test const_2d_bf((0.5, 15.0)) ≈ 100.0

        # Integer arithmetic conversion
        int_func = x -> Int(round(10 * x))
        float_bf = embed_function(Ω1, int_func)
        @test float_bf(0.3) == 3
    end

    @testset "embed_function with BrambleFunction input" begin
        bf1 = embed_function(Ω1, f1)
        bf1_again = embed_function(Ω1, bf1)
        @test bf1_again === bf1
    end

    @testset "Type Inference & Performance" begin
        bf1 = embed_function(Ω1, f1)
        bf2 = embed_function(Ω2, f2)

        @inferred bf1(0.5)
        @inferred bf2((0.5, 10.5))

        # Calling a 1D scalar FunctionWrapper incurs 0 heap allocations
        @test_allocs bf1(0.5)
    end

    @testset "Display / Show" begin
        bf1 = embed_function(Ω1, f1)
        bf1t = embed_function(Ω1, I, f1t)

        io = IOBuffer()
        show(io, bf1)
        str = String(take!(io))
        @test occursin("BrambleFunction(spatial)", str)
        @test occursin("Float64 -> Float64", str)

        show(IOContext(io, :compact => true), bf1)
        @test occursin("BrambleFunction(spatial, Float64 -> Float64)", String(take!(io)))

        show(IOContext(io, :compact => true), bf1t)
        @test occursin("time-dependent", String(take!(io)))
    end
end
@testset "BrambleFunction: additional coverage" begin
    using Bramble: _get_args_type, argstype, codomaintype, BrambleFunction, CartesianProduct
    using Bramble: FunctionWrapper  # re-accessed via Bramble internals

    Ω1 = interval(0.0, 1.0)
    Ω2 = interval(0.0, 1.0) × interval(0.0, 1.0)
    dom = domain(Ω1)

    @testset "_get_args_type: Type-level dispatch (lines 36-40)" begin
        @test _get_args_type(CartesianProduct{1, Float64}) === Float64
        @test _get_args_type(CartesianProduct{2, Float64}) === NTuple{2, Float64}
        @test _get_args_type(typeof(dom)) === Float64  # Domain{CartesianProduct{1,Float64}}
        @test _get_args_type(domain(Ω2)) === NTuple{2, Float64}  # Domain instance dispatch
    end

    @testset "argstype / codomaintype on Type (lines 141-143, 151-153)" begin
        # Construct via embed_function and access wrapped FunctionWrapper internals
        f1 = x -> 2.0 * x
        bf1 = embed_function(Ω1, f1)

        # FunctionWrapper with type-level dispatch (lines 142, 152)
        @test argstype(typeof(bf1.wrapped)) === Float64
        @test codomaintype(typeof(bf1.wrapped)) === Float64

        # argstype/codomaintype on instance (covered by existing tests; add type-level)
        @test argstype(bf1.wrapped) === Float64
        @test codomaintype(bf1.wrapped) === Float64
    end

    @testset "embed_function on Domain and SVector dispatch (lines 72-85)" begin
        # 2D SVector dispatch (line 100)
        f2 = x -> x[1] + x[2]
        bf2 = embed_function(Ω2, f2)
        @test bf2(SVector(0.3, 0.7)) ≈ 1.0

        # 2D SVector with different element type (line 101)
        @test bf2(SVector{2, Float32}(0.3f0, 0.7f0)) ≈ 1.0f0

        # AbstractVector for 2D (line 102)
        @test bf2([0.5, 0.5]) ≈ 1.0
    end

    @testset "BrambleFunction show with time-dependent" begin
        f1(x) = 2.0 * x
        f1t(x, t) = (1.0 + t) * x
        I = interval(0.0, 2.0)
        bf1t = embed_function(Ω1, I, f1t)

        io = IOBuffer()
        show(io, bf1t)
        str = String(take!(io))
        @test occursin("time-dependent", str)

        show(IOContext(io, :compact => true), bf1t)
        @test occursin("time-dependent", String(take!(io)))
    end

    @testset "_get_domains parsing tests (lines 72-87)" begin
        using Bramble: _get_domains
        @test _get_domains(:Ω) == (:Ω, nothing)
        @test _get_domains(:(Ω × I)) == (:Ω, :I)
        @test _get_domains(:(domain(X))) == (:(domain(X)), nothing)
        @test_throws ErrorException _get_domains(:(a = 1))
        @test_throws ErrorException _get_domains(123)
    end
end

@testset "BrambleFunction interface coverage" begin
    @testset "_get_args_type falls back to dim/eltype" begin
        # A mesh is neither a CartesianProduct nor a Domain, so it resolves through
        # the generic Val(dim(X))/eltype(X) path rather than a set-specific method.
        Ω1 = mesh(domain(interval(0.0, 1.0)), 5, true)
        @test Bramble._get_args_type(Ω1) === Float64

        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
        @test Bramble._get_args_type(Ω2) === NTuple{2, Float64}

        # Same answers as the set-based methods for the equivalent domains.
        @test Bramble._get_args_type(Ω1) === Bramble._get_args_type(interval(0.0, 1.0))
        @test Bramble._get_args_type(Ω2) ===
              Bramble._get_args_type(interval(0.0, 1.0) × interval(0.0, 1.0))
    end

    @testset "embed_function is idempotent on an existing BrambleFunction" begin
        X = interval(0.0, 1.0)
        I = interval(0.0, 1.0)
        bf = embed_function(X, x -> 2x)

        # Re-embedding must return the same object, not wrap it a second time,
        # both with and without a time domain.
        @test embed_function(X, bf) === bf
        @test embed_function(X, I, bf) === bf
        @test !has_time(embed_function(X, I, bf))
        @test embed_function(X, I, bf)(2.0) == 4.0
    end
end
