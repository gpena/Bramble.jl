module SpaceInterpolationOutsideTests

using Test
using LinearAlgebra: Diagonal
using Bramble
using Bramble: interpolation_matrix, weights, Innerh, form, assemble, InterpolationNode,
               GridInterpolant, TrialFunction, TestFunction, _bind_interp_spaces

# The interpolation nodes store the `outside` policy as an isbits enum, not a `Symbol`, so
# the nodes stay isbits apart from their arrays and spaces. The public keyword is unchanged:
# these tests read the policy back through behaviour, never through the stored value.

@testset "Interpolation nodes carry no Symbol" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
    W = gridspace(Ωₕ)
    f(x) = 2x + 3
    uₕ = Rₕ(W, f)
    u, v = TrialFunction{1}(), TestFunction{1}()

    @testset "Node is isbits before and after binding" begin
        for o in (:error, :clamp, :extrapolate), leaf in (u, v)

            n = πₕ(leaf; outside = o)
            @test isbitstype(typeof(n))
            b = _bind_interp_spaces((n,), W, W)[1]
            @test b isa InterpolationNode
            @test fieldtype(typeof(b), :outside) !== Symbol
            @test !any(T -> T === Symbol, fieldtypes(typeof(b)))
        end
    end

    @testset "GridInterpolant holds no Symbol" begin
        for o in (:error, :clamp, :extrapolate, 0.0, NaN)
            g = πₕ(uₕ; outside = o).func
            @test g isa GridInterpolant
            @test fieldtype(typeof(g), :outside) !== Symbol
        end
    end

    @testset "Policies still act as named" begin
        outside_value(o) = πₕ(uₕ; outside = o).func(1.5)
        @test_throws ArgumentError πₕ(uₕ).func(1.5)
        @test_throws ArgumentError πₕ(uₕ; outside = :error).func(1.5)
        @test outside_value(:clamp) ≈ f(1.0)
        @test outside_value(:extrapolate) ≈ f(1.5)
        @test outside_value(-7.0) == -7.0
        @test isnan(outside_value(NaN))
        @test πₕ(uₕ; outside = :clamp).func(0.37) ≈ f(0.37)
    end

    @testset "Bilinear node keeps its policy" begin
        # Interpolating onto a mesh that reaches past the source domain: `:error` refuses
        # the point, while `:clamp` and `:extrapolate` assemble to different matrices, each
        # `Hₕ · P` for the `P` that `interpolation_matrix` builds for that policy.
        Wsrc = gridspace(mesh(domain(interval(0.0, 1.0)), 5, true))
        Wdst = gridspace(mesh(domain(interval(0.0, 1.2)), 7, true))
        H = Diagonal(collect(weights(Wdst, Innerh())))
        A(o) = assemble(form(Wsrc, Wdst, (u, v) -> innerₕ(πₕ(u; outside = o), v)))
        @test_throws ArgumentError A(:error)
        @test_throws ArgumentError assemble(form(Wsrc, Wdst, (u, v) -> innerₕ(πₕ(u), v)))
        Ac, Ae = A(:clamp), A(:extrapolate)
        @test Ac ≈ H * interpolation_matrix(Wdst, Wsrc; outside = :clamp)
        @test Ae ≈ H * interpolation_matrix(Wdst, Wsrc; outside = :extrapolate)
        @test Ac != Ae
    end

    @testset "Bad keywords still refused" begin
        for bad in (:wrap, "clamp", 0.0)
            @test_throws ArgumentError πₕ(u; outside = bad)
        end
        @test_throws ArgumentError πₕ(uₕ; outside = :wrap)
    end
end

end # module
