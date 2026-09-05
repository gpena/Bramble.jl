using Test
using Bramble
using ForwardDiff
using Bramble: values, components

# Differentiating through the library.
#
# A grid function's coefficients and the coordinates of the mesh under it are two
# different things. The mesh is geometry and stays in its own element type; the
# coefficients are whatever the computation needs, and for a derivative that is a
# `ForwardDiff.Dual`. `Rₕ` and `avgₕ` take their element type from the function being
# restricted, and `similar` (which every operator allocates its output with) keeps the
# element's own type rather than reaching back to the space's.
#
# None of the arithmetic needed changing: the operators, inner products and norms were
# already generic. What blocked this was containers, in three places, each of which is a
# case below:
#
#   - `Rₕ`/`avgₕ` allocated through `element(Wₕ)`, which takes the backend's type, so a
#     Dual could not be stored in the first place;
#   - `similar(uₕ)` went back to the space, so an operator handed a Dual grid function
#     tried to write its result into a Float64 one;
#   - `avgₕ` built its Gauss rule in the grid function's type, and the rule is tabulated
#     for real types only, so it asked QuadGK for Dual-valued nodes.
#
# Each test compares against a central difference of the same functional evaluated in
# plain Float64, so it checks the derivative is right, not merely that it ran.

@testset "Scalar-tuple product differentiation" begin
    # `a * (uₕ, vₕ)` allocated its output with `similar(vₕ[i])`, taking the element type from
    # the tuple and dropping the scalar's, so a Dual scalar was written into a
    # `Vector{Float64}`:
    #
    #     ERROR: MethodError: no method matching Float64(::ForwardDiff.Dual{...})
    #
    # It threw rather than silently losing the derivative, which is the better failure, but
    # it made `a * (uₕ, vₕ)` unusable under AD. Both tuple methods delegate to broadcasting
    # now. The same held for a Dual-valued `VectorElement` on the left.
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 8, true)
    Wₕ = gridspace(Ωₕ)
    tup = (Rₕ(Wₕ, x -> x), Rₕ(Wₕ, x -> 2x))

    scaled(a) = sum(values((a * tup)[1])) + sum(values((a * tup)[2]))
    @test _matches_fd(scaled)
    @test eltype(values((ForwardDiff.Dual{Nothing}(2.0, 1.0) * tup)[1])) <: ForwardDiff.Dual

    # a Dual-valued element on the left of the tuple, which takes the other method
    weighted(a) = sum(values(((a * Rₕ(Wₕ, x -> 3.0)) * tup)[1]))
    @test _matches_fd(weighted)
end

@testset "Automatic differentiation" begin
    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 21, true)
    Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 8), (true, false))
    Wₕ1, Wₕ2 = gridspace(Ωₕ1), gridspace(Ωₕ2)
    Vₕ2 = gridspace(Ωₕ2, Val(2))

    @testset "Non-differentiated mesh" begin
        # The point of the whole exercise: Dual coefficients over Float64 geometry.
        a = ForwardDiff.Dual{ForwardDiff.Tag{typeof(identity), Float64}}(1.3, 1.0)
        uₕ = Rₕ(Wₕ1, x -> a * sin(x))

        @test eltype(Ωₕ1) === Float64
        @test eltype(points(Ωₕ1)) === Float64
        @test eltype(spacings(Ωₕ1)) === Float64
        @test eltype(values(uₕ)) === typeof(a)
        @test eltype(values(similar(uₕ))) === typeof(a)
        @test eltype(values(D₋ₓ(uₕ))) === typeof(a)
        @test innerₕ(uₕ, uₕ) isa typeof(a)

        # and an ordinary run is untouched: no Dual anywhere, still Float64
        vₕ = Rₕ(Wₕ1, sin)
        @test eltype(values(vₕ)) === Float64
        @test eltype(values(similar(vₕ))) === Float64
        @test eltype(values(D₋ₓ(vₕ))) === Float64
    end

    @testset "Restriction & cell-averaging" begin
        @test _matches_fd(a -> innerₕ(Rₕ(Wₕ1, x -> a * sin(x)),
            Rₕ(Wₕ1, x -> a * x)))
        @test _matches_fd(a -> innerₕ(avgₕ(Wₕ1, x -> a * sin(x)),
            Rₕ(Wₕ1, x -> a * x)))
        # the quadrature rule is the mesh's type at every order, not the field's
        for nq in (2, 3, 6)
            @test _matches_fd(a -> innerₕ(
                avgₕ(Wₕ1, x -> a * sin(x); quad_points = nq), Rₕ(Wₕ1, x -> a * x)))
        end

        # the in-place forms, writing into an element allocated at the Dual type
        @test _matches_fd(function (a)
            uₕ = element(Wₕ1, typeof(a))
            Rₕ!(uₕ, x -> a * sin(x))
            return innerₕ(uₕ, uₕ)
        end)
        @test _matches_fd(function (a)
            uₕ = element(Wₕ1, typeof(a))
            avgₕ!(uₕ, x -> a * sin(x))
            return innerₕ(uₕ, uₕ)
        end)

        # element built from a scalar, and from an existing coefficient vector
        @test _matches_fd(a -> innerₕ(element(Wₕ1, a), element(Wₕ1, a)))

        # markers restrict evaluation and still differentiate
        Ωm = mesh(domain(interval(0.0, 1.0), :left => :left, :right => :right), 21, true)
        Wm = gridspace(Ωm)
        @test _matches_fd(a -> innerₕ(
            avgₕ(Wm, x -> a * sin(x); markers = (:left, :right)), Rₕ(Wm, x -> a * x)))

        # `innerₕ`'s own `markers` keyword (not `avgₕ`'s, above), with one side Dual-valued
        # and the other Float64: `_dot_masked` promotes rather than requiring both sides
        # (and the weight vector) to already share one element type.
        @test _matches_fd(a -> innerₕ(
            element(Wm, a), Rₕ(Wm, x -> x); markers = (:left, :right)))
    end

    @testset "Difference, jump & average" begin
        for (nm, op) in (("D₋ₓ", D₋ₓ), ("D₊ₓ", D₊ₓ), ("diff₋ₓ", diff₋ₓ),
            ("diff₊ₓ", diff₊ₓ), ("jumpₓ", jumpₓ), ("M₋ₓ", M₋ₓ), ("M₊ₓ", M₊ₓ),
            ("Dstar₊ₓ", Dstar₊ₓ), ("Dcₓ", Dcₓ), ("Dₕₓ", Dₕₓ))
            @testset "$nm" begin
                @test _matches_fd(a -> innerₕ(
                    op(Rₕ(Wₕ1, x -> a * sin(x) + a^2 * x)), Rₕ(Wₕ1, x -> x)))
            end
        end

        # the vectorial forms, in 2D, consumed through the ₊ inner product
        for (nm, op) in (("∇₋ₕ", ∇₋ₕ), ("∇₊ₕ", ∇₊ₕ), ("Dcₕ", Dcₕ), ("∇ₕ", ∇ₕ),
            ("Dstar₊ₕ", Dstar₊ₕ), ("M₋ₕ", M₋ₕ), ("jumpₕ", jumpₕ))
            @testset "$nm" begin
                @test _matches_fd(function (a)
                    g = op(Rₕ(Wₕ2, x -> a * sin(x[1]) * x[2] + a^2 * x[1]))
                    return inner₊(g, g)
                end)
            end
        end
    end

    @testset "Inner products & norms" begin
        for (nm, f) in (("innerₕ", uₕ -> innerₕ(uₕ, uₕ)), ("normₕ", normₕ),
            ("snorm₁ₕ", snorm₁ₕ), ("norm₁ₕ", norm₁ₕ),
            ("inner₊", uₕ -> inner₊(uₕ, uₕ)), ("inner₊ₓ", uₕ -> inner₊ₓ(uₕ, uₕ)))
            @testset "$nm" begin
                @test _matches_fd(a -> f(Rₕ(Wₕ1,
                    x -> a * sin(x) + a^2 * x)))
            end
        end
        @test _matches_fd(a -> norm₊(∇₋ₕ(Rₕ(Wₕ2,
            x -> a * sin(x[1]) * x[2]))))
        @test _matches_fd(a -> inner₊ᵧ(Rₕ(Wₕ2, x -> a * x[1] * x[2]),
            Rₕ(Wₕ2, x -> a * x[2])))
    end

    @testset "Composite grid functions" begin
        @test _matches_fd(function (a)
            cₕ = Rₕ(Vₕ2, (x -> a * x[1], x -> a^2 * x[2]))
            dₕ = Dcₓ(cₕ)
            k1, k2 = components(dₕ)
            return innerₕ(k1, k1) + innerₕ(k2, k2)
        end)
    end

    @testset "Arithmetic & broadcasting" begin
        @test _matches_fd(function (a)
            uₕ = Rₕ(Wₕ1, x -> a * sin(x))
            vₕ = Rₕ(Wₕ1, x -> a * x)
            return innerₕ(uₕ + vₕ, 2 .* uₕ .* vₕ)
        end)
        @test _matches_fd(function (a)
            uₕ = Rₕ(Wₕ1, x -> a * sin(x))
            wₕ = similar(uₕ)
            wₕ .= uₕ
            return sum(wₕ) + innerₕ(copy(uₕ), uₕ)
        end)
    end

    @testset "Dual matrix products" begin
        # The matrices are built from the mesh, so they stay Float64; the product with a
        # Dual vector promotes.
        for (nm, op) in (("D₋ₓ", D₋ₓ), ("M₋ₓ", M₋ₓ), ("jumpₓ", jumpₓ))
            @testset "$nm" begin
                @test _matches_fd(a -> sum(op(Ωₕ1) *
                                           values(Rₕ(Wₕ1, x -> a * sin(x)))))
            end
        end
    end

    @testset "Multi-variable gradient" begin
        # Several parameters at once, which is the shape an optimisation actually has.
        function J(p)
            uₕ = Rₕ(Wₕ1, x -> p[1] * sin(x) + p[2] * x^2)
            return normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2
        end
        p0 = [1.3, 0.7]
        g = ForwardDiff.gradient(J, p0)
        @test length(g) == 2
        for k in 1:2
            e = zeros(2)
            e[k] = 1e-6
            @test isapprox(g[k], (J(p0 .+ e) - J(p0 .- e)) / 2e-6; rtol = 1e-5)
        end
    end
end
