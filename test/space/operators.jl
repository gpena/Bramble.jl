module SpaceOperatorsTests

using Test
using Bramble
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
import Bramble: D₊ₓ, D₊ᵧ, D₊₂, D₊, div₊ₕ, curl₊ₕ, forward_star_difference
using Bramble:
               IdentityOperator,
               ZeroOperator,
               OperatorScale,
               GridFunctionScale,
               OperatorAdd,
               is_symbolic,
               space

@testset "Linear operators" begin
    for D in 1:3
        @testset "$(D)D" begin
            I = interval(0.0, 1.0)
            X = domain(reduce(×, ntuple(_ -> I, Val(D))))
            M = mesh(X, ntuple(_ -> 4, Val(D)), ntuple(_ -> false, Val(D)))
            W = gridspace(M)
            u = element(W)
            Rₕ!(u, x -> 1.0)

            x0 = IdentityOperator(W)
            @test space(x0) === W
            @test !is_symbolic(x0)

            z0 = ZeroOperator(W)
            @test space(z0) === W
            @test !is_symbolic(z0)

            # Scalar scaling
            x1 = 2 * x0
            @test x1 isa OperatorScale
            @test x1.scalar == 2
            @test x1.inner_op === x0
            @test !is_symbolic(x1)

            x1_div = x0 / 2
            @test x1_div isa OperatorScale
            @test x1_div.scalar == 0.5

            # Grid function scaling
            x2 = u * x0
            @test x2 isa GridFunctionScale
            @test x2.grid_function === u
            @test x2.inner_op === x0
            @test !is_symbolic(x2)

            # Operator addition and subtraction
            sum_op = x0 + x0
            @test sum_op isa OperatorAdd
            @test sum_op.left_op === x0
            @test sum_op.right_op === x0
            @test !is_symbolic(sum_op)

            diff_op = x0 - x0
            @test diff_op isa OperatorAdd
            @test diff_op.left_op === x0

            # String representation
            buf = IOBuffer()
            show(buf, x0)
            @test String(take!(buf)) == "I"
            show(buf, z0)
            @test String(take!(buf)) == "0"
        end
    end
end

# The discrete vector calculus operators (gpena/Bramble.jl#158).
#
# Each is a contraction of the directional differences, evaluated in one traversal rather
# than as nested operator calls, so the property that matters is that the fused form is the
# composition -- checked entry for entry against the operators it is built from, which are
# themselves pinned by `test/convergence/operators.jl`. Values alone would not catch a wrong
# boundary convention, since a truncated slice is self-consistently wrong.
@testset "Vector calculus operators (#158)" begin
    Ω1 = domain(interval(0.0, 1.0))
    Ω2 = domain(interval(0.0, 1.0) × interval(0.0, 2.0))
    Ω3 = domain(interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 1.5))

    @testset "Δₕ is D̽(D₋(·)) summed over directions" begin
        for (Ω, n, unif) in (
            (Ω1, 21, true), (Ω1, 17, false),
            (Ω2, (9, 8), (true, true)), (Ω2, (11, 9), (false, false)),
            (Ω3, (7, 6, 5), (true, true, true))
        )
            Ωₕ = mesh(Ω, n, unif)
            Wₕ = gridspace(Ωₕ)
            D = dim(Ωₕ)
            uₕ = Rₕ(Wₕ,
                x -> sin(1.3 * x[1]) + (D > 1 ? cos(0.9 * x[2]) : 0.0) +
                     (D > 2 ? x[3]^2 : 0.0))

            ref = zeros(ndofs(Wₕ))
            for d in 1:D
                ref .+= parent(forward_star_difference(D₋(uₕ, Val(d)), Val(d)))
            end
            # the accumulation order differs from the composition's, so 3D agrees to
            # round-off rather than to the bit
            @test parent(Δₕ(uₕ)) ≈ ref atol=1e-14
        end
    end

    @testset "Δₕ! writes what Δₕ returns, and refuses to alias" begin
        Wₕ = gridspace(mesh(Ω2, (9, 8), (true, true)))
        uₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])
        vₕ = element(Wₕ)
        @test parent(Δₕ!(vₕ, uₕ)) == parent(Δₕ(uₕ))
        @test Δₕ!(vₕ, uₕ) === vₕ
        @test_throws ArgumentError Δₕ!(uₕ, uₕ)
    end

    @testset "The summation-by-parts identity holds" begin
        # innerₕ(Δₕ u, v) = -inner₊(∇ₕ u, ∇ₕ v) for grid functions vanishing on the
        # boundary. This is what makes this composition *the* discrete Laplacian rather than
        # one of several plausible five-point stencils, and it fails for a stencil that is
        # off by a spacing anywhere.
        for (n, unif) in (((11, 9), (true, true)), ((13, 11), (false, false)))
            Wₕ = gridspace(mesh(Ω2, n, unif))
            uₕ = Rₕ(Wₕ, x -> sin(π * x[1]) * sin(π * x[2] / 2))
            # deliberately not a mode orthogonal to `uₕ`: with two orthogonal modes both
            # sides are zero and the identity holds vacuously
            vₕ = Rₕ(Wₕ,
                x -> x[1] * (1 - x[1]) * x[2] * (2 - x[2]) * (1 + 0.5x[1] + 0.3x[2]))
            on_boundary = index_in_marker(mesh(Wₕ), :boundary)
            parent(uₕ)[on_boundary] .= 0.0
            parent(vₕ)[on_boundary] .= 0.0
            lhs = innerₕ(Δₕ(uₕ), vₕ)
            @test abs(lhs) > 1e-3
            @test lhs ≈ -inner₊(∇ₕ(uₕ), ∇ₕ(vₕ)) rtol=1e-12
        end
    end

    @testset "divₕ sums the directional differences" begin
        Ωₕ = mesh(Ω2, (9, 8), (true, true))
        Wₕ = gridspace(Ωₕ)
        u1 = Rₕ(Wₕ, x -> x[1]^2 + x[2])
        u2 = Rₕ(Wₕ, x -> sin(x[1]) * x[2])

        @test parent(divₕ((u1, u2))) == parent(D₋ₓ(u1)) .+ parent(D₋ᵧ(u2))
        @test parent(div₊ₕ((u1, u2))) == parent(D₊ₓ(u1)) .+ parent(D₊ᵧ(u2))

        # a composite grid function is the same field, spelled the other way
        Vₕ = gridspace(Ωₕ, Val(2))
        cₕ = Rₕ(Vₕ, (x -> x[1]^2 + x[2], x -> sin(x[1]) * x[2]))
        @test parent(divₕ(cₕ)) == parent(divₕ((u1, u2)))

        # and the in-place form writes the same thing
        wₕ = element(Wₕ)
        @test parent(divₕ!(wₕ, (u1, u2))) == parent(divₕ((u1, u2)))

        # one component per dimension, or it is an error rather than a silent answer
        @test_throws DimensionMismatch divₕ((u1,))
        @test_throws DimensionMismatch divₕ((u1, u2, u1))
    end

    @testset "curlₕ in 2D and 3D" begin
        Ωₕ = mesh(Ω2, (9, 8), (true, true))
        Wₕ = gridspace(Ωₕ)
        u1 = Rₕ(Wₕ, x -> x[1]^2 + x[2])
        u2 = Rₕ(Wₕ, x -> sin(x[1]) * x[2])
        @test parent(curlₕ((u1, u2))) == parent(D₋ₓ(u2)) .- parent(D₋ᵧ(u1))
        @test parent(curl₊ₕ((u1, u2))) == parent(D₊ₓ(u2)) .- parent(D₊ᵧ(u1))

        # curl of a gradient vanishes where both stencils are untruncated -- a discrete
        # identity, not an approximation, since both differences are backward
        w = Rₕ(Wₕ, x -> sin(x[1]) * cos(x[2]))
        interior = reshape(parent(curlₕ(∇ₕ(w))), (9, 8))[3:end, 3:end]
        @test maximum(abs, interior) < 1e-14

        # 3D: three components, each the right combination
        Ω3ₕ = mesh(Ω3, (8, 7, 6), (true, true, true))
        W3 = gridspace(Ω3ₕ)
        f = (Rₕ(W3, x -> sin(x[1]) * x[2]), Rₕ(W3, x -> x[3]^2),
            Rₕ(W3, x -> cos(x[2]) * x[1]))
        c = curlₕ(f)
        @test length(c) == 3
        @test parent(c[1]) == parent(D₋ᵧ(f[3])) .- parent(D₋₂(f[2]))
        @test parent(c[2]) == parent(D₋₂(f[1])) .- parent(D₋ₓ(f[3]))
        @test parent(c[3]) == parent(D₋ₓ(f[2])) .- parent(D₋ᵧ(f[1]))

        # the in-place 3D form writes into a 3-tuple
        dest = ntuple(_ -> element(W3), Val(3))
        curlₕ!(dest, f)
        @test all(k -> parent(dest[k]) == parent(c[k]), 1:3)

        # there is no 1D curl
        W1 = gridspace(mesh(Ω1, 9, true))
        @test_throws ArgumentError curlₕ((Rₕ(W1, sin),))
    end
end

end # module SpaceOperatorsTests
