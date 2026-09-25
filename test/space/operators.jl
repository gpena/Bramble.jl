module SpaceOperatorsTests

using Test
using Bramble
using Bramble: divₕ!, curlₕ!, Δₕ!
using Bramble: D₋
using Bramble: Dcᵧ, Dc₂, Dcₓ, D̃ᵧ, D̃₂, D̃ₓ, D̽ᵧ, D̽₂, D̽ₓ, D₋ᵧ, D₋₂, D₋ₓ, Mᵧ, M₂, Mₓ
using Bramble: index_in_marker, jumpᵧ, jump₂, jumpₓ
using SparseArrays: SparseMatrixCSC, nnz
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
import Bramble: D₊ₓ, D₊ᵧ, D₊₂, D₊, div₊ₕ, curl₊ₕ, forward_star_difference
# `M₊*` is `public`, not `export`ed (average.jl's own note on why); `kronecker_operator_matrix`
# is neither, the oracle `stencil_matrix` (gpena/Bramble.jl#185) is checked against.
import Bramble: M₊ₓ, M₊ᵧ, M₊₂, kronecker_operator_matrix
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

    @testset "Δₕ is D̃(D₋(·)) summed over directions" begin
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

# `stencil_matrix` (gpena/Bramble.jl#185): every family's public per-axis alias now
# routes through the single-pass builder in `src/space/operators/stencil.jl`, rather than
# through the Kronecker products of shift matrices `kronecker_operator_matrix` still
# builds (`src/space/operators/shift.jl`, kept as the retained oracle). Checked entrywise,
# `nnz` included, on non-uniform meshes in 1D/2D/3D so a boundary weight that would only
# coincidentally match on a uniform grid cannot hide a mistake.
@testset "stencil_matrix agrees with the Kronecker oracle (#185)" begin
    meshes = (
        mesh(domain(interval(0.0, 1.0)), 11, false),
        mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (9, 7), false),
        mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (6, 5, 4), false)
    )
    families = (
        (:D₋, (D₋ₓ, D₋ᵧ, D₋₂)),
        (:D₊, (D₊ₓ, D₊ᵧ, D₊₂)),
        (:D̃, (D̃ₓ, D̃ᵧ, D̃₂)),
        (:Dc, (Dcₓ, Dcᵧ, Dc₂)),
        (:D̽, (D̽ₓ, D̽ᵧ, D̽₂)),
        (:jump, (jumpₓ, jumpᵧ, jump₂)),
        (:M, (Mₓ, Mᵧ, M₂)),
        (:M₊, (M₊ₓ, M₊ᵧ, M₊₂))
    )
    for Ωₕ in meshes, (name, ops) in families, d in 1:dim(Ωₕ)
        op = ops[d]
        @testset "$name axis $d, $(dim(Ωₕ))D" begin
            A = op(Ωₕ)
            B = kronecker_operator_matrix(Ωₕ, op)
            @test A isa SparseMatrixCSC
            @test A == B
            @test nnz(A) == nnz(B)
        end
    end
end

# The vectorial aliases (`∇ₕ`, `Mₕ`, `jumpₕ`, ...) destructure and index into the
# per-coordinate aliases they already apply through (#340): `dx, dy = ∇ₕ` and `∇ₕ[1]`,
# `∇ₕ[:x]` are the exact same function object as `D₋ₓ`, not a copy, and the protocol is
# generated once in `@operator_family` for every family with a `vectorial_alias`. Coordinate
# names are due to be demoted from `export` to `public` (#340), so this file reaches them
# through `Bramble.` rather than relying on the bare name staying exported.
@testset "Vectorial operator aliases destructure and index (#340)" begin
    Dₓ, Dᵧ, D₂ = Bramble.D₋ₓ, Bramble.D₋ᵧ, Bramble.D₋₂

    @testset "destructuring and indexing agree with the named aliases" begin
        dx, dy = ∇ₕ
        @test dx === Dₓ && dy === Dᵧ

        dx3, dy3, dz3 = ∇ₕ
        @test (dx3, dy3, dz3) === (Dₓ, Dᵧ, D₂)

        @test ∇ₕ[1] === Dₓ && ∇ₕ[2] === Dᵧ && ∇ₕ[3] === D₂
        @test ∇ₕ[:x] === Dₓ && ∇ₕ[:y] === Dᵧ && ∇ₕ[:z] === D₂
        @test firstindex(∇ₕ) == 1 && lastindex(∇ₕ) == 3
        @test length(∇ₕ) == 3
        @test eltype(∇ₕ) === Function
        @test collect(∇ₕ) == [Dₓ, Dᵧ, D₂]

        @test_throws BoundsError ∇ₕ[0]
        @test_throws BoundsError ∇ₕ[4]
        @test_throws ArgumentError ∇ₕ[:w]
    end

    @testset "indexing folds at compile time with zero allocations" begin
        second(V) = V[2]
        @test only(Base.return_types(second, (typeof(∇ₕ),))) === typeof(Dᵧ)
        @test @inferred(second(∇ₕ)) === Dᵧ
        second(∇ₕ) # warm up before measuring
        @test (@allocated second(∇ₕ)) == 0
    end

    @testset "every family with a vectorial_alias supports the protocol" begin
        families = (
            (∇ₕ, (Bramble.D₋ₓ, Bramble.D₋ᵧ, Bramble.D₋₂)),
            (∇cₕ, (Bramble.Dcₓ, Bramble.Dcᵧ, Bramble.Dc₂)),
            (∇̽ₕ, (Bramble.D̽ₓ, Bramble.D̽ᵧ, Bramble.D̽₂)),
            (∇̃ₕ, (Bramble.D̃ₓ, Bramble.D̃ᵧ, Bramble.D̃₂)),
            (Mₕ, (Bramble.Mₓ, Bramble.Mᵧ, Bramble.M₂)),
            (Mcₕ, (Bramble.Mcₓ, Bramble.Mcᵧ, Bramble.Mc₂)),
            (jumpₕ, (Bramble.jumpₓ, Bramble.jumpᵧ, Bramble.jump₂)),
            (Bramble.∇₊ₕ, (D₊ₓ, D₊ᵧ, D₊₂)),
            (Bramble.M₊ₕ, (M₊ₓ, M₊ᵧ, M₊₂))
        )
        for (V, ops) in families
            a, b, c = V
            @test (a, b, c) === ops
            @test (V[1], V[2], V[3]) === ops
            @test (V[:x], V[:y], V[:z]) === ops
            @test length(V) == 3
        end
    end

    # non-uniform in every direction: uniform is only a special case
    Ω2ₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (9, 8), (false, false))
    W2 = gridspace(Ω2ₕ)

    @testset "a destructured operator works on a VectorElement" begin
        uₕ = Rₕ(W2, x -> x[1]^2 * sin(x[2]))
        dx, dy = ∇ₕ
        @test parent(dx(uₕ)) == parent(Dₓ(uₕ))
        @test parent(dy(uₕ)) == parent(Dᵧ(uₕ))
    end

    @testset "a destructured operator works inside form(...)" begin
        dx, dy = ∇ₕ
        a1 = assemble(form(W2, W2, (u, v) -> innerₕ(dx(u), dx(v)) + innerₕ(dy(u), dy(v))))
        a2 = assemble(form(W2, W2, (u, v) -> innerₕ(Dₓ(u), Dₓ(v)) + innerₕ(Dᵧ(u), Dᵧ(v))))
        @test a1 == a2
    end

    @testset "3D destructuring" begin
        Ω3ₕ = mesh(
            domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (6, 5, 4), (false, false, false)
        )
        W3 = gridspace(Ω3ₕ)
        u3 = Rₕ(W3, x -> x[1] * x[2] + x[3]^2)
        dx3, dy3, dz3 = ∇ₕ
        @test parent(dx3(u3)) == parent(Dₓ(u3))
        @test parent(dy3(u3)) == parent(Dᵧ(u3))
        @test parent(dz3(u3)) == parent(D₂(u3))
    end
end

end # module SpaceOperatorsTests
