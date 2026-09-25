module SpaceCrossWeightedDifferenceTests

using Test
using Bramble
using Bramble: Dcₓ, D̽ᵧ, D̽₂, D̽ₓ, D₋ₓ, VectorElement
using Random
using Bramble: components
using Bramble: div̽ₕ, div̽ₕ!, curl̽ₕ, curl̽ₕ!, ε̽ₕ, ε̽ₕ!, ∇̽ₕ!
using ..TestUtils: alloc_test
using ..SpaceDifferenceTests: test_operator_matrix_equivalence

# The cross-weighted centered difference.
#
#   D̽ₕ(uₕ)(i) = (h_i / (h_i + h_{i+1})) D₋(uₕ)(x_{i+1})
#             + (h_{i+1} / (h_i + h_{i+1})) D₋(uₕ)(x_i)
#
# The same two one-sided differences the centered difference combines, weighted by the
# opposite spacings. That swap is what makes it second order on a non-uniform grid where
# Dc is first, and the two coincide when the spacing is constant.
#
# The property underneath the order is that it differences a quadratic exactly on any
# grid: with u = x², D₋(i) = x_i + x_{i-1}, and the weighted sum telescopes to
# 2x_i(h_i + h_{i+1}) over h_i + h_{i+1}. Dc does not do this unless the grid is uniform,
# which is the whole difference between the two operators.

# The operators as matrices, for `test_operator_matrix_equivalence` (test/space/difference.jl).
cross_weighted_ops(::Val{1}) = (D̽ₓ,)
cross_weighted_ops(::Val{2}) = (D̽ₓ, D̽ᵧ)
cross_weighted_ops(::Val{3}) = (D̽ₓ, D̽ᵧ, D̽₂)

@testset "Cross-weighted difference" begin
    @testset "Definition match" begin
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, unif)
                Wₕ = gridspace(Ωₕ)
                n = npoints(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> x^2 + sin(x))
                dm = parent(D₋ₓ(uₕ))
                h = [spacing(Ωₕ, i) for i in 1:n]

                u = parent(uₕ)
                want = [if i == 1
                            (u[2] - u[1]) / h[1]
                        elseif i == n
                            (u[n] - u[n - 1]) / h[n]
                        else
                            (h[i] / (h[i] + h[i + 1])) * dm[i + 1] +
                            (h[i + 1] / (h[i] + h[i + 1])) * dm[i]
                        end
                        for i in 1:n]
                @test parent(D̽ₓ(uₕ)) ≈ want
            end
        end
    end

    @testset "Boundary is one-sided, not truncated (#183)" begin
        # gpena/Bramble.jl#183: D̽ₕ used to truncate both ends to zero; it now falls back
        # to the one-sided difference the near side still defines, so nothing here is
        # zero for a function with no flat point.
        Random.seed!(20260830)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, false))
        Wₕ = gridspace(Ωₕ)
        n = npoints(Ωₕ, Tuple)
        uₕ = Rₕ(Wₕ, x -> exp(x[1]) * (x[2] + 1))
        u = reshape(parent(uₕ), n)

        rx = reshape(parent(D̽ₓ(uₕ)), n)
        @test !any(iszero, rx)
        hx = [spacing(Ωₕ(1), i) for i in 1:n[1]]
        @test rx[1, :] ≈ (u[2, :] .- u[1, :]) ./ hx[1]
        @test rx[end, :] ≈ (u[end, :] .- u[end - 1, :]) ./ hx[end]

        ry = reshape(parent(D̽ᵧ(uₕ)), n)
        @test !any(iszero, ry)
        hy = [spacing(Ωₕ(2), i) for i in 1:n[2]]
        @test ry[:, 1] ≈ (u[:, 2] .- u[:, 1]) ./ hy[1]
        @test ry[:, end] ≈ (u[:, end] .- u[:, end - 1]) ./ hy[end]
    end

    @testset "Uniform Dc agreement" begin
        # Both are the mean of D₋ and D₊ in the interior; they part company there only
        # where the two spacings differ. At the boundary they now differ regardless of
        # spacing: Dc still truncates to zero, D̽ₕ falls back to a one-sided difference
        # (gpena/Bramble.jl#183).
        Ωu = mesh(domain(interval(0.0, 1.0)), 21, true)
        n = npoints(Ωu)
        uu = Rₕ(gridspace(Ωu), x -> sin(3x))
        dh, dc = parent(D̽ₓ(uu)), parent(Dcₓ(uu))
        @test dh[2:(n - 1)] ≈ dc[2:(n - 1)]
        @test !(dh[1] ≈ dc[1]) && !(dh[n] ≈ dc[n])

        Random.seed!(20260830)
        Ωr = mesh(domain(interval(0.0, 1.0)), 21, false)
        ur = Rₕ(gridspace(Ωr), x -> sin(3x))
        @test !isapprox(parent(D̽ₓ(ur)), parent(Dcₓ(ur)))
    end

    @testset "Exact on quadratics" begin
        # The property that separates it from Dc, and the reason for the second order
        # below. Dc reproduces affine functions on any grid; this reproduces quadratics.
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 15, unif)
                Wₕ = gridspace(Ωₕ)
                n = npoints(Ωₕ)
                x = points(Ωₕ)

                q = parent(D̽ₓ(Rₕ(Wₕ, t -> 5t^2 - 2t + 1)))
                @test all(q[i] ≈ 10x[i] - 2 for i in 2:(n - 1))

                # a cubic is not reproduced, so the test above is not vacuous
                c = parent(D̽ₓ(Rₕ(Wₕ, t -> t^3)))
                @test !all(c[i] ≈ 3x[i]^2 for i in 2:(n - 1))

                # and on a non-uniform grid Dc misses the quadratic, which is what the
                # cross weighting fixes
                unif || @test !all(
                    parent(Dcₓ(Rₕ(Wₕ, t -> 5t^2 - 2t + 1)))[i] ≈ 10x[i] - 2 for
                i in 2:(n - 1)
                )
            end
        end

        # every direction, in three dimensions
        Random.seed!(20260830)
        Ω3 = mesh(
            domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 6, 4), (false, false, false)
        )
        W3 = gridspace(Ω3)
        n3 = npoints(Ω3, Tuple)
        for (d, op) in ((1, D̽ₓ), (2, D̽ᵧ), (3, D̽₂))
            @test all(iszero, parent(op(Rₕ(W3, x -> x[mod1(d + 1, 3)]))))
            r = reshape(parent(op(Rₕ(W3, x -> x[d]^2))), n3)
            xd = points(Ω3)[d]
            interior = CartesianIndices(
                ntuple(k -> k == d ? (2:(n3[k] - 1)) : (1:n3[k]), 3)
            )
            @test all(r[I] ≈ 2 * xd[I[d]] for I in interior)
        end
    end

    @testset "Convergence order" begin
        # Second order on both, which is the point: Dc is first order on a non-uniform
        # grid and this is not.
        function orders(unif; steps = 4)
            Random.seed!(20260830)
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, unif)
            errs = Float64[]
            for k in 0:steps
                k > 0 && iterative_refinement!(Ωₕ)
                Wₕ = gridspace(Ωₕ)
                e = parent(D̽ₓ(Rₕ(Wₕ, sin))) .- parent(Rₕ(Wₕ, cos))
                push!(errs, maximum(abs, e[2:(end - 1)]))
            end
            return [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]
        end

        @test all(o -> abs(o - 2.0) < 0.05, orders(true))

        # The coarsest random pair is not yet asymptotic (measured 1.18 there against
        # 1.97 and better afterwards), so only the refined ones are held to second order.
        orand = orders(false)
        @test all(o -> abs(o - 2.0) < 0.1, orand[2:end])
        @test all(>(1.0), orand)
    end

    @testset "Directional family" begin
        Random.seed!(20260830)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(2))
        uₕ = Rₕ(Wₕ, x -> x[1] * x[2])

        @test D̽ₕ(uₕ) isa NTuple{2, VectorElement}
        @test parent(D̽ₕ(uₕ)[1]) == parent(D̽ₓ(uₕ))
        @test parent(D̽ₕ(uₕ)[2]) == parent(D̽ᵧ(uₕ))

        # in one dimension the tuple and the grid function coincide, as for ∇ₕ
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, true)
        u1 = Rₕ(gridspace(Ω1), sin)
        @test !(D̽ₕ(u1) isa Tuple)
        @test parent(D̽ₕ(u1)) == parent(D̽ₓ(u1))

        # composite grid functions apply componentwise, as the other operators do
        fs = (x -> x[1], x -> x[2]^2)
        cₕ = Rₕ(Vₕ, fs)
        scalars = (Rₕ(Wₕ, fs[1]), Rₕ(Wₕ, fs[2]))
        rₕ = D̽ₓ(cₕ)
        @test length(parent(rₕ)) == length(parent(cₕ))
        for k in 1:2
            @test parent(components(rₕ)[k]) == parent(D̽ₓ(scalars[k]))
        end
    end

    @testset "Type stability and allocations" begin
        Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 33, false)
        Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 8), (true, false))
        u1 = Rₕ(gridspace(Ωₕ1), sin)
        u2 = Rₕ(gridspace(Ωₕ2), x -> x[1] * x[2])

        @test @inferred(D̽ₓ(u1)) isa VectorElement
        @test @inferred(D̽ᵧ(u2)) isa VectorElement
        @test @inferred(D̽ₕ(u2)) isa NTuple{2, VectorElement}

        @test alloc_test(D̽ₓ, u1) == alloc_test(similar, u1)
        @test alloc_test(D̽ᵧ, u2) == alloc_test(similar, u2)
    end

    @testset "Matrix agreement" begin
        # Not one diagonal scaling of one difference, unlike the other two: it is the two
        # one-sided differences it is defined from, each under its own weight:
        # `diag(hᵢ/((hᵢ+hᵢ₊₁)hᵢ₊₁))·diff₊ + diag(hᵢ₊₁/((hᵢ+hᵢ₊₁)hᵢ))·diff₋`. That is worth
        # checking against the kernel rather than trusting the algebra.
        test_operator_matrix_equivalence(cross_weighted_ops)

        Ωm = mesh(domain(interval(0.0, 1.0)), 7, false)
        @test D̽ₓ(gridspace(Ωm)) == D̽ₓ(Ωm)

        n = npoints(Ωm)
        M = Matrix(D̽ₓ(Ωm))
        h = [spacing(Ωm, i) for i in 1:n]
        # no truncated row: row 1 is D₊ at the first point, row n is D₋ at the last
        # (gpena/Bramble.jl#183)
        @test count(!iszero, M[1, :]) == 2
        @test M[1, 1:2] ≈ [-1, 1] ./ h[1]
        @test count(!iszero, M[n, :]) == 2
        @test M[n, (n - 1):n] ≈ [-1, 1] ./ h[n]
        # three points wide in the interior, where the ends are two
        @test count(!iszero, M[4, :]) == 3

        @test_throws ArgumentError D̽ₓ(mesh(domain(interval(0.0, 1.0)), 2, true))
    end
end

# The cross-weighted vector calculus (gpena/Bramble.jl#349): div̽ₕ, curl̽ₕ, ε̽ₕ and ∇̽ₕ! are
# contractions of the co-located D̽ differences, so every identity below is checked against
# the coordinate operators themselves, on meshes non-uniform in every direction.
function _cw_setup(D)
    Random.seed!(349)
    dom = D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 2.0)) :
          domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    Ωₕ = mesh(dom, (9, 8, 7)[1:D], ntuple(_ -> false, D))
    Wₕ = gridspace(Ωₕ)
    fs = D == 2 ? (x -> x[1]^2 * x[2], x -> 100 * sin(x[1] + x[2])) :
         (x -> x[1] * x[2], x -> 100 * (x[3]^2 + x[1]), x -> 10000 * sin(x[2]) * x[3])
    return Ωₕ, Wₕ, map(f -> Rₕ(Wₕ, f), fs), fs
end

_cw_ops(::Val{2}) = (D̽ₓ, D̽ᵧ)
_cw_ops(::Val{3}) = (D̽ₓ, D̽ᵧ, D̽₂)

@testset "Cross-weighted vector calculus (#349)" begin
    @testset "Component-wise identities, $(D)D" for D in 2:3
        Ωₕ, Wₕ, u, fs = _cw_setup(D)
        ops = _cw_ops(Val(D))
        d(k, c) = parent(ops[k](u[c]))

        @test parent(div̽ₕ(u)) ≈ sum(d(k, k) for k in 1:D)
        # never zero at the ends in 2D, where no component has a flat point: no truncated
        # slice, unlike divcₕ
        D == 2 && @test !any(iszero, parent(div̽ₕ(u)))

        cu = curl̽ₕ(u)
        if D == 2
            @test parent(cu) ≈ d(1, 2) .- d(2, 1)
        else
            @test parent(cu[1]) ≈ d(2, 3) .- d(3, 2)
            @test parent(cu[2]) ≈ d(3, 1) .- d(1, 3)
            @test parent(cu[3]) ≈ d(1, 2) .- d(2, 1)
        end

        ε = ε̽ₕ(u)
        for i in 1:D, j in 1:D

            @test parent(ε[i][j]) ≈ (i == j ? d(i, i) : (d(j, i) .+ d(i, j)) ./ 2)
            @test parent(ε[i][j]) == parent(ε[j][i])
        end

        g = ∇̽ₕ(u[1])
        for k in 1:D
            @test parent(g[k]) == d(k, 1)
        end

        # a composite grid function is the same field as the tuple of its leaves
        uc = Rₕ(gridspace(Ωₕ, Val(D)), fs)
        @test parent(div̽ₕ(uc)) ≈ parent(div̽ₕ(u))
        @test parent(ε̽ₕ(uc)[1][2]) ≈ parent(ε[1][2])

        @test_throws DimensionMismatch div̽ₕ(ntuple(_ -> u[1], D + 1))
    end

    @testset "In-place forms, $(D)D" for D in 2:3
        _, Wₕ, u, _ = _cw_setup(D)

        v = element(Wₕ, 0.0)
        @test div̽ₕ!(v, u) === v
        @test parent(v) == parent(div̽ₕ(u))
        @test alloc_test(div̽ₕ!, v, u) == 0

        c = D == 2 ? element(Wₕ, 0.0) : ntuple(_ -> element(Wₕ, 0.0), 3)
        @test curl̽ₕ!(c, u) === c
        want = curl̽ₕ(u)
        @test D == 2 ? parent(c) == parent(want) : all(parent.(c) .== parent.(want))
        @test alloc_test(curl̽ₕ!, c, u) == 0

        dest = ntuple(_ -> ntuple(_ -> element(Wₕ, 0.0), D), D)
        @test ε̽ₕ!(dest, u) === dest
        εu = ε̽ₕ(u)
        @test all(parent(dest[i][j]) == parent(εu[i][j]) for i in 1:D, j in 1:D)
        @test alloc_test(ε̽ₕ!, dest, u) == 0

        g = ntuple(_ -> element(Wₕ, 0.0), D)
        @test ∇̽ₕ!(g, u[1]) === g
        @test all(parent(g[k]) == parent(∇̽ₕ(u[1])[k]) for k in 1:D)
        @test alloc_test(∇̽ₕ!, g, u[1]) == 0
    end

    @testset "1D and curl dimension" begin
        Ω1 = mesh(domain(interval(0.0, 1.0)), 9, false)
        u1 = Rₕ(gridspace(Ω1), x -> x^3)
        @test parent(div̽ₕ(u1)) == parent(D̽ₓ(u1))
        @test_throws ArgumentError curl̽ₕ(u1)

        # a wrong-arity field is reported under the name the caller used
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (false, false))
        u2 = Rₕ(gridspace(Ω2), x -> x[1] * x[2])
        @test_throws DimensionMismatch("curl̽ₕ needs one component per spatial dimension: " *
                                       "got 3 components on a 2D mesh.") curl̽ₕ((u2, u2, u2))
        @test_throws DimensionMismatch("curl̽ₕ! needs one component per spatial dimension: " *
                                       "got 3 components on a 2D mesh.") curl̽ₕ!(similar(u2), (u2, u2, u2))
    end

    @testset "Second-order divergence on a non-uniform grid" begin
        # The leading error of D̽ is h_i h_{i+1} u'''/6, second order on any grid, where
        # Dc's is (h_{i+1} - h_i) u''/2, first order on a random one. The end slices are
        # one-sided, so first order, and are left out of the measured error. The rate is a
        # least-squares slope over five random meshes, since no two share a grading.
        F = (x -> sin(2x[1]) * cos(x[2]), x -> exp(x[1]) * x[2]^3)
        divF = x -> 2cos(2x[1]) * cos(x[2]) + 3exp(x[1]) * x[2]^2
        function rate(op)
            errs, hs = Float64[], Float64[]
            for n in (17, 33, 65, 129, 257)
                Random.seed!(349)
                Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n), (false, false))
                Wₕ = gridspace(Ωₕ)
                r = reshape(parent(op(map(f -> Rₕ(Wₕ, f), F))) .- parent(Rₕ(Wₕ, divF)), (n, n))
                push!(errs, maximum(abs, @view r[2:(end - 1), 2:(end - 1)]))
                push!(hs, maximum(maximum(spacing(Ωₕ(k), i) for i in 2:n) for k in 1:2))
            end
            lh, le = log.(hs), log.(errs)
            lh, le = lh .- sum(lh) / length(lh), le .- sum(le) / length(le)
            return sum(lh .* le) / sum(abs2, lh)
        end
        @test rate(div̽ₕ) > 1.7
        @test rate(divcₕ) < 1.2
    end
end

end # module SpaceCrossWeightedDifferenceTests
