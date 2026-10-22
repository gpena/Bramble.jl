module ConvergenceSpaceOperatorsTests

# Observed orders of the centred and cross-weighted differences, and of the Neumann Poisson
# problem through inner₊. Moved here from test/space/centered_difference.jl,
# cross_weighted_difference.jl and inner_plus_boundary.jl, whose value and identity tests
# stay in the unit group.

using Test
using Bramble
using Bramble: Dcₓ, D̽ₓ
using Random
using ..TestUtils: _observed_orders

@testset "Centered difference" begin
    @testset "Convergence order" begin
        # Second order on a uniform grid, and first order on a non-uniform one: the
        # centered difference approximates the derivative at the midpoint of its stencil,
        # which coincides with xᵢ only when the two spacings match.
        function orders(unif)
            Random.seed!(20260830)
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, unif)
            return first(_observed_orders(Ωₕ, Dcₓ, sin, cos, e -> e[2:(end - 1)]))
        end

        ou = orders(true)
        @test all(o -> abs(o - 2.0) < 0.05, ou)

        # Refinement halves every interval, so the non-uniform grids stay nested and the
        # ratio is an order even though the starting grid is random.
        orand = orders(false)
        @test all(>(0.95), orand)
        @test last(orand) < 1.15
    end
end

@testset "Cross-weighted difference" begin
    @testset "Convergence order" begin
        # Second order on both, which is the point: Dc is first order on a non-uniform
        # grid and this is not.
        function orders(unif)
            Random.seed!(20260830)
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, unif)
            return first(_observed_orders(Ωₕ, D̽ₓ, sin, cos, e -> e[2:(end - 1)]))
        end

        @test all(o -> abs(o - 2.0) < 0.05, orders(true))

        # The coarsest random pair is not yet asymptotic (measured 1.18 there against
        # 1.97 and better afterwards), so only the refined ones are held to second order.
        orand = orders(false)
        @test all(o -> abs(o - 2.0) < 0.1, orand[2:end])
        @test all(>(1.0), orand)
    end

    @testset "Second-order divergence, non-uniform" begin
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

@testset "inner₊ boundary" begin
    @testset "Neumann Poisson MMS: order 2" begin
        # -Δu = f, natural (unconstrained) traction-free boundary everywhere -- inner₊
        # encodes the Neumann condition without any dirichlet_bc! call. Pure Neumann is
        # singular up to an additive constant, pinned by fixing one dof to the exact value
        # there (a device, not a boundary condition of the continuum problem).
        uexact(x) = cos(pi * x[1]) * cos(pi * x[2])
        src(x) = 2 * pi^2 * cos(pi * x[1]) * cos(pi * x[2])

        function solve_neumann(n)
            Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (n, n), (true, true))
            Wₕ = gridspace(Ωₕ)
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
            A = assemble(a)
            F = assemble(l)
            A[1, :] .= 0.0
            A[1, 1] = 1.0
            F[1] = uexact(points(Ωₕ)[1])
            uₕ = element(Wₕ)
            uₕ .= A \ F
            return normₕ(uₕ - Rₕ(Wₕ, uexact)), norm₁ₕ(uₕ - Rₕ(Wₕ, uexact))
        end

        ns = (13, 25, 49, 97)  # each doubles the previous mesh spacing exactly
        errs_h = Float64[]
        errs_1h = Float64[]
        for n in ns
            eh, e1h = solve_neumann(n)
            push!(errs_h, eh)
            push!(errs_1h, e1h)
        end

        rate(e) = [log2(e[i] / e[i + 1]) for i in 1:(length(e) - 1)]
        @test all(>(1.9), rate(errs_h))
        @test all(>(1.9), rate(errs_1h))
    end
end

end # module ConvergenceSpaceOperatorsTests
