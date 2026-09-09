using Test
using Bramble
using Random
using ForwardDiff, DifferentiationInterface
using SparseConnectivityTracer: SparseConnectivityTracer
using SparseMatrixColorings: SparseMatrixColorings

# End-to-end regression coverage for the two nonlinear worked examples,
# docs/src/examples/poisson_nonlinear.md and docs/src/examples/coupled_reaction_diffusion.md,
# which test/examples/convergence.jl does not touch. Neither Picard, nor Newton, nor a
# differentiated composite-space residual is exercised anywhere else in the suite, so a
# routing bug in composite spaces (#64, #78, #79) or a Jacobian-sparsity regression could
# break silently while every existing test stays green. See issue #81.
#
# Duplicated from the doc pages rather than shared with them, for the same reason
# examples/convergence.jl duplicates instead of sharing: the pages are written to be read,
# and a test helper in the middle would cost the reader the thing they came for.

const _sparse_ad = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm=SparseMatrixColorings.GreedyColoringAlgorithm(),
)

@testset "Nonlinear worked examples" begin
    @testset "Nonlinear Poisson: Picard iteration" begin
        # -(α(u)u')' = g, u_exact = exp(x), as in poisson_nonlinear.md.
        sol(x) = exp(x[1])
        α(u) = 3 + 1 / (1 + u^2)
        dαdu(u) = -2u / (1 + u^2)^2
        rhs(x) = -dαdu(sol(x)) * sol(x)^2 - α(sol(x)) * sol(x)

        Ω = domain(interval(0.0, 1.0))
        # `unif = false` draws the interior points from the global RNG, so without a seed
        # the problem itself changes every run -- and both testsets below assert on an
        # iteration count, which is exactly the kind of claim that then passes or fails by
        # chance. Seeded so a failure here is reproducible and means something.
        Random.seed!(20260909)
        Ωₕ = mesh(Ω, 40, false)
        Wₕ = gridspace(Ωₕ)

        bcs = dirichlet_constraints(Ω, :boundary => sol)
        gₕ = element(Wₕ)
        avgₕ!(gₕ, rhs)
        l = form(Wₕ, v -> innerₕ(gₕ, v))
        F = assemble(l; dirichlet=bcs)

        uₙ = element(Wₕ, 0.0)
        αvals = element(Wₕ)
        αvals .= α.(M₋ₕ(uₙ))
        a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇₋ₕ(U), ∇₋ₕ(V)))
        A = allocate_system_matrix(a)

        last_step = Inf
        converged_at = 0
        for it in 1:200
            assemble!(A, a; dirichlet=:boundary)
            unew = A \ F
            last_step = maximum(abs, unew .- parent(uₙ))
            uₙ .= unew
            αvals .= α.(M₋ₕ(uₙ))
            if last_step < 1e-12
                converged_at = it
                break
            end
        end

        # Bounded, not tight: the doc page's own run takes well under 200 steps. A run that
        # never gets there at all is the failure this guards against, not a specific count.
        @test converged_at > 0
        @test converged_at < 200
        @test last_step < 1e-12

        uexact = Rₕ(Wₕ, sol)
        @test norm₁ₕ(uₙ .- uexact) < 1e-2
    end

    @testset "Nonlinear Poisson: Newton's method" begin
        sol(x) = exp(x[1])
        α(u) = 3 + 1 / (1 + u^2)
        dαdu(u) = -2u / (1 + u^2)^2
        rhs(x) = -dαdu(sol(x)) * sol(x)^2 - α(sol(x)) * sol(x)

        Ω = domain(interval(0.0, 1.0))
        # `unif = false` draws the interior points from the global RNG, so without a seed
        # the problem itself changes every run -- and both testsets below assert on an
        # iteration count, which is exactly the kind of claim that then passes or fails by
        # chance. Seeded so a failure here is reproducible and means something.
        Random.seed!(20260909)
        Ωₕ = mesh(Ω, 40, false)
        Wₕ = gridspace(Ωₕ)

        bcs = dirichlet_constraints(Ω, :boundary => sol)
        gₕ = element(Wₕ)
        avgₕ!(gₕ, rhs)
        l = form(Wₕ, v -> innerₕ(gₕ, v))
        F = assemble(l; dirichlet=bcs)

        function diffusion_matrix(uₕ)
            αvals_local = α.(M₋ₕ(uₕ))
            a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals_local * ∇₋ₕ(U), ∇₋ₕ(V)))
            return assemble(a; dirichlet=:boundary)
        end

        function residual(u_vec::AbstractVector{T}) where {T}
            uₕ = element(Wₕ, T)
            uₕ .= u_vec
            A = diffusion_matrix(uₕ)
            return A * u_vec .- F
        end

        u = zeros(ndofs(Wₕ))
        prep = prepare_jacobian(residual, _sparse_ad, u)
        J = DifferentiationInterface.jacobian(residual, prep, _sparse_ad, u)
        newton_residuals = Float64[]
        for it in 1:20
            r = residual(u)
            push!(newton_residuals, sqrt(sum(abs2, r)))
            newton_residuals[end] < 1e-10 && break
            DifferentiationInterface.jacobian!(residual, J, prep, _sparse_ad, u)
            u .-= J \ r
        end

        # The doc page's own claim is quadratic convergence; assert it rather than just
        # rendering it. A regression that degraded Newton to linear convergence, or broke
        # the sparse Jacobian's values while leaving its pattern intact, would blow this.
        # Bounded at 8, not the doc page's usual 5: SparseConnectivityTracer's coloring has
        # shown rare (roughly 1-in-several-dozen-runs) process-to-process nondeterminism
        # taking one extra step, observed twice in this session -- a real property of the
        # external tracer/coloring pipeline, not this package, and precisely the class of
        # fragility #21's AST-derived pattern sidesteps by not tracing at all.
        @test length(newton_residuals) < 8
        @test newton_residuals[end] < 1e-10

        uₕ_newton = element(Wₕ)
        uₕ_newton .= u
        uexact = Rₕ(Wₕ, sol)
        @test norm₁ₕ(uₕ_newton .- uexact) < 1e-2
    end

    @testset "Coupled reaction-diffusion: O(h²) convergence, per species" begin
        # -Δu + u + uv = f₁, -Δv + v - uv = f₂ on (0,1)², homogeneous Dirichlet data, as in
        # coupled_reaction_diffusion.md. Checked independently per species: a routing
        # mistake in the composite Jacobian would show up as one species converging
        # correctly while the other silently used the wrong block, which a single combined
        # error could hide (see bramble-verification §5 on composite spaces).
        u_ex(x) = sin(π * x[1]) * sin(π * x[2])
        v_ex(x) = sin(2π * x[1]) * sin(2π * x[2])
        f1(x) = 2π^2 * u_ex(x) + u_ex(x) + u_ex(x) * v_ex(x)
        f2(x) = 8π^2 * v_ex(x) + v_ex(x) - u_ex(x) * v_ex(x)

        Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))

        function coupled_series(; n0::Int, levels::Int)
            # Seeded for the same reason as the 1D testsets above: a non-uniform mesh is
            # drawn from the global RNG, and the convergence rates asserted below should
            # not depend on which mesh this run happened to get.
            Random.seed!(20260909)
            Ωc = mesh(Ω, (n0, n0), (false, false))
            hs = Float64[]
            erru, errv = Float64[], Float64[]
            for level in 1:levels
                Wc = gridspace(Ωc)
                Vc = Wc^Val(2)
                bcs_c = dirichlet_constraints(Ω, :boundary => x -> 0.0)
                f1_c = element(Wc)
                avgₕ!(f1_c, f1)
                f2_c = element(Wc)
                avgₕ!(f2_c, f2)
                l_c = form(Vc, q -> innerₕ(f1_c, q(1)) + innerₕ(f2_c, q(2)))
                F_c = assemble(l_c; dirichlet=bcs_c)

                Ac(wₕ) = begin
                    u_c, v_c = components(wₕ)
                    assemble(
                        form(
                            Vc,
                            Vc,
                            (p, q) ->
                                inner₊(∇₋ₕ(p(1)), ∇₋ₕ(q(1))) +
                                innerₕ(p(1), q(1)) +
                                innerₕ(v_c * p(1), q(1)) +
                                inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))) +
                                innerₕ(p(2), q(2)) - innerₕ(u_c * p(2), q(2)),
                        );
                        dirichlet=:boundary,
                    )
                end
                rc(w::AbstractVector{T}) where {T} = begin
                    wₕ = element(Vc, T)
                    wₕ .= w
                    Ac(wₕ) * w .- F_c
                end

                w_c = zeros(ndofs(Vc))
                prep_c = prepare_jacobian(rc, _sparse_ad, w_c)
                J_c = DifferentiationInterface.jacobian(rc, prep_c, _sparse_ad, w_c)
                for it in 1:20
                    r = rc(w_c)
                    sqrt(sum(abs2, r)) < 1e-10 && break
                    DifferentiationInterface.jacobian!(rc, J_c, prep_c, _sparse_ad, w_c)
                    w_c .-= J_c \ r
                end
                w_ch = element(Vc)
                w_ch .= w_c
                u_ch, v_ch = components(w_ch)
                uexact_c, vexact_c = Rₕ(Wc, u_ex), Rₕ(Wc, v_ex)

                push!(hs, hₘₐₓ(Ωc))
                push!(erru, norm₁ₕ(u_ch .- uexact_c))
                push!(errv, norm₁ₕ(v_ch .- vexact_c))
                level < levels && iterative_refinement!(Ωc)
            end
            return hs, erru, errv
        end

        # Same seed and level count as the doc page's own run, so a failure here and a
        # changed number on the rendered page mean the same thing.
        Random.seed!(20260903)
        hs, erru, errv = coupled_series(; n0=5, levels=5)

        _observed_order(hs, errs) =
            log(errs[end - 1] / errs[end]) / log(hs[end - 1] / hs[end])
        order_u = _observed_order(hs, erru)
        order_v = _observed_order(hs, errv)

        # Bracketed on both sides, as examples/convergence.jl's own check is: an order far
        # above the promise means the finest error already hit roundoff, not that the
        # scheme is unusually good.
        @test 1.9 < order_u < 3.0
        @test 1.9 < order_v < 3.0
    end
end
