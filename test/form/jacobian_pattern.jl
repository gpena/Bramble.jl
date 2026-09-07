using Test
using Bramble
using Random
using ADTypes: KnownJacobianSparsityDetector
using ForwardDiff, DifferentiationInterface
import SparseConnectivityTracer, SparseMatrixColorings
using SparseArrays: nnz, sparse

# `jacobian_pattern` (form/jacobian_pattern.jl, gpena/Bramble.jl#21) reads a Newton
# residual's Jacobian sparsity off `a`'s AST plus each named coefficient dependency's own
# reach, with no AD tracing. Two properties matter, checked separately:
#
#   - structural: the pattern is a safe superset of the exact one (never drops a true
#     entry), checked against SparseConnectivityTracer's AD-traced pattern as ground truth,
#     across 1D/2D/3D -- not just the doc page's own 1D case.
#   - functional: handing the pattern to `KnownJacobianSparsityDetector` and actually
#     running Newton with it converges to the right answer -- a safe-but-wrong-shaped
#     pattern (e.g. one that accidentally coincides in nnz count) would still pass the
#     structural check alone.

const _traced_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

# `-(α(u)u')' = g`, mirroring docs/src/examples/poisson_nonlinear.md, generalized to D
# dimensions the way examples/nonlinear_convergence.jl and the doc's own `nonlinear_series`
# do. `Ωd` is the domain the mesh `Ωₕ` was built from (kept around separately, the same way
# examples/convergence.jl does, since dirichlet_constraints needs the boundary-marked
# domain rather than the mesh itself). Returns everything a caller needs to build both the
# residual and its native pattern.
function _nonlinear_poisson_setup(D::Int, Ωd, Ωₕ)
    sol(x) = exp(sum(x))
    α(u) = 3 + 1 / (1 + u^2)
    dαdu(u) = -2u / (1 + u^2)^2
    rhs(x) = -D * dαdu(sol(x)) * sol(x)^2 - D * α(sol(x)) * sol(x)

    Wₕ = gridspace(Ωₕ)
    bcs = dirichlet_constraints(Bramble.set(Ωd), :boundary => sol)
    gₕ = element(Wₕ)
    avgₕ!(gₕ, rhs)
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    F = assemble(l; dirichlet_conditions = bcs, dirichlet_labels = :boundary)

    function diffusion_form(uₕ)
        αv = D == 1 ? α.(M₋ₕ(uₕ)) : ntuple(i -> α.(M₋ₕ(uₕ)[i]), D)
        grad(U) = D == 1 ? αv * ∇₋ₕ(U) : ntuple(i -> αv[i] * ∇₋ₕ(U)[i], D)
        return form(Wₕ, Wₕ, (U, V) -> inner₊(grad(U), ∇₋ₕ(V)))
    end

    function residual(u_vec::AbstractVector{T}) where {T}
        uₕ = element(Wₕ, T)
        uₕ .= u_vec
        A = assemble(diffusion_form(uₕ); dirichlet_labels = :boundary)
        return A * u_vec .- F
    end

    return Wₕ, sol, diffusion_form, residual
end

@testset "jacobian_pattern" begin
    @testset "Safe superset of the AD-traced pattern ($D D)" for (D, n) in ((1, 12), (2, 6), (
        3, 4))
        Ω = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
        Ωₕ = mesh(Ω, ntuple(_ -> n, D), ntuple(_ -> false, D))
        Wₕ, sol, diffusion_form, residual = _nonlinear_poisson_setup(D, Ω, Ωₕ)

        u0 = element(Wₕ, 0.0)
        a = diffusion_form(u0)
        mine = jacobian_pattern(a, U -> M₋ₕ(U))

        u_probe = rand(ndofs(Wₕ))
        prep = prepare_jacobian(residual, _traced_ad, u_probe)
        J = DifferentiationInterface.jacobian(residual, prep, _traced_ad, u_probe)
        ground_truth = J .!= 0

        # `mine` must contain every entry the AD trace found; extra (structurally
        # possible but numerically zero) entries are fine -- see the file header.
        @test all(ground_truth .<= mine)
        @test nnz(mine) >= nnz(sparse(ground_truth))
    end

    @testset "Newton with the native pattern reaches the right answer" begin
        Random.seed!(20260907)

        # 1D, a random coarse mesh (not uniform: bramble-verification warns a uniform grid
        # makes exp(sum(x)) nearly exact regardless of correctness).
        Ω1 = domain(interval(0.0, 1.0))
        Ω1ₕ = mesh(Ω1, 10, false)
        iterative_refinement!(Ω1ₕ)
        Wₕ, sol, diffusion_form, residual = _nonlinear_poisson_setup(1, Ω1, Ω1ₕ)

        u0 = element(Wₕ, 0.0)
        a = diffusion_form(u0)
        pattern = jacobian_pattern(a, U -> M₋ₕ(U))
        sparse_ad = AutoSparse(AutoForwardDiff();
            sparsity_detector = KnownJacobianSparsityDetector(pattern),
            coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

        u = zeros(ndofs(Wₕ))
        prep = prepare_jacobian(residual, sparse_ad, u)
        J = DifferentiationInterface.jacobian(residual, prep, sparse_ad, u)
        newton_residuals = Float64[]
        for _ in 1:20
            r = residual(u)
            push!(newton_residuals, sqrt(sum(abs2, r)))
            newton_residuals[end] < 1e-10 && break
            DifferentiationInterface.jacobian!(residual, J, prep, sparse_ad, u)
            u .-= J \ r
        end

        @test length(newton_residuals) < 6
        @test newton_residuals[end] < 1e-10

        uₕ = element(Wₕ)
        uₕ .= u
        @test norm₁ₕ(uₕ .- Rₕ(Wₕ, sol)) < 1e-2

        # 2D, same check: the native pattern must still be exact enough (as a superset) for
        # Newton to reach the same quadratic-convergence behaviour, not just structurally
        # "safe" in isolation.
        Ω2 = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
        Ω2ₕ = mesh(Ω2, (8, 8), (false, false))
        Wₕ2, sol2, diffusion_form2, residual2 = _nonlinear_poisson_setup(2, Ω2, Ω2ₕ)

        u02 = element(Wₕ2, 0.0)
        a2 = diffusion_form2(u02)
        pattern2 = jacobian_pattern(a2, U -> M₋ₕ(U))
        sparse_ad2 = AutoSparse(AutoForwardDiff();
            sparsity_detector = KnownJacobianSparsityDetector(pattern2),
            coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

        u2 = zeros(ndofs(Wₕ2))
        prep2 = prepare_jacobian(residual2, sparse_ad2, u2)
        J2 = DifferentiationInterface.jacobian(residual2, prep2, sparse_ad2, u2)
        newton_residuals2 = Float64[]
        for _ in 1:20
            r = residual2(u2)
            push!(newton_residuals2, sqrt(sum(abs2, r)))
            newton_residuals2[end] < 1e-10 && break
            DifferentiationInterface.jacobian!(residual2, J2, prep2, sparse_ad2, u2)
            u2 .-= J2 \ r
        end

        @test length(newton_residuals2) < 8
        @test newton_residuals2[end] < 1e-10

        uₕ2 = element(Wₕ2)
        uₕ2 .= u2
        @test norm₁ₕ(uₕ2 .- Rₕ(Wₕ2, sol2)) < 1e-1
    end

    @testset "No coefficient dependencies degenerates to A's own pattern" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 10, false)
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (U, V) -> inner₊(∇₋ₕ(U), ∇₋ₕ(V)))

        A = assemble(a)
        @test jacobian_pattern(a) == (A .!= 0)
    end

    @testset "Multiple independent nonlinear coefficients" begin
        # -(α(M₋ₕ(u))u')' + β(D₋ₓ(u))u = g: two terms, each nonlinear through a
        # *different* stencil op. Passing both dependencies together must still be a
        # safe superset of the true pattern -- neither term's reach may be dropped just
        # because the other one was declared too.
        # No manufactured solution needed: only the pattern's structural safety is
        # checked here, not convergence to a known answer, so any boundary/source data
        # will do.
        α(u) = 3 + 1 / (1 + u^2)
        β(u) = 1 + 0.5 * u^2

        Ω = domain(interval(0.0, 1.0))
        Ωₕ = mesh(Ω, 20, false)
        Wₕ = gridspace(Ωₕ)
        bcs = dirichlet_constraints(Bramble.set(Ω), :boundary => x -> exp(x[1]))
        gₕ = element(Wₕ)
        avgₕ!(gₕ, x -> exp(x[1]))
        l = form(Wₕ, v -> innerₕ(gₕ, v))
        F = assemble(l; dirichlet_conditions = bcs, dirichlet_labels = :boundary)

        function build_form(uₕ)
            αv = α.(M₋ₕ(uₕ))
            βv = β.(D₋ₓ(uₕ))
            return form(Wₕ, Wₕ, (U, V) -> inner₊(αv * ∇₋ₕ(U), ∇₋ₕ(V)) + innerₕ(βv * U, V))
        end
        function residual(u_vec::AbstractVector{T}) where {T}
            uₕ = element(Wₕ, T)
            uₕ .= u_vec
            A = assemble(build_form(uₕ); dirichlet_labels = :boundary)
            return A * u_vec .- F
        end

        a = build_form(element(Wₕ, 0.0))
        mine = jacobian_pattern(a, U -> M₋ₕ(U), U -> D₋ₓ(U))

        u_probe = rand(ndofs(Wₕ))
        prep = prepare_jacobian(residual, _traced_ad, u_probe)
        J = DifferentiationInterface.jacobian(residual, prep, _traced_ad, u_probe)
        ground_truth = J .!= 0

        @test all(ground_truth .<= mine)
        @test nnz(mine) >= nnz(sparse(ground_truth))
    end

    @testset "Composite trial space is rejected" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (false, false))
        Wₕ = gridspace(Ωₕ)
        Vₕ = Wₕ^Val(2)
        a = form(Vₕ, Vₕ,
            (p, q) -> inner₊(∇₋ₕ(p(1)), ∇₋ₕ(q(1))) + inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))))

        @test_throws ArgumentError jacobian_pattern(a, U -> M₋ₕ(U))
    end
end
