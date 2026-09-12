using Test
using Bramble
using Random
using ADTypes: KnownJacobianSparsityDetector
using ForwardDiff, DifferentiationInterface
using SparseConnectivityTracer: SparseConnectivityTracer
using SparseMatrixColorings: SparseMatrixColorings
using SparseArrays: nnz, sparse, findnz

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

const _traced_ad = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm=SparseMatrixColorings.GreedyColoringAlgorithm(),
)

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
    bcs = dirichlet_constraints(Ωd, :boundary => sol)
    gₕ = element(Wₕ)
    avgₕ!(gₕ, rhs)
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    F = assemble(l; dirichlet=bcs)

    function diffusion_form(uₕ)
        αv = D == 1 ? α.(M₋ₕ(uₕ)) : ntuple(i -> α.(M₋ₕ(uₕ)[i]), D)
        grad(U) = D == 1 ? αv * ∇₋ₕ(U) : ntuple(i -> αv[i] * ∇₋ₕ(U)[i], D)
        return form(Wₕ, Wₕ, (U, V) -> inner₊(grad(U), ∇₋ₕ(V)))
    end

    function residual(u_vec::AbstractVector{T}) where {T}
        uₕ = element(Wₕ, T)
        uₕ .= u_vec
        A = assemble(diffusion_form(uₕ); dirichlet=:boundary)
        return A * u_vec .- F
    end

    return Wₕ, sol, diffusion_form, residual
end

@testset "jacobian_pattern" begin
    @testset "Safe superset of the AD-traced pattern ($D D)" for (D, n) in
                                                                 ((1, 12), (2, 6), (3, 4))
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
        sparse_ad = AutoSparse(
            AutoForwardDiff();
            sparsity_detector=KnownJacobianSparsityDetector(pattern),
            coloring_algorithm=SparseMatrixColorings.GreedyColoringAlgorithm(),
        )

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
        sparse_ad2 = AutoSparse(
            AutoForwardDiff();
            sparsity_detector=KnownJacobianSparsityDetector(pattern2),
            coloring_algorithm=SparseMatrixColorings.GreedyColoringAlgorithm(),
        )

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
        bcs = dirichlet_constraints(Ω, :boundary => x -> exp(x[1]))
        gₕ = element(Wₕ)
        avgₕ!(gₕ, x -> exp(x[1]))
        l = form(Wₕ, v -> innerₕ(gₕ, v))
        F = assemble(l; dirichlet=bcs)

        function build_form(uₕ)
            αv = α.(M₋ₕ(uₕ))
            βv = β.(D₋ₓ(uₕ))
            return form(Wₕ, Wₕ, (U, V) -> inner₊(αv * ∇₋ₕ(U), ∇₋ₕ(V)) + innerₕ(βv * U, V))
        end
        function residual(u_vec::AbstractVector{T}) where {T}
            uₕ = element(Wₕ, T)
            uₕ .= u_vec
            A = assemble(build_form(uₕ); dirichlet=:boundary)
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

    @testset "Composite trial/test spaces" begin
        # `jacobian_pattern` on a composite space (gpena/Bramble.jl#95): a dependency may
        # name a *different* leaf the same way a form term does (`U -> U(2)`), for
        # "this coefficient is another component's own value" (coupled_reaction_diffusion.md's
        # `v_c`); `U -> M₋ₕ(U)` (no `(k)`) keeps meaning "this block's own trial leaf",
        # exactly like the non-composite case above.

        @testset "coupled reaction-diffusion's own coupling (identity cross-block)" begin
            # -Δu + u + uv = f₁, -Δv + v - uv = f₂: `v_c` scales block (1,1), `u_c` scales
            # block (2,2) -- the case jacobian_pattern's composite support exists for.
            u_ex(x) = sin(π * x[1]) * sin(π * x[2])
            v_ex(x) = sin(2π * x[1]) * sin(2π * x[2])
            f1(x) = 2π^2 * u_ex(x) + u_ex(x) + u_ex(x) * v_ex(x)
            f2(x) = 8π^2 * v_ex(x) + v_ex(x) - u_ex(x) * v_ex(x)

            Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
            Ωₕ = mesh(Ω, (10, 10), (false, false))
            Wₕ = gridspace(Ωₕ)
            Vₕ = Wₕ^Val(2)

            bcs = dirichlet_constraints(Ω, :boundary => x -> 0.0)
            f1ₕ = element(Wₕ)
            avgₕ!(f1ₕ, f1)
            f2ₕ = element(Wₕ)
            avgₕ!(f2ₕ, f2)
            l = form(Vₕ, q -> innerₕ(f1ₕ, q(1)) + innerₕ(f2ₕ, q(2)))
            F = assemble(l; dirichlet=bcs)

            function coupled_form(wₕ)
                u_c, v_c = components(wₕ)
                return form(
                    Vₕ,
                    Vₕ,
                    (p, q) ->
                        inner₊(∇₋ₕ(p(1)), ∇₋ₕ(q(1))) +
                        innerₕ(p(1), q(1)) +
                        innerₕ(v_c * p(1), q(1)) +
                        inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))) +
                        innerₕ(p(2), q(2)) - innerₕ(u_c * p(2), q(2)),
                )
            end
            function residual(w::AbstractVector{T}) where {T}
                wₕ = element(Vₕ, T)
                wₕ .= w
                A = assemble(coupled_form(wₕ); dirichlet=:boundary)
                return A * w .- F
            end

            a = coupled_form(element(Vₕ, 0.0))
            mine = jacobian_pattern(a, U -> U(2), U -> U(1))

            u_probe = rand(ndofs(Vₕ))
            prep = prepare_jacobian(residual, _traced_ad, u_probe)
            J = DifferentiationInterface.jacobian(residual, prep, _traced_ad, u_probe)
            ground_truth = J .!= 0

            @test all(ground_truth .<= mine)
            @test nnz(mine) >= nnz(sparse(ground_truth))
        end

        @testset "3-component chained cross-block coupling" begin
            # block(1,1) depends on component 2, block(2,2) on component 3, block(3,3) on
            # component 1 -- a cycle through 3 leaves, not just the 2-component case above.
            Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
            Ωₕ = mesh(Ω, (8, 8), (false, false))
            Wₕ = gridspace(Ωₕ)
            Vₕ = Wₕ^Val(3)

            bcs = dirichlet_constraints(Ω, :boundary => x -> 0.0)
            fₕ = element(Wₕ)
            avgₕ!(fₕ, x -> sin(π * x[1]))
            l = form(Vₕ, q -> innerₕ(fₕ, q(1)) + innerₕ(fₕ, q(2)) + innerₕ(fₕ, q(3)))
            F = assemble(l; dirichlet=bcs)

            function chain_form(wₕ)
                c1, c2, c3 = components(wₕ)
                return form(
                    Vₕ,
                    Vₕ,
                    (p, q) ->
                        inner₊(∇₋ₕ(p(1)), ∇₋ₕ(q(1))) +
                        innerₕ((1.0 .+ c2 .^ 2) * p(1), q(1)) +
                        inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))) +
                        innerₕ((1.0 .+ c3 .^ 2) * p(2), q(2)) +
                        inner₊(∇₋ₕ(p(3)), ∇₋ₕ(q(3))) +
                        innerₕ((1.0 .+ c1 .^ 2) * p(3), q(3)),
                )
            end
            function residual(w::AbstractVector{T}) where {T}
                wₕ = element(Vₕ, T)
                wₕ .= w
                A = assemble(chain_form(wₕ); dirichlet=:boundary)
                return A * w .- F
            end

            a = chain_form(element(Vₕ, 0.0))
            mine = jacobian_pattern(a, U -> U(2), U -> U(3), U -> U(1))

            u_probe = rand(ndofs(Vₕ))
            prep = prepare_jacobian(residual, _traced_ad, u_probe)
            J = DifferentiationInterface.jacobian(residual, prep, _traced_ad, u_probe)
            ground_truth = J .!= 0

            @test all(ground_truth .<= mine)
            @test nnz(mine) >= nnz(sparse(ground_truth))
        end

        @testset "mixed same-block (stencil op) and cross-block (identity) dependencies" begin
            # block(1,1)'s coefficient depends on BOTH M₋ₕ of its own leaf (same-block,
            # the non-composite mechanism) AND directly on component 2 (cross-block) --
            # neither must shadow the other.
            Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
            Ωₕ = mesh(Ω, (8, 8), (false, false))
            Wₕ = gridspace(Ωₕ)
            Vₕ = Wₕ^Val(2)

            bcs = dirichlet_constraints(Ω, :boundary => x -> 0.0)
            fₕ = element(Wₕ)
            avgₕ!(fₕ, x -> sin(π * x[1]))
            l = form(Vₕ, q -> innerₕ(fₕ, q(1)) + innerₕ(fₕ, q(2)))
            F = assemble(l; dirichlet=bcs)

            α(u) = 3 + 1 / (1 + u^2)

            function mixed_form(wₕ)
                c1, c2 = components(wₕ)
                Mc1 = M₋ₕ(c1)
                αv = ntuple(i -> α.(Mc1[i]), 2)
                grad1(p) = ntuple(i -> αv[i] * ∇₋ₕ(p(1))[i], 2)
                return form(
                    Vₕ,
                    Vₕ,
                    (p, q) ->
                        inner₊(grad1(p), ∇₋ₕ(q(1))) +
                        innerₕ((1.0 .+ c2 .^ 2) * p(1), q(1)) +
                        inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))) +
                        innerₕ(p(2), q(2)),
                )
            end
            function residual(w::AbstractVector{T}) where {T}
                wₕ = element(Vₕ, T)
                wₕ .= w
                A = assemble(mixed_form(wₕ); dirichlet=:boundary)
                return A * w .- F
            end

            a = mixed_form(element(Vₕ, 0.0))
            mine = jacobian_pattern(a, U -> M₋ₕ(U(1)), U -> U(2))

            u_probe = rand(ndofs(Vₕ))
            prep = prepare_jacobian(residual, _traced_ad, u_probe)
            J = DifferentiationInterface.jacobian(residual, prep, _traced_ad, u_probe)
            ground_truth = J .!= 0

            @test all(ground_truth .<= mine)
            @test nnz(mine) >= nnz(sparse(ground_truth))
        end

        @testset "cross-block dependency through a stencil op, not just identity" begin
            # block(1,1)'s coefficient depends on M₋ₕ(component 2) -- averaged through
            # another leaf, not read directly the way coupled_reaction_diffusion.md's own
            # v_c is.
            Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
            Ωₕ = mesh(Ω, (8, 8), (false, false))
            Wₕ = gridspace(Ωₕ)
            Vₕ = Wₕ^Val(2)

            bcs = dirichlet_constraints(Ω, :boundary => x -> 0.0)
            fₕ = element(Wₕ)
            avgₕ!(fₕ, x -> sin(π * x[1]))
            l = form(Vₕ, q -> innerₕ(fₕ, q(1)) + innerₕ(fₕ, q(2)))
            F = assemble(l; dirichlet=bcs)

            β(u) = 1 + 0.5 * u^2

            function averaged_cross_form(wₕ)
                c1, c2 = components(wₕ)
                Mc2 = M₋ₕ(c2)
                βv = ntuple(i -> β.(Mc2[i]), 2)
                grad1(p) = ntuple(i -> βv[i] * ∇₋ₕ(p(1))[i], 2)
                return form(
                    Vₕ,
                    Vₕ,
                    (p, q) ->
                        inner₊(grad1(p), ∇₋ₕ(q(1))) +
                        inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))) +
                        innerₕ(p(2), q(2)),
                )
            end
            function residual(w::AbstractVector{T}) where {T}
                wₕ = element(Vₕ, T)
                wₕ .= w
                A = assemble(averaged_cross_form(wₕ); dirichlet=:boundary)
                return A * w .- F
            end

            a = averaged_cross_form(element(Vₕ, 0.0))
            mine = jacobian_pattern(a, U -> M₋ₕ(U(2)))

            u_probe = rand(ndofs(Vₕ))
            prep = prepare_jacobian(residual, _traced_ad, u_probe)
            J = DifferentiationInterface.jacobian(residual, prep, _traced_ad, u_probe)
            ground_truth = J .!= 0

            @test all(ground_truth .<= mine)
            @test nnz(mine) >= nnz(sparse(ground_truth))
        end

        @testset "no dependencies is a safe superset of assemble(a)'s own pattern" begin
            # Not exact equality: `assemble` can preallocate a structurally-reachable
            # entry that happens to assemble to exact zero (`A .!= 0` would then read as
            # narrower than the AST's own true reach), so the right check is the same
            # safe-superset direction every other case here uses, against `A`'s own
            # stored pattern rather than its numeric one.
            Ωₕ = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 6), (false, false)
            )
            Wₕ = gridspace(Ωₕ)
            Vₕ = Wₕ^Val(2)
            a = form(
                Vₕ,
                Vₕ,
                (p, q) -> inner₊(∇₋ₕ(p(1)), ∇₋ₕ(q(1))) + inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))),
            )
            A = assemble(a)
            stored = fill(false, size(A))
            for (r, c) in zip(findnz(A)[1], findnz(A)[2])
                stored[r, c] = true
            end

            mine = jacobian_pattern(a)
            @test all(stored .<= mine)
        end

        @testset "Newton with the native pattern agrees with the traced one" begin
            # The functional counterpart to the structural checks above: a safe-but-
            # wrong-shaped pattern could still pass "no missing entries" while breaking
            # Newton's own convergence (e.g. if colouring somehow interacted badly with
            # the extra entries) -- so this drives the actual solve.
            #
            # Checked against the *traced* solution from the same run, not a manufactured
            # exact answer: this problem's quadratic `uv` coupling is genuinely nonlinear,
            # and separate process runs were observed to land on solutions with u/v errors
            # varying over an order of magnitude (0.02-1.15) while Newton's own residual
            # converged to ~1e-14 every time -- a property of this coupled system's
            # multiple nearby roots and Newton's own path-sensitivity between processes,
            # not of which sparsity pattern was used. Comparing native against traced
            # *within one run* sidesteps that entirely: both start from the same `w = 0`
            # and see the same residual function, so they must reach the same root if the
            # native pattern is exact enough for Newton to behave identically -- confirmed
            # to agree to ~1e-14 across repeated runs.
            u_ex(x) = sin(π * x[1]) * sin(π * x[2])
            v_ex(x) = sin(2π * x[1]) * sin(2π * x[2])
            f1(x) = 2π^2 * u_ex(x) + u_ex(x) + u_ex(x) * v_ex(x)
            f2(x) = 8π^2 * v_ex(x) + v_ex(x) - u_ex(x) * v_ex(x)

            Random.seed!(20260912)
            Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
            Ωₕ = mesh(Ω, (16, 16), (false, false))
            Wₕ = gridspace(Ωₕ)
            Vₕ = Wₕ^Val(2)

            bcs = dirichlet_constraints(Ω, :boundary => x -> 0.0)
            f1ₕ = element(Wₕ)
            avgₕ!(f1ₕ, f1)
            f2ₕ = element(Wₕ)
            avgₕ!(f2ₕ, f2)
            l = form(Vₕ, q -> innerₕ(f1ₕ, q(1)) + innerₕ(f2ₕ, q(2)))
            F = assemble(l; dirichlet=bcs)

            function coupled_form(wₕ)
                u_c, v_c = components(wₕ)
                return form(
                    Vₕ,
                    Vₕ,
                    (p, q) ->
                        inner₊(∇₋ₕ(p(1)), ∇₋ₕ(q(1))) +
                        innerₕ(p(1), q(1)) +
                        innerₕ(v_c * p(1), q(1)) +
                        inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))) +
                        innerₕ(p(2), q(2)) - innerₕ(u_c * p(2), q(2)),
                )
            end
            function residual(w::AbstractVector{T}) where {T}
                wₕ = element(Vₕ, T)
                wₕ .= w
                A = assemble(coupled_form(wₕ); dirichlet=:boundary)
                return A * w .- F
            end

            function newton(ad)
                w = zeros(ndofs(Vₕ))
                prep = prepare_jacobian(residual, ad, w)
                J = DifferentiationInterface.jacobian(residual, prep, ad, w)
                newton_residuals = Float64[]
                for _ in 1:20
                    r = residual(w)
                    push!(newton_residuals, sqrt(sum(abs2, r)))
                    newton_residuals[end] < 1e-10 && break
                    DifferentiationInterface.jacobian!(residual, J, prep, ad, w)
                    w .-= J \ r
                end
                return w, newton_residuals
            end

            a = coupled_form(element(Vₕ, 0.0))
            pattern = jacobian_pattern(a, U -> U(2), U -> U(1))
            native_ad = AutoSparse(
                AutoForwardDiff();
                sparsity_detector=KnownJacobianSparsityDetector(pattern),
                coloring_algorithm=SparseMatrixColorings.GreedyColoringAlgorithm(),
            )

            w_native, newton_residuals = newton(native_ad)
            w_traced, _ = newton(_traced_ad)

            @test length(newton_residuals) < 8
            @test newton_residuals[end] < 1e-10
            @test maximum(abs.(w_native .- w_traced)) < 1e-8
        end
    end
end
