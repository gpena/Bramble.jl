module ExtChainRulesExtTests

using Test
using Bramble
using ChainRulesCore: ChainRulesCore, rrule
using SparseArrays
using LinearAlgebra: Tridiagonal
using ..TestUtils: _fd

# BrambleChainRulesExt: the `ChainRulesCore.rrule` for `pde_solve` (src/solvers/pde_solve.jl
# explains why this one function is the entire adjoint story -- `assemble`/`dirichlet_bc!`
# are already reverse-mode-differentiable on their own). What is checked here needs no
# reverse-mode AD package at all, only `ChainRulesCore` itself: the rrule's raw pullback
# against finite differences, the `Ā` cotangent's sparsity pattern, and a Dirichlet-value
# gradient chained by hand from real `assemble` output. Not checked against `ForwardDiff`
# here, even though it is known-good through `assemble` in general (#107/#122): UMFPACK's
# sparse factorisation only accepts `Float64`/`ComplexF64`, so `A \ F` fails for a `Dual`-
# valued `F` regardless of this issue (`docs/src/tutorials/autodiff.md` §5 documents this and
# recommends a dense conversion or an iterative solver as the forward-mode workaround) -- the
# adjoint rule sidesteps it entirely since its math is hand-written, not generic-`\`-through-
# Duals, so central differences are the reference instead.
#
# Enzyme composition lives in chainrules_enzyme_ext.jl instead, behind the "ad"/"full" groups
# -- Enzyme is not a `test/Project.toml` dependency (Weekly.yml installs it at runtime only),
# so this file must not need it.
#
# `Bramble.domain`/`Bramble.mesh`/`Bramble.element` are qualified throughout for the reason
# meshes_ext.jl gives: every ext file is included into the same `Main`.

@testset "BrambleChainRulesExt" begin
    @testset "rrule: pullback matches finite differences, Ā never densified" begin
        n = 6
        A = sparse(Tridiagonal(fill(-1.0, n - 1), fill(2.0, n), fill(-1.0, n - 1)))
        F = collect(1.0:n)

        J(f) = sum(abs2, Bramble.pde_solve(A, f))
        u, pullback = rrule(Bramble.pde_solve, A, F)
        @test u ≈ A \ F

        ȳ = 2 .* u
        _, Ā, F̄ = pullback(ȳ)

        # F̄, componentwise, against a central difference on the same loss.
        for i in 1:n
            e = zeros(n)
            e[i] = 1e-6
            dJi = (J(F .+ e) - J(F .- e)) / 2e[i]
            @test F̄[i]≈dJi rtol=1e-5
        end

        # The point of the rule: Ā never densified, only A's own stored entries touched.
        @test nnz(Ā) == nnz(A)
        @test rowvals(Ā) == rowvals(A)
        @test Ā isa SparseMatrixCSC
    end

    @testset "rrule: ∂J/∂A matches a central difference on a genuinely varying A(θ)" begin
        n = 6
        function build_A(θ)
            d = fill(2.0, n)
            d[2] = 2.0 + θ
            return sparse(Tridiagonal(fill(-1.0, n - 1), d, fill(-1.0, n - 1)))
        end
        F = collect(1.0:n)
        loss(θ) = sum(abs2, Bramble.pde_solve(build_A(θ), F))

        θ0 = 0.3
        d_fd = _fd(loss, θ0)

        # Chain the rrule by hand: ∂A/∂θ has a single nonzero (the (2,2) entry), so
        # ⟨Ā, ∂A/∂θ⟩ is just Ā[2,2].
        A0 = build_A(θ0)
        _, pullback = rrule(Bramble.pde_solve, A0, F)
        u = Bramble.pde_solve(A0, F)
        _, Ā, _ = pullback(2 .* u)
        @test Ā[2, 2]≈d_fd rtol=1e-5
    end

    # The Dirichlet-value gradient the issue's own acceptance criteria ask for: `θ` enters
    # `F` through `assemble`'s own `dirichlet` keyword. `ForwardDiff` is *not* the reference
    # here, even though it is known-good through `assemble` in general (#107/#122): UMFPACK's
    # sparse factorisation only accepts `Float64`/`ComplexF64`, so `A \ F` for a `Float64`
    # sparse `A` and a `Dual`-valued `F` fails regardless of this issue --
    # `docs/src/tutorials/autodiff.md` §5 documents this exact limitation and recommends
    # `Matrix(A) \ F` or an iterative solver as the forward-mode workaround. The adjoint rule
    # sidesteps it entirely (its own math is hand-written, not generic-`\`-through-Duals), so
    # central differences -- and the rrule's pullback chained by hand -- are the references.
    @testset "Dirichlet boundary value gradient (1D)" begin
        Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 21, true)
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        fₕ = Bramble.element(Wₕ, 0.0)
        l = form(Wₕ, v -> innerₕ(fₕ, v))

        loss(θ) = sum(abs2, Bramble.pde_solve(assemble(a, l; dirichlet = :boundary => x -> θ)...))

        θ0 = 0.7
        d_fd = _fd(loss, θ0)

        # Chain the rrule by hand against the same reference: F's dependence on θ at the
        # boundary rows is exactly `dg/dθ = 1`, everywhere else `0` -- so ⟨F̄, ∂F/∂θ⟩ collapses
        # to F̄ summed over the two constrained rows.
        A0, F0 = assemble(a, l; dirichlet = :boundary => x -> θ0)
        u0 = Bramble.pde_solve(A0, F0)
        _, pullback = rrule(Bramble.pde_solve, A0, F0)
        _, _, F̄ = pullback(2 .* u0)
        n = ndofs(Wₕ)
        d_manual = F̄[1] + F̄[n]
        @test d_manual≈d_fd rtol=1e-5
    end

    @testset "Dirichlet boundary value gradient (2D)" begin
        Ωd = Bramble.domain(Bramble.interval(0.0, 1.0) × Bramble.interval(0.0, 1.0))
        Ωₕ = Bramble.mesh(Ωd, (11, 11), (true, true))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        fₕ = Bramble.element(Wₕ, 0.0)
        l = form(Wₕ, v -> innerₕ(fₕ, v))

        loss(θ) = sum(abs2, Bramble.pde_solve(assemble(a, l; dirichlet = :boundary => x -> θ)...))

        θ0 = 1.3
        d_fd = _fd(loss, θ0)

        A0, F0 = assemble(a, l; dirichlet = :boundary => x -> θ0)
        u0 = Bramble.pde_solve(A0, F0)
        _, pullback = rrule(Bramble.pde_solve, A0, F0)
        _, _, F̄ = pullback(2 .* u0)
        d_manual = sum(F̄[i] for i in 1:ndofs(Wₕ) if Bramble.index_in_marker(Ωₕ, :boundary)[i])
        @test d_manual≈d_fd rtol=1e-5
    end

    # Issue #228's own acceptance criterion: the adjoint rule costs one extra solve
    # *regardless* of the number of parameters, against a naive scheme -- pushing a
    # perturbation through the solve once per parameter, the cost forward-mode AD through
    # `pde_solve` would pay if it could run at all (it cannot, for a sparse `A`: see the
    # comment above). Not a committed `benchmark/baselines/*.json` entry -- that process
    # tracks Bramble's own performance release to release, not an asymptotic-complexity claim
    # about two different differentiation strategies -- so this is a plain in-test timing
    # sanity check: `nθ` diagonal perturbations to a fixed `A`, one rank-1 direction each, and
    # `@elapsed` after a warm-up call per `bramble-verification`'s measurement discipline.
    @testset "cost is one solve regardless of parameter count" begin
        n = 60
        A_base = sparse(Tridiagonal(fill(-1.0, n - 1), fill(4.0, n), fill(-1.0, n - 1)))
        F = collect(1.0:n)

        function adjoint_time(nθ)
            idx = nθ == 1 ? [2] : round.(Int, range(2, n - 1; length = nθ))
            A = copy(A_base)
            u, pullback = rrule(Bramble.pde_solve, A, F)   # warm-up (compilation, factorisation)
            pullback(u)
            return @elapsed begin
                u, pullback = rrule(Bramble.pde_solve, A, F)
                _, Ā, _ = pullback(u)
                # ⟨Ā, ∂A/∂θₖ⟩ for every θₖ at once, from the one Ā already computed above --
                # the actual point of the rule: no further solve, for any number of idx.
                sum(Ā[i, i] for i in idx)
            end
        end

        function naive_time(nθ)
            idx = nθ == 1 ? [2] : round.(Int, range(2, n - 1; length = nθ))
            A = copy(A_base)
            h = 1e-6
            Bramble.pde_solve(A, F)   # warm-up
            return @elapsed for i in idx
                Aperturbed = copy(A)
                Aperturbed[i, i] += h
                (Bramble.pde_solve(Aperturbed, F) .- Bramble.pde_solve(A, F)) ./ h
            end
        end

        t_adjoint = [adjoint_time(k) for k in (1, 10, 50)]
        t_naive = [naive_time(k) for k in (1, 10, 50)]

        # "Roughly flat" for the adjoint path: the 50-parameter case costs no more than a
        # small constant factor over the 1-parameter case (compilation/GC noise, not growth).
        @test t_adjoint[end] < 5 * t_adjoint[1]
        # The naive path grows with `nθ`, and by the top of this range clearly outpaces the
        # adjoint path solving the identical problem.
        @test t_naive[end] > 10 * t_naive[1]
        @test t_naive[end] > 5 * t_adjoint[end]
    end
end

end # module
