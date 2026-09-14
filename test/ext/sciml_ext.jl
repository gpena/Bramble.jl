module ExtSciMlExtTests

using Test
using Bramble
using SparseArrays
using LinearAlgebra: mul!, I as Identity
using SciMLBase: SciMLBase, ODEProblem, LinearProblem, NonlinearProblem, solve
using OrdinaryDiffEqBDF: FBDF, QNDF
using OrdinaryDiffEqRosenbrock: Rodas5P
using NonlinearSolve: NewtonRaphson
using ADTypes: AutoFiniteDiff
using LinearSolve: KrylovJL_GMRES

# BrambleSciMLExt: the `ODEFunction`/`ODEProblem`/`LinearProblem` wrapping of a
# `Semidiscretization`. The semidiscretisation itself, and its order of convergence, are
# covered without any solver package in test/form/semidiscrete.jl -- what is left to check
# here is that the pieces reach SciMLBase intact and that a stiff solver actually steps the
# differential-algebraic system to the right answer.
#
# Lives in the "ext" group, so `OrdinaryDiffEq*` is precompiled only when that group runs
# and never on the default push CI.
#
# `Bramble.domain`/`Bramble.interval`/`Bramble.mesh`/`Bramble.element` are qualified
# throughout for the reason meshes_ext.jl gives: every ext file is included into the same
# `Main`, where `Meshes.jl` has its own `domain`/`mesh`/`element`.

_sciml_uex(x, t) = exp(-t) * sinpi(x[1])
_sciml_src(x, t) = (pi^2 - 1) * exp(-t) * sinpi(x[1])

function _sciml_setup(n; T = 1.0)
    Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), n)
    Wₕ = gridspace(Ωₕ)
    I = Bramble.interval(0.0, T)
    fₕ = Bramble.element(Wₕ, 0.0)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)
    sd = semidiscretize(
        a, l; dirichlet = bcs, (update_coefficients!) = t -> Rₕ!(fₕ, x -> _sciml_src(x, t))
    )
    return Ωₕ, Wₕ, I, a, l, sd
end

@testset "BrambleSciMLExt" begin
    Ωₕ, Wₕ, I, a, l, sd = _sciml_setup(21)
    n = ndofs(Wₕ)

    @testset "ode_function carries the mass matrix, Jacobian and sparsity" begin
        f = ode_function(sd)
        @test f isa SciMLBase.ODEFunction
        @test f.mass_matrix == mass_matrix(sd)
        @test f.jac_prototype == operator_matrix(sd)

        # The Jacobian closure fills its argument with -A.
        J = jacobian_prototype(sd)
        f.jac(J, zeros(n), nothing, 0.0)
        @test J == -operator_matrix(sd)

        # `jacobian = nothing` leaves `jac` unset, for a solver to build one by AD instead.
        @test ode_function(sd; jacobian = nothing).jac === nothing

        # A caller-supplied prototype is passed through untouched.
        proto = spdiagm(0 => ones(n))
        @test ode_function(sd; jac_prototype = proto).jac_prototype === proto

        # `tgrad = nothing` leaves the field unset, for SciML to build ∂f/∂t by AD instead.
        @test ode_function(sd).tgrad === nothing

        # The plain SciML signature `(dT, u, p, t)` is passed straight through.
        plain_tgrad(dT, u, p, t) = fill!(dT, t)
        f_plain = ode_function(sd; tgrad = plain_tgrad)
        dT = zeros(n)
        f_plain.tgrad(dT, zeros(n), nothing, 2.0)
        @test all(==(2.0), dT)

        # The Bramble-aware signature `(dT, sd, u, p, t)` closes over `sd`.
        aware_tgrad(dT, sd, u, p, t) = fill!(dT, ndofs(space(sd)))
        f_aware = ode_function(sd; tgrad = aware_tgrad)
        fill!(dT, 0.0)
        f_aware.tgrad(dT, zeros(n), nothing, 2.0)
        @test all(==(n), dT)

        # The two-form method builds the semidiscretisation on the way.
        @test ode_function(a, l; dirichlet = :boundary) isa SciMLBase.ODEFunction
    end

    @testset "ode_problem: time domain, and a copied, consistent u₀" begin
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 5 + t)
        sd_bc = semidiscretize(a, l; dirichlet = bcs)

        u₀ = Rₕ(Wₕ, x -> 3.0)
        before = copy(parent(u₀))
        prob = ode_problem(sd_bc, u₀, I)

        @test prob isa ODEProblem
        @test prob.tspan == (0.0, 1.0)
        # A `CartesianProduct{1}` and a plain tuple name the same time domain.
        @test ode_problem(sd_bc, u₀, (0.0, 1.0)).tspan == prob.tspan

        @test parent(u₀) == before                      # u₀ is never mutated
        @test prob.u0 !== parent(u₀)
        @test prob.u0[1] ≈ 5.0 && prob.u0[n] ≈ 5.0      # made consistent at t₀
        @test prob.u0[5] ≈ 3.0                          # interior untouched

        # A plain vector works as the initial condition too.
        @test ode_problem(sd_bc, fill(3.0, n), I).u0[1] ≈ 5.0
        @test ode_problem(a, l, u₀, I; dirichlet = bcs).u0[1] ≈ 5.0
    end

    @testset "linear_problem is the assembled steady system" begin
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
        prob = linear_problem(a, l; dirichlet = :boundary => x -> 0.0)
        @test prob isa LinearProblem
        @test prob.A == A
        @test prob.b == F

        # `symmetrize` is forwarded to `assemble`.
        As, Fs = assemble(a, l; dirichlet = :boundary => x -> 1.0, symmetrize = true)
        prob_s = linear_problem(a, l; dirichlet = :boundary => x -> 1.0, symmetrize = true)
        @test prob_s.A == As
        @test prob_s.b == Fs
    end

    # `element(Wₕ, sol)`/`VectorElement(sol, Wₕ)` unwrap a `LinearSolve` solution into a
    # `VectorElement`, and `solve(a, l; ...)` does assembly, solve and unwrapping in one call
    # -- the three pieces #156 asks for, all reached through the steady system `prob`/`A`/`F`
    # already agree on above.
    @testset "solve: VectorElement from a LinearSolution" begin
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
        expected = A \ F

        prob = linear_problem(a, l; dirichlet = :boundary => x -> 0.0)
        sol = solve(prob)
        @test sol isa SciMLBase.LinearSolution

        uₕ = Bramble.element(Wₕ, sol)
        @test uₕ isa Bramble.VectorElement
        @test space(uₕ) === Wₕ
        @test parent(uₕ) ≈ expected

        vₕ = Bramble.VectorElement(sol, Wₕ)
        @test vₕ isa Bramble.VectorElement
        @test parent(vₕ) ≈ expected

        # The high-level convenience: assemble, solve and unwrap in one call.
        wₕ = solve(a, l; dirichlet = :boundary => x -> 0.0)
        @test wₕ isa Bramble.VectorElement
        @test space(wₕ) === Wₕ
        @test parent(wₕ) ≈ expected

        # `symmetrize` and an explicit `solver` are both forwarded.
        ws = solve(
            a, l; dirichlet = :boundary => x -> 1.0, symmetrize = true,
            solver = KrylovJL_GMRES()
        )
        As, Fs = assemble(a, l; dirichlet = :boundary => x -> 1.0, symmetrize = true)
        @test parent(ws) ≈ As \ Fs
    end

    @testset "nonlinear_problem: residual, jac_prototype, and a copied u0" begin
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)

        residual(u, p) = A * u .- F
        function residual!(res, u, p)
            mul!(res, A, u)
            return res .-= F
        end

        u0 = Rₕ(Wₕ, x -> 3.0)
        before = copy(parent(u0))

        prob = nonlinear_problem(residual, u0)
        @test prob isa NonlinearProblem
        @test parent(u0) == before          # u0 is never mutated
        @test prob.u0 !== parent(u0)

        # A plain vector works as the initial condition too.
        @test nonlinear_problem(residual, fill(3.0, n)) isa NonlinearProblem

        # Out-of-place and in-place residuals reach the same answer, both agreeing with the
        # direct linear solve (a linear residual, so Newton converges in one step exactly).
        sol_oop = solve(nonlinear_problem(residual, zeros(n)), NewtonRaphson())
        sol_iip = solve(nonlinear_problem(residual!, zeros(n)), NewtonRaphson())
        @test SciMLBase.successful_retcode(sol_oop)
        @test SciMLBase.successful_retcode(sol_iip)
        @test sol_oop.u ≈ A \ F
        @test sol_iip.u ≈ A \ F

        # `jac_prototype` and `jacobian` are passed straight through to `NonlinearFunction`.
        proto = spdiagm(0 => ones(n))
        @test nonlinear_problem(residual, zeros(n); jac_prototype = proto).f.jac_prototype === proto
        @test nonlinear_problem(residual, zeros(n)).f.jac === nothing
        # Out-of-place, matching `residual`'s own convention: `NonlinearFunction` requires
        # `jac`'s in-place/out-of-place convention to match `f`'s, never mixed.
        manual_jac(u, p) = A
        @test nonlinear_problem(residual, zeros(n); jacobian = manual_jac).f.jac === manual_jac
    end

    # The heat equation with the manufactured solution `exp(-t) sin(πx)`, stepped to t = 1.
    # The space tolerance is loose and the time tolerance tight, so what is measured is the
    # semidiscretisation's second order rather than the stepper's.
    @testset "order of convergence through OrdinaryDiffEq" begin
        function solve_to(n, alg; kwargs...)
            _, Wₕ, I, _, _, sd = _sciml_setup(n)
            prob = ode_problem(sd, Rₕ(Wₕ, x -> _sciml_uex(x, 0.0)), I; kwargs...)
            sol = solve(prob, alg; reltol = 1e-11, abstol = 1e-13)
            @test SciMLBase.successful_retcode(sol)
            uₕ = Bramble.element(Wₕ)
            parent(uₕ) .= sol.u[end]
            return normₕ(Rₕ(Wₕ, x -> _sciml_uex(x, 1.0)) - uₕ), hₘₐₓ(Bramble.mesh(Wₕ))
        end

        # `Rodas5P` is a Rosenbrock method and needs `∂f/∂t`. The `update_coefficients!`
        # hook writes into a `Float64` element, so that derivative cannot be taken by
        # differentiating through `t` -- hence `AutoFiniteDiff`. The BDF methods need no
        # `∂f/∂t` at all. See the note on `ode_function`.
        for alg in (FBDF(), QNDF(), Rodas5P(; autodiff = AutoFiniteDiff()))
            errors = Float64[]
            spacings = Float64[]
            for n in (11, 21, 41)
                e, h = solve_to(n, alg)
                push!(errors, e)
                push!(spacings, h)
            end
            eoc = [log(errors[i] / errors[i + 1]) / log(spacings[i] / spacings[i + 1]) for
                   i in 1:(length(errors) - 1)]
            @test all(>(1.9), eoc)
            @test issorted(errors; rev = true)
        end

        # An analytical `tgrad` -- exact here since `∂f/∂t = -f` for this manufactured
        # source -- lets the default Rosenbrock `autodiff` run: no `AutoFiniteDiff`, and no
        # differentiation through `t` at all. Built the same way `l` itself is (through
        # `innerₕ`, not raw nodal values), so it carries the same mass weighting as `F(t)`.
        exact_tgrad(dT, sd, u, p, t) = begin
            Wₕ = space(sd)
            gₕ = Rₕ(Wₕ, x -> -_sciml_src(x, t))
            l_t = form(Wₕ, v -> innerₕ(gₕ, v))
            assemble!(dT, l_t)
            return dT
        end
        errors = Float64[]
        spacings = Float64[]
        for n in (11, 21, 41)
            e, h = solve_to(n, Rodas5P(); tgrad = exact_tgrad)
            push!(errors, e)
            push!(spacings, h)
        end
        eoc = [log(errors[i] / errors[i + 1]) / log(spacings[i] / spacings[i + 1]) for
               i in 1:(length(errors) - 1)]
        @test all(>(1.9), eoc)
        @test issorted(errors; rev = true)
    end

    # `semidiscretize(build, l; ...)` (src/form/semidiscrete.jl) is the other half of #107:
    # an operator that genuinely depends on `t` -- not just the source -- built fresh per
    # element type instead of one fixed `Float64`-typed matrix. This is what lets `Rodas5P`'s
    # *default* `autodiff` differentiate through `t` at all: the classic `BilinearForm` path
    # closes over a `Float64` coefficient buffer and throws `InexactError` under that same
    # sweep (the `AutoFiniteDiff`/BDF workarounds above exist because of exactly this).
    @testset "type-cached operator: default autodiff through a t-dependent A(t)" begin
        Ωₕ, Wₕ, I, _, _, _ = _sciml_setup(21)
        α(t) = 1.0 + t   # x-independent: only the value matters for this cross-check

        function build_diffusion(t)
            αₕ = Bramble.element(Wₕ, typeof(t))
            aα = form(Wₕ, Wₕ, (u, v) -> inner₊(αₕ * ∇₋ₕ(u), ∇₋ₕ(v)))
            refill!(t) = (fill!(parent(αₕ), α(t)); nothing)
            return aα, refill!
        end

        # The source is time-independent here on purpose: a `t`-dependent source reached
        # through `update_coefficients!` has the same `Float64`-buffer limitation the note on
        # `ode_function` already documents, and is a separate concern from the operator this
        # testset isolates -- `tgrad`, above, is the fix for that half.
        fₕ = Rₕ(Wₕ, x -> sinpi(x[1]))
        l = form(Wₕ, v -> innerₕ(fₕ, v))
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)

        αₕ_ref = Bramble.element(Wₕ, 0.0)
        a_ref = form(Wₕ, Wₕ, (u, v) -> inner₊(αₕ_ref * ∇₋ₕ(u), ∇₋ₕ(v)))
        sd_ref = semidiscretize(
            a_ref, l; dirichlet = bcs, reassemble = true,
            (update_coefficients!) = t -> (fill!(parent(αₕ_ref), α(t)); nothing)
        )
        u0 = Rₕ(Wₕ, x -> _sciml_uex(x, 0.0))
        sol_ref = solve(
            ode_problem(sd_ref, u0, I), FBDF(); reltol = 1e-11, abstol = 1e-13
        )

        sd = semidiscretize(build_diffusion, l; dirichlet = bcs)
        prob = ode_problem(sd, u0, I)

        # Default `autodiff`: no `AutoFiniteDiff`, no BDF fallback.
        sol = solve(prob, Rodas5P(); reltol = 1e-11, abstol = 1e-13)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u[end] ≈ sol_ref.u[end] rtol = 1e-6

        # A stiff BDF method on the same `build`-based operator agrees too -- the type
        # caching is transparent to a stepper that never asks for `t` as a `Dual`.
        sol_fbdf = solve(prob, FBDF(); reltol = 1e-11, abstol = 1e-13)
        @test sol_fbdf.u[end] ≈ sol_ref.u[end] rtol = 1e-8
    end

    # A constant Dirichlet value is the steady state the parabolic problem relaxes onto, so
    # stepping far enough must reproduce the solution of `A u = F` -- the same system
    # `linear_problem` hands to LinearSolve. This ties the two entry points together.
    @testset "long-time limit is the steady solve" begin
        Ωₕ, Wₕ, _, _, _, _ = _sciml_setup(21)
        gₕ = Rₕ(Wₕ, x -> 1.0)
        a_s = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        l_s = form(Wₕ, v -> innerₕ(gₕ, v))
        bc = :boundary => x -> 0.0

        sd_s = semidiscretize(a_s, l_s; dirichlet = bc)
        prob = ode_problem(sd_s, Bramble.element(Wₕ, 0.0), (0.0, 20.0))
        sol = solve(prob, FBDF(); reltol = 1e-10, abstol = 1e-12)

        A, F = assemble(a_s, l_s; dirichlet = bc)
        @test sol.u[end] ≈ A \ F rtol = 1e-6
    end

    # `M ü + K u = F(t)`: the second-order counterpart of everything above. Manufactured
    # solution `sin(πx) cos(πt)` satisfies the *homogeneous* 1D wave equation with `c = 1`
    # exactly (both ∂ₜ² and ∂ₓ² give `-π² sin(πx) cos(πt)`), zero at both endpoints for every
    # `t` -- so `l ≡ 0` and the only Dirichlet data needed is the constant `0`.
    @testset "second-order semidiscretisation (wave equation)" begin
        _wave_uex(x, t) = sinpi(x[1]) * cospi(t)
        _wave_duex(x, t) = -pi * sinpi(x[1]) * sinpi(t)

        function _wave_setup(n)
            Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), n, true)
            Wₕ = gridspace(Ωₕ)
            Iv = Bramble.interval(0.0, 1.0)
            fₕ = Bramble.element(Wₕ, 0.0)
            K = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            l = form(Wₕ, v -> innerₕ(fₕ, v))
            bcs = dirichlet_constraints(Ωₕ, Iv, :boundary => (x, t) -> 0.0)
            return Ωₕ, Wₕ, Iv, K, l, bcs
        end

        Ωₕ, Wₕ, Iv, K, l, bcs = _wave_setup(21)
        n = ndofs(Wₕ)

        @testset "semidiscretize_second_order: accessors and display" begin
            sd = semidiscretize_second_order(K, l; dirichlet = bcs)
            @test sd isa SecondOrderSemidiscretization
            @test size(stiffness_matrix(sd)) == (n, n)
            @test size(mass_matrix(sd)) == (n, n)
            @test damping_matrix(sd) === nothing

            # Constrained rows: `eₖ` on the stiffness, zero on the mass.
            @test stiffness_matrix(sd)[1, :] == [i == 1 ? 1.0 : 0.0 for i in 1:n]
            @test all(iszero, mass_matrix(sd)[1, :])
            @test all(iszero, mass_matrix(sd)[n, :])

            Mb = block_mass_matrix(sd)
            @test size(Mb) == (2n, 2n)
            @test Mb[(n + 1):(2n), (n + 1):(2n)] == Identity(n)

            # An optional damping form is assembled the same way, zero rows included.
            Cform = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
            sd_c = semidiscretize_second_order(K, l; damping = Cform, dirichlet = bcs)
            @test damping_matrix(sd_c) !== nothing
            @test all(iszero, damping_matrix(sd_c)[1, :])

            @test !isempty(sprint(show, sd))
            @test !isempty(sprint(show, MIME"text/plain"(), sd))
        end

        @testset "second_order_ode_problem: time domain, and a copied, consistent u₀/du₀" begin
            u₀ = Rₕ(Wₕ, x -> _wave_uex(x, 0.0))
            du₀ = Rₕ(Wₕ, x -> _wave_duex(x, 0.0))
            before_u = copy(parent(u₀))
            before_du = copy(parent(du₀))

            sd = semidiscretize_second_order(K, l; dirichlet = bcs)
            prob = second_order_ode_problem(sd, du₀, u₀, Iv)

            @test prob isa SciMLBase.ODEProblem
            @test prob.tspan == (0.0, 1.0)
            @test second_order_ode_problem(sd, du₀, u₀, (0.0, 1.0)).tspan == prob.tspan

            @test parent(u₀) == before_u    # u₀ is never mutated
            @test parent(du₀) == before_du  # du₀ is never mutated
            @test prob.u0.x[2] !== parent(u₀)
            @test prob.u0.x[2][1] ≈ 0.0 && prob.u0.x[2][n] ≈ 0.0   # made consistent at t₀
            @test prob.u0.x[1] == parent(du₀)                       # du₀ passed through as-is

            # The two-form method builds the semidiscretisation on the way.
            @test second_order_ode_problem(K, l, du₀, u₀, Iv; dirichlet = bcs) isa
                  SciMLBase.ODEProblem
        end

        # VelocityVerlet et al. cannot be used at all: `OrdinaryDiffEqCore` refuses any
        # explicit/symplectic solver unless the mass matrix is *exactly* `I`, and a discrete
        # `innerₕ` mass matrix never is (its boundary rows always carry a half-weight) --
        # documented on `SecondOrderSemidiscretization`. `Rodas5P`, already a test dependency
        # for the first-order suite above, is mass-matrix-aware and used here instead.
        @testset "order of convergence through OrdinaryDiffEq" begin
            function solve_to(n)
                _, Wₕ, Iv, K, l, bcs = _wave_setup(n)
                sd = semidiscretize_second_order(K, l; dirichlet = bcs)
                u₀ = Rₕ(Wₕ, x -> _wave_uex(x, 0.0))
                du₀ = Rₕ(Wₕ, x -> _wave_duex(x, 0.0))
                prob = second_order_ode_problem(sd, du₀, u₀, Iv)
                sol = solve(prob, Rodas5P(); reltol = 1e-11, abstol = 1e-13)
                @test SciMLBase.successful_retcode(sol)
                uₕ = Bramble.element(Wₕ, sol.u[end].x[2])
                return normₕ(Rₕ(Wₕ, x -> _wave_uex(x, 1.0)) - uₕ), hₘₐₓ(Bramble.mesh(Wₕ))
            end

            errors = Float64[]
            spacings = Float64[]
            for n in (11, 21, 41)
                e, h = solve_to(n)
                push!(errors, e)
                push!(spacings, h)
            end
            eoc = [log(errors[i] / errors[i + 1]) / log(spacings[i] / spacings[i + 1]) for
                   i in 1:(length(errors) - 1)]
            @test all(>(1.9), eoc)
            @test issorted(errors; rev = true)
        end
    end
end

end # module ExtSciMlExtTests
