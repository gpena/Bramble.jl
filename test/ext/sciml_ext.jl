module ExtSciMlExtTests

using Test
using Bramble
using Bramble: ode_function
using Bramble: SecondOrderSemidiscretization, block_mass_matrix, damping_matrix,
               jacobian_prototype, mass_matrix, operator_matrix, semidiscretize_rhs,
               stiffness_matrix
using SparseArrays
using LinearAlgebra: mul!, I as Identity
using SciMLBase: SciMLBase, ODEProblem, LinearProblem, NonlinearProblem, solve
using OrdinaryDiffEqBDF: FBDF, QNDF
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqTsit5: Tsit5
using ..TestUtils: _check_eoc
using NonlinearSolve: NewtonRaphson
using ADTypes: AutoFiniteDiff, AutoForwardDiff

# Every `NewtonRaphson` below names its Jacobian backend. Left to choose, NonlinearSolve
# picks `AutoPolyesterForwardDiff` whenever PolyesterForwardDiff is loaded (the `full`
# group loads it), and Polyester's closure path fails on the macOS CI runners with
# "closures are not supported on this platform" -- two errors in every Weekly macOS leg
# since v3.4.0, from a choice that depends on which packages happen to be loaded.
const _NEWTON = NewtonRaphson(; autodiff = AutoForwardDiff())
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
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
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

    @testset "ode_problem: p reaches a parametric residual" begin
        # `θ` scales the boundary value and its rate, `(x, t, θ) -> θ[1] + θ[2] * t`, threaded
        # through the residual's own `p` -- gpena/Bramble.jl#239's own gap: nothing before this
        # let a Dirichlet condition see the ODEProblem's parameter at all.
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t, θ) -> θ[1] + θ[2] * t)
        sd_p = semidiscretize(a, l; dirichlet = bcs)
        u₀ = Rₕ(Wₕ, x -> 0.0)
        θ = (1.0, 2.0)

        prob = ode_problem(sd_p, u₀, I; p = θ)
        @test prob.p === θ
        @test prob.u0[1] ≈ 1.0 && prob.u0[n] ≈ 1.0   # dirichlet_bc!'s own consistency step used θ too

        sol = solve(prob, FBDF())
        @test SciMLBase.successful_retcode(sol)
        @test sol.u[end][1] ≈ θ[1] + θ[2] * sol.t[end]
        @test sol.u[end][n] ≈ θ[1] + θ[2] * sol.t[end]

        # A different θ at the same problem reaches a different trajectory -- confirms θ is
        # read fresh from `p` on every residual call, not captured once at `ode_problem` time.
        sol2 = solve(ode_problem(sd_p, u₀, I; p = (0.0, 1.0)), FBDF())
        @test sol2.u[end][1] ≈ sol2.t[end]

        # Omitting `p` on a non-parametric problem is unchanged: `NullParameters`, not an
        # error, exactly as before `p` reached anything.
        bcs_t = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 1 + t)
        sd_t = semidiscretize(a, l; dirichlet = bcs_t)
        @test ode_problem(sd_t, u₀, I).p isa SciMLBase.NullParameters
    end

    # `semidiscretize_rhs` (gpena/Bramble.jl#163): the point of folding `M⁻¹` in ahead of
    # time is reaching solvers that cannot touch a mass matrix at all -- `Tsit5` is one, and
    # is checked to actually reject `sd`'s own `ODEProblem` below, not just to work on
    # `rhs`'s. Agreement against a mass-matrix-aware `Rodas5P` solve of the same system is
    # to solver tolerance, not literal bit-exactness: the two formulations reach the answer
    # through different floating-point operations (`M \ (F - Au)` inside the stepper vs.
    # `M⁻¹` pre-multiplied), so they need not land on the identical last bit.
    @testset "semidiscretize_rhs: matrix-free explicit right-hand side reaches solvers ode_problem(sd, ...) cannot" begin
        sd_free = semidiscretize(a, l)
        @test sd_free.constraints isa Bramble.NoConstraints
        rhs = semidiscretize_rhs(sd_free)

        u₀ = Rₕ(Wₕ, x -> sinpi(x[1]))
        before = copy(parent(u₀))
        prob_rhs = ode_problem(rhs, u₀, I)
        @test prob_rhs isa ODEProblem
        @test prob_rhs.tspan == (0.0, 1.0)
        @test parent(u₀) == before                  # u₀ is never mutated
        @test prob_rhs.u0 !== parent(u₀)

        # `Tsit5` cannot step a system with a mass matrix at all -- confirmed directly
        # against `sd_free`'s own `ODEProblem`, not assumed.
        prob_sd = ode_problem(sd_free, u₀, I)
        @test_throws ErrorException solve(prob_sd, Tsit5())

        sol_rhs = solve(prob_rhs, Tsit5(); reltol = 1e-12, abstol = 1e-14)
        sol_sd = solve(prob_sd, Rodas5P(); reltol = 1e-12, abstol = 1e-14)
        @test SciMLBase.successful_retcode(sol_rhs)
        @test SciMLBase.successful_retcode(sol_sd)
        @test sol_rhs.u[end]≈sol_sd.u[end] atol=1e-10 rtol=1e-10
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
        sol_oop = solve(nonlinear_problem(residual, zeros(n)), _NEWTON)
        sol_iip = solve(nonlinear_problem(residual!, zeros(n)), _NEWTON)
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
            _check_eoc(n -> solve_to(n, alg), (11, 21, 41))
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
        _check_eoc(n -> solve_to(n, Rodas5P(); tgrad = exact_tgrad), (11, 21, 41))
    end

    # `semidiscretize(build, l; ...)` (src/problems/semidiscrete.jl) is the other half of #107:
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
            aα = form(Wₕ, Wₕ, (u, v) -> inner₊(αₕ * ∇ₕ(u), ∇ₕ(v)))
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
        a_ref = form(Wₕ, Wₕ, (u, v) -> inner₊(αₕ_ref * ∇ₕ(u), ∇ₕ(v)))
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
        a_s = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
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
            K = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
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

            _check_eoc(solve_to, (11, 21, 41))
        end
    end

    # v2.13.0's verification gate asks for zero-allocation evaluations *during time
    # stepping*, which is a stronger claim than the one `test/form/semidiscrete.jl` already
    # pins: that file calls `sd(du, u, p, t)` at a state it picked itself, while this samples
    # the states an integrator actually reaches, after it has adapted its step size and, for
    # the BDF and Rosenbrock methods, refactorised at least once. Both measurements sit
    # behind a function barrier, since `@allocated` over globals reports the caller's own
    # boxing rather than the routine's work (`bramble-verification` §1).
    # Measured through these two argument-passing helpers rather than `@allocated` written
    # inline in the loop below: with the state read straight out of `integrator.u` at the
    # measurement site, both readings sat at a constant 16 B and 112 B regardless of mesh
    # size or stepper, the fixed-cost signature of the call boxing its arguments rather than
    # of the routine allocating (`bramble-verification` §1). Passed as arguments, both go to
    # zero, matching what `test/form/semidiscrete.jl` measures the same way.
    _traj_rhs_allocs(sd, du, u, t) = @allocated sd(du, u, nothing, t)
    _traj_jac_allocs(J, sd, u, t) = @allocated Bramble.jacobian!(J, sd, u, nothing, t)

    @testset "zero allocations along an integrator's trajectory" begin
        function _trajectory_allocs(n, alg)
            _, Wₕ, I, _, _, sd = _sciml_setup(n)
            u0 = Rₕ(Wₕ, x -> _sciml_uex(x, 0.0))
            integrator = SciMLBase.init(
                ode_problem(sd, u0, I), alg; reltol = 1e-8, abstol = 1e-10
            )
            du = similar(parent(u0))
            J = jacobian_prototype(sd)

            # Warm both paths at the integrator's own starting state, so what the loop below
            # measures is steady-state work rather than first-call compilation.
            _traj_rhs_allocs(sd, du, integrator.u, integrator.t)
            _traj_jac_allocs(J, sd, integrator.u, integrator.t)

            worst_rhs = 0
            worst_jac = 0
            for _ in 1:8
                SciMLBase.step!(integrator)
                worst_rhs = max(
                    worst_rhs, _traj_rhs_allocs(sd, du, integrator.u, integrator.t)
                )
                worst_jac = max(
                    worst_jac, _traj_jac_allocs(J, sd, integrator.u, integrator.t)
                )
            end
            return worst_rhs, worst_jac
        end

        for alg in (FBDF(), QNDF(), Rodas5P(; autodiff = AutoFiniteDiff()))
            rhs_allocs, jac_allocs = _trajectory_allocs(21, alg)
            @test rhs_allocs == 0
            @test jac_allocs == 0
        end
    end

    # The same gate asks for manufactured solutions verified across the *PDE* solvers, not
    # only the ODE ones. `test/drivers/variable_coefficient_poisson.jl` already pins the
    # assemble -> boundary-condition -> solve pipeline, but it solves with `\`; what that
    # leaves unverified is the order actually delivered through the SciML entry points, which
    # is what a caller of this extension gets. Elsewhere in this file those wrappers are only
    # checked against `A \ F`'s own answer, which cannot catch an error both paths share.
    @testset "manufactured solutions through the steady solvers" begin
        _poisson_uex(x) = sinpi(x[1])
        _poisson_src(x) = pi^2 * sinpi(x[1])

        function _linear_error(n)
            Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), n)
            Wₕ = gridspace(Ωₕ)
            fₕ = Bramble.element(Wₕ)
            avgₕ!(fₕ, _poisson_src)
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            l = form(Wₕ, v -> innerₕ(fₕ, v))
            uₕ = solve(a, l; dirichlet = :boundary => _poisson_uex)
            return normₕ(Rₕ(Wₕ, _poisson_uex) - uₕ), hₘₐₓ(Ωₕ)
        end

        # The nonlinear counterpart: -(α(u) u')' = f with α(u) = 3 + 1/(1 + u²), whose
        # manufactured solution is exp(x). Solved through `nonlinear_problem` and
        # `NewtonRaphson`, so the second order has to survive the Newton path as well as the
        # linear one.
        _nl_sol(x) = exp(x[1])
        _nl_α(u) = 3 + 1 / (1 + u^2)
        _nl_dα(u) = -2u / (1 + u^2)^2
        _nl_src(x) = -_nl_dα(_nl_sol(x)) * _nl_sol(x)^2 - _nl_α(_nl_sol(x)) * _nl_sol(x)

        function _nonlinear_error(n)
            Ω = Bramble.domain(Bramble.interval(0.0, 1.0))
            Ωₕ = Bramble.mesh(Ω, n)
            Wₕ = gridspace(Ωₕ)
            bcs = dirichlet_constraints(Ω, :boundary => _nl_sol)
            gₕ = Bramble.element(Wₕ)
            avgₕ!(gₕ, _nl_src)
            F = assemble(form(Wₕ, v -> innerₕ(gₕ, v)); dirichlet = bcs)

            # `eltype(u)`, not the space's own: `NewtonRaphson` builds its Jacobian by
            # forward-mode AD, so this is called with `ForwardDiff.Dual` state. Typing the
            # element from the space instead is exactly the defect `bramble-verification` §4
            # collects, and shows up here as `MethodError: Float64(::Dual)`.
            function residual!(r, u, p)
                uₕ = Bramble.element(Wₕ, eltype(u))
                parent(uₕ) .= u
                αv = _nl_α.(Mₕ(uₕ))
                A = assemble(
                    form(Wₕ, Wₕ, (U, V) -> inner₊(αv * ∇ₕ(U), ∇ₕ(V))); dirichlet = :boundary
                )
                mul!(r, A, u)
                r .-= F
                return nothing
            end

            sol = solve(nonlinear_problem(residual!, copy(F)), _NEWTON)
            @test SciMLBase.successful_retcode(sol)
            uₕ = Bramble.element(Wₕ)
            parent(uₕ) .= sol.u
            return normₕ(Rₕ(Wₕ, _nl_sol) - uₕ), hₘₐₓ(Ωₕ)
        end

        for (name, errfn) in (("linear_problem", _linear_error),
            ("nonlinear_problem", _nonlinear_error))
            _check_eoc(errfn, (11, 21, 41))
        end
    end
end

end # module ExtSciMlExtTests
