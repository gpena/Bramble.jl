module ConvergenceHeat1dTests

# Orders of the 1D heat equation `uₜ = uₓₓ + f` (TestUtils `_heat_uex`), stepped by hand and
# through the SciML entry points, and of the steady Poisson problems behind
# `linear_problem` and `nonlinear_problem`. Moved here from test/form/semidiscrete.jl and
# test/ext/sciml_ext.jl.
#
# `Bramble.domain`/`Bramble.interval`/`Bramble.mesh`/`Bramble.element` are qualified for the
# reason test/ext/meshes_ext.jl gives.

using Test
using Bramble
using Bramble: mass_matrix, operator_matrix
using LinearAlgebra: mul!
using SciMLBase: SciMLBase, solve
using OrdinaryDiffEqBDF: FBDF, QNDF
using OrdinaryDiffEqRosenbrock: Rodas5P
using NonlinearSolve: NewtonRaphson
using ADTypes: AutoFiniteDiff, AutoForwardDiff
using ..TestUtils: _check_eoc, _heat_src, _heat_uex, _sciml_setup, _sd_problem

const _SD_T = 0.1
const _NEWTON = NewtonRaphson(; autodiff = AutoForwardDiff())

# The semidiscretisation is second order in space. Backward Euler is written out here rather
# than imported so the claim rests on nothing but Bramble: on a constrained row `M` is zero
# and `A` is `eₖ`, so `(M + Δt A) u = M u + Δt F` reduces to `u[i] = g(x_i, t)` and the
# boundary condition is imposed by the same step that advances the interior. `Δt = h²` keeps
# the first-order time error at the order being measured.
@testset "semidiscretize: backward Euler order" begin
    function step_to(n)
        Ωₕ, Wₕ, fₕ, a, l = _sd_problem(n)
        h = hₘₐₓ(Ωₕ)
        bcs = dirichlet_constraints(
            Ωₕ, Bramble.interval(0.0, _SD_T), :boundary => (x, t) -> 0.0
        )
        sd = semidiscretize(
            a, l; dirichlet = bcs, (update_coefficients!) = t -> Rₕ!(fₕ, x -> _heat_src(x, t))
        )

        M = mass_matrix(sd)
        nsteps = ceil(Int, _SD_T / h^2)
        Δt = _SD_T / nsteps
        K = M + Δt * operator_matrix(sd)

        u = collect(parent(Rₕ(Wₕ, x -> _heat_uex(x, 0.0))))
        dirichlet_bc!(u, sd, 0.0)
        F = similar(u)
        zero_u = zero(u)
        for k in 1:nsteps
            t = k * Δt
            sd(F, zero_u, nothing, t)      # F(t), the residual at u = 0
            u = K \ (M * u + Δt * F)
        end

        uₕ = Bramble.element(Wₕ)
        parent(uₕ) .= u
        return normₕ(Rₕ(Wₕ, x -> _heat_uex(x, _SD_T)) - uₕ), h
    end

    eoc = _check_eoc(step_to, (11, 21, 41, 81))
    @test last(eoc) > 1.95
end

@testset "SciML entry points" begin
    # The heat equation with the manufactured solution `exp(-t) sin(πx)`, stepped to t = 1.
    # The space tolerance is loose and the time tolerance tight, so what is measured is the
    # semidiscretisation's second order rather than the stepper's.
    @testset "OrdinaryDiffEq: order of convergence" begin
        function solve_to(n, alg; kwargs...)
            _, Wₕ, I, _, _, sd = _sciml_setup(n)
            prob = ode_problem(sd, Rₕ(Wₕ, x -> _heat_uex(x, 0.0)), I; kwargs...)
            sol = solve(prob, alg; reltol = 1e-11, abstol = 1e-13)
            @test SciMLBase.successful_retcode(sol)
            uₕ = Bramble.element(Wₕ)
            parent(uₕ) .= sol.u[end]
            return normₕ(Rₕ(Wₕ, x -> _heat_uex(x, 1.0)) - uₕ), hₘₐₓ(Bramble.mesh(Wₕ))
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
            gₕ = Rₕ(Wₕ, x -> -_heat_src(x, t))
            l_t = form(Wₕ, v -> innerₕ(gₕ, v))
            assemble!(dT, l_t)
            return dT
        end
        _check_eoc(n -> solve_to(n, Rodas5P(); tgrad = exact_tgrad), (11, 21, 41))
    end

    # The same gate asks for manufactured solutions verified across the *PDE* solvers, not
    # only the ODE ones. `test/convergence/variable_coefficient_poisson.jl` already pins the
    # assemble -> boundary-condition -> solve pipeline, but it solves with `\`; what that
    # leaves unverified is the order actually delivered through the SciML entry points, which
    # is what a caller of this extension gets. Elsewhere in this file those wrappers are only
    # checked against `A \ F`'s own answer, which cannot catch an error both paths share.
    # Manufactured solutions are recovered through the steady solvers.
    @testset "steady solvers: manufactured" begin
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

end # module ConvergenceHeat1dTests
