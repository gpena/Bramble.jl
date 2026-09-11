using Test
using Bramble
using SparseArrays
using SciMLBase: SciMLBase, ODEProblem, LinearProblem, solve
using OrdinaryDiffEqBDF: FBDF, QNDF
using OrdinaryDiffEqRosenbrock: Rodas5P
using ADTypes: AutoFiniteDiff

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

function _sciml_setup(n; T=1.0)
    Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), n)
    Wₕ = gridspace(Ωₕ)
    I = Bramble.interval(0.0, T)
    fₕ = Bramble.element(Wₕ, 0.0)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)
    sd = semidiscretize(
        a, l; dirichlet=bcs, (update_coefficients!)=t -> Rₕ!(fₕ, x -> _sciml_src(x, t))
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
        @test ode_function(sd; jacobian=nothing).jac === nothing

        # A caller-supplied prototype is passed through untouched.
        proto = spdiagm(0 => ones(n))
        @test ode_function(sd; jac_prototype=proto).jac_prototype === proto

        # The two-form method builds the semidiscretisation on the way.
        @test ode_function(a, l; dirichlet=:boundary) isa SciMLBase.ODEFunction
    end

    @testset "ode_problem: time domain, and a copied, consistent u₀" begin
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 5 + t)
        sd_bc = semidiscretize(a, l; dirichlet=bcs)

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
        @test ode_problem(a, l, u₀, I; dirichlet=bcs).u0[1] ≈ 5.0
    end

    @testset "linear_problem is the assembled steady system" begin
        A, F = assemble(a, l; dirichlet=:boundary => x -> 0.0)
        prob = linear_problem(a, l; dirichlet=:boundary => x -> 0.0)
        @test prob isa LinearProblem
        @test prob.A == A
        @test prob.b == F

        # `symmetrize` is forwarded to `assemble`.
        As, Fs = assemble(a, l; dirichlet=:boundary => x -> 1.0, symmetrize=true)
        prob_s = linear_problem(a, l; dirichlet=:boundary => x -> 1.0, symmetrize=true)
        @test prob_s.A == As
        @test prob_s.b == Fs
    end

    # The heat equation with the manufactured solution `exp(-t) sin(πx)`, stepped to t = 1.
    # The space tolerance is loose and the time tolerance tight, so what is measured is the
    # semidiscretisation's second order rather than the stepper's.
    @testset "order of convergence through OrdinaryDiffEq" begin
        function solve_to(n, alg)
            _, Wₕ, I, _, _, sd = _sciml_setup(n)
            prob = ode_problem(sd, Rₕ(Wₕ, x -> _sciml_uex(x, 0.0)), I)
            sol = solve(prob, alg; reltol=1e-11, abstol=1e-13)
            @test SciMLBase.successful_retcode(sol)
            uₕ = Bramble.element(Wₕ)
            parent(uₕ) .= sol.u[end]
            return normₕ(Rₕ(Wₕ, x -> _sciml_uex(x, 1.0)) - uₕ), hₘₐₓ(Bramble.mesh(Wₕ))
        end

        # `Rodas5P` is a Rosenbrock method and needs `∂f/∂t`. The `update_coefficients!`
        # hook writes into a `Float64` element, so that derivative cannot be taken by
        # differentiating through `t` -- hence `AutoFiniteDiff`. The BDF methods need no
        # `∂f/∂t` at all. See the note on `ode_function`.
        for alg in (FBDF(), QNDF(), Rodas5P(; autodiff=AutoFiniteDiff()))
            errors = Float64[]
            spacings = Float64[]
            for n in (11, 21, 41)
                e, h = solve_to(n, alg)
                push!(errors, e)
                push!(spacings, h)
            end
            eoc = [
                log(errors[i] / errors[i + 1]) / log(spacings[i] / spacings[i + 1]) for
                i in 1:(length(errors) - 1)
            ]
            @test all(>(1.9), eoc)
            @test issorted(errors; rev=true)
        end
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

        sd_s = semidiscretize(a_s, l_s; dirichlet=bc)
        prob = ode_problem(sd_s, Bramble.element(Wₕ, 0.0), (0.0, 20.0))
        sol = solve(prob, FBDF(); reltol=1e-10, abstol=1e-12)

        A, F = assemble(a_s, l_s; dirichlet=bc)
        @test sol.u[end] ≈ A \ F rtol = 1e-6
    end
end
