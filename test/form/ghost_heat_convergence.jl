# ghost_heat_convergence.jl: the heat equation `uₜ = Δu + f` on [0, 1] × [0, 2] with
# time-dependent Neumann and Robin data converges in space and in time (gpena/Bramble.jl#776).
# The operator is `innerₕ(-Δₕ(u), v)`, which reads the ghosts (`inner₊(∇ₕ(u), ∇ₕ(v))` does
# not). The data `(x, t) -> ...` and the source come from an exact solution written out by
# hand, so the reference shares no code with the library; the source goes through a function
# of a `Ref` time that `update_coefficients!` sets. Everything runs through `ode_problem`.
#
# Space: three meshes per kind, seeded random and graded, `FBDF` at tight tolerances to a
# final time, the discrete L² error at the physical nodes against `hₘₐₓ`. Time:
# `Rosenbrock23` with `adaptive = false` at halving steps on one graded mesh, against the
# `FBDF` solution of the same semidiscretization, so that the spatial error cancels. That
# run takes the Dual-`t` path of the residual through the solver's own `tgrad`.
module GhostHeatConvergenceTests

using Test
using Bramble
import Bramble
using LinearAlgebra: norm
using ForwardDiff: ForwardDiff
using SciMLBase: solve, successful_retcode
using OrdinaryDiffEqBDF: FBDF
using OrdinaryDiffEqRosenbrock: Rosenbrock23
using ..TestUtils: WITH_SLOW_TESTS

const MIN_ORDER = 1.8
const T_FINAL = 0.5

#------------------------------------------------------------------------------------------#
# The manufactured solution u = a(t) φ₁(x) + b(t) φ₂(x): φ₁ is an eigenfunction of Δ
# (eigenvalue -2) and φ₂ is not, so the source mixes both
#------------------------------------------------------------------------------------------#

a(t) = 1 + sin(2t) / 2
a′(t) = cos(2t)
b(t) = exp(-t)
b′(t) = -exp(-t)
φ₁(x) = exp(x[1] / 2) * cos(3x[2] / 2 + 3 / 10)
φ₂(x) = x[1]^2 * x[2]

uexact(x, t) = a(t) * φ₁(x) + b(t) * φ₂(x)
# `uₜ − Δu`, with Δφ₁ = -2φ₁ and Δφ₂ = 2y.
source(x, t) = (a′(t) + 2a(t)) * φ₁(x) + b′(t) * φ₂(x) - 2b(t) * x[2]
# The partial derivatives, written out.
∂ₓu(x, t) = a(t) * φ₁(x) / 2 + 2b(t) * x[1] * x[2]
∂ᵧu(x, t) = -3a(t) / 2 * exp(x[1] / 2) * sin(3x[2] / 2 + 3 / 10) + b(t) * x[1]^2

# The datum frozen at `t = 0` when `frozen`: the control that shows the data matter.
data(g, frozen) = frozen ? x -> g(x, 0.0) : g

# Neumann on `xmin` and `ymax`, Robin on `xmax` and `ymin`, each as `αu + ∂ₙu = g` with the
# outward normal.
relations(frozen) = u -> (
    :xmin => ∂ₙ(u) => data((x, t) -> -∂ₓu(x, t), frozen),
    :xmax => u + ∂ₙ(u) => data((x, t) -> uexact(x, t) + ∂ₓu(x, t), frozen),
    :ymin => 2.0 * u + ∂ₙ(u) => data((x, t) -> 2uexact(x, t) - ∂ᵧu(x, t), frozen),
    :ymax => ∂ₙ(u) => data((x, t) -> ∂ᵧu(x, t), frozen)
)

#------------------------------------------------------------------------------------------#
# Meshes: seeded random (a base mesh refined by halving) and graded
#------------------------------------------------------------------------------------------#

box() = domain(interval(0.0, 1.0) × interval(0.0, 2.0))

function random_mesh(level; seed = 7, n0 = 9)
    Bramble._seed_mesh1d_rng!(seed)
    Ω = try
        mesh(box(), (n0, n0), (false, false))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
    foreach(_ -> iterative_refinement!(Ω), 1:level)
    return Ω
end

# A smooth grading of each axis, different on the two so no symmetry survives.
function graded_mesh(level; n0 = 9)
    n = (n0 - 1) * 2^level + 1
    Ω = mesh(box(), (n, n), (true, true))
    s = range(0.0, 1.0; length = n)
    Bramble.change_points!(Ω, (collect(s .^ 1.3), collect(2 .* (1 .- (1 .- s) .^ 1.2))))
    return Ω
end

meshes(kind, levels) = [kind === :graded ? graded_mesh(ℓ) : random_mesh(ℓ) for ℓ in levels]

#------------------------------------------------------------------------------------------#
# The ODE problem through `ode_problem`
#------------------------------------------------------------------------------------------#

# `ode_problem` on the constrained space of `Ωₕ`. The source is sampled into a grid function
# on the padded space by the hook, from a `Ref` time. The element type of that function is
# `Float64`, so the hook sets the `Ref` from the value of a `ForwardDiff.Dual` time.
function heat_problem(Ωₕ; frozen = false, tgrad = nothing)
    C = constrained(gridspace(Ωₕ; ghosts = :boundary), relations(frozen))
    fₕ = element(Bramble.padded_space(C))
    tref = Ref(0.0)
    f(x) = source(x, tref[])
    update = t -> (tref[] = ForwardDiff.value(t); Rₕ!(fₕ, f); nothing)
    Wₕ = gridspace(Ωₕ)
    u₀ = collect(parent(Rₕ(Wₕ, x -> uexact(x, 0.0))))
    prob = ode_problem(form(C, C, (u, v) -> innerₕ(-Δₕ(u), v)),
        form(C, v -> innerₕ(fₕ, v)), u₀, (0.0, T_FINAL); (update_coefficients!) = update, tgrad)
    return prob, Wₕ
end

# The reference solution of the semidiscretization, tight but not at the roundoff floor.
reference(prob; reltol = 1e-10, abstol = 1e-12) = solve(prob, FBDF(); reltol, abstol).u[end]

# The discrete L² error of the nodal values `uh` against the exact solution at `T_FINAL`.
function nodal_error(Wₕ, uh)
    uₕ = element(Wₕ)
    parent(uₕ) .= uh
    return normₕ(Rₕ(Wₕ, x -> uexact(x, T_FINAL)) - uₕ)
end

function space_errors(kind, levels; kwargs...)
    Ωs = meshes(kind, levels)
    errs = map(Ωs) do Ωₕ
        prob, Wₕ = heat_problem(Ωₕ; kwargs...)
        nodal_error(Wₕ, reference(prob; reltol = 1e-9, abstol = 1e-11))
    end
    return map(Bramble.hₘₐₓ, Ωs), errs
end

# The least-squares slope of `log err` against `log h`.
function slope(hs, errs)
    X, Y = log.(hs), log.(errs)
    X̄, Ȳ = sum(X) / length(X), sum(Y) / length(Y)
    return sum((X .- X̄) .* (Y .- Ȳ)) / sum(abs2, X .- X̄)
end

function report(label, hs, errs)
    s = slope(hs, errs)
    println("ghost heat ", label, ": slope ", round(s; digits = 3), " (errors ",
        join(round.(errs; sigdigits = 3), ", "), ")")
    return s
end

# The `Rosenbrock23` end values at each fixed step, against `ref`, in the max norm.
function time_errors(prob, ref, dts)
    return map(dts) do dt
        sol = solve(prob, Rosenbrock23(); adaptive = false, dt)
        @test successful_retcode(sol)
        norm(sol.u[end] - ref, Inf)
    end
end

@testset "Ghost heat equation converges (#776)" begin
    @testset "Space order, random meshes" begin
        hs, errs = space_errors(:random, 0:2)
        @test issorted(hs; rev = true)
        @test issorted(errs; rev = true)
        @test report("space random", hs, errs) >= MIN_ORDER
    end

    @testset "Space order, graded meshes" begin
        hs, errs = space_errors(:graded, 0:2)
        @test issorted(hs; rev = true)
        @test issorted(errs; rev = true)
        @test report("space graded", hs, errs) >= MIN_ORDER
    end

    # Data frozen at `t = 0` solve another problem: the error is no longer the scheme's.
    @testset "Time-dependent data matter" begin
        _, errs = space_errors(:graded, 2:2)
        _, frozen = space_errors(:graded, 2:2; frozen = true)
        @test only(frozen) > 10 * only(errs)
    end

    @testset "Time order, Rosenbrock23" begin
        prob, _ = heat_problem(graded_mesh(1))
        ref = reference(prob)
        # The reference is converged well below the smallest time error.
        coarse = reference(prob; reltol = 1e-8, abstol = 1e-10)
        dts = [0.1, 0.05, 0.025, 0.0125]
        errs = time_errors(prob, ref, dts)
        @test norm(ref - coarse, Inf) < errs[end] / 100
        @test issorted(errs; rev = true)
        @test report("time Rosenbrock23", dts, errs) >= MIN_ORDER
    end

    # The ghosts' data carry the time derivative `tgrad` needs: a zero `∂f/∂t` loses the
    # order and the accuracy at the same steps.
    @testset "Time order needs the data's ∂ₜ" begin
        prob, _ = heat_problem(graded_mesh(1))
        ref = reference(prob)
        dts = [0.05, 0.025]
        errs = time_errors(prob, ref, dts)
        zero, _ = heat_problem(graded_mesh(1); tgrad = (dT, u, p, t) -> fill!(dT, 0.0))
        @test minimum(time_errors(zero, ref, dts) ./ errs) > 5
    end

    WITH_SLOW_TESTS && @testset "Space order, four levels" begin
        for kind in (:random, :graded)
            hs, errs = space_errors(kind, 0:3)
            @test report("space $kind (4 levels)", hs, errs) >= MIN_ORDER
        end
    end
end

end
