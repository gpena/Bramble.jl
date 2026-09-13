# # Heat equation
#
# The first time-dependent problem in this manual. Space is discretised exactly as in the
# [linear Poisson example](poisson_linear.md); time is left continuous and handed to a stepper
# from the SciML stack. Every number below was produced by the code shown.
#
# This page is generated from `docs/src/examples/heat_equation.jl` by Literate.jl, and the
# same file runs under `test/examples/heat_equation.jl` with the assertions that the page
# renders but does not check. Lines marked `#src` there are the assertions; they never reach
# the page.
#
# ## Problem
#
# ```math
# \partial_t u = \Delta u + f \text{ in } \Omega \times (0, T], \qquad u = g \text{ on }
# \partial\Omega, \qquad u(\cdot, 0) = u_0,
# ```
#
# on ``\Omega = (0,1)`` with the manufactured solution ``u_{\text{exact}}(x, t) = e^{-t}
# \sin(\pi x)``, which vanishes on the boundary and forces
#
# ```math
# f(x, t) = (\pi^2 - 1)\, e^{-t} \sin(\pi x).
# ```
#
# ## The method of lines
#
# Discretising space alone leaves one ordinary differential equation per degree of freedom,
#
# ```math
# M \frac{\mathrm{d} u_h}{\mathrm{d} t} = F(t) - A u_h,
# ```
#
# where ``A`` is the same discrete Laplacian the steady problem assembles, ``F(t)`` is the
# source at time ``t``, and ``M`` is the mass matrix of the discrete inner product
# ``\langle \cdot, \cdot \rangle_h`` — diagonal, since that inner product is a weighted sum
# over grid points.
#
# [`semidiscretize`](@ref) builds exactly this. The spatial form is written the way the steady
# problem writes it, so the steady state of the system below solves ``A u_h = F``:

using Bramble

uexact(x, t) = exp(-t) * sinpi(x[1])
source(x, t) = (pi^2 - 1) * exp(-t) * sinpi(x[1])

Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωₕ = mesh(Ω, 101)
Wₕ = gridspace(Ωₕ)
I = interval(0.0, 1.0)          # the time domain

fₕ = element(Wₕ, 0.0)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
l = form(Wₕ, v -> innerₕ(fₕ, v))

bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)

sd = semidiscretize(a, l;
    dirichlet = bcs,
    update_coefficients! = t -> Rₕ!(fₕ, x -> source(x, t)))

# Two pieces are worth naming. The time domain passed to [`dirichlet_constraints`](@ref) is
# what lets the boundary values be written as `g(x, t)` rather than `g(x)`; `semidiscretize`
# detects that by the same arity test and re-evaluates them at every step.
# `update_coefficients!` is called with the current time just before each assembly, and is
# where a time-dependent source belongs — `Rₕ!` writes into the `fₕ` the form already holds a
# reference to, so nothing is rebuilt and nothing is allocated.
#
# ## Dirichlet conditions as algebraic constraints
#
# A constrained row of `A` is ``e_k`` and the matching entry of `F(t)` is ``g(x_i, t)`` —
# which is what `assemble` produces for the steady problem too. [`mass_matrix`](@ref) zeroes
# those same rows, so each one reads
#
# ```math
# 0 = g(x_i, t) - u_h[i],
# ```
#
# the boundary condition itself. The system is therefore a differential-algebraic one, and
# needs only ``g`` — never ``\partial_t g``, which prescribing ``u_h'`` on the boundary would
# have required:

M = mass_matrix(sd)
(M[1, 1], M[51, 51], M[101, 101])   # boundary, interior, boundary

# The two constrained rows are zeroed, and the interior one carries the grid weight the  #src
# discrete inner product gives that point, which on this uniform mesh is h.               #src
@test M[1, 1] == 0.0 && M[101, 101] == 0.0                      #src
@test M[51, 51] ≈ hₘₐₓ(Ωₕ)                                      #src

# ## Solving it
#
# [`ode_problem`](@ref) wraps the semidiscretisation, its mass matrix, and its exact Jacobian
# ``-A`` into a problem `OrdinaryDiffEq` can step. Because the mass matrix is singular, the
# method has to be one that admits that — `FBDF` and `QNDF` here, or a Rosenbrock method such
# as `Rodas5P`:

using OrdinaryDiffEqBDF

prob = ode_problem(sd, Rₕ(Wₕ, x -> uexact(x, 0.0)), I)
sol = solve(prob, FBDF(); reltol = 1e-10, abstol = 1e-12)

uₕ = element(Wₕ)
parent(uₕ) .= sol.u[end]
normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ)

# Bracketed on both sides: a solution that collapsed to zero, or to the initial condition, #src
# would sit far outside this window, and so would a first-order one on 101 points.          #src
@test 1.0e-5 < normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ) < 5.0e-5  #src

# A surface plot of the whole time evolution, `x` and `t` the two horizontal axes, `u` as
# height and colour, needs no new solve: `sol` already interpolates continuously in `t`, so
# sampling it at a uniform grid of times is enough to lay one out. The white curve traces
# `u(x, t)` at the current instant and loops on its own:

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide

ts = range(0.0, 1.0; length = 60)
Z = reduce(vcat, (sol(t)' for t in ts))
spacetime_surface_plot(points(Ωₕ), collect(ts), Z; title = "Heat equation, x-t-u")

# The initial condition handed in is copied, never mutated, and the copy is made consistent
# with the algebraic rows at ``t_0`` before stepping starts — an index-1 system whose initial
# condition disagrees with its own constraints is otherwise rejected by the solver or absorbed
# into the first step.
#
# !!! note "Rosenbrock methods need `∂f/∂t`"
#     `Rodas5P` and friends also want the time derivative of the right-hand side, which they
#     build by differentiating through `t`. An `update_coefficients!` hook writing into a
#     `Float64` grid function — the one above does — cannot accept a `ForwardDiff.Dual` time,
#     so pass `Rodas5P(autodiff = AutoFiniteDiff())` from `ADTypes`, use a BDF method (which
#     needs no `∂f/∂t` at all), or supply an analytical `tgrad` — see below.
#
# ## An analytical `tgrad` for Rosenbrock methods
#
# `ode_function`/`ode_problem` take a `tgrad` keyword: an exact `∂f/∂t`, handed straight to
# `ODEFunction` so a Rosenbrock method never has to differentiate through `t` at all. For this
# problem `f(t) = F(t) - A u_h` and `A` does not depend on `t`, so `∂f/∂t = ∂F/∂t` — assembled
# the same way `F` itself is, from the time derivative of the source:

using OrdinaryDiffEqRosenbrock

∂ₜsource(x, t) = -(pi^2 - 1) * exp(-t) * sinpi(x[1])   # ∂ₜ of `source` above

tgrad_heat(dT, sd, u, p, t) = begin
    gₕ = Rₕ(space(sd), x -> ∂ₜsource(x, t))
    l_t = form(space(sd), v -> innerₕ(gₕ, v))
    assemble!(dT, l_t)
end

prob_rosenbrock = ode_problem(sd, Rₕ(Wₕ, x -> uexact(x, 0.0)), I; tgrad = tgrad_heat)
sol_rosenbrock = solve(prob_rosenbrock, Rodas5P(); reltol = 1e-11, abstol = 1e-13)

uₕ_r = element(Wₕ)
parent(uₕ_r) .= sol_rosenbrock.u[end]
normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ_r)

# No `AutoFiniteDiff()` and no BDF fallback: the default forward-mode `autodiff` differentiates
# the Jacobian through `u` only, and `tgrad` supplies the `t`-derivative directly.
@test 1.0e-5 < normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ_r) < 5.0e-5               #src
#
# ## A time-dependent operator
#
# `tgrad` fixes the source half of `∂f/∂t`; the other method of [`semidiscretize`](@ref)
# fixes the operator half, for a spatial operator whose own coefficients vary with `t`. A
# fixed `BilinearForm` closes over a `Float64` coefficient buffer, which cannot hold the
# `Dual` a Rosenbrock stepper reaches for either way. `semidiscretize(build, l; ...)` instead
# takes a `build(t) -> (a, refill!)` factory, called once per element type `t` is ever seen
# at — `Float64` on an ordinary step, `Dual` while `Rodas5P`'s default `autodiff`
# differentiates through `t` — so the coefficient buffer it refills is always the right type.
# The source is held fixed here on purpose, to isolate the operator: a `t`-dependent source
# reached through `update_coefficients!` has the same `Float64`-buffer limitation `tgrad`
# fixes above, for the same reason:

α(t) = 1.0 + 0.5 * sinpi(2t)     # a diffusivity that genuinely varies with t
gₕ = Rₕ(Wₕ, x -> sinpi(x[1]))
l_t = form(Wₕ, v -> innerₕ(gₕ, v))

function build_diffusion(t)
    αₕ = element(Wₕ, typeof(t))
    a_t = form(Wₕ, Wₕ, (u, v) -> inner₊(αₕ * ∇₋ₕ(u), ∇₋ₕ(v)))
    refill!(t) = (fill!(parent(αₕ), α(t)); nothing)
    return a_t, refill!
end

sd_t = semidiscretize(build_diffusion, l_t; dirichlet = bcs)
prob_t = ode_problem(sd_t, Rₕ(Wₕ, x -> uexact(x, 0.0)), I)

sol_t_rosenbrock = solve(prob_t, Rodas5P(); reltol = 1e-11, abstol = 1e-13)
sol_t_fbdf = solve(prob_t, FBDF(); reltol = 1e-11, abstol = 1e-13)
maximum(abs.(sol_t_rosenbrock.u[end] .- sol_t_fbdf.u[end]))

# The same `ODEProblem`, one stepper differentiating through `t` by default and the other
# needing no `∂f/∂t` at all, land on the same answer -- neither `AutoFiniteDiff()` nor a
# hand-written `tgrad` was needed for the operator itself.
@test maximum(abs.(sol_t_rosenbrock.u[end] .- sol_t_fbdf.u[end])) < 1.0e-6              #src
#
# ## Checking the answer
#
# Second order in space is the promise. Refining the mesh while holding the time tolerance far
# below the spatial error isolates it:

function heat_series(ns)
    hs, errs = Float64[], Float64[]
    for n in ns
        Ωc = mesh(Ω, n)
        Wc = gridspace(Ωc)
        fc = element(Wc, 0.0)
        ac = form(Wc, Wc, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        lc = form(Wc, v -> innerₕ(fc, v))
        bc = dirichlet_constraints(Ωc, I, :boundary => (x, t) -> 0.0)

        sdc = semidiscretize(ac, lc;
            dirichlet = bc,
            update_coefficients! = t -> Rₕ!(fc, x -> source(x, t)))

        solc = solve(ode_problem(sdc, Rₕ(Wc, x -> uexact(x, 0.0)), I),
            FBDF(); reltol = 1e-11, abstol = 1e-13)

        uc = element(Wc)
        parent(uc) .= solc.u[end]
        push!(hs, hₘₐₓ(Ωc))
        push!(errs, normₕ(Rₕ(Wc, x -> uexact(x, 1.0)) - uc))
    end
    return hs, errs
end

hs, errs = heat_series((11, 21, 41, 81, 161))
orders = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(errs) - 1)]

#-

all(>(1.95), orders)    # second order is the promise

# The line above renders `true` on the page and asserts nothing there; this is where the   #src
# rate is pinned, so a decay to first order fails the suite instead of publishing `false`.  #src
@test all(>(1.95), orders)                                       #src

# ## Time-dependent boundary data
#
# Nothing above exercised the `(x, t)` in the boundary condition, since the manufactured
# solution vanishes on ``\partial\Omega``. Driving the problem entirely from the boundary
# does: start at zero and raise the left end linearly, with the right end held fixed. The two
# ends are named on the domain above — a bare `domain(interval(...))` registers only
# `:boundary` and `:interior`, so there would be no `:left` to constrain.

gₕ = element(Wₕ, 0.0)
l_drive = form(Wₕ, v -> innerₕ(gₕ, v))
bcs_drive = dirichlet_constraints(Ωₕ, I,
    :left => (x, t) -> t,
    :right => (x, t) -> 0.0)

sd_drive = semidiscretize(a, l_drive; dirichlet = bcs_drive)
sol_drive = solve(ode_problem(sd_drive, element(Wₕ, 0.0), I), FBDF();
    reltol = 1e-10, abstol = 1e-12)

u_end = sol_drive.u[end]
(u_end[1], u_end[51], u_end[101])

# The ends hold g exactly; the midpoint is short of the steady 0.5 for the reason the prose #src
# below gives, and the window is tight enough that reaching the steady line would fail.     #src
@test u_end[1] == 1.0 && u_end[101] == 0.0                       #src
@test 0.43 < u_end[51] < 0.45                                    #src

# At ``t = 1`` the ends hold exactly the prescribed ``g``: one and zero. The interior is
# climbing towards the straight line ``1 - x`` that the source-free steady problem gives, but
# has not arrived — ``0.44`` at the midpoint against the steady ``0.5``. It should not have:
# the left end was still moving over the whole interval, so diffusion is chasing a boundary
# value that never settled. Holding ``g`` fixed and stepping further is what reaches the line,
# and the [steady solve](#Steady-problems-and-LinearSolve) below is its limit.
#
# ## Steady problems and LinearSolve
#
# The same forms describe the steady problem, and [`linear_problem`](@ref) hands it straight to
# `LinearSolve` — with its factorisations, Krylov methods and preconditioners — instead of
# assembling by hand first:
#
# ```julia
# using LinearSolve, IncompleteLU
#
# prob = linear_problem(a, l; dirichlet = :boundary => x -> 0.0)
# sol = solve(prob, KrylovJL_GMRES())
# ```
#
# `linear_problem` takes the `dirichlet`, `dirichlet_components` and `symmetrize` keywords
# [`assemble`](@ref) takes, and returns exactly `LinearProblem(A, F)` for the `A` and `F` that
# [`assemble`](@ref)`(a, l; ...)` produces.
#
# ## See also
#
#   - [`semidiscretize`](@ref), [`ode_problem`](@ref), [`ode_function`](@ref) in the
#     [API reference](../api.md).
#   - [Linear Poisson](poisson_linear.md) for the steady version of the same spatial operator.
#   - [Coupled reaction-diffusion](coupled_reaction_diffusion.md) for systems on a composite
#     space, which `semidiscretize` accepts unchanged.
