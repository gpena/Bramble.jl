# # A transient loop that allocates nothing
#
# Two pieces of this are already covered elsewhere. The [heat equation](heat_equation.md)
# builds `M/Δt + K(t)` once and the [solver tutorial](@ref tutorial_solvers) reuses one
# factorization across steps with [`refactor!`](@ref); [Solving at every time step](@ref
# tutorial_time_stepping) shows the pattern and is not repeated here. This page puts them
# together on a problem with a Robin condition whose data decay in time, through the ghost
# layer, and closes the last gap: the backsolve writes into the solution vector that already
# exists, so a Crank-Nicolson step allocates nothing at all.
#
# This script runs under `test/examples/pages.jl`, where the `#src` assertions execute; it is
# not part of the published manual.
#
# ## Problem
#
# ```math
# \partial_t u = \partial_{xx} u + f \text{ in } (0,1) \times (0, 1], \qquad
# u(0, t) = 0, \qquad u(x, 0) = \sin(\pi x), \qquad
# u + \partial_n u = h(t) \text{ at } x = 1,
# ```
#
# with ``\partial_n`` the outward normal derivative and the manufactured solution
# ``u_{\text{exact}}(x, t) = e^{-t}\sin(\pi x)``, which forces
# ``f(x, t) = (\pi^2 - 1) e^{-t} \sin(\pi x)`` and the Robin data
# ``h(t) = -\pi e^{-t}``. Crank-Nicolson in time:
#
# ```math
# \Bigl(\frac{M}{\Delta t} + \frac{K}{2}\Bigr) u^{n+1} =
# \Bigl(\frac{M}{\Delta t} - \frac{K}{2}\Bigr) u^{n} + F^{n+1/2}
# ```

using Bramble
using Bramble: sparse_factorize
using SuiteSparse
using LinearAlgebra: mul!, ldiv!

uexact(x, t) = exp(-t) * sinpi(x[1])

Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωₕ = mesh(Ω, 201)
Wₕ = gridspace(Ωₕ)

# The mesh is uniform, but the order measured at the end of the page does not rely on that:
# the reference run shares whatever mesh is used.
#
# ## The relation and its data in time
#
# The Robin side goes through a ghost point: `gridspace(Ωₕ; ghosts = (:right,))` adds one
# node outside it, and `constrained` takes a function of the trial function `u` returning one
# `side => relation => data` entry per side. The Dirichlet side is the relation `u => 0` and
# leaves the unknowns. The data may take the time as a second argument, `(x, t) -> ...`, which
# is how the solution is lifted at a given time below. The space the in-place step is built
# from has the data at ``t = 0``, ``-\pi``:

relations(h) = u -> (:left => u => 0.0, :right => u + ∂ₙ(u) => h)
ghosted = gridspace(Ωₕ; ghosts = (:right,))
Cₕ = constrained(ghosted, relations(x -> -π))
Cₜ = constrained(ghosted, relations((x, t) -> -π * exp(-t)))

ndofs(Cₕ), ndofs(Cₜ)

# ## The operators, built once
#
# `assemble!(A, F, a, l)` fills a matrix and a load already allocated, and on a space with
# data it puts the lift of those data on the load: with ``E`` the closure of the relations
# and ``d`` their affine part, ``F = E^\top (F_{\text{pad}} - A_{\text{pad}} d)``. The pair
# form refuses a space whose data depend on `t`, since it lifts with the stored ``d(0)``,
# and the data here are separable, ``d(t) = e^{-t} d(0)``: so the source and the lift scale
# with the same factor ``e^{-t}``, and a single call at unit time factor produces the load
# that every step scales by ``e^{-t}``. The relation also moves in time, which the weak
# form of the time derivative sees as ``-E^\top M_{\text{pad}} d'(t)``, and the mass form
# gives that lift in the same way. Nothing is assembled inside the loop.

sₕ = x -> (π^2 - 1) * sinpi(x[1])

a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
m = form(Cₕ, Cₕ, (u, v) -> innerₕ(u, v))
l = form(Cₕ, v -> innerₕ(sₕ, v))
l0 = form(Cₕ, v -> innerₕ(x -> 0.0, v))

K, g = assemble(a, l)
M, gM = assemble(m, l0)
assemble!(K, g, a, l)                  # the refill form, on the buffers just made
assemble!(M, gM, m, l0)
g .-= gM                               # the lift of d′ = -d, scaled by e⁻ᵗ like the rest

size(K), length(g)

# `A` does not change from step to step here, so it is factorized once and every step is a
# pair of triangular solves. When a coefficient does vary with `t`, refill `A` and call
# [`refactor!`](@ref)`(fact, A)`, which reuses the symbolic analysis and repeats only the
# numeric factorization.

Δt = 1 / 200
A = M / Δt + K / 2
B = M / Δt - K / 2
fact = sparse_factorize(A; sym = :unsymmetric)

u = copy(parent(Rₕ(Wₕ, x -> uexact(x, 0.0)))[2:end])
F = zeros(ndofs(Cₕ))

length(F), u[1]

# ## One step
#
# Three in-place calls: scale the load by the time factor, add `B uⁿ` onto it, and backsolve
# into `u`'s own storage. The solution never moves and nothing is allocated to hold it.

function step!(u, F, g, B, fact, t)
    F .= exp(-t) .* g                            # data and source at t
    mul!(F, B, u, 1.0, 1.0)                      # F += B uⁿ
    ldiv!(u, fact, F)                            # uⁿ⁺¹ into the same storage
    return nothing
end

# The measurement goes through a function of its own, with the buffers passed in as
# arguments. `@allocated` written at top level can report the caller's boxing of the globals
# it reads rather than the routine's own work, so the barrier is the safer habit.

function step_allocation(u, F, g, B, fact, t)
    step!(u, F, g, B, fact, t)                   # warm up: compile first
    return @allocated step!(u, F, g, B, fact, t + 0.5)
end

allocated = step_allocation(u, F, g, B, fact, Δt / 2)

@test allocated == 0                                                                       #src

allocated

# Nothing, and the test asserts exactly that zero. The step reads its time through a plain
# number, so the Robin data moving in time cost no allocation.
#
# ## Running it
#
# Two hundred steps to `t = 1`, reusing the same buffers:

u .= parent(Rₕ(Wₕ, x -> uexact(x, 0.0)))[2:end]

for n in 1:200
    step!(u, F, g, B, fact, (n - 0.5) * Δt)
end

function on_grid(u)
    uₕ = element(Wₕ)
    parent(uₕ)[1] = 0.0
    parent(uₕ)[2:end] .= u
    return uₕ
end

err = normₕ(on_grid(u) - Rₕ(Wₕ, x -> uexact(x, 1.0)))

# Bracketed away from zero: an exactly zero error would mean the scheme had reproduced the #src
# manufactured solution by construction rather than stepped to it.                         #src
@test 1.0e-8 < err < 1.0e-4                                                                #src

err

# ## The relation at the final time
#
# `Bramble.lift(Cₜ, u, t)` returns the solution on the padded grid with the ghost value the
# relation fixes at time `t`. The ghost is fixed by the relation itself, so the check that
# means something is independent of it: the three-point backward difference of the lifted
# solution at ``x = 1`` for the normal derivative must satisfy ``u + \partial_n u = h(1)``
# up to the truncation error, with the data at the final time, not at ``t = 0``:

function robin_residual(Cₜ, u, t)
    U = parent(Bramble.lift(Cₜ, u, t))
    G = Bramble.mesh(Cₜ)
    xs = collect(points(Ωₕ))
    n = length(xs)
    h₁, h₂ = xs[n] - xs[n - 1], xs[n - 1] - xs[n - 2]
    at(i) = U[Bramble.padded_index(G, CartesianIndex(i))]
    dₙ = (2h₁ + h₂) / (h₁ * (h₁ + h₂)) * at(n) - (h₁ + h₂) / (h₁ * h₂) * at(n - 1) +
         h₁ / (h₂ * (h₁ + h₂)) * at(n - 2)
    return abs(at(n) + dₙ + π * exp(-t))
end

residual = robin_residual(Cₜ, u, 1.0)

# A truncation error, not round-off, and far below the size of the data, ``\pi e^{-1}``,
# which the same check at the initial data misses by order one:                              #src
@test 1.0e-9 < residual < 1.0e-3                                                           #src
@test robin_residual(Cₜ, u, 0.0) > 1.0                                                     #src

residual

#-

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
Ωst = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (81, 41), (true, true)) # hide
ust = Rₕ(gridspace(Ωst), x -> exp(-x[2]) * sinpi(x[1])) # hide
surface_plot(ust; title = "u(x, t) = e⁻ᵗ sin(πx), the solution being stepped") # hide

# ## Crank-Nicolson order in time
#
# Crank-Nicolson is second order in `Δt`. Measured against the exact solution the error would
# stop shrinking at the spatial error of this mesh, so it is measured against the same scheme
# on the same mesh with a much smaller step. The spatial error is then the same in both and
# drops out of the difference, which is left with the time error alone:

function crank_nicolson(nsteps)
    Δt = 1 / nsteps
    A = M / Δt + K / 2
    B = M / Δt - K / 2
    fact = sparse_factorize(A; sym = :unsymmetric)
    u = copy(parent(Rₕ(Wₕ, x -> uexact(x, 0.0)))[2:end])
    F = zeros(ndofs(Cₕ))

    for n in 1:nsteps
        step!(u, F, g, B, fact, (n - 0.5) * Δt)
    end

    return on_grid(u)
end

uref = crank_nicolson(6400)
nsteps = [25, 50, 100, 200]
Δts = 1 ./ nsteps
errs = [normₕ(crank_nicolson(n) - uref) for n in nsteps]
orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]

# Every observed order must sit within a few hundredths of two, which a first-order        #src
# stepper, at order one, would miss by far.                                                #src
@test all(o -> 1.95 < o < 2.05, orders)                                                    #src

orders

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(Δts, errs, "Crank-Nicolson", "#5B5FC7")]; # hide
    title = "Crank-Nicolson, ‖·‖ₕ at t = 1 against the Δt = 1/6400 run") # hide

# The orders are two to within a few parts in a thousand, and the errors sit on the
# dashed line of slope two. Against the exact solution the same four errors would flatten
# out and then grow back toward the spatial error of the mesh, about `5e-6`, which says
# nothing about the time stepping.
#
# ## Where to go next
#
# - [Heat equation](heat_equation.md), where [`semidiscretize`](@ref) hands the same problem
#   to an adaptive stepper instead of a fixed loop.
# - [Choosing a linear solver](@ref tutorial_solvers), for the direct solvers behind
#   factorize-and-reuse.
