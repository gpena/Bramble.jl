# # A transient loop that allocates nothing
#
# Two pieces of this are already covered elsewhere. The
# [heat equation](heat_equation.md) builds `M/Δt + K(t)` into a preallocated pattern with
# [`allocate_system_matrix`](@ref) and [`assemble_add!`](@ref), and the
# [solver tutorial](@ref tutorial_solvers) reuses one factorization across steps with
# [`refactor!`](@ref); [Solving at every time step](@ref tutorial_time_stepping) shows the
# pattern and is not repeated here. This page puts them together and closes the last gap: the backsolve
# writes into the solution vector that already exists, so a Crank-Nicolson step allocates
# nothing at all.
#
# This script runs under `test/examples/pages.jl`, where the `#src` assertions execute; it is
# not part of the published manual.
#
# ## Problem
#
# ```math
# \partial_t u = \partial_{xx} u + f \text{ in } (0,1) \times (0, 1], \qquad
# u(0, t) = u(1, t) = 0, \qquad u(x, 0) = \sin(\pi x)
# ```
#
# with the manufactured solution ``u_{\text{exact}}(x, t) = e^{-t}\sin(\pi x)``, which forces
# ``f(x, t) = (\pi^2 - 1) e^{-t} \sin(\pi x)``. Crank-Nicolson in time:
#
# ```math
# \Bigl(\frac{M}{\Delta t} + \frac{K}{2}\Bigr) u^{n+1} =
# \Bigl(\frac{M}{\Delta t} - \frac{K}{2}\Bigr) u^{n} + F^{n+1/2}
# ```

using Bramble
using Bramble: allocate_system_matrix, sparse_factorize
using SuiteSparse
using SparseArrays: nonzeros
using LinearAlgebra: mul!, ldiv!

uexact(x, t) = exp(-t) * sinpi(x[1])

Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωₕ = mesh(Ω, 201)
Wₕ = gridspace(Ωₕ)

# The mesh is uniform, but the order measured at the end of the page does not rely on that:
# the reference run shares whatever mesh is used.

m = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
k = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))

# ## The two operators, built once
#
# Both sides share a sparsity pattern, so one call to [`allocate_system_matrix`](@ref) on a
# form carrying both stencils gives a pattern wide enough for either, and
# [`assemble_add!`](@ref) accumulates the pieces into it with no temporary and no sparse
# addition. `dirichlet_bc!` goes last, after everything that writes to those rows.

Δt = 1 / 200

pattern = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
A = allocate_system_matrix(pattern)
B = similar(A)

fill!(nonzeros(A), 0.0)
assemble_add!(A, m, 1 / Δt)
assemble_add!(A, k, 0.5)
dirichlet_bc!(A, Ωₕ, :boundary)

fill!(nonzeros(B), 0.0)
assemble_add!(B, m, 1 / Δt)
assemble_add!(B, k, -0.5)

size(A), length(nonzeros(A))

# `A` does not change from step to step here, so it is factorized once and every step is a
# pair of triangular solves. When a coefficient does vary with `t`, refill `A` the same way
# and call [`refactor!`](@ref)`(fact, A)`, which reuses the symbolic analysis and repeats only
# the numeric factorization.

fact = sparse_factorize(A; sym = :unsymmetric)

# ## The source, without a closure per step
#
# The source separates as ``f(x,t) = e^{-t} s(x)``, so the spatial part is restricted once and
# the time factor is a `Ref` the form reads live. Nothing is rebuilt between steps; the `Ref`
# keeps the time factor out of the form's type, so the form is built once.

sₕ = Rₕ(Wₕ, x -> (π^2 - 1) * sinpi(x[1]))
decay = Ref(1.0)

l = form(Wₕ, v -> decay * innerₕ(sₕ, v))
bcs = dirichlet_constraints(Ωₕ, :boundary => x -> 0.0)

uₕ = Rₕ(Wₕ, x -> uexact(x, 0.0))
F = zeros(ndofs(Wₕ))

length(F), uₕ[1]

# ## One step
#
# Five in-place calls: set the time factor, refill the load, add `B uⁿ` onto it, overwrite the
# constrained entries, and backsolve into `uₕ`'s own storage. The last one takes the
# [`VectorElement`](@ref) itself as its destination and writes through to the coefficients
# behind it, so the solution never moves and nothing is allocated to hold it.

function step!(uₕ, F, decay, B, fact, l, Ωₕ, bcs, t)
    decay[] = exp(-t)                            # the form reads this live
    assemble!(F, l)                              # refill, no allocation
    mul!(F, B, parent(uₕ), 1.0, 1.0)             # F += B uⁿ
    dirichlet_bc!(F, Ωₕ, bcs, :boundary)
    ldiv!(uₕ, fact, F)                           # uⁿ⁺¹ into the same storage
    return nothing
end

# The measurement goes through a function of its own, with the buffers passed in as
# arguments. `@allocated` written at top level can report the caller's boxing of the globals
# it reads rather than the routine's own work, so the barrier is the safer habit.

function step_allocation(uₕ, F, decay, B, fact, l, Ωₕ, bcs, t)
    step!(uₕ, F, decay, B, fact, l, Ωₕ, bcs, t)             # warm up: compile first
    return @allocated step!(uₕ, F, decay, B, fact, l, Ωₕ, bcs, t + 0.5)
end

allocated = step_allocation(uₕ, F, decay, B, fact, l, Ωₕ, bcs, Δt / 2)

@test allocated == 0                                                                       #src

allocated

# Nothing, and the test asserts exactly that zero. A closure built inside the loop,
# `Rₕ!(fₕ, x -> source(x, t))`, costs nothing here, because `Rₕ!` is compiled for it. A call
# the compiler cannot specialise does cost: the same closure reaching `Rₕ!` through a value of
# uninferred type allocates 96 bytes per step, which a bound of a few hundred bytes would let
# through.
#
# ## Running it
#
# Two hundred steps to `t = 1`, reusing the same five buffers:

parent(uₕ) .= parent(Rₕ(Wₕ, x -> uexact(x, 0.0)))

for n in 1:200
    step!(uₕ, F, decay, B, fact, l, Ωₕ, bcs, (n - 0.5) * Δt)
end

err = normₕ(uₕ - Rₕ(Wₕ, x -> uexact(x, 1.0)))

# Bracketed away from zero: an exactly zero error would mean the scheme had reproduced the #src
# manufactured solution by construction rather than stepped to it.                         #src
@test 1.0e-8 < err < 1.0e-4                                                                #src

err

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
    A = allocate_system_matrix(pattern)
    B = similar(A)

    fill!(nonzeros(A), 0.0)
    assemble_add!(A, m, 1 / Δt)
    assemble_add!(A, k, 0.5)
    dirichlet_bc!(A, Ωₕ, :boundary)

    fill!(nonzeros(B), 0.0)
    assemble_add!(B, m, 1 / Δt)
    assemble_add!(B, k, -0.5)

    fact = sparse_factorize(A; sym = :unsymmetric)
    u = Rₕ(Wₕ, x -> uexact(x, 0.0))
    F = zeros(ndofs(Wₕ))

    for n in 1:nsteps
        step!(u, F, decay, B, fact, l, Ωₕ, bcs, (n - 0.5) * Δt)
    end

    return u
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
# out and then grow back toward the spatial error of the mesh, about `6e-6`, which says
# nothing about the time stepping.
#
# ## Where to go next
#
# - [Heat equation](heat_equation.md), where [`semidiscretize`](@ref) hands the same problem
#   to an adaptive stepper instead of a fixed loop.
# - [Choosing a linear solver](@ref tutorial_solvers), for the direct solvers behind
#   factorize-and-reuse.
