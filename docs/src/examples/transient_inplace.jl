# # A transient loop that allocates nothing
#
# Two pieces of this are already covered elsewhere. The
# [heat equation](heat_equation.md) builds `M/Δt + K(t)` into a preallocated pattern with
# [`allocate_system_matrix`](@ref) and [`assemble_add!`](@ref), and the
# [solver tutorial](../tutorials/solvers.md) reuses one factorization across steps with
# [`refactor!`](@ref). This page puts them together and closes the last gap: the backsolve
# writes into the solution vector that already exists, so a Crank-Nicolson step allocates
# nothing at all.
#
# This page is generated from `docs/src/examples/transient_inplace.jl` by Literate.jl, and
# the same file runs under `test/examples/pages.jl`, where the `#src` assertions execute.
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
using SuiteSparse
using SparseArrays: nonzeros
using LinearAlgebra: mul!, ldiv!

uexact(x, t) = exp(-t) * sinpi(x[1])

Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωₕ = mesh(Ω, 201)
Wₕ = gridspace(Ωₕ)

m = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
k = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))

# ## The two operators, built once
#
# Both sides share a sparsity pattern, so one call to [`allocate_system_matrix`](@ref) on a
# form carrying both stencils gives a pattern wide enough for either, and
# [`assemble_add!`](@ref) accumulates the pieces into it with no temporary and no sparse
# addition. `dirichlet_bc!` goes last, after everything that writes to those rows.

Δt = 1 / 200

pattern = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
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
# the time factor is a `Ref` the form reads live. Nothing is rebuilt between steps, and no
# closure is created inside the loop, which is what a per-step allocation would otherwise come
# from.

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
# constrained entries, and backsolve into `uₕ`'s own storage.

function step!(uₕ, F, decay, B, fact, l, Ωₕ, bcs, t)
    decay[] = exp(-t)                            # the form reads this live
    assemble!(F, l)                              # refill, no allocation
    mul!(F, B, parent(uₕ), 1.0, 1.0)             # F += B uⁿ
    dirichlet_bc!(F, Ωₕ, bcs, :boundary)
    ldiv!(parent(uₕ), fact, F)                   # uⁿ⁺¹ into the same storage
    return nothing
end

# The measurement goes through a function of its own, with the buffers passed in as
# arguments. `@allocated` written at top level reports the caller's boxing of the globals it
# reads rather than the routine's own work, and that alone shows 32 bytes here.

function step_allocation(uₕ, F, decay, B, fact, l, Ωₕ, bcs, t)
    step!(uₕ, F, decay, B, fact, l, Ωₕ, bcs, t)             # warm up: compile first
    return @allocated step!(uₕ, F, decay, B, fact, l, Ωₕ, bcs, t + 0.5)
end

allocated = step_allocation(uₕ, F, decay, B, fact, l, Ωₕ, bcs, Δt / 2)

@test allocated < 1024                                                                     #src

allocated

# Nothing. The bound the test asserts is looser on purpose: the usual way to lose this is to
# build a closure inside the loop, `Rₕ!(fₕ, x -> source(x, t))`, which costs a few hundred
# bytes per step, and the check exists to catch a regression of that size rather than to pin
# an exact number.
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

# ## Order in time
#
# Crank-Nicolson is second order in `Δt`. The spatial mesh is held fixed and fine enough that
# its own error stays below the time error being measured:

function crank_nicolson_error(nsteps)
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

    return normₕ(u - Rₕ(Wₕ, x -> uexact(x, 1.0)))
end

e₁, e₂ = crank_nicolson_error(25), crank_nicolson_error(50)
order = log2(e₁ / e₂)

@test 1.8 < order < 3.0                                                                    #src

e₁, e₂, order

# Above two, because the spatial error is not negligible at the coarser step and cancels part
# of the time error rather than adding to it. Refining `Δt` further would bring the measured
# order back down onto two and then flatten, as the spatial error takes over.
#
# ## See also
#
# - [Heat equation](heat_equation.md), where [`semidiscretize`](@ref) hands the same problem
#   to an adaptive stepper instead of a fixed loop.
# - [Choosing a linear solver](../tutorials/solvers.md), for when factorize-and-reuse beats a
#   warm-started iterative solve.
