# # Heat equation: time stepping details
#
# Three ways to go beyond the plain `semidiscretize` and `ode_problem` call of the
# [heat equation example](heat_equation.md): an analytical `tgrad` for Rosenbrock methods, a
# spatial operator whose coefficients depend on `t`, and a backward Euler loop written by
# hand with [`assemble_add!`](@ref). All three use the problem of that page with a Dirichlet
# condition at both ends: ``\partial_t u = \Delta u + f`` on ``(0, 1)`` with
# ``u_{\text{exact}}(x, t) = e^{-t} \sin(\pi x)``, which vanishes at both ends, and
#
# ```math
# f(x, t) = (\pi^2 - 1)\, e^{-t} \sin(\pi x).
# ```
#
# The setup is the one of that page: a plain space, the constrained rows of
# [`dirichlet_constraints`](@ref) taking `g(x, t)`, and a source refreshed at every step
# through `update_coefficients!`:

using Bramble
using Bramble: allocate_system_matrix
using OrdinaryDiffEqBDF
using OrdinaryDiffEqRosenbrock

uexact(x, t) = exp(-t) * sinpi(x[1])
source(x, t) = (pi^2 - 1) * exp(-t) * sinpi(x[1])

Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 101)
Wₕ = gridspace(Ωₕ)
I = interval(0.0, 1.0)          # the time domain

fₕ = element(Wₕ, 0.0)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wₕ, v -> innerₕ(fₕ, v))

bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)

sd = semidiscretize(a, l;
    dirichlet = bcs,
    update_coefficients! = t -> Rₕ!(fₕ, x -> source(x, t)))

# ## An analytical time derivative for Rosenbrock methods
#
# `Rodas5P` and friends want the time derivative of the right-hand side, which they build by
# differentiating through `t`. An `update_coefficients!` hook writing into a `Float64` grid
# function, as the one above does, cannot accept a `ForwardDiff.Dual` time. Passing
# `Rodas5P(autodiff = AutoFiniteDiff())` from `ADTypes` or using a BDF method (which needs no
# `∂f/∂t` at all) avoids it, and so does an analytical `tgrad`.
#
# `ode_function`/`ode_problem` take a `tgrad` keyword: an exact `∂f/∂t`, handed straight to
# `ODEFunction` so a Rosenbrock method never has to differentiate through `t` at all. For this
# problem `f(t) = F(t) - A u_h` and `A` does not depend on `t`, so `∂f/∂t = ∂F/∂t`, assembled
# the same way `F` itself is, from the time derivative of the source:

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

# No `AutoFiniteDiff()` and no BDF fallback: `ode_problem` already hands the solver the exact
# Jacobian ``-A``, and `tgrad` supplies the `t`-derivative, so `Rodas5P` differentiates nothing.
@test prob_rosenbrock.f.jac !== nothing                                          #src
@test 1.0e-5 < normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ_r) < 5.0e-5               #src
#
# ## A time-dependent operator
#
# `tgrad` fixes the source half of `∂f/∂t`; the other method of [`semidiscretize`](@ref)
# fixes the operator half, for a spatial operator whose own coefficients vary with `t`. It
# keeps the Dirichlet problem, because `semidiscretize(build, l)` on a ghost space is
# refused. A fixed `BilinearForm` closes over a `Float64` coefficient buffer, which cannot
# hold the `Dual` a Rosenbrock stepper reaches for either way. `semidiscretize(build, l; ...)`
# instead takes a `build(t) -> (a, refill!)` factory, called once per element type `t` is ever seen
# at `Float64` on an ordinary step, and at `Dual` while `Rodas5P`'s default `autodiff`
# differentiates through `t`. The coefficient buffer it refills is always the right type.
# The source is held fixed here on purpose, to isolate the operator: a `t`-dependent source
# reached through `update_coefficients!` has the same `Float64`-buffer limitation `tgrad`
# fixes above, for the same reason:

α(t) = 1.0 + 0.5 * sinpi(2t)     # a diffusivity that genuinely varies with t
gₕ = Rₕ(Wₕ, x -> sinpi(x[1]))
l_t = form(Wₕ, v -> innerₕ(gₕ, v))

function build_diffusion(t)
    αₕ = element(Wₕ, typeof(t))
    a_t = form(Wₕ, Wₕ, (u, v) -> inner₊(αₕ * ∇ₕ(u), ∇ₕ(v)))
    refill!(t) = (fill!(parent(αₕ), α(t)); nothing)
    return a_t, refill!
end

sd_t = semidiscretize(build_diffusion, l_t; dirichlet = bcs)
prob_t = ode_problem(sd_t, Rₕ(Wₕ, x -> uexact(x, 0.0)), I)

sol_t_rosenbrock = solve(prob_t, Rodas5P(); reltol = 1e-11, abstol = 1e-13)
sol_t_fbdf = solve(prob_t, FBDF(); reltol = 1e-11, abstol = 1e-13)
maximum(abs.(sol_t_rosenbrock.u[end] .- sol_t_fbdf.u[end]))

# The same `ODEProblem`, one stepper differentiating through `t` by default and the other
# needing no `∂f/∂t` at all, land on the same answer. The operator needed neither
# `AutoFiniteDiff()` nor a hand-written `tgrad`.
@test maximum(abs.(sol_t_rosenbrock.u[end] .- sol_t_fbdf.u[end])) < 1.0e-6              #src
#
# ## Hand-rolled backward Euler
#
# `semidiscretize`/`ode_problem` hide the per-step matrix assembly. This section keeps the
# plain space and the Dirichlet problem, since [`assemble_add!`](@ref) takes forms on a plain
# space only. `assemble_add!` accumulates ``M/\Delta t`` and the ``t``-dependent stiffness piece
# ``K(t)`` into one preallocated matrix, in place, with no temporary and no sparse addition
# per step. Backward Euler for the same varying-diffusivity problem as the section above,
#
# ```math
# \Bigl(\frac{M}{\Delta t} + K(t^{n+1})\Bigr) u^{n+1} = \frac{M}{\Delta t} u^n + F,
# ```
#
# one linear solve per step (``F`` is the static source `gₕ` from `l_t` above, unchanged by
# `t`, so it is assembled once outside the loop):

using SparseArrays: nonzeros

Δt = 1 / 200
nsteps = round(Int, 1 / Δt)

m_form = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
M_mat = assemble(m_form)
F_source = assemble(l_t)

αₕ_be = element(Wₕ, 0.0)
k_form = form(Wₕ, Wₕ, (u, v) -> inner₊(αₕ_be * ∇ₕ(u), ∇ₕ(v)))

# `A_be` takes its pattern from a form that sums the mass and stiffness integrands, built
# for its pattern alone and never assembled. The pattern holds the entries of both pieces.
A_be = allocate_system_matrix(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))

bcs_be = dirichlet_constraints(Ωₕ, :boundary => x -> 0.0)
u_be = parent(Rₕ(Wₕ, x -> uexact(x, 0.0)))
F_be = zeros(ndofs(Wₕ))

for step in 1:nsteps
    t = step * Δt
    fill!(parent(αₕ_be), α(t))       # live coefficient, read by k_form's next assembly

    fill!(nonzeros(A_be), 0.0)
    assemble_add!(A_be, m_form, 1 / Δt)
    assemble_add!(A_be, k_form)
    dirichlet_bc!(A_be, Ωₕ, :boundary)   # constrain once, last; see the note in `assemble_add!`

    F_be .= F_source .+ (M_mat * u_be) ./ Δt
    dirichlet_bc!(F_be, Ωₕ, bcs_be, :boundary)

    global u_be = A_be \ F_be
end

maximum(abs.(u_be .- sol_t_fbdf.u[end]))

# Each step zeroes the stored values of `A_be` and keeps its pattern. `dirichlet_bc!` runs
# after both accumulations and never between them: accumulating a second piece into an
# already-constrained row would add its contribution on top of the identity `dirichlet_bc!`
# wrote there.
#
# Backward Euler at ``\Delta t = 1/200`` is first order in time, so it differs from the
# adaptive `FBDF` solve above by about ``6 \times 10^{-4}``.
@test 5.0e-4 < maximum(abs.(u_be .- sol_t_fbdf.u[end])) < 7.0e-4                         #src
