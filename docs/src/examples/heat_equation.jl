# # Heat equation
#
# The first time-dependent problem in this manual. Space is discretised exactly as in the
# [linear Poisson example](poisson_linear.md); time is left continuous and handed to a stepper
# from the SciML stack.
#
# ## Problem
#
# ```math
# \partial_t u = \Delta u + f \text{ in } \Omega \times (0, T], \qquad u = g \text{ on }
# \texttt{left}, \qquad u + \partial_n u = h \text{ on } \texttt{right}, \qquad
# u(\cdot, 0) = u_0,
# ```
#
# on ``\Omega = (0,1)`` with ``\partial_n`` the outward normal derivative, and the manufactured
# solution ``u_{\text{exact}}(x, t) = e^{-t} \sin(\pi x)``, which forces
#
# ```math
# f(x, t) = (\pi^2 - 1)\, e^{-t} \sin(\pi x),
# ```
#
# vanishes at the left end, so ``g = 0``, and gives the Robin data ``h(x, t) = u_{\text{exact}} +
# \partial_x u_{\text{exact}} = e^{-t} (\sin(\pi x) + \pi \cos(\pi x))`` at the right end. Later
# sections also use the same solution with a Dirichlet condition at both ends, since it
# vanishes at both.
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
# ``\langle \cdot, \cdot \rangle_h``. It is diagonal, since that inner product is a weighted sum
# over grid points.
#
# ## Boundary relations through ghost points
#
# [`semidiscretize`](@ref) builds exactly this. The Robin end goes through the ghost layer, as
# in the [Neumann and Robin Poisson example](neumann_robin_poisson.md): `gridspace(Ωₕ; ghosts)`
# adds a ghost point beyond the end, and [`constrained`](@ref) takes a function of the trial
# function `u` that returns one `side => relation => data` entry per side. The data may take
# the time as a second argument, `(x, t) -> ...`, which `semidiscretize` finds by the arity
# and refreshes at every step. The Dirichlet end is the relation `u => g`, which removes its
# node from the unknowns. The spatial form is written ``\langle -\Delta_h u, v \rangle_h``, which
# reads the ghost values the relations fix:

using Bramble
using Bramble: mass_matrix
using OrdinaryDiffEqBDF

uexact(x, t) = exp(-t) * sinpi(x[1])
source(x, t) = (pi^2 - 1) * exp(-t) * sinpi(x[1])
robin(x, t) = exp(-t) * (sinpi(x[1]) + pi * cospi(x[1]))

Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωₕ = mesh(Ω, 101)
Wₕ = gridspace(Ωₕ)
I = interval(0.0, 1.0)          # the time domain

function robin_problem(Ωₕ, (left, right); T = 1.0)
    Cₕ = constrained(gridspace(Ωₕ; ghosts = right),
        u -> (left => u => (x, t) -> 0.0, right => u + ∂ₙ(u) => robin))
    fₕ = element(Bramble.padded_space(Cₕ), 0.0)
    a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
    l = form(Cₕ, v -> innerₕ(fₕ, v))
    sd = semidiscretize(a, l; update_coefficients! = t -> Rₕ!(fₕ, x -> source(x, t)))
    u₀ = parent(Rₕ(gridspace(Ωₕ), x -> uexact(x, 0.0)))[2:end]   # the left node is no unknown
    sol = solve(ode_problem(sd, u₀, (0.0, T)), FBDF(); reltol = 1e-10, abstol = 1e-12)
    return Cₕ, sd, sol
end

Cₕ, sd_robin, sol_robin = robin_problem(Ωₕ, (:left, :right))

# `update_coefficients!` is called with the current time just before each assembly, and a
# time-dependent source goes there. `Rₕ!` writes into the grid function `fₕ` that the form
# already references, so nothing is rebuilt. `fₕ` lives on the padded space because the
# ghost space has no `Rₕ!` of its own. The initial condition holds the 100 unknowns, the grid
# points except the eliminated left node.
#
# ## Eliminated nodes in the mass matrix
#
# The Dirichlet node is no unknown and the Robin relation fixes only the ghost, so no row of
# the system is algebraic. [`mass_matrix`](@ref) is the diagonal of grid weights on the
# remaining 100 nodes, and the system is an ordinary differential equation:

M_robin = mass_matrix(sd_robin)
(size(M_robin), M_robin[1, 1], M_robin[100, 100])   # first interior node, Robin end

# The first unknown is the node next to the Dirichlet end and carries the full weight h, and #src
# the last is the Robin end, which carries half of it as the end of the mesh.                #src
@test size(M_robin) == (100, 100)                                      #src
@test M_robin[1, 1] ≈ hₘₐₓ(Ωₕ)                                         #src
@test M_robin[100, 100] ≈ hₘₐₓ(Ωₕ) / 2                                 #src

# ## The solution at the final time
#
# `Bramble.lift(Cₕ, u, t)` returns the padded-grid values of the unknowns `u` with the
# ghost value and the eliminated node filled in from the relations at time ``t``. The
# ghost lies beyond the right end, so the physical nodes are its first 101 entries:

using Bramble: lift

Uₚ = parent(lift(Cₕ, sol_robin.u[end], 1.0))
uₕ_robin = element(Wₕ)
parent(uₕ_robin) .= Uₚ[1:101]
normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ_robin)

# The error is bracketed on both sides: a solution that collapsed to zero, or stayed at the   #src
# initial condition, would sit far outside this window, and so would a first-order one.       #src
@test 1.0e-5 < normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ_robin) < 5.0e-5   #src

# The relation is satisfied by the ghost, so the check that means something is one-sided and
# independent of it: the three-point backward difference of the lifted solution at the
# Robin end, with the actual spacings, must satisfy ``u + \partial_n u = h`` up to its
# truncation error, of order ``h^2``:

function robin_residual(Ωₕ, Cₕ, sol)
    U = parent(lift(Cₕ, sol.u[end], 1.0))
    xs = collect(points(Ωₕ))
    n = length(xs)
    h₁, h₂ = xs[n] - xs[n - 1], xs[n - 1] - xs[n - 2]
    dₙ = (2h₁ + h₂) / (h₁ * (h₁ + h₂)) * U[n] - (h₁ + h₂) / (h₁ * h₂) * U[n - 1] +
         h₁ / (h₂ * (h₁ + h₂)) * U[n - 2]
    return abs(U[n] + dₙ - robin(xs[n], 1.0))
end

robin_residual(Ωₕ, Cₕ, sol_robin)

# The residual is the one-sided truncation error plus the error of the solution itself, so   #src
# it is bracketed away from zero as well as above, and the eliminated Dirichlet node of the   #src
# lift holds its datum exactly.                                                               #src
@test 1.0e-5 < robin_residual(Ωₕ, Cₕ, sol_robin) < 1.0e-2                                    #src
@test parent(lift(Cₕ, sol_robin.u[end], 1.0))[1] == 0.0                                      #src

# ## Dirichlet conditions as algebraic constraints
#
# The `dirichlet` keyword of `semidiscretize` belongs to a plain space; a ghost space
# refuses it, and takes the relation `u => g` instead. The sections below need what the ghost
# path does not offer yet, so they keep a plain space and a Dirichlet condition at both ends,
# which the manufactured solution satisfies with ``g = 0``. The time domain passed to
# [`dirichlet_constraints`](@ref) lets the boundary values be written as `g(x, t)`. The
# spatial form is the one the steady problem writes, so the steady state of the system
# solves ``A u_h = F``:

fₕ = element(Wₕ, 0.0)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wₕ, v -> innerₕ(fₕ, v))

bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)

sd = semidiscretize(a, l;
    dirichlet = bcs,
    update_coefficients! = t -> Rₕ!(fₕ, x -> source(x, t)))

# Here the constrained row of `A` is ``e_k`` and the matching entry of `F(t)` is
# ``g(x_i, t)``, which is what `assemble` produces for the steady problem too.
# [`mass_matrix`](@ref) zeroes those same rows, so each one reads
#
# ```math
# 0 = g(x_i, t) - u_h[i],
# ```
#
# the boundary condition itself. The system is therefore a differential-algebraic one, and
# needs only ``g``, never ``\partial_t g``, which prescribing ``u_h'`` on the boundary would
# have required:

M = mass_matrix(sd)
(M[1, 1], M[51, 51], M[101, 101])   # boundary, interior, boundary

# The two constrained rows are zeroed, and the interior one carries the grid weight the  #src
# discrete inner product gives that point, which on this uniform mesh is h.               #src
@test M[1, 1] == 0.0 && M[101, 101] == 0.0                      #src
@test M[51, 51] ≈ hₘₐₓ(Ωₕ)                                      #src

# ## Solving with a BDF stepper
#
# [`ode_problem`](@ref) wraps the semidiscretisation, its mass matrix, and its exact Jacobian
# ``-A`` into a problem `OrdinaryDiffEq` can step. Because the mass matrix is singular, the
# method has to be one that admits that: `FBDF` and `QNDF` here, or a Rosenbrock method such
# as `Rodas5P`:

prob = ode_problem(sd, Rₕ(Wₕ, x -> uexact(x, 0.0)), I)
sol = solve(prob, FBDF(); reltol = 1e-10, abstol = 1e-12)

uₕ = element(Wₕ)
parent(uₕ) .= sol.u[end]
normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ)

# Bracketed on both sides: a solution that collapsed to zero, or to the initial condition, #src
# would sit far outside this window, and so would a first-order one on 101 points.          #src
@test 1.0e-5 < normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ) < 5.0e-5  #src

# The initial condition is copied, never mutated, and the copy is made consistent with the
# algebraic rows at ``t_0`` before stepping.

# A surface plot of the whole time evolution, `x` and `t` the two horizontal axes, `u` as
# height and colour, needs no new solve: `sol` already interpolates continuously in `t`, so
# sampling it at a uniform grid of times is enough to lay one out. The red curve traces
# `u(x, t)` at the current instant and loops on its own:

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide

ts = range(0.0, 1.0; length = 60)
Z = reduce(vcat, (sol(t)' for t in ts))
spacetime_surface_plot(points(Ωₕ), collect(ts), Z; title = "Heat equation, x-t-u")

# `sol` also exports as a ParaView time series; see
# [Plot or export a grid function](../howto/plotting_export.md).
#
# !!! note "Rosenbrock methods and time-dependent sources"
#     `Rodas5P` and friends also want the time derivative of the right-hand side, which they
#     build by differentiating through `t`. An `update_coefficients!` hook writing into a
#     `Float64` grid function, as the one above does, cannot accept a `ForwardDiff.Dual` time,
#     so pass `Rodas5P(autodiff = AutoFiniteDiff())` from `ADTypes`, use a BDF method (which
#     needs no `∂f/∂t` at all), or supply an analytical `tgrad`, as the
#     [time stepping example](@ref "Heat equation: time stepping details") does.
#
# ## Convergence under refinement
#
# Refining the mesh while holding the time tolerance far below the spatial error isolates
# the spatial error. The uniform mesh above would flatter the rate: on it ``\sin(\pi x)`` is
# an exact discrete eigenmode of the Laplacian, a special case that says little about the
# scheme. The series therefore runs on one seeded random coarse mesh,
# refined in place with [`iterative_refinement!`](@ref) so every finer level is the same
# random mesh, dyadically split. The domain carries no `:left` or `:right` marker, which
# the one-argument `iterative_refinement!` would refuse to drop. The uniform mesh stays
# above, where the check on the mass matrix needs a known grid weight ``h``:

using Random

function heat_series(levels; n0 = 11)
    Random.seed!(20260903)
    Ωs = domain(interval(0.0, 1.0))
    Ωc = mesh(Ωs, n0, false)
    hs, errs = Float64[], Float64[]
    for level in 1:levels
        Wc = gridspace(Ωc)
        fc = element(Wc, 0.0)
        ac = form(Wc, Wc, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
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
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, errs
end

hs, errs = heat_series(5)   # 11 up to 161 points
orders = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(errs) - 1)]

#-

all(>(1.95), orders)    # every fitted order above 1.95

# The line above renders `true` on the page and asserts nothing there; this is where the   #src
# rate is pinned, so a decay to first order fails the suite instead of publishing `false`.  #src
@test all(>(1.95), orders)                                       #src

# The Robin problem runs on the same seeded random mesh through `robin_problem`, whose
# sides here are the `:xmin` and `:xmax` of a domain with no markers of its own. Its error
# is the one at the physical nodes of the lifted solution, and the residual of the Robin
# relation, as above, comes with it:

function robin_series(levels; n0 = 11)
    Random.seed!(20260903)
    Ωc = mesh(domain(interval(0.0, 1.0)), n0, false)
    hs, errs, resids = Float64[], Float64[], Float64[]
    for level in 1:levels
        Cc, _, solc = robin_problem(Ωc, (:xmin, :xmax))
        Wc = gridspace(Ωc)
        uc = element(Wc)
        parent(uc) .= parent(lift(Cc, solc.u[end], 1.0))[1:length(parent(uc))]
        push!(hs, hₘₐₓ(Ωc))
        push!(errs, normₕ(Rₕ(Wc, x -> uexact(x, 1.0)) - uc))
        push!(resids, robin_residual(Ωc, Cc, solc))
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, errs, resids
end

hs_r, errs_r, resids_r = robin_series(5)
orders_r = [log(errs_r[i] / errs_r[i + 1]) / log(hs_r[i] / hs_r[i + 1])
            for i in 1:(length(errs_r) - 1)]

#-

all(>(1.95), orders_r)    # second order with a Robin end too

# The ghost layer costs no order, and the Robin residual falls as ``h^2`` as well. The       #src
# bracket keeps both away from zero, so a solve that reproduced the answer by construction  #src
# would fail.                                                                               #src
@test all(>(1.95), orders_r)                                                                #src
@test all(e -> 1.0e-7 < e < 1.0e-1, errs_r)                                                 #src
@test resids_r[1] / resids_r[end] > 100 && resids_r[end] > 1.0e-7                           #src

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(hs, errs, "Dirichlet", "#5B5FC7"), # hide
        (hs_r, errs_r, "Dirichlet and Robin", "#D1495B")]; # hide
    title = "Heat equation, ‖·‖ₕ at t = 1") # hide

# ## Time-dependent boundary data
#
# The Dirichlet data above do not depend on time, since the manufactured solution vanishes
# on ``\partial\Omega``. Driving the problem entirely from the boundary puts `(x, t)` in a
# Dirichlet condition: start at zero and raise the left end linearly, with the right end
# held fixed. The two ends are named on the domain above; a bare `domain(interval(...))`
# registers only `:boundary` and `:interior`, so there would be no `:left` to constrain.

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

# The ends hold g to solver tolerance, since the Dirichlet rows are algebraic constraints   #src
# in the DAE that FBDF's nonlinear solve satisfies up to its convergence criterion. The      #src
# midpoint is short of the steady 0.5 for the reason the prose below gives, with the window  #src
# tight enough that reaching the steady line would fail.                                     #src
@test u_end[1]≈1.0 atol=1e-8 rtol=1e-8                            #src
@test u_end[101]≈0.0 atol=1e-8 rtol=1e-8                          #src
@test 0.43 < u_end[51] < 0.45                                    #src

# At ``t = 1`` the ends hold the prescribed ``g`` to solver tolerance: one and zero. The
# interior is climbing towards the straight line ``1 - x`` that the source-free steady
# problem gives, but has not arrived, with ``0.44`` at the midpoint against the steady
# ``0.5``. The left end was still moving over the whole interval, so diffusion is chasing a
# boundary value that never settled.
#
# ## Where to go next
#
#   - [`semidiscretize`](@ref), [`ode_problem`](@ref) and [`ode_function`](@ref)
#     in the [API reference](../api.md).
#   - [Heat equation: time stepping details](@ref "Heat equation: time stepping details") for
#     an analytical `tgrad`, a time-dependent operator and a hand-written backward Euler loop.
#   - [Linear and bilinear forms](@ref tutorial_form) for the forms and the
#     [Dirichlet conditions](@ref form_dirichlet) used throughout.
#   - [Linear Poisson](poisson_linear.md) for the steady version of the same spatial operator.
