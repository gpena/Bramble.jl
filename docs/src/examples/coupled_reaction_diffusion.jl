# # Coupled nonlinear reaction-diffusion system
#
# Two species, coupled through a quadratic reaction term, solved with Newton's method the same
# way the [nonlinear Poisson example](poisson_nonlinear.md) does — except now the Jacobian
# differentiates through a *composite* space's assembly, not a scalar one. Every number and
# every plot below was produced by the code shown.
#
# ## Problem
#
# ```math
# \begin{aligned}
# -\Delta u + u + uv &= f_1 \\
# -\Delta v + v - uv &= f_2
# \end{aligned}
# \qquad \text{in } \Omega = (0,1)^2, \qquad u = v = 0 \text{ on } \partial\Omega,
# ```
#
# a predator-prey-shaped coupling without the time derivative: `u` grows through the
# interaction term, `v` is depleted by it. The manufactured solutions vanish on the boundary
# already, so homogeneous Dirichlet data is all that is needed:

using Bramble
using Bramble: ast_sparsity_detector
using Random

u_ex(x) = sin(π * x[1]) * sin(π * x[2])
v_ex(x) = sin(2π * x[1]) * sin(2π * x[2])
f1(x) = 2π^2 * u_ex(x) + u_ex(x) + u_ex(x) * v_ex(x)
f2(x) = 8π^2 * v_ex(x) + v_ex(x) - u_ex(x) * v_ex(x)

Random.seed!(20260903)
Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ = mesh(Ω, (24, 24), (false, false))
Wₕ = gridspace(Ωₕ)
Vₕ = Wₕ^Val(2)

bcs = dirichlet_constraints(Ω, :boundary => x -> 0.0)
f1ₕ = element(Wₕ)
avgₕ!(f1ₕ, f1)
f2ₕ = element(Wₕ)
avgₕ!(f2ₕ, f2)
l = form(Vₕ, q -> innerₕ(f1ₕ, q(1)) + innerₕ(f2ₕ, q(2)))
F = assemble(l; dirichlet = bcs)
nothing # hide

# The seed is what makes the numbers below reproducible: `(false, false)` draws the interior
# points from the global RNG, so without it the mesh -- and every figure on this page --
# would differ from build to build, and the suite could not assert what the page prints.
#
# ## Newton's method on a composite residual
#
# `uv` is quadratic in the unknowns, so it cannot sit inside a matrix independent of
# `w = (u, v)` the way the linear terms can — but it *can* sit inside a matrix that depends on
# the current guess, the same trick [the nonlinear Poisson example uses for a single
# species](poisson_nonlinear.md#Fixed-point-%28Picard%29-iteration), extended to a second one.
# Writing the coupling as `v_current * u(1)` in `u`'s own equation and `-u_current * u(2)` in
# `v`'s reproduces `uv` and `-uv` exactly once the trial function is evaluated at the current
# `w` — which is all `A(w)` needs to do. Nothing here works out `∂(uv)/∂u` and `∂(uv)/∂v` by
# hand; `ForwardDiff` differentiates through *how* `A` itself depends on `w` automatically:

function coupled_matrix(wₕ)
    u_c, v_c = components(wₕ)
    a = form(Vₕ,
        Vₕ,
        (p, q) -> inner₊(∇ₕ(p(1)), ∇ₕ(q(1))) + innerₕ(p(1), q(1)) + innerₕ(v_c * p(1), q(1)) +
                  inner₊(∇ₕ(p(2)), ∇ₕ(q(2))) + innerₕ(p(2), q(2)) - innerₕ(u_c * p(2), q(2)))
    return assemble(a; dirichlet = :boundary)
end
nothing # hide

# Its Jacobian is sparse for the same reason the [nonlinear Poisson
# example's](poisson_nonlinear.md#Newton's-method) is — the reaction term couples `u` and `v`
# only pointwise, so it adds nothing to the diffusion stencil's own reach — so the same sparse
# AD setup applies unchanged, just over twice as many unknowns:

using ForwardDiff, DifferentiationInterface
import SparseConnectivityTracer, SparseMatrixColorings

const sparse_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

function residual(w::AbstractVector{T}) where {T}
    wₕ = element(Vₕ, T)
    wₕ .= w
    A = coupled_matrix(wₕ)
    return A * w .- F
end

w = zeros(ndofs(Vₕ))
prep = prepare_jacobian(residual, sparse_ad, w)
J = DifferentiationInterface.jacobian(residual, prep, sparse_ad, w)  # once, for its sparse structure
newton_residuals = Float64[]
for it in 1:20
    r = residual(w)
    push!(newton_residuals, sqrt(sum(abs2, r)))
    newton_residuals[end] < 1e-10 && break
    DifferentiationInterface.jacobian!(residual, J, prep, sparse_ad, w)
    w .-= J \ r
end
length(newton_residuals), newton_residuals

# Quadratic convergence over a composite space, asserted rather than rendered. Bounded at 8 #src
# for the reason poisson_nonlinear.jl gives.                                                #src
@test length(newton_residuals) < 8                                                          #src
@test newton_residuals[end] < 1e-10                                                         #src

# Quadratic convergence, same as the single-species case — the composite space changes what
# the Jacobian differentiates through, not how well Newton converges once it has a correct one.
#
# ## Skipping the tracer here too
#
# `v_c` scaling a term routed into block `(1,1)` is a *different* leaf's component reaching
# into this one — [`jacobian_pattern`](@ref) reads that the same way a form term names a
# component, `U -> U(2)` rather than a stencil op, since `v_c` is read directly rather than
# averaged first. Block `(2,2)`'s own `u_c` dependency is named the same way, `U -> U(1)`:

using ADTypes

u_c0, v_c0 = components(element(Vₕ, 0.0))
a_for_pattern = form(Vₕ,
    Vₕ,
    (p, q) -> inner₊(∇ₕ(p(1)), ∇ₕ(q(1))) + innerₕ(p(1), q(1)) + innerₕ(v_c0 * p(1), q(1)) +
              inner₊(∇ₕ(p(2)), ∇ₕ(q(2))) + innerₕ(p(2), q(2)) - innerₕ(u_c0 * p(2), q(2)))

native_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = ast_sparsity_detector(a_for_pattern, U -> U(2), U -> U(1)),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

w_native = zeros(ndofs(Vₕ))
prep_native = prepare_jacobian(residual, native_ad, w_native)
J_native = DifferentiationInterface.jacobian(residual, prep_native, native_ad, w_native)
newton_residuals_native = Float64[]
for it in 1:20
    r = residual(w_native)
    push!(newton_residuals_native, sqrt(sum(abs2, r)))
    newton_residuals_native[end] < 1e-10 && break
    DifferentiationInterface.jacobian!(residual, J_native, prep_native, native_ad, w_native)
    w_native .-= J_native \ r
end
newton_residuals_native

@test length(newton_residuals_native) < 8                                                   #src
@test newton_residuals_native[end] < 1e-10                                                  #src
@test maximum(abs.(w_native .- w)) < 1e-8                                                   #src

# Same quadratic convergence, no tracing pass paid for it: `v_c0`/`u_c0` are read at `w = 0`
# only to build *some* concrete `BilinearForm` — the pattern is a property of `a`'s AST, not
# of those values. `SparseConnectivityTracer`'s tracer still works here regardless of how
# `coupled_matrix` was built, composite space and all — the case to reach for it is a residual
# whose matrix does not come from a `BilinearForm` in the first place, which is not this one.
#
# ## Solving with NonlinearSolve.jl
#
# Everything above is a hand-written Newton loop, the same as [the nonlinear Poisson
# example](poisson_nonlinear.md). [`nonlinear_problem`](@ref) wraps a residual into the
# `NonlinearProblem` that [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/)
# takes regardless of what space the residual is built over — a composite space changes
# nothing about the wrapping, only what `jac_prototype` looks like once built. `J_native`
# above is already that prototype, the block-sparse pattern [`ast_sparsity_detector`](@ref)
# read off `a_for_pattern`'s two components, so it is handed through unchanged. In place,
# the same way `coupled_matrix` is written out-of-place above and `residual!` here is not:
# `mul!` writes `A * w_vec` into the caller's own `res` rather than allocating a fresh vector
# every evaluation.

using NonlinearSolve
using SciMLBase: SciMLBase
using LinearAlgebra: mul!

function residual!(res::AbstractVector, w_vec::AbstractVector{T}, p) where {T}
    wₕ = element(Vₕ, T)
    wₕ .= w_vec
    A = coupled_matrix(wₕ)
    mul!(res, A, w_vec)
    res .-= F
    return res
end

prob = nonlinear_problem(residual!, zeros(ndofs(Vₕ)); jac_prototype = J_native)
sol_ns = solve(prob, NewtonRaphson(); abstol = 1e-10)
sol_ns.retcode, maximum(abs, sol_ns.u .- w_native)

# The same answer Newton reached by hand above, to the tolerance both were asked for --      #src
# checked per species, not only combined: a routing mistake in the composite Jacobian would  #src
# show up as one species converging while the other silently used the wrong block, which a   #src
# single combined comparison could hide (bramble-verification §5, the same reason the        #src
# exact-solution check above is split by species).                                           #src
@test sol_ns.retcode == SciMLBase.ReturnCode.Success                                          #src
@test maximum(abs, sol_ns.u .- w_native) < 1e-8                                              #src
wₕ_ns = element(Vₕ)                                                                          #src
wₕ_ns .= sol_ns.u                                                                             #src
uₕ_ns, vₕ_ns = components(wₕ_ns)                                                              #src
uₕ_native, vₕ_native = components(element(Vₕ, w_native))                                     #src
@test maximum(abs, parent(uₕ_ns) .- parent(uₕ_native)) < 1e-8                                 #src
@test maximum(abs, parent(vₕ_ns) .- parent(vₕ_native)) < 1e-8                                 #src
#
# Quadratic convergence over a composite space carries over unchanged to `NonlinearSolve.jl`
# too -- the block sparsity pattern is all a composite residual ever needed to hand it, and
# `NewtonRaphson` reaches the same solution the hand-written loops above do. See [the nonlinear
# Poisson example](poisson_nonlinear.md) for a measured Picard-against-NonlinearSolve
# comparison; the same contrast holds here and is not repeated a second time.

wₕ = element(Vₕ)
wₕ .= w
uₕ, vₕ = components(wₕ)
uexact, vexact = Rₕ(Wₕ, u_ex), Rₕ(Wₕ, v_ex)
norm₁ₕ(uₕ .- uexact), norm₁ₕ(vₕ .- vexact)

# Checked per species: a routing mistake in the composite Jacobian shows up as one species  #src
# converging while the other silently used the wrong block, which a single combined error   #src
# would hide (bramble-verification §5).                                                     #src
@test 1.0e-6 < norm₁ₕ(uₕ .- uexact) < 1.0e-1                                                 #src
@test 1.0e-6 < norm₁ₕ(vₕ .- vexact) < 5.0e-1   # v oscillates twice as fast as u              #src

# ## Exploring the system interactively
#
# The panel below is *not* a replay of the solve above, and does not share its manufactured
# solution — it poses a different boundary-value problem in the same two unknowns, chosen so
# every slider has a visible effect rather than being fought back to a fixed answer:
#
# ```math
# \begin{aligned}
# -D_u\Delta u + au + \gamma uv &= 0 \\
# -D_v\Delta v + bv - \gamma uv &= 0
# \end{aligned}
# \qquad \text{in } \Omega = (0,1)^2, \qquad u = v = 1 \text{ on } \partial\Omega,
# ```
#
# with no volumetric source anywhere: `f1 = f2 = 0`. A manufactured right-hand side would
# force the discrete solution back to the same prescribed answer regardless of `a`, `b`, `γ`
# or `D_u/D_v` — only a vanishingly small error field would move. Here the *only* input is a
# constant Dirichlet supply of both species on the boundary (it has to be nonzero for both:
# the reaction terms only ever reach the diagonal of each field's own block, the same way
# `coupled_matrix` above assembles them, so a field with nothing driving it directly solves to
# exactly zero and leaves the other field's `γuv` coupling with nothing to act on). Because
# this boundary-value problem has no closed-form solution, the panel measures its own
# discretization error against a solve on a fixed, much finer uniform reference mesh instead
# of against `u_ex`/`v_ex`.
#
# Each species is its own 2D scalar field — `components(wₕ)` gives a view directly onto it, no
# new solve or copy needed, though the panel below solves its own problem rather than reusing
# `uₕ`/`vₕ`. It runs its own block Gauss-Seidel Picard iteration client-side, so the reaction
# coefficients `a`, `b`, coupling `γ` and diffusion ratio `D_u/D_v` sliders can be swept without
# a round trip to Julia — dragging them re-solves both fields and redraws the linked `u_h`/`v_h`
# heatmaps, the `2×2` Jacobian block-sparsity spy plot, and the cross-section profile in place:

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
coupled_reaction_diffusion_widget(uₕ, vₕ; title = "Coupled reaction-diffusion") # hide

# ## Checking the answer
#
# The same nested-random-mesh pattern as every other example, checking each species' own error
# separately — a routing mistake would show up as one converging correctly while the other
# silently used the wrong block, which a single combined error could hide. A *dense*
# `ForwardDiff.jacobian` over two coupled species would cost `(2n)^2` against the scalar
# examples' `n^2`, and was what forced this example to stay at three small refinement levels
# before switching to sparse AD; with it, this reaches five levels — the same order of tens of
# thousands of degrees of freedom the [linear coupled
# example](convection_diffusion_linear.md) reaches at six — in about a second per level:

Random.seed!(20260903)

function coupled_series(; n0::Int = 5, levels::Int)
    Ωc = mesh(Ω, (n0, n0), (false, false))
    hs = Float64[]
    erru, errv = Float64[], Float64[]
    for level in 1:levels
        Wc = gridspace(Ωc)
        Vc = Wc^Val(2)
        bcs_c = dirichlet_constraints(Ω, :boundary => x -> 0.0)
        f1_c = element(Wc)
        avgₕ!(f1_c, f1)
        f2_c = element(Wc)
        avgₕ!(f2_c, f2)
        l_c = form(Vc, q -> innerₕ(f1_c, q(1)) + innerₕ(f2_c, q(2)))
        F_c = assemble(l_c; dirichlet = bcs_c)

        Ac(wₕ) = begin
            u_c, v_c = components(wₕ)
            assemble(
                form(Vc,
                    Vc,
                    (p, q) -> inner₊(∇ₕ(p(1)), ∇ₕ(q(1))) + innerₕ(p(1), q(1)) +
                              innerₕ(v_c * p(1), q(1)) +
                              inner₊(∇ₕ(p(2)), ∇ₕ(q(2))) + innerₕ(p(2), q(2)) -
                              innerₕ(u_c * p(2), q(2)));
                dirichlet = :boundary)
        end
        rc(w::AbstractVector{T}) where {T} = begin
            wₕ = element(Vc, T)
            wₕ .= w
            Ac(wₕ) * w .- F_c
        end

        w_c = zeros(ndofs(Vc))
        prep_c = prepare_jacobian(rc, sparse_ad, w_c)
        J_c = DifferentiationInterface.jacobian(rc, prep_c, sparse_ad, w_c)
        for it in 1:20
            r = rc(w_c)
            sqrt(sum(abs2, r)) < 1e-10 && break
            DifferentiationInterface.jacobian!(rc, J_c, prep_c, sparse_ad, w_c)
            w_c .-= J_c \ r
        end
        w_ch = element(Vc)
        w_ch .= w_c
        u_ch, v_ch = components(w_ch)
        uexact_c, vexact_c = Rₕ(Wc, u_ex), Rₕ(Wc, v_ex)

        push!(hs, hₘₐₓ(Ωc))
        push!(erru, norm₁ₕ(u_ch .- uexact_c))
        push!(errv, norm₁ₕ(v_ch .- vexact_c))
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, erru, errv
end

hs, erru, errv = coupled_series(; levels = 5)
order_u = log(erru[end - 1] / erru[end]) / log(hs[end - 1] / hs[end])
order_v = log(errv[end - 1] / errv[end]) / log(hs[end - 1] / hs[end])
(order_u, order_v)

#-

order_u > 1.9 && order_v > 1.9

# Per species again, and bracketed above as well as below.                                  #src
@test 1.9 < order_u < 3.0                                                                   #src
@test 1.9 < order_v < 3.0                                                                   #src

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(hs, erru, "u", "#5B5FC7"), (hs, errv, "v", "#0E7C86")]; title = "Coupled nonlinear reaction, ‖·‖₁ₕ") # hide

# Second order for both species, same rate as every other example — the composite space and
# the quadratic coupling change how the residual and its Jacobian are built, not the
# discretization's own accuracy once Newton has converged to it.
#
# `coupled_series` above uses `sparse_ad`, the tracer, at every level — `native_ad`'s
# substitution (`ast_sparsity_detector(a, U -> U(2), U -> U(1))` in place of `sparse_ad`'s
# `sparsity_detector`) works here unchanged too. Not re-run a second time here, the same
# reason [the nonlinear Poisson example](poisson_nonlinear.md#Checking-the-answer) does not
# re-run its own convergence sweep a second time either.
