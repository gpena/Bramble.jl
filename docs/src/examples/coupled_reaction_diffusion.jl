# # Coupled nonlinear reaction-diffusion system
#
# Two species coupled through a quadratic reaction term, solved with Newton's method as in the
# [nonlinear Poisson example](poisson_nonlinear.md), with a Jacobian that differentiates
# through assembly on a composite space.
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
# a predator-prey-shaped coupling without the time derivative: `u` is depleted by the
# interaction term and `v` grows through it. The manufactured solutions vanish on the
# boundary, so homogeneous Dirichlet data is all that is needed.
#
# ## Mesh, space and load

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

# The seed fixes the random mesh, as in the
# [nonlinear Poisson example](poisson_nonlinear.md#The-mesh-and-the-load).
#
# ## Newton's method on a composite residual
#
# `uv` is quadratic in the unknowns, so it cannot sit in a matrix independent of
# `w = (u, v)`, but it can sit in a matrix that depends on the current guess, as in
# [the nonlinear Poisson example](poisson_nonlinear.md#Fixed-point-%28Picard%29-iteration).
# Writing the coupling as `v_current * u(1)` in `u`'s equation and `-u_current * u(2)` in
# `v`'s reproduces `uv` and `-uv` once the trial function is evaluated at the current `w`.
# `ForwardDiff` differentiates how `A` depends on `w`, so no partial derivative of `uv` is
# written by hand:

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
# example's](poisson_nonlinear.md#Newton's-method) is: the reaction term couples `u` and `v`
# only pointwise, so it adds nothing to the diffusion stencil's reach. The same sparse AD
# setup applies, over twice as many unknowns:

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

# The assertions bound the count at 8, for the reason poisson_nonlinear.jl gives, and the   #src
# last residual; the rendered list shows the quadratic decay.                               #src
@test length(newton_residuals) < 8                                                          #src
@test newton_residuals[end] < 1e-10                                                         #src

# Convergence is quadratic, as in the single-species case. The solution, each species with
# the mesh lines of the random grid:

wₕ = element(Vₕ)
wₕ .= w
uₕ, vₕ = components(wₕ)
include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
surface_plot(uₕ; title = "u", mesh_lines = true) # hide

#-

surface_plot(vₕ; title = "v", mesh_lines = true) # hide

# ## Skipping the tracer
#
# The pattern comes from the AST of `a` instead of a tracing pass, as
# [in the nonlinear Poisson example](poisson_nonlinear.md#Skipping-the-tracer). In block
# `(1,1)` the coefficient `v_c` is the value of component 2, which
# [`jacobian_pattern`](@ref) names `U -> U(2)`, with no stencil op because `v_c` is read
# directly. Block `(2,2)` reads `u_c`, named `U -> U(1)`. Each dependency is applied to every
# block, a safe superset:

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

# Convergence is the same, without a tracing pass. `v_c0` and `u_c0` are read at `w = 0` only
# to build a concrete `BilinearForm`, since the pattern depends on the AST of `a`, not on
# those values. The tracer remains the option for a residual whose matrix does not come from a
# `BilinearForm`.
#
# ## NonlinearSolve.jl on the composite system
#
# [`nonlinear_problem`](@ref) wraps a residual into the `NonlinearProblem` that
# [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/) takes, whatever the
# space. `J_native` is the `jac_prototype`: the block-sparse pattern
# [`ast_sparsity_detector`](@ref) read off `a_for_pattern`. `residual!` is the in-place form of
# `residual`: `mul!` writes `A * w_vec` into the caller's `res` instead of allocating the
# product.

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

# The same answer Newton reached by hand above, to the tolerance both were asked for, and   #src
# checked per species as well as combined: a routing mistake in the composite Jacobian would#src
# show up as one species converging while the other silently used the wrong block, which a  #src
# single combined comparison could hide (bramble-verification §5).                          #src
@test sol_ns.retcode == SciMLBase.ReturnCode.Success                                          #src
@test maximum(abs, sol_ns.u .- w_native) < 1e-8                                              #src
wₕ_ns = element(Vₕ)                                                                          #src
wₕ_ns .= sol_ns.u                                                                             #src
uₕ_ns, vₕ_ns = components(wₕ_ns)                                                              #src
uₕ_native, vₕ_native = components(element(Vₕ, w_native))                                     #src
@test maximum(abs, parent(uₕ_ns) .- parent(uₕ_native)) < 1e-8                                 #src
@test maximum(abs, parent(vₕ_ns) .- parent(vₕ_native)) < 1e-8                                 #src
#
# `NewtonRaphson` reaches the same solution as the hand-written loops. Against the exact
# fields, per species:

uexact, vexact = Rₕ(Wₕ, u_ex), Rₕ(Wₕ, v_ex)
norm₁ₕ(uₕ .- uexact), norm₁ₕ(vₕ .- vexact)

# Checked per species: a routing mistake in the composite Jacobian shows up as one species  #src
# converging while the other silently used the wrong block, which a single combined error   #src
# would hide (bramble-verification §5).                                                     #src
@test 1.0e-6 < norm₁ₕ(uₕ .- uexact) < 1.0e-1                                                 #src
@test 1.0e-6 < norm₁ₕ(vₕ .- vexact) < 5.0e-1   # v oscillates twice as fast as u              #src

# ## Exploring the system interactively
#
# The panel below poses a different boundary-value problem in the same two unknowns, so that
# every slider has a visible effect:
#
# ```math
# \begin{aligned}
# -D_u\Delta u + au + \gamma uv &= 0 \\
# -D_v\Delta v + bv - \gamma uv &= 0
# \end{aligned}
# \qquad \text{in } \Omega = (0,1)^2, \qquad u = v = 1 \text{ on } \partial\Omega,
# ```
#
# with no volumetric source: `f1 = f2 = 0`. A manufactured right-hand side would fix the
# solution whatever the parameters. The only input is a constant Dirichlet supply of both
# species on the boundary, nonzero for both: the reaction terms reach only the diagonal of
# each field's own block, so a field with nothing driving it solves to zero and leaves `γuv`
# nothing to act on. The problem has no closed-form solution, so the panel measures its
# discretization error against a solve on a fixed, much finer uniform mesh.
#
# `components(wₕ)` gives each species as a view into `wₕ`. The panel runs its own block
# Gauss-Seidel Picard iteration in JavaScript, so the sliders for `a`, `b`, `γ` and `D_u/D_v`
# need no round trip to Julia. Dragging one re-solves both fields and redraws the linked
# `u_h` and `v_h` heatmaps, the `2×2` Jacobian block-sparsity plot and the cross-section
# profile:

coupled_reaction_diffusion_widget(uₕ, vₕ; title = "Coupled reaction-diffusion") # hide

# ## Convergence of the coupled system
#
# Five levels of the same nested random mesh, each species' error checked separately; the
# finest has 8,450 unknowns (``2 \times 65^2``):

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

# Both species converge at second order.
#
# ## Where to go next
#
#   - [Nonlinear Poisson](poisson_nonlinear.md) for the single-species version, with the
#     Picard comparison and the cached assembly this page leaves out.
#   - The [coupled systems tutorial](@ref tutorial_coupled) for composite spaces and the
#     block structure of their matrices, and the [space tutorial](@ref space_composite) for
#     vector elements.
