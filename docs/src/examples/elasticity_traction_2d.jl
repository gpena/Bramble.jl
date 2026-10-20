# # Elasticity with a traction condition
#
# Linear elasticity in the unit square with a prescribed traction on one side, a
# traction-free side and displacement on the other two. A traction couples the components,
# which is why it goes through ghost points: `constrained` closes each ghost from a relation
# that reads both components of the displacement. The page then checks those relations on
# the lifted solution, and the order of the error on graded grids, where it has to reach two.
# Every number and every plot below was produced by the code shown.
#
# ## Problem
#
# With the stress ``\sigma(u) = 2\mu\,\varepsilon(u) + \lambda\,(\nabla\cdot u)\, I``,
#
# ```math
# -\nabla\cdot\sigma(u) = f \text{ in } \Omega = (0,1)^2, \qquad
# \sigma(u)\, n = t \text{ on } \texttt{xmin} \text{ and } \texttt{ymin}, \qquad
# u = g \text{ on } \texttt{xmax} \text{ and } \texttt{ymax},
# ```
#
# with ``\lambda = 2``, ``\mu = 1/2``. The manufactured displacement is chosen so that the
# traction on `ymin` vanishes, a traction-free side, while the one on `xmin` does not. The
# loads and the data come from it by automatic differentiation, so the reference shares no
# code with the solve.

using Bramble
using Bramble: Dcₓ, Dcᵧ, D̃ₓ, D̃ᵧ, D₋ₓ, D₋ᵧ, change_points!
using ForwardDiff

const λ, μ = 2.0, 0.5
const c = λ / (λ + 2μ)

uᵉ = (x -> sin(x[1] + 0.3) * cos(x[2]) + exp(x[1]) * x[2]^2 / 2,
    x -> -c * cos(x[1] + 0.3) * sin(x[2]) + cos(2x[1]) * x[2]^2)

∇(f, x) = ForwardDiff.gradient(f, collect(Float64, x))
hess(f, x) = ForwardDiff.hessian(f, collect(Float64, x))

# ## The collocated form
#
# The equation is posed at every node as `−μ Δₕ uᵢ − (λ+μ) Σⱼ ∂ᵢ∂ⱼ uⱼ = fᵢ`: the second
# difference of one axis for ``j = i`` and the centred mixed difference for ``j \ne i``. The
# loads are ``-\mu\,\Delta u_i - (\lambda+\mu)\,\partial_i (\nabla\cdot u)``.

second(u, a, b) = a == b ? (a == 1 ? D̃ₓ(D₋ₓ(u)) : D̃ᵧ(D₋ᵧ(u))) : Dcₓ(Dcᵧ(u))

function elasticity(u, v)
    row(i) = innerₕ(-μ * Δₕ(u(i)), v(i)) +
             sum(innerₕ(-(λ + μ) * second(u(j), i, j), v(i)) for j in 1:2)
    return row(1) + row(2)
end

load(i) = x -> -μ * sum(hess(uᵉ[i], x)[a, a] for a in 1:2) -
               (λ + μ) * sum(hess(uᵉ[j], x)[i, j] for j in 1:2)
loadform(v) = innerₕ(load(1), v(1)) + innerₕ(load(2), v(2))

# ## The boundary relations
#
# The traction of component ``i`` on a side is ``\sum_j \partial_n(u_j; K_{ij})``, with the
# conormal tensor ``K_{ij}[d,e] = \lambda\,\delta_{id}\delta_{je} +
# \mu\,(\delta_{ij}\delta_{de} + \delta_{ie}\delta_{jd})``. A displacement side fixes the
# value and, to close its ghost, the second normal derivative. Both come from the
# manufactured solution. On `xmin` and `ymin` the two components' relations meet at the
# corner, where `constrained` solves the four ghost values of the corner's block together.

K(i, j) = [λ * (i == d) * (j == e) + μ * ((i == j) * (d == e) + (i == e) * (j == d))
           for d in 1:2, e in 1:2]
traction(u, i) = ∂ₙ(u(1); K = K(i, 1)) + ∂ₙ(u(2); K = K(i, 2))

const AXIS = (xmin = 1, xmax = 1, ymin = 2, ymax = 2)
const OUTWARD = (xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0)
normal(F) = ntuple(d -> d == AXIS[F] ? OUTWARD[F] : 0.0, 2)

# The traction datum of component `i` on side `F`, ``n \cdot \sigma(u)\, e_i``:

tractiondata(F, i) = x -> sum(normal(F)[d] * K(i, j)[d, e] * ∇(uᵉ[j], x)[e]
for j in 1:2, d in 1:2, e in 1:2)

function relations(u)
    (
        :xmin => traction(u, 1) => tractiondata(:xmin, 1),
        :xmin => traction(u, 2) => tractiondata(:xmin, 2),
        :ymin => traction(u, 1) => x -> 0.0,
        :ymin => traction(u, 2) => x -> 0.0,
        (F => u(i) => uᵉ[i] for F in (:xmax, :ymax) for i in 1:2)...,
        (F => ∂ₙₙ(u(i)) => (x -> hess(uᵉ[i], x)[AXIS[F], AXIS[F]])
        for F in (:xmax, :ymax) for i in 1:2)...)
end

# The traction-free side carries zero data, and the manufactured one does too, to rounding:

(maximum(abs, tractiondata(:ymin, i)((x, 0.0)) for x in 0:0.1:1 for i in 1:2),
    abs(tractiondata(:xmin, 1)((0.0, 0.5))))

# ## Assembling and solving
#
# Mixed differences in the left argument of `innerₕ` read the ghosts of a traction side, and
# the library replaces them by the extrapolation of the nodes inward along the normal. That
# is what keeps the collocated form stable on cells longer across a side than along it.
# `pde_solve` takes the reduced system, and `Bramble.lift` returns the solution with its
# ghost values, component by component.

function solve_on(Ωₕ)
    V = gridspace(Ωₕ; ghosts = :boundary)^Val(2)
    Cₕ = constrained(V, relations)
    A, F = assemble(form(Cₕ, Cₕ, elasticity), form(Cₕ, loadform))
    return Cₕ, Bramble.lift(Cₕ, pde_solve(A, F))
end

# ## Does the solution satisfy the traction?
#
# The relation is checked on the lifted values, with forms that share nothing with the
# closure: per component of the traction, one `inner_Γ` form on the side, summed over the
# components of the displacement, against the weighted datum. Away from a corner these are
# the centred relations the closure imposed. At the corner of the two tractions the
# closure takes the tangential derivative one-sided, so the relation there is written out
# by hand from the mesh points.

paddedvalues(U) = parent(U)
function relation_residual(Ωₕ, U, F, i)
    m = prod(Bramble.size(Ωₕ) .+ 2)
    V = gridspace(Ωₕ; ghosts = :boundary)^Val(2)
    P = Bramble.padded_space(V)
    R = assemble(form(P, P,
        (w, v) -> inner_Γ(∂ₙ(w(1); K = K(i, 1)), v(i); markers = (F,)) +
                  inner_Γ(∂ₙ(w(2); K = K(i, 2)), v(i); markers = (F,))))
    data = assemble(form(P, v -> inner_Γ(tractiondata(F, i), v(i); markers = (F,))))
    flux = R * paddedvalues(U)
    block = ((i - 1) * m + 1):(i * m)
    return (flux - data)[block], flux[block]
end

# The weighted residual of the relation of component `i` on side `F`, on every padded row
# but the corner of the two tractions, against the size of the flux:

function facet_residual(Ωₕ, U, F, i)
    r, flux = relation_residual(Ωₕ, U, F, i)
    n = Bramble.size(Ωₕ)
    keep = trues(length(r))
    P = LinearIndices(n .+ 2)
    I = CartesianIndex(1, 1)
    keep[P[I + oneunit(I)]] = false
    return maximum(abs, r[keep]) / max(1.0, maximum(abs, flux[keep]))
end

Ωchk = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9), (true, true))
Cchk, Uchk = solve_on(Ωchk)
resid = (facet_residual(Ωchk, Uchk, :xmin, 1), facet_residual(Ωchk, Uchk, :xmin, 2))

# The traction-free side `ymin` holds the same way, with zero data:

resid_free = (facet_residual(Ωchk, Uchk, :ymin, 1), facet_residual(Ωchk, Uchk, :ymin, 2))

# Both are rounding, not discretisation error: the relation is an identity of the lifted
# values, whatever the accuracy of the solution.
@test all(<(1.0e-12), resid)                                                                #src
@test all(<(1.0e-12), resid_free)                                                           #src
# The loaded side's flux is far from zero, so the relation is not met by a zero solution.   #src
@test maximum(abs, relation_residual(Ωchk, Uchk, :xmin, 1)[2]) > 0.1                        #src

# ### The corner of the two tractions
#
# At the corner node the ghost across `xmin` reads the tangential derivative along the
# side. The closure takes it one-sided: the derivative at the node of the quadratic through
# the three nodes along the side into the domain, with the normal derivative centred over
# the ghost. This writes that relation from the mesh points and the padded values.

function corner_relation(xs, U, i, d, g)
    e = 3 - d
    n = length.(xs)
    P = LinearIndices(n .+ 2)
    val(j, I) = parent(U)[(j - 1) * prod(n .+ 2) + P[I + oneunit(I)]]
    ud = CartesianIndex(ntuple(a -> a == d ? 1 : 0, 2))
    uex = CartesianIndex(ntuple(a -> a == e ? 1 : 0, 2))
    I = CartesianIndex(1, 1)
    h = xs[d][2] - xs[d][1]
    y = (xs[e][1], xs[e][2], xs[e][3])
    w = (1 / (y[1] - y[2]) + 1 / (y[1] - y[3]),
        (y[1] - y[3]) / ((y[2] - y[1]) * (y[2] - y[3])),
        (y[1] - y[2]) / ((y[3] - y[1]) * (y[3] - y[2])))
    terms = Float64[]
    for j in 1:2
        ∂d = (val(j, I + ud) - val(j, I - ud)) / (2h)
        ∂e = sum(w[k] * val(j, I + (k - 1) * uex) for k in 1:3)
        push!(terms, -K(i, j)[d, d] * ∂d, -K(i, j)[d, e] * ∂e)
    end
    return (sum(terms) - g) / max(1.0, maximum(abs, terms), abs(g))
end

xschk = points(Ωchk)
corner = [corner_relation(xschk, Uchk, i, d, g)
          for i in 1:2 for (d, g) in ((1, tractiondata(:xmin, i)((0.0, 0.0))), (2, 0.0))]
maximum(abs, corner)

# Rounding again, at the corner too:
@test maximum(abs, corner) < 1.0e-10                                                        #src

# ## Second order on graded grids
#
# Three graded grids of 9, 17 and 33 points per axis: the points of a uniform grid pushed
# through a smooth increasing map, installed with [`change_points!`](@ref), the cells
# differing in size by a factor of about six. The error is the discrete ``H^1`` norm of
# each component at the physical nodes. A random grid is left out on purpose: the
# collocated form is stable on it but reaches second order only on a smooth grading.

grade(ξ) = expm1(2ξ) / expm1(2.0)

function graded(n)
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n), (true, true))
    change_points!(Ωₕ, (grade.(points(Ωₕ)[1]), 1 .- grade.(1 .- points(Ωₕ)[2])))
    return Ωₕ
end

function component_errors(Ωₕ)
    _, U = solve_on(Ωₕ)
    n = Bramble.size(Ωₕ)
    m = prod(n .+ 2)
    W = gridspace(Ωₕ)
    return ntuple(2) do i
        uₕ = element(W)
        uₕ .= vec(reshape(parent(U)[((i - 1) * m + 1):(i * m)], n .+ 2)[
            (2:(n[d] + 1) for d in 1:2)...])
        norm₁ₕ(uₕ .- Rₕ(W, uᵉ[i]))
    end
end

levels = [graded(n) for n in (9, 17, 33)]
hs = [hₘₐₓ(Ωₕ) for Ωₕ in levels]
errs = [component_errors(Ωₕ) for Ωₕ in levels]
err1, err2 = getindex.(errs, 1), getindex.(errs, 2)

# The order is the least-squares slope of ``\log e`` against ``\log h_{\max}`` over the three
# levels:

function lsq_order(hs, es)
    X, Y = log.(hs), log.(es)
    return sum((X .- sum(X) / length(X)) .* (Y .- sum(Y) / length(Y))) /
           sum(abs2, X .- sum(X) / length(X))
end

order1, order2 = lsq_order(hs, err1), lsq_order(hs, err2)
(order1, order2)

#-

order1 > 1.8 && order2 > 1.8

# The line above renders its verdict; these assert it, bracketed so that an error at       #src
# roundoff would fail rather than read as a better order.                                  #src
@test 1.8 < order1 < 3.0                                                                   #src
@test 1.8 < order2 < 3.0                                                                   #src
@test all(e -> 1.0e-6 < e < 1.0, err1) && all(e -> 1.0e-6 < e < 1.0, err2)                 #src
# The grading is real, not a uniform grid by another name.                                 #src
@test maximum(diff(points(levels[1])[1])) > 3 * minimum(diff(points(levels[1])[1]))        #src

# The error against ``h_{\max}`` on log-log axes, one curve per component. The dashed line
# has slope 2 and is anchored at the finest point of the first curve.

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot(
    [(hs, err1, "u₁", "#5B5FC7"), (hs, err2, "u₂", "#0E7C86")];
    title = "Elasticity with a traction, ‖·‖₁ₕ") # hide

# Both components run parallel to the slope-2 line, with a loaded traction side, a
# traction-free one and two displacement sides, and with the two tractions meeting at a
# corner.
#
# ## Where to go next
#
#   - [Neumann, Robin and pure Neumann boundaries](neumann_robin_poisson.md) for the same
#     ghost mechanism on a scalar problem.
