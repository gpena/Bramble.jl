# Linear elasticity with traction and displacement facets through ghost points (#793): the
# manufactured displacements, the collocated form, the relations and the H¹ errors that
# test/ghost/traction.jl checks the reduced system with and test/convergence/ghost_traction.jl
# refines. Loaded into `Main` by both, never by the package.
module GhostTraction

using Bramble
import Bramble
using Bramble: Dcₓ, Dcᵧ, Dc₂, D̃ₓ, D̃ᵧ, D̃₂, D₋ₓ, D₋ᵧ, D₋₂
using ForwardDiff: gradient, hessian

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(@__DIR__, "GhostOracleND.jl"))
const O = Main.GhostOracleND

const λ, μ = 2.0, 0.5

Kt(i, j, D) = [λ * (i == d) * (j == e) + μ * ((i == j) * (d == e) + (i == e) * (j == d))
               for d in 1:D, e in 1:D]

# --- The manufactured displacement and its data ---------------------------------------- #

# In 2D the traction on `:ymin` (outward normal −e₂) vanishes: `∂ᵧu₁ = ∂ₓu₂ = 0` there and
# `λ ∂ₓu₁ + (λ + 2μ) ∂ᵧu₂ = 0` through the factor `c = λ / (λ + 2μ)`.
const c = λ / (λ + 2μ)
const UEX2 = (x -> sin(x[1] + 0.3) * cos(x[2]) + exp(x[1]) * x[2]^2 / 2,
    x -> -c * cos(x[1] + 0.3) * sin(x[2]) + cos(2x[1]) * x[2]^2)
const UEX3 = (x -> sin(x[1] + 0.3) * cos(x[2]) * exp(x[3] / 2),
    x -> cos(x[1]) * x[2] * x[3] + exp(x[1] + x[2] / 2),
    x -> sin(x[1] + x[2]) * cos(x[3] + 0.2))

∇(f, x) = gradient(f, collect(Float64, x))
H(f, x) = hessian(f, collect(Float64, x))

# The traction of component i of `uex` on the facet with outward normal `n`.
function traction_data(uex, i, n)
    D = length(uex)
    return x -> sum(n[d] * Kt(i, j, D)[d, e] * ∇(uex[j], x)[e]
    for j in 1:D, d in 1:D, e in 1:D)
end

# The load `−μ Δuᵢ − (λ + μ) ∂ᵢ div u`.
function load(uex, i)
    D = length(uex)
    return x -> -μ * sum(H(uex[i], x)[a, a] for a in 1:D) -
                (λ + μ) * sum(H(uex[j], x)[i, j] for j in 1:D)
end

const AXIS = (xmin = 1, xmax = 1, ymin = 2, ymax = 2, zmin = 3, zmax = 3)
const SIDE = (xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0, zmin = -1.0, zmax = 1.0)
normal(F, D) = ntuple(d -> d == AXIS[F] ? SIDE[F] : 0.0, D)

# --- The form and the relations --------------------------------------------------------- #

const DC = (Dcₓ, Dcᵧ, Dc₂)
const DT = (D̃ₓ, D̃ᵧ, D̃₂)
const DM = (D₋ₓ, D₋ᵧ, D₋₂)

# `∂ₐ∂_b u`: the second difference of axis `a` when `a = b`, else the centred mixed one.
second(u, a, b) = a == b ? DT[a](DM[a](u)) : DC[min(a, b)](DC[max(a, b)](u))

# The form without its mixed differences, and those alone (the facet rule's operand).
function rest_part(D)
    row(u, v, i) = innerₕ(-μ * Δₕ(u(i)), v(i)) + innerₕ(-(λ + μ) * second(u(i), i, i), v(i))
    return (u, v) -> sum(row(u, v, i) for i in 1:D)
end
function mixed_part(D)
    term(u, v, i, j) = innerₕ(-(λ + μ) * second(u(j), i, j), v(i))
    return (u, v) -> sum(term(u, v, i, j) for i in 1:D, j in 1:D if i != j)
end
function elasticity(D)
    rest, mixed = rest_part(D), mixed_part(D)
    return (u, v) -> rest(u, v) + mixed(u, v)
end

load_form(uex) = v -> sum(innerₕ(load(uex, i), v(i)) for i in eachindex(uex))

tr(u, i, D) = sum(∂ₙ(u(j); K = Kt(i, j, D)) for j in 1:D)

# Traction on `tfacets`, the data of `uex` (or zero on `free`), displacement on `dfacets`.
function relations(uex, tfacets, dfacets; free = ())
    D = length(uex)
    tdata(F, i) = F in free ? 0.0 : traction_data(uex, i, normal(F, D))
    ∂ₙₙdata(F, i) = x -> H(uex[i], x)[AXIS[F], AXIS[F]]
    traction(u, F) = ntuple(i -> F => tr(u, i, D) => tdata(F, i), D)
    displacement(u, F) = (ntuple(i -> F => u(i) => uex[i], D)...,
        ntuple(i -> F => ∂ₙₙ(u(i)) => ∂ₙₙdata(F, i), D)...)
    return u -> (Iterators.flatten(traction(u, F) for F in tfacets)...,
        Iterators.flatten(displacement(u, F) for F in dfacets)...)
end

const CASE2 = (tfacets = (:xmin, :ymin), dfacets = (:xmax, :ymax), free = (:ymin,))
const ALL_DISPLACEMENT = (tfacets = (), dfacets = (:xmin, :xmax, :ymin, :ymax), free = ())

function elasticity_space(Ωₕ, uex, case)
    S = gridspace(Ωₕ; ghosts = :boundary)
    V = S^Val(length(uex))
    return V, constrained(V, relations(uex, case.tfacets, case.dfacets; free = case.free))
end

# --- Errors ----------------------------------------------------------------------------- #

# The physical values of component `i` of the padded values `U`, every facet ghosted.
function physical(U, n, i)
    np = n .+ 2
    m = prod(np)
    P = reshape(U[((i - 1) * m + 1):(i * m)], np)
    return P[ntuple(d -> 2:(np[d] - 1), length(n))...]
end

# The discrete H¹ error `(‖e‖_h² + Σₐ ‖D₋ₐ e‖₊²)^{1/2}` at the physical nodes.
function h1_error(xs, e)
    D = length(xs)
    ĥ(x) = [(i > 1 ? x[i] - x[i - 1] : 0.0) / 2 +
            (i < length(x) ? x[i + 1] - x[i] : 0.0) / 2 for i in eachindex(x)]
    along(v, a) = reshape(v, ntuple(d -> d == a ? length(v) : 1, D))
    w = reduce((p, q) -> p .* q, (along(ĥ(xs[a]), a) for a in 1:D))
    total = sum(w .* abs2.(e))
    for a in 1:D
        wa = reduce((p, q) -> p .* q,
            (b == a ? along(1 ./ diff(xs[a]), a) : along(ĥ(xs[b]), b) for b in 1:D))
        total += sum(wa .* abs2.(diff(e; dims = a)))
    end
    return sqrt(total)
end

# `hₘₐₓ` and the H¹ error of each component on `make_mesh(kind, ℓ)`, ℓ in `levels`.
function errors(uex, case, kind, levels; n0, seed = 20261014)
    D = length(uex)
    hs, errs = Float64[], NTuple{D, Float64}[]
    for ℓ in levels
        Ωₕ = O.make_mesh(kind, ℓ; dim = D, n0, seed)
        _, C = elasticity_space(Ωₕ, uex, case)
        A, F = assemble(form(C, C, elasticity(D)), form(C, load_form(uex)))
        U = parent(Bramble.lift(C, A \ F))
        xs = O.axis_points(Ωₕ)
        n = map(length, xs)
        nodes = collect(Iterators.product(xs...))
        push!(hs, maximum(x -> maximum(diff(x)), xs))
        push!(errs, ntuple(i -> h1_error(xs, physical(U, n, i) - uex[i].(nodes)), D))
    end
    return hs, errs
end

function lsq_order(hs, errs)
    X, Y = log.(hs), log.(errs)
    return sum((X .- sum(X) / length(X)) .* (Y .- sum(Y) / length(Y))) /
           sum(abs2, X .- sum(X) / length(X))
end

end # module GhostTraction
