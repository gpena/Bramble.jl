# Plain-matrix 1D oracle for ghost-point Neumann and Robin (#218, #205). It is loaded by
# oracle_1d.jl, never by the package.
#
# Every operator is an explicit sparse matrix on the extended index range `0 … n+1`
# (paper indexing: physical nodes `1 … n`, ghosts `0` and `n+1`), built from the mesh's
# points alone. Nothing here calls the form layer or the runtime operators: those truncate
# at the boundary, and that truncation is what the oracle judges, so it cannot share it.
#
# Conventions, in paper indexing stored 1-based at `k = i + 1` (see `ext`).
#   h_i = x_i − x_{i−1} (2 ≤ i ≤ n), h_1 = h_2, h_0 = h_1, h_{n+1} = h_n.
#   Ghosts x_0 = x_1 − h_2, x_{n+1} = x_n + h_n; outward normals η_1 = −1, η_n = +1.
#   Cell measures ĥ_i = (h_i + h_{i+1})/2 inside, h_2/2 and h_n/2 at the ends.
#
# Systems are row-scaled as a variational assembly would be: a PDE row times ĥ_i, a
# boundary-condition row times its facet weight (1 in 1D). DOF order of an augmented
# (`_b`) system: physical nodes in mesh order, then the xmin ghost, then the xmax ghost.
module GhostOracle1D

using LinearAlgebra: Diagonal, dot
using SparseArrays: SparseMatrixCSC, sparse, spzeros
import Bramble
using Bramble: change_points!, domain, interval, iterative_refinement!, mesh, points

export Dirichlet, Neumann, Robin, augmented_system, cell_measures, discrete_error,
       eliminated_system, extended_points, extended_spacings, ghost_closure, hmax,
       make_mesh, mean_weights, observed_errors, observed_order, operators,
       relation_row, solve, system, system_matrix, variational_system

const PROBLEMS = (:neumann_a, :neumann_b, :robin_a, :robin_b)
const MESH_KINDS = (:uniform, :random, :graded)

# Paper index `i ∈ 0:n+1` to the 1-based storage index.
@inline ext(i::Int) = i + 1

#------------------------------------------------------------------------------------------#
# Geometry
#------------------------------------------------------------------------------------------#

physical_points(Ωₕ) = collect(Float64, points(Ωₕ))

"""
    extended_points(Ωₕ) -> Vector{Float64}

`[x_0, x_1, …, x_n, x_{n+1}]` with `x_0 = x_1 − h_2` and `x_{n+1} = x_n + h_n`.
"""
function extended_points(Ωₕ)
    x = physical_points(Ωₕ)
    n = length(x)
    return vcat(x[1] - (x[2] - x[1]), x, x[n] + (x[n] - x[n - 1]))
end

"""
    extended_spacings(Ωₕ) -> Vector{Float64}

`[h_0, h_1, …, h_{n+1}]` with `h_0 = h_1 = h_2` and `h_{n+1} = h_n`.
"""
function extended_spacings(Ωₕ)
    xp = extended_points(Ωₕ)
    hp = similar(xp)
    hp[2:end] .= diff(xp)
    hp[1] = hp[2]
    return hp
end

"""
    cell_measures(Ωₕ) -> Vector{Float64}

The cell measures `ĥ_1 … ĥ_n` (boundary half cells at the ends): the `(·,·)_{H,*}` weights.
"""
function cell_measures(Ωₕ)
    x = physical_points(Ωₕ)
    n = length(x)
    h = diff(x)                       # h[i - 1] = h_i, 2 ≤ i ≤ n
    ĥ = zeros(n)
    ĥ[1] = h[1] / 2
    ĥ[n] = h[n - 1] / 2
    for i in 2:(n - 1)
        ĥ[i] = (h[i - 1] + h[i]) / 2
    end
    return ĥ
end

"""
    mean_weights(Ωₕ) -> Vector{Float64}

The weights `w` of the discrete mean `Σ ĥ_i u_i` on the physical nodes (length `n`). For an
augmented system, pad with zeros at the two ghost DOFs.
"""
mean_weights(Ωₕ) = cell_measures(Ωₕ)

"""
    hmax(Ωₕ) -> Float64

The largest spacing.
"""
hmax(Ωₕ) = maximum(diff(physical_points(Ωₕ)))

normal(i::Int, n::Int) = i == 1 ? -1.0 : (i == n ? 1.0 : 0.0)

#------------------------------------------------------------------------------------------#
# Operators on V_H* (notes §1.1)
#------------------------------------------------------------------------------------------#

"""
    operators(Ωₕ) -> NamedTuple

Sparse `(n+2) × (n+2)` matrices on the extended range, rows and columns at `ext(i)`; a row
outside an operator's range of definition is empty.

  - `Dminus`: `D₋u_i = (u_i − u_{i−1})/h_i`, rows `1 … n+1`
  - `Dstar`: `D̃u_i = 2(u_{i+1} − u_i)/(h_i + h_{i+1})`, rows `1 … n`
  - `Dc`: `Dc u_i = (u_{i+1} − u_{i−1})/(h_i + h_{i+1})`, rows `1 … n`
  - `Lap`: `Δ_H = D̃ D₋`, rows `1 … n`
  - `Mminus`: `M₋u_i = (u_i + u_{i−1})/2`, rows `1 … n+1`
  - `Jump`: `[[u]]_i = u_{i+1} − u_i`, rows `0 … n`
"""
function operators(Ωₕ)
    hp = extended_spacings(Ωₕ)
    N = length(hp)
    n = N - 2
    Dminus = spzeros(N, N)
    Dstar = spzeros(N, N)
    Dc = spzeros(N, N)
    Mminus = spzeros(N, N)
    Jump = spzeros(N, N)
    h(i) = hp[ext(i)]
    for i in 1:(n + 1)
        Dminus[ext(i), ext(i)] = 1 / h(i)
        Dminus[ext(i), ext(i - 1)] = -1 / h(i)
        Mminus[ext(i), ext(i)] = 0.5
        Mminus[ext(i), ext(i - 1)] = 0.5
    end
    for i in 1:n
        s = h(i) + h(i + 1)
        Dstar[ext(i), ext(i + 1)] = 2 / s
        Dstar[ext(i), ext(i)] = -2 / s
        Dc[ext(i), ext(i + 1)] = 1 / s
        Dc[ext(i), ext(i - 1)] = -1 / s
    end
    for i in 0:n
        Jump[ext(i), ext(i + 1)] = 1.0
        Jump[ext(i), ext(i)] = -1.0
    end
    return (; Dminus, Dstar, Dc, Lap = Dstar * Dminus, Mminus, Jump)
end

#------------------------------------------------------------------------------------------#
# Boundary relations and the ghost closure
#------------------------------------------------------------------------------------------#

"""
    Robin(α, β, g)

`α u + β η u' = g` at a boundary node, discretised as `α u_i + β η_i Dc u_i = g_i`.
"""
struct Robin
    α::Float64
    β::Float64
    g::Float64
end

"""
    Neumann(β, g)

`β η u' = g`, that is `Robin(0, β, g)`.
"""
Neumann(β, g) = Robin(0.0, β, g)

"""
    Dirichlet(g)

`u = g` imposed strongly: the node leaves the free DOFs and its ghost is never read.
"""
struct Dirichlet
    g::Float64
end

node(side::Symbol, n::Int) = side === :xmin ? 1 : side === :xmax ? n :
                                                  throw(ArgumentError("unknown facet $side"))
ghost(side::Symbol, n::Int) = side === :xmin ? 0 : n + 1

"""
    relation_row(rel::Robin, Ωₕ, side::Symbol) -> (r::Vector{Float64}, g::Float64)

The relation `rᵀ u_ext = g` on the extended range at the node of facet `side`
(`:xmin` or `:xmax`): `α e_i + β η_i Dc[i, :]`.
"""
function relation_row(rel::Robin, Ωₕ, side::Symbol)
    Dc = operators(Ωₕ).Dc
    N = size(Dc, 1)
    n = N - 2
    i = node(side, n)
    r = rel.β * normal(i, n) * Vector(Dc[ext(i), :])
    r[ext(i)] += rel.α
    return r, rel.g
end

function free_nodes(relations, n)
    [i for i in 1:n
     if !(i == 1 && relations.xmin isa Dirichlet) &&
        !(i == n && relations.xmax isa Dirichlet)]
end

"""
    ghost_closure(relations, Ωₕ) -> (E::SparseMatrixCSC{Float64, Int}, d::Vector{Float64})

`u_ext = E u_free + d` on the extended range, `u_free` the free physical nodes in mesh order.
`relations = (xmin = rel, xmax = rel)` with `Robin`, `Neumann` or `Dirichlet` entries. A
Robin or Neumann facet solves its relation for the ghost; a Dirichlet node takes `d = g`
and its ghost row stays empty (no free row reads it).
"""
function ghost_closure(relations, Ωₕ)
    n = length(physical_points(Ωₕ))
    N = n + 2
    free = free_nodes(relations, n)
    E = zeros(N, length(free))
    d = zeros(N)
    for (c, i) in enumerate(free)
        E[ext(i), c] = 1.0
    end
    for side in (:xmin, :xmax)
        rel = getfield(relations, side)
        rel isa Dirichlet && (d[ext(node(side, n))] = rel.g)
    end
    for side in (:xmin, :xmax)
        rel = getfield(relations, side)
        rel isa Dirichlet && continue
        r, g = relation_row(rel, Ωₕ, side)
        kg = ext(ghost(side, n))
        cg = r[kg]
        r[kg] = 0.0
        E[kg, :] .= -(E' * r) ./ cg
        d[kg] = (g - dot(r, d)) / cg
    end
    return sparse(E), d
end

#------------------------------------------------------------------------------------------#
# Systems for −u'' = f
#------------------------------------------------------------------------------------------#

"""
    eliminated_system(relations, Ωₕ, f) -> (A, F)

Ghosts eliminated: `ĥ_i (−Δ_H (E u + d))_i = ĥ_i f(x_i)` on the free nodes, so
`A = diag(ĥ) (−Δ_H) E` and `F = ĥ ⊙ (f + Δ_H d)`.
"""
function eliminated_system(relations, Ωₕ, f)
    x = physical_points(Ωₕ)
    n = length(x)
    free = free_nodes(relations, n)
    E, d = ghost_closure(relations, Ωₕ)
    L = operators(Ωₕ).Lap[ext.(free), :]
    W = Diagonal(cell_measures(Ωₕ)[free])
    A = sparse(W * (-L * E))
    F = W * (f.(x[free]) .+ L * d)
    return A, F
end

"""
    variational_system(relations, Ωₕ, f) -> (A, F)

`(D₋u, D₋v)_+ + Σ_{Robin ends} (α/β) u_i v_i = (f, v)_{H,*} + Σ_{Robin ends} (g_i/β) v_i`,
Dirichlet nodes lifted and removed (notes §1.2–§1.4). Reads no ghost.
"""
function variational_system(relations, Ωₕ, f)
    x = physical_points(Ωₕ)
    n = length(x)
    Dm = operators(Ωₕ).Dminus[:, ext.(1:n)]
    hp = extended_spacings(Ωₕ)
    wplus = [2 <= i <= n ? hp[ext(i)] : 0.0 for i in 0:(n + 1)]
    A = Matrix(Dm' * Diagonal(wplus) * Dm)
    F = cell_measures(Ωₕ) .* f.(x)
    ud = zeros(n)
    for side in (:xmin, :xmax)
        rel = getfield(relations, side)
        i = node(side, n)
        if rel isa Dirichlet
            ud[i] = rel.g
        else
            A[i, i] += rel.α / rel.β
            F[i] += rel.g / rel.β
        end
    end
    free = free_nodes(relations, n)
    return sparse(A[free, free]), F[free] .- A[free, :] * ud
end

"""
    augmented_system(relations, Ωₕ, f) -> (A, F)

Ghosts as unknowns (`n + 2` DOFs: physical nodes, then the xmin and xmax ghosts): PDE rows
`ĥ_i (−Δ_H u)_i = ĥ_i f(x_i)` on every physical node, and one relation row per ghost.
Robin and Neumann facets only.
"""
function augmented_system(relations, Ωₕ, f)
    x = physical_points(Ωₕ)
    n = length(x)
    any(r -> r isa Dirichlet, values(relations)) &&
        throw(ArgumentError("augmented_system takes Robin and Neumann facets only"))
    perm = vcat(ext.(1:n), ext(0), ext(n + 1))   # DOF k ↦ extended index
    L = operators(Ωₕ).Lap
    ĥ = cell_measures(Ωₕ)
    rL, gL = relation_row(relations.xmin, Ωₕ, :xmin)
    rR, gR = relation_row(relations.xmax, Ωₕ, :xmax)
    A = vcat(Diagonal(ĥ) * (-L[ext.(1:n), perm]), sparse(rL[perm]'), sparse(rR[perm]'))
    F = vcat(ĥ .* f.(x), gL, gR)
    return sparse(A), F
end

#------------------------------------------------------------------------------------------#
# The four problems
#------------------------------------------------------------------------------------------#

# u = eˣ cos 2x on (0, 1): no symmetry, non-zero data at both ends.
const DEFAULT_EXACT = (u = x -> exp(x) * cos(2x),
    du = x -> exp(x) * (cos(2x) - 2sin(2x)),
    d2u = x -> exp(x) * (-3cos(2x) - 4sin(2x)))

function check_problem(problem::Symbol)
    problem in PROBLEMS || throw(ArgumentError("unknown problem $problem; one of $PROBLEMS"))
    return nothing
end

is_neumann(problem::Symbol) = problem === :neumann_a || problem === :neumann_b
is_augmented(problem::Symbol) = problem === :neumann_b || problem === :robin_b

"""
    problem_relations(problem, Ωₕ; α = (2.0, 0.5), β = (1.5, 3.0), exact) -> NamedTuple

The `(xmin, xmax)` relations of `problem`, data from the exact solution. `α` is ignored for
the Neumann problems.
"""
function problem_relations(problem::Symbol, Ωₕ; α = (2.0, 0.5), β = (1.5, 3.0),
        exact = DEFAULT_EXACT)
    check_problem(problem)
    x = physical_points(Ωₕ)
    n = length(x)
    a = is_neumann(problem) ? (0.0, 0.0) : α
    rel(j, i) = Robin(a[j], β[j], a[j] * exact.u(x[i]) + β[j] * normal(i, n) * exact.du(x[i]))
    return (xmin = rel(1, 1), xmax = rel(2, n))
end

"""
    system(problem::Symbol, Ωₕ; α, β, exact) -> (A::SparseMatrixCSC{Float64, Int}, F)

The `_a` problems give `eliminated_system`, the `_b` ones `augmented_system`, for
`−u'' = f` with `f = −exact.d2u`.
"""
function system(problem::Symbol, Ωₕ; exact = DEFAULT_EXACT, kwargs...)
    relations = problem_relations(problem, Ωₕ; exact, kwargs...)
    f = x -> -exact.d2u(x)
    build = is_augmented(problem) ? augmented_system : eliminated_system
    return build(relations, Ωₕ, f)
end

"""
    system_matrix(problem::Symbol, Ωₕ; kwargs...) -> SparseMatrixCSC{Float64, Int}

The matrix of `system`. A pure-Neumann matrix is the singular operator.
"""
system_matrix(problem::Symbol, Ωₕ; kwargs...) = first(system(problem, Ωₕ; kwargs...))

"""
    solve(problem::Symbol, Ωₕ; kwargs...) -> Vector{Float64}

The discrete solution: physical nodes for `_a`, all `n + 2` DOFs for `_b`. The Neumann
problems add `Σ ĥ_i u_i = 0` through a bordered system (one Lagrange multiplier).
"""
function solve(problem::Symbol, Ωₕ; kwargs...)
    A, F = system(problem, Ωₕ; kwargs...)
    m = size(A, 1)
    is_neumann(problem) || return Matrix(A) \ F
    w = zeros(m)
    w[1:length(mean_weights(Ωₕ))] .= mean_weights(Ωₕ)
    B = [Matrix(A) w; w' 0.0]
    return (B \ vcat(F, 0.0))[1:m]
end

"""
    discrete_error(problem::Symbol, Ωₕ; exact, kwargs...) -> Float64

`(‖D₋e‖_+² + ‖e‖_{H,*}²)^{1/2}` with `e = u_h − R_H u` on the physical nodes; for Neumann,
`R_H u` shifted to discrete mean zero.
"""
function discrete_error(problem::Symbol, Ωₕ; exact = DEFAULT_EXACT, kwargs...)
    x = physical_points(Ωₕ)
    n = length(x)
    uh = solve(problem, Ωₕ; exact, kwargs...)[1:n]
    ue = exact.u.(x)
    ĥ = cell_measures(Ωₕ)
    is_neumann(problem) && (ue .-= dot(ĥ, ue) / sum(ĥ))
    e = uh .- ue
    h = diff(x)
    return sqrt(sum(abs2.(diff(e)) ./ h) + dot(ĥ, abs2.(e)))
end

#------------------------------------------------------------------------------------------#
# Mesh sequences and observed orders
#------------------------------------------------------------------------------------------#

# A smooth, increasing map of [0, 1] onto itself, finer at the left end: x = φ(ξ) on
# uniform ξ, rebuilt per level, so max|h_i − h_{i−1}| = O(hₘₐₓ²) (notes §1.6).
graded_map(ξ) = expm1(2ξ) / expm1(2.0)

"""
    make_mesh(kind::Symbol, level::Int; n0 = 17, seed = 20261014) -> Mesh1D

Level `level` of a sequence on `[0, 1]` starting with `n0` points: `:uniform` and `:random`
(Bramble's random mesh drawn from `seed`) halve every cell `level` times through
`iterative_refinement!`; `:graded` is `graded_map` of `(n0 − 1)·2^level + 1` uniform points.
"""
function make_mesh(kind::Symbol, level::Int; n0::Int = 17, seed::Int = 20261014)
    Ω = domain(interval(0.0, 1.0))
    if kind === :uniform || kind === :random
        if kind === :random
            Bramble._seed_mesh1d_rng!(seed)
            Ωₕ = try
                mesh(Ω, n0, false)
            finally
                Bramble._unseed_mesh1d_rng!()
            end
        else
            Ωₕ = mesh(Ω, n0, true)
        end
        for _ in 1:level
            iterative_refinement!(Ωₕ)
        end
        return Ωₕ
    elseif kind === :graded
        Ωₕ = mesh(Ω, (n0 - 1) * 2^level + 1, true)
        change_points!(Ωₕ, graded_map.(physical_points(Ωₕ)))
        return Ωₕ
    end
    throw(ArgumentError("unknown mesh kind $kind; one of $MESH_KINDS"))
end

"""
    observed_errors(problem, meshkind; dim = 1, levels = 4, n0 = 17, seed, kwargs...)
        -> (hs::Vector{Float64}, errs::Vector{Float64})

`hₘₐₓ` and `discrete_error` over `levels` meshes of `make_mesh(meshkind, ℓ)`.
"""
function observed_errors(problem::Symbol, meshkind::Symbol; dim::Int = 1, levels::Int = 4,
        n0::Int = 17, seed::Int = 20261014, kwargs...)
    dim == 1 || throw(ArgumentError("GhostOracle1D is 1D only (dim = $dim)"))
    check_problem(problem)
    hs = Float64[]
    errs = Float64[]
    for ℓ in 0:(levels - 1)
        Ωₕ = make_mesh(meshkind, ℓ; n0, seed)
        push!(hs, hmax(Ωₕ))
        push!(errs, discrete_error(problem, Ωₕ; kwargs...))
    end
    return hs, errs
end

"""
    observed_order(problem::Symbol, meshkind::Symbol; dim = 1, levels = 4, kwargs...) -> Float64

The least-squares slope of `log(error)` against `log(hₘₐₓ)` over `levels` meshes.
"""
function observed_order(problem::Symbol, meshkind::Symbol; kwargs...)
    hs, errs = observed_errors(problem, meshkind; kwargs...)
    X = log.(hs)
    Y = log.(errs)
    X̄, Ȳ = sum(X) / length(X), sum(Y) / length(Y)
    return sum((X .- X̄) .* (Y .- Ȳ)) / sum(abs2, X .- X̄)
end

end # module GhostOracle1D
