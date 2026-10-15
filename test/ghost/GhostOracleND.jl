# Plain-matrix 2D/3D oracle for ghost-point Neumann and Robin with facet-only ghosts
# (#218, #205). It is loaded by oracle_nd.jl, never by the package.
#
# Every operator is a Kronecker product of GhostOracle1D's verified 1D factors on the
# padded grid (one extra point at each end of every axis), then restricted to the physical
# nodes plus the facet ghosts (notes §2). The padded points outside along two or more axes
# (corners, and edges in 3D) are not ghosts: no row kept here reads them, which the test
# file pins. Nothing calls the form layer or the runtime operators.
#
# Conventions (plan, "Conventions"; notes §2).
#   Physical node I sits at padded index I .+ 1, GhostOracle1D's extended index `ext` on each axis.
#   Facet F = (d, side) has as nodes the physical I with I_d = 1 (min) or n_d (max),
#   edges and corners included; the ghost of (F, I) is I − e_d (min) or I + e_d (max).
#   Facets ordered xmin, xmax, ymin, ymax, zmin, zmax; each facet's ghosts in the
#   column-major (`CartesianIndices`) order of its nodes, as the mesh's linear order.
#   DOF order is physical nodes in mesh linear order, then facet ghosts in that order.
#   Cell measure ĥ_I = Π_d ĥ_d(I_d); facet weight ω_F(I) = Π_{e≠d} ĥ_e(I_e).
#   Rows are scaled as a variational assembly does, a PDE row times ĥ_I and a
#   boundary-condition row times ω_F(I).
#   Neumann and Robin data are per facet (`g_F = α_F u + β_F ∇u·η_F`): a node on k facets
#   (an edge or a corner) has k data and k ghosts, one eliminated along each normal.
module GhostOracleND

using LinearAlgebra: Diagonal, dot, ×
using SparseArrays: SparseMatrixCSC, sparse, spzeros
import Bramble
using Bramble: change_points!, domain, interval, iterative_refinement!, mesh, points

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(@__DIR__, "GhostOracle1D.jl"))
const O1 = Main.GhostOracle1D

const PROBLEMS = O1.PROBLEMS
const MESH_KINDS = O1.MESH_KINDS
const FACETS = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)
const SIDES = (:min, :max)

facet_symbol(d::Int, side::Symbol) = FACETS[2(d - 1) + (side === :min ? 1 : 2)]
outward(side::Symbol) = side === :min ? -1.0 : 1.0
# The facet of the 1D axis mesh that GhostOracle1D's `relation_row` names.
side1d(side::Symbol) = side === :min ? :xmin : :xmax

#------------------------------------------------------------------------------------------#
# Geometry and DOF layout
#------------------------------------------------------------------------------------------#

"""
    axis_meshes(Ωₕ) -> NTuple{D, Mesh1D}

The 1D meshes of the axes, read from `Ωₕ` each time (they follow `change_points!`).
"""
axis_meshes(Ωₕ) = ntuple(d -> Ωₕ(d), Bramble.dim(Ωₕ))

axis_points(Ωₕ) = map(m -> collect(Float64, points(m)), axis_meshes(Ωₕ))

"""
    sizes(Ωₕ) -> NTuple{D, Int}

The number of physical points per axis.
"""
sizes(Ωₕ) = map(length, axis_points(Ωₕ))

"""
    ghost_count(ns::NTuple{D, Int}) -> Int
    ghost_count(Ωₕ) -> Int

`Σ_d 2 Π_{e≠d} n_e`: one ghost per (facet, node), corner and edge points of the padded grid
excluded.
"""
ghost_count(ns::NTuple{D, Int}) where {D} = sum(d -> 2 * (prod(ns) ÷ ns[d]), 1:D)
ghost_count(Ωₕ) = ghost_count(sizes(Ωₕ))

node_point(xs, I::CartesianIndex) = ntuple(e -> xs[e][I[e]], length(xs))

"""
    node_points(Ωₕ) -> Vector{NTuple{D, Float64}}

The physical points in mesh linear order.
"""
function node_points(Ωₕ)
    xs = axis_points(Ωₕ)
    return vec([node_point(xs, I) for I in CartesianIndices(map(length, xs))])
end

# Padded index of a physical node (GhostOracle1D's `ext` along every axis).
padded(I::CartesianIndex) = I + oneunit(I)

function ghost_padded(d::Int, side::Symbol, I::CartesianIndex, ns)
    return CartesianIndex(Base.setindex(Tuple(padded(I)), side === :min ? 1 : ns[d] + 2, d))
end

"""
    ghosts(Ωₕ) -> Vector{Tuple{Int, Symbol, CartesianIndex{D}}}

The facet ghosts `(d, side, I)` in DOF order: axis `d`, `side ∈ (:min, :max)`, the facet
node `I` (physical index) the ghost belongs to.
"""
function ghosts(Ωₕ)
    ns = sizes(Ωₕ)
    D = length(ns)
    list = Tuple{Int, Symbol, CartesianIndex{D}}[]
    for d in 1:D, side in SIDES

        k = side === :min ? 1 : ns[d]
        for I in CartesianIndices(ntuple(e -> e == d ? (k:k) : (1:ns[e]), D))
            push!(list, (d, side, I))
        end
    end
    return list
end

"""
    dof_layout(Ωₕ) -> NamedTuple

Padded linear indices of the DOFs: `phys` (mesh linear order), `ghost` (DOF order), `kept`
(`vcat(phys, ghost)`), and `absent`, the padded corner and edge points; with `ns`, the
padded size `ps` and `ghosts(Ωₕ)`.
"""
function dof_layout(Ωₕ)
    ns = sizes(Ωₕ)
    ps = ns .+ 2
    LP = LinearIndices(ps)
    phys = vec([LP[padded(I)] for I in CartesianIndices(ns)])
    gl = ghosts(Ωₕ)
    ghost = [LP[ghost_padded(d, s, I, ns)] for (d, s, I) in gl]
    kept = vcat(phys, ghost)
    absent = setdiff(1:prod(ps), kept)
    return (; ns, ps, phys, ghost, kept, absent, ghosts = gl)
end

"""
    ghost_points(Ωₕ) -> Vector{NTuple{D, Float64}}

The ghost coordinates in DOF order: the facet node with its normal coordinate replaced by
GhostOracle1D's `x_0 = x_1 − h_2` or `x_{n+1} = x_n + h_n` on that axis.
"""
function ghost_points(Ωₕ)
    xs = axis_points(Ωₕ)
    xe = map(O1.extended_points, axis_meshes(Ωₕ))
    return [ntuple(e -> e == d ? (side === :min ? xe[d][1] : xe[d][end]) : xs[e][I[e]],
                length(xs)) for (d, side, I) in ghosts(Ωₕ)]
end

"""
    cell_measures(Ωₕ) -> Vector{Float64}

`ĥ_I = Π_d ĥ_d(I_d)` in mesh linear order: the `(·,·)_{H,*}` weights.
"""
cell_measures(Ωₕ) = reduce((a, b) -> kron(b, a), map(O1.cell_measures, axis_meshes(Ωₕ)))

"""
    mean_weights(Ωₕ) -> Vector{Float64}

The weights `w` of the discrete mean `Σ ĥ_I u_I` on the physical nodes. For an augmented
system, pad with zeros at the ghost DOFs.
"""
mean_weights(Ωₕ) = cell_measures(Ωₕ)

"""
    facet_weight(Ωₕ, d, I) -> Float64

`ω_F(I) = Π_{e≠d} ĥ_e(I_e)` for a node `I` of a facet normal to axis `d`.
"""
function facet_weight(Ωₕ, d::Int, I::CartesianIndex)
    ĥ = map(O1.cell_measures, axis_meshes(Ωₕ))
    return prod(e == d ? 1.0 : ĥ[e][I[e]] for e in eachindex(ĥ))
end

"""
    hmax(Ωₕ) -> Float64

The largest spacing over all axes.
"""
hmax(Ωₕ) = maximum(map(O1.hmax, axis_meshes(Ωₕ)))

#------------------------------------------------------------------------------------------#
# Operators on the padded grid
#------------------------------------------------------------------------------------------#

speye(n::Int) = sparse(Diagonal(ones(n)))

"""
    lift(A, d, ps) -> SparseMatrixCSC

`A`, an operator on axis `d` of the padded grid of size `ps`, as `I ⊗ … ⊗ A ⊗ … ⊗ I`
(first axis fastest, as the linear order).
"""
function lift(A, d::Int, ps)
    M = d == 1 ? sparse(A) : speye(ps[1])
    for e in 2:length(ps)
        M = kron(e == d ? sparse(A) : speye(ps[e]), M)
    end
    return M
end

"""
    operators(Ωₕ) -> NamedTuple

Sparse matrices on the whole padded grid (`prod(ns .+ 2)` points, padded linear order),
each the `lift` of GhostOracle1D's 1D operator of that axis: `Dminus`, `Dstar`, `Dc`, `Laps` (tuples
over the axes) and `Lap = Σ_d D̃_d D₋_d`. Restrict with `dof_layout`.
"""
function operators(Ωₕ)
    ops = map(O1.operators, axis_meshes(Ωₕ))
    ps = sizes(Ωₕ) .+ 2
    D = length(ps)
    Dminus = ntuple(d -> lift(ops[d].Dminus, d, ps), D)
    Dstar = ntuple(d -> lift(ops[d].Dstar, d, ps), D)
    Dc = ntuple(d -> lift(ops[d].Dc, d, ps), D)
    Laps = ntuple(d -> lift(ops[d].Lap, d, ps), D)
    return (; Dminus, Dstar, Dc, Laps, Lap = sum(Laps))
end

#------------------------------------------------------------------------------------------#
# Facet relations and the ghost closure
#------------------------------------------------------------------------------------------#

"""
    FacetRobin(α, β, g)

`α u + β ∇u·η_F = g(x)` on one facet, `g` a function of the point tuple, discretised at
each facet node as GhostOracle1D's `Robin(α, β, g(x_I))` along the facet normal.
"""
struct FacetRobin{G}
    α::Float64
    β::Float64
    g::G
end

"""
    FacetNeumann(β, g)

`β ∇u·η_F = g(x)`, that is `FacetRobin(0, β, g)`.
"""
FacetNeumann(β, g) = FacetRobin(0.0, β, g)

"""
    relation_row(c::FacetRobin, Ωₕ, d, side, I) -> (cols, vals, g)

The relation `α u_I + β η_F Dc_d u(I) = g(x_I)` at node `I` of facet `(d, side)`, on padded
linear indices `cols`: GhostOracle1D's 1D `relation_row` along axis `d`, lifted to the grid line
through `I`.
"""
function relation_row(c::FacetRobin, Ωₕ, d::Int, side::Symbol, I::CartesianIndex)
    ms = axis_meshes(Ωₕ)
    ps = sizes(Ωₕ) .+ 2
    LP = LinearIndices(ps)
    rel = O1.Robin(c.α, c.β, c.g(node_point(axis_points(Ωₕ), I)))
    r, g = O1.relation_row(rel, ms[d], side1d(side))
    J = Tuple(padded(I))
    cols = Int[]
    vals = Float64[]
    for k in eachindex(r)
        iszero(r[k]) && continue
        push!(cols, LP[CartesianIndex(Base.setindex(J, k, d))])
        push!(vals, r[k])
    end
    return cols, vals, g
end

"""
    ghost_closure(relations, Ωₕ) -> (E::SparseMatrixCSC{Float64, Int}, d::Vector{Float64})

`u_kept = E u_phys + d`, rows in DOF order (physical, then facet ghosts), `u_phys` in mesh
linear order. `relations` maps each facet symbol (`:xmin`, `:xmax`, …) to a `FacetRobin`.
Each ghost solves its own facet relation, which reads only physical nodes besides it.
"""
function ghost_closure(relations, Ωₕ)
    L = dof_layout(Ωₕ)
    np = length(L.phys)
    nk = length(L.kept)
    pos = zeros(Int, prod(L.ps))
    pos[L.kept] .= 1:nk
    rows, cols, vals = collect(1:np), collect(1:np), ones(np)
    dvec = zeros(nk)
    for (k, (ax, side, I)) in enumerate(L.ghosts)
        c = getfield(relations, facet_symbol(ax, side))
        pc, pv, g = relation_row(c, Ωₕ, ax, side, I)
        cg = pv[findfirst(==(L.ghost[k]), pc)]
        for (p, v) in zip(pc, pv)
            p == L.ghost[k] && continue
            1 <= pos[p] <= np || error("a facet relation reads a non-physical point")
            push!(rows, np + k)
            push!(cols, pos[p])
            push!(vals, -v / cg)
        end
        dvec[np + k] = g / cg
    end
    return sparse(rows, cols, vals, nk, np), dvec
end

#------------------------------------------------------------------------------------------#
# Systems for −Δu = f
#------------------------------------------------------------------------------------------#

"""
    eliminated_system(relations, Ωₕ, f) -> (A, F)

Ghosts eliminated: `ĥ_I (−Δ_H (E u + d))_I = ĥ_I f(x_I)` on every physical node, so
`A = diag(ĥ) (−Δ_H) E` and `F = ĥ ⊙ (f + Δ_H d)`, `Δ_H` restricted to physical rows and
kept columns.
"""
function eliminated_system(relations, Ωₕ, f)
    L = dof_layout(Ωₕ)
    E, d = ghost_closure(relations, Ωₕ)
    Lap = operators(Ωₕ).Lap[L.phys, L.kept]
    W = Diagonal(cell_measures(Ωₕ))
    A = sparse(W * (-Lap * E))
    F = W * (f.(node_points(Ωₕ)) .+ Lap * d)
    return A, F
end

"""
    augmented_system(relations, Ωₕ, f) -> (A, F)

Ghosts as unknowns (DOF order): PDE rows `ĥ_I (−Δ_H u)_I = ĥ_I f(x_I)` on every physical
node, then one row per ghost, `ω_F(I)` times its facet relation.
"""
function augmented_system(relations, Ωₕ, f)
    L = dof_layout(Ωₕ)
    np = length(L.phys)
    nk = length(L.kept)
    pos = zeros(Int, prod(L.ps))
    pos[L.kept] .= 1:nk
    ĥ = cell_measures(Ωₕ)
    Apde = Diagonal(ĥ) * (-operators(Ωₕ).Lap[L.phys, L.kept])
    rows, cols, vals = Int[], Int[], Float64[]
    Fg = zeros(length(L.ghost))
    for (k, (ax, side, I)) in enumerate(L.ghosts)
        c = getfield(relations, facet_symbol(ax, side))
        pc, pv, g = relation_row(c, Ωₕ, ax, side, I)
        ω = facet_weight(Ωₕ, ax, I)
        for (p, v) in zip(pc, pv)
            push!(rows, k)
            push!(cols, pos[p])
            push!(vals, ω * v)
        end
        Fg[k] = ω * g
    end
    A = vcat(sparse(Apde), sparse(rows, cols, vals, length(L.ghost), nk))
    return A, vcat(ĥ .* f.(node_points(Ωₕ)), Fg)
end

#------------------------------------------------------------------------------------------#
# The problems
#------------------------------------------------------------------------------------------#

"""
    ExpWave(z, φ)

`u(x) = Re exp(z·x + iφ)`, `z ∈ ℂᴰ`: `∇u = Re(z e^{…})`, `Δu = Re((z·z) e^{…})`. Smooth,
no symmetry, non-zero data on every facet.
"""
struct ExpWave{D}
    z::NTuple{D, ComplexF64}
    φ::Float64
end

phase(w::ExpWave, x) = exp(sum(w.z .* x) + im * w.φ)
value(w::ExpWave, x) = real(phase(w, x))
gradient(w::ExpWave, x) = real.(w.z .* phase(w, x))
laplacian(w::ExpWave, x) = real(sum(w.z .^ 2) * phase(w, x))

default_exact(dim::Int) = ExpWave(ntuple(d -> complex((1.0, -0.5, 0.3)[d], (2.0, 1.0, -1.5)[d]),
        dim), 0.3)

function check_problem(problem::Symbol)
    problem in PROBLEMS ||
        throw(ArgumentError("unknown problem $problem; one of $PROBLEMS"))
end

function check_dim(dim::Int)
    dim in (2, 3) || throw(ArgumentError("GhostOracleND takes dim 2 or 3 (dim = $dim)"))
    return nothing
end

is_neumann(problem::Symbol) = problem === :neumann_a || problem === :neumann_b
is_augmented(problem::Symbol) = problem === :neumann_b || problem === :robin_b

const DEFAULT_α = (2.0, 0.5, 1.0, 3.0, 0.7, 1.5)
const DEFAULT_β = (1.5, 3.0, 2.0, 1.0, 0.8, 2.5)

"""
    problem_relations(problem, Ωₕ; α = DEFAULT_α, β = DEFAULT_β, exact) -> NamedTuple

One `FacetRobin` per facet (`α[j]`, `β[j]` for `FACETS[j]`), data
`g_F = α u + β ∇u·η_F` from the exact solution. `α` is ignored for the Neumann problems.
"""
function problem_relations(problem::Symbol, Ωₕ; α = DEFAULT_α, β = DEFAULT_β,
        exact = default_exact(Bramble.dim(Ωₕ)))
    check_problem(problem)
    D = Bramble.dim(Ωₕ)
    names = FACETS[1:(2D)]
    rels = map(1:(2D)) do j
        d, side = (j + 1) ÷ 2, isodd(j) ? :min : :max
        a = is_neumann(problem) ? 0.0 : α[j]
        b, η = β[j], outward(side)
        FacetRobin(a, b, x -> a * value(exact, x) + b * η * gradient(exact, x)[d])
    end
    return NamedTuple{names}(Tuple(rels))
end

"""
    system(problem::Symbol, Ωₕ; α, β, exact) -> (A::SparseMatrixCSC{Float64, Int}, F)

`_a`: `eliminated_system` (physical DOFs); `_b`: `augmented_system` (physical then facet
ghosts), for `−Δu = f` with `f = −Δ exact`.
"""
function system(problem::Symbol, Ωₕ; exact = default_exact(Bramble.dim(Ωₕ)), kwargs...)
    relations = problem_relations(problem, Ωₕ; exact, kwargs...)
    f = x -> -laplacian(exact, x)
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

The discrete solution on the problem's DOFs. The Neumann problems add `Σ ĥ_I u_I = 0`
through a sparse bordered system (one Lagrange multiplier).
"""
function solve(problem::Symbol, Ωₕ; kwargs...)
    A, F = system(problem, Ωₕ; kwargs...)
    m = size(A, 1)
    is_neumann(problem) || return A \ F
    w = zeros(m)
    w[1:length(mean_weights(Ωₕ))] .= mean_weights(Ωₕ)
    B = [A sparse(reshape(w, :, 1)); sparse(reshape(w, 1, :)) spzeros(1, 1)]
    return (B \ vcat(F, 0.0))[1:m]
end

"""
    discrete_error(problem::Symbol, Ωₕ; exact, kwargs...) -> Float64

`(‖D₋e‖_+² + ‖e‖_{H,*}²)^{1/2}`, `e = u_h − R_H u` on the physical nodes, with
`‖D₋e‖_+² = Σ_d Σ_{I_d ≥ 2} h_d ω_d(I) (D₋_d e)²`; for Neumann `R_H u` is shifted to
discrete mean zero.
"""
function discrete_error(problem::Symbol, Ωₕ; exact = default_exact(Bramble.dim(Ωₕ)),
        kwargs...)
    xs = axis_points(Ωₕ)
    ns = map(length, xs)
    D = length(ns)
    ĥ = cell_measures(Ωₕ)
    uh = solve(problem, Ωₕ; exact, kwargs...)[1:prod(ns)]
    ue = value.(Ref(exact), node_points(Ωₕ))
    is_neumann(problem) && (ue .-= dot(ĥ, ue) / sum(ĥ))
    e = reshape(uh .- ue, ns)
    total = dot(ĥ, abs2.(vec(e)))
    ĥs = map(O1.cell_measures, axis_meshes(Ωₕ))
    shape(v, d) = reshape(v, ntuple(e -> e == d ? length(v) : 1, D))
    for d in 1:D
        w = mapreduce(e -> e == d ? shape(1 ./ diff(xs[d]), d) : shape(ĥs[e], e),
            (a, b) -> a .* b, 1:D)
        total += sum(w .* abs2.(diff(e; dims = d)))
    end
    return sqrt(total)
end

#------------------------------------------------------------------------------------------#
# Mesh sequences and observed orders
#------------------------------------------------------------------------------------------#

# A smooth increasing map of [0, 1] per axis, different on each so no symmetry survives.
function graded_axis(d::Int)
    d == 1 ? O1.graded_map :
    d == 2 ? (ξ -> 1 - O1.graded_map(1 - ξ)) :
    (ξ -> ξ + 0.15 * sin(2π * ξ) / (2π))
end

default_n0(dim::Int) = dim == 2 ? 9 : 5

"""
    make_mesh(kind::Symbol, level::Int; dim = 2, n0 = default_n0(dim), seed = 20261014)

Level `level` of a sequence on `[0, 1]ᵈⁱᵐ` with `n0` points per axis: `:uniform` and
`:random` (Bramble's random mesh drawn from `seed`) refined `level` times by
`iterative_refinement!`; `:graded` is `graded_axis(d)` of `(n0 − 1)·2^level + 1` uniform
points on axis `d`.
"""
function make_mesh(kind::Symbol, level::Int; dim::Int = 2, n0::Int = default_n0(dim),
        seed::Int = 20261014)
    check_dim(dim)
    Ω = domain(foldl(×, ntuple(_ -> interval(0.0, 1.0), dim)))
    if kind === :uniform || kind === :random
        if kind === :random
            Bramble._seed_mesh1d_rng!(seed)
            Ωₕ = try
                mesh(Ω, ntuple(_ -> n0, dim), ntuple(_ -> false, dim))
            finally
                Bramble._unseed_mesh1d_rng!()
            end
        else
            Ωₕ = mesh(Ω, ntuple(_ -> n0, dim), ntuple(_ -> true, dim))
        end
        for _ in 1:level
            iterative_refinement!(Ωₕ)
        end
        return Ωₕ
    elseif kind === :graded
        n = (n0 - 1) * 2^level + 1
        Ωₕ = mesh(Ω, ntuple(_ -> n, dim), ntuple(_ -> true, dim))
        change_points!(Ωₕ, ntuple(d -> graded_axis(d).(axis_points(Ωₕ)[d]), dim))
        return Ωₕ
    end
    throw(ArgumentError("unknown mesh kind $kind; one of $MESH_KINDS"))
end

"""
    observed_errors(problem, meshkind; dim = 2, levels = 4, n0, seed, kwargs...)
        -> (hs::Vector{Float64}, errs::Vector{Float64})

`hₘₐₓ` and `discrete_error` over `levels` meshes of `make_mesh(meshkind, ℓ; dim)`.
"""
function observed_errors(problem::Symbol, meshkind::Symbol; dim::Int = 2, levels::Int = 4,
        n0::Int = default_n0(dim), seed::Int = 20261014, kwargs...)
    check_dim(dim)
    check_problem(problem)
    hs = Float64[]
    errs = Float64[]
    for ℓ in 0:(levels - 1)
        Ωₕ = make_mesh(meshkind, ℓ; dim, n0, seed)
        push!(hs, hmax(Ωₕ))
        push!(errs, discrete_error(problem, Ωₕ; kwargs...))
    end
    return hs, errs
end

"""
    observed_order(problem::Symbol, meshkind::Symbol; dim = 2, levels = 4, kwargs...) -> Float64

The least-squares slope of `log(error)` against `log(hₘₐₓ)` over `levels` meshes. It checks
the slope only, never the size of the errors.
"""
function observed_order(problem::Symbol, meshkind::Symbol; kwargs...)
    hs, errs = observed_errors(problem, meshkind; kwargs...)
    X = log.(hs)
    Y = log.(errs)
    X̄, Ȳ = sum(X) / length(X), sum(Y) / length(Y)
    return sum((X .- X̄) .* (Y .- Ȳ)) / sum(abs2, X .- X̄)
end

end # module GhostOracleND
