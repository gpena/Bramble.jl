# Corner ghosts for a mixed-derivative problem under natural conditions (#218). It is
# loaded by corners.jl, never by the package.
#
# The problem (notes §4): −∇·(K∇u) = f on (0, 1)², K constant SPD with k₁₂ ≠ 0, and the
# conormal Robin relation α_F u + β_F (K∇u)·η_F = g_F on every facet, one datum per facet,
# so a corner node carries two. Discretised at every physical node as
#
#     −(k₁₁ Δ_x + 2k₁₂ X + k₂₂ Δ_y) u = f,   Δ_d = D̃_d D₋_d,
#
# with X the mixed difference `Dc_x Dc_y` (`mixed = :centered`) or the cross-weighted
# `D̽_x D̽_y` (`mixed = :cross_weighted`), and at the node I of facet F (normal axis d,
# tangential axis e) as α_F u_I + β_F η_F (k_dd Dc_d u + k_de Dc_e u)_I = g_F(I).
#
# Every operator is a sparse matrix on the padded grid (paper indices 0 … n_d + 1 per axis,
# x fastest), the Kronecker product of GhostOracle1D's verified 1D factors. Facet ghosts are
# eliminated through the facet relations; at a corner the two relations couple the two facet
# ghosts (`Dc_e` reads the other facet's ghost), a 2×2 solve. X at a corner node reads the
# padded corner point, handled by one of three closures.
#
#   facet_onesided       no corner point, X at the four corner nodes is the inward one-sided
#                        mixed difference D_{±x} D_{±y} on physical nodes only;
#   padded_extrapolate   u_c = u_{c+e_x} + u_{c+e_y} − u_{c+e_x+e_y} (bilinear, notes §4 c2);
#   padded_mixed         the mixed second difference of the corner ghost cell extrapolated
#                        linearly from the three cells around the corner node (two of them
#                        read facet ghosts); exact for bilinear u on any tensor mesh.
#
# Systems are row-scaled as a variational assembly would be, a PDE row times ĥ_x ĥ_y. DOFs
# are the physical nodes in mesh linear order (ghosts eliminated).
module GhostCorners

using LinearAlgebra: Diagonal, I
using SparseArrays: SparseMatrixCSC, sparse, spzeros, kron
import Bramble
using Bramble: change_points!, domain, interval, mesh, points, ×

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(@__DIR__, "GhostOracle1D.jl"))
const O1 = Main.GhostOracle1D

export corner_blocks, cross_weighted, discrete_error, facet_ghost_count, ghost_closure,
       ghost_relations, make_mesh, observed_errors, observed_order, padded_ghost_count,
       pde_operator, solve, system, system_matrix

const PROBLEMS = (:aniso_robin,)
const CLOSURES = (:facet_onesided, :padded_extrapolate, :padded_mixed)
const MIXED = (:centered, :cross_weighted)
const MESH_KINDS = (:uniform, :random, :graded)
const FACETS = (:xmin, :xmax, :ymin, :ymax)

#------------------------------------------------------------------------------------------#
# Counts
#------------------------------------------------------------------------------------------#

"""
    padded_ghost_count(n::NTuple{D, Int}) -> Int

Points a layout padded by one point per axis end adds to an `n[1] × … × n[D]` grid: facet
ghosts, plus edge lines and corners in 3D (corners in 2D).
"""
padded_ghost_count(n::NTuple{D, Int}) where {D} = prod(n .+ 2) - prod(n)

"""
    facet_ghost_count(n::NTuple{D, Int}) -> Int

Facet ghosts only, `Σ_d 2 Π_{e≠d} n_e` (notes §2).
"""
facet_ghost_count(n::NTuple{D, Int}) where {D} = sum(2 * prod(n[e] for e in 1:D if e != d; init = 1) for d in 1:D)

#------------------------------------------------------------------------------------------#
# Geometry and 1D factors
#------------------------------------------------------------------------------------------#

axis_points(Ωₕ, d::Int) = collect(Float64, points(Ωₕ(d)))
sizes(Ωₕ) = (length(points(Ωₕ(1))), length(points(Ωₕ(2))))

"""
    cross_weighted(Ωₕ1) -> SparseMatrixCSC

The cross-weighted difference `D̽u_i = [h_i (u_{i+1} − u_i)/h_{i+1} + h_{i+1} (u_i − u_{i−1})/h_i]
/ (h_i + h_{i+1})` on the extended 1D range (rows `1 … n`), with the extended spacings of
`GhostOracle1D` (`h_1 = h_2`, `h_{n+1} = h_n`, so it equals `Dc` at the two end rows).
"""
function cross_weighted(Ωₕ1)
    hp = O1.extended_spacings(Ωₕ1)
    N = length(hp)
    n = N - 2
    A = spzeros(N, N)
    for i in 1:n
        hi, hf = hp[O1.ext(i)], hp[O1.ext(i + 1)]
        s = hi + hf
        A[O1.ext(i), O1.ext(i + 1)] = hi / (hf * s)
        A[O1.ext(i), O1.ext(i)] = (hf / hi - hi / hf) / s
        A[O1.ext(i), O1.ext(i - 1)] = -hf / (hi * s)
    end
    return A
end

# Padded linear index of the paper index (i, j), i ∈ 0:nx+1, j ∈ 0:ny+1, x fastest.
padded(i::Int, j::Int, nx::Int) = (i + 1) + (nx + 2) * j
# Physical DOF of the node (i, j), 1 ≤ i ≤ nx, 1 ≤ j ≤ ny.
physical(i::Int, j::Int, nx::Int) = i + nx * (j - 1)

physical_padded(nx, ny) = [padded(i, j, nx) for j in 1:ny for i in 1:nx]
corner_nodes(nx, ny) = ((1, 1), (nx, 1), (1, ny), (nx, ny))
corner_points(nx, ny) = ((0, 0), (nx + 1, 0), (0, ny + 1), (nx + 1, ny + 1))

"""
    facet_ghosts(nx, ny) -> Vector{NTuple{4, Int}}

`(facet, node_i, node_j, padded_index)` per facet ghost in DOF order: xmin, xmax, ymin,
ymax, each facet's nodes in lexicographic order.
"""
function facet_ghosts(nx, ny)
    gs = NTuple{4, Int}[]
    for j in 1:ny
        push!(gs, (1, 1, j, padded(0, j, nx)))
    end
    for j in 1:ny
        push!(gs, (2, nx, j, padded(nx + 1, j, nx)))
    end
    for i in 1:nx
        push!(gs, (3, i, 1, padded(i, 0, nx)))
    end
    for i in 1:nx
        push!(gs, (4, i, ny, padded(i, ny + 1, nx)))
    end
    return gs
end

# The padded 2D operators along x and along y, from 1D factors.
along_x(Ax, ny) = kron(sparse(1.0I, ny + 2, ny + 2), Ax)
along_y(Ay, nx) = kron(Ay, sparse(1.0I, nx + 2, nx + 2))

#------------------------------------------------------------------------------------------#
# The problem data
#------------------------------------------------------------------------------------------#

# u = eˣ cos(2y + ½) + x y²: u_xy = −2eˣ sin(2y + ½) + 2y is non-zero at all four corners.
const DEFAULT_EXACT = (u = (x, y) -> exp(x) * cos(2y + 0.5) + x * y^2,
    ux = (x, y) -> exp(x) * cos(2y + 0.5) + y^2,
    uy = (x, y) -> -2exp(x) * sin(2y + 0.5) + 2x * y,
    uxx = (x, y) -> exp(x) * cos(2y + 0.5),
    uxy = (x, y) -> -2exp(x) * sin(2y + 0.5) + 2y,
    uyy = (x, y) -> -4exp(x) * cos(2y + 0.5) + 2x)

const DEFAULT_K = [1.0 0.5; 0.5 2.0]
# Per facet, in the order xmin, xmax, ymin, ymax.
const DEFAULT_α = (2.0, 0.5, 1.0, 1.5)
const DEFAULT_β = (1.5, 3.0, 1.0, 2.0)

facet_normal(F::Int) = F == 1 || F == 3 ? -1.0 : 1.0
facet_axis(F::Int) = F <= 2 ? 1 : 2

function check_problem(problem::Symbol)
    problem in PROBLEMS ||
        throw(ArgumentError("unknown problem $problem; one of $PROBLEMS"))
end
check_closure(c::Symbol) = c in CLOSURES ||
                           throw(ArgumentError("unknown closure $c; one of $CLOSURES"))
check_mixed(m::Symbol) = m in MIXED || throw(ArgumentError("unknown mixed form $m; one of $MIXED"))

#------------------------------------------------------------------------------------------#
# Facet relations and the corner 2×2 systems
#------------------------------------------------------------------------------------------#

"""
    ghost_relations(Ωₕ; K, α, β, exact) -> (R::SparseMatrixCSC, g::Vector{Float64})

One row per facet ghost (DOF order of `facet_ghosts`) on the padded grid:
`α_F e_I + β_F η_F (k_dd Dc_d + k_de Dc_e)[I, :]`, and its data `g_F(I)` from `exact`.
"""
function ghost_relations(Ωₕ; K = DEFAULT_K, α = DEFAULT_α, β = DEFAULT_β,
        exact = DEFAULT_EXACT)
    nx, ny = sizes(Ωₕ)
    x, y = axis_points(Ωₕ, 1), axis_points(Ωₕ, 2)
    Dc = (along_x(O1.operators(Ωₕ(1)).Dc, ny), along_y(O1.operators(Ωₕ(2)).Dc, nx))
    gs = facet_ghosts(nx, ny)
    R = spzeros(length(gs), (nx + 2) * (ny + 2))
    g = zeros(length(gs))
    for (r, (F, i, j, _)) in enumerate(gs)
        d = facet_axis(F)
        e = 3 - d
        η = facet_normal(F)
        p = padded(i, j, nx)
        R[r, :] = β[F] * η * (K[d, d] * Dc[d][p, :] + K[d, e] * Dc[e][p, :])
        R[r, p] += α[F]
        ∇u = (exact.ux(x[i], y[j]), exact.uy(x[i], y[j]))
        g[r] = α[F] * exact.u(x[i], y[j]) + β[F] * η * (K[d, 1] * ∇u[1] + K[d, 2] * ∇u[2])
    end
    return R, g
end

"""
    corner_blocks(Ωₕ; K, α, β) -> Vector{Matrix{Float64}}

At each corner node (order (1,1), (nx,1), (1,ny), (nx,ny)), the 2×2 matrix of the x-facet
and y-facet relations (rows) in the x-facet and y-facet ghosts (columns).
"""
function corner_blocks(Ωₕ; kwargs...)
    nx, ny = sizes(Ωₕ)
    R, _ = ghost_relations(Ωₕ; kwargs...)
    gs = facet_ghosts(nx, ny)
    blocks = Matrix{Float64}[]
    for (i, j) in corner_nodes(nx, ny)
        rx = findfirst(t -> facet_axis(t[1]) == 1 && t[2] == i && t[3] == j, gs)
        ry = findfirst(t -> facet_axis(t[1]) == 2 && t[2] == i && t[3] == j, gs)
        cols = [gs[rx][4], gs[ry][4]]
        push!(blocks, Matrix(R[[rx, ry], cols]))
    end
    return blocks
end

#------------------------------------------------------------------------------------------#
# The PDE operator and the closures
#------------------------------------------------------------------------------------------#

# The inward one-sided mixed difference at corner node (i, j): D_{sx,x} D_{sy,y} with
# sx = +1 at i = 1 (forward) and −1 at i = nx (backward), likewise sy, physical nodes only.
function onesided_mixed_row(i, j, nx, ny, x, y)
    sx = i == 1 ? 1 : -1
    sy = j == 1 ? 1 : -1
    hx = x[i + sx] - x[i]
    hy = y[j + sy] - y[j]
    row = spzeros((nx + 2) * (ny + 2))
    for (a, b, c) in ((0, 0, 1.0), (sx, 0, -1.0), (0, sy, -1.0), (sx, sy, 1.0))
        row[padded(i + a, j + b, nx)] += c / (hx * hy)
    end
    return row
end

"""
    pde_operator(Ωₕ; closure, mixed, K) -> SparseMatrixCSC

`k₁₁ Δ_x + 2k₁₂ X + k₂₂ Δ_y` on the padded grid, rows at the physical nodes (mesh order),
columns padded. `closure = :facet_onesided` replaces X at the four corner nodes by the
inward one-sided mixed difference, so no row reads a padded corner point.
"""
function pde_operator(Ωₕ; closure::Symbol = :padded_extrapolate, mixed::Symbol = :centered,
        K = DEFAULT_K)
    check_closure(closure)
    check_mixed(mixed)
    nx, ny = sizes(Ωₕ)
    ox, oy = O1.operators(Ωₕ(1)), O1.operators(Ωₕ(2))
    Mx, My = mixed === :centered ? (ox.Dc, oy.Dc) :
             (cross_weighted(Ωₕ(1)), cross_weighted(Ωₕ(2)))
    X = kron(My, Mx)
    rows = physical_padded(nx, ny)
    Xp = X[rows, :]
    if closure === :facet_onesided
        x, y = axis_points(Ωₕ, 1), axis_points(Ωₕ, 2)
        for (i, j) in corner_nodes(nx, ny)
            Xp[physical(i, j, nx), :] = onesided_mixed_row(i, j, nx, ny, x, y)
        end
    end
    P = K[1, 1] * along_x(ox.Lap, ny)[rows, :] + K[2, 2] * along_y(oy.Lap, nx)[rows, :]
    return P + 2K[1, 2] * Xp
end

"""
    ghost_closure(Ωₕ; closure, K, α, β, exact) -> (E::SparseMatrixCSC, d::Vector{Float64})

`u_pad = E u + d` on the padded grid, `u` the physical nodes in mesh order. Facet ghosts
solve the facet relations (diagonal except a 2×2 block per corner); corner points follow
`closure` (rows left empty for `:facet_onesided`, which never reads them).
"""
function ghost_closure(Ωₕ; closure::Symbol = :padded_extrapolate, kwargs...)
    check_closure(closure)
    nx, ny = sizes(Ωₕ)
    Np = (nx + 2) * (ny + 2)
    nph = nx * ny
    R, g = ghost_relations(Ωₕ; kwargs...)
    gs = facet_ghosts(nx, ny)
    gcols = [t[4] for t in gs]
    pcols = physical_padded(nx, ny)
    Rg = R[:, gcols]
    Rp = R[:, pcols]
    # Rg is diagonal but for the corner pairs: a sparse LU solves the 2×2 blocks.
    Eg = -(Matrix(Rg) \ Matrix(Rp))
    dg = Rg \ g
    E = spzeros(Np, nph)
    d = zeros(Np)
    for (c, p) in enumerate(pcols)
        E[p, c] = 1.0
    end
    E[gcols, :] = sparse(Eg)
    d[gcols] = dg
    closure === :facet_onesided && return E, d
    # Corner point c, its facet-ghost neighbours a (along x) and b (along y), node n.
    for ((ci, cj), (ni, nj)) in zip(corner_points(nx, ny), corner_nodes(nx, ny))
        c = padded(ci, cj, nx)
        if closure === :padded_extrapolate
            terms = ((ci, nj, 1.0), (ni, cj, 1.0), (ni, nj, -1.0))
        else
            # Cells around node n: the corner ghost cell C, its x-neighbour A (reads the
            # y-facet ghosts), y-neighbour B (reads the x-facet ghosts), the diagonal cell
            # D. The mixed second difference of the cell with x lines (a, a′) and y lines
            # (b, b′), each pair listed outward first, is ∂ = u_ab − u_a′b − u_ab′ + u_a′b′.
            # With the ghost spacings h₁ = h₂ all four cells have equal area and their
            # centres a parallelogram, so ∂C = ∂A + ∂B − ∂D is exact for bilinear u.
            mi = ni == 1 ? 2 : nx - 1          # the next physical lines inward
            mj = nj == 1 ? 2 : ny - 1
            # ∂C = u_c − u_(ni,cj) − u_(ci,nj) + u_n, solved for u_c below.
            ∂A = ((ni, cj, 1.0), (mi, cj, -1.0), (ni, nj, -1.0), (mi, nj, 1.0))
            ∂B = ((ci, nj, 1.0), (ni, nj, -1.0), (ci, mj, -1.0), (ni, mj, 1.0))
            ∂D = ((ni, nj, -1.0), (mi, nj, 1.0), (ni, mj, 1.0), (mi, mj, -1.0))
            terms = ((ci, nj, 1.0), (ni, cj, 1.0), (ni, nj, -1.0), ∂A..., ∂B..., ∂D...)
        end
        for (a, b, w) in terms
            q = padded(a, b, nx)
            E[c, :] += w * E[q, :]
            d[c] += w * d[q]
        end
    end
    return E, d
end

#------------------------------------------------------------------------------------------#
# Systems, solutions, errors
#------------------------------------------------------------------------------------------#

"""
    system(:aniso_robin, Ωₕ; closure = :padded_extrapolate, mixed = :centered, K, α, β,
        exact) -> (A::SparseMatrixCSC{Float64, Int}, F::Vector{Float64})

Ghosts eliminated, rows times `ĥ_x ĥ_y`: `A = W (−P E)`, `F = W (f + P d)` on the physical
nodes in mesh order, `f = −(k₁₁ u_xx + 2k₁₂ u_xy + k₂₂ u_yy)`.
"""
function system(problem::Symbol, Ωₕ; closure::Symbol = :padded_extrapolate,
        mixed::Symbol = :centered, K = DEFAULT_K, exact = DEFAULT_EXACT, kwargs...)
    check_problem(problem)
    x, y = axis_points(Ωₕ, 1), axis_points(Ωₕ, 2)
    P = pde_operator(Ωₕ; closure, mixed, K)
    E, d = ghost_closure(Ωₕ; closure, K, exact, kwargs...)
    W = Diagonal(vec(O1.cell_measures(Ωₕ(1)) * O1.cell_measures(Ωₕ(2))'))
    f = vec([-(K[1, 1] * exact.uxx(xi, yj) + 2K[1, 2] * exact.uxy(xi, yj) +
               K[2, 2] * exact.uyy(xi, yj)) for xi in x, yj in y])
    A = sparse(W * (-P * E))
    F = W * (f .+ P * d)
    return A, F
end

"""
    system_matrix(:aniso_robin, Ωₕ; closure, mixed, kwargs...) -> SparseMatrixCSC{Float64, Int}

The matrix of `system`.
"""
system_matrix(problem::Symbol, Ωₕ; kwargs...) = first(system(problem, Ωₕ; kwargs...))

"""
    solve(:aniso_robin, Ωₕ; kwargs...) -> Vector{Float64}

The discrete solution on the physical nodes, mesh order.
"""
function solve(problem::Symbol, Ωₕ; kwargs...)
    A, F = system(problem, Ωₕ; kwargs...)
    return A \ F
end

"""
    discrete_error(:aniso_robin, Ωₕ; exact, kwargs...) -> Float64

`(‖D₋ₓe‖_+² + ‖D₋ᵧe‖_+² + ‖e‖_{H,*}²)^{1/2}`, `e = u_h − R_H u`, each `+` norm weighted by
the backward spacing along its axis times the cell measure across it.
"""
function discrete_error(problem::Symbol, Ωₕ; exact = DEFAULT_EXACT, kwargs...)
    nx, ny = sizes(Ωₕ)
    x, y = axis_points(Ωₕ, 1), axis_points(Ωₕ, 2)
    e = reshape(solve(problem, Ωₕ; exact, kwargs...), nx, ny) .- [exact.u(a, b) for a in x, b in y]
    ĥx, ĥy = O1.cell_measures(Ωₕ(1)), O1.cell_measures(Ωₕ(2))
    hx, hy = diff(x), diff(y)
    s = sum(ĥx .* abs2.(e) .* ĥy')
    s += sum(abs2.(diff(e; dims = 1)) ./ hx .* ĥy')
    s += sum(ĥx .* abs2.(diff(e; dims = 2)) ./ hy')
    return sqrt(s)
end

#------------------------------------------------------------------------------------------#
# Mesh sequences and observed orders
#------------------------------------------------------------------------------------------#

"""
    make_mesh(kind::Symbol, level::Int; n0 = 17, seed = 20261014) -> MeshnD{2}

The unit square with x points from `GhostOracle1D.make_mesh(kind, level; n0, seed)` and y
points from the same with `seed + 1`, mirrored (`1 − x` reversed) for `:graded` so the two
axes grade towards opposite sides.
"""
function make_mesh(kind::Symbol, level::Int; n0::Int = 17, seed::Int = 20261014)
    px = collect(Float64, points(O1.make_mesh(kind, level; n0, seed)))
    py = collect(Float64, points(O1.make_mesh(kind, level; n0, seed = seed + 1)))
    kind === :graded && (py = reverse(1 .- py))
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (length(px), length(py)),
        (true, true))
    change_points!(Ωₕ, (px, py))
    return Ωₕ
end

hmax(Ωₕ) = max(O1.hmax(Ωₕ(1)), O1.hmax(Ωₕ(2)))

"""
    observed_errors(:aniso_robin, meshkind; dim = 2, levels = 4, n0 = 17, seed, kwargs...)
        -> (hs::Vector{Float64}, errs::Vector{Float64})

`hₘₐₓ` and `discrete_error` over `levels` meshes of `make_mesh(meshkind, ℓ)`.
"""
function observed_errors(problem::Symbol, meshkind::Symbol; dim::Int = 2, levels::Int = 4,
        n0::Int = 17, seed::Int = 20261014, kwargs...)
    dim == 2 || throw(ArgumentError("GhostCorners solves 2D only (dim = $dim)"))
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
    observed_order(:aniso_robin, meshkind; closure, mixed = :centered, dim = 2, levels = 4,
        kwargs...) -> Float64

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

end # module GhostCorners
