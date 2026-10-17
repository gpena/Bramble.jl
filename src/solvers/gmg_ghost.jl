# gmg_ghost.jl: geometric multigrid on ghost spaces (#781).
#
# `build(W)` may return a form on ghost spaces built on `mesh(W)`, a `GhostBilinearForm`;
# every level must then do so, with ghosts on the same facets. Each level is applied by
# the ghost matrix-free operator `E_vᵀ A_pad E_u` (`ghost_matrix_free.jl`), and its
# vectors hold the level's degrees of freedom, not its points.
#
# Every ghost must be eliminated by a relation, so each degree of freedom is the value of
# one physical node (a node a Dirichlet relation fixes is none), read off `E`. The physical
# node's padded row of `E` is that degree of freedom alone, with coefficient 1. A closure
# whose physical rows are not, or a ghost kept as an unknown, is refused. The transfers then
# act on the physical grid, where the levels nest. Prolongation writes the coarse degrees of
# freedom at their nodes (0 at a fixed node, as the closure's linear part gives), runs
# `prolongate!` and reads the fine degrees of freedom at theirs. Restriction is its exact
# transpose through `coarsen!`. A fine physical node lies between coarse physical nodes, so
# interpolating the padded grid instead would give the same fine values: the ghosts never
# enter.
#
# The smoothers read the diagonal of the reduced matrix from `jacobi_preconditioner`, and
# `λmax` from `max_eigenvalue_estimate`, both of which take the ghost operator.
#
# The coarsest level is the assembled reduced matrix, factorised densely. When constants
# lie in its kernel (pure Neumann), it is bordered by the constant, `[A e; eᵀ 0]`, so its
# solve returns the mean-zero solution; a fine right side `b = A x` restricts to a coarse
# one the border keeps solvable.
#
# See also: `GMGPreconditioner`, `GhostMatrixFreeOperator`, `GhostBilinearForm`

# Where the degrees of freedom of each level live on the physical grid, and one work vector
# per level of that grid's size.
struct _GMGGhostTransfer{V <: AbstractVector}
    "degree of freedom `k` of level `l` is the value of physical node `points[l][k]`."
    points::Vector{Vector{Int}}
    "a vector of `npoints(H[l])` entries per level."
    phys::Vector{V}
end

# --- The level forms ---------------------------------------------------------------- #

# A form on ghost spaces returned by `build` on level `l`: square, on ghost spaces over the
# level's mesh, with a closure whose degrees of freedom are physical nodes.
function _gmg_ghost_level_form(a::GhostBilinearForm, Ωl, l)
    Wu, Wv = trial_space(a), test_space(a)
    (physical_mesh(mesh(Wu)) === Ωl && physical_mesh(mesh(Wv)) === Ωl) || throw(
        ArgumentError(
        "gmg_preconditioner: build(W) must return a form on ghost spaces built on mesh(W) " *
        "(level $l): build the space inside build as gridspace(mesh(W); ghosts = ...)",
    ),
    )
    ndofs(Wu) == ndofs(Wv) || throw(
        ArgumentError(
        "gmg_preconditioner: build(W) must return a square form, got $(ndofs(Wv)) × " *
        "$(ndofs(Wu)) on level $l",
    ),
    )
    _gmg_ghost_points(Wu, l) == _gmg_ghost_points(Wv, l) || throw(
        ArgumentError(
        "gmg_preconditioner: the trial and test spaces of build(W) must have the same " *
        "degrees of freedom (level $l): use one constrained space for both",
    ),
    )
    return a
end

# The physical node of each degree of freedom of `W`, in the mesh's linear order.
function _gmg_ghost_points(W::GhostGridSpace, l)
    G = mesh(W)
    plin = LinearIndices(npoints(G, Tuple))
    Et = W.Et
    cp, rv, nz = SparseArrays.getcolptr(Et), rowvals(Et), nonzeros(Et)
    pts = zeros(Int, ndofs(W))
    for (k, J) in enumerate(physical_indices(G))
        i = plin[J]
        m = cp[i + 1] - cp[i]
        m == 0 && continue
        (m == 1 && isone(nz[cp[i]]) && iszero(pts[rv[cp[i]]])) || _throw_gmg_ghost_node(l)
        pts[rv[cp[i]]] = k
    end
    any(iszero, pts) && _throw_gmg_ghost_unknowns(l)
    return pts
end

@noinline function _throw_gmg_ghost_node(l)
    throw(
        ArgumentError(
        "gmg_preconditioner: a physical node of the ghost space of build(W) on level $l " *
        "is not one degree of freedom alone, so the grid transfers cannot act on it. " *
        "Build the space with constrained(gridspace(mesh(W); ghosts = ...), relations).",
    ),
    )
end

@noinline function _throw_gmg_ghost_unknowns(l)
    throw(
        ArgumentError(
        "gmg_preconditioner: the ghost space of build(W) on level $l keeps ghosts as " *
        "unknowns, which no coarser level holds. Give every ghosted facet a relation " *
        "with constrained(gridspace(mesh(W); ghosts = ...), u -> (...)).",
    ),
    )
end

# Every level is a form on ghost spaces or none is; on ghost spaces, each has ghosts on the
# facets the finest one has.
_gmg_check_level_kinds(forms::AbstractVector{<:BilinearForm}) = nothing

function _gmg_check_level_kinds(forms::AbstractVector)
    last(forms) isa GhostBilinearForm || all(a -> a isa BilinearForm, forms) ||
        _throw_gmg_level_kinds()
    if last(forms) isa GhostBilinearForm
        active = mesh(trial_space(last(forms))).active
        for (l, a) in enumerate(forms)
            a isa GhostBilinearForm || _throw_gmg_level_kinds()
            mesh(trial_space(a)).active == active || throw(
                ArgumentError(
                "gmg_preconditioner: build(W) has ghosts on other facets on level $l " *
                "than on the finest level: every level needs the same ghosted facets",
            ),
            )
        end
    end
    return nothing
end

@noinline function _throw_gmg_level_kinds()
    throw(
        ArgumentError(
        "gmg_preconditioner: build(W) returns a form on ghost spaces on some levels and " *
        "on plain spaces on others: return the same kind of form on every level",
    ),
    )
end

# --- Transfers ---------------------------------------------------------------------- #

_gmg_transfer(forms::AbstractVector{<:BilinearForm}, H, ::Type) = nothing

function _gmg_transfer(forms::AbstractVector, H, ::Type{T}) where {T}
    points = [_gmg_ghost_points(trial_space(a), l) for (l, a) in enumerate(forms)]
    phys = [zeros(T, npoints(Ωl)) for Ωl in H]
    return _GMGGhostTransfer{eltype(phys)}(points, phys)
end

# `xf = P xc` and `xc = Pᵀ xf` between levels `l - 1` and `l`: the transfers of the grids
# on a plain hierarchy, through the physical nodes of the degrees of freedom on ghost
# spaces.
@inline _gmg_prolongate!(::Nothing, xf, H, l, xc) = prolongate!(xf, H, l, xc)
@inline _gmg_restrict!(::Nothing, xc, H, l, xf) = coarsen!(xc, H, l, xf)

function _gmg_prolongate!(t::_GMGGhostTransfer, xf, H, l, xc)
    pc, pf = t.phys[l - 1], t.phys[l]
    fill!(pc, zero(eltype(pc)))
    _gmg_scatter!(pc, t.points[l - 1], xc)
    prolongate!(pf, H, l, pc)
    _gmg_gather!(xf, t.points[l], pf)
    return xf
end

function _gmg_restrict!(t::_GMGGhostTransfer, xc, H, l, xf)
    pc, pf = t.phys[l - 1], t.phys[l]
    fill!(pf, zero(eltype(pf)))
    _gmg_scatter!(pf, t.points[l], xf)
    coarsen!(pc, H, l, pf)
    _gmg_gather!(xc, t.points[l - 1], pc)
    return xc
end

@inline function _gmg_scatter!(p, pts, x)
    @inbounds for k in eachindex(pts)
        p[pts[k]] = x[k]
    end
    return p
end

@inline function _gmg_gather!(x, pts, p)
    @inbounds for k in eachindex(pts)
        x[k] = p[pts[k]]
    end
    return x
end

# --- The coarsest level ------------------------------------------------------------- #

# The dense factorisation of `A` bordered by the constant `e`, `[A e; eᵀ 0]`, and the
# vector of `n + 1` entries its solve works in.
struct _GMGBorderedLU{F, V <: AbstractVector}
    lu::F
    w::V
end

function ldiv!(B::_GMGBorderedLU, x::AbstractVector)
    w = B.w
    n = length(x)
    copyto!(w, 1, x, 1, n)
    w[n + 1] = zero(eltype(w))
    ldiv!(B.lu, w)
    copyto!(x, 1, w, 1, n)
    return x
end

# The coarsest level's reduced matrix, factorised densely, and bordered by the constant
# when the constant lies in its kernel.
function _gmg_coarse_factor(::Type{T}, a::GhostBilinearForm) where {T}
    A = Matrix{T}(assemble(a))
    n = size(A, 1)
    _gmg_constant_in_kernel(A) || return lu(A)
    e = ones(T, n)
    B = [A e; transpose(e) zero(T)]
    return _GMGBorderedLU(lu(B), zeros(T, n + 1))
end

# Whether every row of `A` sums to zero up to the rounding of its own entries: then the
# constant is in the kernel (pure Neumann, no reaction). Each row sum is a handful of
# entries whose exact sum is 0, so its rounding is a few `eps` times the row's absolute sum;
# the tolerance `64 eps` per row keeps a weak Robin `α` or a small reaction `c`, which add
# `α` or `c` times a cell weight to some row, out of it down to `α, c ~ 1e-12` relative to
# the row (in `Float64`), where a test on the whole of `A e` against `√eps ‖A‖` bordered a
# nonsingular matrix and the cycle stalled.
function _gmg_constant_in_kernel(A::AbstractMatrix{T}) where {T}
    tol = 64 * eps(real(T))
    for i in axes(A, 1)
        s, m = zero(T), zero(real(T))
        for j in axes(A, 2)
            s += A[i, j]
            m += abs(A[i, j])
        end
        abs(s) <= tol * m || return false
    end
    return true
end

# --- Smoothers ---------------------------------------------------------------------- #

@inline _gmg_trial_space(op::GhostMatrixFreeOperator) = op.trial_space

function red_black_gauss_seidel(::GhostMatrixFreeOperator; sweeps::Integer = 1)
    throw(
        ArgumentError(
        "red_black_gauss_seidel does not take a ghost-space operator: its closure couples " *
        "a facet node to nodes of its own colour through the ghosts. Use jacobi_smoother " *
        "or chebyshev_smoother, as gmg_preconditioner does by default.",
    ),
    )
end
