# kronecker_projection.jl
#
# `_kron_project`: the per-axis factors of one addend of a resolved bilinear form, found by
# projecting the addend onto each axis of a tensor mesh (gpena/Bramble.jl#427).
#
# A directional node along axis `Dim` acts on the grid as `I ⊗ ... ⊗ A ⊗ ... ⊗ I`, with its
# 1D matrix `A` in slot `Dim`. A chain of such nodes is still one Kronecker product, and the
# `innerₕ`/`inner₊` weights are products of per-axis weights, so a product `⟨L(u), R(v)⟩`
# with `L` and `R` chains is `⊗_d (R_dᵀ W_d L_d)`: on axis `d`, the 1D form whose trial
# side keeps only the nodes of `L` along `d`, and likewise for `R`. Each factor is assembled
# from that 1D form on `gridspace(Ωₕ(d))`, so its boundary rows are the ones the 1D
# operator defines, not a second implementation of them. The trial and test sides project
# independently, which is what covers mixed and advection terms. A sum inside a side is
# distributed first, giving one Kronecker product per pair of addends.
#
# Anything without a method here answers `nothing`: a grid-function coefficient (no tensor
# structure in general), a region other than `:interior`, a surface weight, an indexed
# (composite) leaf, an interpolation, a scalar inside a side, and the star and
# cross-weighted differences. A false negative only forgoes the Kronecker fast path; a false
# positive would be a wrong product.

# --- Splitting a side into sum-free chains ------------------------------------------- #

"""
    _kron_split(op::LazyOp) -> Union{Nothing, Tuple}

The addends of one side of a `BilinearProduct` once every `OperatorAdd` in it is
distributed over the nodes around it (`D₋ₓ(u + w)` gives `(D₋ₓ(u), D₋ₓ(w))`), each a chain
of nodes `_kron_axis` can project. `nothing` when some node in `op` has no projection.
"""
_kron_split(::LazyOp) = nothing
_kron_split(op::Union{TrialFunction, TestFunction}) = (op,)
function _kron_split(op::OperatorAdd)
    l = _kron_split(op.left_op)
    l === nothing && return nothing
    r = _kron_split(op.right_op)
    r === nothing && return nothing
    return (l..., r...)
end
function _kron_split(op::RegionRestriction)
    op.region === :interior || return nothing
    return _kron_split_wrapped(op)
end

function _kron_split_wrapped(op)
    inner = _kron_split(op.inner_op)
    inner === nothing && return nothing
    return map(x -> _kron_rewrap(op, x), inner)
end

_kron_rewrap(op::RegionRestriction, x::LazyOp) = restrict_to(op.region, x)

# --- Projecting a chain onto one axis ------------------------------------------------ #

"""
    _kron_axis(op::LazyOp, d::Int, leaf::LazyOp{1}) -> LazyOp{1}

The 1D chain that `op`, a sum-free chain from `_kron_split`, becomes on axis `d`, with its
trial or test function replaced by `leaf`: a node along `d` becomes its 1D counterpart, a
node along another axis becomes the identity, and an `:interior` restriction stays one.
That last step is only exact when the mesh's `:interior` marker is the product of the axes'
own, which a domain can break by redefining `:interior`; `_kron_interior_is_tensor` checks
it before `_kron_project` uses this.
"""
_kron_axis(::Union{TrialFunction, TestFunction}, ::Int, leaf::LazyOp{1}) = leaf
_kron_axis(op::RegionRestriction, d::Int, leaf::LazyOp{1}) = restrict_to(
    op.region, _kron_axis(op.inner_op, d, leaf))

function _kron_axis(op::ShiftNode{D, Dim}, d::Int, leaf::LazyOp{1}) where {D, Dim}
    x = _kron_axis(op.inner_op, d, leaf)
    return d == Dim ? ShiftNode{1, 1, typeof(x)}(op.shift_amount, x) : x
end

_kron_split(op::ShiftNode) = _kron_split_wrapped(op)
_kron_rewrap(op::ShiftNode{D, Dim}, x::LazyOp{D}) where {D, Dim} = ShiftNode{
    D, Dim, typeof(x)}(op.shift_amount, x)

# The node families whose nD node is their 1D node along `Dim` and the identity along every
# other axis; all share the shape `Node{D, Dim, OpType}(inner_op)` (`node_family.jl`).
for N in (:BackwardDifference, :ForwardDifference, :CenteredDifference, :BackwardAverage,
    :ForwardAverage, :CenteredAverage, :JumpNode)
    @eval begin
        _kron_split(op::$N) = _kron_split_wrapped(op)
        _kron_rewrap(::$N{D, Dim}, x::LazyOp{D}) where {D, Dim} = $N{D, Dim, typeof(x)}(x)
        function _kron_axis(op::$N{D, Dim}, d::Int, leaf::LazyOp{1}) where {D, Dim}
            x = _kron_axis(op.inner_op, d, leaf)
            return d == Dim ? $N{1, 1, typeof(x)}(x) : x
        end
    end
end

# --- The 1D inner product on each axis ----------------------------------------------- #

# `InnerH` weighs every axis by its own `innerₕ` weights; `InnerPlus{Dim}` uses `inner₊` on
# `Dim` and `innerₕ` elsewhere, the factorisation `kronecker_operator` already relies on.
_kron_inner(::Type{InnerH}, ::Int, l, r) = innerₕ(l, r)
function _kron_inner(::Type{InnerPlus{Dim}}, d::Int, l, r) where {Dim}
    return d == Dim ? inner₊(l, r) : innerₕ(l, r)
end

_kron_has_inner(::Type) = false
_kron_has_inner(::Type{InnerH}) = true
_kron_has_inner(::Type{<:InnerPlus}) = true

# --- The `:interior` marker -------------------------------------------------------- #

# Whether a sum-free chain from `_kron_split` holds a region restriction.
_kron_restricts(::Union{TrialFunction, TestFunction}) = false
_kron_restricts(::RegionRestriction) = true
_kron_restricts(op::LazyOp) = _kron_restricts(op.inner_op)

# `RegionRestriction` reads `Ωₕ`'s own `:interior` marker, which a domain may redefine
# (`_default_geometric_marker!` keeps a custom one), while each 1D factor reads its axis
# submesh's. The projection is exact only when the first is the `kron` of the others.
function _kron_interior_is_tensor(Ωₕ::MeshnD{D}) where {D}
    m = markers(Ωₕ)
    haskey(m, :interior) || return false
    axes_m = ntuple(d -> markers(Ωₕ(d)), Val(D))
    all(md -> haskey(md, :interior), axes_m) || return false
    return m[:interior] == foldl(kron, reverse(map(md -> Vector(md[:interior]), axes_m)))
end

# --- The projection ------------------------------------------------------------------ #

"""
    _kron_project(term::LazyOp, Ωₕ::MeshnD) -> Union{Nothing, Tuple}

The Kronecker factors of `term`, one addend of `resolve_form_ast` with its scalar
coefficients already stripped by `_kron_leaves`, over the host tensor mesh `Ωₕ`: a tuple
of per-axis factor tuples, each an `NTuple{D, SparseMatrixCSC}` whose `kron` (axis 1
fastest) is one Kronecker product, so that `term` assembles to their sum. The factor on axis
`d` is the 1D form on `gridspace(Ωₕ(d))` that `term` projects to there (see this file's
header). `nothing` when `term` has a node with no projection, or restricts to
`:interior` on a mesh whose `:interior` marker is not the product of its axes' own.

`Ωₕ` must be the mesh both the trial and the test space of `term`'s form live on: the
factors are built on `Ωₕ`'s axes alone, so another mesh of the same size gives other
factors without any error.
"""
_kron_project(::LazyOp, ::Any) = nothing

function _kron_project(term::BilinearProduct{D, I}, Ωₕ::MeshnD{D}) where {D, I}
    _kron_has_inner(I) || return nothing
    ls = _kron_split(term.left_op)
    ls === nothing && return nothing
    rs = _kron_split(term.right_op)
    rs === nothing && return nothing
    if any(_kron_restricts, ls) || any(_kron_restricts, rs)
        _kron_interior_is_tensor(Ωₕ) || return nothing
    end
    spaces = ntuple(d -> gridspace(Ωₕ(d)), Val(D))
    pairs = vec([(l, r) for l in ls, r in rs])
    return Tuple(map(pairs) do (l, r)
        ntuple(Val(D)) do d
            Wd = spaces[d]
            f = (u, v) -> _kron_inner(I, d, _kron_axis(l, d, u), _kron_axis(r, d, v))
            assemble(form(Wd, Wd, f))
        end
    end)
end
