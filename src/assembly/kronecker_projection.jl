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
# distributed first, giving one Kronecker product per pair of addends, and an `inner_Γ`
# weight is a sum over its faces, giving one more per face (`_kron_inners`).
#
# A grid-function coefficient whose values vary along one axis `d` only is the diagonal
# `I ⊗ ... ⊗ diag(g_d) ⊗ ... ⊗ I`, so it projects to its 1D values on `d`, at the same place
# in the chain, and to the identity on every other axis (`_kron_coefs`). Its values are
# read once, when the factors are built.
#
# Anything without a method here answers `nothing`: a grid-function coefficient varying
# along two axes or living on another mesh, a region other than `:interior`, an indexed
# (composite) leaf, an interpolation, a scalar inside a side other than a number or a `Ref`,
# and the star and cross-weighted differences. A false negative only forgoes the Kronecker
# fast path; a false positive would be a wrong product.

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
    _kron_axis(op::LazyOp, ::Val{d}, leaf::LazyOp{1}) -> LazyOp{1}

The 1D chain that `op`, a sum-free chain from `_kron_split`, becomes on axis `d`, with its
trial or test function replaced by `leaf`: a node along `d` becomes its 1D counterpart, a
node along another axis becomes the identity, and an `:interior` restriction stays one.
That last step is only exact when the mesh's `:interior` marker is the product of the axes'
own, which a domain can break by redefining `:interior`; `_kron_interior_is_tensor` checks
it before `_kron_project` uses this. The axis is a `Val`, so that whether a node lies along
it is known when the chain is compiled and the 1D chain has one type.
"""
_kron_axis(::Union{TrialFunction, TestFunction}, ::Val, leaf::LazyOp{1}) = leaf
_kron_axis(op::RegionRestriction, vd::Val, leaf::LazyOp{1}) = restrict_to(
    op.region, _kron_axis(op.inner_op, vd, leaf))

function _kron_axis(op::ShiftNode{D, Dim}, vd::Val{d}, leaf::LazyOp{1}) where {D, Dim, d}
    x = _kron_axis(op.inner_op, vd, leaf)
    return d == Dim ? ShiftNode{1, 1, typeof(x)}(op.shift_amount, x) : x
end

_kron_split(op::ShiftNode{D, Dim}) where {D, Dim} = 1 <= Dim <= D ? _kron_split_wrapped(op) : nothing
_kron_rewrap(op::ShiftNode{D, Dim}, x::LazyOp{D}) where {D, Dim} = ShiftNode{
    D, Dim, typeof(x)}(op.shift_amount, x)

# The node families whose nD node is their 1D node along `Dim` and the identity along every
# other axis; all share the shape `Node{D, Dim, OpType}(inner_op)` (`node_family.jl`). A node
# along an axis the mesh does not have (`Dim > D`) splits to `nothing`: on no axis would it
# meet `d == Dim`, and it would project to the identity.
for N in (:BackwardDifference, :ForwardDifference, :CenteredDifference, :BackwardAverage,
    :ForwardAverage, :CenteredAverage, :JumpNode)
    @eval begin
        function _kron_split(op::$N{D, Dim}) where {D, Dim}
            return 1 <= Dim <= D ? _kron_split_wrapped(op) : nothing
        end
        _kron_rewrap(::$N{D, Dim}, x::LazyOp{D}) where {D, Dim} = $N{D, Dim, typeof(x)}(x)
        function _kron_axis(op::$N{D, Dim}, vd::Val{d}, leaf::LazyOp{1}) where {D, Dim, d}
            x = _kron_axis(op.inner_op, vd, leaf)
            return d == Dim ? $N{1, 1, typeof(x)}(x) : x
        end
    end
end

# --- Constant scalars inside a side ----------------------------------------------- #

# A plain number scaling a node inside a side (what the simplifier leaves when it merges like
# terms, `D₋ₓ(u) + 1.3 * u`) is the same number times the identity on every axis, so it goes
# into the 1D chain on axis 1 alone. A `Ref` there (`u + a * D₋ₓ(u)`, merged from
# `innerₕ(u, v) + a * innerₕ(D₋ₓ(u), v)`) is a constant too, so it commutes with every node
# of its chain, but folded into a factor it would be read once, where a `Ref` coefficient
# must stay live: `_kron_live` takes it out of the chain, and `_kron_project` returns it
# beside the factors. Any other scalar is refused.
_kron_split(op::OperatorScale{D, <:_NormalScalar}) where {D} = _kron_split_wrapped(op)
function _kron_rewrap(op::OperatorScale{D}, x::LazyOp{D}) where {D}
    return OperatorScale{D, typeof(op.scalar), typeof(x)}(op.scalar, x)
end
function _kron_axis(op::OperatorScale, vd::Val{d}, leaf::LazyOp{1}) where {d}
    x = _kron_axis(op.inner_op, vd, leaf)
    return d == 1 ? OperatorScale{1, typeof(op.scalar), typeof(x)}(op.scalar, x) : x
end

"""
    _kron_live(op::LazyOp) -> Tuple{Tuple, LazyOp}

The `Ref` scalars of `op`, a sum-free chain from `_kron_split`, and the chain without them,
which no longer reads a `Ref`. The caller multiplies the term by them at `mul!` time, as it
does a `Ref` that `_kron_leaves` strips from around a whole term.
"""
_kron_live(op::Union{TrialFunction, TestFunction}) = ((), op)
function _kron_live(op::LazyOp)
    scales, inner = _kron_live(op.inner_op)
    return scales, _kron_rewrap(op, inner)
end
function _kron_live(op::OperatorScale{D, <:Base.RefValue}) where {D}
    scales, inner = _kron_live(op.inner_op)
    return (op.scalar, scales...), inner
end

# --- Single-axis grid-function coefficients ----------------------------------------- #

# A `GridFunctionScale` coefficient after `_kron_coefs`: the values along `axis`, the only
# axis the D-dimensional values vary along (any axis for a constant one).
struct _KronCoef{T}
    axis::Int
    values::Vector{T}
end

_kron_split(op::GridFunctionScale) = _kron_split_wrapped(op)
function _kron_rewrap(op::GridFunctionScale{D, V}, x::LazyOp{D}) where {D, V}
    return GridFunctionScale{D, V, typeof(x)}(op.grid_function, x)
end

function _kron_axis(
        op::GridFunctionScale{D, <:_KronCoef}, vd::Val{d}, leaf::LazyOp{1}) where {D, d}
    x = _kron_axis(op.inner_op, vd, leaf)
    c = op.grid_function
    return d == c.axis ? GridFunctionScale(c.values, x) : x
end

"""
    _kron_coef(g, Ωₕ::MeshnD) -> Union{Nothing, _KronCoef}

The 1D coefficient a `GridFunctionScale`'s `g` projects to on `Ωₕ`, or `nothing` when its
values vary along more than one axis or `g` lives on a mesh other than `Ωₕ`. The test is
exact: every slice along the other axes must be `isequal` to the first, which
`Rₕ(Wₕ, x -> f(x[d]))` meets because the points on an axis line share that coordinate
exactly. A tolerance would turn a near miss into a wrong product.
"""
_kron_coef(::Any, ::MeshnD) = nothing
_kron_coef(g::Number, Ωₕ::MeshnD) = _KronCoef(1, fill(g, npoints(Ωₕ, Tuple)[1]))
function _kron_coef(g::VectorElement, Ωₕ::MeshnD)
    space(g) isa ScalarGridSpace && mesh(space(g)) === Ωₕ || return nothing
    return _kron_coef(parent(g), Ωₕ)
end
function _kron_coef(g::AbstractVector, Ωₕ::MeshnD{D}) where {D}
    np = npoints(Ωₕ, Tuple)
    length(g) == prod(np) || return nothing
    A = reshape(collect(g), np)
    # Whether every value equals the one at index 1 along axis `e`, the same point otherwise.
    same_along(e) = all(
        I -> isequal(A[I], A[ntuple(k -> k == e ? 1 : I[k], Val(D))...]), CartesianIndices(A))
    varying = filter(e -> !same_along(e), 1:D)
    length(varying) > 1 && return nothing
    d = isempty(varying) ? 1 : only(varying)
    return _KronCoef(d, A[ntuple(e -> e == d ? Colon() : 1, Val(D))...])
end

"""
    _kron_coefs(op::LazyOp, Ωₕ::MeshnD) -> Union{Nothing, LazyOp}

`op`, a sum-free chain from `_kron_split`, with each `GridFunctionScale` coefficient in it
replaced by its `_KronCoef` on `Ωₕ`; `nothing` when one has none.
"""
_kron_coefs(op::Union{TrialFunction, TestFunction}, ::MeshnD) = op
function _kron_coefs(op::LazyOp, Ωₕ::MeshnD)
    inner = _kron_coefs(op.inner_op, Ωₕ)
    inner === nothing && return nothing
    return _kron_rewrap(op, inner)
end
function _kron_coefs(op::GridFunctionScale{D}, Ωₕ::MeshnD) where {D}
    c = _kron_coef(op.grid_function, Ωₕ)
    c === nothing && return nothing
    inner = _kron_coefs(op.inner_op, Ωₕ)
    inner === nothing && return nothing
    return GridFunctionScale{D, typeof(c), typeof(inner)}(c, inner)
end

# Each chain with its coefficients replaced, or `nothing` when one chain has none.
function _kron_coefs(chains::Tuple, Ωₕ::MeshnD)
    out = map(op -> _kron_coefs(op, Ωₕ), chains)
    # `something` drops `Nothing` from each entry's type, as no entry is `nothing` there.
    return any(isnothing, out) ? nothing : map(something, out)
end

# --- The 1D inner products on each axis --------------------------------------------- #

"""
    _kron_inners(w, Ωₕ::MeshnD{D}) -> Union{Nothing, Tuple, Vector}

One `NTuple{D}` of 1D inner products per Kronecker term the weight `w`, an instance or a
singleton type, expands into, the
entry on axis `d` the product the factor on that axis is assembled with: a tuple of them, a
vector for `InnerGamma`, whose number of faces is known at run time. `nothing` for a
weight with no such expansion.

`InnerH` weighs every axis by its own `innerₕ` weights; `InnerPlus{Dim}` uses `inner₊` on
`Dim` and `innerₕ` elsewhere, the factorisation `kronecker_operator` already relies on.
`InnerGamma` is a sum over the faces its `mask` names, each one Kronecker term: the 1D
`inner_Γ` on that face's endpoint along its axis (weight 1 there, 0 elsewhere) and `innerₕ`
on every other axis, whose weights multiply to the face's transverse measure. A point on two
faces (a corner, a 3D edge) weighs the sum of its faces' measures (`_surface_weight`), so
the face terms add with no correction. `mask` is geometric, resolved from the face symbols
when the form was built, so a domain that redefines `:xmin` does not change it.
"""
_kron_inners(::Type, ::MeshnD) = nothing
_kron_inners(w::AbstractInnerProduct, Ωₕ::MeshnD) = _kron_inners(typeof(w), Ωₕ)
_kron_inners(::Type{InnerH}, ::MeshnD{D}) where {D} = (ntuple(_ -> innerₕ, Val(D)),)
function _kron_inners(::Type{InnerPlus{Dim}}, ::MeshnD{D}) where {Dim, D}
    1 <= Dim <= D || return nothing
    return (ntuple(d -> d == Dim ? inner₊ : innerₕ, Val(D)),)
end

# The 1D product on one axis of an `inner_Γ` face term: `inner_Γ` on the face endpoint
# `side` (1 the minimum, 2 the maximum), or `innerₕ` for `side == 0`. One type for every
# entry, so the faces of a mask, which are known only at run time, share one tuple type;
# `_kron_axis_assemble` turns `side` into a product of one type before it builds a form.
struct _KronFaceInner
    side::Int
end

function _kron_inners(w::InnerGamma, Ωₕ::MeshnD{D}) where {D}
    np = npoints(Ωₕ, Tuple)
    mask = w.mask
    # On a one-point axis both faces are the same point, which the D-dimensional weight
    # counts once and two face terms would count twice.
    any(d -> mask[d][1] && mask[d][2] && np[d] < 2, 1:D) && return nothing
    faces = [ntuple(e -> _KronFaceInner(e == d ? s : 0), Val(D))
             for d in 1:D for s in 1:2 if mask[d][s]]
    return faces
end

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
    _kron_axis_space(Ωₕ::MeshnD, d::Int) -> ScalarGridSpace{1}

The space the axis-`d` factors are assembled on: `gridspace(Ωₕ(d))`, its submesh moved to a
serial host backend (sharing its arrays) when `Ωₕ` runs another policy. A threaded 1D
assembly rounds differently, and the factors must be the same whatever policy the operator
applies them under.
"""
function _kron_axis_space(Ωₕ::MeshnD, d::Int)
    m = Ωₕ(d)
    execution_policy(backend(m)) isa CpuSerial && return gridspace(m)
    return gridspace(_rebackend(m, backend(eltype(m)), identity))
end

"""
    _kron_project(term::LazyOp, Ωₕ::MeshnD) -> Union{Nothing, Tuple}

The Kronecker factors of `term`, one addend of `resolve_form_ast` with its scalar
coefficients already stripped by `_kron_leaves`, over the host tensor mesh `Ωₕ`: a tuple
of `(scales, factors)` pairs, `factors` an `NTuple{D, SparseMatrixCSC}` whose `kron` (axis 1
fastest) is one Kronecker product and `scales` the `Ref` scalars inside its sides
(`_kron_live`), so that `term` assembles to the sum of `_kron_coeff(scales)` times each
product. The factor on axis
`d` is the 1D form on `gridspace(Ωₕ(d))` that `term` projects to there (see this file's
header); there is one tuple per pair of addends of the two sides and per Kronecker term of
the weight (several for `inner_Γ`, one per face). `nothing` when `term` has a node with no
projection, a grid-function coefficient that varies along more than one axis or lives on
another mesh, or restricts to `:interior` on a mesh whose `:interior` marker is not the
product of its axes' own. A coefficient's values are read here, once: editing them later
does not change the factors.

`Ωₕ` must be the mesh both the trial and the test space of `term`'s form live on: the
factors are built on `Ωₕ`'s axes alone, so another mesh of the same size gives other
factors without any error.
"""
_kron_project(::LazyOp, ::Any) = nothing

function _kron_project(term::BilinearProduct{D}, Ωₕ::MeshnD{D}) where {D}
    inners = _kron_inners(term.inner, Ωₕ)
    inners === nothing && return nothing
    ls = _kron_split(term.left_op)
    ls === nothing || (ls = _kron_coefs(ls, Ωₕ))
    ls === nothing && return nothing
    rs = _kron_split(term.right_op)
    rs === nothing || (rs = _kron_coefs(rs, Ωₕ))
    rs === nothing && return nothing
    if any(_kron_restricts, ls) || any(_kron_restricts, rs)
        _kron_interior_is_tensor(Ωₕ) || return nothing
    end
    spaces = ntuple(d -> _kron_axis_space(Ωₕ, d), Val(D))
    return _kron_project_terms(ls, rs, inners, spaces)
end

# The `(scales, factors)` pair of every chain `l` of `ls`, `r` of `rs` and weight `w` of
# `inners`, `l` fastest and `w` slowest, each factor assembled on its space of `spaces`.
# The pairs are built over tuples, the axis as a `Val`, so that every factor's form has one
# type; the faces of an `inner_Γ` weight, a vector, are splatted into the tuple once.
function _kron_project_terms(
        ls::Tuple, rs::Tuple, inners, spaces::Tuple{Vararg{Any, D}}) where {D}
    axes_v = ntuple(Val, Val(D))
    pair(l, r, w) = _kron_project_pair(_kron_live(l), _kron_live(r), w, spaces, axes_v)
    per_w(w) = _kron_cat(map(r -> map(l -> pair(l, r, w), ls), rs)...)
    return _kron_cat_each(per_w, inners)
end

function _kron_project_pair((ls_, l), (rs_, r), w, spaces, axes_v)
    factors = map((Wd, vd) -> _kron_axis_factor(Wd, w, l, r, vd), spaces, axes_v)
    return ((ls_..., rs_...), factors)
end

# The tuple of `f(x)`'s entries for every `x` of `t`, in order: entry by entry over a
# tuple, and over a vector (the faces of `inner_Γ`, known at run time) one splat at the end.
_kron_cat_each(f, t::Tuple) = _kron_cat(map(f, t)...)
_kron_cat_each(f, t::Vector) = (Iterators.flatten(map(f, t))...,)

_kron_cat() = ()
_kron_cat(a::Tuple, b::Tuple...) = (a..., _kron_cat(b...)...)

"""
    _kron_axis_factor(Wd, w, l, r, ::Val{d}) -> SparseMatrixCSC

The factor on axis `d` of the term `w[d](L, R)`: the 1D form on `Wd` whose sides are the
chains `l` and `r` project to on `d` (`_kron_axis`), assembled.
"""
_kron_axis_factor(Wd, w, l, r, vd::Val{d}) where {d} = _kron_axis_assemble(Wd, w[d], l, r, vd)

function _kron_axis_assemble(Wd, inner, l, r, vd::Val)
    return assemble(form(Wd, Wd, (u, v) -> inner(_kron_axis(l, vd, u), _kron_axis(r, vd, v))))
end
function _kron_axis_assemble(Wd, w::_KronFaceInner, l, r, vd::Val)
    w.side == 0 && return _kron_axis_assemble(Wd, innerₕ, l, r, vd)
    w.side == 1 &&
        return _kron_axis_assemble(Wd, (a, b) -> inner_Γ(a, b; markers = :xmin), l, r, vd)
    return _kron_axis_assemble(Wd, (a, b) -> inner_Γ(a, b; markers = :xmax), l, r, vd)
end
