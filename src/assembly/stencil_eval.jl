# stencil_eval.jl

# ==============================================================================
# 1. Zero-Allocation Stencil Evaluators
# ==============================================================================

#=
The three-tap template (gpena/Bramble.jl#70).

Every difference, average and jump node evaluates to the same shape: ordered taps drawn
from {+1, 0, -1}, each the inner operator's stencil either as-is (tap 0) or relabelled to a
neighbour, scaled by a per-node weight and concatenated in order. Eight bodies wrote that
out; a node now declares `_stencil_taps` and `_stencil_weights` in its own file and this is
the only evaluator.

Two properties this has to preserve, both load-bearing:

  - **Taps are a compile-time ordered tuple.** They are `Val`s so `_tap_stencil` can
    dispatch the tap-0 case away from the shifted one -- `shifted_inner_stencil` needs its
    delta as a type. A node declares exactly its own taps and no padding: a zero-weight tap
    added to make shapes uniform would widen `stencil_offsets`, which sizes the sparsity
    pattern and the parallel colouring strides.
  - **Concatenation order is entry-for-entry what it was.** `concatenate_stencils` is
    `(left..., right...)`, so folding right-to-left gives the same flat order the
    hand-written left-nested calls did.

`stencil_offsets` (`ast/stencil_pattern.jl`) now reads the same `_stencil_taps`, so a
node's reach and its stencil cannot disagree -- they used to be spelled out twice.
=#

# One tap. Tap 0 is the inner stencil itself; any other is the inner stencil relabelled to
# that neighbour, which is exactly what `shifted_inner_stencil` decides by trait.
@inline _tap_stencil(op, inner, space, I, markers, ::Val{Dim}, ::Val{0}, w) where {Dim} = scale_stencil(inner, w)

@inline _tap_stencil(
    op, inner, space, I, markers, ::Val{Dim}, ::Val{Delta}, w
) where {Dim, Delta} = scale_stencil(
    shifted_inner_stencil(op.inner_op, inner, space, I, markers, Val(Dim), Val(Delta)),
    w
)

# Recursion rather than a `foldl`, so the tuple length is consumed at compile time and the
# stencil tuple type stays concrete through every step.
@inline _fold_taps(op, inner, space, I, markers, vdim, taps::Tuple{Any}, ws::Tuple{Any}) = _tap_stencil(
    op, inner, space, I, markers, vdim, taps[1], ws[1])

@inline _fold_taps(op, inner, space, I, markers, vdim, taps::Tuple, ws::Tuple) = concatenate_stencils(
    _tap_stencil(op, inner, space, I, markers, vdim, taps[1], ws[1]),
    _fold_taps(op, inner, space, I, markers, vdim, Base.tail(taps), Base.tail(ws))
)

# A node's weights in the type they join with the space's element type. A jump's are
# `Int`s, so over a restriction its tap 0 kept `Int` entries where its shifted tap, zeroed
# in that element type (`shifted_inner_stencil`), had `Float64` ones, and two such jumps
# summed (`jumpₓ(Rₕ(:l, v)) + jumpₓ(Rₕ(:r, v))`) had four stencil types, more than the
# walk splits (27360 B on a 7x9 grid). The conversion is exact; weights of that type are
# left as they are.
@inline _tap_weights(op, space, I) = map(
    w -> convert(promote_type(typeof(w), eltype(space)), w), _stencil_weights(op, space, I))

# The trial function of a mixed difference on a composite constrained as a whole, read at a
# ghost of a facet of `facets` (`facets[d][s]`, `s` 1 the minimum) by extrapolation from
# the nodes inward: the facet rule (below the corner rule), which `form` builds
# (ghost_composite.jl).
struct _FacetExtrapolated{D, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
    facets::NTuple{D, NTuple{2, Bool}}
end

"""
    UnaryWrapper{D}

The AST nodes that wrap exactly one operand in an `inner_op` field.

Fourteen node types, and the reason they are worth naming together: a query about a term
usually has the same answer for a wrapper as for the operand inside it, so each such query
used to be registered against all fourteen by hand -- one line apiece, per query
(gpena/Bramble.jl#52). A query that forgot one inherited a fallback instead, and every
fallback in this family is a plausible wrong answer rather than an error:
`test_component_or_nothing` answering `nothing` sends a term to every block.

Written as a union of concrete types rather than reached through a generic child accessor,
deliberately. Dispatch resolves it at compile time and each method still reads `op.inner_op`
directly, so nothing on the assembly path pays for the generality.

Products and sums are not members. `BilinearProduct` and `LinearProduct` hold two operands
with different roles -- trial on the left, test on the right -- so a query about the test
component reads `right_op` alone, and `OperatorAdd` has to reconcile both sides. Those stay
written out, which is the point: what is left explicit is what genuinely differs.
"""
const UnaryWrapper{D} = Union{
    BackwardDifference{D},
    ForwardDifference{D},
    CenteredDifference{D},
    StarDifference{D},
    CrossWeightedDifference{D},
    SecondDifference{D},
    JumpNode{D},
    BackwardAverage{D},
    ForwardAverage{D},
    CenteredAverage{D},
    ShiftNode{D},
    OperatorScale{D},
    GridFunctionScale{D},
    RegionRestriction{D},
    InterpolationNode{D},
    _FacetExtrapolated{D}
}

# The queries that answer for a wrapper whatever they answer for its operand. One method
# each, where every one of these was previously registered against all fourteen wrapper
# types by hand (gpena/Bramble.jl#52).
#
# They live here rather than beside their own ladders because `UnaryWrapper` names
# `InterpolationNode`, which `operators/interpolation.jl` defines -- so the union, and
# anything dispatching on it, has to come after every operator file. The node-specific
# overrides stay in their own files; `InterpolationNode` deviates from three of these and
# says so there.
#
# Two of these override a fallback declared on the *abstract* type rather than filling a
# gap: `stencil_shift_trait(::LazyOp)` answers translation-invariant and
# `_all_trial_interpolated(::LazyOp)` answers false, so a wrapper reaching either default
# is a wrong answer, not a missing one. That is the shape of mistake this collapse is meant
# to make impossible: there is now one method to get right per query, not fourteen.
stencil_shift_trait(op::UnaryWrapper) = stencil_shift_trait(op.inner_op)
_all_trial_interpolated(op::UnaryWrapper) = _all_trial_interpolated(op.inner_op)
_all_test_interpolated(op::UnaryWrapper) = _all_test_interpolated(op.inner_op)
_has_test_interp(op::UnaryWrapper) = _has_test_interp(op.inner_op)
_has_trial_interp(op::UnaryWrapper) = _has_trial_interp(op.inner_op)
_is_dirac(op::UnaryWrapper) = _is_dirac(op.inner_op)

"""
    TappedNode{D, Dim}

The nodes whose stencil is ordered taps from {+1, 0, -1} along `Dim` with per-node weights:
the one-sided, extended and second differences, the three averages, and the jump. `ShiftNode` is not
one of them -- it relabels its child's whole stencil rather than combining taps.
"""
const TappedNode{D, Dim} = Union{
    BackwardDifference{D, Dim},
    ForwardDifference{D, Dim},
    CenteredDifference{D, Dim},
    StarDifference{D, Dim},
    CrossWeightedDifference{D, Dim},
    SecondDifference{D, Dim},
    BackwardAverage{D, Dim},
    ForwardAverage{D, Dim},
    CenteredAverage{D, Dim},
    JumpNode{D, Dim}
}

@inline function local_stencil(
        op::TappedNode{D, Dim}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D, Dim}
    # The corner rule of a ghost mesh (`_needs_onesided` below); `false` off a ghost mesh.
    _needs_onesided(op, mesh(space), I) &&
        return @noinline _onesided_mixed(op, space, I, markers, lin_idx)
    return _tapped_stencil(op, space, I, markers, lin_idx)
end

@inline function _tapped_stencil(
        op::TappedNode{D, Dim}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D, Dim}
    # A tapped operand over a restriction answers with one length (`_fixed_stencil` below).
    inner = _nests_fixed(op.inner_op) ?
            _fixed_operand(op.inner_op, space, I, markers, lin_idx, Val(Dim),
        _off_tap(_stencil_taps(op))) :
            local_stencil(op.inner_op, space, I, markers, lin_idx)
    return _fold_taps(
        op,
        inner,
        space,
        I,
        markers,
        Val(Dim),
        _stencil_taps(op),
        _tap_weights(op, space, I)
    )
end

# --- The corner rule of a ghost mesh ------------------------------------------------- #
#
# A mixed difference `A_d(B_e(w))` (two first differences along different axes, `w` a trial
# or test function under any constant or grid-function scales) at a 2D corner node, or a 3D
# edge node, of a ghost mesh reads the padded corner beyond it. `A`'s tap to the facet ghost
# along `d` re-evaluates `B` there, and `B`'s tap along `e` lands outside along both axes, a
# point that is no node and no ghost (`is_absent`). At such a node the mixed difference is
# the inward one-sided one along both axes, `D_{s_d,d}(D_{s_e,e}(w))` with `s = +1` (`D₊`) at
# a first node and `-1` (`D₋`) at a last, which reads physical nodes only. `A`'s operand may
# be a linear combination of such terms and others. The rule distributes over it
# (`_rule_terms`): each mixed term that reads the corner takes the one-sided difference, and
# every other term `A`'s own taps. Elsewhere `A` reads the ghosts as the closure gives them,
# but across a facet of the facet rule below (a composite constrained as a whole, whose
# relations there read several components). Off a ghost mesh the predicate is a constant
# `false`.
#
# The one-sided stencil keeps the node's own stencil type: entry for entry in the order of
# `A`'s taps and, within each, of the terms, `A`'s taps past the two it uses and `B`'s
# likewise carry a zero weight on `I`'s own offsets, and each weight is converted to the
# type of the entry it replaces (`_like_entries`), so the walk sees one type at every point.
const _FirstDifference{D, Dim} = Union{
    BackwardDifference{D, Dim},
    ForwardDifference{D, Dim},
    CenteredDifference{D, Dim},
    StarDifference{D, Dim},
    CrossWeightedDifference{D, Dim}
}

# The terms of a difference's operand, in the order its stencil concatenates them: sums
# flattened, and a scale over a sum distributed over the summands.
@inline _rule_terms(op::OperatorAdd) = (_rule_terms(op.left_op)..., _rule_terms(op.right_op)...)
@inline _rule_terms(op::_Scale) = map(t -> _rewrap_inner(op, t), _rule_terms(op.inner_op))
@inline _rule_terms(op) = (op,)

# The axis of a term that is a first difference of a scaled trial or test function, under
# any scales; `0` for any other term. Decided by type.
@inline _cross_axis(op::_FirstDifference{D, E}) where {D, E} = _scaled_leaf(op.inner_op) ? E : 0
@inline _cross_axis(op::_Scale) = _cross_axis(op.inner_op)
@inline _cross_axis(::Any) = 0

@inline _scaled_leaf(::_AnyLeaf) = true
@inline _scaled_leaf(op::_Scale) = _scaled_leaf(op.inner_op)
@inline _scaled_leaf(::Any) = false

@inline _under_scales(op::_Scale) = _under_scales(op.inner_op)
@inline _under_scales(op) = op

# Whether a first difference along `Dim` has a mixed term across another axis to rewrite.
@inline _rule_applies(op::_FirstDifference{D, Dim}) where {D, Dim} = _has_relative_offsets(op) &&
                                                                     !_holds_restriction(op) &&
                                                                     any(
                                                                         t -> _cross_axis(t) ∉ (0, Dim), _rule_terms(op.inner_op))

# Whether the term `t` of a difference along `Dim` with taps `ataps` reads an absent point
# at `I` (`_reads_absent`, `ast/common.jl`).
@inline _term_reads_absent(t, G, I, vdim, ataps) = _cross_reads_absent(
    _under_scales(t), Val(_cross_axis(t)), G, I, vdim, ataps)
@inline _cross_reads_absent(B, ::Val{0}, G, I, ::Val, ataps) = false
@inline _cross_reads_absent(B, ::Val{E}, G, I, ::Val{Dim}, ataps) where {E, Dim} = E != Dim &&
                                                                                   _reads_absent(
    G, I, Val(Dim), ataps, Val(E), _stencil_taps(B))

@inline _any_reads_absent(::Tuple{}, G, I, vdim, ataps) = false
@inline _any_reads_absent(ts::Tuple, G, I, vdim, ataps) = _term_reads_absent(
    first(ts), G, I, vdim, ataps) || _any_reads_absent(Base.tail(ts), G, I, vdim, ataps)

@inline _needs_onesided(op, m, I::CartesianIndex) = false

@inline function _needs_onesided(
        op::_FirstDifference{D, Dim}, G::_GhostMeshLike{D}, I::CartesianIndex{D}
) where {D, Dim}
    _rule_applies(op) || return false
    I in physical_indices(G) || return false
    return _any_reads_absent(_rule_terms(op.inner_op), G, I, Val(Dim), _stencil_taps(op))
end

# The one-sided taps of a node of `N` taps along `d` at `I`, step `s`: `D_s` on taps `0` and
# `s`, `(u_{I+s} - u_I) / (x_{I+s} - x_I)`, and a zero weight on tap `0` for the rest.
@inline function _onesided_taps(m, I::CartesianIndex, d::Int, s::Int, ::Val{N}) where {N}
    h = s == 1 ? forward_spacing(m, I, d) : spacing(m, I, d)
    c = s / h
    taps = ntuple(k -> k == 2 ? s : 0, Val(N))
    ws = ntuple(k -> k == 1 ? -c : (k == 2 ? c : zero(c)), Val(N))
    return taps, ws
end

@noinline function _onesided_mixed(
        op::TappedNode{D, Dim}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D, Dim}
    G = mesh(space)
    terms = _rule_terms(op.inner_op)
    taps = _stencil_taps(op)
    hits = map(t -> _term_reads_absent(t, G, I, Val(Dim), taps), terms)
    as, αs = _onesided_taps(G, I, Dim, _inward_step(G, I, Dim), Val(length(taps)))
    at = (space, I, markers, LinearIndices(indices(G)))
    return _rule_fold(op, terms, hits, at, taps, _stencil_weights(op, space, I), as, αs)
end

# `A`'s taps in order, each the concatenation of the terms' contributions to it.
@inline _rule_fold(op, terms, hits, at, taps::Tuple{Any}, ws, as, αs) = _rule_tap(
    op, terms, hits, at, taps[1], ws[1], as[1], αs[1])
@inline _rule_fold(op, terms, hits, at, taps::Tuple, ws, as, αs) = concatenate_stencils(
    _rule_tap(op, terms, hits, at, taps[1], ws[1], as[1], αs[1]),
    _rule_fold(op, terms, hits, at, Base.tail(taps), Base.tail(ws), Base.tail(as),
        Base.tail(αs)))

@inline _rule_tap(op, terms::Tuple{Any}, hits, at, tap, w, a, α) = _term_tap(
    op, terms[1], hits[1], at, tap, w, a, α)
@inline _rule_tap(op, terms::Tuple, hits, at, tap, w, a, α) = concatenate_stencils(
    _term_tap(op, terms[1], hits[1], at, tap, w, a, α),
    _rule_tap(op, Base.tail(terms), Base.tail(hits), at, tap, w, a, α))

# One term's contribution to one tap of `A`: `A`'s own tap over the term (`_tap_stencil`),
# or, for a mixed term that reads the corner, the one-sided tap `a` (weight `α`) over the
# term's one-sided difference, with the entry types of the tap it replaces (whose values,
# read through the corner, are discarded).
@inline function _term_tap(
        op::TappedNode{D, Dim}, t, hit::Bool, at, tap, w, a::Int, α
) where {D, Dim}
    space, I, markers, lins = at
    S = _tap_stencil(_rewrap_inner(op, t), local_stencil(t, space, I, markers, lins[I]),
        space, I, markers, Val(Dim), tap, w)
    (_cross_axis(t) ∈ (0, Dim) || !hit) && return S
    return _like_entries(S, _onesided_term(t, space, I, markers, lins, Val(Dim), a, α))
end

@inline function _onesided_term(
        t, space, I::CartesianIndex{D}, markers, lins, ::Val{Dim}, a::Int, α
) where {D, Dim}
    G = mesh(space)
    B = _under_scales(t)
    E = _cross_axis(t)
    bs, βs = _onesided_taps(G, I, E, _inward_step(G, I, E), Val(length(_stencil_taps(B))))
    P = CartesianIndex(ntuple(k -> k == Dim ? I[k] + a : I[k], Val(D)))
    inner = _onesided_operand(t, space, P, markers, lins, Val(E), bs, βs)
    return scale_stencil(shift_stencil(inner, Val(Dim), a), α)
end

# `B`'s one-sided stencil at `P` (taps `bs`, weights `βs` along `E`), its offsets from `P`,
# with the scales above it read at `P`.
@inline _onesided_operand(op::_Scale, space, P, markers, lins, vE, bs, βs) = scale_stencil(
    _onesided_operand(op.inner_op, space, P, markers, lins, vE, bs, βs),
    _scale_value(op, lins[P]))

@inline _onesided_operand(op::_FirstDifference, space, P, markers, lins, vE, bs, βs) = _onesided_fold(
    op.inner_op, space, P, markers, lins, vE, bs, βs)

@inline _onesided_fold(w, space, P, markers, lins, vE, bs::Tuple{Any}, βs) = _onesided_leaf(
    w, space, P, markers, lins, vE, bs[1], βs[1])
@inline _onesided_fold(w, space, P, markers, lins, vE, bs::Tuple, βs) = concatenate_stencils(
    _onesided_leaf(w, space, P, markers, lins, vE, bs[1], βs[1]),
    _onesided_fold(w, space, P, markers, lins, vE, Base.tail(bs), Base.tail(βs)))

@inline function _onesided_leaf(
        w, space, P::CartesianIndex{D}, markers, lins, ::Val{E}, b::Int, β
) where {D, E}
    J = CartesianIndex(ntuple(i -> i == E ? P[i] + b : P[i], Val(D)))
    return scale_stencil(shift_stencil(local_stencil(w, space, J, markers, lins[J]), Val(E), b), β)
end

# `O`'s offsets with the entry types of `S`, the regular stencil of the same term and tap.
@inline function _like_entries(S::Tuple, O::Tuple)
    length(S) == length(O) || _throw_onesided_length(length(S), length(O))
    return ntuple(k -> (O[k][1], convert(typeof(S[k][2]), O[k][2])), Val(length(S)))
end

@noinline function _throw_onesided_length(n::Int, m::Int)
    throw(
        ArgumentError(
        "the one-sided corner stencil has $m entries where the mixed difference has $n. " *
        "This is a bug in Bramble's corner rule; please report it with the form.",
    ),
    )
end

# --- The facet rule of a composite constrained as a whole ---------------------------- #
#
# On `constrained(C₁ × C₂, relations)` a facet whose relations read several components (a
# traction) closes its ghosts by a relation with a tangential difference: `u_g` of one
# component is the mirror node's value plus the spacing across the facet times the centred
# tangential difference of another. A mixed difference `A_d(B_e(u(j)))` of the form at a
# node of that facet reads those ghosts, so its row carries a centred tangential difference
# of a centred tangential difference, of the opposite sign to the row's own second
# difference along the facet (`-λ ∂ₑₑ` in elasticity). Once the spacing across the facet is
# about 1.5 times the spacing along it, the reduced matrix has eigenvalues of negative real
# part (traction on one facet, uniform cells; #793).
#
# There `form` (ghost_composite.jl) marks the functions of every mixed difference in a
# product's left argument, the side of the matrix columns (`_FacetExtrapolated`,
# `_mark_mixed`): read at a ghost beyond a node of a marked facet, it is the quadratic
# through its values at the three nodes inward along the normal, evaluated at the ghost
# (`_one_sided_weights`, ghost_relations.jl), and elsewhere its own value. The centred
# difference across the facet then reads the derivative of that quadratic at the node,
# second order on the non-uniform spacing, and no ghost of the facet, whichever of `A` and
# `B` crosses it. The second differences and the relations read the closure's ghosts, and
# at a corner node the corner rule above comes first. Every other space, and a composite
# whose relations each read one component, assembles without the rule. A marked function
# has three entries at every point, the last two zero away from a marked ghost, so the
# walk sees one length.

# A marked function is the corner rule's operand as the function is.
@inline _scaled_leaf(op::_FacetExtrapolated) = _scaled_leaf(op.inner_op)

@inline _three_parts(S₁, S₂, S₃, w) = concatenate_stencils(scale_stencil(S₁, w[1]),
    concatenate_stencils(scale_stencil(S₂, w[2]), scale_stencil(S₃, w[3])))

@inline function local_stencil(
        op::_FacetExtrapolated{D}, space, P::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    d, s = _marked_ghost(op.facets, mesh(space), P)
    iszero(d) || return _extrapolated(op.inner_op, space, P, markers, d, s)
    S = local_stencil(op.inner_op, space, P, markers, lin_idx)
    T = eltype(space)
    return _three_parts(S, S, S, (one(T), zero(T), zero(T)))
end

# The axis `d` and inward step `s` of the marked facet whose ghost `P` is, with three nodes
# inward; `(0, 0)` if it is none (a node, a padded corner point, an unmarked facet's ghost).
@inline function _marked_ghost(
        facets::NTuple{D, NTuple{2, Bool}}, G, P::CartesianIndex{D}) where {D}
    phys = physical_indices(G)
    P in phys && return (0, 0)
    for d in 1:D, t in 1:2

        facets[d][t] || continue
        s = t == 1 ? 1 : -1
        _step(P, d, s) in phys && _step(P, d, 3s) in phys && return (d, s)
    end
    return (0, 0)
end

@inline _step(P::CartesianIndex{D}, d::Int, k::Int) where {D} = CartesianIndex(
    ntuple(a -> a == d ? P[a] + k : P[a], Val(D)))

# The function `w` at the ghost `P` extrapolated from the nodes `P + k s` along `d`, `k` 1
# to 3: their stencils relabelled to offsets from `P` and weighted by the quadratic through
# them evaluated at `P`.
@inline function _extrapolated(
        w, space, P::CartesianIndex{D}, markers, d::Int, s::Int) where {D}
    G = mesh(space)
    T = eltype(space)
    x₁ = _inward_spacing(G, P, d, s)
    x₂ = x₁ + _inward_spacing(G, _step(P, d, s), d, s)
    x₃ = x₂ + _inward_spacing(G, _step(P, d, 2s), d, s)
    c = map(a -> convert(T, a), _one_sided_weights(zero(x₁), x₁, x₂, x₃))
    lins = LinearIndices(indices(G))
    S₁ = _part(w, space, P, markers, lins, d, s)
    S₂ = _part(w, space, P, markers, lins, d, 2s)
    return _three_parts(S₁, S₂, _part(w, space, P, markers, lins, d, 3s), c)
end

@inline _inward_spacing(G, Q, d::Int, s::Int) = s == 1 ? forward_spacing(G, Q, d) :
                                                spacing(G, Q, d)
@inline function _part(w, space, P, markers, lins, d::Int, k::Int)
    Q = _step(P, d, k)
    return _shift_along(local_stencil(w, space, Q, markers, lins[Q]), d, k)
end

@inline _shift_along(S::Tuple, d::Int, k::Int) = _each_entry(
    t -> (shift_offset(t[1], d, k), t[2]), S)

# The facet rule's reads at `J`: the function's at the nodes it extrapolates from.
@inline function _reads_at(
        op::_FacetExtrapolated{D}, space, J::CartesianIndex{D}, facet::Bool) where {D}
    _at_absent(op, space, J) && return true
    d, s = _marked_ghost(op.facets, mesh(space), J)
    iszero(d) && return _reads_at(op.inner_op, space, J, facet)
    return _reads_at(op.inner_op, space, _step(J, d, s), facet) ||
           _reads_at(op.inner_op, space, _step(J, d, 2s), facet) ||
           _reads_at(op.inner_op, space, _step(J, d, 3s), facet)
end

# Its reach: the point, and one to three steps inward across each marked facet.
function stencil_offsets(op::_FacetExtrapolated{D}) where {D}
    inner = stencil_offsets(op.inner_op)
    out = copy(inner)
    for d in 1:D, t in 1:2

        op.facets[d][t] || continue
        for k in 1:3, o in inner

            p = shift_offset(o, d, t == 1 ? k : -k)
            p in out || push!(out, p)
        end
    end
    return sort!(out)
end
# Its margin is the function's: the reads by extrapolation, at a ghost, go inward to nodes,
# and the walk's interior reads no marked ghost without widening (`_face_margins`).
_stencil_margin(op::_FacetExtrapolated) = _stencil_margin(op.inner_op)
expression(op::_FacetExtrapolated) = expression(op.inner_op)

# --- The walk's interior: the term unmarked (#813) ------------------------------------ #
#
# The rule changes a read only at a ghost of a marked facet. The walk keeps its unguarded
# interior `m = _stencil_margin(term)` points in from every face (`_face_margins`). A marked
# function sits only inside a mixed difference (`_mark`), under taps along two axes, and `m`
# counts every tap whatever its axis, so the function reaches at most `m - 1` along any one
# axis: from the interior it reads the first node in from a face, never its ghost layer. So
# no read there reaches a marked ghost, and the walk takes the interior with the term
# unmarked (`_interior_term`): one stencil part where a marked function has three, and no
# `_marked_ghost` test per read. The guarded shell walks the marked term. The recording
# (`_coord_walk!`) and every replay (`_visit_stencil!`, `_SplitTerm`) split alike, so their
# entries agree one for one.

# Whether a term of type `T` holds a marked function where `_unmark` strips it, from the
# type alone (as `_may_restrict`). A field of abstract type answers `false`: that term is
# walked marked everywhere, which reads the same values.
Base.@nospecializeinfer Base.@assume_effects :foldable function _has_marks(
        @nospecialize(T))
    T isa DataType || return false
    T <: _FacetExtrapolated && return true
    T <: UnaryWrapper && return _has_marks(fieldtype(T, :inner_op))
    if T <: Union{BilinearProduct, OperatorAdd}
        return _has_marks(fieldtype(T, :left_op)) || _has_marks(fieldtype(T, :right_op))
    end
    return false
end

# `op` for the walk's interior: unmarked, or `op` itself when it holds no mark.
@inline _interior_term(op) = _has_marks(typeof(op)) ? _unmark(op) : op

_unmark(op) = op
_unmark(op::_FacetExtrapolated) = op.inner_op
_unmark(op::UnaryWrapper) = _rewrap_inner(op, _interior_term(op.inner_op))
# An interpolation reads its operand on another mesh, which this interior does not bound.
_unmark(op::InterpolationNode) = op
function _unmark(op::OperatorAdd{D}) where {D}
    left, right = _interior_term(op.left_op), _interior_term(op.right_op)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end
function _unmark(op::BilinearProduct{D, W}) where {D, W}
    left, right = _interior_term(op.left_op), _interior_term(op.right_op)
    return BilinearProduct{D, W, typeof(left), typeof(right)}(op.inner, left, right)
end

"""
    _face_margins(term, ax::NTuple{D, AbstractUnitRange{Int}}) -> NTuple{D, NTuple{2, Int}}

The walk's guarded margin on each face of the grid with axes `ax`, `(low, high)` per axis:
[`_stencil_margin`](@ref)`(term)` on every face. A marked function of `term` sits only
inside a mixed difference (`_mark`), so it reaches at most `m - 1` along any one axis,
`m = _stencil_margin(term)`, and an interior point `m` in from a marked face reads the first
node in from that face, never its ghost layer: the interior takes the term unmarked
(`_interior_term`) without widening the margin on a marked facet.
"""
function _face_margins(term, ::NTuple{D, AbstractUnitRange{Int}}) where {D}
    m = _stencil_margin(term)
    return ntuple(_ -> (m, m), Val(D))
end

# `op` with the functions of every mixed difference marked for the facet rule on `facets`,
# in the left argument of each product, through sums and wrappers. The left argument's
# stencil gives the matrix columns, the padded values the closure fills, whose ghosts the
# rule reads by extrapolation; the right argument's gives the rows, never marked. Bramble
# places a product's sides by position, not by the leaf type (`u` or `v`). A mixed
# difference is a first difference along one axis of a first difference along another,
# under any sums and scales between them, and the functions of the inner one are marked
# through its own sums and scales (`Dcₓ(Dcᵧ(u(1) + 2.0 * u(2)))` as
# `Dcₓ(Dcᵧ(u(1))) + 2.0 * Dcₓ(Dcᵧ(u(2)))`). `_face_margins` relies on this: under taps
# along two axes a marked function reaches at most `_stencil_margin - 1` along either, so
# the walk's interior reads no marked ghost. Marking the function of a single-axis
# difference (`Dcₓ(u)`) would let it reach the ghost layer from the interior, and
# `_face_margins` must then widen the margin by 1 on every marked facet again.
_mark_mixed(op, facets::NTuple{D, NTuple{2, Bool}}) where {D} = _mark(op, facets)
_mark(op, facets) = op
function _mark(op::OperatorAdd{D}, facets) where {D}
    left, right = _mark(op.left_op, facets), _mark(op.right_op, facets)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end
function _mark(op::BilinearProduct{D, W}, facets) where {D, W}
    left = _mark(op.left_op, facets)
    return BilinearProduct{D, W, typeof(left), typeof(op.right_op)}(op.inner, left, op.right_op)
end
_mark(op::UnaryWrapper, facets) = _rewrap_inner(op, _mark(op.inner_op, facets))
function _mark(op::_FirstDifference{D, Dim}, facets) where {D, Dim}
    return _rewrap_inner(op, _mark_operand(op.inner_op, Val(Dim), facets))
end

# The operand of a first difference along `Dim`: through its sums and scales, a first
# difference along another axis has its functions marked; any other term is walked on.
function _mark_operand(op::OperatorAdd{D}, vdim::Val, facets) where {D}
    left = _mark_operand(op.left_op, vdim, facets)
    right = _mark_operand(op.right_op, vdim, facets)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end
_mark_operand(op::_Scale, vdim::Val, facets) = _rewrap_inner(
    op, _mark_operand(op.inner_op, vdim, facets))
function _mark_operand(op::_FirstDifference{D, E}, ::Val{Dim}, facets) where {D, E, Dim}
    E == Dim && return _mark(op, facets)
    return _rewrap_inner(op, _mark_leaves(op.inner_op, facets))
end
_mark_operand(op, ::Val, facets) = _mark(op, facets)

# The functions of `op` marked through its sums and scales; anything else as it is.
function _mark_leaves(op::OperatorAdd{D}, facets) where {D}
    left, right = _mark_leaves(op.left_op, facets), _mark_leaves(op.right_op, facets)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end
_mark_leaves(op::_Scale, facets) = _rewrap_inner(op, _mark_leaves(op.inner_op, facets))
_mark_leaves(op::_AnyLeaf, facets) = _FacetExtrapolated(op, facets)
_mark_leaves(op, facets) = op

# --- The refusal: any other read of a point the padded grid lacks -------------------- #
#
# A nesting the corner rule does not cover (an average, jump or shift of a cross difference,
# a difference around a second difference, three levels deep, ...) may still read a point
# the padded grid lacks: an absent one near a 2D corner or 3D edge, or a nested operand's
# taps reaching on from a ghost. Under a facet product that includes a point outside the
# grid, beyond the one ghost layer or across a facet without ghosts, where a stencil reads a
# truncation and not the point.
#
# A product (`innerₕ`, `inner_Γ`, ...) on a ghost space asks, at every node of the boundary
# band where its weight is not zero, whether either operand reads one, and throws an
# `ArgumentError` naming it. A read that a zero weight multiplies (a facet operator's
# component off its facet) or a zero scale multiplies is no read. The walk follows the
# evaluation, the corner rule included, through every tap whose own weight is not zero. A
# volume product's operand truncated at the edge of the padded grid is the plain mesh's
# closure there and stands. Off a ghost space the stock `local_stencil` methods
# (`operators/inner.jl`) are the only ones, so a plain mesh compiles as before.
const _GhostSpace{D} = ScalarGridSpace{D, <:Any, <:Any, <:_GhostMeshLike}

@inline function local_stencil(
        op::BilinearProduct{D, InnerType}, space::_GhostSpace{D}, I::CartesianIndex{D},
        markers, lin_idx::Int
) where {D, InnerType}
    _check_product_reads(op, space, I, lin_idx)
    return @invoke local_stencil(op::BilinearProduct{D, InnerType}, space::Any,
        I::CartesianIndex{D}, markers::Any, lin_idx::Int)
end

# The absent-read check of a product at `I`, which the stencil walk above and the factored
# walk (`_factored_product!`, `bilinear_traversal.jl`) both make before reading a factor.
# Off a ghost space there is nothing to check.
@inline _check_product_reads(op, space, I, lin_idx::Int) = nothing
@inline function _check_product_reads(
        op::BilinearProduct{D}, space::_GhostSpace{D}, I::CartesianIndex{D}, lin_idx::Int
) where {D}
    _near_edge(mesh(space), I, _stencil_margin(op)) && _check_absent_reads(op, space, I, lin_idx)
    return nothing
end

@inline function local_stencil(
        op::LinearProduct{D, InnerType}, space::_GhostSpace{D}, I::CartesianIndex{D},
        markers, lin_idx::Int
) where {D, InnerType}
    _near_edge(mesh(space), I, _stencil_margin(op.right_op)) &&
        _check_absent_reads(op, space, I, lin_idx)
    return @invoke local_stencil(op::LinearProduct{D, InnerType}, space::Any,
        I::CartesianIndex{D}, markers::Any, lin_idx::Int)
end

# Whether `I` lies within `margin` steps of the edge of the padded grid.
@inline function _near_edge(G::_GhostMeshLike{D}, I::CartesianIndex{D}, margin::Int) where {D}
    n = size(indices(G))
    near = false
    for d in 1:D
        near |= (I[d] - 1 <= margin) | (n[d] - I[d] <= margin)
    end
    return near
end

# The facet products, whose operands are meant to read ghosts at the node.
@inline _facet_inner(::Union{InnerGamma, InnerGammaNormal, InnerGammaFacet}) = true
@inline _facet_inner(::Any) = false

@noinline function _check_absent_reads(
        op::BilinearProduct{D, InnerType}, space, I::CartesianIndex{D}, lin_idx::Int
) where {D, InnerType}
    iszero(compute_weight(op.inner, space, I, lin_idx)) && return false
    facet = _facet_inner(op.inner)
    (_reads_at(op.left_op, space, I, facet) || _reads_at(op.right_op, space, I, facet)) &&
        _throw_absent_read(op, I - CartesianIndex(mesh(space).lo))
    return false
end

@noinline function _check_absent_reads(
        op::LinearProduct{D, InnerType}, space, I::CartesianIndex{D}, lin_idx::Int
) where {D, InnerType}
    # A dirac's weight is not the volume, so its zero test is where it loads: on a ghost
    # mesh `locate_cell` places every point in a physical cell, so only physical nodes.
    if _is_dirac(op.left_op)
        I in physical_indices(mesh(space)) || return false
    else
        iszero(compute_weight(op.inner, space, I, lin_idx)) && return false
    end
    _reads_at(op.right_op, space, I, _facet_inner(op.inner)) &&
        _throw_absent_read(op, I - CartesianIndex(mesh(space).lo))
    return false
end

@noinline function _throw_absent_read(op, I::CartesianIndex)
    throw(
        ArgumentError(
        "on a ghost space, `$(expression(op))` reads, at the physical node $(Tuple(I)), a " *
        "padded point that does not exist: absent (beyond a 2D corner or a 3D edge, no " *
        "node and no ghost), or outside the padded grid (beyond the one ghost layer, or on " *
        "a facet without ghosts). Only a first difference of first differences of `u` " *
        "across two axes, or a sum or scale of them, has a corner rule (the inward " *
        "one-sided difference); write the term with no nesting that reaches past the " *
        "ghosts, give the space ghosts on the facets it reads, or assemble it on a space " *
        "without ghosts.",
    ),
    )
end

# Whether `op`, evaluated at the padded point `J`, reads a point the grid lacks: at an
# absent one any operand of a trial or test function has no value, and under a facet product
# (`facet`) a node whose taps leave the grid reads past it. One method per node,
# as `local_stencil` has, and the taps a loop over their steps: a recursion through one
# shared helper fails inference's recursion limit and dispatched at run time on every
# refill.
@inline _at_absent(op, space, J) = is_absent(mesh(space), J) && _has_relative_offsets(op)

# Whether a step `delta` along `Dim` from `J` leaves the grid where that counts as a read:
# under a facet product only, as a volume product keeps the plain mesh's truncation.
@inline function _leaves_grid(
        space, J::CartesianIndex, ::Val{Dim}, delta::Int, facet::Bool
) where {Dim}
    facet || return false
    return !(1 <= J[Dim] + delta <= size(indices(mesh(space)), Dim))
end

@inline _reads_at(op, space, J::CartesianIndex, facet::Bool) = _at_absent(op, space, J)
@inline _reads_at(op::RegionRestriction, space, J::CartesianIndex, facet::Bool) = _at_absent(
    op, space, J) || _reads_at(op.inner_op, space, J, facet)
# A zero scale is a zero weight: what it multiplies is no read.
@inline _reads_at(op::_Scale, space, J::CartesianIndex, facet::Bool) = _at_absent(
    op, space, J) ||
                                                                       !iszero(_scale_value(op, LinearIndices(indices(mesh(space)))[J])) &&
                                                                       _reads_at(op.inner_op, space, J, facet)
@inline _reads_at(op::OperatorAdd, space, J::CartesianIndex, facet::Bool) = _at_absent(
                                                                                op, space, J) ||
                                                                            _reads_at(op.left_op, space, J, facet) ||
                                                                            _reads_at(
                                                                                op.right_op, space, J, facet)

@inline function _reads_at(
        op::ShiftNode{D, Dim}, space, J::CartesianIndex{D}, facet::Bool
) where {D, Dim}
    _at_absent(op, space, J) && return true
    _leaves_grid(space, J, Val(Dim), op.shift_amount, facet) && return true
    K = CartesianIndex(ntuple(k -> k == Dim ? J[k] + op.shift_amount : J[k], Val(D)))
    return checkbounds(Bool, indices(mesh(space)), K) && _reads_at(op.inner_op, space, K, facet)
end

@inline function _reads_at(
        op::TappedNode{D, Dim}, space, J::CartesianIndex{D}, facet::Bool
) where {D, Dim}
    _at_absent(op, space, J) && return true
    G = mesh(space)
    taps = _stencil_taps(op)
    ds = map(_shift_delta, taps)
    for k in eachindex(ds)
        _leaves_grid(space, J, Val(Dim), ds[k], facet) && return true
    end
    ws = _stencil_weights(op, space, J)
    # The corner rule: a mixed term that reads the corner takes the one-sided difference,
    # which reads physical nodes only; every other term `A`'s own taps.
    rule = _needs_onesided(op, G, J)
    terms = _rule_terms(op.inner_op)
    for k in eachindex(ds)
        iszero(ws[k]) && continue
        K = CartesianIndex(ntuple(i -> i == Dim ? J[i] + ds[k] : J[i], Val(D)))
        checkbounds(Bool, indices(G), K) || continue
        if rule
            _terms_read(terms, space, J, K, Val(Dim), taps, facet) && return true
        else
            _reads_at(op.inner_op, space, K, facet) && return true
        end
    end
    return false
end

# The terms at `K` that the corner rule leaves to `A`'s taps at `J`.
@inline _terms_read(::Tuple{}, space, J, K, vdim, taps, facet) = false
@inline _terms_read(ts::Tuple, space, J, K, vdim, taps, facet) = (
    !_term_reads_absent(first(ts), mesh(space), J, vdim, taps) &&
    _reads_at(first(ts), space, K, facet)) ||
                                                                 _terms_read(
    Base.tail(ts), space, J, K, vdim, taps, facet)

# A tap reaching a tapped node over a restriction (`D̽ₓ(D̽ᵧ(restrict_to(:l, v)))`) re-evaluated
# it at the neighbour, where the restriction's own tap 0 answers `()` or a full tuple by the
# neighbour's marker; three such taps concatenate to more stencil shapes than inference
# splits, and every refill boxed (83232 B on 7x9, gpena/Bramble.jl#759). The node is instead
# evaluated at the neighbour with that restriction zeroed outside its region, so the tap has
# one length at every point, and relabelled as the trait path relabels it. The predicate
# reads types only, so the branch folds; a source-only operand keeps the trait path.
#
# A tapped node over a scale of bits values over a sum is fixed as one over the sum is
# (`_fixes_operand`). `Mₓ(D₋ₓ(2.0 * (Rₕ(:l, u) + Rₕ(:r, u))))`, where a paired average
# stays whole, boxed (53136 B on 7x9 against one `D₋ₓ(Rₕ(:l, v))`) while
# `Mₓ(D₋ₓ(Rₕ(:l, u) + Rₕ(:r, u)))` did not. Only another node's operand is fixed so. A
# node directly over such a scale, its taps padded, boxed more where its product boxes
# anyway (`innerₕ(2 * D₊ₓ(2.0 * Rₕ(:l, u) + 3 * Rₕ(:r, u)),
# M₊ᵧ(2.0 * (Rₕ(:l, v) + Rₕ(:r, v))))` 49536 -> 63072 B). A `BigFloat` scale is not fixed,
# as each padded zero is one more product that allocates (`Mₓ(D₋ₓ(g * (Rₕ(:l, u) +
# Rₕ(:r, u))))` with `g` a `BigFloat` grid function 557808 B as it was, 816576 B fixed).
@inline _fixes_restriction(op::RegionRestriction) = !_is_source_only(op)
@inline _fixes_restriction(op::_Scale) = !_is_source_only(op) &&
                                                                          (_scales_restriction(op) ||
                                                                           op.inner_op isa TappedNode &&
                                                                           _fixes_restriction(op.inner_op))
@inline _fixes_restriction(op::TappedNode) = _fixes_operand(op.inner_op)
@inline _fixes_restriction(op::OperatorAdd) = _fixes_restriction(op.left_op) ||
                                              _fixes_restriction(op.right_op)
@inline _fixes_restriction(::Any) = false
@inline _fixes_operand(op) = _fixes_restriction(op)
@inline _fixes_operand(op::_Scale) = _fixes_restriction(op) ||
                                                                      !_is_source_only(op) &&
                                                                      op.inner_op isa OperatorAdd &&
                                                                      _bits_weight(op) &&
                                                                      _fixes_restriction(op.inner_op)
@inline _bits_weight(op::OperatorScale) = isbits(_scale_value(op, 0))
@inline _bits_weight(op::GridFunctionScale) = _bits_values(op.grid_function)
@inline _bits_values(g::AbstractArray) = isbitstype(eltype(g))
@inline _bits_values(g) = isbits(g)

@inline function shifted_inner_stencil(
        inner_op::TappedNode{D}, inner, space, I::CartesianIndex{D}, markers, ::Val{Dim}, delta
) where {D, Dim}
    _fixes_restriction(inner_op) || return _shifted_inner_stencil(
        stencil_shift_trait(inner_op), inner_op, inner, space, I, markers, Val(Dim), delta)
    # Out of line, as `_reevaluated_shift` is for any operand deeper than a leaf.
    return @noinline _fixed_shift(inner_op, space, I, markers, Val(Dim), _shift_delta(delta))
end

# `_reevaluated_shift` with the node's stencil at the neighbour of fixed length.
@inline function _fixed_shift(
        op, space, I::CartesianIndex{D}, markers, ::Val{Dim}, delta::Int
) where {D, Dim}
    m = mesh(space)
    Ishift = _clamped_shift(m, I, Val(Dim), delta)
    at = _fixed_stencil(op, space, Ishift, markers, LinearIndices(indices(m))[Ishift])
    T = eltype(space)
    on_grid = Ishift[Dim] == I[Dim] + delta ? one(T) : zero(T)
    return scale_stencil(shift_stencil(at, Val(Dim), delta), on_grid)
end

# `op`'s stencil at `J`, of one length at every point. A tapped node over a restriction, or
# over a chain of scales and restrictions, takes as its tap 0 that chain shifted by zero
# (`_shifted_chain`); outside a region it takes that chain zeroed and relabelled onto one of
# its other taps, entries the point visits anyway, so the pattern gains none
# (`D₋ₓ(D₊ᵧ(Rₕ(:l, v)))` grew 131 -> 176 stored entries zeroed in place). Its other taps are
# fixed by the shift rules already, and any other operand already has one length. A node
# over such a node takes it as its own tap 0 too (`local_stencil` above): left a union of two
# lengths, the product of two sides compiled three long stencils, not one (first assemble of
# `D̽ₓ(D̽ᵧ(Rₕ(:l, D₋ₓ(u))))` on both sides 14.9 s, against 2.4 s with one).
@inline _fixed_stencil(op, space, J, markers, lin::Int) = local_stencil(
    op, space, J, markers, lin)
@inline function _fixed_stencil(
        op::TappedNode{D, Dim}, space, J::CartesianIndex{D}, markers, lin::Int
) where {D, Dim}
    _fixes_restriction(op) || return local_stencil(op, space, J, markers, lin)
    taps = _stencil_taps(op)
    inner = _fixed_operand(op.inner_op, space, J, markers, lin, Val(Dim), _off_tap(taps))
    return _fixed_taps(
        op, inner, space, J, markers, Val(Dim), taps, _tap_weights(op, space, J))
end

# `_fold_taps` for a fixed stencil. A node over a scale of bits values over a sum
# (`_fixes_operand`) shifts the sum by its summands' own rules and scales it, as a scale
# over a restriction is shifted (`shifted_inner_stencil`, region_restriction.jl), where the
# trait path re-evaluated the sum at the neighbour in one of several lengths. Any other
# operand is shifted as `_tap_stencil` shifts it.
@inline _fixed_taps(op, inner, space, I, markers, vdim, taps::Tuple{Any}, ws::Tuple{Any}) = _fixed_tap(
    op, inner, space, I, markers, vdim, taps[1], ws[1])
@inline _fixed_taps(op, inner, space, I, markers, vdim, taps::Tuple, ws::Tuple) = concatenate_stencils(
    _fixed_tap(op, inner, space, I, markers, vdim, taps[1], ws[1]),
    _fixed_taps(op, inner, space, I, markers, vdim, Base.tail(taps), Base.tail(ws)))
@inline _fixed_tap(op, inner, space, I, markers, vdim, ::Val{0}, w) = scale_stencil(inner, w)
@inline _fixed_tap(op, inner, space, I, markers, vdim, delta::Val, w) = scale_stencil(
    _fixed_shift_of(op.inner_op, inner, space, I, markers, vdim, delta), w)
@inline _fixed_shift_of(op, inner, space, I, markers, vdim, delta) = shifted_inner_stencil(
    op, inner, space, I, markers, vdim, delta)
@inline function _fixed_shift_of(op::OperatorScale, inner, space, I, markers, vdim, delta)
    !_fixes_restriction(op) && _fixes_operand(op) ||
        return shifted_inner_stencil(op, inner, space, I, markers, vdim, delta)
    return scale_stencil(
        shifted_inner_stencil(op.inner_op, nothing, space, I, markers, vdim, delta),
        _scale_value(op, 0))
end

# A node over such a node takes its tap 0 the same way (`local_stencil` above), through any
# sum, scale or restriction between them (`D̽ₓ(D̽ᵧ(R) + D₋ₓ(R))`, `D₋ₓ(Rₕ(:c, D₊ᵧ(…)))`):
# a tap 0 left a union made the product of two sides box harder than before (317696 ->
# 463040 B, 106784 -> 146752 B on 7x9) and compile three long stencils, not one.
@inline _nests_fixed(op::TappedNode) = _fixes_restriction(op)
@inline _nests_fixed(op::OperatorAdd) = _nests_fixed(op.left_op) || _nests_fixed(op.right_op)
@inline _nests_fixed(op::Union{RegionRestriction, OperatorScale, GridFunctionScale}) = _nests_fixed(
    op.inner_op) && _fixes_restriction(op)
@inline _nests_fixed(::Any) = false

# A tapped node's tap 0 at `J`, of one length, `off` its first other tap. A restriction (or a
# chain of scales over one) with `J` outside its region answers its own tap at `off` zeroed:
# the entries that tap visits, so the pattern gains none; zeros left in place at `J` stored
# more (`D₋ₓ(D₊ᵧ(Rₕ(:l, v)))` 131 -> 176, `Mₓ(D₊ᵧ(Rₕ(:c, Rₕ(:l, u) + Rₕ(:r, u))))` 189 ->
# 270, a restriction under a sum). Inside, the chain is walked at `J`, a sum or a tapped
# node below it summand by summand; the zero is never multiplied by a scale above it.
@inline _fixed_operand(op, space, J, markers, lin::Int, vdim, off) = local_stencil(
    op, space, J, markers, lin)
@inline _fixed_operand(op::TappedNode, space, J, markers, lin::Int, vdim, off) = _fixed_stencil(
    op, space, J, markers, lin)
@inline function _fixed_operand(op::OperatorAdd, space, J, markers, lin::Int, vdim, off)
    _fixes_restriction(op) || return local_stencil(op, space, J, markers, lin)
    return _concat_operands(_summands(op), space, J, markers, lin, vdim, off)
end
@inline _concat_operands(ts::Tuple{Any}, space, J, markers, lin::Int, vdim, off) = _fixed_operand(
    ts[1], space, J, markers, lin, vdim, off)
@inline _concat_operands(ts::Tuple, space, J, markers, lin::Int, vdim, off) = concatenate_stencils(
    _fixed_operand(ts[1], space, J, markers, lin, vdim, off),
    _concat_operands(Base.tail(ts), space, J, markers, lin, vdim, off))
@inline function _fixed_operand(
        op::Union{RegionRestriction, OperatorScale, GridFunctionScale}, space, J, markers,
        lin::Int, vdim, off
)
    _scales_restriction(op) || return _fixes_operand(op) ?
           scale_stencil(
        _fixed_operand(op.inner_op, space, J, markers, lin, vdim, off), _scale_value(op, lin)) :
           local_stencil(op, space, J, markers, lin)
    _fixes_restriction(op) || return local_stencil(op, space, J, markers, lin)
    _chain_out(op, markers, lin) && return _zero_entries(
        shifted_inner_stencil(op, nothing, space, J, markers, vdim, off))
    return _chain_in(op, space, J, markers, lin, vdim, off)
end

# Whether `lin` lies outside a restriction of the chain of scales and restrictions at `op`.
@inline _chain_out(op::_Scale, markers, lin::Int) = _chain_out(
    op.inner_op, markers, lin)
@inline function _chain_out(op::RegionRestriction, markers, lin::Int)
    _in_region(op, markers, lin) || return true
    return _scales_restriction(op.inner_op) && _chain_out(op.inner_op, markers, lin)
end

# The chain at `J`, every restriction on it holding `J`: scaled at `J`, and below the last
# restriction a sum or a tapped node fixed in turn, anything else shifted by zero there.
@inline _chain_in(op::_Scale, space, J, markers, lin::Int,
    vdim, off) = scale_stencil(
    _chain_in(op.inner_op, space, J, markers, lin, vdim, off), _scale_value(op, lin))
@inline function _chain_in(op::RegionRestriction, space, J, markers, lin::Int, vdim, off)
    inner = op.inner_op
    _scales_restriction(inner) && return _chain_in(inner, space, J, markers, lin, vdim, off)
    inner isa Union{OperatorAdd, TappedNode} && _fixes_restriction(inner) &&
        return _fixed_operand(inner, space, J, markers, lin, vdim, off)
    return shifted_inner_stencil(op, nothing, space, J, markers, vdim, Val(0))
end

# A node's first tap other than 0 (`D₋ₓ`'s is its second): every node with a tap 0 has one.
@inline _off_tap(taps::Tuple{Val{0}, Any, Vararg}) = taps[2]
@inline _off_tap(taps::Tuple) = taps[1]

@inline local_stencil(
    ::TrialFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ((zero_offset(Val(D)), 1),)
@inline local_stencil(
    ::TestFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ((zero_offset(Val(D)), 1),)
@inline local_stencil(
    ::IndexedTrialFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ((zero_offset(Val(D)), 1),)
@inline local_stencil(
    ::IndexedTestFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ((zero_offset(Val(D)), 1),)

@inline function local_stencil(
        op::SourceFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    m = mesh(space)
    x = point(m, I)
    return ((zero_offset(Val(D)), op.func(x)),)
end

@inline function local_stencil(
        op::SourceVector{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    return ((zero_offset(Val(D)), op.vec[lin_idx]),)
end

@inline function local_stencil(
        op::SourceConstant{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    return ((zero_offset(Val(D)), op.value),)
end

@inline function _point_strength_val(s)
    if s isa Function
        return s()
    elseif s isa Base.RefValue
        return s[]
    else
        return s
    end
end

# `outside = :extrapolate`, always: a `DiracSource` point is a modelling choice (where the
# source physically sits), not a query that #223's `outside` policy is about, and this
# keeps its behaviour exactly what it was before that policy existed -- #226's own note
# ("out-of-domain handling consistent with whatever #223 settles for interpolate_at") is
# deliberately left for a follow-up, not folded into #223's own scope (its own "Subsystems
# to Update" list never names this file).
@inline function _interp_cell_frac_pt(Ωₕ::AbstractMeshType{1}, pt::NTuple{1}, ::Val{1})
    c, t = _interp_cell_frac(Ωₕ, pt[1], :extrapolate)
    return (CartesianIndex(c), (t,))
end
@inline _interp_cell_frac_pt(Ωₕ::AbstractMeshType{D}, pt::NTuple{D}, ::Val{D}) where {D} = _interp_cell_frac(
    Ωₕ, pt, :extrapolate
)

@inline _is_corner_inbounds(corner::Tuple) = all(c -> 0 <= c <= 1, corner)

# The weight's type is the mesh's element type promoted with the strength's: a Float32
# space's vector stays Float32 (and so uploads to a Metal space), a `Dual` strength still
# promotes, and a Float64 strength on a Float32 space promotes the way any Float64
# coefficient does. The cell search and fraction stay in Float64, the stored points' own
# precision, so the location is not rounded before the weight is; only the finished product
# is converted, which on a Float64 space is the identity.
@inline _dirac_weight_type(Ωₕ, s) = promote_type(eltype(Ωₕ), typeof(s))

@inline function local_stencil(
        op::DiracSource{D, <:NTuple{D, Float64}}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    Ωₕ = mesh(space)
    idx, ts = _interp_cell_frac_pt(Ωₕ, op.points, Val(D))
    corner = Tuple(I - idx)
    strength_val = _point_strength_val(op.strengths)
    T = _dirac_weight_type(Ωₕ, strength_val)
    if _is_corner_inbounds(corner)
        w = _interp_corner_weight(ts, corner, Val(D))
        return ((zero_offset(Val(D)), convert(T, strength_val * w)),)
    else
        return ((zero_offset(Val(D)), zero(T)),)
    end
end

@inline function local_stencil(
        op::DiracSource{D, <:AbstractVector}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    Ωₕ = mesh(space)
    # Each product takes its own strength's type, never the first one's: a strengths vector
    # with an abstract eltype (thunks, mixed `Ref`s, `Real[...]`) may hold a `Dual` or a
    # wider float further along, and `acc` must widen to it rather than truncate it. It
    # widens at every node, not only near that point, so every node answers with the one
    # type the assembled vector is given from every strength (`_folded_eltype`,
    # gpena/Bramble.jl#370); the widening is a conversion, not an addition, so a concrete
    # strengths eltype sees the identity and HEAD's exact values.
    acc = zero(_dirac_weight_type(Ωₕ, _point_strength_val(first(op.strengths))))
    for k in eachindex(op.points)
        pt = op.points[k]
        s = _point_strength_val(op.strengths[k])
        Tₖ = _dirac_weight_type(Ωₕ, s)
        idx, ts = _interp_cell_frac_pt(Ωₕ, pt, Val(D))
        corner = Tuple(I - idx)
        if _is_corner_inbounds(corner)
            w = _interp_corner_weight(ts, corner, Val(D))
            acc += convert(Tₖ, s * w)
        else
            acc = convert(promote_type(typeof(acc), Tₖ), acc)
        end
    end
    return ((zero_offset(Val(D)), acc),)
end

# Every summand's stencil, concatenated in `_summands` order -- entry for entry what
# pairwise `concatenate_stencils` down the tree gives, walking the flat summand tuple rather
# than the left-deep tree, so no intermediate type carries a whole subtree.
@inline function local_stencil(
        op::OperatorAdd, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    return _concat_summand_stencils(_summands(op), space, I, markers, lin_idx)
end

# One call per summand rather than `_flatten_tuples(map(...))` over all of them: a
# `RegionRestriction` summand's stencil is `()` or a full tuple depending on the point's
# marker, so its inferred type is a `Union`, and a `map` collecting it beside the other
# summands' stencils builds a tuple with a non-concrete element, boxed at every grid point
# (6720 B on a 7x9 grid for `innerₕ(u, v) + innerₕ(u, restrict_to(:boundary, v))`, growing
# with the grid). Passed straight to `concatenate_stencils` instead, the `Union` is split
# at that call and each branch builds a concrete tuple. The split has a limit: the stencil
# has one shape per combination of restricted summands' markers, and past two restricted
# summands in one sum the compiler stops splitting. `form`, bilinear and linear alike,
# therefore splits a product with more than two into products of at most two
# (`_split_restricted_products`, bilinear.jl and linear.jl), each walked apart, and both
# paths assemble with 0 B. A product left whole, as it would split into more than 16 or
# its constant scales keep several types, still boxes per point (gpena/Bramble.jl#727).
@inline _concat_summand_stencils(::Tuple{}, space, I, markers, lin_idx::Int) = ()
@inline function _concat_summand_stencils(ts::Tuple, space, I, markers, lin_idx::Int)
    head = local_stencil(first(ts), space, I, markers, lin_idx)
    return concatenate_stencils(head, _concat_summand_stencils(Base.tail(ts), space, I, markers, lin_idx))
end

@inline function local_stencil(
        op::OperatorScale, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return scale_stencil(inner, op.scalar)
end

@inline function local_stencil(
        op::OperatorScale{D, <:Base.RefValue}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return scale_stencil(inner, op.scalar[])
end

@inline function local_stencil(
        op::GridFunctionScale, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return scale_stencil(inner, _grid_function_value(op.grid_function, lin_idx))
end

@inline local_stencil(
    op::IdentityOperator{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ((zero_offset(Val(D)), 1),)
# No entries at all, the same "contributes nothing here" `RegionRestriction` answers with.
# A one-entry `((offset, 0),)` is a linear-form entry only: a bilinear walk reads each entry
# as `(off_u, off_v, weight)`, and a form whose whole AST simplified to zero (`0 * a`, or a
# runtime `k = 0`) crashed destructuring it with a `BoundsError`.
@inline local_stencil(
    op::ZeroOperator{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ()

# ==============================================================================
# 2. AST Resolution & Thunk Eval
# ==============================================================================

# Each child is resolved once and its value reused for the type parameter. Calling
# `resolve_ast` again inside `typeof(...)` doubles the work per level, `2^depth` calls on a
# left-deep sum: free without coverage, where the compiler removes the duplicate, but under
# `--code-coverage` a 27-term sum's `form` did not finish within 10 GB.
function resolve_ast(op::OperatorAdd{D}) where {D}
    left = resolve_ast(op.left_op)
    right = resolve_ast(op.right_op)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end
function resolve_ast(op::OperatorScale{D}) where {D}
    inner = resolve_ast(op.inner_op)
    return OperatorScale{D, typeof(op.scalar), typeof(inner)}(op.scalar, inner)
end

function resolve_ast(op::GridFunctionScale{D, VType}) where {D, VType}
    inner = resolve_ast(op.inner_op)
    return GridFunctionScale{D, VType, typeof(inner)}(op.grid_function, inner)
end

# A function times a term is read as a zero-argument thunk (`SourceVector`'s docstring), so
# `f * innerₕ(1.0, v)` with `f(x)` a function of position would fail in the call below with
# a `MethodError` naming no cause. Checked here, once per form, never per point.
function resolve_ast(op::GridFunctionScale{D, <:Function}) where {D}
    applicable(op.grid_function) || _throw_not_a_thunk()
    vec = op.grid_function()
    inner = resolve_ast(op.inner_op)
    return GridFunctionScale{D, typeof(vec), typeof(inner)}(vec, inner)
end

resolve_ast(ops::NTuple{N, Any}) where {N} = map(resolve_ast, ops)

@noinline function _throw_not_a_thunk()
    throw(ArgumentError("a function times a term, `f * op`, is read as a zero-argument " *
                        "thunk returning the values to scale by, and this function has " *
                        "no zero-argument method. For a source that is a function of " *
                        "position write `innerₕ(f, v)`; for a coefficient restrict it to " *
                        "the grid first, `Rₕ(W, f) * op`."))
end

# The two scaling wrappers' half of `_bind_interp_spaces` (operators/interpolation.jl),
# beside their `resolve_ast` because they are the same walk. `GridFunctionScale`'s thunk
# form has already been evaluated by `resolve_ast` when binding runs, so one method covers
# both. An `OperatorScale`'s scalar is carried across untouched.
function _bind_interp_spaces(op::OperatorScale{D}, trial_leaf, test_leaf) where {D}
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    return OperatorScale{D, typeof(op.scalar), typeof(inner)}(op.scalar, inner)
end

# A coefficient in device storage is read one point at a time by `local_stencil`, which the
# device refuses, so binding swaps it for a host copy (`_host_array`, `linear.jl`;
# gpena/Bramble.jl#364). Every walk of every fill binds its term afresh (the same sites take
# the walked leaf's `host_weights`), so the copy is new each time and `assemble!` sees a
# coefficient changed on the device since the last call; it is never cached in the form, and
# a device fill allocates it per call. A host coefficient is returned as it is, so the host
# path neither copies nor changes type.
function _bind_interp_spaces(op::GridFunctionScale{D}, trial_leaf, test_leaf) where {D}
    gf = _host_array(op.grid_function)
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    return GridFunctionScale{D, typeof(gf), typeof(inner)}(gf, inner)
end
# The catch-all every node above without its own method falls through to: TrialFunction,
# TestFunction, IndexedTrialFunction, IndexedTestFunction, SourceFunction, SourceVector,
# SourceConstant, IdentityOperator, ZeroOperator, and anything else with nothing to resolve.
# gpena/Bramble.jl#62: those nine used to have their own identity methods here, each
# decorative -- this catch-all made every one of them redundant, since a node not listed
# was never an error, only silently unresolved. Left as one line rather than nine.
resolve_ast(op::Any) = op

# ==============================================================================
# 3. Symbolic AST Traits
# ==============================================================================

# Note: is_symbolic base function is declared in ast.jl

# One method for every node that wraps a single operand, instead of one registration per
# node type (gpena/Bramble.jl#52); see [`UnaryWrapper`](@ref).
is_symbolic(op::UnaryWrapper) = is_symbolic(op.inner_op)
is_symbolic(::TrialFunction) = true
is_symbolic(::TestFunction) = true
is_symbolic(::IndexedTrialFunction) = true
is_symbolic(::IndexedTestFunction) = true
is_symbolic(::SourceFunction) = true
is_symbolic(::SourceVector) = true
is_symbolic(::SourceConstant) = true
is_symbolic(::DiracSource) = true
is_symbolic(op::BilinearProduct) = true
is_symbolic(op::LinearProduct) = true

# A wrapper is source-only when what it wraps is. One method instead of one registration
# per node type (gpena/Bramble.jl#52); `InterpolationNode` overrides it below, since it
# carries a trial function however it is wrapped.
"""
    _is_source_only(op::LazyOp) -> Bool

Whether a `LazyOp` subtree is source-only: built entirely from sources
(`SourceFunction`, `SourceVector`, `SourceConstant`, `DiracSource`) and the plain
operators that wrap them, never bottoming out in a `TrialFunction`/`IndexedTrialFunction`
leaf.

`innerₕ`'s `l::Function`/`l::Number`/`l::VectorElement` overloads (`operators/inner.jl`)
never need this: those three types are never anything but a source, so wrapping them in a
`LinearProduct` is unconditional. The question only exists for an argument that already
arrived as a `LazyOp`: `πₕ(uₕ)` ([`interpolate_at`](@ref)) or `D₋ₓ(πₕ(uₕ))` are
sources too, just already wrapped, and the generic `innerₕ(::LazyOp, ::LazyOp)` used to build
a `BilinearProduct` regardless, which is the wrong AST shape for a `LinearForm`'s assembly
walk, since a `BilinearProduct`'s stencil carries a pair of offsets (trial and test), where
`_scatter_term!` (`assembly/linear.jl`) expects one.

A missing case defaults to `false` (the fallback `::LazyOp` method below), which is conservative,
since that is exactly the behavior every node had before this predicate existed (always
`BilinearProduct`) for anything not explicitly listed as source-only.
"""
_is_source_only(op::UnaryWrapper) = _is_source_only(op.inner_op)
_is_source_only(::TrialFunction) = false
_is_source_only(::TestFunction) = false
_is_source_only(::IndexedTrialFunction) = false
_is_source_only(::IndexedTestFunction) = false
_is_source_only(::SourceFunction) = true
_is_source_only(::SourceVector) = true
_is_source_only(::SourceConstant) = true
_is_source_only(::DiracSource) = true

function _is_source_only(op::OperatorAdd)
    return _is_source_only(op.left_op) && _is_source_only(op.right_op)
end

# A product (whichever kind) is its own thing, not a bare source to route again.
_is_source_only(::BilinearProduct) = false
_is_source_only(::LinearProduct) = false

_is_source_only(::LazyOp) = false

# The value of a source-only subtree at a grid point: `_contracted_left_stencil`
# (operators/inner.jl) reads it from the subtree's own `local_stencil`, correctly
# re-evaluated at every neighbour because a source is `PointDependentStencil`
# (operators/interpolation.jl).

# ==============================================================================
# 4. Walking an OperatorAdd tree: shared by every router in linear.jl/bilinear.jl
# ==============================================================================

# Functions across `linear.jl`/`bilinear.jl`/`jacobian_pattern.jl` (`_check_block_meshes`,
# `_route_terms!`, `_route_terms_parallel!`, `_pattern_blocks_jacobian!`) walk a form's
# `OperatorAdd` tree to send each summand where it belongs, all with the same shape: recurse left, recurse right, done. Recursing the tree
# rather than flattening it into a vector of terms first preserves concrete types: a
# flattened vector is `Vector{Any}` and makes every term a dynamic read, whereas recursing
# keeps each term concretely typed at its own call and costs nothing.
#
# They differ only in how many arguments sit before `op` and what (if anything) the caller
# reads back; nothing does: every call site above these six is a bare statement, the
# return value always discarded. The three shapes below return whatever the six already
# return unread today. Three separate names rather than one overloaded on argument count:
# with `rest...`/untyped leading arguments, overloads sharing a name are genuinely ambiguous
# to the compiler (a call whose second and third arguments both happen to be `OperatorAdd`
# matches two of the three signatures at once). Aqua's ambiguity check catches this even
# though no real call here ever hits it, and three names sidesteps the question rather than
# resolving it with a disambiguating method nothing calls. Each is still picked at its call
# site by hand; what moves out is only the recursion body, and
# `@code_warntype` still sees ordinary calls to `f`, specialized on `F = typeof(f)` like any
# other higher-order call in Julia.

# The summands of a sum, left to right, flattened into one tuple.
@inline _summands(op::OperatorAdd) = (_summands(op.left_op)..., _summands(op.right_op)...)
@inline _summands(op) = (op,)

# The three walks below visit `_summands(op)` with one `map` rather than recursing
# left/right: recursing, every `OperatorAdd` node is its own method instance whose type
# carries its whole subtree, so a left-deep `a + b + c + ...` (what `foldl(+, terms)` and a
# hand-written sum build) is inferred over types of total size quadratic in the term count.
# 3D `innerₕ(εcₕ(u), εcₕ(v))` built with `foldl(+, ...)`, first assemble: 18.6–21.0 s
# recursing, 16.6–17.6 s over `_summands`. Only leaves reach `f`; the walk order is unchanged.

# `op` first, nothing to mutate: `_check_block_meshes`.
@inline function _visit_operator_add1(f::F, op::OperatorAdd, rest...) where {F}
    map(t -> f(t, rest...), _summands(op))
    return nothing
end

# `op` second, one mutated argument returned unchanged: `_route_terms!`,
# `_route_terms_parallel!`, `_assemble_blocks!`.
@inline function _visit_operator_add2(f::F, first_arg, op::OperatorAdd, rest...) where {F}
    map(t -> f(first_arg, t, rest...), _summands(op))
    return first_arg
end

# `op` third, two mutated arguments, nothing returned: `_pattern_blocks_jacobian!`.
@inline function _visit_operator_add3(f::F, a1, a2, op::OperatorAdd, rest...) where {F}
    map(t -> f(a1, a2, t, rest...), _summands(op))
    return nothing
end

# `op` first, an accumulator threaded through both children and returned:
# `_route_terms_contract` (`assembly/linear.jl`).
#
# The `_visit_operator_add*` members above each return one of their own arguments unchanged,
# so they walk for side effects and cannot fold a value. Contraction threads an accumulator,
# so it hand-recursed instead. The family had a hole and its one folding caller quietly
# stepped around it rather than the family gaining this (gpena/Bramble.jl#55). The
# accumulator is the second positional argument, mirroring `_visit_operator_add2`'s
# `first_arg` slot, so the walk order reads the same across the family.
@inline function _fold_operator_add(f::F, op::OperatorAdd, acc, rest...) where {F}
    acc = f(op.left_op, acc, rest...)
    return f(op.right_op, acc, rest...)
end
