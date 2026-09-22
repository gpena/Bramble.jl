# inner.jl
# Contains all inner product traits and logic for Bramble lazy AST
#
# `inner₊` and `innerₕ` mean something different here than they do in
# `src/space/inner_product.jl`. Here they take operators and build an **AST node**; there
# they take grid functions and compute a **number**. CONTEXT.md draws that line at the
# domain level: a form is symbolic, a grid function is data.
#
# What keeps the two families from colliding is the `NTuple{N,<:Tuple}` restriction on the
# tuple overload below: a tuple of grid functions is not a tuple of tuples, so it cannot
# reach this file's method. Widen it, or add a `VectorElement`-shaped overload here, and
# the collision is real. The constraint is asserted in `test/form/inner_products.jl`,
# testset "Symbolic and numeric families stay apart" (gpena/Bramble.jl#60), rather than
# living only in the comment on that overload.

# ==============================================================================
# Struct Definitions
# ==============================================================================

"""
    AbstractInnerProduct

Abstract base type for inner product quadrature weights.
"""
abstract type AbstractInnerProduct end

"""
    InnerH <: AbstractInnerProduct

Quadrature weights for the standard \$L^2\$ inner product using trapezoidal integration.
"""
struct InnerH <: AbstractInnerProduct end

"""
    InnerPlus{Dim} <: AbstractInnerProduct

Quadrature weights for the modified \$L^2_+\$ inner product in a specific coordinate dimension `Dim`.
"""
struct InnerPlus{Dim} <: AbstractInnerProduct end

"""
    InnerPlusSet{S} <: AbstractInnerProduct

Quadrature weights for the modified \$L^2_+\$ inner product staggered in every direction the
set `S` names at once, for `|S| \\geq 2` (gpena/Bramble.jl#115, #234).

The empty set is [`InnerH`](@ref) and a one-element set is [`InnerPlus`](@ref): those two
keep their own node types rather than becoming a special case of this one, because other
code matches on their literal types (`src/form/kronecker.jl`'s separability match, and the
`typeof(inner₊ₓ(id, id)).parameters[2] === InnerPlus{1}`-style pin in
`test/form/inner_products.jl`) and widening what they resolve to would change what those
match. [`inner₊`](@ref)`(u, v, Val(S))` is what builds this node; it is never constructed
for `S` shorter than 2.

`S` is stored sorted (`_canonical_set`, this file) so two callers naming the same set in a
different order -- `Val((1,2))` and `Val((2,1))` -- produce the identical singleton type,
which is what lets the like-term simplifier fold them and `which(inner₊, ...)` resolve one
way regardless of the order the caller wrote `S` in.
"""
struct InnerPlusSet{S} <: AbstractInnerProduct end

"""
    InnerGamma{MASK} <: AbstractInnerProduct

Quadrature weights for the \$(D-1)\$-dimensional surface integral over the grid faces `MASK`
names (gpena/Bramble.jl#157).

`MASK` is the per-axis `(min, max)` face mask `_face_mask` (mesh/queries.jl) resolves the
form's `markers` into, carried as a type parameter rather than a field so the node stays a
singleton the like-term simplifier can fold and `compute_weight` reads a compile-time literal.
It is resolved once, when the form is built.
"""
struct InnerGamma{MASK} <: AbstractInnerProduct end

"""
    BilinearProduct{D,InnerType,LeftType,RightType} <: LazyOp{D}

An AST node representing a bilinear integration term \$(u, v)\$ in a bilinear form.
"""
struct BilinearProduct{
    D, InnerType <: AbstractInnerProduct, LeftType <: LazyOp{D}, RightType <: LazyOp{D}
} <: LazyOp{D}
    left_op::LeftType
    right_op::RightType
end

"""
    LinearProduct{D,InnerType,LeftType,RightType} <: LazyOp{D}

An AST node representing a linear integration term \$(f, v)\$ in a linear form.
"""
struct LinearProduct{
    D, InnerType <: AbstractInnerProduct, LeftType <: LazyOp{D}, RightType <: LazyOp{D}
} <: LazyOp{D}
    left_op::LeftType
    right_op::RightType
end

# ==============================================================================
# Weight Helpers
# ==============================================================================

@inline compute_weight(::InnerH, space, I::CartesianIndex{D}, lin_idx::Int) where {D} = weights(space, Innerh())[lin_idx]

@inline compute_weight(
    ::InnerPlus{ActiveDim}, space, I::CartesianIndex{D}, lin_idx::Int
) where {ActiveDim, D} = weights(space, Innerplus(), ActiveDim)[lin_idx]

# `weights(space, Val(S))` for `length(S) >= 2` is a `SeparableWeights` (scalar_gridspace.jl),
# a lazy per-axis product with no full-grid vector behind it. It answers a `CartesianIndex`
# directly, at whatever point this is called for -- the currently-visited one, or a shifted
# neighbour, whichever `local_stencil` passes in -- with no linear-index division/modulo, so
# `I` is used here rather than `lin_idx` (the dense-vector path above keeps using `lin_idx`,
# unchanged).
@inline compute_weight(
    ::InnerPlusSet{S}, space, I::CartesianIndex{D}, lin_idx::Int
) where {S, D} = weights(space, Val(S))[I]

# The surface weight is computed from the mesh's live half-spacings rather than read from a
# stored vector, so `SpaceWeights` grows no family for it and there is no staleness token to
# keep: the whole contract a weight owes the assembly layer is this one scalar per point, and
# `_surface_weight` (mesh/queries.jl) answers it in a few multiplies. The same function
# answers for the numeric `inner_Γ`, so the two layers agree by construction rather than by
# two implementations being written to match.
@inline compute_weight(
    ::InnerGamma{MASK}, space, I::CartesianIndex{D}, lin_idx::Int
) where {MASK, D} = _surface_weight(mesh(space), MASK, I)

# ==============================================================================
# User-Facing API & Overloads
# ==============================================================================

#=
`markers`, on every `innerₕ`/`inner₊`/`inner₊ₓ`/`inner₊ᵧ`/`inner₊₂` below (both bilinear
and linear forms), restricts the product to the union of the labelled regions: a mask on
which grid points the assembled term contributes to at all, the symbolic counterpart of the
numeric `markers` keyword on `space/inner_product.jl`'s versions of the same names.

Implemented by wrapping the built `BilinearProduct`/`LinearProduct` in `RegionRestriction`
(`restrict_to`). That node returns an empty stencil off-region and the term's own stencil on
it, which is the mask required, and is supported by all existing AST walkers: block routing
(`trial_component_or_nothing`/`test_component_or_nothing`), `resolve_ast`, `is_symbolic`.
=#

@inline _restrict_by_markers(prod::LazyOp{D}, ::NTuple{0, Symbol}) where {D} = prod

# A single marker unwraps to a bare `Symbol` region rather than a one-element tuple, so it
# matches `restrict_to`'s own convention exactly, including `:interior`, which is a keyword
# `RegionRestriction` special-cases only when `region` is literally a `Symbol`, not a tuple
# containing one.
@inline function _restrict_by_markers(prod::LazyOp{D}, markers::NTuple{1, Symbol}) where {D}
    RegionRestriction{D, Symbol, typeof(prod)}(markers[1], prod)
end

@inline function _restrict_by_markers(
        prod::LazyOp{D}, markers::NTuple{N, Symbol}
) where {D, N}
    RegionRestriction{D, typeof(markers), typeof(prod)}(markers, prod)
end

#=
Every public spelling below differs from its neighbours in exactly two ways: which weight
the product carries, and how a bare left operand becomes a source node. The weight is
already a type -- `InnerH`, `InnerPlus{Dim}` -- and already the second parameter of the node
being built, so it is passed as its own singleton rather than encoded a second time in a
function name and then hand-written once per name (gpena/Bramble.jl#58).

`_product` decides linear-versus-bilinear; `_inner` adds the marker restriction on top.
`_linear_source` wraps a `Function`/`Number`/`VectorElement` left operand and is always
linear, since a bare source is never a trial function; `_inner_source` adds the restriction.
Each pair is split that way because the tuple forms fold several terms and have to restrict
the sum once rather than each term separately.
=#

# A source on the left makes the product linear; anything else makes it bilinear.
@inline function _product(
        ::W, left::LazyOp{D}, right::LazyOp{D}
) where {W <: AbstractInnerProduct, D}
    return if _is_source_only(left)
        LinearProduct{D, W, typeof(left), typeof(right)}(left, right)
    else
        _check_one_interpolated_side(left, right)
        BilinearProduct{D, W, typeof(left), typeof(right)}(left, right)
    end
end

# A term may interpolate one side or the other, never both (gpena/Bramble.jl#10, restated in
# #263). Whichever side stays native is the one supplying the mesh with the quadrature
# weights and the grid the sweep walks; with both sides interpolated there is no such mesh,
# and the operators are pure nodal blends carrying no quadrature of their own to stand in for
# one. Decided by type, so an ordinary product pays nothing for the check.
@inline function _check_one_interpolated_side(left, right)
    if _has_trial_interp(left) && _has_test_interp(right)
        _throw_both_sides_interpolated(left, right)
    end
    return nothing
end

@noinline function _throw_both_sides_interpolated(left, right)
    throw(
        ArgumentError(
        "a bilinear term cannot interpolate both sides: got πₕ on the trial side " *
        "($(typeof(left))) and on the test side ($(typeof(right))). One side has to stay " *
        "native, since it is the side whose mesh carries the quadrature weight the product " *
        "integrates against; the interpolation operators are nodal blends and have no " *
        "quadrature of their own. Interpolate the trial side or the test side, not both.",
    ),
    )
end

@inline _inner(w::AbstractInnerProduct, left::LazyOp, right::LazyOp, markers) = _restrict_by_markers(_product(w, left, right), markers)

# `parent` is Julia's own name for the storage a VectorElement delegates to; see
# `src/space/vectorelement.jl`.
@inline _as_source(l::Function, ::Val{D}) where {D} = SourceFunction{D, typeof(l)}(l)
@inline _as_source(l::Number, ::Val{D}) where {D} = source_number(l, Val(D))
@inline _as_source(l::VectorElement, ::Val{D}) where {D} = SourceVector{D, typeof(parent(l))}(parent(l))
@inline _as_source(d::DiracSource{D}, ::Val{D}) where {D} = d

@inline function _linear_source(::W, l, r::LazyOp{D}) where {W <: AbstractInnerProduct, D}
    sf = _as_source(l, Val(D))
    return LinearProduct{D, W, typeof(sf), typeof(r)}(sf, r)
end

@inline _inner_source(w::AbstractInnerProduct, l, r::LazyOp, markers) = _restrict_by_markers(_linear_source(w, l, r), markers)

# One directional source term per dimension, summed, with the sum restricted once. `Val(D)`
# is what keeps `dim` a compile-time literal, so `InnerPlus{dim}` stays a type instead of
# becoming a runtime value -- the same reason the folds below were written with `ntuple`.
@inline function _inner_source_tuple(l, r::NTuple{D, LazyOp{D}}, markers) where {D}
    terms = ntuple(Val(D)) do dim
        _linear_source(InnerPlus{dim}(), l[dim], r[dim])
    end
    return _restrict_by_markers(foldl(+, terms), markers)
end

"""
    inner_plus(left::NTuple{D, LazyOp{D}}, right::NTuple{D, LazyOp{D}}; markers = ()) -> LazyOp{D}

Constructs the sum of directional modified \$L^2_+\$ inner products across all dimensions.

Each dimension's term is a `LinearProduct` or a `BilinearProduct` independently, following
[`_is_source_only`](@ref) on `left[dim]` exactly as [`innerₕ`](@ref) does: a gradient tuple
of interpolated sources (`πₕ(u1), πₕ(u2)`) is source-only dimension by dimension.
"""
function inner_plus(
        left::NTuple{D, LazyOp{D}},
        right::NTuple{D, LazyOp{D}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    terms = ntuple(Val(D)) do dim
        return _product(InnerPlus{dim}(), left[dim], right[dim])
    end
    return _restrict_by_markers(foldl(+, terms), markers)
end

# Support both scalar and NTuple combinations in standard inner products:

"""
    innerₕ(left::LazyOp{D}, right::LazyOp{D}; markers = ()) -> LazyOp{D}

Constructs a symbolic \$L^2\$ inner product between `left` and `right`: a `LinearProduct`
(source × test) if `left` is source-only ([`_is_source_only`](@ref): a source, or a source
wrapped in differences/averages/shifts/jumps/restrictions/scales, never a trial function),
or a `BilinearProduct` (trial × test) otherwise.

This applies specifically when `left` is a `LazyOp`: a bare `Function`/`Number`/`VectorElement`
is unconditionally a source, so those overloads build a `LinearProduct` directly. When
`left` arrives already wrapped (`πₕ(uₕ)` or `D₋ₓ(πₕ(uₕ))`), this check ensures the correct
linear AST node is constructed.

`markers` restricts the assembled term to the union of the labelled regions: a mask on
which grid points it contributes to at all.
"""
function innerₕ(
        left::LazyOp{D}, right::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner(InnerH(), left, right, markers)
end

"""
    inner_Γ(left::LazyOp{D}, right::LazyOp{D}; markers) -> LazyOp{D}
    inner_Γ(g::Union{Function, Number, VectorElement}, v::LazyOp{D}; markers) -> LazyOp{D}

Constructs a symbolic ``(D-1)``-dimensional surface integral over the grid faces `markers`
names, the term a natural boundary condition is written with:

```math
\\int_{\\Gamma_N} g\\, v \\, ds \\qquad\\text{and}\\qquad \\int_{\\Gamma_R} \\beta\\, u\\, v \\, ds
```

A source on the left gives a [`LinearForm`](@ref) contribution (the Neumann flux vector); a
trial function gives a [`BilinearForm`](@ref) one (the Robin boundary mass). The weight is the
same lumped surface weight the numeric [`inner_Γ`](@ref) uses, so `uᵀ A v` equals
`inner_Γ(uₕ, vₕ, markers...)` for the bilinear mass term.

`markers` is a keyword here, matching [`innerₕ`](@ref), where the numeric twin takes its
labels positionally; and it names whole coordinate faces (`:boundary`, `:xmin`…`:zmax`, or a
viewpoint alias), for the reason the numeric one's docstring gives.

The assembled block carries a structural entry at every point the surrounding operators
reach, not only on `Γ`: the weight is zero off the surface rather than the entry being
absent. For a Robin term added to a Laplacian, which is what this is for, those entries are
in the pattern already.

# Examples

```julia
# -Δu = f with a Neumann flux g on the right edge and Robin data on the top one
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + inner_Γ(β * u, v; markers = (:ymax,)))
l = form(Wₕ, v -> innerₕ(fₕ, v) + inner_Γ(g, v; markers = (:xmax,)))
```

See also: [`innerₕ`](@ref), [`inner₊`](@ref)
"""
function inner_Γ(left::LazyOp{D}, right::LazyOp{D}; markers = ()) where {D}
    return _product(_inner_gamma(Val(D), markers), left, right)
end

function inner_Γ(l::Function, r::LazyOp{D}; markers = ()) where {D}
    return _linear_source(_inner_gamma(Val(D), markers), l, r)
end
function inner_Γ(l::Number, r::LazyOp{D}; markers = ()) where {D}
    return _linear_source(_inner_gamma(Val(D), markers), l, r)
end
function inner_Γ(l::VectorElement, r::LazyOp{D}; markers = ()) where {D}
    return _linear_source(_inner_gamma(Val(D), markers), l, r)
end

# A bare symbol is the common spelling and reads better than a one-tuple. Normalized here
# rather than by a second method, since a keyword's type does not take part in dispatch.
@inline _as_labels(s::Symbol) = (s,)
@inline _as_labels(t::NTuple{N, Symbol}) where {N} = t

# Resolving the labels is a runtime computation producing a *type*, so it happens once, here,
# when the form is built -- not once per grid point. The node that comes out is concrete, so
# assembly is as typed as it is for any other weight.
@inline function _inner_gamma(::Val{D}, markers) where {D}
    labels = _as_labels(markers)
    mask = _face_mask(Val(D), labels)
    _no_faces(mask) && _throw_no_surface_labels()
    return InnerGamma{mask}()
end

# There is deliberately no `innerₕ` over gradient tuples. `inner₊` has one because its
# weights are directional and the tuple is what supplies the directions; `InnerH` carries a
# single weight, so summing the components is a plain sum with nothing to infer, and it is
# written out at the call site rather than hidden behind the same spelling as the scalar
# product.

"""
    inner₊(left::LazyOp{1}, right::LazyOp{1}; markers = ())
    inner₊(left::NTuple{D,LazyOp{D}}, right::NTuple{D,LazyOp{D}}; markers = ()) where D

Constructs a symbolic modified \$L^2_+\$ inner product between `left` and `right`.

`markers` restricts the sum as it does for [`innerₕ`](@ref).
"""
function inner₊(
        left::NTuple{D, LazyOp{D}},
        right::NTuple{D, LazyOp{D}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return inner_plus(left, right; markers = markers)
end

"""
    inner₊(left::BackwardDifference{D,Dim}, right::BackwardDifference{D,Dim}) -> LazyOp{D}

`inner₊` of two backward differences taken along the same direction, which is the weight
the product carries: `InnerPlus{Dim}`.

In one dimension there is only one direction, so `inner₊(left, right)` already answers.
Above one dimension a bare `inner₊` of two operators names no direction, and the
weights are directional: the direction is read off the nodes. This is what makes
`inner₊(D₋ₓ(u), D₋ₓ(v))` mean what it reads as.

Backward differences only, as everywhere `inner₊` meets a difference: the weights are those
of the summation-by-parts identity, which pairs them with a backward difference.

Constructs a `LinearProduct` (source × test) if `left` is source-only ([`_is_source_only`](@ref)),
or a `BilinearProduct` otherwise, matching [`innerₕ`](@ref).

`markers` restricts the sum as it does for [`innerₕ`](@ref).
"""
function inner₊(
        left::BackwardDifference{D, Dim},
        right::BackwardDifference{D, Dim};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, Dim, N}
    return _inner(InnerPlus{Dim}(), left, right, markers)
end

"""
    inner₊(left::LazyOp{D}, right::LazyOp{D}) -> LazyOp{D}

Symbolic `inner₊` of two operators neither of which names a direction.

In one dimension there is only one direction to name, so this is the product, with weight
`InnerPlus{1}`. Above one dimension the weights are directional and nothing here supplies
the direction, so it throws an `ArgumentError`.

`markers` restricts the sum as it does for [`innerₕ`](@ref).
"""
function inner₊(
        left::LazyOp{D}, right::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner₊_same_dim(Val(D), left, right, markers)
end

# Split on `Val(D)` rather than branching on `D == 1` at runtime: `D` is a type parameter,
# known at compile time, so the choice belongs at dispatch (gpena/Bramble.jl#59). Kept as an
# inner helper, not a second `inner₊` method on `LazyOp{1}`, `LazyOp{1}`: that concrete-D
# signature is no longer a subtype of the `BackwardDifference{D,Dim}`-paired overloads below
# (unlike this method's shared, still-generic-in-D one), and is genuinely ambiguous against
# them for D=1 -- confirmed by trying it first and watching precompilation fail on exactly
# that call shape.
function _inner₊_same_dim(::Val{1}, left, right, markers::NTuple{N, Symbol}) where {N}
    return _inner(InnerPlus{1}(), left, right, markers)
end

function _inner₊_same_dim(::Val{D}, left, right, markers::NTuple{N, Symbol}) where {D, N}
    return _inner₊_no_direction(left, right, D)
end

@noinline function _inner₊_no_direction(left, right, D::Int)
    throw(
        ArgumentError(
        "inner₊ of two symbolic operators in $D dimensions names no direction, and its " *
        "weights are directional. Write inner₊ₓ, inner₊ᵧ or inner₊₂ for a specific one, " *
        "pass gradient tuples such as inner₊(∇ₕ(u), ∇ₕ(v)) to sum over all of them, or " *
        "difference both sides along the same direction as in inner₊(D₋ₓ(u), D₋ₓ(v)). " *
        "Got $(typeof(left)) and $(typeof(right)).",
    ),
    )
end

"""
    inner₊(left::LazyOp{D}, right::BackwardDifference{D,Dim}) -> LazyOp{D}
    inner₊(left::BackwardDifference{D,Dim}, right::LazyOp{D}) -> LazyOp{D}

`inner₊` where one side is a backward difference and the other is not: the difference names
the direction, so the product carries `InnerPlus{Dim}`.

This is what `inner₊(u, D₋ₓ(v))` means: the common form, and the one the coupled
pressure-velocity terms are written in, `inner₊(p, D₋ₓ(v[1]))` with `p` a symbolic scalar
field. It is not restricted to indexed leaves: a plain `TrialFunction` reads the
direction off the difference just as an `IndexedTrialFunction` does.

Backward differences only, as everywhere `inner₊` meets a difference.

A `LinearProduct` if `left` is source-only ([`_is_source_only`](@ref)), a `BilinearProduct`
otherwise, matching [`innerₕ`](@ref).

`markers` restricts the sum as it does for [`innerₕ`](@ref).
"""
function inner₊(
        left::LazyOp{D},
        right::BackwardDifference{D, Dim};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, Dim, N}
    return _inner(InnerPlus{Dim}(), left, right, markers)
end
function inner₊(
        left::BackwardDifference{D, Dim},
        right::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, Dim, N}
    return _inner(InnerPlus{Dim}(), left, right, markers)
end

"""
    inner₊(left::BackwardDifference{D,Dim1}, right::BackwardDifference{D,Dim2}) where {D,Dim1,Dim2}

Rejects `inner₊` of two backward differences taken along *different* directions.

Each side names a direction and they disagree, so there is no one weight the product
carries. This also has to be written out rather than left to dispatch: with a difference on
either side, the two single-sided methods above tie, and the pair would be an ambiguity
rather than an error the caller can read.
"""
@noinline function inner₊(
        left::BackwardDifference{D, Dim1},
        right::BackwardDifference{D, Dim2};
        markers::NTuple{M, Symbol} = NTuple{0, Symbol}()
) where {D, Dim1, Dim2, M}
    throw(
        ArgumentError(
        "inner₊ of backward differences along different directions ($Dim1 and $Dim2) " *
        "names no single weight. Difference both sides along the same direction, or " *
        "write inner₊ₓ, inner₊ᵧ or inner₊₂ for the one you mean.",
    ),
    )
end

"""
    inner₊(left::NTuple{N,<:Tuple}, right::NTuple{N,<:Tuple}) where N

Vector-field `inner₊`: sums per-component inner products.
Used when `left` and `right` are **tuples of gradient tuples**, e.g.
`inner₊(∇ₕ(u), ∇ₕ(v))` where `u = (u1, u2)` is a velocity tuple.
Each element pair `(left[k], right[k])` is a `D`-tuple of `LazyOp` (a gradient),
which dispatches to the existing `inner₊(::NTuple{D,LazyOp}, ::NTuple{D,LazyOp})`.

This overload is intentionally restricted to `NTuple{N,<:Tuple}` so it does **not**
interfere with `inner₊(NTuple{D,VectorElement}, NTuple{D,VectorElement})` handled by
the `@generated` method in `inner_product.jl`. That restriction is what separates this
file's symbolic family from that file's numeric one; it is asserted in
`test/form/inner_products.jl`, testset "Symbolic and numeric families stay apart".

`markers` restricts the whole sum, not each component separately, as it does for
[`innerₕ`](@ref).
"""
function inner₊(
        left::NTuple{N, <:Tuple},
        right::NTuple{N, <:Tuple};
        markers::NTuple{M, Symbol} = NTuple{0, Symbol}()
) where {N, M}
    return _restrict_by_markers(foldl(+, map(inner₊, left, right)), markers)
end

# ==============================================================================
# The general staggered-set entry point, inner₊(u, v, Val(S))
# ==============================================================================

"""
    inner₊(left::LazyOp{D}, right::LazyOp{D}, ::Val{S}; markers = ()) where {D, S}

Constructs the symbolic modified \$L^2_+\$ inner product staggered in the direction set
`S ⊆ 1:D` (gpena/Bramble.jl#115, #234): the weight at grid index `I` is
``\\prod_{d \\in S} h_d(I_d) \\cdot \\prod_{d \\notin S} h_d(I_d + 1/2)``, matching
[`weights`](@ref)`(Wₕ, Val(S))`.

This is the general entry point [`innerₕ`](@ref) (`S = ()`) and
[`inner₊ₓ`](@ref)/[`inner₊ᵧ`](@ref)/[`inner₊₂`](@ref) (`S` a singleton) are aliases of: for
those two shapes it builds exactly the same [`InnerH`](@ref)/[`InnerPlus`](@ref) node they
do, not a new node under a shared name, so a term written either way folds the same way in
the simplifier and resolves to the same `which(inner₊, ...).file`. Every other `S`
(`|S| ≥ 2`) builds an [`InnerPlusSet`](@ref) node, whose weight at assembly comes from the
lazy `SeparableWeights` [`weights`](@ref)`(Wₕ, Val(S))` returns for those sets.

`S` must be a subset of `1:D` with no axis repeated, and is accepted in any order --
`Val((1,2))` and `Val((2,1))` build the identical node.

Constructs a `LinearProduct` if `left` is source-only ([`_is_source_only`](@ref)), a
`BilinearProduct` otherwise, matching [`innerₕ`](@ref).

`markers` restricts the sum as it does for [`innerₕ`](@ref).

# Examples

```julia
inner₊(u, v, Val((1, 2)))      # xy-edge centres in 3D, an ε_{12}-style placement
inner₊(u, v, Val((1, 2, 3)))   # cell centres in 3D, a divₕ-style placement
inner₊(u, v, Val(()))          # what innerₕ(u, v) is
inner₊(u, v, Val((1,)))        # what inner₊ₓ(u, v) is
```
"""
@inline function inner₊(
        left::LazyOp{D}, right::LazyOp{D}, ::Val{S};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, S, N}
    return _inner₊_val(left, right, Val(D), Val(S), markers)
end

# `S = ()` is innerₕ's own weight -- routed to InnerH rather than InnerPlusSet{()}, so the
# two spellings are the same node (see InnerPlusSet's docstring for why that matters).
@inline _inner₊_val(left, right, ::Val{D}, ::Val{()}, markers) where {D} = _inner(InnerH(), left, right, markers)

@inline function _inner₊_val(left, right, ::Val{D}, ::Val{S}, markers) where {D, S}
    _check_staggered_set(Val(D), Val(S))
    return _inner₊_val(left, right, Val(D), Val(S), Val(length(S)), markers)
end

# A singleton `S` is inner₊ₓ/ᵧ/₂'s own weight -- routed to InnerPlus{only(S)} rather than
# InnerPlusSet{S}, for the same reason as the `S = ()` case above.
@inline _inner₊_val(left, right, ::Val{D}, ::Val{S}, ::Val{1}, markers) where {D, S} = _inner(
    InnerPlus{only(S)}(), left, right, markers
)

# `|S| >= 2`: genuinely a new node, since neither InnerH nor InnerPlus{Dim} names more than
# one (or zero) directions. `_canonical_set` sorts `S` so the node built from `Val((1,2))`
# and `Val((2,1))` is the same type.
@inline function _inner₊_val(left, right, ::Val{D}, ::Val{S}, ::Val{K}, markers) where {D, S, K}
    return _inner(InnerPlusSet{_canonical_set(S)}(), left, right, markers)
end

@inline function _check_staggered_set(::Val{D}, ::Val{S}) where {D, S}
    (allunique(S) && all(d -> 1 <= d <= D, S)) || _throw_invalid_staggered_set(S, D)
    return nothing
end

@noinline function _throw_invalid_staggered_set(S, D)
    throw(
        ArgumentError(
        "inner₊(u, v, Val(S)) got S = $S, which is not a subset of 1:$D with no repeated " *
        "axes: every entry of S must be between 1 and the space's dimension ($D), and name " *
        "each axis at most once.",
    ),
    )
end

# `D <= 3` everywhere in this package (`dim(Wₕ)` is documented 1, 2 or 3), so a valid `S`
# (checked above: distinct entries, each in `1:D`) never has more than 3 entries -- these
# two are the only arities `_canonical_set` needs, both allocation-free sorting networks
# rather than a generic `sort`, since this runs once per `form(...)` call, not per point,
# but still on the same "no unnecessary allocation" footing as the rest of this file.
@inline _canonical_set(S::NTuple{2, Int}) = S[1] < S[2] ? S : (S[2], S[1])

@inline function _canonical_set(S::NTuple{3, Int})
    a, b, c = S
    a, b = a < b ? (a, b) : (b, a)
    b, c = b < c ? (b, c) : (c, b)
    a, b = a < b ? (a, b) : (b, a)
    return (a, b, c)
end

"""
    inner₊ₓ(left::LazyOp{D}, right::LazyOp{D}) where D
    inner₊ᵧ(left::LazyOp{D}, right::LazyOp{D}) where D
    inner₊₂(left::LazyOp{D}, right::LazyOp{D}) where D

Constructs directional modified \$L^2_+\$ inner products in x, y, and z directions.

A `LinearProduct` if `left` is *source-only* ([`_is_source_only`](@ref)), a `BilinearProduct`
otherwise, exactly as [`innerₕ`](@ref) decides.

`markers` restricts the sum as it does for [`innerₕ`](@ref).
"""
function inner₊ₓ(
        left::LazyOp{D}, right::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner(InnerPlus{1}(), left, right, markers)
end
function inner₊ᵧ(
        left::LazyOp{D}, right::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner(InnerPlus{2}(), left, right, markers)
end
function inner₊₂(
        left::LazyOp{D}, right::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner(InnerPlus{3}(), left, right, markers)
end

@inline function source_number(l::Number, ::Val{D}) where {D}
    return SourceConstant{D, typeof(l)}(l)
end

# Linear Forms (e.g. innerₕ(f, v) where f is a Function, Number, or VectorElement and v is
# TestFunction). `markers` restricts each the same way it does the bilinear forms above: a
# mask on which grid points the source term contributes to at all.
function innerₕ(
        l::Function, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerH(), l, r, markers)
end
function innerₕ(
        l::Number, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerH(), l, r, markers)
end
function innerₕ(
        l::VectorElement, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerH(), l, r, markers)
end

function inner₊(
        l::Function, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{1}(), l, r, markers)
end
function inner₊(
        l::Number, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{1}(), l, r, markers)
end
function inner₊(
        l::VectorElement, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{1}(), l, r, markers)
end

function inner₊(
        l::NTuple{D, Function},
        r::NTuple{D, LazyOp{D}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source_tuple(l, r, markers)
end
function inner₊(
        l::NTuple{D, Number},
        r::NTuple{D, LazyOp{D}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source_tuple(l, r, markers)
end
@inline function inner₊(
        l::NTuple{D, VectorElement},
        r::NTuple{D, LazyOp{D}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    if all(is_symbolic, r)
        return _inner_source_tuple(l, r, markers)
    else
        return _inner₊_numeric_tuple_unsupported(r)
    end
end

# The right-hand side in this branch carries no trial or test function, so there is
# nothing for the product to be a form in: `∇ₕ(IdentityOperator(Wₕ))` has no argument to
# differentiate.
@noinline function _inner₊_numeric_tuple_unsupported(r)
    throw(
        ArgumentError(
        "inner₊ of a tuple of grid functions against a tuple of non-symbolic operators " *
        "has no definition: the right-hand side carries no trial or test function, so " *
        "there is nothing for the product to be a form in. Got $(typeof(r)). Pair the " *
        "grid functions with a symbolic gradient such as ∇ₕ(u), or take the numeric " *
        "inner₊ of two grid functions directly.",
    ),
    )
end

function inner₊ₓ(
        l::Function, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{1}(), l, r, markers)
end
function inner₊ᵧ(
        l::Function, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{2}(), l, r, markers)
end
function inner₊₂(
        l::Function, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{3}(), l, r, markers)
end

function inner₊ₓ(
        l::Number, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{1}(), l, r, markers)
end
function inner₊ᵧ(
        l::Number, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{2}(), l, r, markers)
end
function inner₊₂(
        l::Number, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{3}(), l, r, markers)
end

function inner₊ₓ(
        l::VectorElement, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{1}(), l, r, markers)
end
function inner₊ᵧ(
        l::VectorElement, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{2}(), l, r, markers)
end
function inner₊₂(
        l::VectorElement, r::LazyOp{D}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {D, N}
    return _inner_source(InnerPlus{3}(), l, r, markers)
end

# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

# `_same_operator_shape` answers, for `innerₕ(L(u), L(v))` with the same `L`
# on both sides, that `op.left_op` and `op.right_op` are the same operator chain up to
# substituting `TrialFunction` for `TestFunction` at the leaves: exactly the
# condition under which `local_stencil(op.left_op, …)` and `local_stencil(op.right_op, …)`
# compute the identical tuple of `(offset, coefficient)` pairs. The two are then multiplied
# pairwise regardless, so half of those products (`left[i][2]*left[j][2]` and
# `left[j][2]*left[i][2]`) are the same number computed twice.
# `multiply_stencils_bilinear_symmetric` computes each such product once and reuses it for
# both `(i, j)` and `(j, i)`, so the fast path below runs `local_stencil` on one side only and
# still returns the same `N²`-entry tuple `multiply_stencils_bilinear` would have, just with
# `N(N+1)/2` multiplications behind it instead of `N²`.
#
# This check depends only on `op.left_op`/`op.right_op`'s structure, not on which space the
# trial and test argument range over: unlike `issymmetric`/`isposdef`,
# which additionally require the same space, it answers a strictly local question ("do the two
# sides compute the same numbers here") and is safe regardless.
@inline function local_stencil(
        op::BilinearProduct{D, InnerType}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D, InnerType}
    vol = compute_weight(InnerType(), space, I, lin_idx)
    if _same_operator_shape(op.left_op, op.right_op)
        stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
        return multiply_stencils_bilinear_symmetric(stencil, vol)
    end
    left_stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
    right_stencil = local_stencil(op.right_op, space, I, markers, lin_idx)
    return multiply_stencils_bilinear(left_stencil, right_stencil, vol)
end

# The left factor of a linear product is contracted to a scalar:
# `multiply_stencils_linear` keeps only the *right* operand's offsets and multiplies the
# coefficients, so the assembly sums the left stencil's coefficients and discards where each
# one sat. That is exact only when the left stencil is a single entry, at offset zero,
# carrying the factor's true value at this point: an invariant the code relies on, and which
# a source under an operator breaks: `D₋ₓ(f)` stencils as the same value at two offsets with
# opposite signs, so the sum is zero.
#
# Fixed by reading the source-only subtree's own `local_stencil` and discarding its offsets
# (`sum_stencil_values`). That reads correctly via `stencil_shift_trait`:
# a source is `PointDependentStencil`, so every neighbour a wrapping operator reaches is
# obtained by re-evaluating the subtree at that neighbour's own point
# (`shifted_inner_stencil`) rather than by relabelling the offset.
@noinline function _throw_source_not_point_dependent(op)
    throw(
        ArgumentError(
        "`_is_source_only` accepted $(typeof(op)) as a source, but `stencil_shift_trait` " *
        "does not mark it (or a node it wraps) `PointDependentStencil`. Contracting it would " *
        "relabel offsets instead of re-reading the source at each neighbour. " *
        "Add the missing `stencil_shift_trait` method next to the node's definition.",
    ),
    )
end

# Called only on a `LinearProduct`'s own `left_op`, which is source-only by construction:
# every constructor above chooses `LinearProduct` over `BilinearProduct` precisely by
# checking `_is_source_only(left)` first. So the contraction below is unconditional.
@inline function _contracted_left_stencil(
        op, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    stencil_shift_trait(op) isa PointDependentStencil ||
        _throw_source_not_point_dependent(op)
    stencil = local_stencil(op, space, I, markers, lin_idx)
    return ((zero_offset(Val(D)), sum_stencil_values(stencil)),)
end

@inline _is_dirac(::DiracSource) = true
@inline _is_dirac(op::OperatorScale) = _is_dirac(op.inner_op)
@inline _is_dirac(op::GridFunctionScale) = _is_dirac(op.inner_op)
@inline _is_dirac(::LazyOp) = false
@inline _is_dirac(::Any) = false

@inline function local_stencil(
        op::LinearProduct{D, InnerType}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D, InnerType}
    left_stencil = _contracted_left_stencil(op.left_op, space, I, markers, lin_idx)
    right_stencil = local_stencil(op.right_op, space, I, markers, lin_idx)
    vol = _is_dirac(op.left_op) ? 1 : compute_weight(InnerType(), space, I, lin_idx)
    return multiply_stencils_linear(left_stencil, right_stencil, vol)
end

# ==============================================================================
# AST Resolution
# ==============================================================================

function resolve_ast(op::BilinearProduct{D, InnerType}) where {D, InnerType}
    return BilinearProduct{
        D, InnerType, typeof(resolve_ast(op.left_op)), typeof(resolve_ast(op.right_op))
    }(
        resolve_ast(op.left_op), resolve_ast(op.right_op)
    )
end
function resolve_ast(op::LinearProduct{D, InnerType}) where {D, InnerType}
    return LinearProduct{
        D, InnerType, typeof(resolve_ast(op.left_op)), typeof(resolve_ast(op.right_op))
    }(
        resolve_ast(op.left_op), resolve_ast(op.right_op)
    )
end

# Disambiguation for the empty tuple.
#
# The overloads below are written over `NTuple{D, …}` for several element types: LazyOp
# nodes, VectorElements, Functions, Numbers; any two of them overlap at `D = 0`,
# where `Tuple{}` satisfies both and neither signature is more specific. That is ten
# ambiguous pairs, which Aqua fails on.
#
# One method for `Tuple{}` settles all of them. It throws rather than returning zero: an
# empty tuple carries no direction to integrate over, so reaching here means a caller
# built a form with no components, and a silent zero would hide that.
#=
Needs its own `markers` keyword, not just the positional disambiguation: with `markers` added
to every `NTuple{D,...}` overload above, they all tie again at `D = 0` specifically for the
*keyword-call* dispatch (`Tuple{}` matches `NTuple{0,LazyOp{0}}`, `NTuple{0,Function}`,
`NTuple{0,Number}`, `NTuple{0,VectorElement}` and `NTuple{0,<:Tuple}` identically), even
though the plain positional call already resolves through this method with no keywords at
all. Aqua's ambiguity check is what caught it.
=#
@noinline function inner₊(
        ::Tuple{}, ::Tuple{}; markers::NTuple{M, Symbol} = NTuple{0, Symbol}()
) where {M}
    throw(ArgumentError("inner₊ needs at least one component; got two empty tuples"))
end

# ==============================================================================
# Expression rendering (gpena/Bramble.jl#274)
# ==============================================================================

# One name per weight type. `InnerGammaNormal` (normal.jl, a different subplan) adds its own
# method to this same generic function -- no forward declaration needed.
_inner_name(::InnerH) = "innerₕ"
_inner_name(::InnerPlus{Dim}) where {Dim} = "inner₊" * _BRAMBLE_var2symbol[Dim]
_inner_name(::InnerPlusSet{S}) where {S} = "inner₊"
_inner_name(::InnerGamma{MASK}) where {MASK} = "inner_Γ"

# Shared by BilinearProduct/LinearProduct below. `InnerPlusSet` additionally names its
# direction set as a third argument; every other weight renders as a plain two-argument call.
function _product_expression(inner::AbstractInnerProduct, left, right)
    "$(_inner_name(inner))($(expression(left)), $(expression(right)))"
end
function _product_expression(inner::InnerPlusSet{S}, left, right) where {S}
    "$(_inner_name(inner))($(expression(left)), $(expression(right)), $(S))"
end

function expression(op::BilinearProduct{D, InnerType}) where {D, InnerType}
    _product_expression(InnerType(), op.left_op, op.right_op)
end
function expression(op::LinearProduct{D, InnerType}) where {D, InnerType}
    _product_expression(InnerType(), op.left_op, op.right_op)
end
