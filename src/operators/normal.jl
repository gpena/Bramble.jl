# normal.jl
#
# The file holds both halves of the normal family: the numerical operators, which act on grid
# functions, and then the AST nodes that the same names build when handed a `LazyOp`. It is
# included after the AST core (`ast/ast.jl`, `common.jl`, `expression.jl`,
# `operators/node_family.jl`), which the node half needs.

# The outward unit normal as a discrete grid function (gpena/Bramble.jl#213).
#
# `normal_vector(Ωₕ, :xmax)` (mesh/queries.jl) answers the geometric question: the constant
# vector a named face points along. This answers the discrete one: the same vector *sampled
# on the grid*, zero away from the face, so it can multiply a grid function or be exported
# beside one. For a face name the two are the same number; only one of them has a degree of
# freedom per point to put it in. A mesh marker is resolved here as `inner_Γ` resolves it
# (`_normal_face_name`), while the mesh-level query reads every label as a face name.

"""
    normal_vector(Wₕ::ScalarGridSpace{D}, marker::Symbol) -> NTuple{D, VectorElement}

Returns the outward unit normal of the boundary face `marker` selects, as one grid function
per coordinate: component `d` holds that face's normal component `d` at every point of the
face and zero everywhere else.

`marker` is resolved as [`inner_Γ`](@ref) resolves its labels, so the two always agree on the
face. A label the mesh holds as a marker is read off its marked points, which means that
under `domain(S, :left => :ymin)` the label `:left` is the ymin face. Any other label is a
face name (`:xmin`…`:zmax` or a viewpoint alias), whose normal is
`normal_vector(mesh(Wₕ), marker)`. A normal belongs to one face. So a marker that covers
several faces (`:boundary`), or only part of one, throws an `ArgumentError`.

The components are zero off the face rather than the face's normal everywhere, so a product
against one restricts to the face without a second mask.

# Examples

```jldoctest
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
Wₕ = gridspace(Ωₕ)
nₕ = Bramble.normal_vector(Wₕ, :ymin)
(sum(parent(nₕ[1])), sum(parent(nₕ[2])))

# output

(0.0, -5.0)
```

See also: [`normal_vector`](@ref), [`inner_Γ`](@ref)
"""
function normal_vector(Wₕ::ScalarGridSpace{D}, marker::Symbol) where {D}
    Ωₕ = mesh(Wₕ)
    face = _normal_face_name(Ωₕ, marker)
    ν = normal_vector(Ωₕ, face)  # throws the ArgumentError for an unknown face name
    facet = boundary_symbol_to_cartesian(_facet_indices(Ωₕ))[face]
    lin = LinearIndices(indices(Ωₕ))

    T = eltype(Wₕ)
    nₕ = ntuple(_ -> element(Wₕ, zero(T)), Val(D))
    _fill_normal!(locality(backend(Wₕ)), nₕ, facet, ν, lin, Val(D))
    return nₕ
end

# The indices whose faces are the facets: a ghost mesh's physical nodes, in its padded
# index space, so its facets are the physical ones and not the ghost layer.
@inline _facet_indices(Ωₕ::AbstractMeshType) = indices(Ωₕ)
@inline _facet_indices(G::_GhostMeshLike) = physical_indices(G)

# The face name `marker` resolves to under `inner_Γ`'s rule (`_surface_mask`,
# mesh/queries.jl): a marker the mesh holds is read off its points, which must make up exactly
# one face; any other label is already a face name and is returned as it is.
@inline function _normal_face_name(Ωₕ::AbstractMeshType{D}, marker::Symbol) where {D}
    haskey(markers(Ωₕ), marker) || return marker
    mask = _surface_mask(Ωₕ, (marker,), "normal_vector")
    nfaces = sum(d -> mask[d][1] + mask[d][2], 1:D)
    nfaces == 1 || _throw_normal_not_one_face(marker, nfaces)
    axis = findfirst(d -> mask[d][1] || mask[d][2], 1:D)
    return _facet_name(axis, 1 + mask[axis][2])
end

@noinline function _throw_normal_not_one_face(marker::Symbol, nfaces::Int)
    throw(
        ArgumentError(
        "normal_vector: the marker :$marker covers $nfaces boundary faces; a normal is a " *
        "property of one face. Name the face, or a marker covering exactly one face.",
    ),
    )
end

# Host fill: visits only the facet slice (gpena/Bramble.jl#333), never the whole volume.
@inline function _fill_normal!(::HostLocality, nₕ, facet, ν, lin, ::Val{D}) where {D}
    @inbounds for I in facet
        k = lin[I]
        for d in 1:D
            parent(nₕ[d])[k] = ν[d]
        end
    end
    return nothing
end

# Device counterpart (gpena/Bramble.jl#311, #333): the components are already zeroed on the
# device, so only the facet's linear indices -- O(facet), not O(volume) -- are built on the
# host, moved to the device once, and each component is written there by indexed broadcast.
@noinline function _fill_normal!(::DeviceLocality, nₕ, facet, ν, lin, ::Val{D}) where {D}
    T = eltype(parent(nₕ[1]))
    host_idx = Int[lin[I] for I in facet]
    idx = similar(parent(nₕ[1]), Int, length(host_idx))
    copyto!(idx, host_idx)
    for d in 1:D
        view(parent(nₕ[d]), idx) .= T(ν[d])
    end
    return nothing
end

# ==============================================================================
# ==============================================================================
# The AST nodes: the outward normal and the normal derivative
# ==============================================================================
# ==============================================================================

# The outward normal as a symbol inside a form, and the normal derivative built on it
# (gpena/Bramble.jl#213).
#
# Why this is not a node.
#
# A `LazyOp` is scalar-valued and its `local_stencil` returns one coefficient per point per
# column, so `η` cannot be one: it is vector-valued, and what a form does with it is always
# to contract it against another vector-valued quantity. More decisively, the *weight* of a
# normal term is directional. A surface integral of a normal flux is
#
#     ∫_Γ (F · η) v ds = Σ_{faces F_d} ± ∫_{F_d} F_d v ds
#
# and the measure of the face with normal `d` is the transverse product `∏_{e ≠ d} ĥ_e`, not
# the lumped `ω` that `InnerGamma` carries. At a corner the two differ: `ω` sums the incident
# faces' measures into one number, which is right for `∫_Γ g v` and wrong for `∫_Γ (F·η) v`,
# where each face contributes its own component of `F` with its own sign.
#
# So `dot(F, η)` records the tuple, and `inner_Γ` expands it, at the builder, into one
# ordinary product per direction carrying `InnerGammaNormal{M, d}` -- the signed
# transverse measure of the faces with normal `d`. No new node type, no new stencil path, and
# the resulting AST is the sum a caller would have written by hand.

# The type behind `η`. A singleton, so `dot(F, η)` folds at compile time and the symbol costs
# nothing to carry through a form.
struct NormalSymbol end

"""
    η

The outward unit normal, as a symbol usable inside a form.

Only meaningful contracted against a vector-valued quantity whose values are known -- a flux
field, `dot(F, η)` -- and only inside [`inner_Γ`](@ref), which is where a normal has a face to
be normal to.

```julia
l = form(Wₕ, v -> inner_Γ(dot(Fₕ, η), v; markers = (:xmax,)))
```

The normal *derivative* of an unknown is [`∂ₙ`](@ref), which needs ghost points.

See also: [`inner_Γ`](@ref), [`normal_vector`](@ref), [`∂ₙ`](@ref)
"""
const η = NormalSymbol()

"""
    NormalContraction{T}

`dot(F, η)`: the tuple `F`, held until [`inner_Γ`](@ref) expands it into one product per
direction. Not a `LazyOp`, deliberately -- see the note at the head of this file.
"""
struct NormalContraction{T <: Tuple}
    terms::T
end

@inline dot(F::NTuple{D, Any}, ::NormalSymbol) where {D} = NormalContraction(F)
@inline dot(::NormalSymbol, F::NTuple{D, Any}) where {D} = dot(F, η)

# `dot(F, η)` needs no ghosts: `F` is a field whose values are already known at every point,
# so nothing has to be differenced at the face. The normal *derivative* of an unknown does,
# and is the facet operator `∂ₙ` further down.

"""
    InnerGammaNormal{M, DIM} <: AbstractInnerProduct

The quadrature weight of the faces whose normal points along `DIM`, signed by which side of
the axis the point sits on: `-∏_{e ≠ DIM} ĥ_e` on the `min` face, `+` on the `max` one, and
zero anywhere else.

The directional counterpart of [`InnerGamma`](@ref), which lumps every incident face into one
number. Both hold their labels in `mask` until `form` resolves them, once, into the face
mask, a field as `InnerGamma`'s is.
"""
struct InnerGammaNormal{M, DIM} <: AbstractInnerProduct
    mask::M
end

# Built from the labels type or face mask `M`, as `InnerGamma{M}()` is (operators/inner.jl).
@inline function (::Type{InnerGammaNormal{M, DIM}})() where {M, DIM}
    m = _mask_instance(M)
    return InnerGammaNormal{typeof(m), DIM}(m)
end

@inline function compute_weight(
        w::InnerGammaNormal{M, DIM}, space, I::CartesianIndex{D}, lin_idx::Int
) where {M, DIM, D}
    Ωₕ = mesh(space)
    mask = w.mask
    mask[DIM][1] && _on_facet(Ωₕ, I, DIM, 1) && return -_facet_measure(Ωₕ, I, Val(DIM))
    mask[DIM][2] && _on_facet(Ωₕ, I, DIM, 2) && return _facet_measure(Ωₕ, I, Val(DIM))
    return zero(eltype(Ωₕ))
end

# Whether the point `I` of the walked mesh is a node of facet `(d, s)`, `s` 1 for the
# minimum and 2 for the maximum, and the facet's transverse measure there. On a ghost mesh
# the facet is the physical one, inside the padded index space: a ghost is on no facet, and
# the measure is the physical mesh's (the padded mesh gives the end node a whole cell).
@inline function _on_facet(Ωₕ::AbstractMeshType, I::CartesianIndex, d::Int, s::Int)
    return I[d] == (s == 1 ? 1 : npoints(Ωₕ, Tuple)[d])
end
@inline function _on_facet(G::_GhostMeshLike, J::CartesianIndex, d::Int, s::Int)
    J in physical_indices(G) || return false
    return _on_facet(G.physical, J - CartesianIndex(G.lo), d, s)
end

@inline function _facet_measure(Ωₕ::AbstractMeshType{D}, I, ::Val{d}) where {D, d}
    return _transverse_measure(Ωₕ, I, Val(d), Val(D))
end
@inline function _facet_measure(G::_GhostMeshLike{D}, J, ::Val{d}) where {D, d}
    return _transverse_measure(G.physical, J - CartesianIndex(G.lo), Val(d), Val(D))
end

"""
    inner_Γ(F::NormalContraction, v; markers) -> LazyOp

The surface integral of a normal flux, ``\\int_\\Gamma (\\mathbf{F} \\cdot \\boldsymbol{\\eta})\\, v\\, ds``.

Expanded here, at the builder, into one ordinary product per direction, each carrying that
direction's signed transverse measure. The term is the sum a caller would otherwise write out,
so nothing downstream of `form` learns a new node.
"""
function inner_Γ(
        F::NormalContraction{<:NTuple{D, Any}}, right::LazyOp{D}; markers = ()) where {D}
    mask = _normal_mask(Val(D), markers)
    return _normal_terms(F.terms, right, Val(mask), Val(D))
end

# The labels, held until `form` resolves them to faces as it does `InnerGamma`'s
# (`_resolve_surfaces`, operators/inner.jl).
@inline function _normal_mask(::Val{D}, markers) where {D}
    labels = _as_labels(markers)
    isempty(labels) && _throw_no_surface_labels()
    return _SurfaceLabels{labels}
end

function _resolved_inner(::InnerGammaNormal{_SurfaceLabels{L}, DIM}, Ωₕ) where {L, DIM}
    mask = _surface_mask(Ωₕ, L)
    return InnerGammaNormal{typeof(mask), DIM}(mask)
end
_surface_labels(::Type{InnerGammaNormal{_SurfaceLabels{L}, DIM}}) where {L, DIM} = L

# The sum is built with `ntuple(Val(D))` so each weight is a type rather than a runtime value,
# the same reason `_inner_source_tuple` is written that way.
@inline function _normal_terms(terms, right, ::Val{MASK}, ::Val{D}) where {MASK, D}
    return sum(
        ntuple(Val(D)) do d
        _normal_term(InnerGammaNormal{MASK, d}(), terms[d], right)
    end
    )
end

@inline _normal_term(w, left::LazyOp, right::LazyOp) = _product(w, left, right)
@inline _normal_term(w, left, right::LazyOp) = _linear_source(w, left, right)

# --- Components of the normal (gpena/Bramble.jl#341) -------------------------------- #
#
# `ηₓ, ηᵧ = η` names one direction of the normal at a time. A component is a singleton like
# `η` itself, and a product with it is held, not built, until `inner_Γ` expands it into the
# single term `dot(F, η)` would have produced for that direction: the same `_normal_term`
# with the same `InnerGammaNormal{MASK, d}` weight. Sums and scalar multiples of such
# products are held too, as a tuple of products each carrying its own scale, so
# `f1 * ηₓ + f2 * ηᵧ`, integrated face by face, is `dot((f1, f2), η)` term for term, and
# nothing downstream sees a new node.

# What a component, a product with one, or a sum of such products has in common: each is
# meaningful only inside `inner_Γ`.
abstract type AbstractNormalTerm end

"""
    NormalComponent{DIM}

`η[DIM]`: the component of the outward normal along axis `DIM`, as a symbol usable inside a
form. Obtained by indexing or destructuring [`η`](@ref), never constructed directly.
"""
struct NormalComponent{DIM} <: AbstractNormalTerm end

# Three components, like the operator aliases, whatever the dimension: the one a form's
# dimension leaves over is refused inside `inner_Γ`, where the dimension is known.
@inline Base.length(::NormalSymbol) = 3
@inline Base.firstindex(::NormalSymbol) = 1
@inline Base.lastindex(::NormalSymbol) = 3
@inline function Base.getindex(::NormalSymbol, d::Integer)
    _check_direction_type(d)  # `η[true]` would otherwise be `η[1]` (gpena/Bramble.jl#689)
    return _normal_component(Val(Int(d)))
end
@inline Base.getindex(::NormalSymbol, s::Symbol) = _normal_component(Val(_normal_axis(s)))
@inline Base.iterate(::NormalSymbol, state::Int = 1) = state > 3 ? nothing : (η[state], state + 1)

@inline function _normal_component(::Val{d}) where {d}
    1 <= d <= 3 || throw(BoundsError(η, d))
    return NormalComponent{d}()
end

@inline function _normal_axis(s::Symbol)
    s === :x && return 1
    s === :y && return 2
    s === :z && return 3
    throw(ArgumentError("the normal has components :x, :y and :z; got :$s"))
end

"""
    NormalComponentProduct{DIM, T, S}

`c * (g * η[DIM])`: the factor `g` and the scalars `c`, held until [`inner_Γ`](@ref) gives
the factor the signed face selection of direction `DIM` and wraps the term in each scalar,
outermost first, exactly as `c * term` would. The scales are a tuple, empty when there are
none, so an unscaled product lowers to the bare term. Not a `LazyOp`, for the reason `η` is
not one.
"""
struct NormalComponentProduct{DIM, T, S <: Tuple} <: AbstractNormalTerm
    factor::T
    scales::S
end

"""
    NormalComponentSum{T}

A sum of [`NormalComponentProduct`](@ref)s, `f1 * ηₓ + f2 * ηᵧ`, held as a tuple so that
[`inner_Γ`](@ref) can expand each product into its own directional term.
"""
struct NormalComponentSum{T <: Tuple{Vararg{NormalComponentProduct}}} <: AbstractNormalTerm
    terms::T
end

@inline function _normal_product(g, ::NormalComponent{DIM}, scales::Tuple = ()) where {DIM}
    return NormalComponentProduct{DIM, typeof(g), typeof(scales)}(g, scales)
end

for G in (LazyOp, Function, Number, VectorElement)
    @eval @inline Base.:*(g::$G, c::NormalComponent) = _normal_product(g, c)
    @eval @inline Base.:*(c::NormalComponent, g::$G) = _normal_product(g, c)
end

# `(α * ηₓ) * u` and `(f * ηₓ) * u`: the unknown joins the factor, since `η[d]` commutes with
# every scalar-valued thing a surface integral can hold.
@inline function Base.:*(p::NormalComponentProduct{DIM}, u::LazyOp) where {DIM}
    return _normal_product(p.factor * u, NormalComponent{DIM}(), p.scales)
end
@inline Base.:*(u::LazyOp, p::NormalComponentProduct) = p * u
@inline Base.:*(s::NormalComponentSum, u::LazyOp) = NormalComponentSum(map(p -> p * u, s.terms))
@inline Base.:*(u::LazyOp, s::NormalComponentSum) = s * u

# Scalar multiples: a `Number` or a `Ref` to one, the scalars `OperatorScale` carries. Each is
# kept as given and applied at expansion through the same `c * term` a `LazyOp` uses, so an
# integer literal stays an `Integer` (and promotes to the space's element type) and a `Ref`
# stays a runtime scalar (bramble-form §6). `-t` below stores the `Integer` `-1`, whereas
# `A - B` on operators stores the sign `_MinusOne` (ast.jl).
const _NormalScalar = Union{Number, Base.RefValue{<:Number}}

@inline function Base.:*(c::_NormalScalar, p::NormalComponentProduct{DIM}) where {DIM}
    return _normal_product(p.factor, NormalComponent{DIM}(), (c, p.scales...))
end
@inline Base.:*(p::NormalComponentProduct, c::_NormalScalar) = c * p
@inline Base.:*(c::_NormalScalar, s::NormalComponentSum) = NormalComponentSum(map(p -> c * p, s.terms))
@inline Base.:*(s::NormalComponentSum, c::_NormalScalar) = c * s
@inline Base.:*(c::Base.RefValue{<:Number}, nc::NormalComponent) = c * _normal_product(1, nc)
@inline Base.:*(nc::NormalComponent, c::Base.RefValue{<:Number}) = c * nc
@inline Base.:-(t::AbstractNormalTerm) = -1 * _as_normal_sum(t)

# Sums. A bare component in a sum is the product `1 * η[d]`, as it is inside `inner_Γ`: an
# integer one, so it takes the space's element type. `dot(F, η)` joins a sum as its `D`
# directional products.
@inline _normal_terms_of(c::NormalComponent) = (_normal_product(1, c),)
@inline _normal_terms_of(p::NormalComponentProduct) = (p,)
@inline _normal_terms_of(s::NormalComponentSum) = s.terms
@inline function _normal_terms_of(c::NormalContraction)
    D = length(c.terms)
    return ntuple(d -> _normal_product(c.terms[d], NormalComponent{d}()), Val(D))
end
@inline _as_normal_sum(t) = NormalComponentSum(_normal_terms_of(t))

const _NormalSummand = Union{AbstractNormalTerm, NormalContraction}

@inline function Base.:+(a::_NormalSummand, b::_NormalSummand)
    return NormalComponentSum((_normal_terms_of(a)..., _normal_terms_of(b)...))
end
@inline Base.:-(a::_NormalSummand, b::_NormalSummand) = a + (-_as_normal_sum(b))

# A product of two components has no meaning here: the normal enters a surface integral
# linearly, once.
@noinline function _throw_normal_squared()
    throw(
        ArgumentError(
        "a product of two components of the normal η (such as ηₓ * ηᵧ or f * ηₓ * ηᵧ) " *
        "is not a normal flux: η may appear only once, linearly, in each term of inner_Γ",
    ),
    )
end
Base.:*(::AbstractNormalTerm, ::AbstractNormalTerm) = _throw_normal_squared()

"""
    inner_Γ(g * η[d], v; markers) -> LazyOp

The surface integral ``\\int_\\Gamma g\\, \\eta_d\\, v\\, ds``: the direction-`d` term of
`inner_Γ(dot(F, η), v)`, so summing it over `d` with `g = F[d]` gives that integral. A bare
component `η[d]` is `1 * η[d]`, and a linear combination of such products expands to the
sum of their terms, so `inner_Γ(f1 * ηₓ + f2 * ηᵧ, v)` is `inner_Γ(dot((f1, f2), η), v)`.

```julia
ηₓ, ηᵧ = η
l = form(Wₕ, v -> inner_Γ(f * ηₓ, v; markers = (:xmax,)))
a = form(Wₕ, Wₕ, (u, v) -> inner_Γ(u * ηᵧ, v; markers = (:boundary,)))
```
"""
function inner_Γ(t::AbstractNormalTerm, right::LazyOp{D}; markers = ()) where {D}
    mask = _normal_mask(Val(D), markers)
    return _expand_normal(_as_normal_sum(t), right, Val(mask), Val(D))
end

@inline function _expand_normal(s::NormalComponentSum, right, vm::Val, vd::Val)
    return sum(map(p -> _expand_normal(p, right, vm, vd), s.terms))
end

@inline function _expand_normal(
        p::NormalComponentProduct{DIM}, right, ::Val{MASK}, ::Val{D}
) where {DIM, MASK, D}
    DIM <= D || _throw_normal_component_dim(DIM, D)
    return _apply_normal_scale(p.scales, _normal_term(InnerGammaNormal{MASK, DIM}(), p.factor, right))
end

@inline _apply_normal_scale(::Tuple{}, term) = term
@inline _apply_normal_scale(s::Tuple, term) = first(s) * _apply_normal_scale(Base.tail(s), term)

@noinline function _throw_normal_component_dim(dim, D)
    count = D == 1 ? "1 component" : "$D components"
    throw(ArgumentError("η[$dim] has no face in a $(D)D form: the normal has $count here"))
end

@noinline function _throw_normal_on_test_side()
    throw(
        ArgumentError(
        "the normal η belongs on the left of inner_Γ, with the flux: write " *
        "inner_Γ(g * η[d], v; markers = ...), not inner_Γ(v, g * η[d]; ...)",
    ),
    )
end
inner_Γ(::LazyOp, ::AbstractNormalTerm; markers = ()) = _throw_normal_on_test_side()
inner_Γ(::AbstractNormalTerm, ::AbstractNormalTerm; markers = ()) = _throw_normal_squared()

# A normal component has a face to be normal to only inside `inner_Γ`; the volume inner
# products, and a sum with an ordinary form term, have none, so they refuse it by name
# rather than with a bare MethodError.
@noinline function _throw_normal_outside_gamma()
    throw(
        ArgumentError(
        "a component of the normal η is only defined on a boundary face: use it inside " *
        "inner_Γ(g * η[d], v; markers = ...), not in innerₕ or inner₊, and not added to " *
        "an ordinary form term",
    ),
    )
end

for f in (:innerₕ, :inner₊)
    @eval $f(::AbstractNormalTerm, ::LazyOp; markers = ()) = _throw_normal_outside_gamma()
    @eval $f(::LazyOp, ::AbstractNormalTerm; markers = ()) = _throw_normal_outside_gamma()
    @eval $f(::AbstractNormalTerm, ::AbstractNormalTerm; markers = ()) = _throw_normal_outside_gamma()
end
for op in (:+, :-)
    @eval Base.$op(::AbstractNormalTerm, ::LazyOp) = _throw_normal_outside_gamma()
    @eval Base.$op(::LazyOp, ::AbstractNormalTerm) = _throw_normal_outside_gamma()
end

# --- Facet operators: the normal derivative ------------------------------------------- #
#
# `∂ₙ(u)` is `η_F Dc_d(u)` on each facet `F` with normal axis `d`: a value per facet, not per
# node, so a corner node has one for each facet it lies on. It is not a `LazyOp` for that
# reason. It holds one `LazyOp` per (axis, side), already carrying its outward sign, and
# `inner_Γ` expands it into one product per facet, each weighted by that facet's measure
# alone (`InnerGammaFacet`), as it expands `dot(F, η)`; with a facet operator on both sides
# the products pair facet with facet. Nothing downstream of `form` sees a new node. Other
# facet operators are other (axis, side) tuples of `LazyOp`s.

"""
    FacetOperator{D, G, T}

A quantity with one value per boundary facet: `ops[d][s]` is the `LazyOp` it is on the facet
of normal axis `d`, side `s` (1 the minimum, 2 the maximum), outward sign included. `G[d][e]`
says whether the operator of a facet of axis `d` reads the ghosts along axis `e`: beyond its
own facet for `e == d`, beyond both `e` facets at its corners otherwise. `G` depends on
types only. A conormal `K` reaches every axis whatever its values, and [`inner_Γ`](@ref),
at `form`, refuses the operator where those ghosts are missing, except along an axis
`e ≠ d` that each conormal term reads through a zero `K[d, e]`. Built by
[`∂ₙ`](@ref), [`∂ₙₙ`](@ref) and [`jumpₙ`](@ref), and meaningful only inside `inner_Γ`; not a
`LazyOp`, since a node of a corner has a value for each facet it lies on.
"""
struct FacetOperator{D, G, T <: Tuple}
    ops::T
end

@inline function _facet_operator(::Val{G}, ops::NTuple{D, Tuple{Any, Any}}) where {G, D}
    return FacetOperator{D, G, typeof(ops)}(ops)
end

# The value `x` of a `Val{x}`.
@inline _val(::Val{x}) where {x} = x

# The operator of facet `(d, s)`, `s` 1 for the minimum and 2 for the maximum.
@inline _facet_component(F::FacetOperator, ::Val{d}, ::Val{s}) where {d, s} = F.ops[d][s]

@inline _reads_ghosts(::FacetOperator{D, G}) where {D, G} = G

# The axes each facet reads ghosts along, merged: what a sum or a pair of operators reads.
@inline _merge_reach(G, H) = map((g, h) -> map(|, g, h), G, H)

@inline function _map_facets(f::F, op::FacetOperator{D, G}) where {F, D, G}
    return _facet_operator(Val(G), map(p -> (f(p[1]), f(p[2])), op.ops))
end

"""
    ∂ₙ(u; K = nothing)

The outward normal derivative of `u` at the boundary facets, `η_F Dc_d(u)` on the facet `F`
of normal axis `d`, with `Dc_d` the centred difference across the facet. It reads the ghost
beyond the facet, so the space must have ghosts there: build it with
`gridspace(Ωₕ; ghosts = ...)`.

A node on several facets (a 2D corner) has one value per facet. Meaningful inside
[`inner_Γ`](@ref), on either side or both, where each facet's term carries that facet's
weight and `markers` selects the facets (required; `:boundary` is all of them); so
`inner_Γ(∂ₙ(u), v; markers = (:xmin,))` gives a corner only its `xmin` term, and per-facet
coefficients are sums over `markers`. With `∂ₙ` on both sides, facet pairs with facet.

With a constant `D × D` matrix `K`, it is the conormal derivative `η_F · (K Dc u)`, the
centred difference along every axis. A `K` that varies in space is refused with an
`ArgumentError` when `∂ₙ` is called; a facet without ghosts, or a space without any, when
`form` is. So is a nonzero `K[d, e]`, `e ≠ d`, on a facet of axis `d` whose corners lack
the ghosts of the two `e` facets: the tangential difference there reads them.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9), (true, true))
Wₕ = gridspace(Ωₕ; ghosts = :boundary)
a = form(Wₕ, Wₕ, (u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmin, :ymax)))
K = [2.0 0.5; 0.5 1.0]
b = form(Wₕ, Wₕ, (u, v) -> inner_Γ(∂ₙ(u; K), v; markers = :boundary))
```

See also: [`inner_Γ`](@ref), [`η`](@ref)
"""
function ∂ₙ(u::LazyOp{D}; K = nothing) where {D}
    _check_conormal(K, Val(D))
    ops = ntuple(d -> _outward(_normal_flux(u, K, d)), Val(D))
    return _facet_operator(Val(_normal_reach(K, Val(D))), ops)
end

# A facet of axis `d` reads along `d` and, with a matrix `K`, along every axis: a type of
# `K` alone, so that `K`'s values do not reach the type. An axis whose coefficient `K[d, e]`
# is zero is pruned where the check runs (`_check_facet_reach`), on the operator's values.
@inline function _normal_reach(::Nothing, ::Val{D}) where {D}
    return ntuple(d -> ntuple(==(d), Val(D)), Val(D))
end
@inline function _normal_reach(::AbstractMatrix, ::Val{D}) where {D}
    return ntuple(_ -> ntuple(_ -> true, Val(D)), Val(D))
end

@inline _outward(op::LazyOp) = (-op, op)

@inline _normal_flux(u::LazyOp, ::Nothing, d::Int) = Dc(u, Val(d))
# The axes as `Val`s, so each term's difference has its axis as a type, summed by `+`,
# which folds over terms of distinct types where `sum` would not infer.
@inline function _normal_flux(u::LazyOp{D}, K::AbstractMatrix{<:Number}, d::Int) where {D}
    return _tuple_sum(map(e -> K[d, _val(e)] * Dc(u, e), ntuple(Val, Val(D))))
end
@inline _tuple_sum(t::Tuple{Any}) = only(t)
@inline _tuple_sum(t::Tuple) = +(t...)

@inline _check_conormal(::Nothing, ::Val{D}) where {D} = nothing
@inline function _check_conormal(K::AbstractMatrix{<:Number}, ::Val{D}) where {D}
    size(K) == (D, D) || _throw_conormal_tensor(K, D)
    return nothing
end
@noinline _check_conormal(K, ::Val{D}) where {D} = _throw_conormal_tensor(K, D)

@noinline function _throw_conormal_tensor(K, D::Int)
    throw(
        ArgumentError(
        "∂ₙ(u; K) takes a constant $D×$D matrix of numbers, got $(summary(K)). A tensor " *
        "varying in space is not supported: write the conormal flux out per facet.",
    ),
    )
end

"""
    ∂ₙₙ(u)

The second normal derivative of `u` at the boundary facets: on the facet `F` of normal axis
`d`, the second difference `D̃_d D₋_d u` along `d` alone (the per-axis term of
[`Δₕ`](@ref)), the same on both sides since `η_F² = 1`. It reads the ghost beyond the facet,
whose spacing is the first interior one, so the space must have ghosts there: build it with
`gridspace(Ωₕ; ghosts = ...)`.

A node on several facets (a 2D corner) has one value per facet, each reading only the
ghost of its own facet. Meaningful inside [`inner_Γ`](@ref), on either side or both, with
`markers` selecting the facets as for [`∂ₙ`](@ref); a facet without ghosts, or a space
without any, is refused with an `ArgumentError` when `form` is called.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9), (true, true))
Wₕ = gridspace(Ωₕ; ghosts = :boundary)
a = form(Wₕ, Wₕ, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = (:xmin,)))
```

See also: [`∂ₙ`](@ref), [`jumpₙ`](@ref), [`inner_Γ`](@ref)
"""
function ∂ₙₙ(u::LazyOp{D}) where {D}
    ops = ntuple(d -> (s = _second_difference(u, Val(d)); (s, s)), Val(D))
    return _facet_operator(Val(_normal_reach(nothing, Val(D))), ops)
end

"""
    jumpₙ(w)

The inward jump of `w` at the boundary facets, `w_{I − η_F e_d} − w_I` at the node `I` of
the facet `F` of normal axis `d`: the forward jump `w_{I + e_d} − w_I` on a minimum facet,
the backward one `w_{I − e_d} − w_I` on a maximum facet. Both read only the facet node and
its interior neighbour, so `jumpₙ` needs no ghosts of its own and works on any space. Its
operand is evaluated as anywhere else: `jumpₙ(Δₕ(u))` reads the ghosts of a ghost space, and
on a plain space `Δₕ` is truncated at the boundary.

A node on several facets (a 2D corner) has one value per facet. Meaningful inside
[`inner_Γ`](@ref), on either side or both, with `markers` selecting the facets as for
[`∂ₙ`](@ref), so per-facet coefficients are sums over `markers`.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9), (true, true))
Wₕ = gridspace(Ωₕ; ghosts = :boundary)
a = form(Wₕ, Wₕ, (u, v) -> inner_Γ(2.0 * jumpₙ(u), jumpₙ(v); markers = (:xmin,)))
b = form(Wₕ, Wₕ, (u, v) -> inner_Γ(jumpₙ(Δₕ(u)), v; markers = :boundary))
```

See also: [`∂ₙ`](@ref), [`∂ₙₙ`](@ref), [`inner_Γ`](@ref)
"""
function jumpₙ(w::LazyOp{D}) where {D}
    ops = ntuple(d -> (jump(w, Val(d)), backward_shift(w, Val(d)) - w), Val(D))
    return _facet_operator(Val(ntuple(_ -> ntuple(_ -> false, Val(D)), Val(D))), ops)
end

jumpₙ(::FacetOperator) = _throw_jump_of_facet_operator()

@noinline function _throw_jump_of_facet_operator()
    throw(
        ArgumentError(
        "jumpₙ takes a field expression such as u or Δₕ(u), not a facet operator: " *
        "a facet operator already has one value per facet, so jumpₙ(∂ₙ(u)) has no meaning",
    ),
    )
end

# Scalars and sums, facet by facet: `α * u + β * ∂ₙ(u)` is `α u + β η_F Dc_d(u)` on each
# facet, and `c * ∂ₙ(u)` scales every facet's operator.
@inline Base.:*(c::_NormalScalar, F::FacetOperator) = _map_facets(op -> c * op, F)
@inline Base.:*(F::FacetOperator, c::_NormalScalar) = c * F
@inline Base.:/(F::FacetOperator, c::Number) = _map_facets(op -> op / c, F)
@inline Base.:/(::FacetOperator, ::Base.RefValue{<:Number}) = _throw_facet_ref_divisor()

# A `Ref` scale is read at each assembly, and there is no scale that divides by one, so the
# quotient is refused: the reciprocal taken here would freeze the `Ref`'s value at build.
@noinline function _throw_facet_ref_divisor()
    throw(
        ArgumentError(
        "a facet operator cannot be divided by a Ref, which would freeze its value at " *
        "form build: multiply by a Ref holding the reciprocal, `Ref(1 / c) * ∂ₙ(u)`",
    ),
    )
end
@inline Base.:-(F::FacetOperator) = _map_facets(-, F)

@inline function Base.:+(F::FacetOperator{D, G1}, H::FacetOperator{D, G2}) where {D, G1, G2}
    ops = ntuple(d -> (F.ops[d][1] + H.ops[d][1], F.ops[d][2] + H.ops[d][2]), Val(D))
    return _facet_operator(Val(_merge_reach(G1, G2)), ops)
end
@inline Base.:-(F::FacetOperator, H::FacetOperator) = F + (-H)
@inline Base.:+(F::FacetOperator{D}, u::LazyOp{D}) where {D} = _map_facets(op -> op + u, F)
@inline Base.:+(u::LazyOp{D}, F::FacetOperator{D}) where {D} = _map_facets(op -> u + op, F)
@inline Base.:-(F::FacetOperator{D}, u::LazyOp{D}) where {D} = _map_facets(op -> op - u, F)
@inline Base.:-(u::LazyOp{D}, F::FacetOperator{D}) where {D} = _map_facets(op -> u - op, F)

"""
    InnerGammaFacet{M, DIM, SIDE, G} <: AbstractInnerProduct

The quadrature weight of one facet: the transverse measure `∏_{e ≠ DIM} ĥ_e` at the nodes of
the facet of normal axis `DIM`, side `SIDE` (1 the minimum, 2 the maximum), when `mask`
selects it, and zero anywhere else. Unsigned, since a `FacetOperator` carries its
outward sign. `G[e]` says whether the term may read the ghosts along axis `e` (the facet's
own for `e == DIM`, both `e` facets' at its corners otherwise): the facet operator's
structural reach, which no weight reads. `form` checks it on the product's operands,
skipping an axis `e ≠ DIM` that each conormal term there reads through a zero `K[DIM, e]`.
`mask` holds the labels, then the face mask, as `InnerGamma`'s does.
"""
struct InnerGammaFacet{M, DIM, SIDE, G} <: AbstractInnerProduct
    mask::M
end

# Built from the labels type or face mask `M`, as `InnerGamma{M}()` is (operators/inner.jl).
@inline function (::Type{InnerGammaFacet{M, DIM, SIDE, G}})() where {M, DIM, SIDE, G}
    m = _mask_instance(M)
    return InnerGammaFacet{typeof(m), DIM, SIDE, G}(m)
end

@inline function compute_weight(
        w::InnerGammaFacet{M, DIM, SIDE}, space, I::CartesianIndex{D}, lin_idx::Int
) where {M, DIM, SIDE, D}
    Ωₕ = mesh(space)
    w.mask[DIM][SIDE] && _on_facet(Ωₕ, I, DIM, SIDE) &&
        return _facet_measure(Ωₕ, I, Val(DIM))
    return zero(eltype(Ωₕ))
end

function _resolved_inner(::InnerGammaFacet{_SurfaceLabels{L}, DIM, SIDE, G}, Ωₕ,
        product) where {L, DIM, SIDE, G}
    mask = _surface_mask(Ωₕ, L)
    operands = (product.left_op, product.right_op)
    mask[DIM][SIDE] && _check_facet_reach(Ωₕ, G, DIM, SIDE, operands)
    return InnerGammaFacet{typeof(mask), DIM, SIDE, G}(mask)
end
function _surface_labels(
        ::Type{InnerGammaFacet{_SurfaceLabels{L}, DIM, SIDE, G}}) where {L, DIM, SIDE, G}
    return L
end

# The facet's own ghosts first, then, for each other axis `e` the term reads along, the
# ghosts of both `e` facets, which the tangential difference reads at the facet's corners.
# An axis `e` the reach `G` names is read only where a conormal block of an operand has a
# nonzero `K[DIM, e]` (`_conormal_reads`), so a `K` with `K[DIM, e] == 0` ghosts only its
# own facet. Any other read past a corner is the absent-read check's, at `assemble`.
function _check_facet_reach(Ωₕ, G::Tuple, DIM::Int, SIDE::Int, operands::Tuple)
    G[DIM] && _check_facet_ghosts(Ωₕ, DIM, SIDE)
    for e in eachindex(G)
        e != DIM && G[e] && any(o -> _conormal_reads(o, e), operands) || continue
        for s in 1:2
            _check_transverse_ghosts(Ωₕ, DIM, SIDE, e, s)
        end
    end
    return nothing
end

# Whether some conormal block of `op` has a nonzero coefficient along axis `e`. A block is
# the sum `∂ₙ(w; K)` builds on a facet of axis `d`, `K[d, 1] * Dc_1(w) + … + K[d, D] *
# Dc_D(w)`, its terms in axis order; the walk finds it under sums and scales, whatever their
# values, so `0.0 * ∂ₙ(u; K)` and `∂ₙ(u; K) - ∂ₙ(u; K)` still read through `K`. Other
# terms, and the operands of differences, are not blocks: their reads are not `K`'s.
_conormal_reads(::Any, ::Int) = false
_conormal_reads(op::OperatorScale, e::Int) = _conormal_reads(op.inner_op, e)
_conormal_reads(op::GridFunctionScale, e::Int) = _conormal_reads(op.inner_op, e)
function _conormal_reads(op::OperatorAdd, e::Int)
    k = _conormal_coefficients(op)
    k === nothing || return !iszero(k[e])
    return _conormal_reads(op.left_op, e) || _conormal_reads(op.right_op, e)
end

# The coefficients `K[d, :]` of a conormal block, or `nothing` for any other sum.
const _ConormalTerm{D, e} = OperatorScale{D, <:Number, <:CenteredDifference{D, e}}
_conormal_coefficients(::OperatorAdd) = nothing
function _conormal_coefficients(op::OperatorAdd{2, <:_ConormalTerm{2, 1},
        <:_ConormalTerm{2, 2}})
    return (op.left_op.scalar, op.right_op.scalar)
end
function _conormal_coefficients(op::OperatorAdd{3,
        <:OperatorAdd{3, <:_ConormalTerm{3, 1}, <:_ConormalTerm{3, 2}},
        <:_ConormalTerm{3, 3}})
    return (op.left_op.left_op.scalar, op.left_op.right_op.scalar, op.right_op.scalar)
end

@inline function _check_transverse_ghosts(G::_GhostMeshLike, d::Int, s::Int, e::Int, t::Int)
    has_ghosts(G, e, _GHOST_SIDES[t]) || _throw_conormal_unghosted(d, s, e, t)
    return nothing
end
@inline function _check_transverse_ghosts(Ωₕ, d::Int, s::Int, ::Int, ::Int)
    return _check_facet_ghosts(Ωₕ, d, s)
end

@inline _facet_name(d::Int, s::Int) = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)[2d - 2 + s]

@noinline function _throw_conormal_unghosted(d::Int, s::Int, e::Int, t::Int)
    facet, other = _facet_name(d, s), _facet_name(e, t)
    throw(
        ArgumentError(
        "∂ₙ(u; K) on :$facet reads, at its corner on :$other, the ghost beyond :$other " *
        "through K[$d, $e] ≠ 0, and :$other carries no ghosts on this space. Ghost :$other " *
        "too (or use `ghosts = :boundary`), or leave :$facet out of `markers`.",
    ),
    )
end

@inline function _check_facet_ghosts(G::_GhostMeshLike, d::Int, s::Int)
    has_ghosts(G, d, _GHOST_SIDES[s]) || _throw_facet_operator_unghosted(d, s, true)
    return nothing
end
@inline _check_facet_ghosts(Ωₕ, d::Int, s::Int) = _throw_facet_operator_unghosted(d, s, false)

@noinline function _throw_facet_operator_unghosted(d::Int, s::Int, ghost_space::Bool)
    facet = _facet_name(d, s)
    where_ = ghost_space ? "the facet :$facet carries no ghosts on this space" :
             "the space has no ghosts"
    throw(
        ArgumentError(
        "∂ₙ and ∂ₙₙ read the ghost beyond each facet they are integrated on, and $where_. " *
        "Ghost the facet, with `gridspace(Ωₕ; ghosts = ...)` naming :$facet (or " *
        "`ghosts = :boundary`), or leave :$facet out of `markers`.",
    ),
    )
end

"""
    inner_Γ(F::FacetOperator, v; markers) -> LazyOp
    inner_Γ(u, F::FacetOperator; markers) -> LazyOp

The surface integral of a facet operator such as [`∂ₙ`](@ref)`(u)`, facet by facet: one
product per facet `markers` selects, weighted by that facet's measure, so a corner node gets
a term from each selected facet it lies on. With a facet operator on both sides, the
products pair the two on the same facet. `markers` is required, as for the plain
`inner_Γ`: an empty selection is an `ArgumentError`, and `markers = :boundary` selects
every facet.
"""
function inner_Γ(F::FacetOperator{D}, right::LazyOp{D}; markers = ()) where {D}
    return _facet_inner(_normal_mask(Val(D), markers), F, right)
end

function inner_Γ(left::LazyOp{D}, F::FacetOperator{D}; markers = ()) where {D}
    return _facet_inner(_normal_mask(Val(D), markers), left, F)
end

function inner_Γ(F::FacetOperator{D}, H::FacetOperator{D}; markers = ()) where {D}
    return _facet_inner(_normal_mask(Val(D), markers), F, H)
end

for S in (Function, Number, VectorElement)
    @eval function inner_Γ(g::$S, F::FacetOperator{D}; markers = ()) where {D}
        return _facet_inner(_normal_mask(Val(D), markers), g, F)
    end
end

# `inner_Γ` with the labels `MASK` of `markers` as a type, which `form` resolves to a mask
# by reading the mesh (`_resolved_inner`); or, as a `Val`, with the mask already resolved,
# which `form` keeps, as `constrained` passes the face it names (`_relation_mask`). Either
# is a constant, so the term infers where the keyword need not.
@inline function _facet_inner(::Type{MASK}, left, right) where {MASK}
    return _facet_inner(Val(MASK), left, right)
end
@inline function _facet_inner(m::Val, F::FacetOperator{D}, right::LazyOp{D}) where {D}
    return _facet_terms(m, Val(_reads_ghosts(F)), Val(D),
        (d, s) -> _facet_component(F, d, s), (d, s) -> right)
end
@inline function _facet_inner(m::Val, left::LazyOp{D}, F::FacetOperator{D}) where {D}
    return _facet_terms(m, Val(_reads_ghosts(F)), Val(D), (d, s) -> left,
        (d, s) -> _facet_component(F, d, s))
end
@inline function _facet_inner(m::Val, F::FacetOperator{D}, H::FacetOperator{D}) where {D}
    G = _merge_reach(_reads_ghosts(F), _reads_ghosts(H))
    return _facet_terms(m, Val(G), Val(D), (d, s) -> _facet_component(F, d, s),
        (d, s) -> _facet_component(H, d, s))
end
@inline function _facet_inner(m::Val, g::Union{Function, Number, VectorElement},
        F::FacetOperator{D}) where {D}
    return _facet_terms(m, Val(_reads_ghosts(F)), Val(D), (d, s) -> g,
        (d, s) -> _facet_component(F, d, s))
end

# One term per facet, over the axes as `Val`s so each weight is a type.
@inline function _facet_terms(m::Val, g::Val, ::Val{D}, left::L, right::R) where {D, L, R}
    return _tuple_sum(_facet_pairs(m, g, ntuple(Val, Val(D)), left, right))
end

# Unrolled without a closure, which would hold `MASK` as a value of an abstract type.
@inline _facet_pairs(m::Val, g::Val, ::Tuple{}, left, right) = ()
@inline function _facet_pairs(m::Val, g::Val, axes::Tuple, left::L, right::R) where {L, R}
    return (_facet_pair(m, g, first(axes), left, right),
        _facet_pairs(m, g, Base.tail(axes), left, right)...)
end

# The two facets of axis `d`, which `left` and `right` take as `Val`s, so each operand and
# weight has its facet as a type.
@inline function _facet_pair(::Val{MASK}, ::Val{G}, a::Val{d}, left::L,
        right::R) where {MASK, G, d, L, R}
    lo, hi = Val(1), Val(2)
    return _normal_term(InnerGammaFacet{MASK, d, 1, G[d]}(), left(a, lo), right(a, lo)) +
           _normal_term(InnerGammaFacet{MASK, d, 2, G[d]}(), left(a, hi), right(a, hi))
end

# A facet operator has a facet only inside `inner_Γ`.
@noinline function _throw_facet_operator_outside_gamma()
    throw(
        ArgumentError(
        "a facet operator such as ∂ₙ(u) has one value per boundary facet: use it inside " *
        "inner_Γ(∂ₙ(u), v; markers = ...), not in innerₕ or inner₊",
    ),
    )
end

for f in (:innerₕ, :inner₊)
    @eval $f(::FacetOperator, ::LazyOp; markers = ()) = _throw_facet_operator_outside_gamma()
    @eval $f(::LazyOp, ::FacetOperator; markers = ()) = _throw_facet_operator_outside_gamma()
    @eval $f(::FacetOperator, ::FacetOperator; markers = ()) = _throw_facet_operator_outside_gamma()
end

# --- Expression rendering (gpena/Bramble.jl#274) ----------------------------------- #

# `InnerGammaNormal`'s case of `_inner_name` (operators/inner.jl): `inner_Γ` plus the
# direction subscript, as `InnerPlus{Dim}` gets.
_inner_name(::InnerGammaNormal{MASK, DIM}) where {MASK, DIM} = "inner_Γ" * _BRAMBLE_var2symbol[DIM]

# `InnerGammaFacet`'s: `inner_Γ` plus the facet's name.
function _inner_name(::InnerGammaFacet{MASK, DIM, SIDE}) where {MASK, DIM, SIDE}
    return "inner_Γ" * _BRAMBLE_var2symbol[DIM] * (SIDE == 1 ? "⁻" : "⁺")
end
