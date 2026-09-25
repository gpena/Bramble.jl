# normal.jl
# The outward normal as a symbol inside a form, and the normal derivative built on it
# (gpena/Bramble.jl#213).
#
# Why this is not a node.
#
# A `LazyOp` is scalar-valued and its `local_stencil` returns one coefficient per point per
# column, so `n` cannot be one: it is vector-valued, and what a form does with it is always
# to contract it against another vector-valued quantity. More decisively, the *weight* of a
# normal term is directional. A surface integral of a normal flux is
#
#     ∫_Γ (F · n) v ds = Σ_{faces F_d} ± ∫_{F_d} F_d v ds
#
# and the measure of the face with normal `d` is the transverse product `∏_{e ≠ d} ĥ_e`, not
# the lumped `ω` that `InnerGamma` carries. At a corner the two differ: `ω` sums the incident
# faces' measures into one number, which is right for `∫_Γ g v` and wrong for `∫_Γ (F·n) v`,
# where each face contributes its own component of `F` with its own sign.
#
# So `dot(F, n)` records the tuple, and `inner_Γ` expands it, at the builder, into one
# ordinary product per direction carrying `InnerGammaNormal{MASK, d}` -- the signed
# transverse measure of the faces with normal `d`. No new node type, no new stencil path, and
# the resulting AST is the sum a caller would have written by hand.

# The type behind `n`. A singleton, so `dot(F, n)` folds at compile time and the symbol costs
# nothing to carry through a form.
struct NormalSymbol end

"""
    n

The outward unit normal, as a symbol usable inside a form.

Only meaningful contracted against a vector-valued quantity whose values are known -- a flux
field, `dot(F, n)` -- and only inside [`inner_Γ`](@ref), which is where a normal has a face to
be normal to.

```julia
l = form(Wₕ, v -> inner_Γ(dot(Fₕ, n), v; markers = (:xmax,)))
```

The normal *derivative* of an unknown is not built on this yet; see the note below the
`NormalContraction` definition for why it waits for ghost points.

See also: [`inner_Γ`](@ref), [`normal_vector`](@ref)
"""
const n = NormalSymbol()

"""
    NormalContraction{D, T}

`dot(F, n)`: the tuple `F`, held until [`inner_Γ`](@ref) expands it into one product per
direction. Not a `LazyOp`, deliberately -- see the note at the head of this file.
"""
struct NormalContraction{D, T <: Tuple}
    terms::T
end

@inline dot(F::NTuple{D, Any}, ::NormalSymbol) where {D} = NormalContraction{D, typeof(F)}(F)
@inline dot(::NormalSymbol, F::NTuple{D, Any}) where {D} = dot(F, n)

# The normal *derivative* is not here yet, deliberately.
#
# `∂ₙ(u) = ∇u · n` needs a gradient that still has a stencil on a boundary face. `∇ₕ` has
# none there (it is the backward one, truncated on the first slice of each axis) and `Dcₕ`,
# the centered one this should be built on, is truncated to zero on both end slices. What
# makes a centered difference meaningful at a boundary point is a ghost point beyond it, and
# Bramble has no ghost meshes yet -- they are milestone v4.0.0's subject, and the normal
# derivative belongs with the Neumann and Robin assembly that milestone builds
# (gpena/Bramble.jl#27, #30). Defining `∂ₙ` on a one-sided stand-in now would fix the wrong
# convention in the API before the right one exists.
#
# `dot(F, n)` below is a different question and is well defined today: `F` is a field whose
# values are already known at every point, so nothing has to be differenced at the face.

"""
    InnerGammaNormal{MASK, DIM} <: AbstractInnerProduct

The quadrature weight of the faces whose normal points along `DIM`, signed by which side of
the axis the point sits on: `-∏_{e ≠ DIM} ĥ_e` on the `min` face, `+` on the `max` one, and
zero anywhere else.

The directional counterpart of [`InnerGamma`](@ref), which lumps every incident face into one
number. Both are resolved once, when the form is built, and carried as type parameters.
"""
struct InnerGammaNormal{MASK, DIM} <: AbstractInnerProduct end

@inline function compute_weight(
        ::InnerGammaNormal{MASK, DIM}, space, I::CartesianIndex{D}, lin_idx::Int
) where {MASK, DIM, D}
    Ωₕ = mesh(space)
    np = npoints(Ωₕ, Tuple)
    m = _transverse_measure(Ωₕ, I, Val(DIM), Val(D))
    MASK[DIM][1] && I[DIM] == 1 && return -m
    MASK[DIM][2] && I[DIM] == np[DIM] && return m
    return zero(m)
end

"""
    inner_Γ(F::NormalContraction, v; markers) -> LazyOp

The surface integral of a normal flux, ``\\int_\\Gamma (\\mathbf{F} \\cdot \\mathbf{n})\\, v\\, ds``.

Expanded here, at the builder, into one ordinary product per direction, each carrying that
direction's signed transverse measure. The term is the sum a caller would otherwise write out,
so nothing downstream of `form` learns a new node.
"""
function inner_Γ(F::NormalContraction{D}, right::LazyOp{D}; markers = ()) where {D}
    mask = _normal_mask(Val(D), markers)
    return _normal_terms(F.terms, right, Val(mask), Val(D))
end

@inline function _normal_mask(::Val{D}, markers) where {D}
    mask = _face_mask(Val(D), _as_labels(markers))
    _no_faces(mask) && _throw_no_surface_labels()
    return mask
end

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
# `nx, ny = n` names one direction of the normal at a time. A component is a singleton like
# `n` itself, and a product with it is held, not built, until `inner_Γ` expands it into the
# single term `dot(F, n)` would have produced for that direction: the same `_normal_term`
# with the same `InnerGammaNormal{MASK, d}` weight. Sums and scalar multiples of such
# products are held too, as a tuple of products each carrying its own scale, so
# `f1 * nx + f2 * ny`, integrated face by face, is `dot((f1, f2), n)` term for term, and
# nothing downstream sees a new node.

# What a component, a product with one, or a sum of such products has in common: each is
# meaningful only inside `inner_Γ`.
abstract type AbstractNormalTerm end

"""
    NormalComponent{DIM}

`n[DIM]`: the component of the outward normal along axis `DIM`, as a symbol usable inside a
form. Obtained by indexing or destructuring [`n`](@ref), never constructed directly.
"""
struct NormalComponent{DIM} <: AbstractNormalTerm end

# Three components, like the operator aliases, whatever the dimension: the one a form's
# dimension leaves over is refused inside `inner_Γ`, where the dimension is known.
@inline Base.length(::NormalSymbol) = 3
@inline Base.firstindex(::NormalSymbol) = 1
@inline Base.lastindex(::NormalSymbol) = 3
@inline Base.getindex(::NormalSymbol, d::Integer) = _normal_component(Val(Int(d)))
@inline Base.getindex(::NormalSymbol, s::Symbol) = _normal_component(Val(_normal_axis(s)))
@inline Base.iterate(::NormalSymbol, state::Int = 1) = state > 3 ? nothing : (n[state], state + 1)

@inline function _normal_component(::Val{d}) where {d}
    1 <= d <= 3 || throw(BoundsError(n, d))
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

`c * (g * n[DIM])`: the factor `g` and the scalar `c`, held until [`inner_Γ`](@ref) gives
the factor the signed face selection of direction `DIM` and scales the term by `c`. The
scale is `nothing` when there is none, so an unscaled product lowers to the bare term. Not a
`LazyOp`, for the reason `n` is not one.
"""
struct NormalComponentProduct{DIM, T, S} <: AbstractNormalTerm
    factor::T
    scale::S
end

"""
    NormalComponentSum{T}

A sum of [`NormalComponentProduct`](@ref)s, `f1 * nx + f2 * ny`, held as a tuple so that
[`inner_Γ`](@ref) can expand each product into its own directional term.
"""
struct NormalComponentSum{T <: Tuple{Vararg{NormalComponentProduct}}} <: AbstractNormalTerm
    terms::T
end

@inline function _normal_product(g, ::NormalComponent{DIM}, scale = nothing) where {DIM}
    return NormalComponentProduct{DIM, typeof(g), typeof(scale)}(g, scale)
end

for G in (LazyOp, Function, Number, VectorElement)
    @eval @inline Base.:*(g::$G, c::NormalComponent) = _normal_product(g, c)
    @eval @inline Base.:*(c::NormalComponent, g::$G) = _normal_product(g, c)
end

# Scalar multiples. An integer scale is stored as a float: once it is a field, its value is
# a runtime one, and a runtime `Int` must not reach `_wrap_scale` (bramble-form §6).
@inline _normal_scale(c::Integer) = float(c)
@inline _normal_scale(c::Number) = c
@inline _combine_scale(c, ::Nothing) = c
@inline _combine_scale(c, s) = c * s

@inline function Base.:*(c::Number, p::NormalComponentProduct{DIM}) where {DIM}
    return _normal_product(p.factor, NormalComponent{DIM}(), _combine_scale(_normal_scale(c), p.scale))
end
@inline Base.:*(p::NormalComponentProduct, c::Number) = c * p
@inline Base.:*(c::Number, s::NormalComponentSum) = NormalComponentSum(map(p -> c * p, s.terms))
@inline Base.:*(s::NormalComponentSum, c::Number) = c * s
@inline Base.:-(t::AbstractNormalTerm) = -1.0 * _as_normal_sum(t)

# Sums. A bare component in a sum is the product `1.0 * n[d]`, as it is inside `inner_Γ`.
@inline _normal_terms_of(c::NormalComponent) = (_normal_product(1.0, c),)
@inline _normal_terms_of(p::NormalComponentProduct) = (p,)
@inline _normal_terms_of(s::NormalComponentSum) = s.terms
@inline _as_normal_sum(t::AbstractNormalTerm) = NormalComponentSum(_normal_terms_of(t))

@inline function Base.:+(a::AbstractNormalTerm, b::AbstractNormalTerm)
    return NormalComponentSum((_normal_terms_of(a)..., _normal_terms_of(b)...))
end
@inline Base.:-(a::AbstractNormalTerm, b::AbstractNormalTerm) = a + (-b)

# A product of two components has no meaning here: the normal enters a surface integral
# linearly, once.
@noinline function _throw_normal_squared()
    throw(
        ArgumentError(
        "a product of two components of the normal n (such as nx * ny or f * nx * ny) " *
        "is not a normal flux: n may appear only once, linearly, in each term of inner_Γ",
    ),
    )
end
Base.:*(::AbstractNormalTerm, ::AbstractNormalTerm) = _throw_normal_squared()

"""
    inner_Γ(g * n[d], v; markers) -> LazyOp

The surface integral ``\\int_\\Gamma g\\, n_d\\, v\\, ds``: the direction-`d` term of
`inner_Γ(dot(F, n), v)`, so summing it over `d` with `g = F[d]` gives that integral. A bare
component `n[d]` is `1.0 * n[d]`, and a linear combination of such products expands to the
sum of their terms, so `inner_Γ(f1 * nx + f2 * ny, v)` is `inner_Γ(dot((f1, f2), n), v)`.

```julia
nx, ny = n
l = form(Wₕ, v -> inner_Γ(f * nx, v; markers = (:xmax,)))
a = form(Wₕ, Wₕ, (u, v) -> inner_Γ(u * ny, v; markers = (:boundary,)))
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
    return _apply_normal_scale(p.scale, _normal_term(InnerGammaNormal{MASK, DIM}(), p.factor, right))
end

@inline _apply_normal_scale(::Nothing, term) = term
@inline _apply_normal_scale(c, term) = c * term

@noinline function _throw_normal_component_dim(dim, D)
    count = D == 1 ? "1 component" : "$D components"
    throw(ArgumentError("n[$dim] has no face in a $(D)D form: the normal has $count here"))
end

@noinline function _throw_normal_on_test_side()
    throw(
        ArgumentError(
        "the normal n belongs on the left of inner_Γ, with the flux: write " *
        "inner_Γ(g * n[d], v; markers = ...), not inner_Γ(v, g * n[d]; ...)",
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
        "a component of the normal n is only defined on a boundary face: use it inside " *
        "inner_Γ(g * n[d], v; markers = ...), not in innerₕ or inner₊, and not added to " *
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

# --- Expression rendering (gpena/Bramble.jl#274) ----------------------------------- #

# `_inner_name` (src/form/operators/inner.jl, S1.3) gets its `InnerGammaNormal` case here --
# no forward declaration needed, only consistent naming. Named after `InnerGamma`'s own
# `"inner_Γ"` (both are surface integrals) plus the directional subscript
# `InnerPlus{Dim}`/`_inner_name` already appends for its own directional weight, since this
# type's docstring calls itself "the directional counterpart of InnerGamma": the subscript is
# what distinguishes a term carrying one signed transverse measure from the lumped one.
_inner_name(::InnerGammaNormal{MASK, DIM}) where {MASK, DIM} = "inner_Γ" * _BRAMBLE_var2symbol[DIM]
