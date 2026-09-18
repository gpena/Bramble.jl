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
