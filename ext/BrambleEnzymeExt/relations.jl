# A reverse rule for `Bramble._assemble_relation`, the forms of one relation of `constrained`
# (#1153).
#
# Enzyme's own reverse compiles one copy of the forms per facet and does not finish in
# practice; forward mode compiles the function once. The rule is the function's own call;
# its pullback takes one forward derivative of the function per float of the relation `r`
# (its datum, its Robin coefficient: the numbers `r` holds), the tangent a copy of `r` with
# that float one and every other zero, and contracts it with the cotangents of the matrix
# and the load. The relation's floats are the only inputs: the operators of a relation hold
# no data, and the space is not differentiated.

@noinline function _throw_mesh_parameter()
    throw(
        ArgumentError(
        "reverse mode through `constrained` cannot differentiate the space of a relation, " *
        "which has a shadow of its own. That happens when a parameter reaches the mesh, " *
        "or when the differentiated function is annotated `Duplicated` (for example " *
        "DifferentiationInterface's `function_annotation = Enzyme.Duplicated`) and " *
        "captures the space. Annotate the function `Const` (`function_annotation = " *
        "Enzyme.Const`), or, for a parameter in the mesh, use forward mode.",
    ),
    )
end

@noinline function _throw_mutable_relation(r)
    throw(
        ArgumentError(
        "reverse mode through `constrained` cannot differentiate the relation on the " *
        "facet `$(r.name)`: its datum or coefficient captures an array or another mutable " *
        "object. Capture numbers instead (`g = p[1]` before the relation), or use " *
        "forward mode.",
    ),
    )
end

# The space is differentiated only if its shadow is not itself: a `Const`, or runtime
# activity's inactive data, a shadow that is the primal. A shadow of its own comes from a
# parameter in the mesh, or from a `Duplicated` function that captures the space; the rule
# cannot tell them apart, and dropping a mesh shadow would give wrong gradients.
_space_is_data(::Const) = true
_space_is_data(P::Duplicated) = P.dval === P.val
_space_is_data(::Annotation) = false

# A relation reaches the rule as `Active` (an immutable of numbers) or `Const`; one that
# captures a mutable object is `Duplicated` or `MixedDuplicated`.
_check_relation(::Union{Const, Active}) = nothing
_check_relation(r::Annotation) = _throw_mutable_relation(r.val)

# `x` with each float replaced by `f(float, index)`, the floats counted in field order from
# `k`: a `Complex` is its two floats. Every other field is kept.
_remap_floats(f, x::AbstractFloat, k::Base.RefValue{Int}) = f(x, k[] += 1)
function _remap_floats(f, x::T, k::Base.RefValue{Int}) where {T}
    (fieldcount(T) == 0 || !_has_float(T)) && return x
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T,
        Any[_remap_floats(f, getfield(x, i), k) for i in 1:fieldcount(T)], fieldcount(T))::T
end

# The cotangent of the relation `r`: the real inner product of each float's forward
# derivative of the matrix and the load with the matrix's and the load's cotangents.
function _relation_cotangent(config, func, P, r, dR, db)
    mode = Enzyme.set_runtime_activity(Enzyme.Forward, EnzymeRules.runtime_activity(config))
    n = 0
    _remap_floats((x, _) -> (n += 1; x), r, Ref(0))
    vals = zeros(n)
    for l in 1:n
        tangent = _remap_floats((x, i) -> i == l ? one(x) : zero(x), r, Ref(0))
        Ṙ, ḃ = only(Enzyme.autodiff(mode, Const(func), Duplicated, Const(P),
            Duplicated(r, tangent)))
        vals[l] = real(dot(nonzeros(Ṙ), nonzeros(dR))) + real(dot(ḃ, db))
    end
    return _remap_floats((x, i) -> oftype(x, vals[i]), r, Ref(0))
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(Bramble._assemble_relation)},
        ::Type{RT},
        P::Annotation,
        r::Annotation
) where {RT}
    EnzymeRules.width(config) == 1 || _throw_batched("constrained", "reverse", EnzymeRules.width(config))
    _space_is_data(P) || _throw_mesh_parameter()
    _check_relation(r)

    R, b = func.val(P.val, r.val)
    primal = EnzymeRules.needs_primal(config) ? (R, b) : nothing
    # The return's own cotangent accumulates in the shadows, and `reverse` reads it back off
    # the tape.
    shadow = RT <: Const ? nothing : (Enzyme.make_zero(R), Enzyme.make_zero(b))
    # The type is built from the return's annotation, the tape's type left open: the data
    # a function of a global, say, makes the return type abstract, and the concrete
    # `AugmentedReturn` of its values is no subtype of what Enzyme expects.
    ART = EnzymeRules.augmented_rule_return_type(typeof(config), RT, Any)
    return ART(primal, EnzymeRules.needs_shadow(config) ? shadow : nothing, shadow)::ART
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(Bramble._assemble_relation)},
        ::Type{RT},
        tape,
        P::Annotation,
        r::Annotation
) where {RT}
    if tape === nothing
        return (nothing, r isa Active ? _zero_bits(r.val) : nothing)
    end
    dR, db = tape
    r̄ = r isa Active ? _relation_cotangent(config, func.val, P.val, r.val, dR, db) : nothing
    # Enzyme's contract: a custom rule consumes the return cotangent it was handed.
    fill!(nonzeros(dR), zero(eltype(dR)))
    fill!(db, zero(eltype(db)))
    return (nothing, r̄)
end

# The rules for `Bramble._dense_block_solve`, `X = C⁻¹ B` and `x = C⁻¹ b` on one block of
# the ghosts that a conormal `∂ₙ(u; K)` couples (#1176). Enzyme has no derivative for
# LAPACK's `getrf`. Every tangent and adjoint solves against the primal's own LU: forward
# `Ẋ = C⁻¹(Ḃ − Ċ X)`, reverse `Λ = C⁻ᴴ X̄`, `B̄ += Λ`, `C̄ −= Λ Xᴴ`, and likewise for `x`.
# The reverse reads its own copies of `X` and `x` (#1185). The flag of `_nonsingular`
# passes through; a singular block gets zero tangents and cotangents, since its caller
# throws.

# The block's LU and `Bramble._dense_block_solve`'s result, solved on that LU.
function _block_primal(C, B, b)
    F = lu(C; check = false)
    Bramble._nonsingular(F, C) || return F, Bramble._dense_block_solve(C, B, b)
    return F, (F \ B, F \ b, true)
end

# `F \ (Ḃ − Ċ Y)` for one lane; a missing tangent is `nothing`.
function _block_tangent(F, Ċ, Ḃ, Y)
    rhs = Ḃ === nothing ? zero(Y) : copy(Ḃ)
    Ċ === nothing || mul!(rhs, Ċ, Y, -1, 1)
    return F \ rhs
end

function EnzymeRules.forward(
        config::FwdConfig,
        func::Const{typeof(Bramble._dense_block_solve)},
        ::Type{RT},
        C::Annotation,
        B::Annotation,
        b::Annotation
) where {RT}
    _entered!(:_dense_block_solve)
    F, (X, x, ok) = _block_primal(C.val, B.val, b.val)
    RT <: Const && return EnzymeRules.needs_primal(config) ? (X, x, ok) : nothing
    W = EnzymeRules.width(config)
    shadows = ntuple(Val(W)) do w
        ok || return (zero(X), zero(x), ok)
        Ċ = _shadow(C, w)
        return (_block_tangent(F, Ċ, _shadow(B, w), X),
            _block_tangent(F, Ċ, _shadow(b, w), x), ok)
    end
    needs_primal = EnzymeRules.needs_primal(config)
    EnzymeRules.needs_shadow(config) || return needs_primal ? (X, x, ok) : nothing
    shadow = W == 1 ? only(shadows) : shadows
    needs_primal || return shadow
    return W == 1 ? Duplicated((X, x, ok), shadow) : BatchDuplicated((X, x, ok), shadow)
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(Bramble._dense_block_solve)},
        ::Type{RT},
        C::Annotation,
        B::Annotation,
        b::Annotation
) where {RT}
    EnzymeRules.width(config) == 1 || _throw_batched("constrained", "reverse", EnzymeRules.width(config))
    _entered!(:_dense_block_solve)
    F, (X, x, ok) = _block_primal(C.val, B.val, b.val)
    primal = EnzymeRules.needs_primal(config) ? (X, x, ok) : nothing
    # The return's own cotangent accumulates in the shadows, and `reverse` reads it back off
    # the tape.
    shadow = RT <: Const ? nothing : (zero(X), zero(x), ok)
    tape = (F, copy(X), copy(x), shadow)
    needs_shadow = EnzymeRules.needs_shadow(config)
    return AugmentedReturn(primal, needs_shadow ? shadow : nothing, tape)
end

# `C̄ −= Λ Yᴴ`. The caller mixes real and complex blocks and data on purpose (`Tg` in
# `_block_solve`), so a real `C̄` can meet a complex product: it takes the real part.
function _subtract_outer!(C̄, Λ, Y)
    if eltype(C̄) <: Real && !(eltype(Λ) <: Real && eltype(Y) <: Real)
        C̄ .-= real.(Λ * Y')
    else
        mul!(C̄, Λ, Y', -1, 1)
    end
    return nothing
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(Bramble._dense_block_solve)},
        ::Type{RT},
        tape,
        C::Annotation,
        B::Annotation,
        b::Annotation
) where {RT}
    F, X, x, shadow = tape
    shadow === nothing && return (nothing, nothing, nothing)
    X̄, x̄, ok = shadow
    ok || return (nothing, nothing, nothing)
    Λ, λ = F' \ X̄, F' \ x̄
    C̄, B̄, b̄ = _shadow(C, 1), _shadow(B, 1), _shadow(b, 1)
    B̄ === nothing || _accumulate_shadow!(B.val, B̄, Λ)
    b̄ === nothing || _accumulate_shadow!(b.val, b̄, λ)
    if C̄ !== nothing
        _subtract_outer!(C̄, Λ, X)
        _subtract_outer!(C̄, λ, x)
    end
    # Enzyme's contract: a custom rule consumes the return cotangent it was handed.
    fill!(X̄, zero(eltype(X̄)))
    fill!(x̄, zero(eltype(x̄)))
    return (nothing, nothing, nothing)
end

# The reverse rules for `Bramble._corner_entry`, the entry of a relation's row that a
# one-sided corner of tractions moves (`_one_sided_corners!`). With a coefficient of a
# relation's operator active there, Enzyme's reverse derivative of `constrained` on the
# composite is wrong (#1180), though each step of it is right on its own, so the rules
# refuse an active entry. Forward mode needs no rule: it is exact there since
# `_composite_relations` is inlined (#1226).
@noinline function _throw_active_corner()
    throw(
        ArgumentError(
        "Enzyme's reverse mode cannot differentiate `constrained` on a composite through " *
        "a corner where two relations reading several components meet (tractions), with " *
        "a parameter in a relation's operator: its derivative there comes out wrong " *
        "(#1180). Use Enzyme's forward mode or ForwardDiff.",
    ),
    )
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(Bramble._corner_entry)},
        ::Type{RT},
        v::Annotation
) where {RT}
    v isa Const || _throw_active_corner()
    primal = EnzymeRules.needs_primal(config) ? v.val : nothing
    return AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(Bramble._corner_entry)},
        ::Type{RT},
        tape,
        v::Annotation
) where {RT}
    return (nothing,)
end
