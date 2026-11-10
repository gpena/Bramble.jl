# A reverse rule for `Bramble._assemble_relation`, the forms of one relation of `constrained`
# (#1153).
#
# Enzyme's own reverse of the function compiles one copy of the forms per facet and takes
# 700 s and 3.7 GB for one relation, so `constrained` with a parameter in a datum or in a
# Robin coefficient did not finish. Forward mode compiles the same function in under a
# minute. The rule is the function's own call; its pullback takes one forward derivative of
# the function per float of the relation `r` (its datum, its Robin coefficient: the numbers
# `r` holds), the tangent a copy of `r` with that float one and every other zero, and
# contracts it with the cotangents of the matrix and the load. The relation's floats are the
# only inputs: the operators of a relation hold no data, and the space is not differentiated.

@noinline function _throw_batched_relation(width)
    throw(
        ArgumentError(
        "Bramble's Enzyme rule for the relations of `constrained` does not support batched " *
        "(vector-mode) reverse differentiation: got width $(width). Differentiate one " *
        "direction at a time.",
    ),
    )
end

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
    EnzymeRules.width(config) == 1 || _throw_batched_relation(EnzymeRules.width(config))
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
