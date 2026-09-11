# precompile/operator_sessions.jl: difference, jump and average operators, and the inner
# products and norms (src/space/operators/).
#
# This is the part of the space interface where precompilation pays in full.
# Rₕ and avgₕ specialize on the caller's function type, so most of what a
# workload caches for them is thrown away by a user's own function; an
# operator's method instance is fixed by the element type and the direction
# alone, and an inner product's by the element types, so nothing here is
# closure-dependent and every instance is reused verbatim.
#
# Measured on a step function that applies several operators and then takes
# the inner products and norms of the result, which is the shape a scheme
# actually has: first call 624 ms without this workload against 231 ms with it,
# in 1D and 2D together. Calling each operator and product separately at top
# level, 1.52 s against 254 ms.
#
# It is not free: the workload adds about 11 s to the package's precompile time
# and 8 MB to its cache. Set the `precompile_workload` preference to false, as
# documented at the top of `precompile.jl`, to skip all of it while iterating.
#
# The matrix forms of the operators are deliberately left out: they are on the
# way out of the library, and caching them would grow the image for code that
# is being removed.

# `const` so that each tuple has a concrete type and the loops below stay
# inferable, the same reason the operator config tables are `const`.
const _PC_OPS_X = (diff₋ₓ, diff₊ₓ, D₋ₓ, D₊ₓ, jumpₓ, M₋ₓ, M₊ₓ, Dstar₊ₓ, Dcₓ, Dₕₓ)
const _PC_OPS_Y = (diff₋ᵧ, diff₊ᵧ, D₋ᵧ, D₊ᵧ, jumpᵧ, M₋ᵧ, M₊ᵧ, Dstar₊ᵧ, Dcᵧ, Dₕᵧ)
const _PC_OPS_Z = (diff₋₂, diff₊₂, D₋₂, D₊₂, jump₂, M₋₂, M₊₂, Dstar₊₂, Dc₂, Dₕ₂)

# The vectorial aliases, which return a bare element in 1D and a tuple above it,
# so both returns get compiled.
const _PC_OPS_ALL = (∇₋ₕ, ∇₊ₕ, diff₋ₕ, diff₊ₕ, jumpₕ, M₋ₕ, M₊ₕ, Dstar₊ₕ, Dcₕ, ∇ₕ)

# Applied with a plain loop over the tuple, which inference unrolls into a static
# call per operator. Going through `foreach` and a closure instead leaves the
# calls dynamically dispatched, and then only the dispatch site is cached and
# every alias costs its ~7 ms again on first use: measured 254 ms against
# 460 ms over the calls in this file.
function _pc_apply_each(ops, uₕ)
    for op in ops
        op(uₕ)
    end
    return nothing
end

# The directional aliases per coordinate. One method per dimension rather than a
# runtime branch, so that inference never sees D₋ᵧ applied to a 1D element.
_pc_directional_ops(uₕ, ::Val{1}) = _pc_apply_each(_PC_OPS_X, uₕ)

function _pc_directional_ops(uₕ, ::Val{2})
    _pc_directional_ops(uₕ, Val(1))
    return _pc_apply_each(_PC_OPS_Y, uₕ)
end

function _pc_directional_ops(uₕ, ::Val{3})
    _pc_directional_ops(uₕ, Val(2))
    return _pc_apply_each(_PC_OPS_Z, uₕ)
end

const _PC_OPS_X_INPLACE = (
    diff₋ₓ!, diff₊ₓ!, D₋ₓ!, D₊ₓ!, jumpₓ!, M₋ₓ!, M₊ₓ!, Dstar₊ₓ!, Dcₓ!, Dₕₓ!
)
const _PC_OPS_Y_INPLACE = (
    diff₋ᵧ!, diff₊ᵧ!, D₋ᵧ!, D₊ᵧ!, jumpᵧ!, M₋ᵧ!, M₊ᵧ!, Dstar₊ᵧ!, Dcᵧ!, Dₕᵧ!
)
const _PC_OPS_Z_INPLACE = (
    diff₋₂!, diff₊₂!, D₋₂!, D₊₂!, jump₂!, M₋₂!, M₊₂!, Dstar₊₂!, Dc₂!, Dₕ₂!
)

function _pc_apply_each_inplace(ops, vₕ, uₕ)
    for op in ops
        op(vₕ, uₕ)
    end
    return nothing
end

function _pc_directional_ops_inplace(vₕ, uₕ, ::Val{1})
    return _pc_apply_each_inplace(_PC_OPS_X_INPLACE, vₕ, uₕ)
end

function _pc_directional_ops_inplace(vₕ, uₕ, ::Val{2})
    _pc_directional_ops_inplace(vₕ, uₕ, Val(1))
    return _pc_apply_each_inplace(_PC_OPS_Y_INPLACE, vₕ, uₕ)
end

function _pc_directional_ops_inplace(vₕ, uₕ, ::Val{3})
    _pc_directional_ops_inplace(vₕ, uₕ, Val(2))
    return _pc_apply_each_inplace(_PC_OPS_Z_INPLACE, vₕ, uₕ)
end

_pc_vectorial_ops(uₕ) = _pc_apply_each(_PC_OPS_ALL, uₕ)

# innerₕ and the norms built on it take a grid function of a scalar space; inner₊
# and norm₊ take the gradient tuple, which in 1D is the bare element.
#
# innerₕ, normₕ and _dot are all @inline, so no standalone specialization of
# them exists to be cached: they are inlined into whatever calls them, and in a
# user's program that caller is the user's own method. Calling one at top level
# in a fresh session therefore still costs about 9 ms, and nothing this workload
# can do removes that; it is the cost of building a specialization for a call
# that was not inlined into a method.
#
# What the calls below do cache is everything non-inline underneath: the
# directional inner-product kernels, the seminorm machinery, and the operators
# that feed them. That is where the saving is, and it is the case that matters,
# since a scheme calls these from inside its own step function rather than from
# the prompt.
function _pc_inner_products(uₕ, dim_val::Val{D}) where {D}
    innerₕ(uₕ, uₕ)
    normₕ(uₕ)
    snorm₁ₕ(uₕ)
    norm₁ₕ(uₕ)

    gₕ = ∇₋ₕ(uₕ)
    inner₊(gₕ, gₕ)
    norm₊(gₕ)
    inner₊(uₕ, uₕ)

    _pc_directional_inner(uₕ, dim_val)
    return nothing
end

function _pc_directional_inner(uₕ, ::Val{1})
    inner₊ₓ(uₕ, uₕ)
    return nothing
end
function _pc_directional_inner(uₕ, ::Val{2})
    inner₊ₓ(uₕ, uₕ)
    inner₊ᵧ(uₕ, uₕ)
    return nothing
end
function _pc_directional_inner(uₕ, ::Val{3})
    _pc_directional_inner(uₕ, Val(2))
    inner₊₂(uₕ, uₕ)
    return nothing
end

# A composite grid function takes a separate dispatch through the operators, and
# its components are scalar grid functions over a contiguous view, which is a
# distinct element type from the one above and so a distinct set of instances.
function _pc_operator_session(uₕ, cₕ, dim_val::Val)
    _pc_directional_ops(uₕ, dim_val)
    _pc_vectorial_ops(uₕ)
    _pc_inner_products(uₕ, dim_val)

    v_out = similar(uₕ)
    _pc_directional_ops_inplace(v_out, uₕ, dim_val)

    _pc_directional_ops(cₕ, dim_val)
    _pc_vectorial_ops(cₕ)

    vc_out = similar(cₕ)
    _pc_directional_ops_inplace(vc_out, cₕ, dim_val)

    kₕ = components(cₕ)[1]
    _pc_directional_ops(kₕ, dim_val)
    _pc_inner_products(kₕ, dim_val)
    return nothing
end
