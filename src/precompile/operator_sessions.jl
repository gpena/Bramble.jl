# precompile/operator_sessions.jl: difference, jump and average operators, and the inner
# products and norms (src/operators/).
#
# This is the part of the space interface where precompilation pays in full.
# Rₕ and avgₕ specialize on the caller's function type, so most of what a
# workload caches for them is thrown away by a user's own function; an
# operator's method instance is fixed by the element type and the direction
# alone, and an inner product's by the element types, so nothing here is
# closure-dependent and every instance is reused verbatim.
#
# The 1D session applies every operator and product. The 2D and 3D sessions apply only
# what their direction adds (the y and z aliases, the tuple returns, `curlₕ`): an x alias on
# a 2D or 3D grid function runs the methods it runs in 1D, so its instance for that element
# type is left to the caller's first use. Set the `precompile_workload` preference to
# false, as documented at the top of `precompile.jl`, to skip all of it while iterating.
#
# The matrix forms of the operators are deliberately left out: they are on the
# way out of the library, and caching them would grow the image for code that
# is being removed.

# `const` so that each tuple has a concrete type, the same reason the operator
# config tables are `const`. The loops below still dispatch at run time (see
# `_pc_apply_each`).
const _PC_OPS_X = (diff₋ₓ, diff₊ₓ, D₋ₓ, D₊ₓ, jumpₓ, Mₓ, M₊ₓ, D̃ₓ, Dcₓ, D̽ₓ)
const _PC_OPS_Y = (diff₋ᵧ, diff₊ᵧ, D₋ᵧ, D₊ᵧ, jumpᵧ, Mᵧ, M₊ᵧ, D̃ᵧ, Dcᵧ, D̽ᵧ)
const _PC_OPS_Z = (diff₋₂, diff₊₂, D₋₂, D₊₂, jump₂, M₂, M₊₂, D̃₂, Dc₂, D̽₂)

# The vectorial aliases, which return a bare element in 1D and a tuple above it,
# so both returns get compiled.
const _PC_OPS_ALL = (∇ₕ, ∇₊ₕ, diff₋ₕ, diff₊ₕ, jumpₕ, Mₕ, M₊ₕ, D̃ₕ, Dcₕ, D̽ₕ)

# The in-place forms, applied as op(vₕ, uₕ).
const _PC_OPS_X_INPLACE = (
    diff₋ₓ!, diff₊ₓ!, D₋ₓ!, D₊ₓ!, jumpₓ!, Mₓ!, M₊ₓ!, D̃ₓ!, Dcₓ!, D̽ₓ!
)
const _PC_OPS_Y_INPLACE = (
    diff₋ᵧ!, diff₊ᵧ!, D₋ᵧ!, D₊ᵧ!, jumpᵧ!, Mᵧ!, M₊ᵧ!, D̃ᵧ!, Dcᵧ!, D̽ᵧ!
)
const _PC_OPS_Z_INPLACE = (
    diff₋₂!, diff₊₂!, D₋₂!, D₊₂!, jump₂!, M₂!, M₊₂!, D̃₂!, Dc₂!, D̽₂!
)

# Applied with a plain loop over the tuple. The loop variable `op` is inferred as
# `Any`, so each call dispatches at run time: deliberate, because every operator is
# then compiled and cached as a standalone method instance that a user's first call
# reuses. The `foreach` variant, which afoldl unrolls into a static chain, caches
# the chain as one instance instead and every alias costs its compile again on first use.
# `args` is what each operator is called with: (uₕ) out of place, (vₕ, uₕ) in place.
function _pc_apply_each(ops, args...)
    for op in ops
        op(args...)
    end
    return nothing
end

_pc_vectorial_ops(uₕ) = _pc_apply_each(_PC_OPS_ALL, uₕ)

# The vector calculus operators on a 1D grid function.
function _pc_vector_calculus(uₕ::VectorElement)
    gₕ = ∇ₕ(uₕ)
    Δₕ(uₕ)
    Δₕ!(similar(uₕ), uₕ)
    divₕ(gₕ)
    divₕ!(similar(uₕ), gₕ)
    return nothing
end

# innerₕ and the norms built on it take a grid function of a scalar space; inner₊
# and norm₊ take the gradient tuple, which in 1D is the bare element.
#
# innerₕ, normₕ and _dot are all @inline, so the calls below inline them into this
# session and leave no standalone specialization behind. A call at top level in a fresh
# session would then cost about 9 ms of codegen; the explicit `precompile` of each entry
# signature at the end of `_pc_inner_products` forces that standalone instance to exist.
#
# What the calls below do cache is everything non-inline underneath: the
# directional inner-product kernels, the seminorm machinery, and the operators
# that feed them. That is where the saving is, and it is the case that matters,
# since a scheme calls these from inside its own step function rather than from
# the prompt.
function _pc_inner_products(uₕ)
    innerₕ(uₕ, uₕ)
    normₕ(uₕ)
    snorm₁ₕ(uₕ)
    norm₁ₕ(uₕ)
    norminf(uₕ)

    gₕ = ∇ₕ(uₕ)
    inner₊(gₕ, gₕ)
    norm₊(gₕ)
    norminf(gₕ)
    inner₊(uₕ, uₕ)
    inner₊ₓ(uₕ, uₕ)

    # Standalone instances of the inlined entry points, for a call from the prompt.
    precompile(normₕ, (typeof(uₕ),))
    precompile(norm₁ₕ, (typeof(uₕ),))
    precompile(snorm₁ₕ, (typeof(uₕ),))
    precompile(innerₕ, (typeof(uₕ), typeof(uₕ)))
    return nothing
end

# The 1D session, in full. A composite grid function takes a separate dispatch through the
# operators, so the out-of-place operators are applied to one as well.
function _pc_operator_session(uₕ, cₕ, ::Val{1})
    _pc_apply_each(_PC_OPS_X, uₕ)
    _pc_vectorial_ops(uₕ)
    _pc_inner_products(uₕ)
    _pc_vector_calculus(uₕ)
    _pc_apply_each(_PC_OPS_X_INPLACE, similar(uₕ), uₕ)
    _pc_apply_each(_PC_OPS_X, cₕ)
    _pc_shift_form_session(Val(1))
    return nothing
end

# What the second direction adds: the y aliases, the vectorial aliases' tuple return, the
# seminorm, the product and the norm of a gradient tuple, `divₕ` of one, `curlₕ`, the
# strain tensor and the shifts along y.
function _pc_operator_session(uₕ, ::Val{2})
    _pc_apply_each(_PC_OPS_Y, uₕ)
    _pc_apply_each(_PC_OPS_Y_INPLACE, similar(uₕ), uₕ)
    _pc_vectorial_ops(uₕ)
    snorm₁ₕ(uₕ)
    gₕ = ∇ₕ(uₕ)
    inner₊(gₕ, gₕ)
    norminf(gₕ)
    inner₊ᵧ(uₕ, uₕ)
    divₕ(gₕ)
    curlₕ((uₕ, uₕ))
    _pc_strain_tensor_session()
    _pc_shift_form_session(Val(2))
    return nothing
end

# What the third direction adds: the z aliases, `curlₕ` of three components and the
# forward shift along z (#1023).
function _pc_operator_session(uₕ, ::Val{3})
    _pc_apply_each(_PC_OPS_Z, uₕ)
    _pc_apply_each(_PC_OPS_Z_INPLACE, similar(uₕ), uₕ)
    _pc_vectorial_ops(uₕ)
    inner₊₂(uₕ, uₕ)
    curlₕ((uₕ, uₕ, uₕ))
    _pc_shift_form_session(Val(3))
    return nothing
end

# εₕ/εₕ! on a 2D composite field, built small and standalone here. The grid space's type
# carries no marker names, so the separate mesh is for independence from the session's
# fields above, not for type coverage.
function _pc_strain_tensor_session()
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ω = domain(S, :boundary => boundary_symbols(S))
    Ωₕ = mesh(Ω, (2, 2), (false, false))
    Vₕ = gridspace(Ωₕ, Val(2))
    cₕ = element(Vₕ, 1.0)
    E = εₕ(cₕ)
    εₕ!(E, cₕ)
    return nothing
end

# The index shifts. Only the form session below is cached: a first call of a shift on a grid
# function costs about 7 ms, too little to pay for its share of the package build, so the
# grid-function shifts are left to first use.
#
# The index shifts inside a form. The form session reaches no `ShiftNode` otherwise, only
# `shift_op(id, 1, 1)` as a stencil that is never assembled. On a non-uniform mesh. Two
# forms are built. One is the neighbour sum written with the exported vectorial aliases,
# `S₊ₕ(u)[d] + S₋ₕ(u)[d]` (in 1D, `[1]` takes a component of the single node rather than a
# direction, a node of its own), in 1D and 2D only. The other is the forward-shifted unknown
# alone, `innerₕ(S(u), v)`, along the direction each dimension adds: `S₋ₓ(u)` builds the same
# `ShiftNode` type as `S₊ₓ(u)`, with the amount as a field. Compositions (`S(D₋ₓ(u))`,
# `S(jumpₓ(u))`, ...) are left to first use: each is a form type of its own.
function _pc_shift_form_session(::Val{1})
    Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 6, false))
    _pc_neighbour_sum(Wₕ, Val(1))
    _pc_shift_form(Wₕ, S₊ₓ)
    return nothing
end

function _pc_shift_form_session(::Val{2})
    X = interval(0.0, 1.0) × interval(0.0, 1.0)
    Wₕ = gridspace(mesh(domain(X), (5, 4), (false, false)))
    _pc_neighbour_sum(Wₕ, Val(1))
    _pc_shift_form(Wₕ, S₊ᵧ)
    return nothing
end

function _pc_shift_form_session(::Val{3})
    X = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    Wₕ = gridspace(mesh(domain(X), (4, 3, 3), (false, false, false)))
    _pc_shift_form(Wₕ, S₊₂)
    return nothing
end

function _pc_neighbour_sum(Wₕ, ::Val{d}) where {d}
    return assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(S₊ₕ(u)[d], v) + innerₕ(S₋ₕ(u)[d], v)))
end

_pc_shift_form(Wₕ, S) = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(S(u), v)))
