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

# One table triple per form, walked by `_pc_directional_ops`.
const _PC_OPS_TABLES = (_PC_OPS_X, _PC_OPS_Y, _PC_OPS_Z)
const _PC_OPS_TABLES_INPLACE = (_PC_OPS_X_INPLACE, _PC_OPS_Y_INPLACE, _PC_OPS_Z_INPLACE)

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

# The directional aliases per coordinate, over a triple of tables. One method per
# dimension rather than a runtime branch, so that inference never sees D₋ᵧ applied to
# a 1D element.
_pc_directional_ops(tables, ::Val{1}, args...) = _pc_apply_each(tables[1], args...)

function _pc_directional_ops(tables, ::Val{2}, args...)
    _pc_directional_ops(tables, Val(1), args...)
    return _pc_apply_each(tables[2], args...)
end

function _pc_directional_ops(tables, ::Val{3}, args...)
    _pc_directional_ops(tables, Val(2), args...)
    return _pc_apply_each(tables[3], args...)
end

_pc_vectorial_ops(uₕ) = _pc_apply_each(_PC_OPS_ALL, uₕ)

# The vector calculus operators. Each recurses over directions, so a
# session per dimension is what caches the recursion's own specializations; the gradient is
# the field they are most often applied to.
function _pc_vector_calculus(uₕ::VectorElement, ::Val{D}) where {D}
    gₕ = ∇ₕ(uₕ)
    Δₕ(uₕ)
    Δₕ!(similar(uₕ), uₕ)
    divₕ(gₕ)
    divₕ!(similar(uₕ), gₕ)
    D >= 2 && curlₕ(ntuple(_ -> uₕ, Val(D)))
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
function _pc_inner_products(uₕ, dim_val::Val{D}) where {D}
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

    _pc_directional_inner(uₕ, dim_val)

    # Standalone instances of the inlined entry points, for a call from the prompt.
    precompile(normₕ, (typeof(uₕ),))
    precompile(norm₁ₕ, (typeof(uₕ),))
    precompile(snorm₁ₕ, (typeof(uₕ),))
    precompile(innerₕ, (typeof(uₕ), typeof(uₕ)))
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
    _pc_directional_ops(_PC_OPS_TABLES, dim_val, uₕ)
    _pc_vectorial_ops(uₕ)
    _pc_inner_products(uₕ, dim_val)
    _pc_vector_calculus(uₕ, dim_val)

    v_out = similar(uₕ)
    _pc_directional_ops(_PC_OPS_TABLES_INPLACE, dim_val, v_out, uₕ)

    _pc_directional_ops(_PC_OPS_TABLES, dim_val, cₕ)
    _pc_vectorial_ops(cₕ)

    vc_out = similar(cₕ)
    _pc_directional_ops(_PC_OPS_TABLES_INPLACE, dim_val, vc_out, cₕ)

    kₕ = components(cₕ)[1]
    _pc_directional_ops(_PC_OPS_TABLES, dim_val, kₕ)
    _pc_inner_products(kₕ, dim_val)

    # Float64 only: the Float32 2D session below would otherwise rebuild the same
    # Float64 mesh a second time for no extra coverage.
    dim_val isa Val{2} && eltype(uₕ) === Float64 && _pc_strain_tensor_session()
    # 3D too, forward shifts only (#1023): +1.1 MiB of pkgimage, precompile time within its
    # spread, takes each 3D shift form's first `assemble` from 183-245 ms down to 4 ms.
    eltype(uₕ) === Float64 && _pc_shift_form_session(dim_val)
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
# Forward only: `S₋ₓ(u)` builds the same `ShiftNode` type as `S₊ₓ(u)`, the amount a field.
const _PC_SHIFTS = (S₊ₓ, S₊ᵧ, S₊₂)

# The index shifts inside a form: the form session reaches no `ShiftNode` otherwise, only
# `shift_op(id, 1, 1)` as a stencil that is never assembled. On a non-uniform mesh of each
# dimension, along each direction. Two forms are built. One is the neighbour sum written with
# the exported vectorial aliases, `S₊ₕ(u)[d] + S₋ₕ(u)[d]` (in 1D, `[1]` takes a component of
# the single node rather than a direction, a node of its own), in 1D and 2D only. A 3D sum's
# first `assemble` costs about 100 ms, inside form_sessions.jl's 78-114 ms for 3D shapes.
# The other is, per direction, the forward-shifted unknown alone, `innerₕ(S(u), v)`, as its
# own form. Assembly specializes on the whole form, so the sum does not cache the single
# term.
# Compositions (`S(D₋ₓ(u))`, `S(jumpₓ(u))`, ...) are left to first use: each is a form type
# of its own, and caching one would be picking the tests' cases. Float64 only, as the tests
# and the solvers assemble.
function _pc_shift_form_session(::Val{D}) where {D}
    X, n = D == 1 ? (interval(0.0, 1.0), 6) :
           D == 2 ? (interval(0.0, 1.0) × interval(0.0, 1.0), (5, 4)) :
           (box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (4, 3, 3))
    Wₕ = gridspace(mesh(domain(X), n, map(_ -> false, n)))
    # `map` over the tuples rather than a loop: each direction and each shift is its own
    # type, and `map` hands each one on with that type known
    D < 3 && map(dim_val -> _pc_neighbour_sum(Wₕ, dim_val), ntuple(Val, Val(D)))
    map(S -> _pc_shift_form(Wₕ, S), ntuple(k -> _PC_SHIFTS[k], Val(D)))
    return nothing
end

function _pc_neighbour_sum(Wₕ, ::Val{d}) where {d}
    return assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(S₊ₕ(u)[d], v) + innerₕ(S₋ₕ(u)[d], v)))
end

_pc_shift_form(Wₕ, S) = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(S(u), v)))
