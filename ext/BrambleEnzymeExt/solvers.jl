# The rules for the solves `Bramble` adds beside `pde_solve(::SparseMatrixCSC, F)`.
#
# `pde_solve(K::RankOneUpdate, F)` calls `Bramble._bordered_solve` directly, so the rule on
# the sparse-matrix method never fires for it and Enzyme enters the SuiteSparse solve.
# `K = A + σ wₛ wₜᵀ` and `K u = F` give the tangent `u̇ = K⁻¹(Ḟ − K̇ u)` and the pullback
# `λ = K⁻ᴴ ū`, `K̄ = −λ uᴴ`. Every solve is `pde_solve` on `K` or `K'`: no factorisation is
# kept, since the bordered matrix is not `K`'s, and the primal's checks and warnings run once,
# in the primal solve. `RankOneUpdate` is mutable and `σ` its one non-const field, so a
# `Duplicated` shadow holds the cotangents of `A`, `wₛ`, `wₜ` and `σ`.

@noinline function _throw_batched_rank_one(width)
    throw(
        ArgumentError(
        "Bramble's Enzyme rule for `pde_solve(::RankOneUpdate, F)` does not support batched " *
        "(vector-mode) reverse differentiation: got width $(width). Differentiate one " *
        "direction at a time.",
    ),
    )
end

@noinline function _throw_rank_one_matrix(@nospecialize(A))
    throw(
        ArgumentError(
        "Bramble's Enzyme rule for `pde_solve(::RankOneUpdate, F)` needs `K.A` to be a " *
        "`SparseMatrixCSC`: the cotangent of `A` accumulates on its stored pattern. Got a " *
        "$(nameof(typeof(A))). Rebuild `K` with `Bramble.RankOneUpdate(sparse(K.A), K.σ, " *
        "K.ws, K.wt)`.",
    ),
    )
end

# A `K.A` the pattern walk cannot read is refused by name; a `Const` `K` is never walked.
_rank_one_check(::Const) = nothing
function _rank_one_check(K::Annotation)
    K.val.A isa SparseMatrixCSC || _throw_rank_one_matrix(K.val.A)
    return nothing
end

# Under runtime activity a field Enzyme proves constant gets its own array as its shadow, so a
# shadow that is the primal is a zero tangent, not a direction.
_rank_one_tangent(x, ẋ) = ẋ === x ? nothing : ẋ

# `K̇ u` for one lane's shadow `K̇` of `K = A + σ wₛ wₜᵀ`:
# `Ȧ u + σ̇ wₛ (wₜᵀu) + σ ẇₛ (wₜᵀu) + σ wₛ (ẇₜᵀu)`.
function _rank_one_apply(K, K̇, u)
    Ȧ = _rank_one_tangent(K.A, K̇.A)
    ẇₛ = _rank_one_tangent(K.ws, K̇.ws)
    ẇₜ = _rank_one_tangent(K.wt, K̇.wt)
    tu = transpose(K.wt) * u
    r = zero(u)
    r .+= (K̇.σ * tu) .* K.ws
    Ȧ === nothing || (r .+= Ȧ * u)
    ẇₛ === nothing || (r .+= (K.σ * tu) .* ẇₛ)
    ẇₜ === nothing || (r .+= (K.σ * (transpose(ẇₜ) * u)) .* K.ws)
    return r
end

# `Ḟ − K̇ u` for one lane; a lane with neither a shadow of `K` nor of `F` has a zero tangent.
_rank_one_rhs(_, ::Nothing, ::Nothing, u) = zero(u)
_rank_one_rhs(K, K̇, ::Nothing, u) = -_rank_one_apply(K, K̇, u)
_rank_one_rhs(_, ::Nothing, Ḟ, u) = Ḟ
_rank_one_rhs(K, K̇, Ḟ, u) = Ḟ - _rank_one_apply(K, K̇, u)

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        K::Annotation{<:Bramble.RankOneUpdate},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    _rank_one_check(K)
    u = pde_solve(K.val, F.val; kwargs...)
    RT <: Const && return EnzymeRules.needs_primal(config) ? u : nothing
    W = EnzymeRules.width(config)
    shadows = ntuple(Val(W)) do i
        rhs = _rank_one_rhs(K.val, _lane(K, i), _lane(F, i), u)
        return pde_solve(K.val, rhs; kwargs...)
    end
    needs_primal = EnzymeRules.needs_primal(config)
    needs_shadow = EnzymeRules.needs_shadow(config)
    needs_shadow || return needs_primal ? u : nothing
    shadow = W == 1 ? only(shadows) : shadows
    return needs_primal ?
           (W == 1 ? Duplicated(u, shadow) : BatchDuplicated(u, shadow)) : shadow
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        K::Annotation{<:Bramble.RankOneUpdate},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    EnzymeRules.width(config) == 1 || _throw_batched_rank_one(EnzymeRules.width(config))
    _rank_one_check(K)

    u = pde_solve(K.val, F.val; kwargs...)
    # The adjoint reads `K`'s fields at the reverse pass, and a caller may overwrite them
    # after the solve (`EnzymeRules.overwritten` does not flag a matrix scaled in place), so
    # the tape keeps a copy.
    Kₜ = Bramble.RankOneUpdate(copy(K.val.A), K.val.σ, copy(K.val.ws), copy(K.val.wt))
    ū = zero(u)

    primal = EnzymeRules.needs_primal(config) ? u : nothing
    shadow = EnzymeRules.needs_shadow(config) ? ū : nothing
    return AugmentedReturn(primal, shadow, (Kₜ, u, ū))
end

# `pde_solve(Kᴴ, ū; kwargs...)`, with `mean_zero` and `gauge` conjugated as
# `Bramble._pde_solve_adjoint` does: the primal borders `K` as `[K C; Cᵀ 0]`, whose adjoint is
# `[Kᴴ conj(C); Cᴴ 0]`.
function _rank_one_adjoint_solve(K, ū; mean_zero = nothing, gauge = nothing, kwargs...)
    return pde_solve(K', ū; mean_zero = Bramble._conj_border(mean_zero),
        gauge = Bramble._conj_border(gauge), kwargs...)
end

# `x += y` into a shadow that is not the primal itself.
_accumulate_shadow!(x, ẋ, y) = (ẋ === x || (ẋ .+= y); nothing)

function EnzymeRules.reverse(
        config::RevConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        tape,
        K::Annotation{<:Bramble.RankOneUpdate},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    Kₜ, u, ū = tape
    λ = _rank_one_adjoint_solve(Kₜ, ū; kwargs...)

    if !(K isa Const) && K.dval !== K.val
        K̄ = K.dval
        σ = Kₜ.σ
        # `K̄ = −λ uᴴ`: on `A`'s stored pattern entry by entry, and by the product rule on
        # `σ`, `wₛ` and `wₜ`, with `λᵀwₛ` and `uᴴwₜ` the two scalars of the rank-one term.
        if K̄.A !== K.val.A
            Āv = nonzeros(K̄.A)
            rows = rowvals(Kₜ.A)
            @inbounds for j in axes(Kₜ.A, 2), k in nzrange(Kₜ.A, j)

                Āv[k] -= λ[rows[k]] * conj(u[j])
            end
        end
        sλ = transpose(Kₜ.ws) * λ
        tu = dot(u, Kₜ.wt)
        K̄.σ += σ isa Real ? real(-(sλ * tu)) : -(sλ * tu)
        _accumulate_shadow!(K.val.ws, K̄.ws, (-σ * tu) .* λ)
        _accumulate_shadow!(K.val.wt, K̄.wt, (-σ * sλ) .* conj.(u))
    end

    if !(F isa Const)
        F.dval .+= λ
    end

    # A custom rule consumes the return cotangent it was handed (see the rule on a matrix).
    fill!(ū, zero(eltype(ū)))

    return (nothing, nothing)
end

# The rules for a kept factorization (#1165).
#
# Enzyme enters the factorization's `Union` field and stops with `IllegalTypeAnalysisException`,
# so the rules keep it out. With `fact` constant, `u = fact \ F` is linear in `F`. Each lane's
# tangent is `u̇ = fact \ Ḟ`, and the pullback is `λ = fact' \ ū`, `F̄ += λ`, with the
# return's cotangent zeroed after use, as in the rule on `pde_solve(A, F)`. The matrix's
# derivative is not carried, so `sparse_factorize` of an active matrix throws and names
# `pde_solve(A, F)` as the call that carries it. The factorization must be built outside the
# differentiated function: built inside, its inferred `Union` can abort forward mode. The
# rules cover `pde_solve(fact, F)` and `ldiv!(uₕ, fact, F)`. The latter works in place, and
# in reverse it reads the destination's shadow and then zeroes it. The factorization is kept
# by reference, so a `refactor!` between the forward and the reverse pass would change the
# adjoint solve.

@noinline function _throw_adjoint_unavailable(@nospecialize(fact))
    throw(
        ArgumentError(
        "Bramble's Enzyme rule for a kept factorization solves the adjoint system " *
        "`fact' \\ ū` in reverse mode, which a $(nameof(typeof(fact))) does not provide. " *
        "Use forward mode, or differentiate `pde_solve(A, F)` on the matrix.",
    ),
    )
end

@noinline function _throw_active_factorize()
    throw(
        ArgumentError(
        "Bramble's Enzyme rules do not carry the derivative of a matrix through " *
        "`sparse_factorize`: its argument has a shadow, and the factorization would " *
        "silently drop it. Differentiate `pde_solve(A, F)` on the matrix instead, or " *
        "factor a matrix that does not depend on the differentiated inputs.",
    ),
    )
end

@noinline function _throw_active_factorization(@nospecialize(fact))
    throw(
        ArgumentError(
        "Bramble's Enzyme rules for a kept factorization treat it as a constant, but this " *
        "$(nameof(typeof(fact))) has a shadow, and its derivative would be silently " *
        "dropped. Differentiate `pde_solve(A, F)` on the matrix instead, or build the " *
        "factorization outside the differentiated function.",
    ),
    )
end

# A `Bramble` factorization is a constant to Enzyme: its type depends on `sym` at run time, so
# `sparse_factorize` returns a `Union`, which forward mode cannot give a shadow.
EnzymeRules.inactive_type(::Type{<:Bramble.SuiteSparseFactorization}) = true
EnzymeRules.inactive_type(::Type{<:Bramble.AccelerateFactorization}) = true
EnzymeRules.inactive_type(::Type{<:Bramble.MUMPSFactorization}) = true
EnzymeRules.inactive_type(::Type{<:Bramble.SparspakFactorization}) = true

# The factorization the adjoint solve runs against, `nothing` when there is none. A
# `Bramble` SuiteSparse factorization holds the `UmfpackLU` or `CHOLMOD.Factor` that
# `fact'` applies to; the other backends have no adjoint solve.
_adjoint_factor(fact::Factorization) = fact
_adjoint_factor(fact::Bramble.SuiteSparseFactorization) = fact.fact
_adjoint_factor(::Bramble.AccelerateFactorization) = nothing
_adjoint_factor(::Bramble.MUMPSFactorization) = nothing
_adjoint_factor(::Bramble.SparspakFactorization) = nothing

_adjoint_solve(fact, ū) = _adjoint_factor(fact)' \ ū

# Whether `x` carries a shadow that is not the primal itself (see `_rank_one_tangent`).
_has_shadow(::Const) = false
_has_shadow(x::Annotation) = x.dval !== x.val

# Whether any lane's shadow of `A`, a matrix or a factorization, is not `A` itself.
_active_shadow(::Const, _) = false
_active_shadow(A::Annotation, W) = any(i -> _lane(A, i) !== A.val, 1:W)

# What a forward rule returns, from the primal and the shadow it computed.
# `RT`, the annotation Enzyme expects back, builds the pair: a value of a `Union` type must
# come back as that `Union`'s annotation.
function _forward_result(config::FwdConfig, ::Type{RT}, primal, shadow) where {RT}
    needs_primal = EnzymeRules.needs_primal(config)
    EnzymeRules.needs_shadow(config) || return needs_primal ? primal : nothing
    needs_primal || return shadow
    return RT(primal, shadow)
end

# `fact \ Ḟ` for one lane, zero when the load has no tangent.
_load_tangent(fact, ::Nothing, u) = zero(u)
_load_tangent(fact, Ḟ, _) = fact \ Ḟ

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        fact::Annotation{<:Factorization},
        F::Annotation{<:AbstractVector}
) where {RT}
    _active_shadow(fact, EnzymeRules.width(config)) && _throw_active_factorization(fact.val)
    u = pde_solve(fact.val, F.val)
    RT <: Const && return EnzymeRules.needs_primal(config) ? u : nothing
    W = EnzymeRules.width(config)
    shadows = ntuple(Val(W)) do i
        return _load_tangent(fact.val, _rank_one_tangent(F.val, _lane(F, i)), u)
    end
    return _forward_result(config, RT, u, W == 1 ? only(shadows) : shadows)
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        fact::Annotation{<:Factorization},
        F::Annotation{<:AbstractVector}
) where {RT}
    EnzymeRules.width(config) == 1 || _throw_batched(EnzymeRules.width(config))
    _active_shadow(fact, EnzymeRules.width(config)) && _throw_active_factorization(fact.val)
    _has_shadow(F) && _adjoint_factor(fact.val) === nothing &&
        _throw_adjoint_unavailable(fact.val)

    u = pde_solve(fact.val, F.val)
    ū = zero(u)

    primal = EnzymeRules.needs_primal(config) ? u : nothing
    shadow = EnzymeRules.needs_shadow(config) ? ū : nothing
    return AugmentedReturn(primal, shadow, (fact.val, ū))
end

function EnzymeRules.reverse(
        config::RevConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        tape,
        fact::Annotation{<:Factorization},
        F::Annotation{<:AbstractVector}
) where {RT}
    fact₀, ū = tape
    _has_shadow(F) && (F.dval .+= _adjoint_solve(fact₀, ū))

    # A custom rule consumes the return cotangent it was handed (see the rule on a matrix).
    fill!(ū, zero(eltype(ū)))

    return (nothing, nothing)
end

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(ldiv!)},
        ::Type{RT},
        x::Annotation{<:Bramble.VectorElement},
        fact::Annotation{<:Factorization},
        F::Annotation{<:AbstractVector}
) where {RT}
    _active_shadow(fact, EnzymeRules.width(config)) && _throw_active_factorization(fact.val)
    ldiv!(x.val, fact.val, F.val)
    # The destination's tangents are written before any return: a `Const` return still
    # leaves the caller reading `ẋ`, which would otherwise hold whatever it held before.
    for i in 1:EnzymeRules.width(config)
        ẋ = _rank_one_tangent(x.val, _lane(x, i))
        ẋ === nothing && continue
        Ḟ = _rank_one_tangent(F.val, _lane(F, i))
        copyto!(parent(ẋ), _load_tangent(fact.val, Ḟ, parent(ẋ)))
    end
    RT <: Const && return EnzymeRules.needs_primal(config) ? x.val : nothing
    return _forward_result(config, RT, x.val, x.dval)
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        ::Const{typeof(ldiv!)},
        ::Type{RT},
        x::Annotation{<:Bramble.VectorElement},
        fact::Annotation{<:Factorization},
        F::Annotation{<:AbstractVector}
) where {RT}
    EnzymeRules.width(config) == 1 || _throw_batched(EnzymeRules.width(config))
    _active_shadow(fact, EnzymeRules.width(config)) && _throw_active_factorization(fact.val)
    _has_shadow(x) && _has_shadow(F) && _adjoint_factor(fact.val) === nothing &&
        _throw_adjoint_unavailable(fact.val)

    ldiv!(x.val, fact.val, F.val)

    primal = EnzymeRules.needs_primal(config) ? x.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? x.dval : nothing
    return AugmentedReturn(primal, shadow, nothing)
end

# The destination is overwritten, so its cotangent before the call is zero: the cotangent
# after the call is read into `F̄` and cleared. It is cleared before `F̄` is accumulated, since
# `ldiv!(uₕ, fact, parent(uₕ))` shares one shadow between the two. A destination whose
# shadow is itself is a constant and is left alone.
function EnzymeRules.reverse(
        config::RevConfig,
        ::Const{typeof(ldiv!)},
        ::Type{RT},
        tape,
        x::Annotation{<:Bramble.VectorElement},
        fact::Annotation{<:Factorization},
        F::Annotation{<:AbstractVector}
) where {RT}
    if _has_shadow(x)
        ū = parent(x.dval)
        λ = _has_shadow(F) ? _adjoint_solve(fact.val, ū) : nothing
        fill!(ū, zero(eltype(ū)))
        λ === nothing || (F.dval .+= λ)
    end
    return (nothing, nothing, nothing)
end

# `sparse_factorize` is kept out of Enzyme. A matrix without a shadow gives a factorization
# with the matrix's activity, none, and its shadow, if one is asked for, is the factorization
# itself, as runtime activity gives a constant. A matrix with a shadow is refused: nothing
# here carries its derivative, and dropping it would return a gradient that misses it.
function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(Bramble.sparse_factorize)},
        ::Type{RT},
        A::Annotation{<:SparseMatrixCSC};
        kwargs...
) where {RT}
    _active_shadow(A, EnzymeRules.width(config)) && _throw_active_factorize()
    fact = Bramble.sparse_factorize(A.val; kwargs...)
    RT <: Const && return EnzymeRules.needs_primal(config) ? fact : nothing
    W = EnzymeRules.width(config)
    return _forward_result(config, RT, fact, W == 1 ? fact : ntuple(_ -> fact, Val(W)))
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        ::Const{typeof(Bramble.sparse_factorize)},
        ::Type{RT},
        A::Annotation{<:SparseMatrixCSC};
        kwargs...
) where {RT}
    _active_shadow(A, EnzymeRules.width(config)) && _throw_active_factorize()
    fact = Bramble.sparse_factorize(A.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? fact : nothing
    shadow = EnzymeRules.needs_shadow(config) ? fact : nothing
    # The factorization's type depends on `sym` at run time, so `RT` holds a `Union` and the
    # return must be built from it.
    return EnzymeRules.augmented_rule_return_type(config, RT)(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config::RevConfig,
        ::Const{typeof(Bramble.sparse_factorize)},
        ::Type{RT},
        tape,
        A::Annotation{<:SparseMatrixCSC};
        kwargs...
) where {RT}
    return (nothing,)
end

# The rules for `fdm_solve` in its right-hand side (#1167).
#
# Enzyme enters the operator build (`_kron_factor`) and the factorization's `Union` and stops
# with `IllegalTypeAnalysisException`, so the rules sit on `fdm_solve`, the call the user
# writes, and keep both out. `fdm_solve(a, F)` is linear in `F`: each lane's tangent is
# `u̇ = fdm_solve(a, Ḟ)`, and the pullback `F̄ += A⁻ᵀ ū`. The symmetric route has `A⁻ᵀ = A⁻¹`,
# and with `dirichlet = :boundary` the boundary entries of the solution are zero whatever `F`,
# so `fdm_solve(a, ū)` is the adjoint, zero there too. The generalised Schur route (advection
# terms) is not symmetric, so reverse refuses it and forward needs no transpose. The form or
# operator is a constant. Its coefficients and mesh are not differentiated, and a shadow that
# is not the primal is refused, as `fdm_solve` over a `Dual` operator is.

# Enzyme looks a rule up by its arguments' types, never by the return annotation or a keyword's
# value. Inference proves `fdm_solve` only throws for a 1D form, a form on a composite space
# and a ghost form with `dirichlet`, and Enzyme stops in LLVM (`broken gc calling conv fix`)
# on a call to a function that only throws if a rule covers it: the user should get the named
# `ArgumentError`. So the rules cover the forms of a scalar space in 2D and 3D, which `fdm_solve`
# can solve. Without a rule Enzyme differentiates the call, which reports the refusal. A ghost
# form is left out: its refusal depends on a keyword's value, which no signature can name.
const _FDMOperand = Union{
    Bramble.BilinearForm{2, <:Bramble.ScalarGridSpace, <:Bramble.ScalarGridSpace},
    Bramble.BilinearForm{3, <:Bramble.ScalarGridSpace, <:Bramble.ScalarGridSpace}}

# A form built inside the differentiated function holds a fresh `_AssemblyCache`, whose fields
# are integers and a copy of the form's own AST, used only to recognise it again: no value
# `assemble` reads. Enzyme gives that fresh heap object a shadow, so a form of constant
# coefficients reaches a rule as an argument whose floats are constant and whose roots are not,
# and Enzyme stops with `CallingConventionMismatchError` before any rule runs. The cache is
# inactive so such a form is a `Const`. A coefficient that moves still gives the form a shadow
# through its AST.
EnzymeRules.inactive_type(::Type{<:Bramble._AssemblyCache}) = true

@noinline function _throw_active_operator()
    throw(
        ArgumentError(
        "fdm_solve: the form or operator has a shadow, and Bramble's Enzyme rule " *
        "differentiates `fdm_solve` in its right-hand side only, so the derivative with " *
        "respect to a coefficient or the mesh would be silently dropped; to differentiate " *
        "with respect to it, solve with `pde_solve(assemble(a, l)..., F)`",
    ),
    )
end

@noinline function _throw_schur_adjoint()
    throw(
        ArgumentError(
        "fdm_solve: Bramble's Enzyme rule does not differentiate in reverse mode a form " *
        "with a non-symmetric term (advection), whose solve takes the generalised Schur " *
        "route and has no adjoint here. Use forward mode, or differentiate " *
        "`pde_solve(assemble(a, l)..., F)`.",
    ),
    )
end

# The route `fdm_solve` takes is the factorization's type, internal to `BrambleKroneckerExt`.
_fdm_symmetric_route(f) = nameof(typeof(f)) !== :_SchurFactorization

# Forward mode is width 1. `fdm_solve` infers a `Vector` of unknown element type (the
# operator's `T` is not in the form's type), and Enzyme asserts in LLVM, aborting the process,
# when it builds a batched forward call to a rule whose return type is not concrete. A wider
# batch reaches the second method, which refuses by name.
@noinline function _throw_batched_forward(width)
    throw(
        ArgumentError(
        "Bramble's Enzyme rule for `fdm_solve` does not support batched (vector-mode) " *
        "forward differentiation: got width $(width). Differentiate one direction at a " *
        "time.",
    ),
    )
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        ::Const{typeof(Bramble.fdm_solve)},
        ::Type{RT},
        a::Annotation{<:_FDMOperand},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    _active_shadow(a, 1) && _throw_active_operator()
    u = Bramble.fdm_solve(a.val, F.val; kwargs...)
    RT <: Const && return EnzymeRules.needs_primal(config) ? u : nothing
    Ḟ = _rank_one_tangent(F.val, _lane(F, 1))
    u̇ = Ḟ === nothing ? zero(u) : Bramble.fdm_solve(a.val, Ḟ; kwargs...)
    return _forward_result(config, RT, u, u̇)
end

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(Bramble.fdm_solve)},
        ::Type,
        a::Annotation{<:_FDMOperand},
        F::Annotation{<:AbstractVector};
        kwargs...
)
    return _throw_batched_forward(EnzymeRules.width(config))
end

# The reverse sweep solves with the factorization the forward solve built, kept on the tape: a
# coefficient (a `Ref` set between two solves) or the mesh may have changed by then, and the
# pullback is the adjoint of the operator the solve used. The factorization infers a `Union`
# of two types, so the tape is opaque and the primal is solved into an array of its eltype.
function EnzymeRules.augmented_primal(
        config::RevConfig,
        ::Const{typeof(Bramble.fdm_solve)},
        ::Type{RT},
        a::Annotation{<:_FDMOperand},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    EnzymeRules.width(config) == 1 || _throw_batched(EnzymeRules.width(config))
    _active_shadow(a, 1) && _throw_active_operator()

    fact = Bramble.fdm_factorize(a.val; kwargs...)
    # The adjoint is the same solve only on the symmetric route.
    _has_shadow(F) && !_fdm_symmetric_route(fact) && _throw_schur_adjoint()
    u = Bramble.fdm_solve!(Vector{typeof(fact).parameters[1]}(undef, size(fact, 1)), fact,
        F.val)::Vector
    ū = zero(u)

    primal = EnzymeRules.needs_primal(config) ? u : nothing
    shadow = EnzymeRules.needs_shadow(config) ? ū : nothing
    # `fdm_solve` infers a `Vector` of unknown element type, so `RT` is not concrete and the
    # return must be built from it.
    return EnzymeRules.augmented_rule_return_type(config, RT)(primal, shadow, (fact, ū))
end

function EnzymeRules.reverse(
        config::RevConfig,
        ::Const{typeof(Bramble.fdm_solve)},
        ::Type{RT},
        tape,
        a::Annotation{<:_FDMOperand},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    fact, ū = tape
    # `λ` is read before `ū` is cleared, and `F̄` is a different array.
    λ = _has_shadow(F) ? Bramble.fdm_solve!(similar(ū), fact, ū) : nothing
    fill!(ū, zero(eltype(ū)))
    λ === nothing || (F.dval .+= λ)
    return (nothing, nothing)
end

# The rules for `gmg_solve` in its right-hand side and in the coefficients of its form (#1168).
#
# Enzyme stops at the smoother construction in `gmg_preconditioner` with
# `IllegalTypeAnalysisException`. The cycles write the hierarchy's buffers, so they are not
# inactive either; the rules sit on `gmg_solve` and differentiate the converged solve
# implicitly. `A(θ) u = b`, with `θ` what `build` captures, gives the tangent
# `u̇ = A⁻¹(ḃ − Ȧ u)` and the pullback `λ = A⁻ᴴ ū`, `b̄ += λ`, `θ̄ += −λᴴ (∂A/∂θ) u`. The adjoint
# solve is `gmg_solve` itself only when `A` is Hermitian, while `gmg_solve` accepts any `A`
# whose cycle contracts: reverse assembles the finest matrix and refuses any other by name.
# `Ȧ u` and `θ̄` come from a nested Enzyme call on the finest product alone, `_gmg_product`,
# so no cycle is differentiated. One preconditioner per rule call serves every lane, each
# solved to the caller's `tol`, and a zero load returns zero without a cycle. The mesh is a
# constant; a shadow of it that carries data is refused by name rather than dropped.
#
# No signature can name what `build` returns. A `build` whose form `gmg_preconditioner`
# refuses by type alone (one on composite spaces) makes `gmg_solve` infer `Union{}`, and
# Enzyme then stops in LLVM (`broken gc calling conv fix`) instead of reporting the
# `ArgumentError`.

@noinline function _throw_active_gmg(name)
    throw(
        ArgumentError(
        "gmg_solve: `$name` has a shadow, so it may carry a derivative, and Bramble's " *
        "Enzyme rule differentiates `gmg_solve` in its right-hand side and in the " *
        "coefficients `build` captures, not in the mesh. Build the mesh outside the " *
        "differentiated function and pass it in.",
    ),
    )
end

@noinline function _throw_active_ghost_gmg()
    throw(
        ArgumentError(
        "gmg_solve: Bramble's Enzyme rule does not differentiate a `build` on ghost spaces " *
        "in its coefficients. Build it outside the differentiated function if it is " *
        "constant, or differentiate `pde_solve(assemble(build(gridspace(Ωₕ))), b)`.",
    ),
    )
end

# Whether a shadow carries no derivative, walked with its primal. A heap object shared with
# the primal is runtime activity's mark of a constant; anything else is read: every
# floating-point number it holds, through fields, arrays of any element type and `Ref`s, must
# be zero. A closure capturing a constant scalar and a `Ref` reaches forward mode with a shadow
# closure whose scalar is zero and whose `Ref` is the primal's. For a bits value `===`
# compares the data, so a tangent equal to its value is read, not taken for the primal.
_shadow_is_zero(x, ẋ) = _zero_shadow(x, ẋ, Base.IdSet{Any}())

# Whether a value of type `T` can hold a floating-point number.
_holds_floats(::Type{T}) where {T} = T <: Union{AbstractFloat, Complex{<:AbstractFloat}} ||
                                     !isbitstype(T) || any(_holds_floats, fieldtypes(T))

_zero_shadow(_, ẋ::Union{AbstractFloat, Complex{<:AbstractFloat}}, _) = iszero(ẋ)
_zero_shadow(_, ::Union{Symbol, AbstractString, Module, Type}, _) = true
# A number that cannot hold a float (an integer, an irrational) is zero; any other, such as a
# user's `Real` wrapping a `Float64`, is read field by field.
_zero_shadow(x, ẋ::Number, seen) = !_holds_floats(typeof(ẋ)) || _zero_fields(x, ẋ, seen)
function _zero_shadow(x, ẋ::DenseArray, seen)
    ẋ === x && return true
    _holds_floats(eltype(ẋ)) || return true
    (ẋ in seen || (push!(seen, ẋ); false)) && return true
    paired = x isa DenseArray && size(x) == size(ẋ)
    return all(eachindex(ẋ)) do i
        isassigned(ẋ, i) || return true
        xᵢ = paired && isassigned(x, i) ? x[i] : nothing
        return _zero_shadow(xᵢ, ẋ[i], seen)
    end
end
_zero_shadow(x, ẋ, seen) = _zero_fields(x, ẋ, seen)
function _zero_fields(x, ẋ::T, seen) where {T}
    !isbits(ẋ) && ẋ === x && return true
    ismutable(ẋ) && (ẋ in seen || (push!(seen, ẋ); false)) && return true
    paired = typeof(x) === T
    return all(1:fieldcount(T)) do i
        isdefined(ẋ, i) || return true
        xᵢ = paired && isdefined(x, i) ? getfield(x, i) : nothing
        return _zero_shadow(xᵢ, getfield(ẋ, i), seen)
    end
end

# Whether any lane of a forward rule's argument carries a derivative.
_moving_shadow(::Const, _) = false
_moving_shadow(x::Annotation, W) = any(i -> !_shadow_is_zero(x.val, _lane(x, i)), 1:W)

@noinline function _throw_gmg_not_hermitian(gap)
    throw(
        ArgumentError(
        "gmg_solve: Bramble's Enzyme rule differentiates in reverse mode only a form whose " *
        "matrix is symmetric (Hermitian), whose adjoint solve is `gmg_solve` itself; this " *
        "form's is not (‖A − Aᴴ‖∞ / ‖A‖∞ = $gap). Use forward mode, or differentiate " *
        "`pde_solve(assemble(build(gridspace(Ωₕ))), b)`.",
    ),
    )
end

@noinline function _throw_gmg_changed(change)
    throw(
        ArgumentError(
        "gmg_solve: the operator of `build`'s form changed between the solve and Bramble's " *
        "Enzyme reverse pass (relative change $change), as a `Ref` coefficient set after " *
        "the solve makes it: the multigrid operator reads it at each product, so the " *
        "adjoint solve would use the new value. Differentiate " *
        "`pde_solve(assemble(build(gridspace(Ωₕ))), b)`, whose matrix is assembled at the " *
        "solve, or build the form from the `Ref`'s value (`γ[]`).",
    ),
    )
end

# The keywords set the hierarchy, the cycle and the tolerance, none a value the solution is
# differentiated in. A `tol` held in a variable would otherwise reach the rules as active.
EnzymeRules.inactive_kwarg(::typeof(Bramble.gmg_solve), build, Ωₕ, b; kwargs...) = nothing

# The preconditioner and the loop's arguments, with `gmg_solve`'s own keywords and checks.
function _gmg_rule_setup(build, Ωₕ; levels = nothing, cycle::Symbol = :V, ν₁::Integer = 2,
        ν₂::Integer = 2, smoother = Bramble._gmg_default_smoother, tol = nothing,
        maxiters::Integer = 100)
    (tol === nothing || (tol isa Real && tol > 0)) ||
        throw(ArgumentError("gmg_solve needs tol > 0, got $tol"))
    maxiters >= 1 || throw(ArgumentError("gmg_solve needs maxiters >= 1, got $maxiters"))
    P = Bramble.gmg_preconditioner(build, Ωₕ; levels, cycle, ν₁, ν₂, smoother)
    return P, (cycle, tol, Int(maxiters))
end

_gmg_rule_solve(P, b, (cycle, tol, maxiters)) = Bramble._gmg_solve_loop(P, b, cycle, tol,
    maxiters)

# A zero tangent of the solution's type.
_gmg_zero(u) = (z = similar(u); fill!(parent(z), zero(eltype(z))); z)

# The finest product `A(θ) x`, the one call the nested Enzyme calls differentiate. The space
# is made outside it and passed `Const`: built inside, its weights get shadows that forward
# mode leaves unwritten, and the tangent reads them (NaN or 1e277 on a 9×9 mesh).
function _gmg_product(build, W, x)
    op = Bramble.matrix_free_operator(build(W))
    y = zeros(eltype(op), size(op, 1))
    mul!(y, op, x)
    return y
end

# The nested calls on `_gmg_product`.
function _gmg_product_tangent(runtime, build, Wₕ, x)
    mode = Enzyme.set_runtime_activity(Enzyme.Forward, runtime)
    RT = build isa Union{BatchDuplicated, BatchDuplicatedNoNeed} ? BatchDuplicated : Duplicated
    return only(Enzyme.autodiff(mode, Const(_gmg_product), RT, build, Const(Wₕ), Const(x)))
end

# The forward half of the split call, at the solve: its tape holds what the reverse half
# differentiates.
function _gmg_product_split(runtime, build, Wₕ, x)
    mode = Enzyme.set_runtime_activity(Enzyme.ReverseSplitWithPrimal, runtime)
    fwd, rev = Enzyme.autodiff_thunk(mode, Const{typeof(_gmg_product)}, Duplicated,
        typeof(build), Const{typeof(Wₕ)}, Const{typeof(x)})
    tape, _, ȳ = fwd(Const(_gmg_product), build, Const(Wₕ), Const(x))
    return (rev, tape, ȳ, Wₕ, x)
end

# The reverse half, seeded with the product's cotangent `−λ`; returns `build`'s cotangent
# when it is `Active`.
function _gmg_product_pullback((rev, tape, ȳ, Wₕ, x), build, λ)
    ȳ .= .-λ
    return first(first(rev(Const(_gmg_product), build, Const(Wₕ), Const(x), tape)))
end

# Whether `build` may carry a derivative in reverse mode. A scalar it captures arrives
# `Active` or in a `MixedDuplicated` box, so a `Duplicated` shadow that is the primal is
# runtime activity's mark of a constant. Forward mode runs the nested call for any shadow:
# `===` compares a bits field by its data, so a tangent equal to its value would pass for
# the primal, and a zero shadow gives a zero product.
_gmg_build_moves(::Const) = false
_gmg_build_moves(::Active) = true
_gmg_build_moves(build::Annotation) = _active_shadow(build, 1)

# The nested call runs on the forms `gmg_solve` builds from `gridspace(Ωₕ)` alone.
function _gmg_build_check(P)
    last(P.ops) isa Bramble.MatrixFreeOperator || _throw_active_ghost_gmg()
    return nothing
end

# The form the finest operator applies: on ghost spaces, the padded one.
_gmg_op_form(op::Bramble.MatrixFreeOperator) = op.form
_gmg_op_form(op::Bramble.GhostMatrixFreeOperator) = op.padded.form

# Refuses a finest matrix that is not Hermitian. It is assembled once, as `build` makes it on
# `Ωₕ`: a probe of the matrix-free operator can be cancelled by a tuned coefficient, and its
# scale grows with the mesh. A symmetric form assembles to an exactly symmetric matrix (0 on
# every test form measured), so the bound is a few roundings. Returns the matrix of the form
# the finest operator applies, against which the reverse pass checks that the operator is
# still the one the solve used.
function _gmg_hermitian_check(P, build, Ωₕ)
    op = last(P.ops)
    Aₒ = Bramble.assemble(_gmg_op_form(op))
    A = op isa Bramble.MatrixFreeOperator ? Aₒ : Bramble.assemble(build(Bramble.gridspace(Ωₕ)))
    gap = norm(A - A', Inf) / norm(A, Inf)
    gap <= 8 * eps(real(eltype(A))) || _throw_gmg_not_hermitian(gap)
    return Aₒ
end

# The operator's form is assembled again, not `build` called again: a `Ref` that `build`
# reads into a value is frozen in the taped form, and only one the form itself holds moves.
function _gmg_unchanged_check(P, Aₒ)
    A = Bramble.assemble(_gmg_op_form(last(P.ops)))
    change = size(A) == size(Aₒ) ? norm(A - Aₒ, Inf) / norm(Aₒ, Inf) : Inf
    change <= 8 * eps(real(eltype(Aₒ))) || _throw_gmg_changed(change)
    return nothing
end

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(Bramble.gmg_solve)},
        ::Type{RT},
        build::Annotation,
        Ωₕ::Annotation{<:Bramble.AbstractMeshType},
        b::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    W = EnzymeRules.width(config)
    _moving_shadow(Ωₕ, W) && _throw_active_gmg(:Ωₕ)
    P, loop = _gmg_rule_setup(build.val, Ωₕ.val; kwargs...)
    u = _gmg_rule_solve(P, b.val, loop)
    RT <: Const && return EnzymeRules.needs_primal(config) ? u : nothing
    # `Ȧ u` per lane, from the build as it is now, before any lane's solve.
    # On ghost spaces, which the nested call does not cover, a shadow whose data is zero is a
    # constant, as before the nested call.
    Ȧu = nothing
    if !(build isa Const) && last(P.ops) isa Bramble.MatrixFreeOperator
        Ȧu = _gmg_product_tangent(EnzymeRules.runtime_activity(config),
            build, Bramble.gridspace(Ωₕ.val), copy(parent(u)))
    elseif _moving_shadow(build, W)
        _throw_active_ghost_gmg()
    end
    shadows = ntuple(Val(W)) do i
        ḃ = _rank_one_tangent(b.val, _lane(b, i))
        Ȧuᵢ = Ȧu === nothing ? nothing : (W == 1 ? Ȧu : Ȧu[i])
        rhs = Ȧuᵢ === nothing ? ḃ : (ḃ === nothing ? -Ȧuᵢ : Bramble._mf_data(ḃ) - Ȧuᵢ)
        return rhs === nothing ? _gmg_zero(u) : _gmg_rule_solve(P, rhs, loop)
    end
    return _forward_result(config, RT, u, W == 1 ? only(shadows) : shadows)
end

# The reverse sweep solves with the preconditioner the forward solve built, kept on the tape:
# `build` may read a `Ref` whose value has changed by then. A coefficient the form itself holds
# as a `Ref` is read at each product, so the tape cannot keep it; the reverse pass assembles
# the finest operator's form again and refuses it if its matrix has changed since the solve.
# For the same reason the nested call on `build` is split: its forward half runs here, at the
# solve, and the tape it keeps is what its reverse half differentiates.
function EnzymeRules.augmented_primal(
        config::RevConfig,
        ::Const{typeof(Bramble.gmg_solve)},
        ::Type{RT},
        build::Annotation,
        Ωₕ::Annotation{<:Bramble.AbstractMeshType},
        b::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    EnzymeRules.width(config) == 1 || _throw_batched(EnzymeRules.width(config))
    Ωₕ isa Active && _throw_active_gmg(:Ωₕ)
    _active_shadow(Ωₕ, 1) && _throw_active_gmg(:Ωₕ)
    P, loop = _gmg_rule_setup(build.val, Ωₕ.val; kwargs...)
    moves = _gmg_build_moves(build)
    moves && _gmg_build_check(P)
    probe = _has_shadow(b) || moves ? _gmg_hermitian_check(P, build.val, Ωₕ.val) : nothing
    u = _gmg_rule_solve(P, b.val, loop)
    ū = _gmg_zero(u)
    nested = moves ?
             _gmg_product_split(EnzymeRules.runtime_activity(config), build,
        Bramble.gridspace(Ωₕ.val), copy(parent(u))) : nothing

    primal = EnzymeRules.needs_primal(config) ? u : nothing
    shadow = EnzymeRules.needs_shadow(config) ? ū : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(primal, shadow,
        (P, loop, probe, ū, nested))
end

function EnzymeRules.reverse(
        config::RevConfig,
        ::Const{typeof(Bramble.gmg_solve)},
        ::Type{RT},
        tape,
        build::Annotation,
        Ωₕ::Annotation{<:Bramble.AbstractMeshType},
        b::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    P, loop, probe, ū, nested = tape
    λ = nothing
    if _has_shadow(b) || nested !== nothing
        _gmg_unchanged_check(P, probe)
        λ = parent(_gmg_rule_solve(P, parent(ū), loop))
    end
    # `λ` is read before `ū` is cleared, as `b̄` may share its storage.
    fill!(parent(ū), zero(eltype(ū)))
    _has_shadow(b) && (Bramble._mf_data(b.dval) .+= λ)
    nested === nothing && return (nothing, nothing, nothing)
    # `θ̄ += −λᴴ (∂A/∂θ) u`: the product's cotangent is `−λ`. An `Active` `build` gets its
    # cotangent back; Enzyme drops it for one that captures constants only.
    θ̄ = _gmg_product_pullback(nested, build, λ)
    return (build isa Active ? θ̄::typeof(build.val) : nothing, nothing, nothing)
end
