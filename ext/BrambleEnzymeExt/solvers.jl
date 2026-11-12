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
