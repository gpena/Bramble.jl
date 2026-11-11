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
