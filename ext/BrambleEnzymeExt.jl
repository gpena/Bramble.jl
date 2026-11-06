module BrambleEnzymeExt

using Bramble: Bramble, pde_solve, CpuThreaded, KroneckerLinearOperator, StarSpacings,
               _band_range, _difference_band_adjoint!, _difference_spacing_adjoint!,
               _difference_band_spacing_tangent!, _star_spacings_adjoint!
using Enzyme: Enzyme, Const, Annotation, Active, Duplicated, DuplicatedNoNeed,
              BatchDuplicated, BatchDuplicatedNoNeed, MixedDuplicated, BatchMixedDuplicated
using Enzyme.EnzymeRules: EnzymeRules, RevConfig, FwdConfig, AugmentedReturn
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange
using LinearAlgebra: lu

# A native Enzyme reverse rule for `pde_solve`, used instead of `Enzyme.@import_rrule`.
#
# The maths is the adjoint `BrambleChainRulesExt`'s `ChainRulesCore.rrule` states: solve
# `Aᵀ λ = ∂J/∂u` once against the forward solve's own factorisation, then `∂J/∂A = -λuᵀ` on
# `A`'s stored pattern and `∂J/∂F = λ`. The two rules differ in how the cotangent reaches
# Enzyme.
#
# `@import_rrule`'s bridge takes the `SparseMatrixCSC` the rrule returns and merges it into
# Enzyme's own shadow of `A`. That merge drops the cotangent's explicit zeros from `nzval`
# but leaves `colptr` and `rowval`, so the shadow is no longer a well-formed
# `SparseMatrixCSC` and the gradient is wrong without any error. The cotangent has explicit
# zeros whenever some `u[j]` is exactly zero, which a homogeneous Dirichlet problem produces:
# a constrained row's solution entry is its boundary value, so a whole column of `-λuᵀ`
# vanishes. A gradient with respect to a diffusion coefficient is then wrong, while an
# `F`-only (Dirichlet-value) gradient stays correct.
#
# This rule never builds a cotangent object for the bridge to merge. It accumulates into
# `A.dval`'s own `nzval`, entry by entry, on the pattern Enzyme already allocated, so there
# is no explicit zero to drop. A caller on Enzyme must not also call `@import_rrule`: that
# defines a second rule for the same signature. The rrule serves every other
# `ChainRulesCore` consumer.

@noinline function _throw_batched(width)
    throw(
        ArgumentError(
        "Bramble's Enzyme rule for `pde_solve` does not support batched (vector-mode) " *
        "reverse differentiation: got width $(width). Differentiate one direction at a " *
        "time, or open an issue if batching is needed.",
    ),
    )
end

# The forward solve and the adjoint solve, dispatched on the keywords' type so each call
# site has one concrete tape type. Without keywords the forward solve's own factorisation
# serves the adjoint. With them the primal is `pde_solve(A, F; kwargs...)`, no factorisation
# is kept (`fact` is `nothing`), and the adjoint is `pde_solve(Aᴴ, ū; kwargs...)` with the
# border conjugated (`Bramble._pde_solve_adjoint`), as `BrambleChainRulesExt`'s rrule does.
function _forward_solve(A, F, ::@NamedTuple{})
    _entered!(:pde_solve_factor)
    fact = lu(A)
    return fact, fact \ F
end
_forward_solve(A, F, kwargs::NamedTuple) = nothing, pde_solve(A, F; kwargs...)

_adjoint_solve(fact, _, ū, ::@NamedTuple{}) = fact' \ ū
_adjoint_solve(::Nothing, A, ū, kwargs::NamedTuple) = Bramble._pde_solve_adjoint(A, ū; kwargs...)

function EnzymeRules.augmented_primal(
        config::RevConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        A::Annotation{<:SparseMatrixCSC},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    EnzymeRules.width(config) == 1 || _throw_batched(EnzymeRules.width(config))
    Bramble._check_square_for_adjoint(A.val)

    fact, u = _forward_solve(A.val, F.val, values(kwargs))
    # Without a factorisation the adjoint reads `A`'s values, so the forward matrix goes on the
    # tape: a caller may overwrite `A` after the solve, and `EnzymeRules.overwritten` does not
    # flag a matrix scaled in place after the call.
    Aₜ = fact === nothing ? copy(A.val) : nothing
    # The return's own cotangent accumulates here, and `reverse` reads it back off the tape:
    # allocated once, in the pass that knows `u`'s size, rather than reconstructed later.
    ū = zero(u)

    primal = EnzymeRules.needs_primal(config) ? u : nothing
    shadow = EnzymeRules.needs_shadow(config) ? ū : nothing
    return AugmentedReturn(primal, shadow, (fact, Aₜ, u, ū))
end

function EnzymeRules.reverse(
        config::RevConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        tape,
        A::Annotation{<:SparseMatrixCSC},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    fact, Aₜ, u, ū = tape
    A₀ = Aₜ === nothing ? A.val : Aₜ

    # `fact'` reuses the forward solve's own LU factors, exact whether or not `A` is
    # symmetric -- `dirichlet_bc!`'s `eₖ` row replacement makes even a symmetric form
    # structurally asymmetric, so this is not an optimisation for a special case.
    λ = _adjoint_solve(fact, A₀, ū, values(kwargs))

    if !(A isa Const)
        Āv = nonzeros(A.dval)
        rows = rowvals(A₀)
        @inbounds for j in axes(A₀, 2), k in nzrange(A₀, j)

            Āv[k] -= λ[rows[k]] * conj(u[j])
        end
    end

    if !(F isa Const)
        F.dval .+= λ
    end

    # Enzyme's contract: a custom rule consumes the return cotangent it was handed. Leaving
    # it in place would count it again if this rule runs more than once against the same tape.
    fill!(ū, zero(eltype(ū)))

    return (nothing, nothing)
end

# The forward rule: the primal solve is the reverse rule's (`_forward_solve`), and the tangent
# of `A u = F` is `A u̇ = Ḟ − Ȧ u`. Without keywords every tangent solves against the primal's
# own factorisation, so a call factors `A` once whatever the width. With keywords no
# factorisation is kept and each tangent is `pde_solve(A, Ḟ − Ȧ u; kwargs...)`: the bordered
# system `[A C; Cᵀ 0]` has a constant border, so its tangent is the bordered solve of a
# right-hand side with a zero border, and the keywords carry over unchanged. The rule is native,
# not `Enzyme.@import_frule` (the reverse rule avoids the import route too). Without it Enzyme
# differentiates UMFPACK's `lu` and stops on `umfpack_dl_defaults`.

# Lane `i` of the shadow of `x`: `nothing` for a `Const`.
_lane(::Const, ::Int) = nothing
_lane(x::Annotation, i::Int) = x.dval
_lane(x::Union{BatchDuplicated, BatchDuplicatedNoNeed}, i::Int) = x.dval[i]

# `Ḟ − Ȧ u` for one lane; a lane with neither a shadow of `A` nor of `F` has a zero tangent.
_tangent_rhs(::Nothing, ::Nothing, u) = zero(u)
_tangent_rhs(Ȧ, ::Nothing, u) = -(Ȧ * u)
_tangent_rhs(::Nothing, Ḟ, u) = Ḟ
_tangent_rhs(Ȧ, Ḟ, u) = Ḟ - Ȧ * u

_tangent_solve(fact, _, rhs, ::@NamedTuple{}) = fact \ rhs
_tangent_solve(::Nothing, A, rhs, kwargs::NamedTuple) = pde_solve(A, rhs; kwargs...)

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        A::Annotation{<:SparseMatrixCSC},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    Bramble._check_square_for_adjoint(A.val)
    fact, u = _forward_solve(A.val, F.val, values(kwargs))
    RT <: Const && return EnzymeRules.needs_primal(config) ? u : nothing
    W = EnzymeRules.width(config)
    shadows = ntuple(Val(W)) do i
        rhs = _tangent_rhs(_lane(A, i), _lane(F, i), u)
        return _tangent_solve(fact, A.val, rhs, values(kwargs))
    end
    needs_primal = EnzymeRules.needs_primal(config)
    needs_shadow = EnzymeRules.needs_shadow(config)
    needs_shadow || return needs_primal ? u : nothing
    shadow = W == 1 ? only(shadows) : shadows
    return needs_primal ?
           (W == 1 ? Duplicated(u, shadow) : BatchDuplicated(u, shadow)) : shadow
end

# The `CpuThreaded` engine entries' rules.
include("BrambleEnzymeExt/threaded.jl")

end
