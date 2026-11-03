module BrambleEnzymeExt

using Bramble: Bramble, pde_solve, CpuThreaded, KroneckerLinearOperator, StarSpacings,
               _band_range, _difference_band_adjoint!, _difference_spacing_adjoint!,
               _difference_band_spacing_tangent!, _star_spacings_adjoint!
using Enzyme: Enzyme, Const, Annotation, Active, Duplicated, DuplicatedNoNeed,
              BatchDuplicated, BatchDuplicatedNoNeed, MixedDuplicated, BatchMixedDuplicated
using Enzyme.EnzymeRules: EnzymeRules, RevConfig, FwdConfig, AugmentedReturn
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange, sparse
using LinearAlgebra: lu

# A native Enzyme reverse rule for `pde_solve`, used instead of `Enzyme.@import_rrule`
# (gpena/Bramble.jl#240).
#
# The maths is the same adjoint `BrambleChainRulesExt`'s `ChainRulesCore.rrule` states: solve
# `Aᵀ λ = ∂J/∂u` once against the forward solve's own factorisation, then `∂J/∂A = -λuᵀ` on
# `A`'s stored pattern and `∂J/∂F = λ`. What changes is *how the cotangent reaches Enzyme*,
# and that is the entire reason this file exists.
#
# `@import_rrule`'s bridge takes the `SparseMatrixCSC` the rrule *returns* and merges it into
# Enzyme's own shadow of `A`. That merge drops the cotangent's explicit zeros from `nzval`
# while leaving `colptr`/`rowval` untouched, so the shadow stops being a well-formed
# `SparseMatrixCSC`: measured directly on a 31-nonzero operator whose cotangent had 4 exact
# zeros, the shadow came back with `length(nzval) == 27` against `colptr[end] - 1 == 31`, and
# `Matrix(shadow)` then fails its own `_goodbuffers` assertion. Nothing errors during
# differentiation; the gradient is simply wrong, and wrong only when the cotangent happens to
# contain an exact zero -- which a homogeneous Dirichlet problem produces routinely (a
# constrained row's solution entry is exactly its boundary value, so `u[j] == 0` makes a whole
# column of `-λuᵀ` exactly zero). That is why an inverse problem recovering a *diffusion
# coefficient* returned a confidently wrong gradient while the same rule checked by hand, and
# every `F`-only (Dirichlet-value) gradient, stayed correct.
#
# This rule never builds a cotangent object for the bridge to merge. It accumulates straight
# into `A.dval`'s own `nzval`, entry by entry, on the pattern Enzyme already allocated -- so
# there is nothing to restructure and no explicit zero to drop.
#
# `BrambleChainRulesExt`'s rrule stays exactly as it was: it is still what `Zygote` and any
# other `ChainRulesCore` consumer uses, and it is still checked against finite differences on
# its own. A caller on Enzyme must *not* also call `@import_rrule` now -- that would define a
# second rule for this same signature.

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
# is kept (`fact` is `nothing`), and the adjoint is `pde_solve(Aᴴ, ū; kwargs...)`, as
# `BrambleChainRulesExt`'s rrule does.
function _forward_solve(A, F, ::@NamedTuple{})
    fact = lu(A)
    return fact, fact \ F
end
_forward_solve(A, F, kwargs::NamedTuple) = nothing, pde_solve(A, F; kwargs...)

_adjoint_solve(fact, _, ū, ::@NamedTuple{}) = fact' \ ū
_adjoint_solve(::Nothing, A, ū, kwargs::NamedTuple) = pde_solve(sparse(A'), ū; kwargs...)

function EnzymeRules.augmented_primal(
        config::RevConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        A::Annotation{<:SparseMatrixCSC},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    EnzymeRules.width(config) == 1 || _throw_batched(EnzymeRules.width(config))

    fact, u = _forward_solve(A.val, F.val, values(kwargs))
    # The return's own cotangent accumulates here, and `reverse` reads it back off the tape:
    # allocated once, in the pass that knows `u`'s size, rather than reconstructed later.
    ū = zero(u)

    primal = EnzymeRules.needs_primal(config) ? u : nothing
    shadow = EnzymeRules.needs_shadow(config) ? ū : nothing
    return AugmentedReturn(primal, shadow, (fact, u, ū))
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
    fact, u, ū = tape

    # `fact'` reuses the forward solve's own LU factors, exact whether or not `A` is
    # symmetric -- `dirichlet_bc!`'s `eₖ` row replacement makes even a symmetric form
    # structurally asymmetric, so this is not an optimisation for a special case.
    λ = _adjoint_solve(fact, A.val, ū, values(kwargs))

    if !(A isa Const)
        Āv = nonzeros(A.dval)
        rows = rowvals(A.val)
        @inbounds for j in axes(A.val, 2), k in nzrange(A.val, j)

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

# The `CpuThreaded` engine entries' rules (gpena/Bramble.jl#369).
include("BrambleEnzymeExt/threaded.jl")

end
