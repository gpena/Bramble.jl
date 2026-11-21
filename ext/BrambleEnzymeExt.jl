module BrambleEnzymeExt

using Bramble: Bramble, pde_solve, CpuThreaded, KroneckerLinearOperator, StarSpacings,
               _band_range, _difference_band_adjoint!, _difference_spacing_adjoint!,
               _difference_band_spacing_tangent!, _star_spacings_adjoint!
using Enzyme: Enzyme, Const, Annotation, Active, Duplicated, DuplicatedNoNeed,
              BatchDuplicated, BatchDuplicatedNoNeed, MixedDuplicated, BatchMixedDuplicated
using Enzyme.EnzymeRules: EnzymeRules, RevConfig, FwdConfig, AugmentedReturn
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange
using LinearAlgebra: Factorization, dot, ldiv!, lu, mul!, norm

# A native Enzyme reverse rule for `pde_solve`, used instead of `Enzyme.@import_rrule`.
#
# The maths is the adjoint `BrambleChainRulesExt`'s `ChainRulesCore.rrule` states: solve
# `Aᴴ λ = ∂J/∂u` once against the forward solve's own factorisation, then `∂J/∂A = -λuᴴ` on
# `A`'s stored pattern and `∂J/∂F = λ`. The two rules differ in how the cotangent reaches
# Enzyme.
#
# `@import_rrule`'s bridge takes the `SparseMatrixCSC` the rrule returns and merges it into
# Enzyme's own shadow of `A`. That merge drops the cotangent's explicit zeros from `nzval`
# but leaves `colptr` and `rowval`, so the shadow is no longer a well-formed
# `SparseMatrixCSC` and the gradient is wrong without any error. The cotangent has explicit
# zeros whenever some `u[j]` is exactly zero, which a homogeneous Dirichlet problem produces:
# a constrained row's solution entry is its boundary value, so a whole column of `-λuᴴ`
# vanishes. A gradient with respect to a diffusion coefficient is then wrong, while an
# `F`-only (Dirichlet-value) gradient stays correct.
#
# This rule never builds a cotangent object for the bridge to merge. It accumulates into
# `A.dval`'s own `nzval`, entry by entry, on the pattern Enzyme already allocated, so there
# is no explicit zero to drop. A caller on Enzyme must not also call `@import_rrule`: that
# defines a second rule for the same signature. The rrule serves every other
# `ChainRulesCore` consumer.

# The one refusal of a batched (vector-mode) call, by the name of the function the user
# called: `mode` is "forward" or "reverse".
@noinline function _throw_batched(name, mode, width)
    throw(
        ArgumentError(
        "Bramble's Enzyme rule for `$(name)` does not support batched (vector-mode) " *
        "$(mode) differentiation: got width $(width). Differentiate one direction at a " *
        "time, or open an issue if batching is needed.",
    ),
    )
end

# The forward solve and the adjoint solve, dispatched on the keywords' type so each call
# site has one concrete tape type. Without keywords the forward solve's own factorisation
# serves the adjoint. With them the primal is `pde_solve(A, F; kwargs...)`, no factorisation
# is kept (`fact` is `nothing`), and the adjoint is `pde_solve(Aᴴ, ū; kwargs...)` with the
# border conjugated (`Bramble._pde_solve_adjoint`), as `BrambleChainRulesExt`'s rrule does.
# `fact \ F` can be wider than `pde_solve`'s result (`Float64` for `Float32` data), so the
# primal is converted to the type of `a \ f` for one entry of `A` and `F`, which is what
# `pde_solve` returns; the rrule's `_lu_rrule` converts the same way.
function _forward_solve(A, F, ::@NamedTuple{})
    _entered!(:pde_solve_factor)
    fact = lu(A)
    return fact, _solve_as(eltype(A), eltype(F), fact, F)
end
_forward_solve(A, F, kwargs::NamedTuple) = nothing, pde_solve(A, F; kwargs...)

function _solve_as(::Type{TA}, ::Type{TF}, fact, b) where {TA, TF}
    return convert(AbstractVector{typeof(oneunit(TA) \ oneunit(TF))}, fact \ b)
end

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
    EnzymeRules.width(config) == 1 || _throw_batched("pde_solve", "reverse", EnzymeRules.width(config))
    Bramble._check_square_for_adjoint(A.val)

    fact, u = _forward_solve(A.val, F.val, values(kwargs))
    # Without a factorisation the adjoint reads `A`'s values, so the forward matrix goes on the
    # tape: a caller may overwrite `A` after the solve, and `EnzymeRules.overwritten` does not
    # flag a matrix scaled in place after the call.
    Aₜ = fact === nothing ? copy(A.val) : nothing
    # The primal is `u` itself, which the caller may change in place after the solve, and
    # `reverse` reads the solution for `Ā -= λ uᴴ`: the tape keeps its own copy. With no
    # shadow of `A` (a `Const`, or a shadow that is `A` itself) the reverse pass never reads
    # it (`F̄ = λ`), so there the tape holds `u` itself and nothing is copied.
    uₜ = _has_shadow(A) ? copy(u) : u
    # The return's own cotangent accumulates here, and `reverse` reads it back off the tape:
    # allocated once, in the pass that knows `u`'s size, rather than reconstructed later.
    ū = zero(u)

    primal = EnzymeRules.needs_primal(config) ? u : nothing
    shadow = EnzymeRules.needs_shadow(config) ? ū : nothing
    return AugmentedReturn(primal, shadow, (fact, Aₜ, uₜ, ū))
end

# One entry of `-λ uᴴ` for a shadow of element type `T`. A real `A` (with a complex `F`, say)
# has a real shadow, which Enzyme allocated and the rule cannot widen; the derivative of a real
# entry is the real part of the complex one.
_entry_cotangent(::Type{<:Real}, x) = real(x)
_entry_cotangent(::Type, x) = x

function EnzymeRules.reverse(
        config::RevConfig,
        ::Const{typeof(pde_solve)},
        ::Type{RT},
        tape,
        A::Annotation{<:SparseMatrixCSC},
        F::Annotation{<:AbstractVector};
        kwargs...
) where {RT}
    fact, Aₜ, uₜ, ū = tape
    A₀ = Aₜ === nothing ? A.val : Aₜ

    # `fact'` reuses the forward solve's own LU factors, exact whether or not `A` is
    # symmetric -- `dirichlet_bc!`'s `eₖ` row replacement makes even a symmetric form
    # structurally asymmetric, so this is not an optimisation for a special case.
    λ = _adjoint_solve(fact, A₀, ū, values(kwargs))

    # Under runtime activity a constant argument can arrive as `Duplicated` with the primal
    # as its shadow (`_has_shadow`). Accumulating there would overwrite the caller's data.
    if _has_shadow(A)
        Āv = nonzeros(A.dval)
        rows = rowvals(A₀)
        @inbounds for j in axes(A₀, 2), k in nzrange(A₀, j)

            Āv[k] -= _entry_cotangent(eltype(Āv), λ[rows[k]] * conj(uₜ[j]))
        end
    end

    if _has_shadow(F)
        _accumulate_shadow!(F.val, F.dval, λ)
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

_tangent_solve(fact, A, F, rhs, ::@NamedTuple{}) = _solve_as(eltype(A), eltype(F), fact, rhs)
_tangent_solve(::Nothing, A, _, rhs, kwargs::NamedTuple) = pde_solve(A, rhs; kwargs...)

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
        # A lane whose shadow is the primal itself has no tangent (`_rank_one_tangent`).
        Ȧ = _rank_one_tangent(A.val, _lane(A, i))
        Ḟ = _rank_one_tangent(F.val, _lane(F, i))
        rhs = _tangent_rhs(Ȧ, Ḟ, u)
        return _tangent_solve(fact, A.val, F.val, rhs, values(kwargs))
    end
    needs_primal = EnzymeRules.needs_primal(config)
    needs_shadow = EnzymeRules.needs_shadow(config)
    needs_shadow || return needs_primal ? u : nothing
    shadow = W == 1 ? only(shadows) : shadows
    return needs_primal ?
           (W == 1 ? Duplicated(u, shadow) : BatchDuplicated(u, shadow)) : shadow
end

# A ghost mesh's refresh rebuilds the padded mesh's geometry and labels from the physical
# mesh. The rule drops any derivative that passes through a refresh (#1131); a ghost mesh
# built inside the differentiated function keeps its geometry derivatives. Without this
# rule, an assembly into a constant buffer is an inactive call whose nofree copies reach
# the marker warning's `@warn`, whose logger lookup calls `strcmp`, which Enzyme cannot
# copy (#1127).
EnzymeRules.inactive(::typeof(Bramble._refresh!), args...) = nothing

# A mesh `copy` deep-copies its markers, which are `BitVector`s and carry no derivative.
# Enzyme's own `deepcopy` rule has no forward method for a batch of width two or more,
# the width DifferentiationInterface uses for a gradient, and `polygon_spaces` copies the
# axes of the mesh inside the differentiated function (#1212). Only the marker copy is
# inactive: the points `copy` takes keep their derivatives.
EnzymeRules.inactive(::typeof(Bramble._copy_markers), args...) = nothing

# The `CpuThreaded` engine entries' rules.
include("BrambleEnzymeExt/threaded.jl")

# The rules for `constrained`'s relations: their forms, the ghost blocks' dense solves and
# the corner entries.
include("BrambleEnzymeExt/relations.jl")

# The rules for the solves beside `pde_solve(::SparseMatrixCSC, F)`.
include("BrambleEnzymeExt/solvers.jl")

# The rules that refuse a level-set mesh whose `φ` or box depends on the parameter.
include("BrambleEnzymeExt/geometry.jl")

end
