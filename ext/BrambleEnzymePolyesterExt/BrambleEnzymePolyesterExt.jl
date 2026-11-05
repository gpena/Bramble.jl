# ext/BrambleEnzymePolyesterExt/BrambleEnzymePolyesterExt.jl: Enzyme rules for the
# `CpuPolyester` hooks (gpena/Bramble.jl#369).
#
# Enzyme cannot differentiate a `Polyester.@batch` loop: every `_batch_*` hook that
# `BramblePolyesterExt` fills throws `IllegalTypeAnalysisException` in every mode
# (docs/notes/internals/enzyme_policies.md). This extension gives the hooks `EnzymeRules`
# rules whose derivative loops run under `@batch` themselves. No rule falls back to the
# serial engine or to a `CpuSerial` path.
#
# One file per hook group:
#
# - `difference.jl`: `_batch_difference_engine!`, a fixed kernel. Forward runs the engine
#   on the tangents; reverse runs the gather-form `_difference_band_adjoint!`
#   (src/operators/difference.jl).
# - `sweeps.jl`: `_batch_for!`, a closure kernel. Each `@batch` task runs an Enzyme
#   derivative of its slab, through a thunk built on the host.
# - `engines.jl`: the average engines (`_batch_average_engine!`,
#   `_batch_centered_average_engine!`), fixed kernels with gather-form adjoints, and
#   `_batch_run_bands!`, a closure kernel on the route of `sweeps.jl`.
# - `reductions.jl`: the inner products `_batch_dot` and `_batch_dot_masked`.
# - `assembly.jl`: the bilinear replays and linear sweeps of assembly
#   (`_batch_bilinear_band_replay!`, `_batch_bilinear_colour_replay!`,
#   `_batch_linear_band_sweep!`, `_batch_linear_colour_sweep!`) with a replay target.
# - `operators.jl`: the operator products, `_batch_mf_bands!` and the two bilinear replays
#   with a matrix-free action target, `_batch_kron_lines!` and `_batch_csr_spmv!`.
#
# A rule's argument convention follows what Enzyme 0.13 accepts at a custom rule: arrays and
# scalars as separate arguments, never a by-value struct mixing the two. A closure kernel
# reaches `_batch_for!` behind a `Ref` for that reason (`_sweep_for!`), as do a replay or
# action target, a matrix-free plan and the Kronecker terms at their hooks.
module BrambleEnzymePolyesterExt

using Bramble: Bramble, StarSpacings, _band_range, _difference_band_adjoint!,
               _difference_spacing_adjoint!, _difference_band_spacing_tangent!,
               _star_spacings_adjoint!
using Enzyme: Enzyme, Annotation, Const, Active, Duplicated, DuplicatedNoNeed,
              BatchDuplicated, BatchDuplicatedNoNeed, MixedDuplicated,
              BatchMixedDuplicated
using Enzyme.EnzymeRules: EnzymeRules, RevConfig, FwdConfig, AugmentedReturn
using LinearAlgebra: Diagonal
using Polyester: @batch
using SparseArrays: SparseMatrixCSC, nonzeros, sparse

# The shadow of `x` in lane `w`, or `nothing` where `x` is inactive: a `Const`, or under
# runtime activity a shadow that is the primal itself, Enzyme's mark of inactive data.
@inline _shadow(x::Const, w::Int) = nothing
@inline function _shadow(x::Union{Duplicated, DuplicatedNoNeed}, w::Int)
    return x.dval === x.val ? nothing : x.dval
end
@inline function _shadow(x::Union{BatchDuplicated, BatchDuplicatedNoNeed}, w::Int)
    d = x.dval[w]
    return d === x.val ? nothing : d
end

# The lanes of an argument's shadow, one per `EnzymeRules.width(config)`, `nothing` in a
# lane where it is inactive.
@inline _lanes(x, ::Val{W}) where {W} = ntuple(w -> _shadow(x, w), Val(W))

# How many times each hook's rule has been entered (forward, or the augmented primal of
# reverse). test/ext/enzyme_policy_cases.jl reads it to tell that the rule ran, not a route
# around it. Rules run on the calling task, never inside a `@batch` task.
const _ENTRIES = Dict{Symbol, Int}()
const _ENTRIES_LOCK = ReentrantLock()
function _entered!(hook::Symbol)
    @lock _ENTRIES_LOCK _ENTRIES[hook] = get(_ENTRIES, hook, 0) + 1
    return nothing
end
_rule_entries(hook::Symbol) = @lock _ENTRIES_LOCK get(_ENTRIES, hook, 0)

include("difference.jl")
include("engines.jl")
include("reductions.jl")
include("sweeps.jl")
include("assembly.jl")
include("operators.jl")

end
