# ext/BrambleReverseDiffExt.jl: disambiguates `LinearAlgebra.mul!` between
# `KroneckerLinearOperator` (src/assembly/kronecker.jl) and `ReverseDiff.TrackedArray`
# (gpena/Bramble.jl#295).
#
# `KroneckerLinearOperator <: AbstractMatrix`, so it satisfies the unconstrained middle
# argument of `ReverseDiff`'s own `mul!(out::TrackedArray, x::AbstractMatrix, y::TrackedArray{V,
# D, 1})`, while a `TrackedVector` for both `y` and `x` satisfies Bramble's
# `mul!(y::AbstractVector, K::KroneckerLinearOperator, x::AbstractVector)` -- the classic
# diagonal clash where each method wins on a different argument and neither dominates. The
# method below is strictly more specific than both (`KroneckerLinearOperator` narrower than
# `AbstractMatrix`, `TrackedArray` narrower than `AbstractVector` on both `out` and `x`), so it
# resolves the ambiguity rather than adding a third contender.
#
# Only the three-argument form needs it: `ReverseDiff` never defines a five-argument `mul!`
# (`gpena/Bramble.jl#291`'s `mul!(y, K, x, α, β)`), so that one has nothing to clash with here.
#
# `ReverseDiff.record_mul!` -- the same function every `mul!(out::TrackedArray,
# x::AbstractMatrix, y::TrackedArray)` overload forwards to -- is what makes the reverse pass
# correct: the forward value is `K * value(x)` (via `*`, which `KroneckerLinearOperator`
# inherits from `AbstractMatrix` through the very `mul!` this disambiguates), and the pullback
# multiplies the incoming cotangent by `Kᵀ` (`K` is always symmetric, `issymmetric(K) ==
# true`, src/assembly/kronecker.jl) through `LinearAlgebra`'s generic `Transpose`-of-`AbstractMatrix`
# path -- correct, if not on the zero-allocation fast path `mul!(y, K, x; scratch)` gives an
# untracked caller.
module BrambleReverseDiffExt

using Bramble: Bramble, KroneckerLinearOperator
using ReverseDiff: ReverseDiff, TrackedArray
using LinearAlgebra: LinearAlgebra

function LinearAlgebra.mul!(
        out::TrackedArray, K::KroneckerLinearOperator, x::TrackedArray{V, D, 1}
) where {V, D}
    return ReverseDiff.record_mul!(out, K, x)
end

end # module BrambleReverseDiffExt
