module BrambleForwardDiffExt

using Bramble: Bramble
using ForwardDiff: ForwardDiff
using SparseArrays: SparseArrays, SparseMatrixCSC

# A `ForwardDiff.Dual` is linear in its parts (value and partials), so a Sparspak solve may
# split a `Dual` load into one solve per part (src/solvers/sparspak_solver.jl).
Bramble._linear_in_parts(::Type{<:ForwardDiff.Dual}) = true

# The plain solve of a `Dual` matrix or load (`pde_solve(A, F)`, src/solvers/pde_solve.jl).
# The solution's value is `A₀⁻¹ b₀`, where `A₀` is the innermost `Float64` values of `A`,
# and its partial `k` is `A₀⁻¹(b_k − A_k x₀)`. So `A₀` is factored once, by the route of
# its `Float64` solve, and every partial is a solve against that factor.
function Bramble._solve_promoted(
        ::Type{<:ForwardDiff.Dual}, A::SparseMatrixCSC, F::AbstractVector;
        solver::Symbol, sym, kwargs...
)
    if eltype(A) <: ForwardDiff.Dual && size(A, 1) != size(A, 2)
        throw(
            ArgumentError(
            "pde_solve of a ForwardDiff.Dual matrix needs a square A, got size $(size(A)): the " *
            "derivative of a least-squares solve has a residual term that this one omits.",
        ),
        )
    end
    P = promote_type(eltype(A), eltype(F))
    fact = Bramble._primal_factorize(_innermost(A), eltype(_innermost(F)), solver, sym, kwargs)
    An = eltype(A) === P ? A : SparseMatrixCSC{P}(A)
    return _dual_solve(fact, An, convert(Vector{P}, F))
end

# The innermost values: `value` taken until the element type is no longer a `Dual`.
_innermost(x::AbstractArray{<:ForwardDiff.Dual}) = _innermost(ForwardDiff.value.(x))
_innermost(x) = x

# Solves `A x = b` for `A` and `b` of the same `Dual` type `P`, over the factor `fact` of the
# innermost values. A value level that is itself `Dual` (a nested tag) recurses through the
# same factor, so one factorization serves every order.
function _dual_solve(fact, A::AbstractMatrix{P}, b::AbstractVector{P}) where {P <: ForwardDiff.Dual}
    N = ForwardDiff.npartials(P)
    Av = ForwardDiff.value.(A)
    x0 = _level_solve(fact, Av, ForwardDiff.value.(b))
    xk = ntuple(N) do k
        rhs = ForwardDiff.partials.(b, k) .- ForwardDiff.partials.(A, k) * x0
        _level_solve(fact, Av, rhs)
    end
    return [P(x0[i], ForwardDiff.Partials(ntuple(k -> xk[k][i], N))) for i in eachindex(x0)]
end

_level_solve(fact, A::AbstractMatrix{<:ForwardDiff.Dual}, b::AbstractVector) = _dual_solve(fact, A, b)
_level_solve(fact, ::AbstractMatrix, b::AbstractVector) = fact \ b

end # module
