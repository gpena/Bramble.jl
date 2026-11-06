module BrambleForwardDiffExt

using Bramble: Bramble
using ForwardDiff: ForwardDiff
using LinearAlgebra: rank
using SparseArrays: SparseArrays, SparseMatrixCSC

# A `ForwardDiff.Dual` is linear in its parts (value and partials), so a Sparspak solve may
# split a `Dual` load into one solve per part (src/solvers/sparspak_solver.jl).
Bramble._linear_in_parts(::Type{<:ForwardDiff.Dual}) = true

# `(f(x), f′(x))` from one `Dual` evaluation of `f`, for `jacobian_form`'s refill
# (src/assembly/assembled_jacobian.jl). `f::F` so each coefficient function is specialised
# on. An `f` returning a plain number (a constant) has derivative zero.
@inline function Bramble._value_and_derivative(f::F, x::Real) where {F}
    T = typeof(ForwardDiff.Tag(f, typeof(x)))
    return _value_and_partial(f(ForwardDiff.Dual{T}(x, one(x))))
end

@inline _value_and_partial(y::ForwardDiff.Dual) = (ForwardDiff.value(y), ForwardDiff.partials(y, 1))
@inline _value_and_partial(y::Real) = (y, zero(y))

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
    return _run_and_finalize(fact) do fact
        # A rank-deficient least-squares `Dual` solve has a value that is not smooth in the
        # parameter, so the formula below would be silently wrong: refuse it.
        eltype(A) <: ForwardDiff.Dual && solver === :spqr &&
            _check_full_rank(fact, size(A, 2))
        An = eltype(A) === P ? A : SparseMatrixCSC{P}(A)
        _dual_solve(fact, An, convert(Vector{P}, F))
    end
end

# Runs `f(fact)` and then releases `fact` with `finalize`, also when `f` throws. A native
# factor (MUMPS) is freed here, as `_mumps_solve` frees its own, rather than when the GC
# reaches it. A factor without a finalizer is unaffected.
function _run_and_finalize(f, fact)
    try
        return f(fact)
    finally
        finalize(fact)
    end
end

function _check_full_rank(fact, n::Integer)
    r = rank(fact)
    r == n || throw(
        ArgumentError(
        "pde_solve of a ForwardDiff.Dual matrix with solver = :spqr needs a full-rank A, got " *
        "rank $r of $n: the forward derivative of a rank-deficient least-squares solve is not " *
        "the one this formula gives.",
    ),
    )
    return nothing
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
