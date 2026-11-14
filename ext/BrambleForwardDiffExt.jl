module BrambleForwardDiffExt

using Bramble: Bramble
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra, Factorization, rank
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
    fact = Bramble._primal_factorize(_innermost(A), eltype(_innermost(F)), solver, sym, kwargs)
    return _run_and_finalize(fact) do fact
        # A rank-deficient least-squares `Dual` solve has a value that is not smooth in the
        # parameter, so the formula below would be silently wrong: refuse it.
        eltype(A) <: ForwardDiff.Dual && solver === :spqr &&
            _check_full_rank(fact, size(A, 2))
        _promoted_dual_solve(fact, A, F)
    end
end

# `A x = F` over the factor `fact` of the innermost values of `A`, with `A` and `F` promoted
# to their common `Dual` type first (either may be real).
function _promoted_dual_solve(fact, A::SparseMatrixCSC, F::AbstractVector)
    P = promote_type(eltype(A), eltype(F))
    An = eltype(A) === P ? A : SparseMatrixCSC{P}(A)
    return _dual_solve(fact, An, convert(Vector{P}, F))
end

# The solve of a load linear in its parts (`Bramble._solve_by_parts`): the value and each
# partial of `F` are solved by `solve` on their own, and a nested tag recurses. If the solve
# of the value returns tags other than those of the value type `V`, the operator carries a
# tag of its own (a factor of a `Dual` matrix). The parts cannot then be rewrapped in the
# right tag order, so `solve(F)` is taken whole. Only tags count, so a `Float32` operator,
# which returns `Float32` leaves for `Float64` ones, still splits the load.
function Bramble._solve_by_parts(
        solve::S, F::AbstractVector{ForwardDiff.Dual{T, V, N}}) where {S, T, V, N}
    x0 = Bramble._solve_by_parts(solve, ForwardDiff.value.(F))
    _tags(eltype(x0)) === _tags(V) || return solve(F)
    xk = ntuple(k -> Bramble._solve_by_parts(solve, ForwardDiff.partials.(F, k)), N)
    R = eltype(x0)
    return [ForwardDiff.Dual{T}(x0[i], ForwardDiff.Partials{N, R}(ntuple(k -> xk[k][i], N)))
            for i in eachindex(x0)]
end

# The tags of a (nested) `Dual` type, outermost first; none for any other type.
_tags(::Type{ForwardDiff.Dual{T, V, N}}) where {T, V, N} = (T, _tags(V)...)
_tags(::Type) = ()

# A kept factorization of a `Dual` matrix (`sparse_factorize(A)` with `A` a `Dual` matrix):
# `fact` factors the innermost values of `A` by the backend and keywords given, and every
# solve against it is the `Dual` solve of `pde_solve(A, F)`, over that one factor.
struct DualFactorization{P <: ForwardDiff.Dual, FA <: Factorization, M <: SparseMatrixCSC{P}} <:
       Factorization{P}
    fact::FA
    A::M
end

Base.size(f::DualFactorization) = size(f.A)
Base.size(f::DualFactorization, d::Integer) = size(f.A, d)

function Bramble.sparse_factorize(
        A::SparseMatrixCSC{<:ForwardDiff.Dual}; solver::Symbol = :default, sym = :auto, kwargs...
)
    if size(A, 1) != size(A, 2)
        throw(
            ArgumentError(
            "sparse_factorize of a ForwardDiff.Dual matrix needs a square A, got size " *
            "$(size(A)).",
        ),
        )
    end
    fact = Bramble.sparse_factorize(_innermost(A); solver = solver, sym = sym, kwargs...)
    return DualFactorization(fact, A)
end

Base.:\(f::DualFactorization, F::AbstractVector) = _promoted_dual_solve(f.fact, f.A, F)
Bramble.pde_solve(f::DualFactorization, F::AbstractVector) = f \ F
LinearAlgebra.ldiv!(x::AbstractVector, f::DualFactorization, F::AbstractVector) = copyto!(x, f \ F)

# Disambiguates Bramble's `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)` and the
# method above. The load is not split: the solve of a `Dual` matrix is not linear in it.
function LinearAlgebra.ldiv!(uₕ::Bramble.VectorElement, f::DualFactorization, F::AbstractVector)
    copyto!(parent(uₕ), f \ F)
    return uₕ
end

# The bordered solve of a `Dual` matrix or load (`pde_solve(A, F; mean_zero, gauge)`, or a
# `RankOneUpdate`). `B = [A C; Cᵀ 0]` carries a `Dual` only through `A`, since the border
# `C = [w Z]` and the rank-one term are constants. With `B₀` the bordered matrix of the
# innermost values, factored once by the route of its `Float64` solve, partial `k` of
# `[u; μ]` is `B₀⁻¹([Fₖ; 0] − Bₖ[u₀; μ₀])`. A `Dual` border has no such formula and is
# refused. So is a `RankOneUpdate` over a `Dual` matrix, whose `σ` takes the matrix's
# element type.
function Bramble._bordered_promoted(
        ::Type{S}, ::Type{T}, A, F, r, w, Z, solver, kwargs) where {S <: ForwardDiff.Dual, T}
    _check_constant_border(w, Z, r)
    n = size(A, 1)
    A0 = _innermost(A)
    T0 = Bramble._bordered_eltype(A0, r, w, Z)
    B0, bsym = Bramble._bordered_system(T0, A0, r, w, Z)
    S0 = float(promote_type(T0, eltype(_innermost(F))))
    fact = Bramble._bordered_factorize(B0, bsym, T0, S0, solver, kwargs)
    return _run_and_finalize(fact) do fact
        B, _ = Bramble._bordered_system(S, A, r, w, Z)
        b = zeros(S, size(B, 1))
        b[1:n] .= F
        x = _dual_solve(fact, B, b)
        return x[1:n]
    end
end

function _check_constant_border(w, Z, r)
    w !== nothing && eltype(w) <: ForwardDiff.Dual && _throw_dual_border("mean_zero = w")
    Z !== nothing && eltype(Z) <: ForwardDiff.Dual && _throw_dual_border("gauge = Z")
    r !== nothing && Bramble._rank_one_eltype(r) <: ForwardDiff.Dual &&
        _throw_dual_border("the rank-one term of a RankOneUpdate")
    return nothing
end

@noinline function _throw_dual_border(what)
    throw(ArgumentError("pde_solve of a ForwardDiff.Dual system takes $what as a constant, " *
                        "got a ForwardDiff.Dual one: the bordered derivative holds the " *
                        "border fixed."))
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
