module BrambleSuiteSparseExt

using Bramble: Bramble, SuiteSparseFactorization, BilinearForm, LinearForm, assemble, element, trial_space
using SuiteSparse: SuiteSparse, CHOLMOD, UMFPACK
using LinearAlgebra: LinearAlgebra, Factorization, ldiv!, factorize, issymmetric, Symmetric, cholesky, cholesky!, lu,
                     lu!, diag
using SparseArrays: SparseArrays, SparseMatrixCSC

mutable struct ConcreteSuiteSparseFactorization{T, FType <: Factorization} <: SuiteSparseFactorization{T}
    fact::FType
    sym::Symbol
    dim::Int
end

Base.size(fact::ConcreteSuiteSparseFactorization) = (fact.dim, fact.dim)
Base.size(fact::ConcreteSuiteSparseFactorization, d::Integer) = d <= 2 ? fact.dim : 1

function _suitesparse_sym_flag(A::AbstractMatrix, sym)
    if sym === :auto
        if issymmetric(A)
            d = diag(A)
            return (!isempty(d) && all(x -> real(x) > 0, d)) ? :spd : :symmetric
        else
            return :unsymmetric
        end
    elseif sym === :spd || sym === :definite || sym == 1
        return :spd
    elseif sym === :symmetric || sym == 2
        return :symmetric
    elseif sym === :unsymmetric || sym == 0
        return :unsymmetric
    else
        throw(
            ArgumentError(
            "Unknown SuiteSparse symmetry option: $sym. Expected :auto, :spd, :symmetric, or :unsymmetric.",
        ),
        )
    end
end

function Bramble._suitesparse_factorize(A::SparseMatrixCSC; sym = :auto, kwargs...)
    m, n = size(A)
    m == n || throw(DimensionMismatch("Matrix must be square for sparse direct solve, got $(m)×$(n)"))

    sym_flag = _suitesparse_sym_flag(A, sym)

    fact_obj = if sym_flag === :spd
        cholesky(Symmetric(A); kwargs...)
    else
        lu(A; kwargs...)
    end

    return ConcreteSuiteSparseFactorization{eltype(A), typeof(fact_obj)}(fact_obj, sym_flag, n)
end

function LinearAlgebra.ldiv!(
        x::AbstractVector, fact::ConcreteSuiteSparseFactorization, b::AbstractVector
)
    n = fact.dim
    length(b) == n || throw(
        DimensionMismatch("Right-hand side length $(length(b)) does not match matrix dimension $n"),
    )
    length(x) == n || throw(
        DimensionMismatch("Output vector length $(length(x)) does not match matrix dimension $n"),
    )
    ldiv!(x, fact.fact, b)
    return x
end

function LinearAlgebra.ldiv!(
        fact::ConcreteSuiteSparseFactorization, b::AbstractVector
)
    n = fact.dim
    length(b) == n || throw(
        DimensionMismatch("Vector length $(length(b)) does not match matrix dimension $n"),
    )
    if fact.fact isa UMFPACK.UmfpackLU
        ldiv!(fact.fact, b)
    else
        x = similar(b)
        ldiv!(x, fact.fact, b)
        copyto!(b, x)
    end
    return b
end

function Base.:\(fact::ConcreteSuiteSparseFactorization, b::AbstractVector)
    return fact.fact \ b
end

function Bramble._suitesparse_refactor!(fact::ConcreteSuiteSparseFactorization, A::SparseMatrixCSC)
    m, n = size(A)
    (m == fact.dim && n == fact.dim) || throw(
        DimensionMismatch("Matrix size $(m)×$(n) does not match factorization dimension $(fact.dim)"),
    )

    if fact.fact isa CHOLMOD.Factor
        cholesky!(fact.fact, Symmetric(A))
    elseif fact.fact isa UMFPACK.UmfpackLU
        lu!(fact.fact, A)
    else
        throw(ArgumentError("Unsupported underlying factorization type: $(typeof(fact.fact))"))
    end
    return fact
end

function Bramble._suitesparse_solve(A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kwargs...)
    fact = Bramble._suitesparse_factorize(A; sym = sym, kwargs...)
    return fact \ F
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{SuiteSparseFactorization}; kwargs...)
    return Bramble._suitesparse_factorize(A; kwargs...)
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{<:SuiteSparseFactorization}; kwargs...)
    return Bramble._suitesparse_factorize(A; kwargs...)
end

end # module
