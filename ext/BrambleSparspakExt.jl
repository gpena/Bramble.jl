module BrambleSparspakExt

using Bramble: Bramble, SparspakFactorization
using Sparspak: Sparspak, sparspaklu, sparspaklu!
using LinearAlgebra: LinearAlgebra, ldiv!
using SparseArrays: SparseArrays, SparseMatrixCSC

mutable struct ConcreteSparspakFactorization{T, LUT} <: SparspakFactorization{T}
    lu::LUT
    dim::Int
end

Base.size(fact::ConcreteSparspakFactorization) = (fact.dim, fact.dim)
Base.size(fact::ConcreteSparspakFactorization, d::Integer) = d <= 2 ? fact.dim : 1

function Bramble._sparspak_factorize(A::SparseMatrixCSC)
    m, n = size(A)
    m == n || throw(DimensionMismatch("Matrix must be square for sparse direct solve, got $(m)×$(n)"))

    lu = sparspaklu(A)
    return ConcreteSparspakFactorization{eltype(A), typeof(lu)}(lu, n)
end

function LinearAlgebra.ldiv!(
        x::AbstractVector, fact::ConcreteSparspakFactorization, b::AbstractVector
)
    n = fact.dim
    length(b) == n || throw(
        DimensionMismatch("Right-hand side length $(length(b)) does not match matrix dimension $n"),
    )
    length(x) == n || throw(
        DimensionMismatch("Output vector length $(length(x)) does not match matrix dimension $n"),
    )
    ldiv!(x, fact.lu, b)
    return x
end

function LinearAlgebra.ldiv!(
        fact::ConcreteSparspakFactorization, b::AbstractVector
)
    n = fact.dim
    length(b) == n || throw(
        DimensionMismatch("Vector length $(length(b)) does not match matrix dimension $n"),
    )
    ldiv!(fact.lu, b)
    return b
end

function Base.:\(fact::ConcreteSparspakFactorization, b::AbstractVector)
    return fact.lu \ b
end

function Bramble._sparspak_refactor!(fact::ConcreteSparspakFactorization, A::SparseMatrixCSC)
    m, n = size(A)
    (m == fact.dim && n == fact.dim) || throw(
        DimensionMismatch("Matrix size $(m)×$(n) does not match factorization dimension $(fact.dim)"),
    )
    sparspaklu!(fact.lu, A; allow_pattern_change = false)
    return fact
end

function Bramble._sparspak_solve(A::SparseMatrixCSC, F::AbstractVector)
    fact = Bramble._sparspak_factorize(A)
    return fact \ F
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{SparspakFactorization})
    return Bramble._sparspak_factorize(A)
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{<:SparspakFactorization})
    return Bramble._sparspak_factorize(A)
end

end # module
