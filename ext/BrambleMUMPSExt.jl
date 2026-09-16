module BrambleMUMPSExt

using Bramble: Bramble, MUMPSFactorization
using MUMPS:
             MUMPS,
             Mumps,
             associate_matrix!,
             associate_rhs!,
             factorize!,
             get_sol!,
             suppress_display!,
             set_icntl!,
             set_cntl!,
             finalize!
using LinearAlgebra: LinearAlgebra, ldiv!, issymmetric, diag
using SparseArrays: SparseArrays, SparseMatrixCSC

function _ensure_mpi_init()
    if isdefined(MUMPS, :MPI)
        if !MUMPS.MPI.Initialized()
            MUMPS.MPI.Init()
        end
    end
    return nothing
end

mutable struct ConcreteMUMPSFactorization{T, TR} <: MUMPSFactorization{T}
    mumps::Mumps{T, TR}
    sym::Int
    dim::Int
end

Base.size(fact::ConcreteMUMPSFactorization) = (fact.dim, fact.dim)
Base.size(fact::ConcreteMUMPSFactorization, d::Integer) = d <= 2 ? fact.dim : 1

Base.finalize(fact::ConcreteMUMPSFactorization) = finalize!(fact.mumps)

_mumps_val_type(::Type{T}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}} = T
_mumps_val_type(::Type{<:Integer}) = Float64
_mumps_val_type(::Type{<:Real}) = Float64
_mumps_val_type(::Type{<:Complex}) = ComplexF64

function _mumps_sym_flag(A::AbstractMatrix, sym)
    if sym === :auto
        if issymmetric(A)
            # Check if strictly positive diagonal for SPD heuristic
            d = diag(A)
            return (!isempty(d) && all(x -> real(x) > 0, d)) ? 1 : 2
        else
            return 0
        end
    elseif sym === :spd || sym === :definite || sym == 1
        return 1
    elseif sym === :symmetric || sym == 2
        return 2
    elseif sym === :unsymmetric || sym == 0
        return 0
    else
        throw(
            ArgumentError(
            "Unknown MUMPS symmetry option: $sym. Expected :auto, :spd, :symmetric, or :unsymmetric.",
        ),
        )
    end
end

function Bramble._mumps_factorize(
        A::SparseMatrixCSC; sym = :auto, icntl = nothing, cntl = nothing, kwargs...
)
    _ensure_mpi_init()

    sym_flag = _mumps_sym_flag(A, sym)
    Tv = _mumps_val_type(eltype(A))
    A_mat = convert(SparseMatrixCSC{Tv, Int}, A)
    m, n = size(A_mat)
    m == n || throw(DimensionMismatch("Matrix must be square for direct sparse solve, got $(m)×$(n)"))

    mumps = Mumps{Tv}(sym_flag)
    suppress_display!(mumps)

    if icntl !== nothing
        for (k, v) in icntl
            set_icntl!(mumps, Int(k), Int(v); displaylevel = 0)
        end
    end
    if cntl !== nothing
        for (k, v) in cntl
            set_cntl!(mumps, Int(k), Float64(v); displaylevel = 0)
        end
    end

    associate_matrix!(mumps, A_mat)
    factorize!(mumps)

    return ConcreteMUMPSFactorization(mumps, sym_flag, n)
end

function LinearAlgebra.ldiv!(
        x::AbstractVector, fact::ConcreteMUMPSFactorization{T}, b::AbstractVector
) where {T}
    n = fact.dim
    length(b) == n || throw(
        DimensionMismatch("Right-hand side length $(length(b)) does not match matrix dimension $n"),
    )
    length(x) == n || throw(
        DimensionMismatch("Output vector length $(length(x)) does not match matrix dimension $n"),
    )

    associate_rhs!(fact.mumps, b)
    fact.mumps.job = MUMPS.SOLVE
    MUMPS.invoke_mumps!(fact.mumps)
    get_sol!(x, fact.mumps)
    return x
end

function LinearAlgebra.ldiv!(
        fact::ConcreteMUMPSFactorization{T}, b::AbstractVector
) where {T}
    n = fact.dim
    length(b) == n || throw(
        DimensionMismatch("Vector length $(length(b)) does not match matrix dimension $n"),
    )

    associate_rhs!(fact.mumps, b)
    fact.mumps.job = MUMPS.SOLVE
    MUMPS.invoke_mumps!(fact.mumps)
    get_sol!(b, fact.mumps)
    return b
end

function Base.:\(fact::ConcreteMUMPSFactorization{T}, b::AbstractVector) where {T}
    x = similar(b, promote_type(T, eltype(b)))
    return ldiv!(x, fact, b)
end

function Bramble._mumps_solve(
        A::SparseMatrixCSC, F::AbstractVector; sym = :auto, icntl = nothing, cntl = nothing, kwargs...
)
    fact = Bramble._mumps_factorize(A; sym = sym, icntl = icntl, cntl = cntl, kwargs...)
    return fact \ F
end

function Bramble._mumps_refactor!(fact::ConcreteMUMPSFactorization{T}, A::SparseMatrixCSC) where {T}
    m, n = size(A)
    (m == fact.dim && n == fact.dim) || throw(
        DimensionMismatch("Matrix size $(m)×$(n) does not match factorization dimension $(fact.dim)"),
    )
    A_mat = convert(SparseMatrixCSC{T, Int}, A)
    associate_matrix!(fact.mumps, A_mat)
    fact.mumps.job = MUMPS.FACTOR
    MUMPS.invoke_mumps!(fact.mumps)
    return fact
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{MUMPSFactorization}; kwargs...)
    return Bramble._mumps_factorize(A; kwargs...)
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{<:MUMPSFactorization}; kwargs...)
    return Bramble._mumps_factorize(A; kwargs...)
end

end # module
