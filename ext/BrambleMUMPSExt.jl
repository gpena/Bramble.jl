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
using LinearAlgebra: LinearAlgebra, ldiv!, issymmetric
using SparseArrays: SparseArrays, SparseMatrixCSC

function _ensure_mpi_init()
    if isdefined(MUMPS, :MPI)
        if !MUMPS.MPI.Initialized()
            MUMPS.MPI.Init()
        end
    end
    return nothing
end

const MUMPSControls = Tuple{Vector{Pair{Int, Int}}, Vector{Pair{Int, Float64}}}

# `auto` holds the icntl and cntl of a `sym = :auto` request (`nothing` otherwise), so a
# refactorization can repeat the SYM = 1 to SYM = 2 fallback; `failure` holds the exception
# of a refused refactorization until one succeeds, and every solve rethrows it.
mutable struct ConcreteMUMPSFactorization{T, TR, P} <: MUMPSFactorization{T}
    mumps::Mumps{T, TR, P}
    sym::Int
    dim::Int
    auto::Union{Nothing, MUMPSControls}
    failure::Union{Nothing, Exception}
end

_controls(c, ::Type{V}) where {V} = c === nothing ? Pair{Int, V}[] : [Int(k) => V(v) for (k, v) in c]

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
            return 1
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

# The exception for a failed MUMPS call (`INFOG(1) < 0`): -10 is a numerically singular
# matrix, anything else a generic error naming `phase` and both INFOG codes.
function _mumps_error(mumps::Mumps, phase::AbstractString)
    code, info = mumps.infog[1], mumps.infog[2]
    code == -10 && return LinearAlgebra.SingularException(Int(info))
    return ErrorException("MUMPS $phase failed: INFOG(1) = $code, INFOG(2) = $info")
end

_wrong_spd_guess(mumps::Mumps) = mumps.infog[1] == -10 || mumps.infog[12] > 0

function Bramble._mumps_factorize(
        A::SparseMatrixCSC; sym = :auto, icntl = nothing, cntl = nothing
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
    # INFOG(1) = -10 is a zero pivot and INFOG(12) > 0 counts negative ones: under `:auto`
    # either shows the SYM = 1 guess was wrong, so factor again with pivoted LDLᵀ (SYM = 2).
    if sym === :auto && sym_flag == 1 && _wrong_spd_guess(mumps)
        finalize!(mumps)
        return Bramble._mumps_factorize(A; sym = 2, icntl = icntl, cntl = cntl)
    end
    if mumps.infog[1] < 0
        err = _mumps_error(mumps, "factorization")
        finalize!(mumps)
        throw(err)
    end

    auto = sym === :auto ? (_controls(icntl, Int), _controls(cntl, Float64)) : nothing
    return ConcreteMUMPSFactorization(mumps, sym_flag, n, auto, nothing)
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

    fact.failure === nothing || throw(fact.failure)
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

    fact.failure === nothing || throw(fact.failure)
    associate_rhs!(fact.mumps, b)
    fact.mumps.job = MUMPS.SOLVE
    MUMPS.invoke_mumps!(fact.mumps)
    get_sol!(b, fact.mumps)
    return b
end

# Disambiguates the two equally specific candidates for a `VectorElement` destination:
# Bramble's generic `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)` and the
# `ldiv!(::AbstractVector, ::ConcreteMUMPSFactorization, ::AbstractVector)` above.
# `VectorElement <: AbstractVector` and `ConcreteMUMPSFactorization <: Factorization`, so
# neither method is more specific and `ldiv!(uₕ, fact, F)` would otherwise be an ambiguity
# error. Unwraps the destination and returns the `VectorElement`, matching the contract of
# Bramble's method.
function LinearAlgebra.ldiv!(
        uₕ::Bramble.VectorElement, fact::ConcreteMUMPSFactorization{T}, b::AbstractVector
) where {T}
    ldiv!(parent(uₕ), fact, b)
    return uₕ
end

function Base.:\(fact::ConcreteMUMPSFactorization{T}, b::AbstractVector) where {T}
    x = similar(b, promote_type(T, eltype(b)))
    return ldiv!(x, fact, b)
end

# `LinearAlgebra` defines `\(::Factorization{T}, ::Vector{Complex{T}})` for real
# factorizations against a complex right-hand side, and `ConcreteMUMPSFactorization <:
# Factorization`, so that method and the one above are equally specific for a complex vector
# -- the call would be ambiguous without this one.
function Base.:\(
        fact::ConcreteMUMPSFactorization{T}, b::Vector{Complex{T}}
) where {T <: Union{Float32, Float64}}
    return complex.(fact \ real(b), fact \ imag(b))
end

function Bramble._mumps_solve(
        A::SparseMatrixCSC, F::AbstractVector; sym = :auto, icntl = nothing, cntl = nothing
)
    fact = Bramble._mumps_factorize(A; sym = sym, icntl = icntl, cntl = cntl)
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
    mumps = fact.mumps
    if fact.auto !== nothing && fact.sym == 1 && _wrong_spd_guess(mumps)
        # The `:auto` fallback of `_mumps_factorize`: a fresh SYM = 2 instance with the
        # caller's controls replaces the SYM = 1 one
        try
            icntl, cntl = fact.auto
            sym2 = Bramble._mumps_factorize(A_mat; sym = 2, icntl = icntl, cntl = cntl)
            finalize!(mumps)
            fact.mumps, fact.sym = sym2.mumps, 2
        catch err
            fact.failure = err
            rethrow()
        end
    elseif mumps.infog[1] < 0
        # The factors are unusable until a refactorization succeeds; `ldiv!` refuses them
        fact.failure = _mumps_error(mumps, "refactorization")
        throw(fact.failure)
    end
    fact.failure = nothing
    return fact
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{MUMPSFactorization}; kwargs...)
    return Bramble._mumps_factorize(A; kwargs...)
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{<:MUMPSFactorization}; kwargs...)
    return Bramble._mumps_factorize(A; kwargs...)
end

end # module
