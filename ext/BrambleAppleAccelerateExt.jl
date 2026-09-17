module BrambleAppleAccelerateExt

using Bramble: Bramble, AccelerateFactorization
using LinearAlgebra: LinearAlgebra, ldiv!, issymmetric, diag
using SparseArrays: SparseArrays, SparseMatrixCSC

# AppleAccelerate.jl only defines its sparse factorization types (`AAFactorization` and
# friends) on macOS. Referencing one in a struct field type -- even behind a runtime
# `Sys.isapple()` check -- still fails to precompile on Linux/Windows CI, since the field
# type itself is resolved at module load time. `@static if` skips the whole block there
# instead, and the `else` arm below reproduces the same `ArgumentError` the runtime guards
# used to raise, so callers on non-macOS see no behavior change.
@static if Sys.isapple()
    using AppleAccelerate:
                           AppleAccelerate,
                           AAFactorization,
                           factor!,
                           refactor!,
                           SparseFactorizationCholesky,
                           SparseFactorizationLDLT,
                           SparseFactorizationLUTPP,
                           SparseFactorizationQR

    mutable struct ConcreteAccelerateFactorization{T} <: AccelerateFactorization{T}
        aa_fact::AAFactorization{T}
        sym::Symbol
        dim::Int
    end

    Base.size(fact::ConcreteAccelerateFactorization) = (fact.dim, fact.dim)
    Base.size(fact::ConcreteAccelerateFactorization, d::Integer) = d <= 2 ? fact.dim : 1

    function _accelerate_fact_kind(A::AbstractMatrix, sym, kind)
        if kind === :cholesky || sym === :spd || sym === :definite || sym == 1
            return (SparseFactorizationCholesky, :spd)
        elseif kind === :ldlt || sym === :symmetric || sym == 2
            return (SparseFactorizationLDLT, :symmetric)
        elseif kind === :qr
            return (SparseFactorizationQR, :qr)
        elseif kind === :lu || kind === :lutpp || sym === :unsymmetric || sym == 0
            return (SparseFactorizationLUTPP, :unsymmetric)
        elseif (sym === :auto || sym === nothing) && (kind === :auto || kind === nothing)
            if issymmetric(A)
                d = diag(A)
                if !isempty(d) && all(x -> real(x) > 0, d)
                    return (SparseFactorizationCholesky, :spd)
                else
                    return (SparseFactorizationLDLT, :symmetric)
                end
            else
                return (SparseFactorizationLUTPP, :unsymmetric)
            end
        else
            throw(
                ArgumentError(
                "Unknown AppleAccelerate symmetry/factorization option: sym=$sym, kind=$kind.",
            ),
            )
        end
    end

    function Bramble._accelerate_factorize(A::SparseMatrixCSC; sym = :auto, kind = :auto, kwargs...)
        m, n = size(A)
        m == n || throw(DimensionMismatch("Matrix must be square for sparse direct solve, got $(m)×$(n)"))

        fact_kind, sym_flag = _accelerate_fact_kind(A, sym, kind)

        aa = AAFactorization(A)
        factor!(aa, fact_kind)

        return ConcreteAccelerateFactorization{eltype(A)}(aa, sym_flag, n)
    end

    function LinearAlgebra.ldiv!(
            x::AbstractVector, fact::ConcreteAccelerateFactorization, b::AbstractVector
    )
        n = fact.dim
        length(b) == n || throw(
            DimensionMismatch("Right-hand side length $(length(b)) does not match matrix dimension $n"),
        )
        length(x) == n || throw(
            DimensionMismatch("Output vector length $(length(x)) does not match matrix dimension $n"),
        )
        ldiv!(x, fact.aa_fact, b)
        return x
    end

    function LinearAlgebra.ldiv!(
            fact::ConcreteAccelerateFactorization, b::AbstractVector
    )
        n = fact.dim
        length(b) == n || throw(
            DimensionMismatch("Vector length $(length(b)) does not match matrix dimension $n"),
        )
        ldiv!(fact.aa_fact, b)
        return b
    end

    # Disambiguates the two equally specific candidates for a `VectorElement` destination:
    # Bramble's generic `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)` and the
    # `ldiv!(::AbstractVector, ::ConcreteAccelerateFactorization, ::AbstractVector)` above.
    # `VectorElement <: AbstractVector` and `ConcreteAccelerateFactorization <:
    # Factorization`, so neither method is more specific and `ldiv!(uₕ, fact, F)` would
    # otherwise be an ambiguity error. Unwraps the destination and returns the
    # `VectorElement`, matching the contract of Bramble's method.
    function LinearAlgebra.ldiv!(
            uₕ::Bramble.VectorElement, fact::ConcreteAccelerateFactorization, b::AbstractVector
    )
        ldiv!(parent(uₕ), fact, b)
        return uₕ
    end

    function Base.:\(fact::ConcreteAccelerateFactorization, b::AbstractVector)
        return fact.aa_fact \ b
    end

    # `LinearAlgebra` defines `\(::Factorization{T}, ::Vector{Complex{T}})` for real
    # factorizations against a complex right-hand side, and `ConcreteAccelerateFactorization
    # <: Factorization`, so that method and the one above are equally specific for a complex
    # vector -- the call would be ambiguous without this one. It restates what
    # `LinearAlgebra` does: solve the real and imaginary parts separately and recombine
    # them.
    function Base.:\(
            fact::ConcreteAccelerateFactorization{T}, b::Vector{Complex{T}}
    ) where {T <: Union{Float32, Float64}}
        return complex.(fact \ real(b), fact \ imag(b))
    end

    function Bramble._accelerate_refactor!(fact::ConcreteAccelerateFactorization, A::SparseMatrixCSC)
        m, n = size(A)
        (m == fact.dim && n == fact.dim) || throw(
            DimensionMismatch("Matrix size $(m)×$(n) does not match factorization dimension $(fact.dim)"),
        )
        refactor!(fact.aa_fact, A)
        return fact
    end
else
    function Bramble._accelerate_factorize(::SparseMatrixCSC; kwargs...)
        throw(ArgumentError("AppleAccelerate is only supported on macOS (darwin)."))
    end

    function Bramble._accelerate_refactor!(fact, ::SparseMatrixCSC)
        throw(ArgumentError("AppleAccelerate is only supported on macOS (darwin)."))
    end
end

function Bramble._accelerate_solve(A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kind = :auto, kwargs...)
    fact = Bramble._accelerate_factorize(A; sym = sym, kind = kind, kwargs...)
    return fact \ F
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{AccelerateFactorization}; kwargs...)
    return Bramble._accelerate_factorize(A; kwargs...)
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{<:AccelerateFactorization}; kwargs...)
    return Bramble._accelerate_factorize(A; kwargs...)
end

end # module
