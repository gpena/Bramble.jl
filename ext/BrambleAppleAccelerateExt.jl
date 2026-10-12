module BrambleAppleAccelerateExt

using Bramble: Bramble, AccelerateFactorization
using LinearAlgebra: LinearAlgebra, ldiv!, issymmetric, diag, opnorm, norm
using SparseArrays: SparseArrays, SparseMatrixCSC

# This module implements only the *sparse* `_accelerate_factorize`/`_accelerate_refactor!`
# hooks below, over AppleAccelerate.jl's `libSparse` bindings (`AAFactorization` and
# friends). There is no dense counterpart here, and none is needed: AppleAccelerate.jl
# (v0.7.0) has no dense `lu`/`cholesky`/LAPACK entry points to hook. Its `__init__` instead
# calls `BLAS.lbt_forward` to register Accelerate as a libblastrampoline provider, so plain
# `LinearAlgebra.lu`/`cholesky`/`qr` run on Accelerate's BLAS/LAPACK the moment `using
# AppleAccelerate` has been evaluated, with no dispatch through Bramble or this extension.
# The dense `accelerate_factorize(A::AbstractMatrix; ...)` method lives unconditionally in
# `src/solvers/accelerate_solver.jl` for exactly this reason -- see its docstring.

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
                           AASparseMatrix,
                           ATT_ORDINARY,
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
        # Set only when `:auto` fell back from Cholesky to LDLᵀ. The pivoted LDLᵀ never
        # reports a singular matrix, so every solve checks its backward error against it.
        check::Union{Nothing, SparseMatrixCSC{T}}
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

    # AppleAccelerate 0.7.0's `AASparseMatrix(A)` stores a Hermitian `A` as its tagged
    # lower triangle. Cholesky and LDLᵀ read a triangle; LU and QR read the matrix, and
    # factor that triangle alone, so they get the full storage, untagged.
    function _aa_matrix(A::SparseMatrixCSC, sym_flag::Symbol)
        sym_flag === :unsymmetric || sym_flag === :qr || return AASparseMatrix(A)
        return AASparseMatrix(
            size(A)..., Clong.(A.colptr .- 1), Cint.(A.rowval .- 1), copy(A.nzval), ATT_ORDINARY
        )
    end

    function Bramble._accelerate_factorize(A::SparseMatrixCSC; sym = :auto, kind = :auto)
        m, n = size(A)
        m == n || throw(DimensionMismatch("Matrix must be square for sparse direct solve, got $(m)×$(n)"))

        fact_kind, sym_flag = _accelerate_fact_kind(A, sym, kind)

        aa = AAFactorization(_aa_matrix(A, sym_flag))
        check = nothing
        if fact_kind === SparseFactorizationCholesky && sym === :auto && kind === :auto
            # Symmetry and a positive diagonal make `A` a Cholesky candidate only: a failed
            # `factor!` refactors as LDLᵀ on the same object.
            try
                factor!(aa, fact_kind)
            catch e
                e isa ErrorException || rethrow()
                fact_kind, sym_flag = SparseFactorizationLDLT, :symmetric
                factor!(aa, fact_kind)
                check = A
            end
        else
            factor!(aa, fact_kind)
        end

        return ConcreteAccelerateFactorization{eltype(A)}(aa, sym_flag, n, check)
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
        return _aa_check(fact, x, b)
    end

    function LinearAlgebra.ldiv!(
            fact::ConcreteAccelerateFactorization, b::AbstractVector
    )
        n = fact.dim
        length(b) == n || throw(
            DimensionMismatch("Vector length $(length(b)) does not match matrix dimension $n"),
        )
        b0 = fact.check === nothing ? b : copy(b)
        ldiv!(fact.aa_fact, b)
        fact.check === nothing || _aa_check(fact, b, b0)
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

    # The backward error of an LDLᵀ fallback (`fact.check`): a singular matrix gives a
    # large residual, which `ldiv!` and `\` turn into the `SingularException` that the
    # pivoted LDLᵀ does not raise itself.
    function _aa_check(fact, x, b)
        A = fact.check
        A === nothing && return x
        r = A * x - b
        tol = sqrt(eps(real(eltype(A)))) * (opnorm(A, Inf) * norm(x, Inf) + norm(b, Inf))
        norm(r, Inf) <= tol || throw(LinearAlgebra.SingularException(0))
        return x
    end

    function Base.:\(fact::ConcreteAccelerateFactorization, b::AbstractVector)
        return _aa_check(fact, fact.aa_fact \ b, b)
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
        refactor!(fact.aa_fact, _aa_matrix(A, fact.sym))
        fact.check === nothing || (fact.check = A)
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

function Bramble._accelerate_solve(A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kind = :auto)
    fact = Bramble._accelerate_factorize(A; sym = sym, kind = kind)
    return fact \ F
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{AccelerateFactorization}; kwargs...)
    return Bramble._accelerate_factorize(A; kwargs...)
end

function LinearAlgebra.factorize(A::SparseMatrixCSC, ::Type{<:AccelerateFactorization}; kwargs...)
    return Bramble._accelerate_factorize(A; kwargs...)
end

end # module
