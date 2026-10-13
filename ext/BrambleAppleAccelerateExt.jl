module BrambleAppleAccelerateExt

using Bramble: Bramble, AccelerateFactorization
using LinearAlgebra: LinearAlgebra, ldiv!, ishermitian, diag, opnorm, norm
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
        # `checked` marks an `:auto` factorization, which `_aa_singular` tests once per
        # factorization and per `refactor!`. `singular` records a refused `refactor!`, so
        # later solves throw as SuiteSparse's do; testing it costs one branch per solve.
        checked::Bool
        singular::Bool
    end

    Base.size(fact::ConcreteAccelerateFactorization) = (fact.dim, fact.dim)
    Base.size(fact::ConcreteAccelerateFactorization, d::Integer) = d <= 2 ? fact.dim : 1

    # `nothing` spells `:auto` too, for `sym` and for `kind`.
    _aa_auto(sym, kind) = (sym === :auto || sym === nothing) && (kind === :auto || kind === nothing)

    function _accelerate_fact_kind(A::AbstractMatrix, sym, kind)
        if kind === :cholesky || sym === :spd || sym === :definite || sym == 1
            ishermitian(A) || _throw_not_hermitian(A)
            return (SparseFactorizationCholesky, :spd)
        elseif kind === :ldlt || sym === :symmetric || sym == 2
            ishermitian(A) || _throw_not_hermitian(A)
            return (SparseFactorizationLDLT, :symmetric)
        elseif kind === :qr
            return (SparseFactorizationQR, :qr)
        elseif kind === :lu || kind === :lutpp || sym === :unsymmetric || sym == 0
            return (SparseFactorizationLUTPP, :unsymmetric)
        elseif _aa_auto(sym, kind)
            # Hermitian, not symmetric: libSparse's Cholesky and LDLᵀ factor `A = LDLᴴ`, so
            # a complex symmetric matrix that is not Hermitian goes to LU.
            if ishermitian(A)
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

    # libSparse's own refusal is an `ErrorException`; the caller chose the kind, so it is
    # not converted to LU behind their back.
    # A refactor keeps the kind it was built with, so its remedy is a new factorization.
    @noinline function _throw_not_hermitian(A::AbstractMatrix, refactor::Bool = false)
        remedy = refactor ?
            "Refactor onto a Hermitian matrix, or build a new factorization with " *
            "`sym = :auto` or `kind = :lu`." :
            "Use sym = :auto or kind = :lu."
        throw(
            ArgumentError(
            "Cholesky and LDLᵀ factor a Hermitian matrix, and this $(eltype(A)) matrix is " *
            "not Hermitian. " * remedy,
        ),
        )
    end

    # AppleAccelerate 0.7.0's `AASparseMatrix(A)` stores a Hermitian `A` as its tagged
    # lower triangle. Cholesky and LDLᵀ read a triangle; LU and QR read the matrix, and
    # factor that triangle alone, so they get the full storage, untagged.
    # `AASparseMatrix(A)` takes `Int64` indices only, so an `Int32` triangle is widened
    # first (`convert` returns an `Int64` `A` as it is).
    function _aa_matrix(A::SparseMatrixCSC, sym_flag::Symbol)
        sym_flag === :unsymmetric || sym_flag === :qr ||
            return AASparseMatrix(convert(SparseMatrixCSC{eltype(A), Int64}, A))
        return AASparseMatrix(
            size(A)..., Clong.(A.colptr .- 1), Cint.(A.rowval .- 1), copy(A.nzval), ATT_ORDINARY
        )
    end

    function Bramble._accelerate_factorize(A::SparseMatrixCSC; sym = :auto, kind = :auto)
        m, n = size(A)
        m == n || throw(DimensionMismatch("Matrix must be square for sparse direct solve, got $(m)×$(n)"))

        fact_kind, sym_flag = _accelerate_fact_kind(A, sym, kind)

        aa = AAFactorization(_aa_matrix(A, sym_flag))
        auto = _aa_auto(sym, kind)
        if fact_kind === SparseFactorizationCholesky && auto
            # Symmetry and a positive diagonal make `A` a Cholesky candidate only: a failed
            # `factor!` refactors as LDLᵀ on the same object.
            try
                factor!(aa, fact_kind)
            catch e
                e isa ErrorException || rethrow()
                fact_kind, sym_flag = SparseFactorizationLDLT, :symmetric
                factor!(aa, fact_kind)
            end
        else
            factor!(aa, fact_kind)
        end
        auto && _aa_singular(aa, A) && throw(LinearAlgebra.SingularException(0))

        return ConcreteAccelerateFactorization{eltype(A)}(aa, sym_flag, n, auto, false)
    end

    # The pivoted LDLᵀ and LUTPP never report a singular matrix, and Cholesky can succeed on
    # one: they return a solution whatever the right-hand side. A singular `A` has a range
    # that misses almost every vector, so a solve against a fixed vector with no structure
    # (a Weyl sequence, not `ones`, which lies in the range of `[1 1; 1 1]`) leaves a large
    # backward error, while a nonsingular `A` solves it to rounding. Only an exact zero
    # pivot is caught. A nearly singular `A` gives a huge `x` that the tolerance absorbs, and
    # so does a pure-Neumann Laplacian, whose zero pivot the fill-reducing ordering rounds
    # to about 1e-16. No residual threshold separates that from a nonsingular Helmholtz
    # matrix of condition 3.6e16: ‖r‖/‖b‖ is 9.7 to 21 against 2.6 (gpena/Bramble.jl#668).
    function _aa_singular(aa, A::SparseMatrixCSC)
        n = size(A, 1)
        T = eltype(A)
        b = T[0.5 + mod(k * 0.6180339887498949, 1.0) for k in 1:n]
        x = aa \ b
        r = A * x - b
        tol = sqrt(eps(real(T))) * (opnorm(A, Inf) * norm(x, Inf) + norm(b, Inf))
        return !(norm(r, Inf) <= tol)
    end

    function _aa_assert_usable(fact::ConcreteAccelerateFactorization)
        fact.singular && throw(LinearAlgebra.SingularException(0))
        return nothing
    end

    # libSparse solves a strided buffer of the factorization's element type `T` with unit
    # row stride. Any other right-hand side, a row-strided view, an adjoint or another
    # element type, goes through a `T` copy; a complex one on a real factorization is
    # solved as its real and imaginary parts, side by side in one call, as `LinearAlgebra`
    # does for a complex `Vector` or `Matrix`.
    function _aa_solve!(aa::AAFactorization{T}, B::AbstractVecOrMat) where {T}
        B isa StridedVecOrMat{T} && stride(B, 1) == 1 && return ldiv!(aa, B)
        n, k = size(B, 1), size(B, 2)
        if T <: Real && eltype(B) <: Complex
            W = Matrix{T}(undef, n, 2k)
            W[:, 1:k] .= real.(B)
            W[:, (k + 1):(2k)] .= imag.(B)
            ldiv!(aa, W)
            copyto!(B, complex.(view(W, :, 1:k), view(W, :, (k + 1):(2k))))
        else
            W = copyto!(Matrix{T}(undef, n, k), B)
            ldiv!(aa, W)
            copyto!(B, W)
        end
        return B
    end
    function _aa_solve!(
            x::AbstractVecOrMat, aa::AAFactorization{T}, b::AbstractVecOrMat
    ) where {T}
        x isa StridedVecOrMat{T} && b isa StridedVecOrMat{T} &&
            stride(x, 1) == 1 && stride(b, 1) == 1 && return ldiv!(x, aa, b)
        return _aa_solve!(aa, copyto!(x, b))
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
        _aa_assert_usable(fact)
        _aa_solve!(x, fact.aa_fact, b)
        return x
    end

    function LinearAlgebra.ldiv!(
            fact::ConcreteAccelerateFactorization, b::AbstractVector
    )
        n = fact.dim
        length(b) == n || throw(
            DimensionMismatch("Vector length $(length(b)) does not match matrix dimension $n"),
        )
        _aa_assert_usable(fact)
        _aa_solve!(fact.aa_fact, b)
        return b
    end

    # A matrix right-hand side goes to libSparse's multi-column solve in one call;
    # `LinearAlgebra`'s `\(::Factorization, ::AbstractVecOrMat)` reaches it through here.
    # libSparse refuses a right-hand side with no columns, which has nothing to solve.
    function LinearAlgebra.ldiv!(
            fact::ConcreteAccelerateFactorization, B::AbstractMatrix
    )
        n = fact.dim
        size(B, 1) == n || throw(
            DimensionMismatch("Right-hand side has $(size(B, 1)) rows; the matrix dimension is $n"),
        )
        _aa_assert_usable(fact)
        size(B, 2) == 0 && return B
        _aa_solve!(fact.aa_fact, B)
        return B
    end

    function LinearAlgebra.ldiv!(
            X::AbstractMatrix, fact::ConcreteAccelerateFactorization, B::AbstractMatrix
    )
        n = fact.dim
        (size(B, 1) == n && size(X) == size(B)) || throw(
            DimensionMismatch("Output is $(join(size(X), '×')) and right-hand side \
                              $(join(size(B), '×')); the matrix dimension is $n"),
        )
        _aa_assert_usable(fact)
        size(B, 2) == 0 && return X
        _aa_solve!(X, fact.aa_fact, B)
        return X
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
        _aa_assert_usable(fact)
        T = promote_type(eltype(fact), eltype(b))
        return _aa_solve!(fact.aa_fact, copyto!(similar(b, T, size(b)), b))
    end

    # `LinearAlgebra` defines `\(::Factorization{T}, ::Vector{Complex{T}})` for real
    # factorizations against a complex right-hand side, and `ConcreteAccelerateFactorization
    # <: Factorization`, so that method and the one above are equally specific for a complex
    # vector -- the call would be ambiguous without this one.
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
        # A Cholesky or LDLᵀ factorization, `:auto` or explicit, takes a Hermitian matrix
        # only: AppleAccelerate stores a non-Hermitian one in full, against the factored
        # triangle, and it needs LU, a new symbolic factorization. Refused before the
        # factors are touched, so `fact` still solves with the old matrix.
        (fact.sym === :spd || fact.sym === :symmetric) && !ishermitian(A) &&
            _throw_not_hermitian(A, true)
        M = _aa_matrix(A, fact.sym)
        try
            refactor!(fact.aa_fact, M)
        catch e
            (fact.checked && e isa ErrorException) || rethrow()
            # A failed numeric refactor leaves the old factors unusable.
            fact.singular = true
            fact.sym === :spd || rethrow()
            # `:auto` Cholesky on a matrix that is not positive definite: the LDLᵀ fallback
            # `_accelerate_factorize` takes, on a fresh object, since Cholesky's symbolic
            # factorization cannot be refactored as LDLᵀ.
            aa = AAFactorization(M)
            factor!(aa, SparseFactorizationLDLT)
            fact.aa_fact, fact.sym = aa, :symmetric
        end
        fact.singular = fact.checked && _aa_singular(fact.aa_fact, A)
        fact.singular && throw(LinearAlgebra.SingularException(0))
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
