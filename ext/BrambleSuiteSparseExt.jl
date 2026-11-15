module BrambleSuiteSparseExt

using Bramble:
               Bramble,
               SuiteSparseFactorization,
               suitesparse_factorize,
               suitesparse_solve,
               suitesparse_refactor!,
               sparse_factorize,
               refactor!,
               pde_solve,
               domain,
               interval,
               ×,
               mesh,
               gridspace,
               element,
               Rₕ,
               form,
               assemble,
               inner₊,
               ∇ₕ,
               innerₕ,
               boundary_symbols
using SuiteSparse: SuiteSparse, CHOLMOD, UMFPACK
using LinearAlgebra: LinearAlgebra, Factorization, ldiv!, ishermitian, Hermitian, cholesky, cholesky!, lu,
                     lu!, diag, issuccess
using SparseArrays: SparseArrays, SparseMatrixCSC
using PrecompileTools: @setup_workload, @compile_workload

mutable struct ConcreteSuiteSparseFactorization{T, FType <: Factorization} <: SuiteSparseFactorization{T}
    fact::FType
    sym::Symbol
    dim::Int
    auto::Bool
end

# The type `lu(A)` returns: UMFPACK factors `Float64` and `ComplexF64`, widening the rest.
_umfpack_type(A::SparseMatrixCSC{T, Ti}) where {T, Ti} =
    UMFPACK.UmfpackLU{T <: Complex ? ComplexF64 : Float64, Ti}

Base.size(fact::ConcreteSuiteSparseFactorization) = (fact.dim, fact.dim)
Base.size(fact::ConcreteSuiteSparseFactorization, d::Integer) = d <= 2 ? fact.dim : 1

function _suitesparse_sym_flag(A::AbstractMatrix, sym)
    if sym === :auto
        # `ishermitian`, not `issymmetric`: CHOLMOD rejects a complex symmetric, non-Hermitian
        # matrix (Helmholtz with absorption) in `symbolic`, before `issuccess` can fall back
        # to LU. The two tests agree on a real matrix.
        if ishermitian(A)
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

# CHOLMOD reads one triangle. A real matrix keeps the old `Symmetric` behaviour; a complex
# one must be Hermitian, or the other triangle's conjugate would be factored silently.
_cholmod_input(A::SparseMatrixCSC{<:Real}) = Hermitian(A)
function _cholmod_input(A::SparseMatrixCSC)
    ishermitian(A) ||
        throw(ArgumentError("CHOLMOD needs a Hermitian complex matrix; use sym = :unsymmetric"))
    return Hermitian(A)
end

function Bramble._suitesparse_factorize(A::SparseMatrixCSC; sym = :auto, kwargs...)
    m, n = size(A)
    m == n || throw(DimensionMismatch("Matrix must be square for sparse direct solve, got $(m)×$(n)"))

    sym_flag = _suitesparse_sym_flag(A, sym)

    if sym_flag === :spd && sym === :auto
        # A symmetric matrix with a positive diagonal is only a Cholesky candidate: CHOLMOD
        # decides, and a failure falls back to UMFPACK, which reports a singular matrix.
        c = cholesky(Hermitian(A); kwargs..., check = false)
        if issuccess(c)
            # `refactor!` may replace the factor by an LU one, so `FType` is a `Union`.
            FType = Union{typeof(c), _umfpack_type(A)}
            return ConcreteSuiteSparseFactorization{eltype(A), FType}(c, :spd, n, true)
        end
        lu_fact = lu(A; kwargs...)
        return ConcreteSuiteSparseFactorization{eltype(A), typeof(lu_fact)}(
            lu_fact, :symmetric, n, true)
    elseif sym_flag === :spd
        c = cholesky(_cholmod_input(A); kwargs...)
        return ConcreteSuiteSparseFactorization{eltype(A), typeof(c)}(c, :spd, n, false)
    else
        lu_fact = lu(A; kwargs...)
        return ConcreteSuiteSparseFactorization{eltype(A), typeof(lu_fact)}(
            lu_fact, sym_flag, n, sym === :auto)
    end
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
    f = fact.fact
    if f isa UMFPACK.UmfpackLU
        ldiv!(f, b)
    else
        x = similar(b)
        ldiv!(x, f, b)
        copyto!(b, x)
    end
    return b
end

# Disambiguates the two equally specific candidates for a `VectorElement` destination:
# Bramble's generic `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)` and the
# `ldiv!(::AbstractVector, ::ConcreteSuiteSparseFactorization, ::AbstractVector)` above.
# `VectorElement <: AbstractVector` and `ConcreteSuiteSparseFactorization <: Factorization`,
# so neither method is more specific and `ldiv!(uₕ, fact, F)` would otherwise be an
# ambiguity error. Takes Bramble's method, so a `Dual` load (`BrambleForwardDiffExt`) is
# split into parts the same way on every backend.
function LinearAlgebra.ldiv!(
        uₕ::Bramble.VectorElement, fact::ConcreteSuiteSparseFactorization, b::AbstractVector
)
    return invoke(ldiv!, Tuple{Bramble.VectorElement, LinearAlgebra.Factorization, AbstractVector},
        uₕ, fact, b)
end

# UMFPACK factors only `Float64` and `ComplexF64`: `lu` of a `Float32` matrix widens it, and
# so does its solve. The result is converted back to the promoted type of the factorization
# and `b`, as the Accelerate extension does, so a `Float32` system returns `Float32`.
function Base.:\(fact::ConcreteSuiteSparseFactorization{T}, b::AbstractVector) where {T}
    S = float(promote_type(T, eltype(b)))
    return convert(Vector{S}, fact.fact \ b)
end

# `LinearAlgebra` defines `\(::Factorization{T}, ::Vector{Complex{T}})` for real
# factorizations against a complex right-hand side, and `ConcreteSuiteSparseFactorization <:
# Factorization`, so that method and the one above are equally specific for a complex vector
# -- the call would be ambiguous without this one.
function Base.:\(
        fact::ConcreteSuiteSparseFactorization{T}, b::Vector{Complex{T}}
) where {T <: Union{Float32, Float64}}
    return complex.(fact \ real(b), fact \ imag(b))
end

# Only an `:auto` factorization that chose CHOLMOD reaches this, and its `FType` holds the
# `lu(A)` type. `lu` throws for a singular `A` before the assignment, so the failed CHOLMOD
# factor stays in `fact` and refuses to solve.
function _fall_back_to_lu!(fact::ConcreteSuiteSparseFactorization, A::SparseMatrixCSC)
    fact.fact = lu(A)
    fact.sym = :symmetric
    return fact
end

function Bramble.suitesparse_refactor!(fact::ConcreteSuiteSparseFactorization, A::SparseMatrixCSC)
    m, n = size(A)
    (m == fact.dim && n == fact.dim) || throw(
        DimensionMismatch("Matrix size $(m)×$(n) does not match factorization dimension $(fact.dim)"),
    )

    f = fact.fact
    if f isa CHOLMOD.Factor
        # CHOLMOD reads one triangle, so an unsymmetric `A` would be factored as its upper
        # triangle. Refused before `cholesky!`: the factors are untouched and `fact` still
        # solves the previous matrix.
        ishermitian(A) || throw(
            ArgumentError(
                "CHOLMOD factors a Hermitian matrix, and this $(eltype(A)) matrix is not " *
                "Hermitian. Refactor onto a Hermitian matrix, or build a new factorization " *
                "with `sym = :unsymmetric`.",
            ),
        )
        if fact.auto
            # `:auto` only tried Cholesky: a Hermitian matrix that is not positive definite
            # moves the factorization to UMFPACK, as `_suitesparse_factorize` does.
            cholesky!(f, _cholmod_input(A); check = false)
            issuccess(f) || _fall_back_to_lu!(fact, A)
        else
            cholesky!(f, _cholmod_input(A))
        end
    elseif f isa UMFPACK.UmfpackLU
        lu!(f, A)
    else
        throw(ArgumentError("Unsupported underlying factorization type: $(typeof(f))"))
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

# Warms this extension's SuiteSparse entry points -- `suitesparse_factorize`,
# `suitesparse_solve`, `suitesparse_refactor!`, and the two `ldiv!` methods above (one into a
# plain `Vector`, one into a `VectorElement`, which needs its own disambiguating method) --
# plus the core `sparse_factorize`/`refactor!`/`pde_solve` entry points routed to this
# backend, only reachable once `SuiteSparse` is loaded so only this extension's own
# precompile pass reaches them. Covers both `:spd` (CHOLMOD) and `:unsymmetric` (UMFPACK)
# factorizations, and both 1D and 2D assembled systems since the `VectorElement` type the
# second `ldiv!` method dispatches on depends on the mesh dimension.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        systems = map((1, 2)) do D
            S = D == 1 ? interval(0.0, 1.0) : interval(0.0, 1.0) × interval(0.0, 1.0)
            Ω = domain(S, :boundary => boundary_symbols(S))
            Ωₕ = D == 1 ? mesh(Ω, 8, false) : mesh(Ω, (2, 2), (false, false))
            Wₕ = gridspace(Ωₕ)
            a_spd = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
            fₕ = Rₕ(Wₕ, x -> 1.0)
            l = form(Wₕ, v -> innerₕ(fₕ, v))
            A = assemble(a_spd)
            F = assemble(l)
            (; Wₕ, A, F)
        end

        @compile_workload begin
            for sys in systems
                x = zeros(length(sys.F))
                uₕ = element(sys.Wₕ, 0.0)

                for sym in (:spd, :unsymmetric)
                    fact = suitesparse_factorize(sys.A; sym = sym)
                    ldiv!(x, fact, sys.F)
                    ldiv!(uₕ, fact, sys.F)
                    suitesparse_refactor!(fact, sys.A)
                    suitesparse_solve(sys.A, sys.F; sym = sym)
                end

                fs = sparse_factorize(sys.A)
                refactor!(fs, sys.A)
                pde_solve(sys.A, sys.F; solver = :suitesparse)
            end
        end
    end
end

end # module
