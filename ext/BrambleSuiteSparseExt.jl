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
using LinearAlgebra: LinearAlgebra, Factorization, ldiv!, issymmetric, Symmetric, cholesky, cholesky!, lu,
                     lu!, diag
using SparseArrays: SparseArrays, SparseMatrixCSC
using PrecompileTools: @setup_workload, @compile_workload

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

# Disambiguates the two equally specific candidates for a `VectorElement` destination:
# Bramble's generic `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)` and the
# `ldiv!(::AbstractVector, ::ConcreteSuiteSparseFactorization, ::AbstractVector)` above.
# `VectorElement <: AbstractVector` and `ConcreteSuiteSparseFactorization <: Factorization`,
# so neither method is more specific and `ldiv!(uₕ, fact, F)` would otherwise be an
# ambiguity error. Unwraps the destination and returns the `VectorElement`, matching the
# contract of Bramble's method.
function LinearAlgebra.ldiv!(
        uₕ::Bramble.VectorElement, fact::ConcreteSuiteSparseFactorization, b::AbstractVector
)
    ldiv!(parent(uₕ), fact, b)
    return uₕ
end

function Base.:\(fact::ConcreteSuiteSparseFactorization, b::AbstractVector)
    return fact.fact \ b
end

# `LinearAlgebra` defines `\(::Factorization{T}, ::Vector{Complex{T}})` for real
# factorizations against a complex right-hand side, and `ConcreteSuiteSparseFactorization <:
# Factorization`, so that method and the one above are equally specific for a complex vector
# -- the call would be ambiguous without this one. It restates what `LinearAlgebra` does:
# solve the real and imaginary parts separately and recombine them.
function Base.:\(
        fact::ConcreteSuiteSparseFactorization{T}, b::Vector{Complex{T}}
) where {T <: Union{Float32, Float64}}
    return complex.(fact \ real(b), fact \ imag(b))
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

# Warms this extension's SuiteSparse entry points -- `suitesparse_factorize`,
# `suitesparse_solve`, `suitesparse_refactor!`, and the two `ldiv!` methods above (one into a
# plain `Vector`, one into a `VectorElement`, which needs its own disambiguating method) --
# plus the core `sparse_factorize`/`refactor!`/`pde_solve` entry points routed to this
# backend, only reachable once `SuiteSparse` is loaded so only this extension's own
# precompile pass reaches them. Covers both `:spd` (CHOLMOD) and `:unsymmetric` (UMFPACK)
# factorizations, and both 1D and 2D assembled systems since the `VectorElement` type the
# second `ldiv!` method dispatches on depends on the mesh dimension. Not named in
# gpena/Bramble.jl#196; added for gpena/Bramble.jl#284.
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
