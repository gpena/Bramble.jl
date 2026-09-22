module BrambleSparspakExt

using Bramble:
               Bramble,
               SparspakFactorization,
               sparspak_factorize,
               sparspak_solve,
               sparspak_refactor!,
               sparse_factorize,
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
using Sparspak: Sparspak, sparspaklu, sparspaklu!
using LinearAlgebra: LinearAlgebra, ldiv!
using SparseArrays: SparseArrays, SparseMatrixCSC
using PrecompileTools: @setup_workload, @compile_workload

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

# Disambiguates the two equally specific candidates for a `VectorElement` destination:
# Bramble's generic `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)` and the
# `ldiv!(::AbstractVector, ::ConcreteSparspakFactorization, ::AbstractVector)` above.
# `VectorElement <: AbstractVector` and `ConcreteSparspakFactorization <: Factorization`, so
# neither method is more specific and `ldiv!(uₕ, fact, F)` would otherwise be an ambiguity
# error. Unwraps the destination and returns the `VectorElement`, matching the contract of
# Bramble's method.
function LinearAlgebra.ldiv!(
        uₕ::Bramble.VectorElement, fact::ConcreteSparspakFactorization, b::AbstractVector
)
    ldiv!(parent(uₕ), fact, b)
    return uₕ
end

function Base.:\(fact::ConcreteSparspakFactorization, b::AbstractVector)
    return fact.lu \ b
end

# `LinearAlgebra` defines `\(::Factorization{T}, ::Vector{Complex{T}})` for real
# factorizations against a complex right-hand side, and `ConcreteSparspakFactorization <:
# Factorization`, so that method and the one above are equally specific for a complex vector
# -- the call would be ambiguous without this one. It restates what `LinearAlgebra` does:
# solve the real and imaginary parts separately and recombine them.
function Base.:\(
        fact::ConcreteSparspakFactorization{T}, b::Vector{Complex{T}}
) where {T <: Union{Float32, Float64}}
    return complex.(fact \ real(b), fact \ imag(b))
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

# Warms this extension's Sparspak entry points -- `sparspak_factorize`, `sparspak_solve`,
# `sparspak_refactor!`, and the two `ldiv!` methods above (one into a plain `Vector`, one into
# a `VectorElement`, which needs its own disambiguating method) -- plus the core
# `sparse_factorize` entry point routed to this backend, only reachable once `Sparspak` is
# loaded so only this extension's own precompile pass reaches them. Sparspak always factors
# as general unsymmetric LU (no `sym` keyword), so a single SPD system per dimension
# exercises every call. Covers both 1D and 2D assembled systems since the `VectorElement`
# type the second `ldiv!` method dispatches on depends on the mesh dimension. Added for
# gpena/Bramble.jl#284.
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

                fact = sparspak_factorize(sys.A)
                ldiv!(x, fact, sys.F)
                ldiv!(uₕ, fact, sys.F)
                sparspak_refactor!(fact, sys.A)
                sparspak_solve(sys.A, sys.F)
                sparse_factorize(sys.A; solver = :sparspak)
            end
        end
    end
end

end # module
