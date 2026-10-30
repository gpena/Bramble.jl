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
using LinearAlgebra: LinearAlgebra, SingularException, ldiv!
using SparseArrays: SparseArrays, SparseMatrixCSC
using PrecompileTools: @setup_workload, @compile_workload

mutable struct ConcreteSparspakFactorization{T, LUT} <: SparspakFactorization{T}
    lu::LUT
    dim::Int
end

Base.size(fact::ConcreteSparspakFactorization) = (fact.dim, fact.dim)
Base.size(fact::ConcreteSparspakFactorization, d::Integer) = d <= 2 ? fact.dim : 1

# Sparspak never throws `SingularException`. A rank-deficient or structurally singular
# matrix reaches an internal `error("An empty problem. No matrix.")` (its n == 0 guard, hit
# once elimination has no pivot left), and a stored zero pivot or an empty row/column goes
# through the solve unchecked and leaves `Inf`/`NaN` in the solution. Both are reported the
# way SuiteSparse, MUMPS and Accelerate report them. The index 0 is not a pivot position.
const SPARSPAK_EMPTY_PROBLEM = "An empty problem. No matrix."

# Runs `f()` on an `n`×`n` matrix; Sparspak's empty-problem `ErrorException` becomes a
# `SingularException(0)` (matched by message and only for n > 0, where it means no pivot).
@inline function _singular_on_empty(f::F, n::Integer) where {F}
    try
        return f()
    catch err
        n > 0 && err isa ErrorException && err.msg == SPARSPAK_EMPTY_PROBLEM &&
            throw(SingularException(0))
        rethrow()
    end
end

# Throws `SingularException(0)` when a solve of a finite right-hand side left a non-finite
# entry in `x`. A non-finite right-hand side legitimately propagates and is not blamed on `A`.
function _check_finite_solution(x::AbstractVector, finite_rhs::Bool)
    finite_rhs && !all(isfinite, x) && throw(SingularException(0))
    return x
end

function Bramble._sparspak_factorize(A::SparseMatrixCSC)
    m, n = size(A)
    m == n || throw(DimensionMismatch("Matrix must be square for sparse direct solve, got $(m)×$(n)"))

    lu = _singular_on_empty(() -> sparspaklu(A), n)
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
    finite_rhs = all(isfinite, b)
    _singular_on_empty(() -> ldiv!(x, fact.lu, b), n)
    return _check_finite_solution(x, finite_rhs)
end

function LinearAlgebra.ldiv!(
        fact::ConcreteSparspakFactorization, b::AbstractVector
)
    n = fact.dim
    length(b) == n || throw(
        DimensionMismatch("Vector length $(length(b)) does not match matrix dimension $n"),
    )
    finite_rhs = all(isfinite, b)
    _singular_on_empty(() -> ldiv!(fact.lu, b), n)
    return _check_finite_solution(b, finite_rhs)
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
    finite_rhs = all(isfinite, b)
    x = _singular_on_empty(() -> fact.lu \ b, fact.dim)
    return _check_finite_solution(x, finite_rhs)
end

# `LinearAlgebra` defines `\(::Factorization{T}, ::Vector{Complex{T}})` for real
# factorizations against a complex right-hand side, and `ConcreteSparspakFactorization <:
# Factorization`, so that method and the one above are equally specific for a complex vector
# -- the call would be ambiguous without this one.
function Base.:\(
        fact::ConcreteSparspakFactorization{T}, b::Vector{Complex{T}}
) where {T <: Union{Float32, Float64}}
    return complex.(fact \ real(b), fact \ imag(b))
end

function Bramble.sparspak_refactor!(fact::ConcreteSparspakFactorization, A::SparseMatrixCSC)
    m, n = size(A)
    (m == fact.dim && n == fact.dim) || throw(
        DimensionMismatch("Matrix size $(m)×$(n) does not match factorization dimension $(fact.dim)"),
    )
    _singular_on_empty(() -> sparspaklu!(fact.lu, A; allow_pattern_change = false), n)
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
# type the second `ldiv!` method dispatches on depends on the mesh dimension.
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
