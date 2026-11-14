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

# `matrix` copies the factored matrix, which `lu.p` only aliases: a load that must be
# solved against the matrix refactored at its own type (`_solve_wider`) sees the matrix as
# factored even if the caller has since overwritten its values or its pattern.
mutable struct ConcreteSparspakFactorization{T, LUT} <: SparspakFactorization{T}
    lu::LUT
    dim::Int
    matrix::SparseMatrixCSC{T, Int}
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
    return ConcreteSparspakFactorization{eltype(A), typeof(lu)}(
        lu, n, SparseMatrixCSC{eltype(A), Int}(A)
    )
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
# error. Takes Bramble's method, so a `Dual` load (`BrambleForwardDiffExt`) is split
# into parts the same way on every backend.
function LinearAlgebra.ldiv!(
        uₕ::Bramble.VectorElement, fact::ConcreteSparspakFactorization, b::AbstractVector
)
    return invoke(ldiv!, Tuple{Bramble.VectorElement, LinearAlgebra.Factorization, AbstractVector},
        uₕ, fact, b)
end

# Sparspak's triangular solve takes a right-hand side of the matrix's own element type only,
# so a real load against a `Dual` (or other wider) matrix is promoted first. A load that
# already has that type is passed as is: the all-`Float64` path copies it once, as before.
# A load wider than the matrix is solved part by part (`_solve_parts`) or, when it does not
# split into parts of type `T`, against `A` refactored at the load's type.
function Base.:\(fact::ConcreteSparspakFactorization{T}, b::AbstractVector) where {T}
    finite_rhs = all(isfinite, b)
    S = promote_type(T, eltype(b))
    x = if S === T
        b′ = convert(AbstractVector{T}, b)
        _singular_on_empty(() -> fact.lu \ b′, fact.dim)
    else
        _solve_wider(fact, Vector{S}(b))
    end
    return _check_finite_solution(x, finite_rhs)
end

function _solve_wider(fact::ConcreteSparspakFactorization{T}, b::Vector{S}) where {T, S}
    parts = _parts(S, T)
    parts === nothing || return _solve_parts(fact, b, parts)
    A = SparseMatrixCSC{S}(fact.matrix)
    return _singular_on_empty(() -> sparspaklu(A) \ b, fact.dim)
end

# The solve is linear in the load, and a scalar of type `T` multiplies each part of a
# `Complex` or `Dual` load on its own, so `T⁻¹`-solving every part gives the solution. Part
# `k` of `b[j]` is the `T` whose leaves are `leaves(b[j])[parts[:, k]]`.
function _solve_parts(
        fact::ConcreteSparspakFactorization{T}, b::Vector{S}, parts::Matrix{Int}
) where {T, S}
    R = _leaf_type(T)
    n = length(b)
    B = reshape(reinterpret(R, b), :, n)
    x = similar(b)
    X = reshape(reinterpret(R, x), :, n)
    y = Vector{T}(undef, n)
    Y = reshape(reinterpret(R, y), :, n)
    for k in axes(parts, 2)
        for j in 1:n, i in axes(parts, 1)

            Y[i, j] = B[parts[i, k], j]
        end
        _singular_on_empty(() -> ldiv!(fact.lu, y), fact.dim)
        for j in 1:n, i in axes(parts, 1)

            X[parts[i, k], j] = Y[i, j]
        end
    end
    return x
end

# The primitive type every leaf of the bits type `T` has, or `nothing` if they differ.
function _leaf_type(T::Type)
    isprimitivetype(T) && return T
    (T isa DataType && isbitstype(T) && fieldcount(T) > 0) || return nothing
    leaves = map(_leaf_type, fieldtypes(T))
    R = first(leaves)
    return all(==(R), leaves) ? R : nothing
end

_nleaves(T::Type, R::Type) = sizeof(T) ÷ sizeof(R)

# Splits the bits number type `S` into parts of type `T`, both with the same primitive
# leaves: returns `parts` with column `k` listing the leaves of `S` that form part `k`, in
# `T`'s leaf order, or `nothing` if `S` has no such split. `S` is either built from fields
# that split into `T` (a `Dual` whose value and partials have type `T`) or is `T`'s own
# wrapper with the same non-numeric parameters and wider fields (a `Dual` with `T`'s tag
# over a wider value type), recursively.
function _parts(S::Type, T::Type)
    (S isa DataType && T isa DataType && S <: Number && T <: Number) || return nothing
    R = _leaf_type(T)
    (R === nothing || _leaf_type(S) !== R) && return nothing
    return _split(S, T, R)
end

function _split(S::DataType, T::DataType, R::DataType)
    S === T && return reshape(collect(1:_nleaves(T, R)), :, 1)
    S <: Number && !Bramble._linear_in_parts(S) && return nothing
    isprimitivetype(S) && return nothing
    outer = _split_fields(S, T, R)
    outer === nothing || return outer
    return _split_wrapper(S, T, R)
end
_split(::Type, ::Type, ::Type) = nothing

# Every field of `S` holds whole parts of type `T`.
function _split_fields(S::DataType, T::DataType, R::DataType)
    blocks = Matrix{Int}[]
    offset = 0
    for F in fieldtypes(S)
        P = _split(F, T, R)
        P === nothing && return nothing
        push!(blocks, P .+ offset)
        offset += _nleaves(F, R)
    end
    return isempty(blocks) ? zeros(Int, _nleaves(T, R), 0) : reduce(hcat, blocks)
end

# `S` and `T` share their wrapper and non-numeric parameters (a `Dual`'s tag and length),
# and each field of `S` splits into the matching field of `T` in as many parts.
function _split_wrapper(S::DataType, T::DataType, R::DataType)
    (S.name === T.name && length(S.parameters) == length(T.parameters)) || return nothing
    for (p, q) in zip(S.parameters, T.parameters)
        p == q && continue
        (p isa Type && q isa Type && p <: Number && q <: Number) || return nothing
    end
    blocks = Matrix{Int}[]
    offset = 0
    for (F, G) in zip(fieldtypes(S), fieldtypes(T))
        P = _split(F, G, R)
        P === nothing && return nothing
        push!(blocks, P .+ offset)
        offset += _nleaves(F, R)
    end
    isempty(blocks) && return nothing
    K = size(first(blocks), 2)
    all(P -> size(P, 2) == K, blocks) || return nothing
    return reduce(vcat, blocks)
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
    copy!(fact.matrix.colptr, SparseArrays.getcolptr(A))
    copy!(fact.matrix.rowval, SparseArrays.rowvals(A))
    copy!(fact.matrix.nzval, SparseArrays.nonzeros(A))
    return fact
end

function Bramble._sparspak_solve(A::SparseMatrixCSC, F::AbstractVector)
    T, S = eltype(A), promote_type(eltype(A), eltype(F))
    # A load with no split into parts of type `T` is solved against `A` at its own type.
    A′ = S === T || _parts(S, T) !== nothing ? A : SparseMatrixCSC{S}(A)
    fact = Bramble._sparspak_factorize(A′)
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
