# matrix_free_preconditioners.jl: preconditioners built from a `BilinearForm` without its
# matrix (gpena/Bramble.jl#327). Jacobi reads the form's diagonal off one walk of
# `visit_bilinear_stencil` with a `DiagonalSink`, through the serial unit walk
# `matrix_free_operator`'s `mul!` takes (`_mf_apply!`, matrix_free.jl), so it costs one vector
# and never assembles `A`.

"""
    AbstractMatrixFreePreconditioner{T}

A preconditioner `P ≈ A` for the matrix `A` of a [`BilinearForm`](@ref), applied as `P⁻¹`
without storing `A`. A subtype supplies `ldiv!(y, P, x)` (`y = P⁻¹ x`) and `ldiv!(P, x)` (in
place); `P \\ x` returns `P⁻¹ x` in a new vector. That is the interface `LinearSolve`'s `Pl`
and `Pr` keywords take, so any subtype preconditions a Krylov solve of
[`assemble`](@ref)`(a)` or of [`matrix_free_operator`](@ref)`(a)`.

# Type parameters
- `T`: The element type of `P⁻¹ x` for a vector `x` of element type `T`.

See also: [`JacobiPreconditioner`](@ref), [`jacobi_preconditioner`](@ref).
"""
abstract type AbstractMatrixFreePreconditioner{T} end

Base.eltype(::Type{<:AbstractMatrixFreePreconditioner{T}}) where {T} = T

function Base.:\(P::AbstractMatrixFreePreconditioner{T}, x::AbstractVector) where {T}
    y = similar(x, promote_type(T, eltype(x)))
    return ldiv!(y, P, x)
end

"""
    JacobiPreconditioner{T, V <: AbstractVector{T}} <: AbstractMatrixFreePreconditioner{T}

The Jacobi preconditioner `P = diag(A)`: `ldiv!(y, P, x)` sets `y[i] = x[i] / A[i, i]`.
Stores the reciprocal diagonal, so an application is one multiply per entry and allocates
nothing. A Dirichlet row of `A` is the identity row, so its entry is `1`. Build one with
[`jacobi_preconditioner`](@ref).

A zero diagonal entry gives an infinite reciprocal, as `x ./ diag(A)` does.

# Type parameters
- `T`: The element type of `A`.
- `V`: The vector type of the stored reciprocal diagonal.

See also: [`jacobi_preconditioner`](@ref), [`AbstractMatrixFreePreconditioner`](@ref).
"""
struct JacobiPreconditioner{T, V <: AbstractVector{T}} <: AbstractMatrixFreePreconditioner{T}
    inv_diagonal::V
end

Base.size(P::JacobiPreconditioner) = (n = length(P.inv_diagonal); (n, n))

"""
    jacobi_preconditioner(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing, policy = execution_policy(trial_space(a))) -> JacobiPreconditioner
    jacobi_preconditioner(op::MatrixFreeOperator) -> JacobiPreconditioner

The Jacobi preconditioner of `A = assemble(a; dirichlet, dirichlet_components)`, or of the
matrix `op` stands for, from its diagonal alone: one walk of the form's stencil keeps the
entries with `row == col`, and `A` is never built. Construction allocates the diagonal and
the Dirichlet row mask, both of length `ndofs`. The walk runs on the calling task under any
CPU policy.

# Arguments
- `a`: A square bilinear form ([`ndofs`](@ref) of trial and test space equal), on scalar or
  composite spaces.
- `op`: A square [`MatrixFreeOperator`](@ref); its Dirichlet rows are kept.

# Keywords
- `dirichlet`, `dirichlet_components`, `policy`: As in [`matrix_free_operator`](@ref).

# Returns
- [`JacobiPreconditioner`](@ref): `ldiv!(y, P, x)` gives `x ./ diag(A)`.

# Throws
- `ArgumentError`: `policy` is a [`GpuPolicy`](@ref); device execution is tracked on
  milestone v4.4.0.
- `ArgumentError`: `dirichlet_components` names a leaf the test space does not have.
- `DimensionMismatch`: `A` is not square.

# Examples
```jldoctest
using Bramble, LinearAlgebra
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11, false))
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
P = jacobi_preconditioner(a; dirichlet = :boundary)
x = rand(ndofs(Wₕ))
P \\ x ≈ x ./ diag(assemble(a; dirichlet = :boundary))

# output
true
```

See also: [`JacobiPreconditioner`](@ref), [`matrix_free_operator`](@ref).
"""
function jacobi_preconditioner(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
        policy = execution_policy(trial_space(a))
)
    op = matrix_free_operator(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components, policy = policy
    )
    return jacobi_preconditioner(op)
end

function jacobi_preconditioner(op::MatrixFreeOperator{T}) where {T <: Number}
    n = op.nrows
    n == op.ncols || _throw_jacobi_nonsquare(op)
    d = zeros(T, n)
    _mf_apply!(CpuSerial(), DiagonalSink(d), op.form)
    mask = _mf_mask(op)
    if mask !== nothing
        @inbounds for i in eachindex(mask)
            mask[i] && (d[i] = one(T))
        end
    end
    d .= inv.(d)
    return JacobiPreconditioner{T, typeof(d)}(d)
end

@noinline function _throw_jacobi_nonsquare(op)
    throw(DimensionMismatch("jacobi_preconditioner needs a square form, got $(size(op))"))
end

@noinline function _throw_jacobi_dimmismatch(P, x, y)
    throw(
        DimensionMismatch(
        "JacobiPreconditioner of size $(size(P)) cannot divide a vector of length " *
        "$(length(x)) into one of length $(length(y))",
    ),
    )
end

function ldiv!(y::AbstractVector, P::JacobiPreconditioner, x::AbstractVector)
    d = P.inv_diagonal
    (length(x) == length(d) && length(y) == length(d)) || _throw_jacobi_dimmismatch(P, x, y)
    yd = _mf_data(y)
    xd = _mf_data(x)
    Base.require_one_based_indexing(yd, xd)
    @inbounds for i in eachindex(d)
        yd[i] = d[i] * xd[i]
    end
    return y
end

ldiv!(P::JacobiPreconditioner, x::AbstractVector) = ldiv!(x, P, x)

"""
    FDMPreconditioner{T, F} <: AbstractMatrixFreePreconditioner{T}

The fast-diagonalisation preconditioner of a separable [`BilinearForm`](@ref): `ldiv!(y, P,
x)` applies the exact inverse of the form's Laplacian-like part (every term that differs
from one mass per axis on at most one axis), factorised once, and allocates nothing on host
vectors. Under `dirichlet = :boundary` it is the identity on the boundary rows. Build one
with [`fdm_preconditioner`](@ref), which needs `using Kronecker`.

# Type parameters
- `T`: The element type of the form's Kronecker operator.
- `F`: The type of the stored factorisation, the one `BrambleKroneckerExt` builds.

See also: [`fdm_preconditioner`](@ref), [`AbstractMatrixFreePreconditioner`](@ref).
"""
struct FDMPreconditioner{T, F} <: AbstractMatrixFreePreconditioner{T}
    factorization::F
end

# The factorisation's `size` and both `ldiv!` methods come from `BrambleKroneckerExt`, the
# only place a three-argument `ldiv!` exists for `ldiv!(P, x)` to forward to.
Base.size(P::FDMPreconditioner) = size(P.factorization)
Base.size(P::FDMPreconditioner, i::Integer) = size(P.factorization, i)

# --- The diagonal walk ------------------------------------------------------------------ #

"""
    DiagonalSink(d::AbstractVector)

A sink that adds the weight of every stencil entry with `row == col` into `d[row]`, so one
walk of [`visit_bilinear_stencil`](@ref) accumulates `diag(A)` without storing `A`. Off-
diagonal entries are dropped. Dirichlet rows are not skipped: the caller overwrites them.
"""
struct DiagonalSink{V <: AbstractVector}
    d::V
end

# `row` lands inside the matrix (the walk's guard) and `d` has its row count.
@inline function _sink_entry!(s::DiagonalSink, row::Int, col::Int, weight, ::Int)
    row == col && @inbounds(s.d[row] += weight)
    return nothing
end
