# matrix_free_preconditioners.jl: preconditioners built from a `BilinearForm` without its
# matrix (gpena/Bramble.jl#327). Jacobi reads the form's diagonal off one walk of
# `visit_bilinear_stencil` with a `DiagonalSink`, through the serial unit walk
# `matrix_free_operator`'s `mul!` takes (`_mf_apply!`, matrix_free.jl), so it costs one vector
# and never assembles `A`.
#
# On a ghost-space form (#781) the matrix is the reduced `E_vᵀ A_pad E_u`, whose diagonal
# differs from the padded one's wherever a closure row reaches a ghost: entry `j` sums
# `E_v[r, j] a_rc E_u[c, j]` over every padded entry `a_rc`, off-diagonal ones included.
# One walk of the padded form with `ReducedDiagonalSink` adds those, reading each padded
# row and column's closure row off the stored `Eᵀ`, so `A_pad` is not stored either.

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

# The policy a form-level preconditioner or smoother runs under by default: the trial
# space's, or the padded space's for a form on ghost spaces.
@inline _default_policy(a::BilinearForm) = execution_policy(trial_space(a))
@inline _default_policy(a::GhostBilinearForm) = _ghost_policy(a.trial_space)

"""
    jacobi_preconditioner(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing, policy = execution_policy(trial_space(a))) -> JacobiPreconditioner
    jacobi_preconditioner(a::GhostBilinearForm; dirichlet = nothing, dirichlet_components = nothing, policy) -> JacobiPreconditioner
    jacobi_preconditioner(op::MatrixFreeOperator) -> JacobiPreconditioner

The Jacobi preconditioner of `A = assemble(a; dirichlet, dirichlet_components)`, or of the
matrix `op` stands for, from its diagonal alone: one walk of the form's stencil keeps the
entries with `row == col`, and `A` is never built. Construction allocates the diagonal and
the Dirichlet row mask, both of length `ndofs`. The walk runs on the calling task under any
CPU policy.

On a form over ghost spaces ([`GhostGridSpace`](@ref)), `A` is the reduced matrix
`E_vᵀ A_pad E_u` that `assemble(a)` returns, and `op` the operator
[`matrix_free_operator`](@ref)`(a)` returns for it: one walk of the padded form sums, for
each degree of freedom, the padded entries whose row and column closures both reach it,
so neither `A_pad` nor `A` is built. `policy` defaults to the padded space's.

# Arguments
- `a`: A square bilinear form ([`ndofs`](@ref) of trial and test space equal), on scalar,
  composite or ghost spaces.
- `op`: A square [`MatrixFreeOperator`](@ref), or the operator `matrix_free_operator`
  returns for a ghost-space form; its Dirichlet rows are kept.

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
        a::Union{BilinearForm, GhostBilinearForm}; dirichlet = nothing,
        dirichlet_components = nothing, policy = _default_policy(a)
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

function jacobi_preconditioner(op::GhostMatrixFreeOperator{T}) where {T <: Number}
    n = size(op, 1)
    n == size(op, 2) || _throw_jacobi_nonsquare(op)
    d = zeros(T, n)
    Eu, Ev = op.trial_space.Et, op.test_space.Et
    sink = ReducedDiagonalSink(d, SparseArrays.getcolptr(Ev), rowvals(Ev), nonzeros(Ev),
        SparseArrays.getcolptr(Eu), rowvals(Eu), nonzeros(Eu))
    _mf_apply!(CpuSerial(), sink, op.padded.form)
    @inbounds for i in eachindex(op.mask)
        op.mask[i] && (d[i] = one(T))
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
with [`fdm_preconditioner`](@ref), which needs `using Kronecker`; on a form over ghost
spaces it inverts the Laplacian-like part of the reduced Kronecker operator.

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

"""
    ReducedDiagonalSink(d, vptr, vidx, vnz, uptr, uidx, unz)

A sink that accumulates the diagonal of a ghost-space form's reduced matrix
`E_vᵀ A_pad E_u` from one walk of its padded form: for a padded entry `(row, col, weight)`
it adds `E_v[row, j] * weight * E_u[col, j]` into `d[j]` for every degree of freedom `j`
that both closure rows reach. `vptr`, `vidx`, `vnz` are the column pointers, row indices
and values of `E_vᵀ` (one column per padded point), and `uptr`, `uidx`, `unz` those of
`E_uᵀ`. Dirichlet rows are not skipped: the caller overwrites them.
"""
struct ReducedDiagonalSink{V <: AbstractVector, IV <: AbstractVector{Int},
    NV <: AbstractVector, IU <: AbstractVector{Int}, NU <: AbstractVector}
    d::V
    vptr::IV
    vidx::IV
    vnz::NV
    uptr::IU
    uidx::IU
    unz::NU
end

# A closure row holds a few entries (one for a physical point, the relation's terms for a
# ghost), so the two rows are matched pairwise.
@inline function _sink_entry!(s::ReducedDiagonalSink, row::Int, col::Int, weight, ::Int)
    @inbounds for p in s.vptr[row]:(s.vptr[row + 1] - 1)
        j = s.vidx[p]
        for q in s.uptr[col]:(s.uptr[col + 1] - 1)
            s.uidx[q] == j && (s.d[j] += s.vnz[p] * weight * s.unz[q])
        end
    end
    return nothing
end
