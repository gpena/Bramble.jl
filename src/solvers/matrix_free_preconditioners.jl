# matrix_free_preconditioners.jl: preconditioners built from a `BilinearForm` without its
# matrix (gpena/Bramble.jl#327). Jacobi reads the form's diagonal off one walk of
# `visit_bilinear_stencil` with a `DiagonalSink`, the walk `matrix_free_operator`'s `mul!`
# takes (matrix_free.jl), so it costs one vector and never assembles `A`.

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

function jacobi_preconditioner(op::MatrixFreeOperator{T}) where {T}
    n = op.nrows
    n == op.ncols || _throw_jacobi_nonsquare(op)
    d = zeros(T, n)
    _diagonal_walk!(DiagonalSink(d), op.form)
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
    @inbounds for i in eachindex(d)
        yd[i] = d[i] * xd[i]
    end
    return y
end

ldiv!(P::JacobiPreconditioner, x::AbstractVector) = ldiv!(x, P, x)

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

# The unit walk of `_mf_apply!` (matrix_free.jl), serial, with every summand walked alone: a
# transposed pair's second term is walked as itself instead of as the first's transpose,
# which gives the same entries. `map` over the summand tuple, as `stencil_eval.jl`'s walks,
# keeps each term's call concrete without allocating.
function _diagonal_walk!(s::DiagonalSink, a::BilinearForm)
    Wu, Wv, ast = trial_space(a), test_space(a), a.ast
    if _is_block_pair(Wu, Wv)
        trial_leaves = leaf_spaces_offsets(Wu)
        test_leaves = leaf_spaces_offsets(Wv)
        map(t -> _diagonal_blocks!(s, t, trial_leaves, test_leaves), _summands(ast))
        return nothing
    end
    bound = _bind_interp_spaces(ast, Wu, Wv)
    _check_block_meshes(bound, Wu, Wv)
    sp = host_weights(_walked_leaf(bound, Wu, Wv))
    map(t -> _diagonal_summand!(s, t, sp), _summands(bound))
    return nothing
end

@noinline function _diagonal_summand!(s::DiagonalSink, term::TERM, sp) where {TERM}
    visit_bilinear_stencil(s, term, sp, 0, 0)
    return nothing
end

@noinline function _diagonal_blocks!(
        s::DiagonalSink, term::TERM, trial_leaves, test_leaves
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        bound = _bind_interp_spaces(term, blk.trial_leaf, blk.test_leaf)
        _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
        sp = host_weights(_walked_leaf(bound, blk.trial_leaf, blk.test_leaf))
        visit_bilinear_stencil(s, bound, sp, blk.row_offset, blk.col_offset)
    end
    return nothing
end
