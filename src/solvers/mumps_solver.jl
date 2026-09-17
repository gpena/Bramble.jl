# mumps_solver.jl
#
# `mumps_factorize`/`mumps_solve`: MUMPS (MUltifrontal Massively Parallel sparse direct
# Solver) parallel multifrontal LU, LDLᵀ, and Cholesky factorizations for the sparse
# linear systems Bramble discretizes. Implemented in `BrambleMUMPSExt`, same
# underscored-fallback idiom as `amg_preconditioner` (solvers/amg_preconditioner.jl) and
# `linear_problem` (form/semidiscrete_problems.jl).

"""
    MUMPSFactorization{T} <: Factorization{T}

Wrapper type representing a factorized MUMPS linear system, supporting in-place solves
via `LinearAlgebra.ldiv!`, back-substitution via `\\`, and non-allocating reuse for
transient PDE time stepping.
"""
abstract type MUMPSFactorization{T} <: Factorization{T} end

"""
    mumps_factorize(A::AbstractMatrix; sym = :auto, icntl = nothing, cntl = nothing, kwargs...) -> MUMPSFactorization
    mumps_factorize(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
                    symmetrize = false, sym = :auto, kwargs...) -> MUMPSFactorization

Compute the sparse direct multifrontal factorization of `A` (or the assembled matrix of `a`)
using [MUMPS.jl](https://github.com/lruthotto/MUMPS.jl).

# Symmetry options
- `:auto` (default): automatically detects symmetry. If `A` is symmetric and positive definite
  (or `symmetrize = true`), uses symmetric positive-definite factorization (`sym = 1`). If
  symmetric, uses general symmetric ``LDL^T`` (`sym = 2`). Otherwise, uses unsymmetric LU (`sym = 0`).
- `:spd`, `:definite`, or `1`: symmetric positive definite (Cholesky / ``LL^T``).
- `:symmetric` or `2`: general symmetric (``LDL^T`` with Bunch-Kaufman pivoting).
- `:unsymmetric` or `0`: general unsymmetric LU.

# Control parameters
- `icntl`: Optional dictionary or collection of pairs of integer control parameters (e.g., `7 => 1` for user/METIS ordering, `14 => 30` for memory relaxation).
- `cntl`: Optional dictionary or collection of pairs of real control parameters (e.g., `1 => 0.01` for numerical pivoting threshold).

Requires [MUMPS.jl](https://github.com/lruthotto/MUMPS.jl); call `using MUMPS` before
calling this function.

# Examples

```julia
using MUMPS

A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)
fact = mumps_factorize(A; sym = :spd)
u = fact \\ F
```

See also [`pde_solve`](@ref), [`mumps_solve`](@ref), [`assemble`](@ref).
"""
function mumps_factorize(A::SparseMatrixCSC; kwargs...)
    return _mumps_factorize(A; kwargs...)
end

function mumps_factorize(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
        sym = :auto, kwargs...
)
    A = assemble(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components
    )
    return _mumps_factorize(A; sym = sym, kwargs...)
end

"""
    mumps_solve(A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kwargs...) -> Vector
    mumps_solve(a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
                symmetrize = false, sym = :auto, kwargs...) -> VectorElement

Directly solve `A u = F` (or `assemble(a, l)` system) using MUMPS direct factorization.

# Symmetry options
- `:auto` (default): automatically detects symmetry.
- `:spd`, `:definite`, or `1`: symmetric positive definite.
- `:symmetric` or `2`: general symmetric.
- `:unsymmetric` or `0`: general unsymmetric.

Requires [MUMPS.jl](https://github.com/lruthotto/MUMPS.jl); call `using MUMPS` before
calling this function.

# Examples

```julia
using MUMPS

A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
u = mumps_solve(A, F)
```

See also [`mumps_factorize`](@ref), [`refactor!`](@ref), [`pde_solve`](@ref).
"""
function mumps_solve(A::SparseMatrixCSC, F::AbstractVector; kwargs...)
    return _mumps_solve(A, F; kwargs...)
end

function mumps_solve(
        a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
        symmetrize::Bool = false, sym = :auto, kwargs...
)
    A, F = assemble(
        a, l; dirichlet = dirichlet, dirichlet_components = dirichlet_components, symmetrize = symmetrize
    )
    u = _mumps_solve(A, F; sym = sym, kwargs...)
    return element(trial_space(a), u)
end

"""
    mumps_refactor!(fact::MUMPSFactorization, A::SparseMatrixCSC) -> MUMPSFactorization

Recompute the numeric factorization of `A` inside `fact` **reusing the existing symbolic
factorization** (fill-reducing analysis and ordering). `A` must have the exact same
sparsity pattern as the matrix originally factored.

See also [`refactor!`](@ref), [`mumps_factorize`](@ref).
"""
function mumps_refactor!(fact::MUMPSFactorization, A::SparseMatrixCSC)
    return _mumps_refactor!(fact, A)
end

function _mumps_factorize(::Any; kwargs...)
    return error(
        "mumps_factorize requires MUMPS.jl. Add `using MUMPS` before calling this function.",
    )
end

function _mumps_solve(::Any, ::Any; kwargs...)
    return error(
        "mumps_solve requires MUMPS.jl. Add `using MUMPS` before calling this function.",
    )
end

function _mumps_refactor!(::Any, ::Any)
    return error(
        "mumps_refactor! requires MUMPS.jl. Add `using MUMPS` before calling this function.",
    )
end
