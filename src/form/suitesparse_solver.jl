# suitesparse_solver.jl
#
# `suitesparse_factorize`/`suitesparse_solve`/`suitesparse_refactor!`: SuiteSparse (CHOLMOD
# Cholesky/LDLT and UMFPACK LU) sparse direct factorizations for the sparse linear systems
# Bramble discretizes. Implemented in `BrambleSuiteSparseExt`.

"""
    SuiteSparseFactorization{T} <: Factorization{T}

Wrapper type representing a factorized SuiteSparse linear system (CHOLMOD Cholesky or
UMFPACK LU), supporting in-place solves via `LinearAlgebra.ldiv!`, back-substitution
via `\\`, and non-allocating numeric refactoring via `suitesparse_refactor!`.
"""
abstract type SuiteSparseFactorization{T} <: Factorization{T} end

"""
    suitesparse_factorize(A::AbstractMatrix; sym = :auto, kwargs...) -> SuiteSparseFactorization
    suitesparse_factorize(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
                          symmetrize = false, sym = :auto, kwargs...) -> SuiteSparseFactorization

Compute the sparse direct factorization of `A` (or the assembled matrix of `a`) using
SuiteSparse (CHOLMOD for symmetric positive-definite systems, UMFPACK for unsymmetric).

# Symmetry options (`sym`)
- `:auto` (default): automatically detects symmetry. If `A` is symmetric and positive definite
  (or `symmetrize = true`), uses CHOLMOD sparse Cholesky. Otherwise, uses UMFPACK sparse LU.
- `:spd`, `:definite`, or `1`: symmetric positive definite (CHOLMOD Cholesky).
- `:symmetric` or `2`: symmetric factorization.
- `:unsymmetric` or `0`: general unsymmetric (UMFPACK LU).

Requires `SuiteSparse.jl`; call `using SuiteSparse` before calling this function.

# Examples

```julia
using SuiteSparse

A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)
fact = suitesparse_factorize(A; sym = :spd)
u = fact \\ F
```

See also [`suitesparse_solve`](@ref), [`suitesparse_refactor!`](@ref), [`pde_solve`](@ref), [`assemble`](@ref).
"""
function suitesparse_factorize(A::SparseMatrixCSC; kwargs...)
    return _suitesparse_factorize(A; kwargs...)
end

function suitesparse_factorize(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
        sym = :auto, kwargs...
)
    A = assemble(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components
    )
    return _suitesparse_factorize(A; sym = sym, kwargs...)
end

"""
    suitesparse_solve(A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kwargs...) -> Vector
    suitesparse_solve(a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
                      symmetrize = false, sym = :auto, kwargs...) -> VectorElement

Directly solve `A u = F` (or `assemble(a, l)` system) using SuiteSparse factorization.

Requires `SuiteSparse.jl`; call `using SuiteSparse` before calling this function.

See also [`suitesparse_factorize`](@ref), [`refactor!`](@ref), [`pde_solve`](@ref).
"""
function suitesparse_solve(A::SparseMatrixCSC, F::AbstractVector; kwargs...)
    return _suitesparse_solve(A, F; kwargs...)
end

function suitesparse_solve(
        a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
        symmetrize::Bool = false, sym = :auto, kwargs...
)
    A, F = assemble(
        a, l; dirichlet = dirichlet, dirichlet_components = dirichlet_components, symmetrize = symmetrize
    )
    sym_effective = (sym === :auto && symmetrize) ? :spd : sym
    u = _suitesparse_solve(A, F; sym = sym_effective, kwargs...)
    return element(trial_space(a), u)
end

"""
    suitesparse_refactor!(fact::SuiteSparseFactorization, A::SparseMatrixCSC) -> SuiteSparseFactorization

Recompute the numeric factorization of `A` inside `fact` **reusing the existing symbolic
factorization** (fill-reducing ordering and elimination tree). `A` must have the exact same
sparsity pattern as the matrix originally factored.

See also [`refactor!`](@ref), [`suitesparse_factorize`](@ref).
"""
function suitesparse_refactor!(fact::SuiteSparseFactorization, A::SparseMatrixCSC)
    return _suitesparse_refactor!(fact, A)
end

function _suitesparse_factorize(::Any; kwargs...)
    return error(
        "suitesparse_factorize requires SuiteSparse.jl. Add `using SuiteSparse` before calling this function.",
    )
end

function _suitesparse_solve(::Any, ::Any; kwargs...)
    return error(
        "suitesparse_solve requires SuiteSparse.jl. Add `using SuiteSparse` before calling this function.",
    )
end

function _suitesparse_refactor!(::Any, ::Any)
    return error(
        "suitesparse_refactor! requires SuiteSparse.jl. Add `using SuiteSparse` before calling this function.",
    )
end
