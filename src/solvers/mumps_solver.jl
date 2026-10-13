# mumps_solver.jl
#
# `mumps_factorize`/`mumps_solve`: MUMPS (MUltifrontal Massively Parallel sparse direct
# Solver) parallel multifrontal LU, LDLᵀ, and Cholesky factorizations for the sparse
# linear systems Bramble discretizes. Implemented in `BrambleMUMPSExt`, same
# `_`-prefixed fallback idiom as `amg_preconditioner` (solvers/amg_preconditioner.jl) and
# `linear_problem` (form/semidiscrete_problems.jl).

"""
    MUMPSFactorization{T} <: Factorization{T}

Wrapper type representing a factorized MUMPS linear system, supporting in-place solves
via `LinearAlgebra.ldiv!`, back-substitution via `\\`, and non-allocating reuse for
transient PDE time stepping.
"""
abstract type MUMPSFactorization{T} <: Factorization{T} end

"""
    mumps_factorize(A::SparseMatrixCSC; sym = :auto, icntl = nothing, cntl = nothing) -> MUMPSFactorization
    mumps_factorize(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
                    sym = :auto, icntl = nothing, cntl = nothing) -> MUMPSFactorization

Compute the sparse direct multifrontal factorization of `A` (or the assembled matrix of `a`)
using [MUMPS.jl](https://github.com/lruthotto/MUMPS.jl).

# Symmetry options
- `:auto`: the default, which detects symmetry. A symmetric `A` is factored as symmetric
  positive definite (`sym = 1`, no pivoting), and when MUMPS reports a zero or negative
  pivot (`INFOG(1) = -10` or `INFOG(12) > 0`) it is factored again as general symmetric
  ``LDL^T`` (`sym = 2`). Any other `A` uses unsymmetric LU (`sym = 0`).
- `:spd`, `:definite`, `1`: symmetric positive definite (Cholesky / ``LL^T``).
- `:symmetric`, `2`: general symmetric (``LDL^T`` with Bunch-Kaufman pivoting).
- `:unsymmetric`, `0`: general unsymmetric LU.

A failed factorisation (`INFOG(1) < 0`) throws for every `sym`: a `SingularException` for a
zero pivot (`INFOG(1) = -10`), an error otherwise.

The factorization's MUMPS memory is freed once it is unreachable; `finalize(fact)` frees it
at once, after which `fact` must not be used.

# Control parameters
- `icntl`: Optional dictionary or collection of pairs of integer control parameters (e.g.,
  `7 => 1` for user/METIS ordering, `14 => 30` for memory relaxation).
- `cntl`: Optional dictionary or collection of pairs of real control parameters (e.g.,
  `1 => 0.01` for numerical pivoting threshold).

On a matrix, any other keyword throws a `MethodError`. The form method also takes `dirichlet`
and `dirichlet_components`, and refuses `symmetrize` with an `ArgumentError`.

A `BilinearForm` on a dense or CSR backend is converted to `SparseMatrixCSC` first.

Requires [MUMPS.jl](https://github.com/lruthotto/MUMPS.jl); call `using MUMPS` before
calling this function.

# Examples

```julia
using Bramble: mumps_factorize
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
    _reject_factorize_symmetrize(:mumps_factorize, kwargs)
    A = assemble(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components
    )
    A = _as_csc(A)
    return _mumps_factorize(A; sym = sym, kwargs...)
end

"""
    mumps_solve(A::SparseMatrixCSC, F::AbstractVector; sym = :auto, icntl = nothing, cntl = nothing) -> Vector
    mumps_solve(a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
                symmetrize = false, sym = :auto, icntl = nothing, cntl = nothing) -> VectorElement

Directly solve `A u = F` (or `assemble(a, l)` system) using MUMPS direct factorization.

# Symmetry options
- `:auto`: the default, as in [`mumps_factorize`](@ref): `symmetrize = true` does not
  choose positive definite, and a symmetric indefinite `A` is refactored as ``LDL^T``.
- `:spd`, `:definite`, `1`: symmetric positive definite.
- `:symmetric`, `2`: general symmetric.
- `:unsymmetric`, `0`: general unsymmetric.

`icntl` and `cntl` are as in [`mumps_factorize`](@ref). Any other keyword throws a
`MethodError`.

A `BilinearForm` on a dense or CSR backend is converted to `SparseMatrixCSC` first.

Requires [MUMPS.jl](https://github.com/lruthotto/MUMPS.jl); call `using MUMPS` before
calling this function.

# Examples

```julia
using Bramble: mumps_solve
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
    A = _as_csc(A)
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

function _mumps_factorize(A; kwargs...)
    _ext_loaded(:BrambleMUMPSExt) && _throw_unsupported_matrix("mumps_factorize", A)
    return error(
        "mumps_factorize requires MUMPS.jl. Add `using MUMPS` before calling this function.",
    )
end

function _mumps_solve(A, ::Any; kwargs...)
    _ext_loaded(:BrambleMUMPSExt) && _throw_unsupported_matrix("mumps_solve", A)
    return error(
        "mumps_solve requires MUMPS.jl. Add `using MUMPS` before calling this function.",
    )
end

function _mumps_refactor!(::Any, ::Any)
    return error(
        "mumps_refactor! requires MUMPS.jl. Add `using MUMPS` before calling this function.",
    )
end
