# sparspak_solver.jl
#
# `sparspak_factorize`/`sparspak_solve`: pure-Julia sparse direct LU factorization (the
# Waterloo Sparse Linear Equations Package, George & Liu) for the sparse linear systems
# Bramble discretizes, with zero binary dependencies. Implemented in `BrambleSparspakExt`,
# same `_`-prefixed fallback idiom as `mumps_factorize` (solvers/mumps_solver.jl) and
# `accelerate_factorize` (solvers/accelerate_solver.jl). Unlike SuiteSparse or MUMPS, Sparspak's
# factorization is generic over the matrix element type, so it also factors matrices of
# `Float32`, `BigFloat`, or `ForwardDiff.Dual` entries.

"""
    SparspakFactorization{T} <: Factorization{T}

Wrapper type representing a factorized [Sparspak.jl](https://github.com/JuliaSparse/Sparspak.jl)
sparse LU system, supporting in-place solves via `LinearAlgebra.ldiv!`, back-substitution via
`\\`, and non-allocating reuse for transient PDE time stepping.
"""
abstract type SparspakFactorization{T} <: Factorization{T} end

"""
    sparspak_factorize(A::SparseMatrixCSC) -> SparspakFactorization
    sparspak_factorize(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing) -> SparspakFactorization

Compute the pure-Julia sparse direct LU factorization of `A` (or the assembled matrix of `a`)
using [Sparspak.jl](https://github.com/JuliaSparse/Sparspak.jl).

Sparspak has no binary dependency, so it factors matrices whose entries are not
`Float64`/`ComplexF64` -- `Float32`, `BigFloat`, or a `ForwardDiff.Dual` -- where SuiteSparse
and MUMPS cannot.

A singular matrix throws a `SingularException`, as it does for SuiteSparse, MUMPS and
Accelerate, from the factorization or from the solve: a rank-deficient or structurally
singular matrix at factorization, and a stored zero pivot or an empty row or column when a
solve returns a non-finite entry for a finite right-hand side. Sparspak's own failure is
a misleading `ErrorException` or an `Inf` in the solution. The index of the
`SingularException` is `0`, not a pivot position.

A `BilinearForm` on a dense or CSR backend is converted to `SparseMatrixCSC` first.

Requires [Sparspak.jl](https://github.com/JuliaSparse/Sparspak.jl); call `using Sparspak`
before calling this function.

# Examples

```julia
using Bramble: sparspak_factorize
using Sparspak

A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
fact = sparspak_factorize(A)
u = fact \\ F
```

See also [`sparspak_solve`](@ref), [`sparspak_refactor!`](@ref), [`pde_solve`](@ref), [`assemble`](@ref).
"""
function sparspak_factorize(A::SparseMatrixCSC)
    return _sparspak_factorize(A)
end

function sparspak_factorize(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing
)
    A = assemble(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components
    )
    A = _as_csc(A)
    return _sparspak_factorize(A)
end

"""
    sparspak_solve(A::SparseMatrixCSC, F::AbstractVector) -> Vector
    sparspak_solve(a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
                   symmetrize = false) -> VectorElement

Directly solve `A u = F` (or `assemble(a, l)` system) using Sparspak's pure-Julia sparse
direct LU factorization.

A singular matrix throws a `SingularException` with index `0` (see
[`sparspak_factorize`](@ref)).

A `BilinearForm` on a dense or CSR backend is converted to `SparseMatrixCSC` first.

Requires [Sparspak.jl](https://github.com/JuliaSparse/Sparspak.jl); call `using Sparspak`
before calling this function.

# Examples

```julia
using Bramble: sparspak_solve
using Sparspak

A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
u = sparspak_solve(A, F)
```

See also [`sparspak_factorize`](@ref), [`sparspak_refactor!`](@ref), [`pde_solve`](@ref).
"""
function sparspak_solve(A::SparseMatrixCSC, F::AbstractVector)
    return _sparspak_solve(A, F)
end

function sparspak_solve(
        a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
        symmetrize::Bool = false
)
    A, F = assemble(
        a, l; dirichlet = dirichlet, dirichlet_components = dirichlet_components, symmetrize = symmetrize
    )
    A = _as_csc(A)
    u = _sparspak_solve(A, F)
    return element(trial_space(a), u)
end

"""
    sparspak_refactor!(fact::SparspakFactorization, A::SparseMatrixCSC) -> SparspakFactorization

Recompute the numeric factorization of `A` inside `fact` **reusing the existing symbolic
factorization** (fill-reducing ordering). `A` must have the exact same sparsity pattern as
the matrix originally factored. A singular `A` throws a `SingularException` with index `0`.

See also [`refactor!`](@ref), [`sparspak_factorize`](@ref).
"""
function sparspak_refactor! end

function _sparspak_factorize(A)
    _ext_loaded(:BrambleSparspakExt) && _throw_unsupported_matrix("sparspak_factorize", A)
    return error(
        "sparspak_factorize requires Sparspak.jl. Add `using Sparspak` before calling this function.",
    )
end

function _sparspak_solve(A, ::Any)
    _ext_loaded(:BrambleSparspakExt) && _throw_unsupported_matrix("sparspak_solve", A)
    return error(
        "sparspak_solve requires Sparspak.jl. Add `using Sparspak` before calling this function.",
    )
end

function sparspak_refactor!(::SparspakFactorization, A)
    _ext_loaded(:BrambleSparspakExt) && _throw_unsupported_matrix("sparspak_refactor!", A)
    return error(
        "sparspak_refactor! requires Sparspak.jl. Add `using Sparspak` before calling this function.",
    )
end
