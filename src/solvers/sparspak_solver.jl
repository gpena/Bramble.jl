# sparspak_solver.jl
#
# `sparspak_factorize`/`sparspak_solve`: pure-Julia sparse direct LU factorization (the
# Waterloo Sparse Linear Equations Package, George & Liu) for the sparse linear systems
# Bramble discretizes, with zero binary dependencies. Implemented in `BrambleSparspakExt`,
# same underscored-fallback idiom as `mumps_factorize` (solvers/mumps_solver.jl) and
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
    sparspak_factorize(A::AbstractMatrix) -> SparspakFactorization
    sparspak_factorize(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing) -> SparspakFactorization

Compute the pure-Julia sparse direct LU factorization of `A` (or the assembled matrix of `a`)
using [Sparspak.jl](https://github.com/JuliaSparse/Sparspak.jl).

Sparspak has no binary dependency, so it factors matrices whose entries are not
`Float64`/`ComplexF64` -- `Float32`, `BigFloat`, or a `ForwardDiff.Dual` -- where SuiteSparse
and MUMPS cannot.

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
    return _sparspak_factorize(A)
end

"""
    sparspak_solve(A::SparseMatrixCSC, F::AbstractVector) -> Vector
    sparspak_solve(a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
                   symmetrize = false) -> VectorElement

Directly solve `A u = F` (or `assemble(a, l)` system) using Sparspak's pure-Julia sparse
direct LU factorization.

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
    u = _sparspak_solve(A, F)
    return element(trial_space(a), u)
end

"""
    sparspak_refactor!(fact::SparspakFactorization, A::SparseMatrixCSC) -> SparspakFactorization

Recompute the numeric factorization of `A` inside `fact` **reusing the existing symbolic
factorization** (fill-reducing ordering). `A` must have the exact same sparsity pattern as
the matrix originally factored.

See also [`refactor!`](@ref), [`sparspak_factorize`](@ref).
"""
function sparspak_refactor!(fact::SparspakFactorization, A::SparseMatrixCSC)
    return _sparspak_refactor!(fact, A)
end

function _sparspak_factorize(::Any)
    return error(
        "sparspak_factorize requires Sparspak.jl. Add `using Sparspak` before calling this function.",
    )
end

function _sparspak_solve(::Any, ::Any)
    return error(
        "sparspak_solve requires Sparspak.jl. Add `using Sparspak` before calling this function.",
    )
end

function _sparspak_refactor!(::Any, ::Any)
    return error(
        "sparspak_refactor! requires Sparspak.jl. Add `using Sparspak` before calling this function.",
    )
end
