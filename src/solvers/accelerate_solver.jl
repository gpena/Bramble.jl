# accelerate_solver.jl
#
# `accelerate_factorize`/`accelerate_solve`/`accelerate_refactor!`: Apple Accelerate
# (`libSparse`) sparse Cholesky, LDLᵀ, and LUTPP factorizations for the sparse linear systems
# Bramble discretizes on macOS. Implemented in `BrambleAppleAccelerateExt`.

"""
    AccelerateFactorization{T} <: Factorization{T}

Wrapper type representing a factorized Apple Accelerate (`libSparse`) linear system,
supporting in-place solves via `LinearAlgebra.ldiv!`, back-substitution via `\\`,
and non-allocating symbolic reuse via `accelerate_refactor!`. Available only on macOS.
"""
abstract type AccelerateFactorization{T} <: Factorization{T} end

"""
    accelerate_factorize(A::AbstractMatrix; sym = :auto, kind = :auto, kwargs...) -> AccelerateFactorization
    accelerate_factorize(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
                         symmetrize = false, sym = :auto, kind = :auto, kwargs...) -> AccelerateFactorization

Compute the sparse direct factorization of `A` (or the assembled matrix of `a`) using Apple
Accelerate's native `libSparse` on macOS.

# Symmetry and factorization options
- `sym = :auto` (default): automatically detects symmetry. If `A` is symmetric with strictly positive
  diagonal, uses Cholesky (`SparseFactorizationCholesky`). If symmetric, uses ``LDL^T`` (`SparseFactorizationLDLT`).
  Otherwise uses threshold partial pivoting LU (`SparseFactorizationLUTPP`).
- `sym = :spd`, `:definite`, or `1` (or `kind = :cholesky`): symmetric positive definite Cholesky.
- `sym = :symmetric` or `2` (or `kind = :ldlt`): symmetric indefinite ``LDL^T``.
- `sym = :unsymmetric` or `0` (or `kind = :lu` / `:lutpp`): general unsymmetric LU with threshold partial pivoting.
- `kind = :qr`: sparse QR factorization.

Requires macOS and [AppleAccelerate.jl](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl);
call `using AppleAccelerate` before calling this function.

# Examples

```julia
using AppleAccelerate

A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)
fact = accelerate_factorize(A; sym = :spd)
u = fact \\ F
```

See also [`accelerate_solve`](@ref), [`accelerate_refactor!`](@ref), [`pde_solve`](@ref), [`assemble`](@ref).
"""
function accelerate_factorize(A::SparseMatrixCSC; kwargs...)
    if !Sys.isapple()
        throw(ArgumentError("AppleAccelerate is only supported on macOS (darwin)."))
    end
    return _accelerate_factorize(A; kwargs...)
end

function accelerate_factorize(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
        sym = :auto, kind = :auto, kwargs...
)
    if !Sys.isapple()
        throw(ArgumentError("AppleAccelerate is only supported on macOS (darwin)."))
    end
    A = assemble(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components
    )
    return _accelerate_factorize(A; sym = sym, kind = kind, kwargs...)
end

"""
    accelerate_solve(A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kind = :auto, kwargs...) -> Vector
    accelerate_solve(a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
                     symmetrize = false, sym = :auto, kind = :auto, kwargs...) -> VectorElement

Directly solve `A u = F` (or `assemble(a, l)` system) using Apple Accelerate direct factorization.

Requires macOS and [AppleAccelerate.jl](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl);
call `using AppleAccelerate` before calling this function.

See also [`accelerate_factorize`](@ref), [`refactor!`](@ref), [`pde_solve`](@ref).
"""
function accelerate_solve(A::SparseMatrixCSC, F::AbstractVector; kwargs...)
    if !Sys.isapple()
        throw(ArgumentError("AppleAccelerate is only supported on macOS (darwin)."))
    end
    return _accelerate_solve(A, F; kwargs...)
end

function accelerate_solve(
        a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
        symmetrize::Bool = false, sym = :auto, kind = :auto, kwargs...
)
    if !Sys.isapple()
        throw(ArgumentError("AppleAccelerate is only supported on macOS (darwin)."))
    end
    A, F = assemble(
        a, l; dirichlet = dirichlet, dirichlet_components = dirichlet_components, symmetrize = symmetrize
    )
    sym_effective = (sym === :auto && symmetrize) ? :spd : sym
    u = _accelerate_solve(A, F; sym = sym_effective, kind = kind, kwargs...)
    return element(trial_space(a), u)
end

"""
    accelerate_refactor!(fact::AccelerateFactorization, A::SparseMatrixCSC) -> AccelerateFactorization

Recompute the numeric factorization of `A` stored in `fact` **reusing the existing symbolic
factorization** (fill-reducing ordering and sparsity analysis). `A` must have the exact same
sparsity pattern as the matrix originally factored.

See also [`refactor!`](@ref), [`accelerate_factorize`](@ref).
"""
function accelerate_refactor!(fact::AccelerateFactorization, A::SparseMatrixCSC)
    if !Sys.isapple()
        throw(ArgumentError("AppleAccelerate is only supported on macOS (darwin)."))
    end
    return _accelerate_refactor!(fact, A)
end

function _accelerate_factorize(::Any; kwargs...)
    return error(
        "accelerate_factorize requires AppleAccelerate.jl. Add `using AppleAccelerate` before calling this function.",
    )
end

function _accelerate_solve(::Any, ::Any; kwargs...)
    return error(
        "accelerate_solve requires AppleAccelerate.jl. Add `using AppleAccelerate` before calling this function.",
    )
end

function _accelerate_refactor!(::Any, ::Any)
    return error(
        "accelerate_refactor! requires AppleAccelerate.jl. Add `using AppleAccelerate` before calling this function.",
    )
end
