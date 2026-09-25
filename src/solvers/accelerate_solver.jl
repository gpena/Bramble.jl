# accelerate_solver.jl
#
# `accelerate_factorize`/`accelerate_solve`/`accelerate_refactor!`: Apple Accelerate
# (`libSparse`) sparse Cholesky, LDLᵀ, and LUTPP factorizations for the sparse linear systems
# Bramble discretizes on macOS. Implemented in `BrambleAppleAccelerateExt`.
#
# The dense `accelerate_factorize(A::AbstractMatrix; ...)` method below is different: it is
# plain `LinearAlgebra.lu`/`cholesky`/`qr`, defined unconditionally here (no macOS guard, no
# `BrambleAppleAccelerateExt` involvement). AppleAccelerate.jl has no dense `lu`/`cholesky`/
# LAPACK bindings to call -- its `__init__` instead registers itself with
# libblastrampoline, so ordinary `LinearAlgebra` calls already run on Accelerate's BLAS/LAPACK
# once `using AppleAccelerate` has been evaluated. This method only gives dense callers the
# same name and `sym`/`kind` vocabulary as the sparse methods; see its own docstring.

"""
    AccelerateFactorization{T} <: Factorization{T}

Wrapper type representing a factorized Apple Accelerate (`libSparse`) linear system,
supporting in-place solves via `LinearAlgebra.ldiv!`, back-substitution via `\\`,
and non-allocating symbolic reuse via `accelerate_refactor!`. Available only on macOS.
"""
abstract type AccelerateFactorization{T} <: Factorization{T} end

"""
    accelerate_factorize(A::SparseMatrixCSC; sym = :auto, kind = :auto, kwargs...) -> AccelerateFactorization
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
    accelerate_factorize(A::AbstractMatrix; sym = :auto, kind = :auto, kwargs...) -> LinearAlgebra.Factorization

Dense factorization of `A`, via ordinary `LinearAlgebra.lu`, `LinearAlgebra.cholesky`, or
`LinearAlgebra.qr` -- **not** Apple Accelerate's `libSparse` used by the `SparseMatrixCSC`
method above, and not gated on macOS.

AppleAccelerate.jl (checked against v0.7.0) defines no dense `lu`, `cholesky`, `getrf`,
`potrf`, `gemm`, or other LAPACK/BLAS bindings for Bramble to call. What its `__init__` does
is register Accelerate with libblastrampoline (`BLAS.lbt_forward`), so plain
`LinearAlgebra.lu`/`cholesky`/`qr` are *already* running on Accelerate's BLAS/LAPACK the
moment `using AppleAccelerate` has been evaluated, on every call site in Bramble or anywhere
else, with no Bramble dispatch involved. This method adds no acceleration beyond that; it
exists so dense callers can spell `accelerate_factorize` with the same `sym`/`kind`
vocabulary as the sparse methods above. `kwargs...` is accepted but unused.

# Symmetry and factorization options
- `sym = :auto` (default): `LinearAlgebra.cholesky` if `A` is symmetric positive definite,
  otherwise `LinearAlgebra.lu`.
- `sym = :spd`, `:definite`, or `1` (or `kind = :cholesky`): `LinearAlgebra.cholesky`.
- `sym = :unsymmetric` or `0` (or `kind = :lu`): `LinearAlgebra.lu`.
- `kind = :qr`: `LinearAlgebra.qr`.
- `sym = :symmetric`, `2`, or `kind = :ldlt`: not implemented here. The dense analogue of a
  sparse `LDLᵀ` factorization is `LinearAlgebra.bunchkaufman`, which this method does not
  wrap; call it directly.

See also [`accelerate_solve`](@ref).
"""
function accelerate_factorize(A::AbstractMatrix; sym = :auto, kind = :auto, kwargs...)
    if kind === :cholesky || sym === :spd || sym === :definite || sym == 1
        return cholesky(A)
    elseif kind === :lu || sym === :unsymmetric || sym == 0
        return lu(A)
    elseif kind === :qr
        return qr(A)
    elseif kind === :ldlt || sym === :symmetric || sym == 2
        throw(
            ArgumentError(
            "accelerate_factorize does not wrap dense symmetric indefinite factorization; call LinearAlgebra.bunchkaufman(A) directly.",
        ),
        )
    elseif (sym === :auto || sym === nothing) && (kind === :auto || kind === nothing)
        return issymmetric(A) && isposdef(A) ? cholesky(A) : lu(A)
    else
        throw(ArgumentError("Unknown factorization option for accelerate_factorize: sym=$sym, kind=$kind."))
    end
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
