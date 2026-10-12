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
    accelerate_factorize(A::SparseMatrixCSC; sym = :auto, kind = :auto) -> AccelerateFactorization
    accelerate_factorize(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
                         sym = :auto, kind = :auto) -> AccelerateFactorization

Compute the sparse direct factorization of `A` (or the assembled matrix of `a`) using Apple
Accelerate's native `libSparse` on macOS.

# Symmetry and factorization options
- `sym = :auto` (default) detects symmetry. A symmetric `A` with a positive diagonal is
  tried with Cholesky (`SparseFactorizationCholesky`), and when that fails (`A` is not
  positive definite) it is refactored as ``LDL^T`` (`SparseFactorizationLDLT`) on the same
  object. A symmetric `A` with a non-positive diagonal uses ``LDL^T`` directly. Any other
  `A` uses threshold partial pivoting LU (`SparseFactorizationLUTPP`). None of these
  reports a singular matrix itself, so `:auto` (and `nothing`) solves once against a fixed
  vector and throws a `SingularException` when the backward error is large. A singular `A`
  with a zero pivot throws, whatever the right-hand side. A nearly singular one may return
  a large solution without an error, and so may a singular one whose zero pivot rounds to
  about `eps`, as a pure-Neumann Laplacian's does.
- `sym = :spd`, `:definite`, or `1` (or `kind = :cholesky`) selects symmetric positive definite Cholesky.
- `sym = :symmetric` or `2` (or `kind = :ldlt`) selects symmetric indefinite ``LDL^T``.
- `sym = :unsymmetric` or `0` (or `kind = :lu` / `:lutpp`) selects general unsymmetric LU with threshold partial pivoting.
- `kind = :qr` selects sparse QR factorization.

Any keyword other than `sym` and `kind` throws a `MethodError`.

A `BilinearForm` on a dense or CSR backend is converted to `SparseMatrixCSC` first, so a
dense-backed form uses `libSparse`; `accelerate_factorize(::Matrix)` itself stays dense
LAPACK.

Requires macOS and [AppleAccelerate.jl](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl);
call `using AppleAccelerate` before calling this function.

# Examples

```julia
using Bramble: accelerate_factorize
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
    _reject_factorize_symmetrize(:accelerate_factorize, kwargs)
    A = assemble(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components
    )
    A = _as_csc(A)
    return _accelerate_factorize(A; sym = sym, kind = kind, kwargs...)
end

"""
    accelerate_factorize(A::AbstractMatrix; sym = :auto, kind = :auto) -> LinearAlgebra.Factorization

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
vocabulary as the sparse methods above.

# Symmetry and factorization options
- `sym = :auto` (default) uses `LinearAlgebra.cholesky` if `A` is symmetric positive definite,
  otherwise `LinearAlgebra.lu`.
- `sym = :spd`, `:definite`, or `1` (or `kind = :cholesky`) uses `LinearAlgebra.cholesky`.
- `sym = :unsymmetric` or `0` (or `kind = :lu`) uses `LinearAlgebra.lu`.
- `kind = :qr` uses `LinearAlgebra.qr`.
- `sym = :symmetric`, `2`, or `kind = :ldlt` is not implemented here. The dense analogue of a
  sparse `LDLᵀ` factorization is `LinearAlgebra.bunchkaufman`, which this method does not
  wrap; call it directly.

See also [`accelerate_solve`](@ref).
"""
function accelerate_factorize(A::AbstractMatrix; sym = :auto, kind = :auto)
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
    accelerate_solve(A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kind = :auto) -> Vector
    accelerate_solve(a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
                     symmetrize = false, sym = :auto, kind = :auto) -> VectorElement

Directly solve `A u = F` (or `assemble(a, l)` system) using Apple Accelerate direct factorization.
`sym` is as in [`accelerate_factorize`](@ref): `symmetrize = true` does not choose Cholesky,
`:auto` still tries it and falls back to ``LDL^T``. Any keyword other than `sym` and `kind`
throws a `MethodError`.

A `BilinearForm` on a dense or CSR backend is converted to `SparseMatrixCSC` first, so a
dense-backed form uses `libSparse`; `accelerate_factorize(::Matrix)` itself stays dense
LAPACK.

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
    A = _as_csc(A)
    u = _accelerate_solve(A, F; sym = sym, kind = kind, kwargs...)
    return element(trial_space(a), u)
end

"""
    accelerate_refactor!(fact::AccelerateFactorization, A::SparseMatrixCSC) -> AccelerateFactorization

Recompute the numeric factorization of `A` stored in `fact` **reusing the existing symbolic
factorization** (fill-reducing ordering and sparsity analysis). `A` must have the exact same
sparsity pattern as the matrix originally factored.

On a factorization from `sym = :auto`, a singular `A` with a zero pivot throws a
`SingularException`, and so does every later solve until `fact` is refactored onto a
nonsingular matrix. When `:auto` chose Cholesky and `A` is not positive definite, `A` is
factored afresh as ``LDL^T``, as `accelerate_factorize` would, without the symbolic reuse.

See also [`refactor!`](@ref), [`accelerate_factorize`](@ref).
"""
function accelerate_refactor!(fact::AccelerateFactorization, A::SparseMatrixCSC)
    if !Sys.isapple()
        throw(ArgumentError("AppleAccelerate is only supported on macOS (darwin)."))
    end
    return _accelerate_refactor!(fact, A)
end

function _accelerate_factorize(A; kwargs...)
    _ext_loaded(:BrambleAppleAccelerateExt) && _throw_unsupported_matrix("accelerate_factorize", A)
    return error(
        "accelerate_factorize requires AppleAccelerate.jl. Add `using AppleAccelerate` before calling this function.",
    )
end

function _accelerate_solve(A, ::Any; kwargs...)
    _ext_loaded(:BrambleAppleAccelerateExt) && _throw_unsupported_matrix("accelerate_solve", A)
    return error(
        "accelerate_solve requires AppleAccelerate.jl. Add `using AppleAccelerate` before calling this function.",
    )
end

function _accelerate_refactor!(::Any, ::Any)
    return error(
        "accelerate_refactor! requires AppleAccelerate.jl. Add `using AppleAccelerate` before calling this function.",
    )
end
