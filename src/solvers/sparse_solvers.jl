# sparse_solvers.jl
#
# Unified interface for sparse linear solvers and factorization reuse:
# - `sparse_factorize`: dispatches to SuiteSparse, AppleAccelerate, MUMPS, or Sparspak
# - `sparse_refactor!`: recomputes numerical factorization in-place reusing symbolic analysis

"""
    sparse_factorize(A::SparseMatrixCSC; solver::Symbol = :default, sym = :auto, kwargs...) -> Factorization
    sparse_factorize(a::BilinearForm; solver::Symbol = :default, sym = :auto,
                     dirichlet = nothing, dirichlet_components = nothing, kwargs...) -> Factorization

Compute the sparse direct factorization of `A` (or the assembled matrix of `a`) using the
requested solver backend.

# Solvers (`solver`)
- `:default` or `:suitesparse`: SuiteSparse (CHOLMOD for SPD/symmetric, UMFPACK for unsymmetric).
- `:accelerate`: Apple Accelerate native `libSparse` on macOS (Cholesky, LDLᵀ, LUTPP, QR).
- `:mumps`: MUMPS multifrontal parallel direct solver.
- `:sparspak`: pure-Julia sparse direct LU, zero binary dependencies.

# Symmetry options
- `:auto` detects matrix symmetry. A symmetric matrix is tried as positive definite
  (SuiteSparse and Apple Accelerate only when its diagonal is positive) and refactored as
  symmetric indefinite when that fails. On SuiteSparse and Apple Accelerate, a singular
  matrix with a zero pivot throws a `SingularException`; a nearly singular one may return a
  large solution without an error. So may a singular one on Apple Accelerate, whose ordering
  can round the zero pivot of a pure-Neumann Laplacian to about `eps`. See the docstring of
  each backend's `*_factorize`.
- `:spd`, `:definite`, or `1` selects symmetric positive definite.
- `:symmetric` or `2` selects general symmetric.
- `:unsymmetric` or `0` selects general unsymmetric.

Ignored by `:sparspak`, which always factors as general unsymmetric LU. It too throws a
`SingularException` for a singular matrix, with index `0` rather than a pivot position.

`kwargs` go to the chosen backend's `*_factorize`, which throws on a keyword it does not
take. `:sparspak` takes none and throws an `ArgumentError`.

A `SparseMatrixCSR` (`SparseMatricesCSR.jl`) is also accepted: converted to `SparseMatrixCSC`
first (no backend has a native CSR solve path -- see `docs/src/internals/csr_solvers.md`),
then factored exactly as above.

The form method refuses `symmetrize` with an `ArgumentError`: it also changes the load
vector. Call `A, F = assemble(a, l; symmetrize = true)` and factor the matrix `A`.

# Examples

```julia
using Bramble: sparse_factorize
fact = sparse_factorize(A; solver = :suitesparse, sym = :spd)
u = fact \\ F
```

See also [`refactor!`](@ref), [`pde_solve`](@ref), [`suitesparse_factorize`](@ref),
[`suitesparse_qr_factorize`](@ref), [`accelerate_factorize`](@ref), [`mumps_factorize`](@ref),
[`sparspak_factorize`](@ref).
"""
function sparse_factorize(A::SparseMatrixCSC; solver::Symbol = :default, sym = :auto, kwargs...)
    if solver === :default || solver === :suitesparse
        return suitesparse_factorize(A; sym = sym, kwargs...)
    elseif solver === :accelerate
        return accelerate_factorize(A; sym = sym, kwargs...)
    elseif solver === :mumps
        return mumps_factorize(A; sym = sym, kwargs...)
    elseif solver === :sparspak
        _reject_unknown_kwargs(:sparse_factorize, "solver = :sparspak", kwargs)
        return sparspak_factorize(A)
    else
        throw(
            ArgumentError(
            "Unknown solver: $solver. Expected :default, :suitesparse, :accelerate, :mumps, or :sparspak.",
        ),
        )
    end
end

# A backend that takes no keyword beyond `sym` cannot ignore the rest: a typo would pass.
# `allowed` lists the keywords the route does take.
function _reject_unknown_kwargs(name::Symbol, route::String, kwargs, allowed = ())
    bad = [k for k in keys(kwargs) if !(k in allowed)]
    isempty(bad) || throw(
        ArgumentError(
        "$name($route) does not accept the keyword(s) $(join(bad, ", ")). " *
        (isempty(allowed) ? "It takes none." : "It takes $(join(allowed, ", ")).")
    ),
    )
    return nothing
end

# `symmetrize` belongs to `assemble(a, l; symmetrize = true)`, which also symmetrises the
# load vector. A factorization of `a` alone has no matching `F`, so it refuses the keyword.
function _reject_factorize_symmetrize(name::Symbol, kwargs)
    haskey(kwargs, :symmetrize) && throw(
        ArgumentError(
        "$name(a::BilinearForm) does not accept `symmetrize`: it also changes the load " *
        "vector, which a factorization of `a` alone cannot match. Assemble with " *
        "`A, F = assemble(a, l; symmetrize = true)` and call $name(A) on the matrix."
    ),
    )
    return nothing
end

function sparse_factorize(
        a::BilinearForm; solver::Symbol = :default, sym = :auto,
        dirichlet = nothing, dirichlet_components = nothing, kwargs...
)
    _reject_factorize_symmetrize(:sparse_factorize, kwargs)
    A = assemble(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components
    )
    A = _as_csc(A)
    return sparse_factorize(A; solver = solver, sym = sym, kwargs...)
end

# `SparseMatricesCSR` is a weakdep, so `SparseMatrixCSR` cannot be named in `src/`: `_is_csr`
# asks whether `BrambleSparseMatricesCSRExt` is loaded and whether `A` is an instance of the type
# that extension bound. No backend solves a CSR matrix natively
# (`docs/src/internals/csr_solvers.md`), so a CSR `A` is converted to CSC in O(nnz) and the CSC
# methods take it from there.
_csr_extension() = Base.get_extension(Bramble, :BrambleSparseMatricesCSRExt)

function _is_csr(A)
    ext = _csr_extension()
    return ext !== nothing && A isa ext.SparseMatrixCSR
end

function _csr_to_csc(A::AbstractMatrix)
    m, n = size(A)
    rowptr, colval, nzval = A.rowptr, A.colval, A.nzval
    # a zero-based `sparsecsr(Val(0), ...)` stores `rowptr` and `colval` from 0: read the
    # base off the data and shift both to one-based (`o == 0` for the usual one-based case)
    o = 1 - Int(rowptr[1])
    I = Vector{Int}(undef, length(nzval))
    @inbounds for i in 1:m, k in (rowptr[i] + o):(rowptr[i + 1] + o - 1)

        I[k] = i
    end
    return sparse(I, o == 0 ? colval : colval .+ o, nzval, m, n)
end

# The assembled matrix of a `BilinearForm` is CSR on a `csr_backend()` mesh, and no backend
# solver takes one (`qr` would silently densify it): the form methods pass it through here.
_as_csc(A) = _is_csr(A) ? _csr_to_csc(A) : A

# A dense-backed form assembles into a `Matrix`, which no backend takes either. The scan is
# O(n²), once per solve, on a matrix whose assembly is already O(n²).
_as_csc(A::Matrix) = sparse(A)

# The `_X_factorize`/`_X_solve` fallbacks of the backend solvers (and the ILU ones) say
# "requires X.jl" only when the extension is missing. With it loaded, a matrix type its
# methods do not take (a `Diagonal`, say) gets this error naming the type instead.
_ext_loaded(ext::Symbol) = Base.get_extension(Bramble, ext) !== nothing

function _throw_unsupported_matrix(name::String, A)
    throw(ArgumentError("$name takes a SparseMatrixCSC; got a $(typeof(A))."))
end

# Only a `SparseMatrixCSC` is supported (test/solvers/sparse_solvers.jl keeps the `MethodError`).
# This method states that, and converts a CSR matrix and delegates back, so every
# `AbstractMatrix` argument ends at a `SparseMatrixCSC` method or a `MethodError`.
function sparse_factorize(A::AbstractMatrix; kwargs...)
    _is_csr(A) && return sparse_factorize(_csr_to_csc(A); kwargs...)
    throw(MethodError(sparse_factorize, (A,)))
end

"""
    refactor!(fact::Factorization, A::SparseMatrixCSC) -> Factorization
    refactor!(fact::Factorization, a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing) -> Factorization

Recompute the numerical values of `fact` for updated matrix `A` (or assembled bilinear form `a`)
**reusing the existing symbolic factorization** (fill-reducing ordering and elimination tree).
`A` must have the exact same sparsity pattern as the matrix originally factored.

Dispatches automatically via multiple dispatch to the appropriate backend.
- [`SuiteSparseFactorization`](@ref) updates CHOLMOD or UMFPACK numeric values.
- [`AccelerateFactorization`](@ref) updates Apple Accelerate `libSparse` numeric values.
- [`MUMPSFactorization`](@ref) updates MUMPS multifrontal numerical factorization (`job = 2`).
- [`SparspakFactorization`](@ref) updates Sparspak's numeric LU values.

See also [`sparse_factorize`](@ref), [`pde_solve`](@ref).
"""
function refactor!(fact::SuiteSparseFactorization, A::SparseMatrixCSC)
    return suitesparse_refactor!(fact, A)
end

function refactor!(fact::AccelerateFactorization, A::SparseMatrixCSC)
    return accelerate_refactor!(fact, A)
end

function refactor!(fact::MUMPSFactorization, A::SparseMatrixCSC)
    return mumps_refactor!(fact, A)
end

function refactor!(fact::SparspakFactorization, A::SparseMatrixCSC)
    return sparspak_refactor!(fact, A)
end

function refactor!(
        fact::Factorization, a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing
)
    A = _as_csc(assemble(a; dirichlet = dirichlet, dirichlet_components = dirichlet_components))
    return refactor!(fact, A)
end

function refactor!(fact::Factorization, A::Any)
    _is_csr(A) && return refactor!(fact, _csr_to_csc(A))
    throw(
        ArgumentError(
        "refactor! is not supported for factorization of type $(typeof(fact)) and matrix type $(typeof(A)). Expected SparseMatrixCSC.",
    ),
    )
end

"""
    sparse_refactor!(fact::Factorization, A::SparseMatrixCSC) -> Factorization
    sparse_refactor!(fact::Factorization, a::BilinearForm; kwargs...) -> Factorization

Alias for [`refactor!`](@ref). Recomputes numerical values of `fact` reusing the existing symbolic factorization.
"""
const sparse_refactor! = refactor!
