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

With `ForwardDiff.jl` loaded, a square `ForwardDiff.Dual` matrix (or a form whose
coefficients are `Dual`) is factored through its `Float64` values, by the backend and
keywords above, and the result keeps the matrix: `pde_solve(fact, F)`, `fact \\ F` and
`ldiv!(uₕ, fact, F)` give the derivative of `pde_solve(A, F)`, every partial solved against
that one factor. A non-square `Dual` matrix throws an `ArgumentError`, as does `refactor!` of
such a factorization.

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
    A = _assembled_csc(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components
    )
    return sparse_factorize(A; solver = solver, sym = sym, kwargs...)
end

# The assembled matrix of a `BilinearForm` is CSR on a `csr_backend()` mesh, and no backend
# solver takes one (`qr` would silently densify it): the form methods pass it through
# `_as_csc`. `SparseMatricesCSR` is a weakdep, so `BrambleSparseMatricesCSRExt` adds the
# `SparseMatrixCSR` methods of `_as_csc`, `sparse_factorize`, `refactor!` and `pde_solve`:
# an O(nnz) conversion to CSC, after which the CSC methods take over
# (`docs/src/internals/csr_solvers.md`).
_as_csc(A) = A

# A dense-backed form assembles into a `Matrix`, which no backend takes either. The scan is
# O(n²), once per solve, on a matrix whose assembly is already O(n²).
_as_csc(A::Matrix) = sparse(A)

# The `_X_factorize`/`_X_solve` fallbacks of the backend solvers (and the ILU and AMG ones) say
# "requires X.jl" only when the extension is missing. With it loaded, an argument type its
# methods do not take (a `Diagonal`, say) gets this error naming the type instead.
_ext_loaded(ext::Symbol) = Base.get_extension(Bramble, ext) !== nothing

# The form methods of every backend assemble, then convert the matrix with `_as_csc`. With a
# load, the pair `(A, F)` comes back and only `A` is converted.
_assembled_csc(a; kwargs...) = _as_csc(assemble(a; kwargs...))

function _assembled_csc(a, l; kwargs...)
    A, F = assemble(a, l; kwargs...)
    return _as_csc(A), F
end

# One reading of the `sym` vocabulary shared by the backends: `:spd`, `:symmetric`,
# `:unsymmetric` or `:auto` for the accepted spellings (`nothing` is `:auto`, the integers
# 1, 2 and 0 the first three), `:unknown` otherwise. Each backend keeps its own error text
# for `:unknown` and its own rule for `:auto`.
function _normalize_sym(sym)
    (sym === :spd || sym === :definite || sym == 1) && return :spd
    (sym === :symmetric || sym == 2) && return :symmetric
    (sym === :unsymmetric || sym == 0) && return :unsymmetric
    (sym === :auto || sym === nothing) && return :auto
    return :unknown
end

# The `!Sys.isapple()` refusal of the Accelerate entry points.
function _require_macos()
    Sys.isapple() || throw(ArgumentError("AppleAccelerate is only supported on macOS (darwin)."))
    return nothing
end

# The body of the 16 stubs a backend or preconditioner keeps for its extension. With the
# extension loaded, an argument its methods do not take lands here, and the error names the
# type. Without it, the error names the package to load. `ext` is the extension module name
# (`:BrambleSparspakExt`), from which the package name follows.
function _ext_fallback(ext::Symbol, name::AbstractString, A, expected::AbstractString = "a SparseMatrixCSC")
    _ext_loaded(ext) && _throw_unsupported_matrix(name, A, expected)
    pkg = String(ext)[(length("Bramble") + 1):(end - length("Ext"))]
    return error("$name requires $pkg.jl. Add `using $pkg` before calling this function.")
end

function _throw_unsupported_matrix(name::String, A, expected::String = "a SparseMatrixCSC")
    throw(ArgumentError("$name takes $expected; got a $(typeof(A))."))
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

# Throws
Besides the same-pattern rule, a backend refuses a refactor in these cases. The backend
docstrings give the detail and the exception types.
- A Cholesky or symmetric factorization refuses a matrix that is not symmetric (not Hermitian
  for CHOLMOD and Accelerate) with an `ArgumentError`, because it reads one triangle. It
  checks first, so `fact` keeps its factors and still solves the previous matrix. MUMPS
  (`:spd`, `:symmetric` and `:auto`, also after its fallback) and Accelerate's Cholesky and
  ``LDL^T`` refuse this way. SuiteSparse does when it holds CHOLMOD. A SuiteSparse
  factorization that fell back to UMFPACK is an LU one and takes any matrix of its pattern,
  as Sparspak's does. To refactor an unsymmetric matrix, factorize with
  `sym = :unsymmetric`.
- An `:auto` factorization follows an indefinite matrix where a chosen `:spd` one throws:
  MUMPS and Accelerate move to ``LDL^T`` and SuiteSparse to LU.
- A refactor refused in the numeric step (a singular matrix, or one that is not positive
  definite under `:spd`) leaves `fact` refusing every solve until a refactor succeeds.

`refactor!(fact, a; dirichlet)` assembles `a` without symmetrising it, so a symmetric
factorization refuses a matrix with Dirichlet rows cleared. Assemble with
`symmetrize = true` and call the matrix method instead.

See also [`sparse_factorize`](@ref), [`pde_solve`](@ref), [`suitesparse_refactor!`](@ref),
[`accelerate_refactor!`](@ref), [`mumps_refactor!`](@ref), [`sparspak_refactor!`](@ref).
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
    A = _assembled_csc(a; dirichlet = dirichlet, dirichlet_components = dirichlet_components)
    return refactor!(fact, A)
end

function refactor!(fact::Factorization, A::Any)
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
