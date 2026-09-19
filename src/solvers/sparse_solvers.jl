# sparse_solvers.jl
#
# Unified interface for sparse linear solvers and factorization reuse:
# - `sparse_factorize`: dispatches to SuiteSparse, AppleAccelerate, or MUMPS
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
- `:auto`: automatically detect matrix symmetry (and diagonal positivity).
- `:spd`, `:definite`, or `1`: symmetric positive definite.
- `:symmetric` or `2`: general symmetric.
- `:unsymmetric` or `0`: general unsymmetric.

Ignored by `:sparspak`, which always factors as general unsymmetric LU.

# Examples

```julia
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
        return sparspak_factorize(A)
    else
        throw(
            ArgumentError(
            "Unknown solver: $solver. Expected :default, :suitesparse, :accelerate, :mumps, or :sparspak.",
        ),
        )
    end
end

function sparse_factorize(
        a::BilinearForm; solver::Symbol = :default, sym = :auto,
        dirichlet = nothing, dirichlet_components = nothing, kwargs...
)
    A = assemble(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components
    )
    return sparse_factorize(A; solver = solver, sym = sym, kwargs...)
end

# `assemble(a::BilinearForm)` is generic over the backend's matrix type (gpena/Bramble.jl#12)
# and a dense-backed form assembles into a `Matrix`, not a `SparseMatrixCSC` -- so the call
# above genuinely can reach here. `sparse_factorize` only ever supported `SparseMatrixCSC`
# (test/form/sparse_solvers.jl: "Type safety: sparse_factorize only accepts SparseMatrixCSC",
# `@test_throws MethodError`), and this states that as an actual method instead of leaving it
# an inference-only gap: same exception a plain dispatch failure would raise, just reachable
# from an analysis that has to consider every backend a `BilinearForm` could name.
function sparse_factorize(A::AbstractMatrix; kwargs...)
    throw(MethodError(sparse_factorize, (A,)))
end

"""
    refactor!(fact::Factorization, A::SparseMatrixCSC) -> Factorization
    refactor!(fact::Factorization, a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing) -> Factorization

Recompute the numerical values of `fact` for updated matrix `A` (or assembled bilinear form `a`)
**reusing the existing symbolic factorization** (fill-reducing ordering and elimination tree).
`A` must have the exact same sparsity pattern as the matrix originally factored.

Dispatches automatically via multiple dispatch to the appropriate backend:
- [`SuiteSparseFactorization`](@ref): updates CHOLMOD or UMFPACK numeric values.
- [`AccelerateFactorization`](@ref): updates Apple Accelerate `libSparse` numeric values.
- [`MUMPSFactorization`](@ref): updates MUMPS multifrontal numerical factorization (`job = 2`).
- [`SparspakFactorization`](@ref): updates Sparspak's numeric LU values.

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
    A = assemble(a; dirichlet = dirichlet, dirichlet_components = dirichlet_components)
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
