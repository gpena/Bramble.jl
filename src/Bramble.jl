module Bramble

import Base: eltype, length
import Base: show, first, last, getindex, setindex!, iterate, size, firstindex, lastindex, axes, eachindex

using SparseArrays: SparseArrays, SparseMatrixCSC, spdiagm, spzeros, rowvals, nonzeros, nzrange, sparse, sparse!,
                    blockdiag,
                    dropzeros!

using LinearAlgebra: I, Diagonal
import LinearAlgebra: mul!, issymmetric, isposdef, ldiv!, Factorization, ×, qr, dot, lu, cholesky, ⋅, norm

import Base: copy
using Base: @propagate_inbounds
import Random
using Random: rand!

using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference
using QuadGK: gauss
import GPUArraysCore

include("api.jl")

# --- Extension Stubs ---
"""
    fdm_solve(a::BilinearForm, F::AbstractVector; dirichlet = nothing) -> Vector
    fdm_solve(K::KroneckerLinearOperator, F::AbstractVector) -> Vector

Directly solve `assemble(a) \\ F` (or the linear system `K` represents) for a Laplacian-like
`BilinearForm`, without assembling `a`'s matrix. A form is Laplacian-like when it is
separable and every term differs from one mass per axis on at most one axis. Symmetric 1D
operators are solved by fast diagonalisation; non-symmetric ones (advection terms) by a
complex generalised Schur factorisation per axis and a triangular back substitution.
A mixed-derivative form is refused: precondition a Krylov solver with
[`fdm_preconditioner`](@ref) instead. [`fdm_factorize`](@ref) and [`fdm_solve!`](@ref) split
the factorisation from the solve.

`dirichlet`, when given, must request homogeneous Dirichlet conditions on the whole mesh
boundary; `K` alone carries no boundary handling, since a `KroneckerLinearOperator` has no
tensor structure of its own to restrict.

Requires [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl); call `using Kronecker`
before calling this function.

See also: [`kronecker_operator`](@ref), [`KroneckerLinearOperator`](@ref), [`is_separable`](@ref).
"""
function fdm_solve end

"""
    fdm_preconditioner(a::BilinearForm; dirichlet = nothing) -> FDMPreconditioner

A preconditioner for `assemble(a; dirichlet)` that applies the fast-diagonalisation inverse
of `a`'s Laplacian-like part, factorised once, without assembling `a`'s matrix. The
Laplacian-like part keeps the terms [`fdm_solve`](@ref) can solve: with one mass per axis,
every term that differs from the masses on at most one axis. Every term differing on two or
more axes (a mixed derivative such as `innerₕ(D₋ₓ(D₋ᵧ(u)), v)`, or a coefficient varying
along two axes) is left out of the preconditioner, though the Krylov solver still sees it in
`assemble(a)`. For a form `fdm_solve` accepts, the preconditioner is the exact inverse.

Use it for a mixed-derivative form whose cross term the diffusion dominates, so that what
it leaves out is small beside what it inverts. The larger the cross term, the more work it
leaves to the Krylov solver. A form `fdm_solve` accepts needs no Krylov solver at all.

`dirichlet` is `nothing` (the unconstrained system) or `:boundary` (homogeneous Dirichlet on
the whole mesh boundary): there the boundary rows of `assemble(a; dirichlet = :boundary)`
are identity rows, so the preconditioner is the identity on them and the interior
factorisation inside. Construction costs `O(n_d^3)` per axis; an `ldiv!` on host vectors
costs `O(N Σ_d n_d)` and allocates nothing. The preconditioner is host-only.

Requires [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl); call `using Kronecker`
before calling this function.

# Throws

  - `ArgumentError` saying `fdm_preconditioner` does not support the form, with the reason,
    for a form with no usable Laplacian-like part: `a` is not separable, it is posed on a
    composite space, a mass is not symmetric positive definite, an axis has no term of its
    own once the two-axis terms are left out (the form has no Laplacian-like part), or that
    part is singular. These are `fdm_solve`'s refusals, a two-axis term apart.
  - `ArgumentError`: `dirichlet` is neither `nothing` nor `:boundary`.

# Examples

```julia
using Bramble, Kronecker, LinearSolve
using Bramble: D₋ₓ, D₋ᵧ
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (33, 25), (false, false))
Wₕ = gridspace(Ωₕ)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                           0.25 * innerₕ(D₋ₓ(D₋ᵧ(u)), v))
P = fdm_preconditioner(a)
A = assemble(a)
F = rand(ndofs(Wₕ))
sol = solve(LinearProblem(A, F), KrylovJL_GMRES(); Pl = P)
```

See also: [`FDMPreconditioner`](@ref), [`fdm_solve`](@ref), [`jacobi_preconditioner`](@ref).
"""
function fdm_preconditioner end

"""
    fdm_factorize(a::BilinearForm; dirichlet = nothing)
    fdm_factorize(K::KroneckerLinearOperator; dirichlet = nothing)

Factorise, once, the system [`fdm_solve`](@ref) solves, so that each further right-hand side
costs only the solve. The result supports [`fdm_solve!`](@ref), `ldiv!(x, f, F)` (the same
solve) and `size`; its type is internal. `fdm_solve(a, F)` is `fdm_factorize(a)` followed by
one solve.

`a` must be a form `fdm_solve` accepts, and `fdm_factorize(a; dirichlet)` refuses what
`fdm_solve(a, F; dirichlet)` refuses. `K` must be Laplacian-like, as for `fdm_solve(K, F)`;
unlike that method, `fdm_factorize(K)` also takes `dirichlet = :boundary`. With
`dirichlet = :boundary`, each right-hand side must be zero on the boundary and the solution
is zero there. The factorisation is built from the mesh and coefficients as they are when
`fdm_factorize` is called: after [`change_points!`](@ref) or a change of a coefficient (a
`Ref` one included), refill it with [`fdm_factorize!`](@ref), or call `fdm_factorize` again.
Construction costs `O(n_d^3)` per axis; a solve on host vectors costs `O(N Σ_d n_d)` and
allocates nothing. The factorisation is host-only.

Requires [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl); call `using Kronecker`
before calling this function.

# Throws

  - `ArgumentError` saying `fdm_solve` does not support the form or operator, with the
    reason, for one that is not Laplacian-like or is singular.
  - `ArgumentError`: `dirichlet` is neither `nothing` nor `:boundary`.
  - `ArgumentError` naming `change_points!`: `K`'s mesh was mutated in place after `K` was
    built; build the operator again.

# Examples

```julia
using Bramble, Kronecker
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (25, 19), (false, false))
Wₕ = gridspace(Ωₕ)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
f = fdm_factorize(a)
x = zeros(ndofs(Wₕ))
for k in 1:3
    F = rand(ndofs(Wₕ))
    fdm_solve!(x, f, F)   # x ≈ assemble(a) \\ F
end
```

See also: [`fdm_solve!`](@ref), [`fdm_solve`](@ref), [`fdm_preconditioner`](@ref).
"""
function fdm_factorize end

"""
    fdm_factorize!(f, a::BilinearForm) -> f
    fdm_factorize!(f, K::KroneckerLinearOperator) -> f

Refill the factorisation `f`, which [`fdm_factorize`](@ref) returned, from `a` or `K`, and
return `f`. Use it in place of a new `fdm_factorize` when the structure is the same and only
the numbers changed: a coefficient that is a scalar or a `Ref` took a new value, or
[`change_points!`](@ref) moved the grid points and `a` is a form on a new `gridspace` (or `K`
a new [`kronecker_operator`](@ref)) built after it. `f` keeps its `dirichlet`: there is no
keyword here. Afterwards `fdm_solve!(x, f, F)` solves the new system, and agrees bitwise with
the solve of a fresh `fdm_factorize`, unless a term's coefficient is now zero: that term
keeps its place, and the two agree to rounding. Only `Float32` and `Float64` are supported.

With `K`, a refill allocates nothing once warm. With `a`, it allocates nothing once warm for
the form `f` was built (or last refilled) from, when only scalar or `Ref` coefficients
changed. A different form is projected again, and so is one that reads a grid-function
coefficient, so that an edit made with `Rₕ!` is read and never ignored; both allocate. The
operator or form must have the structure `f` was built for: the same dimension, sizes,
eltype, number of terms, mass term on each axis, terms left out at build (a zero
coefficient, or a factor that is zero on the solved points), and symmetric or non-symmetric
(Schur) solve. Anything else needs a new `fdm_factorize`. A `KroneckerLinearOperator` backed by a
device is refused: the refill is host-only. An [`FDMPreconditioner`](@ref) is not refilled.

Requires [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl); call `using Kronecker`
before calling this function.

# Throws

  - `ArgumentError` starting `fdm_factorize! cannot refill this factorisation`, then the
    reason: `a` or `K` differs from `f` in dimension, size, eltype, number of terms, mass
    term, a term left out at build that is present now, or whether the solve is symmetric
    or Schur, or `K` is backed by a device. `f` is left as it was and still solves.
  - `ArgumentError`: the refilled system is one `fdm_solve` refuses, singular or with a mass
    that is not symmetric positive definite. `f` then cannot solve, and
    [`fdm_solve!`](@ref) says so, until a later refill succeeds.
  - `ArgumentError` naming `change_points!`: the mesh of `a` or `K` was mutated in place
    after it was built; build the form on a new `gridspace` (or the operator again).
  - `ArgumentError` saying `fdm_solve` does not support the form, with the reason: `a` is on
    a composite space, or is not separable.

# Examples

```julia
using Bramble, Kronecker
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (25, 19), (false, false))
Wₕ = gridspace(Ωₕ)
c = Ref(1.0)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + c * inner₊(∇ₕ(u), ∇ₕ(v)))
f = fdm_factorize(a)
F = rand(ndofs(Wₕ))
x = zeros(ndofs(Wₕ))
for k in 1:3
    c[] = 10.0^k
    fdm_factorize!(f, a)   # no allocation once warm: only a Ref changed
    fdm_solve!(x, f, F)    # x ≈ assemble(a) \\ F
end
```

See also: [`fdm_factorize`](@ref), [`fdm_solve!`](@ref), [`fdm_solve`](@ref).
"""
function fdm_factorize! end

"""
    fdm_solve!(x::AbstractVector, f, F::AbstractVector) -> x

Solve into `x`, with the factorisation `f` that [`fdm_factorize`](@ref) returned, the system
it was built for, right-hand side `F`. With `dirichlet = :boundary`, `x` is zero on the
boundary. `x` may alias `F`. On host vectors it allocates nothing.

Requires [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl); call `using Kronecker`
before calling this function.

# Throws

  - `DimensionMismatch`: `x` or `F` does not have `size(f, 1)` entries.
  - `ArgumentError`: `x` or `F` is not 1-based.
  - `ArgumentError` naming `fdm_factorize!`: the last [`fdm_factorize!`](@ref) of `f` was
    refused because the system is singular or its mass is not symmetric positive definite.
    Refill `f` before solving again.

See also: [`fdm_factorize`](@ref), [`fdm_factorize!`](@ref), [`fdm_solve`](@ref).
"""
function fdm_solve! end

"""
    _launch_spmv_csr!(y, rowPtr, colVal, nzVal, x, α, β) -> Nothing

Row-parallel sparse matrix-vector product `y .= α .* (A * x) .+ β .* y`, where `A`'s
storage is given as the raw CSR arrays `rowPtr`, `colVal`, `nzVal` -- never a struct
wrapping them, since a struct nesting a device array fails `KernelAbstractions` kernel
compilation. One work item owns one output row, so there are no write conflicts and no
atomics.

Requires `using KernelAbstractions`; the real method is supplied by
`BrambleKernelAbstractionsExt`.

# Throws
- `ErrorException`: if `KernelAbstractions` is not loaded.
"""
function _launch_spmv_csr!(y, rowPtr, colVal, nzVal, x, α, β)
    return _throw_no_ka_sparse_kernel("_launch_spmv_csr!")
end

"""
    _launch_spmm_csr!(C, rowPtr, colVal, nzVal, B, α, β) -> Nothing

Row-parallel sparse matrix-matrix product `C .= α .* (A * B) .+ β .* C` for a dense
right-hand side `B`, where `A`'s storage is given as the raw CSR arrays `rowPtr`, `colVal`,
`nzVal`. See [`_launch_spmv_csr!`](@ref) for the extension contract and why the arrays are
passed separately rather than as a struct.

Requires `using KernelAbstractions`; the real method is supplied by
`BrambleKernelAbstractionsExt`.

# Throws
- `ErrorException`: if `KernelAbstractions` is not loaded.
"""
function _launch_spmm_csr!(C, rowPtr, colVal, nzVal, B, α, β)
    return _throw_no_ka_sparse_kernel("_launch_spmm_csr!")
end

@noinline function _throw_no_ka_sparse_kernel(name::String)
    return error(
        "$name has no method loaded. Add `using KernelAbstractions` before calling " *
        "Metal sparse `mul!`.",
    )
end

# --- Submodule Includes ---
include("utils/macros.jl")
include("utils/backend.jl")
include("utils/device_kernels.jl")
include("utils/linear_algebra.jl")
include("utils/backend_profile.jl")

include("geometry/pretty_print.jl")
include("geometry/set.jl")
include("geometry/marker.jl")
include("geometry/domain.jl")

include("mesh/interface.jl")
include("mesh/indices.jl")
include("mesh/constructors.jl")
include("mesh/queries.jl")
include("mesh/marker.jl")
include("mesh/pretty_print.jl")
include("mesh/mesh1d.jl")
include("mesh/meshnd.jl")
# The split of a walk argument names the mesh types and their walk states.
include("utils/batch_split.jl")

include("space/gridspace.jl")
include("space/scalar_gridspace.jl")
include("space/vector_gridspace.jl")
include("space/vectorelement.jl")

include("operators/projection.jl")
include("operators/restriction.jl")
include("operators/cell_average.jl")
include("operators/stencil.jl")
include("operators/stencil_matrix.jl")

# The AST core comes before the first merged operator family (gpena/Bramble.jl#350): a file
# under src/operators/ defines its stencil and its AST nodes together, and the nodes need
# `LazyOp` and `@node_family`. The numerical files that consume differences follow it.
include("ast/ast.jl")
include("ast/common.jl")
include("ast/expression.jl")
include("operators/node_family.jl")
include("operators/difference.jl")
include("operators/shift.jl")

include("operators/jump.jl")
include("operators/average.jl")
include("operators/vector_calculus.jl")
include("space/inner_product.jl")

# The families whose AST half reads the symbolic inner products follow inner.jl:
# interpolation's nodes extend `BilinearProduct`'s methods.
include("operators/region_restriction.jl")
include("operators/inner.jl")
include("operators/normal.jl")
include("operators/skew.jl")
include("operators/interpolation.jl")
include("assembly/stencil_eval.jl")
include("ast/component.jl")
include("ast/stencil_pattern.jl")
include("assembly/block_extract.jl")
include("ast/simplifier.jl")
include("assembly/dirichlet_constraints.jl")
include("assembly/linear.jl")
include("assembly/bilinear.jl")
include("postprocessing/reaction.jl")
include("assembly/bilinear_traversal.jl")
include("assembly/bilinear_pattern.jl")
include("assembly/bilinear_execution.jl")
include("assembly/kronecker_projection.jl")
include("assembly/kronecker.jl")
include("assembly/kronecker_block.jl")
include("assembly/assemble_add.jl")
include("assembly/jacobian_pattern.jl")
include("assembly/type_cached_assemble.jl")
include("assembly/symmetry.jl")
include("assembly/matrix_free.jl")
include("problems/semidiscrete_constraints.jl")
include("problems/semidiscrete.jl")
include("problems/semidiscrete_rhs.jl")
include("problems/semidiscrete_problems.jl")
include("problems/second_order_semidiscrete.jl")
include("problems/nonlinear_problem.jl")
include("solvers/amg_preconditioner.jl")
include("solvers/ilu_preconditioner.jl")
include("solvers/matrix_free_preconditioners.jl")
include("solvers/chebyshev.jl")
include("solvers/gmg_hierarchy.jl")
include("solvers/gmg_transfer.jl")
include("solvers/gmg_smoothers.jl")
include("solvers/gmg_cycles.jl")
include("solvers/suitesparse_solver.jl")
include("solvers/accelerate_solver.jl")
include("solvers/mumps_solver.jl")
include("solvers/sparspak_solver.jl")
include("solvers/sparse_solvers.jl")
include("solvers/pde_solve.jl")

include("exporters/vtk_export.jl")
include("exporters/pgfplots_export.jl")

include("precompile.jl")
end
