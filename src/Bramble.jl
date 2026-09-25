module Bramble

import Base: eltype, length
import Base: show, first, last, getindex, setindex!, iterate, size, firstindex, lastindex, axes, eachindex

using SparseArrays: SparseArrays, SparseMatrixCSC, spdiagm, spzeros, rowvals, nonzeros, nzrange, sparse, sparse!,
                    blockdiag,
                    dropzeros!

using LinearAlgebra: I, Diagonal
import LinearAlgebra: mul!, issymmetric, isposdef, ldiv!, Factorization, ×, qr, dot, lu, cholesky, ⋅

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

Directly solve `assemble(a) \\ F` (or the linear system `K` represents) for a separable,
constant-coefficient `BilinearForm` by fast diagonalisation, without assembling `a`'s matrix.

`dirichlet`, when given, must request homogeneous Dirichlet conditions on the whole mesh
boundary; `K` alone carries no boundary handling, since a `KroneckerLinearOperator` has no
tensor structure of its own to restrict.

Requires [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl); call `using Kronecker`
before calling this function.

See also: [`kronecker_operator`](@ref), [`KroneckerLinearOperator`](@ref), [`is_separable`](@ref).
"""
function fdm_solve end

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

include("space/gridspace.jl")
include("space/scalar_gridspace.jl")
include("space/vector_gridspace.jl")
include("space/vectorelement.jl")

include("space/operators/projection.jl")
include("space/operators/restriction.jl")
include("space/operators/cell_average.jl")
include("space/operators/shift.jl")
include("space/operators/stencil.jl")
include("space/operators/stencil_matrix.jl")
include("space/operators/difference.jl")
include("space/operators/jump.jl")
include("space/operators/average.jl")
include("space/operators/interpolation.jl")
include("space/operators/vector_calculus.jl")
include("space/operators/normal.jl")
include("space/inner_product.jl")

include("form/ast.jl")
include("form/common.jl")
include("form/expression.jl")
include("form/operators/node_family.jl")
include("form/operators/difference.jl")
include("form/operators/jump.jl")
include("form/operators/average.jl")
include("form/operators/restriction.jl")
include("form/operators/inner.jl")
include("form/operators/normal.jl")
include("form/operators/skew.jl")
include("form/operators/interpolation.jl")
include("form/stencil_eval.jl")
include("form/component.jl")
include("form/stencil_pattern.jl")
include("form/block_extract.jl")
include("form/simplifier.jl")
include("form/dirichlet_constraints.jl")
include("form/linear.jl")
include("form/bilinear.jl")
include("form/reaction.jl")
include("form/bilinear_traversal.jl")
include("form/bilinear_pattern.jl")
include("form/bilinear_execution.jl")
include("form/kronecker.jl")
include("form/assemble_add.jl")
include("form/jacobian_pattern.jl")
include("form/type_cached_assemble.jl")
include("form/symmetry.jl")
include("form/semidiscrete_constraints.jl")
include("form/semidiscrete.jl")
include("form/semidiscrete_rhs.jl")
include("form/semidiscrete_problems.jl")
include("form/second_order_semidiscrete.jl")
include("form/nonlinear_problem.jl")
include("solvers/amg_preconditioner.jl")
include("solvers/ilu_preconditioner.jl")
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
