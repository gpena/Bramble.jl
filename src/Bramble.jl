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

# --- Backend & Execution Policies ---
export backend, gpu_backend, metal_backend, csr_backend
export Serial, Parallel, vector_type, matrix_type, backend_types, execution_policy

public ExecutionPolicy, CpuPolicy, CpuSerial, CpuThreaded, CpuPolyester, CpuBatch
public GpuPolicy, GpuKernel, GpuAsync
public locality, Locality, HostLocality, DeviceLocality
public vector, matrix, metal_sparse_csr, metal_sparse_csc

# --- Domain & Geometry ---
export box, interval, ×, ⋅, dim, boundary_symbols
export domain, markers, labels

public center, projection, point, topo_dim

# --- Mesh ---
export Mesh1D, MeshnD
export mesh, npoints, points, hₘₐₓ, hₘᵢₙ, iterative_refinement!, normal_vector
export spacing, forward_spacing

public change_points!, set_points!, is_uniform
public half_spacing, spacings, cell_measure, half_point, half_points
public indices, boundary_indices, interior_indices, is_boundary_index, index_in_marker

# --- Spaces & Grid Functions ---
export gridspace, vector_gridspace, space, spaces
export ndofs, ncomponents
export element, components
export Rₕ, Rₕ!, avgₕ, avgₕ!
export interpolate_at, πₕ, πₕ!

export innerₕ, inner_Γ, dirac
export n
export inner₊
export snorm₁ₕ, norm₁ₕ, norm₊, normₕ, norm∞ₕ

public ScalarGridSpace, CompositeGridSpace, VectorGridSpace, VectorElement
public component_range, component_ranges, skew_symmetric
public inner₊ₓ, inner₊ᵧ, inner₊₂
public norminf_h
public weights, interpolation_matrix

# --- Discrete Differential & Difference Operators ---
export ∇ₕ, D₋
export divₕ, divₕ!, curlₕ, curlₕ!, Δₕ, Δₕ!
export εₕ, εₕ!

export D̃, D̃ₕ
export ∇̃ₕ, ∇̃ₕ!, diṽₕ, diṽₕ!, curl̃ₕ, curl̃ₕ!

export Dc, Dcₕ
export ∇cₕ, ∇cₕ!, divcₕ, divcₕ!, curlcₕ, curlcₕ!, εcₕ, εcₕ!

export D̽ₕ, ∇̽ₕ
export div̽ₕ, div̽ₕ!, curl̽ₕ, curl̽ₕ!, ε̽ₕ, ε̽ₕ!, ∇̽ₕ!

export jump, jumpₕ

export Mₕ, Mcₕ

# Coordinate aliases (destructure from the vectorial entities above; public for tests and
# extensions, unexported from default namespace)
public D₋ₓ, D₋ᵧ, D₋₂, D₋ₓ!, D₋ᵧ!, D₋₂!
public D̃ₓ, D̃ᵧ, D̃₂, D̃ₓ!, D̃ᵧ!, D̃₂!
public Dcₓ, Dcᵧ, Dc₂, Dcₓ!, Dcᵧ!, Dc₂!
public D̽ₓ, D̽ᵧ, D̽₂, D̽ₓ!, D̽ᵧ!, D̽₂!
public jumpₓ, jumpᵧ, jump₂, jumpₓ!, jumpᵧ!, jump₂!
public Mₓ, Mᵧ, M₂, Mₓ!, Mᵧ!, M₂!, Mcₓ, Mcᵧ, Mc₂, Mcₓ!, Mcᵧ!, Mc₂!

# Forward operators (public for tests and extensions, unexported from default namespace)
public D₊ₓ, D₊ᵧ, D₊₂, ∇₊ₕ, D₊
public D₊ₓ!, D₊ᵧ!, D₊₂!
public div₊ₕ, div₊ₕ!, curl₊ₕ, curl₊ₕ!, ε₊ₕ, ε₊ₕ!
public M₊ₓ, M₊ᵧ, M₊₂, M₊ₕ
public M₊ₓ!, M₊ᵧ!, M₊₂!

# --- Forms, Assembly & Problems ---
export dirichlet_constraints, dirichlet_bc!, symmetrize!
export form, assemble, assemble!, assemble_add!
export expression
export is_separable, kronecker_operator, KroneckerLinearOperator
export pde_solve
export semidiscretize, semidiscretize_second_order
export ode_function, ode_problem, linear_problem, nonlinear_problem
export second_order_ode_function, second_order_ode_problem
export amg_preconditioner
export ilu_preconditioner

public jacobian!, jacobian_prototype, jacobian_pattern, ast_sparsity_detector
public reaction, reaction_density, reaction!, reaction_density!
public allocate_system_matrix, type_cached_assemble!, evaluate!, assemble_parallel!
public Semidiscretization, SemidiscretizeRHS, SecondOrderSemidiscretization
public mass_matrix, operator_matrix, damping_matrix, stiffness_matrix, block_mass_matrix
public SuiteSparseFactorization, suitesparse_factorize
public AccelerateFactorization
public MUMPSFactorization
public SparspakFactorization
public sparse_factorize, sparse_refactor!, refactor!
public DirichletConstraint

# --- Exporters ---
export export_vtk
export export_pgfplots

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
export fdm_solve

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
