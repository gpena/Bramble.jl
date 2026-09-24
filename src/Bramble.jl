module Bramble

import Base: eltype, length
import Base: show, first, last, getindex, setindex!, iterate, size, firstindex, lastindex, axes, eachindex

using SparseArrays: SparseArrays, SparseMatrixCSC, spdiagm, spzeros, rowvals, nonzeros, nzrange, sparse, sparse!,
                    blockdiag,
                    dropzeros!

using LinearAlgebra: I, Diagonal
import LinearAlgebra: mul!, issymmetric, isposdef, ldiv!, Factorization, ×, qr, dot

import Base: copy
using Base: @propagate_inbounds
using Random: rand!

using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference
using QuadGK: gauss

# --- Backend & Execution Policies ---
export backend, gpu_backend, metal_backend, vector_type, matrix_type, backend_types
export csr_backend
export ExecutionPolicy, Serial, Parallel, execution_policy
export CpuPolicy, CpuSerial, CpuThreaded, GpuPolicy, GpuAsync
export CpuBatch

# Backend extension hooks and traits
public _batch_for!, _batch_axis_for!, _batch_scatter_for!, _batch_dot, _batch_dot_masked
public _batch_bilinear_colour_sweep!, _batch_bilinear_band_sweep!
public _batch_linear_colour_sweep!, _batch_linear_band_sweep!
public _allocate_from_pattern, _scatter_position, _scatter_add!, _zero_stored!
public vector, matrix, backend_eye, backend_zeros
public metal_sparse_csr, metal_sparse_csc
public supports_undef_construction
public ka_device, ka_synchronize
public locality, Locality, HostLocality, DeviceLocality
public PRECOMPILE_WORKLOAD

# GPU kernel launch hooks (extended by BrambleKernelAbstractionsExt)
public _gpu_for!, _gpu_scatter_for!
public _launch_half_points!, _launch_spacing!, _launch_half_spacing!
public _launch_refine_indices!
public _launch_restriction!, _launch_restriction_scatter!
public _launch_restriction_nd!, _launch_restriction_scatter_nd!
public _launch_cell_average!, _launch_cell_average_scatter!
public _launch_cell_average_nd!, _launch_cell_average_scatter_nd!
public _launch_difference_onesided!, _launch_difference_centered!, _launch_average_engine!
public _launch_uniform_mesh1d_init!, _launch_nonuniform_mesh1d_metrics!
public _launch_fused_divergence!, _launch_fused_curl2d!, _launch_fused_curl3d!
public _launch_fused_laplacian!, _launch_fused_strain_offdiag!
public _launch_spmv_csr!, _launch_spmm_csr!
public _launch_kron_fused!

# --- Domain & Geometry ---
export box, interval, ×, dim, topo_dim, extrema, point, center, projection, boundary_symbols
export domain, markers, labels
public set, is_collapsed, point_type

# --- Mesh ---
export Mesh1D, MeshnD
export mesh, submeshes, hₘₐₓ, stepsize, locate_cell, iterative_refinement!, change_points!, set_points!
export npoints, points, point, half_points, half_point
export spacing, forward_spacing, half_spacing, spacings, forward_spacings, cell_measure
export indices, boundary_indices, interior_indices, is_boundary_index, index_in_marker, is_uniform
export normal_vector

public AbstractMeshType, MeshMarkers
public mesh_type, hₘᵢₙ, half_spacings, cell_measures
public host_spacings, host_half_spacings, host_points

# --- Spaces & Grid Functions ---
export gridspace, vector_gridspace, space, spaces, ScalarGridSpace, CompositeGridSpace
export ndofs, ncomponents, weights
export VectorElement, element, parent, reshape, components, component_range, component_ranges
export ldiv!
export *
export Rₕ, Rₕ!, avgₕ, avgₕ!
export interpolate_at, interpolation_matrix, πₕ, πₕ!

export innerₕ, inner_Γ, dirac, skew_symmetric
export n
export inner₊, inner₊ₓ, inner₊ᵧ, inner₊₂
export snorm₁ₕ, norm₁ₕ, norm₊, normₕ, norminf_h, norm∞ₕ

public VectorGridSpace, space_type, host_weights

# --- Discrete Differential & Difference Operators ---
export D₋ₓ, D₋ᵧ, D₋₂, ∇ₕ, D₋
export divₕ, divₕ!, curlₕ, curlₕ!, Δₕ, Δₕ!
export εₕ, εₕ!
export D₋ₓ!, D₋ᵧ!, D₋₂!

export D̽ₓ, D̽ᵧ, D̽₂, D̽ₕ, D̽
export D̽ₓ!, D̽ᵧ!, D̽₂!, ∇̽ₕ, ∇̽ₕ!, div̽ₕ, div̽ₕ!, curl̽ₕ, curl̽ₕ!

export Dcₓ, Dcᵧ, Dc₂, Dcₕ, Dc
export Dcₓ!, Dcᵧ!, Dc₂!, ∇cₕ, ∇cₕ!, divcₕ, divcₕ!, curlcₕ, curlcₕ!, εcₕ, εcₕ!

export Dₕₓ, Dₕᵧ, Dₕ₂, Dₕ
export Dₕₓ!, Dₕᵧ!, Dₕ₂!

export jumpₓ, jumpᵧ, jump₂, jumpₕ, jump
export jumpₓ!, jumpᵧ!, jump₂!

export Mₓ, Mᵧ, M₂, Mₕ, Mcₓ, Mcᵧ, Mc₂, Mcₕ
export Mₓ!, Mᵧ!, M₂!, Mcₓ!, Mcᵧ!, Mc₂!

# Forward operators (public for tests and extensions, unexported from default namespace)
public D₊ₓ, D₊ᵧ, D₊₂, ∇₊ₕ, D₊
public D₊ₓ!, D₊ᵧ!, D₊₂!
public div₊ₕ, div₊ₕ!, curl₊ₕ, curl₊ₕ!, ε₊ₕ, ε₊ₕ!
public M₊ₓ, M₊ᵧ, M₊₂, M₊ₕ
public M₊ₓ!, M₊ᵧ!, M₊₂!

# --- Forms, Assembly & Problems ---
export dirichlet_constraints, dirichlet_bc!, symmetrize!
export reaction, reaction_density, reaction!, reaction_density!
export form, assemble, assemble!, assemble_parallel!, allocate_system_matrix, evaluate!
export expression
export is_separable, kronecker_operator, KroneckerLinearOperator
export bandwidths, blockbandwidths
export assemble_add!
export jacobian_pattern, ast_sparsity_detector
export type_cached_assemble!
export Semidiscretization, semidiscretize, mass_matrix, operator_matrix
export SemidiscretizeRHS, semidiscretize_rhs
export SecondOrderSemidiscretization,
       semidiscretize_second_order, damping_matrix, stiffness_matrix, block_mass_matrix
public jacobian!
export jacobian_prototype
export ode_function, ode_problem, linear_problem, nonlinear_problem
export amg_preconditioner
export ilu_preconditioner
export second_order_ode_function, second_order_ode_problem
export SuiteSparseFactorization, suitesparse_factorize, suitesparse_solve, suitesparse_refactor!
export suitesparse_qr_factorize, suitesparse_qr_solve
export AccelerateFactorization, accelerate_factorize, accelerate_solve, accelerate_refactor!
export MUMPSFactorization, mumps_factorize, mumps_solve, mumps_refactor!
export SparspakFactorization, sparspak_factorize, sparspak_solve, sparspak_refactor!
export sparse_factorize, sparse_refactor!, refactor!
export pde_solve
export issymmetric, isposdef
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
