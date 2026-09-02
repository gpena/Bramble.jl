module Bramble

using DocStringExtensions

import Base: eltype, length
import Base: show, first, last, getindex, setindex!, iterate, size, ndims, firstindex,
             lastindex, axes, eachindex

using SparseArrays: SparseMatrixCSC, SparseVector, spdiagm, spzeros,
                    rowvals, nonzeros, nzrange, sparse, sparse!

using FunctionWrappers: FunctionWrapper

using StaticArrays: SVector, @SVector

using LinearAlgebra: I, transpose, dot
import LinearAlgebra: issymmetric, isposdef

import Base: copy
using Base: @propagate_inbounds
using Random: rand!

using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference
using QuadGK: gauss

# Utilities
export backend, metal_backend, vector, matrix, vector_type, matrix_type, backend_types,
       backend_eye, backend_zeros
export ExecutionPolicy, Serial, Parallel, execution_policy
export BrambleFunction, embed_function, has_time

# domain/interval handling functions
export box, interval, ×, dim, topo_dim, tails, point, cartesian_product, center, projection,
       is_collapsed, point_type, get_boundary_symbols, set
export domain, markers, labels

# Mesh handling
export AbstractMeshType, Mesh1D, MeshnD, MeshMarkers
export mesh, mesh_type, submeshes, hₘₐₓ, hₘᵢₙ, stepsize, locate_cell, normal_vector,
       iterative_refinement!, change_points!, set_points!
export npoints, points, point, half_points, half_point
export spacing, forward_spacing, half_spacing, spacings, half_spacings, cell_measure,
       cell_measures
export spacings_iterator, forward_spacings_iterator, half_spacings_iterator,
       points_iterator, half_points_iterator, cell_measures_iterator
export indices, boundary_indices, interior_indices, is_boundary_index, index_in_marker,
       is_uniform

# Space handling
export gridspace, vector_gridspace, space, space_type, spaces, ScalarGridSpace,
       CompositeGridSpace,
       VectorGridSpace
export ndofs, ncomponents, weights
export VectorElement, element, to_matrix, values, values!, component, components,
       component_range, component_ranges
export Rₕ, Rₕ!, avgₕ, avgₕ!
export interpolate, interpolate!, interpolate_at, interpolation_matrix, πₕ

export innerₕ
export inner₊, inner₊ₓ, inner₊ᵧ, inner₊₂
export snorm₁ₕ, norm₁ₕ, norm₊, normₕ

export diff₋ₓ, diff₋ᵧ, diff₋₂, diff₋ₕ
export diff₊ₓ, diff₊ᵧ, diff₊₂, diff₊ₕ
export diff₋ₓ!, diff₋ᵧ!, diff₋₂!
export diff₊ₓ!, diff₊ᵧ!, diff₊₂!

export D₋ₓ, D₋ᵧ, D₋₂, ∇₋ₕ
export D₊ₓ, D₊ᵧ, D₊₂, ∇₊ₕ
export D₋ₓ!, D₋ᵧ!, D₋₂!
export D₊ₓ!, D₊ᵧ!, D₊₂!

export Dstar₊ₓ, Dstar₊ᵧ, Dstar₊₂, Dstar₊ₕ
export Dstar₊ₓ!, Dstar₊ᵧ!, Dstar₊₂!

export Dcₓ, Dcᵧ, Dc₂, Dcₕ
export Dcₓ!, Dcᵧ!, Dc₂!

export Dₕₓ, Dₕᵧ, Dₕ₂, ∇ₕ
export Dₕₓ!, Dₕᵧ!, Dₕ₂!

export jumpₓ, jumpᵧ, jump₂, jumpₕ
export jumpₓ!, jumpᵧ!, jump₂!

export M₋ₓ, M₋ᵧ, M₋₂, M₋ₕ
export M₊ₓ, M₊ᵧ, M₊₂, M₊ₕ
export M₋ₓ!, M₋ᵧ!, M₋₂!
export M₊ₓ!, M₊ᵧ!, M₊₂!

export dirichlet_constraints, dirichlet_bc!, symmetrize!, DirichletConstraint
export form, assemble, assemble!, assemble_parallel!, allocate_system_matrix, evaluate!
export issymmetric, isposdef

export export_vtk
export export_pgfplots

include("utils/macros.jl")
include("utils/backend.jl")
include("utils/linear_algebra.jl")

include("geometry/pretty_print.jl")
include("geometry/set.jl")
include("geometry/marker.jl")
include("geometry/domain.jl")
include("utils/bramble_function.jl")

include("mesh/interface.jl")
include("mesh/marker.jl")
include("mesh/pretty_print.jl")
include("mesh/mesh1d.jl")
include("mesh/meshnd.jl")

include("space/gridspace.jl")
include("space/scalar_gridspace.jl")
include("space/vector_gridspace.jl")
include("space/vectorelement.jl")

include("space/operators/restriction.jl")
include("space/operators/cell_average.jl")
include("space/operators/linear_operators.jl")
include("space/operators/shift.jl")
include("space/operators/difference.jl")
include("space/operators/jump.jl")
include("space/operators/average.jl")
include("space/operators/interpolation.jl")
include("space/inner_product.jl")

include("form/common.jl")
include("form/component.jl")
include("form/stencil_pattern.jl")
include("form/block_extract.jl")
include("form/dirichlet_constraints.jl")
include("form/linear.jl")
include("form/bilinear.jl")
include("form/symmetry.jl")

include("exporters/vtk_export.jl")
include("exporters/pgfplots_export.jl")

include("precompile.jl")
end
