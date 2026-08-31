module Bramble

#using StyledStrings: styled, @styled_str

using DocStringExtensions

import Base: eltype, length
import Base: show, first, last, getindex, setindex!, iterate, size, ndims, firstindex,
             lastindex, axes, eachindex

# `rowvals`, `nonzeros`, `nzrange` and `sparse` are for the form layer: it
# edits sparse structure directly when imposing Dirichlet rows, and the import here is
# selective, so a name missing from this list is simply undefined in the module. That is
# what `dirichlet_bc!` hit on its first run — `UndefVarError: rowvals`.
using SparseArrays: SparseMatrixCSC, SparseVector, spdiagm, spzeros,
                    rowvals, nonzeros, nzrange, sparse, sparse!

using FunctionWrappers: FunctionWrapper

using StaticArrays: SVector, @SVector

using LinearAlgebra: Diagonal, I, transpose, mul!, dot
using FillArrays: Eye, Ones

import Base: copy
using Base: @propagate_inbounds
using Random: rand!

using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference
using QuadGK: gauss

# Utilities
export backend, metal_backend, vector, matrix, vector_type, matrix_type, backend_types,
       backend_eye, backend_zeros
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
export gridspace, vector_gridspace, space, spaces, ScalarGridSpace, CompositeGridSpace,
       VectorGridSpace
export ndofs, ncomponents, weights
export VectorElement, element, to_matrix, values, values!, component, components,
       component_range, component_ranges
export Rₕ, Rₕ!, avgₕ, avgₕ!

export innerₕ
export inner₊, inner₊ₓ, inner₊ᵧ, inner₊₂
export snorm₁ₕ, norm₁ₕ, norm₊, normₕ

export diff₋ₓ, diff₋ᵧ, diff₋₂, diff₋ₕ
export diff₊ₓ, diff₊ᵧ, diff₊₂, diff₊ₕ
# The in-place forms. Each writes into its first argument and returns it, so the allocating
# form above is one call to it on a `similar`.
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

#=
export ⋅

export form, assemble, assemble!
=#
#=
# Exporters
export ExporterVTK, addScalarDataset!, datasets, save2file, close
=#

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

# Rₕ and avgₕ come first: they need only the element, while the stencil operators below
# share a traversal defined in difference.jl.
include("space/operators/restriction.jl")
include("space/operators/cell_average.jl")
include("space/operators/linear_operators.jl")
include("space/operators/shift.jl")
include("space/operators/difference.jl")
include("space/operators/jump.jl")
include("space/operators/average.jl")
include("space/inner_product.jl")

# The form layer is being brought back a file at a time, auxiliary files first so that
# every name is defined before the file that consumes it.
#
# common.jl carries the symbolic AST nodes — TrialFunction, TestFunction, their indexed
# forms and SourceFunction — all subtypes of the LazyOp restored in
# space/operators/linear_operators.jl.
include("form/common.jl")

# block_extract.jl adds the component walks a coupled form routes by: which component a term
# names on each side, and the block that pair picks out.
# component.jl indexes a node by component, and needs every node type in its signatures, so
# it comes after common.jl and the operator files it includes.
include("form/component.jl")

# stencil_pattern.jl reads the sparsity pattern off a built AST, so it needs every node
# type that carries a stencil — which means after common.jl and its operator includes.
include("form/stencil_pattern.jl")

# Shared by both assembly files, so it belongs to neither. Included ahead of them because

include("form/block_extract.jl")

# dirichlet_constraints.jl comes last of the three: it is the user-facing one, and its
# composite methods are what need the component walks above.
include("form/dirichlet_constraints.jl")

# linear.jl assembles a right-hand side. It comes after dirichlet_constraints.jl, whose
# `dirichlet_bc!` it applies.
include("form/linear.jl")

include("form/bilinear.jl")
#=
include("exporter/types.jl")
include("exporter/exporter_vtk.jl")
=#

include("precompile.jl")
end
