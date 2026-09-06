module Bramble

using DocStringExtensions

import Base: eltype, length
import Base: show, first, last, getindex, setindex!, iterate, size, firstindex,
             lastindex, axes, eachindex

using SparseArrays: SparseMatrixCSC, spdiagm, spzeros,
                    rowvals, nonzeros, nzrange, sparse, sparse!

using LinearAlgebra: I, dot, mul!
import LinearAlgebra: issymmetric, isposdef

import Base: copy
using Base: @propagate_inbounds
using Random: rand!

using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference
using QuadGK: gauss

# Utilities
export backend, metal_backend, vector_type, matrix_type, backend_types
export ExecutionPolicy, Serial, Parallel, execution_policy

# `vector`/`matrix` build a raw backend array (point 70): real API, but two of the most
# generic nouns in the language, and a beginner's own top-level `vector = [...]` after
# `using Bramble` errors rather than shadows. `Bramble.vector(...)` still reaches them.
public vector, matrix
# Backend-extension plumbing (point 70): identity/zero matrices tied to a `Backend` — real,
# tested, reached while implementing a new backend rather than while using one.
public backend_eye, backend_zeros

# domain/interval handling functions
export box, interval, ×, dim, topo_dim, tails, point, cartesian_product, center, projection,
       get_boundary_symbols
export domain, markers, labels

# `set` is `CartesianProduct`'s identity accessor — real, but the single most generic noun
# in the language, same reasoning as `vector`/`matrix` above. `is_collapsed`/`point_type`
# are queries about a `CartesianProduct`'s own internal shape, reached while building
# geometry helpers, not while using one (point 70).
public set, is_collapsed, point_type

# Mesh handling
export Mesh1D, MeshnD
export mesh, submeshes, hₘₐₓ, stepsize, locate_cell,
       iterative_refinement!, change_points!, set_points!
export npoints, points, point, half_points, half_point
export spacing, forward_spacing, half_spacing, spacings, cell_measure
export indices, boundary_indices, interior_indices, is_boundary_index, index_in_marker,
       is_uniform

# `AbstractMeshType`/`MeshMarkers` are extension points for a new mesh type, not everyday
# vocabulary; `mesh_type`/`normal_vector`/`hₘᵢₙ`/`half_spacings`/`cell_measures` and the six
# `*_iterator` functions are the same layer — real, tested, reached while implementing a
# mesh or a boundary-facing operator rather than while using one (point 70).
public AbstractMeshType, MeshMarkers
public mesh_type, hₘᵢₙ, normal_vector, half_spacings, cell_measures
public spacings_iterator, forward_spacings_iterator, half_spacings_iterator,
       points_iterator, half_points_iterator, cell_measures_iterator

# Space handling
export gridspace, vector_gridspace, space, spaces, ScalarGridSpace,
       CompositeGridSpace
export ndofs, ncomponents, weights

# `VectorGridSpace` is a type alias for `CompositeGridSpace{N}`; `space_type` reads a
# space's type back off a `VectorElement`. Neither appears in a tutorial — both are for
# code written *against* a space's type, not for building one (point 70).
public VectorGridSpace, space_type
export VectorElement, element, to_matrix, values, values!, component, components,
       component_range, component_ranges
export Rₕ, Rₕ!, avgₕ, avgₕ!
export interpolate_at, interpolation_matrix, πₕ, πₕ!

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

export dirichlet_constraints, dirichlet_bc!, symmetrize!
export form, assemble, assemble!, assemble_parallel!, allocate_system_matrix, evaluate!

# `DirichletConstraint` is `dirichlet_constraints(...)`'s own return type, reached for an
# `isa` check rather than constructed by name — the tests already reach it as
# `import Bramble: DirichletConstraint` rather than through `using` (point 70).
public DirichletConstraint
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
include("space/operators/stencil.jl")
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
