module Bramble

import Base: eltype, length
import Base: show, first, last, getindex, setindex!, iterate, size, firstindex, lastindex, axes, eachindex

using SparseArrays: SparseArrays, SparseMatrixCSC, spdiagm, spzeros, rowvals, nonzeros, nzrange, sparse, sparse!

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
include("mesh/ghost_mesh.jl")
# The split of a walk argument names the mesh types and their walk states.
include("utils/batch_split.jl")

include("space/gridspace.jl")
include("space/scalar_gridspace.jl")
include("space/vector_gridspace.jl")
include("space/vectorelement.jl")
include("space/ghost_gridspace.jl")

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
include("assembly/ghost_assembly.jl")
include("assembly/ghost_relations.jl")
include("problems/semidiscrete_constraints.jl")
include("problems/semidiscrete.jl")
include("problems/semidiscrete_rhs.jl")
include("problems/semidiscrete_problems.jl")
include("problems/nonlinear_problem.jl")
include("solvers/amg_preconditioner.jl")
include("solvers/ilu_preconditioner.jl")
include("solvers/matrix_free_preconditioners.jl")
include("solvers/fdm.jl")
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

include("exporters/common.jl")
include("exporters/vtk_export.jl")
include("exporters/pgfplots_export.jl")

include("precompile.jl")
end
