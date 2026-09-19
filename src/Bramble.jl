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

# Utilities
export backend, metal_backend, vector_type, matrix_type, backend_types
# The CSR backend constructor for the memory-scaling milestone (gpena/Bramble.jl#214): a
# `metal_backend`-style stub whose real method arrives with the SparseMatricesCSR extension.
export csr_backend
export ExecutionPolicy, Serial, Parallel, execution_policy
# The CPU/GPU split of the policy hierarchy (gpena/Bramble.jl#191). `Serial` and `Parallel`
# stay exported above: they are aliases of the first two of these, and every call site,
# test and benchmark key in this repository spells them that way.
export CpuPolicy, CpuSerial, CpuThreaded, GpuPolicy, GpuAsync
# The Polyester-backed policy (gpena/Bramble.jl#190): the type ships here, the sweeps it
# selects arrive with the BramblePolyesterExt package extension.
export CpuBatch

# `vector`/`matrix` build a raw backend array (point 70): real API, but two of the most
# generic nouns in the language, and a beginner's own top-level `vector = [...]` after
# `using Bramble` errors rather than shadows. `Bramble.vector(...)` still reaches them.
public vector, matrix
# Backend-extension plumbing (point 70): identity/zero matrices tied to a `Backend`, real,
# tested, reached while implementing a new backend rather than while using one.
public backend_eye, backend_zeros
# The contract a custom backend array type implements (gpena/Bramble.jl#100): declared by
# whoever adds an array type, never called by a user of one, so `public` rather than
# exported, alongside the allocators that read it.
public supports_undef_construction
# Read by every package extension's own `@compile_workload` gate (gpena/Bramble.jl#196), so
# a user's `set_preferences!(Bramble, "precompile_workload" => false)` disables the
# extensions' workloads along with the core one, not just the core one.
public PRECOMPILE_WORKLOAD

# domain/interval handling functions
export box, interval, ×, dim, topo_dim, extrema, point, center, projection, boundary_symbols
export domain, markers, labels

# `set` is `CartesianProduct`'s identity accessor, real, but the single most generic noun
# in the language, same reasoning as `vector`/`matrix` above. `is_collapsed`/`point_type`
# are queries about a `CartesianProduct`'s own internal shape, reached while building
# geometry helpers, not while using one (point 70).
public set, is_collapsed, point_type

# Mesh handling
export Mesh1D, MeshnD
export mesh, submeshes, hₘₐₓ, stepsize, locate_cell, iterative_refinement!, change_points!, set_points!
export npoints, points, point, half_points, half_point
export spacing, forward_spacing, half_spacing, spacings, forward_spacings, cell_measure
export indices, boundary_indices, interior_indices, is_boundary_index, index_in_marker, is_uniform

# `AbstractMeshType`/`MeshMarkers` are extension points for a new mesh type, not everyday
# vocabulary; `mesh_type`/`normal_vector`/`hₘᵢₙ`/`half_spacings`/`cell_measures` are the
# same layer, real, tested, reached while implementing a mesh or a boundary-facing
# operator rather than while using one (point 70).
public AbstractMeshType, MeshMarkers
public mesh_type, hₘᵢₙ, half_spacings, cell_measures

# Exported since v3.1 (gpena/Bramble.jl#213): the outward normal is part of writing a
# Neumann or Robin term, not an internal query, now that `inner_Γ` exists.
export normal_vector

# Space handling
export gridspace, vector_gridspace, space, spaces, ScalarGridSpace, CompositeGridSpace
export ndofs, ncomponents, weights

# `VectorGridSpace` is a type alias for `CompositeGridSpace{N}`; `space_type` reads a
# space's type back off a `VectorElement`. Neither appears in a tutorial: both are for
# code written *against* a space's type, not for building one (point 70).
public VectorGridSpace, space_type
export VectorElement, element, parent, reshape, components, component_range, component_ranges
export ldiv!
# `*` is already in scope from Base regardless (a fundamental operator, never shadowed by
# `using Bramble`), so this export is for documentation purposes alone -- the same reason
# `parent`/`reshape` above are re-exported despite being Base's own functions too: it is what
# keeps this file's `Base.:*(::Function, ::VectorElement)` docstring "public" from
# Documenter's perspective, so `docs/src/internals/space.md`'s `Public = false` autodocs
# sweep of `space/vectorelement.jl` does not also pick it up and conflict with its explicit
# `@docs` entry in `api.md` (gpena/Bramble.jl#197).
export *
export Rₕ, Rₕ!, avgₕ, avgₕ!
export interpolate_at, interpolation_matrix, πₕ, πₕ!

export innerₕ, inner_Γ, dirac, skew_symmetric
export n
export inner₊, inner₊ₓ, inner₊ᵧ, inner₊₂
export snorm₁ₕ, norm₁ₕ, norm₊, normₕ, norminf_h, norm∞ₕ

# Three families are internal in v3.0 (gpena/Bramble.jl#211): the unscaled differences
# `diff₋*`/`diff₊*`, the forward differences `D₊*`/`∇₊ₕ`, and the forward averages `M₊*`.
# They keep their definitions, their docstrings and their entries in the API reference, and
# are reached as `Bramble.D₊ₓ`; what they lose is a place on the surface `using Bramble`
# brings in.
#
# The reasons differ by family. The unscaled differences have no form-layer node, so they
# cannot appear inside a bilinear form, and `diff₊` is the same arithmetic as `jump`, which
# carries the intent a caller reaching for it usually means. The forward difference and the
# forward average are the duals the backward ones are built and checked against: Bramble
# discretises with the backward operator paired with `inner₊`, and a user writing a form
# reaches for `D₋ₓ` and `Mₓ`. Keeping their forward partners exported offered a choice that
# the discretisation does not actually leave open.

# The dimensional entry points (gpena/Bramble.jl#74) travel with their family: `D₋` is
# exported because `D₋ₓ` is, `D₊` is `public` because `D₊ₓ` is, and `diff₋`/`diff₊` are
# neither because their subscripts are neither. Two families have no entry point of their
# own: the averages put theirs on `Mₕ`/`M₊ₕ`, already listed below, rather than mint a bare
# `M` that `using Bramble` would take away from a caller's mass matrix.
#
# This is also where gpena/Bramble.jl#74 and gpena/Bramble.jl#211 have to be reconciled. #74
# was written before #211 and says the subscript names are retained "permanently"; #211 then
# took 28 of them off this surface. #211 wins: every forwarder below is still *defined*, and
# `D₋(uₕ, 2)` reaches the same method `D₋ᵧ(uₕ)` does, but only the survivors are exported or
# `public`.
export D₋ₓ, D₋ᵧ, D₋₂, ∇ₕ, D₋

# The vector calculus operators built on those differences (gpena/Bramble.jl#158). The
# backward-difference spellings are exported beside `∇ₕ`, which is also the backward one;
# the forward twins are `public` for the same reason `∇₊ₕ` is (#211).
export divₕ, divₕ!, curlₕ, curlₕ!, Δₕ, Δₕ!
public div₊ₕ, div₊ₕ!, curl₊ₕ, curl₊ₕ!

# The small-strain tensor (gpena/Bramble.jl#234): `εₕ`/`εₕ!` are the runtime pair over a
# `VectorElement` or composite grid function, the same shape as `divₕ`/`divₕ!` above.
# `εₕ` also has a builder-only symbolic method returning a `Bramble._StrainTensor`, consumed
# immediately by `inner₊` inside a `form(...)` body and never a grid function -- that method
# has no in-place counterpart, but the runtime one does, and both share this export.
export εₕ, εₕ!
export D₋ₓ!, D₋ᵧ!, D₋₂!

# `public` rather than nothing at all, unlike the unscaled differences above: the forward
# difference and the forward average are what the backward ones are checked against, so
# `Bramble.D₊ₓ` is a supported thing to reach for -- the summation-by-parts tests and
# `ext/BrambleILUZeroExt.jl`'s workload both do. Declaring them keeps that access honest
# under `ExplicitImports.check_all_qualified_accesses_are_public`, and keeps them in
# `names(Bramble)`, which is what requires them to stay documented.
public D₊ₓ, D₊ᵧ, D₊₂, ∇₊ₕ, D₊
public D₊ₓ!, D₊ᵧ!, D₊₂!

export D̽ₓ, D̽ᵧ, D̽₂, D̽ₕ, D̽
export D̽ₓ!, D̽ᵧ!, D̽₂!

export Dcₓ, Dcᵧ, Dc₂, Dcₕ, Dc
export Dcₓ!, Dcᵧ!, Dc₂!

export Dₕₓ, Dₕᵧ, Dₕ₂, Dₕ
export Dₕₓ!, Dₕᵧ!, Dₕ₂!

export jumpₓ, jumpᵧ, jump₂, jumpₕ, jump
export jumpₓ!, jumpᵧ!, jump₂!

export Mₓ, Mᵧ, M₂, Mₕ
export Mₓ!, Mᵧ!, M₂!

public M₊ₓ, M₊ᵧ, M₊₂, M₊ₕ
public M₊ₓ!, M₊ᵧ!, M₊₂!

export dirichlet_constraints, dirichlet_bc!, symmetrize!
export reaction, reaction_density
export form, assemble, assemble!, assemble_parallel!, allocate_system_matrix, evaluate!
export is_separable, kronecker_operator, KroneckerLinearOperator
# `bandwidths`/`blockbandwidths` (gpena/Bramble.jl#175): the lower/upper (block) bandwidth a
# form's matrix occupies, read from the resolved AST alone. Their only caller was the banded
# backend extension, removed 2026-09-19 (v3.3.0 plan, "Removed: the banded backends"); they
# stay and are exported because they answer a question about a form independently of any
# storage type.
export bandwidths, blockbandwidths
export assemble_add!
export jacobian_pattern, ast_sparsity_detector
export type_cached_assemble!
export Semidiscretization, semidiscretize, mass_matrix, operator_matrix
export SemidiscretizeRHS, semidiscretize_rhs
export SecondOrderSemidiscretization,
       semidiscretize_second_order, damping_matrix, stiffness_matrix, block_mass_matrix
# `jacobian!` is `public` rather than exported, the same call as `diff₋ₓ` above:
# `DifferentiationInterface` exports a `jacobian!` of its own, and the two are ambiguous in
# any session holding both -- which the test suite is, and any user pairing a Bramble
# semidiscretisation with sparse AD would be. Reached as `Bramble.jacobian!`; it is also the
# default `jacobian` of `ode_function`, so it rarely needs naming at all.
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
# `fdm_solve` (gpena/Bramble.jl#259): a direct solve for a separable, constant-coefficient
# `BilinearForm` by fast diagonalisation, implemented in `ext/BrambleKroneckerExt.jl` (S5.2).
# A package extension cannot introduce a new binding into its parent module, so this stub
# gives it one to add methods to, the way `csr_backend`/`_csr_backend` above and
# `sparspak_solve`/`_sparspak_solve` (`src/solvers/sparspak_solver.jl`) let their extensions
# extend a name this module already owns.
#
# Verified 2026-09-19 (S9.1) that this alone is *not* enough here: unlike those two, the
# extension's `fdm_solve(a::BilinearForm, ...)`/`fdm_solve(K::KroneckerLinearOperator, ...)`
# are written unqualified, and `using Bramble: Bramble, ...` (no `fdm_solve` in that list)
# does not let an unqualified `function fdm_solve(...)` extend this stub -- Julia only
# extends a parent's function through `import Parent: name` or a dot-qualified
# `function Parent.name(...)` definition, confirmed by a minimal repro of the same shape.
# After `using Bramble, Kronecker`, `methods(Bramble.fdm_solve)` is empty and
# `Bramble.fdm_solve !== Base.get_extension(Bramble, :BrambleKroneckerExt).fdm_solve`: the
# plain spelling does not yet work, and reaching the real implementation still needs
# `Base.get_extension(Bramble, :BrambleKroneckerExt).fdm_solve`. Fixing this needs an edit
# inside `ext/BrambleKroneckerExt.jl` (outside S9.1's ownership) to import `fdm_solve` from
# `Bramble` or qualify its two method definitions as `Bramble.fdm_solve`; reported to the
# integrator rather than done here.
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

# `DirichletConstraint` is `dirichlet_constraints(...)`'s own return type, reached for an
# `isa` check rather than constructed by name: the tests already reach it as
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
