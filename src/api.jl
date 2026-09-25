# This file is Bramble's whole public interface: every `export` and `public` statement in
# the package lives here (#344).

# --- Backend & Execution Policies ---
export backend, gpu_backend, metal_backend, csr_backend
export Serial, Parallel, vector_type, matrix_type, backend_types, execution_policy

public ExecutionPolicy, CpuPolicy, CpuSerial, CpuThreaded, CpuPolyester, CpuBatch
public GpuPolicy, GpuKernel, GpuAsync
public locality, Locality, HostLocality, DeviceLocality
public vector, matrix, metal_sparse_csr, metal_sparse_csc

# Read by every package extension's own `@compile_workload` gate (gpena/Bramble.jl#196), so
# a user's `set_preferences!(Bramble, "precompile_workload" => false)` disables the
# extensions' workloads along with the core one, not just the core one.
public PRECOMPILE_WORKLOAD

# --- Domain & Geometry ---
export box, interval, ×, ⋅, dim, boundary_symbols
export domain, markers, labels

public center, projection, point, topo_dim

# --- Mesh ---
export mesh, npoints, points, hₘₐₓ, hₘᵢₙ, iterative_refinement!, normal_vector
export spacing, forward_spacing

public Mesh1D, MeshnD
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
export η
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
export fdm_solve
