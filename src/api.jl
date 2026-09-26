# This file is Bramble's whole public interface: every `export` and `public` statement in
# the package lives here (#344).

# --- Backend & Execution Policies ---
export backend, gpu_backend, metal_backend, csr_backend
export Serial, Parallel

public ExecutionPolicy, CpuPolicy, CpuSerial, CpuThreaded, CpuPolyester, CpuBatch
public GpuPolicy, GpuKernel, GpuAsync
public locality, Locality, HostLocality, DeviceLocality
public vector, matrix, metal_sparse_csr, metal_sparse_csc
public vector_type, matrix_type, backend_types, execution_policy

# Read by every package extension's own `@compile_workload` gate (gpena/Bramble.jl#196), so
# a user's `set_preferences!(Bramble, "precompile_workload" => false)` disables the
# extensions' workloads along with the core one, not just the core one.
public PRECOMPILE_WORKLOAD

# --- Domain & Geometry ---
export box, interval, ×, ⋅, dim, boundary_symbols
export domain, markers, labels

public center, projection, point, topo_dim

# --- Mesh ---
export mesh, npoints, points, hₘₐₓ, iterative_refinement!
export spacing, forward_spacing

public Mesh1D, MeshnD, hₘᵢₙ, normal_vector
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
export norm, snorm₁ₕ, norm₁ₕ, normₕ, norminf

public ScalarGridSpace, CompositeGridSpace, VectorGridSpace, VectorElement
public component_range, component_ranges, skew_symmetric
public inner₊ₓ, inner₊ᵧ, inner₊₂
public norm₊
public weights, interpolation_matrix

# --- Discrete Differential & Difference Operators ---
export ∇ₕ
export divₕ, curlₕ, Δₕ, εₕ
export ∇̃ₕ, diṽₕ, curl̃ₕ
export ∇cₕ, divcₕ, curlcₕ, εcₕ
export ∇̽ₕ, div̽ₕ, curl̽ₕ, ε̽ₕ

export jumpₕ

export Mₕ, Mcₕ

# Dimensional entry points (the direction as an argument; `∇ₕ[d]` is the exported route)
public D₋, D̃, Dc, jump

# The `D*ₕ` spellings of the exported gradients (`D̃ₕ === ∇̃ₕ`, `Dcₕ === ∇cₕ`, `D̽ₕ === ∇̽ₕ`)
public D̃ₕ, Dcₕ, D̽ₕ

# In-place vector calculus
public divₕ!, curlₕ!, Δₕ!, εₕ!
public ∇̃ₕ!, diṽₕ!, curl̃ₕ!
public ∇cₕ!, divcₕ!, curlcₕ!, εcₕ!
public ∇̽ₕ!, div̽ₕ!, curl̽ₕ!, ε̽ₕ!

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
export is_separable, kronecker_operator
export pde_solve
export semidiscretize, semidiscretize_second_order
export ode_problem, linear_problem, nonlinear_problem
export second_order_ode_problem
export amg_preconditioner
export ilu_preconditioner

public KroneckerLinearOperator, ode_function, second_order_ode_function
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
