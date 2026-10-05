# Backends, exporters, time integration and solvers

Part of the `bramble` skill. Names written `Bramble.name` are `public` but not exported.

## Execution policies and backends

```julia
b = backend()                          # default: Vector{Float64}, SparseMatrixCSC, CpuSerial
b = backend(Float32; policy = Bramble.CpuThreaded())
# public only (write Bramble.vector_type or import):
Bramble.vector_type(b), Bramble.matrix_type(b), Bramble.backend_types(b)
Bramble.execution_policy(b)                    # the EP instance the backend carries
Bramble.locality(b)                            # HostLocality() or DeviceLocality()
```

`ExecutionPolicy` (`public`, like every policy type here except the exported
`Serial` and `Parallel`) splits into `CpuPolicy` (`CpuSerial`, `CpuThreaded`, and `CpuBatch`,
which needs `using Polyester`) and `GpuPolicy` (`GpuKernel`; `GpuAsync` is its deprecated alias). `Serial`/`Parallel` alias
`CpuSerial`/`CpuThreaded`. `metal_backend`/`gpu_backend` build a device backend (need
`using Metal`); `csr_backend` selects a CSR sparse matrix type.

Vector-type locality and policy locality must agree, or construction raises (`SKILL.md`, rule 9).

## Exporters

```julia
export_vtk(filename, Ωₕ, :u => uₕ, :v => vₕ)
export_pgfplots(filename, Ωₕ, :u => uₕ)
```

## Time integration and solvers (SciML stack)

Detail in the documentation's SciML API page; each needs the named package loaded.

```julia
sd = semidiscretize(a, l; dirichlet = bcs)   # M uₕ' = F(t) - A uₕ  (needs SciMLBase)
Bramble.mass_matrix(sd), Bramble.operator_matrix(sd), Bramble.jacobian_prototype(sd)
prob = ode_problem(sd, u₀, tspan), fn = Bramble.ode_function(sd)   # hand off to OrdinaryDiffEq; ode_function public only

sd2 = semidiscretize_second_order(k, l; dirichlet = bcs)   # M üₕ + C u̇ₕ + K uₕ = F(t)
Bramble.mass_matrix(sd2), Bramble.damping_matrix(sd2), Bramble.stiffness_matrix(sd2), Bramble.block_mass_matrix(sd2)
prob2 = second_order_ode_problem(sd2, v₀, u₀, tspan)   # velocity, then displacement

linear_problem(a, l; kwargs...)              # steady linear system, to LinearSolve
nonlinear_problem(residual, u0; kwargs...)   # steady nonlinear residual, to NonlinearSolve
```

Each `*_solve` below also takes `(a::BilinearForm, l::LinearForm; dirichlet,
dirichlet_components, symmetrize, kwargs...)`, assembling and solving in one call
(`suitesparse_solve(a, l; dirichlet = bcs)`).

```julia
pde_solve(A, F)                              # A \ F with a ChainRulesCore adjoint rule
amg_preconditioner(A), ilu_preconditioner(A) # multigrid / ILU(0), each its own package

Bramble.sparse_factorize(A; solver = :default, sym = :auto)   # dispatches to a backend below
Bramble.refactor!(fact, A)                           # reuse a factorization's symbolic structure
Bramble.suitesparse_factorize(A), suitesparse_qr_factorize(A) # CHOLMOD/UMFPACK, SPQR
accelerate_factorize(A)                      # macOS libSparse
mumps_factorize(A)                           # parallel multifrontal
sparspak_factorize(A)                        # pure Julia, works with Dual/BigFloat
# each *_factorize has a matching *_solve and *_refactor!

Bramble.type_cached_assemble!(...)                   # one sparsity pattern per coefficient element type
```
