# Variational forms and assembly

Part of the `bramble` skill. Names written `Bramble.name` are `public` but not exported.

## Variational forms & assembly

```julia
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(∇ₕ(u), ∇ₕ(v)))   # bilinear
l = form(Wₕ, v -> innerₕ(fₕ, v))                    # linear

bcs = dirichlet_constraints(Ωₕ, :boundary => g)

A = assemble(a; dirichlet = :boundary)              # allocates
F = assemble(l; dirichlet = bcs)
A, F = assemble(a, l; dirichlet = bcs, symmetrize = true)

Bramble.allocate_system_matrix(a)                           # build a matrix's sparsity pattern once...
assemble!(A, a; dirichlet = bcs)                    # ...then refill it, no allocation
Bramble.assemble_parallel!(A, a)                             # threaded fill
assemble_add!(A, a)                                  # accumulate onto A, no fill! first

dirichlet_bc!(A, Ωₕ, :boundary)                     # matrix overload
dirichlet_bc!(F, Wₕ, bcs, :boundary)                # vector overload
symmetrize!(A, F, Ωₕ, :boundary)

Bramble.reaction(A_unconstrained, F_unconstrained, uₕ; marker = :boundary)   # flux a constraint had to supply
Bramble.reaction_density(...)                               # pointwise counterpart, for export_vtk

Bramble.jacobian_pattern(a)                                  # Jacobian sparsity straight off the AST
bandwidths(a), blockbandwidths(a)                    # storage the assembled matrix needs
issymmetric(a), isposdef(a)                          # symbolic checks on a BilinearForm
```

`kronecker_operator`/`KroneckerLinearOperator`/`is_separable`/`fdm_solve` are a matrix-free
Kronecker path for a separable `BilinearForm` on a `MeshnD` (`fdm_solve` needs `using
Kronecker`). `dirac`/`DiracSource` build a point source. `dirichlet_constraints` returns a
`DirichletConstraint`, the pairs `assemble`/`dirichlet_bc!` consume.
