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

`dirac`/`DiracSource` build a point source. `dirichlet_constraints` returns a
`DirichletConstraint`, the pairs `assemble`/`dirichlet_bc!` consume.

## Kronecker path

`kronecker_operator(a)` builds a matrix-free `KroneckerLinearOperator` for a separable
`BilinearForm` on a `MeshnD` (`is_separable(a)`). The `fdm_*` functions need `using
Kronecker`; `fdm_solve` and `fdm_factorize` need a Laplacian-like form: separable, and every term differs from one mass per axis on at
most one axis. Symmetric axes are solved by fast diagonalisation, advection terms by a
generalised Schur factorisation. `dirichlet` is `nothing` or `:boundary` (homogeneous, the
whole boundary). All of them are host-only.

```julia
x = fdm_solve(a, F; dirichlet = :boundary)       # the assembled system's solution, no matrix
fact = fdm_factorize(a; dirichlet = :boundary)   # factorise once...
fdm_solve!(x, fact, F)                           # ...then each right-hand side, 0 B on host
fdm_factorize!(fact, a)                          # refill: a scalar or Ref coefficient changed
P = fdm_preconditioner(a)                        # a Bramble.FDMPreconditioner, for Krylov
```

`fdm_factorize!` needs the structure `fact` was built for (sizes, eltype, terms, symmetric
or Schur); otherwise call `fdm_factorize` again. After `Bramble.change_points!`, refill from a
form built on a new `gridspace`. `fdm_solve` refuses a mixed-derivative form
(`innerₕ(D₋ₓ(D₋ᵧ(u)), v)`, or a coefficient varying along two axes): assemble it and
precondition a Krylov solver with `fdm_preconditioner(a)`, which inverts the Laplacian-like
part and works when diffusion dominates the cross term.
