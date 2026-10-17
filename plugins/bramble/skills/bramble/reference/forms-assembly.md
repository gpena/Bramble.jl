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
issymmetric(a), isposdef(a)                          # symbolic checks on a BilinearForm
```

`dirac`/`DiracSource` build a point source. `dirichlet_constraints` returns a
`DirichletConstraint`, the pairs `assemble`/`dirichlet_bc!` consume.

### Ghost spaces

On a constrained ghost space (`meshes-spaces.md`) the load is `Eᵀ(F − A d)`, so it needs the
matrix: write `assemble(a, l)` (or `assemble!(A, F, a, l)`), never `assemble(l)` where the
closure has `d ≠ 0`; `assemble(l)`, `l(v)` and `evaluate!` refuse with an `ArgumentError`
naming `assemble(a, l)`. `dirichlet` and `symmetrize` work when every ghost is eliminated by a
relation and none by `u => g`. `assemble_parallel!` refuses (`assemble!` already runs under
the mesh's policy). `matrix_free_operator(a)` works on ghost forms (one product at a time per
operator), and `kronecker_operator(a)` and the `fdm_*` solvers where the closure is
separable; elsewhere they refuse with an `ArgumentError`. The table `Support on ghost spaces`
in the documentation's `internals/space.md` lists every operation.

### Rank-one terms

`σ * innerₕ(g, u) * innerₕ(h, v)` (a penalty, such as `innerₕ(1, u) * innerₕ(1, v)` to fix
the constant of pure Neumann) is a `Bramble.RankOneUpdate`: `assemble` returns `A` plus the
rank-one part, never a dense matrix, and `pde_solve` solves the bordered system. One
rank-one term per form, scalar spaces only (a composite ghost space refuses).
`assemble!(K, a)` rereads a `Ref` σ; written `r[]` it is frozen at `form`. A large σ costs
nothing; a small one loses digits, so take σ of order `1 / hₘₐₓ(Ωₕ)` or more.

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
