```@meta
CurrentModule = Bramble
CollapsedDocStrings = false
```

# AD coverage: every public feature under each kept backend and policy

[`ad_backend_evaluation.md`](ad_backend_evaluation.md) chose the AD backends, and
[`ad_di_evaluation.md`](ad_di_evaluation.md) chose to call them through
DifferentiationInterface (DI). This page records what the kept backends differentiate. Each
row is one public feature's smallest differentiable call, and each cell is what that call
did on this branch.

The cells are pinned by `test/space/ad_coverage.jl`. It builds every call on a uniform and
on a seeded non-uniform mesh, runs it through DI under `CpuSerial`, `CpuThreaded` and
`CpuPolyester`, and compares the gradient with ForwardDiff's on the same policy. ForwardDiff
is itself compared with a central difference, and every component of that reference is
clearly non-zero, so a backend that drops a parameter cannot pass. Enzyme runs in child
Julia processes, one per policy and group of cells: an Enzyme abort ends the process that
met it. A cell reads `works`, `throws` with the error's name, or `not run` with the reason. The
failures that have no cell are under "Open failures". A cell never reads `works` on a wrong gradient.

## What was measured

| Item | Value |
|:--|:--|
| Date | 2026-11-03 for the first matrix; the three-summand cells 2026-11-04; the cells changed in v4.2.2 and the compile table 2026-11-07; the ghost load and `Δₕ!` cells and the new `centered_difference!` cell 2026-11-08 and 2026-11-09; the cells changed in v4.2.4 2026-11-10, by the reproducers of the notes in `ad_reductions/` (the same calls as the cells); the geometric-input cells (`polygon`, `dirac`, `levelset`) and the ghost relation cells (`ghost_data`, `ghost_robin`) 2026-11-11, by the cells themselves, one child per policy and group; the solver cells (`factor_*`, `rank_one`, `fdm_*`, `gmg_*`, `chebyshev`) 2026-11-13, by the cells themselves, one child per policy and group |
| Tree | branch `v4-2-4`, with the v4.2.4 AD fixes (the v4.2.2 cells were measured on `v4-2-2`); the 2026-11-11 cells on branch `v4-2-4-geometry` at 39f604bf, macbook; the solver cells on branch `v4-2-4-solvers` at 87a509f4, macbook |
| Machine | Apple M2, Julia 1.13.3, on mains power |
| Enzyme | 0.13.219 for the solver cells (2026-11-13), the 2026-11-11 cells, the v4.2.4 cells (2026-11-10) and for the ghost load, `Δₕ!` and `centered_difference!` cells (2026-11-08); 0.13.218 for the other v4.2.2 cells (2026-11-07), the newest in the depot, which a temporary environment without a manifest loads; 0.13.216 for the 2026-11-03 cells |
| Threads | 4, the test's thread count, so both threaded policies run on 4 threads; the 2026-11-03 figures and the v4.2.4 reproducers ran on 2 (1 for the `constrained` reductions); the 2026-11-11 cells ran on 2 under load, in `--optimize=1` children; the solver cells ran on 2, alone on the machine, in `--optimize=1` children |
| Problem | 9 x 9 mesh in 2D, 17 nodes in 1D, 5 x 5 x 5 mesh in 3D, three parameters, `Float64`; the `fdm_*` cells have one parameter, and the `fdm_*`, `gmg_*` and `chebyshev` cells run on 17 x 17 (uniform) and 9 x 9 (non-uniform) meshes |
| Agreement | within 1e-8 relative to ForwardDiff on the same policy |
| After 8b423d93 | The `CpuThreaded` runs of `gmg_coefficient` reverse and `chebyshev` gave wrong gradients before commit 8b423d93; it fixed them (#1168), and both cells were measured again after it |
| Julia version | The solver cells and the re-measurement after 8b423d93 ran on Julia 1.13.3. juliaup then installed 1.13.4 (2026-11-13, 22:46) and removed 1.13.3; with it, Enzyme 0.13.219 aborts every child within 13 s (`unsupported tag julia.gc_alloc_ptr_offsets`, `UNREACHABLE executed at GradientUtils.cpp:364`), `chebyshev` under `CpuPolyester` among them, so nothing was re-measured on 1.13.4 and the pins above are the 1.13.3 results |
| Idle run | `gmg_load` under `CpuPolyester` (2 threads, `--optimize=1`): 1501 s wall for 88 s of CPU, its cells normal; the cause is unknown, see "Compile cost of the solver cells" |

The `dirac_location` cell compares with a central difference where ForwardDiff cannot run
(rel. 1e-5). The `gmg_*` cells compare with ForwardDiff of the direct solve `pde_solve(assemble(…), b)` to 1e-6, since the multigrid iteration stops at `tol = 1e-10`; `rank_one` compares with ForwardDiff of the plain solve of `A + w wᵀ` to 1e-8, since ForwardDiff refuses the `RankOneUpdate`.

## Kept backends

ForwardDiff, through DI, and Enzyme, called directly and through DI. PolyesterForwardDiff,
ReverseDiff and Mooncake are dropped by the AD backend evaluation
(`ad_backend_evaluation.md`).

## The matrix

ForwardDiff has one column: every cell is the same under the three policies, because a
`Dual` is a plain value and each thread writes its own entries. Enzyme has one column per
policy. Each Enzyme cell holds both modes, reverse then forward, unless it says otherwise.

| Feature | Call | ForwardDiff | Enzyme `CpuSerial` | Enzyme `CpuThreaded` | Enzyme `CpuPolyester` |
|:--|:--|:--|:--|:--|:--|
| mesh | `D₋ₓ(Rₕ(…))` on `mesh(…, [0, b] × [0, 1])`, gradient in `b` | works | works | works | works |
| mesh | `assemble` of a form on that mesh, gradient in `b` | works | works | not run | not run |
| operator | `D₋ₓ`, `Mₓ` and `Mᵧ`, `Δₕ`, `innerₕ` with `normₕ` | works | works | works | works |
| operator | `εₕ` on a composite space | works | works | works | works |
| operator | `D₋ₓ` and `D₋₂` on a grid function, 3D | works | works | not run | not run |
| operator | `M₂` on a grid function, 3D | works | works | works | works |
| interpolation | `Rₕ`, `avgₕ` | works | works | works | works |
| assemble | `assemble` of a linear form | works | works | works | works |
| assemble | `assemble` of `inner₊(κ∇U, ∇V)` | works | works | works | works |
| assemble | the same with `Mₕ(κ)` | works | works | works | works |
| assemble | `assemble(a, l; dirichlet)`, 1D | works | works | works | works |
| assemble | the same in 2D, non-uniform mesh | works | works | works | works |
| assemble | three summands, the third scaled by a constant | works | works | works | works |
| assemble | the linear form `innerₕ(u, v)`, 3D | works | works | not run | not run |
| assemble | `inner₊(κ∇U, ∇V)`, 3D | works | works | not run | not run |
| assemble | the same with `Mₕ(κ)`, 3D | works | works | works | works |
| pde_solve | `pde_solve(A, F)` with `A` from a 1D `assemble` of `κ` | works (the `Float64` factor of the values, solved per partial) | works | works | works |
| pde_solve | `pde_solve(sparse_factorize(A), F)`, `A` from a 1D `assemble` of `κ` and active, the factorization built inside the function | works | throws `IllegalTypeAnalysisException` in both modes (D8, see the limits) | throws `IllegalTypeAnalysisException` in both modes | throws `IllegalTypeAnalysisException` in both modes |
| pde_solve | `pde_solve(fact, F)`, `fact = sparse_factorize(A₀)` built outside, a load of the parameters | works | works | works | works |
| pde_solve | `ldiv!(uₕ, fact, F)`, the same factorization and load, into a grid function | works | works | works | works |
| pde_solve | `pde_solve(RankOneUpdate(A, 1.0, w, w), F)`, `A` and `F` active | throws `ArgumentError`, naming `RankOneUpdate` (D3) | works | works | works |
| semidiscretize | `semidiscretize_rhs(sd)(du, u, p, t)` | works | works | works | works |
| matrix_free | `mul!` with `matrix_free_operator` | works | works | works | works |
| kronecker | `mul!` with `kronecker_operator` | works | works | works | works |
| fdm_solve | `fdm_solve(a, F; dirichlet = :boundary)` on a constant form built inside the function, `F` active, 2D | works | works | works | works |
| fdm_solve | the same with a coefficient of the form active | throws `ArgumentError` | throws `ArgumentError` in both modes (D4) | throws `ArgumentError` in both modes | throws `ArgumentError` in both modes |
| gmg_solve | `gmg_solve(build, Ωₕ, b)`, `b` active, 2D | works | works | reverse works, batched forward ends the process (LLVM assertion, #1232, see the limits) | reverse works, batched forward ends the process (#1232) |
| gmg_solve | the same with a scalar and a grid-function coefficient of `build` active, and the load scaled | works | reverse works, forward throws `ArgumentError` naming "forward mode" (D9) | reverse works, batched forward ends the process (#1232) | reverse works, batched forward ends the process (#1232) |
| chebyshev | `chebyshev_preconditioner(a) \ x` with the coefficients of the form and the load scale active, 2D | works | throws `IllegalTypeAnalysisException` in both modes | works | works |
| ghost | `assemble(a, l)` on a constrained ghost space, parameters scale the form's coefficients, `Δₕ` in the form | works | works | works | works |
| ghost | the same with `inner₊` and `innerₕ` terms only | works | reverse works, forward not run | not run | not run |
| ghost | the load of `assemble(a, l)` alone, parameters in the linear form | works | works | works | works |
| ghost | relation data that depend on the parameter | works | works | throws `EnzymeNoTypeError` in both modes (#1179, see the limits) | works |
| ghost | a Robin coefficient that depends on the parameter | works | works | throws `EnzymeNoTypeError` in both modes (#1179) | works |
| ghost | `Δₕ!` on a `GhostVector` | works | works | works | works |
| ghost | `centered_difference!` on a `GhostVector` | works | works | works | works |
| sparse Jacobian | `AutoSparse` with `ast_sparsity_detector`, 1D diffusion residual | works | works | works | works |
| polygon | the datum `p₁ x₁ + p₂ x₂²` of `polygon_spaces`, mesh built outside the function | works | reverse works, forward throws `MethodError` (#1212) | reverse works, forward throws `MethodError` | reverse works, forward throws `MethodError` |
| polygon | the coefficient and load of a form on the `polygon_spaces`, datum fixed | works | reverse works, forward throws `MethodError` (#1212) | reverse works, forward throws `MethodError` | reverse works, forward throws `MethodError` |
| polygon | `polygon(V)`, the vertices as the parameters | throws `MethodError` | throws `IllegalTypeAnalysisException` | not run | not run |
| polygon | the mesh of `polygon(V)` built inside the function, vertices fixed | works | throws `IllegalTypeAnalysisException` | not run | not run |
| dirac | `assemble` of `innerₕ(dirac((p[1], p[2]), 2.0), v)`, the location as the parameters | throws `MethodError` | works | not run | not run |
| dirac | `assemble` of `innerₕ(dirac(DIRAC_POINT, p[1]), v)`, the strength as the parameter | works | works | not run | not run |
| levelset | `levelset(φ)` with `φ = x[1]^2 + x[2]^2 - p[1]^2`, mesh built inside the function | throws `ArgumentError` (#1172) | not run (#1211, see the limits) | not run | not run |

More facts about the table:

- ForwardDiff and Enzyme agree on every `works` cell to 1e-8 relative or better. The worst
  cell, the mesh endpoint, agrees to 5e-11. The rest agree to 1e-15 and 1e-16, and the
  cells measured on 2026-11-07 (the 2D Dirichlet, the ghost and three-summand reverse cells)
  agree to 6e-16 or better.
- The solver cells agree with their reference to 9e-16 or better (`factor_load`, `factor_ldiv`, `fdm_load`, both modes, all policies), to 3.8e-14 (`rank_one`) and to 2e-12 (`gmg_load`, `gmg_coefficient` where they work, against ForwardDiff of the direct solve, asserted to 1e-6). `chebyshev` under `CpuThreaded` and `CpuPolyester` agrees to 5.3e-16. The factorization cells use a 1D mesh of 17 nodes, `fdm_*`, `gmg_*` and `chebyshev` a 17 x 17 uniform and a 9 x 9 seeded non-uniform mesh.
- The 3D cells run on a `5 x 5 x 5` mesh in the `threed` group. The averaging operator `M₂`
  on a grid function and `Mₕ(κ)` in a form run under all three policies, at Enzyme's default
  type-analysis bounds; the other 3D cells run under `CpuSerial` only.
- Four fixes of v4.2.4 flipped cells. `SpaceWeights` is held by reference (3aefc79f), which
  makes the 3D `Mₕ(κ)` form fit Enzyme's type analysis (gpena/Bramble.jl#1154, six cells
  measured in `ad_reductions/1154_3d_threaded.md`) and removed the wrong forward gradient of the
  three-summand form (gpena/Bramble.jl#1124, `ad_reductions/1124_forward_offset.md`). The
  boundary memset of the averages is typed (gpena/Bramble.jl#1155, `ad_reductions/1155_m2_memset.md`),
  and the walk space reaches each piece of a ghost form not through a closure (9faf6345,
  gpena/Bramble.jl#1152).
- The ghost cells run on a non-uniform mesh only, since the ghost test helper builds those.
  The `polygon_datum` and `polygon_coefficient` cells run on both meshes under `CpuSerial`
  and on the non-uniform one under the threaded policies.
- Every `works` cell matches ForwardDiff to 1e-8 relative or better except
  `dirac_location`, whose ForwardDiff half throws: Enzyme matches a central difference
  there (1.2e-13 or better in both modes, asserted to 1e-5). The polygon `works` cells, reverse,
  match ForwardDiff to 6.6e-16 or better, and `ghost_data` and `ghost_robin` to 1.3e-16 or
  better (`ad_reductions/1171_geometric_inputs.md` and `ad_reductions/1153_constrained.md`).
- The ghost load, `Δₕ!` and `centered_difference!` cells differentiate in both modes under
  all three policies. Enzyme reverse of the load under the threaded policies was
  gpena/Bramble.jl#1157, `Δₕ!` reverse under `CpuPolyester` was gpena/Bramble.jl#1127, and
  forward of the load, in the layout the test uses (a closure over the form), crashed until
  gpena/Bramble.jl#1159; all three are fixed in v4.2.2. `centered_difference!` failed under
  `CpuSerial` on 2026-11-03 and passes now; the likely cause is the ghost facet band
  rewrite it predates (not bisected). The
  measurements are in `ad_reductions/1157_ghost_load.md` and `ad_reductions/1158_ghost_stencil.md`
  (gpena/Bramble.jl#1158).

## Enzyme under the threaded policies

Enzyme differentiates `CpuSerial`, `CpuThreaded` and `CpuPolyester`, and the derivative runs
threaded too. It never enters a `Threads.@threads` or `@batch` body: each engine hook has
its own forward and reverse rule in `ext/BrambleEnzymeExt.jl`, and the rule runs the
derivative threaded itself. The evidence is `test/ext/enzyme_threaded_ext.jl` (196
passing in its last full run; its three-summand reverse cases, added after, pass in their
own run under both threaded policies), `test/ext/enzyme_polyester_hooks.jl` (66 passing)
and the table above, where every feature
that ran under `CpuSerial` also ran under both threaded policies, except the cells marked
`not run` and the two relation cells, which throw under `CpuThreaded` (gpena/Bramble.jl#1179).

Inside a user's own threaded region, a `Threads.@threads` loop of the caller's, the
derivative runs its bands in order, as the primal does: a `:static` loop cannot start there.
The nested reverse and forward modes of `test/ext/enzyme_policy_cases.jl` pin this.

## DI settings Enzyme needs

Measured on `difference` (`D₋ₓ`) and on a closure that fills a captured buffer, `CpuSerial`,
non-uniform mesh. The reference of the buffer cell is a central difference, since ForwardDiff
cannot write a `Dual` into a `Float64` buffer.

| `AutoEnzyme` | Mode | Result |
|:--|:--|:--|
| `AutoEnzyme()` | reverse and forward | throws `EnzymeMutabilityException` |
| `mode = set_runtime_activity(…)` | reverse and forward | throws `EnzymeMutabilityException` |
| `mode = set_runtime_activity(…), function_annotation = Enzyme.Const` | reverse and forward | works |

For a closure that writes the parameters into a captured buffer, which the differentiated
function then reads:

| `function_annotation` | Mode | Result |
|:--|:--|:--|
| `Enzyme.Const` | reverse and forward | returns a zero gradient, no error |
| `Enzyme.Duplicated` | reverse and forward | works (1.9e-10 against the central difference) |

So: `AutoEnzyme(; mode = set_runtime_activity(Reverse), function_annotation = Enzyme.Const)`
for an ordinary closure, and `function_annotation = Enzyme.Duplicated` for a closure that
captures a buffer the input is written into. The plain call is an error, never a wrong
value. The `Const` annotation on a buffer-writing closure is a wrong value: Enzyme treats
the captured buffer as constant, so its derivative is dropped. The forward mode is
`set_runtime_activity(Forward)`.

## Sparse Jacobians

`AutoSparse(AutoEnzyme(; mode = set_runtime_activity(Forward), function_annotation =
Enzyme.Const); sparsity_detector = Bramble.ast_sparsity_detector(a, U -> Mₕ(U)),
coloring_algorithm = GreedyColoringAlgorithm())` returns the Jacobian of the 1D diffusion
residual `A(u) u - F` that ForwardDiff's dense Jacobian does, to 1e-15, under the three
policies and on both meshes. Reverse mode works too. The detector never runs the residual,
so a sparse cell's verdict is its inner backend's.

## Accepted limits

- **`idxs` must be unique.** A sweep over repeated indices has no defined reverse.
- **A closure that reaches `v` only through a global gets no gradient.** Per-slab Enzyme
  compilations treat a global as constant. `test/ext/enzyme_threaded_ext.jl` pins it with a
  `@test_broken`.
- **Ghost spaces whose relation data depend on the parameter differentiate under
  `CpuSerial` and `CpuPolyester`, not under `CpuThreaded`.** In Enzyme reverse the cost was
  compile time, not a missing derivative (gpena/Bramble.jl#1153, fixed; measured in
  `ad_reductions/1153_constrained.md`): one facet of `_assemble_relation`
  (`src/assembly/ghost_relations.jl`) compiled for 732 s and 3.7 GB, and `constrained` passed
  the 10 GB cap at about 310 s. Since PR #1182 a reverse rule over `_assemble_relation`
  (`ext/BrambleEnzymeExt/relations.jl`) takes one forward derivative of the function per
  float of the relation, so reverse never compiles its forms: one facet takes 52 s and
  1.6 GB, `constrained` 95 s and 2.1 GB. The `ghost_data` and `ghost_robin` cells match
  ForwardDiff in both modes under `CpuSerial` and `CpuPolyester` (0 to 1.3e-16). Under
  `CpuThreaded` both modes throw `EnzymeNoTypeError` in the forms of the relation
  ([gpena/Bramble.jl#1179](https://github.com/gpena/Bramble.jl/issues/1179)), and the cells
  pin the throw. The rule has open limits: reverse `constrained` on a composite space fails in
  `_relation_rows` ([gpena/Bramble.jl#1180](https://github.com/gpena/Bramble.jl/issues/1180)),
  and a relation that couples the ghosts of a node has no derivative of its dense `lu`
  ([gpena/Bramble.jl#1181](https://github.com/gpena/Bramble.jl/issues/1181)). The rule
  refuses with an `ArgumentError` a parameter in the mesh, a `Duplicated` annotation that
  captures the space, a relation that captures a mutable, and batched reverse.
- **Ghost assembly works in both Enzyme modes under all three policies.** The cell with `Δₕ`
  in the form and the parameters in the coefficients matches ForwardDiff to 8.5e-16: forward
  and reverse under `CpuThreaded` and `CpuPolyester` in 115 to 169 s and 2.3 to 3.3 GB, and
  under `CpuSerial` since 9faf6345 (5.1e-16 in reverse; it failed with `EnzymeNoTypeError` for
  the commits from 3aefc79f up to 9faf6345, `ad_reductions/1152_ghost_matrix.md`). The cell
  without `Δₕ` differentiates in reverse under `CpuSerial` (8.5e-16 after 9faf6345, 370 s
  and 4.9 GB on 2026-11-07); its forward mode and the threaded policies were not run.
- **The level-set parameter has no Enzyme verdict.** The cell is `:skip` under the three
  policies because the verdict is unstable: the first fresh process returned a gradient of
  zeros in both modes (relative error 1.0 against the central difference), and five later
  ones aborted in LLVM (`Instruction does not dominate all uses!`, signal 6) with the cell
  and the file unchanged (gpena/Bramble.jl#1211). ForwardDiff throws the named
  `ArgumentError` of gpena/Bramble.jl#1172, since a `Float64` grid has no derivative to
  give. The cell has a group of its own, `levelset`, so that an abort ends no other cell.
- **The polygon `works` cells fail in Enzyme forward.** Reverse works under the three
  policies; forward throws `MethodError`, since Enzyme has no forward rule for `deepcopy`
  (gpena/Bramble.jl#1212). The datum of `polygon_spaces` and the coefficients differentiate
  with the mesh built outside the function.
- **A polygon mesh built inside the function, and the polygon's vertices, fail in Enzyme.**
  Both modes throw `IllegalTypeAnalysisException` in the exact `Rational` and `BigFloat`
  arithmetic of the grid closure, under `CpuSerial`, the only policy run. ForwardDiff
  differentiates the mesh built inside; the vertices are `Rational{BigInt}`, which a `Dual`
  cannot become, so it throws `MethodError` there.
- **Dirac location has no ForwardDiff derivative.** `dirac` converts its point to `Float64`,
  so ForwardDiff throws `MethodError`; Enzyme differentiates the location and the strength
  in both modes, measured under `CpuSerial` only. The cells with `not run` under the
  threaded policies (`dirac`, the polygon vertices and the mesh built inside) were not
  measured there.
- **Plan geometry is inactive.** The plans that carry the mesh geometry (the replay plans of
  the threaded assembly and the matrix-free plan `_mf_plan`) are inactive for Enzyme
  (`EnzymeRules.inactive` on `_mf_plan` in `ext/BrambleEnzymeExt/threaded.jl`), so a
  coordinate read only through a plan gets no derivative. The `mesh` row that
  assembles in the endpoint `b` works under `CpuSerial` in both modes (6e-10 and 2e-10
  against ForwardDiff); that route was not measured through a matrix-free operator or under
  the threaded policies.
- **Compile memory.** `_batch_mf_bands_coef`, the matrix-free reverse with a coefficient,
  compiles at 5.96 GB and 747 s under `CpuPolyester`, against 2.46 GB and 215 s under
  `CpuSerial`, measured on 2026-11-02 by the `CpuPolyester` matrix-free rule's own memory
  run, and not re-run here.
- **Enzyme does not carry `Ā` through a kept factorization (departure D1).**
  `pde_solve(fact, F)` and `ldiv!(uₕ, fact, F)` differentiate in `F` only (the `factor_load`
  and `factor_ldiv` cells, both modes, all three policies, 9e-16 or better). A factorization
  object has no shadow that can carry `Ā`, so a rule would need the factorization to keep `A`
  and a second store for its cotangent across two rules. A factorization with a shadow is
  refused by name, and so is a standard-library dense `lu` with an all-zero shadow. ForwardDiff
  does differentiate `A` through `pde_solve(sparse_factorize(A), F)`. Differentiate
  `pde_solve(A, F)` under Enzyme.
- **A factorization built inside the differentiated function is unsupported under Enzyme
  (departure D8).** The `factor_matrix` cell throws `IllegalTypeAnalysisException` in both
  modes, both meshes and all three policies, before the rule on `sparse_factorize` that names
  `pde_solve(A, F)` runs: `sparse_factorize` infers a `Union` (across backends once
  AppleAccelerate loads), which Enzyme's type analysis rejects. Each fix measured, one concrete
  SuiteSparse type or an inference barrier, passes reverse mode and aborts the process in forward
  mode on an LLVM assertion (`invertedReturn->getType() == gutils->getShadowType(...)`).
  Build the factorization outside the function, as the `factor_load` and `factor_ldiv` cells do,
  or differentiate `pde_solve(A, F)`. Reverse mode also needs the factorization's adjoint solve,
  and a backend without one is refused by name.
- **`fact \ F` is not routed, and `factorize` of a `Dual` matrix overflows (departure D2).**
  Each backend defines its own `\`: call `pde_solve(fact, F)`. `ldiv!(uₕ, factorize(A), F)`
  over a `Dual` matrix overflows inside the standard library's `factorize`, upstream of Bramble.
- **ForwardDiff refuses a `RankOneUpdate` over a `Dual` matrix (departure D3).** The `rank_one`
  cell throws an `ArgumentError` naming `RankOneUpdate`, and Enzyme differentiates it in both
  modes under all three policies (1e-14 or better against ForwardDiff of the plain solve of
  `A + w wᵀ`). Under ForwardDiff call `pde_solve(A, F; mean_zero = w)`. The Enzyme rule needs
  `K.A` to be a `SparseMatrixCSC` (any other type is refused by name) and refuses batched reverse
  mode.
- **`fdm_solve` has no derivative in its coefficients or mesh (departure D4).** Both backends
  refuse them by name (`fdm_coefficient`). Call `pde_solve(assemble(a, l)..., F)`. In `F` both
  modes work (`fdm_load`), with limits under Enzyme: forward mode has width 1 only (a wider batch
  is refused by name), reverse mode refuses a form with a non-symmetric term (advection) by
  name, a `KroneckerLinearOperator` operand has no rule, and only a 2D or 3D scalar
  `BilinearForm` gets a rule, so ghost, 1D and composite forms have none and Enzyme reports
  `fdm_solve`'s own `ArgumentError` for them.
- **`gmg_solve`: ForwardDiff differentiates the cycles, Enzyme an implicit rule (departures D5
  and D9).** Under ForwardDiff an active coefficient differentiates the cycles themselves, which
  agrees with the direct solve's derivative to 1e-6 at `tol = 1e-10`. Under Enzyme `b` is
  differentiated in both modes, and the coefficients `build` captures in reverse mode only
  (`gmg_coefficient`: a scalar and a grid function together). Under `CpuSerial` forward mode
  with a coefficient active throws an `ArgumentError` naming "forward mode". Under Enzyme:
  - Reverse mode refuses a non-Hermitian form by name, and a form whose operator changed between
    the solve and the reverse pass (a `Ref` the form holds, set after the solve).
  - An active mesh is refused by name: build the mesh outside the differentiated function. A
    mesh built inside a reverse loss reaches the rule with a shadow and is refused the same way
    (measured on a 9 x 9 mesh, 2026-11-13).
  - A composite-space `build` stops in LLVM (`broken gc calling conv fix`), not with an
    `ArgumentError`: `build` is an opaque closure type, so no rule signature can exclude it.
  - A ghost-space `build` with active coefficients is refused in both modes, and a constant
    ghost-space `build` that captures a scalar is refused in reverse mode even with only `b`
    active.
  - Forward mode refuses any `build` whose shadow carries data (departure D9). Under runtime
    activity Enzyme can pass a constant closure's primal, captured values included, as its
    shadow, so no rule can tell a constant `build` from an active one whose tangent equals its
    value. Three forward designs gave silent wrong tangents (relative error up to 0.74). Forward
    mode with only `b` active still works.
  - Forward mode over-refuses a constant number captured by value from a non-`const` global, and
    `gmg_mixed(0.3, <untyped global Ref>)`. Use a `const` global, a `Ref`, or a `Const` or
    `DI.Constant` argument. `DI.derivative` in a coefficient is refused too.
  - An active value stored into a shared mutable that `build` captures (a global `Ref`, a `const`
    global vector, a `Const` vector argument) gets 0 from Enzyme in both modes, with no Bramble
    rule involved: Enzyme drops it on a plain `c -> (GR[] = c; h(GR))`. Close over the value
    instead.
  For a form any of these refuse, differentiate `pde_solve(assemble(build(gridspace(Ωₕ))), b)`.
- **Batched forward through `gmg_solve` ends the process under the threaded policies
  ([gpena/Bramble.jl#1232](https://github.com/gpena/Bramble.jl/issues/1232), departure D7).** The
  issues ask for `CpuSerial`; the cells ran under all three policies. Under `CpuThreaded` and
  `CpuPolyester`, Enzyme forward mode of `gmg_load` and `gmg_coefficient` aborts the process with
  the LLVM assertion `invertedReturn->getType() == gutils->getShadowType(call.getType())` when
  the forward call is batched, which DI's gradient is: its width is the number of parameters (3
  in the cells). The abort comes before the rule runs, so the named forward refusal of
  `gmg_coefficient` is not reached either. It was there before the coefficient rule (at
  03d4c9c6), width 1 works under `CpuThreaded`, and `CpuSerial` works at any width. `gmg_solve`
  infers `Any` under both threaded policies and a concrete type under `CpuSerial`; that is a likely
  precondition, measured but not shown to be the cause. Both forward halves are `:aborts` under
  the threaded policies, each in a child of its own. To differentiate in forward mode there, use
  width 1: a gradient with a single parameter, or `DI.pushforward` with one tangent (DI's
  pushforward takes a tuple of tangents and its width is the tuple's length; only the
  single-parameter case was measured). Reverse mode is not affected. `chebyshev` under Enzyme
  throws `IllegalTypeAnalysisException` under `CpuSerial` only; the two threaded policies match
  ForwardDiff.
- **Plain `Enzyme.Reverse` needs runtime activity (`set_runtime_activity`).** Without it `ldiv!`
  throws `EnzymeRuntimeActivityError`, and DI's plain `AutoEnzyme` throws
  `EnzymeMutabilityException` (the table of the DI settings above).
- **Follow-ups.** `gmg_solve` with `Dual`s of different tags
  ([gpena/Bramble.jl#1227](https://github.com/gpena/Bramble.jl/issues/1227)), and a CHOLMOD
  factorization into a `VectorElement` load, which fails with a `MethodError` before any AD is
  involved ([gpena/Bramble.jl#1228](https://github.com/gpena/Bramble.jl/issues/1228)).

## Open failures

- A relation that couples the ghosts of a node (`conormal`, `∂ₙ(u; K)` with a full `K`)
  reaches the dense branch of `_block_solve` (`src/assembly/ghost_relations.jl`, `lu` at line
  1218), which has no Enzyme derivative: `EnzymeNoDerivativeError` on `dgetrf`, in forward and
  in reverse, at `_block_solve`, `_relation_closure` and `constrained` alike (13 to 54 s).
  The relations of the cells above never take that branch
  ([gpena/Bramble.jl#1181](https://github.com/gpena/Bramble.jl/issues/1181)). No cell.
- A 3D ghost `Δₕ` under `CpuSerial` in reverse aborts in LLVM verification
  (`Instruction does not dominate all uses!`). It is there at 501128cb, before this plan's
  changes. No issue yet, and no cell.

## Compile cost of the cells

Measured 2026-11-07 on mains power, the machine otherwise idle, at default `-O`, 4 threads;
the v4.2.4 figures follow the table.
Each child is one policy and one group, run alone as the test runs it: Enzyme compiles every
cell of the group on both meshes in both modes, and the wall time includes the ForwardDiff
reference. Peak is the child's peak memory footprint (`/usr/bin/time -l`), the figure the
local 10 GB cap reads; the resident set is lower. Each unchanged child ran twice and the table
gives both runs, wall time first; the children split on 2026-11-07 ran once.

| Group | Cells | `CpuSerial` | `CpuThreaded` | `CpuPolyester` |
|:--|:--|:--|:--|:--|
| dense | mesh, operators, interpolation | 181, 182 s; 3.2 GB | 206, 197 s; 3.4 GB | 198, 201 s; 3.4 GB |
| assemble | the 1D and 2D assemblies, 1D Dirichlet | see below | 533 s; 6.5 GB | 315 s; 4.9 GB |
| three | the three-summand form | see below | 530 s; 6.3 GB | see below |
| boundary | 2D Dirichlet | 181 s; 3.4 GB | 291 s; 4.2 GB | 197 s; 3.5 GB |
| geometry | the mesh endpoint | 132, 132 s; 2.8 GB | 94, 94 s; 2.4 GB | 94, 96 s; 2.4 GB |
| solve | `pde_solve`, `semidiscretize` | 142, 143 s; 3.0 GB | 178, 178 s; 3.3 GB | 168, 170 s; 3.2 GB |
| operators | matrix-free, Kronecker | 112, 112 s; 2.7 GB | 216, 217 s; 3.7 GB | 205, 207 s; 3.6 GB |
| ghost_fast | the ghost cells, before the relation cells left it | 702, 709 s; 5.4 GB | 114, 114 s; 2.6 GB | 120, 121 s; 2.6 GB |
| threed | the 3D cells | 2514, 2394 s (1235, 1203 s CPU); 8.8 GB | see below | see below |
| sparse Jacobian | the residual, four calls | 130 s; 3.1 GB | 166, 167 s; 3.5 GB | 143, 146 s; 3.1 GB |
| DI settings | ten calls | 101, 101 s; 2.5 GB | | |

Before the split, `assemble` held the `three` and `boundary` cells: `CpuSerial` 662 and 664 s,
6.6 GB; `CpuPolyester` 663 and 669 s, 8.5 GB; `CpuThreaded` was stopped by the 10 GB cap at
11.3 GB. The `CpuSerial` and `CpuPolyester` children of the split `assemble` and `three`
groups were not run again. The `threed` wall time is twice its CPU time in both runs; the
second `CpuSerial` sparse run, 2053 s for 123 s of CPU, is left out as a stall.

The `ghost_fast` row predates the load and `centered_difference!` cells' Enzyme runs: each of the
latter's six calls compiled in under 30 s and 1.6 GB (measured alone, 2026-11-08, 2 threads),
and the child was re-timed on 2026-11-11 (second table). Every child stays under the 10 GB cap;
the largest of the first table is `CpuSerial` `threed`, at 8.8 GB, and of the second
`CpuThreaded` `polygon`, at 9.80 GB. A cell that ends in an Enzyme failure still pays its
compile. Before 2026-11-07 the ghost assembly cells cost more than 22 minutes each before they
threw; they now differentiate in 6 minutes.

The groups added or split on 2026-11-11 were timed as the test runs them (one child per
policy and group, 2 threads, `--optimize=1`, under load, so the times are upper bounds; every
exit normal and every non-skip cell line equal to its pin). `ghost_fast` here holds
`ghost_assemble`, `ghost_inner`, `ghost_load`, `ghost_runtime` and `ghost_stencil`, and
replaces the row above:

| Group | Cells | `CpuSerial` | `CpuThreaded` | `CpuPolyester` |
|:--|:--|:--|:--|:--|
| inputs | `dirac` location and strength, `polygon(V)` vertices, polygon mesh built inside | 611 s; 8.88 GB | not run | not run |
| levelset | `levelset(φ)`, the ForwardDiff half only | 79 s; 2.29 GB | not run | not run |
| polygon | `polygon_spaces` datum and coefficients | 581 s; 7.21 GB | 777 s; 9.80 GB | 696 s; 8.99 GB |
| relation_data | `ghost_data` | 409 s; 5.23 GB | 736 s; 8.72 GB | 570 s; 7.83 GB |
| relation_robin | `ghost_robin` | 452 s; 5.22 GB | 620 s; 8.93 GB | 539 s; 8.09 GB |
| ghost_fast | the five ghost cells | 499 s; 5.22 GB | 480 s; 5.68 GB | 576 s; 5.47 GB |

`inputs` under `CpuSerial` is 1.1 GB under the 10 GB cap and `polygon` under `CpuThreaded`
0.2 GB, the group to split if the cap kills it under load. The relation cells were split
from `ghost_fast` because together they passed the cap (`CpuPolyester` 10.23 GB, `CpuThreaded`
killed at 10.55 GB). `levelset` has a group of its own so that an Enzyme abort ends no other
cell; the two `not run` entries of its row and of `inputs` are cells that are `:skip` under
the threaded policies.

None of the children of the first table was re-timed for v4.2.4, so the rows above do not include the cells it
changed. Their compile figures, each from a single run of the note's reproducer at 2
threads (one process a cell, with another Enzyme job on the machine, so indicative):

| Cell | Policy and mode | Time | Peak |
|:--|:--|:--|:--|
| `assemble_averaged_3d` (`Mₕ(κ)`, 3D) | `CpuSerial` reverse, forward | 119 s, 87 s | 2.45 GB, 1.85 GB |
| | `CpuThreaded` reverse, forward | 206 s, 178 s | 2.62 GB, 2.69 GB |
| | `CpuPolyester` reverse, forward | 160 s, 122 s | 2.37 GB, 2.25 GB |
| `ghost_assemble` | `CpuThreaded` reverse, forward | 169 s, 153 s | 2.70 GB, 2.31 GB |
| | `CpuPolyester` reverse, forward | 153 s, 115 s | 3.29 GB, 2.77 GB |
| `average_z_3d` (`M₂`) | each policy and mode | under 17 s | not recorded |
| `ghost_data`, one facet of the relation, before the rule of #1182 | `CpuSerial` reverse | 732 s | 3.7 GB |
| `ghost_robin`, one facet of the relation, before the rule | `CpuSerial` reverse | 744 s | 3.2 GB |
| `constrained` with parameter data, before the rule | `CpuSerial` reverse | stopped at 311 s | 11.6 GB |
| `ghost_data` cell, with the rule | `CpuSerial` reverse, forward | 247 s, 249 s | 4.35 GB, 3.34 GB |
| | `CpuPolyester` reverse, forward | 396 s, 309 s | 6.07 GB, 4.87 GB |
| | `CpuThreaded` reverse, forward (both throw) | 421 s, 334 s | 6.36 GB, 5.39 GB |
| `ghost_robin` cell, with the rule | `CpuSerial` reverse, forward | 259 s, 200 s | 4.31 GB, 3.44 GB |
| | `CpuPolyester` reverse, forward | 384 s, 300 s | 6.10 GB, 4.96 GB |
| | `CpuThreaded` reverse, forward (both throw) | 442 s, 355 s | 6.37 GB, 5.49 GB |

The `ghost_data` and `ghost_robin` rows give the wall time of the process, package loading
(about 70 s) included.

The `ghost_assemble` serial cell was 139 s at 501128cb and 120 s and 84 s (reverse, forward) to
its failure at e3fcca1d; its time after 9faf6345 was not recorded. `constrained` reverse
was stopped by the 10 GB cap of `julia_slot.sh`, so its peak is what the process had reached.
The `threed` child under the threaded policies now holds the `M₂` and `Mₕ(κ)` cells on both
meshes in both modes; it was not timed as a child.

The 2026-11-03 figures, at 2 threads and as maximum resident set, were: dense 179 s, 3.4 GB (`CpuSerial`); assemble 578 s,
3.9 GB; solve 176 s, 3.9 GB; operators 118 s, 2.5 GB; sparse Jacobian 133 s, 1.8 GB; DI
settings 100 s, 2.8 GB.

## Compile cost of the solver cells

Measured 2026-11-13, at 2 threads in `--optimize=1` children, alone on the machine, one child
per policy and group as the test runs it (wall time in seconds, `/usr/bin/time -l` peak memory
footprint, one run each). The wall time includes about 70 s of package loading. Every group
stays under the 10 GB cap and every wall time under 10 minutes, except one run noted below.

| Group | Cells | `CpuSerial` | `CpuThreaded` | `CpuPolyester` |
|:--|:--|:--|:--|:--|
| factor_matrix | `factor_matrix` | 112 s; 2.70 GB | 119 s; 2.98 GB | 128 s; 3.11 GB |
| factor_kept | `factor_load`, `factor_ldiv` | 85 s; 2.22 GB | 86 s; 2.24 GB | 86 s; 2.25 GB |
| rank_one | `rank_one` | 116 s; 2.54 GB | 131 s; 2.68 GB | 124 s; 2.72 GB |
| fdm | `fdm_load`, `fdm_coefficient` | 96 s; 2.37 GB | 96 s; 2.38 GB | 96 s; 2.45 GB |
| gmg_load | `gmg_load` | 89 s; 2.27 GB | 91 s; 2.27 GB, plus 84 s; 2.25 GB for the forward child | 1501 s (88 s CPU); 2.17 GB, plus 87 s; 2.21 GB for the forward child |
| gmg_coefficient | `gmg_coefficient` | 131 s; 2.47 GB | 342 s; 3.79 GB, plus 84 s; 2.27 GB for the forward child | 300 s; 3.53 GB, plus 87 s; 2.31 GB for the forward child |
| chebyshev | `chebyshev` | 124 s; 2.91 GB | 306 s; 4.29 GB | 273 s; 4.07 GB |

The forward half of the two `gmg` cells under the threaded policies ends the process, so the
test gives each its own child (`:aborts`): the figures after "plus" are that child, which ran
`nonuniform` forward only. The main `gmg` figures are from the run that skips the aborting
forward calls, as the test runs it.

`gmg_load` under `CpuPolyester` took 1501 s wall for 88 s of CPU (user 87.9 s, system 1.5 s),
2 threads, `--optimize=1`: about 23 minutes in which the process did not compute. Its cells
ended normally and agree with the pins. The cause is unknown; the same child under `CpuSerial`
(89 s) and `CpuThreaded` (87 s), and the first `CpuPolyester` run of the group (92 s, which
ended in the forward abort), did not stall. It is a measurement of that run, and has not been
reproduced.

Under `CpuSerial`, the cell that ends in an Enzyme failure still pays its compile: `factor_matrix`
and `chebyshev` took 112 s and 124 s to throw `IllegalTypeAnalysisException`.

## Reproducing

```bash
# ForwardDiff cells, no Enzyme needed
julia --project=test -t 2 -e 'using Test; include("test/TestUtils.jl"); include("test/space/ad_coverage.jl")'
```

With Enzyme in the environment (the weekly `ad-*` shards install it into the test project),
the same file runs the Enzyme cells in child processes, with `BRAMBLE_TEST_GROUP=ad` and
`BRAMBLE_TEST_FILE=space/ad_coverage.jl`. No cell is marked slow now (`ghost_data` was, until
gpena/Bramble.jl#1153 was fixed), so `BRAMBLE_AD_COVERAGE_SLOW=true` adds nothing.
