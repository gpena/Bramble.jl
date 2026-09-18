```@meta
CollapsedDocStrings = false
```

# Evaluating Gridap.jl's subsystems for Bramble

A decision record, from the evaluation [gpena/Bramble.jl#219](https://github.com/gpena/Bramble.jl/issues/219)
asked for. Five of Gridap.jl's subsystems were held against what Bramble needs; this page
records what was decided about each and why, so the question is not reopened from scratch.

Gridap is a finite element library and Bramble is a finite difference one, so most of what
makes Gridap what it is — reference elements, shape function bases, coordinate maps, mesh
adaptivity — has no counterpart here and is not discussed. What is worth borrowing is the
handful of subsystems that solve problems a Cartesian finite difference library also has.

## 1. Static tensor values

**Gridap has** `VectorValue`, `TensorValue`, `SymTensorValue` and `ExteriorFormValue{K,D}`:
bitstype, stack-allocated, with unrolled algebra (`⋅`, `⊗`, `∧`, `⋆`, `tr`, `dev`).

**Bramble has** flat coefficient vectors and `NTuple{D, VectorElement}`, one grid function per
component.

**Decision: not adopted, and the gap is smaller than it looks.** The place a tensor type would
pay is a point evaluation that carries several components at once, and Bramble's operators are
written the other way round — one traversal per component over the whole grid, which is what
makes them vectorise and allocate nothing. A `VectorValue` per grid point would replace a
contiguous `Float64` sweep with a sweep over a struct, and the component loop is already
unrolled by `Val`-recursion.

Where a tensor genuinely appears — `εₕ(u)` in the elasticity example, the gradient tensor of a
composite trial function — the quantity that matters is not a point value but a *block* of the
assembled matrix, and [#234](https://github.com/gpena/Bramble.jl/issues/234) specifies it as
nested Julia tuples expanded at the form builder into the same single-block products the
hand-written version produces. That keeps the stencil algebra scalar, which is the property
`local_stencil` is built on.

If a point-wise tensor is ever needed, `StaticArrays.SVector`/`SMatrix` is the answer rather
than a bespoke type: the algebra is the same, and Bramble would be adding a dependency it can
drop again rather than a type hierarchy it has to maintain.

## 2. Continuous differential forms

**Gridap has** exact analytical ``k``-forms with exterior derivatives, codifferentials and
Koszul contractions, used to verify discrete complexes against continuous ones.

**Decision: not adopted as code; the verification idea is worth keeping.** What a de Rham
verification harness would check is that restriction commutes with differentiation,
``R_h d = d_h R_h``. Bramble already checks the discrete half of every such statement directly
— `test/space/commutation.jl`, `test/space/sbp_identities.jl`,
`test/space/discrete_calculus_identities.jl` — against closed forms and with Supposition over
random meshes. A continuous form algebra would let those be written more compactly; it would
not let them check anything they do not already check, and it is a large subsystem to carry
for notation.

## 3. First-class facet topology

**Gridap has** `BoundaryTriangulation` and `SkeletonTriangulation`: ``(D-1)``-dimensional
topologies with their own normals, measures and paired cell indices.

**Decision: not adopted, and [#157](https://github.com/gpena/Bramble.jl/issues/157) is the
reason.** This was the strongest candidate before the surface integral was built. A facet mesh
would have carried the weights, the normals and the incidence for `inner_Γ`.

What the implementation found is that on a Cartesian grid none of the three needs a structure.
The surface weight is `ω(p) = Σ_{d ∈ N(p)} ∏_{e≠d} ĥ_e`, a product of numbers `half_spacings`
already holds, computed per point with no stored vector and no staleness; the normal of a
coordinate face is a constant known from the face's index; and "which faces pass through this
point" is the index arithmetic `I[d] == 1 || I[d] == n_d`. A `BoundaryFacetMesh` would be an
object whose only content is answers that are cheaper to recompute than to look up.

This is a statement about Cartesian grids, not about the design. The moment facets stop being
axis-aligned — the cut cells [#219](https://github.com/gpena/Bramble.jl/issues/219) lists as
out of scope, or the polygonal sub-domains of the v5.x line — the incidence *is* data and has
to be stored, and this decision should be revisited there rather than inherited.

DG jump and average terms ([#212](https://github.com/gpena/Bramble.jl/issues/212)) are the
other caller a skeleton topology would serve. Bramble's `jumpₓ` already works pointwise on the
same principle, and the same argument applies until the interfaces stop being grid faces.

## 4. Constrained and quotient spaces

**Gridap has** `ZeroMeanFESpace` and `FESpacesWithConstantFixed`, which quotient out the
kernel of a pure Neumann problem at the space level.

**Decision: worth having, and it belongs with the boundary conditions that create the
problem.** A pure Neumann Laplacian is singular — constants are in its kernel — and with
`inner_Γ` in place it is now expressible, so the problem is reachable in a way it was not
before. The two fixes are pinning one degree of freedom and enforcing ``\sum_I H_I u_I = 0``;
the second is the better conditioned one and is a rank-one modification of the assembled
matrix, not a new space type.

Filed as [#272](https://github.com/gpena/Bramble.jl/issues/272), against the milestone that
owns Neumann and Robin assembly rather than built here, since that is where a user meets the
singularity. The compatibility condition on the data is the other half of the same problem and
is now checkable, both integrals being computable with `inner_Γ`.

## 5. Dirac point sources

**Gridap has** `DiracDeltas` for point loads in weak forms.

**Decision: already done.** `dirac(x₀, strength)` builds a `DiracSource` node
(`src/form/ast.jl`), documented, tested in `test/form/dirac.jl`, and used in the
`point_sources_flux` worked example. The evaluation's own list had it as a prototype to write;
it predates the issue.

## Summary

| Subsystem | Decision |
|:---|:---|
| `TensorValues` | Not adopted. Per-component traversal is what makes the operators fast; a tensor per point undoes it. `StaticArrays` if ever needed. |
| `DifferentialForms` | Not adopted. The commutation properties are already tested directly. |
| `BoundaryTriangulation` / `SkeletonTriangulation` | Not adopted for Cartesian grids: weights, normals and incidence are all index arithmetic. Revisit for cut cells and polygonal sub-domains. |
| `ZeroMeanFESpace` | Worth having; filed against the Neumann/Robin milestone. |
| `DiracDeltas` | Already implemented as `dirac`. |
