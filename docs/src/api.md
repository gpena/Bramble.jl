```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# API reference

Docstrings of the exported and `public` names of the core library, grouped by stage of a
computation. Start with the tutorials; look up a signature or an option here. The SciML,
automatic differentiation and solver names are on [Scientific computing: SciML, AD and
solvers](api_sciml.md). The solver pages also document some names that are neither exported
nor `public`; they are called as `Bramble.name`. Private docstrings for contributors are under
Internals.

- [Utilities](api/utilities.md): linear algebra backends and other package-wide helpers.
- [Geometry](api/geometry.md): sets, intervals, markers and domains, which describe where a
  problem lives and how its boundary is labelled.
- [Meshes](api/meshes.md): mesh types and constructors, points and spacings, indexing and
  boundaries, adaptation, and ghost meshes.
- [Grid spaces](api/spaces.md): function spaces on a mesh, their degrees of freedom, grid
  functions, restriction and averaging, interpolation between spaces, and spaces with ghost
  points and boundary relations.
- [Difference, jump and average operators](api/operators.md): the discrete derivatives and
  related operators that act on grid functions, and the normal derivatives of a facet.
- [Inner products and norms](api/inner_products.md): discrete inner products and the norms
  built from them.
- [Forms](api/forms.md): building, assembling and solving bilinear and linear forms,
  including matrix-free operators, Dirichlet conditions and sparsity analysis.
- [Exporters](api/exporters.md): writing grid functions to files for visualisation.
