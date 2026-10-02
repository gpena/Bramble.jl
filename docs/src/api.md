```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# API reference

This reference lists the docstrings of the public names in `Bramble.jl`'s core library,
grouped by the stage of a computation they belong to. The SciML, automatic differentiation
and solver integrations are on [Scientific computing: SciML, AD and
solvers](api_sciml.md). A typical program follows the pages in order. It describes the domain,
builds a mesh on it, defines a grid space, applies difference operators, and assembles and
solves a form. If you are new to the library, start with the tutorials and come here to look
up a signature or an option.

- [Utilities](api/utilities.md): linear algebra backends and other package-wide helpers.
- [Geometry](api/geometry.md): sets, intervals, markers and domains, which describe where a
  problem lives and how its boundary is labelled.
- [Meshes](api/meshes.md): mesh types and constructors, points and spacings, indexing and
  boundaries, and adaptation.
- [Grid spaces](api/spaces.md): function spaces on a mesh, their degrees of freedom, grid
  functions, restriction and averaging, and interpolation between spaces.
- [Difference, jump and average operators](api/operators.md): the discrete derivatives and
  related operators that act on grid functions.
- [Inner products and norms](api/inner_products.md): discrete inner products and the norms
  built from them.
- [Forms](api/forms.md): building, assembling and solving bilinear and linear forms,
  including matrix-free operators, Dirichlet conditions and sparsity analysis.
- [Exporters](api/exporters.md): writing grid functions to files for visualisation.
