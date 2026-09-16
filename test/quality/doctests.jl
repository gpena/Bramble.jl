module QualityDoctestsTests

using Test
using Bramble
using Documenter

# Decoupled from the documentation build (gpena/Bramble.jl#251): `docs/make.jl` sets
# `doctest = false` and points here in a comment. `Documenter.doctest` only needs the module
# it inspects and its own package loaded -- none of the six `jldoctest` blocks currently in
# `src/` reach for anything outside `Bramble`/`SparseArrays` (already a test dependency) -- so
# running it here does not pull the rest of `docs/Project.toml`'s heavier dependencies
# (Enzyme, Makie, NonlinearSolve, ...) into the test environment for a handful of doctests.
#
# Three of those six doctests (marker.jl's `boundary_symbol_to_cartesian`, geometry/marker.jl's
# `symbols`/`conditions`, backend.jl's `Backend`) call internal, non-exported names bare. A
# full `docs/make.jl` build resolves those because `docs/src/api.md` sets `CurrentModule =
# Bramble`, evaluating API docstrings' doctests as if written inside the module itself --
# which also means those three were never actually exercised by a full build either, since
# `boundary_symbol_to_cartesian` (an internal helper with no page of its own, per the
# `missing_docs` warning `docs/make.jl` already carries) is never pulled into any `@docs`
# block for `CurrentModule` to apply to. `Documenter.doctest(Bramble)` here checks every
# docstring regardless of whether a page references it, so it needs those names imported
# explicitly instead of relying on page context that doesn't exist standalone.
DocMeta.setdocmeta!(
    Bramble, :DocTestSetup,
    :(using Bramble; using Bramble: boundary_symbol_to_cartesian, symbols, conditions, Backend);
    recursive = true
)

@testset "Doctests" begin
    Documenter.doctest(Bramble)
end

end # module QualityDoctestsTests
