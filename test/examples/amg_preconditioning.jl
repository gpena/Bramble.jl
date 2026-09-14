module ExamplesAmgPreconditioningTests

using Test
using ..TestUtils: _run_example_page

# The AMG preconditioning page, docs/src/examples/amg_preconditioning.jl, run the same way
# the other worked-example pages are (test/examples/pages.jl) -- but from the `ext` group
# rather than every push, since it needs `LinearSolve` and `AlgebraicMultigrid` loaded, a
# cost test/ext/sciml_ext.jl and test/ext/algebraicmultigrid_ext.jl deliberately keep off
# the push path.
#
# Its `#src` assertions pin that LU, unpreconditioned CG and AMG-preconditioned CG all reach
# the same answer, and that the AMG-preconditioned iteration count stays essentially flat
# under refinement where the unpreconditioned one grows.

@testset "AMG preconditioning page" begin
    _run_example_page(:amg_preconditioning)
end

end # module ExamplesAmgPreconditioningTests
