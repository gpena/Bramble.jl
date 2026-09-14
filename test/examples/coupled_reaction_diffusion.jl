module ExamplesCoupledReactionDiffusionTests

using Test
using ..TestUtils: _run_example_page

# The coupled reaction-diffusion page, docs/src/examples/coupled_reaction_diffusion.jl, run
# the same way the every-push pages are (test/examples/pages.jl) -- but moved to the `ext`
# group rather than every push once it grew a `nonlinear_problem`/NonlinearSolve.jl section
# (#119), the same reasoning that keeps poisson_nonlinear and heat_equation out of that path.
#
# Its `#src` assertions pin the tracer-based and native (`ast_sparsity_detector`) Newton
# iteration counts and errors already covered before this page grew that section, plus that
# `nonlinear_problem`'s NewtonRaphson solve agrees with the manual native-Newton loop to
# machine precision -- checked per species, not only combined (bramble-verification §5): a
# routing mistake in the composite Jacobian shows up as one species converging while the
# other silently used the wrong block, which a single combined comparison would hide.

@testset "Coupled reaction-diffusion page" begin
    _run_example_page(:coupled_reaction_diffusion)
end

end # module ExamplesCoupledReactionDiffusionTests
