module ExamplesPoissonNonlinearTests

using Test
using ..TestUtils: _run_example_page

# The nonlinear Poisson page, docs/src/examples/poisson_nonlinear.jl, run the same way the
# every-push pages are (test/examples/pages.jl) -- but from the `ext` group rather than every
# push: its `nonlinear_problem` section solves through NonlinearSolve.jl's `NewtonRaphson`,
# and loading `NonlinearSolve` is a cost the push path does not otherwise pay (the same
# reasoning that keeps heat_equation, which needs `OrdinaryDiffEqBDF`, out of that path too).
#
# Its `#src` assertions pin the Picard and manual-Newton iteration counts and errors already
# covered before this page grew a third method, plus that `nonlinear_problem`'s NewtonRaphson
# solve agrees with the manual Newton loop to machine precision and the Picard-vs-NonlinearSolve
# timing/allocation ratio it prints is finite and positive -- not a specific value, since a
# single-run wall-clock ratio is not something to pin a regression threshold to
# (bramble-verification).

@testset "Nonlinear Poisson page" begin
    _run_example_page(:poisson_nonlinear)
end

end # module ExamplesPoissonNonlinearTests
