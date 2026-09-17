module ExamplesExtPagesTests

using Test
using ..TestUtils: _run_example_page

# The four worked-example pages that need more than the every-push test environment, run
# the same way the others are (test/examples/pages.jl) but from the `ext` group. Each is a
# Literate script under docs/src/examples/ whose `#src` assertions execute here, so the
# numbers a reader sees on the page are the numbers checked.
#
# They sit in this group rather than on the push path because of what they load, not
# because of what they assert: `OrdinaryDiffEqBDF` for the differential-algebraic step,
# `NonlinearSolve` for the two pages that grew a `nonlinear_problem` section, and
# `LinearSolve` with `AlgebraicMultigrid` for the preconditioning comparison. This file is
# included last in that group, after the ext tests that already pay those costs.
#
# The two pages behind the `ad`/`ext` groups for a *backend* rather than a solver --
# inverse_diffusion (Enzyme) and transient_inverse_problem (SciMLSensitivity) -- keep their
# own files: each guards on `_have` and skips rather than runs when the backend is absent.

@testset "Worked example pages (extensions)" begin
    # Steps a differential-algebraic system with `FBDF`. Pins the second-order rate the
    # page prints, the two zeroed mass-matrix rows against the interior weight, the error
    # at `t = 1`, and the driven boundary values.
    @testset "Heat equation page" begin
        _run_example_page(:heat_equation)
    end

    # Pins the Picard and manual-Newton iteration counts and errors, that
    # `nonlinear_problem`'s NewtonRaphson solve agrees with the manual Newton loop to
    # machine precision, and that the Picard-vs-NonlinearSolve timing/allocation ratio the
    # page prints is finite and positive -- not a specific value, since a single-run
    # wall-clock ratio is not something to pin a regression threshold to
    # (bramble-verification).
    @testset "Nonlinear Poisson page" begin
        _run_example_page(:poisson_nonlinear)
    end

    # Pins the tracer-based and native (`ast_sparsity_detector`) Newton iteration counts and
    # errors, plus that `nonlinear_problem`'s NewtonRaphson solve agrees with the manual
    # native-Newton loop to machine precision -- checked per species, not only combined
    # (bramble-verification §5): a routing mistake in the composite Jacobian shows up as one
    # species converging while the other silently used the wrong block, which a single
    # combined comparison would hide.
    @testset "Coupled reaction-diffusion page" begin
        _run_example_page(:coupled_reaction_diffusion)
    end

    # Pins that LU, unpreconditioned CG and AMG-preconditioned CG all reach the same answer,
    # and that the AMG-preconditioned iteration count stays essentially flat under
    # refinement where the unpreconditioned one grows.
    @testset "AMG preconditioning page" begin
        _run_example_page(:amg_preconditioning)
    end
end

end # module ExamplesExtPagesTests
