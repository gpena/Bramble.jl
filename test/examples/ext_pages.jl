module ExamplesExtPagesTests

using Test
using ..TestUtils: _run_example_page

# The worked-example pages that need more than the every-push test environment, run
# the same way the others are (test/examples/pages.jl) and in the same `examples` group.
# Each is a Literate script under docs/src/examples/ whose `#src` assertions execute here,
# so the numbers a reader sees on the page are the numbers checked.
#
# They sit in this file rather than in pages.jl because of what they load, not
# because of what they assert: `OrdinaryDiffEqBDF` for the differential-algebraic step,
# `NonlinearSolve` for the two pages that grew a `nonlinear_problem` section, and
# `LinearSolve` with `AlgebraicMultigrid` for the preconditioning comparison. This file is
# included after pages.jl in the `examples` group.
#
# The two pages behind the `ad` group for a *backend* rather than a solver --
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

    # The second-order (wave) counterpart of the heat-equation page: it steps a
    # `SecondOrderODEProblem` with `Rodas5P`, so it needs `OrdinaryDiffEqRosenbrock` rather
    # than the BDF stepper above. Its assertions pin the error at `t = 1` against the exact
    # standing wave, bracketed away from zero so a solution reproduced by construction would
    # fail; the relative energy drift the page reports; and the second-order spatial rate
    # over three meshes.
    @testset "2D wave equation page" begin
        _run_example_page(:wave_equation_2d)
    end

    # Loads `LinearSolve` for CG. Pins that `kronecker_operator` refuses its non-separable
    # form; that the serial and `CpuThreaded()` matrix-free products match `assemble(a) * x`
    # on a graded non-uniform 2D mesh with Dirichlet rows; that CG through `KrylovJL_CG` with
    # a matrix-free preconditioner converges to the assembled direct solution in fewer
    # iterations than unpreconditioned CG; and that a `GpuKernel()` policy throws
    # `ArgumentError`.
    @testset "Matrix-free operator page" begin
        _run_example_page(:matrix_free_operator)
    end
end

end # module ExamplesExtPagesTests
