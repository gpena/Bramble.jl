module ExamplesPagesTests

using Test
using ..TestUtils: _run_example_page

# The three worked-example pages that need nothing beyond the every-push test environment.
# Each is a Literate script under docs/src/examples/, run here through `_run_example_page`
# (test/runtests.jl), which is where its `#src` assertions execute: the convergence rates the
# pages print, the iteration counts Picard and Newton reach, and the errors against each
# manufactured solution.
#
# This replaces test/examples/convergence.jl's "Linear Poisson" and "Convection-diffusion"
# testsets and the whole of nonlinear_convergence.jl, which mirrored these pages line by line
# so that the rendered numbers were checked somewhere. There is no second copy to keep in
# step now (gpena/Bramble.jl#117). What those files covered that the pages do not -- a
# variable-coefficient operator no page uses -- stays in convergence.jl.
#
# Four more pages run in the `ext` group instead (test/examples/ext_pages.jl), for what they
# load rather than what they assert, and two more behind a differentiation backend
# (test/examples/inverse_diffusion.jl, test/examples/transient_inverse_problem.jl).

@testset "Worked example pages" begin
    @testset "Linear Poisson" begin
        _run_example_page(:poisson_linear)
    end

    @testset "Convection-diffusion" begin
        _run_example_page(:convection_diffusion_linear)
    end

    @testset "3D linear elasticity" begin
        _run_example_page(:elasticity_3d)
    end

    # Point sources and the flux the Dirichlet condition supplies: its `#src` assertions
    # pin the solution against the rectangle's Green's function at three points away from
    # the singularity, and the two conservation statements that hold on any mesh rather
    # than in the limit -- one well's boundary flux equals its strength, and an
    # injector/producer pair's net flux is zero.
    @testset "Point sources and boundary flux" begin
        _run_example_page(:point_sources_flux)
    end

    # The in-place Crank-Nicolson loop: pattern reuse, one factorization, and a backsolve
    # into the solution's own storage. Its assertions pin the per-step allocation under a
    # bound loose enough to survive a compiler change but tight enough to catch a closure
    # built inside the loop (a few hundred bytes per step), the error at t = 1 bracketed
    # away from zero, and the order in time.
    @testset "Transient loop, in place" begin
        _run_example_page(:transient_inplace)
    end

    # Uniform against graded points on a convection-diffusion layer at ε = 1e-3. Its
    # assertions pin the three-orders-of-magnitude gap at equal degrees of freedom, that
    # the graded mesh at 41 points beats the uniform one at 641, and that the best grading
    # strength is an interior point of the sweep rather than its weakest end.
    @testset "Graded mesh for a boundary layer" begin
        _run_example_page(:boundary_layer_graded)
    end
end

end # module ExamplesPagesTests
