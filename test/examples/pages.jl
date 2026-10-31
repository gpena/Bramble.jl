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
# step now. What those files covered that the pages do not -- a
# variable-coefficient operator no page uses -- stays in convergence.jl.
#
# More pages run from test/examples/ext_pages.jl, in this same `examples` group, for what
# they load rather than what they assert, and one more behind a differentiation backend
# (test/examples/inverse_diffusion.jl).

@testset "Worked example pages" begin
    @testset "Linear Poisson" begin
        _run_example_page(:poisson_linear)
    end

    @testset "Poisson on a polygon" begin
        _run_example_page(:poisson_polygon)
    end

    # The unit disk as a level set, on the closure path: its `#src` assertions pin the
    # fitted second order in the max and H¹ₕ errors over four uniform grids, the boundary
    # nodes on the circle to 1e-12, and a random-seed draw within a few times the uniform
    # error.
    @testset "Poisson on a disk" begin
        _run_example_page(:poisson_circle)
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
    # into the solution's own storage. Its assertions pin the per-step allocation at zero
    # bytes (a closure reaching `Rₕ!` uninferred costs 96), the error at t = 1 bracketed
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

    # Poisson with Neumann and Robin relations through the ghost layer, on a random
    # non-uniform grid: its assertions pin the order of the error at second, for a mixed
    # Robin and Neumann boundary, a Robin one on every side and a pure Neumann one solved
    # with `mean_zero`, and that the pure Neumann solution has discrete mean zero.
    @testset "Neumann and Robin boundaries" begin
        _run_example_page(:neumann_robin_poisson)
    end

    # The heat equation on the same kind of grid with Neumann and Robin data that change in
    # time, through `semidiscretize` and `FBDF`: its assertions pin the order of the error
    # at the final time at second, bracketed away from the integrator's tolerance, and that
    # the solution has moved from its initial state, so frozen boundary data would fail.
    @testset "Heat equation, Neumann and Robin data" begin
        _run_example_page(:ghost_heat_equation)
    end

    # Elasticity with a traction through ghost points, collocated form: its assertions pin
    # the traction relations on the lifted padded values, as rounding, on the loaded and
    # the traction-free side and at the corner of the two, and the order of the H1 error of
    # both components above 1.8 on three graded grids.
    @testset "Elasticity with a traction" begin
        _run_example_page(:elasticity_traction_2d)
    end

    # The matrix-free Kronecker operator worked example.
    # It needs `using Kronecker`, but that is a lightweight, pure-Julia weak dependency --
    # unlike the stiff DAE solver, `NonlinearSolve` and `AlgebraicMultigrid` that justify
    # holding the pages in test/examples/ext_pages.jl in their own file -- and
    # `LinearSolve`, which this page also uses, is already a test dependency loaded by other
    # pages in this file. It stays here rather than there.
    @testset "Memory scaling: matrix-free Kronecker" begin
        _run_example_page(:memory_scaling)
    end
end

end # module ExamplesPagesTests
