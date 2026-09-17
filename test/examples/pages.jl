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
end

end # module ExamplesPagesTests
