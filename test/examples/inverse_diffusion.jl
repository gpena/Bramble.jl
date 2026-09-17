module ExamplesInverseDiffusionTests

using Test
using ..TestUtils: _run_example_page, _have

# The inverse-diffusion page, docs/src/examples/inverse_diffusion.jl, run the same way the
# other worked-example pages are -- but from the "ad"/"full" groups rather than "ext" or the
# every-push path, since it needs `Enzyme` (§4 of the page itself): the gradient descent
# driving the parameter recovery is `Enzyme.gradient` through `pde_solve`'s adjoint rule.
# `Enzyme` is not a `test/Project.toml` dependency -- `Weekly.yml` installs it at CI runtime
# only, the same reasoning `test/space/autodiff_heavy.jl` documents -- so this file skips
# rather than errors when it is absent locally.
#
# Its `#src` assertions pin the recovered diffusion coefficient against the true one (within
# the synthetic observation noise) and that the loss actually decreased from the starting
# guess.


@testset "Inverse diffusion page" begin
    if _have(:Enzyme)
        _run_example_page(:inverse_diffusion)
    else
        @test_skip "Enzyme not in this environment"
    end
end

end # module ExamplesInverseDiffusionTests
