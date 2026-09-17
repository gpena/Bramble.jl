module ExamplesTransientInverseProblemTests

using Test
using ..TestUtils: _run_example_page

# The transient-inverse-problem page, docs/src/examples/transient_inverse_problem.jl, run
# the same way the other worked-example pages are -- but from the "ext"/"full" groups rather
# than the every-push path, since it needs `SciMLSensitivity`: the gradient descent driving
# the parameter recovery is `Bramble.adjoint_sensitivities`, `BrambleSciMLSensitivityExt`'s
# own wrapper. `SciMLSensitivity` is not a `test/Project.toml` dependency -- it is exactly as
# heavy as Enzyme/Mooncake (Zygote, Tracker, ReverseDiff, NNlib all load behind it), so
# `Weekly.yml` installs it at CI runtime only, the same reasoning
# `test/ext/sciml_sensitivity_ext.jl` documents -- so this file skips rather than errors when
# it is absent locally.
#
# Its `#src` assertions pin both recovered parameters (an initial-condition amplitude and a
# boundary-value ramp rate) against their true values, within the synthetic observation
# noise, and that the loss actually decreased from the starting guess.

_have(mod::Symbol) = Base.identify_package(String(mod)) !== nothing

@testset "Transient inverse problem page" begin
    if _have(:SciMLSensitivity)
        _run_example_page(:transient_inverse_problem)
    else
        @test_skip "SciMLSensitivity not in this environment"
    end
end

end # module ExamplesTransientInverseProblemTests
