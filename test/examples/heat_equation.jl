using Test

# The heat-equation page, docs/src/examples/heat_equation.jl, run the same way the other four
# are (test/examples/pages.jl) -- but from the `ext` group rather than every push: the script
# steps a differential-algebraic system with `FBDF`, and loading `OrdinaryDiffEqBDF` is the
# cost test/ext/sciml_ext.jl deliberately keeps off the push path.
#
# Its `#src` assertions pin the second-order rate the page prints, the two zeroed mass-matrix
# rows against the interior weight, the error at `t = 1`, and the driven boundary values.

@testset "Heat equation page" begin
    _run_example_page(:heat_equation)
end
