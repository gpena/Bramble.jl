using Test

# The heat-equation page is a Literate script, `docs/src/examples/heat_equation.jl`:
# docs/make.jl renders it to markdown and this file runs it, so the page a reader sees and
# the code the suite executes are one file. The script's `#src` lines are the assertions --
# stripped on the way to the page, executed from here -- which is what closes the gap
# convergence.jl describes at its top: a Documenter `@example` block renders the value of
# `all(>(1.95), orders)`, it does not assert it, so a rate that decayed to first order would
# publish `false` and fail nothing (gpena/Bramble.jl#117).
#
# Runs in the `ext` group rather than on every push because the script solves with `FBDF`,
# and loading `OrdinaryDiffEqBDF` is the cost sciml_ext.jl keeps out of the push path.
#
# Included into a throwaway module rather than into `Main`: the script writes the bare
# `domain`/`mesh`/`element` a reader would type, and every ext file shares one `Main` with
# Meshes.jl, which exports those same three names.
const _heat_example_script = joinpath(
    @__DIR__, "..", "..", "docs", "src", "examples", "heat_equation.jl"
)

@testset "Heat equation example (docs/src/examples/heat_equation.jl)" begin
    @test isfile(_heat_example_script)

    # `@eval module` rather than `Module(...)`: a module built by hand has no `include` of
    # its own, and the script includes the shared plotting helper the same way the rendered
    # page does.
    @eval module HeatEquationExample
    using Test
    include($_heat_example_script)
    end
end
