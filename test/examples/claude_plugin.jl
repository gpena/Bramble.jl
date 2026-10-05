module ExamplesClaudePluginTests

using Test

# The worked programs bundled with the Claude Code plugin (plugins/bramble/skills/bramble/
# examples/). An assistant copies them as starting points, so each ends in `@assert`s against
# an independent value (a manufactured solution, a convergence order, a conservation law);
# running a file is the test. Each runs in its own module, as a user's script would.
# The names they and the skill's Markdown use are checked in test/quality/claude_plugin.jl.

const _EXAMPLES = normpath(joinpath(@__DIR__, "..", "..", "plugins", "bramble", "skills",
    "bramble", "examples"))

@testset "Claude Code plugin examples" begin
    files = sort!(filter(endswith(".jl"), readdir(_EXAMPLES)))
    @test !isempty(files)
    for file in files
        @testset "$file" begin
            m = Module(Symbol(:PluginExample_, first(splitext(file))))
            Core.eval(m, :(include(x) = Base.include($m, x)))
            @test (Base.include(m, joinpath(_EXAMPLES, file)); true)
        end
    end
end

end # module
