module QualitySourceImportsTests

using Test

# gpena/Bramble.jl#337: every `using`/`import` statement lives at the top of `src/Bramble.jl`
# so a sub-file relies on the module's namespace instead of importing packages on its own.
# Three had drifted (`src/mesh/mesh1d.jl`, `src/solvers/accelerate_solver.jl`,
# `src/utils/backend.jl`); this is the ratchet that keeps a fourth from doing the same.
#
# Deliberately Test-only: no `using Bramble`, so this file runs standalone (the CHECK
# includes it directly) and never needs the package to load to catch a scattered import.
#
# Parses each file with `Meta.parseall` rather than grepping text, so a docstring or comment
# that merely mentions `using Bramble` (as plenty do, as an example) is not a false positive
# -- it is a `String`/`Expr(:string, ...)` node in the parsed AST, never an
# `Expr(:using, ...)`/`Expr(:import, ...)` one.

const _SRC_DIR = normpath(joinpath(@__DIR__, "..", "..", "src"))
const _BRAMBLE_JL = normpath(joinpath(_SRC_DIR, "Bramble.jl"))

function _src_files()
    files = String[]
    for (root, _, names) in walkdir(_SRC_DIR)
        for name in names
            endswith(name, ".jl") || continue
            push!(files, normpath(joinpath(root, name)))
        end
    end
    return files
end

# Recursively walks a parsed expression, looking for a `using`/`import` node at any depth
# (inside a function body, an `if`, a macro call, ...), not just at the top level.
function _find_import(ex)
    ex isa Expr || return nothing
    if ex.head === :using || ex.head === :import
        return ex
    end
    for arg in ex.args
        found = _find_import(arg)
        found === nothing || return found
    end
    return nothing
end

@testset "No scattered using/import in src/" begin
    files = _src_files()
    @test !isempty(files)

    for file in files
        file == _BRAMBLE_JL && continue

        text = read(file, String)
        parsed = Meta.parseall(text; filename = file)
        found = _find_import(parsed)

        rel = relpath(file, _SRC_DIR)
        @testset "$rel" begin
            @test found === nothing
        end
    end
end

end # module
