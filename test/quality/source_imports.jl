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

# `PRECOMPILE_WORKLOAD` (src/precompile.jl) "disables every workload, core and extensions
# alike", so each extension that runs a `@compile_workload` must reference it. The scan is
# on the parsed AST, so a comment or docstring that only mentions either name does not count
# (`ext/BrambleKernelAbstractionsExt.jl` names `@compile_workload` in a comment and has no
# workload, so it needs no exception).
const _EXT_DIR = normpath(joinpath(@__DIR__, "..", "..", "ext"))

_is_workload_macro(name) = name === Symbol("@compile_workload")

# Does the expression hold a call of the macro `@compile_workload`, qualified or not?
function _has_workload_call(ex)
    ex isa QuoteNode && return _has_workload_call(ex.value)
    ex isa Expr || return false
    if ex.head === :macrocall && !isempty(ex.args)
        name = ex.args[1]
        if name isa Expr && name.head === :. && length(name.args) == 2
            name = name.args[2] isa QuoteNode ? name.args[2].value : name.args[2]
        end
        _is_workload_macro(name) && return true
    end
    return any(_has_workload_call, ex.args)
end

# Does the expression mention the symbol `PRECOMPILE_WORKLOAD` anywhere?
function _mentions_gate(ex)
    ex === :PRECOMPILE_WORKLOAD && return true
    ex isa QuoteNode && return _mentions_gate(ex.value)
    ex isa Expr || return false
    return any(_mentions_gate, ex.args)
end

# Returns `(gated, ungated)` for one source text: whether it holds a workload at all, and
# whether that workload is missing the gate.
function _workload_gate_status(text, file)
    parsed = Meta.parseall(text; filename = file)
    has_workload = _has_workload_call(parsed)
    return has_workload, has_workload && !_mentions_gate(parsed)
end

@testset "Ext workloads gate on the flag" begin
    # Positive controls: the scanner must flag an ungated workload and accept a gated one,
    # or the loop below could pass by finding nothing.
    ungated = "using PrecompileTools\n@compile_workload begin\n    f()\nend\n"
    gated = "if Bramble.PRECOMPILE_WORKLOAD\n    @compile_workload begin f() end\nend\n"
    comment_only = "# a `@compile_workload` in a comment\nf() = 1\n"
    @test _workload_gate_status(ungated, "ungated.jl") == (true, true)
    @test _workload_gate_status(gated, "gated.jl") == (true, false)
    @test _workload_gate_status(comment_only, "comment.jl") == (false, false)

    files = sort!(filter(f -> endswith(f, ".jl"), readdir(_EXT_DIR; join = true)))
    @test !isempty(files)

    n_workloads = 0
    for file in files
        has_workload, is_ungated = _workload_gate_status(read(file, String), file)
        n_workloads += has_workload
        @testset "$(basename(file))" begin
            @test !is_ungated
        end
    end
    # An empty scan must not pass: at least one extension has a gated workload.
    @test n_workloads >= 1
end

end # module
