module QualityTestsetNamesTests

using Test

# Every `@testset` name fits in 40 display columns, writes an issue tag as `(#N)`, and differs
# from its siblings' literal names, so the test report stays readable and unambiguous.
#
# Deliberately Test-only: no `using Bramble`, so this file runs standalone. Parses each file
# with `Meta.parseall` rather than a regex, which would misread escaped quotes, options such
# as `verbose = true`, and interpolation. Width is `textwidth` (display columns, so `ₕ` is
# one), with each interpolation (`$x`, `$(expr)`) counted as 4 columns: it renders as a short
# value (`2D`, `Float32`), not as its source text.

const _TEST_DIR = normpath(joinpath(@__DIR__, ".."))
const _LIMIT = 40
const _INTERP_WIDTH = 4

_is_testset(ex) = ex isa Expr && ex.head === :macrocall &&
                  ex.args[1] in (Symbol("@testset"), GlobalRef(Main, Symbol("@testset")))

# The first positional string argument; options (`verbose = true`) are skipped and a
# non-literal first argument (a custom testset type, a variable) means no measurable name.
function _first_name(ex)
    for a in ex.args[2:end]
        a isa LineNumberNode && continue
        a isa Expr && a.head === :(=) && continue
        (a isa String || (a isa Expr && a.head === :string)) && return a
        return nothing
    end
    return nothing
end

_name_width(arg::String) = textwidth(arg)
_name_width(arg::Expr) = sum(a -> a isa String ? textwidth(a) : _INTERP_WIDTH, arg.args;
                             init = 0)

_name_text(arg::String) = arg
_name_text(arg::Expr) = join((a isa String ? a : "\$(…)" for a in arg.args))

function _duplicates!(out, path, siblings)
    seen = Set{String}()
    for s in siblings
        s in seen ? push!(out, "$path: duplicate sibling testset \"$s\"") : push!(seen, s)
    end
    return out
end

# `siblings` collects the literal names of testsets directly under the current testset (or
# the file), so duplicates are judged per parent.
function _walk!(out, path, ex, siblings, line)
    ex isa Expr || return line
    if _is_testset(ex)
        for a in ex.args
            a isa LineNumberNode && (line = a.line)
        end
        arg = _first_name(ex)
        if arg !== nothing
            w = _name_width(arg)
            txt = _name_text(arg)
            w > _LIMIT && push!(out, "$path:$line: testset name is $w columns: \"$txt\"")
            occursin("gpena/Bramble.jl#", txt) &&
                push!(out, "$path:$line: write the issue tag as (#N): \"$txt\"")
            arg isa String && push!(siblings, arg)
        end
        kids = String[]
        l = line
        for a in ex.args
            a isa LineNumberNode && (l = a.line; continue)
            l = _walk!(out, path, a, kids, l)
        end
        _duplicates!(out, path, kids)
        return line
    end
    l = line
    for a in ex.args
        a isa LineNumberNode && (l = a.line; continue)
        l = _walk!(out, path, a, siblings, l)
    end
    return l
end

"""
    testset_name_violations(path) -> Vector{String}

One message per `@testset` name in the file at `path` that is over 40 display columns,
carries a `gpena/Bramble.jl#` tag, or repeats a sibling's literal name.
"""
function testset_name_violations(path)
    ast = Meta.parseall(read(path, String); filename = path)
    out = String[]
    top = String[]
    _walk!(out, path, ast, top, 0)
    _duplicates!(out, path, top)
    return out
end

function _test_files()
    files = String[]
    for (root, _, names) in walkdir(_TEST_DIR)
        for name in names
            endswith(name, ".jl") && push!(files, normpath(joinpath(root, name)))
        end
    end
    return files
end

@testset "Testset names" begin
    files = _test_files()
    @test !isempty(files)
    for file in files
        bad = testset_name_violations(file)
        isempty(bad) || foreach(println, bad)
        @test isempty(bad)
    end
end

end # module
