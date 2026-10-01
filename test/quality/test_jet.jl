module QualityTestJetTests

using Test
using JET

# JET's error analysis over the test code itself: test/TestUtils.jl and every test module
# under test/ (gpena/Bramble.jl#390). It catches what only fails when a branch runs: an
# undefined name, a possible `UndefVarError`, a call with the wrong arity.
#
# `BRAMBLE_TEST_JET_FILES` (colon-separated paths) overrides the file list.
#
# How each file is analysed. A test module reaches its helpers as `using ..TestUtils` (and
# a few as `using ..ExtSolverContracts` or `using ..ExtSparseAdExtTests`), so
# `JET.report_file` on the file alone reports every helper as undefined: an artifact of the
# standalone analysis, not a finding. Instead each file is analysed through a small wrapper
# script that `include`s TestUtils.jl, then the module files the target names via `using ..X`,
# then the target, so the parent module JET builds holds the same siblings as `Main` does in
# a real run. Only reports whose top frame lies in the target file are kept: the
# prerequisites are analysed in their own `@test`.

const TEST_DIR = normpath(joinpath(@__DIR__, ".."))
const TESTUTILS = joinpath(TEST_DIR, "TestUtils.jl")

_is_test_module(path) = endswith(path, ".jl") && any(startswith("module "), eachline(path))

function _default_files()
    files = String[TESTUTILS]
    for (root, _, names) in walkdir(TEST_DIR), name in sort(names)

        path = joinpath(root, name)
        path == TESTUTILS || path == abspath(@__FILE__) || !_is_test_module(path) ||
            push!(files, path)
    end
    files
end

const FILES = let s = get(ENV, "BRAMBLE_TEST_JET_FILES", "")
    isempty(s) ? _default_files() : abspath.(filter(!isempty, split(s, ':')))
end

# Module name -> defining file, for the sibling modules a test file reaches as `using ..X`.
const MODULE_FILES = let d = Dict{String, String}()
    for path in _default_files(), line in eachline(path)

        m = match(r"^module\s+(\w+)", line)
        m === nothing || (d[m[1]] = path)
    end
    d
end

function _prerequisites(path)
    pre = String[]
    for line in eachline(path), m in eachmatch(r"(?:using|import)\s+\.\.(\w+)", line)

        name = m[1]
        name == "TestUtils" && continue
        dep = get(MODULE_FILES, name, nothing)
        dep === nothing || dep == path || dep in pre || push!(pre, dep)
    end
    pre
end

function _wrapper(path)
    io = IOBuffer()
    # `Base.include`, not `include`: TestUtils.jl rebinds `Main.include` to its tracing
    # wrapper, which JET does not follow as an include and analyses as a call instead.
    files = path == TESTUTILS ? [path] : [TESTUTILS; _prerequisites(path); path]
    foreach(p -> println(io, "Base.include(@__MODULE__, ", repr(p), ")"), files)
    w = joinpath(mktempdir(), "jet_wrapper.jl")
    write(w, take!(io))
    w
end

_same(f, path) = abspath(String(f)) == abspath(path)

# A report counts against a file when the error itself is raised in that file: the last
# frame of its stack (or, for a toplevel error, its location) lies there. A report whose error
# site is in Base, a stdlib or a package is an imprecision of inference over a call the test
# code makes with concrete arguments, which the call's own tests own, not a bug in the test.
_in_file(r::JET.ToplevelErrorReport, path) = _same(r.file, path)
_in_file(r, path) = !isempty(r.vst) && _same(r.vst[end].file, path)
_line(r::JET.ToplevelErrorReport) = r.line
_line(r) = r.vst[end].line
function _message(r::JET.ToplevelErrorReport)
    hasfield(typeof(r), :err) ? first(split(sprint(showerror, r.err), '\n')) : string(typeof(r))
end
_message(r) = sprint(JET.print_report_message, r)

# Line ranges of every `@test_throws` call in `path`.
function _test_throws_lines(path)
    ranges = UnitRange{Int}[]
    lines(ex) = ex isa LineNumberNode ? [ex.line] :
                ex isa Expr ? reduce(vcat, map(lines, ex.args); init = Int[]) : Int[]
    function walk(ex)
        ex isa Expr || return
        if ex.head === :macrocall && ex.args[1] === Symbol("@test_throws")
            ls = lines(ex)
            isempty(ls) || push!(ranges, minimum(ls):maximum(ls))
        end
        foreach(walk, ex.args)
    end
    walk(Meta.parseall(read(path, String); filename = path))
    ranges
end

# Reports dropped before counting, each for a stated reason. Errors raised inside Test's
# own macro expansions never reach here: their error site is in Test's source (see
# `_in_file`).
function _filtered(r, path, throws_lines)
    # `@test_throws E body` runs `body` expecting it to throw: a call there that JET proves
    # always errors is the point of the test, not a bug.
    any(rg -> _line(r) in rg, throws_lines) && return true
    # JET interprets toplevel code partially: a value it chose not to evaluate (concretize)
    # but a later definition needs is reported as missing. That is a limit of JET's toplevel
    # interpreter, not a property of the test code.
    r isa JET.MissingConcretizationErrorReport && return true
    # Same cause, seen from the other side: a closure written in toplevel `@testset` code
    # whose definition JET did not evaluate is reported as its generated name (`#12#13`,
    # `#f#f##0`) being undefined. A user-written name never starts with `#`.
    r isa JET.UndefVarErrorReport && occursin(r"\.#[^.]*`", _message(r)) && return true
    false
end

# Known findings JET cannot avoid: file (relative to test/) => [(JET message fragment, reason)].
# A package missing from the test environment or gated off, or a limit of JET itself. A
# finding in working test code is fixed in the test, not listed here.
const EXCEPTIONS = Dict{String, Vector{Tuple{String, String}}}(
    "ext/ad_backend_verification.jl" => [
        ("Package ReverseDiff not found",
        "8: `using ReverseDiff` at toplevel, but ReverseDiff is not in test/Project.toml"),
    ],
    "form/autodiff.jl" => [
        ("Package ReverseDiff not found",
        "6: `using ReverseDiff` at toplevel, but ReverseDiff is not in test/Project.toml"),
    ],
    "ext/appleaccelerate_ext.jl" => [
        ("non-boolean (JET.AbstractBindingState)",
        "7: `if !isdefined(Main, :ExtSolverContracts)`: JET cannot evaluate `isdefined` " *
        "on Main from its virtual module; JET limitation, the guard is fine at runtime"),
    ],
    "ext/metal_assembly_replay.jl" => [
        ("Package Metal not found", "6: Metal is out of the test env since 2026-09-27"),
        ("`Metal` not defined", "23: follows from the missing Metal package")
    ],
    "ext/metal_ext.jl" => [
        ("Package Metal not found", "6: Metal is out of the test env since 2026-09-27"),
        ("Package GPUArrays not found", "9: GPUArrays is not in test/Project.toml"),
        ("`Metal` not defined", "258: follows from the missing Metal package")
    ],
    "ext/metal_form_assembly.jl" => [
        ("Package Metal not found", "6: Metal is out of the test env since 2026-09-27"),
        ("`Metal` not defined", "22: follows from the missing Metal package")
    ],
    "ext/metal_fullstack.jl" => [
        ("Package Metal not found", "8: Metal is out of the test env since 2026-09-27"),
        ("`Metal` not defined", "27: follows from the missing Metal package")
    ],
    "space/autodiff_heavy.jl" => [
        ("Enzyme` is not defined",
        "20 (reported at the testset): `Enzyme` is bound by `@eval import Enzyme` at " *
        "line 47 only when `_have(:Enzyme)`"),
    ],
    "utils/macros.jl" => [
        ("Syntax: @forward T.x f",
        "50: line 152 deliberately `@eval`s a malformed `@forward` inside a " *
        "`try`/`catch` to test its error; JET expands it eagerly"),
    ]
)

function _excepted(r, path)
    rel = relpath(path, TEST_DIR)
    msg = _message(r)
    any(e -> occursin(e[1], msg), get(EXCEPTIONS, rel, Tuple{String, String}[]))
end

@testset "JET on test code" begin
    elapsed = @elapsed for path in FILES
        # Toplevel analysis, not `analyze_from_definitions`: it follows the `@testset`
        # bodies with the arguments they actually pass, where analysing every method from
        # its declared signature buries the real reports under inference noise.
        reports = try
            JET.get_reports(JET.report_file(_wrapper(path); toplevel_logger = nothing))
        catch e
            @error "JET could not analyse $path" exception = e
            @test false
            continue
        end
        throws_lines = _test_throws_lines(path)
        mine = filter(r -> _in_file(r, path) && !_filtered(r, path, throws_lines), reports)
        bad = filter(r -> !_excepted(r, path), mine)
        for r in bad
            println("JET ", relpath(path, TEST_DIR), ":", _line(r), ": ", _message(r))
        end
        @test isempty(bad)
    end
    println("JET on test code: ", length(FILES), " files in ", round(elapsed; digits = 1), " s")
end

end # module QualityTestJetTests
