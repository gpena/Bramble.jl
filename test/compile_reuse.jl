# Compile-reuse measurement for the test suite (#888, docs/notes/test-closure-reuse.md).
#
# `CompileReuse.measure(files)` includes test files cold into `Main` under
# SnoopCompileCore's `@snoop_inference` and counts the Bramble method instances inferred.
# Run as a script it measures one rewrite group and writes its rows to
# docs/notes/test-closure-reuse.tsv:
#
#     julia --startup-file=no --project=test test/compile_reuse.jl S3.1
#
# Included as a file it only defines the module `CompileReuse`.

module CompileReuse

using SnoopCompileCore
using SnoopCompile: SnoopCompile, flatten, exclusive
using Test
import Bramble

const ROOT = dirname(@__DIR__)
const TSV = joinpath(ROOT, "docs", "notes", "test-closure-reuse.tsv")
const COLUMNS = ("id", "kind", "files", "mis", "mis_own", "mis_coef", "mis_coef_shared",
    "passes", "broken", "test_group", "t_infer", "shared_ratio", "commit", "julia")

# The rewrite groups: the OWNS lists of S3.1-S3.5 in .claude/plans/test-compile-reuse.md.
const GROUPS = Dict(
    "S3.1" => ["test/form/linear.jl", "test/form/source_operators.jl",
        "test/form/rank_one.jl", "test/form/linear_restriction.jl"],
    "S3.2" => ["test/form/semidiscrete.jl", "test/form/normal.jl", "test/form/bilinear.jl",
        "test/form/dirichlet_constraints.jl", "test/form/inner_products.jl",
        "test/form/ghost_rank_one.jl"],
    "S3.3" => ["test/space/inner_product.jl", "test/space/vector_elements.jl",
        "test/space/operators.jl", "test/space/star_difference.jl",
        "test/space/interpolation_bounds.jl", "test/space/sbp_identities.jl"],
    "S3.4" => ["test/mesh/meshes.jl", "test/mesh/markers.jl", "test/mesh/meshnd.jl",
        "test/mesh/levelset_class.jl"],
    "S3.5" => ["test/ghost/cross_component.jl", "test/ghost/polygon_spaces.jl",
        "test/ghost/composite.jl"])

# Files that import a test module another file defines (`using ..SpaceDifferenceTests`),
# so `measure` includes the prerequisite first: inside the measurement when it is one of
# the files asked for, otherwise uncounted, as the suite's earlier files would have run it.
const PREREQUISITES = Dict(
    "test/space/inner_product.jl" => ["test/space/vector_elements.jl"],
    "test/space/star_difference.jl" => ["test/space/difference.jl"])

# Closure bodies S2 turned into TestUtils functions, with the argument named `x`.
const SHARED_BODIES = Set([
    "x[1]", "x[2]", "x[3]", "1 + x[1]", "x[1] < 0.5", "x[1] + x[2]",
    "sin(x[1]) * x[2]", "x[1] * x[2]", "x", "exp(x[1])", "x[1]^2", "sin(x[1])"])
const CONSTANT_BODY = r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$"

abspath_of(f) = isabspath(f) ? f : joinpath(ROOT, f)

# Modules (nested ones too) that `Main` holds. Callers after an include use `invokelatest`:
# a binding the include created is invisible to the caller's world age.
function _modules_of(m::Module, acc = Dict{Module, Nothing}())
    for n in names(m; all = true)
        isdefined(m, n) || continue
        v = getfield(m, n)
        (v isa Module && v !== m && v !== Base && parentmodule(v) === m && !haskey(acc, v)) ||
            continue
        acc[v] = nothing
        _modules_of(v, acc)
    end
    return acc
end

# First type in `T` (depth first, at any depth of type parameters) defined in `mods`.
function _owned(T, mods, depth = 0)
    depth > 16 && return nothing
    if T isa Union
        r = _owned(T.a, mods, depth + 1)
        return r === nothing ? _owned(T.b, mods, depth + 1) : r
    elseif T isa UnionAll
        return _owned(Base.unwrap_unionall(T), mods, depth + 1)
    elseif T isa TypeVar
        return _owned(T.ub, mods, depth + 1)
    elseif T isa DataType
        T.name.module in mods && return T
        for p in T.parameters
            q = p isa Union{Type, TypeVar} ? p : (p isa Function ? typeof(p) : nothing)
            q === nothing && continue
            r = _owned(q, mods, depth + 1)
            r === nothing || return r
        end
    end
    return nothing
end

# `(arg, body)` of every `x -> body` / `p -> body` on a source line; the body ends at the
# first comma or semicolon outside brackets, at a comment, or at a bracket closing one
# opened before it.
function _closure_bodies(line::AbstractString)
    out = Tuple{String, String}[]
    for m in eachmatch(r"(?<![\w.])([xp])\s*->\s*", line)
        s = m.offset + ncodeunits(m.match)
        depth = 0
        e = lastindex(line)
        for i in s:lastindex(line)
            i in eachindex(line) || continue
            c = line[i]
            if c in ('(', '[', '{')
                depth += 1
            elseif c in (')', ']', '}')
                depth == 0 && (e = prevind(line, i); break)
                depth -= 1
            elseif c == '#' || ((c == ',' || c == ';') && depth == 0)
                e = prevind(line, i)
                break
            end
        end
        body = strip(line[s:e])
        push!(out, (m.captures[1], String(body)))
    end
    return out
end

function _matches_shared(arg, body)
    arg == "x" || (body = replace(body, Regex("(?<![\\w.])" * arg * "(?![\\w])") => "x"))
    return body in SHARED_BODIES || occursin(CONSTANT_BODY, body)
end

const _LINES = Dict{String, Vector{String}}()
function _source_line(file, line)
    file = String(file)
    lines = get!(_LINES, file) do
        isfile(file) ? readlines(file) : String[]
    end
    return 1 <= line <= length(lines) ? lines[line] : ""
end

# The one-argument closure, argument `x` or `p`, a type is, as `(method, line text)`.
function _coef_closure(T)
    startswith(String(nameof(T)), "#") || return nothing
    ms = Base._methods_by_ftype(Tuple{T, Vararg{Any}}, -1, Base.get_world_counter())
    (ms === nothing || isempty(ms)) && return nothing
    m = ms[1].method
    m.nargs == 2 || return nothing
    Base.method_argnames(m)[2] in (:x, :p) || return nothing
    return m
end

function _count(tinf, mods)
    mis = mis_own = mis_coef = mis_coef_shared = 0
    t_infer = 0.0
    cache = Dict{Any, Union{Nothing, Bool}}()  # owned type => nothing | shared?
    for f in flatten(tinf)
        mi = SnoopCompile.MethodInstance(f)
        mi.def isa Method || continue
        mi.def.module === Bramble || continue
        mis += 1
        t_infer += exclusive(f)
        T = _owned(mi.specTypes, mods)
        T === nothing && continue
        mis_own += 1
        shared = get!(cache, T) do
            m = _coef_closure(T)
            m === nothing && return nothing
            bodies = _closure_bodies(_source_line(m.file, Int(m.line)))
            any(b -> _matches_shared(b...), bodies)
        end
        shared === nothing && continue
        mis_coef += 1
        shared && (mis_coef_shared += 1)
    end
    return (; mis, mis_own, mis_coef, mis_coef_shared, t_infer)
end

function _load_testutils()
    isdefined(Main, :TestUtils) || Base.include(Main, joinpath(@__DIR__, "TestUtils.jl"))
    return nothing
end

# One file, included cold into `Main` inside `@snoop_inference` and a `@testset`.
function _measure_file(file)
    path = abspath_of(file)
    isfile(path) || error("compile_reuse: no such file $file")
    before = _modules_of(Main)
    outer = Test.DefaultTestSet("compile_reuse")
    tinf = Test.@with_testset outer begin
        @snoop_inference @testset "$file" begin
            Base.include(Main, path)
        end
    end
    loaded = Set(Base.loaded_modules_array())
    mine = Set(m for m in keys(Base.invokelatest(_modules_of, Main)) if !haskey(before, m) && !(m in loaded))
    c = Test.get_test_counts(outer)
    return merge(_count(tinf, mine),
        (; passes = c.cumulative_passes,
            broken = c.cumulative_broken, fails = c.cumulative_fails + c.cumulative_errors))
end

"""
    measure(files) -> NamedTuple

Include each of `files` (paths from the repository root, or absolute) cold into `Main`,
one `@snoop_inference` and `@testset` each, and count the Bramble method instances
inferred: `mis` (all), `mis_own` (specTypes mention a type a file's own modules define),
`mis_coef` (own, keyed on a one-argument closure whose argument is `x` or `p`) and
`mis_coef_shared` (of those, closures whose source line holds a body S2 turned into a
TestUtils function). Also `passes`, `broken`, `fails`, `t_infer` (exclusive inference
seconds of the Bramble frames, never gated), `test_group` and `perfile`. A file that imports
another test module (`PREREQUISITES`) gets that file included first, uncounted when it is
not in `files`.
"""
function measure(files)
    _load_testutils()
    wanted = String.(files)
    done = Set{String}()
    per = []
    function visit(f)
        f in done && return
        push!(done, f)
        foreach(visit, get(PREREQUISITES, f, String[]))
        if f in wanted
            push!(per, (; file = f, _measure_file(f)...))
        else
            _measure_file(f)  # prerequisite outside `files`: run, not counted
        end
    end
    foreach(visit, wanted)
    tot(k) = sum(getfield(p, k) for p in per)
    group = Base.invokelatest(() -> Main.TestUtils.TEST_GROUP)
    return (; files = wanted, mis = tot(:mis), mis_own = tot(:mis_own),
        mis_coef = tot(:mis_coef), mis_coef_shared = tot(:mis_coef_shared),
        passes = tot(:passes), broken = tot(:broken), fails = tot(:fails),
        t_infer = tot(:t_infer), test_group = group, perfile = per)
end

function _read_rows()
    isfile(TSV) || return Vector{Dict{String, String}}()
    lines = filter(!isempty, readlines(TSV))
    isempty(lines) && return Vector{Dict{String, String}}()
    cols = split(lines[1], '\t')
    return [Dict(zip(cols, split(l, '\t'))) for l in lines[2:end]]
end

"""
    baseline(group) -> NamedTuple

The recorded row of `group` (`"S3.1"` to `"S3.5"`) in docs/notes/test-closure-reuse.tsv,
with `files`, `mis`, `mis_own`, `mis_coef`, `mis_coef_shared`, `passes`, `broken`,
`test_group`, `t_infer`, `shared_ratio`, `commit` and `julia`.
"""
function baseline(group)
    rows = filter(r -> r["id"] == group && r["kind"] == "group", _read_rows())
    isempty(rows) && error("compile_reuse: no baseline row for $group in $TSV")
    r = only(rows)
    i(k) = parse(Int, r[k])
    return (; id = r["id"], files = String.(split(r["files"], ';')), mis = i("mis"),
        mis_own = i("mis_own"), mis_coef = i("mis_coef"),
        mis_coef_shared = i("mis_coef_shared"), passes = i("passes"), broken = i("broken"),
        test_group = r["test_group"], t_infer = parse(Float64, r["t_infer"]),
        shared_ratio = parse(Float64, r["shared_ratio"]), commit = r["commit"],
        julia = r["julia"])
end

function _row(id, kind, files, r, commit, group)
    Dict(
        "id" => id, "kind" => kind, "files" => join(files, ';'), "mis" => string(r.mis),
        "mis_own" => string(r.mis_own), "mis_coef" => string(r.mis_coef),
        "mis_coef_shared" => string(r.mis_coef_shared), "passes" => string(r.passes),
        "broken" => string(r.broken), "test_group" => group,
        "t_infer" => string(round(r.t_infer; digits = 1)),
        "shared_ratio" => string(round(r.mis_coef == 0 ? 0.0 : r.mis_coef_shared / r.mis_coef;
            digits = 3)),
        "commit" => commit, "julia" => string(VERSION))
end

# Measure `group` in this (cold) process and replace its rows in the TSV.
function record(group)
    haskey(GROUPS, group) || error("compile_reuse: unknown group $group (S3.1 to S3.5)")
    r = measure(GROUPS[group])
    r.fails == 0 || error("compile_reuse: $(r.fails) failing or erroring tests in $group")
    commit = try
        readchomp(`git -C $ROOT rev-parse --short HEAD`)
    catch
        "unknown"
    end
    new = [_row(group, "group", r.files, r, commit, r.test_group)]
    for p in r.perfile
        push!(new, _row(group, "file", [p.file], p, commit, r.test_group))
    end
    keep = filter(x -> x["id"] != group, _read_rows())
    rows = vcat(keep, new)
    sort!(rows; by = x -> (x["id"], x["kind"] == "group" ? 0 : 1, x["files"]))
    mkpath(dirname(TSV))
    open(TSV, "w") do io
        println(io, join(COLUMNS, '\t'))
        for x in rows
            println(io, join((x[c] for c in COLUMNS), '\t'))
        end
    end
    for x in new
        println(join((x[c] for c in COLUMNS), '\t'))
    end
    return r
end

end # module CompileReuse

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia --project=test test/compile_reuse.jl S3.1")
    CompileReuse.record(ARGS[1])
end
