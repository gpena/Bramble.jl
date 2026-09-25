#!/usr/bin/env julia
#
# v3.12 operator rename (gpena/Bramble.jl#349).
#
# Swaps the starred forward family and the cross-weighted family's names: the starred family
# (built on the combining mark U+033D, `D̽`) becomes the tilde family (U+0303, `D̃`), and the
# cross-weighted family (plain `Dₕ`, `Dₕₓ`, `Dₕᵧ`, `Dₕ₂`) takes the freed `D̽` names. Run from
# the repository root:
#
#     julia --project=. dev/rename_v3_12.jl [--check] [paths...]
#
# `--check` reports what would change without writing anything. Given no paths, it walks the
# default roots below.
#
# Bramble users can point it at their own code: the mapping is the whole of the rename, and
# nothing in the script is specific to this repository except the default roots.
#
# ## Why one `replace` call rather than a pass per step
#
# `D̽ₓ` (old starred) and `Dₕₓ` (old cross-weighted) both need to move in the same run --
# the old starred family onto `D̃ₓ`, the old cross-weighted family onto `D̽ₓ` -- and a
# pass-per-step application would have the second pass rescan names the first pass just
# wrote (`Dₕₓ -> D̽ₓ` followed by `D̽ₓ -> D̃ₓ` would send the cross-weighted family all the way
# to the tilde name). `RENAME_REGEX`'s alternation walks the text once left to right, taking
# the first matching pattern at each position and never re-examining replacement text, so both
# families land in one pass without either clobbering the other. It is a hand-written loop
# rather than a plain `Base.replace(s, pairs...)` call (as `dev/rename_v3.jl` uses) because
# each match also needs the identifier-boundary check below, which a bare `replace` pass
# cannot express.
#
# This is a silent semantic swap, not a plain rename: code left unrenamed keeps compiling and
# silently starts computing the other family's values. There is no `D̽`-based marker left
# untouched by the rename that a preflight could use to detect a prior run, the way
# `dev/rename_v3.jl` uses `D̽` itself -- a second run over already-renamed code would rename
# the tilde family a second time onto nothing useful and re-route `Dₕ`-shaped names again, so
# only run this once per tree.
#
# ## Identifier boundaries
#
# Unlike `dev/rename_v3.jl`'s four names, `D̽` and `Dₕ` are short enough to occur as a prefix
# of an unrelated user identifier (`xDₕ`, `LDₕ`). A match only renames when it is not preceded
# by an identifier character (letter, digit, `_`, `!`, or a combining mark -- `Base.is_id_char`
# covers all of these) and not followed by one either, except a trailing `!`, which belongs to
# the same bang-form identifier rather than starting a new one. A following subscript that
# extends the match into a longer pattern (`Dₕ` immediately before `ₓ`) never reaches this
# check: the alternation below lists the longer, more specific patterns first, so the regex
# already prefers them at that position. The LaTeX rule is not an identifier and is exempt
# (`NO_BOUNDARY_CHECK` below).

const RENAMES = [
    # Longest/most specific key first: the alternation tries each pattern in order at a given
    # position and uses the first one that matches, and the bare stems (`D̽`, `Dₕ`) are
    # prefixes of their own subscripted and vector-calculus forms, so those must be listed
    # after everything built on them.
    "D̽ₓ" => "D̃ₓ",         # starred forward, x component
    "D̽ᵧ" => "D̃ᵧ",         # starred forward, y component
    "D̽₂" => "D̃₂",         # starred forward, z/second component
    "∇̽ₕ" => "∇̃ₕ",         # starred forward gradient
    "div̽ₕ" => "diṽₕ",     # starred forward divergence
    "curl̽ₕ" => "curl̃ₕ",   # starred forward curl
    "D̽ₕ" => "D̃ₕ",         # starred forward, mesh-wide alias
    "D̽" => "D̃",           # starred forward, bare stem
    "Dₕₓ" => "D̽ₓ",        # cross-weighted centered, x component
    "Dₕᵧ" => "D̽ᵧ",        # cross-weighted centered, y component
    "Dₕ₂" => "D̽₂",        # cross-weighted centered, z/second component
    "Dₕ" => "D̽ₕ",         # cross-weighted centered, bare stem
    raw"\overset{\times}{" => raw"\tilde{", # LaTeX: the starred operator's math mark, whatever it wraps
]

# `!` forms need no rules of their own: `D̽ₓ!` contains `D̽ₓ`, so it renames with it (see
# "Identifier boundaries" above -- the trailing `!` is explicitly let through the check).

const RENAME_DICT = Dict(RENAMES)

# The LaTeX rule is exempt from the identifier-boundary check: `\overset{\times}{...}` is not
# an identifier, and every existing use is the old starred mark, followed by whatever the mark
# wraps (`\textrm{D}`, `\mathrm{D}`, bare `D`, ...) and then that content's own subscript
# (`_{+x}`), which would otherwise fail the boundary check on the trailing `{` or `_`.
const NO_BOUNDARY_CHECK = Set([raw"\overset{\times}{"])

_escape_regex(pat) = replace(pat, r"([\\^$.|?*+()\[\]{}])" => s"\\\1")

const RENAME_REGEX = Regex(join(("(?:$(_escape_regex(old)))" for (old, _) in RENAMES), "|"))

# `.claude/skills` is Claude Code project configuration, not part of a user's own tree; the
# skills there are renamed by the integrator separately, not by this script.
const DEFAULT_ROOTS = [
    "src", "test", "ext", "benchmark", "docs/src", "docs/make.jl",
    "docs/generate_benchmarks.jl", "manual", "README.md", "CONTEXT.md",
    "STANDARDS.md", "DOCUMENTATION_PLAN.md"
]

const EXTENSIONS = (".jl", ".md", ".tex", ".toml")

# Generated or settled files the text pass must not touch, plus the v3.0 rename script, which
# is history and never gets rewritten by this one.
const SKIP = [
    "docs/build", "graphify-out", "test/space/operator_docstrings.txt", "benchmark/baselines",
    "dev/rename_v3.jl", "dev/rename_v3_12.jl", "dev/rename_baselines.jl"
]

function should_skip(path)
    any(occursin(skip, path) for skip in SKIP) && return true
    # The generated examples only; their Literate sources are renamed like any other file.
    return occursin("docs/src/examples/", path) && endswith(path, ".md")
end

function collect_files(roots)
    files = String[]
    for root in roots
        if isfile(root)
            should_skip(root) || push!(files, root)
            continue
        end
        isdir(root) || continue
        for (dir, _, names) in walkdir(root), name in names

            path = joinpath(dir, name)
            (any(e -> endswith(name, e), EXTENSIONS) && !should_skip(path)) &&
                push!(files, path)
        end
    end
    return files
end

"""
Whether a match of `RENAME_REGEX` at `[mstart, mstop)` (codeunit indices into `s`, half-open)
sits on an identifier boundary: not preceded by an identifier character, and not followed by
one except a trailing `!`.
"""
function _boundary_ok(s::String, mstart::Int, mstop::Int)
    prev_ok = mstart == firstindex(s) || !Base.is_id_char(s[prevind(s, mstart)])
    next_ok = if mstop > lastindex(s)
        true
    else
        c = s[mstop]
        c == '!' || !Base.is_id_char(c)
    end
    return prev_ok && next_ok
end

"""
Applies the rename to `s`, returning `(result, n_renamed)`.
"""
function _rename_counted(s::String)
    io = IOBuffer()
    pos = firstindex(s)
    n = 0
    for m in eachmatch(RENAME_REGEX, s)
        mstart = m.offset
        write(io, s[pos:prevind(s, mstart)])
        mstop = mstart + ncodeunits(m.match)
        if m.match in NO_BOUNDARY_CHECK || _boundary_ok(s, mstart, mstop)
            write(io, RENAME_DICT[m.match])
            n += 1
        else
            write(io, m.match)
        end
        pos = mstop
    end
    write(io, s[pos:lastindex(s)])
    return String(take!(io)), n
end

"""
    rename_v3_12(s::String)::String

Applies the v3.12 operator rename (gpena/Bramble.jl#349) to a single string: the starred
forward family (`D̽`) becomes the tilde family (`D̃`), and the cross-weighted centered family
(`Dₕ`) takes the freed `D̽` names, in one simultaneous pass. A match only renames when it sits
on an identifier boundary (see "Identifier boundaries" above), so a user name like `xDₕ` or
`myD̽ₓ` is left untouched.
"""
rename_v3_12(s::String) = _rename_counted(s)[1]

function main(args)
    check_only = "--check" in args
    roots = filter(a -> !startswith(a, "--"), args)
    isempty(roots) && (roots = DEFAULT_ROOTS)

    files = collect_files(roots)

    changed = 0
    total = 0
    for path in files
        text = Base.Unicode.normalize(read(path, String), :NFC)
        new_text, renamed = _rename_counted(text)
        renamed == 0 && continue

        changed += 1
        total += renamed
        check_only || write(path, new_text)
        println(check_only ? "would rename" : "renamed", " $renamed in $path")
    end

    println("\n$(check_only ? "would touch" : "touched") $changed files, $total occurrences")
    check_only && println("run without --check to write the changes")
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
