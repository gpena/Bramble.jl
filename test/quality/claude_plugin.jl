module QualityClaudePluginTests

using Test
using Bramble

# The Claude Code plugin under plugins/bramble/ teaches an assistant the public API. It goes
# stale silently: a renamed function, or one that stops being exported, leaves the skill
# telling users to write code that throws an `UndefVarError`. These checks read the skill's
# Markdown and fail on the three ways that has happened:
#
# 1. `Bramble.name` for a name that is neither exported nor `public`.
# 2. A `public`-only name written bare in a `julia` block, where `using Bramble` does not
#    bring it into scope (the block may import it with `using Bramble: name`).
# 3. A name in Bramble's notation (subscripts such as `ₕ`, `ₓ`, `₊`, `₋`, or `∇`, `Δ`) that
#    Bramble does not define and the snippet does not assign. This is how a made-up spelling
#    such as `Dₕ` was found.
#
# The examples under plugins/bramble/skills/bramble/examples/ run in the `examples` group
# (test/examples/claude_plugin.jl), which is where their own assertions execute.

const _ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const _PLUGIN = joinpath(_ROOT, "plugins", "bramble")
const _SKILL = joinpath(_PLUGIN, "skills", "bramble")

# An identifier, including subscripts and combining marks (`D̽ₕ`, `∇cₕ`, `inner₊ₓ`).
const _ID = r"(?<![\w.:∇\p{M}])([\p{L}_∇][\w\p{M}!′₀-₉ₐ-ₜ₊₋ᵢ-ᵪ∇]*)"
# Bramble's notation: a subscript direction or sign, or a leading `∇`.
const _NOTATION = r"[ₕₓᵧ₂₊₋]|^∇"
_is_subscript_only(s) = all(c -> c in "ₕₓᵧ₂₊₋", s)          # prose naming a subscript: `ₕ`

# The first capture of a match the pattern guarantees; `something` narrows it for JET.
_cap(m::RegexMatch) = something(m.captures[1])

_markdown_files() = [joinpath(dir, f) for (dir, _, files) in walkdir(_SKILL)
                     for f in files if endswith(f, ".md")]

_julia_blocks(text) = [_cap(m) for m in eachmatch(r"```julia\n(.*?)```"s, text)]
_inline_spans(text) = [_cap(m)
                       for m in eachmatch(r"`([^`\n]+)`", replace(text, r"```.*?```"s => ""))]
_strip_comments(code) = replace(code, r"#[^\n]*" => "")

function _owned_api_name(s::Symbol)
    isdefined(Bramble, s) || return false
    return Base.which(Bramble, s) === Bramble
end
_exported(s) = Base.isexported(Bramble, s)
_public(s) = Base.ispublic(Bramble, s)

# Names a snippet binds itself: assignments (`x = …`, `a, b = …`), function definitions,
# loop variables and anonymous-function arguments.
function _local_names(code)
    names = Set{String}()
    for m in eachmatch(r"^\s*([^=\n#]+?)\s*=(?!=)"m, code)
        lhs = replace(_cap(m), r"\(.*" => "")   # `f(x) = …` binds `f`
        for t in eachmatch(_ID, lhs)
            push!(names, _cap(t))
        end
    end
    for re in (r"function\s+([^\s(]+)", r"for\s+([^=\n]+?)\s+in\b", r"\(([^()]*)\)\s*->",
        r"([\p{L}_][\w\p{M}]*)\s*->")
        for m in eachmatch(re, code), t in eachmatch(_ID, _cap(m))

            push!(names, _cap(t))
        end
    end
    return names
end

function _imported(code)
    Set(_cap(t) for m in eachmatch(r"using Bramble:\s*([^\n]+)", code)
    for t in eachmatch(_ID, _cap(m)))
end

# Conventional variable names the reference files use without assigning them first.
const _NOTATION_VARIABLES = Set(["Ωₕ", "Wₕ", "Vₕ", "Zₕ", "uₕ", "vₕ", "wₕ", "zₕ", "fₕ", "gₕ", "αₕ",
    "uₓ", "uᵧ", "u₂", "Ωₕ_1d", "Ωₕ_mx", "Wₕ_dest", "Wₕ_src", "u_avg", "hₘₐₓ_",
    "sd₂", "∇ₕ_"])

function _problems(path)
    text = read(path, String)
    rel = relpath(path, _ROOT)
    found = String[]
    for m in eachmatch(r"Bramble\.([\p{L}_][\w\p{M}!₀-₉ₐ-ₜ₊₋ᵢ-ᵪ]*)", text)
        s = _cap(m)
        s == "jl" && continue
        s == "name" && continue                       # prose: "write `Bramble.name`"
        _public(Symbol(s)) || push!(found, "$rel: `Bramble.$s` is not exported or public")
    end
    blocks = _julia_blocks(text)
    for block in blocks
        code = _strip_comments(block)
        imported = _imported(block)
        locals = _local_names(code)
        for line in split(code, '\n')
            startswith(lstrip(line), "using") && continue
            for t in eachmatch(_ID, line)
                s = _cap(t)
                sym = Symbol(s)
                if _owned_api_name(sym) && _public(sym) && !_exported(sym) && s ∉ imported
                    push!(found, "$rel: public-only `$s` written bare; use `Bramble.$s`")
                elseif occursin(_NOTATION, s) && !_is_subscript_only(s) && !isdefined(Bramble, sym) &&
                       s ∉ locals &&
                       s ∉ _NOTATION_VARIABLES
                    push!(found, "$rel: `$s` is not defined by Bramble")
                end
            end
        end
    end
    for span in _inline_spans(text)
        occursin(r"^[\w./-]+\.(jl|md)$", span) && continue      # a file path
        code = _strip_comments(span)
        locals = _local_names(code)
        for t in eachmatch(_ID, code)
            s = _cap(t)
            prefix = SubString(code, 1, prevind(code, t.offset))
            endswith(prefix, "Bramble.") && continue
            if occursin(_NOTATION, s) && !_is_subscript_only(s) && !isdefined(Bramble, Symbol(s)) &&
               s ∉ locals &&
               s ∉ _NOTATION_VARIABLES
                push!(found, "$rel: `$s` in `$span` is not defined by Bramble")
            end
        end
    end
    return found
end

@testset "Claude Code plugin" begin
    @testset "Manifests" begin
        version = _cap(something(match(r"^version\s*=\s*\"([^\"]+)\""m,
            read(joinpath(_ROOT, "Project.toml"), String))))
        plugin = read(joinpath(_PLUGIN, ".claude-plugin", "plugin.json"), String)
        @test occursin("\"name\": \"bramble\"", plugin)
        # The plugin's version is the package's, so a release that changes the API bumps both.
        @test _cap(something(match(r"\"version\":\s*\"([^\"]+)\"", plugin))) == version
        marketplace = read(joinpath(_ROOT, ".claude-plugin", "marketplace.json"), String)
        @test occursin("\"source\": \"./plugins/bramble\"", marketplace)
    end

    @testset "Bundled files and links" begin
        skill = read(joinpath(_SKILL, "SKILL.md"), String)
        linked = Set(m.match for m in eachmatch(r"(reference|examples)/[\w-]+\.(md|jl)", skill))
        on_disk = Set(joinpath(d, f) for d in ("reference", "examples")
        for f in readdir(joinpath(_SKILL, d)))
        @test sort!(collect(setdiff(linked, on_disk))) == String[]
        @test sort!(collect(setdiff(on_disk, linked))) == String[]
    end

    @testset "Names and their visibility" begin
        problems = reduce(vcat, (_problems(p) for p in _markdown_files()); init = String[])
        isempty(problems) || @info "Stale names in plugins/bramble" problems
        @test problems == String[]
    end
end

end # module
