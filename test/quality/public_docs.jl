module QualityPublicDocsTests

using Test, Bramble

# A public name (exported or declared `public`) is only reachable in the built manual if
# some `@docs` or `@autodocs` block lists it. Documenter's own `missing_docs` check tests
# the wrong side of this gap: it verifies a docstring exists in *source*, not that any page
# includes it (gpena/Bramble.jl#314). `host_weights` and `metal_sparse_csr`/`metal_sparse_csc`
# carried real docstrings and were still unreachable, and a `[`name`](@ref)` to one of them
# fails the docs build with "Cannot resolve @ref" rather than rendering nowhere quietly.
#
# This scans the *text* of `docsrc`'s Markdown, not a live Documenter build: a name in a
# ```@docs block counts if a line there names it (after stripping a leading module prefix
# and a call signature), and a ```@autodocs block with `Modules = [Bramble]` counts every
# name whose docstring's source file matches its `Pages = [...]` list (or every name, with
# no `Pages` filter). `Public`/`Private`/`Filter` restrictions on an `@autodocs` block are
# not modelled -- this is deliberately a coarse reachability check, not a reimplementation
# of Documenter's own filtering.

# Names re-exported from Base or another package (×, ⋅, close, ...) are documented there.
function _is_ours(mod::Module, name::Symbol)
    isdefined(mod, name) || return false
    v = getproperty(mod, name)
    v isa Function || return true
    return parentmodule(v) === mod
end

function _public_names(mod::Module)
    return filter(n -> n !== nameof(mod) && _is_ours(mod, n), names(mod))
end

# A `@docs`/`@autodocs` line names a symbol, possibly `Mod.name` or `name(::T)`; strip both.
function _line_symbol(line::AbstractString)
    s = strip(line)
    isempty(s) && return nothing
    startswith(s, "#") && return nothing
    if (i = findfirst('(', s)) !== nothing
        s = s[1:prevind(s, i)]
    end
    if (i = findlast('.', s)) !== nothing
        s = s[nextind(s, i):end]
    end
    s = strip(s)
    isempty(s) && return nothing
    return Symbol(s)
end

function _docs_md_files(docsrc::AbstractString)
    files = String[]
    for (root, _, fnames) in walkdir(docsrc)
        for f in fnames
            endswith(f, ".md") && push!(files, joinpath(root, f))
        end
    end
    return files
end

# Parses fenced ```@docs and ```@autodocs blocks out of one Markdown file's text.
function _fenced_blocks(text::AbstractString, tag::AbstractString)
    blocks = Vector{String}[]
    lines = split(text, '\n')
    i = 1
    while i <= length(lines)
        if strip(lines[i]) == "```$tag"
            body = String[]
            j = i + 1
            while j <= length(lines) && strip(lines[j]) != "```"
                push!(body, lines[j])
                j += 1
            end
            push!(blocks, body)
            i = j + 1
        else
            i += 1
        end
    end
    return blocks
end

function _autodocs_field(block::Vector{String}, key::AbstractString)
    for line in block
        m = match(Regex("^\\s*" * key * "\\s*=\\s*(.*)\$"), line)
        m === nothing && continue
        return m.captures[1]
    end
    return nothing
end

function _parse_string_list(s::AbstractString)
    return [String(m.match) for m in eachmatch(r"\"([^\"]*)\"", s)]
end

# Names covered by ```@docs blocks (direct name match) and ```@autodocs blocks with
# `Modules = [Bramble]` (every name whose docstring's source file matches `Pages`, or every
# name of `mod` with no `Pages` filter).
function _documented_names(mod::Module, docsrc::AbstractString)
    covered = Set{Symbol}()
    all_names = _public_names(mod)
    doc_file = Dict{Symbol, String}()
    for n in all_names
        b = Docs.Binding(mod, n)
        meta = Docs.meta(mod)
        haskey(meta, b) || continue
        m = meta[b]
        isempty(m.order) && continue
        ds = m.docs[first(m.order)]
        path = get(ds.data, :path, nothing)
        path === nothing || (doc_file[n] = String(path))
    end

    for file in _docs_md_files(docsrc)
        text = read(file, String)
        for block in _fenced_blocks(text, "@docs")
            for line in block
                sym = _line_symbol(line)
                sym === nothing || push!(covered, sym)
            end
        end
        for block in _fenced_blocks(text, "@autodocs")
            modules_field = _autodocs_field(block, "Modules")
            (modules_field === nothing || !occursin(string(nameof(mod)), modules_field)) &&
                continue
            pages_field = _autodocs_field(block, "Pages")
            if pages_field === nothing
                union!(covered, all_names)
            else
                pages = _parse_string_list(pages_field)
                for n in all_names
                    path = get(doc_file, n, nothing)
                    path === nothing && continue
                    any(p -> endswith(path, p), pages) && push!(covered, n)
                end
            end
        end
    end
    return covered
end

"""
    missing_from_docs(mod::Module, docsrc::AbstractString) -> Vector{Symbol}

The names `mod` exports or declares `public` (excluding names owned by another module, such
as Base/LinearAlgebra re-exports) that no ```@docs``` or ```@autodocs``` block under `docsrc`
lists, scanning every `.md` file recursively. Sorted.
"""
function missing_from_docs(mod::Module, docsrc::AbstractString)
    covered = _documented_names(mod, docsrc)
    missing_names = filter(n -> !(n in covered), _public_names(mod))
    return sort(missing_names)
end

const _docsrc = joinpath(@__DIR__, "..", "..", "docs", "src")

@testset "Public names are reachable in the manual (#314)" begin
    missing_names = missing_from_docs(Bramble, _docsrc)
    if !isempty(missing_names)
        @info "public names in no @docs/@autodocs block" missing_names
    end
    @test isempty(missing_names)
end

module Fake
public zzz_undoc
zzz_undoc() = 1
end

@testset "missing_from_docs positive/negative controls" begin
    @test :zzz_undoc in missing_from_docs(Fake, _docsrc)
    # `form` is listed in a ```@docs block in docs/src/api.md and must not be reported.
    @test :form ∉ missing_from_docs(Bramble, _docsrc)
end

end # module QualityPublicDocsTests
