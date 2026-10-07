module QualityTestEnvTests

using Test

# The test environment (`test/Project.toml`) lists only packages the
# tests load, and every non-stdlib entry carries a `[compat]` bound. `Aqua.test_stale_deps`
# cannot do this, since it takes a package module and `test/` is a project, not a package.
#
# Deliberately Test-only: no `using Bramble`, so this file runs standalone. Scans with
# `Meta.parseall` rather than a regex, so a docstring or comment that mentions `using Foo` is
# not a load. The project and the scanned roots are parameters so the check has controls that
# leave the real files alone: `BRAMBLE_TEST_ENV_PROJECT` (default `test/Project.toml`) and
# `BRAMBLE_TEST_ENV_ROOTS` (colon-separated; default `test/` and `docs/src/examples/`, which
# the example tests run).

import TOML

const _REPO = normpath(joinpath(@__DIR__, "..", ".."))
const _PROJECT = get(ENV, "BRAMBLE_TEST_ENV_PROJECT", joinpath(_REPO, "test", "Project.toml"))
const _ROOTS = let r = get(ENV, "BRAMBLE_TEST_ENV_ROOTS", "")
    isempty(r) ? [joinpath(_REPO, "test"), joinpath(_REPO, "docs", "src", "examples")] :
    String.(split(r, ':'; keepempty = false))
end

# Dependencies the tests need without a `using`/`import` line, each with its reason.
# CpuId is indirect (Polyester -> PolyesterWeave -> CPUSummary), listed only to hold it at
# 0.3.1: 0.3.2's precompile workload throws on aarch64, so the Polyester stack loads
# without cached code on Apple Silicon. Remove it with its compat entry once a CpuId
# release fixes m-j-w/CpuId.jl#67.
const _NEEDED_WITHOUT_USING = Dict{String, String}(
    "CpuId" => "pinned below 0.3.2, which cannot precompile on aarch64 (CpuId.jl#67)")

# Bramble is the package under test, reached through `[sources]`.
const _SELF = "Bramble"

const _STDLIBS = Set(readdir(Sys.STDLIB))

function _jl_files(root)
    files = String[]
    isfile(root) && return endswith(root, ".jl") ? [root] : files
    for (dir, _, names) in walkdir(root)
        for name in names
            endswith(name, ".jl") && push!(files, joinpath(dir, name))
        end
    end
    return files
end

# Top-level package names of every `using`/`import` node, at any depth. Relative paths
# (`using .Foo`, `import ..Foo`) name modules, not packages, and so do interpolated ones
# (`using $pkg`, inside a quoted block); both are skipped.
function _loaded!(out, ex)
    ex isa Expr || return out
    if ex.head === :using || ex.head === :import
        for a in ex.args
            a.head === :(:) && (a = a.args[1])   # `using A: x` names A
            a.head === :as && (a = a.args[1])    # `import A as B` names A
            name = first(a.args)
            name isa Symbol && name !== :(.) && push!(out, String(name))
        end
        return out
    end
    for a in ex.args
        _loaded!(out, a)
    end
    return out
end

function _loaded_packages(roots)
    out = Set{String}()
    for root in roots, file in _jl_files(root)

        _loaded!(out, Meta.parseall(read(file, String); filename = file))
    end
    return out
end

@testset "test environment (#390)" begin
    project = TOML.parsefile(_PROJECT)
    deps = sort!(collect(keys(get(project, "deps", Dict{String, Any}()))))
    compat = get(project, "compat", Dict{String, Any}())
    loaded = _loaded_packages(_ROOTS)

    @testset "every dependency is loaded" begin
        unused = [d for d in deps
                  if d != _SELF && !(d in loaded) && !haskey(_NEEDED_WITHOUT_USING, d)]
        @test isempty(unused)
        isempty(unused) || @info "test dependencies no `using`/`import` loads" unused
    end

    @testset "non-stdlib dependencies have compat" begin
        missing_compat = [d for d in deps
                          if d != _SELF && !(d in _STDLIBS) && !haskey(compat, d)]
        @test isempty(missing_compat)
        isempty(missing_compat) ||
            @info "non-stdlib test dependencies without [compat]" missing_compat
    end
end

end # module
