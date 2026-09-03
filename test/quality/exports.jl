using Test
using Bramble

# Properties of the reachable surface: every exported or `public` name carries a docstring,
# and none shadows a different function of the same name in Base.
#
# `names(Bramble)` (default `all = false`) already returns both kinds since Julia 1.11 — a
# `public` name is documented API, just not brought into scope by a bare `using Bramble`
# (point 70) — so this file needed no change to start covering it too.
#
# Documenter's `missing_docs` check is the wrong tool for this: it reports every internal
# helper it cannot find a page for, so turning it into an error would mean adding `@docs`
# stubs to silence it rather than because they help anyone. The rule worth enforcing is
# narrower and has no false positives — a name a user can reach should say what it does.
#
# `Docs.meta` is read directly rather than going through `Base.Docs.doc`, which needs the
# REPL stdlib loaded to have a method for a function object.

const _DOC_META = Docs.meta(Bramble)

function _has_docstring(name::Symbol)
    b = Docs.Binding(Bramble, name)
    haskey(_DOC_META, b) || return false
    m = _DOC_META[b]
    return !isempty(m.order) && !isempty(strip(join(m.docs[first(m.order)].text, "")))
end

# Names re-exported from Base or another package are documented there, not here.
function _is_ours(name::Symbol)
    isdefined(Bramble, name) &&
        parentmodule(getproperty(Bramble, name)) === Bramble
end

@testset "Docstrings exist" begin
    exported = filter(!=(:Bramble), names(Bramble))
    @test !isempty(exported)

    # Operators and types defined elsewhere and re-exported (⋅, ×, close, …) carry their
    # documentation in the module that owns them.
    ours = filter(exported) do n
        v = getproperty(Bramble, n)
        v isa Function ? _is_ours(n) : true
    end

    undocumented = filter(n -> !_has_docstring(n), ours)
    if !isempty(undocumented)
        @info "exported names with no docstring" undocumented
    end
    @test isempty(undocumented)
end

@testset "Base shadowing" begin
    # Exporting a name that Base also exports, bound to a *different* function, makes that
    # name ambiguous for the whole session: after `using Bramble` a call to it raises an
    # UndefVarError naming two modules, and the user loses the Base one everywhere.
    #
    # `values` was in exactly that position. It is defined as a method on `Base.values`
    # now, which is an extension rather than a new function, and not piracy because
    # `VectorElement` belongs to this package.
    #
    # A name Base defines but does not export (`tails`) is fine: `using Bramble` resolves
    # it to this package's without ambiguity.
    clashes = Symbol[]
    for n in names(Bramble)
        n === :Bramble && continue
        (isdefined(Base, n) && isdefined(Bramble, n)) || continue
        Base.isexported(Base, n) || continue
        getproperty(Bramble, n) === getproperty(Base, n) || push!(clashes, n)
    end
    if !isempty(clashes)
        @info "exported names shadowing a different Base function" clashes
    end
    @test isempty(clashes)

    # the case that motivated this: both meanings reachable after `using Bramble`
    @test Bramble.values === Base.values
    @test collect(values(Dict(1 => 2))) == [2]
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
    @test values(Rₕ(gridspace(Ωₕ), x -> x^2)) ≈ [0.0, 0.0625, 0.25, 0.5625, 1.0]
end
