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
    # `parent`/`reshape` are in exactly that position, on `VectorElement`, and `extrema` is
    # too, on `CartesianProduct`/`Domain`. All are defined as methods on
    # `Base.parent`/`Base.reshape`/`Base.extrema`, extensions rather than new functions,
    # and not piracy because `VectorElement`/`CartesianProduct`/`Domain` belong to this
    # package.
    #
    # A name Base defines but does not export is fine too: `using Bramble` would resolve
    # it to this package's own definition without ambiguity, the same way `tails` (removed
    # in gpena/Bramble.jl#76) used to share a name with a non-exported Base internal.
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
    @test Bramble.parent === Base.parent
    @test Bramble.reshape === Base.reshape
    @test Bramble.extrema === Base.extrema
    v = [1, 2, 3]
    @test parent(v) === v  # generic AbstractArray fallback, untouched
    @test extrema(v) == (1, 3)  # generic AbstractArray reduction, untouched
    I = interval(0.0, 2.0)
    @test extrema(I) == (0.0, 2.0)
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
    uₕ = Rₕ(gridspace(Ωₕ), x -> x^2)
    @test parent(uₕ) ≈ [0.0, 0.0625, 0.25, 0.5625, 1.0]
    @test reshape(uₕ) == parent(uₕ)
end
