using Test
using Bramble

# Every exported name carries a docstring.
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
_is_ours(name::Symbol) = isdefined(Bramble, name) &&
                         parentmodule(getproperty(Bramble, name)) === Bramble

@testset "Every exported name is documented" begin
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
