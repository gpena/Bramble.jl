module QualityInvalidationsSciMLSnoopTests

#===========================================================================#
# Run in a fresh process by invalidations.jl. SciMLBase (and ForwardDiff) are
# loaded before the snoop, as in a SciML session that loads Bramble last, so
# `using Bramble` also loads BrambleSciMLExt inside it: what the extension's
# methods invalidate is counted with the core package's.
#
# Prints `SCIML_EXT_LOADED=<true|false>`, `SCIML_OWNED_TREES=<n>` (trees whose
# triggering method is Bramble's or an extension's), `SCIML_OWNED_INVALIDATED=
# <n>` (the MethodInstances those trees invalidate), then each owned tree.
#===========================================================================#

using SciMLBase, ForwardDiff
using SnoopCompileCore

# Bound first: `@snoop_invalidations` assigns in a branch JET cannot see is always taken.
invalidations = nothing
invalidations = @snoop_invalidations begin
    using Bramble
end

using SnoopCompile

trees = invalidation_trees(invalidations)

# On Julia 1.13 `t.method` can be a `Core.Binding`; `nothing` marks SnoopCompile's
# unknown-reason trees, which have no inserting method to blame.
_owner_module(m::Method) = m.module
_owner_module(b::Core.Binding) = b.globalref.mod

owned = empty(trees)
for t in trees
    t.method === nothing && continue
    startswith(string(_owner_module(t.method)), "Bramble") && push!(owned, t)
end

println("SCIML_EXT_LOADED=", Base.get_extension(Bramble, :BrambleSciMLExt) !== nothing)
println("SCIML_OWNED_TREES=", length(owned))
println("SCIML_OWNED_INVALIDATED=", sum(SnoopCompile.countchildren, owned; init = 0))
for t in owned
    println(t)
end

end # module QualityInvalidationsSciMLSnoopTests
