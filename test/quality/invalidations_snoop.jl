module QualityInvalidationsSnoopTests

#===========================================================================#
# Run in a fresh process by invalidations.jl (never `include`d into a session
# that has already loaded Bramble -- `@snoop_invalidations` only sees
# invalidations triggered while `using Bramble` actually runs `__init__`/
# method definitions, which a second `using` in an already-loaded session
# skips entirely).
#
# Prints `OWNED_COUNT=<n>` followed by one line per package-owned invalidation
# tree, so the parent process can assert on the count and still show the
# offending tree(s) when it fails.
#===========================================================================#

using SnoopCompileCore

invalidations = @snoop_invalidations begin
    using Bramble
end

using SnoopCompile

trees = invalidation_trees(invalidations)

# Package-owned: the method (or, on Julia 1.13+, the binding) whose insertion
# triggered the tree is defined in Bramble itself or one of its package
# extensions (BrambleMakieExt, etc.) -- never in Base, Core, or a standard
# library, which is what every other tree here (OrderedCollections,
# SparseArrays, Dates precompiled elsewhere in the depot) reflects instead.
#
# On Julia 1.13, `t.method` can be a `Core.Binding` (a binding invalidation)
# rather than a `Method` -- it has no `.module` field, only `.globalref`, whose
# `.mod` is the owning module. One method per type keeps the ownership check
# the same for both.
_owner_module(m::Method) = m.module
_owner_module(b::Core.Binding) = b.globalref.mod

# `t.method` can also be `nothing`: SnoopCompile's own :unknown-reason trees, built when
# a root `MethodInstance` surfaces invalidated at the C level with no method that
# inserted/deleted it to blame (SnoopCompile's `invalidations.jl`, the "unknown nothing"
# case). What such a tree names instead are the *superseded* MethodInstances -- code
# Bramble itself had cached, now invalidated by something else's load -- not the method
# that caused the invalidation, so attributing the tree to their module would blame
# Bramble for its own code being knocked out of the cache rather than for inserting a
# method that broke someone else's. With no inserting method to blame, such a tree is
# never package-owned; it only gets logged, under UNATTRIBUTED_COUNT, so it stays
# visible without failing the gate.

owned = empty(trees)
unattributed = empty(trees)
for t in trees
    if t.method === nothing
        push!(unattributed, t)
    else
        mod = _owner_module(t.method)
        startswith(string(mod), "Bramble") && push!(owned, t)
    end
end

println("OWNED_COUNT=", length(owned))
for t in owned
    println(t)
end
println("UNATTRIBUTED_COUNT=", length(unattributed))
for t in unattributed
    println(t)
end

end # module QualityInvalidationsSnoopTests
