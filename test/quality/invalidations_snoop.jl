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

owned = filter(t -> startswith(string(_owner_module(t.method)), "Bramble"), trees)

println("OWNED_COUNT=", length(owned))
for t in owned
    println(t)
end

end # module QualityInvalidationsSnoopTests
