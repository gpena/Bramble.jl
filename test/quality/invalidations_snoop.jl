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

# Package-owned: the method whose insertion triggered the tree is defined in
# Bramble itself or one of its package extensions (BrambleMakieExt, etc.) --
# never in Base, Core, or a standard library, which is what every other tree
# here (OrderedCollections, SparseArrays, Dates precompiled elsewhere in the
# depot) reflects instead.
owned = filter(t -> startswith(string(t.method.module), "Bramble"), trees)

println("OWNED_COUNT=", length(owned))
for t in owned
    println(t)
end
