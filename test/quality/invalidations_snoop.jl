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

# On Julia 1.12, `t.method` can also be `nothing`: SnoopCompile's own :unknown-reason
# trees, built when a root `MethodInstance` surfaces invalidated at the C level with no
# method that inserted/deleted it to blame (SnoopCompile's `invalidations.jl`, the
# "unknown nothing" case). Such a tree is only ever pushed when it still carries at
# least one `InstanceNode` (in `backedges` or `mt_backedges`), so fall back to that
# instance's own defining method for a module. If somehow neither is present, the tree
# can't be attributed at all; count it separately instead of guessing.
function _owner_module(::Nothing, backedges, mt_backedges)
    node = !isempty(backedges) ? first(backedges) :
           !isempty(mt_backedges) ? last(first(mt_backedges)) : nothing
    node === nothing && return nothing
    def = node.mi.def
    return def isa Method ? def.module : nothing
end

owned = empty(trees)
n_unattributed = 0
for t in trees
    mod = t.method === nothing ? _owner_module(t.method, t.backedges, t.mt_backedges) :
          _owner_module(t.method)
    if mod === nothing
        global n_unattributed += 1
    elseif startswith(string(mod), "Bramble")
        push!(owned, t)
    end
end

println("OWNED_COUNT=", length(owned))
for t in owned
    println(t)
end
println("UNATTRIBUTED_COUNT=", n_unattributed)

end # module QualityInvalidationsSnoopTests
