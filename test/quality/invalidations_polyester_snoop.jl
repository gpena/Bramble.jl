module QualityInvalidationsPolyesterSnoopTests

#===========================================================================#
# Run in a fresh process (never `include`d into a session that has already
# loaded Polyester). Bramble is loaded outside the snoop, so only what
# `using Polyester` invalidates -- BramblePolyesterExt's methods being
# inserted -- is counted.
#
# Prints `POLYESTER_OWNED=<n>` (trees whose triggering method lives in
# BramblePolyesterExt), `CPUPOLYESTER_INVALIDATED=<n>` (distinct invalidated
# MethodInstances whose specTypes mention CpuPolyester), then each owned tree.
#===========================================================================#

using Bramble
using SnoopCompileCore

# Bound first: `@snoop_invalidations` assigns in a branch JET cannot see is always taken.
invalidations = nothing
invalidations = @snoop_invalidations begin
    using Polyester
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
    startswith(string(_owner_module(t.method)), "BramblePolyesterExt") && push!(owned, t)
end

_mentions_cpupolyester(mi) = occursin("CpuPolyester", string(mi.specTypes))

_collect!(seen, node::Pair) = _collect!(seen, node.second)
function _collect!(seen, node)
    mi = node.mi
    _mentions_cpupolyester(mi) && push!(seen, mi)
    for c in node.children
        _collect!(seen, c)
    end
    return seen
end

seen = Set{Core.MethodInstance}()
for t in trees
    for node in t.backedges
        _collect!(seen, node)
    end
    for node in t.mt_backedges
        _collect!(seen, node)
    end
end

println("POLYESTER_OWNED=", length(owned))
println("CPUPOLYESTER_INVALIDATED=", length(seen))
for t in owned
    println(t)
end

end # module QualityInvalidationsPolyesterSnoopTests
