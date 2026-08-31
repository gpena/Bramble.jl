# parallel_workspace.jl
#
# The preallocated partition a parallel assembly walks.
#
# This lived in bilinear.jl, and both assembly files need it: `linear.jl` names it as the
# type of a `LinearForm` field, which a struct definition has to resolve at definition time
# rather than at call time. So unlocking `linear.jl` on its own was impossible while the
# type sat in the other file — `linear.jl`'s own note at the top said as much and left it
# unresolved.
#
# It belongs here rather than in either: a colouring of the grid into independent groups is
# a property of the mesh and the stencil, not of whether the form being assembled is linear
# or bilinear. Both files use it the same way.

"""
	ParallelWorkspace{D}

Preallocated structure containing coordinate indices partitioned into lock-free/independent
colour groups, together with the per-thread scratch buffers an assembly writes through.

The colouring is what makes the assembly lock free: no two indices within a group write to
the same matrix entry, so a group can be walked in parallel without synchronisation. Shared
by [`LinearForm`](@ref) and [`BilinearForm`](@ref), which is why it lives in a file of its
own rather than in either of theirs.
"""
struct ParallelWorkspace{D}
    color_groups::Vector{Vector{CartesianIndex{D}}}
    thread_buffers::Vector{Vector{Float64}}

    function ParallelWorkspace{D}(color_groups::Vector{Vector{CartesianIndex{D}}}) where {D}
        new{D}(color_groups, Vector{Float64}[])
    end

    function ParallelWorkspace{D}(color_groups::Vector{Vector{CartesianIndex{D}}},
            thread_buffers::Vector{Vector{Float64}}) where {D}
        new{D}(color_groups, thread_buffers)
    end
end
