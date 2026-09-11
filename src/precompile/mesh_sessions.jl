# precompile/mesh_sessions.jl: one full pass over the mesh interface (src/mesh/) --
# construction, indexed queries, iteration, mutation and display. Ordinary methods, so
# PrecompileTools caches them along with everything they call.

# 1D meshes are indexed by a scalar, nD meshes by the index tuple.
function _pc_indexed(Ωₕ::AbstractMeshType{1}, idx_tup)
    i = idx_tup[1]
    point(Ωₕ, i)
    spacing(Ωₕ, i)
    forward_spacing(Ωₕ, i)
    half_spacing(Ωₕ, i)
    half_point(Ωₕ, i)
    cell_measure(Ωₕ, i)
    return nothing
end

function _pc_indexed(Ωₕ::AbstractMeshType, idx_tup)
    point(Ωₕ, idx_tup)
    spacing(Ωₕ, idx_tup)
    forward_spacing(Ωₕ, idx_tup)
    half_spacing(Ωₕ, idx_tup)
    half_point(Ωₕ, idx_tup)
    cell_measure(Ωₕ, idx_tup)
    return nothing
end

# One full pass over the mesh interface: construction, queries, iteration,
# mutation and display.
function _pc_mesh_session(Ω, npts, unif, be, label::Symbol)
    Ωₕ = mesh(Ω, npts, unif; backend=be)

    idx = first(indices(Ωₕ))
    _pc_indexed(Ωₕ, Tuple(idx))
    point(Ωₕ, idx)
    spacing(Ωₕ, idx)
    forward_spacing(Ωₕ, idx)

    dim(Ωₕ)
    dim(typeof(Ωₕ))
    eltype(Ωₕ)
    eltype(typeof(Ωₕ))
    topo_dim(Ωₕ)
    set(Ωₕ)
    backend(Ωₕ)
    markers(Ωₕ)
    npoints(Ωₕ)
    npoints(Ωₕ, Tuple)
    size(Ωₕ)
    length(Ωₕ)
    axes(Ωₕ)
    points(Ωₕ)
    half_points(Ωₕ)
    half_spacings(Ωₕ)
    hₘₐₓ(Ωₕ)
    hₘᵢₙ(Ωₕ)
    spacings(Ωₕ)
    cell_measures(Ωₕ)
    is_uniform(Ωₕ)
    is_collapsed(Ωₕ(1))

    indices(Ωₕ)
    boundary_indices(Ωₕ)
    interior_indices(Ωₕ)
    is_boundary_index(Ωₕ, idx)
    index_in_marker(Ωₕ, label)
    boundary_symbol_to_dict(indices(Ωₕ))

    # gpena/Bramble.jl#75: warms the plain accessors and the mesh's own iteration
    # protocol, not the deprecated `*_iterator` aliases these replace.
    for iter in (
        points(Ωₕ),
        half_points(Ωₕ),
        spacings(Ωₕ),
        forward_spacings(Ωₕ),
        half_spacings(Ωₕ),
        (cell_measure(Ωₕ, idx) for idx in indices(Ωₕ)),
    )
        isempty(iter) || first(iter)
    end
    for p in Ωₕ
        p
    end

    Ωₕ[idx]
    # Both display paths: the embeddable one-liner and the detailed `MIME"text/plain"`
    # block, which are now separate methods rather than one flag-switched body
    # (gpena/Bramble.jl#45).
    sprint(show, Ωₕ)
    sprint(show, MIME"text/plain"(), Ωₕ)

    return Ωₕ
end

# Mesh mutation is a separate pass so the queries above stay on a pristine mesh.
function _pc_mesh_mutation(Ωₕ, dm)
    # The one-argument path is for a mesh with no custom labels to begin with; every
    # `Ωₕ` reaching this function carries `dm`'s own, so calling it directly would throw
    # (correctly) rather than drop them on precompilation. Strip them first so this
    # exercises the intended no-custom-labels case; the two-argument call right below
    # already exercises the marked-mesh path. `MeshnD` carries labels twice: once on the
    # multidimensional mesh itself, and once on each dimension's `Mesh1D` submesh, so both need
    # clearing, not just the outer one.
    bare = deepcopy(Ωₕ)
    bare.markers = MeshMarkers()
    if bare isa MeshnD
        for sm in bare.submeshes
            sm.markers = MeshMarkers()
        end
    end
    iterative_refinement!(bare)
    iterative_refinement!(deepcopy(Ωₕ), dm)
    pts = points(Ωₕ)
    change_points!(deepcopy(Ωₕ), pts)
    change_points!(deepcopy(Ωₕ), dm, pts)
    return nothing
end
