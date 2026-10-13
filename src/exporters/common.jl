# Both exporters write a composite's leaves side by side against one mesh: a VTK vector
# field has one array per component, a table has one row per grid point. Leaves on
# different meshes have no shared "point i", so they are refused rather than written as
# values that pair up unrelated points. `fname` names the exporter in the message.
function _require_one_mesh(fname, uₕ::VectorElement{<:CompositeGridSpace})
    _shares_one_mesh(components(uₕ)) && return nothing
    n = length(components(uₕ))
    k = length(unique(objectid, map(c -> mesh(space(c)), components(uₕ))))
    throw(
        ArgumentError(
        "$fname needs every component of a composite element on the same mesh, but " *
        "its $n components live on $k different meshes. Export each component against " *
        "its own mesh instead.",
    ),
    )
end

# The Plots and Makie extensions share code only through Bramble, so both refuse a grid
# function they cannot plot through this one pair. A composite element has no single
# reading as one curve or one grid, so plot each of its components(...) separately.
@noinline function _throw_plot_composite()
    throw(
        ArgumentError(
        "plotting a composite element directly has no single reading — plot each of its " *
        "components(...) separately.",
    ),
    )
end

# Only 1D curves and 2D grids are wired up. Plots.jl's `surface`/`heatmap` plot a height
# field over a 2D domain, not a true 3D volume, and Makie could render one (`Makie.volume`),
# so for Makie this is a scope decision, not a limit. See `export_vtk`.
@noinline function _throw_plot_dim(D)
    throw(
        ArgumentError(
        "plotting a $(D)D grid function directly is not implemented — only 1D and 2D. " *
        "See export_vtk for a full 3D field.",
    ),
    )
end
