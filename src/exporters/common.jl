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
