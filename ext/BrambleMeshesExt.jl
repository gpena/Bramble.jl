module BrambleMeshesExt
using Bramble:
               Bramble, CartesianProduct, dim, Mesh1D, MeshnD, points, domain, interval, ×, mesh

using Meshes: Meshes
using PrecompileTools: @setup_workload, @compile_workload

# `Meshes.viz` has no method for a geometry until a Makie backend is loaded, so the call
# fails with a `MethodError` on `Meshes.viz` itself. That is the only failure turned into a
# warning: any other error (a conversion bug, a Meshes error, an interrupt) propagates.
function _is_no_backend_error(err, geometry)
    err isa MethodError || return false
    called = err.f === Core.kwcall ? err.args[2] : err.f
    return called === Meshes.viz && any(arg -> arg === geometry, err.args)
end

function _viz_or_warn(geometry; kwargs...)
    try
        return Meshes.viz(geometry; kwargs...)
    catch err
        _is_no_backend_error(err, geometry) || rethrow()
        @warn "Meshes.viz needs a Makie backend: install `GLMakie` and add `using GLMakie` \
               to your preamble."
        return nothing
    end
end

function _check_viz_dimension(D)
    D >= 2 || throw(ArgumentError("viz needs a mesh of dimension >= 2, got $D"))
    return nothing
end

"""
    Meshes.viz(X::CartesianProduct)

Visualize a [`CartesianProduct`](@ref) of dimension at least 2; a 1D product throws an
`ArgumentError`. Keyword arguments are forwarded to `Meshes.viz`. Without a Makie backend
loaded it warns and returns `nothing`.
"""
function Meshes.viz(X::CartesianProduct; kwargs...)
    D = dim(X)
    _check_viz_dimension(D)

    mins = ntuple(i -> X.box[i][1], D)
    maxs = ntuple(i -> X.box[i][2], D)
    mesh_set = Meshes.Box(mins, maxs)

    return _viz_or_warn(mesh_set; kwargs...)
end

"""
    Meshes.viz(Ωₕ::MeshnD{D}; kwargs...)

Visualize a [`MeshnD`](@ref). Keyword arguments are forwarded to `Meshes.viz`, with
`showsegments = true` by default. Without a Makie backend loaded it warns and returns
`nothing`.
"""
function Meshes.viz(M::MeshnD{D}; kwargs...) where {D}
    _check_viz_dimension(D)

    pts = points(M)
    grid = Meshes.RectilinearGrid(pts...)

    return _viz_or_warn(grid; showsegments = true, kwargs...)
end

"""
    Meshes.viz(Ωₕ::Mesh1D; kwargs...)

Throws an `ArgumentError`: 1D meshes cannot be visualized.
"""
Meshes.viz(::Mesh1D; kwargs...) = _check_viz_dimension(1)

# Warms both `Meshes.viz` methods above (not the 1D one, which only ever throws) for a 2D
# `CartesianProduct` and a 2D `MeshnD`. Each already routes its real `Meshes.viz` call
# through `_viz_or_warn`, which warns when no Makie backend is installed -- precompiling
# through that path is no riskier than an ordinary call from a session without GLMakie
# loaded, which this code is already designed to tolerate. Only reachable once `Meshes` is
# loaded, so only this extension's own precompile pass reaches it.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        X = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωₕ = mesh(domain(X), (4, 4), (true, true))

        @compile_workload begin
            Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                Meshes.viz(X)
                Meshes.viz(Ωₕ)
            end
        end
    end
end

end
