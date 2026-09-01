module BrambleMakieExt

using Bramble: Bramble, VectorElement, ScalarGridSpace, CompositeGridSpace,
               mesh, points, to_matrix, values

import Makie

# A composite element has no single reading as one curve or one grid — plot each of its
# components(...) separately.
function _makie_error_composite()
    throw(ArgumentError(
        "plotting a composite element directly has no single reading — plot each of its " *
        "components(...) separately."))
end

# Unlike PGFPlots, Makie genuinely can render a true 3D volume (`Makie.volume`), so this
# refusal is a scope decision for this extension, not a limit of Makie itself: only 1D
# curves and 2D grids are wired up here.
function _makie_error_dim(D)
    throw(ArgumentError(
        "plotting a $(D)D grid function directly is not implemented — only 1D and 2D. " *
        "See export_vtk for a full 3D field."))
end

# 1D: `lines(uₕ)`, `scatter(uₕ)`. `PointBased` plot types want a vector of points, not two
# separate coordinate vectors — that is a Makie 0.10+ requirement, not a style choice.
function Makie.convert_arguments(::Makie.PointBased, uₕ::VectorElement{<:ScalarGridSpace{1}})
    (Makie.Point2f.(points(mesh(uₕ)), values(uₕ)),)
end

# `VectorElement <: AbstractVector`, so before `convert_arguments` above is ever reached,
# Makie's own `expand_dimensions(::PointBased, ::AbstractVector{<:Real})` intercepts it —
# a generic "bare numeric vector" fallback meant for plain arrays, which auto-generates an
# index axis. It matches `VectorElement` by that generic type alone, and the pipeline it
# routes into fails several layers down with `Cannot convert Point{2,Float64} to Float32`,
# an error that names neither this extension nor `VectorElement` and gives no hint that the
# real cause is dispatch happening one stage earlier than `convert_arguments`.
#
# Found by rendering, not by reading the conversion alone: calling `convert_arguments`
# directly, bypassing the full pipeline, gave the correct data every time and never
# reproduced this. Confirmed with `methods(Makie.expand_dimensions)`, which listed the
# exact intercepting signature. This override matches `VectorElement` more specifically, so
# it wins ordinary dispatch, and returns the two coordinate vectors `convert_arguments`
# above expects to have been given directly.
function Makie.expand_dimensions(::Makie.PointBased, uₕ::VectorElement{<:ScalarGridSpace{1}})
    (
        points(mesh(uₕ)), values(uₕ))
end

# 2D: `heatmap(uₕ)` and `surface(uₕ)`/`contour(uₕ)` alike. Each fixes its own conversion
# trait — `Heatmap` wants `CellGrid`, `Surface` and `Contour` want `VertexGrid` — but both
# read the same three arguments the same way, so one method serves both.
function Makie.convert_arguments(
        ::Union{Makie.CellGrid, Makie.VertexGrid}, uₕ::VectorElement{<:ScalarGridSpace{2}})
    (
        points(mesh(uₕ))..., to_matrix(uₕ))
end

function Makie.convert_arguments(::Makie.PointBased, ::VectorElement{<:CompositeGridSpace})
    _makie_error_composite()
end
function Makie.convert_arguments(
        ::Union{Makie.CellGrid, Makie.VertexGrid}, ::VectorElement{<:CompositeGridSpace})
    _makie_error_composite()
end

function Makie.convert_arguments(
        ::Makie.PointBased, ::VectorElement{<:ScalarGridSpace{D}}) where {D}
    _makie_error_dim(D)
end
function Makie.convert_arguments(::Union{Makie.CellGrid, Makie.VertexGrid},
        ::VectorElement{<:ScalarGridSpace{D}}) where {D}
    _makie_error_dim(D)
end

end
