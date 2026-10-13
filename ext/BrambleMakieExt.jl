module BrambleMakieExt

using Bramble:
               Bramble,
               VectorElement,
               ScalarGridSpace,
               CompositeGridSpace,
               mesh,
               gridspace,
               points,
               domain,
               interval,
               ×,
               Rₕ

using Makie: Makie
using PrecompileTools: @setup_workload, @compile_workload

# 1D: `lines(uₕ)`, `scatter(uₕ)`. `PointBased` plot types want a vector of points, not two
# separate coordinate vectors. Makie 0.10+ requires this.
function Makie.convert_arguments(
        ::Makie.PointBased, uₕ::VectorElement{<:ScalarGridSpace{1}}
)
    return (Makie.Point2f.(points(mesh(uₕ)), parent(uₕ)),)
end

# `VectorElement <: AbstractVector`, so Makie's generic
# `expand_dimensions(::PointBased, ::AbstractVector{<:Real})` (an index axis for a bare
# numeric vector) fires before `convert_arguments` is reached, and the pipeline fails
# several layers down with `Cannot convert Point{2,Float64} to Float32`. This more
# specific method wins and hands `convert_arguments` the two coordinate vectors.
function Makie.expand_dimensions(
        ::Makie.PointBased, uₕ::VectorElement{<:ScalarGridSpace{1}}
)
    return (points(mesh(uₕ)), parent(uₕ))
end

# 2D: `heatmap(uₕ)` and `surface(uₕ)`/`contour(uₕ)` alike. Each fixes its own conversion
# trait (`Heatmap` wants `CellGrid`, `Surface` and `Contour` want `VertexGrid`), but both
# read the same three arguments the same way, so one method serves both.
function Makie.convert_arguments(
        ::Union{Makie.CellGrid, Makie.VertexGrid}, uₕ::VectorElement{<:ScalarGridSpace{2}}
)
    return (points(mesh(uₕ))..., reshape(uₕ))
end

function Makie.convert_arguments(::Makie.PointBased, ::VectorElement{<:CompositeGridSpace})
    return Bramble._throw_plot_composite()
end
function Makie.convert_arguments(
        ::Union{Makie.CellGrid, Makie.VertexGrid}, ::VectorElement{<:CompositeGridSpace}
)
    return Bramble._throw_plot_composite()
end

function Makie.convert_arguments(
        ::Makie.PointBased, ::VectorElement{<:ScalarGridSpace{D}}
) where {D}
    return Bramble._throw_plot_dim(D)
end
function Makie.convert_arguments(
        ::Union{Makie.CellGrid, Makie.VertexGrid}, ::VectorElement{<:ScalarGridSpace{D}}
) where {D}
    return Bramble._throw_plot_dim(D)
end

# The same refusals one step earlier: Makie's generic `expand_dimensions` matches any
# `VectorElement` (an `AbstractVector{<:Real}`) and would plot the DOF vector against its
# index, so `lines(u)` on a composite or 3D element never reached the refusals above. The
# 1D method above stays more specific.
function Makie.expand_dimensions(
        ::Makie.PointBased, ::VectorElement{<:CompositeGridSpace}
)
    return Bramble._throw_plot_composite()
end
function Makie.expand_dimensions(
        ::Makie.PointBased, ::VectorElement{<:ScalarGridSpace{D}}
) where {D}
    return Bramble._throw_plot_dim(D)
end

# Warms `convert_arguments`/`expand_dimensions` for the 1D and 2D grid-function cases above
# -- pure data conversion, no window or active display needed, so safe on a headless CI
# runner. Only reachable once `Makie` is loaded, so only this extension's own precompile
# pass reaches it.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        W1 = gridspace(mesh(domain(interval(0.0, 1.0)), 5, true))
        W2 = gridspace(
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
        )
        u1 = Rₕ(W1, x -> x[1])
        u2 = Rₕ(W2, x -> x[1] * x[2])

        @compile_workload begin
            Makie.convert_arguments(Makie.PointBased(), u1)
            Makie.expand_dimensions(Makie.PointBased(), u1)
            Makie.convert_arguments(Makie.CellGrid(), u2)
        end
    end
end

end
