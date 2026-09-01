module BramblePlotsExt

using Bramble: Bramble, VectorElement, ScalarGridSpace, CompositeGridSpace,
               mesh, points, to_matrix, values

using RecipesBase

# Scoped the same way as the PGFPlots exporter, and for the same reason: Plots.jl's own
# `surface`/`heatmap` plot a height field over a 2D domain, not a true 3D volume, so there
# is no faithful Plots.jl representation of a field over an actual 3D mesh. See
# `export_vtk`.
function _plots_error_composite()
    throw(ArgumentError(
        "plotting a composite element directly has no single reading — plot each of its " *
        "components(...) separately."))
end
function _plots_error_dim(D)
    throw(ArgumentError(
        "plotting a $(D)D grid function directly is not implemented — only 1D and 2D. " *
        "See export_vtk for a full 3D field."))
end

@recipe function f(uₕ::VectorElement{<:ScalarGridSpace{1}})
    seriestype --> :line
    return points(mesh(uₕ)), values(uₕ)
end

# Plots.jl's `heatmap(x, y, z)`/`surface(x, y, z)` read `z` as an image matrix: the first
# index of `z` is the row, plotted against `y`, the second is the column, plotted against
# `x` — so `size(z) == (length(y), length(x))`. `to_matrix(uₕ)` is `(nx, ny)`, x fastest,
# so it needs transposing to match; without it the plot would be silently rotated.
@recipe function f(uₕ::VectorElement{<:ScalarGridSpace{2}})
    seriestype --> :heatmap
    x, y = points(mesh(uₕ))
    return x, y, permutedims(to_matrix(uₕ))
end

@recipe function f(::VectorElement{<:CompositeGridSpace})
    _plots_error_composite()
end

@recipe function f(::VectorElement{<:ScalarGridSpace{D}}) where {D}
    _plots_error_dim(D)
end

end
