module BramblePlotsExt

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

using RecipesBase: RecipesBase, @recipe
using PrecompileTools: @setup_workload, @compile_workload

@recipe function f(uₕ::VectorElement{<:ScalarGridSpace{1}})
    seriestype --> :line
    return points(mesh(uₕ)), parent(uₕ)
end

# Plots.jl's `heatmap(x, y, z)`/`surface(x, y, z)` read `z` as an image matrix: the first
# index of `z` is the row, plotted against `y`, the second is the column, plotted against
# `x`, so `size(z) == (length(y), length(x))`. `reshape(uₕ)` is `(nx, ny)`, x fastest,
# so it needs transposing to match; without it the plot would be silently rotated.
@recipe function f(uₕ::VectorElement{<:ScalarGridSpace{2}})
    seriestype --> :heatmap
    x, y = points(mesh(uₕ))
    return x, y, permutedims(reshape(uₕ))
end

@recipe function f(::VectorElement{<:CompositeGridSpace})
    Bramble._throw_plot_composite()
end

@recipe function f(::VectorElement{<:ScalarGridSpace{D}}) where {D}
    Bramble._throw_plot_dim(D)
end

# Warms `RecipesBase.apply_recipe`, the method each `@recipe` block above expands into, for
# the 1D and 2D grid-function cases -- the ones with a real plot, not the error-throwing
# fallbacks. Only reachable once `RecipesBase` is loaded, so only this extension's own
# precompile pass reaches it. `apply_recipe` needs no plotting backend or display: it just
# builds `RecipesBase.RecipeData`, the same as calling `plot(uₕ)` would before Plots.jl
# itself takes over.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        W1 = gridspace(mesh(domain(interval(0.0, 1.0)), 5, true))
        W2 = gridspace(
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
        )
        u1 = Rₕ(W1, x -> x[1])
        u2 = Rₕ(W2, x -> x[1] * x[2])

        @compile_workload begin
            RecipesBase.apply_recipe(Dict{Symbol, Any}(), u1)
            RecipesBase.apply_recipe(Dict{Symbol, Any}(), u2)
        end
    end
end

end
