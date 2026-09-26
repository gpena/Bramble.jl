# The deformed-solid plot for the 3D elasticity example: Plotly.js `mesh3d` over the six outer
# faces of a structured 3D grid, every node moved to `x + scale * u(x)`. Mirrors
# `solution_plot.jl`'s structure exactly; see `plotly_common.jl` for the shared CDN-loading and
# theming infrastructure.
#
# `mesh3d` rather than Plotly's `volume`/`isosurface`, which take a scalar field on a structured
# lattice and run marching cubes over it: a deformed solid is exactly the case that rules out,
# since every node has moved to `x + u(x)` and the grid is no longer Cartesian. A triangle list
# does not care where its vertices went.
#
# Only the faces of the drawn region are triangulated. The interior nodes stay in the vertex
# list, unused, because the triangles index the grid's own column-major linear order and
# renumbering to a surface-only list would buy a smaller payload at the cost of an index map that
# has to agree with `reshape(uₕ)` in three places.

# Guarded for the same reason `solution_plot.jl` guards it: a page that draws both a deformed
# solid and a convergence plot includes both files, and both reach for plotly_common.jl.
isdefined(@__MODULE__, :plotlyjs_head) ||
    include(joinpath(@__DIR__, "..", "plotly_common.jl"))

import Bramble: point

struct DeformedPlot
    html::String
end
Base.show(io::IO, ::MIME"text/html", p::DeformedPlot) = print(io, p.html)

const _DEFORMED_PLOT_COUNTER = Ref(0)
_next_deformed_plot_id() = "bramble_dp_$(_DEFORMED_PLOT_COUNTER[] += 1)"

# Two triangles per exposed quad. A face is drawn when the cell behind it is kept and the cell
# in front of it is not — which for `keep = all` is the outer boundary, and for a cut-away is the
# outer boundary plus the interior planes the removed cells expose. Indices follow the grid's own
# column-major linear order, 0-based as Plotly's `i`/`j`/`k` want them.
function _boundary_triangles(nx::Int, ny::Int, nz::Int, keep)
    lin(i, j, k) = (i - 1) + nx * ((j - 1) + ny * (k - 1))
    I, J, K = Int[], Int[], Int[]
    quad(a, b, c, d) = (push!(I, a); push!(J, b); push!(K, c);
        push!(I, a); push!(J, c); push!(K, d))
    kept(i, j, k) = 1 <= i <= nx - 1 && 1 <= j <= ny - 1 && 1 <= k <= nz - 1 && keep(i, j, k)

    for i in 1:(nx - 1), j in 1:(ny - 1), k in 1:(nz - 1)
        kept(i, j, k) || continue
        kept(i - 1, j, k) ||
            quad(lin(i, j, k), lin(i, j, k + 1), lin(i, j + 1, k + 1), lin(i, j + 1, k))
        kept(i + 1, j, k) ||
            quad(lin(i + 1, j, k), lin(i + 1, j + 1, k), lin(i + 1, j + 1, k + 1),
                lin(i + 1, j, k + 1))
        kept(i, j - 1, k) ||
            quad(lin(i, j, k), lin(i + 1, j, k), lin(i + 1, j, k + 1), lin(i, j, k + 1))
        kept(i, j + 1, k) ||
            quad(lin(i, j + 1, k), lin(i, j + 1, k + 1), lin(i + 1, j + 1, k + 1),
                lin(i + 1, j + 1, k))
        kept(i, j, k - 1) ||
            quad(lin(i, j, k), lin(i, j + 1, k), lin(i + 1, j + 1, k), lin(i + 1, j, k))
        kept(i, j, k + 1) ||
            quad(lin(i, j, k + 1), lin(i + 1, j, k + 1), lin(i + 1, j + 1, k + 1),
                lin(i, j + 1, k + 1))
    end
    return I, J, K
end

# Five significant digits: the payload is three coordinates and one intensity per node, and
# Documenter refuses to render a page past `size_threshold` (400 KB, docs/make.jl).
_jsnum(v) = "[" * join((string(round(x, sigdigits = 5)) for x in v), ",") * "]"
_jsint(v) = "[" * join(v, ",") * "]"
_jsnan(v) = "[" * join((isnan(x) ? "null" : string(round(x, sigdigits = 5)) for x in v), ",") * "]"

# The twelve edges of the undeformed bounding box as one polyline, `null` breaking it between
# edges so a single trace draws all twelve.
function _box_wireframe(X, Y, Z)
    x0, x1 = extrema(X)
    y0, y1 = extrema(Y)
    z0, z1 = extrema(Z)
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    wx, wy, wz = Float64[], Float64[], Float64[]
    for (a, b) in [(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5),
        (1, 5), (2, 6), (3, 7), (4, 8)]
        for c in (corners[a], corners[b])
            push!(wx, c[1])
            push!(wy, c[2])
            push!(wz, c[3])
        end
        push!(wx, NaN)
        push!(wy, NaN)
        push!(wz, NaN)
    end
    return wx, wy, wz
end

"""
    deformed_plot(uₕ, values; label = "", scale = 1.0, cut = nothing, title = "",
                  width = 560, height = 440)

The deformed solid a three-component grid function `uₕ` describes: the faces of its mesh drawn at
`x + scale * u(x)`, coloured by the nodal field `values`, over a wireframe of the undeformed box.

`scale = 1` draws the solid where the computation put it. A larger value exaggerates a
displacement too small to see at true scale, and should be stated wherever the plot appears.

`cut` takes the undeformed coordinates of a cell centre and answers whether that cell is drawn.
Removing a block of cells exposes the interior planes they bordered, which is how a field that
varies through the solid — a bending stress, say — becomes visible at all. The cut is applied in
the *undeformed* configuration, so the same cells are removed however far the solid moves.
"""
function deformed_plot(
        uₕ,
        values::AbstractVector;
        label::AbstractString = "",
        scale::Real = 1.0,
        cut = nothing,
        title::AbstractString = "",
        width::Int = 560,
        height::Int = 440
)
    Ωₕ = mesh(space(uₕ))
    nx, ny, nz = npoints(Ωₕ, Tuple)
    ux, uy, uz = (vec(reshape(c)) for c in components(uₕ))

    X, Y, Z = Float64[], Float64[], Float64[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        p = point(Ωₕ, CartesianIndex(i, j, k))
        push!(X, p[1])
        push!(Y, p[2])
        push!(Z, p[3])
    end
    ## The cut is read at the cell centre, halfway along each edge of the cell whose corners are
    ## nodes `(i,j,k)` and `(i+1,j+1,k+1)`.
    keep = if cut === nothing
        (i, j, k) -> true
    else
        (i, j, k) -> cut(
            (point(Ωₕ(1), i) + point(Ωₕ(1), i + 1)) / 2,
            (point(Ωₕ(2), j) + point(Ωₕ(2), j + 1)) / 2,
            (point(Ωₕ(3), k) + point(Ωₕ(3), k + 1)) / 2
        )
    end
    I, J, K = _boundary_triangles(nx, ny, nz, keep)
    wx, wy, wz = _box_wireframe(X, Y, Z)

    div_id = _next_deformed_plot_id()
    title_js = isempty(title) ? "''" : "'$title'"

    html = """
    $(plotlyjs_head())
    <div id="$div_id" style="width:100%; max-width:$(width)px; height:$(height)px; margin: 1em 0;"></div>
    <script>
    (function () {
      const theme = window.bramblePlotlyTheme();
      const axis = {
        color: theme.text, gridcolor: theme.grid, zerolinecolor: theme.grid,
        showbackground: false, tickfont: { size: 9 },
      };
      const data = [{
        type: 'mesh3d',
        x: $(_jsnum(X .+ scale .* ux)),
        y: $(_jsnum(Y .+ scale .* uy)),
        z: $(_jsnum(Z .+ scale .* uz)),
        i: $(_jsint(I)), j: $(_jsint(J)), k: $(_jsint(K)),
        intensity: $(_jsnum(values)),
        colorscale: 'Viridis', showscale: true, hoverinfo: 'skip',
        colorbar: { title: { text: '$label', font: { color: theme.text, size: 10 } },
                    tickfont: { color: theme.text, size: 9 }, thickness: 11, len: 0.55 },
        lighting: { ambient: 0.64, diffuse: 0.85, specular: 0.15, roughness: 0.62 },
      }, {
        type: 'scatter3d', mode: 'lines',
        x: $(_jsnan(wx)), y: $(_jsnan(wy)), z: $(_jsnan(wz)),
        line: { color: theme.grid, width: 2 }, hoverinfo: 'skip', showlegend: false,
      }];
      const themedLayout = function () {
        const t = window.bramblePlotlyTheme();
        const ax = { color: t.text, gridcolor: t.grid, zerolinecolor: t.grid,
                     showbackground: false, tickfont: { size: 9 } };
        return {
          'title.font.color': t.text,
          'scene.xaxis': Object.assign({}, ax, { title: { text: 'x', font: { size: 11 } } }),
          'scene.yaxis': Object.assign({}, ax, { title: { text: 'y', font: { size: 11 } } }),
          'scene.zaxis': Object.assign({}, ax, { title: { text: 'z', font: { size: 11 } } }),
        };
      };
      const layout = {
        title: { text: $title_js, font: { color: theme.text, size: 14 } },
        paper_bgcolor: theme.bg, plot_bgcolor: theme.bg,
        margin: { l: 0, r: 0, t: 30, b: 0 },
        scene: {
          xaxis: Object.assign({}, axis, { title: { text: 'x', font: { size: 11 } } }),
          yaxis: Object.assign({}, axis, { title: { text: 'y', font: { size: 11 } } }),
          zaxis: Object.assign({}, axis, { title: { text: 'z', font: { size: 11 } } }),
          aspectmode: 'data',
          camera: { eye: { x: -0.62, y: -3.25, z: 0.9 } },
        },
      };
      Plotly.newPlot('$div_id', data, layout, { displayModeBar: false, responsive: true });
      window.brambleRegisterPlotlyChart('$div_id', themedLayout);
    })();
    </script>
    """
    return DeformedPlot(html)
end
