# Solution-field plots via Plotly.js (CDN), no bundler, no assets wiring in make.jl —
# `@example` blocks across the worked examples `include` this rather than each redefining it.
# Mirrors `convergence_plot.jl`'s structure exactly; see `plotly_common.jl` for the shared
# CDN-loading/theming infrastructure.

# Guarded: a page that draws both a solution surface and a convergence plot includes this
# file and its sibling, and both reach for plotly_common.jl. Including it twice into one
# module redefines `plotlyjs_head` and warns about replacing its docstring.
isdefined(@__MODULE__, :plotlyjs_head) ||
    include(joinpath(@__DIR__, "..", "plotly_common.jl"))

import Bramble: point

struct SolutionPlot
    html::String
end
Base.show(io::IO, ::MIME"text/html", p::SolutionPlot) = print(io, p.html)

const _SOLUTION_PLOT_COUNTER = Ref(0)
_next_solution_plot_id() = "bramble_sp_$(_SOLUTION_PLOT_COUNTER[] += 1)"

"""
    surface_plot(uₕ; title = "", width = 480, height = 420)

A 3D surface plot of a 2D scalar grid function `uₕ`, in physical mesh coordinates (not index
space), with an orthographic camera looking straight down the `z` axis — reads as a flat
colour map at rest, and drags to tilt the field into view as elevation.
"""
function surface_plot(uₕ; title::AbstractString = "", width::Int = 480, height::Int = 420)
    Ωₕ = mesh(space(uₕ))
    nx, ny = npoints(Ωₕ, Tuple)
    xs = [point(Ωₕ(1), i) for i in 1:nx]
    ys = [point(Ωₕ(2), j) for j in 1:ny]

    # reshape(uₕ) is (nx, ny) — Plotly's z wants z[row][col] with row = y, col = x, so
    # transpose rather than reindex by hand.
    M = permutedims(reshape(uₕ))

    div_id = _next_solution_plot_id()
    x_js = "[" * join(xs, ",") * "]"
    y_js = "[" * join(ys, ",") * "]"
    z_js = "[" * join(("[" * join(row, ",") * "]" for row in eachrow(M)), ",") * "]"
    title_js = isempty(title) ? "''" : "'$title'"

    html = """
    $(plotlyjs_head())
    <div id="$div_id" style="width:100%; max-width:$(width)px; height:$(height)px; margin: 1em 0;"></div>
    <script>
    (function () {
      const theme = window.bramblePlotlyTheme();
      const data = [{
        type: 'surface',
        x: $x_js,
        y: $y_js,
        z: $z_js,
        colorscale: 'Viridis',
        showscale: true,
        colorbar: { tickfont: { color: theme.text } },
        contours: { z: { show: false } },
      }];
      const layout = {
        title: { text: $title_js, font: { color: theme.text, size: 14 } },
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        font: { color: theme.text },
        scene: {
          camera: {
            eye: { x: 0, y: 0, z: 2.1 },
            up: { x: 0, y: 1, z: 0 },
            projection: { type: 'orthographic' },
          },
          xaxis: { title: 'x', color: theme.text, gridcolor: theme.grid, backgroundcolor: theme.bg },
          yaxis: { title: 'y', color: theme.text, gridcolor: theme.grid, backgroundcolor: theme.bg },
          zaxis: { title: 'u', color: theme.text, gridcolor: theme.grid, backgroundcolor: theme.bg },
        },
        margin: { t: 40, l: 10, r: 10, b: 10 },
      };
      Plotly.newPlot('$div_id', data, layout, { displayModeBar: false, responsive: true });
      window.brambleRegisterPlotlyChart('$div_id', function () {
        const t = window.bramblePlotlyTheme();
        return {
          'font.color': t.text,
          'scene.xaxis.color': t.text, 'scene.xaxis.gridcolor': t.grid, 'scene.xaxis.backgroundcolor': t.bg,
          'scene.yaxis.color': t.text, 'scene.yaxis.gridcolor': t.grid, 'scene.yaxis.backgroundcolor': t.bg,
          'scene.zaxis.color': t.text, 'scene.zaxis.gridcolor': t.grid, 'scene.zaxis.backgroundcolor': t.bg,
          'title.font.color': t.text,
        };
      });
    })();
    </script>
    """
    return SolutionPlot(html)
end

"""
    spacetime_surface_plot(xs, ts, Z; title = "", width = 480, height = 480)

A 3D surface over `(x, t)` for a 1D time-dependent solution, `Z[j, i]` the value at
`(xs[i], ts[j])` — the whole time evolution in one static surface. A white profile curve
(`u(x, t_k)` at the current `t_k`) loops continuously along it, so the field is legible at
rest as a surface and legible in motion as a travelling cross-section, without ever
redrawing the surface itself.

`xs`/`ts` are the coordinates a caller already has (a mesh's `points`, and however many
times `ts` the solution was sampled at, e.g. `sol.(range(first(I), last(I); length = 60))`
for a SciML `sol`) — this function only lays them out and does not solve anything itself.
"""
function spacetime_surface_plot(
        xs::AbstractVector, ts::AbstractVector, Z::AbstractMatrix; title::AbstractString = "", width::Int = 480, height::Int = 480
)
    size(Z) == (length(ts), length(xs)) ||
        throw(DimensionMismatch("Z is $(size(Z)), expected (length(ts), length(xs)) = ($(length(ts)), $(length(xs)))"))

    div_id = _next_solution_plot_id()
    x_js = "[" * join(xs, ",") * "]"
    t_js = "[" * join(ts, ",") * "]"
    z_js = "[" * join(("[" * join(row, ",") * "]" for row in eachrow(Z)), ",") * "]"
    title_js = isempty(title) ? "''" : "'$title'"
    zmin, zmax = extrema(Z)

    html = """
    $(plotlyjs_head())
    <div id="$div_id" style="width:100%; max-width:$(width)px; height:$(height)px; margin: 1em 0;"></div>
    <script>
    (function () {
      const theme = window.bramblePlotlyTheme();
      // Viridis has no red anywhere in its range (purple through blue, green, yellow), so a
      // crimson curve stays legible against every point on the surface it crosses -- unlike
      // a colour picked from the map itself, which blends in exactly where the curve needs
      // to read most clearly. White cleared the surface but washed out against a light
      // page's own background (the scene's axis panes use the page background colour, and
      // the curve runs near them) -- this red is dark/saturated enough to hold contrast on
      // a light ground and bright enough to hold it on a dark one, so it stays fixed
      // regardless of theme rather than switching with the page chrome.
      const curveColor = '#e8464f';

      const xs = $x_js, ts = $t_js, zFull = $z_js;
      const lift = ($(zmax) - $(zmin)) * 0.01 || 0.01;

      function curveAt(k) {
        const yk = ts[k];
        return { x: xs, y: xs.map(() => yk), z: zFull[k].map((v) => v + lift) };
      }

      const data = [
        {
          type: 'surface',
          x: xs, y: ts, z: zFull,
          colorscale: 'Viridis',
          showscale: true,
          colorbar: { tickfont: { color: theme.text } },
          contours: { z: { show: false } },
          opacity: 0.96,
        },
        {
          type: 'scatter3d',
          mode: 'lines',
          ...curveAt(0),
          line: { color: curveColor, width: 7 },
          showlegend: false,
        },
      ];

      const frames = [];
      const sliderSteps = [];
      const tickEvery = Math.max(1, Math.round(ts.length / 6));
      for (let k = 0; k < ts.length; k++) {
        frames.push({ name: String(k), traces: [1], data: [curveAt(k)] });
        sliderSteps.push({
          method: 'animate',
          label: k % tickEvery === 0 ? ts[k].toFixed(2) : '',
          args: [[String(k)], { mode: 'immediate', frame: { duration: 0, redraw: true }, transition: { duration: 0 } }],
        });
      }

      const layout = {
        title: { text: $title_js, font: { color: theme.text, size: 14 } },
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        font: { color: theme.text },
        scene: {
          camera: { eye: { x: 1.5, y: -1.7, z: 0.85 } },
          xaxis: { title: 'x', color: theme.text, gridcolor: theme.grid, backgroundcolor: theme.bg },
          yaxis: { title: 't', color: theme.text, gridcolor: theme.grid, backgroundcolor: theme.bg },
          zaxis: { title: 'u', color: theme.text, gridcolor: theme.grid, backgroundcolor: theme.bg },
          aspectratio: { x: 1, y: 1, z: 0.6 },
        },
        margin: { t: 40, l: 10, r: 10, b: 10 },
        sliders: [{
          active: 0,
          x: 0.06, y: 0, len: 0.9,
          pad: { t: 30 },
          font: { color: theme.text, size: 10 },
          currentvalue: { prefix: 't = ', font: { color: theme.text, size: 12 }, xanchor: 'left' },
          bgcolor: theme.bg,
          activebgcolor: curveColor,
          bordercolor: theme.grid,
          tickcolor: theme.grid,
          steps: sliderSteps,
        }],
      };

      // `addFrames` is chained with `return`, not fired alongside `loopPlay`: a chart with
      // no frames registered yet can't animate them, and `addFrames` is itself async, so
      // starting `loopPlay` without waiting on it resolving first races the two.
      Plotly.newPlot('$div_id', data, layout, { displayModeBar: false, responsive: true })
        .then(function () {
          return Plotly.addFrames('$div_id', frames);
        })
        .then(function () {
          const frameNames = frames.map((f) => f.name);
          (function loopPlay() {
            Plotly.animate('$div_id', frameNames, {
              frame: { duration: 60, redraw: true },
              transition: { duration: 0 },
              mode: 'immediate',
            }).then(loopPlay);
          })();
        });

      window.brambleRegisterPlotlyChart('$div_id', function () {
        const t = window.bramblePlotlyTheme();
        return {
          'font.color': t.text,
          'scene.xaxis.color': t.text, 'scene.xaxis.gridcolor': t.grid, 'scene.xaxis.backgroundcolor': t.bg,
          'scene.yaxis.color': t.text, 'scene.yaxis.gridcolor': t.grid, 'scene.yaxis.backgroundcolor': t.bg,
          'scene.zaxis.color': t.text, 'scene.zaxis.gridcolor': t.grid, 'scene.zaxis.backgroundcolor': t.bg,
          'title.font.color': t.text,
        };
      });
    })();
    </script>
    """
    return SolutionPlot(html)
end

"""
    poisson_interactive_widget(uₕ; title = "", width = 720, height = 640) -> SolutionPlot

An interactive 2D linear Poisson panel, running entirely client-side: a resolution slider,
uniform/random mesh toggle and manufactured-solution picker drive an embedded
finite-difference solver, with a solution/error heatmap next to a matrix-sparsity or
mesh-nodes view and a table of discrete errors, degrees of freedom and a condition estimate.

`uₕ` only sets the slider's starting resolution (its mesh's point count, clamped to the
widget's `[8, 48]` range) — the panel resolves its own problem in JavaScript rather than
replaying the Julia solve, since every control on the page needs a solve of its own. Runs in
a sandboxed `iframe` (`srcdoc`, `allow-scripts` only) so its script and styling stay isolated
from the surrounding page and every other chart on it.
"""
function poisson_interactive_widget(uₕ; title::AbstractString = "", width::Int = 720, height::Int = 640)
    Ωₕ = mesh(space(uₕ))
    nx, _ = npoints(Ωₕ, Tuple)
    default_N = clamp(nx - 1, 8, 48)

    raw_html = read(joinpath(@__DIR__, "assets", "widgets", "poisson_interactive.html"), String)
    # Escape for use as a double-quoted HTML attribute value (order matters: `&` first, or
    # the ampersands introduced by escaping `"` would themselves get escaped).
    escaped = replace(raw_html, "&" => "&amp;")
    escaped = replace(escaped, "\"" => "&quot;")
    escaped = replace(escaped, "<body>" => "<body>\n<script>window.__BRAMBLE_INITIAL_N__ = $default_N;</script>")

    div_id = _next_solution_plot_id()
    title_html = isempty(title) ? "" : "<div style=\"font-weight: 500; margin-bottom: 6px;\">$title</div>"

    html = """
    $title_html
    <iframe id="$div_id" srcdoc="$escaped" width="100%" height="$height"
        style="max-width: $(width)px; border: 1px solid var(--pre-border-color, #d8d8d4); border-radius: 6px;"
        sandbox="allow-scripts" loading="lazy"></iframe>
    """
    return SolutionPlot(html)
end

"""
    coupled_reaction_diffusion_widget(uₕ, vₕ; title = "", width = 760, height = 760) -> SolutionPlot

An interactive panel posing a *different* boundary-value problem in `u`/`v` than the worked
example solves: source-free (`f1 = f2 = 0`), driven only by a constant Dirichlet supply
(`u = v = 1`) on the boundary, so every slider has a visible effect instead of being fought
back to a fixed manufactured answer. Shows synchronized `u_h`/`v_h` heatmaps with a linked
cursor, sliders for the reaction coefficients `a`, `b`, coupling `γ` and diffusion ratio
`D_u/D_v`, a `2×2` Jacobian block-sparsity spy plot (`A_uu`, `A_vv` diagonal blocks, `A_uv`,
`A_vu` pointwise coupling blocks), and a real-time 1D cross-section profile along `x` or `y`.

Like [`poisson_interactive_widget`](@ref), the panel resolves its own block Gauss-Seidel
Picard iteration in JavaScript rather than replaying the Julia solve — `uₕ`/`vₕ` only set the
slider's starting resolution (clamped to the widget's `[8, 28]` range). This boundary-value
problem has no closed-form solution, so discretization error is measured against a solve on a
fixed, much finer uniform reference mesh instead. Runs in a sandboxed `iframe` (`srcdoc`,
`allow-scripts` only).
"""
function coupled_reaction_diffusion_widget(
        uₕ, vₕ; title::AbstractString = "", width::Int = 760, height::Int = 760
)
    Ωₕ = mesh(space(uₕ))
    nx, _ = npoints(Ωₕ, Tuple)
    default_N = clamp(nx - 1, 8, 28)

    raw_html = read(joinpath(@__DIR__, "assets", "widgets", "reaction_diffusion_interactive.html"), String)
    escaped = replace(raw_html, "&" => "&amp;")
    escaped = replace(escaped, "\"" => "&quot;")
    escaped = replace(escaped, "<body>" => "<body>\n<script>window.__BRAMBLE_INITIAL_N__ = $default_N;</script>")

    div_id = _next_solution_plot_id()
    title_html = isempty(title) ? "" : "<div style=\"font-weight: 500; margin-bottom: 6px;\">$title</div>"

    html = """
    $title_html
    <iframe id="$div_id" srcdoc="$escaped" width="100%" height="$height"
        style="max-width: $(width)px; border: 1px solid var(--pre-border-color, #d8d8d4); border-radius: 6px;"
        sandbox="allow-scripts" loading="lazy"></iframe>
    """
    return SolutionPlot(html)
end

"""
    convection_diffusion_interactive_widget(uₕ; title = "", width = 760, height = 640) -> SolutionPlot

An interactive 2D linear convection-diffusion panel: sliders for the Péclet number and flow
angle, and a centered/upwind stencil switch, drive a matrix-free BiCGSTAB solve in
JavaScript, with a solution heatmap (a quiver overlay draws the constant advection
direction) next to a matrix-sparsity or mesh-nodes view, and diagnostics for the cell Péclet
number, an asymptotic boundary-layer width estimate, a matrix-asymmetry ratio, and the
discrete ``L^2`` error against a solve on a ``3\\times`` finer grid — no closed-form solution
exists for this problem (homogeneous equation, ``u = 1`` on the inflow edge and ``u = 0``
elsewhere on the boundary, chosen so the field itself moves with Pe and the flow angle
instead of always resolving to the same prescribed answer), so accuracy is measured the
other standard way. The centered stencil is second order but develops the classic grid-scale
oscillation once the cell Péclet number passes ``O(1)``; upwind stays first order and
monotone at every Péclet number.

Like [`poisson_interactive_widget`](@ref), `uₕ` only seeds the slider's starting resolution
(clamped to the widget's `[8, 48]` range) — the panel resolves its own problem in JavaScript
rather than replaying the Julia solve. Runs in a sandboxed `iframe` (`srcdoc`,
`allow-scripts` only).
"""
function convection_diffusion_interactive_widget(
        uₕ; title::AbstractString = "", width::Int = 760, height::Int = 640
)
    Ωₕ = mesh(space(uₕ))
    nx, _ = npoints(Ωₕ, Tuple)
    default_N = clamp(nx - 1, 8, 48)

    raw_html = read(joinpath(@__DIR__, "assets", "widgets", "convection_diffusion_interactive.html"), String)
    escaped = replace(raw_html, "&" => "&amp;")
    escaped = replace(escaped, "\"" => "&quot;")
    escaped = replace(escaped, "<body>" => "<body>\n<script>window.__BRAMBLE_INITIAL_N__ = $default_N;</script>")

    div_id = _next_solution_plot_id()
    title_html = isempty(title) ? "" : "<div style=\"font-weight: 500; margin-bottom: 6px;\">$title</div>"

    html = """
    $title_html
    <iframe id="$div_id" srcdoc="$escaped" width="100%" height="$height"
        style="max-width: $(width)px; border: 1px solid var(--pre-border-color, #d8d8d4); border-radius: 6px;"
        sandbox="allow-scripts" loading="lazy"></iframe>
    """
    return SolutionPlot(html)
end
