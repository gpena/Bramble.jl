# A dependency-free log-log convergence plot: inline SVG, no plotting package, no JS.
# `@example` blocks across the worked examples `include` this rather than each redefining it.
#
# Deliberately no background colour on the container — `var(--documenter-bg, ...)`, copied
# from `docs/generate_benchmarks.jl`'s own charts on a first attempt, does not resolve to
# anything in this Documenter theme and silently falls back to solid white regardless of
# light/dark mode. Every other element here uses `currentColor`, which *does* correctly track
# the surrounding page's text colour — so a white box drawn underneath them made the axes,
# reference line and labels invisible against it in dark mode, leaving only the markers
# (explicit hex colours) visible. Leaving the container transparent instead means there is no
# theme-dependent background to get wrong: everything just inherits whatever is already
# correct around it.

struct ConvergencePlot
    svg::String
end
Base.show(io::IO, ::MIME"text/html", p::ConvergencePlot) = print(io, p.svg)

"""
    convergence_plot(series; title = "", reference_slope = 2)

`series` is a vector of `(hs, errs, label, color)` tuples, one per curve — e.g. one per
spatial dimension. Points only, no connecting line: with several curves on one plot a
connecting line adds nothing a reader doesn't already get from the markers being in order
left to right, and it competes visually with the one reference line that matters. That one
dashed line has slope `reference_slope` (what the scheme promises) and is labelled with the
number, anchored through the finest point of the first series.
"""
function convergence_plot(series; title::AbstractString = "", reference_slope::Real = 2,
        width::Int = 480, height::Int = 340)
    pad_l, pad_r, pad_t, pad_b = 46, 16, (isempty(title) ? 14 : 30), 34
    legend_h = 18
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b - legend_h

    all_lx = reduce(vcat, [log2.(hs) for (hs, _, _, _) in series])
    all_ly = reduce(vcat, [log2.(errs) for (_, errs, _, _) in series])
    xmin, xmax = extrema(all_lx)
    ymin, ymax = extrema(all_ly)
    xspan = max(xmax - xmin, 1e-12)
    yspan = max(ymax - ymin, 1e-12)
    xmin -= 0.1 * xspan
    xmax += 0.1 * xspan
    ymin -= 0.1 * yspan
    ymax += 0.1 * yspan

    px(x) = pad_l + (x - xmin) / (xmax - xmin) * plot_w
    py(y) = pad_t + (ymax - y) / (ymax - ymin) * plot_h

    curves_svg = String[]
    legend_svg = String[]
    legend_y = height - legend_h + 10
    x_cursor = pad_l
    for (hs, errs, label, color) in series
        lx = log2.(hs)
        ly = log2.(errs)
        pts = [(px(lx[i]), py(ly[i])) for i in eachindex(lx)]
        circles = join(
            ("<circle cx=\"$(round(x, digits = 1))\" cy=\"$(round(y, digits = 1))\" r=\"5\" " *
             "fill=\"$color\"/>"
            for (x, y) in pts))
        push!(curves_svg, circles)

        # One row, entries laid out left to right — width estimated from character count
        # (no text-metrics access here), generous enough not to overlap for short labels.
        push!(legend_svg,
            "<circle cx=\"$(x_cursor + 4)\" cy=\"$(legend_y - 3)\" r=\"4\" fill=\"$color\"/>" *
            "<text x=\"$(x_cursor + 12)\" y=\"$(legend_y)\" font-size=\"10\" fill=\"currentColor\">" *
            "$label</text>")
        x_cursor += 16 + 6.2 * length(label) + 14
    end

    # One reference line for the whole plot, anchored through the first series' finest
    # (last) point — the number printed is the claim; the markers either sit on it or not.
    hs1, errs1, = series[1]
    x0, y0 = log2(hs1[end]), log2(errs1[end])
    rx1, ry1 = px(xmin), py(y0 + reference_slope * (xmin - x0))
    rx2, ry2 = px(xmax), py(y0 + reference_slope * (xmax - x0))
    slope_label = "slope $(isinteger(reference_slope) ? Int(reference_slope) : reference_slope)"
    mx, my = px((xmin + xmax) / 2), py(y0 + reference_slope * ((xmin + xmax) / 2 - x0))
    reference_svg = "<line x1=\"$(round(rx1, digits = 1))\" y1=\"$(round(ry1, digits = 1))\" " *
                     "x2=\"$(round(rx2, digits = 1))\" y2=\"$(round(ry2, digits = 1))\" " *
                     "stroke=\"currentColor\" stroke-opacity=\"0.55\" stroke-width=\"1.3\" stroke-dasharray=\"4,3\"/>" *
                     "<text x=\"$(round(mx, digits = 1))\" y=\"$(round(my - 6, digits = 1))\" " *
                     "font-size=\"10\" fill=\"currentColor\" fill-opacity=\"0.75\" text-anchor=\"middle\">$slope_label</text>"
    push!(curves_svg, reference_svg)

    title_svg = isempty(title) ? "" :
                "<text x=\"$(width / 2)\" y=\"14\" font-size=\"12\" font-weight=\"600\" " *
                "fill=\"currentColor\" text-anchor=\"middle\">$(title)</text>"

    svg = """
    <div style="width:100%; max-width:$(width)px; border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:0.6em; box-sizing:border-box;">
    <svg viewBox="0 0 $width $height" width="100%" xmlns="http://www.w3.org/2000/svg" style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:currentColor;">
      $title_svg
      <line x1="$pad_l" y1="$pad_t" x2="$pad_l" y2="$(height - pad_b - legend_h)" stroke="currentColor" stroke-opacity="0.35"/>
      <line x1="$pad_l" y1="$(height - pad_b - legend_h)" x2="$(width - pad_r)" y2="$(height - pad_b - legend_h)" stroke="currentColor" stroke-opacity="0.35"/>
      $(join(curves_svg))
      <text x="$(pad_l)" y="$(pad_t - 4)" font-size="10" fill="currentColor" fill-opacity="0.7" text-anchor="start">log₂(error)</text>
      <text x="$(width - pad_r)" y="$(height - pad_b - legend_h + 20)" font-size="10" fill="currentColor" fill-opacity="0.7" text-anchor="end">log₂(h) →</text>
      $(join(legend_svg))
    </svg>
    </div>
    """
    return ConvergencePlot(svg)
end
