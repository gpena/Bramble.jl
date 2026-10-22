# Hand-built oracle for the scheme of Ferreira and Grigorieff (2006) on a grid-fitted
# polygon (#204). Plain vectors and loops over the grid lines `xs`, `ys`; it calls no code
# of the package under test, so a fault there cannot cancel against the same fault here.
# It is loaded by polygon_oracle.jl, never by the package.
#
# Conventions.
#   Nodes (j, l) at (xs[j], ys[l]); unknowns are the interior nodes in column-major order
#   (x fastest), the order of the mesh's linear indices.
#   `classify` labels nodes 1 (interior), 0 (on the boundary Γ) and −1 (exterior).
#   `fg_system` is (3.1) at every interior node: −D½x(a D½x u) − D½y(c D½y u) with a and
#   c at the edge midpoints, minus the centred mixed term Dx(b Dy u) + Dy(b Dx u) with b at
#   the nodes, plus the optional convection Dx(d u) (centred, d at the nodes) and reaction
#   q u. The load is the mean of f over the dual cell (Gauss-Legendre 3×3, exact for degree
#   5), less the boundary columns times g. A diagonal neighbour p = (j + sx, l + sy) read by
#   the mixed term and exterior is replaced by (3.2) of this equation,
#   u_p = u(j + sx, l) + u(j, l + sy) − u(j, l); with `exterior = :zero` it reads 0 instead
#   (the positive control). Rows are not scaled by the dual-cell area.
module FGOracle

using SparseArrays: sparse

edges(V) = [(V[k], V[mod1(k + 1, length(V))]) for k in eachindex(V)]

function segdist(p, a, b)
    vx, vy = b[1] - a[1], b[2] - a[2]
    t = clamp(((p[1] - a[1]) * vx + (p[2] - a[2]) * vy) / (vx^2 + vy^2), 0.0, 1.0)
    return hypot(p[1] - a[1] - t * vx, p[2] - a[2] - t * vy)
end

# Crossing number: whether p is inside the polygon V (p off Γ).
function inside(p, V)
    c = false
    for (a, b) in edges(V)
        if (a[2] > p[2]) != (b[2] > p[2])
            x = a[1] + (p[2] - a[2]) * (b[1] - a[1]) / (b[2] - a[2])
            p[1] < x && (c = !c)
        end
    end
    return c
end

# 1 interior, 0 Γ (within `tol` of an edge), −1 exterior. The tolerance is absolute, a
# fraction of the polygon's size: a closure grid's crossings sit within round-off of Γ,
# and its slivers (1e-9 wide) are far wider than that.
function classify(xs, ys, V; tol = 1e-12 * max(
        maximum(first, V) - minimum(first, V), maximum(last, V) - minimum(last, V)))
    cls = zeros(Int, length(xs), length(ys))
    for j in eachindex(xs), l in eachindex(ys)

        p = (xs[j], ys[l])
        d = minimum(segdist(p, a, b) for (a, b) in edges(V))
        cls[j, l] = d <= tol ? 0 : (inside(p, V) ? 1 : -1)
    end
    return cls
end

# Bounds of the dual cell of node j on the lines x.
function halfcell(x, j)
    lo = j > 1 ? (x[j - 1] + x[j]) / 2 : x[j]
    hi = j < length(x) ? (x[j] + x[j + 1]) / 2 : x[j]
    return lo, hi
end

const GL3 = ([-sqrt(3 / 5), 0.0, sqrt(3 / 5)], [5 / 9, 8 / 9, 5 / 9])

# Mean of f over [xlo, xhi] × [ylo, yhi], Gauss-Legendre 3×3.
function cellmean(f, xlo, xhi, ylo, yhi)
    s = 0.0
    for (ξ, wξ) in zip(GL3...), (η, wη) in zip(GL3...)

        x = (xlo + xhi) / 2 + (xhi - xlo) / 2 * ξ
        y = (ylo + yhi) / 2 + (yhi - ylo) / 2 * η
        s += wξ * wη * f((x, y))
    end
    return s / 4
end

# Dual-cell areas of the interior nodes `int` (linear indices).
function dual_areas(xs, ys, int)
    C = CartesianIndices((length(xs), length(ys)))
    return map(int) do k
        j, l = Tuple(C[k])
        ((xs[j + 1] - xs[j - 1]) / 2) * ((ys[l + 1] - ys[l - 1]) / 2)
    end
end

"""
    fg_system(xs, ys, cls, a, b, c, f, g; exterior = :extend, d = nothing, q = nothing)
        -> (A, F, int)

The system (3.1) and (3.2) on the interior nodes of `cls`, `int` their linear indices.
`a`, `b`, `c`, `f`, `g`, `d`, `q` are functions of a point `(x, y)`.
"""
function fg_system(xs, ys, cls, a, b, c, f, g; exterior = :extend, d = nothing,
        q = nothing)
    exterior in (:extend, :zero) ||
        throw(ArgumentError("exterior must be :extend or :zero, got $(repr(exterior))"))
    nx, ny = length(xs), length(ys)
    lin = LinearIndices((nx, ny))
    int = [lin[j, l] for l in 1:ny for j in 1:nx if cls[j, l] == 1]
    pos = zeros(Int, nx * ny)
    pos[int] .= eachindex(int)
    I, J, Vv = Int[], Int[], Float64[]
    F = zeros(length(int))
    for (r, k) in enumerate(int)
        j, l = Tuple(CartesianIndices((nx, ny))[k])
        coeffs = Dict{Tuple{Int, Int}, Float64}()
        add!(jj, ll, v) = (coeffs[(jj, ll)] = get(coeffs, (jj, ll), 0.0) + v)
        hm, hp = xs[j] - xs[j - 1], xs[j + 1] - xs[j]
        km, kp = ys[l] - ys[l - 1], ys[l + 1] - ys[l]
        hs, ks = (hm + hp) / 2, (km + kp) / 2
        ap = a(((xs[j] + xs[j + 1]) / 2, ys[l]))
        am = a(((xs[j - 1] + xs[j]) / 2, ys[l]))
        cp = c((xs[j], (ys[l] + ys[l + 1]) / 2))
        cm = c((xs[j], (ys[l - 1] + ys[l]) / 2))
        # −D½x(a D½x u) − D½y(c D½y u)
        add!(j + 1, l, -ap / hp / hs)
        add!(j, l, (ap / hp + am / hm) / hs)
        add!(j - 1, l, -am / hm / hs)
        add!(j, l + 1, -cp / kp / ks)
        add!(j, l, (cp / kp + cm / km) / ks)
        add!(j, l - 1, -cm / km / ks)
        # −Dx(b Dy u) − Dy(b Dx u), centred: u(j + sx, l + sy) gets −sx sy (b(j + sx, l) +
        # b(j, l + sy)) / ((x_{j+1} − x_{j−1})(y_{l+1} − y_{l−1})).
        for (sx, sy) in ((1, 1), (1, -1), (-1, 1), (-1, -1))
            bx = b((xs[j + sx], ys[l]))
            by = b((xs[j], ys[l + sy]))
            v = -sx * sy * (bx + by) / ((xs[j + 1] - xs[j - 1]) * (ys[l + 1] - ys[l - 1]))
            jj, ll = j + sx, l + sy
            if cls[jj, ll] != -1
                add!(jj, ll, v)
            elseif exterior === :extend
                add!(jj, l, v)
                add!(j, ll, v)
                add!(j, l, -v)
            end
        end
        if d !== nothing   # Dx(d u), centred, d at the nodes
            w = xs[j + 1] - xs[j - 1]
            add!(j + 1, l, d((xs[j + 1], ys[l])) / w)
            add!(j - 1, l, -d((xs[j - 1], ys[l])) / w)
        end
        q === nothing || add!(j, l, q((xs[j], ys[l])))
        lo_x, hi_x = halfcell(xs, j)
        lo_y, hi_y = halfcell(ys, l)
        F[r] = cellmean(f, lo_x, hi_x, lo_y, hi_y)
        for ((jj, ll), v) in coeffs
            if cls[jj, ll] == 1
                push!(I, r)
                push!(J, pos[lin[jj, ll]])
                push!(Vv, v)
            elseif cls[jj, ll] == 0
                F[r] -= v * g((xs[jj], ys[ll]))
            else
                error("row ($j, $l) reads the exterior node ($jj, $ll) after (3.2)")
            end
        end
    end
    return sparse(I, J, Vv, length(int), length(int)), F, int
end

end # module
