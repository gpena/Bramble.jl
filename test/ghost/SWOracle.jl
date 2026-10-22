# Hand-built oracle for the Shortley-Weller scheme on the level set φ < 0 (#834). Plain
# vectors and loops over the grid lines `xs`, `ys`, with its own bisection for the
# crossings; it calls no code of the package under test, so a fault there cannot cancel
# against the same fault here. It is loaded by the level-set tests, never by the package.
#
# Conventions.
#   Nodes (i, j) at (xs[i], ys[j]). The curve is φ = 0 and the domain φ < 0.
#   A crossing is the root of φ on a grid edge whose end values have opposite signs,
#   found by 80 bisection steps on the segment parameter.
#   A node is a boundary node when φ = 0 there or when a crossing lies within √ε · h of
#   it on an incident edge, h being that edge's length and ε = eps(Float64). It carries
#   no unknown and its value is g at the node itself. An interior node has φ < 0 and is
#   not a boundary node; `inside[i, j]` marks them.
#   Unknowns are the interior nodes in column-major order (x fastest).
#   `sw_system(φ, xs, ys, g)` returns `(A, b, inside)`, with `A u = f + b` for -Δu = f.
#   Each row is the pointwise scheme, not multiplied by a dual-cell area, per axis
#   -2 / (hₗ + hᵣ) [(uᵣ - uₖ) / hᵣ - (uₖ - uₗ) / hₗ].
#   An arm to a non-interior neighbour that is a boundary node keeps its full length and
#   reads g at that node. An arm to any other non-interior neighbour is shortened to the
#   crossing on that edge and reads g there. The weight times g goes into b.
#   The curve must stay off the outermost nodes, which have no neighbour beyond them.
module SWOracle

using SparseArrays: sparse

# Parameter t ∈ (0, 1) of the root of φ on p → q, where φ(p) < 0 < φ(q) or the reverse.
function crossing(φ, p, q)
    neg = φ(p) < 0
    lo, hi = 0.0, 1.0
    for _ in 1:80
        m = (lo + hi) / 2
        pt = (p[1] + m * (q[1] - p[1]), p[2] + m * (q[2] - p[2]))
        (φ(pt) < 0) == neg ? (lo = m) : (hi = m)
    end
    return (lo + hi) / 2
end

const STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))

function boundary_nodes(φ, xs, ys, vals)
    nx, ny = length(xs), length(ys)
    tol = sqrt(eps(Float64))
    bnd = falses(nx, ny)
    for j in 1:ny, i in 1:nx

        if vals[i, j] == 0
            bnd[i, j] = true
            continue
        end
        for (di, dj) in STEPS
            a, c = i + di, j + dj
            (1 <= a <= nx && 1 <= c <= ny) || continue
            vals[i, j] * vals[a, c] < 0 || continue
            h = abs(xs[a] - xs[i]) + abs(ys[c] - ys[j])
            t = crossing(φ, (xs[i], ys[j]), (xs[a], ys[c]))
            t * h <= tol * h && (bnd[i, j] = true)
        end
    end
    return bnd
end

function sw_system(φ, xs, ys, g)
    nx, ny = length(xs), length(ys)
    vals = [φ((x, y)) for x in xs, y in ys]
    bnd = boundary_nodes(φ, xs, ys, vals)
    inside = (vals .< 0) .& .!bnd
    id = zeros(Int, nx, ny)
    id[inside] = 1:count(inside)
    I, J, V = Int[], Int[], Float64[]
    b = zeros(count(inside))
    for j in 1:ny, i in 1:nx

        inside[i, j] || continue
        r = id[i, j]
        p = (xs[i], ys[j])
        for axis in 1:2
            arms = map((-1, 1)) do s
                a, c = axis == 1 ? (i + s, j) : (i, j + s)
                q = (xs[a], ys[c])
                h = abs(q[1] - p[1]) + abs(q[2] - p[2])
                inside[a, c] && return (h, id[a, c], 0.0)
                bnd[a, c] && return (h, 0, g(q))
                t = crossing(φ, p, q)
                return (t * h, 0, g((p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2]))))
            end
            w = 2 / (arms[1][1] + arms[2][1])
            for (h, col, gv) in arms
                push!(I, r)
                push!(J, r)
                push!(V, w / h)
                if col > 0
                    push!(I, r)
                    push!(J, col)
                    push!(V, -w / h)
                else
                    b[r] += w / h * gv
                end
            end
        end
    end
    n = count(inside)
    return sparse(I, J, V, n, n), b, inside
end

end # module
