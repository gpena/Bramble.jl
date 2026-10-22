"""
    Polygon{T}

Simple polygon in ``\\mathbb{R}^2``, possibly non-convex, with scalar coordinate type `T`.

Build it with `polygon`, which validates the vertices and stores them counter-clockwise.

# Fields
- `vertices`: the vertices in counter-clockwise order; the last one joins the first, and no
  vertex is repeated.
"""
struct Polygon{T}
    vertices::Vector{NTuple{2, T}}
end

@noinline _throw_polygon_error(msg) = throw(ArgumentError("Invalid polygon: " * msg))

"""
    polygon(V::AbstractVector{<:NTuple{2, Real}}) -> Polygon{T}

Construct the simple polygon whose boundary visits the vertices `V` in order and returns from
the last to the first. `V` may run clockwise or counter-clockwise; the result is stored
counter-clockwise. Coordinates are converted to the promoted floating-point type `T`.

A vertex in the middle of a straight edge is kept as a vertex.

# Throws
- `ArgumentError`: if `V` has fewer than three vertices, a coordinate is not finite, two
  consecutive vertices coincide (the last and the first included), the polygon has zero
  area, or its boundary crosses or touches itself (a bowtie, an edge folding back on the
  previous one).
"""
function polygon(V::AbstractVector{<:NTuple{2, Real}})
    T = _polygon_eltype(V)
    W = NTuple{2, T}[(T(v[1]), T(v[2])) for v in V]
    _validate_polygon(W) < 0 && reverse!(W)
    return Polygon{T}(W)
end

# The floating-point coordinate type of the vertices `V`: from the element type when it is
# concrete, from the values otherwise (a `Vector{Tuple{Real, Real}}`).
function _polygon_eltype(V::AbstractVector{<:NTuple{2, Real}})
    S = promote_type(fieldtypes(eltype(V))...)
    isconcretetype(S) && return float(S)
    return float(mapreduce(v -> promote_type(map(typeof, v)...), promote_type, V; init = Bool))
end

# Twice the signed area of the closed vertex chain `V` (shoelace formula): positive when `V`
# runs counter-clockwise.
function _signed_area2(V::Vector{NTuple{2, T}}) where {T}
    n = length(V)
    s = zero(T)
    @inbounds for k in 1:n
        a, b = V[k], V[mod1(k + 1, n)]
        s += a[1] * b[2] - b[1] * a[2]
    end
    return s
end

# Orientation of the triple (a, b, c): the cross product (b − a) × (c − a).
@inline _orient(a, b, c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])

# Whether `c`, collinear with `a` and `b`, lies in the closed bounding box of segment `ab`.
@inline function _in_box(a, b, c)
    return min(a[1], b[1]) <= c[1] <= max(a[1], b[1]) && min(a[2], b[2]) <= c[2] <= max(a[2], b[2])
end

# Whether the closed segments `ab` and `cd` share a point.
function _segments_meet(a, b, c, d)
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    (sign(o1) * sign(o2) < 0 && sign(o3) * sign(o4) < 0) && return true
    return (iszero(o1) && _in_box(a, b, c)) || (iszero(o2) && _in_box(a, b, d)) ||
           (iszero(o3) && _in_box(c, d, a)) || (iszero(o4) && _in_box(c, d, b))
end

# Validate the vertices `W` and return the sign of the enclosed area (positive when `W` runs
# counter-clockwise). Every predicate is evaluated exactly on the given floats, in
# `Rational{BigInt}`: rounding would otherwise flip the orientation of nearly collinear
# vertices (accepting an edge that folds back) or of a small polygon far from the origin.
# Validation runs once, at construction, so the cost is affordable.
function _validate_polygon(W::Vector{NTuple{2, T}}) where {T}
    n = length(W)
    n >= 3 || _throw_polygon_error("expected at least 3 vertices, got $n")
    all(v -> isfinite(v[1]) && isfinite(v[2]), W) || _throw_polygon_error("a vertex coordinate is not finite")
    V = [(Rational{BigInt}(v[1]), Rational{BigInt}(v[2])) for v in W]

    for k in 1:n
        V[k] == V[mod1(k + 1, n)] &&
            _throw_polygon_error("vertices $k and $(mod1(k + 1, n)) coincide at $(V[k]) (zero-length edge)")
    end

    # Zero area, relative to the squared extent so that a scaled or translated copy is judged
    # alike.
    L = max(maximum(v -> v[1], V) - minimum(v -> v[1], V), maximum(v -> v[2], V) - minimum(v -> v[2], V))
    A = _signed_area2(V)
    abs(A) > 8 * n * eps(T) * L^2 || _throw_polygon_error("the vertices enclose zero area")

    # Consecutive edges a → b → c meet only at b: an edge folding back overlaps the previous.
    for k in 1:n
        a, b, c = V[mod1(k - 1, n)], V[k], V[mod1(k + 1, n)]
        if iszero(_orient(a, b, c)) && (a[1] - b[1]) * (c[1] - b[1]) + (a[2] - b[2]) * (c[2] - b[2]) > 0
            _throw_polygon_error("the edges at vertex $k fold back on each other")
        end
    end

    # Edges that are not consecutive never meet.
    for i in 1:n, j in (i + 2):n

        (i == 1 && j == n) && continue
        if _segments_meet(V[i], V[i + 1], V[j], V[mod1(j + 1, n)])
            _throw_polygon_error("edges $i and $j cross or touch (the polygon is not simple)")
        end
    end
    return sign(A)
end

# Distance from the point `p` to the closed segment `ab` (`a != b`). When the foot of the
# perpendicular falls inside the segment, the distance is |(b − a) × (p − a)| / |b − a|: a
# point on the edge whose offsets from `a` are exact gives exactly zero, where the foot
# `a + t (b − a)` would leave a rounding residue.
@inline function _segment_distance(p, a, b)
    vx, vy = b[1] - a[1], b[2] - a[2]
    wx, wy = p[1] - a[1], p[2] - a[2]
    s = wx * vx + wy * vy
    s <= 0 && return hypot(wx, wy)
    s >= vx * vx + vy * vy && return hypot(p[1] - b[1], p[2] - b[2])
    return abs(vx * wy - vy * wx) / hypot(vx, vy)
end

"""
    signed_distance(P::Polygon, x) -> Real

Signed Euclidean distance from the point `x = (x₁, x₂)` to the boundary of `P`: negative
inside, zero on the boundary, positive outside.

The distance is the minimum over the edges of the exact distance from `x` to the edge; the
sign comes from the crossing number of a horizontal ray from `x`, so non-convex polygons and
reflex vertices are handled.
"""
function signed_distance(P::Polygon{T}, x) where {T}
    R = float(promote_type(T, typeof(x[1]), typeof(x[2])))
    p = (R(x[1]), R(x[2]))
    V = P.vertices
    n = length(V)
    d = typemax(R)
    inside = false
    @inbounds for k in 1:n
        a, b = V[k], V[ifelse(k == n, 1, k + 1)]
        d = min(d, _segment_distance(p, a, b))
        if (a[2] > p[2]) != (b[2] > p[2])
            xc = a[1] + (p[2] - a[2]) * (b[1] - a[1]) / (b[2] - a[2])
            p[1] < xc && (inside = !inside)
        end
    end
    return (inside && !iszero(d)) ? -d : d
end
