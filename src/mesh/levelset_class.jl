# Classification of a level-set curve φ = 0 inside a box before any grid line is built
# (Ferreira–Grigorieff grids, #834). Every horizontal and vertical line must meet the curve at
# most twice, so the vertex set of a fitted grid is closed under σᵥ (the other crossing of the
# vertical line through a point) and σₕ (the same for the horizontal line). The crossing map
# T = σₕ∘σᵥ is then a circle homeomorphism: of finite order (closure path), conjugate to an
# irrational rotation (orbit path), or locked or nearly locked (no grid fits; named error).
# Everything runs in Float64.

@noinline function _throw_levelset_touches_box()
    throw(ArgumentError("level-set grid: φ = 0 reaches the domain's bounding box, or φ < 0 on it; the curve must lie strictly inside the box."))
end

@noinline function _throw_levelset_empty()
    throw(ArgumentError("level-set grid: φ is nonnegative at every sampled point of the domain's bounding box; the region φ < 0 is empty or too small to resolve."))
end

@noinline function _throw_levelset_not_orthoconvex(axis, v, k)
    name = axis == 1 ? "x" : "y"
    throw(ArgumentError("level-set grid: the grid line $name = $(round(v; sigdigits = 4)) meets φ = 0 in $k points; the Ferreira–Grigorieff grid needs every horizontal and vertical line to meet the curve at most twice (the domain must be orthogonally convex)."))
end

@noinline function _throw_levelset_not_star(c, θ, k)
    throw(ArgumentError("level-set grid: the ray from $(round.(c; sigdigits = 4)) (the minimum of φ) at angle $(round(θ; sigdigits = 4)) meets φ = 0 in $k points; the curve must be star-shaped about that point."))
end

@noinline function _throw_levelset_locked(p, q)
    throw(ArgumentError("level-set grid: the curve's crossing map is phase-locked at rotation number $p/$q without being periodic (it has an inscribed axis-parallel polygon that attracts the others); no Cartesian grid fits this curve at every refinement. Rotate or symmetrise the domain."))
end

@noinline function _throw_levelset_nearly_locked(ρ)
    throw(ArgumentError("level-set grid: the curve's crossing map is nearly phase-locked (rotation number ≈ $(round(ρ; digits = 8)), not converged); a fitted grid would need far more lines than requested."))
end

# The other crossing of φ = 0 on the line {x_axis = p[axis]} through the crossing p; p itself
# when the line does not meet the curve twice (a tangent line).
function _levelset_σ(φ, box, axis::Int, p::NTuple{2, Float64})
    lo, hi = box[3 - axis]
    r = _crossings(φ, axis, p[axis], lo, hi)
    length(r) == 2 || return p
    o = abs(r[1] - p[3 - axis]) > abs(r[2] - p[3 - axis]) ? r[1] : r[2]
    return axis == 1 ? (p[1], o) : (o, p[2])
end

_levelset_T(φ, box, p) = _levelset_σ(φ, box, 2, _levelset_σ(φ, box, 1, p))

# Polar angle of p about c in [0, 2π).
_levelset_angle(c, p) = mod(atan(p[2] - c[2], p[1] - c[1]), 2π)

# The point of φ = 0 on the ray from c at angle θ: the ray is sampled up to the box and its
# single sign change bisected. More than one sign change: the curve is not star-shaped about c.
function _levelset_ray(φ, box, c::NTuple{2, Float64}, θ::Float64; M::Int = 129)
    e = (cos(θ), sin(θ))
    tmax = Inf
    for d in 1:2
        iszero(e[d]) && continue
        tmax = min(tmax, ((e[d] > 0 ? box[d][2] : box[d][1]) - c[d]) / e[d])
    end
    f = t -> φ((c[1] + t * e[1], c[2] + t * e[2]))
    ts = range(0.0, tmax; length = M)
    k, changes = 0, 0
    fprev = f(ts[1])
    for j in 2:M
        fj = f(ts[j])
        if (fprev < 0) != (fj < 0)
            changes += 1
            k = j
        end
        fprev = fj
    end
    changes == 1 || _throw_levelset_not_star(c, θ, changes)
    t = _bisect_root(f, ts[k - 1], ts[k], true)
    return (c[1] + t * e[1], c[2] + t * e[2])
end

# Whether T^q = id, to 1e-10 L, at every point of `pts`.
function _levelset_periodic(φ, box, pts, q::Int, L::Float64)
    for p in pts
        x = p
        for _ in 1:q
            x = _levelset_T(φ, box, x)
        end
        hypot(x[1] - p[1], x[2] - p[2]) <= 1e-10 * L || return false
    end
    return true
end

"""
    _levelset_class(φ, box) -> (path = :closure | :orbit, ρ)

Classify the curve `φ = 0` inside `box = ((x₀, x₁), (y₀, y₁))` for a Ferreira–Grigorieff grid
and return its path with the rotation number `ρ` of the crossing map `T = σₕ∘σᵥ`:

  - `:closure` when `T^q = id` for some `q ≤ 4` (any curve with one grid-aligned symmetry
    axis has `T² = id`); `ρ = p/q`.
  - `:orbit` when the weighted Birkhoff rotation number converges (8000 and 32000 steps agree
    to 1e-10) and is not `p/q` with `q ≤ 60`.

Throws a named `ArgumentError` when the curve reaches the box, a sampled grid line meets it
more than twice, it is not star-shaped about the minimum of `φ`, `T` is phase-locked
(`ρ = p/q`, `T^q ≠ id`) or nearly locked (the average does not converge).
"""
function _levelset_class(φ, box)
    B = ((Float64(box[1][1]), Float64(box[1][2])), (Float64(box[2][1]), Float64(box[2][2])))
    (x₀, x₁), (y₀, y₁) = B
    L = max(x₁ - x₀, y₁ - y₀)

    # The curve lies strictly inside the box: φ > 0 at 2001 samples of each side, and no
    # crossing found on it.
    for (axis, v, lo, hi) in ((1, x₀, y₀, y₁), (1, x₁, y₀, y₁), (2, y₀, x₀, x₁), (2, y₁, x₀, x₁))
        for t in range(lo, hi; length = 2001)
            φ(axis == 1 ? (v, t) : (t, v)) <= 0 && _throw_levelset_touches_box()
        end
        isempty(_crossings(φ, axis, v, lo, hi)) || _throw_levelset_touches_box()
    end

    # Orthogonal convexity on 2001 sampled lines per axis.
    nl = 2001
    for axis in 1:2
        a, b = B[axis]
        lo, hi = B[3 - axis]
        for i in 1:nl
            v = a + i * (b - a) / (nl + 1)
            k = length(_crossings(φ, axis, v, lo, hi))
            k > 2 && _throw_levelset_not_orthoconvex(axis, v, k)
        end
    end

    # Centre: the minimum of φ on a sample of the box; star-shaped about it.
    ns = 129
    c, fmin = (x₀, y₀), Inf
    for i in 1:ns, j in 1:ns

        p = (x₀ + (i - 1) * (x₁ - x₀) / (ns - 1), y₀ + (j - 1) * (y₁ - y₀) / (ns - 1))
        fp = φ(p)
        if fp < fmin
            c, fmin = p, fp
        end
    end
    fmin < 0 || _throw_levelset_empty()
    for k in 0:719
        _levelset_ray(φ, B, c, 2π * k / 720 + 0.001)
    end

    # Finite order: T^q = id at 9 points, q ≤ 4.
    pts = [_levelset_ray(φ, B, c, 0.3 + 2π * (k - 1) / 9) for k in 1:9]
    for q in 1:4
        _levelset_periodic(φ, B, pts, q, L) && return (path = :closure, ρ = _levelset_ρq(φ, B, c, pts[1], q))
    end

    # Weighted Birkhoff rotation number (Das–Sander–Yorke), weights exp(-1/(t(1-t))), on one
    # orbit: the first 8000 steps and all 32000.
    N₁, N₂ = 8000, 32000
    p = pts[1]
    s₁ = W₁ = s₂ = W₂ = 0.0
    for k in 1:(N₂ - 1)
        p₂ = _levelset_T(φ, B, p)
        δ = mod(_levelset_angle(c, p₂) - _levelset_angle(c, p), 2π)
        t = k / N₂
        w = exp(-1 / (t * (1 - t)))
        s₂ += w * δ
        W₂ += w
        if k < N₁
            t = k / N₁
            w = exp(-1 / (t * (1 - t)))
            s₁ += w * δ
            W₁ += w
        end
        p = p₂
    end
    ρ₁, ρ₂ = s₁ / (2π * W₁), s₂ / (2π * W₂)
    abs(ρ₁ - ρ₂) > 1e-10 && _throw_levelset_nearly_locked(ρ₂)
    for q in 1:60
        abs(q * ρ₂ - round(q * ρ₂)) < 1e-10 || continue
        _levelset_periodic(φ, B, pts, q, L) && return (path = :closure, ρ = _levelset_ρq(φ, B, c, pts[1], q))
        _throw_levelset_locked(round(Int, q * ρ₂), q)   # q is the smallest, so p/q is reduced
    end
    return (path = :orbit, ρ = ρ₂)
end

# The rotation number p/q of a T of order q, from the angle T turns p through in q steps.
function _levelset_ρq(φ, box, c, p, q::Int)
    s = 0.0
    for _ in 1:q
        p₂ = _levelset_T(φ, box, p)
        s += mod(_levelset_angle(c, p₂) - _levelset_angle(c, p), 2π)
        p = p₂
    end
    return round(s / 2π) / q
end
