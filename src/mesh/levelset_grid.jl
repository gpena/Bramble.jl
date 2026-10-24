# Fitted grids for a level-set curve φ = 0 (Ferreira and Grigorieff, #834): grid lines whose
# nodes include the vertices of a polygon inscribed in the curve, every polygon edge a cell
# diagonal or a cell side, so that the paper's (Geom) holds. `_levelset_class` picks the path:
# a crossing map T = σₕ∘σᵥ of finite order closes the seeds (`_grid_closure`); a T conjugate
# to an irrational rotation gets periodic orbits of T closed by least squares. Both return
# the lines and the inscribed polygon, whose labels and `polygon_spaces` are reused. The
# construction runs in Float64; lines and polygon are converted to the seeds' type and
# checked in it.

@noinline function _throw_levelset_too_many(cap, nin, ρ, cmax)
    throw(ArgumentError("level-set grid: too many lines: no N ≤ $cap lines per axis inside the curve's extent has N‖Nρ‖ ≤ $cmax (rotation number $(round(ρ; digits = 8))) for the $nin lines asked there."))
end

@noinline function _throw_levelset_too_coarse(nx, ny, nin)
    throw(ArgumentError("level-set grid: too coarse: $nx × $ny points give $nin lines inside the curve's extent, and the fit needs at least 6; use more."))
end

@noinline function _throw_levelset_check_failed(reason)
    throw(ArgumentError("level-set grid: check failed: the fitted grid does not fit the level set ($reason), so the curve would cross a grid line off the nodes. Use more points per axis."))
end

"""
    _levelset_grid(φ, xs, ys) -> (X::Vector{T}, Y::Vector{T}, P::Polygon{T})

Fit grid lines to the curve `φ = 0` from the seed lines `xs` and `ys` (end points included,
which bound the box) and return them with the polygon inscribed in the curve whose vertices
are the boundary nodes. `T` is the floating-point type of the seeds.

  - Closure path (`T` of finite order): `_grid_closure` closes the seeds; the vertices are
    the crossings, each on a node. Every seed is kept.
  - Orbit path (`T` conjugate to an irrational rotation): `_levelset_orbit_grid`.

The returned grid passes the post-check `_levelset_fit`: consecutive vertices are corners
of one cell, every other node's `φ` sign agrees with its polygon label, no grid edge joins
an interior and an exterior node, and every vertex is within `h²/(4ℓ)` of `φ = 0`, `ℓ`
half the larger extent of the vertices.

# Throws

  - `ArgumentError`: the errors of `_levelset_class` and of `_levelset_orbit_grid`, or, on
    the closure path, "check failed" or `_grid_closure`'s cap error.
"""
function _levelset_grid(φ, xs::AbstractVector, ys::AbstractVector)
    B = _levelset_box(xs, ys)
    path, ρ = _levelset_class(φ, B)
    path === :orbit && return _levelset_orbit_grid(φ, xs, ys, ρ)
    T = float(promote_type(eltype(xs), eltype(ys)))
    X, Y = _grid_closure(LevelSet(φ), xs, ys)
    c, _ = _levelset_centre(φ, B)
    V = NTuple{2, T}[]
    for (a, L, M) in ((1, X, Y), (2, Y, X)), v in L

        for w in _crossings(φ, a, v, M[1], M[end])
            s = _levelset_snap(M, w)
            push!(V, a == 1 ? (v, s) : (s, v))
        end
    end
    r = _levelset_fit(φ, c, X, Y, unique!(V))
    r isa String && _throw_levelset_check_failed(r)
    return r
end

# The box ((x₀, x₁), (y₀, y₁)) of the seed lines, in Float64.
function _levelset_box(xs, ys)
    return ((Float64(minimum(xs)), Float64(maximum(xs))), (Float64(minimum(ys)), Float64(maximum(ys))))
end

# The line of the sorted lines L nearest to w.
function _levelset_snap(L::AbstractVector, w)
    i = searchsortedfirst(L, w)
    i > length(L) && return L[end]
    i == 1 && return L[1]
    return w - L[i - 1] < L[i] - w ? L[i - 1] : L[i]
end

"""
    _levelset_orbit_grid(φ, xs, ys, ρ; cmax = 2.0) -> (X::Vector{T}, Y::Vector{T}, P::Polygon{T})

The orbit-path grid of the curve `φ = 0` whose crossing map `T = σₕ∘σᵥ` has rotation number
`ρ`. With `nin` the larger of the two axes' seed counts inside the curve's extent (in
proportion; the larger, so that no axis is coarser than asked), the number
`N` of lines per axis inside the extent minimises the cost ``N\\|N\\rho\\|`` (which predicts
the vertices' distance to the curve over ``h^2``) in ``[nin/1.25, 1.25 nin]``, or is the
least `N` up to `4 nin` with cost at most `cmax`. With ``p`` the nearest integer to
``N\\rho``, ``m = \\gcd(N, p)`` orbits of period ``q = N/m`` start at phases ``1/(2N)``
apart in the conjugacy of `T` to the rotation, a quarter of that off the leftmost point;
Levenberg-Marquardt closes each orbit on the vertical lines alone (`_levelset_close!`), its
anchor scaled by the curve's length scale `ℓ` (half its larger extent), so that a scaled
curve with scaled seeds gives the scaled grid. The distinct seeds farther than a quarter of
their spacing from the curve's extent are kept; the others give way. A grid that fails the
post-check is rebuilt once with the next-best `N`.

A rational `ρ = p/q` (reached only by tests, to build a grid that fails the post-check)
samples the conjugacy from orbits started at equally spaced angles.

# Throws

  - `ArgumentError`: "too coarse" when fewer than 6 lines are asked inside the extent, "too
    many lines" when no `N` up to `4 nin` costs at most `cmax`, "check failed" when both
    grids fail the post-check.
"""
function _levelset_orbit_grid(φ, xs::AbstractVector, ys::AbstractVector, ρ::Real; cmax::Real = 2.0)
    B = _levelset_box(xs, ys)
    sx, sy = unique!(sort!(Float64.(collect(xs)))), unique!(sort!(Float64.(collect(ys))))
    c, _ = _levelset_centre(φ, B)
    θₗ, ex, ey = _levelset_extent(φ, B, c)
    nin = round(Int, max(length(sx) * (ex[2] - ex[1]) / (B[1][2] - B[1][1]), length(sy) * (ey[2] - ey[1]) /
                                                                             (B[2][2] - B[2][1])))
    nin < 6 && _throw_levelset_too_coarse(length(sx), length(sy), nin)
    ℓ = max(ex[2] - ex[1], ey[2] - ey[1]) / 2
    N = _levelset_N(Float64(ρ), nin, Float64(cmax), 0)
    iszero(N) && _throw_levelset_too_many(4nin, nin, ρ, cmax)
    kx, ky = _levelset_kept(sx, ex), _levelset_kept(sy, ey)
    T = float(promote_type(eltype(xs), eltype(ys)))
    r = ""
    for _ in 1:2
        X, Y, V = _levelset_orbits(φ, B, c, θₗ, Float64(ρ), N, (ex[2] - ex[1]) / ℓ)
        r = _levelset_fit(φ, c, sort!([X; kx]), sort!([Y; ky]), V, T)
        r isa String || return r
        N = _levelset_N(Float64(ρ), nin, Float64(cmax), N)
        iszero(N) && break
    end
    return _throw_levelset_check_failed(r)
end

# The seeds farther than a quarter of their mean spacing from the extent e = (lo, hi), and
# the end points, which bound the box.
function _levelset_kept(s::Vector{Float64}, e::NTuple{2, Float64})
    d = (s[end] - s[1]) / (length(s) - 1)
    return [v for (k, v) in enumerate(s) if k == 1 || k == length(s) || v < e[1] - d / 4 || v > e[2] + d / 4]
end

# The cost N‖Nρ‖, or Inf when the nearest integer to Nρ is 0 or N (a period-1 orbit).
function _levelset_cost(ρ::Float64, N::Int)
    p = round(Int, N * ρ)
    (p <= 0 || p >= N) && return Inf
    return N * abs(N * ρ - p)
end

# The number of lines inside the extent for nin asked, other than `skip`: the least weighted
# cost in [nin/1.25, 1.25 nin] when it is at most cmax, otherwise the least N up to 4 nin
# whose cost is; 0 when there is none.
function _levelset_N(ρ::Float64, nin::Int, cmax::Float64, skip::Int)
    lo, hi = max(3, floor(Int, nin / 1.25)), ceil(Int, 1.25 * nin)
    best, wbest = 0, Inf
    for N in lo:hi
        N == skip && continue
        w = _levelset_cost(ρ, N) * (1 + 0.05 * abs(log(N / nin)))
        w < wbest && ((best, wbest) = (N, w))
    end
    best > 0 && _levelset_cost(ρ, best) <= cmax && return best
    for N in (hi + 1):(4nin)
        N != skip && _levelset_cost(ρ, N) <= cmax && return N
    end
    return 0
end

# The angle about c of the curve's leftmost point and the curve's x- and y-extents: 2001
# sampled angles, each extreme refined by golden section.
function _levelset_extent(φ, B, c::NTuple{2, Float64})
    M = 2001
    θs = [2π * (k - 1) / M for k in 1:M]
    P = [_levelset_ray(φ, B, c, θ) for θ in θs]
    function extreme(d, sgn)
        k = argmin(k -> sgn * P[k][d], 1:M)
        a, b = θs[k] - 2π / M, θs[k] + 2π / M
        f = θ -> sgn * _levelset_ray(φ, B, c, θ)[d]
        invφ = (sqrt(5.0) - 1) / 2
        for _ in 1:100
            m1, m2 = b - invφ * (b - a), a + invφ * (b - a)
            f(m1) < f(m2) ? (b = m2) : (a = m1)
        end
        θ = (a + b) / 2
        return θ, sgn * min(f(θ), sgn * P[k][d])
    end
    θₗ, x₀ = extreme(1, 1.0)
    return θₗ, (x₀, extreme(1, -1.0)[2]), (extreme(2, 1.0)[2], extreme(2, -1.0)[2])
end

# The crossings (lower, upper) of the vertical line x with φ = 0, or NaNs when there are not two.
function _levelset_vcross(φ, B, x::Float64)
    r = _crossings(φ, 1, x, B[2][1], B[2][2])
    length(r) == 2 || return (NaN, NaN)
    return (r[1], r[2])
end

# Sorted polar angles of K points of T-orbits: one orbit, or, when ρ = p/q with q ≤ 60, K/q
# orbits of length q started at equally spaced angles. Their empirical distribution is the
# conjugacy H of T to the rotation by ρ.
function _levelset_conjugacy(φ, B, c::NTuple{2, Float64}, ρ::Float64, K::Int)
    q = something(findfirst(q -> abs(q * ρ - round(q * ρ)) < 1e-12, 1:60), K)
    s = K ÷ q
    θs = Float64[]
    sizehint!(θs, s * q)
    for j in 1:s
        p = _levelset_ray(φ, B, c, 0.123 + 2π * (j - 1) / s)
        for _ in 1:q
            push!(θs, _levelset_angle(c, p))
            p = _levelset_T(φ, B, p)
        end
    end
    return sort!(θs)
end

_levelset_H(θs::Vector{Float64}, θ) = (searchsortedfirst(θs, mod(θ, 2π)) - 1) / length(θs)

function _levelset_Hinv(θs::Vector{Float64}, ψ)
    K = length(θs)
    t = mod(ψ, 1.0) * K
    k = floor(Int, t)
    a = k == 0 ? θs[end] - 2π : θs[k]
    b = k >= K ? θs[1] + 2π : θs[k + 1]
    return a + (t - k) * (b - a)
end

# The vertical lines X, horizontal lines Y and vertices V of N = m q lines per axis inside
# the extent (width w in units of the curve's length scale): m periodic orbits of T, each
# closed by `_levelset_close!`.
function _levelset_orbits(φ, B, c, θₗ::Float64, ρ::Float64, N::Int, w::Float64)
    m = gcd(N, round(Int, N * ρ))
    q = N ÷ m
    p = round(Int, N * ρ) ÷ m
    θs = _levelset_conjugacy(φ, B, c, ρ, max(20_000, 400N))
    ψₑ = _levelset_H(θs, θₗ)
    X, Y, V = Float64[], Float64[], NTuple{2, Float64}[]
    for j in 0:(m - 1)
        ψ₀ = ψₑ - 1 / (4N) + j / N
        Xj = zeros(q)
        inb = zeros(Int, q)
        for k in 1:q
            v = _levelset_ray(φ, B, c, _levelset_Hinv(θs, ψ₀ + (k - 1) * p / q))
            r = _levelset_vcross(φ, B, v[1])
            Xj[k] = v[1]
            inb[k] = abs(r[1] - v[2]) <= abs(r[2] - v[2]) ? 1 : 2
        end
        _levelset_close!(φ, B, Xj, inb, 0.5 * w / N)
        Yj = zeros(q)
        for k in 1:q
            k₂ = mod1(k + 1, q)
            Yj[k] = (_levelset_vcross(φ, B, Xj[k])[3 - inb[k]] + _levelset_vcross(φ, B, Xj[k₂])[inb[k₂]]) / 2
        end
        for k in 1:q
            push!(V, (Xj[k], Yj[mod1(k - 1, q)]), (Xj[k], Yj[k]))
        end
        append!(X, Xj)
        append!(Y, Yj)
    end
    return X, Y, V
end

# Levenberg-Marquardt on the q vertical lines X of one orbit: vertex k is the crossing of
# branch inb[k] (1 lower, 2 upper) on X[k] and enters σᵥ exactly; the residuals are the
# horizontal mismatches r_k = y_out(X_k) − y_in(X_{k+1}) (cyclic) and the anchor λ(X − X₀),
# which keeps the orbits apart near the ghost of a periodic orbit. A step that shrinks a
# gap X_{k+1} − X_k below a quarter of its start is rejected: it heads for the degenerate
# zero where a horizontal line pairs a vertex with itself. The normal matrix is cyclic
# tridiagonal (`_levelset_cyclic_solve!`). Stops when a step gains less than 1e-12 of the
# sum of squares, or after 60 steps.
function _levelset_close!(φ, B, X::Vector{Float64}, inb::Vector{Int}, λ::Float64)
    q = length(X)
    X₀ = copy(X)
    δfd = 1e-6 * max(B[1][2] - B[1][1], B[2][2] - B[2][1])
    gap₀ = [X[mod1(k + 1, q)] - X[k] for k in 1:q]
    ycr = [_levelset_vcross(φ, B, x) for x in X]
    function sumsq(X, ycr)
        s = 0.0
        for k in 1:q
            k₂ = mod1(k + 1, q)
            s += (ycr[k][3 - inb[k]] - ycr[k₂][inb[k₂]])^2 + (λ * (X[k] - X₀[k]))^2
        end
        return s
    end
    function slope(x, y)
        gx = (φ((x + δfd, y)) - φ((x - δfd, y))) / 2δfd
        gy = (φ((x, y + δfd)) - φ((x, y - δfd))) / 2δfd
        return -gx / gy
    end
    S = sumsq(X, ycr)
    a, b, g, d, e, δ, X₂ = (zeros(q) for _ in 1:7)
    work = zeros(4q)
    μ = 1e-8
    for _ in 1:60
        isfinite(S) || break
        for k in 1:q
            a[k] = slope(X[k], ycr[k][3 - inb[k]])
            b[k] = slope(X[k], ycr[k][inb[k]])
        end
        for j in 1:q
            j₋, j₊ = mod1(j - 1, q), mod1(j + 1, q)
            rⱼ = ycr[j][3 - inb[j]] - ycr[j₊][inb[j₊]]
            r₋ = ycr[j₋][3 - inb[j₋]] - ycr[j][inb[j]]
            g[j] = a[j] * rⱼ - b[j] * r₋ + λ^2 * (X[j] - X₀[j])
            e[j] = -a[j] * b[j₊]
        end
        improved = false
        for _ in 1:40
            for j in 1:q
                d[j] = a[j]^2 + b[j]^2 + λ^2 + μ
            end
            _levelset_cyclic_solve!(δ, d, e, g, work)
            @. X₂ = X - δ
            if all(k -> (X₂[mod1(k + 1, q)] - X₂[k]) / gap₀[k] > 0.25, 1:q)
                ycr₂ = [_levelset_vcross(φ, B, x) for x in X₂]
                S₂ = sumsq(X₂, ycr₂)
                if S₂ < S
                    improved = S - S₂ > 1e-12 * S
                    X .= X₂
                    ycr, S = ycr₂, S₂
                    μ = max(μ / 10, 1e-16)
                    break
                end
            end
            μ *= 10
        end
        improved || break
    end
    return X
end

# Solve A x = r for the symmetric cyclic tridiagonal A with diagonal d and A[j, j+1] = e[j]
# (indices mod q, q ≥ 2; for q = 2 the two couplings add): eliminate the last unknown and
# solve the tridiagonal rest twice (Thomas). `work` holds 4q entries.
function _levelset_cyclic_solve!(x, d, e, r, work)
    q = length(d)
    n = q - 1
    y, z, cc, cp = view(work, 1:n), view(work, (q + 1):(q + n)), view(work, (2q + 1):(2q + n)),
    view(work, (3q + 1):(3q + n))
    cc .= 0
    cc[n] += e[n]
    cc[1] += e[q]
    # Thomas: factor once, apply to r[1:n] (into y) and to cc (into z).
    y[1], z[1] = r[1], cc[1]
    piv = d[1]
    cp[1] = n > 1 ? e[1] / piv : 0.0
    y[1] /= piv
    z[1] /= piv
    for j in 2:n
        piv = d[j] - e[j - 1] * cp[j - 1]
        cp[j] = j < n ? e[j] / piv : 0.0
        y[j] = (r[j] - e[j - 1] * y[j - 1]) / piv
        z[j] = (cc[j] - e[j - 1] * z[j - 1]) / piv
    end
    for j in (n - 1):-1:1
        y[j] -= cp[j] * y[j + 1]
        z[j] -= cp[j] * z[j + 1]
    end
    s, t = 0.0, 0.0
    for j in 1:n
        s += cc[j] * y[j]
        t += cc[j] * z[j]
    end
    x[q] = (r[q] - s) / (d[q] - t)
    for j in 1:n
        x[j] = y[j] - z[j] * x[q]
    end
    return x
end

"""
    _levelset_fit(φ, c, X, Y, V[, T]) -> (X::Vector{T}, Y::Vector{T}, P::Polygon{T}) or String

Convert the lines `X`, `Y` and the vertices `V` (each on a node) to `T` (default: the lines'
type), order the vertices by angle about the interior point `c` into a counter-clockwise
polygon, and check the grid fits it (architect §3.3 step 9). Returns the reason as a
`String` when it does not:

  - the lines are not strictly increasing in `T`, or a vertex is not a node;
  - two consecutive vertices are not corners of one cell;
  - a node other than a vertex has a polygon label that disagrees with the sign of `φ`, or
    a grid edge joins an interior and an exterior node;
  - a vertex is farther than ``h^2/(4\\ell)`` from `φ = 0` (``|\\varphi|/|\\nabla\\varphi|``),
    ``h`` the largest spacing between the lines within the vertices' extent and ``\\ell``
    half the larger side of that extent: ``h^2/4`` in units of ``\\ell``.

The polygon is built without `polygon`'s validation (quadratic in the vertices): the check
certifies it.
"""
function _levelset_fit(φ, c::NTuple{2, Float64}, X::AbstractVector, Y::AbstractVector, V::AbstractVector)
    return _levelset_fit(φ, c, X, Y, V, float(promote_type(eltype(X), eltype(Y))))
end

# `T` is positional so that it specialises: as a keyword it is a `DataType` (#892).
function _levelset_fit(φ, c::NTuple{2, Float64}, X::AbstractVector, Y::AbstractVector, V::AbstractVector,
        ::Type{T}) where {T}
    XT, YT = T.(X), T.(Y)
    W = NTuple{2, T}[(T(v[1]), T(v[2])) for v in V]
    all(isfinite, XT) && all(isfinite, YT) && all(v -> isfinite(v[1]) && isfinite(v[2]), W) ||
        return "a line is not finite"
    (all(>(0), diff(XT)) && all(>(0), diff(YT))) || return "two lines coincide"
    length(W) >= 3 || return "fewer than three vertices"
    sort!(W; by = v -> _levelset_angle(c, (Float64(v[1]), Float64(v[2]))))
    nx, ny = length(XT), length(YT)
    ij = Vector{NTuple{2, Int}}(undef, length(W))
    for (k, v) in enumerate(W)
        i, j = searchsortedfirst(XT, v[1]), searchsortedfirst(YT, v[2])
        (i <= nx && j <= ny && XT[i] == v[1] && YT[j] == v[2]) || return "a vertex is not a node"
        ij[k] = (i, j)
    end
    # Labels: 0 at a vertex, -1 inside the polygon, 1 outside. A vertical ray up from a node
    # crosses the edges whose ends straddle the node's line under the half-open rule
    # (one end at x > X[i]); each such edge has its other end on column i.
    lbl = ones(Int8, nx, ny)
    tog = falses(nx, ny)
    nv = length(ij)
    for k in 1:nv
        (i, j), (i₂, j₂) = ij[k], ij[mod1(k + 1, nv)]
        (abs(i - i₂) <= 1 && abs(j - j₂) <= 1 && (i, j) != (i₂, j₂)) ||
            return "vertices $(W[k]) and $(W[mod1(k + 1, nv)]) are consecutive but not corners of one cell"
        if i != i₂
            iₑ, jₑ = i < i₂ ? (i, j) : (i₂, j₂)
            tog[iₑ, jₑ] = !tog[iₑ, jₑ]
        end
    end
    for i in 1:nx
        s = false
        for j in ny:-1:1
            lbl[i, j] = s ? -1 : 1
            s ⊻= tog[i, j]
        end
    end
    for (i, j) in ij
        lbl[i, j] = 0
    end
    for j in 1:ny, i in 1:nx

        l = lbl[i, j]
        iszero(l) && continue
        f = φ((XT[i], YT[j]))
        (f < 0) == (l < 0) && !iszero(f) ||
            return "the node ($(XT[i]), $(YT[j])) has polygon label $(l < 0 ? "interior" : "exterior") but φ = $f"
        i < nx && l * lbl[i + 1, j] < 0 && return "a grid edge joins an interior and an exterior node"
        j < ny && l * lbl[i, j + 1] < 0 && return "a grid edge joins an interior and an exterior node"
    end
    il, ih = extrema(first, ij)
    jl, jh = extrema(last, ij)
    h = max(il < ih ? maximum(diff(XT[il:ih])) : zero(T), jl < jh ? maximum(diff(YT[jl:jh])) : zero(T))
    ℓ = Float64(max(XT[ih] - XT[il], YT[jh] - YT[jl])) / 2
    δfd = 1e-6 * Float64(max(XT[end] - XT[1], YT[end] - YT[1]))
    for v in W
        x, y = Float64(v[1]), Float64(v[2])
        gx = (φ((x + δfd, y)) - φ((x - δfd, y))) / 2δfd
        gy = (φ((x, y + δfd)) - φ((x, y - δfd))) / 2δfd
        dist = abs(φ((x, y))) / hypot(gx, gy)
        dist * ℓ <= Float64(h)^2 / 4 ||
            return "the vertex $v is $(round(dist * ℓ / h^2; sigdigits = 3)) h²/ℓ from φ = 0, above h²/(4ℓ)"
    end
    return XT, YT, Polygon{T}(W)
end
