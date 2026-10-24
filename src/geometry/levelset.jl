"""
    LevelSet{F}

Region of ``\\mathbb{R}^2`` described by a level-set function `φ`: the set where `φ < 0`,
bounded by the curve `φ = 0`.

Build it with [`levelset`](@ref). It is a marker value, not a predicate: a bare function in a
domain is a predicate marker, so the wrapper tells the two apart.

# Fields
- `φ`: the level-set function, called as `φ(x)` with `x = (x₁, x₂)`.
"""
struct LevelSet{F}
    φ::F
end

@inline (L::LevelSet)(x) = L.φ(x)

"""
    levelset(φ) -> LevelSet

Wrap the function `φ(x)`, `x = (x₁, x₂)`, as a 2D level-set marker. In a domain,
`:D => levelset(φ)` gives a 2D mesh the labels `D` (nodes where `φ < 0`), `D_boundary`
(nodes on the curve `φ = 0`) and `D_exterior` (nodes where `φ > 0`).

`φ` must return a real number, not a `Bool`. It need not be a signed distance; a mesh
samples each grid line at 129 points and bisects every sign change, so `φ` only has to
change sign across the curve. Two roots on a line closer than its sample spacing are
missed. With a `time_set`, `φ` still takes `x` alone.

# Examples
```jldoctest
using Bramble
L = levelset(p -> hypot(p[1], p[2]) - 1)
Ω = domain(interval(-1.25, 1.25) × interval(-1.25, 1.25), :D => L)
labels(Ω)

# output
(:D, :D_boundary, :D_exterior)
```

See also: [`polygon`](@ref), [`domain`](@ref).
"""
levelset(φ) = LevelSet(φ)

# Bisection for the root of f in [a, b] where f(a) < 0 differs in sign from f(b) (`neg_a`
# says whether f(a) < 0). Stops once the bracket is within eps of its initial width, about
# 53 halvings in Float64 and 24 in Float32, or when it cannot be halved, as happens first
# far from 0. Halving until then would run a root near 0 to subnormals (#893). Half-widths
# keep a bracket as wide as the type's range from overflowing.
function _bisect_root(f, a, b, neg_a::Bool)
    tol = eps(float(typeof(b - a))) * (b / 2 - a / 2)
    while b / 2 - a / 2 > tol
        m = (a + b) / 2
        (m <= a || m >= b) && break
        (f(m) < 0) == neg_a ? (a = m) : (b = m)
    end
    return (a + b) / 2
end

# All roots of φ on the line {x_axis = v}, the other coordinate in [lo, hi], sorted. φ is
# sampled at `M` equispaced points and every sign change is bisected. When φ is nonnegative
# at every sample, a golden-section search on the sampled minimum finds a near-tangent line
# that dips below zero between samples, and the two roots on either side are bisected.
# Roots closer than the sample spacing, a pair of them between two samples with the same
# sign, are missed, except for that one dip.
function _crossings(φ, axis::Integer, v::Real, lo::Real, hi::Real; M::Integer = 129)
    (axis == 1 || axis == 2) || throw(ArgumentError("levelset crossings: axis must be 1 or 2, got $axis."))
    M >= 3 || throw(ArgumentError("levelset crossings: need at least 3 samples, got $M."))
    T = float(promote_type(typeof(v), typeof(lo), typeof(hi)))
    f = axis == 1 ? t -> φ((T(v), t)) : t -> φ((t, T(v)))
    ts = range(T(lo), T(hi); length = M)
    fs = map(f, ts)
    out = T[]
    for k in 1:(M - 1)
        (fs[k] < 0) == (fs[k + 1] < 0) && continue
        push!(out, _bisect_root(f, ts[k], ts[k + 1], fs[k] < 0))
    end
    if isempty(out) && all(>=(0), fs)
        k = argmin(fs)
        a, b = ts[max(k - 1, 1)], ts[min(k + 1, M)]
        invφ = (sqrt(T(5)) - 1) / 2
        for _ in 1:100
            m1, m2 = b - invφ * (b - a), a + invφ * (b - a)
            f(m1) < f(m2) ? (b = m2) : (a = m1)
        end
        m = (a + b) / 2
        if f(m) < 0
            push!(out, _bisect_root(f, T(lo), m, false), _bisect_root(f, m, T(hi), true))
        end
    end
    return out
end

function Base.show(io::IO, ::LevelSet)
    print(io, "LevelSet(<function>)")
end
