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
locates each crossing of `φ = 0` along an edge by bracketing the root, so `φ` only has to
change sign across the curve. With a `time_set`, `φ` still takes `x` alone.

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

@noinline function _throw_crossing_error(a, b)
    throw(ArgumentError("levelset crossing: φ must have opposite signs at the two nodes, got $a and $b."))
end

# The parameter t ∈ [0, 1] of the root of φ on the segment p → q, for nodes where φ has
# opposite signs (either order): bisection on the bracket [lo, hi], stopped when it cannot be
# halved in the floating-point type. Bisection cannot leave the bracket, which a secant or
# Newton step can on a φ that is not a signed distance. A zero of φ at an endpoint is
# returned as 0 or 1.
function _crossing(L::LevelSet, p, q)
    T = float(promote_type(typeof(p[1]), typeof(p[2]), typeof(q[1]), typeof(q[2])))
    x1, x2 = T(p[1]), T(p[2])
    d1, d2 = T(q[1]) - x1, T(q[2]) - x2
    φ(t) = L.φ((x1 + t * d1, x2 + t * d2))
    fa, fb = φ(zero(T)), φ(one(T))
    (fa * fb > 0 || isnan(fa) || isnan(fb)) && _throw_crossing_error(fa, fb)
    iszero(fa) && return zero(T)
    iszero(fb) && return one(T)
    lo, hi = zero(T), one(T)
    sa = signbit(fa)
    while true
        mid = (lo + hi) / 2
        (mid <= lo || mid >= hi) && break
        fm = φ(mid)
        iszero(fm) && return mid
        if signbit(fm) == sa
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

function Base.show(io::IO, ::LevelSet)
    print(io, "LevelSet(<function>)")
end
