# interpolation.jl
# The symbolic counterpart of interpolate_at/interpolate! (space/operators/interpolation.jl):
# wraps a grid function's interpolant as a SourceFunction, so it composes with the rest of
# the AST layer exactly the way any other source does.

"""
	πₕ(uₕ::VectorElement) -> LazyOp

The interpolant of `uₕ`, as a symbolic source term — usable anywhere a source is, including
inside another operator: `innerₕ(D₋ₓ(πₕ(uₕ)), D₋ₓ(v))` differentiates the interpolated field
the same way `D₋ₓ` differentiates any other source, `innerₕ(M₋ₓ(πₕ(uₕ)), v)` averages it,
and so on. This is what lets a coupled form read a leaf's own grid function on a *different*
leaf's mesh — the case point 24 unlocked and point 25 exists to use.

Built as `source_function(x -> interpolate_at(uₕ, x), Val(D))`: a `SourceFunction`'s own
`local_stencil` already evaluates its function at the *current* point of whichever mesh is
being walked, so nothing else needs to know `uₕ` came from another leaf at all — the
interpolation happens once per point, inside `interpolate_at`, exactly where any other
source's function call would happen.
"""
function πₕ(uₕ::VectorElement{<:ScalarGridSpace{D}}) where {D}
    return source_function(x -> interpolate_at(uₕ, x), Val(D))
end
