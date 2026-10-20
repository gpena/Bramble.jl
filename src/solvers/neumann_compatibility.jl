# neumann_compatibility.jl: `neumann_compatibility`, the check that the data of a pure
# Neumann problem satisfies the compatibility condition, warning on an incompatible datum.
#
# One mesh cannot tell an incompatible datum from a quadrature error, since on a coarse
# random mesh the multiplier of compatible data is as large as an O(1) incompatibility's.
# The check therefore builds the problem again on copies of the mesh refined once and
# twice at the midpoints and compares the load the multiplier removes on the two refined
# meshes. A quadrature error falls (by about 4 per halving of the spacings), an
# incompatibility stays. The coarse mesh does not decide, because its quadrature error can
# be pre-asymptotic both for compatible data (a ratio of 1.63 at a defect of 0.036) and
# over an incompatibility (λ = 0.88 against a refined -14.8 for a source shifted by -20).
#
# See also: `pde_solve` (its `mean_zero` keyword), `iterative_refinement!`

# The thresholds of the rule; the docstring of `neumann_compatibility` gives the
# measurements behind them, on random, graded and one-axis-uniform meshes.
const _NEUMANN_RATIO_WINDOW = (0.4, 2.5)
const _NEUMANN_DEFECT_FLOOR = 5e-3

"""
    neumann_compatibility(problem, Ωₕ[, domain_markers]) -> NamedTuple

Check that the data of a pure-Neumann problem are compatible, by building the problem on
the mesh `Ωₕ` and on copies of it refined once and twice at the midpoints. `problem(Ωₕ)`
returns `(A, F, w)`: the assembled matrix `A`, whose kernel holds the constants, the load
`F` and the weights `w` of the discrete average (for instance the assembled `innerₕ(1, v)`).
`Ωₕ` itself is not changed. The copies are refined by [`iterative_refinement!`](@ref): pass
the domain's `domain_markers` (`markers(Ω)`) when `Ωₕ` carries markers of its own, which
the refinement re-evaluates on the new points; without them such a mesh is refused.

On each mesh the system is solved with [`pde_solve`](@ref)`(A, F; mean_zero = w)`, and the
bordered solve's multiplier ``\\lambda = w^\\top (F - A u) / w^\\top w`` measures the part of
`F` that no `u` reaches. The signed load it removes and its relative defect are

```math
\\mu = \\lambda \\sum_i |w_i|, \\qquad \\delta = \\frac{|\\mu|}{\\sum_i |F_i|},
```

both independent of the scale of `w` (``\\delta`` is zero for zero data). For compatible
data ``\\mu`` is a quadrature error and falls under refinement, by about 4 when the
spacings halve; an incompatible datum, such as a source with a nonzero mean added to
compatible flux data, leaves a ``\\mu`` that tends to the datum's integral and keeps its
sign.

The data are reported incompatible when the defect ``\\delta_f`` on the refined mesh is
above `5e-3` and the ratio ``\\mu_f / \\mu_{ff}`` of the removed loads on the once- and
twice-refined meshes lies in ``[0.4, 2.5]``; `Ωₕ` itself does not decide, since its
quadrature error can be pre-asymptotic. The thresholds come from measurement on random,
graded and one-axis-uniform 2D meshes of the unit square and of ``[0, 1] \\times [0, 2]``,
with 9 to 129 points per axis: the ratio was negative or at least 3.39 for compatible
harmonic data, 0.99 to 1.01 for a source shifted by 0.2, and 0.64 to 1.9 for a shift of 5
or ±20 against the larger quadrature error of ``e^{5x} \\cos 5y`` fluxes (random meshes
of the square, 9 to 33 points).

The check has known limits:

- An incompatibility whose refined defect is below `5e-3`, or which is small against the
  quadrature error of the refined meshes (seen on coarse meshes of the rectangle), is not
  detected.
- Compatible data whose flux is singular at a boundary point (as for
  ``u = \\operatorname{Re} z^{\\alpha}`` near a corner, ``\\alpha < 1``) can be reported
  incompatible: their quadrature error falls like ``h^{\\alpha}``, a ratio of
  ``2^{\\alpha} < 2`` per refinement that no ratio of two or three meshes tells from an
  incompatibility.
- When ``\\delta_f`` is above `5e-3` the problem is built and solved on the twice-refined
  mesh, with 16 times the points of `Ωₕ`: at 129 points per axis that took 20.8 s and
  3.4 GiB, against 1.4 s when the floor decides.

An incompatible datum logs a warning reporting the relative defects, the multipliers and
the ratios; nothing is thrown. The result is the `NamedTuple`
`(; defect, refined_defect, λ, refined_λ, ratio, second_ratio, compatible)`, with
`ratio = μ / μ_f`, the removed loads on `Ωₕ` and its refinement (infinite when `μ_f` is
zero), `second_ratio = μ_f / μ_ff`, which decides, or `nothing` when the floor decided and
the second refinement was not built, and `compatible` the negation of the rule above.

`neumann_compatibility` is public, not exported: write `Bramble.neumann_compatibility`.

Throws an `ArgumentError` when `A`, on any of the meshes, does not annihilate the constants
(``\\| A \\mathbf{1} \\|_\\infty > 1000\\, \\varepsilon \\| A \\|_\\infty``, with
``\\varepsilon`` the machine epsilon of `A`'s element type), and the errors of
[`iterative_refinement!`](@ref) for a mesh it cannot refine.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 17), (false, false))
function problem(Ωₕ)   # -Δu = f + s with ∂ₙu = ∂ₙ(cos πx cos πy): compatible only for s = 0
    Wₕ = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0,
              :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0))
    f = x -> 2π^2 * cos(π * x[1]) * cos(π * x[2]) + 0.2
    A, F = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(-Δₕ(u), v)), form(Wₕ, v -> innerₕ(f, v)))
    w = assemble(form(gridspace(Ωₕ), v -> innerₕ(x -> 1.0, v)))
    return A, F, w
end
r = Bramble.neumann_compatibility(problem, Ωₕ)   # warns: λ ≈ 0.2 on every mesh
r.compatible                                    # false
```
"""
function neumann_compatibility(problem, Ωₕ::AbstractMeshType)
    return _neumann_compatibility(problem, Ωₕ, iterative_refinement!)
end

function neumann_compatibility(problem, Ωₕ::AbstractMeshType, domain_markers::DomainMarkers)
    return _neumann_compatibility(problem, Ωₕ, Rₕ -> iterative_refinement!(Rₕ, domain_markers))
end

function _neumann_compatibility(problem, Ωₕ, refine!)
    Rₕ = copy(Ωₕ)
    refine!(Rₕ)
    λ, μ, δ = _neumann_defect(problem(Ωₕ)...)
    λf, μf, δf = _neumann_defect(problem(Rₕ)...)
    ratio = _removed_load_ratio(μ, μf)
    second_ratio = nothing
    compatible = δf <= _NEUMANN_DEFECT_FLOOR
    if !compatible
        refine!(Rₕ)
        _, μff, _ = _neumann_defect(problem(Rₕ)...)
        second_ratio = _removed_load_ratio(μf, μff)
        compatible = !_in_ratio_window(second_ratio)
    end
    if !compatible
        @warn "neumann_compatibility: the pure-Neumann data look incompatible: the relative " *
              "defect is $δ on the mesh and $δf on its refinement, with λ = $λ and $λf; the " *
              "removed load λ Σ|w| changes by the ratio $ratio, then $second_ratio at a " *
              "second refinement, within $(_NEUMANN_RATIO_WINDOW), where a quadrature error " *
              "would fall by about 4. Check that the source integrates to the boundary flux."
    end
    return (; defect = δ, refined_defect = δf, λ, refined_λ = λf, ratio, second_ratio,
        compatible)
end

_removed_load_ratio(μ, μf) = iszero(μf) ? oftype(μ, Inf) : μ / μf

_in_ratio_window(ρ) = _NEUMANN_RATIO_WINDOW[1] <= ρ <= _NEUMANN_RATIO_WINDOW[2]

# The bordered solve's multiplier `λ`, the signed load it removes, `μ = λ Σ|w|`, which does
# not depend on the scale of `w`, and the relative defect `|μ| / Σ|F|`, zero for zero data.
function _neumann_defect(A, F, w)
    _check_annihilates_constants(A)
    u = pde_solve(A, F; mean_zero = w)
    λ = dot(w, F - A * u) / dot(w, w)
    μ = λ * sum(abs, w)
    load = sum(abs, F)
    return λ, μ, iszero(load) ? abs(μ) : abs(μ) / load
end

function _check_annihilates_constants(A)
    tol = 1000 * eps(real(float(eltype(A))))
    norm(A * ones(eltype(A), size(A, 2)), Inf) <= tol * norm(A, Inf) ||
        _throw_not_pure_neumann(A)
    return nothing
end

@noinline function _throw_not_pure_neumann(A)
    throw(
        ArgumentError(
        "neumann_compatibility needs a pure-Neumann matrix, whose kernel holds the " *
        "constants (‖A 1‖∞ ≤ 1000 eps ‖A‖∞), but the $(size(A, 1))×$(size(A, 2)) matrix " *
        "`problem` returned does not annihilate them. A problem with Dirichlet or Robin " *
        "conditions has no compatibility condition; solve it with pde_solve(A, F).",
    ),
    )
end
