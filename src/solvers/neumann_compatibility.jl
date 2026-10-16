# neumann_compatibility.jl: `neumann_compatibility`, the check that the data of a pure
# Neumann problem satisfies the compatibility condition, warning on an incompatible datum.
# Filled in by the ghost-dynamics plan (gpena/Bramble.jl#777).
#
# One mesh cannot tell an incompatible datum from a quadrature error: on a coarse random
# mesh the multiplier of compatible data is as large as an O(1) incompatibility's. The
# check therefore builds the problem again on a copy of the mesh refined at the midpoints
# and compares the two multipliers: a quadrature error falls (by about 4 per halving of
# the spacings), an incompatibility stays. When the first pair looks like an
# incompatibility, a second refinement confirms it, since a coarse mesh's quadrature error
# can be pre-asymptotic (seen with a ratio of 1.63 at a defect of 0.036).
#
# See also: `pde_solve` (its `mean_zero` keyword), `iterative_refinement!`

# The thresholds of the rule, measured on harmonic (e^{kx} cos ky, k = 5, 8) and shifted
# (tutorial problem, source + 0.2) pure-Neumann data on [0,1]² and [0,1]×[0,2], random and
# one-axis-uniform meshes, n = 9, 17, 33, seeds 1:20, and the test file's cases up to
# n = 129. The signed ratio of the multipliers of a mesh and its refinement was 0.74 to 1.24
# for the shift at both refinements; for compatible data it was 1.33, 1.63 and 1.75 on three
# meshes, negative or above 2.3 on the rest, and at the second refinement negative or at
# least 2.1 on every mesh. A refined defect at or below the floor (also rounding) is never
# reported.
const _NEUMANN_RATIO_WINDOW = (1 / 2, 2)
const _NEUMANN_DEFECT_FLOOR = 5e-3

"""
    neumann_compatibility(problem, Ωₕ[, domain_markers]) -> NamedTuple

Check that the data of a pure-Neumann problem are compatible, by comparing the problem on
the mesh `Ωₕ` with the problem on copies of `Ωₕ` refined at the midpoints. `problem(Ωₕ)`
returns `(A, F, w)`: the assembled matrix `A`, whose kernel holds the constants, the load
`F` and the weights `w` of the discrete average (for instance the assembled `innerₕ(1, v)`).
`Ωₕ` itself is not changed. The copies are refined by [`iterative_refinement!`](@ref): pass
the domain's `domain_markers` (`markers(Ω)`) when `Ωₕ` carries markers of its own, which
the refinement re-evaluates on the new points; without them such a mesh is refused.

On each mesh the system is solved with [`pde_solve`](@ref)`(A, F; mean_zero = w)`, and the
bordered solve's multiplier ``\\lambda = w^\\top (F - A u) / w^\\top w`` measures the part of
`F` that no `u` reaches. Its relative defect is

```math
\\delta = \\frac{|\\lambda| \\sum_i |w_i|}{\\sum_i |F_i|},
```

the fraction of the load the multiplier removes (zero for zero data). For compatible data
``\\lambda`` is a quadrature error and falls under refinement, by about 4 when the spacings
halve; an incompatible datum, such as a source with a nonzero mean added to compatible flux
data, leaves a ``\\lambda`` that tends to the datum's mean and keeps its sign.

The data are reported incompatible when the refined defect ``\\delta_f`` is above `5e-3`
and the signed ratio ``\\lambda / \\lambda_f`` of the multipliers on `Ωₕ` and on its
refinement lies in ``[1/2, 2]``, and, refining once more, the ratio ``\\lambda_f /
\\lambda_{ff}`` lies there too. The second refinement is built only when the first ratio is
in the window. The thresholds come from measurement on random, graded and one-axis-uniform
2D meshes of the unit square and of ``[0, 1] \\times [0, 2]``, with 9 to 129 points per
axis: the ratios of an incompatible datum were 0.74 to 1.24; those of compatible data were
negative or above 2.3 at the first refinement on all but three meshes (1.33, 1.63 and 1.75,
on meshes of 9 and 17 points) and negative or above 2.1 at the second on every mesh. An
incompatibility whose relative defect is below `5e-3` is therefore not detected.

An incompatible datum logs a warning reporting the relative defects, the multipliers and
the ratios; nothing is thrown. The result is the `NamedTuple`
`(; defect, refined_defect, λ, refined_λ, ratio, second_ratio, compatible)`, with
`ratio = λ / refined_λ` (infinite when `refined_λ` is zero), `second_ratio` the second
refinement's ratio, or `nothing` when it was not built, and `compatible` the negation of
the rule above.

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
    λ, δ = _neumann_defect(problem(Ωₕ)...)
    λf, δf = _neumann_defect(problem(Rₕ)...)
    ratio = _multiplier_ratio(λ, λf)
    second_ratio = nothing
    compatible = δf <= _NEUMANN_DEFECT_FLOOR || !_in_ratio_window(ratio)
    if !compatible
        refine!(Rₕ)
        λff, _ = _neumann_defect(problem(Rₕ)...)
        second_ratio = _multiplier_ratio(λf, λff)
        compatible = !_in_ratio_window(second_ratio)
    end
    if !compatible
        @warn "neumann_compatibility: the pure-Neumann data look incompatible: the relative " *
              "defect is $δ on the mesh and $δf on its refinement, with λ = $λ and $λf " *
              "(ratio $ratio, then $second_ratio at a second refinement, both within " *
              "$(_NEUMANN_RATIO_WINDOW)), where a quadrature error would fall by about 4. " *
              "Check that the source integrates to the boundary flux."
    end
    return (; defect = δ, refined_defect = δf, λ, refined_λ = λf, ratio, second_ratio,
        compatible)
end

_multiplier_ratio(λ, λf) = iszero(λf) ? oftype(λ, Inf) : λ / λf

_in_ratio_window(ρ) = _NEUMANN_RATIO_WINDOW[1] <= ρ <= _NEUMANN_RATIO_WINDOW[2]

# The bordered solve's multiplier `λ` and the relative defect `|λ| Σ|w| / Σ|F|`, zero for
# zero data.
function _neumann_defect(A, F, w)
    _check_annihilates_constants(A)
    u = pde_solve(A, F; mean_zero = w)
    λ = dot(w, F - A * u) / dot(w, w)
    load = sum(abs, F)
    return λ, iszero(load) ? abs(λ) : abs(λ) * sum(abs, w) / load
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
