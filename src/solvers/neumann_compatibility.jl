# neumann_compatibility.jl: `neumann_compatibility`, the check that the data of a pure
# Neumann problem satisfies the compatibility condition, warning on an incompatible datum.
# Filled in by the ghost-dynamics plan (gpena/Bramble.jl#777).
#
# One mesh cannot tell an incompatible datum from a quadrature error: on a coarse random
# mesh the multiplier of compatible data is as large as an O(1) incompatibility's. The
# check therefore builds the problem again on a copy of the mesh refined at the midpoints
# and compares the two defects: a quadrature error falls (by about 4 per halving of the
# spacings), an incompatibility stays.
#
# See also: `pde_solve` (its `mean_zero` keyword), `iterative_refinement!`

# The thresholds of the rule, measured on the harmonic and shifted pure-Neumann cases of
# `test/solvers/neumann_compatibility.jl` (random, x²-graded and one-axis-uniform meshes,
# n = 9 to 129, 25 random seeds at n ≤ 33): a refinement divides a compatible defect by 3.1
# to 19 (once by 1.31 and once by 0.77, both at a defect below 2e-3), and an incompatible
# one by 0.82 to 1.18. A defect at or below the floor (also rounding) is never reported.
const _NEUMANN_DEFECT_RATIO = 2
const _NEUMANN_DEFECT_FLOOR = 5e-3

"""
    neumann_compatibility(problem, Ωₕ) -> NamedTuple

Check that the data of a pure-Neumann problem are compatible, by comparing the problem on
the mesh `Ωₕ` with the problem on a copy of `Ωₕ` refined at the midpoints. `problem(Ωₕ)`
returns `(A, F, w)`: the assembled matrix `A`, whose kernel holds the constants, the load
`F` and the weights `w` of the discrete average (for instance the assembled `innerₕ(1, v)`).
`Ωₕ` itself is not changed.

On each mesh the system is solved with [`pde_solve`](@ref)`(A, F; mean_zero = w)`, and the
bordered solve's multiplier ``\\lambda = w^\\top (F - A u) / w^\\top w`` measures the part of
`F` that no `u` reaches. Its relative defect is

```math
\\delta = \\frac{|\\lambda| \\sum_i |w_i|}{\\sum_i |F_i|},
```

the fraction of the load the multiplier removes. For compatible data ``\\delta`` is a
quadrature error and falls under refinement, by about 4 when the spacings halve; an
incompatible datum, such as a source with a nonzero mean added to compatible flux data,
leaves a ``\\lambda`` that tends to the datum's mean and a ``\\delta`` that stays. The data
are reported incompatible when the refined defect ``\\delta_f`` is above `5e-3` and the
ratio ``\\delta / \\delta_f`` is below 2. Both thresholds come from measurement on random,
graded and one-axis-uniform 2D meshes with 9 to 129 points per axis: a compatible defect
fell by a factor 3.1 to 19 (below 2 only for defects under `2e-3`, on meshes of 9 and 17
points), an incompatible one by 0.8 to 1.2. An incompatibility whose relative defect is
below `5e-3` is therefore not detected.

An incompatible datum logs a warning reporting both relative defects, ``\\lambda`` and the
ratio; nothing is thrown. The result is the `NamedTuple`
`(; defect, refined_defect, λ, refined_λ, ratio, compatible)`, with `compatible` the
negation of the rule above.

`neumann_compatibility` is public, not exported: write `Bramble.neumann_compatibility`.

Throws an `ArgumentError` when `A`, on either mesh, does not annihilate the constants
(``\\| A \\mathbf{1} \\|_\\infty > 10^{-10} \\| A \\|_\\infty``), and the errors of
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
r = Bramble.neumann_compatibility(problem, Ωₕ)   # warns: λ ≈ 0.2 on both meshes
r.compatible                                    # false
```
"""
function neumann_compatibility(problem, Ωₕ::AbstractMeshType)
    Rₕ = copy(Ωₕ)
    iterative_refinement!(Rₕ)
    λ, δ = _neumann_defect(problem(Ωₕ)...)
    λf, δf = _neumann_defect(problem(Rₕ)...)
    ratio = iszero(δf) ? oftype(δf, Inf) : δ / δf
    compatible = δf <= _NEUMANN_DEFECT_FLOOR || ratio >= _NEUMANN_DEFECT_RATIO
    if !compatible
        @warn "neumann_compatibility: the pure-Neumann data look incompatible: the relative " *
              "defect is $δ on the mesh and $δf on its refinement (ratio $ratio, below " *
              "$(_NEUMANN_DEFECT_RATIO)), with λ = $λ and $λf, where a quadrature error " *
              "would fall by about 4. Check that the source integrates to the boundary flux."
    end
    return (; defect = δ, refined_defect = δf, λ, refined_λ = λf, ratio, compatible)
end

# The bordered solve's multiplier `λ` and the relative defect `|λ| Σ|w| / Σ|F|`.
function _neumann_defect(A, F, w)
    _check_annihilates_constants(A)
    u = pde_solve(A, F; mean_zero = w)
    λ = dot(w, F - A * u) / dot(w, w)
    return λ, abs(λ) * sum(abs, w) / sum(abs, F)
end

function _check_annihilates_constants(A)
    norm(A * ones(eltype(A), size(A, 2)), Inf) <= 1e-10 * norm(A, Inf) ||
        _throw_not_pure_neumann(A)
    return nothing
end

@noinline function _throw_not_pure_neumann(A)
    throw(
        ArgumentError(
        "neumann_compatibility needs a pure-Neumann matrix, whose kernel holds the " *
        "constants (‖A 1‖∞ ≤ 1e-10 ‖A‖∞), but the $(size(A, 1))×$(size(A, 2)) matrix " *
        "`problem` returned does not annihilate them. A problem with Dirichlet or Robin " *
        "conditions has no compatibility condition; solve it with pde_solve(A, F).",
    ),
    )
end
