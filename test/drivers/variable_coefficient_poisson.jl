module DriversVariableCoefficientPoissonTests

using Test
using Bramble
using Random

# Order of convergence for an operator no worked example uses: -div(κ(x)∇u) with a smooth,
# spatially varying κ. Every page in docs/src/examples/ states a constant coefficient in its
# own problem statement, so none of them reaches the live grid-coefficient path (the same
# κₕ * ∇₋ₕ(u) mechanism the nonlinear Poisson page's α(u) uses) under an independently-known
# exact answer.
#
# test/convergence/operators.jl already pins the *operators* to the derivatives they approximate.
# This file pins the whole pipeline -- assemble, impose boundary conditions, solve -- which is
# a different property: a consistent operator can still be wrecked by the assembly or the
# Dirichlet path, and the resulting solution would be self-consistently wrong at first order
# with nothing to say so.
#
# The linear-Poisson and convection-diffusion cases that lived here mirrored
# docs/src/examples/poisson_linear.md and convection_diffusion_linear.md line by line, so
# that the orders those pages rendered were asserted somewhere. The pages are Literate
# scripts now and the suite runs them directly (test/examples/pages.jl), so the copies are
# gone and only this case, which was never a copy of anything, remains
# (gpena/Bramble.jl#117).
#
# Method, as on the pages: a manufactured solution, one *random* coarse mesh per dimension
# refined in place with `iterative_refinement!` so every finer level is the same random mesh
# dyadically split rather than an independent draw with its own noise. Random grids are the
# point -- a uniform grid makes `exp(sum(x))` nearly exact for this scheme at any mesh size,
# so a uniform check passes on a broken implementation as readily as a correct one. Seeded,
# for the reason spelled out at the top of convergence/operators.jl.

# Observed order from the finest pair. The coarse levels are not yet asymptotic, so only
# the last ratio is worth asserting on.
_observed_order(hs, errs) = log(errs[end - 1] / errs[end]) / log(hs[end - 1] / hs[end])

# Solve the manufactured problem on `levels` successively halved random meshes, returning
# the mesh size and the discrete H¹ error at each. `assemble_form` builds the bilinear form
# for a given space; `rhs` is the manufactured source.
function _series(assemble_form, rhs, D::Int, n0::Int, levels::Int)
    sol_d(x) = exp(sum(x))

    Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
    Ωc = mesh(Ωd, ntuple(_ -> n0, D), ntuple(_ -> false, D))

    hs, errs = Float64[], Float64[]
    for level in 1:levels
        Wc = gridspace(Ωc)
        bcs_c = dirichlet_constraints(Ωd, :boundary => sol_d)

        A_c = assemble(assemble_form(Wc); dirichlet = :boundary)
        g_c = element(Wc)
        avgₕ!(g_c, x -> rhs(x, D))
        l_c = form(Wc, v -> innerₕ(g_c, v))
        F_c = assemble(l_c; dirichlet = bcs_c)

        u_c = element(Wc)
        u_c .= A_c \ F_c

        push!(hs, hₘₐₓ(Ωc))
        push!(errs, norm₁ₕ(u_c .- Rₕ(Wc, sol_d)))
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, errs
end

# The seed and the level counts are the ones the worked-example pages use, so this case is
# measured on the same meshes they are.
const _SEED = 20260903
const _LEVELS = (1 => (6, 7), 2 => (5, 6), 3 => (5, 4))   # D => (n0, levels)

function _orders(assemble_form, rhs)
    map((1, 2, 3)) do D
        n0, levels = _LEVELS[D].second
        Random.seed!(_SEED)
        hs, errs = _series(assemble_form, rhs, D, n0, levels)
        return _observed_order(hs, errs)
    end
end

# An order far *above* the promise is not reassuring: it means the finest error has reached
# roundoff and the ratio no longer measures the scheme. Bracketed on both sides so that case
# fails loudly rather than passing as an excellent result.
_asymptotically_second_order(p, lower) = lower < p < 3.0

@testset "Variable-coefficient Poisson" begin
    # -∇·(κ(x)∇u) = f with κ(x) = 1 + 0.5sin(πx₁) > 0. Same manufactured solution the
    # worked-example pages use, exp(∑x), so f is worked out by hand
    # from div(κ∇u) = (κ'(x₁) + D·κ(x₁))·exp(∑x): κ depends only on x₁, so the x₁ term
    # picks up κ' from the product rule while each of the other D-1 directions
    # contributes a plain κ·exp(∑x) (∂ᵢexp(∑x) = exp(∑x) for every i).
    κ(x) = 1.0 + 0.5 * sin(pi * x[1])
    dκ(x) = 0.5 * pi * cos(pi * x[1])

    p1, p2, p3 = _orders(
        Wc -> begin
            # Evaluating κ at the nodes and multiplying it straight into the nodal
            # gradient degrades to first order: `∇₋ₕ(u)` lives at the staggered
            # half-points, so a nodal κ is an O(h) mismatch in *location*, not just a
            # discretization choice. `M₋ₕ` -- the same averaging poisson_nonlinear.jl
            # uses to move its solution-dependent α onto the staggered grid -- moves κ
            # there too, direction by direction.
            κₕ = Rₕ(Wc, κ)
            D = dim(Wc)
            κf = M₋ₕ(κₕ)
            gradκ(u) = D == 1 ? κf * ∇₋ₕ(u) : ntuple(i -> κf[i] * ∇₋ₕ(u)[i], D)
            form(Wc, Wc, (u, v) -> inner₊(gradκ(u), ∇₋ₕ(v)))
        end,
        (x, D) -> -(dκ(x) + D * κ(x)) * exp(sum(x))
    )

    @test _asymptotically_second_order(p1, 1.9)
    @test _asymptotically_second_order(p2, 1.9)
    @test _asymptotically_second_order(p3, 1.8)
end

end # module DriversVariableCoefficientPoissonTests
