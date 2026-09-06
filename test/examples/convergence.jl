using Test
using Bramble
using Random

# End-to-end order of convergence for the worked examples in docs/src/examples/.
#
# test/space/convergence.jl already pins the *operators* to the derivatives they
# approximate. This file pins the whole pipeline — assemble, impose boundary conditions,
# solve — which is a different property: a consistent operator can still be wrecked by the
# assembly or the Dirichlet path, and the resulting solution would be self-consistently
# wrong at first order with nothing to say so.
#
# The examples themselves already compute these orders and already end on a line reading
# `order1 > 1.9 && order2 > 1.9 && order3 > 1.8`. But a Documenter `@example` block renders
# the value of that expression; it does not assert it. A rate that decayed to first order
# would publish `false` in the built documentation and fail nothing. That is the gap this
# file closes, so the numbers on those pages become load-bearing.
#
# Each case mirrors its example deliberately rather than sharing code with it: the pages
# are written to be read, and routing them through a test helper would cost the reader the
# thing they came for. The duplication is the price, and it is the right way round — the
# test's job is to pin the library's order, not the prose.
#
# Method, as in the examples: a manufactured solution, one *random* coarse mesh per
# dimension refined in place with `iterative_refinement!` so every finer level is the same
# random mesh dyadically split rather than an independent draw with its own noise. Random
# grids are the point — a uniform grid makes `exp(sum(x))` nearly exact for this scheme at
# any mesh size, so a uniform check passes on a broken implementation as readily as a
# correct one. Seeded, for the reason spelled out at the top of space/convergence.jl.

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
        bcs_c = dirichlet_constraints(Bramble.set(Ωd), :boundary => sol_d)

        A_c = assemble(assemble_form(Wc); dirichlet_labels = :boundary)
        g_c = element(Wc)
        avgₕ!(g_c, x -> rhs(x, D))
        l_c = form(Wc, v -> innerₕ(g_c, v))
        F_c = assemble(l_c; dirichlet_conditions = bcs_c, dirichlet_labels = :boundary)

        u_c = element(Wc)
        u_c .= A_c \ F_c

        push!(hs, hₘₐₓ(Ωc))
        push!(errs, norm₁ₕ(u_c .- Rₕ(Wc, sol_d)))
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, errs
end

# The seed and the level counts are the examples' own, so a failure here and a changed
# number on the rendered page mean the same thing.
const _SEED = 20260903
const _LEVELS = (1 => (6, 7), 2 => (5, 6), 3 => (5, 4))   # D => (n0, levels)

function _orders(assemble_form, rhs)
    map((1, 2, 3)) do D
        n0, levels = _LEVELS[D].second
        Random.seed!(_SEED)
        hs, errs = _series(assemble_form, rhs, D, n0, levels)
        _observed_order(hs, errs)
    end
end

@testset "Worked example convergence" begin
    # An order far *above* the promise is not reassuring: it means the finest error has
    # reached roundoff and the ratio no longer measures the scheme. Bracketed on both
    # sides so that case fails loudly rather than passing as an excellent result.
    _asymptotically_second_order(p, lower) = lower < p < 3.0

    @testset "Linear Poisson" begin
        # -Δu = g, u_exact = exp(∑x), so g = -D·u_exact
        p1, p2, p3 = _orders(Wc -> form(Wc, Wc, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v))),
            (x, D) -> -D * exp(sum(x)))

        @test _asymptotically_second_order(p1, 1.9)
        @test _asymptotically_second_order(p2, 1.9)
        @test _asymptotically_second_order(p3, 1.8)   # 3D runs fewer levels
    end

    @testset "Convection-diffusion" begin
        ϵ, b = 1.0, 0.1
        # ϵ·(-Δu) + b·∇u with the same manufactured solution: rhs = -D·u_exact·(b + ϵ)
        p1, p2, p3 = _orders(
            Wc -> form(Wc, Wc,
                (u, v) -> ϵ * inner₊(∇₋ₕ(u), ∇₋ₕ(v)) + b * inner₊(M₋ₕ(u), ∇₋ₕ(v))),
            (x, D) -> -D * exp(sum(x)) * (b + ϵ))

        @test _asymptotically_second_order(p1, 1.9)
        @test _asymptotically_second_order(p2, 1.9)
        @test _asymptotically_second_order(p3, 1.8)
    end

    @testset "Variable-coefficient Poisson" begin
        # -∇·(κ(x)∇u) = f with a smooth, spatially varying κ(x) = 1 + 0.5sin(πx₁) > 0 --
        # every case above uses a constant coefficient (docs/src/examples/*.md's own
        # problem statements say so explicitly), so none of them exercises the live grid
        # coefficient path (the same κₕ * ∇₋ₕ(u) mechanism poisson_nonlinear.md's α(u)
        # uses) under an independently-known exact answer.
        #
        # Same manufactured solution as "Linear Poisson" above, so f is worked out by hand
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
                # discretization choice. `M₋ₕ` -- the same averaging poisson_nonlinear.md
                # uses to move its solution-dependent α onto the staggered grid -- moves κ
                # there too, direction by direction.
                κₕ = Rₕ(Wc, κ)
                D = dim(Wc)
                κf = M₋ₕ(κₕ)
                gradκ(u) = D == 1 ? κf * ∇₋ₕ(u) : ntuple(i -> κf[i] * ∇₋ₕ(u)[i], D)
                form(Wc, Wc, (u, v) -> inner₊(gradκ(u), ∇₋ₕ(v)))
            end,
            (x, D) -> -(dκ(x) + D * κ(x)) * exp(sum(x)))

        @test _asymptotically_second_order(p1, 1.9)
        @test _asymptotically_second_order(p2, 1.9)
        @test _asymptotically_second_order(p3, 1.8)
    end
end
