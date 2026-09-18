module ConvergenceOperatorsTests

using Test
using Bramble
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
import Bramble: D₊ₓ
using Random

# Convergence order of the finite difference operators.
#
# The value-comparison tests elsewhere check that an operator computes what its formula
# says. They cannot tell whether the formula is the right one: a wrong spacing, or the
# wrong neighbour, still produces a self-consistent set of numbers. Order of convergence
# is the property that pins the operator to the derivative it approximates.
#
# Refinement is done with iterative_refinement!, which halves every interval, so the
# meshes are nested and the ratio of successive errors is an order even when the starting
# grid is arbitrary. That lets the non-uniform cases start from a random grid, which is
# what mesh(Ω, n, false) produces.
#
# The random grids are seeded. An order measured on the coarsest pair is not yet
# asymptotic and varies with how uneven the draw happens to be: over forty draws the
# smallest ratio ranged from 0.948 to 0.991, and an unseeded run occasionally fell below
# the bound asserted here. Seeding keeps the test reproducible, so a failure is a real
# regression rather than an unlucky grid.

# Error of `op` against the exact derivative `df`, over the points whose stencil is not
# truncated. The max norm is used so the result does not depend on the quadrature weights.
function _interior_error(Ωₕ, op, f, df, drop)
    Wₕ = gridspace(Ωₕ)
    e = parent(op(Rₕ(Wₕ, f))) .- parent(Rₕ(Wₕ, df))
    dims = npoints(Ωₕ, Tuple)
    return maximum(abs, drop(reshape(e, dims)))
end

# Successive halvings of the mesh give log2 of the error ratio as the observed order.
# Returns the per-step ratios alongside the raw errors, so a caller can also fit a slope
# across every level (`_lsq_order`) rather than only reading the last pair.
function _orders(Ωₕ, op, f, df, drop; steps = 4)
    errs = Float64[]
    for k in 0:steps
        k > 0 && iterative_refinement!(Ωₕ)
        push!(errs, _interior_error(Ωₕ, op, f, df, drop))
    end
    ords = [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]
    return ords, errs
end

# Least-squares order across every refinement level, not just the last pair: since each
# level halves every spacing, err_k ≈ C·h₀^p·2^(-pk), so log2(err_k) is linear in the level
# index k with slope -p. Fitting the whole series is more robust than the two-point ratio
# when an early level (still pre-asymptotic) is noisier than the rest.
function _lsq_order(errs)
    k = 0:(length(errs) - 1)
    y = log2.(errs)
    kbar, ybar = sum(k) / length(k), sum(y) / length(y)
    slope = sum((k .- kbar) .* (y .- ybar)) / sum(abs2, k .- kbar)
    return -slope
end

@testset "Difference convergence" begin
    @testset "1D order" begin
        # A backward difference has no stencil at the first point and a forward one none
        # at the last, so those are dropped. Both uniform and random starting grids.
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                for (opname, op, drop) in (
                    ("D₋ₓ", D₋ₓ, e -> @view e[2:end]),
                    ("D₊ₓ", D₊ₓ, e -> @view e[1:(end - 1)])
                )
                    Random.seed!(20250829)
                    Ωₕ = mesh(domain(interval(0.0, 1.0)), 51, unif)
                    ords, errs = _orders(Ωₕ, op, sin, cos, drop)
                    @test all(>(0.9), ords)
                    @test 0.95 < last(ords) < 1.05
                    @test 0.95 < _lsq_order(errs) < 1.05
                end
            end
        end
    end

    @testset "2D order" begin
        f = x -> sin(x[1]) * exp(x[2])
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                for (opname, op, df, drop) in (
                    ("D₋ₓ", D₋ₓ, x -> cos(x[1]) * exp(x[2]), e -> @view e[2:end, :]),
                    ("D₋ᵧ", D₋ᵧ, x -> sin(x[1]) * exp(x[2]), e -> @view e[:, 2:end])
                )
                    Random.seed!(20250829)
                    Ωₕ = mesh(
                        domain(interval(0.0, 1.0) × interval(0.0, 1.0)),
                        (17, 17),
                        (unif, unif)
                    )
                    ords, errs = _orders(Ωₕ, op, f, df, drop; steps = 3)
                    @test all(>(0.9), ords)
                    @test 0.95 < last(ords) < 1.05
                    @test 0.95 < _lsq_order(errs) < 1.05
                end
            end
        end
    end

    @testset "Extreme aspect ratio" begin
        # Δy/Δx ~ 1e4 on a 2D grid, the kind of elongated domain a boundary-layer or thin-
        # channel geometry produces. `f` is rescaled in y so its values stay O(1) despite
        # the domain spanning four orders of magnitude in that direction; what is under
        # test is whether the operator's own order degrades from the resulting spacing
        # disparity, not whether the manufactured values themselves stay reasonable.
        f = x -> sin(x[1]) * exp(x[2] / 1.0e4)
        for (opname, op, df, drop) in (
            ("D₋ₓ", D₋ₓ, x -> cos(x[1]) * exp(x[2] / 1.0e4), e -> @view e[2:end, :]),
            (
            "D₋ᵧ",
            D₋ᵧ,
            x -> sin(x[1]) * exp(x[2] / 1.0e4) / 1.0e4,
            e -> @view e[:, 2:end]
        )
        )
            @testset "$opname" begin
                Random.seed!(20250829)
                Ωₕ = mesh(
                    domain(interval(0.0, 1.0) × interval(0.0, 1.0e4)),
                    (17, 17),
                    (true, true)
                )
                ords, _ = _orders(Ωₕ, op, f, df, drop; steps = 3)
                @test all(>(0.9), ords)
                @test 0.95 < last(ords) < 1.05
            end
        end
    end

    @testset "Truncation boundary order" begin
        # The trap this exists to document. D₋ₓ is zero at the first point while the
        # derivative is not, so that one point contributes an O(1) error at every
        # refinement. It carries a weight of about h/2 in the discrete L² norm, so it
        # alone contributes about sqrt(h/2), and the measured order is one half rather
        # than one however fine the grid gets.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 51, true)
        errs = Float64[]
        for k in 0:4
            k > 0 && iterative_refinement!(Ωₕ)
            Wₕ = gridspace(Ωₕ)
            push!(errs, normₕ(D₋ₓ(Rₕ(Wₕ, sin)) - Rₕ(Wₕ, cos)))
        end
        ords = [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]
        @test all(o -> abs(o - 0.5) < 0.02, ords)

        # and the error at that point stays O(1) rather than shrinking
        Wₕ = gridspace(Ωₕ)
        @test parent(D₋ₓ(Rₕ(Wₕ, sin)))[1] == 0.0
        @test abs(parent(Rₕ(Wₕ, cos))[1] - 1.0) < 1e-12
    end
end

# Natural boundary conditions through `inner_Γ` (gpena/Bramble.jl#157).
#
# A wrong surface weight is invisible to a value comparison -- the assembled boundary term is
# self-consistent whatever weight it carries -- and invisible to a pure Dirichlet problem,
# which never uses it. What exposes it is the order of the solution: a weight off by a factor
# of two, or missing its corner share, drops a second-order scheme to first order.
@testset "Natural boundary conditions converge at order two" begin
    @testset "2D, mixed Dirichlet-Neumann-Robin, graded mesh" begin
        u_ex(x) = cos(2.2 * x[1]) * exp(0.7 * x[2])
        ux(x) = -2.2 * sin(2.2 * x[1]) * exp(0.7 * x[2])
        uy(x) = 0.7 * cos(2.2 * x[1]) * exp(0.7 * x[2])
        f(x) = (2.2^2 - 0.7^2 + 1.0) * u_ex(x)
        β = 1.7
        # the outward normal is -y on :ymin and +y on :ymax, so the two Robin data differ
        gN(x) = ux(x)
        gR_min(x) = -uy(x) + β * u_ex(x)
        gR_max(x) = uy(x) + β * u_ex(x)

        Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0),
            :xmin => :xmin, :xmax => :xmax, :ymin => :ymin, :ymax => :ymax)
        bcs = dirichlet_constraints(Ω, :xmin => u_ex)

        errs = map((9, 17, 33, 65)) do n
            Wₕ = gridspace(mesh(Ω, (n, n), (true, true)))
            a = form(Wₕ, Wₕ,
                (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v) +
                          inner_Γ(β * u, v; markers = (:ymin, :ymax)))
            l = form(Wₕ,
                v -> innerₕ(f, v) + inner_Γ(gN, v; markers = (:xmax,)) +
                     inner_Γ(gR_min, v; markers = (:ymin,)) +
                     inner_Γ(gR_max, v; markers = (:ymax,)))
            A, F = assemble(a, l; dirichlet = bcs)
            uₕ = element(Wₕ)
            parent(uₕ) .= Matrix(A) \ Vector(F)
            return normₕ(uₕ - Rₕ(Wₕ, u_ex))
        end

        rates = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
        @test all(r -> 1.95 < r < 2.05, rates)
        @test _lsq_order(errs) > 1.95
    end

    @testset "1D Neumann, where the weight is 1 and not half a cell" begin
        # The case a 2D intuition gets wrong: a `(D-1)`-face is a point, of measure 1, so any
        # `h/2` at the endpoint would break exactly this order.
        u_ex(x) = sin(1.3 * x[1]) + 0.4 * x[1]^2
        du(x) = 1.3 * cos(1.3 * x[1]) + 0.8 * x[1]
        f(x) = 1.3^2 * sin(1.3 * x[1]) - 0.8 + u_ex(x)

        Ω = domain(interval(0.0, 1.0), :xmin => :xmin, :xmax => :xmax)
        bcs = dirichlet_constraints(Ω, :xmin => u_ex)

        errs = map((9, 17, 33, 65)) do n
            Wₕ = gridspace(mesh(Ω, n, true))
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
            l = form(Wₕ, v -> innerₕ(f, v) + inner_Γ(du, v; markers = (:xmax,)))
            A, F = assemble(a, l; dirichlet = bcs)
            uₕ = element(Wₕ)
            parent(uₕ) .= Matrix(A) \ Vector(F)
            return normₕ(uₕ - Rₕ(Wₕ, u_ex))
        end

        rates = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
        @test all(r -> 1.9 < r < 2.1, rates)

# The discrete Laplacian (gpena/Bramble.jl#158). Its stencil is truncated at both ends of
# every axis, so the error is measured away from them, as it is for the differences above,
# and the meshes are refined by halving rather than redrawn, for the reason the header of
# this file gives.
#
# The two orders differ, and the difference is the point. On a uniform mesh the conservative
# form is second order pointwise. On a non-uniform one it is only first order pointwise: the
# flux difference is centred on the cell face, not on the node, and the two half-cells have
# different widths. That is not a defect of this implementation -- the same stencil is second
# order in the discrete `normₕ`, which is what a solved problem sees (supraconvergence), and
# what `test/space/inner_product.jl` and the MMS suites measure. Asserting 2 here would be
# asserting something false about the operator.
@testset "Δₕ convergence" begin
    f2(x) = sin(1.7 * x[1]) * exp(0.6 * x[2])
    lap2(x) = (-1.7^2 + 0.6^2) * f2(x)

    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    drop_rim(e) = @view e[2:(end - 1), 2:(end - 1)]

    @testset "uniform: second order" begin
        Ωₕ = mesh(Ω, (9, 9), (true, true))
        _, errs = _orders(Ωₕ, Δₕ, f2, lap2, drop_rim; steps = 3)
        @test 1.9 < _lsq_order(errs) < 2.1
    end

    @testset "non-uniform: first order pointwise" begin
        Random.seed!(20250829)
        Ωₕ = mesh(Ω, (9, 9), (false, false))
        _, errs = _orders(Ωₕ, Δₕ, f2, lap2, drop_rim; steps = 3)
        @test 0.9 < _lsq_order(errs) < 1.6
    end
end

end # module ConvergenceOperatorsTests
