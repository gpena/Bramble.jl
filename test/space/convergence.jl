using Test
using Bramble
using Bramble: values

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

# Error of `op` against the exact derivative `df`, over the points whose stencil is not
# truncated. The max norm is used so the result does not depend on the quadrature weights.
function _interior_error(Ωₕ, op, f, df, drop)
    Wₕ = gridspace(Ωₕ)
    e = values(op(Rₕ(Wₕ, f))) .- values(Rₕ(Wₕ, df))
    dims = npoints(Ωₕ, Tuple)
    return maximum(abs, drop(reshape(e, dims)))
end

# Successive halvings of the mesh give log2 of the error ratio as the observed order.
function _orders(Ωₕ, op, f, df, drop; steps = 4)
    errs = Float64[]
    for k in 0:steps
        k > 0 && iterative_refinement!(Ωₕ)
        push!(errs, _interior_error(Ωₕ, op, f, df, drop))
    end
    return [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]
end

@testset "Convergence order of the finite differences" begin
    @testset "1D, first order away from the truncated point" begin
        # A backward difference has no stencil at the first point and a forward one none
        # at the last, so those are dropped. Both uniform and random starting grids.
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                for (opname, op, drop) in (("D₋ₓ", D₋ₓ, e -> @view e[2:end]),
                    ("D₊ₓ", D₊ₓ, e -> @view e[1:(end - 1)]))
                    Ωₕ = mesh(domain(interval(0.0, 1.0)), 51, unif)
                    ords = _orders(Ωₕ, op, sin, cos, drop)
                    @test all(>(0.9), ords)
                    @test 0.95 < last(ords) < 1.05
                end
            end
        end
    end

    @testset "2D, first order along each direction" begin
        f = x -> sin(x[1]) * exp(x[2])
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                for (opname, op, df, drop) in (
                    ("D₋ₓ", D₋ₓ, x -> cos(x[1]) * exp(x[2]), e -> @view e[2:end, :]),
                    ("D₋ᵧ", D₋ᵧ, x -> sin(x[1]) * exp(x[2]), e -> @view e[:, 2:end]))
                    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 17),
                        (unif, unif))
                    ords = _orders(Ωₕ, op, f, df, drop; steps = 3)
                    @test all(>(0.9), ords)
                    @test 0.95 < last(ords) < 1.05
                end
            end
        end
    end

    @testset "including the truncated point halves the observed order" begin
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
        @test values(D₋ₓ(Rₕ(Wₕ, sin)))[1] == 0.0
        @test abs(values(Rₕ(Wₕ, cos))[1] - 1.0) < 1e-12
    end
end
