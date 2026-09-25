module SpaceDiscreteCalculusIdentitiesTests

using Test
using Bramble
using Bramble: D₋
using Bramble: Dcₓ, D₋ₓ, Mₓ, inner₊ₓ, norminf_h, set_points!
using Random
using Supposition
using Bramble: cell_measures, AbstractMeshType, weights, Innerplus
using ..TestUtils: WITH_SLOW_TESTS
using ..TestUtils: _nonuniform_points, _zero_boundary!

# The discrete calculus identities and inequalities of Propositions 2.1-2.4
# (gpena/Bramble.jl#188), for grid functions vanishing on the boundary.
#
#   2.1  (Δₕu, v)ₕ = -(∇ₕu, ∇ₕv)₊                       discrete Green's first identity
#   2.2  ‖uₕ‖ₕ ≤ ‖∇ₕuₕ‖₊                                Poincaré-Friedrichs
#   2.3  ‖uₕ‖ₕ,∞ ≤ ‖uₕ‖ₕ / √Hₘᵢₙ  and two relatives      inverse and embedding inequalities
#   2.4  (∇c,ₕ·uₕ, vₕ)ₕ = -(Mₕuₕ, ∇ₕvₕ)₊                centered divergence duality
#
# 2.4 is not here: it is `test/space/sbp_identities.jl`'s subject, tested there in 1D, 2D and
# 3D with its own Supposition suite. Worth knowing when reading the issue, which states it
# with the *forward* average M₊: that spelling does not close the identity, and that file
# pins the failure as well as the correction. The average that pairs with `inner₊`'s backward
# weighting is the backward one, for the indexing reason its header sets out. The
# cross-reference is exercised below rather than only asserted in prose.
#
# 2.1 also has a deterministic counterpart in `test/space/operators.jl`, where `Δₕ` is
# defined; what is here is the property-based version over arbitrary meshes and fields.

# The smallest cell measure on the grid. Separable, so the minimum of the product is the
# product of the per-axis minima.
_Hmin(Ωₕ::AbstractMeshType{1}) = minimum(cell_measures(Ωₕ))
_Hmin(Ωₕ::AbstractMeshType) = prod(minimum, cell_measures(Ωₕ))

# ‖uₕ‖_{1,H,∞}, the maximum norm of the function and of every backward difference of it.
_w1inf(uₕ, D) = max(norminf_h(uₕ), maximum(d -> norminf_h(D₋(uₕ, Val(d))), 1:D))

# The smallest *staggered* weight, over every direction. `inner₊`'s weight along `d` is
# `h_d(i) ∏_{e≠d} ĥ_e(i_e)` -- a full spacing along `d` where the cell measure carries a half
# one -- so on a graded mesh it can be orders of magnitude below `Hₘᵢₙ`, which is what makes
# it and not `Hₘᵢₙ` the constant in the gradient inverse inequality (see 2.3 below). The
# zeros are the truncated slices, where the difference has no stencil and contributes
# nothing, so they are skipped rather than taken as the minimum.
function _wplus_min(Wₕ, D)
    return minimum(1:D) do d
        w = weights(Wₕ, Innerplus(), d)
        return minimum(x for x in w if x > 0)
    end
end

function _boundary_vanishing(Wₕ, raw, dims)
    a = reshape(copy(raw[1:prod(dims)]), dims)
    return element(Wₕ, vec(_zero_boundary!(a)))
end

@testset "Discrete calculus identities (Propositions 2.1-2.4)" begin
    Random.seed!(20260918)

    # Compared with an absolute floor as well as a relative one: on fields that happen not to
    # vary much both sides are small, and a purely relative comparison reports a large error
    # on two values of order 1e-17.
    agree(a, b) = isapprox(a, b; atol = 1e-11, rtol = 1e-11)
    holds(lhs, rhs) = lhs <= rhs * (1 + 1e-10) + 1e-12

    _unit(D) = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D))))

    function _random_vanishing(Wₕ, dims)
        a = randn(dims)
        return element(Wₕ, vec(_zero_boundary!(a)))
    end

    @testset "2.1 Discrete Green's first identity" begin
        for D in 1:3, unif in (true, false)

            @testset "$(D)D $(unif ? "uniform" : "non-uniform")" begin
                Ωₕ = mesh(_unit(D), ntuple(_ -> 9, Val(D)), ntuple(_ -> unif, Val(D)))
                Wₕ = gridspace(Ωₕ)
                dims = npoints(Ωₕ, Tuple)
                for _ in 1:3
                    uₕ = _random_vanishing(Wₕ, dims)
                    vₕ = _random_vanishing(Wₕ, dims)
                    lhs = innerₕ(Δₕ(uₕ), vₕ)
                    rhs = -inner₊(∇ₕ(uₕ), ∇ₕ(vₕ))
                    @test agree(lhs, rhs)
                    # and the quantity is not accidentally zero, which would make the
                    # identity hold vacuously
                    @test abs(lhs) > 1e-6
                end
            end
        end
    end

    @testset "2.2 Poincaré-Friedrichs" begin
        for D in 1:3, unif in (true, false)

            Ωₕ = mesh(_unit(D), ntuple(_ -> 9, Val(D)), ntuple(_ -> unif, Val(D)))
            Wₕ = gridspace(Ωₕ)
            dims = npoints(Ωₕ, Tuple)
            for _ in 1:3
                uₕ = _random_vanishing(Wₕ, dims)
                @test holds(normₕ(uₕ), norm₊(∇ₕ(uₕ)))
            end
        end
    end

    @testset "2.3 Inverse and embedding inequalities" begin
        # The first is sharp as written here: ‖u‖ₕ² = Σ H_I u_I² ≥ Hₘᵢₙ max u_I², so the
        # constant is 1/√Hₘᵢₙ and a spike on the smallest cell nearly attains it (the worst
        # ratio measured over a hundred random meshes is 0.79). The issue states it with
        # 1/Hₘᵢₙ, which is looser on a mesh finer than one cell across; both are asserted, so
        # the sharper statement is the one that would break first.
        #
        # The third one is stated differently from the issue, which writes it with 1/Hₘᵢₙ.
        # That constant is wrong on a graded mesh, and the next testset holds the
        # counterexample: the quantity being bounded is a *difference*, so it is divided by a
        # spacing, and what controls it is the smallest weight `inner₊` carries -- a full
        # spacing along the difference's own direction -- not the smallest cell measure,
        # which carries a half one. The two coincide to within a factor on a quasi-uniform
        # mesh and separate by orders of magnitude on a graded one.
        for D in 1:3, unif in (true, false)

            Ωₕ = mesh(_unit(D), ntuple(_ -> 9, Val(D)), ntuple(_ -> unif, Val(D)))
            Wₕ = gridspace(Ωₕ)
            dims = npoints(Ωₕ, Tuple)
            Hm = _Hmin(Ωₕ)
            wm = _wplus_min(Wₕ, D)
            for _ in 1:3
                uₕ = _random_vanishing(Wₕ, dims)
                g = norm₊(∇ₕ(uₕ))
                @test holds(norminf_h(uₕ), normₕ(uₕ) / sqrt(Hm))
                @test holds(norminf_h(uₕ), normₕ(uₕ) / Hm)
                @test holds(norminf_h(uₕ), g / sqrt(2 * Hm))
                @test holds(_w1inf(uₕ, D), max(normₕ(uₕ) / sqrt(Hm), g / sqrt(wm)))
                # on these meshes, which are not strongly graded, the issue's constant holds
                # too -- it is the graded case below that separates them
                @test holds(_w1inf(uₕ, D), g / Hm)
            end
        end
    end

    @testset "2.3's gradient inverse inequality needs the staggered weight (#188)" begin
        # A mesh with one cell three orders of magnitude thinner than its neighbours, and a
        # single spike beside it. `Hₘᵢₙ` barely moves -- a cell measure averages the two
        # spacings around a point, so one thin cell between two fat ones leaves every measure
        # of order 1 -- while the difference across the thin cell is enormous. This is the
        # case the issue's `‖u‖₁,∞ ≤ ‖∇u‖₊ / Hₘᵢₙ` gets wrong.
        px = [0.0, 0.4348, 0.4354, 1.0]
        py = [0.0, 1 / 3, 2 / 3, 1.0]
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (false, false))
        set_points!(Ωₕ(1), px)
        set_points!(Ωₕ(2), py)
        Wₕ = gridspace(Ωₕ)

        a = zeros(4, 4)
        a[2, 2] = 1.0
        uₕ = element(Wₕ, vec(a))
        Hm = _Hmin(Ωₕ)
        wm = _wplus_min(Wₕ, 2)
        g = norm₊(∇ₕ(uₕ))

        # the two constants are far apart on this mesh, which is the whole point
        @test wm < Hm / 100
        # the issue's form fails
        @test _w1inf(uₕ, 2) > g / Hm
        # the staggered-weight form holds
        @test holds(_w1inf(uₕ, 2), max(normₕ(uₕ) / sqrt(Hm), g / sqrt(wm)))
    end

    @testset "2.4 lives in sbp_identities.jl" begin
        # One live call, so the cross-reference above cannot go stale silently: the centered
        # divergence pairs with the *backward* average through `inner₊`.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, false)
        Wₕ = gridspace(Ωₕ)
        uₕ = _random_vanishing(Wₕ, npoints(Ωₕ, Tuple))
        vₕ = _random_vanishing(Wₕ, npoints(Ωₕ, Tuple))
        @test agree(innerₕ(Dcₓ(uₕ), vₕ), -inner₊ₓ(Mₓ(uₕ), D₋ₓ(vₕ)))
    end

    WITH_SLOW_TESTS && @testset "Random grids (Supposition)" begin
        positive_h = Data.Floats{Float64}(;
            minimum = 0.01, maximum = 10.0, nans = false, infs = false
        )
        field_val = Data.Floats{Float64}(;
            minimum = -100.0, maximum = 100.0, nans = false, infs = false
        )
        # Scale-normalised, because a random grid can put several orders of magnitude between
        # the smallest and the largest cell and the two sides are then sums of terms much
        # larger than their difference.
        scaled_agree(lhs, rhs) = isapprox(
            lhs, rhs; atol = 1e-10 * max(abs(lhs), abs(rhs), 1.0), rtol = 1e-10
        )
        #holds(lhs, rhs) = lhs <= rhs * (1 + 1e-10) + 1e-12

        @check function check_green_1d(
                h = Data.Vectors(positive_h; min_size = 3, max_size = 30),
                u_raw = Data.Vectors(field_val; min_size = 31, max_size = 31),
                v_raw = Data.Vectors(field_val; min_size = 31, max_size = 31)
        )
            pts = _nonuniform_points(h)
            n = length(pts)
            Ωₕ = mesh(domain(interval(0.0, 1.0)), n, false)
            set_points!(Ωₕ, pts)
            Wₕ = gridspace(Ωₕ)
            uₕ = element(Wₕ, _zero_boundary!(copy(u_raw[1:n])))
            vₕ = element(Wₕ, _zero_boundary!(copy(v_raw[1:n])))
            scaled_agree(innerₕ(Δₕ(uₕ), vₕ), -inner₊(∇ₕ(uₕ), ∇ₕ(vₕ)))
        end

        @check function check_green_2d(
                hx = Data.Vectors(positive_h; min_size = 3, max_size = 8),
                hy = Data.Vectors(positive_h; min_size = 3, max_size = 8),
                u_raw = Data.Vectors(field_val; min_size = 81, max_size = 81),
                v_raw = Data.Vectors(field_val; min_size = 81, max_size = 81)
        )
            px, py = _nonuniform_points(hx), _nonuniform_points(hy)
            dims = (length(px), length(py))
            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), dims, (false, false))
            set_points!(Ωₕ(1), px)
            set_points!(Ωₕ(2), py)
            Wₕ = gridspace(Ωₕ)
            uₕ = _boundary_vanishing(Wₕ, u_raw, dims)
            vₕ = _boundary_vanishing(Wₕ, v_raw, dims)
            scaled_agree(innerₕ(Δₕ(uₕ), vₕ), -inner₊(∇ₕ(uₕ), ∇ₕ(vₕ)))
        end

        @check function check_green_3d(
                hx = Data.Vectors(positive_h; min_size = 3, max_size = 5),
                hy = Data.Vectors(positive_h; min_size = 3, max_size = 5),
                hz = Data.Vectors(positive_h; min_size = 3, max_size = 5),
                u_raw = Data.Vectors(field_val; min_size = 216, max_size = 216),
                v_raw = Data.Vectors(field_val; min_size = 216, max_size = 216)
        )
            px, py, pz = _nonuniform_points(hx), _nonuniform_points(hy),
            _nonuniform_points(hz)
            dims = (length(px), length(py), length(pz))
            Ωₕ = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
                dims, (false, false, false))
            set_points!(Ωₕ(1), px)
            set_points!(Ωₕ(2), py)
            set_points!(Ωₕ(3), pz)
            Wₕ = gridspace(Ωₕ)
            uₕ = _boundary_vanishing(Wₕ, u_raw, dims)
            vₕ = _boundary_vanishing(Wₕ, v_raw, dims)
            scaled_agree(innerₕ(Δₕ(uₕ), vₕ), -inner₊(∇ₕ(uₕ), ∇ₕ(vₕ)))
        end

        # The inequalities, over the same arbitrary partitions: 2.2 and the three of 2.3 at
        # once, since they share the mesh and the field.
        @check function check_inequalities_2d(
                hx = Data.Vectors(positive_h; min_size = 3, max_size = 8),
                hy = Data.Vectors(positive_h; min_size = 3, max_size = 8),
                u_raw = Data.Vectors(field_val; min_size = 81, max_size = 81)
        )
            px, py = _nonuniform_points(hx), _nonuniform_points(hy)
            dims = (length(px), length(py))
            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), dims, (false, false))
            set_points!(Ωₕ(1), px)
            set_points!(Ωₕ(2), py)
            Wₕ = gridspace(Ωₕ)
            uₕ = _boundary_vanishing(Wₕ, u_raw, dims)
            Hm = _Hmin(Ωₕ)
            wm = _wplus_min(Wₕ, 2)
            g = norm₊(∇ₕ(uₕ))
            holds(normₕ(uₕ), g) &&
                holds(norminf_h(uₕ), normₕ(uₕ) / sqrt(Hm)) &&
                holds(norminf_h(uₕ), g / sqrt(2 * Hm)) &&
                holds(_w1inf(uₕ, 2), max(normₕ(uₕ) / sqrt(Hm), g / sqrt(wm)))
        end
    end
end

end # module SpaceDiscreteCalculusIdentitiesTests
