module SpaceSobolevInequalitiesTests

using Test
using Bramble
using Random
using Supposition
using ..TestUtils: WITH_SLOW_TESTS
using ..TestUtils: _nonuniform_points, _zero_boundary!

# The discrete Poincaré and Sobolev embedding inequalities (gpena/Bramble.jl#187), for grid
# functions vanishing on the boundary of the unit domain:
#
#   ‖vₕ‖ₕ    ≤ ‖D₋ₓ vₕ‖₊ₓ        (Poincaré, in every dimension)
#   ‖vₕ‖ₕ,∞  ≤ ‖D₋ₓ vₕ‖₊ₓ        (embedding into L^∞, in 1D only -- see below)
#
# An inequality is a poor thing to test with a handful of smooth functions: what violates one
# is the jagged, badly scaled field nobody writes by hand, which is what the Supposition
# blocks generate -- arbitrary non-uniform partitions and arbitrary values.
#
# **The second inequality does not generalise per direction, and the issue's statement of it
# is wrong for D > 1.** The argument behind it is one-dimensional: |v(x)| = |∫₀ˣ ∂ₓv| ≤
# ‖∂ₓv‖_{L¹(line)} ≤ ‖∂ₓv‖_{L²(line)} by Cauchy-Schwarz on an interval of length 1. In 2D and
# 3D the right-hand side of the claim is the norm over the whole *domain*, which is an average
# of those line norms, and a field concentrated on one line beats the average. The ratio grows
# with dimension even for a smooth bump -- 0.43, 0.59, 0.81 in 1D, 2D, 3D for
# `∏ xᵢ(1-xᵢ)` -- and passes 1 once the field is concentrated, which the last testset here
# pins as a counterexample so the claim cannot be reinstated by accident.
#
# The generalisation that is true reads `‖v‖_∞ ≤ ‖∂₁⋯∂_D v‖`, with the *mixed* derivative:
# `|v(x,y)| = |∫∫ ∂ₓ∂ᵧv|` over the rectangle behind the point. Its right-hand side is a
# quantity staggered in every direction at once, and the inner product that weights those is
# the `2^D`-member staggered family that does not exist yet (#234) -- `innerₕ` and the three
# singletons are what Bramble ships. So that form is left unasserted rather than asserted
# against a weight that is not its own.

# ‖D₋_d vₕ‖ along direction `d` alone. Spelled with `inner₊ₓ` and its siblings rather than
# with `norm₊`, which sums over every direction in 2D and 3D.
_dir_gradient_norm(vₕ, ::Val{1}) = sqrt(inner₊ₓ(D₋ₓ(vₕ), D₋ₓ(vₕ)))
_dir_gradient_norm(vₕ, ::Val{2}) = sqrt(inner₊ᵧ(D₋ᵧ(vₕ), D₋ᵧ(vₕ)))
_dir_gradient_norm(vₕ, ::Val{3}) = sqrt(inner₊₂(D₋₂(vₕ), D₋₂(vₕ)))

# A field on the unit domain vanishing on every boundary plane, from a raw draw.
function _boundary_vanishing(Wₕ, raw, dims)
    a = reshape(copy(raw[1:prod(dims)]), dims)
    return element(Wₕ, vec(_zero_boundary!(a)))
end

@testset "Discrete Poincaré and Sobolev inequalities" begin
    # An inequality is asserted with slack that scales with the quantities compared, not with
    # a bare epsilon: on a badly graded mesh both sides can be large, and equality is
    # approached from below.
    holds(lhs, rhs) = lhs <= rhs * (1 + 1e-10) + 1e-12

    @testset "Poincaré, per direction, 1D/2D/3D" begin
        for D in 1:3
            @testset "$(D)D" begin
                Random.seed!(20260918)
                Ω = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D))))
                for unif in (true, false)
                    Ωₕ = mesh(Ω, ntuple(_ -> 9, Val(D)), ntuple(_ -> unif, Val(D)))
                    Wₕ = gridspace(Ωₕ)
                    dims = npoints(Ωₕ, Tuple)

                    for f in (
                        x -> prod(xi * (1 - xi) for xi in x),
                        x -> sin(π * x[1]) * prod(xi * (1 - xi) for xi in x),
                        x -> 1000 * prod(xi * (1 - xi) for xi in x)^3
                    )
                        vₕ = element(Wₕ, vec(_zero_boundary!(reshape(
                            copy(parent(Rₕ(Wₕ, f))), dims))))
                        for d in 1:D
                            @test holds(normₕ(vₕ), _dir_gradient_norm(vₕ, Val(d)))
                        end
                    end
                end
            end
        end
    end

    @testset "The L^∞ embedding, in 1D" begin
        Random.seed!(20260918)
        for unif in (true, false), n in (9, 33)

            Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), n, unif))
            for f in (
                x -> x[1] * (1 - x[1]),
                x -> sin(π * x[1]),
                x -> 1000 * (x[1] * (1 - x[1]))^3
            )
                vₕ = Rₕ(Wₕ, f)
                parent(vₕ)[1] = parent(vₕ)[end] = 0.0
                @test holds(norminf_h(vₕ), _dir_gradient_norm(vₕ, Val(1)))
            end
        end
    end

    @testset "Both inequalities are sharp enough to be worth asserting" begin
        # A constant-free inequality says nothing unless the two sides are within an order of
        # magnitude for some field; otherwise any bound would pass.
        Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 65, true))
        vₕ = Rₕ(Wₕ, x -> sin(π * x[1]))
        parent(vₕ)[1] = parent(vₕ)[end] = 0.0
        g = _dir_gradient_norm(vₕ, Val(1))
        @test 0.2 < normₕ(vₕ) / g < 1.0
        @test 0.2 < norminf_h(vₕ) / g < 1.0
    end

    @testset "The L^∞ embedding does not generalise per direction (#187)" begin
        # The counterexample, kept as a test so the per-direction claim cannot come back: a
        # field concentrated enough that its maximum beats the domain-averaged directional
        # gradient norm. In 3D the plain product bump already comes within 20% of the bound,
        # and cubing it crosses over.
        Ωₕ = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (9, 9, 9), (true, true, true))
        Wₕ = gridspace(Ωₕ)
        vₕ = Rₕ(Wₕ, x -> 1000 * prod(xi * (1 - xi) for xi in x)^3)
        parent(vₕ)[vec(index_in_marker(Ωₕ, :boundary))] .= 0.0

        for d in 1:3
            @test norminf_h(vₕ) > _dir_gradient_norm(vₕ, Val(d))
        end
        # while Poincaré, which is not a line-wise argument, still holds on the same field
        @test holds(normₕ(vₕ), _dir_gradient_norm(vₕ, Val(1)))
    end

    WITH_SLOW_TESTS && @testset "Random grids (Supposition)" begin
        positive_h = Data.Floats{Float64}(;
            minimum = 0.01, maximum = 10.0, nans = false, infs = false
        )
        field_val = Data.Floats{Float64}(;
            minimum = -100.0, maximum = 100.0, nans = false, infs = false
        )
        holds(lhs, rhs) = lhs <= rhs * (1 + 1e-10) + 1e-12

        # 1D carries both inequalities, which is where the L^∞ one is stated.
        @check function check_poincare_1d(
                h = Data.Vectors(positive_h; min_size = 3, max_size = 40),
                v_raw = Data.Vectors(field_val; min_size = 41, max_size = 41)
        )
            pts = _nonuniform_points(h)
            n = length(pts)
            Ωₕ = mesh(domain(interval(0.0, 1.0)), n, false)
            set_points!(Ωₕ, pts)
            Wₕ = gridspace(Ωₕ)
            vₕ = element(Wₕ, _zero_boundary!(copy(v_raw[1:n])))
            g = _dir_gradient_norm(vₕ, Val(1))
            holds(normₕ(vₕ), g) && holds(norminf_h(vₕ), g)
        end

        @check function check_poincare_2d(
                hx = Data.Vectors(positive_h; min_size = 3, max_size = 8),
                hy = Data.Vectors(positive_h; min_size = 3, max_size = 8),
                v_raw = Data.Vectors(field_val; min_size = 81, max_size = 81)
        )
            px, py = _nonuniform_points(hx), _nonuniform_points(hy)
            nx, ny = length(px), length(py)
            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (nx, ny),
                (false, false))
            set_points!(Ωₕ(1), px)
            set_points!(Ωₕ(2), py)
            Wₕ = gridspace(Ωₕ)
            vₕ = _boundary_vanishing(Wₕ, v_raw, (nx, ny))
            holds(normₕ(vₕ), _dir_gradient_norm(vₕ, Val(1))) &&
                holds(normₕ(vₕ), _dir_gradient_norm(vₕ, Val(2)))
        end

        @check function check_poincare_3d(
                hx = Data.Vectors(positive_h; min_size = 3, max_size = 5),
                hy = Data.Vectors(positive_h; min_size = 3, max_size = 5),
                hz = Data.Vectors(positive_h; min_size = 3, max_size = 5),
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
            vₕ = _boundary_vanishing(Wₕ, v_raw, dims)
            holds(normₕ(vₕ), _dir_gradient_norm(vₕ, Val(1))) &&
                holds(normₕ(vₕ), _dir_gradient_norm(vₕ, Val(3)))
        end
    end
end

end # module SpaceSobolevInequalitiesTests
