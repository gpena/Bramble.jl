module ConvergenceGhostOraclesTests

# Observed orders of the plain-matrix ghost-point oracles: the 1D one (GhostOracle1D.jl,
# notes §1), the 2D/3D facet-only one (GhostOracleND.jl, notes §2) and the corner ghosts of
# the mixed-derivative problem (GhostCorners.jl, notes §4). Moved here from
# test/ghost/oracle_1d.jl, oracle_nd.jl and corners.jl, which pin the oracles' rows.

using Test
using Bramble
using ..TestUtils: _order_header, _order_row

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(@__DIR__, "..", "ghost", "GhostOracle1D.jl"))
isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(@__DIR__, "..", "ghost", "GhostOracleND.jl"))
isdefined(Main, :GhostCorners) ||
    Base.include(Main, joinpath(@__DIR__, "..", "ghost", "GhostCorners.jl"))

@testset "1D oracle" begin
    O = Main.GhostOracle1D
    @testset "Second order" begin
        for p in O.PROBLEMS, kind in O.MESH_KINDS

            r = O.observed_order(p, kind)
            @test r > 1.8
        end
    end
end

@testset "ND oracle" begin
    O = Main.GhostOracleND
    @testset "Orders" begin
        # Every mesh kind in 2D and 3D on four levels, gated on uniform and graded meshes;
        # refined random meshes printed. The errors must also fall at every level and end small
        # (the slope alone hides a blow-up).
        levels = 4
        dims = (2, 3)
        kinds = O.MESH_KINDS
        _order_header("ND oracle: discrete H¹ error, $levels levels",
            ("dim", "mesh", "problem"), (5, 9, 11))
        for dim in dims, kind in kinds, p in O.PROBLEMS
            hs, errs = O.observed_errors(p, kind; dim, levels)
            X, Y = log.(hs), log.(errs)
            X̄, Ȳ = sum(X) / levels, sum(Y) / levels
            r = sum((X .- X̄) .* (Y .- Ȳ)) / sum(abs2, X .- X̄)
            _order_row(("$(dim)D", kind, p), r, errs, (5, 9, 11);
                gate = kind === :random ? nothing : 1.8)
            @test all(errs[2:end] .< 0.5 .* errs[1:(end - 1)])
            @test errs[end] < 1e-2
            kind === :random || @test r > 1.8
        end
    end
end

@testset "Corner oracle" begin
    C = Main.GhostCorners
    @testset "Convergence" begin
        # Every closure, mixed form and mesh kind on four levels. The one-sided corner closure
        # with the centred mixed form on graded meshes is held to second order, the rest to
        # first.
        levels = 4
        _order_header("Corner oracle, aniso_robin: discrete H¹ error, $levels levels from 17 \
                       points per axis", ("closure", "mixed", "mesh"), (20, 16, 9))
        for closure in C.CLOSURES, mixed in C.MIXED, kind in C.MESH_KINDS
            hs, errs = C.observed_errors(:aniso_robin, kind; closure, mixed, levels)
            r = C.observed_order(:aniso_robin, kind; closure, mixed, levels)
            gate = (closure, mixed, kind) == (:facet_onesided, :centered, :graded) ? 1.8 : 1.0
            _order_row((closure, mixed, kind), r, errs, (20, 16, 9); gate)
            @test all(diff(errs) .< 0)
            @test errs[end] < errs[1] / 8
            @test r > gate
        end
    end
end

end # module ConvergenceGhostOraclesTests
