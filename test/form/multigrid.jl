module TestFormMultigrid

using Test
using Bramble
using Bramble: GeometricMeshHierarchy, set_markers!, spacings
using Random

# Geometric multigrid (gpena/Bramble.jl#329). Meshes are non-uniform throughout: on a uniform
# mesh rebuilding each level from the domain would nest too, and hide a hierarchy that does
# not take every other point.

const MG_SEED = 3291

# The markers a mesh gets from `Ω` at its own points, for comparison with the carried ones.
function _mg_reevaluated_markers(Ωₕ, Ω)
    Ωr = copy(Ωₕ)
    set_markers!(Ωr, markers(Ω); warn_marker_mismatch = false)
    return markers(Ωr)
end

_mg_axes(p::AbstractVector{<:Number}) = (p,)
_mg_axes(p::Tuple) = p

@testset "gmg: nested hierarchy" begin
    Random.seed!(MG_SEED)
    cases = (
        (domain(interval(0.0, 1.0), :inlet => :left, :near => x -> x[1] < 0.3), 33, 4),
        (
            domain(interval(0.0, 1.0) × interval(0.0, 2.0), :walls => (:top, :bottom),
                :blob => x -> (x[1] - 0.5)^2 + (x[2] - 1.0)^2 < 0.2),
            (17, 33),
            3),
        (
            domain(interval(0.0, 1.0) × interval(-1.0, 1.0) × interval(0.0, 2.0),
                :inlet => :left, :half => x -> x[3] < 1.0),
            (9, 5, 17),
            3)
    )
    for (Ω, n, L) in cases
        Ωf = mesh(Ω, n, false)
        H = GeometricMeshHierarchy(Ωf, L)
        @test length(H) == L && lastindex(H) == L && firstindex(H) == 1
        @test H[end] === Ωf
        @test all(Ωₗ -> Ωₗ isa typeof(Ωf), H)
        @test collect(H) == [H[l] for l in 1:L]
        for l in 1:(L - 1)
            pc, pf = _mg_axes(points(H[l])), _mg_axes(points(H[l + 1]))
            # Exact equality: the coarse points are copies of the fine ones, not recomputed.
            @test all(d -> pc[d] == pf[d][1:2:end], eachindex(pc))
            @test npoints(H[l], Tuple) == map(k -> (k - 1) ÷ 2 + 1, npoints(H[l + 1], Tuple))
            # Cached metrics follow the new points, not the uniform ones built first.
            @test all(d -> _mg_axes(spacings(H[l]))[d] ≈ [pc[d][2] - pc[d][1]; diff(pc[d])], eachindex(pc))
        end
        @test npoints(H[1], Tuple) == map(k -> (k - 1) ÷ 2^(L - 1) + 1, npoints(Ωf, Tuple))
        # Custom markers carry over, and agree with re-evaluating the domain at each level.
        for l in 1:L
            @test markers(H[l]) == _mg_reevaluated_markers(H[l], Ω)
        end
        @test keys(markers(H[1])) == keys(markers(Ωf))
        # A space on the coarsest level sees its own points.
        @test ndofs(gridspace(H[1])) == npoints(H[1])
    end

    # The element type and backend of the finest mesh are kept on every level.
    Ω32 = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (9, 17), false)
    H32 = GeometricMeshHierarchy(Ω32, 4)
    @test all(Ωₗ -> eltype(Ωₗ) == Float32 && backend(Ωₗ) == backend(Ω32), H32)
    @test npoints(H32[1], Tuple) == (2, 3)

    # One level is the mesh alone.
    Ω1 = mesh(domain(interval(0.0, 1.0)), 10, false)
    @test only(collect(GeometricMeshHierarchy(Ω1, 1))) === Ω1

    # A collapsed axis stays one point.
    Ωc = mesh(domain(interval(0.0, 1.0) × interval(0.5, 0.5)), (9, 4), false)
    @test npoints(GeometricMeshHierarchy(Ωc, 3)[1], Tuple) == (3, 1)

    # (n - 1) not divisible by 2^(levels - 1) on some axis, and levels < 1.
    @test_throws ArgumentError GeometricMeshHierarchy(mesh(domain(interval(0.0, 1.0)), 18, false), 2)
    @test_throws ArgumentError GeometricMeshHierarchy(
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 11), false), 3)
    @test_throws ArgumentError GeometricMeshHierarchy(
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9, 7), false), 3)
    @test_throws ArgumentError GeometricMeshHierarchy(Ω1, 0)
    Ω33 = mesh(domain(interval(0.0, 1.0)), 33, false)
    @test_throws ArgumentError GeometricMeshHierarchy(Ω33, 65)
    @test eachindex(GeometricMeshHierarchy(Ω33, 2)) == 1:2

    @test sprint(show, GeometricMeshHierarchy(Ω32, 3)) ==
          "GeometricMeshHierarchy{2D, 3 levels, (3, 5) to (9, 17) pts}"
end

end # module TestFormMultigrid
