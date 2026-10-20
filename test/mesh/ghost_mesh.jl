module MeshGhostMeshTests

using Test
using Random: MersenneTwister
using Bramble
using Bramble: GhostMesh, nghosts, ghost_index, boundary_node, is_absent, has_ghosts,
               physical_mesh, physical_indices, padded_index, boundary_indices,
               index_in_marker, markers, _surface_mask, _surface_weight, _walk_mesh,
               _mesh_version, change_points!, half_spacing, spacing, forward_spacing
using ..TestUtils: WITH_3D_TESTS, @test_allocs

# A ghost mesh pads only the facets it names, and maps each ghost to its facet node in
# O(1). Its padded mesh is a real mesh whose points follow the physical mesh's
# `change_points!`; the labels and facet queries name the physical nodes.

# The error `f` throws, or `nothing`.
function thrown(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

_unit(::Val{D}, d) where {D} = CartesianIndex(ntuple(e -> Int(e == d), Val(D)))

# The padded index of the ghost `g`, from the maps alone.
function ghost_position(G, g, ::Val{D}) where {D}
    I, d, side = boundary_node(G, g)
    return padded_index(G, I) + (side === :min ? -1 : 1) * _unit(Val(D), d)
end

# Every padded index is exactly one of a physical node, a ghost, or absent, and the ghosts
# number `nghosts(G)`, each at its own index.
function partition_is_exact(G, ::Val{D}) where {D}
    phys = physical_indices(G)
    ghost_at = [ghost_position(G, g, Val(D)) for g in 1:nghosts(G)]
    length(unique(ghost_at)) == nghosts(G) || return false
    any(J -> J in phys || is_absent(G, J), ghost_at) && return false
    others = count(J -> !(J in phys) && !is_absent(G, J), CartesianIndices(npoints(G, Tuple)))
    return others == nghosts(G)
end

# The coordinates of the padded axis from the physical ones, ghost at `x₁ - h₂`, `xₙ + hₙ`.
function extended(x, lo::Bool, hi::Bool)
    out = collect(x)
    lo && pushfirst!(out, x[1] - (x[2] - x[1]))
    hi && push!(out, x[end] + (x[end] - x[end - 1]))
    return out
end

@testset "Ghost mesh" begin
    @testset "1D" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
        G = GhostMesh(Ωₕ; ghosts = :boundary)
        @test G isa Bramble.AbstractMeshType{1}
        @test npoints(G, Tuple) == (11,)
        @test npoints(G) == 11
        @test nghosts(G) == 2
        @test physical_mesh(G) === Ωₕ
        x = collect(points(Ωₕ))
        @test collect(points(G)) ≈ extended(x, true, true)
        @test collect(points(G))[2:(end - 1)] == x
        @test boundary_node(G, 1) == (CartesianIndex(1), 1, :min)
        @test boundary_node(G, 2) == (CartesianIndex(9), 1, :max)
        @test ghost_index(G, CartesianIndex(1), 1, :min) == 1
        @test ghost_index(G, (9,), 1, :max) == 2
        @test !any(J -> is_absent(G, J), CartesianIndices((11,)))
        @test partition_is_exact(G, Val(1))
        @test G(1) === G(1)
        @test G(1) !== Ωₕ
        @test collect(points(Ωₕ)) == x
    end

    @testset "2D counts and maps" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, true))
        G = GhostMesh(Ωₕ; ghosts = :boundary)
        @test npoints(G, Tuple) == (7, 9)
        @test npoints(G) == 63
        @test nghosts(G) == 2 * 5 + 2 * 7
        @test size(Bramble.indices(G)) == (7, 9)
        @test count(J -> is_absent(G, J), CartesianIndices((7, 9))) == 4
        @test all(J -> is_absent(G, J), CartesianIndex.([1, 7, 1, 7], [1, 1, 9, 9]))
        @test partition_is_exact(G, Val(2))
        @test all(1:nghosts(G)) do g
            ghost_index(G, boundary_node(G, g)...) == g
        end
        # Every facet node maps to a ghost number, and back.
        for d in 1:2, (side, edge) in ((:min, 1), (:max, size(Ωₕ, d)))

            for I in CartesianIndices(npoints(Ωₕ, Tuple))
                I[d] == edge || continue
                g = ghost_index(G, I, d, side)
                @test boundary_node(G, g) == (I, d, side)
            end
        end
        # The ghost DOF order: facets xmin, xmax, ymin, ymax, each in column-major order.
        order = [boundary_node(G, g) for g in 1:nghosts(G)]
        @test order[1:7] == [(CartesianIndex(1, j), 1, :min) for j in 1:7]
        @test order[8:14] == [(CartesianIndex(5, j), 1, :max) for j in 1:7]
        @test order[15:19] == [(CartesianIndex(i, 1), 2, :min) for i in 1:5]
        @test order[20:24] == [(CartesianIndex(i, 7), 2, :max) for i in 1:5]
    end

    @testset "Smallest axes" begin
        # Two points on an axis: the two facets of it are different nodes.
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 2), false)
        G = GhostMesh(Ωₕ; ghosts = :boundary)
        @test npoints(G, Tuple) == (5, 4)
        @test [boundary_node(G, g) for g in 1:nghosts(G)] == [
            (CartesianIndex(1, 1), 1, :min), (CartesianIndex(1, 2), 1, :min),
            (CartesianIndex(3, 1), 1, :max), (CartesianIndex(3, 2), 1, :max),
            (CartesianIndex(1, 1), 2, :min), (CartesianIndex(2, 1), 2, :min),
            (CartesianIndex(3, 1), 2, :min),
            (CartesianIndex(1, 2), 2, :max), (CartesianIndex(2, 2), 2, :max),
            (CartesianIndex(3, 2), 2, :max)]
        @test partition_is_exact(G, Val(2))
        @test all(g -> ghost_index(G, boundary_node(G, g)...) == g, 1:nghosts(G))
        Ω1 = mesh(domain(interval(0.0, 1.0)), 2, true)
        G1 = GhostMesh(Ω1; ghosts = :boundary)
        @test collect(points(G1)) ≈ [-1.0, 0.0, 1.0, 2.0]
        @test nghosts(G1) == 2
    end

    @testset "Named facets only" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, true))
        Gx = GhostMesh(Ωₕ; ghosts = (:xmin,))
        @test npoints(Gx, Tuple) == (6, 7)
        @test nghosts(Gx) == 7
        @test has_ghosts(Gx, 1, :min)
        @test !has_ghosts(Gx, 1, :max) && !has_ghosts(Gx, 2, :min)
        @test !any(J -> is_absent(Gx, J), CartesianIndices((6, 7)))
        @test partition_is_exact(Gx, Val(2))
        # `x` is padded below only, `y` not at all.
        @test collect(points(Gx(1))) ≈ extended(collect(points(Ωₕ(1))), true, false)
        @test collect(points(Gx(2))) == collect(points(Ωₕ(2)))

        Gxy = GhostMesh(Ωₕ; ghosts = (:xmax, :ymin))
        @test npoints(Gxy, Tuple) == (6, 8)
        @test nghosts(Gxy) == 7 + 5
        @test count(J -> is_absent(Gxy, J), CartesianIndices((6, 8))) == 1
        @test boundary_node(Gxy, 1) == (CartesianIndex(5, 1), 1, :max)
        @test boundary_node(Gxy, 8) == (CartesianIndex(1, 1), 2, :min)
        @test partition_is_exact(Gxy, Val(2))

        # A facet by its alias, a single name, and none at all.
        @test npoints(GhostMesh(Ωₕ; ghosts = (:left,)), Tuple) == (6, 7)
        @test npoints(GhostMesh(Ωₕ; ghosts = :ymax), Tuple) == (5, 8)
        G0 = GhostMesh(Ωₕ; ghosts = ())
        @test npoints(G0, Tuple) == (5, 7) && nghosts(G0) == 0
        @test collect.(points(G0)) == collect.(points(Ωₕ))
        @test_throws ArgumentError ghost_index(Gx, CartesianIndex(5, 1), 1, :max)
        @test_throws ArgumentError ghost_index(Gx, CartesianIndex(1, 1), 2, :min)
    end

    WITH_3D_TESTS && @testset "3D" begin # 3D meshes are built in `slow` and nightly only
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (4, 5, 6), (false, false, false))
        G = GhostMesh(Ωₕ; ghosts = :boundary)
        @test npoints(G, Tuple) == (6, 7, 8)
        @test nghosts(G) == 2 * (5 * 6 + 4 * 6 + 4 * 5) == 148
        @test count(J -> is_absent(G, J), CartesianIndices((6, 7, 8))) == 68
        @test partition_is_exact(G, Val(3))
        @test all(g -> ghost_index(G, boundary_node(G, g)...) == g, 1:nghosts(G))
        order = [boundary_node(G, g) for g in 1:nghosts(G)]
        @test order[1] == (CartesianIndex(1, 1, 1), 1, :min)
        @test order[2] == (CartesianIndex(1, 2, 1), 1, :min)
        @test order[30] == (CartesianIndex(1, 5, 6), 1, :min)
        @test order[31] == (CartesianIndex(4, 1, 1), 1, :max)
        @test order[end] == (CartesianIndex(4, 5, 6), 3, :max)

        # A single facet pads one axis and leaves no absent point.
        Gz = GhostMesh(Ωₕ; ghosts = (:zmin,))
        @test npoints(Gz, Tuple) == (4, 5, 7)
        @test nghosts(Gz) == 20
        @test !any(J -> is_absent(Gz, J), CartesianIndices((4, 5, 7)))
        @test partition_is_exact(Gz, Val(3))
        # Two facets of different axes: one absent edge.
        Gxy = GhostMesh(Ωₕ; ghosts = (:xmin, :ymax))
        @test npoints(Gxy, Tuple) == (5, 6, 6)
        @test nghosts(Gxy) == 5 * 6 + 4 * 6
        @test count(J -> is_absent(Gxy, J), CartesianIndices((5, 6, 6))) == 6
        @test partition_is_exact(Gxy, Val(3))
    end

    @testset "Ghost coordinates" begin
        rng = MersenneTwister(7)
        pts(n, lo, hi) = sort!([lo; lo .+ (hi - lo) .* rand(rng, n - 2); hi])
        for unif in ((true, true), (true, false), (false, true), (false, false))
            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(-1.0, 2.0)), (8, 9), unif)
            Bramble.change_points!(Ωₕ,
                (unif[1] ? collect(points(Ωₕ(1))) : pts(8, 0.0, 1.0),
                    unif[2] ? collect(points(Ωₕ(2))) : pts(9, -1.0, 2.0)))
            G = GhostMesh(Ωₕ; ghosts = :boundary)
            for d in 1:2
                x = collect(points(Ωₕ(d)))
                xp = collect(points(G(d)))
                @test xp == extended(x, true, true)
                @test xp[1] ≈ x[1] - spacing(Ωₕ(d), 2)
                @test xp[end] ≈ x[end] + forward_spacing(Ωₕ(d), length(x))
                @test issorted(xp; lt = <=)
            end
            # The padded mesh's own spacing at a physical node is the physical one.
            @test spacing(G, (3, 4), 1) == spacing(Ωₕ, (2, 4), 1)
            @test spacing(G, (3, 4), 2) == spacing(Ωₕ, (2, 3), 2)
        end
        # A uniform axis stays uniform when padded.
        Ωu = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (8, 9), (true, false))
        Gu = GhostMesh(Ωu; ghosts = :boundary)
        @test Bramble.is_uniform(Gu(1))
        @test !Bramble.is_uniform(Gu(2))
        # The submeshes are the padded mesh's own, and the physical ones keep their points.
        @test Gu(1) !== Ωu(1) && Gu(2) !== Ωu(2)
        @test npoints(Ωu, Tuple) == (8, 9)
    end

    @testset "Follows change_points!" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, true))
        G = GhostMesh(Ωₕ; ghosts = (:xmin, :ymax))
        x = collect(points(Ωₕ(1)))
        y = collect(points(Ωₕ(2)))
        v0 = _mesh_version(G)
        walk0 = _walk_mesh(G)
        Bramble.change_points!(Ωₕ, (x .^ 2, y))
        @test collect(points(G(1))) ≈ extended(x .^ 2, true, false)
        @test collect(points(G(1)))[2:end] == x .^ 2
        @test collect(points(G(2))) ≈ extended(y, false, true)
        @test _mesh_version(G) > v0
        # The walk reads the moved points.
        @test collect(points(_walk_mesh(G)(1))) == collect(points(G(1)))
        @test collect(points(walk0(1))) == collect(points(G(1)))
        # And the mesh geometry the operators read: spacings at a physical node.
        @test spacing(G, (2, 3), 1) ≈ x[2]^2 - x[1]^2
        @test half_spacing(G, (3, 3))[1] ≈ (x[3]^2 - x[1]^2) / 2
        # The physical end node has a full cell on the padded mesh, a half cell on the other.
        @test half_spacing(G, (2, 3))[1] ≈ x[2]^2 - x[1]^2
        @test forward_spacing(G, (6, 3), 1) ≈ x[5]^2 - x[4]^2
        # Moving only one axis, then the other.
        Bramble.change_points!(Ωₕ, (x .^ 2, y .^ 2 ./ 2))
        @test collect(points(G(2)))[1:(end - 1)] == y .^ 2 ./ 2
        @test collect(points(G(2)))[end] ≈ y[end]^2 / 2 + (y[end]^2 - y[end - 1]^2) / 2
        @test collect(points(physical_mesh(G)(1))) == x .^ 2
        # A moved 1D mesh.
        Ω1 = mesh(domain(interval(0.0, 1.0)), 6, false)
        G1 = GhostMesh(Ω1; ghosts = :boundary)
        x1 = collect(points(Ω1))
        Bramble.change_points!(Ω1, x1 .^ 3)
        @test collect(points(G1)) ≈ extended(x1 .^ 3, true, true)
    end

    @testset "Refinement makes it stale" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, true))
        G = GhostMesh(Ωₕ; ghosts = :boundary)
        Bramble.iterative_refinement!(Ωₕ)
        @test_throws ArgumentError points(G)
        @test_throws ArgumentError G(1)
        @test_throws ArgumentError _walk_mesh(G)
        @test_throws ArgumentError Bramble.locate_cell(G, (0.5, 0.5))
        err = thrown(() -> points(G))
        @test err isa ArgumentError
        @test occursin("refined", err.msg) && occursin("gridspace", err.msg)
        # A fresh ghost mesh on the refined mesh works.
        G2 = GhostMesh(Ωₕ; ghosts = :boundary)
        @test npoints(G2, Tuple) == npoints(Ωₕ, Tuple) .+ 2
    end

    @testset "Labels name physical facets" begin
        Ωₕ = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 2.0), :left => :xmin,
                :disk => x -> (x[1] - 0.5)^2 + (x[2] - 1.0)^2 < 0.3),
            (6, 7), (false, true))
        G = GhostMesh(Ωₕ; ghosts = :boundary)
        pn = npoints(G, Tuple)
        lin = LinearIndices(pn)
        phys = CartesianIndices((6, 7))
        shift = CartesianIndex(1, 1)
        onx = [lin[I + shift] for I in phys if I[1] == 1]
        for label in (:xmin, :left)
            bits = index_in_marker(G, label)
            @test length(bits) == prod(pn)
            @test findall(bits) == sort(onx)
        end
        bnd = index_in_marker(G, :boundary)
        @test count(bnd) == 2 * (6 + 7) - 4
        @test all(I -> bnd[lin[I + shift]] == (I[1] in (1, 6) || I[2] in (1, 7)), phys)
        # No ghost and no absent point carries a label.
        outside = [J for J in CartesianIndices(pn) if !(J in physical_indices(G))]
        for label in (:boundary, :interior, :left, :disk)
            @test !any(J -> markers(G)[label][lin[J]], outside)
        end
        @test count(index_in_marker(G, :interior)) == (6 - 2) * (7 - 2)
        # The physical mesh's own label keeps its nodes, at their padded indices.
        disk = findall(index_in_marker(Ωₕ, :disk))
        @test !isempty(disk)
        padded_disk = findall(markers(G)[:disk])
        @test padded_disk == sort([lin[phys[k] + shift] for k in disk])
        @test all(padded_disk) do k
            collect(Bramble.point(G, CartesianIndices(pn)[k])) ==
            collect(Bramble.point(Ωₕ, CartesianIndices(pn)[k] - shift))
        end
        # Stored index-derived markers, never predicates.
        @test isempty(Bramble._predicate_labels(G(1))) && isempty(Bramble._predicate_labels(G(2)))
        # The facets are the physical ones.
        faces = boundary_indices(G)
        @test faces[1] == CartesianIndices((2:2, 2:8)) && faces[2] == CartesianIndices((7:7, 2:8))
        @test faces[3] == CartesianIndices((2:7, 2:2)) && faces[4] == CartesianIndices((2:7, 8:8))
        @test Bramble.is_boundary_index(G, CartesianIndex(2, 3))
        @test !Bramble.is_boundary_index(G, CartesianIndex(1, 2))
        @test !Bramble.is_boundary_index(G, CartesianIndex(3, 3))
        @test Bramble.interior_indices(G) == CartesianIndices((3:6, 3:7))
    end

    @testset "Labels follow the physical mesh" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, true))
        G = GhostMesh(Ωₕ; ghosts = :boundary)
        @test !haskey(markers(G), :corner)
        stamp0 = Bramble._marker_stamp(G)
        mm = copy(markers(Ωₕ))
        bits = falses(35)
        bits[1] = true
        mm[:corner] = bits
        Bramble.markers!(Ωₕ, mm)
        @test haskey(markers(G), :corner)
        @test findall(markers(G)[:corner]) == [LinearIndices((7, 9))[2, 2]]
        # The stamp changed with the store, so a cached marker read notices.
        @test Bramble._marker_stamp(G) != stamp0
    end

    @testset "Surface masks and weights" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0), :left => :xmin),
            (5, 7), (false, true))
        G = GhostMesh(Ωₕ; ghosts = :boundary)
        for labels in ((:boundary,), (:xmin,), (:left,), (:ymax,), (:xmin, :ymax))
            @test _surface_mask(G, labels) == _surface_mask(Ωₕ, labels)
        end
        shift = CartesianIndex(1, 1)
        for labels in ((:boundary,), (:xmin,), (:xmin, :ymax))
            mask = _surface_mask(G, labels)
            for J in CartesianIndices(npoints(G, Tuple))
                w = _surface_weight(G, mask, J)
                if J in physical_indices(G)
                    @test w == _surface_weight(Ωₕ, mask, J - shift)
                else
                    @test w == 0
                end
            end
        end
        # Only whole facets are surfaces: a partial label is refused.
        mm = copy(markers(Ωₕ))
        part = falses(35)
        part[3] = true
        mm[:part] = part
        Bramble.markers!(Ωₕ, mm)
        @test_throws ArgumentError _surface_mask(G, (:part,))
        @test_throws ArgumentError _surface_mask(Ωₕ, (:part,))
        # A face is named whether or not it carries ghosts.
        Gx = GhostMesh(Ωₕ; ghosts = (:xmin,))
        @test _surface_mask(Gx, (:ymax,)) == _surface_mask(Ωₕ, (:ymax,))
        w = _surface_weight(Gx, _surface_mask(Gx, (:ymax,)), CartesianIndex(2, 7))
        @test w == _surface_weight(Ωₕ, _surface_mask(Ωₕ, (:ymax,)), CartesianIndex(1, 7))
        # Too few points for a thin surface: counted on the physical axis, not the padded one.
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (2, 4), false)
        G2 = GhostMesh(Ω2; ghosts = :boundary)
        @test_throws ArgumentError Bramble._check_surface_is_thin(G2, ((true, true), (false, false)))
        @test Bramble._check_surface_is_thin(G2, ((false, false), (true, true))) === nothing
    end

    @testset "Refusals" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, true))
        G = GhostMesh(Ωₕ; ghosts = (:xmin,))
        @test_throws ArgumentError GhostMesh(Ωₕ; ghosts = (:zmin,))
        @test_throws ArgumentError GhostMesh(Ωₕ; ghosts = (:nowhere,))
        @test_throws ArgumentError GhostMesh(Ωₕ; ghosts = (1,))
        err = thrown(() -> GhostMesh(Ωₕ; ghosts = (:nowhere,)))
        @test occursin(":xmin", err.msg) && occursin(":boundary", err.msg)
        # The facet carries no ghosts: the message names the fix.
        err = thrown(() -> ghost_index(G, CartesianIndex(5, 2), 1, :max))
        @test err isa ArgumentError && occursin("ghosts =", err.msg)
        # The node is not on the facet.
        @test_throws ArgumentError ghost_index(G, CartesianIndex(2, 2), 1, :min)
        @test_throws ArgumentError ghost_index(G, CartesianIndex(1, 2), 1, :middle)
        @test_throws BoundsError ghost_index(G, (1, 99), 1, :min)
        @test_throws ArgumentError boundary_node(G, 0)
        @test_throws ArgumentError boundary_node(G, nghosts(G) + 1)
        # An axis too short to place a ghost from.
        Ωc = mesh(domain(interval(0.0, 1.0) × interval(2.0, 2.0)), (5, 1), true)
        @test npoints(Ωc, Tuple) == (5, 1)
        @test_throws ArgumentError GhostMesh(Ωc; ghosts = :boundary)
        @test_throws ArgumentError GhostMesh(Ωc; ghosts = (:ymin,))
        @test npoints(GhostMesh(Ωc; ghosts = (:xmin, :xmax)), Tuple) == (7, 1)
    end

    @testset "Inference and allocations" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, true))
        G = GhostMesh(Ωₕ; ghosts = :boundary)
        I = CartesianIndex(1, 3)
        J = CartesianIndex(3, 3)
        @test @inferred(ghost_index(G, I, 1, :min)) isa Int
        @test @inferred(boundary_node(G, 4)) isa Tuple{CartesianIndex{2}, Int, Symbol}
        @test @inferred(is_absent(G, J)) isa Bool
        @test @inferred(nghosts(G)) isa Int
        @test @inferred(npoints(G, Tuple)) == (7, 9)
        @test @inferred(GhostMesh(Ωₕ; ghosts = :boundary)) isa GhostMesh
        @test @inferred(GhostMesh(Ωₕ; ghosts = (:xmin,))) isa GhostMesh
        @test typeof(GhostMesh(Ωₕ; ghosts = :boundary)) == typeof(GhostMesh(Ωₕ; ghosts = (:xmin,)))
        @test @inferred(boundary_indices(G)) isa NTuple{4, CartesianIndices{2}}
        @test @inferred(_walk_mesh(G)) isa Bramble.GhostMeshState
        @test @inferred(_surface_weight(G, ((true, true), (true, true)), J)) isa Float64
        @test @inferred(half_spacing(G, (3, 3))) isa NTuple{2, Float64}
        @test @inferred(Bramble.point(G, J)) isa NTuple{2, Float64}

        @test_allocs ghost_index(G, I, 1, :min)
        @test_allocs boundary_node(G, 4)
        @test_allocs is_absent(G, J)
        @test_allocs nghosts(G)
        @test_allocs npoints(G, Tuple)
        @test_allocs G(1)
        @test_allocs points(G)
        @test_allocs _walk_mesh(G)
        @test_allocs _mesh_version(G)
        @test_allocs boundary_indices(G)
        @test_allocs _surface_weight(G, ((true, true), (true, true)), J)
        @test_allocs half_spacing(G, (3, 3))
        @test_allocs spacing(G, (3, 3), 1)
        @test_allocs forward_spacing(G, (3, 3), 2)
        @test_allocs Bramble.point(G, J)

        G1 = GhostMesh(mesh(domain(interval(0.0, 1.0)), 6, false); ghosts = :boundary)
        @test @inferred(ghost_index(G1, CartesianIndex(1), 1, :min)) isa Int
        @test @inferred(boundary_node(G1, 2)) isa Tuple{CartesianIndex{1}, Int, Symbol}
        @test_allocs ghost_index(G1, CartesianIndex(6), 1, :max)
        @test_allocs boundary_node(G1, 2)
        @test_allocs points(G1)
    end
end

end # module MeshGhostMeshTests
