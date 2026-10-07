module MeshMeshStateTests

using Test
using Random
using Bramble
using Bramble: markers, markers!, index_in_marker, is_uniform, change_points!, set_points!,
               iterative_refinement!, points, npoints, spacing, half_spacing, cell_measure,
               point, indices, half_points, spacings, half_spacings, _walk_mesh,
               _marker_words, _marker_id, _mesh_version, Mesh1D, Mesh1DState, MeshnDState
using ..TestUtils: alloc_test, @test_allocs

# The immutable mesh state (gpena/Bramble.jl#437): geometry, version, uniformity flag and
# marker words, held as plain arrays and isbits values; label-set changes rebuild it whole.

# Every leaf of `x` is an isbits value or a dense array of an isbits element type.
function plain_leaves(x)
    (x isa AbstractDict || x isa BitArray || x isa Symbol || x isa AbstractString) && return false
    x isa DenseArray && return isbitstype(eltype(x))
    isbits(x) && return true
    ismutable(x) && return false
    return all(i -> !isdefined(x, i) || plain_leaves(getfield(x, i)), 1:nfields(x))
end

# Column `_marker_id(Ωₕ, l)` of the word matrix holds the chunks of `markers(Ωₕ)[l]`.
function words_agree(Ωₕ)
    W = _marker_words(_walk_mesh(Ωₕ))
    return W isa Matrix{UInt64} && size(W, 2) == length(markers(Ωₕ)) &&
           all(((l, bv),) -> W[:, _marker_id(Ωₕ, l)] == bv.chunks, markers(Ωₕ))
end

I = interval(0.0, 1.0)
function _meshes()
    (Random.seed!(41);
        (
            mesh(domain(I, :l => :left, :c => x -> x < 0.4), 70, false),
            mesh(domain(I × I, :l => :left, :c => x -> x[1] + x[2] < 0.7), (9, 8), (false, false)),
            mesh(domain(I × I × I, :c => x -> x[1] < 0.5), (5, 4, 6), (false, true, false))))
end

@testset "Mesh state (#437)" begin
    @testset "State: plain, immutable, typed" begin
        # Positive control: the predicate rejects a Dict and a mutable struct.
        @test !plain_leaves((1, Dict(:a => 1))) && !plain_leaves((Ref(1),))
        for Ωₕ in _meshes()
            s = _walk_mesh(Ωₕ)
            @test !ismutable(s)
            @test plain_leaves(s)
            @test s isa Bramble.AbstractMeshType{Bramble.dim(Ωₕ)}
            @test _walk_mesh(s) === s
        end
        @test _walk_mesh(_meshes()[1]) isa Mesh1DState
        @test _walk_mesh(_meshes()[2]) isa MeshnDState{2}
    end

    @testset "State answers the mesh accessors" begin
        for Ωₕ in _meshes()
            s = _walk_mesh(Ωₕ)
            @test npoints(s) == npoints(Ωₕ) && indices(s) == indices(Ωₕ)
            @test points(s) == points(Ωₕ) && spacings(s) == spacings(Ωₕ)
            @test half_spacings(s) == half_spacings(Ωₕ)
            @test _mesh_version(s) == _mesh_version(Ωₕ)
            @test is_uniform(s) == is_uniform(Ωₕ)
            for idx in indices(Ωₕ)
                i = Bramble.dim(Ωₕ) == 1 ? idx[1] : idx
                @test point(s, i) == point(Ωₕ, i)
                @test spacing(s, i) == spacing(Ωₕ, i)
                @test half_spacing(s, i) == half_spacing(Ωₕ, i)
                @test cell_measure(s, i) == cell_measure(Ωₕ, i)
            end
        end
    end

    @testset "Uniform flag follows the points" begin
        Ωₕ = mesh(domain(I), 11, true)
        @test _walk_mesh(Ωₕ).uniform
        s0 = _walk_mesh(Ωₕ)
        @test is_uniform(Ωₕ) && _walk_mesh(Ωₕ) === s0   # a query never writes the mesh
        p = collect(points(Ωₕ))
        p[4] += 0.02
        change_points!(Ωₕ, p)
        @test !_walk_mesh(Ωₕ).uniform && !is_uniform(Ωₕ)
        @test is_uniform(Ωₕ; tol = 1.0)
        change_points!(Ωₕ, collect(range(0.0, 1.0; length = 11)))
        @test _walk_mesh(Ωₕ).uniform
        @test !hasfield(Mesh1D, :_uniform_cache)
    end

    @testset "Point changes: aliasing and version" begin
        # Asserted: `I` is a non-const global, so inference sees `Mesh1D` or `MeshnD` here,
        # and `set_points!` has no `MeshnD` method (JET on the test code).
        Ωₕ = _meshes()[1]::Mesh1D
        s0 = _walk_mesh(Ωₕ)
        p = collect(points(Ωₕ))
        p[2] = (p[1] + p[2]) / 2
        change_points!(Ωₕ, p)
        s1 = _walk_mesh(Ωₕ)
        # Same length: the arrays are updated in place, the version tells staleness.
        @test points(s1) === points(s0) && points(s1) == p
        @test _mesh_version(s1) == _mesh_version(s0) + 1
        @test s1.uid == s0.uid
        # New length: new arrays and indices.
        q = collect(range(0.0, 1.0; length = 9))
        set_points!(Ωₕ, q)
        s2 = _walk_mesh(Ωₕ)
        @test points(s2) === q && npoints(s2) == 9 && length(indices(s2)) == 9
        @test points(s0) !== q && length(points(s0)) == 70
    end

    @testset "Word matrix: build and rebuild" begin
        for Ωₕ in _meshes()
            @test words_agree(Ωₕ)
            W0 = _marker_words(Ωₕ)
            W0copy = copy(W0)
            mm = copy(markers(Ωₕ))
            mm[:extra] = .!mm[:c]
            delete!(mm, :c)
            markers!(Ωₕ, mm)
            @test haskey(markers(Ωₕ), :extra) && !haskey(markers(Ωₕ), :c)
            @test words_agree(Ωₕ)
            # Rebuilt, never resized in place: the old matrix is untouched.
            @test _marker_words(Ωₕ) !== W0 && W0 == W0copy
            @test_throws KeyError _marker_id(Ωₕ, :c)
        end
    end

    @testset "Refinement rebuilds the words" begin
        for Ωₕ in _meshes()
            D = Bramble.dim(Ωₕ)
            dm = markers(domain(reduce(×, ntuple(_ -> I, D)), :r => :right,
                :far => x -> x[1] > 0.3))
            n0 = npoints(Ωₕ)
            iterative_refinement!(Ωₕ, dm)
            @test npoints(Ωₕ) > n0 && haskey(markers(Ωₕ), :far)
            @test !haskey(markers(Ωₕ), :c)
            @test size(_marker_words(Ωₕ), 1) == cld(npoints(Ωₕ), 64)
            @test words_agree(Ωₕ)
        end
    end

    @testset "GMG coarsening keeps the words" begin
        for Ωₕ in _meshes()
            Ωc = Bramble._coarsen_mesh(Ωₕ)
            @test npoints(Ωc) < npoints(Ωₕ)
            @test keys(markers(Ωc)) == keys(markers(Ωₕ))
            @test words_agree(Ωc)
        end
    end

    @testset "Identity: stable and distinct" begin
        a, b = _meshes()[2], _meshes()[2]
        @test isbits(_walk_mesh(a).uid)
        @test _walk_mesh(a).uid == _walk_mesh(a).uid
        @test _walk_mesh(a).uid != _walk_mesh(b).uid
        @test _walk_mesh(copy(a)).uid != _walk_mesh(a).uid
        @test _walk_mesh(copy(a(1))).uid != _walk_mesh(a(1)).uid
        # A deepcopy is an independent mesh: after one point change each, the original
        # and the copy hold different points, so they must not share (uid, version).
        Ω1 = mesh(domain(I), 5, true)
        d1 = deepcopy(Ω1)
        change_points!(Ω1, [0.0, 0.1, 0.5, 0.7, 1.0])
        change_points!(d1, [0.0, 0.4, 0.5, 0.9, 1.0])
        s, t = _walk_mesh(Ω1), _walk_mesh(d1)
        @test s.version == t.version && points(Ω1) != points(d1)
        @test s.uid != t.uid
        W = gridspace(a)
        Wd = deepcopy(W)
        @test Bramble.mesh(Wd) !== a
        @test _walk_mesh(Bramble.mesh(Wd)).uid != _walk_mesh(a).uid
        @test all(i -> _walk_mesh(Bramble.mesh(Wd)(i)).uid != _walk_mesh(a(i)).uid, 1:2)
        @test _mesh_version(Bramble.mesh(Wd)) == _mesh_version(a)
        @test points(Bramble.mesh(Wd)) == points(a)
        # A copied walk state keeps its uid: the in-task rebuild case.
        @test deepcopy(_walk_mesh(a)).uid == _walk_mesh(a).uid
        # Kept across a point change.
        u = _walk_mesh(a).uid
        change_points!(a, map(collect, points(a)))
        @test _walk_mesh(a).uid == u
    end

    @testset "Rebackend shares or copies arrays" begin
        Ωₕ = _meshes()[1]::Mesh1D   # as above: `_rebackend` has no `MeshnD` method
        be = Bramble.backend(Ωₕ)
        c = Bramble._rebackend(Ωₕ, be, identity)
        @test points(c) === points(Ωₕ) && spacings(c) === spacings(Ωₕ)
        @test _mesh_version(c) == _mesh_version(Ωₕ) && !_walk_mesh(c).uniform
        @test markers(c) === markers(Ωₕ) && words_agree(c)
        @test _walk_mesh(c).uid != _walk_mesh(Ωₕ).uid
        d = Bramble._rebackend(Ωₕ, be, copy)
        @test points(d) == points(Ωₕ) && points(d) !== points(Ωₕ)
    end

    @testset "State access allocates nothing" begin
        for Ωₕ in _meshes()
            @test_allocs _walk_mesh(Ωₕ)
            @test_allocs _mesh_version(_walk_mesh(Ωₕ))
            @test_allocs is_uniform(Ωₕ)
            @test_allocs index_in_marker(Ωₕ, :c)
            @test_allocs _marker_id(Ωₕ, :c)
        end
    end
end

end # module MeshMeshStateTests
