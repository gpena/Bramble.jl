module MeshMeshStateTests

using Test
using Random
using Bramble
using Bramble: markers, markers!, index_in_marker, is_uniform, change_points!, set_points!,
               iterative_refinement!, points, npoints, spacing, half_spacing, cell_measure,
               point, indices, half_points, spacings, half_spacings, _walk_mesh,
               _marker_words, _marker_id, _marker_stamp, _mesh_version, Mesh1D, Mesh1DState,
               MeshnDState, set_markers!, restrict_to
using ..TestUtils: alloc_test, WITH_3D_TESTS, @test_allocs

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
# The 3D mesh is built in `slow` and by the nightly run (gpena/Bramble.jl#650); callers index
# only the first two.
function _meshes()
    (Random.seed!(41);
        (
            mesh(domain(I, :l => :left, :c => x -> x < 0.4), 70, false),
            mesh(domain(I × I, :l => :left, :c => x -> x[1] + x[2] < 0.7), (9, 8), (false, false)),
            (WITH_3D_TESTS ?
             (mesh(domain(I × I × I, :c => x -> x[1] < 0.5), (5, 4, 6), (false, true, false)),) :
             ())...))
end

# Sets and predicates for re-marking a mesh after its points move, 3D in `slow` as above.
_c1(x) = x < 0.4
_c2(x) = x[1] + x[2] < 0.7
_c3(x) = x[1] < 0.5
# A second predicate, of another type: re-marking walks a heterogeneous tuple of them.
_d(x) = first(x) > 0.8
_remark_cases() = ((I, _c1), (I × I, _c2), (WITH_3D_TESTS ? ((I × I × I, _c3),) : ())...)
# `f` on each axis of a mesh's points, a vector in 1D and a tuple of vectors otherwise.
_axes_map(f, p) = p isa Tuple ? map(f, p) : f(p)
# `c` at every point of `Ωₕ`, in linear order.
_evaluated(c, Ωₕ) = BitVector(vec([c(point(Ωₕ, i)) for i in indices(Ωₕ)]))

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
        # The mesh carries the predicate label :c, so the three-argument form moves it.
        change_points!(Ωₕ, markers(domain(I, :l => :left, :c => x -> x < 0.4)), p)
        s1 = _walk_mesh(Ωₕ)
        # Same length: the arrays are updated in place, the version tells staleness.
        @test points(s1) === points(s0) && points(s1) == p
        @test _mesh_version(s1) == _mesh_version(s0) + 1
        @test s1.uid == s0.uid
        # New length, new arrays and indices.
        q = collect(range(0.0, 1.0; length = 9))
        # The mesh carries the custom labels :l and :c, which a resize has no domain to
        # re-evaluate, so the refusal comes before anything changes.
        v1 = _mesh_version(Ωₕ)
        @test_throws ArgumentError set_points!(Ωₕ, q)
        @test npoints(Ωₕ) == 70 && _mesh_version(Ωₕ) == v1 && words_agree(Ωₕ)
        mm = Bramble.MeshMarkers()
        mm[:boundary] = markers(Ωₕ)[:boundary]
        mm[:interior] = markers(Ωₕ)[:interior]
        markers!(Ωₕ, mm)
        set_points!(Ωₕ, q)
        s2 = _walk_mesh(Ωₕ)
        @test points(s2) === q && npoints(s2) == 9 && length(indices(s2)) == 9
        @test points(s0) !== q && length(points(s0)) == 70
        # The markers and words follow the new grid; `cld(n, 64)` alone would not tell
        # (70 and 9 points both fit one word), so the lengths are asserted too.
        @test words_agree(Ωₕ)
        @test size(_marker_words(Ωₕ), 1) == cld(9, 64)
        @test length(markers(Ωₕ)[:boundary]) == 9
    end

    @testset "Moving points: predicate labels" begin
        # The 1D probe of the bug: `:c` is `x < 0.5` at the old points, and the squared
        # points put two more of them below 0.5.
        Ω = domain(I, :c => x -> x[1] < 0.5, :l => :left)
        pts = collect(range(0.0, 1.0; length = 7)) .^ 2
        Ωₕ = mesh(Ω, 7, true)
        c0 = copy(markers(Ωₕ)[:c])
        v0, p0 = _mesh_version(Ωₕ), copy(points(Ωₕ))
        @test c0 == Bool[1, 1, 1, 0, 0, 0, 0]
        err = try
            change_points!(Ωₕ, pts)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("change_points!(Ωₕ, markers(Ω), pts)", err.msg)
        @test occursin("(:c,)", err.msg)
        # Refused before anything changed.
        @test points(Ωₕ) == p0 && _mesh_version(Ωₕ) == v0
        @test markers(Ωₕ)[:c] == c0
        # The three-argument form re-evaluates the predicate at the new points.
        change_points!(Ωₕ, markers(Ω), pts)
        @test markers(Ωₕ)[:c] == Bool[1, 1, 1, 1, 1, 0, 0] && words_agree(Ωₕ)
        # ... and a mesh it moved is still refused by the two-argument form.
        @test_throws ArgumentError change_points!(Ωₕ, pts)

        # Face labels are index-based and stay valid, in 1D and 2D.
        Ωf = domain(I, :l => :left, :r => (:left, :right))
        Mf = mesh(Ωf, 7, true)
        change_points!(Mf, pts)
        @test points(Mf) == pts && markers(Mf)[:l] == Bool[1, 0, 0, 0, 0, 0, 0]
        Ω2 = domain(I × I, :l => :left, :b => :bottom)
        M2 = mesh(Ω2, (5, 6), (true, true))
        q = (collect(range(0.0, 1.0; length = 5)) .^ 2,
            collect(range(0.0, 1.0; length = 6)) .^ 2)
        l0 = copy(markers(M2)[:l])
        change_points!(M2, q)
        @test points(M2) == q && markers(M2)[:l] == l0 && words_agree(M2)

        # A 2D predicate is refused, before any axis moves.
        M3 = mesh(domain(I × I, :l => :left, :c => x -> x[1] < 0.5), (5, 6), (true, true))
        v3 = _mesh_version(M3)
        @test_throws ArgumentError change_points!(M3, q)
        @test _mesh_version(M3) == v3
    end

    @testset "Predicate labels follow the mesh" begin
        Ω = domain(I, :c => x -> x[1] < 0.5, :l => :left)
        Ωₕ = mesh(Ω, 7, true)
        pts = collect(range(0.0, 1.0; length = 7)) .^ 2
        # Copies, host mirrors and refinement keep the refusal.
        @test_throws ArgumentError change_points!(copy(Ωₕ), pts)
        @test_throws ArgumentError change_points!(deepcopy(Ωₕ), pts)
        be = Bramble.backend(Ωₕ)
        @test_throws ArgumentError change_points!(Bramble._rebackend(Ωₕ, be, copy), pts)
        r = copy(Ωₕ)
        iterative_refinement!(r, markers(Ω))
        @test_throws ArgumentError change_points!(r, collect(range(0.0, 1.0; length = 13)))
        # A hand-set label set has no predicate to re-evaluate, and a re-evaluation by
        # `set_markers!` of face labels alone clears the record.
        mm = Bramble.MeshMarkers()
        mm[:boundary] = copy(markers(Ωₕ)[:boundary])
        mm[:interior] = copy(markers(Ωₕ)[:interior])
        markers!(Ωₕ, mm)
        change_points!(Ωₕ, pts)
        @test points(Ωₕ) == pts
        Bramble.set_markers!(Ωₕ, markers(domain(I, :l => :left)))
        change_points!(Ωₕ, collect(range(0.0, 1.0; length = 7)))
        @test points(Ωₕ) == range(0.0, 1.0; length = 7)
    end

    @testset "Submesh moves: predicate parent" begin
        # `Ωₕ(1)` is the parent's shared submesh, and the parent's `:c` depends on its
        # points: a move of it alone is refused in every form, before anything changes.
        Ω = domain(I × I, :c => x -> x[1] < 0.5, :l => :left)
        M = mesh(Ω, (7, 5), (true, true))
        q = collect(range(0.0, 1.0; length = 7)) .^ 2
        c0, p0, v0 = copy(markers(M)[:c]), copy(points(M(1))), _mesh_version(M)
        msgs = ["a submesh of a mesh carrying predicate markers (:c,)",
            "change_points!(Ωₕ, markers(Ω), pts)", "points of every axis"]
        @test_throws msgs set_points!(M(1), q)
        @test_throws msgs change_points!(M(1), q)
        @test_throws msgs change_points!(M(1), markers(interval(0.0, 1.0)), q)
        @test points(M(1)) == p0 && _mesh_version(M) == v0 && markers(M)[:c] == c0
        # A copy stamps its own submeshes, and the original stays refused.
        @test_throws ArgumentError set_points!(copy(M)(1), q)
        # The parent-level three-argument call moves both and re-evaluates `:c`.
        qy = collect(range(0.0, 1.0; length = 5))
        change_points!(M, markers(Ω), (q, qy))
        @test markers(M)[:c] == (repeat(q, 5) .< 0.5)
        # `markers!` on the parent drops the predicate labels, and the submesh is free.
        markers!(M, copy(markers(M)))
        set_points!(M(1), collect(range(0.0, 1.0; length = 7)))
        # Face labels and bare parents: submeshes move as before.
        Mf = mesh(domain(I × I, :l => :left), (7, 5), (true, true))
        set_points!(Mf(1), q)
        change_points!(Mf(2), qy .^ 2)
        @test points(Mf(1)) == q && markers(Mf)[:l] == markers(copy(Mf))[:l]
        Mb = mesh(domain(I × I), (7, 5), (true, true))
        set_points!(Mb(1), q)
        @test points(Mb(1)) == q
    end

    @testset "Same count keeps face labels" begin
        Ωₕ = mesh(domain(I, :a => :left), 70, false)::Mesh1D
        a0 = copy(markers(Ωₕ)[:a])
        # x^1.2 keeps both ends and the order of the points, and moves the interior ones.
        p = collect(points(Ωₕ)) .^ 1.2
        set_points!(Ωₕ, p)
        @test points(Ωₕ) == p && npoints(Ωₕ) == 70
        @test markers(Ωₕ)[:a] == a0
        @test words_agree(Ωₕ)
    end

    @testset "set_points! refuses predicate labels" begin
        Ω = domain(I, :c => x -> x < 0.4, :l => :left)
        Ωₕ = mesh(Ω, 70, false)::Mesh1D
        c0, p0, v0 = copy(markers(Ωₕ)[:c]), copy(points(Ωₕ)), _mesh_version(Ωₕ)
        p = p0 .^ 1.2
        @test_throws ["set_points!(Ωₕ, pts)", "(:c,)",
            "change_points!(Ωₕ, markers(Ω), pts)"] set_points!(Ωₕ, p)
        # Refused before anything changed.
        @test points(Ωₕ) == p0 && _mesh_version(Ωₕ) == v0 && markers(Ωₕ)[:c] == c0
        # The three-argument form moves it and re-evaluates the label.
        change_points!(Ωₕ, markers(Ω), p)
        @test points(Ωₕ) == p && markers(Ωₕ)[:c] == (p .< 0.4) && words_agree(Ωₕ)
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
        Ω2 = domain(I × I, :l => :left, :c => x -> x[1] + x[2] < 0.7)
        change_points!(a, markers(Ω2), map(collect, points(a)))
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

    @testset "Re-marking keeps the storage" begin
        for (X, c) in _remark_cases(), k in (1, 4)

            D = Bramble.dim(X)
            Ω = domain(X, :l => :left, :w => (:left, :right), :c => c, :d => _d)
            dm = markers(Ω)
            Ωₕ = D == 1 ? mesh(Ω, 9k, false) : mesh(Ω, ntuple(d -> 3k + d, D), ntuple(_ -> false, D))
            mm, W = markers(Ωₕ), _marker_words(Ωₕ)
            cbits = mm[:c]

            # Uniform points, then their square roots: every point but the ends moves right
            # and `:c` shrinks, so a stale bit would survive without a clear.
            u = _axes_map(q -> collect(range(0.0, 1.0; length = length(q))), points(Ωₕ))
            change_points!(Ωₕ, dm, u)
            c0 = copy(mm[:c])
            p = _axes_map(q -> sqrt.(q), u)
            change_points!(Ωₕ, dm, p)
            @test markers(Ωₕ) === mm && mm[:c] === cbits && _marker_words(Ωₕ) === W
            @test mm[:c] == _evaluated(c, Ωₕ)
            @test mm[:c] != c0
            @test mm[:interior] == .!mm[:boundary] && count(mm[:l]) > 0
            @test words_agree(Ωₕ)

            @test_allocs set_markers!(Ωₕ, dm)
            @test_allocs change_points!(Ωₕ, dm, p)

            # The stamp is still redrawn, so a cached assembly re-reads the emptied region.
            Wₕ = gridspace(Ωₕ)
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(restrict_to(:c, u), v))
            A = assemble(a)
            s0 = _marker_stamp(Ωₕ)
            nowhere = _ -> false
            set_markers!(Ωₕ, markers(domain(X, :l => :left, :w => (:left, :right), :c => nowhere, :d => _d)))
            @test markers(Ωₕ) === mm && mm[:c] == _evaluated(nowhere, Ωₕ)
            @test mm[:c] != _evaluated(c, Ωₕ)
            @test _marker_stamp(Ωₕ) != s0
            assemble!(A, a)
            @test A == assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(restrict_to(:c, u), v)))

            # A different label set still rebuilds the table.
            set_markers!(Ωₕ, markers(domain(X, :r => :right)))
            @test markers(Ωₕ) !== mm && haskey(markers(Ωₕ), :r) && !haskey(markers(Ωₕ), :c)
            @test words_agree(Ωₕ)
        end
    end
end

end # module MeshMeshStateTests
