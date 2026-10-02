module FormThreadedReplayTests

using Test
using Bramble
using Random
using SparseArrays
using SparseArrays: getcolptr
using Bramble: Serial, Parallel, CpuPolyester, backend, assemble_parallel!,
               allocate_system_matrix, D₋ₓ, D₊ᵧ, πₕ, inner₊ₓ, CompositeGridSpace
using ..TestUtils: WITH_SLOW_TESTS

# The threaded refill replays the form's recording: `assemble!` on a
# `Parallel()` form and `assemble_parallel!` from any policy write through the recorded
# `nzval` positions, band by band, instead of searching for each entry. Every check here
# goes against a serial `assemble` of the same form on the same (non-uniform) mesh, never
# against another threaded fill. That no threaded refill searches at all is pinned by the
# plan's own check, which makes the search throw; here the evidence is the cache recording
# the matrix being filled, and agreement to round-off.

# Allocation checks behind a function barrier (bramble-verification §1).
_alloc(f::F, args...) where {F} = (f(args...); @allocated f(args...))

const _DOMAINS = (
    domain(interval(0.0, 1.0)),
    domain(interval(0.0, 1.0) × interval(0.0, 2.0)),
    domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0))
)

# The same random non-uniform mesh under either policy: the seed fixes the points.
function _mesh(D, n, policy; seed = 338)
    Random.seed!(seed)
    return mesh(
        _DOMAINS[D], ntuple(_ -> n, D), ntuple(_ -> false, D); backend = backend(policy = policy)
    )
end

# `assemble` and `allocate_system_matrix` infer a union that includes a dense `Matrix`
# (their element-type promotion is not inferable here), which has no `nonzeros`; the matrices
# these tests fill are always `SparseMatrixCSC`, so the assertion narrows the type for JET.
function _fillnz!(A, v)
    @assert A isa SparseMatrixCSC
    return fill!(nonzeros(A), v)
end

_same_structure(A, B) = getcolptr(A) == getcolptr(B) && rowvals(A) == rowvals(B)
_agrees(A, R) = _same_structure(A, R) && isapprox(A, R; rtol = 1e-12)
# `s .* R` may drop stored zeros, which `_agrees` would read as a different structure.
_scaled(R, s) = (S = copy(R); _fillnz!(S, 0.0); nonzeros(S) .= s .* nonzeros(R); S)

# Scalar sums, a transposed pair ⟨Au, Bv⟩ + ⟨Bu, Av⟩ (one recorded unit, written twice per
# entry), and their composite counterparts, off-diagonal blocks and a block pair included.
_scalar(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(u), v)
_pair(u, v) = innerₕ(D₋ₓ(u), v) + innerₕ(u, D₋ₓ(v)) + innerₕ(u, v)
_composite(u, v) = innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) + innerₕ(D₋ₓ(u(1)), v(2))
_block_pair(u, v) = innerₕ(D₋ₓ(u(1)), v(2)) + innerₕ(u(2), D₋ₓ(v(1))) + innerₕ(u(1), v(1)) +
                    innerₕ(u(2), v(2))

const _SIZES = (41, 13, 7)

# Compile time per form and dimension dominates this file, not grid size. `unit` keeps the
# 2D composite and block-pair forms (the only per-leaf replay tests; their sweeps also walk
# the scalar kernels) and the mixed-leaf cases; the scalar and pair forms and the 1D/3D
# sweeps run under `slow`.
const _DIMS = WITH_SLOW_TESTS ? (1:3) : (2,)
const _FORMS = (("composite", _composite, 2), ("block pair", _block_pair, 2))
const _ALL_FORMS = WITH_SLOW_TESTS ? ((("scalar", _scalar, 1), ("pair", _pair, 1))..., _FORMS...) :
                   _FORMS

# Threaded refill replays the recording.
@testset "threaded refill replays (#338)" begin
    @testset "$(D)D, $(nm)" for D in _DIMS, (nm, f, comps) in _ALL_FORMS

        n = _SIZES[D]
        Ωs = _mesh(D, n, Serial())
        Ωp = _mesh(D, n, Parallel())
        @test points(Ωs) == points(Ωp)
        space(Ω) = comps == 1 ? gridspace(Ω) : gridspace(Ω, Val(comps))
        as = form(space(Ωs), space(Ωs), f)
        ap = form(space(Ωp), space(Ωp), f)
        R = assemble(as)

        # `Parallel()` through `assemble!`: the first fill records, later ones replay.
        P = assemble(ap)
        @test _agrees(P, R)
        @test ap.cache.valid && ap.cache.A_id == objectid(P)
        _fillnz!(P, NaN)
        assemble!(P, ap)
        @test _agrees(P, R)

        # `assemble_parallel!` from a serial form: `assemble` already recorded this matrix,
        # so even its first threaded fill replays.
        B = assemble(as)
        _fillnz!(B, NaN)
        assemble_parallel!(B, as)
        @test _agrees(B, R)
        @test as.cache.A_id == objectid(B)

        # ... and against a matrix it has never seen: records once, then replays.
        C = copy(R)
        _fillnz!(C, NaN)
        assemble_parallel!(C, as)
        @test _agrees(C, R)
        @test as.cache.A_id == objectid(C)
        _fillnz!(C, NaN)
        assemble_parallel!(C, as)
        @test _agrees(C, R)

        # The serial path reads the recording the threaded one made, and the other way round.
        _fillnz!(C, NaN)
        assemble!(C, as)
        @test _agrees(C, R)
    end

    @testset "$(D)D: live coefficient through Rₕ!" for D in _DIMS
        n = _SIZES[D]
        Ωs = _mesh(D, n, Serial())
        Ωp = _mesh(D, n, Parallel())
        Ws = gridspace(Ωs)
        Wp = gridspace(Ωp)
        cs = Rₕ(Ws, x -> 1.0 + first(x)^2)
        cp = Rₕ(Wp, x -> 1.0 + first(x)^2)
        as = form(Ws, Ws, (u, v) -> innerₕ(cs * D₋ₓ(u), D₋ₓ(v)) + innerₕ(cs * u, v))
        ap = form(Wp, Wp, (u, v) -> innerₕ(cp * D₋ₓ(u), D₋ₓ(v)) + innerₕ(cp * u, v))

        P = assemble(ap)
        assemble!(P, ap)
        B = assemble(as)
        @test _agrees(P, assemble(as))

        Rₕ!(cs, x -> 2.0 + sin(3 * first(x)))
        Rₕ!(cp, x -> 2.0 + sin(3 * first(x)))
        R = assemble(as)
        assemble!(P, ap)
        @test _agrees(P, R)
        assemble_parallel!(B, as)
        @test _agrees(B, R)
    end

    @testset "A second matrix object re-records" begin
        Ωp = _mesh(2, 13, Parallel())
        Ωs = _mesh(2, 13, Serial())
        ap = form(gridspace(Ωp), gridspace(Ωp), _scalar)
        R = assemble(form(gridspace(Ωs), gridspace(Ωs), _scalar))

        P1 = assemble(ap)
        P2 = copy(P1)
        _fillnz!(P2, NaN)
        assemble!(P2, ap)
        @test ap.cache.A_id == objectid(P2)
        @test _agrees(P2, R)

        # Back to the first: its recording was replaced, so it records again, correctly.
        _fillnz!(P1, NaN)
        assemble!(P1, ap)
        @test ap.cache.A_id == objectid(P1)
        @test _agrees(P1, R)
        _fillnz!(P1, NaN)
        assemble!(P1, ap)
        @test _agrees(P1, R)
    end

    @testset "2D transposed pair records one unit" begin
        Ωp = _mesh(2, 13, Parallel())
        Ωs = _mesh(2, 13, Serial())
        g(u, v) = innerₕ(D₋ₓ(u), D₊ᵧ(v)) + innerₕ(D₊ᵧ(u), D₋ₓ(v))
        ap = form(gridspace(Ωp), gridspace(Ωp), g)
        R = assemble(form(gridspace(Ωs), gridspace(Ωs), g))
        P = assemble(ap)
        @test length(ap.cache.segments) == 1
        _fillnz!(P, NaN)
        assemble!(P, ap)
        @test _agrees(P, R)
    end

    # Whether a unit replays is decided from the leaf its sweep walks, not from the form's
    # trial space: a composite's leaves, or a cross-mesh form's two meshes, can carry
    # different policies. `other` is the second leaf's (or the test mesh's) policy; every
    # case is checked against the same form with every leaf serial.
    function _mixed_cases(other)
        n = 33
        leaf(policy, seed) = gridspace(_mesh(1, n, policy; seed = seed))
        comp(p1, p2) = CompositeGridSpace((leaf(p1, 1), leaf(p2, 2)))
        f(u, v) = innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(2)), D₋ₓ(v(2))) +
                  innerₕ(D₋ₓ(u(1)), v(2)) + innerₕ(u(2), D₋ₓ(v(1))) +   # pair across leaves
                  innerₕ(D₋ₓ(u(2)), v(2)) + innerₕ(u(2), D₋ₓ(v(2)))     # pair on leaf 2
        g(u, v) = innerₕ(πₕ(u), v)
        Wu(p) = gridspace(_mesh(1, 17, p; seed = 3))
        Wv(p) = gridspace(_mesh(1, n, p; seed = 4))
        return (
            ("composite", form(comp(Parallel(), other), comp(Parallel(), other), f),
                assemble(form(comp(Serial(), Serial()), comp(Serial(), Serial()), f))),
            ("cross-mesh", form(Wu(Parallel()), Wv(other), g),
                assemble(form(Wu(Serial()), Wv(Serial()), g)))
        )
    end

    function _check_mixed(other)
        for (nm, a, R) in _mixed_cases(other), refill! in (assemble!, assemble_parallel!)

            A = copy(R)
            for _ in 1:2   # record, then replay
                _fillnz!(A, NaN)
                refill!(A, a)
                @test _agrees(A, R)
            end
        end
    end

    # Mixed leaf policies: CpuThreaded beside CpuSerial.
    @testset "mixed policies: Threaded + Serial" begin
        _check_mixed(Serial())
    end

    # The Threaded + Polyester mixed case lives in test/ext/polyester_ext.jl.

    # Test-side interpolation replays on one thread.
    @testset "test-side interpolation: one thread" begin
        Ω = domain(interval(0.0, 1.0))
        Random.seed!(263)
        Wu = gridspace(mesh(Ω, 9, true))
        Wv = gridspace(mesh(Ω, 6, false))
        a = form(Wu, Wv, (u, v) -> innerₕ(u, πₕ(v)) + inner₊ₓ(D₋ₓ(u), D₋ₓ(πₕ(v))))
        R = assemble(a)
        A = allocate_system_matrix(a)
        assemble_parallel!(A, a)
        @test _agrees(A, R)
        _fillnz!(A, NaN)
        assemble_parallel!(A, a)
        @test _agrees(A, R)
    end

    # A leaf whose policy does not replay is searched point by point (`_scatter_point!`).
    # Without `BramblePolyesterExt`, `CpuPolyester` is such a policy, and the only sweeps it
    # can run are the one-thread fallbacks of a test-side interpolation; every other unit
    # stops at a `_batch_*` hook naming Polyester. With the extension it replays instead,
    # and every fill below must still agree with the serial one.
    has_polyester = Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing
    function leaf1(n, policy, seed)
        Random.seed!(seed)
        return gridspace(
            mesh(domain(interval(0.0, 1.0)), n, false; backend = backend(policy = policy))
        )
    end
    interp(u, v) = innerₕ(u, πₕ(v)) + inner₊ₓ(D₋ₓ(u), D₋ₓ(πₕ(v)))
    function polyester_error(f, args...)
        try
            f(args...)
            return nothing
        catch e
            return e
        end
    end

    @testset "no leaf replays: the searching sweep" begin
        R = assemble(form(leaf1(9, Serial(), 3), leaf1(6, Serial(), 4), interp))
        a = form(leaf1(9, CpuPolyester(), 3), leaf1(6, CpuPolyester(), 4), interp)
        for refill! in (assemble!, assemble_parallel!)
            A = copy(R)
            _fillnz!(A, NaN)
            refill!(A, a)
            @test _agrees(A, R)
        end
        # searched, so nothing was recorded (the extension's replay records)
        @test a.cache.valid == has_polyester

        # composite: each block searched at its own offsets, against the scalar cross-mesh
        # forms; 1 and 100 so that a block landing in the wrong place cannot pass
        f(u, v) = innerₕ(u(1), πₕ(v(2))) + 100 * innerₕ(u(2), πₕ(v(1)))
        comp(p) = CompositeGridSpace((leaf1(9, p, 5), leaf1(6, p, 6)))
        Rc = assemble(form(comp(Serial()), comp(Serial()), f))
        C = copy(Rc)
        _fillnz!(C, NaN)
        assemble_parallel!(C, form(comp(CpuPolyester()), comp(CpuPolyester()), f))
        @test _agrees(C, Rc)
        g(u, v) = innerₕ(u, πₕ(v))
        B21 = Matrix(assemble(form(leaf1(9, Serial(), 5), leaf1(6, Serial(), 6), g)))
        B12 = Matrix(assemble(form(leaf1(6, Serial(), 6), leaf1(9, Serial(), 5), g)))
        M = Matrix(C)
        @test M[10:15, 1:9] ≈ B21
        @test M[1:9, 10:15] ≈ 100 * B12
        @test iszero(M[1:9, 1:9]) && iszero(M[10:15, 10:15])
    end

    @testset "no replay: units reach Polyester hooks" begin
        h(u, v) = innerₕ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(u, v)
        # 9 points band the grid; 3 are too few for two bands, so the sweep colours points
        for (n, hook) in ((9, "_batch_bilinear_band_sweep!"), (3, "_batch_bilinear_colour_sweep!"))
            R = assemble(form(leaf1(n, Serial(), 7), leaf1(n, Serial(), 7), h))
            a = form(leaf1(n, CpuPolyester(), 7), leaf1(n, CpuPolyester(), 7), h)
            A = copy(R)
            _fillnz!(A, NaN)
            if has_polyester
                assemble_parallel!(A, a)
                @test _agrees(A, R)
            else
                err = polyester_error(assemble_parallel!, A, a)
                @test err isa ArgumentError
                @test occursin(hook, err.msg)
            end
        end
        # a composite's plain block the same way
        comp(p) = CompositeGridSpace((leaf1(9, p, 5), leaf1(6, p, 6)))
        k(u, v) = innerₕ(u, v)
        Rc = assemble(form(comp(Serial()), comp(Serial()), k))
        ac = form(comp(CpuPolyester()), comp(CpuPolyester()), k)
        C = copy(Rc)
        _fillnz!(C, NaN)
        if has_polyester
            assemble_parallel!(C, ac)
            @test _agrees(C, Rc)
        else
            err = polyester_error(assemble_parallel!, C, ac)
            @test err isa ArgumentError
            @test occursin("_batch_bilinear_band_sweep!", err.msg)
        end
        # the replay hooks are reached only once the extension opts the policy in; their
        # `src/` methods still answer an untyped call naming Polyester
        for (hook, nargs) in ((Bramble._batch_bilinear_colour_replay!, 8),
            (Bramble._batch_bilinear_band_replay!, 11),
            (Bramble._batch_bilinear_colour_sweep!, 9),
            (Bramble._batch_bilinear_band_sweep!, 12))
            err = polyester_error(hook, ntuple(_ -> nothing, nargs)...)
            @test err isa ArgumentError
            @test occursin(string(nameof(hook)), err.msg)
        end
    end

    @testset "one leaf replays, the other searches" begin
        # The form takes the recording because its test leaf replays; the unit walks the
        # trial leaf (a test-side interpolation), which cannot, so that unit searches.
        R = assemble(form(leaf1(9, Serial(), 3), leaf1(6, Serial(), 4), interp))
        a = form(leaf1(9, CpuPolyester(), 3), leaf1(6, Parallel(), 4), interp)
        A = copy(R)
        for _ in 1:2   # record, then replay
            _fillnz!(A, NaN)
            assemble!(A, a)
            @test _agrees(A, R)
        end
        @test a.cache.valid && a.cache.A_id == objectid(A)
    end

    @testset "pair leaf fallbacks, called directly" begin
        # A transposed pair never carries an interpolation (`_pairable_type`), so the
        # threaded pair leaf's one-thread branches are reached here by calling it with an
        # interpolating term as both halves. Its recorded segment is the term's own, which
        # the first half (`half = 1`) reads exactly as a `ReplaySink` would.
        tr(u, v) = innerₕ(πₕ(u), v)
        Rt = assemble(form(leaf1(9, Serial(), 3), leaf1(6, Serial(), 4), tr))
        Wu, Wv = leaf1(9, Parallel(), 3), leaf1(6, Parallel(), 4)
        at = form(Wu, Wv, tr)
        P = assemble(at)
        seg = only(at.cache.segments)
        @test !seg.is_diagonal
        bound = Bramble._bind_interp_spaces(at.ast, Wu, Wv)
        sp = Bramble._walked_leaf(bound, Wu, Wv)
        _fillnz!(P, 0.0)
        Bramble._replay_pair_unit!(
            Bramble._ThreadedReplay(), P, bound, bound, sp, 0, 0, (0, 0), seg, 2.0, 0.0, 1)
        @test _agrees(P, _scaled(Rt, 2.0))

        # a leaf that cannot replay searches the pair as its two terms, each at its own
        # offsets and with its own scaling (`seg` is not read there)
        g(u, v) = innerₕ(u, πₕ(v))
        if !has_polyester
            Rg = assemble(form(leaf1(9, Serial(), 3), leaf1(6, Serial(), 4), g))
            Wpu, Wpv = leaf1(9, CpuPolyester(), 3), leaf1(6, CpuPolyester(), 4)
            ag = form(Wpu, Wpv, g)
            bg = Bramble._bind_interp_spaces(ag.ast, Wpu, Wpv)
            spg = Bramble._walked_leaf(bg, Wpu, Wpv)
            A = copy(Rg)
            _fillnz!(A, 0.0)
            Bramble._replay_pair_unit!(
                Bramble._ThreadedReplay(), A, bg, bg, spg, 0, 0, (0, 0), seg, 2.0, 3.0, 0)
            @test _agrees(A, _scaled(Rg, 5.0))
        end
    end

    @testset "inside a threaded region" begin
        # A `:static` loop cannot start inside another threaded region, so every colour
        # and band runs in order on the calling task, to the same answer. One point is a
        # single colour of the whole grid (no difference: its spacing is 0), 3 points are
        # too few to band, 33 band.
        m(u, v) = innerₕ(u, v)
        h(u, v) = innerₕ(u, v) + innerₕ(D₋ₓ(u), D₋ₓ(v))
        for (n, g) in ((1, m), (3, h), (33, h))
            R = assemble(form(leaf1(n, Serial(), 11), leaf1(n, Serial(), 11), g))
            as = form(leaf1(n, Serial(), 11), leaf1(n, Serial(), 11), g)
            ap = form(leaf1(n, Parallel(), 11), leaf1(n, Parallel(), 11), g)
            P = assemble(ap)
            B = assemble(as)
            _fillnz!(P, NaN)
            _fillnz!(B, NaN)
            Threads.@threads for _ in 1:1
                assemble!(P, ap)
                assemble_parallel!(B, as)
            end
            @test _agrees(P, R)
            @test _agrees(B, R)
        end
    end

    # Threaded tasks allocate per call, so a warmed refill is not 0 B; what it must not do
    # is grow with the grid (the plan's O13).
    @testset "$(D)D: refill allocs flat in ndofs" for D in _DIMS
        sizes = D == 1 ? (200, 800) : D == 2 ? (24, 64) : (10, 20)
        bytes_par = map(sizes) do n
            Ω = _mesh(D, n, Parallel())
            a = form(gridspace(Ω), gridspace(Ω), _scalar)
            A = assemble(a)
            _alloc(assemble!, A, a)
        end
        bytes_forced = map(sizes) do n
            Ω = _mesh(D, n, Serial())
            a = form(gridspace(Ω, Val(2)), gridspace(Ω, Val(2)), _block_pair)
            A = assemble(a)
            _alloc(assemble_parallel!, A, a)
        end
        @test bytes_par[1] == bytes_par[2]
        @test bytes_forced[1] == bytes_forced[2]
    end
end

end # module
