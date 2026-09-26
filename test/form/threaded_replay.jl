module FormThreadedReplayTests

using Test
using Bramble
using Random
using SparseArrays
using SparseArrays: getcolptr
using Bramble: Serial, Parallel, backend, assemble_parallel!, allocate_system_matrix, D₋ₓ,
               D₊ᵧ, πₕ, inner₊ₓ, CompositeGridSpace, CpuPolyester

# The threaded refill replays the form's recording (gpena/Bramble.jl#338): `assemble!` on a
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

_same_structure(A, B) = getcolptr(A) == getcolptr(B) && rowvals(A) == rowvals(B)
_agrees(A, R) = _same_structure(A, R) && isapprox(A, R; rtol = 1e-12)

# Scalar sums, a transposed pair ⟨Au, Bv⟩ + ⟨Bu, Av⟩ (one recorded unit, written twice per
# entry), and their composite counterparts, off-diagonal blocks and a block pair included.
_scalar(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(u), v)
_pair(u, v) = innerₕ(D₋ₓ(u), v) + innerₕ(u, D₋ₓ(v)) + innerₕ(u, v)
_composite(u, v) = innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) + innerₕ(D₋ₓ(u(1)), v(2))
_block_pair(u, v) = innerₕ(D₋ₓ(u(1)), v(2)) + innerₕ(u(2), D₋ₓ(v(1))) + innerₕ(u(1), v(1)) +
                    innerₕ(u(2), v(2))

const _SIZES = (41, 13, 7)

@testset "Threaded refill replays the recording (#338)" begin
    @testset "$(D)D, $(nm)" for D in 1:3,
        (nm, f, comps) in (
            ("scalar", _scalar, 1), ("pair", _pair, 1),
            ("composite", _composite, 2), ("block pair", _block_pair, 2)
        )

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
        fill!(nonzeros(P), NaN)
        assemble!(P, ap)
        @test _agrees(P, R)

        # `assemble_parallel!` from a serial form: `assemble` already recorded this matrix,
        # so even its first threaded fill replays.
        B = assemble(as)
        fill!(nonzeros(B), NaN)
        assemble_parallel!(B, as)
        @test _agrees(B, R)
        @test as.cache.A_id == objectid(B)

        # ... and against a matrix it has never seen: records once, then replays.
        C = copy(R)
        fill!(nonzeros(C), NaN)
        assemble_parallel!(C, as)
        @test _agrees(C, R)
        @test as.cache.A_id == objectid(C)
        fill!(nonzeros(C), NaN)
        assemble_parallel!(C, as)
        @test _agrees(C, R)

        # The serial path reads the recording the threaded one made, and the other way round.
        fill!(nonzeros(C), NaN)
        assemble!(C, as)
        @test _agrees(C, R)
    end

    @testset "$(D)D: live coefficient through Rₕ!" for D in 1:3
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
        fill!(nonzeros(P2), NaN)
        assemble!(P2, ap)
        @test ap.cache.A_id == objectid(P2)
        @test _agrees(P2, R)

        # Back to the first: its recording was replaced, so it records again, correctly.
        fill!(nonzeros(P1), NaN)
        assemble!(P1, ap)
        @test ap.cache.A_id == objectid(P1)
        @test _agrees(P1, R)
        fill!(nonzeros(P1), NaN)
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
        fill!(nonzeros(P), NaN)
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
                fill!(nonzeros(A), NaN)
                refill!(A, a)
                @test _agrees(A, R)
            end
        end
    end

    @testset "Mixed leaf policies: CpuThreaded beside CpuSerial" begin
        _check_mixed(Serial())
    end

    # A `CpuPolyester` leaf cannot replay (until `BramblePolyesterExt` fills the replay
    # hooks), so its units search while the `CpuThreaded` leaf's units replay. Building a
    # `CpuPolyester` space needs Polyester, which the `unit` group deliberately does not load
    # (test/space/inner_product.jl checks the error without it), so this runs only where the
    # extension is already loaded.
    if Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing
        @testset "Mixed leaf policies: CpuThreaded beside CpuPolyester" begin
            _check_mixed(CpuPolyester())
        end
    end

    @testset "Test-side interpolation replays on one thread" begin
        Ω = domain(interval(0.0, 1.0))
        Random.seed!(263)
        Wu = gridspace(mesh(Ω, 9, true))
        Wv = gridspace(mesh(Ω, 6, false))
        a = form(Wu, Wv, (u, v) -> innerₕ(u, πₕ(v)) + inner₊ₓ(D₋ₓ(u), D₋ₓ(πₕ(v))))
        R = assemble(a)
        A = allocate_system_matrix(a)
        assemble_parallel!(A, a)
        @test _agrees(A, R)
        fill!(nonzeros(A), NaN)
        assemble_parallel!(A, a)
        @test _agrees(A, R)
    end

    # Threaded tasks allocate per call, so a warmed refill is not 0 B; what it must not do
    # is grow with the grid (the plan's O13).
    @testset "$(D)D: warmed refill allocation is independent of ndofs" for D in 1:3
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
