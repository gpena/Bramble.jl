module ExtMetalAssemblyReplayTests

using Test
using Bramble
using Bramble: D₋ᵧ, D₋ₓ, change_points!, inner₊ᵧ, inner₊ₓ
using Metal
using SparseArrays
using ..TestUtils: _run_gpu_tests

# A refill `assemble!` into a Metal CSR matrix replays the positions its form recorded
# against the matrix's own host mirror, and never searches (gpena/Bramble.jl#318). Only the
# recording fill -- the first `assemble` -- searches `A.mirror`'s `rowptr`/`colval`.
#
# The proof scrubs `A.mirror.colval` to zeros after the recording fill: any search on a
# refill then misses every entry and throws (`add_to_sparse!`'s missing-pattern error), while
# a replay never reads `colval` and fills the matrix correctly. The mirror's `colval` is a
# host copy only the search reads (the device array was uploaded from it when the matrix was
# built), and it is restored before the matrix is read back, so nothing outside the refills
# sees the scrub.
#
# Gated like `test/ext/metal_fullstack.jl`, for the same reason: a silently skipped file was
# issue #84's failure mode.
if !Metal.functional() || !_run_gpu_tests()
    @warn "Skipping Metal assembly replay tests: Metal.functional() is false, or GPU tests are skipped in CI"
    @test_skip "Metal assembly replay tests not exercised: Metal.functional() is false, or GPU tests are skipped in CI"
else
    const _RTOL = 1.0f-4

    # A deterministic, strictly increasing stretch of [0, 1], mirrored onto the CPU and the
    # device mesh with `change_points!` so both sides see the same coordinates
    # (`metal_fullstack.jl`'s `_stretch_points`).
    function _stretch_points(n::Int)
        t = Float32[(i - 1) / (n - 1) for i in 1:n]
        raw = Float32.(t .+ 0.15f0 .* sin.(Float32(pi) .* t))
        return (raw .- raw[1]) ./ (raw[end] - raw[1])
    end

    function _matched_spaces(npts::NTuple{D, Int}) where {D}
        Ω = domain(D == 1 ? interval(0.0f0, 1.0f0) :
                   interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0))
        unif = D == 1 ? false : ntuple(_ -> false, Val(D))
        n = D == 1 ? npts[1] : npts
        Ωc = mesh(Ω, n, unif)
        Ωg = mesh(Ω, n, unif; backend = metal_backend())
        for d in 1:D
            pts = _stretch_points(npts[d])
            change_points!(D == 1 ? Ωc : Ωc(d), copy(pts))
            change_points!(D == 1 ? Ωg : Ωg(d), Metal.MtlVector(pts))
        end
        return gridspace(Ωc), gridspace(Ωg)
    end

    _relerr(Ag, Ac) = maximum(abs, Array(Ag) .- Float32.(Matrix(Ac))) / maximum(abs, Ac)

    # Runs `refill!()` `nrefills` times with `A.mirror.colval` scrubbed, restoring it after.
    function _refills_without_search!(refill!, A, nrefills::Int)
        saved = copy(A.mirror.colval)
        fill!(A.mirror.colval, zero(eltype(A.mirror.colval)))
        try
            # The scrub really does blind a search: the diagonal is in the pattern.
            @test Bramble._scatter_position(A, 1, 1) == 0
            for _ in 1:nrefills
                refill!()
            end
        finally
            copyto!(A.mirror.colval, saved)
        end
        return A
    end

    @testset "Metal assembly replay (gpena/Bramble.jl#318)" begin
        @testset "1D non-uniform: refills replay, match the CPU" begin
            a(u, v) = inner₊ₓ(D₋ₓ(u), D₋ₓ(v))
            for n in (33, 513)
                Wc, Wg = _matched_spaces((n,))
                Fg = form(Wg, Wg, a)
                Ag = assemble(Fg)
                Ac = assemble(form(Wc, Wc, a))
                @test _relerr(Ag, Ac) < _RTOL
                _refills_without_search!(() -> assemble!(Ag, Fg), Ag, 5)
                @test _relerr(Ag, Ac) < _RTOL
            end
        end

        @testset "2D non-uniform, two terms: refills replay, match the CPU" begin
            a(u, v) = inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v))
            Wc, Wg = _matched_spaces((17, 23))
            Fg = form(Wg, Wg, a)
            Ag = assemble(Fg)
            Ac = assemble(form(Wc, Wc, a))
            @test _relerr(Ag, Ac) < _RTOL
            _refills_without_search!(() -> assemble!(Ag, Fg), Ag, 5)
            @test _relerr(Ag, Ac) < _RTOL
        end

        # Replay skips only the search: weights are still evaluated on every refill, so a
        # live scalar changed between refills shows up in the replayed matrix.
        @testset "1D non-uniform: a replayed refill sees a live coefficient" begin
            β = Ref(1.0f0)
            a(u, v) = inner₊ₓ(β * D₋ₓ(u), D₋ₓ(v))
            Wc, Wg = _matched_spaces((65,))
            Fg = form(Wg, Wg, a)
            Ag = assemble(Fg)
            Ac = assemble(form(Wc, Wc, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))
            β[] = 3.0f0
            _refills_without_search!(() -> assemble!(Ag, Fg), Ag, 3)
            @test _relerr(Ag, 3 .* Ac) < _RTOL
        end
    end
end

end # module
