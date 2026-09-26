module ExtMetalFormAssemblyTests

using Test
using Bramble
using Bramble: D₋ᵧ, D₋ₓ, change_points!, innerₕ
using Metal
using SparseArrays
using ..TestUtils: _run_gpu_tests

# A bilinear form on a composite space assembles on a Metal backend (gpena/Bramble.jl#361):
# the element-type probe reads each leaf's `host_weights`, as the pattern walk and the fill
# already do, and the matrix is filled on the host and uploaded once (#317). The first
# `assemble` must match the host assembly, and a refill `assemble!` must replay the recorded
# positions rather than search (#318), proved as in `metal_assembly_replay.jl`: the mirror's
# `colval` is scrubbed during the refills, so any search throws.
#
# Two leaves with an off-diagonal coupling term, so a block other than the diagonal ones is
# assembled, on mirrored non-uniform meshes (`metal_fullstack.jl`).
#
# Gated like `test/ext/metal_fullstack.jl`, for the same reason: a silently skipped file was
# issue #84's failure mode.
if !Metal.functional() || !_run_gpu_tests()
    @warn "Skipping Metal form assembly tests: Metal.functional() is false, or GPU tests are skipped in CI"
    @test_skip "Metal form assembly tests not exercised: Metal.functional() is false, or GPU tests are skipped in CI"
else
    const _RTOL = 1.0f-4

    # A deterministic, strictly increasing stretch of [0, 1], mirrored onto the CPU and the
    # device mesh with `change_points!` so both sides see the same coordinates.
    function _stretch_points(n::Int)
        t = Float32[(i - 1) / (n - 1) for i in 1:n]
        raw = Float32.(t .+ 0.15f0 .* sin.(Float32(pi) .* t))
        return (raw .- raw[1]) ./ (raw[end] - raw[1])
    end

    function _matched_composites(npts::NTuple{D, Int}) where {D}
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
        Wc, Wg = gridspace(Ωc), gridspace(Ωg)
        return Wc × Wc, Wg × Wg
    end

    _relerr(Ag, Ac) = maximum(abs, Array(Ag) .- Float32.(Matrix(Ac))) / maximum(abs, Ac)

    # Runs `refill!()` `nrefills` times with `A.mirror.colval` scrubbed, restoring it after.
    function _refills_without_search!(refill!, A, nrefills::Int)
        saved = copy(A.mirror.colval)
        fill!(A.mirror.colval, zero(eltype(A.mirror.colval)))
        try
            @test Bramble._scatter_position(A, 1, 1) == 0
            for _ in 1:nrefills
                refill!()
            end
        finally
            copyto!(A.mirror.colval, saved)
        end
        return A
    end

    function _check_composite(a, npts)
        Xc, Xg = _matched_composites(npts)
        Fg = form(Xg, Xg, a)
        Ag = assemble(Fg)
        Ac = assemble(form(Xc, Xc, a))
        @test nameof(typeof(Ag)) === :MetalSparseMatrixCSR
        @test size(Ag) == size(Ac)
        @test _relerr(Ag, Ac) < _RTOL
        # The coupling block really is assembled: rows of the second leaf, columns of the first.
        n = ndofs(first(Bramble.leaf_spaces_offsets(Xc))[1])
        @test any(!iszero, Array(Ag)[(n + 1):end, 1:n])
        _refills_without_search!(() -> assemble!(Ag, Fg), Ag, 5)
        @test _relerr(Ag, Ac) < _RTOL
    end

    @testset "Metal composite form assembly (gpena/Bramble.jl#361)" begin
        @testset "1D non-uniform, two leaves with coupling" begin
            a(U, V) = innerₕ(U, V) + innerₕ(D₋ₓ(U[1]), D₋ₓ(V[1])) + innerₕ(U[1], V[2])
            _check_composite(a, (33,))
        end

        @testset "2D non-uniform, two leaves with coupling" begin
            a(U, V) = innerₕ(U, V) + innerₕ(D₋ₓ(U[1]), D₋ₓ(V[1])) +
                      innerₕ(D₋ᵧ(U[2]), D₋ᵧ(V[2])) + innerₕ(U[1], V[2])
            _check_composite(a, (17, 23))
        end
    end
end

end # module
