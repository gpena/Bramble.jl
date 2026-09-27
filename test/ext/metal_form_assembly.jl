module ExtMetalFormAssemblyTests

using Test
using Bramble
using Bramble: D₋ᵧ, D₋ₓ, change_points!, dirac, dirichlet_bc!, innerₕ, Rₕ, πₕ
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

    function _matched_composites(npts)
        Wc, Wg = _matched_spaces(npts)
        return Wc × Wc, Wg × Wg
    end

    _relerr(Ag, Ac) = maximum(abs, Array(Ag) .- Float32.(Matrix(Ac))) / maximum(abs, Ac)
    _vrelerr(bg, bc) = maximum(abs, Array(bg) .- bc) / maximum(abs, bc)

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

    # A linear form on a device test space is swept on a host mirror and uploaded once
    # (gpena/Bramble.jl#361). `assemble` must return device storage matching the host
    # vector, and `assemble!` must refill a zeroed device vector to the same values, with
    # and without a Dirichlet condition written into it.
    function _check_linear(l, sc, sg; kwargs...)
        bc = assemble(form(sc, l); kwargs...)
        lg = form(sg, l)
        bg = assemble(lg; kwargs...)
        @test bg isa Metal.MtlVector{Float32}
        @test length(bg) == length(bc)
        @test _vrelerr(bg, bc) < _RTOL
        fill!(bg, 0.0f0)
        assemble!(bg, lg; kwargs...)
        @test _vrelerr(bg, bc) < _RTOL
        return bc
    end

    _f(x) = sin(3.0f0 * x[1]) + 1.0f0
    _g(x) = 2.0f0 + x[1]

    @testset "Metal linear form assembly (gpena/Bramble.jl#361)" begin
        for npts in ((33,), (17, 23))
            @testset "$(length(npts))D non-uniform" begin
                Wc, Wg = _matched_spaces(npts)
                Xc, Xg = Wc × Wc, Wg × Wg
                composite = V -> innerₕ(_f, V[1]) + innerₕ(2.0f0, V[2])

                @testset "scalar, constant source" begin
                    _check_linear(v -> innerₕ(one(Float32), v), Wc, Wg)
                end
                @testset "scalar, function source (lowered through Rₕ on the device)" begin
                    _check_linear(v -> innerₕ(_f, v), Wc, Wg)
                end
                @testset "scalar, source under a difference of the test function" begin
                    _check_linear(v -> innerₕ(_f, D₋ₓ(v)), Wc, Wg)
                end
                @testset "composite" begin
                    bc = _check_linear(composite, Xc, Xg)
                    n = ndofs(Wc)
                    @test any(!iszero, bc[1:n]) && any(!iszero, bc[(n + 1):end])
                end
                @testset "scalar, Dirichlet on the vector" begin
                    bc = _check_linear(
                        v -> innerₕ(_f, v), Wc, Wg; dirichlet = :boundary => _g)
                    @test any(==(2.0f0), bc)   # the condition was written, not only swept
                end
                @testset "composite, Dirichlet on one component" begin
                    _check_linear(composite, Xc, Xg; dirichlet = :boundary => _g,
                        dirichlet_components = 1)
                end
                @testset "device grid-function scale on the test side stays live" begin
                    uc, ug = Rₕ(Wc, _g), Rₕ(Wg, _g)
                    for l in (u -> (v -> innerₕ(_f, u * v)),
                        u -> (v -> innerₕ(_f, u * D₋ₓ(v))))
                        lg = form(Wg, l(ug))
                        bg = assemble(lg)
                        @test bg isa Metal.MtlVector{Float32}
                        @test _vrelerr(bg, assemble(form(Wc, l(uc)))) < _RTOL
                        parent(ug) .*= 3.0f0
                        parent(uc) .*= 3.0f0
                        fill!(bg, 0.0f0)
                        assemble!(bg, lg)
                        @test _vrelerr(bg, assemble(form(Wc, l(uc)))) < _RTOL
                    end
                end
                @testset "device grid-function source stays live across assemble!" begin
                    uc, ug = Rₕ(Wc, _f), Rₕ(Wg, _f)
                    lg = form(Wg, v -> innerₕ(ug, v))
                    bg = assemble(lg)
                    parent(ug) .*= 3.0f0
                    parent(uc) .*= 3.0f0
                    assemble!(bg, lg)
                    @test _vrelerr(bg, assemble(form(Wc, v -> innerₕ(uc, v)))) < _RTOL
                end
            end
        end
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

    # Dirichlet conditions on a device CSR matrix (gpena/Bramble.jl#361, S4): the marked rows
    # are rewritten by one kernel on the device's own arrays, never by scalar `setindex!`, and
    # only those rows are touched -- a value changed on the device since the last assembly
    # must survive `dirichlet_bc!`, which a re-flush of the host mirror would overwrite.
    @testset "Metal Dirichlet conditions (gpena/Bramble.jl#361)" begin
        g = x -> 1.0f0 + x[1]
        f = x -> sin(3.0f0 * x[1]) + 1.0f0
        dir = :boundary => g
        for npts in ((33,), (17, 23))
            Wc, Wg = _matched_spaces(npts)
            cases = (
                ("scalar", Wc, Wg, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(u, v),
                    v -> innerₕ(f, v), nothing),
                ("composite", Wc × Wc, Wg × Wg,
                    (U, V) -> innerₕ(U, V) + innerₕ(D₋ₓ(U[1]), D₋ₓ(V[1])) + innerₕ(U[2], V[1]),
                    V -> innerₕ(f, V[1]) + innerₕ(2.0f0, V[2]), 1)
            )
            for (label, Sc, Sg, a, l, comps) in cases
                @testset "$(length(npts))D $label" begin
                    Ac = assemble(form(Sc, Sc, a); dirichlet = dir, dirichlet_components = comps)
                    ag = form(Sg, Sg, a)
                    Ag = assemble(ag; dirichlet = dir, dirichlet_components = comps)
                    @test _relerr(Ag, Ac) < _RTOL
                    # The host mirror carries the same rows, so a later accumulation agrees.
                    @test maximum(abs, Ag.mirror.nzval .- Array(Ag.nzVal)) < _RTOL
                    assemble!(Ag, ag; dirichlet = dir, dirichlet_components = comps)
                    @test _relerr(Ag, Ac) < _RTOL

                    Ac2 = 2.0f0 .* assemble(form(Sc, Sc, a))
                    dirichlet_bc!(Ac2, Sc, :boundary; components = comps)
                    Ag2 = assemble(ag)
                    Ag2.nzVal .*= 2.0f0
                    dirichlet_bc!(Ag2, Sg, :boundary; components = comps)
                    @test _relerr(Ag2, Ac2) < _RTOL

                    Acc, Fcc = assemble(form(Sc, Sc, a), form(Sc, l); dirichlet = dir,
                        dirichlet_components = comps)
                    Agc, Fgc = assemble(ag, form(Sg, l); dirichlet = dir,
                        dirichlet_components = comps)
                    @test _relerr(Agc, Acc) < _RTOL
                    @test Fgc isa Metal.MtlVector{Float32}
                    @test _vrelerr(Fgc, Fcc) < _RTOL
                    @test_throws ArgumentError assemble(ag, form(Sg, l); dirichlet = dir,
                        dirichlet_components = comps, symmetrize = true)
                end
            end
        end

        # Only the device CSR type takes the row kernel; a dense device matrix keeps the
        # generic body, which works under `@allowscalar`.
        @testset "dense MtlMatrix under @allowscalar" begin
            Wc, Wg = _matched_spaces((9,))
            a = (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(u, v)
            Ac = Matrix(assemble(form(Wc, Wc, a)))
            Ad = Metal.MtlMatrix(Ac)
            Metal.@allowscalar dirichlet_bc!(Ad, Wg, :boundary)
            dirichlet_bc!(Ac, Wc, :boundary)
            @test Array(Ad) == Ac
        end

        @testset "constrained row without a stored diagonal" begin
            S = sparse([1, 2, 2, 3], [2, 2, 3, 3], Float32[1, 2, 3, 4], 3, 3)
            A = Bramble.metal_sparse_csr(S)
            copyto!(A.mirror.nzval, Array(A.nzVal))
            err = try
                Bramble._dirichlet_bc_indices!(A, BitVector([true, false, true]))
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("row 1", sprint(showerror, err))
            # Nothing is written before the check fails.
            @test Array(A) == Matrix(S)
            @test A.mirror.nzval == Array(A.nzVal)
        end
    end

    # `πₕ` across two device meshes (gpena/Bramble.jl#363): each walk binds the interpolation
    # to its source leaf's `host_weights` mirror, so the cell search reads host points. The
    # source and target meshes differ in size, both non-uniform and mirrored.
    @testset "Metal interpolation across device meshes (gpena/Bramble.jl#363)" begin
        for (ns, nt) in (((7,), (11,)), ((7, 9), (11, 8)))
            Wsc, Wsg = _matched_spaces(ns)
            Wtc, Wtg = _matched_spaces(nt)
            cases = (
                ("trial πₕ", Wsc, Wtc, Wsg, Wtg, (u, v) -> innerₕ(πₕ(u), v)),
                ("D₋ₓ of trial πₕ", Wsc, Wtc, Wsg, Wtg,
                    (u, v) -> innerₕ(D₋ₓ(πₕ(u)), D₋ₓ(v))),
                ("test πₕ", Wtc, Wsc, Wtg, Wsg, (u, v) -> innerₕ(u, πₕ(v))),
                ("composite with a cross-mesh block", Wsc × Wtc, Wtc × Wtc, Wsg × Wtg,
                    Wtg × Wtg,
                    (U, V) -> innerₕ(πₕ(U[1]), V[1]) + innerₕ(U[2], V[2]) +
                              innerₕ(U[2], V[1]))
            )
            for (label, trc, tec, trg, teg, a) in cases
                @testset "$(length(ns))D $label" begin
                    Ac = assemble(form(trc, tec, a))
                    Fg = form(trg, teg, a)
                    Ag = assemble(Fg)
                    @test nameof(typeof(Ag)) === :MetalSparseMatrixCSR
                    @test size(Ag) == size(Ac)
                    @test _relerr(Ag, Ac) < _RTOL
                    _refills_without_search!(() -> assemble!(Ag, Fg), Ag, 3)
                    @test _relerr(Ag, Ac) < _RTOL
                end
            end
        end
    end

    # A grid-function coefficient on a device space (gpena/Bramble.jl#364): each fill binds
    # a fresh host copy of it, so the first `assemble` matches the host and a refill after
    # the coefficient changes on the device sees the new values. Scalar and composite
    # bilinear forms, a linear source, and a coefficient scaling the test function under an
    # average and a difference, on mirrored non-uniform meshes.
    @testset "Metal grid-function coefficients (gpena/Bramble.jl#364)" begin
        for npts in ((13,), (13, 11))
            Wc, Wg = _matched_spaces(npts)
            cc, cg = Rₕ(Wc, _g), Rₕ(Wg, _g)
            cases = (
                ("scalar", Wc, Wg,
                    c -> ((u, v) -> innerₕ(c * u, v) + innerₕ(D₋ₓ(u), D₋ₓ(v)))),
                ("composite", Wc × Wc, Wg × Wg,
                    c -> ((U, V) -> innerₕ(c * U[1], V[1]) + innerₕ(U[2], V[2]) +
                                    innerₕ(c * U[2], V[1])))
            )
            for (label, sc, sg, a) in cases
                @testset "$(length(npts))D $label bilinear" begin
                    Fg = form(sg, sg, a(cg))
                    Ag = assemble(Fg)
                    @test nameof(typeof(Ag)) === :MetalSparseMatrixCSR
                    @test _relerr(Ag, assemble(form(sc, sc, a(cc)))) < _RTOL
                    parent(cg) .*= 3.0f0
                    parent(cc) .*= 3.0f0
                    assemble!(Ag, Fg)
                    @test _relerr(Ag, assemble(form(sc, sc, a(cc)))) < _RTOL
                    parent(cg) ./= 3.0f0
                    parent(cc) ./= 3.0f0
                end
            end
            @testset "$(length(npts))D linear source" begin
                lg = form(Wg, v -> innerₕ(cg, v))
                bg = assemble(lg)
                @test _vrelerr(bg, assemble(form(Wc, v -> innerₕ(cc, v)))) < _RTOL
                parent(cg) .*= 2.0f0
                parent(cc) .*= 2.0f0
                assemble!(bg, lg)
                @test _vrelerr(bg, assemble(form(Wc, v -> innerₕ(cc, v)))) < _RTOL
                parent(cg) ./= 2.0f0
                parent(cc) ./= 2.0f0
            end
            for (label, l) in (("Mₓ(c * v)", c -> (v -> innerₕ(1.0f0, Bramble.Mₓ(c * v)))),
                ("D₋ₓ(c * v)", c -> (v -> innerₕ(1.0f0, D₋ₓ(c * v)))))
                @testset "$(length(npts))D linear $label" begin
                    bg = assemble(form(Wg, l(cg)))
                    @test bg isa Metal.MtlVector{Float32}
                    @test _vrelerr(bg, assemble(form(Wc, l(cc)))) < _RTOL
                end
            end
        end
    end

    # `assemble_add!` and `assemble_parallel!` on a linear form call the sweep cores
    # directly, so they split on locality themselves (gpena/Bramble.jl#361): the device
    # vector comes to the host, the contribution is added on a host mirror, and the sum is
    # uploaded once. Unscaled and scaled accumulation onto an assembled vector and a threaded
    # refill must match the host, scalar and composite, on mirrored non-uniform meshes. A
    # vector of the wrong length is refused before anything is written, host and device.
    @testset "Metal linear assemble_add! and assemble_parallel! (gpena/Bramble.jl#361)" begin
        for npts in ((13,), (13, 11))
            Wc, Wg = _matched_spaces(npts)
            cases = (("scalar", Wc, Wg, v -> innerₕ(_f, v) + innerₕ(1.0f0, D₋ₓ(v))),
                ("composite", Wc × Wc, Wg × Wg, V -> innerₕ(_f, V[1]) + innerₕ(2.0f0, V[2])))
            for (label, sc, sg, l) in cases
                @testset "$(length(npts))D $label" begin
                    lc, lg = form(sc, l), form(sg, l)
                    bc, bg = assemble(lc), assemble(lg)
                    Bramble.assemble_add!(bc, lc)
                    Bramble.assemble_add!(bg, lg)
                    @test bg isa Metal.MtlVector{Float32}
                    @test _vrelerr(bg, bc) < _RTOL
                    Bramble.assemble_add!(bc, lc, Ref(0.5f0))
                    Bramble.assemble_add!(bg, lg, Ref(0.5f0))
                    @test _vrelerr(bg, bc) < _RTOL
                    fill!(bg, 1.0f0)
                    Bramble.assemble_parallel!(bg, lg)
                    @test _vrelerr(bg, assemble(lc)) < _RTOL
                end
            end
        end
        @testset "wrong-length vector refused before any write" begin
            Wc, Wg = _matched_spaces((13,))
            for (label, W, zeros_) in (("host", Wc, zeros), ("device", Wg, Metal.zeros))
                l = form(W, v -> innerₕ(1.0f0, v))
                for m in (ndofs(W) - 5, ndofs(W) + 5)
                    b = zeros_(Float32, m)
                    @testset "$label, length $m" begin
                        @test_throws DimensionMismatch assemble!(b, l)
                        @test_throws DimensionMismatch Bramble.assemble_add!(b, l)
                        @test_throws DimensionMismatch Bramble.assemble_add!(b, l, 2.0f0)
                        @test_throws DimensionMismatch Bramble.assemble_parallel!(b, l)
                        @test all(iszero, Array(b))
                    end
                end
            end
        end
    end

    # The target vector may be a device `element(Wg, 0f0)` or a strided view of a device
    # vector, as on the host. The device path downloads from and uploads to the element's
    # storage, not through the wrapper, which would index the device array one scalar at a
    # time (gpena/Bramble.jl#361).
    _hostvec(u) = Array(u isa Bramble.VectorElement ? parent(u) : u)
    @testset "Metal linear assembly into an element or a strided view (gpena/Bramble.jl#361)" begin
        for npts in ((13,), (13, 11))
            Wc, Wg = _matched_spaces(npts)
            for (label, sc, sg, l) in (("scalar", Wc, Wg, v -> innerₕ(_f, v)),
                ("composite", Wc × Wc, Wg × Wg, V -> innerₕ(_f, V[1]) + innerₕ(2.0f0, V[2])))
                lc, lg = form(sc, l), form(sg, l)
                bc = assemble(lc)
                n = length(bc)
                targets = (("element", () -> Bramble.element(sg, 0.0f0)),
                    ("strided view", () -> view(Metal.zeros(Float32, 2n), 1:2:(2n))))
                for (tlabel, target) in targets
                    @testset "$(length(npts))D $label, $tlabel" begin
                        u = target()
                        assemble!(u, lg)
                        @test _vrelerr(_hostvec(u), bc) < _RTOL
                        Bramble.assemble_add!(u, lg)
                        Bramble.assemble_add!(u, lg, 0.5f0)
                        @test _vrelerr(_hostvec(u), 2.5f0 .* bc) < _RTOL
                        Bramble.assemble_parallel!(u, lg)
                        @test _vrelerr(_hostvec(u), bc) < _RTOL
                    end
                end
            end
        end
    end

    # A `dirac` source assembles on a Metal space: its weight is computed in the mesh's
    # element type promoted with the strength's, so a Float32 strength on a Float32 space
    # gives a Float32 vector the device path can upload (a Float64 hardcode made it fail).
    # The sources hold host points, which the host-mirror sweep reads as they are
    # (gpena/Bramble.jl#361).
    @testset "Metal dirac sources (gpena/Bramble.jl#361)" begin
        for npts in ((13,), (13, 11))
            Wc, Wg = _matched_spaces(npts)
            D = length(npts)
            p1 = D == 1 ? 0.37f0 : (0.37f0, 0.61f0)
            pts = D == 1 ? [(0.2f0,), (0.55f0,), (0.8f0,)] :
                  [(0.2f0, 0.3f0), (0.55f0, 0.7f0)]
            for (label, l) in (("one point", v -> innerₕ(dirac(p1, 2.0f0), v)),
                ("points", v -> innerₕ(dirac(pts, [1.5f0, -0.5f0, 2.0f0][1:length(pts)]), v)))
                @testset "$(D)D $label" begin
                    bc = assemble(form(Wc, l))
                    @test eltype(bc) === Float32
                    lg = form(Wg, l)
                    bg = assemble(lg)
                    @test bg isa Metal.MtlArray
                    @test _vrelerr(bg, bc) < _RTOL
                    fill!(bg, 0.0f0)
                    assemble!(bg, lg)
                    @test _vrelerr(bg, bc) < _RTOL
                end
            end
        end
    end
end

end # module
