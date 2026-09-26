module ExtMetalFullstackTests

using Test
using Bramble
using Bramble: divₕ!, curlₕ!, Δₕ!
using Bramble: D₋ᵧ, D₋ₓ, D₋ₓ!, Mᵧ, Mₓ, change_points!, half_points, inner₊ᵧ, inner₊ₓ,
               jumpᵧ, jumpₓ, spacings, weights
using Metal
using SparseArrays
using LinearAlgebra
using Random
using ..TestUtils: _run_gpu_tests

# The Metal full stack's own test file (gpena/Bramble.jl#94, S2.7): every layer S2.1-S4.2
# built -- mesh, grid space, Rₕ!/avgₕ!, difference/jump/average operators, inner products,
# operator matrices and an assembled system matrix -- checked against the `Float64` CPU
# result within `Float32` tolerance, on a genuine Metal device.
#
# `Metal.functional() && _run_gpu_tests()` gates every testset below. Unlike
# `test/ext/metal_ext.jl`'s `@test_skip`-only skip path, a host without a functional device
# `@warn`s here too: a silently skipped file was exactly issue #84's failure mode, once
# already repeated in this milestone (see S3.3), and this file must not repeat it a second
# time. `_run_gpu_tests()` (TestUtils.jl) is the explicit CI opt-out on top of
# `Metal.functional()`: GitHub's hosted macOS runners are real Apple Silicon hardware, so
# `Metal.functional()` alone would let this file actually execute GPU kernels, unattended,
# in CI.
if !Metal.functional() || !_run_gpu_tests()
    @warn "Skipping Metal full-stack tests: Metal.functional() is false, or GPU tests are skipped in CI"
    @test_skip "Metal full-stack tests not exercised: Metal.functional() is false, or GPU tests are skipped in CI"
else
    const _TOL = 1.0f-4

    f1(x) = sin(x[1])
    f2(x) = sin(x[1]) * cos(x[2])

    # 1D CPU/Metal mesh & grid space pair
    n1 = 33
    Ωc1 = mesh(domain(interval(0.0, 1.0)), n1, true)
    Ωg1 = mesh(domain(interval(0.0f0, 1.0f0)), n1, true; backend = metal_backend())
    Wc1 = gridspace(Ωc1)
    Wg1 = gridspace(Ωg1)

    # 2D CPU/Metal mesh & grid space pair
    n2 = (9, 7)
    Ωc2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), n2, (true, true))
    Ωg2 = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), n2, (true, true);
        backend = metal_backend())
    Wc2 = gridspace(Ωc2)
    Wg2 = gridspace(Ωg2)

    # S15 (gpena/Bramble.jl#174, #302-#306): kernel fusion, asynchrony and CPU/GPU
    # equivalence for the mesh and vector-calculus kernels S8-S12 shipped.
    #
    # `rtol = 1.0f-5` throughout, per the subplan's own instruction -- no bitwise assertion
    # below. Bramble's own non-uniform point generator draws i.i.d. random `Float32`
    # coordinates (`rand!`), so two independently-built non-uniform meshes are never compared
    # directly: every non-uniform helper below mirrors one side's realized coordinates onto
    # the other with `change_points!` instead (bramble-metal §3), using a deterministic
    # stretch rather than `rand!` so the KA/Metal RNG-advancing quirk (#320) never enters the
    # picture at all.
    const _RTOL5 = 1.0f-5
    _close(a, b) = all(isapprox.(a, b; rtol = _RTOL5, atol = _RTOL5))

    # A mild, deterministic, strictly increasing non-uniform stretch over [0, 1] (spacing
    # ratio ~2.8 -- genuinely non-uniform, nothing pathological), the same construction
    # `.agents/plans/checks/s12-stencils.jl`'s `_stretch_points`-style helper uses.
    function _stretch_points(n::Int)
        t = Float32[(i - 1) / (n - 1) for i in 1:n]
        raw = Float32.(t .+ 0.15f0 .* sin.(Float32(pi) .* t))
        raw = (raw .- raw[1]) ./ (raw[end] - raw[1])
        return raw
    end

    # Mirrors `uc`'s (CPU) coefficients onto a device grid function on `Wg` via one bulk
    # `copyto!` -- never independently evaluated on each backend, since `sin`/`cos` disagree
    # at the ULP level between Metal's device math library and libm, which a difference
    # operator's division by `h` amplifies past `rtol = 1f-5` (S11's own finding).
    function _mirror_to_device(Wg, uc)
        ug = element(Wg)
        copyto!(parent(ug), parent(uc))
        return ug
    end

    # A CPU/device mesh pair over `interval(0, 1)` with `n` points, coordinates matched
    # exactly: uniform on both sides, or the same deterministic stretch mirrored onto both
    # via `change_points!` for non-uniform.
    function _matched_meshes_1d(n::Int, uniform::Bool)
        Ω = domain(interval(0.0f0, 1.0f0))
        Ωc = mesh(Ω, n, uniform)
        Ωg = mesh(Ω, n, uniform; backend = metal_backend())
        if !uniform
            pts = _stretch_points(n)
            change_points!(Ωc, copy(pts))
            change_points!(Ωg, Metal.MtlVector(pts))
        end
        return Ωc, Ωg
    end

    # Same, for `D` in {2, 3}: an `n^D` CPU/device mesh pair over the unit square/cube, each
    # axis mirrored independently.
    function _matched_meshes_nd(n::Int, D::Int, uniform::Bool)
        if D == 2
            Ω = domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0))
            unif = (uniform, uniform)
        else
            Ω = domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0))
            unif = (uniform, uniform, uniform)
        end
        npts = ntuple(_ -> n, Val(D))
        Ωc = mesh(Ω, npts, unif)
        Ωg = mesh(Ω, npts, unif; backend = metal_backend())
        if !uniform
            for d in 1:D
                pts = _stretch_points(n)
                change_points!(Ωc(d), copy(pts))
                change_points!(Ωg(d), Metal.MtlVector(pts))
            end
        end
        return Ωc, Ωg
    end

    function _mesh_arrays(Ωₕ)
        return (
            pts = Array(points(Ωₕ)), half_pts = Array(half_points(Ωₕ)),
            sp = Array(spacings(Ωₕ)), hsp = Array(Bramble.half_spacings(Ωₕ))
        )
    end

    # #304/#305: unlike `_matched_meshes_1d`/`_matched_meshes_nd` above (a deterministic
    # stretch forced onto both sides, used where the two sides just need to agree on SOME
    # non-uniform coordinates), this builds the device mesh's non-uniform coordinates the
    # normal way -- its own device point-generation kernel (#304), the same call every real
    # caller makes -- then reads those realized points back and mirrors them onto a CPU
    # mesh via `change_points!`. That is what lets the mesh-kernel testset below actually
    # exercise #304's own kernel rather than bypass it, while still comparing against a CPU
    # reference for the exact coordinates it produced (bramble-metal §3; the same
    # discipline S10's own check script uses).
    function _mirror_device_nonuniform_1d(n::Int)
        Ωg = mesh(domain(interval(0.0f0, 1.0f0)), n, false; backend = metal_backend())
        pts = Array(points(Ωg))
        Ωc = mesh(domain(interval(0.0f0, 1.0f0)), n, true)
        change_points!(Ωc, copy(pts))
        return Ωc, Ωg
    end

    function _mirror_device_nonuniform_nd(n::Int, D::Int)
        if D == 2
            Ω = domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0))
        else
            Ω = domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0))
        end
        npts = ntuple(_ -> n, Val(D))
        Ωg = mesh(Ω, npts, ntuple(_ -> false, Val(D)); backend = metal_backend())
        Ωc = mesh(Ω, npts, ntuple(_ -> true, Val(D)))
        for d in 1:D
            pts = Array(points(Ωg(d)))
            change_points!(Ωc(d), copy(pts))
        end
        return Ωc, Ωg
    end

    @testset "Metal full stack" begin
        @testset "mesh: point coordinates match CPU" begin
            @test isapprox(Array(points(Ωg1)), Float32.(points(Ωc1)); atol = 1.0f-6)
            for d in 1:2
                @test isapprox(
                    Array(points(Ωg2(d))), Float32.(points(Ωc2(d)));
                    atol = 1.0f-6
                )
            end
        end

        @testset "grid space: quadrature weights match CPU" begin
            wg1 = weights(Wg1)
            wc1 = weights(Wc1)
            @test isapprox(Array(wg1.innerh.factors[1]), Float32.(wc1.innerh.factors[1]); atol = 1.0f-6)
            @test isapprox(Array(wg1.aligned[1]), Float32.(wc1.aligned[1]); atol = 1.0f-6)
            @test isapprox(Array(wg1.cellfactor[1]), Float32.(wc1.cellfactor[1]); atol = 1.0f-6)

            wg2 = weights(Wg2)
            wc2 = weights(Wc2)
            for d in 1:2
                @test isapprox(
                    Array(wg2.innerh.factors[d]), Float32.(wc2.innerh.factors[d]); atol = 1.0f-6
                )
                @test isapprox(Array(wg2.aligned[d]), Float32.(wc2.aligned[d]); atol = 1.0f-6)
            end
        end

        # `element(Wₕ, α)` filled through the `VectorElement` wrapper, so Base's generic
        # `fill!` stored one point at a time and scalar-indexed the device array.
        # `Metal.allowscalar(false)` is the default, so the call raised
        # "Scalar indexing is disallowed." rather than running slowly.
        @testset "element(Wₕ, α) fills device storage without scalar indexing" begin
            Metal.allowscalar(false)
            u1 = element(Wg1, 2.0f0)
            @test parent(u1) isa MtlVector{Float32}
            @test all(==(2.0f0), Array(parent(u1)))

            u2 = element(Wg2, 0.0f0)
            @test all(==(0.0f0), Array(parent(u2)))

            # An `Int` fill still promotes against the backend's `Float32`.
            u3 = element(Wg2, 1)
            @test eltype(parent(u3)) === Float32
            @test all(==(1.0f0), Array(parent(u3)))

            # Same for the vector constructor, from host and from device memory alike.
            u4 = element(Wg1, ones(Float32, ndofs(Wg1)))
            @test all(==(1.0f0), Array(parent(u4)))

            u5 = element(Wg1, Metal.ones(Float32, ndofs(Wg1)))
            @test all(==(1.0f0), Array(parent(u5)))
        end

        @testset "Rₕ!/avgₕ!: projection and cell average match CPU" begin
            uc1 = element(Wc1)
            ug1 = element(Wg1)
            Rₕ!(uc1, f1)
            Rₕ!(ug1, f1)
            @test isapprox(Array(parent(ug1)), Float32.(parent(uc1)); atol = _TOL)

            ac1 = element(Wc1)
            ag1 = element(Wg1)
            avgₕ!(ac1, f1)
            avgₕ!(ag1, f1)
            @test isapprox(Array(parent(ag1)), Float32.(parent(ac1)); atol = _TOL)

            uc2 = element(Wc2)
            ug2 = element(Wg2)
            Rₕ!(uc2, f2)
            Rₕ!(ug2, f2)
            @test isapprox(Array(parent(ug2)), Float32.(parent(uc2)); atol = _TOL)

            ac2 = element(Wc2)
            ag2 = element(Wg2)
            avgₕ!(ac2, f2)
            avgₕ!(ag2, f2)
            @test isapprox(Array(parent(ag2)), Float32.(parent(ac2)); atol = _TOL)
        end

        # A masked call (`markers` non-empty) has no device kernel by design (#297, deferred to
        # v3.5.0) and must raise, not silently run on the CPU or produce a wrong value.
        @testset "Rₕ!/avgₕ!: masked call raises (issue #297)" begin
            u = element(Wg1)
            err = try
                Rₕ!(u, f1; markers = (:boundary,))
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("no device kernel", sprint(showerror, err))

            a = element(Wg1)
            err2 = try
                avgₕ!(a, f1; markers = (:boundary,))
                nothing
            catch e
                e
            end
            @test err2 isa ArgumentError
            @test occursin("no device kernel", sprint(showerror, err2))
        end

        @testset "difference / jump / average operators match CPU" begin
            uc1 = Rₕ(Wc1, f1)
            ug1 = Rₕ(Wg1, f1)
            @test isapprox(Array(parent(D₋ₓ(ug1))), Float32.(parent(D₋ₓ(uc1))); atol = _TOL)
            @test isapprox(Array(parent(jumpₓ(ug1))), Float32.(parent(jumpₓ(uc1))); atol = _TOL)
            @test isapprox(Array(parent(Mₓ(ug1))), Float32.(parent(Mₓ(uc1))); atol = _TOL)

            uc2 = Rₕ(Wc2, f2)
            ug2 = Rₕ(Wg2, f2)
            @test isapprox(Array(parent(D₋ᵧ(ug2))), Float32.(parent(D₋ᵧ(uc2))); atol = _TOL)
            @test isapprox(Array(parent(jumpᵧ(ug2))), Float32.(parent(jumpᵧ(uc2))); atol = _TOL)
            @test isapprox(Array(parent(Mᵧ(ug2))), Float32.(parent(Mᵧ(uc2))); atol = _TOL)
        end

        @testset "the three inner products match CPU" begin
            uc1 = Rₕ(Wc1, f1)
            ug1 = Rₕ(Wg1, f1)
            @test isapprox(Float64(innerₕ(ug1, ug1)), innerₕ(uc1, uc1); rtol = _TOL)
            @test isapprox(Float64(inner₊(ug1, ug1, Val(()))), inner₊(uc1, uc1, Val(())); rtol = _TOL)
            @test isapprox(Float64(inner₊ₓ(ug1, ug1)), inner₊ₓ(uc1, uc1); rtol = _TOL)
        end

        @testset "operator matrices come back in the backend's matrix type" begin
            Ag1 = D₋ₓ(Wg1)
            Ac1 = D₋ₓ(Wc1)
            @test Ag1 isa MtlMatrix
            @test isapprox(Array(Ag1), Float32.(Array(Ac1)); atol = _TOL)

            Ag2 = D₋ᵧ(Wg2)
            Ac2 = D₋ᵧ(Wc2)
            @test Ag2 isa MtlMatrix
            @test isapprox(Array(Ag2), Float32.(Array(Ac2)); atol = _TOL)
        end

        @testset "assembled system matrix matches CPU" begin
            a1(u, v) = inner₊ₓ(D₋ₓ(u), D₋ₓ(v))
            Ag1 = assemble(form(Wg1, Wg1, a1))
            Ac1 = assemble(form(Wc1, Wc1, a1))
            @test isapprox(Array(SparseMatrixCSC(Ag1)), Array(Ac1); rtol = _TOL)

            a2(u, v) = inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v))
            Ag2 = assemble(form(Wg2, Wg2, a2))
            Ac2 = assemble(form(Wc2, Wc2, a2))
            @test isapprox(Array(SparseMatrixCSC(Ag2)), Array(Ac2); rtol = _TOL)
        end

        # The device scatter mirror used to live in a module-global `Dict` keyed on
        # `objectid(A)`, threaded through ten sweep functions by hand as a trailing `mirror`
        # argument (gpena/Bramble.jl#94, S4.2). Issue #313 moved it onto the matrix itself: a
        # `MetalSparseMatrixCSR` now carries its own `mirror` field (host `rowptr`/`colval`
        # copies in the matrix's own index type, plus the `nzval` staging vector), and
        # `_scatter_position`/`_scatter_add!` read it off `A` directly instead of resolving it
        # from a cache. This testset checks that field is actually there and actually reused,
        # not just that assembly still gives the right answer (the testsets above already
        # cover that).
        @testset "the device CSR carries its own scatter mirror (issue #313)" begin
            a1(u, v) = inner₊ₓ(D₋ₓ(u), D₋ₓ(v))
            Ag1 = assemble(form(Wg1, Wg1, a1))
            Ac1 = assemble(form(Wc1, Wc1, a1))

            @test hasproperty(Ag1, :mirror)
            @test eltype(Ag1.mirror.nzval) == eltype(Ag1.nzVal)
            @test eltype(Ag1.mirror.rowptr) == eltype(Ag1.rowPtr)
            @test eltype(Ag1.mirror.colval) == eltype(Ag1.colVal)
            @test length(Ag1.mirror.nzval) == length(Ag1.nzVal)

            # A second `assemble!` on the same matrix must reuse the same mirror object rather
            # than rebuild it: the module-global cache this replaced could rebuild mid-sweep
            # and lose entries already scattered into the discarded instance (S4.2, round 7).
            # With the mirror a field of `A`, there is nothing left to rebuild.
            mirror_before = Ag1.mirror
            assemble!(Ag1, form(Wg1, Wg1, a1))
            @test Ag1.mirror === mirror_before
            @test isapprox(Array(SparseMatrixCSC(Ag1)), Array(Ac1); rtol = _TOL)
        end

        # Two device-scatter races (both now fixed in `src/assembly/`) only ever surfaced at a grid
        # large enough to give the sweep many scattered entries, and only across many repeated
        # assemblies -- a single small assembly (the testset above, n=33/(9,7)) always looked
        # correct even while both races were live. `_zero_stored!`'s device `fill!` over
        # `A.nzVal` used to race the later flush and could wipe entries already written, and the
        # host mirror used to be re-resolved from a global cache on every scattered entry and
        # could be rebuilt mid-sweep, discarding entries already scattered into the old
        # instance. This testset follows the same "repeat at a size that can fail" shape S4.2's
        # own CHECK was rewritten to use, so a regression of either race shows up here as a
        # mismatch rather than looking correct forever. No `Metal.synchronize()`: this is exactly
        # the class of bug an explicit synchronise would mask, not fix.
        @testset "assembled system matrix matches CPU, repeated at a size that can race" begin
            n1r = 513
            Wg1r = gridspace(mesh(domain(interval(0.0f0, 1.0f0)), n1r, true; backend = metal_backend()))
            Wc1r = gridspace(mesh(domain(interval(0.0, 1.0)), n1r, true))
            a1r(u, v) = inner₊ₓ(D₋ₓ(u), D₋ₓ(v))
            Ac1r = Array(assemble(form(Wc1r, Wc1r, a1r)))
            for _ in 1:40
                Ag1r = assemble(form(Wg1r, Wg1r, a1r))
                @test isapprox(Array(SparseMatrixCSC(Ag1r)), Ac1r; rtol = _TOL)
            end

            n2r = (65, 63)
            Wg2r = gridspace(mesh(
                domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), n2r, (true, true);
                backend = metal_backend()
            ))
            Wc2r = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), n2r, (true, true)))
            a2r(u, v) = inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v))
            Ac2r = Array(assemble(form(Wc2r, Wc2r, a2r)))
            for _ in 1:40
                Ag2r = assemble(form(Wg2r, Wg2r, a2r))
                @test isapprox(Array(SparseMatrixCSC(Ag2r)), Ac2r; rtol = _TOL)
            end
        end

        # #303 (uniform init), #305 (non-uniform metrics), #304 (non-uniform point
        # generation on device): the fused mesh kernels' points, spacings, half points and
        # half spacings must match the CPU arrays. Uniform is compared directly (both sides
        # built from the same `a`, `n`); non-uniform mirrors the device mesh's own realized
        # coordinates onto the CPU side, never two independently-generated non-uniform
        # meshes (see the file-level note above).
        @testset "fused mesh kernels match the CPU arrays (#303, #304, #305)" begin
            # Uniform (#303): points, spacings, half points and half spacings all computed
            # independently on each side, from `a`, `h`, `n` alone.
            for n in (5, 1025)
                Ωc, Ωg = _matched_meshes_1d(n, true)
                c, g = _mesh_arrays(Ωc), _mesh_arrays(Ωg)
                @test _close(g.pts, c.pts)
                @test _close(g.sp, c.sp)
                @test _close(g.half_pts, c.half_pts)
                @test _close(g.hsp, c.hsp)
            end
            for D in (2, 3), n in (5, 9)

                Ωc, Ωg = _matched_meshes_nd(n, D, true)
                for d in 1:D
                    c, g = _mesh_arrays(Ωc(d)), _mesh_arrays(Ωg(d))
                    @test _close(g.pts, c.pts)
                    @test _close(g.sp, c.sp)
                    @test _close(g.half_pts, c.half_pts)
                    @test _close(g.hsp, c.hsp)
                end
            end

            # Non-uniform (#304 device point generation, #305 device metrics): the device
            # mesh builds its own random coordinates the normal way, and only THOSE
            # coordinates, read back, are mirrored onto the CPU reference.
            for n in (5, 1025)
                Ωc, Ωg = _mirror_device_nonuniform_1d(n)
                c, g = _mesh_arrays(Ωc), _mesh_arrays(Ωg)
                @test _close(g.pts, c.pts)
                @test _close(g.sp, c.sp)
                @test _close(g.half_pts, c.half_pts)
                @test _close(g.hsp, c.hsp)
            end
            for D in (2, 3), n in (5, 9)

                Ωc, Ωg = _mirror_device_nonuniform_nd(n, D)
                for d in 1:D
                    c, g = _mesh_arrays(Ωc(d)), _mesh_arrays(Ωg(d))
                    @test _close(g.pts, c.pts)
                    @test _close(g.sp, c.sp)
                    @test _close(g.half_pts, c.half_pts)
                    @test _close(g.hsp, c.hsp)
                end
            end
        end

        # #320 (S1.2): `_seed_mesh1d_rng!` isolates non-uniform mesh point generation onto a
        # package-local RNG immune to whatever a device kernel launch does to the global
        # stream -- so seeding it, building a non-uniform mesh on the host, seeding it again
        # with the same seed, and building the "same" non-uniform mesh on a Metal backend
        # produces matching interior coordinates. Before this fix, the device path's other
        # kernel launches (metrics, fused init, ...) had already burned draws from
        # `Random.default_rng()` by the time point generation ran, so the two builds
        # disagreed even under the same `Random.seed!`.
        @testset "seeded non-uniform mesh matches across host/Metal backends (#320)" begin
            n = 33
            seed = 20260920

            Bramble._seed_mesh1d_rng!(seed)
            Ωc_seeded = mesh(domain(interval(0.0, 1.0)), n, false)
            Bramble._seed_mesh1d_rng!(seed)
            Ωg_seeded = mesh(domain(interval(0.0f0, 1.0f0)), n, false; backend = metal_backend())
            Bramble._unseed_mesh1d_rng!()

            # Host draws are Float64, device draws get converted to Float32 (see
            # mesh1d.jl's `_generate_random_points!`), so the comparison tolerates that
            # rounding rather than demanding exact equality.
            @test isapprox(
                Array(points(Ωg_seeded)), Float32.(points(Ωc_seeded)); rtol = 1.0f-6
            )

            # Negative control: without arming `_seed_mesh1d_rng!`, `Random.seed!(N)` alone
            # still controls a non-uniform mesh build exactly as before -- the pattern the
            # ~27 other test files already rely on -- proving the opt-in design didn't
            # regress legacy behavior.
            Random.seed!(20260921)
            a = mesh(domain(interval(0.0, 1.0)), n, false)
            Random.seed!(20260921)
            b = mesh(domain(interval(0.0, 1.0)), n, false)
            @test collect(points(a)) == collect(points(b))
        end

        # #302: no eager `synchronize` remains under `GpuKernel` (S11 removed all 21 call
        # sites), so a chained sequence with no explicit sync between calls must still match
        # the CPU on EVERY repetition, not just the first -- two real device races in this
        # repository were invisible at one small run and only showed up across 40
        # repetitions at n >= 1025 (bramble-metal §3).
        @testset "chained asynchronous sequence, 40 reps at n = 1025x1025, matches CPU every time (#302)" begin
            n = 1025
            Ωc, Ωg = _matched_meshes_nd(n, 2, false)
            Wc, Wg = gridspace(Ωc), gridspace(Ωg)
            f1 = x -> sin(Float32(pi) * x[1]) * cos(Float32(pi) * x[2])
            f2 = x -> cos(Float32(pi) * x[1]) * x[2]^2
            u1c, u2c = Rₕ(Wc, f1), Rₕ(Wc, f2)
            u1g, u2g = _mirror_to_device(Wg, u1c), _mirror_to_device(Wg, u2c)

            # CPU references, built once: deterministic pointwise stencils, compared every
            # repetition below, not rebuilt every repetition.
            dx_c = Array(parent(D₋ₓ(u1c)))
            div_c = Array(parent(divₕ((u1c, u2c))))
            lap_c = Array(parent(Δₕ(u1c)))
            curl_c = Array(parent(curlₕ((u1c, u2c))))

            # Destinations preallocated once, outside the loop, so nothing but the four
            # launches themselves sits between them. Every repetition below launches all
            # four in-place device operators back to back with NO host transfer in between
            # -- and so no synchronization in between, `Array(parent(...))` is itself a
            # synchronization point -- and only reads the four buffers back once every
            # launch has already been enqueued. An earlier version of this testset read each
            # result back right after its own launch, which forced every operator to
            # complete before the next was even enqueued: four independent one-launch
            # sequences repeated 40 times, not the one four-deep unsynchronised chain #302 is
            # actually about (a queued device write from an earlier stage still in flight
            # when a later stage reads the same buffer).
            dx_dest = element(Wg)
            div_dest = element(Wg)
            lap_dest = element(Wg)
            curl_dest = element(Wg)

            bad = 0
            first_bad = 0
            worst = 0.0f0
            for rep in 1:40
                D₋ₓ!(dx_dest, u1g)
                divₕ!(div_dest, (u1g, u2g))
                Δₕ!(lap_dest, u1g)
                curlₕ!(curl_dest, (u1g, u2g))

                # One host transfer per buffer, only now that all four launches are queued.
                dx_g = Array(parent(dx_dest))
                div_g = Array(parent(div_dest))
                lap_g = Array(parent(lap_dest))
                curl_g = Array(parent(curl_dest))

                ok = _close(dx_g, dx_c) && _close(div_g, div_c) &&
                     _close(lap_g, lap_c) && _close(curl_g, curl_c)
                if !ok
                    bad += 1
                    first_bad == 0 && (first_bad = rep)
                    rep_worst = maximum(
                        (
                        maximum(abs, dx_g .- dx_c), maximum(abs, div_g .- div_c),
                        maximum(abs, lap_g .- lap_c), maximum(abs, curl_g .- curl_c)
                    )
                    )
                    worst = max(worst, rep_worst)
                end
            end
            # A failure prints `(bad, first_bad, worst)` against `(0, 0, 0.0f0)`, so which
            # repetition failed first and by how much is visible in the test report itself,
            # not just that some repetition did.
            @test (bad, first_bad, worst) == (0, 0, 0.0f0)
        end

        # #306: the fused vector-calculus operators (divₕ, div₊ₕ, curlₕ, curl₊ₕ, Δₕ, εₕ)
        # match the CPU in 1D, 2D and 3D, on non-uniform meshes -- non-uniform is the case
        # this package exists for, not the degenerate uniform special case.
        @testset "fused vector-calculus operators match CPU in 1D/2D/3D, non-uniform (#306)" begin
            # 1D: curlₕ/εₕ are not defined in 1D (curlₕ raises rather than returning a
            # zero), so only divₕ and Δₕ apply.
            for n in (9, 1025)
                Ωc, Ωg = _matched_meshes_1d(n, false)
                Wc, Wg = gridspace(Ωc), gridspace(Ωg)
                f = x -> sin(3.0f0 * x)
                uc = Rₕ(Wc, f)
                ug = _mirror_to_device(Wg, uc)
                @test _close(Array(parent(divₕ(ug))), Array(parent(divₕ(uc))))
                @test _close(Array(parent(Δₕ(ug))), Array(parent(Δₕ(uc))))
            end

            # 2D: every operator this subplan fused a device kernel for.
            for n in (9, 33)
                Ωc, Ωg = _matched_meshes_nd(n, 2, false)
                Wc, Wg = gridspace(Ωc), gridspace(Ωg)
                f1 = x -> sin(Float32(pi) * x[1]) * cos(Float32(pi) * x[2])
                f2 = x -> cos(Float32(pi) * x[1]) * x[2]^2
                u1c, u2c = Rₕ(Wc, f1), Rₕ(Wc, f2)
                u1g, u2g = _mirror_to_device(Wg, u1c), _mirror_to_device(Wg, u2c)

                @test _close(Array(parent(divₕ((u1g, u2g)))), Array(parent(divₕ((u1c, u2c)))))
                @test _close(
                    Array(parent(Bramble.div₊ₕ((u1g, u2g)))), Array(parent(Bramble.div₊ₕ((u1c, u2c))))
                )
                @test _close(Array(parent(curlₕ((u1g, u2g)))), Array(parent(curlₕ((u1c, u2c)))))
                @test _close(
                    Array(parent(Bramble.curl₊ₕ((u1g, u2g)))), Array(parent(Bramble.curl₊ₕ((u1c, u2c))))
                )
                @test _close(Array(parent(Δₕ(u1g))), Array(parent(Δₕ(u1c))))

                epsc, epsg = εₕ((u1c, u2c)), εₕ((u1g, u2g))
                for i in 1:2, j in 1:2

                    @test _close(Array(parent(epsg[i][j])), Array(parent(epsc[i][j])))
                end
            end

            # 3D: divₕ, div₊ₕ, Δₕ, the 3-component curlₕ, and εₕ.
            for n in (9, 17)
                Ωc, Ωg = _matched_meshes_nd(n, 3, false)
                Wc, Wg = gridspace(Ωc), gridspace(Ωg)
                fs = (
                    x -> sin(Float32(pi) * x[1]) * cos(Float32(pi) * x[2]) * x[3],
                    x -> cos(Float32(pi) * x[1]) * x[2]^2 * x[3],
                    x -> x[1] * x[2] * sin(Float32(pi) * x[3])
                )
                ucs = Tuple(Rₕ(Wc, f) for f in fs)
                ugs = Tuple(_mirror_to_device(Wg, uc) for uc in ucs)

                @test _close(Array(parent(divₕ(ugs))), Array(parent(divₕ(ucs))))
                @test _close(Array(parent(Bramble.div₊ₕ(ugs))), Array(parent(Bramble.div₊ₕ(ucs))))
                @test _close(Array(parent(Δₕ(ugs[1]))), Array(parent(Δₕ(ucs[1]))))

                curlc, curlg = curlₕ(ucs), curlₕ(ugs)
                for k in 1:3
                    @test _close(Array(parent(curlg[k])), Array(parent(curlc[k])))
                end

                epsc, epsg = εₕ(ucs), εₕ(ugs)
                for i in 1:3, j in 1:3

                    @test _close(Array(parent(epsg[i][j])), Array(parent(epsc[i][j])))
                end
            end
        end

        # #174's own remaining criterion: CPU and GPU agree within `Float32` tolerance
        # (`rtol = 1f-5`) for the difference operators and `πₕ` -- not bitwise. Non-uniform,
        # mirrored coefficients, same discipline as the rest of this file.
        @testset "#174: CPU/GPU agree within Float32 tolerance (rtol = 1f-5), not bitwise" begin
            n = 129
            Ωc, Ωg = _matched_meshes_nd(n, 2, false)
            Wc, Wg = gridspace(Ωc), gridspace(Ωg)
            f = x -> sin(Float32(pi) * x[1]) * cos(Float32(pi) * x[2])
            uc = Rₕ(Wc, f)
            ug = _mirror_to_device(Wg, uc)

            @test isapprox(Array(parent(D₋ₓ(ug))), Array(parent(D₋ₓ(uc))); rtol = _RTOL5, atol = _RTOL5)
            @test isapprox(Array(parent(D₋ᵧ(ug))), Array(parent(D₋ᵧ(uc))); rtol = _RTOL5, atol = _RTOL5)

            # πₕ, interpolating a non-uniform source grid function onto a different-
            # resolution non-uniform destination mesh, CPU vs device -- source and
            # destination coordinates each mirrored independently across backends.
            n_src, n_dst = 65, 33
            Ωc_src, Ωg_src = _matched_meshes_1d(n_src, false)
            Ωc_dst, Ωg_dst = _matched_meshes_1d(n_dst, false)
            Wc_src, Wg_src = gridspace(Ωc_src), gridspace(Ωg_src)
            Wc_dst, Wg_dst = gridspace(Ωc_dst), gridspace(Ωg_dst)

            fπ = x -> sin(2.0f0 * x) * x^2
            uc_src = Rₕ(Wc_src, fπ)
            ug_src = _mirror_to_device(Wg_src, uc_src)

            # `πₕ!` (in place), not the allocating `πₕ(Wₕ, src)`: the latter goes through a
            # per-point `Rₕ`/`interpolate_at` evaluation that scalar-indexes its device
            # source, exactly the failure mode `test/ext/metal_ext.jl`'s own interpolation
            # testset (#312) avoids the same way.
            uc_dst = element(Wc_dst)
            πₕ!(uc_dst, uc_src)
            ug_dst = element(Wg_dst)
            πₕ!(ug_dst, ug_src)
            @test isapprox(Array(parent(ug_dst)), Array(parent(uc_dst)); rtol = _RTOL5, atol = _RTOL5)
        end
    end
end # if Metal.functional() && _run_gpu_tests()

end # module ExtMetalFullstackTests
