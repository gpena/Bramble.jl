module ExtMetalFullstackTests

using Test
using Bramble
using Metal
using SparseArrays
using LinearAlgebra

# The Metal full stack's own test file (gpena/Bramble.jl#94, S2.7): every layer S2.1-S4.2
# built -- mesh, grid space, Rₕ!/avgₕ!, difference/jump/average operators, inner products,
# operator matrices and an assembled system matrix -- checked against the `Float64` CPU
# result within `Float32` tolerance, on a genuine Metal device.
#
# `Metal.functional()` gates every testset below. Unlike `test/ext/metal_ext.jl`'s
# `@test_skip`-only skip path, a host without a functional device `@warn`s here too:
# a silently skipped file was exactly issue #84's failure mode, once already repeated in
# this milestone (see S3.3), and this file must not repeat it a second time.
if !Metal.functional()
    @warn "Skipping Metal full-stack tests: Metal.functional() is false on this host"
    @test_skip "Metal full-stack tests not exercised: Metal.functional() is false"
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

        # Two device-scatter races (both now fixed in `src/form/`) only ever surfaced at a grid
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
    end
end # if Metal.functional()

end # module ExtMetalFullstackTests
