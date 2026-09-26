module FormAssembleAddTests

using Test
using Bramble
using SparseArrays
using Random
using Bramble: Serial, Parallel, backend, execution_policy, allocate_system_matrix

# `assemble_add!` (gpena/Bramble.jl#231) accumulates a form's contribution into an already
# filled matrix/vector, in place, without the `fill!` `assemble!` does first. Every check
# here goes against an independent reference built from `assemble` alone (never against
# another call to the code under test), following bramble-verification.

# Allocation checks go behind a function barrier (bramble-verification §1): measured at
# top-level or `@testset` scope, global-variable access alone can report spurious bytes
# that have nothing to do with the kernel under test.
_alloc(f::F, args...) where {F} = (f(args...); @allocated f(args...))

@testset "assemble_add! (#231)" begin
    @testset "Bilinear: unscaled accumulation matches assemble-then-add, 1D/2D/3D" begin
        cases = (
            ("1D", mesh(domain(interval(0.0, 1.0)), 21, true)),
            ("2D", mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 11), (true, true))),
            (
                "3D",
                mesh(
                    domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
                    (5, 6, 7),
                    (true, true, true)
                )
            )
        )
        for (lbl, Ωₕ) in cases
            @testset "$lbl" begin
                Wₕ = gridspace(Ωₕ)
                m_form = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
                k_form = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
                wide = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))

                A = allocate_system_matrix(wide)
                fill!(nonzeros(A), 0.0)
                assemble_add!(A, m_form)
                assemble_add!(A, k_form)

                Aref = Matrix(assemble(m_form)) .+ Matrix(assemble(k_form))
                @test Matrix(A) ≈ Aref
            end
        end
    end

    @testset "Bilinear: scaled accumulation, Number and RefValue" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (11, 13), (true, true))
        Wₕ = gridspace(Ωₕ)
        m_form = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        k_form = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        wide = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        M = Matrix(assemble(m_form))
        K = Matrix(assemble(k_form))

        @testset "plain Number" begin
            A = allocate_system_matrix(wide)
            fill!(nonzeros(A), 0.0)
            assemble_add!(A, m_form, 1 / 0.01)
            assemble_add!(A, k_form, 0.75)
            @test Matrix(A) ≈ (1 / 0.01) .* M .+ 0.75 .* K
        end

        @testset "RefValue, changing between calls, replaying from cache" begin
            θ = Ref(1.0)
            A = allocate_system_matrix(wide)
            fill!(nonzeros(A), 0.0)
            assemble_add!(A, m_form)
            assemble_add!(A, k_form, θ)
            @test Matrix(A) ≈ M .+ 1.0 .* K

            θ[] = 3.5
            fill!(nonzeros(A), 0.0)
            assemble_add!(A, m_form)
            assemble_add!(A, k_form, θ)
            @test Matrix(A) ≈ M .+ 3.5 .* K
        end
    end

    @testset "Bilinear: zero allocation on warm replay (unscaled and RefValue-scaled)" begin
        function _warm_and_measure()
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 31, true)
            Wₕ = gridspace(Ωₕ)
            m_form = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
            k_form = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            wide = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
            A = allocate_system_matrix(wide)
            θ = Ref(2.0)

            fill!(nonzeros(A), 0.0)
            assemble_add!(A, m_form)      # cold: records
            assemble_add!(A, k_form, θ)   # cold: records

            allocs_unscaled = _alloc(assemble_add!, A, m_form)
            allocs_scaled = _alloc(assemble_add!, A, k_form, θ)
            return allocs_unscaled, allocs_scaled
        end
        allocs_unscaled, allocs_scaled = _warm_and_measure()
        @test allocs_unscaled == 0
        @test allocs_scaled == 0
    end

    @testset "Bilinear: composite space, one contribution per leaf" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, true)
        Wₕ = gridspace(Ωₕ)
        W = Wₕ × Wₕ

        a1 = form(W, W, (u, v) -> innerₕ(u(1), v(1)))
        a2 = form(W, W, (u, v) -> inner₊(∇ₕ(u(2)), ∇ₕ(v(2))))
        wide = form(
            W, W, (u, v) -> innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2)))
        )

        A = allocate_system_matrix(wide)
        fill!(nonzeros(A), 0.0)
        assemble_add!(A, a1)
        assemble_add!(A, a2, 2.0)

        Aref = Matrix(assemble(a1)) .+ 2.0 .* Matrix(assemble(a2))
        @test Matrix(A) ≈ Aref
    end

    @testset "Bilinear: a pattern too narrow raises, naming the entry" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, true)
        Wₕ = gridspace(Ωₕ)
        narrow_form = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))  # diagonal-only pattern
        wide_form = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))  # off-diagonal too

        A = allocate_system_matrix(narrow_form)
        @test_throws ArgumentError assemble_add!(A, wide_form)
    end

    @testset "Bilinear: correct and deterministic under Parallel() execution" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωₕ = mesh(domain(S, :walls => boundary_symbols(S)), (9, 11), (true, true))
        Ω_par = mesh(
            domain(S, :walls => boundary_symbols(S)),
            (9, 11),
            (true, true);
            backend = backend(policy = Parallel())
        )
        Wₕ = gridspace(Ωₕ)
        W_par = gridspace(Ω_par)
        @test execution_policy(W_par) isa Parallel

        m_form = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        m_par = form(W_par, W_par, (u, v) -> innerₕ(u, v))
        k_form = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        k_par = form(W_par, W_par, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        wide_par = form(
            W_par, W_par, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
        )

        Aref = Matrix(assemble(m_form)) .+ 2.0 .* Matrix(assemble(k_form))

        results = Matrix{Float64}[]
        for _ in 1:3  # repeat-run determinism
            A = allocate_system_matrix(wide_par)
            fill!(nonzeros(A), 0.0)
            assemble_add!(A, m_par)
            assemble_add!(A, k_par, 2.0)
            push!(results, Matrix(A))
        end
        @test all(R -> R ≈ Aref, results)
        @test all(R -> R == results[1], results)  # bit-for-bit repeat-run agreement
    end

    # S3.4 (gpena/Bramble.jl#338): a warmed Parallel() `assemble_add!` replays the recorded
    # positions instead of searching -- checked here against non-uniform 1D/2D/3D meshes (the
    # searching sweep and the replay must agree on a mesh whose stencils are not all the same
    # width), against the same accumulation run serially, at rtol 1e-12 (bramble-verification:
    # never checked against another call to the code under test alone -- the Serial forms are
    # an independently-executed reference, not a repeat of the Parallel path).
    @testset "Bilinear: Parallel() replay matches Serial, non-uniform 1D/2D/3D (#338)" begin
        cases = (
            (
                "1D",
                domain(interval(0.0, 1.0)),
                9,
                false
            ),
            (
                "2D",
                domain(
                    interval(0.0, 1.0) × interval(0.0, 1.0),
                    :walls => boundary_symbols(interval(0.0, 1.0) × interval(0.0, 1.0))
                ),
                (9, 8),
                (false, false)
            ),
            (
                "3D",
                domain(
                    box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                    :walls => boundary_symbols(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
                ),
                (7, 6, 5),
                (false, false, false)
            )
        )
        for (lbl, Ω, npts, unif) in cases
            @testset "$lbl" begin
                # Non-uniform point placement is drawn from the default RNG (mesh1d.jl):
                # reseed identically before each of the two otherwise-independent `mesh`
                # calls so the Serial and Parallel meshes share the same points, and only
                # the execution policy differs.
                Random.seed!(3384)
                Ωₕ = mesh(Ω, npts, unif)
                Random.seed!(3384)
                Ω_par = mesh(Ω, npts, unif; backend = backend(policy = Parallel()))
                Wₕ = gridspace(Ωₕ)
                W_par = gridspace(Ω_par)
                @test execution_policy(W_par) isa Parallel

                m_ser = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
                k_ser = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
                wide_ser = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
                m_par = form(W_par, W_par, (u, v) -> innerₕ(u, v))
                k_par = form(W_par, W_par, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
                wide_par = form(
                    W_par, W_par, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
                )

                @testset "θ = Float64" begin
                    θ = 2.5
                    A_ser = allocate_system_matrix(wide_ser)
                    A_par = allocate_system_matrix(wide_par)
                    for _ in 1:3  # repeated refills: cold record, then warm replays
                        fill!(nonzeros(A_ser), 0.0)
                        assemble_add!(A_ser, m_ser)
                        assemble_add!(A_ser, k_ser, θ)
                        fill!(nonzeros(A_par), 0.0)
                        assemble_add!(A_par, m_par)
                        assemble_add!(A_par, k_par, θ)
                        @test Matrix(A_par) ≈ Matrix(A_ser) rtol = 1e-12
                    end
                end

                @testset "θ = Ref" begin
                    θ_ser = Ref(1.0)
                    θ_par = Ref(1.0)
                    A_ser = allocate_system_matrix(wide_ser)
                    A_par = allocate_system_matrix(wide_par)
                    for scale in (1.0, 0.3, 4.0)  # repeated refills, changing θ between them
                        θ_ser[] = scale
                        θ_par[] = scale
                        fill!(nonzeros(A_ser), 0.0)
                        assemble_add!(A_ser, m_ser)
                        assemble_add!(A_ser, k_ser, θ_ser)
                        fill!(nonzeros(A_par), 0.0)
                        assemble_add!(A_par, m_par)
                        assemble_add!(A_par, k_par, θ_par)
                        @test Matrix(A_par) ≈ Matrix(A_ser) rtol = 1e-12
                    end
                end

                @testset "assemble_add! after assemble! of another form on the same A" begin
                    A_ser = allocate_system_matrix(wide_ser)
                    A_par = allocate_system_matrix(wide_par)
                    assemble!(A_ser, wide_ser)
                    assemble!(A_par, wide_par)
                    assemble_add!(A_ser, m_ser, 0.5)
                    assemble_add!(A_par, m_par, 0.5)

                    Aref = Matrix(assemble(wide_ser)) .+ 0.5 .* Matrix(assemble(m_ser))
                    @test Matrix(A_par) ≈ Aref rtol = 1e-12
                    @test Matrix(A_par) ≈ Matrix(A_ser) rtol = 1e-12
                end
            end
        end
    end

    @testset "Linear: unscaled and scaled accumulation, 1D/2D" begin
        for (lbl, Ωₕ) in (
            ("1D", mesh(domain(interval(0.0, 1.0)), 25, true)),
            ("2D", mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 8), (true, true)))
        )
            @testset "$lbl" begin
                Wₕ = gridspace(Ωₕ)
                fₕ = Rₕ(Wₕ, x -> 1.0)
                gₕ = Rₕ(Wₕ, x -> 2.0)
                l1 = form(Wₕ, v -> innerₕ(fₕ, v))
                l2 = form(Wₕ, v -> innerₕ(gₕ, v))

                F = zeros(ndofs(Wₕ))
                assemble_add!(F, l1)
                assemble_add!(F, l2, 3.0)

                Fref = assemble(l1) .+ 3.0 .* assemble(l2)
                @test F ≈ Fref
            end
        end
    end

    @testset "Linear: zero allocation on warm replay" begin
        function _warm_and_measure()
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 31, true)
            Wₕ = gridspace(Ωₕ)
            fₕ = Rₕ(Wₕ, x -> 1.0)
            l = form(Wₕ, v -> innerₕ(fₕ, v))
            F = zeros(ndofs(Wₕ))
            assemble_add!(F, l)  # warm-up
            θ = Ref(1.5)
            return _alloc(assemble_add!, F, l), _alloc(assemble_add!, F, l, θ)
        end
        allocs_unscaled, allocs_scaled = _warm_and_measure()
        @test allocs_unscaled == 0
        @test allocs_scaled == 0
    end

    @testset "Linear: composite space, one contribution per leaf" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, true)
        Wₕ = gridspace(Ωₕ)
        W = Wₕ × Wₕ
        fₕ = Rₕ(Wₕ, x -> 1.0)
        gₕ = Rₕ(Wₕ, x -> 2.0)

        l1 = form(W, v -> innerₕ(fₕ, v(1)))
        l2 = form(W, v -> innerₕ(gₕ, v(2)))

        F = zeros(ndofs(W))
        assemble_add!(F, l1)
        assemble_add!(F, l2, 0.5)

        Fref = assemble(l1) .+ 0.5 .* assemble(l2)
        @test F ≈ Fref
    end

    @testset "Linear: correct under Parallel() execution" begin
        Ω_par = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0)),
            (9, 8),
            (true, true);
            backend = backend(policy = Parallel())
        )
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 8), (true, true))
        Wₕ = gridspace(Ωₕ)
        W_par = gridspace(Ω_par)

        fₕ = Rₕ(Wₕ, x -> 1.0)
        fₕ_par = Rₕ(W_par, x -> 1.0)
        l = form(Wₕ, v -> innerₕ(fₕ, v))
        l_par = form(W_par, v -> innerₕ(fₕ_par, v))

        Fref = 2.0 .* assemble(l)

        F = zeros(ndofs(W_par))
        assemble_add!(F, l_par, 2.0)
        @test F ≈ Fref
    end

    # gpena/Bramble.jl#283: the explicit-scale, cached `assemble_add!(A, a, α)` /
    # `assemble_add!(F, l, α)` path is inferred and allocates nothing on warm replay, 1D and
    # 2D. `@inferred` reads no global binding here, so it stays in the barrier alongside the
    # warm-up rather than being pulled out to top level.
    @testset "Bilinear and linear: cached, explicit-scale path is inferred and allocation-free (#283)" begin
        function _bilinear_scaled_alloc(A, a, α)
            assemble_add!(A, a, α)             # cold: records
            @inferred assemble_add!(A, a, α)
            return _alloc(assemble_add!, A, a, α)
        end

        function _linear_scaled_alloc(F, l, α)
            assemble_add!(F, l, α)             # cold: records
            @inferred assemble_add!(F, l, α)
            return _alloc(assemble_add!, F, l, α)
        end

        for (lbl, Ωₕ) in (
            ("1D", mesh(domain(interval(0.0, 1.0)), 21, true)),
            ("2D", mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 11), (true, true)))
        )
            @testset "$lbl" begin
                Wₕ = gridspace(Ωₕ)
                m_form = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
                A = allocate_system_matrix(m_form)
                fill!(nonzeros(A), 0.0)
                @test _bilinear_scaled_alloc(A, m_form, 0.5) == 0

                fₕ = Rₕ(Wₕ, x -> 1.0)
                l_form = form(Wₕ, v -> innerₕ(fₕ, v))
                F = zeros(ndofs(Wₕ))
                @test _linear_scaled_alloc(F, l_form, 0.5) == 0
            end
        end
    end

    @testset "Dirichlet interaction is left to the caller, not applied here" begin
        # assemble_add! never zeros or constrains rows -- accumulating twice doubles
        # every entry, exactly what a raw additive scatter should do; dirichlet_bc! is a
        # separate, explicit step the caller runs once, last (see src/assembly/assemble_add.jl).
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, true)
        Wₕ = gridspace(Ωₕ)
        m_form = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        A = allocate_system_matrix(m_form)
        fill!(nonzeros(A), 0.0)
        assemble_add!(A, m_form)
        assemble_add!(A, m_form)
        @test Matrix(A) ≈ 2 .* Matrix(assemble(m_form))
    end
end

end # module
