module FormTypeCachedAssembleTests

using Test
using Bramble
using ForwardDiff, DifferentiationInterface
using SparseConnectivityTracer: SparseConnectivityTracer
using SparseMatrixColorings: SparseMatrixColorings
using SparseArrays: nnz
using Random
using Bramble: Mₓ, Mᵧ, Mₓ!, Mᵧ!, type_cached_assemble!
using ..TestUtils: @test_allocs

# `type_cached_assemble!` (assembly/type_cached_assemble.jl): caches a
# coefficient-dependent BilinearForm's sparsity pattern per element type, so a Newton
# residual generic over `T` only pays allocate_system_matrix's own cost once per type
# instead of on every call.

const _traced_ad = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
)

@testset "type_cached_assemble!" begin
    # Non-uniform (`false`): mesh1d.jl's `_generate_random_points!` draws from the global
    # RNG, so an unseeded run's point placement differs run to run -- and the Newton solve
    # below asserts a specific iteration bound (`length(newton_residuals) < 6`), which an
    # unlucky draw could plausibly miss. Seeded so the test is reproducible, not just
    # usually passing.
    Random.seed!(20260912)
    Ω = domain(interval(0.0, 1.0))
    Ωₕ = mesh(Ω, 20, false)
    Wₕ = gridspace(Ωₕ)
    bcs = dirichlet_constraints(Ω, :boundary => x -> exp(x[1]))
    gₕ = element(Wₕ)
    avgₕ!(gₕ, x -> exp(x[1]))
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    F = assemble(l; dirichlet = bcs)

    α(u) = 3 + 1 / (1 + u^2)

    # The direct (uncached) equivalent, one fresh matrix per call -- the ground truth
    # `type_cached_assemble!` must reproduce exactly, values and all.
    function diffusion_matrix_direct(uₕ)
        αvals = α.(Mₕ(uₕ))
        a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))
        return assemble(a; dirichlet = :boundary)
    end

    # `build` is a named, top-level function (not a `do ... end` literal written inside a
    # repeatedly-called function) precisely so passing it doesn't allocate a fresh closure
    # every call -- the point the docstring itself warns about. `refill!` uses `Mₓ!`
    # (in place) rather than `Mₓ`/`Mₕ`, which would allocate a fresh result every call --
    # exactly the cost the "allocates nothing" test below checks was not reintroduced.
    function _build_diffusion(uₕ)
        Mu = element(Wₕ, eltype(uₕ))
        αvals = element(Wₕ, eltype(uₕ))
        a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))
        refill!(uₕ) = begin
            Mₓ!(Mu, uₕ)
            αvals .= α.(Mu)
        end
        return a, refill!
    end

    # Matches the direct (uncached) result, at Float64 and at Dual.
    @testset "matches direct, Float64 and Dual" begin
        cache = Dict{DataType, Any}()
        diffusion_matrix_cached(uₕ) = type_cached_assemble!(_build_diffusion, cache, uₕ; dirichlet = :boundary)

        u0 = element(Wₕ, 0.0)
        @test diffusion_matrix_cached(u0) == diffusion_matrix_direct(u0)

        # A different Float64 value: the cache must be *refilled*, not stuck on u0.
        u1 = element(Wₕ)
        Rₕ!(u1, x -> exp(x[1]))
        @test diffusion_matrix_cached(u1) == diffusion_matrix_direct(u1)

        # Now a genuine Dual-typed call, through real AD machinery rather than a
        # hand-rolled Dual literal -- a different element type sharing the same cache.
        residual_cached(u) = diffusion_matrix_cached(
            let uₕ = element(Wₕ, eltype(u))
            uₕ .= u
            uₕ
        end,
        ) * u
        residual_direct(u) = diffusion_matrix_direct(
            let uₕ = element(Wₕ, eltype(u))
            uₕ .= u
            uₕ
        end,
        ) * u

        u1_vec = parent(u1)
        J_cached = DifferentiationInterface.jacobian(
            residual_cached, AutoForwardDiff(), u1_vec
        )
        J_direct = DifferentiationInterface.jacobian(
            residual_direct, AutoForwardDiff(), u1_vec
        )
        @test J_cached == J_direct

        # Back to Float64: the Dual call must not have corrupted the Float64 entry.
        @test diffusion_matrix_cached(u1) == diffusion_matrix_direct(u1)
    end

    # Matches assemble(a)'s own pattern, at every type seen.
    @testset "structural: pattern matches assemble(a)" begin
        # SparseConnectivityTracer's own tracer element type: the one case that would
        # crash outright (UndefRefError) if `refill!` ran after allocate_system_matrix
        # instead of before, since a tracer is not `isbits` and starts genuinely
        # unassigned rather than merely holding arbitrary bits.
        cache = Dict{DataType, Any}()
        diffusion_matrix_cached(uₕ) = type_cached_assemble!(_build_diffusion, cache, uₕ; dirichlet = :boundary)
        function residual_cached(u_vec::AbstractVector{T}) where {T}
            uₕ = element(Wₕ, T)
            uₕ .= u_vec
            return diffusion_matrix_cached(uₕ) * u_vec .- F
        end

        u_probe = rand(ndofs(Wₕ))
        prep = prepare_jacobian(residual_cached, _traced_ad, u_probe)
        J = DifferentiationInterface.jacobian(residual_cached, prep, _traced_ad, u_probe)
        @test nnz(J) > 0
    end

    # A repeated call at an already-seen type allocates nothing, in 1D and in 2D.
    @testset "seen type: warm hit allocates 0 B" begin
        # Entries live in the cache as `Any`, so a hit makes one dynamic call. Everything
        # that call receives is already on the heap and it returns `nothing`, so it boxes
        # nothing, and `refill!` and `assemble!` past it work in place. An untyped `Dict()`
        # here, the way callers write it.
        hit = let cache = Dict()
            uₕ -> type_cached_assemble!(_build_diffusion, cache, uₕ; dirichlet = :boundary)
        end
        u0 = element(Wₕ)
        Rₕ!(u0, x -> exp(x[1]))
        hit(u0)   # first call: builds and caches
        @test_allocs hit(u0)
        @test hit(u0) == diffusion_matrix_direct(u0)

        # A constraints struct for `dirichlet` travels in the entry's typed argument slot,
        # as a `Symbol` does, so it is never boxed either.
        hit_bcs = let cache = Dict()
            uₕ -> type_cached_assemble!(_build_diffusion, cache, uₕ; dirichlet = bcs)
        end
        hit_bcs(u0)
        @test_allocs hit_bcs(u0)

        # 2D, on a non-uniform mesh: one coefficient per direction, each averaged across the
        # other, so `refill!` needs both `Mₓ!` and `Mᵧ!`.
        Random.seed!(20260912)
        Ω2 = domain(interval(0.0, 1.0) × interval(0.0, 2.0))
        W2 = gridspace(mesh(Ω2, (8, 9), (false, false)))
        _flux2(ax, ay) = form(
            W2, W2, (U, V) -> inner₊((ax * ∇ₕ(U)[1], ay * ∇ₕ(U)[2]), ∇ₕ(V)))
        function _build_flux(uₕ)
            Mx, My, ax, ay = (element(W2, eltype(uₕ)) for _ in 1:4)
            refill!(uₕ) = begin
                Mₓ!(Mx, uₕ)
                Mᵧ!(My, uₕ)
                ax .= α.(My)
                ay .= α.(Mx)
            end
            return _flux2(ax, ay), refill!
        end
        hit2 = let cache = Dict()
            uₕ -> type_cached_assemble!(_build_flux, cache, uₕ; dirichlet = :boundary)
        end
        u2 = element(W2)
        Rₕ!(u2, x -> 1 + x[1] * x[2])
        hit2(u2)
        @test_allocs hit2(u2)
        @test hit2(u2) == assemble(_flux2(α.(Mᵧ(u2)), α.(Mₓ(u2))); dirichlet = :boundary)
    end

    # Drives a full Newton solve to the same answer as the direct approach.
    @testset "Newton solve matches direct" begin
        sol(x) = exp(x[1])
        dαdu(u) = -2u / (1 + u^2)^2
        rhs(x) = -dαdu(sol(x)) * sol(x)^2 - α(sol(x)) * sol(x)

        bcs_sol = dirichlet_constraints(Ω, :boundary => sol)
        gₕ_sol = element(Wₕ)
        avgₕ!(gₕ_sol, rhs)
        l_sol = form(Wₕ, v -> innerₕ(gₕ_sol, v))
        F_sol = assemble(l_sol; dirichlet = bcs_sol)

        cache = Dict{DataType, Any}()
        diffusion_matrix_cached(uₕ) = type_cached_assemble!(_build_diffusion, cache, uₕ; dirichlet = :boundary)
        function residual_cached(u_vec::AbstractVector{T}) where {T}
            uₕ = element(Wₕ, T)
            uₕ .= u_vec
            return diffusion_matrix_cached(uₕ) * u_vec .- F_sol
        end
        function residual_direct(u_vec::AbstractVector{T}) where {T}
            uₕ = element(Wₕ, T)
            uₕ .= u_vec
            αvals = α.(Mₕ(uₕ))
            a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))
            A = assemble(a; dirichlet = :boundary)
            return A * u_vec .- F_sol
        end

        function newton(residual)
            u = zeros(ndofs(Wₕ))
            prep = prepare_jacobian(residual, _traced_ad, u)
            J = DifferentiationInterface.jacobian(residual, prep, _traced_ad, u)
            newton_residuals = Float64[]
            for _ in 1:20
                r = residual(u)
                push!(newton_residuals, sqrt(sum(abs2, r)))
                newton_residuals[end] < 1e-10 && break
                DifferentiationInterface.jacobian!(residual, J, prep, _traced_ad, u)
                u .-= J \ r
            end
            return u, newton_residuals
        end

        u_cached, newton_residuals = newton(residual_cached)
        u_direct, _ = newton(residual_direct)

        @test length(newton_residuals) < 6
        @test newton_residuals[end] < 1e-10
        @test maximum(abs.(u_cached .- u_direct)) < 1e-10

        uₕ = element(Wₕ)
        uₕ .= u_cached
        @test norm₁ₕ(uₕ .- Rₕ(Wₕ, sol)) < 1e-2
    end
end

end # module FormTypeCachedAssembleTests
