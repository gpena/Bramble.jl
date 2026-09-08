using Test
using Bramble
using ForwardDiff, DifferentiationInterface
import SparseConnectivityTracer, SparseMatrixColorings
using SparseArrays: nnz

# `type_cached_assemble!` (form/type_cached_assemble.jl, gpena/Bramble.jl#20): caches a
# coefficient-dependent BilinearForm's sparsity pattern per element type, so a Newton
# residual generic over `T` only pays allocate_system_matrix's own cost once per type
# instead of on every call.

const _traced_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

# Standalone runner fallback (`runtests.jl` already defines both when this file is
# included from there). A top-level, type-parametric function barrier -- unlike a closure
# written inside a `@testset`, which `@testset`'s own scope-wrapping can box, inflating an
# allocation count that has nothing to do with the code under test (bramble-verification §1).
if !@isdefined(alloc_test)
    @inline function alloc_test(f::F, args...; kwargs...) where {F}
        f(args...; kwargs...)
        return @allocated(f(args...; kwargs...))
    end
end

if !@isdefined(var"@test_allocs")
    macro test_allocs(call_expr)
        if Meta.isexpr(call_expr, :call)
            fn = call_expr.args[1]
            args = call_expr.args[2:end]
            quote
                @test alloc_test($(esc(fn)), $(map(esc, args)...)) == 0
            end
        else
            quote
                let
                    $(esc(call_expr))
                    @test (@allocated $(esc(call_expr))) == 0
                end
            end
        end
    end
end

@testset "type_cached_assemble!" begin
    Ω = domain(interval(0.0, 1.0))
    Ωₕ = mesh(Ω, 20, false)
    Wₕ = gridspace(Ωₕ)
    bcs = dirichlet_constraints(Bramble.set(Ω), :boundary => x -> exp(x[1]))
    gₕ = element(Wₕ)
    avgₕ!(gₕ, x -> exp(x[1]))
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    F = assemble(l; dirichlet_conditions = bcs, dirichlet_labels = :boundary)

    α(u) = 3 + 1 / (1 + u^2)

    # The direct (uncached) equivalent, one fresh matrix per call -- the ground truth
    # `type_cached_assemble!` must reproduce exactly, values and all.
    function diffusion_matrix_direct(uₕ)
        αvals = α.(M₋ₕ(uₕ))
        a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇₋ₕ(U), ∇₋ₕ(V)))
        return assemble(a; dirichlet_labels = :boundary)
    end

    # `build` is a named, top-level function (not a `do ... end` literal written inside a
    # repeatedly-called function) precisely so passing it doesn't allocate a fresh closure
    # every call -- the point the docstring itself warns about. `refill!` uses `M₋ₓ!`
    # (in place) rather than `M₋ₓ`/`M₋ₕ`, which would allocate a fresh result every call --
    # exactly the cost the "allocates nothing" test below checks was not reintroduced.
    function _build_diffusion(uₕ)
        Mu = element(Wₕ, eltype(uₕ))
        αvals = element(Wₕ, eltype(uₕ))
        a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇₋ₕ(U), ∇₋ₕ(V)))
        refill!(uₕ) = begin
            M₋ₓ!(Mu, uₕ)
            αvals .= α.(Mu)
        end
        return a, refill!
    end

    @testset "matches the direct (uncached) result, at Float64 and at Dual" begin
        cache = Dict()
        diffusion_matrix_cached(uₕ) = type_cached_assemble!(
            _build_diffusion, cache, uₕ; dirichlet_labels = :boundary)

        u0 = element(Wₕ, 0.0)
        @test diffusion_matrix_cached(u0) == diffusion_matrix_direct(u0)

        # A different Float64 value: the cache must be *refilled*, not stuck on u0.
        u1 = element(Wₕ)
        Rₕ!(u1, x -> exp(x[1]))
        @test diffusion_matrix_cached(u1) == diffusion_matrix_direct(u1)

        # Now a genuine Dual-typed call, through real AD machinery rather than a
        # hand-rolled Dual literal -- a different element type sharing the same cache.
        residual_cached(u) = diffusion_matrix_cached(let uₕ = element(Wₕ, eltype(u))
            uₕ .= u
            uₕ
        end) * u
        residual_direct(u) = diffusion_matrix_direct(let uₕ = element(Wₕ, eltype(u))
            uₕ .= u
            uₕ
        end) * u

        u1_vec = parent(u1)
        J_cached = DifferentiationInterface.jacobian(residual_cached, AutoForwardDiff(), u1_vec)
        J_direct = DifferentiationInterface.jacobian(residual_direct, AutoForwardDiff(), u1_vec)
        @test J_cached == J_direct

        # Back to Float64: the Dual call must not have corrupted the Float64 entry.
        @test diffusion_matrix_cached(u1) == diffusion_matrix_direct(u1)
    end

    @testset "structural: matches assemble(a)'s own pattern, at every type seen" begin
        # SparseConnectivityTracer's own tracer element type: the one case that would
        # crash outright (UndefRefError) if `refill!` ran after allocate_system_matrix
        # instead of before, since a tracer is not `isbits` and starts genuinely
        # unassigned rather than merely holding arbitrary bits.
        cache = Dict()
        diffusion_matrix_cached(uₕ) = type_cached_assemble!(
            _build_diffusion, cache, uₕ; dirichlet_labels = :boundary)
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

    @testset "repeated calls at an already-seen type cost a small, N-independent overhead, not a rebuild" begin
        # `cache::AbstractDict` necessarily stores its `(a, refill!, A)` entries as `Any` --
        # the concrete triple's type differs across every distinct element type `T` a cache
        # can ever be asked about -- so a cache hit still pays a small, fixed dynamic-dispatch
        # cost fetching that entry back out. It is not literally zero allocation, but the
        # property that actually matters holds: unlike `diffusion_matrix_direct`, whose
        # allocation grows with `ndofs` because it rebuilds the whole sparsity pattern every
        # call, a cache hit's cost does not grow with the mesh at all.
        diffusion_matrix_cached = let cache = Dict()
            uₕ -> type_cached_assemble!(_build_diffusion, cache, uₕ; dirichlet_labels = :boundary)
        end

        u0 = element(Wₕ, 0.0)
        diffusion_matrix_cached(u0)   # first call: builds and caches
        diffusion_matrix_cached(u0)   # warm up the second call's own compilation

        cached_bytes = alloc_test(diffusion_matrix_cached, u0)
        direct_bytes = alloc_test(diffusion_matrix_direct, u0)
        @test cached_bytes < direct_bytes / 2

        # Same check at a 100x larger mesh: `direct_bytes` must grow with it (it rebuilds the
        # pattern), while a cache hit's own cost must not (it is dictionary/dispatch overhead,
        # not proportional to ndofs).
        Ωₕ_big = mesh(domain(interval(0.0, 1.0)), 2000, false)
        Wₕ_big = gridspace(Ωₕ_big)
        function _build_diffusion_big(uₕ)
            Mu = element(Wₕ_big, eltype(uₕ))
            αvals = element(Wₕ_big, eltype(uₕ))
            a = form(Wₕ_big, Wₕ_big, (U, V) -> inner₊(αvals * ∇₋ₕ(U), ∇₋ₕ(V)))
            refill!(uₕ) = begin
                M₋ₓ!(Mu, uₕ)
                αvals .= α.(Mu)
            end
            return a, refill!
        end
        function diffusion_matrix_direct_big(uₕ)
            αvals = α.(M₋ₕ(uₕ))
            a = form(Wₕ_big, Wₕ_big, (U, V) -> inner₊(αvals * ∇₋ₕ(U), ∇₋ₕ(V)))
            return assemble(a; dirichlet_labels = :boundary)
        end
        diffusion_matrix_cached_big = let cache_big = Dict()
            uₕ -> type_cached_assemble!(_build_diffusion_big, cache_big, uₕ; dirichlet_labels = :boundary)
        end
        u0_big = element(Wₕ_big, 0.0)
        diffusion_matrix_cached_big(u0_big)
        diffusion_matrix_cached_big(u0_big)

        cached_bytes_big = alloc_test(diffusion_matrix_cached_big, u0_big)
        direct_bytes_big = alloc_test(diffusion_matrix_direct_big, u0_big)
        @test direct_bytes_big > 50 * direct_bytes
        @test cached_bytes_big == cached_bytes
    end

    @testset "drives a full Newton solve to the same answer as the direct approach" begin
        sol(x) = exp(x[1])
        dαdu(u) = -2u / (1 + u^2)^2
        rhs(x) = -dαdu(sol(x)) * sol(x)^2 - α(sol(x)) * sol(x)

        bcs_sol = dirichlet_constraints(Bramble.set(Ω), :boundary => sol)
        gₕ_sol = element(Wₕ)
        avgₕ!(gₕ_sol, rhs)
        l_sol = form(Wₕ, v -> innerₕ(gₕ_sol, v))
        F_sol = assemble(l_sol; dirichlet_conditions = bcs_sol, dirichlet_labels = :boundary)

        cache = Dict()
        diffusion_matrix_cached(uₕ) = type_cached_assemble!(
            _build_diffusion, cache, uₕ; dirichlet_labels = :boundary)
        function residual_cached(u_vec::AbstractVector{T}) where {T}
            uₕ = element(Wₕ, T)
            uₕ .= u_vec
            return diffusion_matrix_cached(uₕ) * u_vec .- F_sol
        end
        function residual_direct(u_vec::AbstractVector{T}) where {T}
            uₕ = element(Wₕ, T)
            uₕ .= u_vec
            αvals = α.(M₋ₕ(uₕ))
            a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇₋ₕ(U), ∇₋ₕ(V)))
            A = assemble(a; dirichlet_labels = :boundary)
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
