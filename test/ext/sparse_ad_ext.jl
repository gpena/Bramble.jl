using Test
using Bramble
using ADTypes
using ForwardDiff, DifferentiationInterface
import SparseMatrixColorings

# BrambleSparseADExt's `ast_sparsity_detector` (gpena/Bramble.jl#13): the ergonomic wrapper
# around `jacobian_pattern` (gpena/Bramble.jl#21) that plugs straight into
# `AutoSparse(...; sparsity_detector = ...)`, in place of `ADTypes.KnownJacobianSparsityDetector`
# or a tracer. Loaded here as an "ext" group test since it needs `ADTypes` -- a weak
# dependency, not part of the always-run unit-test environment's precompilation cost.
#
# `Bramble.domain`/`Bramble.interval`/`Bramble.mesh`/`Bramble.element` are qualified
# throughout, the same reason meshes_ext.jl is: this file runs after
# makie_ext.jl/meshes_ext.jl in the same "Package extensions" testset, and
# `using Bramble, Makie, Meshes` together (all `include`d into the same `Main`) makes the
# generic-sounding bare names ambiguous (Meshes.jl has its own `domain`/`mesh`/`element`).

@testset "BrambleSparseADExt" begin
    sol(x) = exp(x[1])
    α(u) = 3 + 1 / (1 + u^2)
    dαdu(u) = -2u / (1 + u^2)^2
    rhs(x) = -dαdu(sol(x)) * sol(x)^2 - α(sol(x)) * sol(x)

    Ω = Bramble.domain(Bramble.interval(0.0, 1.0))
    Ωₕ = Bramble.mesh(Ω, 20, false)
    Wₕ = gridspace(Ωₕ)

    bcs = dirichlet_constraints(Bramble.set(Ω), :boundary => sol)
    gₕ = Bramble.element(Wₕ)
    avgₕ!(gₕ, rhs)
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    F = assemble(l; dirichlet_conditions = bcs, dirichlet_labels = :boundary)

    function diffusion_form(uₕ)
        αv = α.(M₋ₕ(uₕ))
        return form(Wₕ, Wₕ, (U, V) -> inner₊(αv * ∇₋ₕ(U), ∇₋ₕ(V)))
    end

    function residual(u_vec::AbstractVector{T}) where {T}
        uₕ = Bramble.element(Wₕ, T)
        uₕ .= u_vec
        A = assemble(diffusion_form(uₕ); dirichlet_labels = :boundary)
        return A * u_vec .- F
    end

    a = diffusion_form(Bramble.element(Wₕ, 0.0))
    detector = ast_sparsity_detector(a, U -> M₋ₕ(U))

    @testset "isa AbstractSparsityDetector" begin
        @test detector isa ADTypes.AbstractSparsityDetector
    end

    @testset "matches jacobian_pattern directly" begin
        u_probe = zeros(ndofs(Wₕ))
        @test ADTypes.jacobian_sparsity(residual, u_probe, detector) ==
              jacobian_pattern(a, U -> M₋ₕ(U))
    end

    @testset "drives Newton to the right answer through AutoSparse" begin
        sparse_ad = AutoSparse(AutoForwardDiff();
            sparsity_detector = detector,
            coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

        u = zeros(ndofs(Wₕ))
        prep = prepare_jacobian(residual, sparse_ad, u)
        J = DifferentiationInterface.jacobian(residual, prep, sparse_ad, u)
        newton_residuals = Float64[]
        for _ in 1:20
            r = residual(u)
            push!(newton_residuals, sqrt(sum(abs2, r)))
            newton_residuals[end] < 1e-10 && break
            DifferentiationInterface.jacobian!(residual, J, prep, sparse_ad, u)
            u .-= J \ r
        end

        @test length(newton_residuals) < 6
        @test newton_residuals[end] < 1e-10

        uₕ = Bramble.element(Wₕ)
        uₕ .= u
        @test norm₁ₕ(uₕ .- Rₕ(Wₕ, sol)) < 1e-2
    end
end
