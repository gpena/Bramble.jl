module ExtADBackendVerificationTests

using Test
using Bramble
using ADTypes
using ForwardDiff, ReverseDiff, FiniteDiff, DifferentiationInterface
using Random
using LinearAlgebra: mul!
using ..ExtSparseAdExtTests: nonlinear_diffusion_problem

# Verifies two things gpena/Bramble.jl#122 asks for: that `jacobian_pattern` never misses a
# structural nonzero the real Jacobian has (a false zero would silently corrupt a sparse
# Newton solve), and that independent AD backends agree on that same Jacobian, scalar and
# composite space alike.
#
# The issue's own proposal was to write this against DifferentiationInterfaceTest.jl, the
# harness `DifferentiationInterface`'s own authors ship for exactly this. It cannot be added
# here: its registered release (0.11.0) lists JET only as a `[weakdeps]` extra (for its own
# optional type-stability checks) with `compat` `"0.9,0.10,0.11"`, and Julia's resolver
# enforces a weak dependency's compat bounds on every package sharing an environment with it,
# whether or not the extension that needs it ever loads -- confirmed directly:
# `Pkg.resolve()` on a copy of `test/Project.toml` with `DifferentiationInterfaceTest` added
# reports JET unsatisfiable against Bramble's own `JET = "0.12"` (Project.toml). Downgrading
# that pin to unblock a test harness was not on the table -- it gates the package's own
# static-analysis quality check (test/quality/jet.jl). So this file drives
# `DifferentiationInterface.jacobian` directly across backends and checks agreement by hand,
# which needs nothing beyond what `test/ext/sparse_ad_ext.jl` and
# `test/space/autodiff_backends.jl` already depend on plus `FiniteDiff`.
#
# ForwardDiff is the reference in every comparison below, not because it is assumed correct,
# but because it is the backend already established (bramble-performance §8) to work
# unconditionally through Bramble's mutating operators; agreement with two independent
# backends (ReverseDiff, a different differentiation direction; FiniteDiff, no
# differentiation machinery at all) is what actually corroborates it.

# The scalar nonlinear problem is sparse_ad_ext.jl's `nonlinear_diffusion_problem`, which
# the ext group includes first -- at n = 8 here, small enough for the dense
# `ForwardDiff.jacobian` reference below to cost nothing.
function _scalar_residual_problem(n = 8)
    p = nonlinear_diffusion_problem(n)
    return p.residual, p.a, ndofs(p.Wₕ)
end

u_ex2d(x) = sin(π * x[1]) * sin(π * x[2])
v_ex2d(x) = sin(2π * x[1]) * sin(2π * x[2])
f1_2d(x) = 2π^2 * u_ex2d(x) + u_ex2d(x) + u_ex2d(x) * v_ex2d(x)
f2_2d(x) = 8π^2 * v_ex2d(x) + v_ex2d(x) - u_ex2d(x) * v_ex2d(x)

# The coupled reaction-diffusion residual from docs/src/examples/coupled_reaction_diffusion.jl,
# at a size small enough for a dense reference Jacobian (5x5 per species, 50 dofs total).
function _composite_residual_problem(n = 5)
    Ω = Bramble.domain(Bramble.interval(0.0, 1.0) × Bramble.interval(0.0, 1.0))
    Ωₕ = Bramble.mesh(Ω, (n, n), (false, false))
    Wₕ = gridspace(Ωₕ)
    Vₕ = Wₕ^Val(2)

    bcs = dirichlet_constraints(Ω, :boundary => x -> 0.0)
    f1ₕ = Bramble.element(Wₕ)
    avgₕ!(f1ₕ, f1_2d)
    f2ₕ = Bramble.element(Wₕ)
    avgₕ!(f2ₕ, f2_2d)
    l = form(Vₕ, q -> innerₕ(f1ₕ, q(1)) + innerₕ(f2ₕ, q(2)))
    F = assemble(l; dirichlet = bcs)

    coupled_matrix(wₕ) = begin
        u_c, v_c = components(wₕ)
        a = form(
            Vₕ,
            Vₕ,
            (p, q) -> inner₊(∇ₕ(p(1)), ∇ₕ(q(1))) + innerₕ(p(1), q(1)) +
                      innerₕ(v_c * p(1), q(1)) +
                      inner₊(∇ₕ(p(2)), ∇ₕ(q(2))) + innerₕ(p(2), q(2)) -
                      innerₕ(u_c * p(2), q(2))
        )
        assemble(a; dirichlet = :boundary)
    end

    residual(w::AbstractVector{T}) where {T} = begin
        wₕ = Bramble.element(Vₕ, T)
        wₕ .= w
        A = coupled_matrix(wₕ)
        A * w .- F
    end

    u_c0, v_c0 = components(Bramble.element(Vₕ, 0.0))
    a_for_pattern = form(
        Vₕ,
        Vₕ,
        (p, q) -> inner₊(∇ₕ(p(1)), ∇ₕ(q(1))) + innerₕ(p(1), q(1)) +
                  innerₕ(v_c0 * p(1), q(1)) +
                  inner₊(∇ₕ(p(2)), ∇ₕ(q(2))) + innerₕ(p(2), q(2)) -
                  innerₕ(u_c0 * p(2), q(2))
    )
    return residual, a_for_pattern, ndofs(Vₕ)
end

@testset "BrambleSparseADExt/DifferentiationInterface backend verification (gpena/Bramble.jl#122)" begin
    @testset "jacobian_pattern is a safe superset of the real Jacobian" begin
        Random.seed!(20260914)

        @testset "scalar" begin
            residual, a_for_pattern, n = _scalar_residual_problem()
            pattern = jacobian_pattern(a_for_pattern, U -> Mₕ(U))
            for _ in 1:3
                J = ForwardDiff.jacobian(residual, rand(n))
                @test all(iszero(J[i, j]) || pattern[i, j] for i in axes(J, 1), j in axes(J, 2))
            end
        end

        @testset "composite (coupled species)" begin
            residual, a_for_pattern, n = _composite_residual_problem()
            pattern = jacobian_pattern(a_for_pattern, U -> U(2), U -> U(1))
            for _ in 1:3
                J = ForwardDiff.jacobian(residual, rand(n))
                @test all(iszero(J[i, j]) || pattern[i, j] for i in axes(J, 1), j in axes(J, 2))
            end
        end
    end

    @testset "ForwardDiff, ReverseDiff and FiniteDiff agree on the Jacobian" begin
        Random.seed!(20260914)

        @testset "scalar" begin
            residual, _, n = _scalar_residual_problem()
            u0 = rand(n)
            Jref = ForwardDiff.jacobian(residual, u0)
            @test DifferentiationInterface.jacobian(residual, AutoReverseDiff(), u0) ≈
                  Jref rtol=1e-8
            @test DifferentiationInterface.jacobian(residual, AutoFiniteDiff(), u0) ≈
                  Jref rtol=1e-4
        end

        @testset "composite (coupled species)" begin
            residual, _, n = _composite_residual_problem()
            w0 = rand(n)
            Jref = ForwardDiff.jacobian(residual, w0)
            @test DifferentiationInterface.jacobian(residual, AutoReverseDiff(), w0) ≈
                  Jref rtol=1e-8
            @test DifferentiationInterface.jacobian(residual, AutoFiniteDiff(), w0) ≈
                  Jref rtol=1e-4
        end
    end
end

@testset "ReverseDiff/KroneckerLinearOperator mul! disambiguation (gpena/Bramble.jl#295)" begin
    # A small non-uniform 2D separable form -- `is_separable`'s own recognised shape
    # (`innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))`, see src/form/kronecker.jl). Exercises exactly
    # what `ext/BrambleReverseDiffExt.jl`'s disambiguating `mul!` needs to get right: the
    # extension only loads (and only needs to resolve the ambiguity) once `ReverseDiff` --
    # already `using`'d above -- is loaded alongside Bramble.
    @assert Base.get_extension(Bramble, :BrambleReverseDiffExt) !== nothing "BrambleReverseDiffExt did not load"

    Random.seed!(20260923)
    Ω = domain(Bramble.interval(0.0, 1.0) × Bramble.interval(0.0, 1.0))
    Ωₕ = mesh(Ω, (9, 7), (false, false))
    Wₕ = gridspace(Ωₕ)
    a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
    K = kronecker_operator(a)
    A = assemble(a)
    n = size(K, 1)
    x0 = rand(n)

    @testset "tracked mul!" begin
        ty = ReverseDiff.track(zeros(n))
        tx = ReverseDiff.track(x0)
        mul!(ty, K, tx)
        @test ReverseDiff.value(ty) ≈ A * x0
    end

    @testset "gradient through K * x" begin
        g = ReverseDiff.gradient(x -> sum(abs2, K * x), x0)
        @test g ≈ 2 * (A' * (A * x0))
    end
end

end # module ExtADBackendVerificationTests
