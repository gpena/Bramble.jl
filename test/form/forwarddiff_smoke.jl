module FormForwarddiffSmokeTests

# The one ForwardDiff check that runs while AD tests are switched off (TestUtils.WITH_AD_TESTS):
# bilinear and linear assembly, a residual Jacobian and a semidiscrete residual, each with
# dual numbers flowing through it, on non-uniform meshes. Every oracle is exact: the forms are
# linear in the differentiated parameter, so the derivative is itself an assembled form.

using Test
using Bramble
using Bramble: semidiscretize
using ForwardDiff
using LinearAlgebra: Diagonal

_smoke_meshes() = (mesh(domain(interval(0.0, 1.0)), 11, false),
    mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9), (false, true)))

@testset "ForwardDiff smoke" begin
    for Ωₕ in _smoke_meshes()
        Wₕ = gridspace(Ωₕ)
        D = dim(Ωₕ)
        n = ndofs(Wₕ)
        M = Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))))
        K = Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))))

        @testset "bilinear, $(D)D" begin
            A(s) = Matrix(assemble(form(Wₕ, Wₕ,
                (u, v) -> innerₕ(u, v) + Ref(s) * inner₊(∇ₕ(u), ∇ₕ(v)))))
            @test eltype(A(ForwardDiff.Dual(2.0, 1.0))) <: ForwardDiff.Dual
            @test ForwardDiff.derivative(A, 2.0) ≈ K rtol = 1e-12
        end

        @testset "linear, $(D)D" begin
            f = x -> 1.0 + x[1]
            b(s) = assemble(form(Wₕ, v -> innerₕ(1.0, v) + Ref(s) * innerₕ(f, v)))
            @test ForwardDiff.derivative(b, 2.0) ≈ assemble(form(Wₕ, v -> innerₕ(f, v))) rtol = 1e-12
        end

        @testset "residual Jacobian, $(D)D" begin
            resid(w) = assemble(form(Wₕ, v -> innerₕ(element(Wₕ, w .^ 2), v)))
            w0 = collect(range(0.3, 1.7; length = n))
            @test ForwardDiff.jacobian(resid, w0) ≈ M * Diagonal(2 .* w0) rtol = 1e-12
        end

        @testset "semidiscrete residual, $(D)D" begin
            sd = semidiscretize(form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))),
                form(Wₕ, v -> innerₕ(1.0, v)))
            r(u) = (du = similar(u); sd(du, u, nothing, 0.0); du)
            # The residual is affine in u, so its Jacobian has columns r(eᵢ) - r(0).
            r0 = r(zeros(n))
            J = reduce(hcat, [r(Float64.((1:n) .== i)) .- r0 for i in 1:n])
            @test ForwardDiff.jacobian(r, zeros(n)) ≈ J rtol = 1e-10
        end
    end
end

end # module FormForwarddiffSmokeTests
