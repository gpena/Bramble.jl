using Test
using Bramble
using SparseArrays
using Bramble: values

# The element type of the backend survives the whole library.
#
# The backend decides the numeric type — the domain's own eltype does not propagate, so a
# Float32 run is asked for by constructing the backend with Float32 array types. From
# there every mesh, grid space, grid function, operator, matrix and inner product must
# stay in that type.
#
# A single Float64 literal anywhere in a formula silently promotes whatever it touches,
# and the promotion is invisible in a Float64 run, which is every other test in this
# suite. `add_half_shift` multiplied by `0.5` and returned Float64 averaging matrices
# while the rest of the library stayed Float32; nothing failed, the numbers were simply
# in the wrong type. These tests are what makes that visible.

const F32_BACKEND = backend(vector_type = Vector{Float32},
    matrix_type = SparseMatrixCSC{Float32, Int})

@testset "Element type preservation" begin
    @testset "Meshes & spaces" begin
        @test eltype(F32_BACKEND) === Float32

        Ωₕ1 = mesh(domain(interval(0.0f0, 1.0f0)), 11, true; backend = F32_BACKEND)
        Ωₕ2 = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (6, 7),
            (true, false); backend = F32_BACKEND)

        for Ωₕ in (Ωₕ1, Ωₕ2)
            @test eltype(Ωₕ) === Float32
            @test eltype(hₘₐₓ(Ωₕ)) === Float32
            @test eltype(hₘᵢₙ(Ωₕ)) === Float32
        end
        @test eltype(points(Ωₕ1)) === Float32
        @test eltype(spacings(Ωₕ1)) === Float32
        @test eltype(half_spacings(Ωₕ1)) === Float32
        @test eltype(cell_measures(Ωₕ1)) === Float32

        @test eltype(gridspace(Ωₕ1)) === Float32
        @test eltype(gridspace(Ωₕ2, Val(2))) === Float32
    end

    @testset "Functions & operators" begin
        Ωₕ = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (6, 7),
            (true, false); backend = F32_BACKEND)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])

        @test eltype(values(uₕ)) === Float32
        @test eltype(values(avgₕ(Wₕ, x -> sin(x[1]) * x[2]))) === Float32

        for op in (diff₋ₓ, diff₊ₓ, D₋ₓ, D₊ₓ, jumpₓ, M₋ₓ, M₊ₓ,
            Dstar₊ₓ, Dcₓ, Dₕₓ, D₋ᵧ, M₊ᵧ, Dcᵧ, Dₕᵧ)
            @test eltype(values(op(uₕ))) === Float32
        end
        for op in (∇₋ₕ, ∇₊ₕ, Dstar₊ₕ, Dcₕ, ∇ₕ, M₋ₕ, jumpₕ)
            @test all(g -> eltype(values(g)) === Float32, op(uₕ))
        end
    end

    @testset "Matrix forms" begin
        # The averaging matrices are where the Float64 literal was: every other family
        # was already exact, so a bound on all of them would not have caught it.
        Ωₕ = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (6, 7),
            (true, false); backend = F32_BACKEND)

        for op in (D₋ₓ, D₊ₓ, diff₋ₓ, diff₊ₓ, jumpₓ, M₋ₓ, M₊ₓ, M₋ᵧ, M₊ᵧ)
            @test eltype(op(Ωₕ)) === Float32
        end
    end

    @testset "Inner products & norms" begin
        Ωₕ = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (6, 7),
            (true, false); backend = F32_BACKEND)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])
        gₕ = ∇₋ₕ(uₕ)

        @test innerₕ(uₕ, uₕ) isa Float32
        @test normₕ(uₕ) isa Float32
        @test snorm₁ₕ(uₕ) isa Float32
        @test norm₁ₕ(uₕ) isa Float32
        @test inner₊(uₕ, uₕ) isa Float32
        @test inner₊(gₕ, gₕ) isa Float32
        @test norm₊(gₕ) isa Float32
        @test inner₊ₓ(uₕ, uₕ) isa Float32
        @test inner₊ᵧ(uₕ, uₕ) isa Float32
    end
end
