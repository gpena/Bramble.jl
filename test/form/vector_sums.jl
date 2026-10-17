module FormVectorSumsTests

using Test
using Bramble
using Bramble: D₋ₓ, D₋ᵧ
using ..TestUtils: WITH_3D_TESTS

# Sums and differences of operator tuples (gpena/Bramble.jl#767).
#
# `0.1 * ∇ₕ(v) + 3 * ∇ₕ(D₋ₓ(v))` adds two tuples of lazy operators entry by entry. The
# reference is the same form with the scalars pulled out of `inner₊` by hand, which never
# adds two tuples, on non-uniform meshes.

@testset "Vector sums" begin
    W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9),
        (false, false)))
    asm(f) = Matrix(assemble(form(W, W, f)))

    @testset "sum" begin
        A = asm((u, v) -> inner₊(∇ₕ(u), ∇ₕ(v) + ∇ₕ(D₋ᵧ(v))))
        B = asm((u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + inner₊(∇ₕ(u), ∇ₕ(D₋ᵧ(v))))
        @test A ≈ B
        @test A ≉ asm((u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    end

    @testset "difference" begin
        A = asm((u, v) -> inner₊(∇ₕ(u), ∇ₕ(v) - 0.5 * ∇ₕ(D₋ₓ(v))))
        B = asm((u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) - 0.5 * inner₊(∇ₕ(u), ∇ₕ(D₋ₓ(v))))
        @test A ≈ B
        # a `-` read as `+` would give this one instead
        @test A ≉ asm((u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + 0.5 * inner₊(∇ₕ(u), ∇ₕ(D₋ₓ(v))))
    end

    @testset "scaled sum" begin
        A = asm((u, v) -> inner₊(∇ₕ(u), 0.1 * ∇ₕ(v) + 3 * ∇ₕ(D₋ₓ(v))))
        B = asm((u, v) -> 0.1 * inner₊(∇ₕ(u), ∇ₕ(v)) + 3 * inner₊(∇ₕ(u), ∇ₕ(D₋ₓ(v))))
        @test A ≈ B
        @test A ≉ asm((u, v) -> 3 * inner₊(∇ₕ(u), ∇ₕ(v)) + 0.1 * inner₊(∇ₕ(u), ∇ₕ(D₋ₓ(v))))
    end

    @testset "unequal lengths" begin
        @test_throws DimensionMismatch form(W, W,
            (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v) + (first(∇ₕ(v)),)))
        @test_throws DimensionMismatch form(W, W,
            (u, v) -> inner₊(∇ₕ(u), (first(∇ₕ(v)),) - ∇ₕ(v)))
    end

    WITH_3D_TESTS && @testset "3D" begin
        W3 = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) ×
                                   interval(0.0, 1.0)), (5, 6, 4), (false, false, false)))
        asm3(f) = Matrix(assemble(form(W3, W3, f)))
        A = asm3((u, v) -> inner₊(∇ₕ(u), 2 * ∇ₕ(v) - 0.25 * ∇ₕ(D₋ₓ(v))))
        B = asm3((u, v) -> 2 * inner₊(∇ₕ(u), ∇ₕ(v)) - 0.25 * inner₊(∇ₕ(u), ∇ₕ(D₋ₓ(v))))
        @test A ≈ B
        @test A ≉ asm3((u, v) -> 2 * inner₊(∇ₕ(u), ∇ₕ(v)))
    end
end

end # module
