module TestAppleAccelerateExt

using Test
using Bramble
using LinearAlgebra
using SparseArrays

if Sys.isapple()
    using AppleAccelerate
end

@testset "AppleAccelerate extension" begin
    if !Sys.isapple()
        @testset "Non-macOS guard" begin
            A = spdiagm(0 => [2.0, 2.0], 1 => [-1.0], -1 => [-1.0])
            F = [1.0, 1.0]
            @test_throws ArgumentError accelerate_factorize(A)
            @test_throws ArgumentError accelerate_solve(A, F)
            @test_throws ArgumentError pde_solve(A, F; solver = :accelerate)
        end
    else
        @testset "1D/2D/3D Poisson (SPD, sym = :spd)" begin
            # 1D
            Ω1 = mesh(domain(interval(0.0, 1.0), :boundary => boundary_symbols(interval(0.0, 1.0))), 21, true)
            W1 = gridspace(Ω1)
            a1 = form(W1, W1, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            f1 = Rₕ(W1, x -> sin(π * x))
            l1 = form(W1, v -> innerₕ(f1, v))
            A1, F1 = assemble(a1, l1; dirichlet = :boundary => x -> 0.0, symmetrize = true)

            u_ref1 = A1 \ F1
            u_acc1 = pde_solve(A1, F1; solver = :accelerate, sym = :spd)
            @test isapprox(u_acc1, u_ref1; atol = 1e-12)

            # 2D
            I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
            Ω2 = mesh(domain(I2, :boundary => boundary_symbols(I2)), (12, 12), (true, true))
            W2 = gridspace(Ω2)
            a2 = form(W2, W2, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            f2 = Rₕ(W2, x -> sin(π * x[1]) * sin(π * x[2]))
            l2 = form(W2, v -> innerₕ(f2, v))
            A2, F2 = assemble(a2, l2; dirichlet = :boundary => x -> 0.0, symmetrize = true)

            u_ref2 = A2 \ F2
            u_acc2 = accelerate_solve(A2, F2; sym = :spd)
            @test isapprox(u_acc2, u_ref2; atol = 1e-12)

            # Direct form solve returning VectorElement
            u_elem = accelerate_solve(a2, l2; dirichlet = :boundary => x -> 0.0, symmetrize = true, sym = :spd)
            @test u_elem isa Bramble.VectorElement
            @test isapprox(parent(u_elem), u_ref2; atol = 1e-12)

            # Factorize directly from bilinear form
            fact_form = accelerate_factorize(a2; dirichlet = :boundary => x -> 0.0)
            @test fact_form isa AccelerateFactorization
            @test isapprox(fact_form \ F2, u_ref2; atol = 1e-12)

            # 3D
            I3 = interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)
            Ω3 = mesh(domain(I3, :boundary => boundary_symbols(I3)), (6, 6, 6), (true, true, true))
            W3 = gridspace(Ω3)
            a3 = form(W3, W3, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            f3 = Rₕ(W3, x -> sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3]))
            l3 = form(W3, v -> innerₕ(f3, v))
            A3, F3 = assemble(a3, l3; dirichlet = :boundary => x -> 0.0, symmetrize = true)

            u_ref3 = A3 \ F3
            u_acc3 = pde_solve(A3, F3; solver = :accelerate)
            @test isapprox(u_acc3, u_ref3; atol = 1e-12)
        end

        @testset "Symmetric LDLᵀ and QR" begin
            I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
            Ω2 = mesh(domain(I2, :boundary => boundary_symbols(I2)), (8, 8), (true, true))
            W2 = gridspace(Ω2)
            a2 = form(W2, W2, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            l2 = form(W2, v -> innerₕ(Rₕ(W2, x -> 1.0), v))
            A2, F2 = assemble(a2, l2; dirichlet = :boundary => x -> 0.0, symmetrize = true)

            u_ldlt = accelerate_solve(A2, F2; kind = :ldlt)
            @test isapprox(u_ldlt, A2 \ F2; atol = 1e-12)

            u_qr = accelerate_solve(A2, F2; kind = :qr)
            @test isapprox(u_qr, A2 \ F2; atol = 1e-12)
        end

        @testset "Unsymmetric convection-diffusion (sym = :unsymmetric)" begin
            I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
            Ω = mesh(domain(I2, :boundary => boundary_symbols(I2)), (10, 10), (true, true))
            W = gridspace(Ω)

            a = form(
                W, W,
                (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)) +
                          2.0 * innerₕ(D₊ₓ(u), v) +
                          1.0 * innerₕ(D₊ᵧ(u), v)
            )
            l = form(W, v -> innerₕ(Rₕ(W, x -> 1.0), v))
            A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = false)

            @test !issymmetric(A)
            u_ref = A \ F
            u_acc = accelerate_solve(A, F; sym = :unsymmetric)
            @test isapprox(u_acc, u_ref; atol = 1e-12)

            # Auto-detection correctly selects unsymmetric LUTPP
            u_auto = pde_solve(A, F; solver = :accelerate, sym = :auto)
            @test isapprox(u_auto, u_ref; atol = 1e-12)
        end

        @testset "Factorization reuse and refactoring (accelerate_refactor!)" begin
            I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
            Ω = mesh(domain(I2, :boundary => boundary_symbols(I2)), (8, 8), (true, true))
            W = gridspace(Ω)
            a = form(W, W, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            l = form(W, v -> innerₕ(Rₕ(W, x -> 1.0), v))
            A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)

            fact = accelerate_factorize(A; sym = :spd)
            @test size(fact) == (size(A, 1), size(A, 2))
            @test size(fact, 1) == size(A, 1)

            b = copy(F)
            x = similar(b)
            ldiv!(x, fact, b)
            @test isapprox(x, A \ F; atol = 1e-12)

            b_in_place = copy(F)
            ldiv!(fact, b_in_place)
            @test isapprox(b_in_place, x; atol = 1e-12)

            # In-place refactor reusing symbolic analysis via unique refactor! driver
            A_mod = copy(A)
            A_mod[1, 1] += 5.0
            refactor!(fact, A_mod)
            @test isapprox(fact \ F, A_mod \ F; atol = 1e-12)

            # Unified sparse_factorize and refactor! on Matrix and BilinearForm
            fact_unified = sparse_factorize(A; solver = :accelerate, sym = :spd)
            @test fact_unified isa AccelerateFactorization
            @test isapprox(fact_unified \ F, A \ F; atol = 1e-12)
            refactor!(fact_unified, A_mod)
            @test isapprox(fact_unified \ F, A_mod \ F; atol = 1e-12)

            # Direct accelerate_refactor! call
            accelerate_refactor!(fact, A_mod)
            @test isapprox(fact \ F, A_mod \ F; atol = 1e-12)

            # Alias sparse_refactor! check
            sparse_refactor!(fact_unified, A_mod)
            @test isapprox(fact_unified \ F, A_mod \ F; atol = 1e-12)

            # Refactor unsymmetric factorization directly with BilinearForm
            A_unsym, F_unsym = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = false)
            fact_lu = accelerate_factorize(A_unsym; sym = :unsymmetric)
            refactor!(fact_lu, a; dirichlet = :boundary => x -> 0.0)
            @test isapprox(fact_lu \ F_unsym, A_unsym \ F_unsym; atol = 1e-12)

            # Alias with BilinearForm
            sparse_refactor!(fact_lu, a; dirichlet = :boundary => x -> 0.0)
            @test isapprox(fact_lu \ F_unsym, A_unsym \ F_unsym; atol = 1e-12)

            # Type safety: reject dense matrices
            @test_throws ArgumentError refactor!(fact, Matrix(A_mod))
            @test_throws ArgumentError sparse_refactor!(fact, Matrix(A_mod))
        end

        @testset "Error handling & validation" begin
            I1 = interval(0.0, 1.0)
            Ω1 = mesh(domain(I1, :boundary => boundary_symbols(I1)), 5, true)
            W1 = gridspace(Ω1)
            a = form(W1, W1, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            l = form(W1, v -> innerₕ(Rₕ(W1, x -> 1.0), v))
            A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)

            # Invalid symmetry option
            @test_throws ArgumentError accelerate_solve(A, F; sym = :invalid_sym)

            # Non-square matrix
            A_rect = spzeros(5, 4)
            @test_throws DimensionMismatch accelerate_factorize(A_rect)

            # Dimension mismatch in ldiv!
            fact = accelerate_factorize(A)
            @test_throws DimensionMismatch ldiv!(fact, rand(length(F) + 1))
            @test_throws DimensionMismatch ldiv!(rand(length(F) + 1), fact, F)

            # Dimension mismatch in refactor!
            @test_throws DimensionMismatch accelerate_refactor!(fact, spzeros(length(F) + 1, length(F) + 1))
        end
    end
end

end # module
