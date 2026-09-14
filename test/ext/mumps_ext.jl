module TestMUMPSExt

using Test
using Bramble
using LinearAlgebra
using SparseArrays
using MUMPS

@testset "MUMPS extension" begin
    @testset "1D/2D/3D Poisson (SPD, sym = :spd)" begin
        # 1D
        Ω1 = mesh(domain(interval(0.0, 1.0), :boundary => boundary_symbols(interval(0.0, 1.0))), 21, true)
        W1 = gridspace(Ω1)
        a1 = form(W1, W1, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        f1 = Rₕ(W1, x -> sin(π * x))
        l1 = form(W1, v -> innerₕ(f1, v))
        A1, F1 = assemble(a1, l1; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        u_ss1 = A1 \ F1
        u_mumps1 = pde_solve(A1, F1; solver = :mumps, sym = :spd)
        @test isapprox(u_mumps1, u_ss1; atol = 1e-12)

        # 2D
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω2 = mesh(domain(I2, :boundary => boundary_symbols(I2)), (12, 12), (true, true))
        W2 = gridspace(Ω2)
        a2 = form(W2, W2, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        f2 = Rₕ(W2, x -> sin(π * x[1]) * sin(π * x[2]))
        l2 = form(W2, v -> innerₕ(f2, v))
        A2, F2 = assemble(a2, l2; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        u_ss2 = A2 \ F2
        u_mumps2 = mumps_solve(A2, F2; sym = :spd)
        @test isapprox(u_mumps2, u_ss2; atol = 1e-12)

        # Direct form solve returning VectorElement
        u_elem = mumps_solve(a2, l2; dirichlet = :boundary => x -> 0.0, symmetrize = true, sym = :spd)
        @test u_elem isa Bramble.VectorElement
        @test isapprox(parent(u_elem), u_ss2; atol = 1e-12)

        # 3D
        I3 = interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω3 = mesh(domain(I3, :boundary => boundary_symbols(I3)), (6, 6, 6), (true, true, true))
        W3 = gridspace(Ω3)
        a3 = form(W3, W3, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        f3 = Rₕ(W3, x -> sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3]))
        l3 = form(W3, v -> innerₕ(f3, v))
        A3, F3 = assemble(a3, l3; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        u_ss3 = A3 \ F3
        u_mumps3 = pde_solve(A3, F3; solver = :mumps)
        @test isapprox(u_mumps3, u_ss3; atol = 1e-12)
    end

    @testset "Symmetric indefinite (Helmholtz / saddle point, sym = :symmetric)" begin
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω2 = mesh(domain(I2, :boundary => boundary_symbols(I2)), (10, 10), (true, true))
        W2 = gridspace(Ω2)
        # Shifted Helmholtz: -Δu - k²u = f (indefinite symmetric matrix)
        k2 = 50.0
        a_helm = form(W2, W2, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)) - k2 * innerₕ(u, v))
        f_helm = Rₕ(W2, x -> 1.0)
        l_helm = form(W2, v -> innerₕ(f_helm, v))
        A_h, F_h = assemble(a_helm, l_helm; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        u_ss = A_h \ F_h
        fact_h = mumps_factorize(A_h; sym = :symmetric)
        u_mumps = fact_h \ F_h
        @test isapprox(u_mumps, u_ss; atol = 1e-12)
    end

    @testset "Unsymmetric (Convection-diffusion, sym = :unsymmetric)" begin
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω2 = mesh(domain(I2, :boundary => boundary_symbols(I2)), (12, 12), (true, true))
        W2 = gridspace(Ω2)
        # Convection-diffusion: -Δu + β ⋅ ∇u = f
        βx = 5.0
        βy = 2.0
        a_cd = form(
            W2,
            W2,
            (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)) + βx * innerₕ(Bramble.D₊ₓ(u), v) + βy * innerₕ(Bramble.D₊ᵧ(u), v)
        )
        f_cd = Rₕ(W2, x -> 1.0)
        l_cd = form(W2, v -> innerₕ(f_cd, v))
        A_cd, F_cd = assemble(a_cd, l_cd; dirichlet = :boundary => x -> 0.0)

        @test !issymmetric(A_cd)
        u_ss = A_cd \ F_cd
        u_mumps = pde_solve(A_cd, F_cd; solver = :mumps, sym = :unsymmetric)
        @test isapprox(u_mumps, u_ss; atol = 1e-12)
    end

    @testset "Factorization reuse for time stepping" begin
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω2 = mesh(domain(I2, :boundary => boundary_symbols(I2)), (8, 8), (true, true))
        W2 = gridspace(Ω2)
        a = form(W2, W2, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        l = form(W2, v -> innerₕ(Rₕ(W2, x -> 1.0), v))
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        fact = factorize(A, MUMPSFactorization; sym = :spd)
        @test fact isa MUMPSFactorization

        u1 = similar(F)
        ldiv!(u1, fact, F)
        @test isapprox(u1, A \ F; atol = 1e-12)

        # Repeated solves with different RHS vectors
        F2 = rand(length(F))
        u2 = similar(F2)
        ldiv!(u2, fact, F2)
        @test isapprox(u2, A \ F2; atol = 1e-12)

        # In-place mutating vector solve
        F3 = copy(F2)
        ldiv!(fact, F3)
        @test isapprox(F3, u2; atol = 1e-12)
    end

    @testset "Control options (icntl, cntl)" begin
        I1 = interval(0.0, 1.0)
        Ω1 = mesh(domain(I1, :boundary => boundary_symbols(I1)), 15, true)
        W1 = gridspace(Ω1)
        a = form(W1, W1, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        l = form(W1, v -> innerₕ(Rₕ(W1, x -> 1.0), v))
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        # Custom memory relaxation and METIS ordering (7 => 5)
        fact = mumps_factorize(A; sym = :spd, icntl = (7 => 5, 14 => 35))
        u = fact \ F
        @test isapprox(u, A \ F; atol = 1e-12)

        # pde_solve with MUMPSFactorization directly
        @test isapprox(pde_solve(fact, F), u; atol = 1e-12)
    end

    @testset "Error handling & validation" begin
        I1 = interval(0.0, 1.0)
        Ω1 = mesh(domain(I1, :boundary => boundary_symbols(I1)), 5, true)
        W1 = gridspace(Ω1)
        a = form(W1, W1, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        l = form(W1, v -> innerₕ(Rₕ(W1, x -> 1.0), v))
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        # Invalid symmetry option
        @test_throws ArgumentError mumps_solve(A, F; sym = :invalid_sym)
        @test_throws ArgumentError pde_solve(A, F; solver = :invalid_solver)

        # Non-square matrix
        A_rect = spzeros(5, 4)
        @test_throws DimensionMismatch mumps_factorize(A_rect)

        # Dimension mismatch in ldiv!
        fact = mumps_factorize(A)
        @test_throws DimensionMismatch ldiv!(fact, rand(length(F) + 1))
        @test_throws DimensionMismatch ldiv!(rand(length(F) + 1), fact, F)
    end
end

end # module
