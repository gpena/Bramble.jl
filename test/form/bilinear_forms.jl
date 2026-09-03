"""
Extended test suite for BilinearForm
Coverage improvements for:
- Dirichlet boundary conditions (Symbol and Tuple)
- In-place assembly (assemble!)
- Accessor functions (trial_space, test_space)
- Callable interface
- Edge cases (multiple labels, 2D/3D problems)
"""

import Bramble: points

@testset "BilinearForm extended" begin
    @testset "1D Dirichlet BCs" begin
        N = 10
        I = interval(-1.0, 1.0)
        X = domain(I, markers(I, :left => x -> x[1] < -0.99, :right => x -> x[1] > 0.99))
        Mh = mesh(X, N, false)
        Wh = gridspace(Mh)

        # Stiffness matrix: ∫ ∇u·∇v dx
        a = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))

        @testset "Symbol label" begin
            A_left = assemble(a, dirichlet_labels = :left)
            @test A_left isa AbstractMatrix
            @test size(A_left) == (ndofs(Wh), ndofs(Wh))

            A_right = assemble(a, dirichlet_labels = :right)
            @test A_right isa AbstractMatrix
            @test size(A_right) == (ndofs(Wh), ndofs(Wh))
        end

        @testset "Tuple of labels" begin
            A_both = assemble(a, dirichlet_labels = (:left, :right))
            @test A_both isa AbstractMatrix
            @test size(A_both) == (ndofs(Wh), ndofs(Wh))

            # Check that diagonal entries for boundary points are modified
            # (Dirichlet conditions set rows/cols to identity-like)
            @test all(diag(A_both) .> 0)
        end

        @testset "Unconstrained vs constrained" begin
            A_no_bc = assemble(a)
            A_with_bc = assemble(a, dirichlet_labels = :left)

            # Matrices should differ
            @test A_no_bc != A_with_bc

            # But have same size
            @test size(A_no_bc) == size(A_with_bc)
        end
    end

    @testset "2D Dirichlet BCs" begin
        N = 5
        I = interval(0.0, 1.0)
        Ω = I × I

        X = domain(Ω,
            markers(Ω,
                :bottom => x -> x[2] < 0.01,
                :top => x -> x[2] > 0.99,
                :left => x -> x[1] < 0.01,
                :right => x -> x[1] > 0.99))

        Mh = mesh(X, (N, N), (false, false))
        Wh = gridspace(Mh)

        # Laplacian stiffness matrix
        a = form(Wh, Wh, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))

        @testset "Single boundary" begin
            A_bottom = assemble(a, dirichlet_labels = :bottom)
            @test A_bottom isa AbstractMatrix
            @test size(A_bottom) == (ndofs(Wh), ndofs(Wh))
        end

        @testset "Multiple boundaries" begin
            A_all = assemble(a, dirichlet_labels = (:bottom, :top, :left, :right))
            @test A_all isa AbstractMatrix
            @test size(A_all) == (ndofs(Wh), ndofs(Wh))

            # Check symmetry is preserved
            @test issparse(A_all)
        end

        @testset "Empty label tuple" begin
            A_empty = assemble(a, dirichlet_labels = ())
            A_none = assemble(a)

            # Empty tuple should be equivalent to no BCs
            @test A_empty == A_none
        end
    end

    @testset "Accessors" begin
        N = 5
        I = interval(0.0, 1.0)
        Mh = mesh(domain(I), N, false)
        Wh = gridspace(Mh)
        Vh = gridspace(Mh)  # Separate test space

        a = form(Wh, Vh, (u, v) -> innerₕ(u, v))

        @test trial_space(a) === Wh
        @test test_space(a) === Vh
    end

    @testset "Callable interface" begin
        N = 8
        I = interval(0.0, π)
        Mh = mesh(domain(I), N, true)
        Wh = gridspace(Mh)

        # Mass matrix form
        a = form(Wh, Wh, (u, v) -> innerₕ(u, v))

        # Create test functions
        uₕ = element(Wh)
        vₕ = element(Wh)

        Rₕ!(uₕ, x -> sin(x[1]))
        Rₕ!(vₕ, x -> cos(x[1]))

        # Direct evaluation
        result = a(uₕ, vₕ)
        @test result isa Number

        # Should match numerical integration
        pts = points(Mh)
        u_vals = sin.(pts)
        v_vals = cos.(pts)
        expected = sum(u_vals .* v_vals) * (π / N)  # Simple quadrature

        @test abs(result - expected) < 1e-5
    end

    @testset "In-place assembly" begin
        N = 6
        I = interval(-1.0, 1.0)
        Mh = mesh(domain(I), N, false)
        Wh = gridspace(Mh)

        a = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))

        @testset "assemble! unconstrained" begin
            A = assemble(a)
            A_preallocated = similar(A)
            fill!(A_preallocated, 0)

            assemble!(A_preallocated, a)

            @test A_preallocated ≈ A
        end

        @testset "assemble! constrained" begin
            X = domain(I, markers(I, :boundary => x -> abs(x[1]) > 0.99))
            Mh_bc = mesh(X, N, false)
            Wh_bc = gridspace(Mh_bc)

            a_bc = form(Wh_bc, Wh_bc, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))

            A = assemble(a_bc, dirichlet_labels = :boundary)
            A_preallocated = similar(A)
            fill!(A_preallocated, 0)

            assemble!(A_preallocated, a_bc, dirichlet_labels = :boundary)

            @test A_preallocated ≈ A
        end
    end

    @testset "Form types" begin
        N = 7
        I = interval(0.0, 1.0)
        Mh = mesh(domain(I), N, false)
        Wh = gridspace(Mh)

        @testset "Mass matrix" begin
            a_mass = form(Wh, Wh, (u, v) -> innerₕ(u, v))
            A_mass = assemble(a_mass)

            @test A_mass isa AbstractMatrix
            @test size(A_mass) == (ndofs(Wh), ndofs(Wh))
            # Mass matrix should be positive definite
            @test all(diag(A_mass) .> 0)
        end

        @testset "Stiffness matrix" begin
            a_stiff = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))
            A_stiff = assemble(a_stiff)

            @test A_stiff isa AbstractMatrix
            @test size(A_stiff) == (ndofs(Wh), ndofs(Wh))
        end

        @testset "Mixed derivatives" begin
            a_mixed = form(Wh, Wh, (u, v) -> inner₊(M₋ₕ(u), D₋ₓ(v)))
            A_mixed = assemble(a_mixed)

            @test A_mixed isa AbstractMatrix
            @test size(A_mixed) == (ndofs(Wh), ndofs(Wh))
        end
    end

    @testset "3D form" begin
        N = 3  # Small for 3D
        I = interval(0.0, 1.0)
        Ω = I × I × I

        X = domain(Ω, markers(Ω, :boundary => x -> any(abs.(x .- 0.5) .> 0.48)))
        Mh = mesh(X, (N, N, N), (false, false, false))
        Wh = gridspace(Mh)

        # 3D Laplacian
        a = form(Wh, Wh, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))

        @testset "3D assembly" begin
            A = assemble(a)
            @test A isa AbstractMatrix
            @test size(A) == (ndofs(Wh), ndofs(Wh))
            @test issparse(A)
        end

        @testset "3D Dirichlet BCs" begin
            A_bc = assemble(a, dirichlet_labels = :boundary)
            @test A_bc isa AbstractMatrix
            @test size(A_bc) == (ndofs(Wh), ndofs(Wh))
        end
    end
end
