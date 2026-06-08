using Test
using Bramble
using SparseArrays
using LinearAlgebra
using Bramble: Innerplus, backward_difference_matrix, weights, CompositeGridSpace

@testset "CompositeGridSpace & Forms Assembly" begin
    # 1. Setup
    N = 10
    I = interval(-1.0, 1.0)
    X = domain(I, markers(I, :left => x -> x[1] < -0.99, :right => x -> x[1] > 0.99))
    Mh = mesh(X, N, false)
    Wh = gridspace(Mh)
    
    # Cartesian product space
    Vh = Wh × Wh
    
    @testset "Composite Space Properties" begin
        @test Vh isa CompositeGridSpace{2}
        @test ndofs(Vh) == 2 * ndofs(Wh)
        @test dim(Vh) == dim(Wh)
        @test mesh(Vh) === mesh(Wh)
        @test eltype(Vh) === eltype(Wh)
        @test ndofs(Vh, Tuple) == (ndofs(Wh), ndofs(Wh))
    end
    
    @testset "Bilinear Form Block Diagonal Assembly" begin
        # Composite bilinear form: ∫ ∇u · ∇v dx
        a = form(Vh, Vh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))
        A = assemble(a)
        
        @test size(A) == (2 * ndofs(Wh), 2 * ndofs(Wh))
        
        # Assemble single-component form for comparison
        a_single = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))
        A_single = assemble(a_single)
        
        n = ndofs(Wh)
        # Extract blocks
        A11 = A[1:n, 1:n]
        A12 = A[1:n, n+1:end]
        A21 = A[n+1:end, 1:n]
        A22 = A[n+1:end, n+1:end]
        
        # Verify block diagonal structure
        @test A11 ≈ A_single
        @test A22 ≈ A_single
        @test norm(A12) == 0.0
        @test norm(A21) == 0.0
    end
    
    @testset "Bilinear Form with Dirichlet BCs" begin
        a = form(Vh, Vh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))
        A_bc = assemble(a, dirichlet_labels = (:left, :right))
        
        # Verify that Dirichlet boundary conditions are applied component-wise
        a_single = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))
        A_single_bc = assemble(a_single, dirichlet_labels = (:left, :right))
        
        n = ndofs(Wh)
        @test A_bc[1:n, 1:n] ≈ A_single_bc
        @test A_bc[n+1:end, n+1:end] ≈ A_single_bc
        @test norm(A_bc[1:n, n+1:end]) == 0.0
        @test norm(A_bc[n+1:end, 1:n]) == 0.0
    end
    
    @testset "Linear Form Assembly" begin
        # Composite linear form
        l = form(Vh, v -> innerₕ(1.0, v))
        b = assemble(l)
        
        @test length(b) == 2 * ndofs(Wh)
        
        l_single = form(Wh, v -> innerₕ(1.0, v))
        b_single = assemble(l_single)
        
        n = ndofs(Wh)
        @test b[1:n] ≈ b_single
        @test b[n+1:end] ≈ b_single
    end
    
    @testset "Linear Form with Dirichlet BCs" begin
        l = form(Vh, v -> innerₕ(1.0, v))
        bcs = dirichlet_constraints(Vh, :left => x -> -5.0, :right => x -> 5.0)
        
        b_bc = assemble(l, dirichlet_conditions = bcs, dirichlet_labels = (:left, :right))
        
        l_single = form(Wh, v -> innerₕ(1.0, v))
        bcs_single = dirichlet_constraints(Wh, :left => x -> -5.0, :right => x -> 5.0)
        b_single_bc = assemble(l_single, dirichlet_conditions = bcs_single, dirichlet_labels = (:left, :right))
        
        n = ndofs(Wh)
        @test b_bc[1:n] ≈ b_single_bc
        @test b_bc[n+1:end] ≈ b_single_bc
    end
end
