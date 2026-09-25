module TestSparseSolversInterface

using Test
using Bramble
using LinearAlgebra
using SparseArrays
using Bramble: refactor!, sparse_factorize, sparse_refactor!

@testset "Sparse solver interface (core fallback & validation)" begin
    I1 = interval(0.0, 1.0)
    Ω1 = mesh(domain(I1, :boundary => boundary_symbols(I1)), 10, true)
    W1 = gridspace(Ω1)
    a1 = form(W1, W1, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    l1 = form(W1, v -> innerₕ(Rₕ(W1, x -> sin(π * x)), v))
    A, F = assemble(a1, l1; dirichlet = :boundary => x -> 0.0, symmetrize = true)

    # 1. Alias equivalence
    @test sparse_refactor! === refactor!

    # 2. Default pde_solve
    u_default = pde_solve(A, F)
    @test isapprox(u_default, A \ F; atol = 1e-12)

    u_default_sym = pde_solve(A, F; solver = :default)
    @test isapprox(u_default_sym, A \ F; atol = 1e-12)

    # 3. pde_solve on Factorization object
    fact_lu = lu(A)
    @test isapprox(pde_solve(fact_lu, F), u_default; atol = 1e-12)

    # 4. Unknown solver error
    @test_throws ArgumentError sparse_factorize(A; solver = :nonexistent_solver)
    @test_throws ArgumentError sparse_factorize(a1; solver = :nonexistent_solver)
    @test_throws ArgumentError pde_solve(A, F; solver = :nonexistent_solver)

    # 5. Type safety: refactor! only accepts SparseMatrixCSC (or BilinearForm)
    @test_throws ArgumentError refactor!(fact_lu, Matrix(A))
    @test_throws ArgumentError sparse_refactor!(fact_lu, Matrix(A))
    @test_throws ArgumentError refactor!(fact_lu, [1.0, 2.0])

    # 6. Type safety: sparse_factorize only accepts SparseMatrixCSC
    @test_throws MethodError sparse_factorize(Matrix(A))
end

end # module
