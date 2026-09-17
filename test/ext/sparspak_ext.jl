module TestSparspakExt

using Test
using Bramble
using LinearAlgebra
using SparseArrays
using Sparspak
using ForwardDiff

@testset "Sparspak extension" begin
    @testset "1D/2D/3D Poisson" begin
        # 1D
        Ω1 = mesh(domain(interval(0.0, 1.0), :boundary => boundary_symbols(interval(0.0, 1.0))), 21, true)
        W1 = gridspace(Ω1)
        a1 = form(W1, W1, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        f1 = Rₕ(W1, x -> sin(π * x))
        l1 = form(W1, v -> innerₕ(f1, v))
        A1, F1 = assemble(a1, l1; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        u_ref1 = A1 \ F1
        u_sp1 = pde_solve(A1, F1; solver = :sparspak)
        @test isapprox(u_sp1, u_ref1; atol = 1e-10)

        # 2D
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω2 = mesh(domain(I2, :boundary => boundary_symbols(I2)), (12, 12), (true, true))
        W2 = gridspace(Ω2)
        a2 = form(W2, W2, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        f2 = Rₕ(W2, x -> sin(π * x[1]) * sin(π * x[2]))
        l2 = form(W2, v -> innerₕ(f2, v))
        A2, F2 = assemble(a2, l2; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        u_ref2 = A2 \ F2
        u_sp2 = sparspak_solve(A2, F2)
        @test isapprox(u_sp2, u_ref2; atol = 1e-10)

        # Direct form solve returning VectorElement
        u_elem = sparspak_solve(a2, l2; dirichlet = :boundary => x -> 0.0, symmetrize = true)
        @test u_elem isa Bramble.VectorElement
        @test isapprox(parent(u_elem), u_ref2; atol = 1e-10)

        # Factorize directly from bilinear form
        fact_form = sparspak_factorize(a2; dirichlet = :boundary => x -> 0.0)
        @test fact_form isa SparspakFactorization
        @test isapprox(fact_form \ F2, u_ref2; atol = 1e-10)

        # 3D
        I3 = interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω3 = mesh(domain(I3, :boundary => boundary_symbols(I3)), (6, 6, 6), (true, true, true))
        W3 = gridspace(Ω3)
        a3 = form(W3, W3, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        f3 = Rₕ(W3, x -> sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3]))
        l3 = form(W3, v -> innerₕ(f3, v))
        A3, F3 = assemble(a3, l3; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        u_ref3 = A3 \ F3
        u_sp3 = pde_solve(A3, F3; solver = :sparspak)
        @test isapprox(u_sp3, u_ref3; atol = 1e-10)
    end

    @testset "Unsymmetric convection-diffusion" begin
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
        u_sp = pde_solve(A, F; solver = :sparspak)
        @test isapprox(u_sp, u_ref; atol = 1e-10)
    end

    @testset "Factorization reuse and refactoring (sparspak_refactor!)" begin
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω = mesh(domain(I2, :boundary => boundary_symbols(I2)), (8, 8), (true, true))
        W = gridspace(Ω)
        a = form(W, W, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        l = form(W, v -> innerₕ(Rₕ(W, x -> 1.0), v))
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        fact = sparspak_factorize(A)
        @test size(fact) == (size(A, 1), size(A, 2))
        @test size(fact, 1) == size(A, 1)

        b = copy(F)
        x = similar(b)
        ldiv!(x, fact, b)
        @test isapprox(x, A \ F; atol = 1e-10)

        b_in_place = copy(F)
        ldiv!(fact, b_in_place)
        @test isapprox(b_in_place, x; atol = 1e-10)

        # Regression: a `VectorElement` destination used to hit a method ambiguity
        # between Bramble's `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)`
        # and this extension's three-argument `ldiv!` on an `AbstractVector` destination.
        uₕ = element(W)
        @test ldiv!(uₕ, fact, F) === uₕ
        @test isapprox(parent(uₕ), A \ F; atol = 1e-10)

        # Regression: a complex right-hand side used to be ambiguous between this
        # extension's `\(::Concrete...Factorization, ::AbstractVector)` and
        # `LinearAlgebra`'s `\(::Factorization{T}, ::Vector{Complex{T}})`. The answer is
        # the real solve applied to each part.
        F_complex = complex.(F, 2 .* F)
        u_complex = fact \ F_complex
        @test u_complex isa Vector{ComplexF64}
        @test isapprox(real(u_complex), A \ F; atol = 1e-10)
        @test isapprox(imag(u_complex), A \ (2 .* F); atol = 1e-10)

        # Modify values with same sparsity pattern via unique refactor! driver
        A_mod = copy(A)
        A_mod[1, 1] += 5.0
        refactor!(fact, A_mod)
        @test isapprox(fact \ F, A_mod \ F; atol = 1e-10)

        # Unified sparse_factorize and refactor! on Matrix and BilinearForm
        fact_unified = sparse_factorize(A; solver = :sparspak)
        @test fact_unified isa SparspakFactorization
        @test isapprox(fact_unified \ F, A \ F; atol = 1e-10)
        refactor!(fact_unified, A_mod)
        @test isapprox(fact_unified \ F, A_mod \ F; atol = 1e-10)

        # Direct sparspak_refactor! call
        sparspak_refactor!(fact, A_mod)
        @test isapprox(fact \ F, A_mod \ F; atol = 1e-10)

        # Alias sparse_refactor! check
        sparse_refactor!(fact_unified, A_mod)
        @test isapprox(fact_unified \ F, A_mod \ F; atol = 1e-10)

        # Refactor directly with BilinearForm
        refactor!(fact, a; dirichlet = :boundary => x -> 0.0)
        @test isapprox(fact \ F, A \ F; atol = 1e-10)

        # Type safety: reject dense matrices
        @test_throws ArgumentError refactor!(fact, Matrix(A_mod))
        @test_throws ArgumentError sparse_refactor!(fact, Matrix(A_mod))
    end

    @testset "Non-Float64 element types (no binary dependency)" begin
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω = mesh(domain(I2, :boundary => boundary_symbols(I2)), (8, 8), (true, true))
        W = gridspace(Ω)
        a = form(W, W, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        l = form(W, v -> innerₕ(Rₕ(W, x -> 1.0), v))
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        # Float32
        A32 = SparseMatrixCSC{Float32}(A)
        F32 = Float32.(F)
        u_ref32 = Float32.(A \ F)
        u_sp32 = sparspak_solve(A32, F32)
        @test isapprox(u_sp32, u_ref32; atol = 1.0f-3)

        # ForwardDiff.Dual: Sparspak's factorization is generic over the matrix element
        # type (`sparspaklu(m::SparseMatrixCSC{FT,IT}; ...) where {FT,IT}`), so a matrix
        # assembled from a `Dual`-valued coefficient factors and solves directly, which
        # SuiteSparse/MUMPS cannot do (both require `Float64`/`ComplexF64` matrix entries).
        # Sparspak's triangular solve additionally requires the right-hand side to share
        # the matrix's own element type exactly, hence the explicit `eltype(Aθ).(...)`.
        function g(θ)
            aθ = form(W, W, (u, v) -> (1.0 + θ) * inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            Aθ, Fθ0 = assemble(aθ, l; dirichlet = :boundary => x -> 0.0)
            Fθ = eltype(Aθ).(Fθ0)
            uθ = sparspak_solve(Aθ, Fθ)
            return sum(uθ)
        end
        d = ForwardDiff.derivative(g, 1.0)
        d_fd = (g(1.0 + 1.0e-6) - g(1.0 - 1.0e-6)) / 2.0e-6
        @test isapprox(d, d_fd; atol = 1e-6, rtol = 1e-6)
    end

    @testset "Error handling & validation" begin
        I1 = interval(0.0, 1.0)
        Ω1 = mesh(domain(I1, :boundary => boundary_symbols(I1)), 5, true)
        W1 = gridspace(Ω1)
        a = form(W1, W1, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        l = form(W1, v -> innerₕ(Rₕ(W1, x -> 1.0), v))
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)

        # Non-square matrix
        A_rect = spzeros(5, 4)
        @test_throws DimensionMismatch sparspak_factorize(A_rect)

        # Dimension mismatch in ldiv!
        fact = sparspak_factorize(A)
        @test_throws DimensionMismatch ldiv!(fact, rand(length(F) + 1))
        @test_throws DimensionMismatch ldiv!(rand(length(F) + 1), fact, F)

        # Dimension mismatch in refactor!
        @test_throws DimensionMismatch sparspak_refactor!(fact, spzeros(length(F) + 1, length(F) + 1))
    end
end

end # module
