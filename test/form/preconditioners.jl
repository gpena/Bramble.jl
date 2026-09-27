module TestFormPreconditioners

using Test
using Bramble
using Bramble: AbstractMatrixFreePreconditioner, JacobiPreconditioner, restrict_to, jumpₓ, M₊ₓ, D₋ₓ
using LinearAlgebra: ldiv!, diag, norm, cond, Symmetric
using LinearSolve: LinearProblem, KrylovJL_CG, solve
using Random

# Matrix-free preconditioners (gpena/Bramble.jl#327). Jacobi reads diag(A) off one stencil
# walk, so every check compares it with `diag(assemble(a; dirichlet))`. Meshes are
# non-uniform throughout, and the coefficient varies in space: a uniform constant-coefficient
# Laplacian has a constant diagonal, which Jacobi only rescales.

const PC_SEED = 3271

_pc_agree(a, b) = isapprox(a, b; rtol = 1e-12, atol = 1e-12 * max(1.0, norm(b, Inf)))

# Inside functions so `@allocated` measures the call alone.
_pc_alloc3(y, P, x) = (ldiv!(y, P, x); @allocated ldiv!(y, P, x))
_pc_alloc2(P, x) = (ldiv!(P, x); @allocated ldiv!(P, x))
_pc_alloc_build(a, kw) = (jacobi_preconditioner(a; kw...); @allocated jacobi_preconditioner(a; kw...))

function _pc_spaces()
    Random.seed!(PC_SEED)
    return (
        gridspace(mesh(domain(interval(0.0, 1.0)), 17, false)),
        gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (9, 11), (false, true))),
        gridspace(mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7, 5), false))
    )
end

# (name, form, dirichlet): variable diffusion, a transposed difference pair, a region
# restriction per dimension, then a composite form with a crossed block.
function _pc_cases()
    out = Any[]
    for W in _pc_spaces()
        D = dim(W)
        κ = Rₕ(W, x -> 1 + sum(abs2, x))
        push!(out, ("$(D)D diffusion", form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))), :boundary))
        push!(out, (
            "$(D)D pair", form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(D₋ₓ(u), v) + 2.0 * innerₕ(u, D₋ₓ(v))), nothing))
        push!(out, (
            "$(D)D restricted", form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(κ * u, restrict_to(:boundary, v))), nothing))
    end
    W = _pc_spaces()[2]
    V = W × W
    push!(out, ("composite",
        form(V, V, (u, v) -> innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) + innerₕ(u(1), v(2))), :boundary))
    return out
end

_pc_kw(dl) = dl === nothing ? (;) : (; dirichlet = dl)

function _pc_cg(A, b; kw...)
    sol = solve(LinearProblem(A, b), KrylovJL_CG(); reltol = 1e-8, abstol = 0.0, maxiters = 20_000, kw...)
    return sol.iters, norm(A * sol.u - b) / norm(b)
end

function _pc_spd_form(n)
    Random.seed!(PC_SEED)
    W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n), (false, false)))
    κ = Rₕ(W, x -> 1 + 10 * sum(abs2, x))
    return form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v)))
end

@testset "matrix-free preconditioners (#327)" begin
    @testset "jacobi: matrix-free diagonal" begin
        for (name, a, dl) in _pc_cases()
            @testset "$name" begin
                kw = _pc_kw(dl)
                A = assemble(a; kw...)
                n = size(A, 1)
                P = jacobi_preconditioner(a; kw...)
                @test P isa JacobiPreconditioner{Float64}
                @test P isa AbstractMatrixFreePreconditioner{Float64}
                @test size(P) == (n, n)
                x = randn(n)
                y = similar(x)
                ldiv!(y, P, x)
                @test _pc_agree(y, x ./ diag(A))
                @test P \ x == y
                z = copy(x)
                ldiv!(P, z)
                @test z == y
                ldiv!(z, jacobi_preconditioner(matrix_free_operator(a; kw...)), x)
                @test z == y
                # O(ndofs): the diagonal, its mask and a fixed setup, never a matrix.
                @test _pc_alloc_build(a, kw) < 4 * 8 * n + 4096
                @test _pc_alloc3(y, P, x) == 0
                @test _pc_alloc2(P, z) == 0
            end
        end
    end

    @testset "jacobi: dirichlet_components" begin
        W = _pc_spaces()[1]
        V = W × W
        a = form(V, V, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + 3.0 * innerₕ(u(2), v(2)) + innerₕ(u(1), v(2)))
        x = randn(ndofs(V))
        for comps in (1, 2, (1, 2), nothing)
            A = assemble(a; dirichlet = :boundary, dirichlet_components = comps)
            P = jacobi_preconditioner(a; dirichlet = :boundary, dirichlet_components = comps)
            @test _pc_agree(P \ x, x ./ diag(A))
        end
        @test_throws ArgumentError jacobi_preconditioner(a; dirichlet = :boundary, dirichlet_components = 3)
    end

    @testset "jacobi: errors" begin
        W = _pc_spaces()[1]
        a = form(W, W, (u, v) -> innerₕ(u, v))
        err = try
            jacobi_preconditioner(a; policy = Bramble.GpuKernel())
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("v4.4.0", sprint(showerror, err))
        Wf = gridspace(mesh(domain(interval(0.0, 1.0)), 9, false))
        @test_throws DimensionMismatch jacobi_preconditioner(form(Wf, W, (u, v) -> innerₕ(πₕ(u), v)))
        P = jacobi_preconditioner(a)
        @test_throws DimensionMismatch ldiv!(zeros(3), P, zeros(ndofs(W)))
    end

    # Jacobi equilibrates a variable coefficient on a non-uniform mesh, so it lowers both the
    # condition number and the CG iteration count, on the assembled matrix and the operator.
    @testset "jacobi: LinearSolve Pl" begin
        a = _pc_spd_form(17)
        A = Matrix(assemble(a))
        s = 1 ./ sqrt.(diag(A))
        @test cond(Symmetric(s .* A .* s')) < cond(Symmetric(A)) / 2
        a = _pc_spd_form(49)
        op = matrix_free_operator(a)
        b = randn(size(op, 1))
        P = jacobi_preconditioner(a)
        for M in (op, assemble(a))
            i0, r0 = _pc_cg(M, b)
            i1, r1 = _pc_cg(M, b; Pl = P)
            @test r0 < 1e-7
            @test r1 < 1e-7
            @test i1 < i0
        end
    end
end

end # module
