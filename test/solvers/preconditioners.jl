module SolversPreconditionersTests

using Test
using Bramble
using Bramble: AbstractMatrixFreePreconditioner, JacobiPreconditioner, ChebyshevPreconditioner, max_eigenvalue_estimate,
               restrict_to, jumpₓ, M₊ₓ, D₋ₓ
using LinearAlgebra: LinearAlgebra, ldiv!, diag, norm, cond, Symmetric, eigmax, isposdef
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

# One space's three cases, in a function of its own so each call is compiled for one concrete
# space type: looping over the 1D/2D/3D tuple inline makes inference pair a trial function of
# one dimension with a test function of another, which never happens.
function _pc_push_dim_cases!(out, W)
    D = dim(W)
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    push!(out, ("$(D)D diffusion", form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))), :boundary))
    push!(out, (
        "$(D)D pair", form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(D₋ₓ(u), v) + 2.0 * innerₕ(u, D₋ₓ(v))), nothing))
    push!(out, (
        "$(D)D restricted", form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(κ * u, restrict_to(:boundary, v))), nothing))
    return out
end

# (name, form, dirichlet): variable diffusion, a transposed difference pair, a region
# restriction per dimension, then a composite form with a crossed block.
function _pc_cases()
    out = Tuple{String, Bramble.BilinearForm, Union{Nothing, Symbol, Tuple{Vararg{Symbol}}}}[]
    for W in _pc_spaces()
        _pc_push_dim_cases!(out, W)
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

# Chebyshev (#327): mass plus variable diffusion on a non-uniform mesh in 1D-3D, SPD with no
# Dirichlet rows, the problem the preconditioner is built for.
function _pc_cheb_form(D, n)
    Random.seed!(PC_SEED)
    I = interval(0.0, 1.0)
    Ω = D == 1 ? I : D == 2 ? I × I : I × I × I
    W = gridspace(mesh(domain(Ω), ntuple(_ -> n, D), ntuple(_ -> false, D)))
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    return form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v)))
end

# P⁻¹ as a dense matrix from the definition, valid with Dirichlet rows (A unsymmetric): with
# B = D⁻¹A and t = (θ - B)/δ, B q(B) = 1 - T_k(t) / T_k(θ/δ), so P⁻¹ = q(B) D⁻¹ = A⁻¹ D (1 - T_k(t) / T_k(θ/δ)) D⁻¹.
function _pc_cheb_dense(A, λmin, λmax, k)
    d = diag(A)
    θ, δ = (λmax + λmin) / 2, (λmax - λmin) / 2
    Id = Matrix{Float64}(LinearAlgebra.I, size(A))
    t = (θ * Id - A ./ d) / δ
    T0, T1 = Id, t
    for _ in 2:k
        T0, T1 = T1, 2 * t * T1 - T0
    end
    σ = θ / δ
    Tσ = cosh(k * acosh(σ))
    return A \ (d .* (Id - T1 / Tσ) ./ d')
end

function _pc_cheb_matrix(P, n)
    M = Matrix{Float64}(undef, n, n)
    for j in 1:n
        M[:, j] = P \ [Float64(i == j) for i in 1:n]
    end
    return M
end

# A vector indexed from 0, to check that `ldiv!` refuses offset axes.
struct _PcZeroBased <: AbstractVector{Float64}
    p::Vector{Float64}
end
Base.size(v::_PcZeroBased) = size(v.p)
Base.axes(v::_PcZeroBased) = (Base.IdentityUnitRange(0:(length(v.p) - 1)),)
Base.getindex(v::_PcZeroBased, i::Int) = v.p[i + 1]
Base.setindex!(v::_PcZeroBased, x, i::Int) = (v.p[i + 1] = x)

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
    # The polynomial is the one its docstring states, SPD, and applied without allocating;
    # the form route with Dirichlet rows agrees with the operator route.
    @testset "chebyshev: the polynomial" begin
        for D in 1:3
            a = _pc_cheb_form(D, (17, 9, 5)[D])
            A = Matrix(assemble(a; dirichlet = :boundary))
            n = size(A, 1)
            for k in (1, 2, 4)
                P = chebyshev_preconditioner(a; dirichlet = :boundary, degree = k, λmax = 2.5, ratio = 20)
                @test P isa ChebyshevPreconditioner{Float64}
                @test P isa AbstractMatrixFreePreconditioner{Float64}
                @test size(P) == (n, n)
                @test isapprox(_pc_cheb_matrix(P, n), _pc_cheb_dense(A, 2.5 / 20, 2.5, k); rtol = 1e-9)
                # Without Dirichlet rows A is SPD, and so is P⁻¹.
                M = _pc_cheb_matrix(chebyshev_preconditioner(a; degree = k, λmax = 2.5, ratio = 20), n)
                @test norm(M - M') <= 1e-12 * norm(M)
                @test isposdef(Symmetric(M))
            end
            op = matrix_free_operator(a; dirichlet = :boundary)
            P = chebyshev_preconditioner(op)
            @test P.λmax == max_eigenvalue_estimate(op; preconditioner = jacobi_preconditioner(op))
            @test P.λmin == P.λmax / 30
            x = randn(n)
            y = similar(x)
            ldiv!(y, P, x)
            @test y == chebyshev_preconditioner(a; dirichlet = :boundary) \ x
            z = copy(x)
            ldiv!(P, z)
            @test z == y
            @test _pc_alloc3(y, P, x) == 0
            @test _pc_alloc2(P, z) == 0
        end
    end

    @testset "chebyshev: dirichlet_components" begin
        W = _pc_spaces()[1]
        V = W × W
        a = form(V, V, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + 3.0 * innerₕ(u(2), v(2)) + innerₕ(u(1), v(1)))
        x = randn(ndofs(V))
        for comps in (1, 2, nothing)
            A = Matrix(assemble(a; dirichlet = :boundary, dirichlet_components = comps))
            P = chebyshev_preconditioner(a; dirichlet = :boundary, dirichlet_components = comps, degree = 3,
                λmax = 2.0, ratio = 10)
            @test isapprox(P \ x, _pc_cheb_dense(A, 0.2, 2.0, 3) * x; rtol = 1e-9)
        end
        @test_throws ArgumentError chebyshev_preconditioner(a; dirichlet = :boundary, dirichlet_components = 3)
    end

    @testset "chebyshev: errors" begin
        W = _pc_spaces()[1]
        a = form(W, W, (u, v) -> innerₕ(u, v))
        op = matrix_free_operator(a)
        @test_throws ArgumentError chebyshev_preconditioner(op; degree = 0)
        @test_throws ArgumentError chebyshev_preconditioner(op; ratio = 1)
        @test_throws ArgumentError chebyshev_preconditioner(op; λmax = -1.0)
        @test_throws ArgumentError chebyshev_preconditioner(op; λmax = Inf)
        @test_throws ArgumentError chebyshev_preconditioner(a; policy = Bramble.GpuKernel())
        @test_throws ArgumentError max_eigenvalue_estimate(op; iterations = 0)
        Wf = gridspace(mesh(domain(interval(0.0, 1.0)), 9, false))
        rect = matrix_free_operator(form(Wf, W, (u, v) -> innerₕ(πₕ(u), v)))
        @test_throws DimensionMismatch chebyshev_preconditioner(rect)
        @test_throws DimensionMismatch max_eigenvalue_estimate(rect)
        P = chebyshev_preconditioner(op)
        n = ndofs(W)
        @test_throws DimensionMismatch ldiv!(zeros(3), P, zeros(n))
        @test_throws ArgumentError ldiv!(zeros(n), P, _PcZeroBased(zeros(n)))
        @test_throws ArgumentError ldiv!(_PcZeroBased(zeros(n)), P, zeros(n))
    end

    # Power iteration from below, lifted by the documented factor 1.1: it must land above the
    # top eigenvalue (which the polynomial needs) and not far past it, for A and for D⁻¹A. For
    # symmetric A it is at most 1.1λ up to rounding; D⁻¹A is not symmetric, so it may pass that.
    @testset "chebyshev: λmax estimate" begin
        for D in 1:3
            a = _pc_cheb_form(D, (65, 17, 9)[D])
            op = matrix_free_operator(a)
            A = Matrix(assemble(a))
            s = 1 ./ sqrt.(diag(A))
            λ = eigmax(Symmetric(A))
            λs = eigmax(Symmetric(s .* A .* s'))
            est = max_eigenvalue_estimate(op)
            @test λ <= est <= 1.1λ * (1 + 1e-12)
            @test est == max_eigenvalue_estimate(op)
            @test λ <= max_eigenvalue_estimate(op; iterations = 40) <= 1.1λ * (1 + 1e-12)
            est = max_eigenvalue_estimate(op; preconditioner = jacobi_preconditioner(op))
            @test λs <= est <= 1.15λs
        end
    end

    # Degree 4 costs three products per application; it must at least halve the iterations
    # of plain CG, on the operator and on the assembled matrix.
    @testset "chebyshev: CG iterations halve" begin
        for D in 1:3
            a = _pc_cheb_form(D, (129, 33, 11)[D])
            op = matrix_free_operator(a)
            b = randn(size(op, 1))
            P = chebyshev_preconditioner(a)
            for M in (op, assemble(a))
                i0, r0 = _pc_cg(M, b)
                i1, r1 = _pc_cg(M, b; Pl = P)
                @test r0 < 1e-7
                @test r1 < 1e-7
                @test 2 * i1 <= i0
            end
        end
    end
end

end # module
