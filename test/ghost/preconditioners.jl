module GhostPreconditionersTests

# Preconditioners and the fast-diagonalisation entry points on forms over ghost spaces
# (#781). Each entry point either works, checked against the same object built from the
# assembled reduced matrix (or a dense oracle) and by a preconditioned CG solve that reaches
# `pde_solve`'s solution, or refuses with an `ArgumentError` naming `assemble(a)`. `SUPPORT`
# lists every entry point with its status; the testsets below, named `works:` or
# `refuses:`, assert its rows. The meshes are seeded non-uniform ones, with Neumann, Robin,
# and Dirichlet and Robin relations together, a Dirichlet facet as a relation `u => g`.

using Test
using Bramble
import Bramble
using Bramble: D₋ₓ, closure, padded_space
using AlgebraicMultigrid: aspreconditioner
using ILUZero: ILUZero
using Kronecker: Kronecker
using LinearAlgebra: Diagonal, Symmetric, diag, dot, eigen, isposdef, ldiv!, norm
using Random: MersenneTwister

using ..TestUtils: WITH_3D_TESTS

# Entry point => status on a ghost-space form; "where separable" means the closure and
# every term factor into 1D (see `kronecker_operator`), and the form is Laplacian-like.
const SUPPORT = (
    "jacobi_preconditioner(a)" => :works,
    "jacobi_preconditioner(matrix_free_operator(a))" => :works,
    "chebyshev_preconditioner(a)" => :works,
    "chebyshev_preconditioner(matrix_free_operator(a))" => :works,
    "max_eigenvalue_estimate(matrix_free_operator(a))" => :works,
    "ilu_preconditioner(a)" => :works,
    "amg_preconditioner(a)" => :works,
    "fdm_solve(a, F) where separable" => :works,
    "fdm_factorize(a), fdm_solve!, fdm_factorize!(f, a) where separable" => :works,
    "fdm_preconditioner(a) where separable" => :works,
    "fdm_* on a closure that is not separable (conormal K)" => :refuses,
    "fdm_* on a term that is not separable (Δₕ)" => :refuses,
    "fdm_* on a 1D form" => :refuses,
    "fdm_* with dirichlet" => :refuses
)

function seeded_mesh(ns::NTuple{D, Int}) where {D}
    Ω = domain(foldl(×, ntuple(d -> interval(0.0, Float64(d)), D)))
    Bramble._seed_mesh1d_rng!(20261017)
    try
        return mesh(Ω, D == 1 ? only(ns) : ns, D == 1 ? false : ntuple(_ -> false, D))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

nonuniform(Ωₕ) = all(d -> !allequal(round.(diff(Bramble.points(Ωₕ(d))); digits = 12)),
    1:Bramble.dim(Ωₕ))

relerr(x, y) = norm(x - y) / norm(y)

# `-Δₕ` reads the ghosts, so the relations shape the reduced matrix; it is not separable
# into Kronecker factors (`laplacian` serves the refusal). `separable` is, and reads the
# `:xmin` ghost through `D₋ₓ`; `stiffness` is its symmetric part.
laplacian(u, v) = innerₕ(-1.0 * Δₕ(u), v) + innerₕ(u, v)
stiffness(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
separable(u, v) = stiffness(u, v) + innerₕ(D₋ₓ(u), v)
source(C) = form(C, v -> innerₕ(x -> 1.0 + x[1]^2, v))

const K = [2.0 1.0; 1.0 2.0]

CASES_1D = (
    "Neumann" => ((15,), :boundary, u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙ(u) => 0.5)),
    "Robin" => ((15,), :boundary,
        u -> (:xmin => u + ∂ₙ(u) => 2.0, :xmax => 3.0 * u + ∂ₙ(u) => 1.0)),
    "Dirichlet–Robin" => ((15,), (:xmax,),
        u -> (:xmin => u => 1.0, :xmax => 2.0 * u + ∂ₙ(u) => 0.5))
)

CASES_2D = (
    "Neumann" => ((13, 11), :boundary,
        u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0,
            :ymax => ∂ₙ(u) => x -> x[1])),
    "Robin" => ((13, 11), :boundary,
        u -> (:xmin => 2.0 * u + ∂ₙ(u) => 0.0, :xmax => 0.5 * u + 3.0 * ∂ₙ(u) => 1.0,
            :ymin => u + ∂ₙ(u) => 0.0, :ymax => 4.0 * u + ∂ₙ(u) => x -> x[1])),
    "Dirichlet–Robin" => ((13, 11), (:xmin, :xmax, :ymin),
        u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => 2.0 * u + ∂ₙ(u) => x -> x[2],
            :ymin => u + ∂ₙ(u) => 0.5, :ymax => u => x -> x[1]))
)

CASES_3D = (
    "Robin and Dirichlet" => ((5, 6, 4), (:xmin, :xmax, :ymin, :ymax, :zmax),
    u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[3],
        :ymin => ∂ₙ(u) => 0.0, :ymax => 2.0 * u + ∂ₙ(u) => 0.0,
        :zmin => u => x -> x[1], :zmax => ∂ₙ(u) => 1.0)),
)

function space(ns, ghosts, rel)
    Ωₕ = seeded_mesh(ns)
    @test nonuniform(Ωₕ)
    return constrained(gridspace(Ωₕ; ghosts), rel)
end

# `P⁻¹ x` through the three-argument `ldiv!` every preconditioner here supplies.
apply(P, x) = ldiv!(similar(x), P, x)

# Preconditioned conjugate gradients, written out so that it shares no code with the
# preconditioners: the iterate once the residual is `1e-12` of `b`, or `nothing`.
function pcg(A, b, P; maxiter = 2000)
    x = zero(b)
    r = copy(b)
    z = apply(P, r)
    p = copy(z)
    rz = dot(r, z)
    for _ in 1:maxiter
        q = A * p
        α = rz / dot(p, q)
        x .+= α .* p
        r .-= α .* q
        norm(r) <= 1e-12 * norm(b) && return x
        z = apply(P, r)
        rz₊ = dot(r, z)
        p .= z .+ (rz₊ / rz) .* p
        rz = rz₊
    end
    return nothing
end

# Checks that `x` is the solution `pde_solve` gives of the assembled system.
function converges(P, a, l)
    A, F = assemble(a, l)
    u = pde_solve(A, F)
    x = pcg(A, F, P)
    @test x !== nothing
    x === nothing || @test relerr(x, u) <= 1e-8
    return nothing
end

# `p(D⁻¹A) D⁻¹ x` from the eigendecomposition of `S = D^{-1/2} A D^{-1/2}`, with `p` the
# Chebyshev polynomial whose residual `1 - λ p(λ)` is `T_k((θ - λ) / δ) / T_k(θ / δ)`.
function chebyshev_oracle(A, x, λmax, ratio, degree)
    D = diag(A)
    s = 1 ./ sqrt.(D)
    λ, V = eigen(Symmetric(Matrix(s .* A .* s')))
    λmin = λmax / ratio
    θ, δ = (λmax + λmin) / 2, (λmax - λmin) / 2
    T(k, t) = abs(t) <= 1 ? cos(k * acos(t)) : cosh(k * acosh(abs(t))) * sign(t)^k
    p = [(1 - T(degree, (θ - μ) / δ) / T(degree, θ / δ)) / μ for μ in λ]
    return s .* (V * (p .* (V' * (s .* x))))
end

# Jacobi, Chebyshev, ILU(0) and AMG on the form `laplacian` over `C`: each against what
# the assembled matrix gives, and in a CG solve. `reaches` says that some ghost's closure
# row holds the degree of freedom of its own boundary node (a Robin relation's does; a
# Neumann ghost is its mirror node alone).
function check_preconditioners(C; dirichlet = nothing, reaches = true)
    a = form(C, C, laplacian)
    A = assemble(a; dirichlet)
    symmetric = dirichlet === nothing
    symmetric &&
        @test isposdef(Symmetric(Matrix(A))) && norm(A - A', Inf) <= 1e-12 * norm(A, Inf)
    x = rand(MersenneTwister(7), size(A, 1))
    l = source(C)

    # The reduced diagonal differs from the padded diagonal reduced by `E`: entry `j` takes
    # the padded entries between a node and the ghosts whose closure reaches it.
    E, _ = closure(C)
    Ap = assemble(form(padded_space(C), padded_space(C), laplacian))
    naive = vec(sum(abs2.(Matrix(E)) .* diag(Ap); dims = 1))
    symmetric && reaches &&
        @test maximum(abs, naive - diag(A)) > 1e-3 * maximum(abs, diag(A))

    @testset "jacobi_preconditioner" begin
        P = jacobi_preconditioner(a; dirichlet)
        @test P isa Bramble.JacobiPreconditioner
        @test relerr(apply(P, x), x ./ diag(A)) <= 1e-12
        M = matrix_free_operator(a; dirichlet)
        fill!(M.xpad, 1e300)
        fill!(M.ypad, 1e300)
        @test relerr(apply(jacobi_preconditioner(M), x), x ./ diag(A)) <= 1e-12
        symmetric && converges(P, a, l)
    end

    # Degree 1 is Jacobi over `θ = (λmax + λmin) / 2`, whatever the symmetry of `A`; a
    # higher degree is checked against the oracle on a symmetric `A`.
    @testset "chebyshev_preconditioner" begin
        P = chebyshev_preconditioner(a; dirichlet, degree = 1, λmax = 2.0, ratio = 3)
        @test relerr(apply(P, x), (x ./ diag(A)) ./ (4 / 3)) <= 1e-12
        M = matrix_free_operator(a; dirichlet)
        P = chebyshev_preconditioner(M)
        @test P.op === M
        if symmetric
            S = Symmetric(Matrix(A ./ sqrt.(diag(A)) ./ sqrt.(diag(A))'))
            λmax = 1.05 * maximum(eigen(S).values)
            P = chebyshev_preconditioner(a; degree = 4, λmax, ratio = 20)
            fill!(P.op.xpad, 1e300)
            fill!(P.op.ypad, 1e300)
            @test relerr(apply(P, x), chebyshev_oracle(A, x, λmax, 20, 4)) <= 1e-10
            λ = maximum(eigen(Symmetric(Matrix(A))).values)
            @test 0.95λ <= Bramble.max_eigenvalue_estimate(M) <= 1.2λ
            converges(chebyshev_preconditioner(a), a, l)
        end
    end

    @testset "ilu_preconditioner" begin
        P = ilu_preconditioner(a; dirichlet)
        @test P isa ILUZero.ILU0Precon
        @test apply(P, x) == apply(ilu_preconditioner(A), x)
        symmetric && converges(P, a, l)
    end

    @testset "amg_preconditioner" begin
        P = aspreconditioner(amg_preconditioner(a; dirichlet))
        @test apply(P, x) == apply(aspreconditioner(amg_preconditioner(A)), x)
        symmetric && converges(P, a, l)
    end
    return nothing
end

# `fdm_solve`, the factorisation and its refill, and the preconditioner on the separable
# forms over `C`, against the assembled reduced matrix.
function check_fdm(C)
    for f in (stiffness, separable)
        a = form(C, C, f)
        A, F = assemble(a, source(C))
        u = A \ F
        @test relerr(fdm_solve(a, F), u) <= 1e-10
        x = similar(F)
        fac = fdm_factorize(a)
        @test relerr(fdm_solve!(x, fac, F), u) <= 1e-10
        @test relerr(apply(fdm_preconditioner(a), F), u) <= 1e-10
    end
    c = Ref(1.0)
    a = form(C, C, (u, v) -> innerₕ(u, v) + c * inner₊(∇ₕ(u), ∇ₕ(v)))
    fac = fdm_factorize(a)
    c[] = 7.0
    fdm_factorize!(fac, a)
    A, F = assemble(a, source(C))
    x = similar(F)
    @test relerr(fdm_solve!(x, fac, F), A \ F) <= 1e-10
    a = form(C, C, stiffness)
    converges(fdm_preconditioner(a), a, source(C))
    return nothing
end

# Every fdm entry point on `a` throws an `ArgumentError` whose message names `fix`.
function check_fdm_refuses(a, fix; kwargs...)
    F = ones(ndofs(Bramble.trial_space(a)))
    calls = (() -> fdm_solve(a, F; kwargs...), () -> fdm_factorize(a; kwargs...),
        () -> fdm_preconditioner(a; kwargs...))
    for call in calls
        err = try
            call()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        err isa ArgumentError && @test occursin(fix, err.msg)
    end
    return nothing
end

@testset "Ghost preconditioners" begin
    @test all(((_, s),) -> s in (:works, :refuses), SUPPORT)

    @testset "works: 1D $name" for (name, (ns, ghosts, rel)) in CASES_1D
        check_preconditioners(space(ns, ghosts, rel); reaches = name != "Neumann")
    end

    @testset "works: 2D $name" for (name, (ns, ghosts, rel)) in CASES_2D
        C = space(ns, ghosts, rel)
        check_preconditioners(C; reaches = name != "Neumann")
        @testset "fdm" begin
            check_fdm(C)
        end
    end

    # `dirichlet` as identity rows, on a space whose unknowns are its physical nodes.
    @testset "works: dirichlet rows" begin
        _, (ns, ghosts, rel) = CASES_2D[2]
        check_preconditioners(space(ns, ghosts, rel); dirichlet = (:ymin, :xmax))
    end

    @testset "refuses: fdm_* without separable factors" begin
        conormal = u -> (:xmin => ∂ₙ(u; K) => 1.0, :xmax => u + ∂ₙ(u; K) => 0.0,
            :ymin => ∂ₙ(u; K) => 0.0, :ymax => 2.0 * u + ∂ₙ(u; K) => 1.0)
        C = space((13, 11), :boundary, conormal)
        a = form(C, C, stiffness)
        check_fdm_refuses(a, "assemble(a)")
        _, (ns, ghosts, rel) = CASES_2D[2]
        C2 = space(ns, ghosts, rel)
        fac = fdm_factorize(form(C2, C2, stiffness))
        @test_throws ArgumentError fdm_factorize!(fac, a)
        _, (ns, ghosts, rel) = CASES_2D[3]
        C = space(ns, ghosts, rel)
        check_fdm_refuses(form(C, C, laplacian), "assemble(a)")
    end

    @testset "refuses: fdm_* on a 1D form" begin
        _, (ns, ghosts, rel) = CASES_1D[2]
        C = space(ns, ghosts, rel)
        check_fdm_refuses(form(C, C, stiffness), "assemble(a)")
    end

    @testset "refuses: fdm_* with dirichlet" begin
        _, (ns, ghosts, rel) = CASES_2D[2]
        C = space(ns, ghosts, rel)
        check_fdm_refuses(form(C, C, stiffness), "u => g"; dirichlet = :boundary)
    end

    WITH_3D_TESTS && @testset "works: 3D $name" for (name, (ns, ghosts, rel)) in CASES_3D
        C = space(ns, ghosts, rel)
        check_preconditioners(C)
        check_fdm(C)
    end
end

end
