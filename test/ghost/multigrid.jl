module GhostMultigridTests

# Geometric multigrid on ghost spaces (#781): `build(W)` returns a form on a constrained
# ghost space built on `mesh(W)`, and `gmg_preconditioner` and `gmg_solve` run on its
# degrees of freedom. Neumann, Robin, mixed Dirichlet and Robin, and pure Neumann, in
# 1D and 2D (3D behind `WITH_3D_TESTS`), on seeded non-uniform meshes: cycles to 1e-8 within
# a bracket and the same on every mesh size, the solution against `pde_solve`, the
# transfers against `interpolation_matrix`, the smoothers' diagonal against `assemble`, a
# symmetric V-cycle that preconditions CG, a weak Robin `α` or reaction left unbordered,
# warm cycles at 0 B, and the refusals naming their fix.

using Test
using Bramble
import Bramble
using Bramble: change_points!, v_cycle!, D₋ₓ, D₊ᵧ
using LinearAlgebra: diag, dot, ldiv!, norm
using LinearSolve: LinearProblem, KrylovJL_CG, solve
using Random: MersenneTwister

using ..TestUtils: WITH_3D_TESTS

# A mesh of `n` points per axis on [0, 1]^D, each interior point moved by up to `±0.3h`
# (seeded), so its cells stay within a bounded aspect ratio on every level.
function jittered(::Val{D}, n; seed = 7) where {D}
    rng = MersenneTwister(seed)
    h = 1 / (n - 1)
    axis() = (p = collect(range(0.0, 1.0; length = n));
        p[2:(end - 1)] .+= 0.3h .* (2 .* rand(rng, n - 2) .- 1); p)
    Ωd = domain(foldl(×, ntuple(_ -> interval(0.0, 1.0), Val(D))))
    Ω = mesh(Ωd, D == 1 ? n : ntuple(_ -> n, Val(D)), D == 1 ? true : ntuple(_ -> true, Val(D)))
    change_points!(Ω, D == 1 ? axis() : ntuple(_ -> axis(), Val(D)))
    return Ω
end

# A 1D mesh graded as `t^1.5`.
function graded(n)
    Ω = mesh(domain(interval(0.0, 1.0)), n, true)
    change_points!(Ω, range(0.0, 1.0; length = n) .^ 1.5)
    return Ω
end

facets(D) = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)[1:(2D)]
lower(F) = F in (:xmin, :ymin, :zmin)

# The four problem kinds on the facets `Fs`: relations, ghosted facets and the mass weight.
function kinds(Fs)
    return (
        neumann_robin = (u -> map(F -> F => (F in (:xmin, :xmax) ? ∂ₙ(u) : 2.0 * u + ∂ₙ(u)) => 0.0, Fs),
            :boundary, 1.0),
        robin = (u -> map(F -> F => u + ∂ₙ(u) => 0.0, Fs), :boundary, 0.0),
        dirichlet_robin = (u -> map(F -> F => (lower(F) ? u : 3.0 * u + ∂ₙ(u)) => 0.0, Fs),
            filter(!lower, Fs), 0.0),
        neumann = (u -> map(F -> F => ∂ₙ(u) => 0.0, Fs), :boundary, 0.0))
end

# `-Δₕ u` reads the ghosts at every facet node, so each relation's closure enters the
# matrix (`inner₊(∇ₕ u, ∇ₕ v)` with physical weights never reads a ghost).
builder(rels, ghosts, c) = W -> (C = constrained(gridspace(mesh(W); ghosts), rels);
    form(C, C, (u, v) -> innerₕ(-1.0 * Δₕ(u), v) + c * innerₕ(u, v)))

# The V-cycles from zero that bring `‖b - A x‖` to `1e-8 ‖b‖`, at most `cap`.
function cycles(P, A, b; cap = 40)
    x = zeros(length(b))
    k = 0
    while norm(b - A * x) > 1.0e-8 * norm(b) && k < cap
        v_cycle!(x, P, b)
        k += 1
    end
    return k
end

# The cycle counts of one problem kind over the meshes `meshes`, with `b = A x`.
function counts(kind, meshes; build = builder(kind...))
    return map(meshes) do Ω
        A = assemble(build(gridspace(Ω)))
        b = A * rand(MersenneTwister(1), size(A, 2))
        cycles(gmg_preconditioner(build, Ω), A, b)
    end
end

@testset "Ghost multigrid" begin
    @testset "Level-free cycle counts ($(D)D, $label)" for (D, label, meshes, lo, hi) in (
        (1, "graded", [graded(n) for n in (33, 129, 513)], 3, 7),
        (1, "jittered", [jittered(Val(1), n) for n in (33, 129, 513)], 3, 7),
        (2, "jittered", [jittered(Val(2), n) for n in (17, 33, 65)], 4, 9))
        for (name, kind) in pairs(kinds(facets(D)))
            ks = counts(kind, meshes)
            @test all(k -> lo <= k <= hi, ks)
            @test maximum(ks) - minimum(ks) <= 1
        end
    end

    # On meshes graded as `t^(1 + d/4)`, whose cell aspect ratio grows with n, point
    # smoothers slow down on every space: the ghost cycles stay within 3 of the plain
    # mass plus diffusion problem's on the same mesh.
    @testset "Graded 2D: no worse than plain" begin
        plainbuild = W -> form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
        for n in (17, 33, 65)
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n), (true, true))
            change_points!(Ω, ntuple(d -> range(0.0, 1.0; length = n) .^ (1 + 0.25d), 2))
            kp = only(counts((nothing, nothing, nothing), [Ω]; build = plainbuild))
            for (name, kind) in pairs(kinds(facets(2)))
                @test only(counts(kind, [Ω])) <= kp + 3
            end
        end
    end

    if WITH_3D_TESTS
        # 3D warm products allocate until X9, so no allocation is checked here.
        @testset "Level-free cycle counts (3D)" begin
            meshes = [jittered(Val(3), n) for n in (9, 17)]
            for (name, kind) in pairs(kinds(facets(3)))
                ks = counts(kind, meshes)
                @test all(k -> 3 <= k <= 12, ks)
                @test maximum(ks) - minimum(ks) <= 2
            end
        end
    end

    @testset "gmg_solve matches pde_solve ($(D)D)" for D in (1, 2)
        Ω = jittered(Val(D), D == 1 ? 65 : 33)
        g = x -> 1 + x[1]
        mixed = u -> map(F -> F => (lower(F) ? u : 3.0 * u + ∂ₙ(u)) => (lower(F) ? g : 0.5),
            facets(D))
        fixed = u -> map(F -> F => u => g, facets(D))
        # Dirichlet on every facet: the ghosted one also carries the relation for its ghosts.
        dirichlet = u -> (fixed(u)..., :xmax => ∂ₙ(u) => 0.0)
        for (rels, ghosts) in ((mixed, filter(!lower, facets(D))), (dirichlet, (:xmax,)))
            build = W -> (C = constrained(gridspace(mesh(W); ghosts), rels);
                form(C, C, (u, v) -> innerₕ(-1.0 * Δₕ(u), v)))
            a = build(gridspace(Ω))
            A, F = assemble(a, form(Bramble.trial_space(a), v -> innerₕ(x -> sum(x), v)))
            uₕ = gmg_solve(build, Ω, F; tol = 1.0e-12)
            @test norm(parent(uₕ) - pde_solve(A, F)) <= 1.0e-9 * norm(pde_solve(A, F))
        end
        # Pure Neumann: a compatible right side converges, to its solution plus a constant.
        rels, ghosts, c = kinds(facets(D)).neumann
        build = builder(rels, ghosts, c)
        A = assemble(build(gridspace(Ω)))
        x = rand(MersenneTwister(2), size(A, 2))
        uₕ = gmg_solve(build, Ω, A * x; tol = 1.0e-11)
        u = parent(uₕ)
        @test norm(A * u - A * x) <= 1.0e-11 * norm(A * x)
        @test norm((u .- sum(u) / length(u)) - (x .- sum(x) / length(x))) <= 1.0e-8 * norm(x)
    end

    @testset "Transfers are interpolation, transposed" begin
        Ω = jittered(Val(2), 17)
        rels, ghosts, c = kinds(facets(2)).dirichlet_robin
        P = gmg_preconditioner(builder(rels, ghosts, c), Ω)
        H = P.hierarchy
        @test P.transfer !== nothing
        for l in 2:length(H)
            # the nodes no Dirichlet relation fixes: neither coordinate is 0
            free(Ωl) = findall(vec([x > 0 && y > 0 for x in points(Ωl)[1], y in points(Ωl)[2]]))
            ff, fc = free(H[l]), free(H[l - 1])
            R = Matrix(Bramble.interpolation_matrix(gridspace(H[l]), gridspace(H[l - 1])))[ff, fc]
            xc, xf = rand(length(fc)), rand(length(ff))
            @test Bramble._gmg_prolongate!(P.transfer, zeros(length(ff)), H, l, xc) ≈ R * xc
            @test Bramble._gmg_restrict!(P.transfer, zeros(length(fc)), H, l, xf) ≈ R' * xf
        end
        # A plain hierarchy keeps the transfers of the grid.
        @test gmg_preconditioner(W -> form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)),
            Ω).transfer === nothing
    end

    @testset "Smoothers read the reduced diagonal" begin
        Ω = jittered(Val(2), 9)
        K = [2.0 0.5; 0.5 1.0]
        rels = u -> (:xmin => ∂ₙ(u; K) => 0.0, :xmax => u + ∂ₙ(u; K) => 0.0,
            :ymin => 2.0 * u + ∂ₙ(u; K) => 0.0, :ymax => ∂ₙ(u; K) => 0.0)
        C = constrained(gridspace(Ω; ghosts = :boundary), rels)
        # the mixed difference reads the corner ghosts
        a = form(C, C, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(D₊ᵧ(u)), v) + innerₕ(u, v))
        op = matrix_free_operator(a)
        @test jacobi_smoother(op).inv_diagonal ≈ 1 ./ diag(Matrix(assemble(a)))
        @test chebyshev_smoother(op).inv_diagonal ≈ 1 ./ diag(Matrix(assemble(a)))
        @test_throws ArgumentError red_black_gauss_seidel(op)
        e = try
            red_black_gauss_seidel(op)
        catch err
            err
        end
        @test occursin("chebyshev_smoother", sprint(showerror, e))
    end

    # A V-cycle with ν₁ = ν₂ is symmetric when restriction is the transpose of
    # prolongation and the fixed nodes stay 0 in both, so CG can use it.
    @testset "Symmetric V-cycle, PCG ($(D)D)" for D in (1, 2)
        Ω = jittered(Val(D), D == 1 ? 65 : 33)
        for name in (:dirichlet_robin, :neumann)
            build = builder(kinds(facets(D))[name]...)
            A = assemble(build(gridspace(Ω)))
            P = gmg_preconditioner(build, Ω)
            @test P.smoothers[end].inv_diagonal ≈ 1 ./ diag(Matrix(A))
            @test (P.coarse isa Bramble._GMGBorderedLU) == (name === :neumann)
            x, y = rand(MersenneTwister(3), size(A, 2)), rand(MersenneTwister(4), size(A, 2))
            @test dot(P \ x, y) ≈ dot(x, P \ y) rtol = 1.0e-12
            b = A * rand(MersenneTwister(5), size(A, 2))
            sol = solve(LinearProblem(A, b), KrylovJL_CG(); Pl = P, reltol = 1.0e-11,
                abstol = 0.0, maxiters = 50)
            u = sol.u
            @test norm(A * u - b) <= 1.0e-10 * norm(b)
            name === :dirichlet_robin && @test u ≈ pde_solve(A, b) rtol = 1.0e-8
        end
    end

    # A weak Robin α or a small reaction leaves the coarsest matrix nonsingular: it is not
    # bordered, and the cycle converges to the unique solution 1.
    @testset "Weak Robin and reaction: unbordered" begin
        Ω = jittered(Val(2), 33)
        for (rels, c) in ((u -> map(F -> F => 1.0e-8 * u + ∂ₙ(u) => 0.0, facets(2)), 0.0),
            (u -> map(F -> F => ∂ₙ(u) => 0.0, facets(2)), 1.0e-7))
            build = builder(rels, :boundary, c)
            A = assemble(build(gridspace(Ω)))
            b = A * ones(size(A, 2))
            P = gmg_preconditioner(build, Ω)
            @test !(P.coarse isa Bramble._GMGBorderedLU)
            x = zeros(length(b))
            for _ in 1:15
                v_cycle!(x, P, b)
            end
            @test norm(b - A * x) <= 1.0e-5 * norm(b)
        end
    end

    @testset "Warm cycles allocate 0 B ($(D)D)" for D in (1, 2)
        Ω = jittered(Val(D), D == 1 ? 65 : 17)
        rels, ghosts, c = kinds(facets(D)).dirichlet_robin
        P = gmg_preconditioner(builder(rels, ghosts, c), Ω)
        n = first(size(P))
        x, b, y = zeros(n), rand(n), zeros(n)
        cyc() = v_cycle!(x, P, b)
        cyc()
        @test (@allocated cyc()) == 0
        div() = ldiv!(y, P, b)
        div()
        @test (@allocated div()) == 0
    end

    @testset "Refusals name the fix" begin
        Ω = jittered(Val(2), 17)
        f = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)
        rels = kinds(facets(2)).robin[1]
        message(build; kw...) = try
            gmg_preconditioner(build, Ω; kw...)
            ""
        catch err
            err isa ArgumentError ? sprint(showerror, err) : rethrow()
        end
        # ghosts left as unknowns
        kept = W -> (C = gridspace(mesh(W); ghosts = :boundary); form(C, C, f))
        @test_throws ArgumentError gmg_preconditioner(kept, Ω)
        @test occursin("constrained", message(kept))
        # a ghost form on the finest level only
        mixedkinds = W -> npoints(mesh(W)) == npoints(Ω) ?
            (C = constrained(gridspace(mesh(W); ghosts = :boundary), rels); form(C, C, f)) :
            form(W, W, f)
        @test occursin("same kind of form on every level", message(mixedkinds))
        # other ghosted facets below the finest level
        partial = u -> (:xmin => u + ∂ₙ(u) => 0.0, :xmax => u => 0.0, :ymin => u => 0.0,
            :ymax => u => 0.0)
        facetsdiffer = W -> npoints(mesh(W)) == npoints(Ω) ?
            (C = constrained(gridspace(mesh(W); ghosts = :boundary), rels); form(C, C, f)) :
            (C = constrained(gridspace(mesh(W); ghosts = (:xmin,)), partial); form(C, C, f))
        @test occursin("same ghosted facets", message(facetsdiffer))
        # the finest space captured instead of one built on mesh(W)
        Cfine = constrained(gridspace(Ω; ghosts = :boundary), rels)
        captured = W -> form(Cfine, Cfine, f)
        @test occursin("gridspace(mesh(W)", message(captured))
        # red-black Gauss-Seidel on a ghost level
        build = builder(rels, :boundary, 1.0)
        @test occursin("jacobi_smoother", message(build; smoother = red_black_gauss_seidel))
        # relation data in time read `E` alone, so the cycle runs on them
        timed = W -> (C = constrained(gridspace(mesh(W); ghosts = :boundary),
            u -> map(F -> F => u + ∂ₙ(u) => (x, t) -> t * x[1], facets(2)));
            form(C, C, f))
        A = assemble(timed(gridspace(Ω)))
        b = A * rand(MersenneTwister(3), size(A, 2))
        @test cycles(gmg_preconditioner(timed, Ω), A, b) <= 9
    end
end

end
