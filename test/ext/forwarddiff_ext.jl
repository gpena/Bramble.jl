module TestForwardDiffExt

using Test
using Bramble
using Bramble: sparspak_factorize, sparse_factorize
using LinearAlgebra
using Random
using SparseArrays
using Sparspak
using SuiteSparse
using MUMPS
using DifferentiationInterface: DifferentiationInterface, AutoForwardDiff
using ForwardDiff: ForwardDiff
using ..TestUtils: alloc_test, alloc_ok

if Sys.isapple()
    using AppleAccelerate
end

# A stand-in for a native factor: its finalizer counts the releases.
mutable struct ReleaseProbe end
const RELEASED = Ref(0)
function release_probe()
    p = ReleaseProbe()
    finalizer(_ -> RELEASED[] += 1, p)
    return p
end

# The `ForwardDiff.Dual` route of `pde_solve` (`BrambleForwardDiffExt`): the `Float64`
# values of the matrix are factored once, by the route the `Float64` solve takes, and each
# partial is one solve against that factor. The oracles are the direct Sparspak factor of
# the `Dual` matrix (the route `pde_solve` took before), the dense solve, and a central
# difference of a `Float64` derivative, which uses no `Dual` arithmetic at all.
@testset "ForwardDiff extension" begin
    m = 30
    A0 = spdiagm(0 => fill(4.0, m), 1 => fill(-1.0, m - 1), -1 => fill(-1.2, m - 1))
    D = ForwardDiff.Dual{:t}
    Ad = SparseMatrixCSC(A0.m, A0.n, A0.colptr, A0.rowval, D.(A0.nzval, 0.5 .* A0.nzval))
    b = ones(m)
    x0 = A0 \ b
    p1 = A0 \ (-(0.5 .* A0) * x0)

    @testset "every solver, Dual matrix" begin
        solvers = [:default, :suitesparse, :spqr, :sparspak]
        Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing &&
            push!(solvers, :accelerate)
        Base.get_extension(Bramble, :BrambleMUMPSExt) !== nothing && push!(solvers, :mumps)
        for solver in solvers
            x = pde_solve(Ad, b; solver = solver)
            @test ForwardDiff.value.(x) ≈ x0 rtol = 1.0e-12
            @test ForwardDiff.partials.(x, 1) ≈ p1 rtol = 1.0e-12
        end
    end

    @testset "matches the direct Sparspak factor" begin
        x = pde_solve(Ad, b; solver = :sparspak)
        old = sparspak_factorize(Ad) \ b
        @test ForwardDiff.value.(x) ≈ ForwardDiff.value.(old) rtol = 1.0e-12
        @test ForwardDiff.partials.(x, 1) ≈ ForwardDiff.partials.(old, 1) rtol = 1.0e-10
    end

    # A symmetric positive definite matrix is the case `:default` sends to Accelerate on
    # macOS, and to `factorize`'s Cholesky elsewhere.
    @testset "symmetric matrix under :default" begin
        S0 = spdiagm(0 => fill(4.0, m), 1 => fill(-1.0, m - 1), -1 => fill(-1.0, m - 1))
        Sd = SparseMatrixCSC(S0.m, S0.n, S0.colptr, S0.rowval, D.(S0.nzval, 0.5 .* S0.nzval))
        xs0 = S0 \ b
        x = pde_solve(Sd, b)
        @test ForwardDiff.value.(x) ≈ xs0 rtol = 1.0e-12
        @test ForwardDiff.partials.(x, 1) ≈ S0 \ (-(0.5 .* S0) * xs0) rtol = 1.0e-12
    end

    @testset "real matrix with a Dual load" begin
        bd = D.(b, collect(1.0:m))
        for solver in (:default, :suitesparse, :sparspak)
            xb = pde_solve(A0, bd; solver = solver)
            @test ForwardDiff.value.(xb) ≈ x0 rtol = 1.0e-12
            @test ForwardDiff.partials.(xb, 1) ≈ A0 \ collect(1.0:m) rtol = 1.0e-12
        end
    end

    # A kept factorization (`sparse_factorize`) on every backend `sparse_factorize` takes.
    # The oracles are `pde_solve(A, F)` of the same `Dual` system and the dense solve.
    fact_solvers = [:default, :suitesparse, :sparspak]
    Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing &&
        push!(fact_solvers, :accelerate)
    Base.get_extension(Bramble, :BrambleMUMPSExt) !== nothing && push!(fact_solvers, :mumps)
    Wm = gridspace(mesh(domain(interval(0.0, 1.0)), m, true))

    # A real factor and a `Dual` load: the solve is linear in the load, so the value and the
    # partial are each solved against the factor, by `pde_solve` and by `ldiv!` alike.
    @testset "kept real factor, Dual load" begin
        bd = D.(b, collect(1.0:m))
        for solver in fact_solvers
            fact = sparse_factorize(A0; solver = solver)
            for x in (pde_solve(fact, bd), parent(ldiv!(element(Wm, eltype(bd)), fact, bd)))
                @test ForwardDiff.value.(x) ≈ x0 rtol = 1.0e-12
                @test ForwardDiff.partials.(x, 1) ≈ A0 \ collect(1.0:m) rtol = 1.0e-12
            end
        end
        # A factor of the `Dual` matrix itself carries the load's tag, so the load is not
        # split: its own `\` is the oracle.
        fd = sparspak_factorize(Ad)
        ref = fd \ bd
        for x in (pde_solve(fd, bd), parent(ldiv!(element(Wm, eltype(bd)), fd, bd)))
            @test ForwardDiff.value.(x) ≈ ForwardDiff.value.(ref) rtol = 1.0e-12
            @test ForwardDiff.partials.(x, 1) ≈ ForwardDiff.partials.(ref, 1) rtol = 1.0e-12
        end
    end

    # A factor over `Dual` that is not `sparse_factorize`'s (Sparspak's own, stdlib `lu`)
    # with a load of another tag, in both nesting orders: the load is solved whole, so the
    # derivative matches the nested derivative of `pde_solve(A, F)`.
    @testset "factor of a Dual matrix, load of another tag" begin
        n = 6
        Am(p) = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n) .+ p .* (1:n),
            1 => fill(-1.0, n - 1))
        Fv(p) = (1 .+ p .* (1:n)) ./ n
        inner_A(s) = ForwardDiff.derivative(
            p1 -> ForwardDiff.derivative(p2 -> sum(abs2, s(Am(p2), Fv(p1))), 0.3), 0.7)
        outer_A(s) = ForwardDiff.derivative(
            p2 -> ForwardDiff.derivative(p1 -> sum(abs2, s(Am(p2), Fv(p1))), 0.7), 0.3)
        for d in (inner_A, outer_A)
            ref = d((A, F) -> pde_solve(A, F))
            @test d((A, F) -> pde_solve(sparspak_factorize(A), F)) ≈ ref rtol = 1.0e-10
            @test d((A, F) -> pde_solve(lu(Matrix(A)), F)) ≈ ref rtol = 1.0e-10
            @test d((A, F) -> pde_solve(sparse_factorize(A), F)) ≈ ref rtol = 1.0e-10
        end
    end

    # A `Dual` matrix factored once and kept: every solve against it, by `pde_solve`, `\`
    # and `ldiv!`, is the derivative of `pde_solve(A, F)`. Two solves against one factor
    # catch a factor released after its first solve (MUMPS frees a native one).
    @testset "kept factor of a Dual matrix" begin
        bd = D.(b, collect(1.0:m))
        xd = pde_solve(Ad, bd)
        for solver in fact_solvers
            fact = sparse_factorize(Ad; solver = solver)
            @test fact isa Factorization
            @test size(fact) == size(Ad)
            for _ in 1:2
                for x in (pde_solve(fact, b), fact \ b, parent(ldiv!(element(Wm, D), fact, b)))
                    @test ForwardDiff.value.(x) ≈ x0 rtol = 1.0e-12
                    @test ForwardDiff.partials.(x, 1) ≈ p1 rtol = 1.0e-12
                end
                x = pde_solve(fact, bd)
                @test ForwardDiff.value.(x) ≈ ForwardDiff.value.(xd) rtol = 1.0e-12
                @test ForwardDiff.partials.(x, 1) ≈ ForwardDiff.partials.(xd, 1) rtol = 1.0e-12
            end
        end
        rect = SparseMatrixCSC(3, 2, [1, 3, 5], [1, 2, 2, 3], D.([1.0, 0.5, 2.0, 1.0], 1.0))
        @test_throws ArgumentError sparse_factorize(rect)
        @test_throws ArgumentError sparse_factorize(Ad; solver = :bogus)
        @test_throws ArgumentError Bramble.refactor!(sparse_factorize(Ad), Ad)
    end

    # The calls of the subplan's probe, on a 1D form whose coefficient and load scale are the
    # parameters: a factor of the form or of its matrix, and of the matrix at the fixed
    # parameter (only the load active). The oracles are the `Dual` solve `pde_solve(A, F)`
    # and a central difference, on a uniform and a non-uniform mesh.
    @testset "kept factor, form problem, uniform: $unif" for unif in (true, false)
        Random.seed!(1234)
        W = gridspace(mesh(domain(interval(0.0, 1.0)), 17, unif))
        function bf(p)
            κ = element(W, eltype(p))
            parent(κ) .= 2.0 .+ p[1]
            return form(W, W, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V)) + innerₕ(U, V))
        end
        lf = form(W, v -> innerₕ(x -> 1.0 + x[1], v))
        function sys(p)
            A, F = assemble(bf(p), lf; dirichlet = :boundary => x -> 0.0)
            return A, (1 + p[2]) .* F
        end
        rel(g, r) = maximum(abs, g .- r) / maximum(abs, r)
        cdg(f, p) = [(f(setindex!(copy(p), p[k] + 1.0e-5, k)) -
                      f(setindex!(copy(p), p[k] - 1.0e-5, k))) / 2.0e-5 for k in eachindex(p)]
        function by_ldiv(fact, F, T)
            u = element(W, T)
            ldiv!(u, fact, F)
            return sum(abs2, parent(u))
        end
        p = [0.5, 0.25]
        A0p = first(sys(p))
        gA = ForwardDiff.gradient(q -> sum(abs2, pde_solve(sys(q)...)), p)
        gF = ForwardDiff.gradient(q -> sum(abs2, pde_solve(A0p, last(sys(q)))), p)
        @test minimum(abs, gA) > 1.0e-6 * maximum(abs, gA)
        calls = (
            (q -> ((A, F) = sys(q); sum(abs2, pde_solve(sparse_factorize(A), F))), gA),
            (q -> sum(abs2, pde_solve(sparse_factorize(bf(q); dirichlet = :boundary),
                    last(sys(q)))), gA),
            (q -> ((A, F) = sys(q); by_ldiv(sparse_factorize(A), F, eltype(q))), gA),
            (q -> sum(abs2, pde_solve(sparse_factorize(A0p), last(sys(q)))), gF),
            (q -> by_ldiv(sparse_factorize(A0p), last(sys(q)), eltype(q)), gF)
        )
        for (f, g) in calls
            d = ForwardDiff.gradient(f, p)
            @test rel(d, g) <= 1.0e-8
            @test rel(d, cdg(f, p)) <= 1.0e-6
        end
    end

    # The `Float64` route is unchanged by the load split: the same result type, and an
    # `ldiv!` into a `VectorElement` allocates what the backend's own `ldiv!` does, nothing
    # for SuiteSparse and Accelerate.
    @testset "Float64 kept factor unchanged" begin
        for solver in fact_solvers
            fact = sparse_factorize(A0; solver = solver)
            @test @inferred(pde_solve(fact, b)) ≈ x0 rtol = 1.0e-12
            u = element(Wm)
            @test @inferred(ldiv!(u, fact, b)) === u
            @test parent(u) ≈ x0 rtol = 1.0e-12
            bytes = alloc_test(ldiv!, u, fact, b)
            @test bytes == alloc_test(ldiv!, parent(u), fact, b)
            solver in (:default, :suitesparse, :accelerate) && @test alloc_ok(bytes)
        end
    end

    # The second derivative of a nested tag: the outer tag `s` perturbs the matrix by 0.1 A0,
    # the inner `t` scales it by 1 + t / 2. The mixed partial comes from one factor of the
    # value, and the Sparspak factor of the nested matrix is the oracle.
    @testset "nested tags recurse over one factor" begin
        E = ForwardDiff.Dual{:s}
        An = SparseMatrixCSC(A0.m, A0.n, A0.colptr, A0.rowval,
            E.(Ad.nzval, D.(0.1 .* A0.nzval, 0.0)))
        xn = pde_solve(An, b)
        xr = sparspak_factorize(An) \ b
        mixed = ForwardDiff.partials.(ForwardDiff.partials.(xn, 1), 1)
        @test mixed ≈ ForwardDiff.partials.(ForwardDiff.partials.(xr, 1), 1) rtol = 1.0e-10
        @test ForwardDiff.value.(ForwardDiff.partials.(xn, 1)) ≈
              ForwardDiff.value.(ForwardDiff.partials.(xr, 1)) rtol = 1.0e-10

        # The first derivative in `s` in closed form, `-A(t)⁻¹ 0.1 A0 x(t)`, with no `Dual`
        # arithmetic: its central difference in `t` is the mixed partial.
        ds(τ) = (A0 .* (1 + 0.5τ)) \ (-(0.1 .* A0) * ((A0 .* (1 + 0.5τ)) \ b))
        h = 1.0e-5
        @test ForwardDiff.value.(mixed) ≈ (ds(h) - ds(-h)) / (2h) rtol = 1.0e-6
    end

    # The same nested tag through a kept factor: of the nested matrix, whose solve recurses
    # over the factor of its innermost values, and of the real matrix with a nested load,
    # whose split recurses over the parts of each level.
    @testset "nested tags through a kept factor" begin
        E = ForwardDiff.Dual{:s}
        An = SparseMatrixCSC(A0.m, A0.n, A0.colptr, A0.rowval,
            E.(Ad.nzval, D.(0.1 .* A0.nzval, 0.0)))
        xr = sparspak_factorize(An) \ b
        for solver in fact_solvers
            xn = pde_solve(sparse_factorize(An; solver = solver), b)
            @test ForwardDiff.partials.(ForwardDiff.partials.(xn, 1), 1) ≈
                  ForwardDiff.partials.(ForwardDiff.partials.(xr, 1), 1) rtol = 1.0e-10
        end
        bn = E.(D.(b, collect(1.0:m)), D.(sin.(1:m), cos.(1:m)))
        xb = pde_solve(A0, bn)
        for solver in fact_solvers
            fact = sparse_factorize(A0; solver = solver)
            for x in (pde_solve(fact, bn), parent(ldiv!(element(Wm, eltype(bn)), fact, bn)))
                @test ForwardDiff.partials.(ForwardDiff.partials.(x, 1), 1) ≈
                      ForwardDiff.partials.(ForwardDiff.partials.(xb, 1), 1) rtol = 1.0e-12
                @test ForwardDiff.value.(ForwardDiff.partials.(x, 1)) ≈ A0 \ sin.(1:m) rtol = 1.0e-12
            end
        end
    end

    # The factor is shared: a Dual with N partials costs N solves and one factorization, not
    # N factorizations. A factorization of this matrix allocates about 1.8 MB and a partial
    # about 0.5 MB, so the bound of half a factorization per partial fails a refactor on
    # every partial.
    @testset "one factorization for every partial" begin
        n = 3000
        T0 = spdiagm(0 => fill(4.0, n), 1 => fill(-1.0, n - 1), -1 => fill(-1.0, n - 1))
        dual(N) = SparseMatrixCSC(T0.m, T0.n, T0.colptr, T0.rowval,
            [ForwardDiff.Dual{Nothing, Float64, N}(v, ForwardDiff.Partials(ntuple(
                 k -> k * v, N))) for v in T0.nzval])
        rhs = ones(n)
        solve(A) = pde_solve(A, rhs; solver = :sparspak)
        per_partial = (alloc_test(solve, dual(8)) - alloc_test(solve, dual(1))) / 7
        factor = alloc_test(sparspak_factorize, T0)
        @test per_partial < factor / 2

        # A kept factor of a `Dual` matrix factors once whatever the partial count, and its
        # solves and those of a `Dual` load against a real factor refactor nothing.
        kept(A) = sparse_factorize(A; solver = :sparspak)
        @test (alloc_test(kept, dual(8)) - alloc_test(kept, dual(1))) / 7 < factor / 2
        kept_solve(f) = pde_solve(f, rhs)
        @test (alloc_test(kept_solve, kept(dual(8))) - alloc_test(kept_solve, kept(dual(1)))) / 7 <
              factor / 2
        f0 = kept(T0)
        load(N) = [ForwardDiff.Dual{Nothing, Float64, N}(1.0, ForwardDiff.Partials(ntuple(
                       k -> k / n, N))) for _ in 1:n]
        load_solve(F) = pde_solve(f0, F)
        @test (alloc_test(load_solve, load(8)) - alloc_test(load_solve, load(1))) / 7 < factor / 2
    end

    # A rank-deficient `A` under `:spqr` is a least-squares solve whose value is not smooth in
    # the parameter: for `[1 0; t 0]` and `F = [0, 1]` the solution is `x₁ = t / (1 + t²)`,
    # with derivative 1 at `t = 0`, and the forward formula gives 0. A `Dual` matrix of rank
    # below its size is refused, naming the rank. A real rank-deficient matrix with a `Dual`
    # load is linear in the load, so it stays allowed.
    @testset "rank-deficient spqr refuses" begin
        At = SparseMatrixCSC(2, 2, [1, 3, 3], [1, 2], [D(1.0, 0.0), D(0.0, 1.0)])
        err = try
            pde_solve(At, [0.0, 1.0]; solver = :spqr)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("rank 1", sprint(showerror, err))
        Ar = SparseMatrixCSC(2, 2, [1, 2, 2], [1], [1.0])
        @test pde_solve(Ar, D.([0.0, 1.0], [1.0, 0.0]); solver = :spqr) isa Vector
    end

    # A `Float32` matrix with a `Float64`-partials load: the primal factor is built at the
    # common scalar type, `Float64`, so each partial is a solve against it. A factor left at
    # `Float32` would be refactored at `Float64` for every partial. The oracle is the dense
    # solve of the same matrix in `Float64`.
    @testset "Float32 matrix, Float64 Dual load" begin
        A32 = SparseMatrixCSC{Float32}(A0)
        A64 = Float64.(A32)
        x = pde_solve(A32, D.(b, collect(1.0:m)); solver = :sparspak)
        @test ForwardDiff.value.(x) ≈ A64 \ b rtol = 1.0e-12
        @test ForwardDiff.partials.(x, 1) ≈ A64 \ collect(1.0:m) rtol = 1.0e-12

        n = 3000
        T0 = spdiagm(0 => fill(4.0f0, n), 1 => fill(-1.0f0, n - 1), -1 => fill(-1.0f0, n - 1))
        dual(N) = [ForwardDiff.Dual{Nothing, Float64, N}(1.0, ForwardDiff.Partials(
                       ntuple(k -> float(k), N))) for _ in 1:n]
        solve(rhs) = pde_solve(T0, rhs; solver = :sparspak)
        per_partial = (alloc_test(solve, dual(8)) - alloc_test(solve, dual(1))) / 7
        factor = alloc_test(sparspak_factorize, Float64.(T0))
        @test per_partial < factor / 2
    end

    # The primal factor is released once the solve is done, as `_mumps_solve` releases its
    # native factor, and also when the solve throws: otherwise native MUMPS factors outlive
    # the gradient call. MUMPS is in the test environment, but a finalizer on a probe object
    # tests the release without a MUMPS factor: `_run_and_finalize` is the helper that
    # `BrambleForwardDiffExt` runs its work through.
    @testset "primal factor released after the solve" begin
        ext = Base.get_extension(Bramble, :BrambleForwardDiffExt)
        before = RELEASED[]
        @test ext._run_and_finalize(_ -> 42, release_probe()) == 42
        @test RELEASED[] == before + 1
        @test_throws ErrorException ext._run_and_finalize(
            _ -> error("solve failed"), release_probe())
        @test RELEASED[] == before + 2
    end

    # The bordered solve (`mean_zero`, `gauge`) of a `Dual` matrix and load: the bordered
    # matrix of the values is factored once and each partial solves its bordered right-hand
    # side, with `w` and `Z` constants. The oracle is ForwardDiff through the dense bordered
    # matrix, which factors a `Dual` matrix with generic LU and shares no code with the sparse
    # route. The parameters are `p = (t, s)`: the matrix is `A₀ + t A₁` (`A₁` keeps the kernel
    # of `A₀`, so the border stays valid) and the load `F₀ + s F₁`.
    @testset "bordered solve, Dual matrix and load" begin
        bordered_oracle(A0, A1, F0, F1, C, p) = begin
            Cm = Matrix(reshape(C, size(A0, 1), :))
            k = size(Cm, 2)
            B = [Matrix(A0) + p[1] * Matrix(A1) Cm; transpose(Cm) zeros(k, k)]
            return (B \ [F0 + p[2] * F1; zeros(k)])[1:size(A0, 1)]
        end
        sparse_route(A0, A1, F0, F1, p; kwargs...) = pde_solve(A0 + p[1] * A1, F0 + p[2] * F1; kwargs...)

        # pure Neumann: the Laplacian of a conductance `c` has the constants in its kernel
        lap(c) = spdiagm(0 => [c[1]; c[1:(end - 1)] .+ c[2:end]; c[end]], 1 => -c, -1 => -c)
        c0 = 1.0 .+ 0.5 .* sin.(1:(m - 1))
        c1 = 0.3 .+ 0.2 .* cos.(1:(m - 1))
        L0, L1 = lap(c0), lap(c1)
        F0 = cos.(1:m)
        F1 = sin.(2 .* (1:m))
        w = collect(1.0 .+ 0.1 .* (1:m))
        p = [0.4, 1.1]
        solvers = [:default, :suitesparse, :sparspak]
        Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing &&
            push!(solvers, :accelerate)
        Base.get_extension(Bramble, :BrambleMUMPSExt) !== nothing && push!(solvers, :mumps)
        expected = ForwardDiff.jacobian(q -> bordered_oracle(L0, L1, F0, F1, w, q), p)
        for solver in solvers
            J = ForwardDiff.jacobian(
                q -> sparse_route(L0, L1, F0, F1, q; mean_zero = w, solver = solver), p)
            @test J ≈ expected rtol = 1.0e-9
        end
        # a Dual matrix with a `Float64` load, and a Dual load with a `Float64` matrix
        Jt = ForwardDiff.jacobian(
            q -> pde_solve(L0 + q[1] * L1, F0 + p[2] * F1; mean_zero = w), [p[1]])
        @test Jt ≈ expected[:, 1:1] rtol = 1.0e-9
        Js = ForwardDiff.jacobian(q -> pde_solve(L0 + p[1] * L1, F0 + q[1] * F1; mean_zero = w),
            [p[2]])
        @test Js ≈ expected[:, 2:2] rtol = 1.0e-9

        # a ghost-only kernel: `gauge`, then `mean_zero` and `gauge` together
        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (6, 7), (true, true))
        Wg = gridspace(Ω; ghosts = :boundary)
        Ag0 = assemble(form(Wg, Wg, (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)))
        Ag1 = assemble(form(Wg, Wg, (u, v) -> innerₕ(u, v)))
        Zg = Bramble.ghost_kernel(Ag0, Wg)
        ng = size(Ag0, 1)
        G0 = cos.(1:ng)
        G1 = sin.(2 .* (1:ng))
        gexpected = ForwardDiff.jacobian(q -> bordered_oracle(Ag0, Ag1, G0, G1, Zg, q), p)
        for solver in (:default, :suitesparse)
            J = ForwardDiff.jacobian(
                q -> sparse_route(Ag0, Ag1, G0, G1, q; gauge = Zg, solver = solver), p)
            @test J ≈ gexpected rtol = 1.0e-5
        end

        Cn = constrained(Wg, u -> (:xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0))
        Bn0 = assemble(form(Cn, Cn, (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + inner₊(∇ₕ(u), ∇ₕ(v))))
        Bn1 = assemble(form(Cn, Cn, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))))
        wn = assemble(form(Cn, v -> innerₕ(x -> 1.0, v)))
        Zn = Bramble.ghost_kernel(Bn0, Cn)
        nn = size(Bn0, 1)
        H0 = cos.(1:nn)
        H1 = sin.(2 .* (1:nn))
        nexpected = ForwardDiff.jacobian(
            q -> bordered_oracle(Bn0, Bn1, H0, H1, [wn Zn], q), p)
        for solver in (:default, :suitesparse)
            J = ForwardDiff.jacobian(
                q -> sparse_route(
                    Bn0, Bn1, H0, H1, q; mean_zero = wn, gauge = Zn, solver = solver), p)
            @test J ≈ nexpected rtol = 1.0e-5
        end

        # the rank-one term of a `RankOneUpdate` is a constant: a real one takes the bordered
        # route with a `Dual` load (over a `Dual` matrix `σ` is `Dual`, refused below)
        ws, wt = collect(1.0 .+ 0.1 .* (1:m)), cos.(1:m)
        σ = 0.7
        K = Bramble.RankOneUpdate(A0, σ, ws, wt)
        dK = ForwardDiff.derivative(t -> pde_solve(K, b .+ t .* F1), 0.4)
        @test dK ≈ (Matrix(A0) + σ * ws * transpose(wt)) \ F1 rtol = 1.0e-9
    end

    # The bordered primal system is factored once: a Dual with N partials costs N solves.
    # Factoring the bordered matrix of this ghost space allocates about 3 MB and a partial
    # about 0.5 MB, so the bound of half a factorization per partial fails a refactor on
    # every partial.
    @testset "bordered solve, one factorization" begin
        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (30, 30), (true, true))
        Wg = gridspace(Ω; ghosts = :boundary)
        Ag = assemble(form(Wg, Wg, (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)))
        A1 = assemble(form(Wg, Wg, (u, v) -> innerₕ(u, v)))
        Zg = Bramble.ghost_kernel(Ag, Wg)
        k = size(Zg, 2)
        dual(N) = Ag + ForwardDiff.Dual{Nothing, Float64, N}(
            0.0, ForwardDiff.Partials(ntuple(float, N))) * A1
        rhs = ones(size(Ag, 1))
        solve(A) = pde_solve(A, rhs; gauge = Zg)
        per_partial = (alloc_test(solve, dual(8)) - alloc_test(solve, dual(1))) / 7
        factor = alloc_test(lu, [Ag Zg; transpose(Zg) zeros(k, k)])
        @test per_partial < factor / 2
    end

    @testset "bordered solve refuses a Dual border" begin
        w = ones(m)
        Z = reshape(ones(m), m, 1)
        wd = D.(w, w)
        @test_throws ArgumentError pde_solve(Ad, b; mean_zero = wd)
        @test_throws ArgumentError pde_solve(A0, b; mean_zero = wd)
        @test_throws ArgumentError pde_solve(Ad, b; gauge = D.(Z, Z))
        err = try
            pde_solve(Ad, b; mean_zero = wd)
        catch e
            e
        end
        @test occursin("mean_zero", sprint(showerror, err))
        K = Bramble.RankOneUpdate(Ad, 0.7, w, w)
        @test_throws ArgumentError pde_solve(K, b)
        @test occursin("RankOneUpdate", sprint(showerror, try
            pde_solve(K, b)
        catch e
            e
        end))
        @test_throws ArgumentError pde_solve(Ad, b; mean_zero = w, solver = :bogus)
        @test_throws ArgumentError pde_solve(Ad, b; mean_zero = w, solver = :spqr)
        @test_throws DimensionMismatch pde_solve(Ad, b; mean_zero = w[1:(end - 1)])
    end

    @testset "the Dual route refuses bad input" begin
        rect = SparseMatrixCSC(3, 2, [1, 3, 5], [1, 2, 2, 3],
            D.([1.0, 0.5, 2.0, 1.0], [1.0, 0.0, 0.0, 1.0]))
        @test_throws ArgumentError pde_solve(rect, ones(3); solver = :spqr)
        @test_throws ArgumentError pde_solve(Ad, b; solver = :bogus)
        @test_throws ArgumentError pde_solve(Ad, b; solver = :sparspak, bogus = 1)
        @test_throws ArgumentError pde_solve(Ad, b; bogus = 1)
        # A singular Dual matrix reaches the primal factorization, which throws.
        sing = SparseMatrixCSC(2, 2, [1, 3, 5], [1, 2, 1, 2],
            D.([1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]))
        @test_throws SingularException pde_solve(sing, ones(2); solver = :sparspak)
    end
end

end # module
