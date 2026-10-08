module TestKroneckerRefactorize

using Test
using Bramble
using Bramble: kronecker_operator, D₋ₓ
using LinearAlgebra: LinearAlgebra, Symmetric, UpperTriangular, eigen, norm
using Random
using SparseArrays: sparse
# `CpuPolyester` meshes below, as in test/ext/kronecker_ext.jl.
using Polyester
using Bramble: CpuPolyester, Serial

# An existing fast-diagonalisation factorisation recomputes its per-axis decompositions
# (`_fdm_decompose!`) in place from the dense mass and operator it keeps, through stored
# LAPACK workspaces: 0 bytes, and bitwise the factorisation a fresh build returns
# (gpena/Bramble.jl#474).
const KronExt = Base.get_extension(Bramble, :BrambleKroneckerExt)
@assert KronExt !== nothing "BrambleKroneckerExt did not load -- is Kronecker.jl a test dependency?"

const KRON_REFACTOR_SEED = 20261008

# Uniform, then moved to `t^(1 + d/4)` along axis `d`, so no two axes share their nodes.
function graded_space(::Type{T}, n::NTuple{D, Int}, p) where {T, D}
    Ω = mesh(domain(reduce(×, ntuple(_ -> interval(zero(T), one(T)), D))), n,
        ntuple(_ -> false, D); backend = backend(policy = p))
    x(d) = T.(range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d))
    Bramble.change_points!(Ω, ntuple(x, D))
    return gridspace(Ω)
end

symmetric_form(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
advection_form(u, v) = innerₕ(u, v) + 0.1 * inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(u), v)

# The refill's forms, each with a live `Ref` coefficient: symmetric; Schur; Schur with the
# `Ref` on the mass term, so `c_m` changes; and a negative multiple of the axis-1 mass
# (`-0.5 * innerₕ(D₋ᵧ(u), v)`), which the classification matches proportionally and
# represents negated.
const KRON_REFILL_C = Ref(2.5)
const KRON_REFILL_FORMS = (
    (u, v) -> innerₕ(u, v) + KRON_REFILL_C * inner₊(∇ₕ(u), ∇ₕ(v)),
    (u, v) -> innerₕ(u, v) + KRON_REFILL_C * inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(u), v),
    (u, v) -> KRON_REFILL_C * innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
              Bramble.inner₊ₓ(D₋ₓ(u), v),
    (u, v) -> -0.5 * innerₕ(Bramble.D₋ᵧ(u), v) + innerₕ(u, v) +
              KRON_REFILL_C * inner₊(∇ₕ(u), ∇ₕ(v)))

# The arrays `_fdm_decompose!` writes, so zeroing them first keeps a no-op from passing.
outputs(f::KronExt._FDMFactorization) = (f.Q..., f.Λ)
outputs(f::KronExt._SchurFactorization) = (f.Qc..., f.Zt..., f.S..., f.Tr..., f.Λ)
wipe!(f) = foreach(A -> fill!(A, zero(eltype(A))), outputs(f))
decompose_bytes(f) = (KronExt._fdm_decompose!(f); @allocated KronExt._fdm_decompose!(f))
refill_bytes(f, K) = (fdm_factorize!(f, K); @allocated fdm_factorize!(f, K))

function rhs(W, ::Type{T}, dirichlet) where {T}
    F = rand(MersenneTwister(KRON_REFACTOR_SEED), T, ndofs(W))
    dirichlet === :boundary && (F[Bramble._combined_mask(mesh(W), (:boundary,))] .= 0)
    return F
end

@testset "Kronecker refactorisation" begin
    @testset "_fdm_decompose!: 0 B, bitwise" begin
        routes = ((symmetric_form, KronExt._FDMFactorization),
            (advection_form, KronExt._SchurFactorization))
        for T in (Float64, Float32), p in (Serial(), CpuPolyester()),
            n in ((17, 13), (7, 6, 5)), (L, route) in routes, dir in (nothing, :boundary)
            W = graded_space(T, n, p)
            f = fdm_factorize(form(W, W, L); dirichlet = dir)
            @test f isa route
            F = rhs(W, T, dir)
            x0 = fdm_solve!(similar(F), f, F)
            saved = map(copy, outputs(f))
            wipe!(f)
            @test decompose_bytes(f) == 0
            @test all(map(==, outputs(f), saved))
            @test fdm_solve!(similar(F), f, F) == x0
        end
    end

    # The symmetric route is the call `eigen(Symmetric(A_d), Symmetric(M_d))` makes, through
    # the same LAPACK, so a fresh factorisation keeps its eigenpairs bit for bit.
    @testset "symmetric route equals eigen" begin
        for T in (Float64, Float32), dir in (nothing, :boundary)

            W = graded_space(T, (17, 13), Serial())
            f = fdm_factorize(form(W, W, symmetric_form); dirichlet = dir)
            for d in 1:2
                e = eigen(Symmetric(f.A[d]), Symmetric(f.M[d]))
                @test f.Q[d] == e.vectors
                @test f.lapack[d].w == e.values
            end
            c, w₁, w₂ = f.c_m[], f.lapack[1].w, f.lapack[2].w
            @test f.Λ == [c + w₁[i] + w₂[j] for i in eachindex(w₁), j in eachindex(w₂)]
        end
    end

    # `Q_d' A_d Z_d = S_d` and `Q_d' M_d Z_d = T_d`, both upper triangular, `Q_d`, `Z_d`
    # unitary: the generalised Schur form, recomputed from `M_d`, `A_d` alone.
    @testset "Schur route is a Schur form" begin
        for T in (Float64, Float32)
            W = graded_space(T, (17, 13), Serial())
            f = fdm_factorize(form(W, W, advection_form); dirichlet = :boundary)
            wipe!(f)
            KronExt._fdm_decompose!(f)
            tol = 100 * eps(T)
            for d in 1:2
                Q, Z = conj(f.Qc[d]), transpose(f.Zt[d])
                @test f.S[d] == UpperTriangular(f.S[d])
                @test f.Tr[d] == UpperTriangular(f.Tr[d])
                @test norm(Q' * Q - LinearAlgebra.I) <= tol * size(Q, 1)
                @test norm(Z' * Z - LinearAlgebra.I) <= tol * size(Z, 1)
                @test norm(Q' * f.A[d] * Z - f.S[d]) <= tol * norm(f.A[d]) * size(Q, 1)
                @test norm(Q' * f.M[d] * Z - f.Tr[d]) <= tol * norm(f.M[d]) * size(Q, 1)
            end
        end
    end

    # `fdm_factorize!(f, K)` after `change_points!` and a new `Ref` value: 0 B once warm,
    # and the solve bitwise that of a fresh factorisation of the new operator. Both changes
    # move every buffer, so a refill that wrote nothing would fail the bitwise test.
    @testset "fdm_factorize!(f, K): 0 B, bitwise" begin
        for T in (Float64, Float32), p in (Serial(), CpuPolyester()),
            n in ((17, 13), (7, 6, 5)), (k, L) in enumerate(KRON_REFILL_FORMS),
            dir in (nothing, :boundary)
            KRON_REFILL_C[] = 2.5
            W = graded_space(T, n, p)
            f = fdm_factorize(kronecker_operator(form(W, W, L)); dirichlet = dir)
            @test f isa (k == 1 ? KronExt._FDMFactorization : KronExt._SchurFactorization)
            x(d) = T.(range(0.0, 1.0; length = n[d]) .^ (1 + 0.4d))
            Bramble.change_points!(mesh(W), ntuple(x, length(n)))
            W2 = gridspace(mesh(W))
            a2 = form(W2, W2, L)
            KRON_REFILL_C[] = 0.7
            K2 = kronecker_operator(a2)
            @test fdm_factorize!(f, K2) === f
            @test refill_bytes(f, K2) == 0
            F = rhs(W2, T, dir)
            x2 = fdm_solve!(similar(F), f, F)
            g = fdm_factorize(K2; dirichlet = dir)
            @test x2 == fdm_solve!(similar(F), g, F)
            @test f.M == g.M && f.A == g.A && f.c_m[] == g.c_m[]
            T === Float64 && @test x2 ≈ assemble(a2; dirichlet = dir) \ F rtol = 1e-10
        end
        KRON_REFILL_C[] = 2.5
    end

    # The recipe of each form: which terms differ on which axis, and how the masses matched.
    @testset "fdm_factorize!: recorded recipe" begin
        W = graded_space(Float64, (17, 13), Serial())
        recipe(L) = fdm_factorize(kronecker_operator(form(W, W, L))).recipe
        r = map(recipe, KRON_REFILL_FORMS)
        @test map(q -> q.on, r) == ([0, 1, 2], [0, 1, 1, 2], [0, 1, 1, 2], [2, 0, 1, 2])
        @test map(q -> q.proportional, r) == (false, false, false, true)
        @test r[4].mass == (1, 2) && r[4].neg == (true, false)
        @test all(q -> !any(q.neg), r[1:3])
    end

    # A stale operator is refused as `fdm_solve(K, F)` refuses it, and an empty interior has
    # nothing to refill.
    @testset "fdm_factorize!: stale, empty" begin
        W = graded_space(Float64, (17, 13), Serial())
        K = kronecker_operator(form(W, W, symmetric_form))
        f = fdm_factorize(K)
        Bramble.change_points!(mesh(W), (range(0.0, 1.0; length = 17) .^ 2,
            range(0.0, 1.0; length = 13) .^ 3))
        @test_throws ArgumentError fdm_factorize!(f, K)
        @test_throws "change_points!" fdm_factorize!(f, K)
        V = graded_space(Float64, (2, 9), Serial())
        KV = kronecker_operator(form(V, V, symmetric_form))
        fV = fdm_factorize(KV; dirichlet = :boundary)
        @test fdm_factorize!(fV, KV) === fV
        @test refill_bytes(fV, KV) == 0
    end

    # A different structure is refused before anything is written, each case differing from
    # `f` in one respect only, so `f` still solves exactly as before.
    @testset "fdm_factorize!: structure refused" begin
        c = Ref(2.5)
        sym = (u, v) -> innerₕ(u, v) + c * inner₊(∇ₕ(u), ∇ₕ(v))
        adv = (u, v) -> sym(u, v) + innerₕ(D₋ₓ(u), v)
        W = graded_space(Float64, (17, 13), Serial())
        K(Wₕ, L) = kronecker_operator(form(Wₕ, Wₕ, L))
        fs = fdm_factorize(K(W, sym))
        fa = fdm_factorize(K(W, adv); dirichlet = :boundary)
        # The advection term's coefficient is zero at build, so it is dropped and `f0` is
        # symmetric; nonzero, it needs the Schur route.
        c3 = Ref(0.0)
        L3 = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + c3 * innerₕ(D₋ₓ(u), v)
        f0 = fdm_factorize(K(W, L3); dirichlet = :boundary)
        @test f0 isa KronExt._FDMFactorization && -1 in f0.recipe.on
        F = rhs(W, Float64, :boundary)
        xs, xa, x0 = (fdm_solve!(similar(F), f, F) for f in (fs, fa, f0))
        c3[] = 1.0
        W32 = gridspace(mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)),
            (17, 13), (false, false)))
        # The mass coefficient varies along axis 1, so term 1 no longer matches the mass
        # the other terms carry there: same terms, same route.
        gh = Rₕ(W, x -> 1 + x[1])
        cases = ((fs, K(W, adv), "has 4 terms"), (fa, K(W, sym), "has 3 terms"),
            (fs, K(graded_space(Float64, (9, 7), Serial()), sym), "size"),
            (fs, K(graded_space(Float64, (7, 6, 5), Serial()), sym), "dimension 3"),
            (fs, K(W32, sym), "eltype Float32"),
            (fs, K(W, (u, v) -> sym(u, v) + 0.3 * innerₕ(D₋ₓ(u), D₋ₓ(v))), "terms"),
            (fs, K(W, (u, v) -> innerₕ(gh * u, v) + c * inner₊(∇ₕ(u), ∇ₕ(v))), "mass"),
            (f0, K(W, L3), "generalised Schur route"))
        for (f, k, reason) in cases
            @test_throws ArgumentError fdm_factorize!(f, k)
            @test_throws "fdm_factorize! cannot refill this factorisation" fdm_factorize!(f, k)
            @test_throws reason fdm_factorize!(f, k)
        end
        @test fdm_solve!(similar(F), fs, F) == xs
        @test fdm_solve!(similar(F), fa, F) == xa
        @test fdm_solve!(similar(F), f0, F) == x0
        # The Schur route needs some axis operator that is not symmetric: with the advection
        # coefficient zero (a hand-made operator, as a `Ref` cannot scale that term), every
        # one is.
        Ka = K(W, adv)
        terms = map(t -> Bramble._kron_term(t.symmetric ? t.scales : (0.0,), t.factors),
            Ka.terms)
        Kz = Bramble.KroneckerLinearOperator{Float64, 2, typeof(terms), typeof(Ka.policy)}(
            terms, Ka.dims, Ka.n, Ka.policy)
        @test_throws ArgumentError fdm_factorize!(fa, Kz)
        @test_throws "every axis operator of `K` is symmetric" fdm_factorize!(fa, Kz)
        @test fdm_solve!(similar(F), fa, F) == xa
        # The same operator on device-backed factors (host vectors stand in for device ones).
        dev(M::LinearAlgebra.Diagonal) = Bramble._KronDeviceDiagonal(collect(M.diag))
        dev(M) = Bramble._KronDeviceSparse(Int32.(M.colptr), Int32.(M.rowval), M.nzval)
        Ks = K(W, sym)
        dterms = map(Ks.terms) do t
            fd = map(dev, t.factors)
            return Bramble.KroneckerTerm{2, typeof(t.scales), typeof(fd), Nothing, Nothing}(
                t.scales, fd, nothing, nothing, t.symmetric)
        end
        Kd = Bramble.KroneckerLinearOperator{Float64, 2, typeof(dterms), typeof(Ks.policy)}(
            dterms, Ks.dims, Ks.n, Ks.policy)
        @test_throws ArgumentError fdm_factorize!(fs, Kd)
        @test_throws "backed by a device" fdm_factorize!(fs, Kd)
        @test fdm_solve!(similar(F), fs, F) == xs
    end

    # A term dropped at build because a factor vanished on the interior (`gv` is zero at
    # the one interior node of a 3-point axis) is refused once it is present, not silently
    # left out; dropped again, it refills at 0 B, as a zero-coefficient one does.
    @testset "fdm_factorize!: dropped term" begin
        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 7), (false, false))
        Bramble.change_points!(Ω, (collect(range(0.0, 1.0; length = 3)),
            collect(range(0.0, 1.0; length = 7))))
        function op()
            W = gridspace(Ω)
            gv = Rₕ(W, x -> x[1] - 0.5)
            return kronecker_operator(form(W, W, (u, v) -> innerₕ(u, v) +
                                                           inner₊(∇ₕ(u), ∇ₕ(v)) +
                                                           innerₕ(gv * u, v)))
        end
        K = op()
        f = fdm_factorize(K; dirichlet = :boundary)
        @test -1 in f.recipe.on
        @test refill_bytes(f, K) == 0
        W = graded_space(Float64, (17, 13), Serial())
        c3 = Ref(0.0)
        K0 = kronecker_operator(form(W, W,
            (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + c3 * innerₕ(D₋ₓ(u), v)))
        f0 = fdm_factorize(K0; dirichlet = :boundary)
        @test -1 in f0.recipe.on
        @test refill_bytes(f0, K0) == 0
        Bramble.change_points!(Ω, (range(0.0, 1.0; length = 3) .^ 1.25,
            range(0.0, 1.0; length = 7) .^ 1.5))
        K = op()
        @test_throws ArgumentError fdm_factorize!(f, K)
        dropped = r"cannot refill this factorisation: term \d of `K` was dropped"
        @test_throws dropped fdm_factorize!(f, K)
    end

    # A refill refused for a numerical reason, after `f` was written, leaves `f` unable to
    # solve until a refill succeeds: singular (no mass term left), or an axis-1 mass that is
    # not symmetric positive definite (in a hand-made operator), on both routes: negated, or
    # not symmetric though its upper triangle, all `sygvd` reads, is definite.
    @testset "fdm_factorize!: refused refill" begin
        W = graded_space(Float64, (17, 13), Serial())
        F = rhs(W, Float64, nothing)
        c = Ref(2.5)
        K = kronecker_operator(form(W, W, (u, v) -> c * innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))
        f = fdm_factorize(K)
        c[] = 0.0
        @test_throws ArgumentError fdm_factorize!(f, K)
        @test_throws "singular" fdm_factorize!(f, K)
        @test_throws ArgumentError fdm_solve!(similar(F), f, F)
        @test_throws "fdm_factorize!" fdm_solve!(similar(F), f, F)
        c[] = 1.0
        @test fdm_factorize!(f, K) === f
        @test fdm_solve!(similar(F), f, F) == fdm_solve!(similar(F), fdm_factorize(K), F)
        for (L, route) in ((symmetric_form, KronExt._FDMFactorization),
            (advection_form, KronExt._SchurFactorization))
            Kp = kronecker_operator(form(W, W, L))
            fp = fdm_factorize(Kp)
            @test fp isa route
            m = Kp.terms[1].factors[1]
            neg = LinearAlgebra.Diagonal(-collect(m.diag))
            skew = sparse(Matrix(m))
            skew[1, 2] = 0.3 * skew[1, 1]
            for bad in (neg, skew)
                terms = map(t -> Bramble._kron_term(t.scales,
                        map(M -> M === m ? bad : M, t.factors)), Kp.terms)
                Kn = Bramble.KroneckerLinearOperator{Float64, 2, typeof(terms),
                    typeof(Kp.policy)}(terms, Kp.dims, Kp.n, Kp.policy)
                @test_throws ArgumentError fdm_factorize!(fp, Kn)
                notspd = "axis-1 mass is not symmetric positive definite"
                @test_throws notspd fdm_factorize!(fp, Kn)
                @test_throws "fdm_factorize!" fdm_solve!(similar(F), fp, F)
                fdm_factorize!(fp, Kp)
                x = fdm_solve!(similar(F), fp, F)
                @test x == fdm_solve!(similar(F), fdm_factorize(Kp), F)
            end
        end
    end

    # A 2-point axis under `:boundary` leaves no interior: nothing to decompose.
    @testset "_fdm_decompose!: empty interior" begin
        W = graded_space(Float64, (2, 9), Serial())
        f = fdm_factorize(form(W, W, symmetric_form); dirichlet = :boundary)
        @test isempty(f.Λ)
        @test decompose_bytes(f) == 0
        @test iszero(fdm_solve!(zeros(ndofs(W)), f, ones(ndofs(W))))
    end

    # The singularity refusal reruns on the recomputed `Λ_total`: with the mass coefficient
    # gone, a Neumann stiffness is singular.
    @testset "_fdm_decompose! refuses singular" begin
        W = graded_space(Float64, (17, 13), Serial())
        f = fdm_factorize(form(W, W, symmetric_form))
        f.c_m[] = 0.0
        @test_throws ArgumentError KronExt._fdm_decompose!(f)
        @test_throws "singular" KronExt._fdm_decompose!(f)
        f.c_m[] = 1.0
        KronExt._fdm_decompose!(f)
        F = rhs(W, Float64, nothing)
        @test fdm_solve!(similar(F), f, F) ==
              fdm_solve!(similar(F), fdm_factorize(form(W, W, symmetric_form)), F)
    end

    # With no OpenBLAS handle, the Schur route calls the active LAPACK's `xgges3` through
    # libblastrampoline. Accelerate's fails to converge on this pencil
    # (gpena/Bramble.jl#443): an ArgumentError naming the active LAPACK, as
    # test/ext/kronecker_ext.jl pins for a fresh build on the same mesh and form.
    @testset "_fdm_decompose!: no OpenBLAS" begin
        X = interval(0.0, 1.0) × interval(0.0, 1.0)
        W = gridspace(mesh(domain(X), (65, 17), (true, true)))
        a = form(W, W, (u, v) -> (1 / 1e6) * inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(u), v) +
                                 innerₕ(Bramble.D₋ᵧ(u), v))
        f = fdm_factorize(a; dirichlet = :boundary)
        @test f isa KronExt._SchurFactorization
        F = rhs(W, Float64, :boundary)
        x = fdm_solve!(similar(F), f, F)
        accelerate = any(lib -> occursin("Accelerate", lib.libname),
            LinearAlgebra.BLAS.get_config().loaded_libs)
        found = KronExt._OPENBLAS_GGES3[]
        try
            KronExt._OPENBLAS_GGES3[] = (C_NULL, C_NULL)
            wipe!(f)
            if accelerate
                @test_throws ArgumentError KronExt._fdm_decompose!(f)
                @test_throws "active LAPACK" KronExt._fdm_decompose!(f)
            else
                @test decompose_bytes(f) == 0
                @test norm(fdm_solve!(similar(F), f, F) - x) <= 1e-10 * norm(x)
            end
        finally
            KronExt._OPENBLAS_GGES3[] = found
        end
        KronExt._fdm_decompose!(f)
        @test fdm_solve!(similar(F), f, F) == x
    end
end

end # module TestKroneckerRefactorize
