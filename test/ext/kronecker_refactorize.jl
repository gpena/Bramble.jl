module TestKroneckerRefactorize

using Test
using Bramble
using Bramble: kronecker_operator, D₋ₓ
using LinearAlgebra: LinearAlgebra, Symmetric, UpperTriangular, eigen, norm
using Random
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

# The arrays `_fdm_decompose!` writes, so zeroing them first keeps a no-op from passing.
outputs(f::KronExt._FDMFactorization) = (f.Q..., f.Λ)
outputs(f::KronExt._SchurFactorization) = (f.Qc..., f.Zt..., f.S..., f.Tr..., f.Λ)
wipe!(f) = foreach(A -> fill!(A, zero(eltype(A))), outputs(f))
decompose_bytes(f) = (KronExt._fdm_decompose!(f); @allocated KronExt._fdm_decompose!(f))

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
