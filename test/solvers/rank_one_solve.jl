module RankOneSolveTests

# `pde_solve(K, F)` for a `RankOneUpdate` `K = A + σ wₛ wₜᵀ`, through the bordered system
# `[A wₛ; wₜᵀ −1/σ]`, against references that share no code with it: the dense solve of
# `Matrix(A) + σ wₛ wₜᵀ`, and for compatible data (`1ᵀF = 0`)
# `pde_solve(A, F; mean_zero = w)`, which that solve equals for every σ ≠ 0. Data with `1ᵀF ≠ 0` tell the corner `−1/σ` apart
# from the `mean_zero` border's zero: then `wᵀu = 1ᵀF / (σ 1ᵀw)`, which depends on σ.
#
# Accuracy (measured on these meshes): against `mean_zero` the relative error is about
# `1e-14 / γ` (σ = γ / hₘₐₓ): 1e-15 for γ ≥ 1, 1e-12 at γ = 1e-2, 1e-8 at γ = 1e-6, 1e-4
# at γ = 1e-10. A small σ puts `−1/σ` in the corner, far above `A`'s entries, and the
# factorisation loses what is left. The dense reference degrades the other way, with the
# conditioning of `A + σ w wᵀ` (1e-8 at γ = 1e9). Float32 reaches 2e-5 for γ ≥ 1.

using Test
using Bramble
import Bramble
using Bramble: RankOneUpdate
using LinearAlgebra: norm, dot, lu, I
using SparseArrays: SparseMatrixCSC, sparse, spzeros, spdiagm
using Random: Random
using ..TestUtils: WITH_3D_TESTS, WITH_SLOW_TESTS

function random_mesh(seed, ranges, n)
    Bramble._seed_mesh1d_rng!(seed)
    try
        return mesh(domain(foldl(×, ranges)), n, ntuple(_ -> false, length(n)))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

const ∇form = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))

# A pure-Neumann system on `Ω`: `A` annihilates constants, `w` the discrete average's
# weights, `F` a load with `1ᵀF ≠ 0` and `Fc` its compatible part.
function neumann_system(Ω)
    W = gridspace(Ω)
    A = assemble(form(W, W, ∇form))
    w = assemble(form(W, v -> innerₕ(x -> 1.0, v)))
    F = assemble(form(W, v -> innerₕ(x -> sum(x) + x[1]^2, v)))
    return W, A, w, F, F .- sum(F) / length(F)
end

relerr(a, b) = norm(a - b) / norm(b)

const MESHES = [
    (1, random_mesh(1, (interval(0.0, 1.0),), (40,))),
    (2, random_mesh(2, (interval(0.0, 1.0), interval(0.0, 2.0)), (17, 19)))
]

@testset "Rank-one solve: dense and mean_zero" begin
    for (D, Ω) in MESHES
        W, A, w, F, Fc = neumann_system(Ω)
        um = pde_solve(A, Fc; mean_zero = w)
        for γ in (1.0e-2, 1.0, 1.0e3, 1.0e6)
            σ = γ / hₘₐₓ(Ω)
            K = assemble(form(W, W, (u, v) -> ∇form(u, v) + σ * innerₕ(1, u) * innerₕ(1, v)))
            @test K isa RankOneUpdate
            Md = Matrix(A) + σ * w * w'
            u = pde_solve(K, Fc)
            @test u isa Vector{Float64}
            @test relerr(u, Md \ Fc) <= 1.0e-9
            @test relerr(u, um) <= 1.0e-11
            ui = pde_solve(K, F)
            @test relerr(ui, Md \ F) <= 1.0e-9
            @test dot(w, ui) ≈ sum(F) / (σ * sum(w)) rtol = 1.0e-8
            @test relerr(ui, um) > 1.0e-6  # the corner is −1/σ, not the mean_zero zero
        end
    end
end

@testset "Rank-one solve: negative and complex σ" begin
    _, Ω = MESHES[2]
    W, A, w, F, Fc = neumann_system(Ω)
    um = pde_solve(A, Fc; mean_zero = w)
    for σ in (-1.0, -1.0e3) ./ hₘₐₓ(Ω)
        K = RankOneUpdate(A, σ, w, w)
        @test relerr(pde_solve(K, Fc), um) <= 1.0e-11
        @test relerr(pde_solve(K, F), (Matrix(A) + σ * w * w') \ F) <= 1.0e-10
    end
    σ = (1.0 + 2.0im) / hₘₐₓ(Ω)
    K = RankOneUpdate(A, σ, w, w)
    u = pde_solve(K, F)
    @test u isa Vector{ComplexF64}
    @test relerr(u, (Matrix(A) + σ * w * w') \ F) <= 1.0e-10
    @test relerr(pde_solve(K, complex.(Fc)), um) <= 1.0e-11
    # a complex border is transposed, not conjugated
    wc = w .* (1.0 .+ 0.5im)
    u = pde_solve(RankOneUpdate(A, 2.0, wc, wc), F)
    @test relerr(u, (Matrix(A) + 2.0 * wc * transpose(wc)) \ F) <= 1.0e-10
end

@testset "Rank-one solve: wₛ ≠ wₜ" begin
    for (D, Ω) in MESHES
        W, A, w, F, _ = neumann_system(Ω)
        σ = 3.0 / hₘₐₓ(Ω)
        K = assemble(form(W, W,
            (u, v) -> ∇form(u, v) + σ * innerₕ(x -> 1 + x[1], u) * innerₕ(x -> 2 - x[D], v)))
        @test K.ws != K.wt
        ws = assemble(form(W, v -> innerₕ(x -> 2 - x[D], v)))
        wt = assemble(form(W, v -> innerₕ(x -> 1 + x[1], v)))
        @test relerr(pde_solve(K, F), (Matrix(A) + σ * ws * wt') \ F) <= 1.0e-9
        # swapped, the solve changes: the column is wₛ and the row wₜ
        Ks = RankOneUpdate(A, σ, wt, ws)
        @test relerr(pde_solve(Ks, F), (Matrix(A) + σ * wt * ws') \ F) <= 1.0e-9
        @test relerr(pde_solve(Ks, F), pde_solve(K, F)) > 1.0e-6
    end
end

# A symmetric route (Accelerate and MUMPS `LDLᵀ`) reads one triangle, so the bordered
# matrix is symmetric only when `wₛ == wₜ`, `A` is symmetric and everything is real.
@testset "Rank-one solve: bordered symmetry" begin
    _, Ω = MESHES[1]
    _, A, w, _, _ = neumann_system(Ω)
    n = size(A, 1)
    C0 = spzeros(n, 0)
    B, s = Bramble._bordered_matrix(Float64, A, C0, (w, w, -0.5))
    @test s === :symmetric
    @test Matrix(B) == [Matrix(A) w; w' -0.5]
    w2 = 2 .* w
    B, s = Bramble._bordered_matrix(Float64, A, C0, (w, w2, -0.5))
    @test s === :unsymmetric
    @test Matrix(B) == [Matrix(A) w; w2' -0.5]
    An = copy(A)
    An[1, 2] *= 2
    @test Bramble._bordered_matrix(Float64, An, C0, (w, w, -0.5))[2] === :unsymmetric
    @test Bramble._bordered_matrix(ComplexF64, A, spzeros(ComplexF64, n, 0),
        (w, w, -0.5im))[2] === :unsymmetric
    # with a mean_zero column after wₛ, and the corner only on the rank-one row
    B, s = Bramble._bordered_matrix(Float64, A, sparse(reshape(w2, n, 1)), (w, w, -0.5))
    @test s === :symmetric
    @test Matrix(B) == [Matrix(A) w w2; w' -0.5 0; w2' 0 0]
end

# `K.σ` is read when the solve runs; a `Ref` factor reaches it through `assemble!`.
@testset "Rank-one solve: σ from a Ref" begin
    _, Ω = MESHES[2]
    W, A, w, F, _ = neumann_system(Ω)
    r = Ref(2.0)
    a = form(W, W, (u, v) -> ∇form(u, v) + r * innerₕ(1, u) * innerₕ(1, v))
    K = assemble(a)
    u2 = pde_solve(K, F)
    @test relerr(u2, (Matrix(A) + 2.0 * w * w') \ F) <= 1.0e-10
    r[] = 50.0
    @test pde_solve(K, F) == u2
    assemble!(K, a)
    u50 = pde_solve(K, F)
    @test relerr(u50, (Matrix(A) + 50.0 * w * w') \ F) <= 1.0e-10
    @test relerr(u50, u2) > 1.0e-3
    K.σ = 7.0
    @test relerr(pde_solve(K, F), (Matrix(A) + 7.0 * w * w') \ F) <= 1.0e-10
end

@testset "Rank-one solve: σ = 0 is the plain solve" begin
    _, Ω = MESHES[2]
    W, A, w, F, Fc = neumann_system(Ω)
    Ar = A + 1.0I  # nonsingular
    @test pde_solve(RankOneUpdate(Ar, 0.0, w, w), F) == pde_solve(Ar, F)
    @test pde_solve(RankOneUpdate(Ar, 0.0, w, 2 .* w), F) == pde_solve(Ar, F)
    K0 = RankOneUpdate(A, 0.0, w, w)
    @test pde_solve(K0, Fc; mean_zero = w) == pde_solve(A, Fc; mean_zero = w)
    # the plain path's warning for a pure-Neumann A, which σ = 0 does not fix
    @test_logs (:warn, r"mean_zero") match_mode = :any (try
        pde_solve(K0, F)
    catch
    end)
    @test_logs min_level = Base.CoreLogging.Warn pde_solve(RankOneUpdate(A, 1.0, w, w), F)
end

nnz_row(A::SparseMatrixCSC, i) = count(!iszero, A[i, :])

@testset "Rank-one solve: Dirichlet rows" begin
    _, Ω = MESHES[2]
    W = gridspace(Ω)
    σ = 5.0 / hₘₐₓ(Ω)
    gD = x -> 1 + x[1] * x[2]
    a = form(W, W, (u, v) -> ∇form(u, v) + σ * innerₕ(1, u) * innerₕ(1, v))
    l = form(W, v -> innerₕ(x -> sum(x), v))
    K, F = assemble(a, l; dirichlet = :boundary => gD)
    # the reference: the Dirichlet rows of A are identity rows, and the term is absent there
    Ad, Fd = assemble(form(W, W, ∇form), l; dirichlet = :boundary => gD)
    w = assemble(form(W, v -> innerₕ(x -> 1.0, v)))
    bd = [nnz_row(Ad, i) == 1 && Ad[i, i] == 1 for i in axes(Ad, 1)]
    @test count(bd) == 2 * (17 + 19) - 4
    ws = ifelse.(bd, 0.0, w)
    ud = (Matrix(Ad) + σ * ws * w') \ Fd
    u = pde_solve(K, F)
    @test relerr(u, ud) <= 1.0e-10
    @test u[bd] ≈ Fd[bd] rtol = 1.0e-12
    @test norm(K.A * u + σ * K.ws * dot(K.wt, u) - F) <= 1.0e-9 * norm(F)
    Ks, Fs = assemble(a, l; dirichlet = :boundary => gD, symmetrize = true)
    @test relerr(pde_solve(Ks, Fs), ud) <= 1.0e-10
end

# A ghost-point Neumann matrix: constants in its right kernel only (its boundary rows carry
# the doubled neighbour `-2/h²`).
function ghost_neumann(N)
    h = 1 / N
    A = spdiagm(-1 => fill(-1.0, N), 0 => fill(2.0, N + 1), 1 => fill(-1.0, N)) / h^2
    A[1, 2] = -2 / h^2
    A[N + 1, N] = -2 / h^2
    return A
end

@testset "Rank-one solve: an unfixed singular A" begin
    _, Ω = MESHES[2]
    _, A, w, F, Fc = neumann_system(Ω)
    n = size(A, 1)
    z = collect(range(-1.0, 1.0; length = n))
    z .-= sum(z) / n  # 1ᵀz = 0: z does not see constants
    e = @test_throws ArgumentError pde_solve(RankOneUpdate(A, 1.0, w, z), F)
    @test occursin("wₜ", e.value.msg) && occursin("mean_zero", e.value.msg)
    e = @test_throws ArgumentError pde_solve(RankOneUpdate(A, 1.0, z, w), F)
    @test occursin("wₛ", e.value.msg) && occursin("innerₕ(1, u)", e.value.msg)
    @test_throws ArgumentError pde_solve(RankOneUpdate(A, 1.0, w, zeros(n)), F)
    # mean_zero fixes the constant itself: no check, and the border carries both rows
    u = pde_solve(RankOneUpdate(A, 1.0, w, z), Fc; mean_zero = w)
    @test abs(dot(w, u)) <= 1.0e-12 * norm(u)
    r = A * u + w * dot(z, u) - Fc
    @test norm(r - w * (dot(w, r) / dot(w, w))) <= 1.0e-9 * norm(Fc)
    # an A without constants in its kernel is solved whatever the functionals' sums
    Ar = A + 1.0I
    @test relerr(pde_solve(RankOneUpdate(Ar, 1.0, z, z), F),
        (Matrix(Ar) + z * z') \ F) <= 1.0e-10
    # constants in the right kernel only: wₜ is checked, wₛ is not
    G = ghost_neumann(10)
    @test norm(G * ones(11), Inf) == 0 && norm(G' * ones(11), Inf) > 1
    m = size(G, 1)
    y = zeros(m)  # zero sum, and not orthogonal to G's left kernel (trapezoid weights)
    y[1], y[2] = 1.0, -1.0
    o = ones(m)
    b = collect(range(0.0, 1.0; length = m))
    @test relerr(pde_solve(RankOneUpdate(G, 1.0, y, o), b), (Matrix(G) + y * o') \ b) <= 1.0e-10
    @test_throws ArgumentError pde_solve(RankOneUpdate(G, 1.0, o, y), b)
    Gt = sparse(G')
    @test relerr(pde_solve(RankOneUpdate(Gt, 1.0, o, y), b), (Matrix(Gt) + o * y') \ b) <= 1.0e-10
    @test_throws ArgumentError pde_solve(RankOneUpdate(Gt, 1.0, y, o), b)
end

# The check's tolerance, `8 ε ‖A‖∞` on `‖A 1‖∞`, from both sides: pure-Neumann matrices
# read 0.2 to 0.65 ε (refused with a zero-sum functional), and `−Δ` plus a reaction of
# `1e-10` reads 50 ε (nonsingular, solved). `zerosum` is a functional blind to constants.
zerosum(n) = (z = collect(range(-1.0, 1.0; length = n)); z .- sum(z) / n)

@testset "Rank-one solve: check tolerance" begin
    for Ω in (random_mesh(4, (interval(0.0, 1.0),), (2000,)),
        random_mesh(5, (interval(0.0, 1.0), interval(0.0, 2.0)), (120, 130)))
        _, A, w, F, _ = neumann_system(Ω)
        z = zerosum(size(A, 1))
        @test_throws ArgumentError pde_solve(RankOneUpdate(A, 1.0, w, z), F)
        @test_throws ArgumentError pde_solve(
            RankOneUpdate(Float32.(A), 1.0f0, Float32.(w), Float32.(z)), Float32.(F))
    end
    _, Ω = MESHES[2]
    _, A, w, F, _ = neumann_system(Ω)
    n = size(A, 1)
    z = zerosum(n)
    Aδ = A + 1.0e-10 * spdiagm(0 => w)
    ub = (big.(Matrix(Aδ)) + big.(z) * big.(z)') \ big.(F)
    u = pde_solve(RankOneUpdate(Aδ, 1.0, z, z), F)
    # badly conditioned: the bordered solve keeps three digits (6e-4), as the dense LU does
    @test Float64(norm(u - ub) / norm(ub)) <= 2.0e-3
    # the threshold itself: a reaction `c w` puts `‖A 1‖∞` at about ρ ε ‖A‖∞; ρ = 5 counts
    # as constants in the kernel (refused) and ρ = 12 does not (solved)
    ρ(B) = norm(B * ones(n), Inf) / (eps() * norm(B, Inf))
    Aρ(r) = A + (r * eps() * norm(A, Inf) / maximum(w)) * spdiagm(0 => w)
    @test 4.5 <= ρ(Aρ(5)) <= 5.5 && 11.5 <= ρ(Aρ(12)) <= 12.5
    @test_throws ArgumentError pde_solve(RankOneUpdate(Aρ(5), 1.0, w, z), F)
    @test all(isfinite, pde_solve(RankOneUpdate(Aρ(12), 1.0, w, z), F))
    # the functional's side: refused when |1ᵀwₜ| ≤ 8 ε ‖wₜ‖₁, solved above it
    s(x) = abs(sum(x)) / (eps() * norm(x, 1))
    zs(r) = z .+ r * eps() * norm(z, 1) / n
    @test 3 <= s(zs(4)) <= 5 && s(zs(1.0e4)) > 9.0e3
    @test_throws ArgumentError pde_solve(RankOneUpdate(A, 1.0, w, zs(4)), F)
    u = pde_solve(RankOneUpdate(A, 1.0, w, zs(1.0e4)), F)
    # 1ᵀ(A + wₛ wₜᵀ) u = 1ᵀF with 1ᵀA = 0 fixes wₜᵀu; with 1ᵀwₜ near 1e-12 the system is
    # that ill conditioned, and two digits survive (4e-3)
    @test dot(zs(1.0e4), u) ≈ sum(F) / sum(w) rtol = 2.0e-2
end

@testset "Rank-one solve: K' and transpose(K)" begin
    _, Ω = MESHES[2]
    _, A, w, F, _ = neumann_system(Ω)
    v = assemble(form(gridspace(Ω), v -> innerₕ(x -> 1 + x[1], v)))
    n = size(A, 1)
    # unsymmetric, and complex, so that A, Aᵀ, Aᴴ and conj(A) all differ
    Au = A + spdiagm(1 => fill(0.7, n - 1))
    Ac = A + spdiagm(1 => fill(0.7 + 0.4im, n - 1))
    for B in (Au, Ac), σ in (2.0, 2.0 - 1.0im), ws in (w, 1.0im .* w)
        K = RankOneUpdate(B, σ, ws, v)
        M = Matrix(B) + σ * ws * transpose(v)
        @test relerr(pde_solve(K', F), M' \ F) <= 1.0e-10
        @test relerr(pde_solve(transpose(K), F), transpose(M) \ F) <= 1.0e-10
        @test relerr(pde_solve(transpose(K'), F), conj(M) \ F) <= 1.0e-10
        @test relerr(pde_solve(transpose(K)', F), conj(M) \ F) <= 1.0e-10
        # a sparse part built lazily wrapped, by hand, is materialised
        Ka = RankOneUpdate(B', σ, ws, v)
        @test relerr(pde_solve(Ka, F), (Matrix(B)' + σ * ws * transpose(v)) \ F) <= 1.0e-10
        Kb = RankOneUpdate(transpose(B), σ, ws, v)
        @test relerr(pde_solve(Kb, F),
            (transpose(Matrix(B)) + σ * ws * transpose(v)) \ F) <= 1.0e-10
    end
    # σ and the vectors are read from the parent at solve time, after a refill
    W = gridspace(Ω)
    r = Ref(2.0 + 0.0im)
    a = form(W, W, (u, v) -> ∇form(u, v) + r * innerₕ(x -> 1 + x[1], u) * innerₕ(1, v))
    K = assemble(a)
    Kt, Kh = transpose(K), K'
    r[] = 5.0 - 3.0im
    assemble!(K, a)
    M = Matrix(A) + (5.0 - 3.0im) * w * transpose(v)
    @test relerr(pde_solve(Kh, F), M' \ F) <= 1.0e-10
    @test relerr(pde_solve(Kt, F), transpose(M) \ F) <= 1.0e-10
end

@testset "Rank-one solve: with a gauge" begin
    Ω = random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7))
    Wg = gridspace(Ω; ghosts = :boundary)
    Ag = assemble(form(Wg, Wg, (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)))
    Z = Bramble.ghost_kernel(Ag, Wg)
    n = size(Ag, 1)
    np = npoints(Ω)
    wg = collect(range(1.0, 2.0; length = n))
    wg[(np + 1):end] .= 0  # a physical functional: the ghost kernel stays A's
    x = collect(range(-1.0, 1.0; length = n)) .^ 2
    K = RankOneUpdate(Ag, 3.0, wg, wg)
    F = K * x
    u = pde_solve(K, F; gauge = Z)
    @test norm(transpose(Z) * u, Inf) <= 1.0e-10 * norm(u, Inf)
    r = K * u - F
    @test norm(r - Z * (Matrix(Z) \ r)) <= 1.0e-7 * norm(F)
    @test relerr(u[1:np], x[1:np]) <= 1.0e-6
    @test_throws ArgumentError pde_solve(K, F; gauge = Z, solver = :sparspak)
end

@testset "Rank-one solve: Float32 and inference" begin
    _, Ω = MESHES[2]
    _, A, w, F, Fc = neumann_system(Ω)
    um = pde_solve(A, Fc; mean_zero = w)
    σ = 1.0 / hₘₐₓ(Ω)
    K32 = RankOneUpdate(Float32.(A), Float32(σ), Float32.(w), Float32.(w))
    u32 = pde_solve(K32, Float32.(Fc))
    @test u32 isa Vector{Float32}
    @test relerr(u32, um) <= 1.0e-4
    K = RankOneUpdate(A, σ, w, w)
    @test (@inferred pde_solve(K, F)) isa Vector{Float64}
    Ki = RankOneUpdate(SparseMatrixCSC{Int, Int}([1 -1 0; -1 2 -1; 0 -1 1]), 1, [1, 1, 1],
        [1, 1, 1])
    ui = pde_solve(Ki, [2, 0, -1])
    @test ui isa Vector{Float64}
    @test ui ≈ ([1 -1 0; -1 2 -1; 0 -1 1] + ones(3, 3)) \ [2.0, 0.0, -1.0] atol = 1.0e-12
    @test pde_solve(RankOneUpdate(spzeros(0, 0), 1.0, Float64[], Float64[]), Float64[]) ==
          Float64[]
end

@testset "Rank-one solve: refusals" begin
    _, Ω = MESHES[1]
    _, A, w, F, _ = neumann_system(Ω)
    K = RankOneUpdate(A, 1.0, w, w)
    e = @test_throws ArgumentError pde_solve(K, F; solver = :spqr)
    @test occursin("RankOneUpdate", e.value.msg) && occursin(":suitesparse", e.value.msg)
    @test_throws ArgumentError pde_solve(K, F; solver = :bogus)
    @test_throws ArgumentError pde_solve(K, F; bogus = 1)
    @test_throws DimensionMismatch pde_solve(K, F[1:(end - 1)])
    @test_throws DimensionMismatch pde_solve(K, F; mean_zero = w[1:(end - 1)])
    e = @test_throws ArgumentError pde_solve(RankOneUpdate(Matrix(A), 1.0, w, w), F)
    @test occursin("sparse(K.A)", e.value.msg)
    # one right-hand side, as for a matrix
    @test_throws MethodError pde_solve(K, hcat(F, F))
    @test_throws MethodError pde_solve(A, hcat(F, F))
end

# The plain and mean_zero paths are the bordered solve they were before rank-one terms: the
# same sparse matrix, factored by the same LU, bit for bit (without Accelerate's route).
@testset "Rank-one solve: plain paths unchanged" begin
    _, Ω = MESHES[2]
    _, A, w, F, Fc = neumann_system(Ω)
    n = size(A, 1)
    if Base.get_extension(Bramble, :BrambleAppleAccelerateExt) === nothing
        ws = SparseMatrixCSC{Float64, Int}(reshape(w, n, 1))
        B = [A ws; transpose(ws) spzeros(1, 1)]
        @test pde_solve(A, Fc; mean_zero = w) == (lu(B) \ [Fc; 0.0])[1:n]
        σ = 4.0
        Bσ = [A ws; transpose(ws) sparse(fill(-inv(σ), 1, 1))]
        @test pde_solve(RankOneUpdate(A, σ, w, w), F) == (lu(Bσ) \ [F; 0.0])[1:n]
    end
    Ar = A + 1.0I
    @test pde_solve(Ar, F) == Ar \ F
end

# The dense matrix is 25 MB here; the solve stays within the bordered system's size.
@testset "Rank-one solve: sparse memory" begin
    Ω = random_mesh(6, (interval(0.0, 1.0), interval(0.0, 1.0)), (41, 43))
    W, A, w, F, Fc = neumann_system(Ω)
    K = assemble(form(W, W, (u, v) -> ∇form(u, v) + innerₕ(1, u) * innerₕ(1, v)))
    pde_solve(K, Fc)
    bytes = @allocated pde_solve(K, Fc)
    @test bytes < 64 * Base.summarysize(A)
    @test 8 * length(F)^2 > 2 * 64 * Base.summarysize(A)
    @test relerr(pde_solve(K, Fc), pde_solve(A, Fc; mean_zero = w)) <= 1.0e-10
end

# Small σ: the corner −1/σ dwarfs A's entries and the bordered solve loses about
# `1e-14 / γ` against `mean_zero`. These bounds hold with a factor of 10 to spare.
@testset "Rank-one solve: small σ" begin
    for (D, Ω) in MESHES
        W, A, w, F, Fc = neumann_system(Ω)
        um = pde_solve(A, Fc; mean_zero = w)
        for (γ, tol) in ((1.0e-6, 1.0e-7), (1.0e-10, 1.0e-3))
            u = pde_solve(RankOneUpdate(A, γ / hₘₐₓ(Ω), w, w), Fc)
            @test relerr(u, um) <= tol
        end
    end
end

# The backends loaded in this session, then (slow group) a child that loads every backend
# the environment has, Sparspak included.
@testset "Rank-one solve: each solver backend" begin
    _, Ω = MESHES[2]
    W, A, w, F, Fc = neumann_system(Ω)
    σ = 2.0 / hₘₐₓ(Ω)
    loaded = filter(((s, ext),) -> Base.get_extension(Bramble, ext) !== nothing,
        [(:suitesparse, :BrambleSuiteSparseExt), (:mumps, :BrambleMUMPSExt),
            (:accelerate, :BrambleAppleAccelerateExt), (:sparspak, :BrambleSparspakExt)])
    wt = 2 .* w
    for solver in (:default, first.(loaded)...), (ws, wt) in ((w, w), (w, wt))

        u = pde_solve(RankOneUpdate(A, σ, ws, wt), F; solver = solver)
        @test relerr(u, (Matrix(A) + σ * ws * wt') \ F) <= 1.0e-9
    end
    if WITH_SLOW_TESTS
        code = raw"""
        using Bramble, LinearAlgebra, SparseArrays, SuiteSparse, MUMPS, Sparspak
        Sys.isapple() && @eval using AppleAccelerate
        Bramble._seed_mesh1d_rng!(2)
        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (17, 19), (false, false))
        Bramble._unseed_mesh1d_rng!()
        W = gridspace(Ω)
        A = assemble(form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))))
        w = assemble(form(W, v -> innerₕ(x -> 1.0, v)))
        F = assemble(form(W, v -> innerₕ(x -> sum(x) + x[1]^2, v)))
        σ = 2.0 / hₘₐₓ(Ω)
        for s in (:default, :suitesparse, :mumps, :sparspak,
                (Sys.isapple() ? (:accelerate,) : ())...), wt in (w, 2 .* w)
            u = pde_solve(Bramble.RankOneUpdate(A, σ, w, wt), F; solver = s)
            ud = (Matrix(A) + σ * w * wt') \ F
            println(s, " ", norm(u - ud) / norm(ud))
        end
        """
        project = dirname(something(Base.active_project()))
        cmd = `$(Base.julia_cmd()) --project=$project --startup-file=no --threads=1 -e $code`
        lines = split(read(cmd, String), '\n'; keepempty = false)
        @test length(lines) == 2 * (Sys.isapple() ? 5 : 4)
        for line in lines
            s, e = split(line)
            @test parse(Float64, e) <= 1.0e-9
        end
    end
end

# 3D runs nightly.
WITH_3D_TESTS && @testset "Rank-one solve: 3D" begin
    Ω = random_mesh(3, (interval(0.0, 1.0), interval(0.0, 2.0), interval(0.0, 1.5)), (5, 6, 7))
    W, A, w, F, Fc = neumann_system(Ω)
    um = pde_solve(A, Fc; mean_zero = w)
    for γ in (1.0e-2, 1.0, 1.0e3, 1.0e6)
        σ = γ / hₘₐₓ(Ω)
        K = assemble(form(W, W, (u, v) -> ∇form(u, v) + σ * innerₕ(1, u) * innerₕ(1, v)))
        @test relerr(pde_solve(K, Fc), um) <= 1.0e-10
        @test relerr(pde_solve(K, F), (Matrix(A) + σ * w * w') \ F) <= 1.0e-9
    end
    @test_throws ArgumentError pde_solve(RankOneUpdate(A, 1.0, w, zerosum(size(A, 1))), F)
    # a true pure-Neumann matrix above 1 ε: 40³ random points read ρ = 1.12
    Random.seed!(7)
    Ωb = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
        (40, 40, 40), (false, false, false))
    Wb = gridspace(Ωb)
    Ab = assemble(form(Wb, Wb, ∇form))
    nb = size(Ab, 1)
    @test norm(Ab * ones(nb), Inf) > eps() * norm(Ab, Inf)
    @test_throws ArgumentError pde_solve(RankOneUpdate(Ab, 1.0, ones(nb), zerosum(nb)),
        ones(nb))
end

end # module RankOneSolveTests
