module FormGhostRankOneTests

# Rank-one terms `σ * innerₕ(g, u) * innerₕ(h, v)` on ghost spaces (#780). On the padded
# unknown `E_u u + d` the term is `σ wₛ_pad wₜ_padᵀ`, so the reduced matrix is
# `E_vᵀ (A_pad + σ wₛ_pad wₜ_padᵀ) E_u` and the load `E_vᵀ (F_pad − (A_pad + σ wₛ_pad wₜ_padᵀ) d)`.
#
# The references share no reduction code with the library: `A_pad`, `wₛ_pad`, `wₜ_pad` and
# `F_pad` are forms on the padded spaces assembled apart, densified and multiplied by the
# closures. The trial data `g` and the test data `h` differ, and so do the trial and test
# spaces' closures where a case allows, so a swap of the two sides changes the answer. Every
# mesh is non-uniform and seeded.

using Test
using Bramble
import Bramble
using Bramble: RankOneUpdate, padded_space, closure, D₋ₓ
using LinearAlgebra: norm, dot, I
using SparseArrays: SparseMatrixCSC, nonzeros, nnz
using Random: seed!
using ..TestUtils: WITH_3D_TESTS

relerr(A, B) = norm(A - B) / max(norm(B), eps(real(eltype(B))))

lap(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))
const gt = x -> 1 + x[1]          # trial functional data
const hs = x -> 2 - x[end]        # test functional data
const fl = x -> x[1] + 1          # load data

function rmesh(D, T; seed = 780, n = (9, (7, 9), (5, 6, 4))[D])
    seed!(seed)
    I1 = interval(zero(T), one(T))
    Ω = D == 1 ? domain(I1) : D == 2 ? domain(I1 × I1) : domain(I1 × I1 × I1)
    return mesh(Ω, n, D == 1 ? false : ntuple(_ -> false, D))
end

msg(f) =
    try
        f()
        ""
    catch err
        err isa ArgumentError ? sprint(showerror, err) : "not an ArgumentError: $err"
    end

# The parts of the reference on the padded spaces of the trial space `Cu` and test space `Cv`.
function reference(Cu, Cv, σ; f = lap, g = gt, h = hs, load = fl)
    Pu, Pv = padded_space(Cu), padded_space(Cv)
    Ap = Matrix(assemble(form(Pu, Pv, f)))
    wsp = assemble(form(Pv, v -> innerₕ(h, v)))
    wtp = assemble(form(Pu, v -> innerₕ(g, v)))
    Fp = assemble(form(Pv, v -> innerₕ(load, v)))
    Eu, du = closure(Cu)
    Ev, _ = closure(Cv)
    Ar = Ap + σ * wsp * wtp'
    return (; K = Ev' * Ar * Eu, F = Ev' * (Fp - Ar * du), ws = Ev' * wsp, wt = Eu' * wtp,
        A = Ev' * Ap * Eu, Fplain = Ev' * (Fp - Ap * du), du, wtp)
end

rank_form(Cu, Cv, σ; f = lap, g = gt, h = hs) = form(Cu, Cv, (u, v) -> f(u, v) + σ * innerₕ(g, u) * innerₕ(h, v))

# Every output entry overwritten by a refill: start from NaN.
function poison!(K, F = nothing)
    fill!(nonzeros(K.A), NaN)
    fill!(K.ws, NaN)
    fill!(K.wt, NaN)
    F === nothing || fill!(F, NaN)
    return K
end

refill(K, a) = (assemble!(K, a); assemble!(K, a); @allocated assemble!(K, a))
refill(K, F, a, l) = (assemble!(K, F, a, l); assemble!(K, F, a, l);
    @allocated assemble!(K, F, a, l))
function refill_rows(K, F, a, l, dir)
    (assemble!(K, F, a, l; dirichlet = dir, symmetrize = true);
        @allocated assemble!(K, F, a, l; dirichlet = dir, symmetrize = true))
end

# The spaces of each case: `(trial, test)`, with relations of every kind and `d ≠ 0`.
function cases(D, T)
    Ωₕ = rmesh(D, T)
    o, two = one(T), T(2)
    if D == 1
        W = gridspace(Ωₕ; ghosts = (:xmin, :xmax))
        Cu = constrained(W, u -> (:xmin => ∂ₙ(u) => o, :xmax => two * u + ∂ₙ(u) => T(0.5)))
        Cv = constrained(W, u -> (:xmin => ∂ₙ(u) => o, :xmax => ∂ₙ(u) => -o))
        Cd = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
            u -> (:xmin => u + ∂ₙ(u) => o, :xmax => u => two))
        return (("Neumann and Robin, two closures", Cu, Cv),
            ("u => g at xmax", Cd, Cd))
    elseif D == 2
        W = gridspace(Ωₕ; ghosts = (:xmin, :xmax, :ymin))
        Cmix = constrained(W,
            u -> (:xmin => ∂ₙ(u) => o, :xmax => two * u + ∂ₙ(u) => x -> x[2],
                :ymin => ∂ₙ(u) => T(0.5), :ymax => u => x -> x[1]))
        Cv = constrained(W, u -> (:xmin => u + ∂ₙ(u) => o, :xmax => ∂ₙ(u) => o,
            :ymin => ∂ₙ(u) => -o, :ymax => u => x -> x[1]))
        K = T[2 1; 1 2]
        Wb = gridspace(Ωₕ; ghosts = :boundary)
        Ccon = constrained(Wb, u -> map(F -> F => ∂ₙ(u; K) => x -> x[1] - x[2],
            (:xmin, :xmax, :ymin, :ymax)))
        return (("mixed Dirichlet–Neumann–Robin, two closures", Cmix, Cv),
            ("conormal", Ccon, Ccon))
    else
        W = gridspace(Ωₕ; ghosts = (:xmin, :zmax))
        C = constrained(W, u -> (:xmin => ∂ₙ(u) => o, :zmax => u + ∂ₙ(u) => x -> x[2],
            :ymin => u => o))
        return (("Neumann, Robin and u => g", C, C),)
    end
end

const DIMS = WITH_3D_TESTS ? (1, 2, 3) : (1, 2)

@testset "matrix and load vs padded reference" begin
    for D in DIMS, T in (Float64, Float32), (name, Cu, Cv) in cases(D, T)
        tol = T == Float64 ? 1e-12 : 2e-5
        σ = T(3) / hₘₐₓ(Bramble.physical_mesh(Bramble.mesh(Cv)))
        ref = reference(Cu, Cv, σ)
        a = rank_form(Cu, Cv, σ)
        l = form(Cv, v -> innerₕ(fl, v))
        @testset "$(D)D $T $name" begin
            @test norm(ref.du) > 0                 # the load's lift is exercised
            K = assemble(a)
            @test K isa RankOneUpdate{T}
            @test K.A isa SparseMatrixCSC{T}
            @test size(K) == size(ref.K)
            @test nnz(K.A) == nnz(assemble(form(Cu, Cv, lap)))   # no dense block
            @test K.σ == σ
            @test relerr(Matrix(K), ref.K) <= tol
            @test relerr(Matrix(K.A), ref.A) <= tol
            @test relerr(K.ws, ref.ws) <= tol
            @test relerr(K.wt, ref.wt) <= tol
            K2, F = assemble(a, l)
            @test Matrix(K2) == Matrix(K)
            @test eltype(F) == T
            @test relerr(F, ref.F) <= tol
            # the term's lift is the difference from the load without it
            @test relerr(ref.Fplain - F, σ * dot(ref.wtp, ref.du) * ref.ws) <= 10tol
            poison!(K)
            @test refill(K, a) == 0
            @test relerr(Matrix(K), ref.K) <= tol
            poison!(K2, F)
            @test refill(K2, F, a, l) == 0
            @test relerr(Matrix(K2), ref.K) <= tol
            @test relerr(F, ref.F) <= tol
        end
    end
end

@testset "trial and test spaces differ" begin
    # The trial space fixes the xmax nodes by `u => g`, the test space solves their ghosts:
    # the matrix is rectangular, so swapping the two closures or vectors cannot fit.
    Ωₕ = rmesh(2, Float64)
    W = gridspace(Ωₕ; ghosts = (:xmin, :xmax))
    Cu = constrained(W, u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u => x -> x[2],
        :xmax => ∂ₙ(u) => 0.5))
    Cv = constrained(W, u -> (:xmin => 2.0 * u + ∂ₙ(u) => 1.0, :xmax => ∂ₙ(u) => 0.0))
    @test Bramble.ndofs(Cu) != Bramble.ndofs(Cv)
    σ = 2.5
    ref = reference(Cu, Cv, σ)
    a = rank_form(Cu, Cv, σ)
    l = form(Cv, v -> innerₕ(fl, v))
    K, F = assemble(a, l)
    @test size(K) == (Bramble.ndofs(Cv), Bramble.ndofs(Cu))
    @test length(K.ws) == Bramble.ndofs(Cv) && length(K.wt) == Bramble.ndofs(Cu)
    @test relerr(K.ws, ref.ws) <= 1e-12
    @test relerr(K.wt, ref.wt) <= 1e-12
    @test relerr(Matrix(K), ref.K) <= 1e-12
    @test relerr(F, ref.F) <= 1e-12
    @test relerr(Matrix(assemble(a)), ref.K) <= 1e-12
end

@testset "term alone, live Ref scale" begin
    Ωₕ = rmesh(2, Float64; seed = 3)
    C = constrained(gridspace(Ωₕ; ghosts = (:ymax,)), u -> :ymax => u + ∂ₙ(u) => x -> x[1])
    zero_form = (u, v) -> 0.0 * innerₕ(u, v)
    ref = reference(C, C, 1.5; f = zero_form)
    a = form(C, C, (u, v) -> 1.5 * innerₕ(gt, u) * innerₕ(hs, v))
    K, F = assemble(a, form(C, v -> innerₕ(fl, v)))
    @test relerr(Matrix(K), ref.K) <= 1e-12
    @test relerr(F, ref.F) <= 1e-12
    r = Ref(2.0)
    b = form(C, C, (u, v) -> lap(u, v) + r * innerₕ(gt, u) * innerₕ(hs, v))
    Kb = assemble(b)
    r[] = 5.0
    @test refill(Kb, b) == 0
    @test Kb.σ == 5.0
    @test relerr(Matrix(Kb), reference(C, C, 5.0).K) <= 1e-12
    # the pair's refill reads σ too, in the matrix and in the load's lift (`d ≠ 0`)
    lb = form(C, v -> innerₕ(fl, v))
    Kp, Fp = assemble(b, lb)
    r[] = 7.0
    poison!(Kp, Fp)
    @test refill(Kp, Fp, b, lb) == 0
    ref7 = reference(C, C, 7.0)
    @test norm(ref7.du) > 0
    @test Kp.σ == 7.0
    @test relerr(Matrix(Kp), ref7.K) <= 1e-12
    @test relerr(Fp, ref7.F) <= 1e-12
end

@testset "ghost values never read: sentinel data" begin
    # The padded functionals weigh physical nodes only: data of 1e300 at the ghosts and
    # beyond the domain give the same system as data of 1 there.
    Ωₕ = rmesh(2, Float64; seed = 11)
    C = constrained(gridspace(Ωₕ; ghosts = (:xmin, :ymax)),
        u -> (:xmin => ∂ₙ(u) => 1.0, :ymax => u + ∂ₙ(u) => 0.5, :xmax => u => x -> x[2]))
    P = padded_space(C)
    G = Bramble.mesh(P)
    lin = LinearIndices(Bramble.npoints(G, Tuple))
    phys = falses(length(lin))
    phys[vec([lin[J] for J in Bramble.physical_indices(G)])] .= true
    @test !all(phys)
    systems = map((1.0, 1.0e300)) do s
        gₕ = element(P)
        parent(gₕ) .= ifelse.(phys, 2.0, s)
        out = x -> all(c -> 0 <= c <= 1, x) ? 1 + x[1] : s
        a = form(C, C, (u, v) -> lap(u, v) + 2.0 * innerₕ(gₕ, u) * innerₕ(out, v))
        l = form(C, v -> innerₕ(out, v))
        K, F = assemble(a, l)
        (Matrix(K), F)
    end
    @test all(isfinite, systems[2][1]) && all(isfinite, systems[2][2])
    @test systems[1][1] == systems[2][1]
    @test systems[1][2] == systems[2][2]
end

@testset "dirichlet rows and symmetrize" begin
    # Rows are the physical nodes: every ghost eliminated, no `u => g`.
    Ωₕ = rmesh(2, Float64; seed = 5)
    C = constrained(gridspace(Ωₕ; ghosts = (:xmin, :ymax)),
        u -> (:xmin => ∂ₙ(u) => 1.0, :ymax => u + ∂ₙ(u) => x -> x[1]))
    σ = 4.0
    ref = reference(C, C, σ)
    a = rank_form(C, C, σ)
    l = form(C, v -> innerₕ(fl, v))
    gD = x -> 3.0
    Ω0 = Bramble.physical_mesh(Bramble.mesh(C))
    n = Bramble.npoints(Ω0, Tuple)
    mask = vec([J[1] == n[1] for J in CartesianIndices(n)])
    rows = findall(mask)
    @test !isempty(rows)
    # the matrix alone: Dirichlet rows of `K` are identity rows, `wₛ` zero there
    K = assemble(a; dirichlet = :xmax)
    Kd = copy(ref.K)
    Kd[rows, :] .= 0
    Kd[rows, rows] .= Matrix(I, length(rows), length(rows))
    @test relerr(Matrix(K), Kd) <= 1e-12
    @test all(iszero, K.ws[rows]) && !any(iszero, K.ws[.!mask])
    @test relerr(K.wt, ref.wt) <= 1e-12          # columns keep their entries
    poison!(K)
    assemble!(K, a; dirichlet = :xmax)
    @test relerr(Matrix(K), Kd) <= 1e-12
    @test (assemble!(K, a; dirichlet = :xmax); @allocated assemble!(K, a; dirichlet = :xmax)) == 0
    # the pair: the load's Dirichlet rows hold the data, the others the lifted load
    Kp, F = assemble(a, l; dirichlet = :xmax => gD)
    @test relerr(Matrix(Kp), Kd) <= 1e-12
    @test all(==(3.0), F[rows])
    @test relerr(F[.!mask], ref.F[.!mask]) <= 1e-12
    # symmetrize: the same solution, the Dirichlet columns cleared off their rows
    Ks, Fs = assemble(a, l; dirichlet = :xmax => gD, symmetrize = true)
    Ms = Matrix(Ks)
    @test all(iszero, Ms[.!mask, rows])
    @test Ms[rows, rows] == Matrix(I, length(rows), length(rows))
    @test all(iszero, Ks.wt[rows]) && all(iszero, Ks.ws[rows])
    @test relerr(Ms \ Fs, Matrix(Kp) \ F) <= 1e-10
    poison!(Ks, Fs)
    @test refill_rows(Ks, Fs, a, l, :xmax => gD) == 0
    @test relerr(Matrix(Ks), Ms) <= 1e-12
    @test relerr(Ms \ Fs, Matrix(Kp) \ F) <= 1e-10
    # two labels: the columns of both move into the load
    dir2 = (:xmax => gD, :ymin => gD)
    mask2 = mask .| vec([J[2] == 1 for J in CartesianIndices(n)])
    rows2 = findall(mask2)
    K2, F2 = assemble(a, l; dirichlet = dir2)
    Ks2, Fs2 = assemble(a, l; dirichlet = dir2, symmetrize = true)
    @test iszero(Ks2.wt[rows2]) && iszero(Ks2.ws[rows2])
    @test iszero(Matrix(Ks2)[.!mask2, rows2])
    @test relerr(Matrix(Ks2) \ Fs2, Matrix(K2) \ F2) <= 1e-10
    # by hand, as on a plain space
    Kh, Fh = assemble(a, l; dirichlet = :xmax => gD)
    symmetrize!(Kh, Fh, C, :xmax)
    @test relerr(Matrix(Kh), Ms) <= 1e-12
    @test relerr(Fh, Fs) <= 1e-12
end

@testset "lift before Dirichlet rows" begin
    # A trial functional reading ghost values, `innerₕ(g, D₋ₓ(u))`, gives `wₜ_padᵀ d ≠ 0` on a
    # Robin and Neumann space, whose rows `dirichlet` may replace: those rows of `F` hold the
    # data exactly, the term's lift taken before them, and the other rows the lifted load.
    for D in (1, 2)
        Ωₕ = rmesh(D, Float64; seed = 5)
        C = constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax)),
            u -> (:xmin => 3.0 * u + ∂ₙ(u) => 1.0, :xmax => ∂ₙ(u) => 4.0))
        gD = u -> innerₕ(x -> 1 + x[1], D₋ₓ(u))
        a = form(C, C, (u, v) -> lap(u, v) + 2.0 * gD(u) * innerₕ(hs, v))
        l = form(C, v -> innerₕ(fl, v))
        P = padded_space(C)
        E, d = closure(C)
        wtp = assemble(form(P, gD))
        @test abs(dot(wtp, d)) > 1e-3
        Ar = Matrix(assemble(form(P, P, lap))) + 2.0 * assemble(form(P, v -> innerₕ(hs, v))) * wtp'
        Fr = E' * (assemble(form(P, v -> innerₕ(fl, v))) - Ar * d)
        n = Bramble.npoints(Bramble.physical_mesh(Bramble.mesh(C)), Tuple)
        mask = vec([J[1] == n[1] for J in CartesianIndices(n)])
        K, F = assemble(a, l; dirichlet = :xmax => (x -> 7.0))
        @test all(==(7.0), F[mask])
        @test relerr(F[.!mask], Fr[.!mask]) <= 1e-12
        poison!(K, F)
        assemble!(K, F, a, l; dirichlet = :xmax => (x -> 7.0))
        @test all(==(7.0), F[mask])
        @test relerr(F[.!mask], Fr[.!mask]) <= 1e-12
    end
end

@testset "time-dependent relation data" begin
    # `assemble(a)` reads `E` only and works; the load needs `d(t)` and is refused as for
    # a form without the term.
    Ωₕ = rmesh(2, Float64; seed = 9)
    C = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
        u -> (:xmin => ∂ₙ(u) => (x, t) -> t * x[2], :xmax => u => 1.0))
    Ct = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => u => 1.0))
    a = rank_form(C, C, 2.0)
    l = form(C, v -> innerₕ(fl, v))
    @test relerr(Matrix(assemble(a)), reference(Ct, Ct, 2.0).K) <= 1e-12
    K = assemble(a)
    @test occursin("depend on time", msg(() -> assemble(a, l)))
    @test occursin("depend on time", msg(() -> assemble!(K, zeros(size(K, 1)), a, l)))
    @test occursin("no semidiscretisation", msg(() -> semidiscretize(a, l)))
end

@testset "refusals" begin
    Ωₕ = rmesh(2, Float64; seed = 13)
    C = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
        u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u => 2.0))
    a = rank_form(C, C, 2.0)
    l = form(C, v -> innerₕ(fl, v))
    K, F = assemble(a, l)
    # the load alone still needs the bilinear form when d ≠ 0
    @test occursin("needs the bilinear form", msg(() -> assemble(l)))
    target = "assembles into the RankOneUpdate"
    @test occursin(target, msg(() -> assemble!(copy(K.A), a)))
    @test occursin(target, msg(() -> assemble!(copy(K.A), copy(F), a, l)))
    @test occursin(target, msg(() -> assemble!(K, form(C, C, lap))))
    @test occursin("does not take a form holding a rank-one term",
        msg(() -> matrix_free_operator(a)))
    # `dirichlet` on a space whose unknowns are not the physical nodes
    @test occursin("given both ways", msg(() -> assemble(a; dirichlet = :xmax)))
    @test occursin("given both ways", msg(() -> assemble(a, l; dirichlet = :xmax => (x -> 2.0))))
    @test occursin("given both ways",
        msg(() -> assemble!(K, F, a, l; dirichlet = :xmax => (x -> 2.0))))
    @test occursin("given both ways", msg(() -> symmetrize!(K, F, C, :xmax)))
    @test occursin("without", msg(() -> assemble(a, l; symmetrize = true)))
    # a refill target of another pattern, or a load of a narrower type
    b = form(C, C, (u, v) -> innerₕ(u, v) + 2.0 * innerₕ(gt, u) * innerₕ(hs, v))
    @test occursin("pattern", msg(() -> assemble!(assemble(b), F, a, l)))
    @test occursin("element type", msg(() -> assemble!(K, Float32.(F), a, l)))
    @test_throws DimensionMismatch assemble!(K, zeros(length(F) + 1), a, l)
    # a load on another space
    C2 = constrained(gridspace(rmesh(2, Float64; seed = 14); ghosts = (:xmin,)),
        u -> (:xmin => ∂ₙ(u) => 1.0,))
    @test_throws DimensionMismatch assemble(a, form(C2, v -> innerₕ(fl, v)))
    # a plain load with a ghost form, and a plain space beside a ghost one
    W = gridspace(Ωₕ)
    @test_throws MethodError assemble(a, form(W, v -> innerₕ(fl, v)))
    @test occursin("ghost spaces on both sides",
        msg(() -> form(C, W, (u, v) -> lap(u, v) + innerₕ(1, u) * innerₕ(1, v))))
end

end
