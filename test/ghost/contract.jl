module GhostContractTests

# `a(u, v)`, `l(v)` and `evaluate!` on forms over ghost spaces give the values of the
# reduced system (#781): `vᴴ A u` and `Fᵀ v` for `A = E_vᵀ A_pad E_u` and `F = E_vᵀ F_pad`,
# checked against dense products of the closure and the padded system, and without
# assembling the reduced matrix. A closure with an affine part `d ≠ 0` has no load alone,
# so `l(v)` and `evaluate!` refuse it naming `assemble(a, l)`.

using Test
using Bramble
import Bramble
using Bramble: evaluate!, lift
using LinearAlgebra: dot, norm
using Random: Xoshiro

using ..TestUtils: WITH_3D_TESTS

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND

const FACETS = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)

# A seeded non-uniform mesh on the unit `D`-cube: random in 1D and 3D, graded in 2D.
function nonuniform_mesh(D::Int)
    D == 1 && return O.O1.make_mesh(:random, 0; n0 = 9)
    D == 2 && return O.make_mesh(:graded, 0; dim = 2, n0 = 7)
    return O.make_mesh(:random, 0; dim = 3, n0 = 5)
end

# Every facet but the last ghosted: a Robin relation on the first, Neumann on the others,
# and a Dirichlet relation `u => g` on the last, with data `g` (all zero by default).
function ghost_space(Ωₕ, D::Int; g = zeros(2D), β = 2.0)
    fs = FACETS[1:(2D)]
    rel = u -> ntuple(
        k -> k == 1 ? fs[k] => β * u + ∂ₙ(u) => g[k] :
             k == 2D ? fs[k] => u => g[k] : fs[k] => ∂ₙ(u) => g[k], 2D)
    return constrained(gridspace(Ωₕ; ghosts = fs[1:(end - 1)]), rel)
end

const f_a = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)
const f_c = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + (1.0 + 0.5im) * innerₕ(u, v)
const f_l = v -> innerₕ(x -> 1 + x[1], v)

# The reduced matrix and load from the padded system and the dense closure, sharing
# nothing with the contraction or the reduction map.
function reference(C, fa = f_a)
    P = Bramble.padded_space(C)
    E = Matrix(C.E)
    A_pad = Matrix(assemble(form(P, P, fa)))
    F_pad = assemble(form(P, f_l))
    return transpose(E) * A_pad * E, transpose(E) * F_pad
end

function random_element(C, rng, T = Float64)
    v = element(C, T)
    v .= rand(rng, T, length(v))
    return v
end

@testset "Ghost contract: $(D)D reduced values" for D in 1:3
    (D < 3 || WITH_3D_TESTS) || continue
    Ωₕ = nonuniform_mesh(D)
    C = ghost_space(Ωₕ, D)
    a = form(C, C, f_a)
    l = form(C, f_l)
    A_ref, F_ref = reference(C)
    rng = Xoshiro(D)
    u, v = random_element(C, rng), random_element(C, rng)

    @test a(u, v) ≈ dot(parent(v), A_ref * parent(u)) rtol = 1e-12
    @test a(u, v) ≈ dot(parent(v), assemble(a) * parent(u)) rtol = 1e-12
    @test a(parent(u), parent(v)) == a(u, v)                 # plain vectors too
    @test l(v) ≈ dot(F_ref, parent(v)) rtol = 1e-12
    scratch = fill(1e300, ndofs(C))
    @test evaluate!(scratch, l, v) ≈ dot(F_ref, parent(v)) rtol = 1e-12
    @test scratch ≈ F_ref rtol = 1e-12                       # the reduced load
end

@testset "Ghost contract: assembles nothing" begin
    C = ghost_space(nonuniform_mesh(2), 2)
    a = form(C, C, f_a)
    l = form(C, f_l)
    rng = Xoshiro(7)
    u, v = random_element(C, rng), random_element(C, rng)
    A, F = assemble(a), assemble(l)
    a(u, v), l(v)
    # Assembling the reduced matrix would cost at least its size; the padded vectors
    # `E u` and `E v` already exceed a quarter of it on this mesh.
    @test (@allocated a(u, v)) < Base.summarysize(A) ÷ 4
    @test (@allocated l(v)) < Base.summarysize(F)
    # the figures the docstrings give: the one cell of the plain form's sum, and nothing
    P = Bramble.padded_space(C)
    ap = form(P, P, f_a)
    up, vp = element(P, C.E * parent(u)), element(P, C.E * parent(v))
    ap(up, vp)
    @test (@allocated a(u, v)) == (@allocated ap(up, vp))
    @test (@allocated l(v)) == 0
    scratch = similar(F)
    evaluate!(scratch, l, v)
    @test (@allocated evaluate!(scratch, l, v)) < Base.summarysize(F)
end

@testset "Ghost contract: a closure with d ≠ 0" begin
    D = 2
    Ωₕ = nonuniform_mesh(D)
    C = ghost_space(Ωₕ, D; g = [1.0, 0.0, 0.5, 0.0])
    @test any(!iszero, C.d)
    a = form(C, C, f_a)
    l = form(C, f_l)
    rng = Xoshiro(3)
    u, v = random_element(C, rng), random_element(C, rng)
    # `a(u, v)` is the contraction of `assemble(a)`, the closure's linear part.
    A_ref, _ = reference(C)
    @test a(u, v) ≈ dot(parent(v), A_ref * parent(u)) rtol = 1e-12
    for call in (() -> l(v), () -> evaluate!(zeros(ndofs(C)), l, v))
        err = try
            call()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("assemble(a, l)", sprint(showerror, err))
    end
end

# A pair reduces with the test closure of the bilinear form and loads with the linear
# form's. Two closures of one size but different kept nodes (Dirichlet at x = 0 against
# y = 0) give an inconsistent system, so the pair is refused unless the two test spaces are
# one space or have equal closures.
@testset "Ghost contract: pairs share a closure" begin
    Ωₕ = O.make_mesh(:random, 0; dim = 2, n0 = 7)
    S = gridspace(Ωₕ; ghosts = :boundary)
    dir = F -> (u -> (F => u => 0.0, F => ∂ₙₙ(u) => 0.0,
        (G => ∂ₙ(u) => 0.0 for G in (:xmin, :xmax, :ymin, :ymax) if G !== F)...))
    Ca, Cb = constrained(S, dir(:xmin)), constrained(S, dir(:ymin))
    @test ndofs(Ca) == ndofs(Cb)
    fs = (u, v) -> innerₕ(-Δₕ(u), v) + innerₕ(u, v)
    a = form(Ca, Ca, fs)
    la, lb = form(Ca, f_l), form(Cb, f_l)
    A, F = assemble(a, la)
    @test_throws DimensionMismatch assemble(a, lb)
    @test_throws DimensionMismatch assemble!(copy(A), copy(F), a, lb)
    err = try
        assemble(a, lb)
        nothing
    catch e
        e
    end
    @test occursin("closures (E, d) are not equal", sprint(showerror, err))
    # the load on `Cb` alone is a different vector, so accepting the pair would be wrong
    @test norm(assemble(lb) - assemble(la)) > 1.0e-3 * norm(assemble(la))
    # a space rebuilt alike has an equal closure and pairs, with the same system
    Ca2 = constrained(S, dir(:xmin))
    @test Ca2 !== Ca
    @test assemble(a, form(Ca2, f_l)) == (A, F)
    A2, F2 = copy(A), fill(NaN, length(F))
    assemble!(A2, F2, a, la)
    @test F2 == F
    @test (@allocated assemble!(A2, F2, a, la)) == 0
    # data in time against zero data: one `E` and one stored `d(0)`, two relation sets
    S1 = gridspace(Ωₕ; ghosts = (:xmax,))
    C0 = constrained(S1, u -> (:xmin => u => 0.0, :xmax => ∂ₙ(u) => 0.0))
    Ct = constrained(S1, u -> (:xmin => u => (x, t) -> t, :xmax => ∂ₙ(u) => 0.0))
    @test C0.E == Ct.E && C0.d == Ct.d
    a0, lt = form(C0, C0, f_a), form(Ct, f_l)
    A0, F0 = assemble(a0, form(C0, f_l))
    @test_throws ArgumentError assemble(a0, lt)
    @test_throws ArgumentError assemble!(copy(A0), copy(F0), a0, lt)
end

# The guard of the load alone reads its test space only. Petrov-Galerkin pairs on one ghost
# set: with data on the trial side alone, `assemble(l)` is `E_vᵀ F_pad`, accepted, and the
# pair's load adds the lift `−E_vᵀ A_pad d_u`; with data on the test side alone, the pair's
# load is `E_vᵀ F_pad`, and the load alone refuses.
@testset "Ghost contract: load alone, two sides" begin
    Ωₕ = O.make_mesh(:random, 0; dim = 2, n0 = 7)
    S = gridspace(Ωₕ; ghosts = (:xmax,))
    rels = g -> (u -> (:xmin => u => g, :xmax => ∂ₙ(u) => 0.0))
    C1, C0 = constrained(S, rels(1.0)), constrained(S, rels(0.0))
    @test any(!iszero, C1.d) && all(iszero, C0.d)
    P = Bramble.padded_space(S)
    A_pad = Matrix(assemble(form(P, P, f_a)))
    F_pad = assemble(form(P, f_l))
    E1, E0 = Matrix(C1.E), Matrix(C0.E)

    # trial data, test none: the load alone is accepted and misses the lift
    l0 = form(C0, f_l)
    F_alone = assemble(l0)
    @test F_alone ≈ transpose(E0) * F_pad rtol = 1e-12
    _, F_pair = assemble(form(C1, C0, f_a), l0)
    @test F_pair ≈ transpose(E0) * (F_pad - A_pad * C1.d) rtol = 1e-12
    @test norm(F_pair - F_alone) > 1.0e-2 * norm(F_alone)

    # test data, trial none: the pair's load has no lift, and the load alone refuses
    l1 = form(C1, f_l)
    _, F_pair = assemble(form(C0, C1, f_a), l1)
    @test F_pair ≈ transpose(E1) * F_pad rtol = 1e-12
    v = random_element(C1, Xoshiro(11))
    @test_throws ArgumentError assemble(l1)
    @test_throws ArgumentError l1(v)
    @test_throws ArgumentError evaluate!(zeros(ndofs(C1)), l1, v)
end

# Two Dirichlet data in time meeting at a corner agree at `t = 0`; at any other `t` the
# corner takes the datum of the first relation given, and the other nodes of each facet
# their own.
@testset "Ghost contract: corner datum in time" begin
    Ωₕ = O.make_mesh(:random, 0; dim = 2, n0 = 7)
    S = gridspace(Ωₕ; ghosts = (:xmax, :ymax))
    for (first_rel, corner) in ((:xmin, 1.0), (:ymin, 2.0))
        rels = u -> begin
            rx = :xmin => u => (x, t) -> t
            ry = :ymin => u => (x, t) -> 2t
            ((first_rel === :xmin ? (rx, ry) : (ry, rx))..., :xmax => ∂ₙ(u) => 0.0,
                :ymax => ∂ₙ(u) => 0.0)
        end
        C = constrained(S, rels)
        G = mesh(C)
        uₕ = parent(lift(C, zeros(ndofs(C)), 1.0))
        plin = LinearIndices(npoints(G, Tuple))
        at(I) = uₕ[plin[Bramble.padded_index(G, I)]]
        @test at(CartesianIndex(1, 1)) == corner
        @test at(CartesianIndex(1, 3)) == 1.0        # :xmin
        @test at(CartesianIndex(3, 1)) == 2.0        # :ymin
    end
end

@testset "Ghost contract: data depending on time" begin
    Ωₕ = nonuniform_mesh(1)
    C = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
        u -> (:xmin => ∂ₙ(u) => (x, t) -> sin(t), :xmax => u => 0.0))
    l = form(C, f_l)
    v = random_element(C, Xoshiro(4))
    err = try
        l(v)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("semidiscretize", sprint(showerror, err))
    a = form(C, C, f_a)
    @test a(v, v) ≈ dot(parent(v), assemble(a) * parent(v)) rtol = 1e-12
end

# A complex closure needs a complex mesh, which `mesh` does not build: the closure stays
# real, the form and the vectors are complex, and `v` is conjugated as in `vᴴ A u`.
@testset "Ghost contract: complex form" begin
    T = ComplexF64
    C = ghost_space(nonuniform_mesh(2), 2)
    a = form(C, C, f_c)
    rng = Xoshiro(9)
    u, v = random_element(C, rng, T), random_element(C, rng, T)
    A_ref, _ = reference(C, f_c)
    @test a(u, v) ≈ dot(parent(v), A_ref * parent(u)) rtol = 1e-12
    @test a(u, v) ≈ dot(parent(v), assemble(a) * parent(u)) rtol = 1e-12
end

# One ghost at the end of a 1D mesh is the last padded point, the last degree of freedom:
# the closure is the identity, and each value is the plain one on the padded space.
@testset "Ghost contract: identity closure" begin
    Ωₕ = nonuniform_mesh(1)
    W = gridspace(Ωₕ; ghosts = (:xmax,))
    P = Bramble.padded_space(W)
    @test Matrix(W.E) == [i == j for i in 1:ndofs(P), j in 1:ndofs(P)]
    rng = Xoshiro(5)
    u, v = random_element(W, rng), random_element(W, rng)
    up, vp = element(P, parent(u)), element(P, parent(v))
    @test form(W, W, f_a)(u, v) == form(P, P, f_a)(up, vp)
    @test form(W, f_l)(v) ≈ form(P, f_l)(vp) rtol = 1e-14
    s, sp = zeros(ndofs(W)), zeros(ndofs(P))
    @test evaluate!(s, form(W, f_l), v) ≈ evaluate!(sp, form(P, f_l), vp) rtol = 1e-14
    @test s == sp
end

@testset "Ghost contract: lengths, bare vectors" begin
    Wg = ghost_space(nonuniform_mesh(2), 2)
    a = form(Wg, Wg, f_a)
    l = form(Wg, f_l)
    vₕ = element(Bramble.padded_space(Wg), 1.0)          # padded, not the degrees of freedom
    u = element(Wg, 1.0)
    @test_throws DimensionMismatch a(vₕ, u)
    @test_throws DimensionMismatch a(u, vₕ)
    @test_throws DimensionMismatch l(vₕ)
    @test_throws DimensionMismatch evaluate!(zeros(ndofs(Wg)), l, vₕ)
    @test_throws ArgumentError l(parent(u))
    msg = f -> try
        f()
        ""
    catch e
        sprint(showerror, e)
    end
    n = ndofs(Wg)
    @test occursin("a linear form on a ghost space with $n degrees of freedom cannot " *
                   "contract a vector of length $(length(vₕ))", msg(() -> l(vₕ)))
    @test occursin("on ghost spaces with $(n)×$n degrees", msg(() -> a(vₕ, u)))
    # a refill matrix of another size names both sizes
    @test occursin("of size (3, 3)", msg(() -> assemble!(zeros(3, 3), a)))
    @test occursin("its size ($n, $n)", msg(() -> assemble!(zeros(3, 3), a)))
    # a plain space on one side is named
    W = gridspace(nonuniform_mesh(2))
    @test occursin("the test space, ", msg(() -> form(Wg, W, f_a)))
    @test occursin("the trial space, ", msg(() -> form(W, Wg, f_a)))
end

end
