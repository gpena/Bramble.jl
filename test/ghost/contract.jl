module GhostContractTests

# `a(u, v)`, `l(v)` and `evaluate!` on forms over ghost spaces give the values of the
# reduced system (#781): `vᴴ A u` and `Fᵀ v` for `A = E_vᵀ A_pad E_u` and `F = E_vᵀ F_pad`,
# checked against dense products of the closure and the padded system, and without
# assembling the reduced matrix. A closure with an affine part `d ≠ 0` has no load alone,
# so `l(v)` and `evaluate!` refuse it naming `assemble(a, l)`.

using Test
using Bramble
import Bramble
using Bramble: evaluate!
using LinearAlgebra: dot
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
end

end
