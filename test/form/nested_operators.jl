module FormNestedOperatorsTests

using Test
using Bramble
using Random
using LinearAlgebra: dot
using ..TestUtils: @test_allocs
import Bramble: D₋ₓ, D₊ₓ, Dcₓ, D̽ₓ, Dₕₓ, Mₓ, M₊ₓ, Mcₓ, jumpₓ, D₋ᵧ, Mcᵧ, Dₕᵧ, restrict_to

# A random non-uniform mesh on the unit square (or interval): the relabelling bug this file
# guards against (gpena/Bramble.jl#287, S7) is invisible on a uniform one.
function _nonuniform_space(D)
    dom = D == 1 ? domain(interval(0.0, 1.0)) : domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Ωₕ = D == 1 ? mesh(dom, 11, false) : mesh(dom, (11, 9), (false, false))
    return gridspace(Ωₕ)
end

_random_element(Wₕ) = (uₕ = element(Wₕ); parent(uₕ) .= randn(length(parent(uₕ))); uₕ)

const XOPS = (("D₋ₓ", D₋ₓ), ("D₊ₓ", D₊ₓ), ("Dcₓ", Dcₓ), ("D̽ₓ", D̽ₓ), ("Dₕₓ", Dₕₓ),
              ("Mₓ", Mₓ), ("M₊ₓ", M₊ₓ), ("Mcₓ", Mcₓ), ("jumpₓ", jumpₓ))
const YOPS = (("D₋ᵧ", D₋ᵧ), ("Mcᵧ", Mcᵧ), ("Dₕᵧ", Dₕᵧ))

# `op1(op2(·))` in a form on each side, against the runtime composition on grid functions.
function _check_pair(Wₕ, u, w, f, o1, o2)
    op = v -> o1(o2(v))
    A = assemble(form(Wₕ, Wₕ, (p, q) -> innerₕ(op(p), q)))
    @test dot(parent(w), A * parent(u)) ≈ innerₕ(op(u), w) rtol = 1e-10 atol = 1e-12
    B = assemble(form(Wₕ, Wₕ, (p, q) -> innerₕ(p, op(q))))
    @test dot(parent(w), B * parent(u)) ≈ innerₕ(u, op(w)) rtol = 1e-10 atol = 1e-12
    F = assemble(form(Wₕ, q -> innerₕ(f, op(q))))
    @test dot(F, parent(w)) ≈ innerₕ(f, op(w)) rtol = 1e-10 atol = 1e-12
end

@testset "Nested stencil operators in forms (#287)" begin
    Random.seed!(2877)
    for D in (1, 2)
        Wₕ = _nonuniform_space(D)
        u, w, f = _random_element(Wₕ), _random_element(Wₕ), _random_element(Wₕ)
        pairs = [(a, b) for a in XOPS for b in XOPS]
        D == 2 && append!(pairs, [(a, b) for a in XOPS for b in YOPS],
            [(a, b) for a in YOPS for b in XOPS])
        @testset "$(D)D" begin
            for ((n1, o1), (n2, o2)) in pairs
                @testset "$n1($n2(u))" begin
                    _check_pair(Wₕ, u, w, f, o1, o2)
                end
            end
        end
    end

    # A tap reaching a restricted operand reads the region at the neighbour, not at the
    # point itself; oracle: the operator applied to `u` zeroed outside `:interior`.
    @testset "$(D)D Mₓ(D₋ₓ(restrict_to(:interior, u)))" for D in (1, 2)
        Wₕ = _nonuniform_space(D)
        u, w = _random_element(Wₕ), _random_element(Wₕ)
        a = form(Wₕ, Wₕ, (p, q) -> innerₕ(Mₓ(D₋ₓ(restrict_to(:interior, p))), q))
        A = assemble(a)
        um = copy(u)
        interior = markers(mesh(Wₕ))[:interior]
        parent(um)[.!vec(collect(interior))] .= 0
        @test dot(parent(w), A * parent(u)) ≈ innerₕ(Mₓ(D₋ₓ(um)), w) rtol = 1e-10 atol = 1e-12
        assemble!(A, a)
        @test_allocs assemble!(A, a)
    end

    @testset "refilling a nested form allocates nothing" begin
        Wₕ = _nonuniform_space(2)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(Mcₓ(D₋ₓ(u)), D₊ₓ(Mₓ(v))))
        A = assemble(a)
        B = copy(A)
        assemble!(B, a)
        @test A ≈ B
        @test_allocs assemble!(B, a)
    end
end

end
