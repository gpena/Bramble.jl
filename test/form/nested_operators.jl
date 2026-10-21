module FormNestedOperatorsTests

using Test
using Bramble
using Random
using LinearAlgebra: dot
using ..TestUtils: @test_allocs, _grid_sample, _random_element, WITH_SLOW_TESTS
import Bramble: D₋ₓ, D₊ₓ, Dcₓ, D̃ₓ, D̽ₓ, Mₓ, M₊ₓ, Mcₓ, jumpₓ, D₋ᵧ, Mcᵧ, D̽ᵧ, restrict_to,
                jumpₙ

# A random non-uniform mesh on the unit square (or interval): the relabelling bug this file
# guards against is invisible on a uniform one.
function _nonuniform_space(D)
    dom = D == 1 ? domain(interval(0.0, 1.0)) : domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Ωₕ = D == 1 ? mesh(dom, 11, false) : mesh(dom, (11, 9), (false, false))
    return gridspace(Ωₕ)
end

const XOPS = (("D₋ₓ", D₋ₓ), ("D₊ₓ", D₊ₓ), ("Dcₓ", Dcₓ), ("D̃ₓ", D̃ₓ), ("D̽ₓ", D̽ₓ),
    ("Mₓ", Mₓ), ("M₊ₓ", M₊ₓ), ("Mcₓ", Mcₓ), ("jumpₓ", jumpₓ))
const YOPS = (("D₋ᵧ", D₋ᵧ), ("Mcᵧ", Mcᵧ), ("D̽ᵧ", D̽ᵧ))

# `op1(op2(·))` in a form on each side, against the runtime composition on grid functions.
# The full XOPS×XOPS (plus, in 2D, XOPS×YOPS and YOPS×XOPS) grid is 81/135 pairs -- 648
# fresh form compilations across both dimensions, ~309 s and 1.5 GB locally -- about half
# the Forms group's time, and the largest addition when the macOS CI unit job began being
# killed. The cover is not a sample: a cyclic walk (op i with op i+1 and op i+3, wrapping)
# puts every operator through as outer twice and as inner twice, at a fraction of the cost.
# `unit` runs the cover, `slow` the cover plus a random draw from the rest, and `grids` all
# of it (`TestUtils._grid_sample`, gpena/Bramble.jl#789).
function _pairs(D)
    ops = D == 1 ? XOPS : (XOPS..., YOPS...)
    n = length(ops)
    cover = Tuple{Tuple{String, Function}, Tuple{String, Function}}[]
    for i in 1:n
        push!(cover, (ops[i], ops[mod1(i + 1, n)]))
        push!(cover, (ops[i], ops[mod1(i + 3, n)]))
    end
    full = [(a, b) for a in XOPS for b in XOPS]
    D == 2 && append!(full, [(a, b) for a in XOPS for b in YOPS],
        [(a, b) for a in YOPS for b in XOPS])
    return _grid_sample("op1(op2(u)) $(D)D", union(full, cover), in(cover), D == 1 ? 12 : 16)
end

function _check_pair(Wₕ, u, w, f, o1, o2)
    op = v -> o1(o2(v))
    A = assemble(form(Wₕ, Wₕ, (p, q) -> innerₕ(op(p), q)))
    @test dot(parent(w), A*parent(u))≈innerₕ(op(u), w) rtol=1e-10 atol=1e-12
    B = assemble(form(Wₕ, Wₕ, (p, q) -> innerₕ(p, op(q))))
    @test dot(parent(w), B*parent(u))≈innerₕ(u, op(w)) rtol=1e-10 atol=1e-12
    F = assemble(form(Wₕ, q -> innerₕ(f, op(q))))
    @test dot(F, parent(w))≈innerₕ(f, op(w)) rtol=1e-10 atol=1e-12
end

@testset "Nested stencil operators in forms (#287)" begin
    Random.seed!(2877)
    for D in (1, 2)
        Wₕ = _nonuniform_space(D)
        u, w, f = _random_element(Wₕ), _random_element(Wₕ), _random_element(Wₕ)
        pairs = _pairs(D)
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
        @test dot(parent(w), A*parent(u))≈innerₕ(Mₓ(D₋ₓ(um)), w) rtol=1e-10 atol=1e-12
        assemble!(A, a)
        @test_allocs assemble!(A, a)
    end

    @testset "nested form: refill allocates nothing" begin
        Wₕ = _nonuniform_space(2)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(Mcₓ(D₋ₓ(u)), D₊ₓ(Mₓ(v))))
        A = assemble(a)
        B = copy(A)
        assemble!(B, a)
        @test A ≈ B
        @test_allocs assemble!(B, a)
    end
end

# The Laplacian `Δₕ` (gpena/Bramble.jl#774) nested with every operator, as the outer one and
# as the inner one, and with itself: the same `_check_pair` oracle, the runtime composition
# on grid functions. The cover takes each of D₋ₓ, Dcₓ, Mₓ and jumpₓ (and in 2D D₋ᵧ and Mcᵧ)
# once on either side, plus `Δₕ(Δₕ(u))`; the rest of the operators are drawn as for `_pairs`.
function _laplacian_pairs(D)
    ops = D == 1 ? XOPS : (XOPS..., YOPS...)
    lap = ("Δₕ", Δₕ)
    pairs = Tuple{Tuple{String, Function}, Tuple{String, Function}}[(lap, lap)]
    for op in ops
        push!(pairs, (lap, op), (op, lap))
    end
    cover_ops = D == 1 ? (XOPS[1], XOPS[3], XOPS[6], XOPS[9]) :
                (XOPS[1], XOPS[3], XOPS[6], XOPS[9], YOPS[1], YOPS[2])
    incover(p) = p == (lap, lap) || p[1] in cover_ops || p[2] in cover_ops
    return _grid_sample("Δₕ pairs $(D)D", pairs, incover, 4)
end

# `jumpₙ(op(u))` in `inner_Γ` on a plain space: the inward jump of the runtime `op(u)` along
# the facet's axis, zero elsewhere (the facet weight selects the nodes on the facet).
function _inward_jump(g, d, side)
    G = zeros(eltype(g), size(g))
    n = size(g, d)
    if side == 1
        selectdim(G, d, 1:(n - 1)) .= selectdim(g, d, 2:n) .- selectdim(g, d, 1:(n - 1))
    else
        selectdim(G, d, 2:n) .= selectdim(g, d, 1:(n - 1)) .- selectdim(g, d, 2:n)
    end
    return G
end

const FACET_NAMES = ((:xmin, 1, 1), (:xmax, 1, 2), (:ymin, 2, 1), (:ymax, 2, 2))
const JUMP_KINDS_2D = (("Δₕ", :xmin), ("D̽ᵧ", :ymin))

# `jumpₙ(op(u))` on facet `F`, trial and test side, against the same jump applied to the
# runtime `op(u)` and weighed by `inner_Γ` on a grid function.
function _check_jump(Wₕ, dims, u, w, op, F, d, s)
    jump_data(x, d, s) = element(Wₕ, vec(_inward_jump(reshape(parent(op(x)), dims), d, s)))
    A = assemble(form(Wₕ, Wₕ, (p, q) -> inner_Γ(jumpₙ(op(p)), q; markers = (F,))))
    Gu = jump_data(u, d, s)
    l = assemble(form(Wₕ, q -> inner_Γ(Gu, q; markers = (F,))))
    @test dot(parent(w), A*parent(u))≈dot(l, parent(w)) rtol=1e-10 atol=1e-12
    B = assemble(form(Wₕ, Wₕ, (p, q) -> inner_Γ(p, jumpₙ(op(q)); markers = (F,))))
    Gw = jump_data(w, d, s)
    m = assemble(form(Wₕ, q -> inner_Γ(Gw, q; markers = (F,))))
    @test dot(parent(w), B*parent(u))≈dot(m, parent(u)) rtol=1e-10 atol=1e-12
end

# In unit the 2D block keeps one case per composition kind: Δₕ∘Δₕ, Δₕ∘XOP, XOP∘Δₕ, Δₕ∘YOP
# and YOP∘Δₕ. The other 2D pairs run in `slow` and `grids` only.
const LAP_KINDS_2D = (("Δₕ", "Δₕ"), ("Δₕ", "D₋ₓ"), ("D₋ₓ", "Δₕ"), ("Δₕ", "D₋ᵧ"),
    ("D₋ᵧ", "Δₕ"))

@testset "Δₕ nested in forms (#774)" begin
    Random.seed!(7741)
    for D in (1, 2)
        Wₕ = _nonuniform_space(D)
        u, w, f = _random_element(Wₕ), _random_element(Wₕ), _random_element(Wₕ)
        @testset "$(D)D" begin
            for ((n1, o1), (n2, o2)) in _laplacian_pairs(D)
                (WITH_SLOW_TESTS || D == 1 || (n1, n2) in LAP_KINDS_2D) &&
                    @testset "$n1($n2(u))" begin
                        _check_pair(Wₕ, u, w, f, o1, o2)
                    end
            end
        end
    end
end

@testset "jumpₙ nested in inner_Γ (#205)" begin
    Random.seed!(2057)
    for D in (1, 2)
        Wₕ = _nonuniform_space(D)
        dims = D == 1 ? (11,) : (11, 9)
        u, w = _random_element(Wₕ), _random_element(Wₕ)
        # Each (operator, facet) check is two fresh compilations: the cover is two operators
        # on every facet, and the rest are drawn as for `_pairs`.
        ops = D == 1 ? (("Δₕ", Δₕ), ("Dcₓ", Dcₓ), ("D₋ₓ", D₋ₓ)) :
              (("Δₕ", Δₕ), ("D̽ᵧ", D̽ᵧ), ("D₋ₓ", D₋ₓ), ("Dcₓ", Dcₓ), ("D₋ᵧ", D₋ᵧ))
        checks = [(op, facet) for op in ops for facet in FACET_NAMES[1:(2D)]]
        incover(c) = c[1] in ops[1:2]
        cases = _grid_sample("jumpₙ $(D)D", checks, incover, D == 1 ? 1 : 4)
        # In unit the 2D block keeps one facet per composition kind, jumpₙ∘Δₕ and jumpₙ∘D̽ᵧ.
        kept = filter(c -> D == 1 || WITH_SLOW_TESTS || (c[1][1], c[2][1]) in JUMP_KINDS_2D,
            cases)
        @testset "$(D)D jumpₙ($name(u)) on $F" for ((name, op), (F, d, s)) in kept
            _check_jump(Wₕ, dims, u, w, op, F, d, s)
        end
    end
end

end
