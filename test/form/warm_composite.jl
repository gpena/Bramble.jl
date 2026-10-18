module FormWarmCompositeTests

# Warm assembly of the elasticity form (#765). The recording and the replays walk a product
# term factor by factor (`_factored_walk`, bilinear_traversal.jl) instead of building its
# product stencil at every point. The tests pin that the walk is taken, that it hands a sink
# the very entries the product stencil holds, bit for bit and in order, and that the 3D
# matrix is the one 303eafcb, before the change, assembled.
#
# The entry check writes each entry to its own slot: a `ReplaySink` over a dense column
# whose positions are `1:n`, with each point's slice laid out from `local_stencil`, the
# product stencil the walk no longer builds. Every mesh is non-uniform and seeded.

using Test
using Bramble
import Bramble
using Bramble: ReplaySink, _factored_walk, visit_bilinear_stencil, local_stencil,
               entry_offsets, entry_weights, matrix_free_operator, restrict_to, πₕ,
               padded_space, D₋ₓ, D₋ᵧ, Dcₓ, Mₓ, Mᵧ
using LinearAlgebra: norm
using SparseArrays: nnz
using Random: seed!
using ..TestUtils: WITH_3D_TESTS

elasticity(u, v) = 2.0 * inner₊(εₕ(u), εₕ(v)) + 2.0 * inner₊(divₘₕ(u), divₘₕ(v))

function rmesh(n; seed = 20260903)
    seed!(seed)
    S = foldl(×, ntuple(_ -> interval(0.0, 1.0), length(n)))
    return mesh(domain(S), n, ntuple(_ -> false, length(n)))
end

# Every (term, block) unit of `a`, as the replay walks it: the term (a pair's first product
# alone) and the leaf it walks.
function units(a)
    out = Any[]
    Bramble._foreach_unit(a.trial_space, a.test_space, a.ast) do term, sp, _...
        push!(out, (term, sp))
    end
    return out
end

# Whether `term` survives its entry `(off_u, off_v)` at `I`: both its row and column land.
inside(lin, I, o) = checkbounds(Bool, lin, I + CartesianIndex(o))

# The surviving entries of the product stencil at every point, laid out point after point,
# and the `point_ptr` giving each point's first slot.
function stencil_entries(term, sp)
    grid = Bramble.indices(Bramble.mesh(sp))
    lin = LinearIndices(grid)
    ptr = zeros(Int, length(lin) + 1)
    vals = Float64[]
    for I in grid
        st = local_stencil(term, sp, I, nothing, lin[I])
        ptr[lin[I]] = length(vals) + 1
        for (off, w) in zip(entry_offsets(st), entry_weights(st))
            inside(lin, I, off[1]) && inside(lin, I, off[2]) && push!(vals, w)
        end
    end
    ptr[end] = length(vals) + 1
    return ptr, vals
end

# What the walk writes, one entry per slot. `-0.0` is the additive identity for every
# value, a signed zero included, so each slot ends holding its entry exactly.
function walked_entries(term, sp, ptr, n)
    out = fill(-0.0, n, 1)
    visit_bilinear_stencil(ReplaySink(out, ptr, collect(1:n), 1.0), term, sp, 0, 0)
    return vec(out)
end

factored(term, sp) = _factored_walk(
    ReplaySink{Vector{Float64}, Vector{Int}, Float64}, typeof(term), typeof(sp))

@testset "Factored walk is taken (#765)" begin
    @test isdefined(Bramble, :_factored_walk)
    W = gridspace(rmesh((9, 7)))
    V = W^Val(2)
    us = units(form(V, V, elasticity))
    @test length(us) == 10
    @test all(((t, sp),) -> factored(t, sp), us)
    # The stencil walk stays for what the factored one does not cover.
    Wm = gridspace(mesh(
        domain(interval(0.0, 1.0) × interval(0.0, 1.0),
            :walls => boundary_symbols(interval(0.0, 1.0) × interval(0.0, 1.0))),
        (9, 7), (false, false)))
    restricted = form(Wm, Wm, (u, v) -> innerₕ(u, restrict_to(:walls, v))).ast
    @test !factored(restricted, Wm)
    Wc = gridspace(rmesh((5, 4); seed = 3))
    interp = form(Wc, W, (u, v) -> innerₕ(πₕ(u), v)).ast
    @test !factored(interp, W)
    # A ghost space's padded leaf keeps it too: its products check absent reads.
    Wg = gridspace(rmesh((9, 7)); ghosts = (:xmin,))
    C = constrained(Wg, u -> (:xmin => ∂ₙ(u) => 1.0,))
    padded = form(C, C, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))).padded
    @test !factored(padded.ast, padded_space(Wg))
    # The matrix-free sinks keep the stencil walk, so they check this one independently.
    @test !_factored_walk(Bramble.ActionSink, typeof(us[1][1]), typeof(us[1][2]))
end

refill_bytes(A, a) = (assemble!(A, a); @allocated assemble!(A, a))

# Each term's entries against its product stencil on `W`, in slot order.
function check_entries(term, W)
    @test factored(term, W)
    ptr, vals = stencil_entries(term, W)
    @test isequal(walked_entries(term, W, ptr, length(vals)), vals)
end

@testset "Factored entries, bit for bit (#765)" begin
    W = gridspace(rmesh((9, 7)))
    c = Rₕ(W, x -> 1.0 + x[1]^2 + x[2])
    s = Ref(2.5)
    V = W^Val(2)
    f(g) = form(W, W, g).ast
    terms = Any[t for (t, _) in units(form(V, V, elasticity))]
    push!(terms, f((u, v) -> s * innerₕ(u, v) + innerₕ(c * D₋ₓ(u), D₋ᵧ(v))),
        f((u, v) -> 1.5 * (inner₊(∇ₕ(u), ∇ₕ(v)) + (1 // 3) * innerₕ(u, v))),
        # A grid function scaling a whole product, read at each point: a two-entry
        # symmetric product (through its symmetric stencil) and a four-entry one (looped).
        f((u, v) -> c * innerₕ(D₋ₓ(u), D₋ₓ(v))),
        f((u, v) -> c * innerₕ(Mₓ(D₋ᵧ(u)), Mₓ(D₋ᵧ(v)))),
        # Factors whose weight matrix is not symmetric, so the entry order shows: a sum
        # with repeated offsets beside a difference, and a three-tap difference.
        f((u, v) -> innerₕ(Mₓ(u) + D₋ₓ(u), D₋ᵧ(v))),
        f((u, v) -> innerₕ(Dcₓ(u), Mᵧ(D₋ᵧ(v)))),
        # `Int` and `Float64` weights in one factor: built as the product stencil, plain
        # and symmetric.
        f((u, v) -> innerₕ(u, v + 0.1 * D₋ₓ(v))),
        f((u, v) -> innerₕ(u + 0.1 * D₋ₓ(u), v + 0.1 * D₋ₓ(v))))
    for term in terms
        check_entries(term, W)
    end
    # Mixed weights indexed by a loop variable would box at every point.
    for g in ((u, v) -> innerₕ(u, v + 0.1 * D₋ₓ(v)),
        (u, v) -> innerₕ(u + 0.1 * D₋ₓ(u), v + 0.1 * D₋ₓ(v)))
        a = form(W, W, g)
        @test refill_bytes(assemble(a), a) == 0
    end
    W1 = gridspace(rmesh((17,)))
    c1 = Rₕ(W1, x -> 2.0 + sin(3 * x[1]))
    for g in ((u, v) -> c1 * innerₕ(D₋ₓ(u), D₋ₓ(v)),
        (u, v) -> c1 * innerₕ(Mₓ(u) + D₋ₓ(u), Mₓ(v) + D₋ₓ(v)),
        (u, v) -> innerₕ(Mₓ(u) + D₋ₓ(u), Dcₓ(v)))
        check_entries(form(W1, W1, g).ast, W1)
    end
end

# The 3D form of the PROBE: 9x8x7 points, every unit at a boundary on some side.
WITH_3D_TESTS && @testset "3D elasticity matrix pinned (#765)" begin
    Vc = gridspace(rmesh((9, 8, 7)))^Val(3)
    a = form(Vc, Vc, elasticity)
    A = assemble(a)
    # 303eafcb, before the factored walk, assembled these.
    @test nnz(A) == 94050
    @test norm(A) ≈ 115.80367650755502 rtol = 1e-12
    M = matrix_free_operator(a)
    n = size(A, 2)
    for j in (1, n ÷ 3, n ÷ 2, n)
        e = zeros(n)
        e[j] = 1.0
        @test norm(M * e - A[:, j], Inf) <= 1e-12 * max(1.0, norm(A[:, j], Inf))
    end
    for (t, sp) in units(a)
        ptr, vals = stencil_entries(t, sp)
        @test isequal(walked_entries(t, sp, ptr, length(vals)), vals)
    end
    W3 = gridspace(rmesh((9, 8, 7)))
    c3 = Rₕ(W3, x -> 1.0 + x[1] * x[3])
    check_entries(form(W3, W3, (u, v) -> c3 * innerₕ(Mᵧ(D₋ₓ(u)), Mᵧ(D₋ₓ(v)))).ast, W3)
    B = copy(A)
    assemble!(B, a)
    @test (@allocated assemble!(B, a)) == 0
    @test B == A
end

end
