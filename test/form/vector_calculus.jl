module TestFormVectorCalculus

using Test
using Bramble
using Bramble: OperatorAdd, block_of, resolve_form_ast
using Bramble: ∇₊ₕ, div₊ₕ, curl₊ₕ
using LinearAlgebra: dot, ⋅, ×

# gpena/Bramble.jl#234 (v3.3.0 plan S6.5): `∇ₕ`/`∇₊ₕ`/`εₕ`/`divₕ` accept a composite trial or
# test function and expand, at the builder, into the same single-block products a user would
# otherwise write by hand. This file checks that expansion two ways:
#
#   1. the compact 3D elasticity form `2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₕ(u), divₕ(v))`
#      assembles to the same matrix, entry for entry, as the hand-expanded form
#      `docs/src/examples/elasticity_3d.jl` uses (transcribed below under different names, so
#      as not to shadow the package's own `εₕ`/`divₕ`); same in 2D.
#   2. `inner₊(∇ₕ(u), ∇ₕ(v))` over a composite trial/test function agrees with the same sum
#      written out term by term.
#
# and that every leaf of the assembled AST still answers `block_of`: the architecture promises
# no new node type, so the expansion must bottom out in ordinary single-block products.

# --- Transcribed from docs/src/examples/elasticity_3d.jl (S6.6 owns that file; not imported
# from it, since it is a `#src`-driven documentation page, not a library module) ------------ #

const _Dm3 = (D₋ₓ, D₋ᵧ, D₋₂)
const _Mm3 = (Mₓ, Mᵧ, M₂)
const _Dm2 = (D₋ₓ, D₋ᵧ)
const _Mm2 = (Mₓ, Mᵧ)

# `scale * wₛ / wₕ` as a grid function, where `wₛ` is the quadrature weight of a quantity
# staggered in the directions `S` and `wₕ` is the one `innerₕ` carries -- the same ratio
# `docs/src/examples/elasticity_3d.jl`'s `stagger_ratio` builds, generalised over `D` here
# since this file needs it in both 2D and 3D.
function hand_stagger_ratio(Wₕ, S, scale, ::Val{D}) where {D}
    Ωₕ = mesh(Wₕ)
    npts = npoints(Ωₕ, Tuple)
    r = fill(float(scale), npts)
    for d in S
        h = [i == 1 ? 0.0 : spacing(Ωₕ(d), i) for i in 1:npts[d]]
        ratio = h ./ [half_spacing(Ωₕ(d), i) for i in 1:npts[d]]
        r .*= reshape(ratio, ntuple(k -> k == d ? npts[d] : 1, Val(D)))
    end
    cₕ = element(Wₕ)
    copyto!(parent(cₕ), vec(r))
    return cₕ
end

# One entry of the hand-expanded strain tensor, `εₕ(p, i, j)` in the docs page.
hand_strain(p, Dm, Mm, i, j) = i == j ? Dm[i](p(i)) :
                               0.5 * Mm[i](Dm[j](p(i))) + 0.5 * Mm[j](Dm[i](p(j)))

# One term of the hand-expanded staggered divergence, `divₜ(p, i)` in the docs page.
function hand_div_term(p, Dm, Mm, i, ::Val{D}) where {D}
    foldl(
        (op, d) -> Mm[d](op), Iterators.filter(!=(i), 1:D); init = Dm[i](p(i))
    )
end

# The 27-term (3D) / 8-term (2D) hand-expanded elasticity form: every `(i, j)` pair of the
# strain and the divergence written out as its own `innerₕ` against a precomputed weight
# ratio, exactly as `docs/src/examples/elasticity_3d.jl`'s `elasticity_form` does.
function hand_elasticity_form(Vₕ, μ, λ, Dm, Mm, ::Val{D}) where {D}
    Wₕ = first(spaces(Vₕ))
    cε = Dict(
        (i == j ? (i,) : minmax(i, j)) => hand_stagger_ratio(
            Wₕ, i == j ? (i,) : minmax(i, j), 2μ, Val(D)
        )
    for i in 1:D, j in 1:D
    )
    cdiv = hand_stagger_ratio(Wₕ, ntuple(identity, Val(D)), λ, Val(D))
    return form(
        Vₕ, Vₕ,
        (p, q) -> sum(
            innerₕ(
                cε[i == j ? (i,) : minmax(i, j)] * hand_strain(p, Dm, Mm, i, j),
                hand_strain(q, Dm, Mm, i, j)
            )
        for i in 1:D, j in 1:D) +
                  sum(
            innerₕ(cdiv * hand_div_term(p, Dm, Mm, i, Val(D)), hand_div_term(q, Dm, Mm, j, Val(D)))
        for i in 1:D, j in 1:D)
    )
end

# The compact form the issue asks for.
compact_elasticity_form(Vₕ, μ, λ) = form(
    Vₕ, Vₕ, (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₕ(u), divₕ(v))
)

# The gradient-tensor inner product, hand-expanded as a plain double sum over component and
# direction, versus the compact `inner₊(∇ₕ(u), ∇ₕ(v))` over a composite trial/test function.
function hand_grad_form(Vₕ, Dm, ::Val{D}) where {D}
    form(
        Vₕ, Vₕ, (u, v) -> sum(inner₊(Dm[d](u(c)), Dm[d](v(c))) for c in 1:D, d in 1:D)
    )
end
compact_grad_form(Vₕ) = form(Vₕ, Vₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))

# --- block_of over every leaf of an assembled AST ------------------------------------------- #

_leaf_terms(op::OperatorAdd) = (_leaf_terms(op.left_op)..., _leaf_terms(op.right_op)...)
_leaf_terms(op) = (op,)

# Every leaf must route to a definite, in-range block: the architecture promises no new node
# type, so `block_of` -- which only ever understands `BilinearProduct`/`LinearProduct` and the
# wrappers around them -- must still answer for whatever the builder produced.
function check_single_block_leaves(a, D::Int)
    leaves = _leaf_terms(resolve_form_ast(a))
    @test !isempty(leaves)
    for term in leaves
        blk = block_of(term, D, D)
        @test blk !== nothing
        tc, sc = blk
        @test 1 <= tc <= D
        @test 1 <= sc <= D
    end
    return nothing
end

@testset "Composite ∇ₕ, εₕ, divₕ (S6.5)" begin
    @testset "3D elasticity: compact equals hand-expanded" begin
        Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 4, 3), (false, true, false))
        Vₕ = gridspace(Ωₕ)^Val(3)
        μ, λ = 1.7, 0.9

        a_compact = compact_elasticity_form(Vₕ, μ, λ)
        a_hand = hand_elasticity_form(Vₕ, μ, λ, _Dm3, _Mm3, Val(3))

        A_compact = assemble(a_compact)
        A_hand = assemble(a_hand)

        # Not `==`: the two sides reach the same staggered weight through different
        # arithmetic (the compact form reads it from `SpaceWeights` directly, the hand form
        # multiplies `innerₕ`'s weight by a separately computed ratio), so the last bit or
        # two of a handful of entries can differ. `atol = 1e-12` comfortably clears the
        # largest observed gap (~1e-15) while still catching a real discretisation mismatch.
        @test isapprox(A_compact, A_hand; atol = 1.0e-12)

        check_single_block_leaves(a_compact, 3)
    end

    @testset "2D elasticity: compact equals hand-expanded" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 5), (false, true))
        Vₕ = gridspace(Ωₕ)^Val(2)
        μ, λ = 2.3, 1.1

        a_compact = compact_elasticity_form(Vₕ, μ, λ)
        a_hand = hand_elasticity_form(Vₕ, μ, λ, _Dm2, _Mm2, Val(2))

        A_compact = assemble(a_compact)
        A_hand = assemble(a_hand)

        # See the 3D testset above for why `isapprox` rather than `==`.
        @test isapprox(A_compact, A_hand; atol = 1.0e-12)

        check_single_block_leaves(a_compact, 2)
    end

    @testset "inner₊(∇ₕ(u), ∇ₕ(v)) over a composite function" begin
        for (D, Dm, mk_mesh) in (
            (2, _Dm2, () -> mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 5), (true, false))),
            (3, _Dm3, () -> mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 3), (true, false, true)))
        )
            Ωₕ = mk_mesh()
            Vₕ = gridspace(Ωₕ)^Val(D)

            a_compact = compact_grad_form(Vₕ)
            a_hand = hand_grad_form(Vₕ, Dm, Val(D))

            A_compact = assemble(a_compact)
            A_hand = assemble(a_hand)

            @test A_compact == A_hand

            check_single_block_leaves(a_compact, D)
        end
    end
end

# gpena/Bramble.jl#341 (S5.4): `∇ₕ ⋅ u`/`∇ₕ × u` (and the four sibling gradient aliases)
# contract to `divₕ(u)`/`curlₕ(u)` via `LinearAlgebra.dot`/`×`. Every mesh here is
# non-uniform in every direction, as elsewhere in the suite.
const _CONTRACT_FAMILIES = (
    (∇ₕ, divₕ, curlₕ), (∇cₕ, divcₕ, curlcₕ), (∇̽ₕ, div̽ₕ, curl̽ₕ),
    (∇̃ₕ, diṽₕ, curl̃ₕ), (∇₊ₕ, div₊ₕ, curl₊ₕ)
)

@testset "∇ₕ ⋅ u and ∇ₕ × u contract to div/curl (S5.4, #341)" begin
    @testset "numeric, 2D non-uniform" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (9, 7), (false, false))
        Wₕ = gridspace(Ωₕ)
        u1 = Rₕ(Wₕ, x -> x[1]^2 + 0.3x[2])
        u2 = Rₕ(Wₕ, x -> sin(x[1]) * x[2])

        for (G, Dv, Cu) in _CONTRACT_FAMILIES
            @test parent(G ⋅ (u1, u2)) == parent(Dv((u1, u2)))
            @test parent(dot(G, (u1, u2))) == parent(Dv((u1, u2)))
            @test parent(G × (u1, u2)) == parent(Cu((u1, u2)))
        end
    end

    @testset "numeric, 3D non-uniform" begin
        Ω3 = mesh(
            domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (6, 5, 7), (false, false, false)
        )
        W3 = gridspace(Ω3)
        a = Rₕ(W3, x -> x[1] * x[2])
        b = Rₕ(W3, x -> x[3]^2 + x[1])
        c = Rₕ(W3, x -> sin(x[2]))

        for (G, Dv, Cu) in _CONTRACT_FAMILIES
            @test parent(G ⋅ (a, b, c)) == parent(Dv((a, b, c)))
            cu1 = G × (a, b, c)
            cu2 = Cu((a, b, c))
            @test all(i -> parent(cu1[i]) == parent(cu2[i]), 1:3)
        end
    end

    @testset "composite VectorElement, not just an NTuple" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (8, 6), (false, false))
        Vₕ = gridspace(Ωₕ)^Val(2)
        uₕ = Rₕ(Vₕ, (x -> x[1]^2, x -> x[1] * x[2]))

        for (G, Dv, Cu) in _CONTRACT_FAMILIES
            @test parent(G ⋅ uₕ) == parent(Dv(uₕ))
            @test parent(G × uₕ) == parent(Cu(uₕ))
        end
    end

    @testset "forms: ∇ₕ ⋅ u in a bilinear form on a composite space" begin
        # Only the backward, centered and cross-weighted families have a *symbolic* `div`
        # over a trial/test function (src/form/operators/difference.jl); `∇̃ₕ`/`∇₊ₕ` have
        # none, so `∇̃ₕ ⋅ u`/`∇₊ₕ ⋅ u` inside a form is out of scope here -- the contraction
        # still reaches whatever `diṽₕ`/`div₊ₕ` themselves support, symbolic or not. The
        # staggered `divₕ` builds an `inner₊`-only container; the collocated `divcₕ`/`div̽ₕ`
        # are plain operator sums that `innerₕ` accepts directly.
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (7, 6), (false, false))
        Vₕ = gridspace(Ωₕ)^Val(2)

        a1 = assemble(form(Vₕ, Vₕ, (u, v) -> inner₊(∇ₕ ⋅ u, ∇ₕ ⋅ v)))
        a2 = assemble(form(Vₕ, Vₕ, (u, v) -> inner₊(divₕ(u), divₕ(v))))
        @test a1 == a2

        for (G, Dv) in ((∇cₕ, divcₕ), (∇̽ₕ, div̽ₕ))
            b1 = assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(G ⋅ u, G ⋅ v)))
            b2 = assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(Dv(u), Dv(v))))
            @test b1 == b2
        end
    end

    @testset "in-place divₕ!/curlₕ! stay allocation-free" begin
        # `⋅`/`×` forward to the allocating `divₕ`/`curlₕ`, never the `!` forms; this
        # confirms the in-place paths are untouched by adding those two methods.
        function alloc_counts()
            Ωₕ = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (8, 6), (false, false)
            )
            Wₕ = gridspace(Ωₕ)
            u1 = Rₕ(Wₕ, x -> x[1]^2 + 0.3x[2])
            u2 = Rₕ(Wₕ, x -> sin(x[1]) * x[2])
            v = similar(u1)
            divₕ!(v, (u1, u2))   # warm up
            dv = @allocated divₕ!(v, (u1, u2))
            w = similar(u1)
            curlₕ!(w, (u1, u2))  # warm up
            cw = @allocated curlₕ!(w, (u1, u2))
            return dv, cw
        end

        dv, cw = alloc_counts()
        @test dv == 0
        @test cw == 0
    end
end

end # module
