module FormRebindTests

# `_replace_grid_function(form, old => new)` (src/ast/rebind.jl, gpena/Bramble.jl#985): the
# form rebuilt to read `new` wherever its AST read `old`. Each rebound form is judged
# against the form written with `new` from the start, on non-uniform 2D meshes. A swap of
# one element type for itself must change the values too (`_rewrap_inner` keeps a node
# whose operand keeps its type), so the coefficients swapped differ everywhere.

using Test
using Bramble
import Bramble
using Bramble: D₋ₓ, D₋ᵧ, M₊ₓ, restrict_to, CpuSerial
using ForwardDiff: Dual, value, partials
using LinearAlgebra: ×
using ..TestUtils: ghost_mesh

const rebind = Bramble._replace_grid_function
const Dl = Dual{Nothing, Float64, 1}

ss(x) = 1.0 + x[1] + 2x[2]^2
ts(x) = 3.0 - x[1] * x[2]

# A `Dual` copy of `s` on its space, seeded with derivative one.
function seeded(s)
    sd = element(space(s), Dl)
    parent(sd) .= Dual{Nothing}.(parent(s), 1.0)
    return sd
end

@testset "rebind: plain bilinear form" begin
    W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (6, 7), (false, false)))
    s, t, r = Rₕ(W, ss), Rₕ(W, ts), Rₕ(W, x -> 5.0 + x[2])
    # `s` in two terms, one under a difference; `r` beside it, not swapped.
    f(c) = (u, v) -> inner₊(c * ∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(c * u), D₋ᵧ(v)) +
                     innerₕ(r * u, v) + innerₕ(M₊ₓ(c * u), v)
    a = form(W, W, f(s))
    A = assemble(a)
    b = rebind(a, s => t)
    @test assemble(b) == assemble(form(W, W, f(t)))
    @test assemble(b) != A
    # the original form, its cache included, is untouched by the rebound one's assembly
    @test assemble(a) == A
    B = assemble(b)
    assemble!(B, b)
    @test B == assemble(form(W, W, f(t)))

    @testset "only the matched element is replaced" begin
        b2 = rebind(a, r => t)
        g = (u, v) -> inner₊(s * ∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(s * u), D₋ᵧ(v)) +
                      innerₕ(t * u, v) + innerₕ(M₊ₓ(s * u), v)
        @test assemble(b2) == assemble(form(W, W, g))
    end

    @testset "a form not reading old comes back itself" begin
        @test rebind(a, t => s) === a
        # an element equal in value is a different object
        @test rebind(a, copy(s) => t) === a
    end

    @testset "Dual coefficient" begin
        bd = rebind(a, s => seeded(s))
        Ad = assemble(bd)
        @test eltype(Ad) == Dl
        @test value.(Ad) ≈ A
        # `A` is affine in `s`: the derivative along a unit seed is the form at `s = 1`,
        # without the `r` term, which does not read `s`
        one_s = element(W, 1.0)
        g = (u, v) -> inner₊(one_s * ∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(one_s * u), D₋ᵧ(v)) +
                      innerₕ(M₊ₓ(one_s * u), v)
        @test map(x -> partials(x, 1), Ad) ≈ assemble(form(W, W, g))
    end
end

@testset "rebind: plain linear form" begin
    W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (7, 6), (false, false)))
    s, t = Rₕ(W, ss), Rₕ(W, ts)
    src(c) = Bramble._as_source(c, Val(2))
    f(c) = v -> innerₕ(c, v) + innerₕ(πₕ(c), D₋ₓ(v)) +
                innerₕ(restrict_to(:boundary, src(c)), v)
    l = form(W, f(s))
    F = assemble(l)
    m = rebind(l, s => t)
    @test assemble(m) ≈ assemble(form(W, f(t))) atol = 1e-14
    @test assemble(m) != F
    @test assemble(l) == F
    @test rebind(l, t => s) === l

    @testset "Dual source" begin
        Fd = assemble(rebind(l, s => seeded(s)))
        @test eltype(Fd) == Dl
        @test value.(Fd) ≈ F
        @test partials.(Fd, 1) ≈ assemble(form(W, f(element(W, 1.0))))
    end
end

@testset "rebind: composite space" begin
    W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (6, 7), (false, false)))
    C = W × W
    n = ndofs(W)
    s, t, r = element(W, 1.0), element(W, 100.0), element(W, 10000.0)
    s .= Rₕ(W, ss)
    f(c) = (u, v) -> innerₕ(c * u(1), v(1)) + innerₕ(r * u(2), v(2)) +
                     inner₊(c * ∇ₕ(u(2)), ∇ₕ(v(1)))
    a = form(C, C, f(s))
    A = assemble(a)
    B = assemble(rebind(a, s => t))
    Bref = assemble(form(C, C, f(t)))
    blk(M, i, j) = M[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n)]
    for i in 1:2, j in 1:2

        @test blk(B, i, j) == blk(Bref, i, j)
    end
    # `s` is read by blocks (1, 1) and (1, 2); block (2, 2) reads `r` only
    @test blk(B, 1, 1) != blk(A, 1, 1)
    @test blk(B, 1, 2) != blk(A, 1, 2)
    @test blk(B, 2, 2) == blk(A, 2, 2)
    @test assemble(a) == A

    g(c) = v -> innerₕ(c, v(1)) + innerₕ(r, v(2))
    l = form(C, g(s))
    F = assemble(l)
    G = assemble(rebind(l, s => t))
    @test G == assemble(form(C, g(t)))
    @test G[1:n] != F[1:n]
    @test G[(n + 1):end] == F[(n + 1):end]

    Ad = assemble(rebind(a, s => seeded(s)))
    @test eltype(Ad) == Dl
    @test value.(Ad) ≈ A
end

@testset "rebind: ghost space" begin
    G = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = :boundary)
    P = Bramble.padded_space(G)
    s, t = Rₕ(P, ss), Rₕ(P, ts)
    f(c) = (u, v) -> inner₊(c * ∇ₕ(u), ∇ₕ(v)) + innerₕ(c * u, v)
    g(c) = v -> innerₕ(c, v)
    a, l = form(G, G, f(s)), form(G, g(s))
    A, F = assemble(a), assemble(l)
    b, m = rebind(a, s => t), rebind(l, s => t)
    @test b isa Bramble.GhostBilinearForm
    @test m isa Bramble.GhostLinearForm
    @test assemble(b) == assemble(form(G, G, f(t)))
    @test assemble(m) == assemble(form(G, g(t)))
    @test assemble(b) != A
    # the padded buffers are the rebound form's own
    @test b.A_pad !== a.A_pad
    @test m.F_pad !== l.F_pad
    @test assemble(a) == A
    @test assemble(l) == F
    @test rebind(a, t => s) === a
    @test rebind(l, t => s) === l
end

@testset "rebind: composite of ghost spaces" begin
    G = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = :boundary)
    C = G × G
    P = Bramble.padded_space(G)
    s, t, r = Rₕ(P, ss), Rₕ(P, ts), Rₕ(P, x -> 10000.0)
    f(c) = (u, v) -> innerₕ(c * u(1), v(1)) + inner₊(r * ∇ₕ(u(2)), ∇ₕ(v(2))) +
                     innerₕ(c * u(2), v(1))
    g(c) = v -> innerₕ(c, v(1)) + innerₕ(r, v(2))
    a, l = form(C, C, f(s)), form(C, g(s))
    A, F = assemble(a), assemble(l)
    b, m = rebind(a, s => t), rebind(l, s => t)
    @test b isa Bramble.GhostCompositeBilinearForm
    @test m isa Bramble.GhostCompositeLinearForm
    @test assemble(b) == assemble(form(C, C, f(t)))
    @test assemble(m) == assemble(form(C, g(t)))
    @test assemble(b) != A
    @test assemble(a) == A
    @test assemble(l) == F
    @test rebind(a, t => s) === a
end

# A constant scale summed with a coefficient is widened by `form` to the coefficient's type
# where it differs (`_split_restricted_products`, assembly/bilinear.jl); a Float64 `s`
# rebound to a complex one leaves the scale Float64 unless the rebound tree runs that pass
# again, and every refill then boxes. Bytes against the form built with `new` directly.
@testset "rebind: complex coefficient by a scale" begin
    bytes(x) = (M = assemble(x); assemble!(M, x); @allocated assemble!(M, x))
    complexified(s) = (c = element(space(s), ComplexF64); parent(c) .= parent(s) .+ 2im; c)
    W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (6, 7), (false, false)))
    s = Rₕ(W, ss)
    sc = complexified(s)
    fs = (c -> (u, v) -> innerₕ(c * u + 0.5 * D₋ₓ(u), v),
        c -> (u, v) -> innerₕ(c * u + 2 * u, v),
        c -> (u, v) -> inner₊(c * ∇ₕ(u) + 0.5 * ∇ₕ(u), ∇ₕ(v)))
    @testset "plain, form $k" for (k, f) in enumerate(fs)
        b, d = rebind(form(W, W, f(s)), s => sc), form(W, W, f(sc))
        @test assemble(b) == assemble(d)
        @test bytes(b) == bytes(d)
    end

    @testset "composite" begin
        C = W × W
        f(c) = (u, v) -> innerₕ(c * u(1) + 0.5 * D₋ₓ(u(1)), v(2)) + innerₕ(u(2), v(1))
        b, d = rebind(form(C, C, f(s)), s => sc), form(C, C, f(sc))
        @test assemble(b) == assemble(d)
        @test bytes(b) == bytes(d)
    end

    @testset "ghost" begin
        G = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = :boundary)
        P = Bramble.padded_space(G)
        sp = Rₕ(P, ss)
        spc = complexified(sp)
        f(c) = (u, v) -> innerₕ(c * u + 0.5 * D₋ₓ(u), v)
        b, d = rebind(form(G, G, f(sp)), sp => spc), form(G, G, f(spc))
        @test assemble(b) == assemble(d)
        @test bytes(b) == bytes(d)
    end
end

end
