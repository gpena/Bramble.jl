module FormScalarCoefficientsTests

# On a Float64 space a non-`Integer` real coefficient is converted once at `form` to a
# `Float64` (`_normalise_scalars`, bilinear.jl, gpena/Bramble.jl#912), so `2.0f0`, `π`,
# `1 // 2` and `2.0` give one AST type and compile `assemble` once. The matrices are pinned
# bit for bit against the `Float64` spelling, which they matched before the conversion too.
# Where converting would move a bit (a Float32 space, two scales meeting in one product)
# the form is kept as built, pinned against references spelled with literals or a `Ref`,
# which is never converted. An `Int` is kept, so `simplify_ast` still folds it as a literal.
# Every mesh is graded.

using Test
using Bramble
import Bramble
using Bramble: change_points!, OperatorScale, restrict_to, D₋ₓ
using LinearAlgebra: issymmetric
using ..TestUtils: WITH_3D_TESTS, OPTIMIZED, alloc_test

grading(n, a) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2 * a)

function graded_space(n, ::Type{T} = Float64; markers = ()) where {T}
    Iv = interval(zero(T), one(T))
    Ω = mesh(domain(foldl(×, ntuple(_ -> Iv, length(n))), markers...), n,
        ntuple(_ -> false, length(n)); backend = backend(T))
    change_points!(Ω, ntuple(d -> T.(grading(n[d], d)), length(n)))
    return gridspace(Ω)
end

# The scalar types of every constant scale in the AST type `T`.
function scalar_types(T, out = Set{Any}())
    T isa DataType || return out
    T <: OperatorScale && push!(out, T.parameters[2])
    foreach(p -> p isa Type && scalar_types(p, out), T.parameters)
    return out
end

@testset "One AST type per coefficient (#912)" begin
    W = graded_space((9, 11))
    Wf = graded_space((9, 11), Float32)
    a(c) = form(W, W, (u, v) -> c * inner₊(∇ₕ(u), ∇ₕ(v)))
    l(c) = form(W, v -> c * innerₕ(1.0, v))
    T = typeof(a(2.0).ast)
    L = typeof(l(2.0).ast)

    @testset "2D Laplacian and a linear form, $c" for c in (2.0f0, π, 1 // 2, 1 // 3)
        @test typeof(a(c).ast) === T
        @test typeof(l(c).ast) === L
        @test assemble(a(c)) == assemble(a(Float64(c)))
        @test assemble(l(c)) == assemble(l(Float64(c)))
    end

    @testset "An Int stays an Int" begin
        @test typeof(a(3).ast) !== T
        @test Int in scalar_types(typeof(a(3).ast))
        @test assemble(a(3)) == assemble(a(3.0))
    end

    @testset "No loose scalar: the form itself" begin
        for ast in (a(2.0).ast, a(3).ast, l(2.0).ast)
            @test Bramble._normalise_scalars(ast, Float64) === ast
        end
        u = Bramble.trial_function(W)
        for c in (2 + 1im, 2.0 + 0im, Ref(2.0), 2)
            op = OperatorScale(c, u)
            @test Bramble._normalise_scalars(op, Float64) === op
        end
        @test Bramble._normalise_scalars(OperatorScale(1 // 2, u), Float64) ===
              OperatorScale(0.5, u)
    end

    @testset "A restricted sum of an Int and 1 // 3" begin
        Wr = graded_space((9, 11); markers = (:l => :left, :r => :right))
        R(c) = form(Wr, Wr,
            (u, v) -> innerₕ(u, 2 * restrict_to(:l, v) + c * restrict_to(:r, v)))
        @test typeof(R(1 // 3).ast) === typeof(R(1 / 3).ast)
        @test assemble(R(1 // 3)) == assemble(R(1 / 3))
    end

    @testset "Kept as built on a Float32 space" begin
        af(c) = form(Wf, Wf, (u, v) -> c * inner₊(∇ₕ(u), ∇ₕ(v)))
        for c in (π, 1 // 2, 2.0f0, 2.0)
            @test scalar_types(typeof(af(c).ast)) == Set([typeof(c)])
        end
        # A `1 // 3` in a term a `2.0` widens to `Float64` is rounded once, in `Float64`.
        A = assemble(form(Wf, Wf, (u, v) -> innerₕ(D₋ₓ(u) / 3, 2.0 * D₋ₓ(v))))
        @test eltype(A) === Float64
        @test A ==
              assemble(form(Wf, Wf, (u, v) -> innerₕ((1 / 3) * D₋ₓ(u), 2.0 * D₋ₓ(v))))
        @test assemble(form(Wf, v -> innerₕ(2.0, D₋ₓ(v)) / 3)) ==
              assemble(form(Wf, v -> (1 / 3) * innerₕ(2.0, D₋ₓ(v))))
        # `π` on a transposed pair is scaled as a `Ref(π)` is.
        pair(c) = form(Wf, Wf, (u, v) -> c * innerₕ(D₋ₓ(u), v) + c * innerₕ(u, D₋ₓ(v)))
        @test assemble(pair(π)) == assemble(pair(Ref(π)))
    end

    @testset "Kept as built on W × Wf" begin
        # `eltype(W × Wf)` is `Float64`, its second component's weights `Float32` ones.
        M = W × Wf
        m(c) = form(M, M, (u, v) -> c * innerₕ(D₋ₓ(u(2)), D₋ₓ(v(2))))
        @test Rational{Int} in scalar_types(typeof(m(1 // 3).ast))
        @test assemble(m(1 // 3)) == assemble(m(Ref(1 // 3)))
    end

    @testset "Two scales in one product" begin
        # `2.0f0 * (1 // 3)` is formed in `Float32`, as the literal product is.
        p(c, d) = form(W, W,
            (u, v) -> innerₕ(D₋ₓ(u) * c, v) * d + innerₕ(u, D₋ₓ(v) * c) * d)
        @test assemble(p(1 // 3, 2.0f0)) == assemble(p(1, 2.0f0 * (1 // 3)))
        @test Rational{Int} in scalar_types(typeof(p(1 // 3, 2.0f0).ast))
    end

    # Two nested `Rational`s partner a transposed term by their exact product (#633).
    @testset "Nested Rational pairs, $(eltype(S))" for S in (W, Wf)
        for (m, n) in ((3, 3), (3, 10), (7, 3))
            a = form(S, S,
                (u, v) -> innerₕ(D₋ₓ(u), v) / m / n + innerₕ(u, D₋ₓ(v)) / (m * n))
            @test issymmetric(a)
            @test issymmetric(assemble(a))
        end
    end
end

# `εₕ` holds `1 // 2` in its off-diagonal pieces; the same tensor with `0.5` spelled.
function strain_half(u::Bramble.LazyOp{D}) where {D}
    entries = ntuple(Val(D)) do i
        ntuple(Val(D)) do j
            i == j && return (Bramble._vc_bwd(u(i), i),)
            return (0.5 * Bramble._vc_avg(Bramble._vc_bwd(u(i), j), i),
                0.5 * Bramble._vc_avg(Bramble._vc_bwd(u(j), i), j))
        end
    end
    return Bramble._StrainTensor{D, typeof(entries)}(entries)
end

# The two `1 // 2` lifted out of a product's sides stay nested, as `simplify_ast` never
# folds a `Rational` (#633), where two `0.5` fold to `0.25`. Two scales meet in that
# product, so the form is kept as built, its `Rational`s with it.
function check_strain(V)
    a = form(V, V, (u, v) -> inner₊(εₕ(u), εₕ(v)))
    b = form(V, V, (u, v) -> inner₊(strain_half(u), strain_half(v)))
    @test Rational{Int} in scalar_types(typeof(a.ast))
    @test assemble(a) == assemble(b)
end

# An element of a composite space holds one block per component, so as a coefficient it
# is refused at `form`, in every spelling, and its component is the spelling that works.
@testset "Composite coefficient refused (#978)" begin
    # `graded_space` passes a tuple of points, which a 1D mesh refuses.
    Ω₁ = mesh(domain(interval(0.0, 1.0)), 9, false)
    change_points!(Ω₁, grading(9, 1))
    @testset "$(dim(W))D" for W in (gridspace(Ω₁), graded_space((9, 11)))
        V = W × W
        w = element(V)
        w(1) .= 1.0
        w(2) .= 2.0
        refused = (
            "on a component" => () -> form(V, V, (u, v) -> innerₕ(w * u(2), v(2))),
            "on a plain space" => () -> form(W, W, (u, v) -> innerₕ(w * u, v)),
            "as a thunk" => () -> form(W, W, (u, v) -> innerₕ((() -> w) * u, v)),
            "on the right" => () -> form(W, W, (u, v) -> innerₕ(u * w, v)),
            "under a gradient" => () -> form(W, W,
                (u, v) -> inner₊(w * ∇ₕ(u), ∇ₕ(v))),
            "in a linear form" => () -> form(W, v -> innerₕ(1.0, w * v)))
        @testset "$name" for (name, mk) in refused
            e = @test_throws ArgumentError mk()
            @test occursin("wₕ(i) * u(i)", sprint(showerror, e.value))
        end
        a = assemble(form(V, V, (u, v) -> innerₕ(w(2) * u(2), v(2))))
        b = assemble(form(W, W, (u, v) -> innerₕ(2.0 * u, v)))
        @test a[(ndofs(W) + 1):end, (ndofs(W) + 1):end] == b
    end

    # A constrained composite's mesh is its padded leaf mesh, which passes the mesh check,
    # so its element was read across the block boundary before it was refused.
    @testset "Constrained composite, $name" for (name, mk) in (
        "on a component" => (C, S, wg) -> form(C, C, (u, v) -> innerₕ(wg * u(2), v(2))),
        "as a thunk" => (C, S, wg) -> form(C, C,
        (u, v) -> innerₕ((() -> wg) * u(2), v(2))),
        "in a linear form" => (C, S, wg) -> form(C, v -> innerₕ(1.0, wg * v(1))),
        "on its leaf space" => (C, S, wg) -> form(S, S, (u, v) -> innerₕ(wg * u, v)))
        S = gridspace(mesh(graded_space((7, 8))); ghosts = (:xmin, :xmax))
        C = constrained(S × S,
            u -> (:xmin => ∂ₙ(u(1)) => 0.0, :xmax => ∂ₙ(u(1)) => 0.0,
                :xmin => ∂ₙ(u(2)) => 0.0, :xmax => ∂ₙ(u(2)) => 0.0))
        wg = element(C)
        wg .= 1.0
        e = @test_throws ArgumentError mk(C, S, wg)
        @test occursin("wₕ(i) * u(i)", sprint(showerror, e.value))
    end
end

# `form` with a grid-function coefficient, and a linear form whose grid source meets a
# scaled two-component sum, allocate as on main (160 B and 0 B, at `-O1` up): the sign
# test `_sign_clash` (simplifier.jl) once took them to 672 B and 1216 B (#978).
grid_coefficient_form(W, c) = form(W, W, (u, v) -> innerₕ(c * u, v))
grid_source_form(V, g) = form(V, v -> innerₕ(g, 3.0 * (v(1) + v(2))))
least_allocs(f, args...) = minimum(alloc_test(f, args...) for _ in 1:5)

@testset "Form construction allocations (#978)" begin
    Ω = mesh(domain(interval(0.0, 1.0)), 9, false)
    change_points!(Ω, grading(9, 1))
    W = gridspace(Ω)
    c = Rₕ(W, x -> 1.0 + x[1])
    g = Rₕ(W, x -> 2.0)
    @test !OPTIMIZED || least_allocs(grid_coefficient_form, W, c) <= 160
    @test !OPTIMIZED || least_allocs(grid_source_form, W × W, g) == 0
end

@testset "2D strain, 1 // 2 and 0.5 (#912)" begin
    check_strain(graded_space((9, 11))^Val(2))
end

# Nightly only, like every 3D mesh (gpena/Bramble.jl#650).
WITH_3D_TESTS && @testset "3D strain, 1 // 2 and 0.5 (#912)" begin
    check_strain(graded_space((5, 6, 7))^Val(3))
end

end
