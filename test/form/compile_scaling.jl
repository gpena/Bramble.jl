module FormCompileScalingTests

using Test
using Bramble
using Random
using LinearAlgebra: dot, norm
using SparseArrays: SparseMatrixCSC, nnz
import Bramble: D₋ₓ, Dcᵧ, D̽ᵧ, Mₓ
using Bramble: OperatorAdd, OperatorScale, BilinearProduct, LinearProduct

# S3 (gpena/Bramble.jl, this milestone) fixed a compile blow-up in `resolve_ast`. Each
# summand's AST was being resolved twice per node -- once inside `typeof(resolve_ast(x))`,
# once more for the value itself -- roughly 2^26 calls for a 27-term 3D form, invisible
# without coverage because the compiler dead-code-eliminated the duplicate call once
# instrumentation was off. The fix resolves each child once into a local, and walks a scalar
# form's top-level sum through `_summands`, recording/replaying one execution segment per
# summand rather than recursing into the fused stencil. This file is S4's guard: it
# re-measures the scaling directly (model: .agents/plans/checks/nterm-form.jl), so a
# regression shows up as a test failure rather than a SIGKILLed CI job.
#
# Measured wall time of the first `assemble` after a one-term warm-up, 2 threads.
#   before the fix:  N=9   5.2 s   N=27  44.4 s   (ratio ~8.5, worse than linear)
#   after the fix:   N=9   3.7 s   N=27  15.1 s   (ratio ~4.1, close to linear)
# The assertion below allows slack around that ratio (4.5x plus a fixed 1s) rather than
# pinning it exactly, since compile time is noisy across machines and Julia versions.
#
# Timed in a fresh process: in the suite, earlier files have already compiled the first nine
# pairs (k = 0, each operator with itself), so `t9` shrank to 0.6 s while `t27` stayed at
# 9.5 s, a ratio of 17 with no change to the compile. Coverage on, 3 threads: fresh, the
# ratio is 2.6.
const NTERM_TIMES = raw"""
using Bramble, Random
import Bramble: D₋ₓ, D₊ₓ, Dcₓ, D₋ᵧ, D₊ᵧ, Dcᵧ, D₋₂, D₊₂, Dc₂
Random.seed!(287)
Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (9, 8, 7), (false, false, false))
Wₕ = gridspace(Ωₕ)
OPS = (D₋ₓ, D₊ₓ, Dcₓ, D₋ᵧ, D₊ᵧ, Dcᵧ, D₋₂, D₊₂, Dc₂)
# (i, i+k) mod 9 for k = 0, 1, 2, ...: every N ≤ 81 gets N distinct operator pairs.
PAIRS = [(OPS[i], OPS[mod1(i + k, 9)]) for k in 0:8 for i in 1:9]
# Warm up shared machinery (mesh/space construction, the simplifier, `assemble` itself)
# with a one-term form, so the timed runs below measure only the N-term compile.
assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v))))
function assemble_nterms(n)
    ps = Tuple(PAIRS[1:n])
    f = form(Wₕ, Wₕ, (u, v) -> foldl(+, map(p -> innerₕ(p[1](u), p[2](v)), ps)))
    t = @elapsed A = assemble(f)
    return t, A
end
t9, _ = assemble_nterms(9)
t27, A27 = assemble_nterms(27)
# Cheap correctness guard alongside the timing one: the 27-term matrix must equal the
# sum of its 27 single-term matrices -- catches a fused or short-circuited sum that
# happens to compile fast but silently drops or double-counts a term.
singles = [assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(p[1](u), p[2](v)))) for p in PAIRS[1:27]]
println(t9, " ", t27, " ", isapprox(Matrix(A27), sum(Matrix.(singles))))
"""

@testset "N-term form compiles near-linearly" begin
    project = Base.active_project()
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=2 --project=$project
        -e $NTERM_TIMES`
    out = split(readchomp(cmd))
    t9, t27 = parse(Float64, out[1]), parse(Float64, out[2])
    @test t27 <= 4.5 * t9 + 1
    @test out[3] == "true"
end

# The Laplacian `Δₕ` (gpena/Bramble.jl#774) nests as a node of its own, so `Δₕ(Δₕ(Δₕ(u)))`
# is a sum of 2^3 = 8 three-deep products in 2D and 3^3 = 27 in 3D. Wall time of the first
# `assemble` of `innerₕ(Δₕ^k(u), v)` at 1 thread in a fresh session, with depths 1 and 2 in
# 2D already warm from the precompile workload:
#   2D  k=3   4.4 s   k=4  78.8 s
#   3D  k=2   2.6 s   k=3  41.5 s
# Each level multiplies the compile time by about 16, more than its term count (2 in 2D, 3
# in 3D) accounts for, and this case records that cost. Depth 3 in 2D must stay within
# 180 s, about 2.2 times the 80 s the macOS nightly runner takes (coverage, --optimize=1,
# --check-bounds=yes; about 2.3 times slower than ubuntu) and about 40 times the local
# 4.4 s. A depth-4 blow-up (78.8 s locally, about 16 times per level) still fails it by a
# wide margin. The matrix is also compared with the runtime composition, so a fast compile
# that drops a term fails too.
@testset "Δₕ to depth 3 compiles in a bound" begin
    Random.seed!(774)
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 8), (false, false))
    Wₕ = gridspace(Ωₕ)
    assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v))))

    a = form(Wₕ, Wₕ, (u, v) -> innerₕ(Δₕ(Δₕ(Δₕ(u))), v))
    t = @elapsed A = assemble(a)
    @test t <= 180

    u = element(Wₕ, rand(ndofs(Wₕ)))
    w = element(Wₕ, rand(ndofs(Wₕ)))
    @test dot(parent(w), A*parent(u))≈innerₕ(Δₕ(Δₕ(Δₕ(u))), w) rtol=1e-10 atol=1e-12
end

# #763 step 2. The first `form` of 3D elasticity compiled 5803 method instances module-wide
# at a70e3159 and 3485 at 018c6224, most of them in the mesh-check walks over every partial
# sum and term (`_check_walked_meshes`, `_check_centered_mesh`, `_check_source_mesh`) and in
# `simplify_ast`: one method per partial sum, a scale distributed over a sum by simplifying
# every partial sum again, a zero test on every node and component-class queries under each
# difference and average. The bound, 2900, is half the first count. Counted as
# compile_assembly.jl counts its own, in a fresh process, so no earlier test file has
# compiled any of them.
const ELASTICITY_FORM_COUNT = raw"""
using Bramble, Random
nspec(f) = sum(m -> count(!isnothing, Base.specializations(m)), methods(f); init = 0)
fns = [getfield(Bramble, n) for n in names(Bramble; all = true)
       if isdefined(Bramble, n) && getfield(Bramble, n) isa Function]
total() = sum(nspec, fns)
Random.seed!(20260903)
Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 5, 5), (false, false, false))
Vₕ = gridspace(Ωₕ)^Val(3)
μ, λ = 1.0, 2.0
c0 = total()
a = form(Vₕ, Vₕ, (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v)))
n = total() - c0
println(n, " ", a isa Bramble.BilinearForm)
"""

@testset "Elasticity first form bound (#763)" begin
    project = Base.active_project()
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$project
        -e $ELASTICITY_FORM_COUNT`
    out = split(readchomp(cmd))
    @test parse(Int, out[1]) <= 2900
    @test out[2] == "true"
end

# What #763's changes to `form` must not move. A scale over a sum whose terms name different
# components is distributed by `_scale_over_sum` in one generated body, and the mesh checks
# skip a term that holds nothing they read (`_reads_meshes`). Each form below is compared
# with the tree 018c6224 builds for it, written out by hand, bit for bit, and a few with
# the type and norm that commit gives. The second argument of each pair is already in that
# shape, so it never reaches the distribution under test.
function _same_bits(A::SparseMatrixCSC, B::SparseMatrixCSC)
    A.colptr == B.colptr &&
        A.rowval == B.rowval &&
        reinterpret(UInt64, A.nzval) ==
        reinterpret(UInt64, B.nzval)
end
_same_bits(a::AbstractVector, b::AbstractVector) = reinterpret(UInt64, a) ==
                                                   reinterpret(UInt64, b)

@testset "Scaled mixed sums keep their tree (#763)" begin
    Random.seed!(763)
    Wₕ = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7),
        (false, false)))
    Vₕ = Wₕ × Wₕ
    r = Ref(1.7)
    fₕ = Rₕ(Wₕ, x -> 1.0 + x[1] * x[2])
    δ = dirac((0.35, 0.6), 2.0)
    pairs = (
        # A nested static scale folds into the distributed one (rule 4).
        "fold" => ((u, v) -> 2.0 * (3.0 * innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))),
            (u, v) -> 6.0 * innerₕ(u(1), v(1)) + 2.0 * innerₕ(u(2), v(2))),
        # A `Rational` never folds.
        "rational" => ((u, v) -> 2.0 * ((1 // 3) * innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))),
            (u, v) -> 2.0 * ((1 // 3) * innerₕ(u(1), v(1))) + 2.0 * innerₕ(u(2), v(2))),
        # A sum naming one component stays whole under the scale.
        "same component" => (
            (u, v) -> 2.0 * ((innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(1)), v(1))) +
                             innerₕ(u(2), v(2))),
            (u, v) -> 2.0 * (innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(1)), v(1))) +
                      2.0 * innerₕ(u(2), v(2))),
        # An `Integer` zero drops every term.
        "integer zero" => (
            (u, v) -> 0 * (innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))) +
                      innerₕ(D₋ₓ(u(1)), D₋ₓ(v(1))),
            (u, v) -> innerₕ(D₋ₓ(u(1)), D₋ₓ(v(1)))),
        # A product left with a scale operand is simplified again, as 018c6224 did.
        "ref over integer" => (
            (u, v) -> 3.0 * (innerₕ(r * (2 * u(1)), v(1)) + innerₕ(u(2), v(2))),
            (u, v) -> 3.0 * (r * (2 * innerₕ(u(1), v(1)))) + 3.0 * innerₕ(u(2), v(2))),
        "rational over integer" => (
            (u, v) -> 0.3 * (innerₕ((1 // 3) * (3 * u(1)), D₋ₓ(v(1))) +
                             innerₕ(u(2), v(2))),
            (u, v) -> 0.3 * ((1 // 3) * (3 * innerₕ(u(1), D₋ₓ(v(1))))) +
                      0.3 * innerₕ(u(2), v(2))),
        "rational over float" => (
            (u, v) -> 2 * (innerₕ(u(1), (1 // 7) * (0.3 * v(1))) + innerₕ(u(2), v(1))),
            (u, v) -> 2 * ((1 // 7) * (0.3 * innerₕ(u(1), v(1)))) + 2 * innerₕ(u(2), v(1)))
    )
    for (name, (f, g)) in pairs
        @testset "$name" begin
            a, b = form(Vₕ, Vₕ, f), form(Vₕ, Vₕ, g)
            @test typeof(a) == typeof(b)
            @test _same_bits(assemble(a), assemble(b))
        end
    end

    # The same, pinned to 018c6224's own figures rather than to a second form.
    a = form(Vₕ, Vₕ, (u, v) -> 2.0 * (3.0 * innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))))
    @test a.ast isa OperatorAdd{2, <:OperatorScale{2, Float64, <:BilinearProduct},
        <:OperatorScale{2, Float64, <:BilinearProduct}}
    @test a.ast.left_op.scalar === 6.0 && a.ast.right_op.scalar === 2.0
    a = form(Vₕ, Vₕ, (u, v) -> 2.0 * ((1 // 3) * innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))))
    @test a.ast.left_op isa
          OperatorScale{2, Float64, <:OperatorScale{2, Rational{Int}, <:BilinearProduct}}
    a = form(Vₕ, Vₕ, (u, v) -> 0 * (innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))) +
                               innerₕ(D₋ₓ(u(1)), D₋ₓ(v(1))))
    @test a.ast isa BilinearProduct
    a = form(Vₕ, Vₕ, (u, v) -> 0.3 * (innerₕ((1 // 3) * (3 * u(1)), D₋ₓ(v(1))) +
                                      innerₕ(u(2), v(2))))
    @test a.ast.left_op isa OperatorScale{2, Float64,
        <:OperatorScale{2, Rational{Int}, <:OperatorScale{2, Int, <:BilinearProduct}}}
    A = assemble(a)
    @test nnz(A) == 119
    @test isapprox(norm(A), 0.5279233254079588; rtol = 1e-12)

    # A Dirac split leaves `0.3 * πₕ(fₕ)` inside its product, which a distributed scale
    # simplifies again.
    l = form(Vₕ, v -> 3.0 * (innerₕ(0.3 * πₕ(fₕ) + δ, v(1)) + innerₕ(fₕ, v(2))))
    m = form(Vₕ, v -> 3.0 * (0.3 * innerₕ(πₕ(fₕ), v(1)) + innerₕ(δ, v(1))) +
                      3.0 * innerₕ(fₕ, v(2)))
    @test typeof(l) == typeof(m)
    @test _same_bits(assemble(l), assemble(m))
end

# A centered difference on an axis of two points is refused at `form` wherever it sits:
# bare, under an average, as the last of twelve terms, and in a linear form.
@testset "Short-axis Dc refused at form (#763)" begin
    Random.seed!(764)
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 2), (false, true))
    Wₕ = gridspace(Ωₕ)
    many = (u, v) -> foldl(+, ntuple(k -> (k + 0.5) * innerₕ(D₋ₓ(u), v), 11)) +
                     innerₕ(Dcᵧ(u), v)
    for f in ((u, v) -> innerₕ(Dcᵧ(u), v), (u, v) -> innerₕ(Mₓ(Dcᵧ(u)), v),
        (u, v) -> innerₕ(u, D̽ᵧ(v)), many)
        @test_throws "needs at least 3 points" form(Wₕ, Wₕ, f)
        @test_throws ArgumentError form(Wₕ, Wₕ, f)
    end
    @test_throws ArgumentError form(Wₕ, v -> innerₕ(x -> 1.0, Mₓ(Dcᵧ(v))))
    @test_throws ArgumentError form(Wₕ × Wₕ, Wₕ × Wₕ,
        (u, v) -> innerₕ(u(1), v(1)) + innerₕ(Dcᵧ(u(2)), v(2)))
end

end # module FormCompileScalingTests
