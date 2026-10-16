module FormCompileScalingTests

using Test
using Bramble
using Random
using LinearAlgebra: dot
import Bramble: D₋ₓ, D₊ₓ, Dcₓ, D₋ᵧ, D₊ᵧ, Dcᵧ, D₋₂, D₊₂, Dc₂

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
@testset "N-term form compiles near-linearly" begin
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

    @test t27 <= 4.5 * t9 + 1

    # Cheap correctness guard alongside the timing one: the 27-term matrix must equal the
    # sum of its 27 single-term matrices -- catches a fused or short-circuited sum that
    # happens to compile fast but silently drops or double-counts a term.
    singles = [assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(p[1](u), p[2](v)))) for p in PAIRS[1:27]]
    @test isapprox(Matrix(A27), sum(Matrix.(singles)))
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

end # module FormCompileScalingTests
