module FormCompileScalingTests

using Test
using Bramble
using Random
import Bramble: D₋ₓ, D₊ₓ, Dcₓ, D₋ᵧ, D₊ᵧ, Dcᵧ, D₋₂, D₊₂, Dc₂

# S3 (gpena/Bramble.jl, this milestone) fixed a compile blow-up in `resolve_ast`: each
# summand's AST was being resolved twice per node -- once inside `typeof(resolve_ast(x))`,
# once more for the value itself -- roughly 2^26 calls for a 27-term 3D form, invisible
# without coverage because the compiler dead-code-eliminated the duplicate call once
# instrumentation was off. The fix resolves each child once into a local, and walks a scalar
# form's top-level sum through `_summands`, recording/replaying one execution segment per
# summand rather than recursing into the fused stencil. This file is S4's guard: it
# re-measures the scaling directly (model: .agents/plans/checks/nterm-form.jl), so a
# regression shows up as a test failure rather than a SIGKILLed CI job.
#
# Measured wall time of the first `assemble` after a one-term warm-up, 2 threads:
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

end # module FormCompileScalingTests
