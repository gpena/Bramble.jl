module QualityJetTests

using Test
using JET
using Bramble

@testset "JET" begin
    if isempty(VERSION.prerelease)
        try
            jet_report = JET.report_package(
                Bramble; target_modules = (Bramble,), toplevel_logger = nothing
            )
            reports = JET.get_reports(jet_report)
            @test length(reports) == 0
        catch e
            # A JET that cannot analyse the package is a failed gate, not a skip.
            @error "JET could not analyse Bramble" exception = e
            @test false
        end
    else
        @test_skip "JET full-package static analysis is skipped on prerelease Julia"
    end
end

# A ghost-space relation that dispatches at runtime shows up as a JET report with a
# `constrained` frame on its stack: `constrained` builds the relation closure's forms,
# and the closure's type must reach it as a constant. The relation closures capture
# locals (`let`), never non-const globals, which infer `Any` for reasons outside Bramble.
_in_constrained(f) = (mi = f.linfo; mi isa Core.MethodInstance && mi.def isa Method &&
                                        mi.def.name === :constrained)
_constrained_reports(rs) = count(r -> any(_in_constrained, r.vst), JET.get_reports(rs))
function _constrained_reports(W, rels)
    _constrained_reports(
        JET.report_opt(Bramble.constrained, (typeof(W), typeof(rels)); target_modules = (Bramble,)))
end

@testset "JET constrained frames" begin
    if isempty(VERSION.prerelease)
        @testset "precompile workload $D" for D in (1, 2)
            rs = JET.report_opt(Bramble._pc_form_nested, (Val{D},); target_modules = (Bramble,))
            @test _constrained_reports(rs) == 0
        end
        @testset "relations" begin
            W1 = gridspace(mesh(domain(interval(0.0, 1.0)), 9, false); ghosts = :boundary)
            W2 = gridspace(
                mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, false));
                ghosts = :boundary)
            conormal = let K = [2.0 0.5; 0.5 1.0], F4 = (:xmin, :xmax, :ymin, :ymax)
                u -> map(F -> F => 2.0 * u + ∂ₙ(u; K) => 1.0, F4)
            end
            traction = let K = [2.0 0.5; 0.5 1.0], F4 = (:xmin, :xmax, :ymin, :ymax)
                u -> (map(F -> F => ∂ₙ(u(1); K) + ∂ₙ(u(2); K) => 0.0, F4)...,
                    map(F -> F => ∂ₙ(u(1); K) - ∂ₙ(u(2); K) => 0.0, F4)...)
            end
            cross = u -> (:xmin => ∂ₙ(u(3)) - ∂ₙ(u(1)) - ∂ₙ(u(2)) => 0.0,
                :xmax => u(3) + ∂ₙ(u(3)) => 0.0)
            @test _constrained_reports(W2, conormal) == 0
            @test _constrained_reports(W2 × W2, traction) == 0
            @test _constrained_reports(W1 × W1 × W1, cross) == 0
        end
    else
        @test_skip "JET constrained-frame count is skipped on prerelease Julia"
    end
end

# The ghost Kronecker and matrix-free operators infer without an inference barrier in the
# precompile workload (#894): a closure that is not separable throws, never `nothing`.
@testset "JET ghost operators" begin
    if isempty(VERSION.prerelease)
        Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (6, 6), (true, true))
        C = constrained(gridspace(Ωₕ; ghosts = :xmin), u -> :xmin => 1.0 * u + ∂ₙ(u) => 0.0)
        a = form(C, C, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        @test isempty(JET.get_reports(JET.report_call(kronecker_operator, (typeof(a),))))
        @test isempty(JET.get_reports(JET.report_call(matrix_free_operator, (typeof(a),))))
    else
        @test_skip "JET ghost-operator reports are skipped on prerelease Julia"
    end
end

end # module QualityJetTests
