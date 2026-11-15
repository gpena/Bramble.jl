module ExtEnzymeRelationsCompositeTests

# Enzyme through `constrained` on a composite, `constrained(W × W, …)` (gpena/Bramble.jl#1180).
# These cells were in test/ext/enzyme_relations.jl until they pushed that child past 10 GB.
# Reverse mode used to stop with an `EnzymeInternalError` where `constrained` builds the
# rows of all components in one pass, constant relations too. Checked: constant relations,
# and a datum on one component, against `AutoForwardDiff()` to 1e-8 relative on the 7 by 6
# non-uniform `ghost_mesh`. Tractions meeting at a corner with a parameter in their
# operators get a wrong derivative from Enzyme in both modes; the rules of
# `Bramble._corner_entry` refuse it, and a cell pins the refusal.
#
# Enzyme is not in test/Project.toml: load it first, through a temporary environment
# (test/ext/enzyme_policy_cases.jl says how). Without it the file runs nothing, and with it
# the file runs only as its own `ENZYME_FILES` child, so its compiled code is released.

using Test
using Bramble
using Bramble: CpuSerial
using ForwardDiff
using SparseArrays: nonzeros
import DifferentiationInterface as DI
using ..TestUtils: _have, ghost_mesh, TEST_FILE

const RUN_ENZYME = _have(:Enzyme) && TEST_FILE == "ext/enzyme_relations_composite.jl"
RUN_ENZYME && @eval using Enzyme

const TOL = 1e-8
const X0 = [1.3]

_relerr(g, ref) = maximum(abs, g .- ref) / maximum(abs, ref)

_auto(mode) = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(mode),
    function_annotation = Enzyme.Const)

# `f`'s gradient by Enzyme in `mode`, relative to ForwardDiff's.
function _enzyme_vs_forwarddiff(f, mode)
    ref = DI.gradient(f, DI.AutoForwardDiff(), X0)
    return _relerr(DI.gradient(f, _auto(mode), X0), ref), ref
end

# The loss of `constrained(W × W, …)`: the closure of relations on both components, the
# datum on `:xmax` of the first `p[1]` (`:data`), or every relation constant and the closure
# scaled by `p[1]^2` (`:constant`).
function _composite_loss(W, what::Symbol)
    function f(p)
        g = what === :data ? p[1] : 1.0
        C = constrained(W × W,
            u -> (:xmin => ∂ₙ(u(1)) => 0.0, :xmax => 2.0 * u(1) + ∂ₙ(u(1)) => g,
                :ymin => u(1) => 0.0, :ymax => ∂ₙ(u(1)) => 1.0,
                :xmin => ∂ₙ(u(2)) => 0.0, :xmax => ∂ₙ(u(2)) => 1.0,
                :ymin => u(2) => 0.0, :ymax => ∂ₙ(u(2)) => 0.5))
        E, d = Bramble.closure(C)
        s = sum(abs2, nonzeros(E)) + sum(abs2, d)
        return what === :constant ? p[1]^2 * s : s
    end
    return f
end

# The loss of tractions on `:xmin` and `:ymin` of `W × W`, with `λ = p[1]` in the
# operators' `K`, the other facets clamped: their corner moves its entries through the
# one-sided difference (`Bramble._corner_entry`), whose rules refuse this case (#1180).
function _corner_loss(W)
    K(i, j, λ) = [λ * (i == d) * (j == e) + 0.5 * ((i == j) * (d == e) + (i == e) * (j == d))
                  for d in 1:2, e in 1:2]
    t(u, i, λ) = ∂ₙ(u(1); K = K(i, 1, λ)) + ∂ₙ(u(2); K = K(i, 2, λ))
    clamped(F, u) = (F => u(1) => 0.0, F => u(2) => 0.0, F => ∂ₙₙ(u(1)) => 0.0,
        F => ∂ₙₙ(u(2)) => 0.0)
    function f(p)
        λ = p[1]
        C = constrained(W × W,
            u -> (:xmin => t(u, 1, λ) => 1.0, :xmin => t(u, 2, λ) => 0.0,
                :ymin => t(u, 1, λ) => 0.0, :ymin => t(u, 2, λ) => 0.3,
                clamped(:xmax, u)..., clamped(:ymax, u)...))
        E, d = Bramble.closure(C)
        return sum(abs2, nonzeros(E)) + sum(abs2, d)
    end
    return f
end

@testset verbose=true "constrained composite, Enzyme (#1180)" begin
    if RUN_ENZYME
        W = gridspace(ghost_mesh(CpuSerial(), (7, 6)); ghosts = (:xmin, :xmax, :ymax))
        modes = ((:reverse, Enzyme.Reverse), (:forward, Enzyme.Forward))
        cells = ((:constant, modes[1]...), (:data, modes[1]...), (:data, modes[2]...))
        @testset "composite, $what, $mname (#1180)" for (what, mname, mode) in cells
            err, ref = _enzyme_vs_forwarddiff(_composite_loss(W, what), mode)
            @test abs(only(ref)) > 1e-6
            @test err < TOL
        end

        @testset "corner coefficient refused, $mname (#1180)" for (mname, mode) in modes
            Wb = gridspace(ghost_mesh(CpuSerial(), (7, 6)); ghosts = :boundary)
            grad() = DI.gradient(_corner_loss(Wb), _auto(mode), [2.0])
            @test_throws ArgumentError grad()
            @test_throws "parameter in a relation's operator" grad()
        end
    end
end

end
