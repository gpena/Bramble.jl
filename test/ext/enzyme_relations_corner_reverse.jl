module ExtEnzymeRelationsCornerReverseTests

# Enzyme's reverse mode through `constrained` on a composite where tractions meet at a
# one-sided corner, with a parameter in their operators (`λ` in `K`, `β` in `β * u(i)`):
# its derivative there comes out wrong (gpena/Bramble.jl#1180), the rules of
# `Bramble._corner_entry` refuse it, and these cells pin the refusal. Forward mode is exact
# there (#1226), checked in test/ext/enzyme_relations_corner.jl. These cells were in
# test/ext/enzyme_relations_composite.jl until they pushed that child past 10 GB: one
# reverse compile of this loss peaks near 7.3 GB.
#
# Enzyme is not in test/Project.toml: load it first, through a temporary environment
# (test/ext/enzyme_policy_cases.jl says how). Without it the file runs nothing, and with it
# the file runs only as its own `ENZYME_FILES` child, so its compiled code is released.

using Test
using Bramble
using Bramble: CpuSerial
using SparseArrays: nonzeros
import DifferentiationInterface as DI
using ..TestUtils: _have, ghost_mesh, TEST_FILE

const RUN_ENZYME = _have(:Enzyme) && TEST_FILE == "ext/enzyme_relations_corner_reverse.jl"
RUN_ENZYME && @eval using Enzyme

_auto(mode) = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(mode),
    function_annotation = Enzyme.Const)

_kt(i, j, λ) = [λ * (i == d) * (j == e) + 0.5 * ((i == j) * (d == e) + (i == e) * (j == d))
                for d in 1:2, e in 1:2]
_trac(u, i, λ) = ∂ₙ(u(1); K = _kt(i, 1, λ)) + ∂ₙ(u(2); K = _kt(i, 2, λ))
_clamped(F, u) = (F => u(1) => 0.0, F => u(2) => 0.0, F => ∂ₙₙ(u(1)) => 0.0,
    F => ∂ₙₙ(u(2)) => 0.0)

# The loss of tractions on `:xmin` and `:ymin` of `W × W`, with `λ = p[1]` in the
# operators' `K` (`:λ`) or `β = p[1]` in `β * u(i)` on one traction of each facet (`:β`),
# the other facets clamped: their corner moves its entries through the one-sided difference
# (`Bramble._corner_entry`), whose reverse rules refuse this case (#1180). One closure
# type for both, so Enzyme compiles its reverse pass once.
function _corner_loss(W, what::Symbol)
    function f(p)
        λ = what === :λ ? p[1] : 2.0
        β = what === :β ? p[1] : 0.0
        C = constrained(W × W,
            u -> (:xmin => β * u(1) + _trac(u, 1, λ) => 1.0,
                :xmin => _trac(u, 2, λ) => 0.0, :ymin => _trac(u, 1, λ) => 0.0,
                :ymin => β * u(2) + _trac(u, 2, λ) => 0.3, _clamped(:xmax, u)...,
                _clamped(:ymax, u)...))
        E, d = Bramble.closure(C)
        return sum(abs2, nonzeros(E)) + sum(abs2, d)
    end
    return f
end

@testset verbose=true "one-sided corner, reverse (#1180)" begin
    if RUN_ENZYME
        Wb = gridspace(ghost_mesh(CpuSerial(), (7, 6)); ghosts = :boundary)
        @testset "refused, $what, reverse (#1180)" for what in (:β, :λ)
            grad() = DI.gradient(_corner_loss(Wb, what), _auto(Enzyme.Reverse), [2.0])
            @test_throws ArgumentError grad()
            @test_throws "Enzyme's reverse mode cannot differentiate `constrained`" grad()
        end
    end
end

end
