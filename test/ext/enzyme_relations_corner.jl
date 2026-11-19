module ExtEnzymeRelationsCornerTests

# Enzyme's forward mode through `constrained` on a composite where tractions meet at a
# one-sided corner (gpena/Bramble.jl#1226), against `AutoForwardDiff()` to 1e-8 relative on
# the 7 by 6 non-uniform `ghost_mesh`: a parameter in the operators (`λ` in `K`, `β` in
# `β * u(i)`) or in a datum, at widths 1 and 2.
# Forward mode once gave the datum of the third and fourth of four tractions a tangent equal
# to its value, a wrong gradient with no error, while `_composite_relations` returned the
# relations' tuple from a call; cells on the relations alone pin that tangent in each slot.
# These cells were in test/ext/enzyme_relations_composite.jl until they pushed that child
# past 10 GB. That file checks `λ` forward with tractions on `:xmin` alone, and
# test/ext/enzyme_relations_corner_reverse.jl pins the reverse-mode refusal at this corner.
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

const RUN_ENZYME = _have(:Enzyme) && TEST_FILE == "ext/enzyme_relations_corner.jl"
RUN_ENZYME && @eval using Enzyme

const TOL = 1e-8

_relerr(g, ref) = maximum(abs, g .- ref) / maximum(abs, ref)

_auto(mode) = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(mode),
    function_annotation = Enzyme.Const)

# The loss of tractions on `:xmin` and `:ymin` of `W × W`, the other facets clamped, so
# their corner is one-sided (`Bramble._corner_entry`). `p = [λ, β, g]` holds `λ` in the
# operators' `K`, `β` in `β * u(i)` on one traction of each facet, and the `:xmin` datum
# `g`. The other data are literals, as a user writes them. Data passed in from a variable
# hid the wrong tangent of #1226 in the slot cells, and so did a `:ymin` datum that is a
# parameter, which gets a tangent of its own. Every case is a seed of a pushforward of this
# one loss, so Enzyme compiles it once per width; one compile per case passed the 10 GB
# that julia_slot allows a process.
_kt(i, j, λ) = [λ * (i == d) * (j == e) + 0.5 * ((i == j) * (d == e) + (i == e) * (j == d))
                for d in 1:2, e in 1:2]
_trac(u, i, λ) = ∂ₙ(u(1); K = _kt(i, 1, λ)) + ∂ₙ(u(2); K = _kt(i, 2, λ))
_clamped(F, u) = (F => u(1) => 0.0, F => u(2) => 0.0, F => ∂ₙₙ(u(1)) => 0.0,
    F => ∂ₙₙ(u(2)) => 0.0)
function _corner_loss(W)
    function f(p)
        λ, β, g = p[1], p[2], p[3]
        C = constrained(W × W,
            u -> (:xmin => β * u(1) + _trac(u, 1, λ) => g, :xmin => _trac(u, 2, λ) => 0.0,
                :ymin => _trac(u, 1, λ) => 0.5, :ymin => β * u(2) + _trac(u, 2, λ) => 0.3,
                _clamped(:xmax, u)..., _clamped(:ymax, u)...))
        E, d = Bramble.closure(C)
        return sum(abs2, nonzeros(E)) + sum(abs2, d)
    end
    return f
end

# The seeds of `p = [λ, β, g]` of each case: one parameter at width 1, and at width 2 `β`
# or `λ` with the datum, the `p = [β, g]` of #1226.
const _E = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
const _SEEDS = (("λ", (_E[1],)), ("β", (_E[2],)), ("datum", (_E[3],)),
    ("β and datum", (_E[2], _E[3])), ("λ and datum", (_E[1], _E[3])))

# Four tractions, each `β * u(i) + t(u, i)` with its own datum: the critic's sets of #1226,
# `:xmin` first (`_TractionsX`) or `:ymin` first (`_TractionsY`). The data are literals
# here: passed in as a tuple, the wrong tangent did not show.
_t(u, i) = _trac(u, i, 2.0)
struct _TractionsX end
struct _TractionsY end
function (::_TractionsX)(β)
    u -> (:xmin => β * u(1) + _t(u, 1) => 1.0,
        :xmin => β * u(2) + _t(u, 2) => 0.2, :ymin => β * u(1) + _t(u, 1) => 0.5,
        :ymin => β * u(2) + _t(u, 2) => 0.3)
end
function (::_TractionsY)(β)
    u -> (:ymin => β * u(1) + _t(u, 1) => 0.5,
        :ymin => β * u(2) + _t(u, 2) => 0.3, :xmin => β * u(1) + _t(u, 1) => 1.0,
        :xmin => β * u(2) + _t(u, 2) => 0.2)
end

# The `β` of a traction `β * u(i) + …` and its datum, through a call, as `constrained` reads
# a relation.
@noinline _scalar_and_datum(r) = ([_beta(r.op)], [r.data])
_beta(op::Bramble.FacetOperator) = op.ops[1][1].left_op.scalar

# `β² + datum` of the relation in slot `K` of the tuple `Bramble._composite_relations`
# parses from `R(β)`, on the padded meshes of `W × W`. Its derivative is `2β`; Enzyme's
# forward mode added the datum in slots 3 and 4 when the tuple came back from a call (#1226).
# The slot is a type parameter, so it folds as `constrained`'s own indexing does.
struct _SlotLoss{R, K, G, P}
    Gs::G
    P::P
end
_SlotLoss{R, K}(Gs, P) where {R, K} = _SlotLoss{R, K, typeof(Gs), typeof(P)}(Gs, P)
function (f::_SlotLoss{R, K})(p) where {R, K}
    r = Bramble._composite_relations(f.Gs, R()(p[1])(Bramble.trial_function(f.P)))[K]
    x, y = _scalar_and_datum(r)
    return sum(abs2, x) + sum(y)
end

@testset verbose=true "one-sided corner, forward (#1226)" begin
    if RUN_ENZYME
        Wb = gridspace(ghost_mesh(CpuSerial(), (7, 6)); ghosts = :boundary)
        Gs = map(((C, _, _),) -> mesh(C), Bramble._padded_leaves(Wb × Wb))
        P = Bramble.padded_space(Wb × Wb)
        tractions = (_TractionsX, _TractionsY)
        @testset "slot $k, $(nameof(R)), forward (#1226)" for R in tractions,
            k in 1:4

            f = _SlotLoss{R, k}(Gs, P)
            ref = DI.gradient(f, DI.AutoForwardDiff(), [2.0])
            @test only(ref) ≈ 4.0
            @test _relerr(DI.gradient(f, _auto(Enzyme.Forward), [2.0]), ref) < TOL
        end

        f, x = _corner_loss(Wb), [2.0, 2.0, 1.0]
        @testset "corner $what, width $(length(tx)) (#1226)" for (what,
                tx) in _SEEDS
            ref = collect(DI.pushforward(f, DI.AutoForwardDiff(), x, tx))
            @test minimum(abs, ref) > 1e-6
            @test _relerr(collect(DI.pushforward(f, _auto(Enzyme.Forward), x, tx)), ref) < TOL
        end
    end
end

end
