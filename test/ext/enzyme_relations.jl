module ExtEnzymeRelationsTests

# Enzyme through `constrained` with relation data that depend on the parameter
# (gpena/Bramble.jl#1153). Enzyme's own reverse of the forms of a relation compiled one copy
# per facet and did not finish; ext/BrambleEnzymeExt/relations.jl is the reverse rule for
# `Bramble._assemble_relation` that replaces it. Forward mode never needed the rule.
#
# The gradients are checked through DifferentiationInterface against `AutoForwardDiff()` to
# 1e-8 relative, under `CpuSerial` on the 7 by 6 non-uniform mesh of `build_ghost_param`
# (test/space/ad_coverage.jl), with a reference clearly non-zero so a backend that drops the
# parameter cannot pass. The reverse cells are the costly ones: the first compiles in about
# 7 minutes and 3.6 GB at one thread, and the others reuse what it compiled where the types
# match (`robin` after `data`). The refusals of the rule are called on
# `Bramble._assemble_relation` itself, since DifferentiationInterface offers neither a space
# with a shadow nor a batch of a reverse gradient cheaply.
#
# A composite of constrained spaces is not pinned: full reverse `constrained` on a composite
# fails with an `EnzymeInternalError` in `_relation_rows`, which the rule does not touch.
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

const RUN_ENZYME = _have(:Enzyme) && TEST_FILE == "ext/enzyme_relations.jl"
RUN_ENZYME && @eval using Enzyme

const TOL = 1e-8
const X0 = [1.3]

_relerr(g, ref) = maximum(abs, g .- ref) / maximum(abs, ref)

_auto(mode) = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(mode),
    function_annotation = Enzyme.Const)

# The loss of `build_ghost_param` (test/space/ad_coverage.jl): relation data `g` (`:data`)
# or a Robin coefficient `β` (`:robin`) the parameter, on the space `W`. `datum` maps `g`
# to the datum of the `:xmax` relation: `identity` gives a number, a function of `g` a
# function of the point that captures `g` as a number.
function _loss(W, what::Symbol; datum = identity)
    function f(p)
        β = what === :robin ? p[1] : 2.0
        g = what === :data ? p[1] : 1.0
        d = datum(g)
        C = constrained(W,
            u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => β * u + ∂ₙ(u) => d,
                :ymin => u => 0.0, :ymax => ∂ₙ(u) => 1.0))
        a = form(C, C, (U, V) -> innerₕ(-Δₕ(U), V) + innerₕ(U, V))
        A, F = assemble(a, form(C, v -> innerₕ(x -> 1.0 + x[1], v)))
        return sum(abs2, nonzeros(A)) + sum(abs2, F)
    end
    return f
end

# `f`'s gradient by Enzyme in `mode`, relative to ForwardDiff's.
function _enzyme_vs_forwarddiff(f, mode)
    ref = DI.gradient(f, DI.AutoForwardDiff(), X0)
    return _relerr(DI.gradient(f, _auto(mode), X0), ref), ref
end

# The forms of the `:xmax` relation of the padded space `P`, its datum `d`, as `constrained`
# hands them to `Bramble._assemble_relation`.
function _xmax_relation(P, d)
    rels = Bramble._facet_relations(mesh(P),
        (u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => 2.0 * u + ∂ₙ(u) => d,
            :ymin => u => 0.0, :ymax => ∂ₙ(u) => 1.0))(Bramble.trial_function(P)))
    return Bramble._ghost_relations(rels)[2]
end

function _relation_loss(P, r)
    R, b = Bramble._assemble_relation(P, r)
    return sum(abs2, nonzeros(R)) + sum(abs2, b)
end
_scaled_loss(P, r, v) = v[1] * _relation_loss(P, r)

# Reverse `Active` on the scalar loss `_scaled_loss`, the relation `r` and the space `P`
# annotated as given, with `v` (a scalar vector or a batch of them) the `Duplicated` rest.
function _reverse(P, r, v)
    return Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(_scaled_loss), Enzyme.Active, P, r, v)
end

@testset verbose=true "constrained relations, Enzyme (#1153)" begin
    if RUN_ENZYME
        W = gridspace(ghost_mesh(CpuSerial(), (7, 6)); ghosts = (:xmin, :xmax, :ymax))

        @testset "refusals" begin
            P = Bramble.padded_space(W)
            r = _xmax_relation(P, 1.0)
            v = Enzyme.Duplicated([1.5], [0.0])

            @testset "array captured by a datum" begin
                arr = [0.7]
                ra = _xmax_relation(P, x -> arr[1])
                dup = Enzyme.Duplicated(ra, Enzyme.make_zero(ra))
                @test_throws ArgumentError _reverse(Enzyme.Const(P), dup, v)
                @test_throws "facet `xmax`" _reverse(Enzyme.Const(P), dup, v)
            end

            @testset "batch of two tangents" begin
                batch = Enzyme.BatchDuplicated([1.5], ([0.0], [0.0]))
                ar = Enzyme.Active(r)
                @test_throws ArgumentError _reverse(Enzyme.Const(P), ar, batch)
                @test_throws "does not support batched" _reverse(Enzyme.Const(P), ar, batch)
            end

            @testset "space with a shadow of its own" begin
                dP = Enzyme.Duplicated(P, Enzyme.make_zero(P))
                @test_throws ArgumentError _reverse(dP, Enzyme.Active(r), v)
                @test_throws "shadow of its own" _reverse(dP, Enzyme.Active(r), v)
            end
        end

        modes = ((:reverse, Enzyme.Reverse), (:forward, Enzyme.Forward))
        @testset "$what, $mname" for (mname, mode) in modes, what in (:data, :robin)

            err, ref = _enzyme_vs_forwarddiff(_loss(W, what), mode)
            @test abs(only(ref)) > 1e-6
            @test err < TOL
        end

        @testset "datum of a captured number, reverse" begin
            f = _loss(W, :data; datum = g -> (x -> g * x[2]))
            err, ref = _enzyme_vs_forwarddiff(f, Enzyme.Reverse)
            @test abs(only(ref)) > 1e-6
            @test err < TOL
        end
    end
end

end
