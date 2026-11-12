module ExtEnzymeRelationsThreadedTests

# Enzyme through the relations of `constrained` under `CpuThreaded` (gpena/Bramble.jl#1179).
# These cells sit apart from test/ext/enzyme_relations.jl, which they would push past
# 10 GB: a reverse cell compiles in about 10 minutes and 6 GB.
#
# The forms of a relation (`Bramble._assemble_relation`) are facet forms, whose weight
# holds the facet's `Bool` mask. In `_static_or_serial`'s `try` the band sweep's point
# was spilled with that mask and its padding as one `i64`, which Enzyme cannot type:
# forward stopped with `EnzymeNoTypeError`, and the reverse of `constrained`, which pulls
# back through that forward derivative (ext/BrambleEnzymeExt/relations.jl), with it.
# `Bramble._threaded_bands_ref!` now carries the call, ruled as the band runners are.
#
# Checked against `AutoForwardDiff()` to 1e-8 relative on the 7 by 6 non-uniform
# `ghost_mesh`, with a reference clearly non-zero: forward through `_assemble_relation`
# with the Robin coefficient and the datum the parameters, and `constrained` with either
# the parameter, in both modes. Each cell also checks that the rule of
# `_threaded_bands_ref!` ran, so no serial fallback passes. It needs two threads or more,
# since at one the policy runs its bands on the calling task.
#
# Enzyme is not in test/Project.toml: load it first, through a temporary environment
# (test/ext/enzyme_policy_cases.jl says how). Without it the file runs nothing, and with it
# the file runs only as its own `ENZYME_FILES` child, so its compiled code is released.

using Test
using Bramble
using Bramble: CpuThreaded
using ForwardDiff
using SparseArrays: nonzeros
import DifferentiationInterface as DI
using ..TestUtils: _have, ghost_mesh, TEST_FILE

const RUN_ENZYME = _have(:Enzyme) && TEST_FILE == "ext/enzyme_relations_threaded.jl"
RUN_ENZYME && @eval using Enzyme

const TOL = 1e-8

_relerr(g, ref) = maximum(abs, g .- ref) / maximum(abs, ref)

_auto(mode) = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(mode),
    function_annotation = Enzyme.Const)

_entries() = Base.get_extension(Bramble, :BrambleEnzymeExt)._rule_entries(:_threaded_bands_ref!)

_relations(β, g) = u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => β * u + ∂ₙ(u) => g,
    :ymin => u => 0.0, :ymax => ∂ₙ(u) => 1.0)

# The forms of the `:xmax` relation on the padded space `P`, Robin coefficient `p[1]` and
# datum `p[2]`, as `constrained` hands them to `Bramble._assemble_relation`.
function _relation_loss(P)
    function f(p)
        rels = Bramble._facet_relations(mesh(P),
            _relations(p[1], p[2])(Bramble.trial_function(P)))
        R, b = Bramble._assemble_relation(P, Bramble._ghost_relations(rels)[2])
        return sum(abs2, nonzeros(R)) + sum(abs2, b)
    end
    return f
end

# The loss of `constrained` on `W`: the datum of `:xmax` (`:data`) or its Robin coefficient
# (`:robin`) the parameter.
function _loss(W, what::Symbol)
    function f(p)
        β = what === :robin ? p[1] : 2.0
        g = what === :data ? p[1] : 1.0
        C = constrained(W, _relations(β, g))
        A, F = assemble(form(C, C, (U, V) -> innerₕ(-Δₕ(U), V) + innerₕ(U, V)),
            form(C, v -> innerₕ(x -> 1.0 + x[1], v)))
        return sum(abs2, nonzeros(A)) + sum(abs2, F)
    end
    return f
end

# `f`'s gradient by Enzyme in `mode` at `x`, relative to ForwardDiff's, and how many times
# the rule of `_threaded_bands_ref!` was entered.
function _check(f, mode, x)
    ref = DI.gradient(f, DI.AutoForwardDiff(), x)
    n = _entries()
    err = _relerr(DI.gradient(f, _auto(mode), x), ref)
    return err, ref, _entries() - n
end

@testset verbose=true "constrained, CpuThreaded, Enzyme (#1179)" begin
    if RUN_ENZYME && Threads.nthreads() > 1
        W = gridspace(ghost_mesh(CpuThreaded(), (7, 6)); ghosts = (:xmin, :xmax, :ymax))

        @testset "_assemble_relation, forward" begin
            err, ref, n = _check(_relation_loss(Bramble.padded_space(W)), Enzyme.Forward,
                [2.0, 1.3])
            @test minimum(abs, ref) > 1e-6
            @test err < TOL
            @test n > 0
        end

        modes = ((:reverse, Enzyme.Reverse), (:forward, Enzyme.Forward))
        @testset "$what, $mname" for what in (:data, :robin), (mname, mode) in modes

            err, ref, n = _check(_loss(W, what), mode, [1.3])
            @test abs(only(ref)) > 1e-6
            @test err < TOL
            @test n > 0
        end
    end
end

end
