# How much of a warm stencil evaluation is duplicated work (gpena/Bramble.jl#347).
#
# For each #347 form, the time to evaluate every summand's local stencil over the grid (what
# the term-outer replay does) is set against the time spent re-evaluating sub-operators that
# an earlier summand, or an earlier operand of the same summand, already evaluated at that
# point. A sub-operator is counted once at its outermost shared occurrence, so a shared
# `D₋ₓ(u)` inside a shared `M₊ᵧ(D₋ₓ(u))` is not counted twice. The duplicated share is the
# upper bound on what common-subexpression evaluation could save.
#
# Run: julia --project=benchmark --startup-file=no --threads=4 benchmark/shared_suboperators.jl

using Bramble, Random
using Bramble: D₋ₓ, D₋ᵧ, M₊ᵧ, local_stencil, markers, indices, LazyOp, BilinearProduct,
               leaf_spaces_offsets, blocks, _bind_interp_spaces, _walked_leaf, _summands

# The forms of `.claude/plans/v3-16-0-to-v3-20-0-checks/cse_forms.jl`, verbatim and seeded.
Random.seed!(347)
const V = gridspace(
    mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)), (24, 24, 24),
        (false, false, false)),
    Val(3))
const εc = form(V, V, (u, v) -> innerₕ(εcₕ(u), εcₕ(v)))
const W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (201, 201),
    (false, false)))
const κ = Rₕ(W, x -> 1 + x[1] * x[2])
const sc = form(W, W,
    (u, v) -> innerₕ(M₊ᵧ(D₋ₓ(u)), v) + innerₕ(D₋ₓ(u), D₋ₓ(v)) +
              innerₕ(κ * D₋ₓ(u), D₋ᵧ(v)))

# Every (summand, walked leaf) unit the replay evaluates, blocks expanded, pairs not fused.
function units(f)
    if f.trial_space isa Bramble.CompositeGridSpace
        tl, sl = leaf_spaces_offsets(f.trial_space), leaf_spaces_offsets(f.test_space)
        out = Any[]
        for t in _summands(f.ast), blk in blocks(t, tl, sl)
            b = _bind_interp_spaces(t, blk.trial_leaf, blk.test_leaf)
            push!(out, (b, _walked_leaf(b, blk.trial_leaf, blk.test_leaf)))
        end
        return out
    end
    b = _bind_interp_spaces(f.ast, f.trial_space, f.test_space)
    sp = _walked_leaf(b, f.trial_space, f.test_space)
    return Any[(t, sp) for t in _summands(b)]
end

# One sweep of `op`'s stencil over the grid, the coefficients summed so nothing is elided.
@noinline function sweep(op, sp)
    m = mesh(sp)
    mk = markers(m)
    idx = indices(m)
    lin = LinearIndices(idx)
    acc = 0.0
    @inbounds for I in idx
        for e in local_stencil(op, sp, I, mk, lin[I])
            acc += sum(last(e))
        end
    end
    return acc
end

best(op, sp; n = 30) = (sweep(op, sp); minimum(@elapsed(sweep(op, sp)) for _ in 1:n))

children(op) = Tuple(getfield(op, k) for k in fieldnames(typeof(op))
if getfield(op, k) isa LazyOp)

# Outermost repeated sub-operators: walk each operand top-down; a node already seen on the
# same leaf is duplicated work and is not descended into. Leaves cost nothing and are skipped.
function duplicated_time(us)
    seen = Set{Any}()
    dup = 0.0
    walk(op, sp) = begin
        ch = children(op)
        isempty(ch) && return
        key = (op, objectid(sp))
        if key in seen
            dup += best(op, sp)
            return
        end
        push!(seen, key)
        foreach(c -> walk(c, sp), ch)
    end
    for (t, sp) in us
        while t isa Bramble.OperatorScale
            t = t.inner_op  # a summand's scalar weight, not a shared sub-operator
        end
        t isa BilinearProduct || error("unexpected summand $(typeof(t))")
        walk(t.left_op, sp)
        walk(t.right_op, sp)
    end
    return dup
end

function report(label, f)
    us = units(f)
    total = sum(best(t, sp) for (t, sp) in us)
    dup = duplicated_time(us)
    println(label, ": duplicated=", round(100 * dup / total; digits = 1), "% of ",
        round(total; sigdigits = 4), " s (", length(us), " units)")
end

report("εc 3D", εc)
report("scalar 2D", sc)
