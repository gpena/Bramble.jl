# Measurements moved out of `docs/src/internals/form.md`

## Factoring a shared argument: compile time

Fewer products means fewer compiled terms. The 3D scalar form of 27 distinct `innerₕ` terms in
`nterm-form.jl`, whose pairs share trial operators three at a time, compiles its first
assemble in 6.6–6.7 s factored against 16.4–17.1 s unfactored (one measurement at
commit `d35ba91a`, two interleaved runs each, 2 threads; the script is not in the repository
and no test checks the ratio).

## Shared sub-operators across terms

A composite form such as the symmetric gradient ⟨ε(u), ε(v)⟩ over a vector space repeats
the same operand sub-operators across its summands (gpena/Bramble.jl#347). Could warm
assembly evaluate each shared operand stencil once per point? `benchmark/shared_suboperators.jl` (three runs, 4
threads) bounds the saving: in 3D εc, 48–54% of warm evaluation (2.5–4.4 ms, 15 replay
units) is duplicated; in a 2D scalar form, 28–35% of 0.62–0.76 ms (3 units). A prototype
grouped consecutive term-outer units that share an operand type (at most 4 per group; 3D εc
gave three groups of 3 units), checked once per fill that the operands are identical, the
members write disjoint blocks and share a walked leaf, then evaluated each distinct operand
stencil and quadrature weight once per point and fed every member's product. It covered
serial composite replay from 2D up; its matrices were bitwise identical to the unshared
evaluation and refills allocated 0 B (as recorded in commit `76c0eb25` and
[gpena/Bramble.jl#347](https://github.com/gpena/Bramble.jl/issues/347); the prototype code is
not in the tree, so these results cannot be rerun from it). The rule, set before measuring, was to adopt it iff
the εc warm ratio is ≤ 0.90, the scalar warm ratio ≤ 1.02 and both first-assemble ratios
≤ 1.05. Interleaved ratios (after/before, run alone):

| form           | `assemble` (first)          | `assemble!` (warm)            |
|:-------------- |:--------------------------- |:----------------------------- |
| εc, 3D         | 1.133 (7.07 s → 8.01 s)     | 0.907 (3.64 ms → 3.30 ms)     |
| scalar, 2D     | 0.994                       | 0.995                         |

The decision is not to adopt it. It fails on εc first assembly (13.3% slower) and narrowly
on εc warm assembly (9.3% faster, short of 10%). Compile cost rose because term-outer units
of equal type share one compiled kernel. Grouping gives each group its own sweep and member
preparation, which breaks that sharing. The warm gain of about 9% is
well below the ~50% duplicated share, so most of that share is not recovered by evaluating
the operand stencils once. The replay stays term-outer, one unit at a time as described in
[One setup walk per term](@ref), and the prototype was not merged.

