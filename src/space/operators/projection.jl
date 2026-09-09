##############################################################################
#                                                                            #
#            Projecting a function onto the space of grid functions          #
#                                                                            #
##############################################################################

#=
# projection.jl

`CONTEXT.md` defines restriction and cell average as the same operation with different
rules: both project a continuous function onto the space of grid functions, one by
evaluating it at the mesh points, the other by averaging it over each cell. This file is
that shared operation, with the rule as an argument (gpena/Bramble.jl#51).

The driver decides the three things neither rule should have an opinion about -- how the
space is shaped (scalar, composite on one mesh, composite across several), whether a marker
mask applies, and whether the sweep is threaded -- and asks the rule only for a per-point
kernel. `Rₕ!` (`restriction.jl`) and `avgₕ!` (`cell_average.jl`) keep their own public
signatures and docstrings and become thin wrappers over `project!`.

**The threading trap this closes.** Before this, the masked paths of both families were
plain serial loops that ignored `execution_policy`, while the unmasked ones threaded. A
caller who added `markers = (...)` silently lost threading, with nothing in either
interface saying so. Masking is now a property of the kernel (`_MaskedKernel` below), not
of the sweep, so both go through the same `_cpu_threaded_for!` and the policy is honoured
either way.
=#

"""
    ProjectionRule

Supertype of the rules [`project!`](@ref) can apply: what a grid function's value at a
point should be, given a continuous function. [`PointValue`](@ref) evaluates at the mesh
point, [`CellAverage`](@ref) averages over the cell.
"""
abstract type ProjectionRule end

"""
    PointValue(f)

Project `f` by evaluating it at each mesh point. The rule behind [`Rₕ!`](@ref).
"""
struct PointValue{F} <: ProjectionRule
    f::F
end

"""
    CellAverage(f, nq::Val)

Project `f` by averaging it over each cell with `nq` quadrature points per direction. The
rule behind [`avgₕ!`](@ref).
"""
struct CellAverage{F,NQ} <: ProjectionRule
    f::F
    nq::Val{NQ}
end

# --- what a rule has to provide, implemented in each rule's own file --------------- #

"""
    _rule_kernel(rule, space) -> callable

A concretely typed callable mapping a linear grid index to the scalar value `rule` gives
at that point. Kept a named struct per rule rather than a closure: measured, an anonymous
closure over the captures here takes a miscompiled path that allocates per grid point
(gpena/Bramble.jl#64).
"""
function _rule_kernel end

"""
    _rule_scatter_kernel(rule, space, ::Val{NC}) -> callable

As [`_rule_kernel`](@ref), for a rule whose function returns all `NC` leaf values at once,
so it is evaluated once per point and scattered across the leaves.
"""
function _rule_scatter_kernel end

"""
    _rule_component(rule, k) -> ProjectionRule

`rule` restricted to leaf `k` of a composite space whose leaves do not share one mesh, so
there is no shared grid point to evaluate once and scatter.
"""
function _rule_component end

# --- masking, as a property of the kernel rather than of the sweep ----------------- #

# The marker masks, as a tuple whose length is a type parameter so the `||` chain below
# unrolls. `index_in_marker` is a dictionary lookup returning the mesh's own array, so this
# copies nothing.
@inline _marker_masks(Ωₕ, markers::NTuple{N,Symbol}) where {N} =
    ntuple(i -> index_in_marker(Ωₕ, markers[i]), Val(N))

@inline _in_any_marker(::Tuple{}, i) = false
@inline _in_any_marker(masks::Tuple, i) =
    (@inbounds masks[1][i]) || _in_any_marker(Base.tail(masks), i)

# Off-region entries are written as zero rather than skipped. The serial loops this
# replaces did `fill!(raw, 0)` first and then wrote only the masked entries, which leaves
# exactly the same array -- but writing every entry is what lets the masked sweep be the
# same threaded sweep as the unmasked one.
struct _MaskedKernel{K,M,Z}
    kernel::K
    masks::M
    zeroval::Z
end

@inline (mk::_MaskedKernel)(i) = _in_any_marker(mk.masks, i) ? mk.kernel(i) : mk.zeroval

# --- the driver -------------------------------------------------------------------- #

"""
    project!(uₕ::VectorElement, rule, markers = ()) -> uₕ

Project onto `uₕ` in place according to `rule`, optionally restricted to the union of the
labelled marker regions, leaving every other entry zero.

Handles the space's shape and the execution policy; `rule` supplies only the per-point
value. See [`PointValue`](@ref) and [`CellAverage`](@ref), and [`Rₕ!`](@ref) / [`avgₕ!`](@ref)
for the public spellings.
"""
function project! end

@inline function project!(
    uₕ::VectorElement{<:ScalarGridSpace},
    rule::ProjectionRule,
    markers::NTuple{N,Symbol}=NTuple{0,Symbol}(),
) where {N}
    sp = space(uₕ)
    Ωₕ = mesh(sp)
    raw = parent(uₕ)
    n = length(indices(Ωₕ))
    kernel = _rule_kernel(rule, sp)
    _cpu_threaded_for!(
        execution_policy(sp), raw, 1:n, _apply_mask(kernel, Ωₕ, markers, eltype(raw))
    )
    return uₕ
end

@inline function project!(
    uₕ::VectorElement{<:CompositeGridSpace},
    rule::ProjectionRule,
    markers::NTuple{N,Symbol}=NTuple{0,Symbol}(),
) where {N}
    comps = components(uₕ)
    if _shares_one_mesh(comps)
        sp = space(uₕ)
        Ωₕ = mesh(sp)
        raws = map(parent, comps)
        n = length(indices(Ωₕ))
        NC = length(comps)
        kernel = _rule_scatter_kernel(rule, sp, Val(NC))
        zeros_nc = ntuple(_ -> zero(eltype(first(raws))), Val(NC))
        _cpu_threaded_scatter_for!(
            execution_policy(sp), raws, 1:n, _apply_mask(kernel, Ωₕ, markers, zeros_nc)
        )
    else
        # No shared grid point exists, so `rule`'s function is re-evaluated at each leaf's
        # own points, keeping only that leaf's entry (gpena/Bramble.jl#78).
        ntuple(
            k -> (project!(comps[k], _rule_component(rule, k), markers); nothing),
            Val(length(comps)),
        )
    end
    return uₕ
end

# One rule per leaf: each is already independent. `map` over both tuples rather than
# `ntuple` indexing a shared count -- it needs no leaf count, stays correct under any
# nesting, and errors on a length mismatch the way it always did.
@inline function project!(
    uₕ::VectorElement{<:CompositeGridSpace},
    rules::Tuple,
    markers::NTuple{N,Symbol}=NTuple{0,Symbol}(),
) where {N}
    map((c, r) -> project!(c, r, markers), components(uₕ), rules)
    return uₕ
end

# A one-component space is a scalar space, so a 1-tuple of rules must still work.
@inline project!(
    uₕ::VectorElement{<:ScalarGridSpace},
    rules::Tuple{Any},
    markers::NTuple{N,Symbol}=NTuple{0,Symbol}(),
) where {N} = project!(uₕ, rules[1], markers)

# Unmasked stays the bare kernel, so the common path carries no mask check at all.
@inline _apply_mask(kernel, Ωₕ, ::NTuple{0,Symbol}, zeroval) = kernel
@inline _apply_mask(kernel, Ωₕ, markers::NTuple{N,Symbol}, zeroval) where {N} =
    _MaskedKernel(kernel, _marker_masks(Ωₕ, markers), _zero_of(zeroval))

@inline _zero_of(::Type{T}) where {T} = zero(T)
@inline _zero_of(z::Tuple) = z
