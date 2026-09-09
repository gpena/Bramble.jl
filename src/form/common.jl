# common.jl

# ==============================================================================
# 1. Zero-Allocation Spacing & Tuple Utilities
# ==============================================================================

@inline _get_component(x::Tuple, dim::Int) = x[dim]
@inline _get_component(x::Number, dim::Int) = x

"""
    get_spacing(mesh, I, dim::Int) -> Real

Grid spacing in coordinate direction `dim` at Cartesian index `I`.
"""
@inline get_spacing(mesh, I, dim::Int) = _get_component(spacing(mesh, I), dim)

"""
    get_forward_spacing(mesh, I, dim::Int) -> Real

Forward grid spacing in coordinate direction `dim` at Cartesian index `I`.
"""
@inline get_forward_spacing(mesh, I, dim::Int) =
    _get_component(forward_spacing(mesh, I), dim)

"""
    get_half_spacing(mesh, I, dim::Int) -> Real

Half-grid spacing in coordinate direction `dim` at Cartesian index `I`.
"""
@inline get_half_spacing(mesh, I, dim::Int) = _get_component(half_spacing(mesh, I), dim)

"""
    shift_offset(offset::NTuple{D, Int}, dim::Int, delta::Int) -> NTuple{D, Int}

Shifts a Cartesian offset tuple by `delta` in dimension `dim`.
"""
@inline shift_offset(offset::NTuple{D,Int}, dim::Int, delta::Int) where {D} =
    ntuple(i -> i == dim ? offset[i] + delta : offset[i], Val(D))

"""
    zero_offset(::Val{D}) -> NTuple{D, Int}

Zero-initialized offset tuple of dimension `D`.
"""
@inline zero_offset(::Val{D}) where {D} = ntuple(x -> 0, Val(D))

"""
    shift_stencil(inner::Tuple, ::Val{Dim}, delta)

Shifts all coordinates in a stencil tuple by `delta` in dimension `Dim`.

`map` over a `Tuple` unrolls and stays type-stable at compile time in Julia — measured
against a `@generated` version this once was (gpena/Bramble.jl#63): identical zero
allocations and identical inferred return type, so the code generation bought nothing
here.
"""
@inline shift_stencil(inner::Tuple, ::Val{Dim}, ::Val{Delta}) where {Dim,Delta} =
    map(t -> (shift_offset(t[1], Dim, Delta), t[2]), inner)

@inline shift_stencil(inner::Tuple, ::Val{Dim}, delta::Int) where {Dim} =
    map(t -> (shift_offset(t[1], Dim, delta), t[2]), inner)

# `(left..., right...)` is already resolved at compile time for tuples; no metaprogramming
# needed (gpena/Bramble.jl#63).
@inline concatenate_stencils(left::Tuple, right::Tuple) = (left..., right...)

# Recursion on tuple structure (`_flatten_tuples`, below) rather than `Iterators.flatten`:
# measured (gpena/Bramble.jl#63), the latter does not stay type-stable for a tuple-of-tuples
# and allocates (752 B for a 2×3 outer product), where this and the `@generated` version it
# replaces both allocate 0.
@inline _flatten_tuples(::Tuple{}) = ()
@inline _flatten_tuples(t::Tuple) = (first(t)..., _flatten_tuples(Base.tail(t))...)

@inline function multiply_stencils_bilinear(left::Tuple, right::Tuple, vol::Number)
    _flatten_tuples(map(l -> map(r -> (l[1], r[1], l[2] * r[2] * vol), right), left))
end

@inline function multiply_stencils_linear(left::Tuple, right::Tuple, vol::Number)
    _flatten_tuples(map(l -> map(r -> (r[1], l[2] * r[2] * vol), right), left))
end

@inline scale_stencil(inner::Tuple, scalar::Number) =
    map(t -> (Base.front(t)..., t[end] * scalar), inner)

"""
    sum_stencil_values(stencil::Tuple)

The sum of a stencil's coefficients, ignoring its offsets entirely.

Required by `_contracted_left_stencil` (`form/operators/inner.jl`) for a source-only
subtree's own `local_stencil`: not the offsets, which mean nothing for a value that
contributes no matrix structure, only their total. `false` rather than `0` or `zero(T)` is
the empty-stencil answer: [`RegionRestriction`](@ref) can legitimately produce `()` for a
point outside its region, and there is no `T` to call `zero` on when there are no entries to
read one from; `false` promotes to whatever numeric type the other entries (or, empty, the
caller's own multiplication) turn out to have — exactly `sum(f, itr; init = false)`'s own
behavior, which is what this calls. This used to be its own `@generated` unrolled fold
"like every other stencil-algebra primitive" in this file; measured against `sum` directly
(gpena/Bramble.jl#63), identical zero allocations and identical inferred type, so the
`@generated` version bought nothing that `sum` was not already providing.
"""
@inline sum_stencil_values(stencil::Tuple) = sum(t -> t[end], stencil; init=false)

# ==============================================================================
# 2. Abstract Syntax Tree (AST) Nodes
# ==============================================================================

"""
    TrialFunction{D} <: LazyOp{D}

An AST node representing the symbolic trial function \$u\$ in a bilinear form.
"""
struct TrialFunction{D} <: LazyOp{D} end

"""
    TestFunction{D} <: LazyOp{D}

An AST node representing the symbolic test function \$v\$ in a form.
"""
struct TestFunction{D} <: LazyOp{D} end

"""
    IndexedTrialFunction{D} <: LazyOp{D}

An AST node representing the symbolic trial function for a specific **component**
of a composite trial space. Carries a runtime `component_idx` identifying which
leaf scalar space (1-based, depth-first order) it belongs to. Used by
a coupled form to route stencil contributions to the correct block.
"""
struct IndexedTrialFunction{D} <: LazyOp{D}
    component_idx::Int
end

"""
    IndexedTestFunction{D} <: LazyOp{D}

An AST node representing the symbolic test function for a specific **component**
of a composite test space. Carries a runtime `component_idx`. Used by
a coupled form to route stencil contributions to the correct block.
"""
struct IndexedTestFunction{D} <: LazyOp{D}
    component_idx::Int
end

"""
    SourceFunction{D,F} <: LazyOp{D}

An AST node representing a source term defined by a continuous function.
"""
struct SourceFunction{D,F} <: LazyOp{D}
    func::F
end

"""
    SourceVector{D,VType} <: LazyOp{D}

An AST node representing a source term defined by a discrete vector of values.

Note the division of labour with `GridFunctionScale`, which also carries values per
grid point. A `SourceFunction` holds a function of position, `f(x)`, evaluated at the
point. A `Function` inside a `GridFunctionScale` is something else entirely: a
zero-argument thunk returning the vector or number to scale by, called as `f()` both
here and in `resolve_ast`. It defers building that vector until the form is resolved.

So `(x -> x[1]) * D₋ₓ(u)` does not do what it reads as: the thunk call fails, because the
function wants a point. A function of position belongs in a `SourceFunction`, or should be
restricted to the grid with `Rₕ` first and passed as the vector it becomes.
"""
struct SourceVector{D,VType<:AbstractVector} <: LazyOp{D}
    vec::VType
end

"""
    SourceConstant{D, T} <: LazyOp{D}

An AST node representing a source term that is the same number everywhere on the mesh.

`SourceFunction` reaches this value the general way, through `f(point(m, I))`: a real
cost when `f` is `x -> l`, discarding the point it just computed, at every grid point of
every assembly. `SourceConstant` skips `point` entirely; measured behind a function
barrier, assembling a constant source is 1.6–2.6× faster than through `SourceFunction`,
the ratio growing with `ndofs` rather than staying fixed, so this is a per-point saving
rather than one-off overhead. `source_number` is what builds one from a literal `Number`.
"""
struct SourceConstant{D,T} <: LazyOp{D}
    value::T
end

"""
    AbsoluteColumn

A stencil entry's trial slot, naming a column of the trial space directly rather than an
offset from the point being evaluated.

Every other node's stencil says "this many points from here, on the mesh being walked", which
is what lets `shift_stencil` compose operators by relabelling. An interpolation cannot say
that: the trial degrees of freedom it reaches live on a different mesh, and which ones
depends on where the point falls (`locate_cell`). So it names them outright, and the bilinear
consumers resolve the two kinds of entry by dispatch.
"""
struct AbsoluteColumn
    col::Int
end

# --- Whether an operator's stencil may be shifted by relabelling its offsets -------- #
#
# Every wrapper that reaches a neighbour (the differences, the averages, Sₓ, the jumps)
# evaluates its inner operator once, at the point being visited, and then produces the
# neighbour's contribution by adding a constant to the offsets (`shift_stencil`). That is
# exact whenever the inner stencil is the same shape everywhere, which is to say for a trial
# or test function however deeply wrapped: relabelling `(0,)` as `(-1,)` says precisely what
# evaluating at `I - e` would have said, and evaluating once instead of twice is why an
# operator tower costs nothing to compose.
#
# Two kinds of node break that. An interpolation's entries name absolute columns chosen by
# `locate_cell` from the point's own coordinates, and a source's entries carry the function's
# value at the point: for neither does adding one to an offset produce what the neighbour
# holds. Such a node has to be re-evaluated at the shifted point instead, which is what this
# trait selects between. `stencil_shift_trait`'s ladder lives in
# `form/operators/interpolation.jl`, after every node type it has to answer for exists:
# marking a source point-dependent there is also what a source-only subtree's own
# contraction (`_contracted_left_stencil`, `form/operators/inner.jl`) reads its values through.
#
# A Holy trait rather than a `Bool` predicate on purpose: the choice is made by dispatch on a
# singleton, so neither branch is ever compiled into the other's code path, and the
# translation-invariant path stays the single `shift_stencil` call it is today.
abstract type StencilShiftTrait end

"""
    TranslationInvariantStencil <: StencilShiftTrait

The operator's stencil has the same shape at every point, so a neighbour's contribution is
its own stencil with the offsets relabelled ([`shift_stencil`](@ref)).
"""
struct TranslationInvariantStencil <: StencilShiftTrait end

"""
    PointDependentStencil <: StencilShiftTrait

The operator's stencil depends on *where* it is evaluated in a way relabelling cannot
express, so a neighbour's contribution has to be obtained by evaluating the operator again at
the neighbour's own point.
"""
struct PointDependentStencil <: StencilShiftTrait end

# A sum is translation invariant only if both summands are.
@inline _combine_shift_traits(
    ::TranslationInvariantStencil, ::TranslationInvariantStencil
) = TranslationInvariantStencil()
@inline _combine_shift_traits(::StencilShiftTrait, ::StencilShiftTrait) =
    PointDependentStencil()

@inline stencil_shift_trait(::LazyOp) = TranslationInvariantStencil()

@inline _shift_delta(::Val{Delta}) where {Delta} = Delta
@inline _shift_delta(delta::Int) = delta

# The point `delta` steps away in direction `Dim`, clamped to the mesh.
#
# Clamping is safe at most callers: an operator that reaches outside masks its own
# out-of-range half to a zero coefficient (`mask = I[Dim] == 1 ? 0 : 1` and its twins), so the
# clamped point's entries are multiplied by zero and only ever contribute an explicit zero.
# Evaluating without clamping is what is not safe: `point(m, I)` off the grid is out of
# bounds, where the offsets a translation-invariant shift produces are merely filtered later.
#
# `ShiftNode` is the one caller this does not fully cover: it carries no mask of its own
# (unlike every difference, average and jump), and relies for its offset path on the
# assembly's own bounds check dropping an out-of-range offset: a fallback with nothing left
# to check once a `PointDependentStencil` has already reduced the shift to a bare value. For a
# source specifically, it checks [`_in_grid`](@ref) itself rather than trusting the clamp.
# An interpolation is the other `PointDependentStencil` node and is unaffected: clamping is
# its own already-correct behaviour (`locate_cell` extrapolates by design), so `ShiftNode`
# only takes the `_in_grid` branch when its inner operand is source-only.
@inline function _clamped_shift(
    m, I::CartesianIndex{D}, ::Val{Dim}, delta::Int
) where {D,Dim}
    dims = npoints(m, Tuple)
    j = clamp(I[Dim] + delta, 1, dims[Dim])
    return CartesianIndex(ntuple(d -> d == Dim ? j : I[d], Val(D)))
end

"""
    _in_grid(space, I::CartesianIndex) -> Bool

Whether `I` names a real point of `space`'s mesh.

The check [`ShiftNode`](@ref)'s own `local_stencil` makes for a `PointDependentStencil` inner
operator, in place of trusting `_clamped_shift`'s clamp; see the note there for why that
trust does not extend to this one caller.
"""
@inline _in_grid(space, I::CartesianIndex{D}) where {D} =
    checkbounds(Bool, LinearIndices(indices(mesh(space))), I)

"""
    shifted_inner_stencil(inner_op, inner, space, I, markers, ::Val{Dim}, delta)

The stencil `inner_op` contributes `delta` points away in direction `Dim`, given `inner`, its
stencil already evaluated at `I`.

The one place the "shift by relabelling" assumption is made, so the one place a node that
cannot be relabelled has to be handled: [`TranslationInvariantStencil`](@ref) relabels
`inner`'s offsets and never touches `inner_op` again, [`PointDependentStencil`](@ref)
discards `inner` and evaluates `inner_op` at the shifted point instead. Both produce a tuple
of the same static length, since it is the same operator either way, so the callers'
`concatenate_stencils` sees exactly the shape it always did.
"""
@inline function shifted_inner_stencil(
    inner_op, inner, space, I::CartesianIndex{D}, markers, ::Val{Dim}, delta
) where {D,Dim}
    return _shifted_inner_stencil(
        stencil_shift_trait(inner_op), inner_op, inner, space, I, markers, Val(Dim), delta
    )
end

@inline _shifted_inner_stencil(
    ::TranslationInvariantStencil,
    inner_op,
    inner,
    space,
    I::CartesianIndex{D},
    markers,
    ::Val{Dim},
    delta,
) where {D,Dim} = shift_stencil(inner, Val(Dim), delta)

@inline function _shifted_inner_stencil(
    ::PointDependentStencil,
    inner_op,
    inner,
    space,
    I::CartesianIndex{D},
    markers,
    ::Val{Dim},
    delta,
) where {D,Dim}
    m = mesh(space)
    Ishift = _clamped_shift(m, I, Val(Dim), _shift_delta(delta))
    return local_stencil(
        inner_op, space, Ishift, markers, LinearIndices(indices(m))[Ishift]
    )
end

# ==============================================================================
# 3. Form API & Bramble Standard Mapping
# ==============================================================================

# Indexing a node by component (`v(1)`, and distribution through whatever is built on
# top of it) lives in form/component.jl, included after the operator files because it needs
# every node type in its signatures.

"""
    trial_function(::Val{D}) -> TrialFunction{D}

Constructs a `TrialFunction` of dimension `D`.
"""
trial_function(::Val{D}) where {D} = TrialFunction{D}()

"""
    test_function(::Val{D}) -> TestFunction{D}

Constructs a `TestFunction` of dimension `D`.
"""
test_function(::Val{D}) where {D} = TestFunction{D}()

"""
    source_function(f, ::Val{D}) -> SourceFunction{D, typeof(f)}

Constructs a `SourceFunction` wrapping function `f`.
"""
source_function(f, ::Val{D}) where {D} = SourceFunction{D,typeof(f)}(f)
