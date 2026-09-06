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
@inline get_forward_spacing(mesh, I, dim::Int) = _get_component(forward_spacing(mesh, I), dim)

"""
    get_half_spacing(mesh, I, dim::Int) -> Real

Half-grid spacing in coordinate direction `dim` at Cartesian index `I`.
"""
@inline get_half_spacing(mesh, I, dim::Int) = _get_component(half_spacing(mesh, I), dim)

"""
    shift_offset(offset::NTuple{D, Int}, dim::Int, delta::Int) -> NTuple{D, Int}

Shifts a Cartesian offset tuple by `delta` in dimension `dim`.
"""
@inline shift_offset(offset::NTuple{D, Int}, dim::Int, delta::Int) where {D} = ntuple(
    i -> i == dim ? offset[i] + delta : offset[i], Val(D))

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
@inline shift_stencil(inner::Tuple, ::Val{Dim}, ::Val{Delta}) where {Dim, Delta} = map(
    t -> (shift_offset(t[1], Dim, Delta), t[2]), inner)

@inline shift_stencil(inner::Tuple, ::Val{Dim}, delta::Int) where {Dim} = map(
    t -> (shift_offset(t[1], Dim, delta), t[2]), inner)

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

@inline scale_stencil(inner::Tuple, scalar::Number) = map(
    t -> (Base.front(t)..., t[end] * scalar), inner)

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
@inline sum_stencil_values(stencil::Tuple) = sum(t -> t[end], stencil; init = false)

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
struct SourceFunction{D, F} <: LazyOp{D}
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
struct SourceVector{D, VType <: AbstractVector} <: LazyOp{D}
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
struct SourceConstant{D, T} <: LazyOp{D}
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
    ::TranslationInvariantStencil, ::TranslationInvariantStencil) = TranslationInvariantStencil()
@inline _combine_shift_traits(::StencilShiftTrait, ::StencilShiftTrait) = PointDependentStencil()

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
@inline function _clamped_shift(m, I::CartesianIndex{D}, ::Val{Dim}, delta::Int) where {
        D, Dim}
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
@inline _in_grid(space, I::CartesianIndex{D}) where {D} = checkbounds(
    Bool, LinearIndices(indices(mesh(space))), I)

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
@inline function shifted_inner_stencil(inner_op, inner, space, I::CartesianIndex{D},
        markers, ::Val{Dim}, delta) where {D, Dim}
    return _shifted_inner_stencil(stencil_shift_trait(inner_op), inner_op, inner, space, I,
        markers, Val(Dim), delta)
end

@inline _shifted_inner_stencil(::TranslationInvariantStencil, inner_op, inner, space,
    I::CartesianIndex{D}, markers, ::Val{Dim}, delta) where {D, Dim} = shift_stencil(
    inner, Val(Dim), delta)

@inline function _shifted_inner_stencil(::PointDependentStencil, inner_op, inner, space,
        I::CartesianIndex{D}, markers, ::Val{Dim}, delta) where {D, Dim}
    m = mesh(space)
    Ishift = _clamped_shift(m, I, Val(Dim), _shift_delta(delta))
    return local_stencil(inner_op, space, Ishift, markers,
        LinearIndices(indices(m))[Ishift])
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
source_function(f, ::Val{D}) where {D} = SourceFunction{D, typeof(f)}(f)

# Import modularized operator and product logic
include("operators/difference.jl")
include("operators/jump.jl")
include("operators/average.jl")
include("operators/restriction.jl")
include("operators/inner.jl")
include("operators/interpolation.jl")

# ==============================================================================
# 4. Zero-Allocation Stencil Evaluators
# ==============================================================================

@inline local_stencil(
    ::TrialFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D} = ((
    zero_offset(Val(D)), 1),)
@inline local_stencil(
    ::TestFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D} = ((
    zero_offset(Val(D)), 1),)
@inline local_stencil(::IndexedTrialFunction{D}, space, I::CartesianIndex{D},
    markers, lin_idx::Int) where {D} = ((zero_offset(Val(D)), 1),)
@inline local_stencil(::IndexedTestFunction{D}, space, I::CartesianIndex{D},
    markers, lin_idx::Int) where {D} = ((zero_offset(Val(D)), 1),)

@inline function local_stencil(
        op::SourceFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    m = mesh(space)
    x = point(m, I)
    return ((zero_offset(Val(D)), op.func(x)),)
end

@inline function local_stencil(
        op::SourceVector{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    return ((zero_offset(Val(D)), op.vec[lin_idx]),)
end

@inline function local_stencil(
        op::SourceConstant{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    return ((zero_offset(Val(D)), op.value),)
end

@inline function local_stencil(
        op::OperatorAdd, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    left_stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
    right_stencil = local_stencil(op.right_op, space, I, markers, lin_idx)
    return concatenate_stencils(left_stencil, right_stencil)
end

@inline function local_stencil(
        op::OperatorScale, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return scale_stencil(inner, op.scalar)
end

@inline function local_stencil(
        op::OperatorScale{D, <:Base.RefValue}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return scale_stencil(inner, op.scalar[])
end

@inline function local_stencil(
        op::GridFunctionScale, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    grid_fn = op.grid_function
    local_val = if grid_fn isa Function
        val = grid_fn()
        val isa Number ? val : val[lin_idx]
    else
        grid_fn isa Number ? grid_fn : grid_fn[lin_idx]
    end
    return scale_stencil(inner, local_val)
end

@inline local_stencil(op::IdentityOperator{D}, space, I::CartesianIndex{D},
    markers, lin_idx::Int) where {D} = ((zero_offset(Val(D)), 1),)
@inline local_stencil(
    op::ZeroOperator{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D} = ((
    zero_offset(Val(D)), 0),)

# ==============================================================================
# 5. AST Resolution & Thunk Eval
# ==============================================================================

function resolve_ast(op::OperatorAdd{D}) where {D}
    OperatorAdd{D, typeof(resolve_ast(op.left_op)), typeof(resolve_ast(op.right_op))}(
        resolve_ast(op.left_op), resolve_ast(op.right_op))
end
function resolve_ast(op::OperatorScale{D}) where {D}
    OperatorScale{D, typeof(op.scalar), typeof(resolve_ast(op.inner_op))}(op.scalar, resolve_ast(op.inner_op))
end

function resolve_ast(op::GridFunctionScale{D, VType}) where {D, VType}
    GridFunctionScale{D, VType, typeof(resolve_ast(op.inner_op))}(op.grid_function, resolve_ast(op.inner_op))
end

function resolve_ast(op::GridFunctionScale{D, <:Function}) where {D}
    vec = op.grid_function()
    return GridFunctionScale{D, typeof(vec), typeof(resolve_ast(op.inner_op))}(vec, resolve_ast(op.inner_op))
end

resolve_ast(ops::NTuple{N, Any}) where {N} = map(resolve_ast, ops)
# The catch-all every node above without its own method falls through to: TrialFunction,
# TestFunction, IndexedTrialFunction, IndexedTestFunction, SourceFunction, SourceVector,
# SourceConstant, IdentityOperator, ZeroOperator, and anything else with nothing to resolve.
# gpena/Bramble.jl#62: those nine used to have their own identity methods here, each
# decorative -- this catch-all made every one of them redundant, since a node not listed
# was never an error, only silently unresolved. Left as one line rather than nine.
resolve_ast(op::Any) = op

# ==============================================================================
# 6. Symbolic AST Traits
# ==============================================================================

# Note: is_symbolic base function is declared in linear_operators.jl

is_symbolic(::TrialFunction) = true
is_symbolic(::TestFunction) = true
is_symbolic(::IndexedTrialFunction) = true
is_symbolic(::IndexedTestFunction) = true
is_symbolic(::SourceFunction) = true
is_symbolic(::SourceVector) = true
is_symbolic(::SourceConstant) = true
is_symbolic(op::BilinearProduct) = true
is_symbolic(op::LinearProduct) = true

is_symbolic(op::BackwardDifference) = is_symbolic(op.inner_op)
is_symbolic(op::ForwardDifference) = is_symbolic(op.inner_op)

is_symbolic(op::BackwardAverage) = is_symbolic(op.inner_op)
is_symbolic(op::ForwardAverage) = is_symbolic(op.inner_op)
is_symbolic(op::ShiftNode) = is_symbolic(op.inner_op)

is_symbolic(op::RegionRestriction) = is_symbolic(op.inner_op)

is_symbolic(op::CenteredDifference) = is_symbolic(op.inner_op)
is_symbolic(op::StarDifference) = is_symbolic(op.inner_op)
is_symbolic(op::CrossWeightedDifference) = is_symbolic(op.inner_op)
is_symbolic(op::JumpNode) = is_symbolic(op.inner_op)

"""
    _is_source_only(op::LazyOp) -> Bool

Whether a `LazyOp` subtree is source-only: built entirely from sources
(`SourceFunction`/`SourceVector`) and the plain operators that wrap them, never bottoming
out in a `TrialFunction`/`IndexedTrialFunction` leaf.

`innerₕ`'s `l::Function`/`l::Number`/`l::VectorElement` overloads (`operators/inner.jl`)
never need this: those three types are never anything but a source, so wrapping them in a
`LinearProduct` is unconditional. The question only exists for an argument that already
arrived as a `LazyOp`: `πₕ(uₕ)` ([`interpolate_at`](@ref)) or `D₋ₓ(πₕ(uₕ))` are
sources too, just already wrapped, and the generic `innerₕ(::LazyOp, ::LazyOp)` used to build
a `BilinearProduct` regardless, which is the wrong AST shape for a `LinearForm`'s assembly
walk: a `BilinearProduct`'s stencil carries a pair of offsets (trial and test), where
`_scatter_term!` (`form/linear.jl`) expects one.

A missing case defaults to `false` (the fallback `::LazyOp` method below): conservative,
since that is exactly the behavior every node had before this predicate existed (always
`BilinearProduct`) for anything not explicitly listed as source-only.
"""
_is_source_only(::TrialFunction) = false
_is_source_only(::TestFunction) = false
_is_source_only(::IndexedTrialFunction) = false
_is_source_only(::IndexedTestFunction) = false
_is_source_only(::SourceFunction) = true
_is_source_only(::SourceVector) = true
_is_source_only(::SourceConstant) = true

_is_source_only(op::BackwardDifference) = _is_source_only(op.inner_op)
_is_source_only(op::ForwardDifference) = _is_source_only(op.inner_op)
_is_source_only(op::CenteredDifference) = _is_source_only(op.inner_op)
_is_source_only(op::StarDifference) = _is_source_only(op.inner_op)
_is_source_only(op::CrossWeightedDifference) = _is_source_only(op.inner_op)

_is_source_only(op::BackwardAverage) = _is_source_only(op.inner_op)
_is_source_only(op::ForwardAverage) = _is_source_only(op.inner_op)
_is_source_only(op::ShiftNode) = _is_source_only(op.inner_op)
_is_source_only(op::JumpNode) = _is_source_only(op.inner_op)
_is_source_only(op::RegionRestriction) = _is_source_only(op.inner_op)

_is_source_only(op::OperatorScale) = _is_source_only(op.inner_op)
_is_source_only(op::GridFunctionScale) = _is_source_only(op.inner_op)
function _is_source_only(op::OperatorAdd)
    _is_source_only(op.left_op) &&
        _is_source_only(op.right_op)
end

# A product (whichever kind) is its own thing, not a bare source to route again.
_is_source_only(::BilinearProduct) = false
_is_source_only(::LinearProduct) = false

_is_source_only(::LazyOp) = false

# The value of a source-only subtree at a grid point: `_contracted_left_stencil`
# (form/operators/inner.jl) reads it from the subtree's own `local_stencil`, correctly
# re-evaluated at every neighbour because a source is `PointDependentStencil`
# (form/operators/interpolation.jl).

# ==============================================================================
# 7. Walking an OperatorAdd tree: shared by every router in linear.jl/bilinear.jl
# ==============================================================================

# Six functions across `linear.jl`/`bilinear.jl` (`_check_block_meshes`,
# `_route_terms!`, `_route_terms_parallel!`, `_pattern_blocks!`, `_assemble_blocks!`,
# `_assemble_blocks_parallel!`) walk a form's `OperatorAdd` tree to send each summand where
# it belongs, all with the same shape: recurse left, recurse right, done. Recursing the tree
# rather than flattening it into a vector of terms first preserves concrete types: a
# flattened vector is `Vector{Any}` and makes every term a dynamic read, whereas recursing
# keeps each term concretely typed at its own call and costs nothing.
#
# They differ only in how many arguments sit before `op` and what (if anything) the caller
# reads back; nothing does: every call site above these six is a bare statement, the
# return value always discarded. The three shapes below return whatever the six already
# return unread today. Three separate names rather than one overloaded on argument count:
# with `rest...`/untyped leading arguments, overloads sharing a name are genuinely ambiguous
# to the compiler (a call whose second and third arguments both happen to be `OperatorAdd`
# matches two of the three signatures at once). Aqua's ambiguity check catches this even
# though no real call here ever hits it, and three names sidesteps the question rather than
# resolving it with a disambiguating method nothing calls. Each is still picked at its call
# site by hand; what moves out is only the recursion body, and
# `@code_warntype` still sees ordinary calls to `f`, specialized on `F = typeof(f)` like any
# other higher-order call in Julia.

# `op` first, nothing to mutate: `_check_block_meshes`.
@inline function _visit_operator_add1(f::F, op::OperatorAdd, rest...) where {F}
    f(op.left_op, rest...)
    f(op.right_op, rest...)
    return nothing
end

# `op` second, one mutated argument returned unchanged: `_route_terms!`,
# `_route_terms_parallel!`, `_assemble_blocks!`, `_assemble_blocks_parallel!`.
@inline function _visit_operator_add2(f::F, first_arg, op::OperatorAdd, rest...) where {F}
    f(first_arg, op.left_op, rest...)
    f(first_arg, op.right_op, rest...)
    return first_arg
end

# `op` third, two mutated arguments, nothing returned: `_pattern_blocks!`.
@inline function _visit_operator_add3(f::F, a1, a2, op::OperatorAdd, rest...) where {F}
    f(a1, a2, op.left_op, rest...)
    f(a1, a2, op.right_op, rest...)
    return nothing
end
