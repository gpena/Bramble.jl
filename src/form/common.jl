# common.jl

# ==============================================================================
# 1. Zero-Allocation Spacing & Tuple Utilities
# ==============================================================================

"""
    shift_offset(offset::NTuple{D, Int}, dim::Int, delta::Int) -> NTuple{D, Int}

Shifts a Cartesian offset tuple by `delta` in dimension `dim`.
"""
@inline shift_offset(offset::NTuple{D, Int}, dim::Int, delta::Int) where {D} = ntuple(
    i -> i == dim ?
         offset[i] + delta :
         offset[i], Val(D))

"""
    zero_offset(::Val{D}) -> NTuple{D, Int}

Zero-initialized offset tuple of dimension `D`.
"""
@inline zero_offset(::Val{D}) where {D} = ntuple(x -> 0, Val(D))

"""
    shift_stencil(inner::Tuple, ::Val{Dim}, delta)

Shifts all coordinates in a stencil tuple by `delta` in dimension `Dim`.

`map` over a `Tuple` unrolls and stays type-stable at compile time in Julia, measured
against a `@generated` version this once was (gpena/Bramble.jl#63): identical zero
allocations and identical inferred return type, so the code generation bought nothing
here.
"""
@inline shift_stencil(inner::Tuple, ::Val{Dim}, ::Val{Delta}) where {Dim, Delta} = map(
    t -> (
        shift_offset(t[1], Dim, Delta), t[2]), inner)

@inline shift_stencil(inner::Tuple, ::Val{Dim}, delta::Int) where {Dim} = map(
    t -> (
        shift_offset(t[1], Dim, delta), t[2]), inner)

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

@inline scale_stencil(inner::Tuple, scalar::Number) = map(t -> (Base.front(t)..., t[end] * scalar), inner)

"""
    entry_offsets(stencil::Tuple)
    entry_weights(stencil::Tuple)

A stencil's offsets and its coefficients, as two separate containers: `entry_offsets` keeps
each entry's offsets alone (`(off_u, off_v)` for a bilinear entry, `(off_v,)` for a linear
one) and `entry_weights` keeps the coefficients.

Taken apart for `Enzyme`, which cannot type a stencil entry's mixed `Int`/`Float64` tuple
in a function that reads *both* halves of it and is not inlined into the function being
differentiated (gpena/Bramble.jl#249). All three conditions are needed, measured one at a
time: the same loop written inside the differentiated closure compiles, so does one that
reads only the offsets, and so does one that reads only the weights -- what fails is the
combination, with `EnzymeNoTypeError` inside `_visit_guarded_region!`. Reading the offsets
from a container of `Int`s and the weights from a container of `Float64`s removes it, for a
coefficient scaling the form and for a `VectorElement` coefficient alike, in 1D, 2D and 3D.

The stencil itself is unchanged: `local_stencil` returns what it always did, and the split
happens where the entries are consumed (`_visit_entries`). That is enough, and the
narrower change: every `local_stencil` method, the stencil algebra above and the tests that
compare stencils against literal tuples all stay as they are. Enzyme differentiates
`local_stencil` and `scale_stencil` themselves correctly at any stencil size measured, up
to 63 machine words -- the size threshold gpena/Bramble.jl#249 was filed against is really
`_peelable` selecting the guarded walk, not an aggregate Enzyme cannot type.

`map` over a `Tuple` unrolls and stays type-stable, the same property the rest of the
stencil algebra in this file relies on, so neither call allocates.
"""
@inline entry_offsets(stencil::Tuple) = map(Base.front, stencil)

@inline entry_weights(stencil::Tuple) = map(last, stencil)

@doc (@doc entry_offsets) entry_weights

"""
    sum_stencil_values(stencil::Tuple)

The sum of a stencil's coefficients, ignoring its offsets entirely.

Required by `_contracted_left_stencil` (`form/operators/inner.jl`) for a source-only
subtree's own `local_stencil`: not the offsets, which mean nothing for a value that
contributes no matrix structure, only their total. `false` rather than `0` or `zero(T)` is
the empty-stencil answer: [`RegionRestriction`](@ref) can legitimately produce `()` for a
point outside its region, and there is no `T` to call `zero` on when there are no entries to
read one from; `false` promotes to whatever numeric type the other entries (or, empty, the
caller's own multiplication) turn out to have: exactly `sum(f, itr; init = false)`'s own
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
    TrialFunction{D, N} <: LazyOp{D}

An AST node representing the symbolic trial function \$u\$ in a bilinear form over a
`D`-dimensional space with `N` components.
"""
struct TrialFunction{D, N} <: LazyOp{D} end

"""
    TestFunction{D, N} <: LazyOp{D}

An AST node representing the symbolic test function \$v\$ in a form over a
`D`-dimensional space with `N` components.
"""
struct TestFunction{D, N} <: LazyOp{D} end

@inline TrialFunction{D}() where {D} = TrialFunction{D, nothing}()
@inline TestFunction{D}() where {D} = TestFunction{D, nothing}()

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

"""
    AbsoluteRow

A stencil entry's test slot, naming a row of the test space directly rather than an offset
from the point being evaluated.

The mirror of [`AbsoluteColumn`](@ref), and it exists for the mirror reason: a test-side
interpolation (`innerₕ(u, πₕ(w))`, gpena/Bramble.jl#263) determines which rows a point's
contribution scatters into, on a mesh other than the one being walked, so it names them
outright. The walked mesh is then the trial function's -- whichever side stays native is the
side that supplies the quadrature weight and the grid being swept.
"""
struct AbsoluteRow
    row::Int
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
@inline function _clamped_shift(
        m, I::CartesianIndex{D}, ::Val{Dim}, delta::Int
) where {D, Dim}
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
@inline _in_grid(space, I::CartesianIndex{D}) where {D} = checkbounds(Bool, LinearIndices(indices(mesh(space))), I)

"""
    shifted_inner_stencil(inner_op, inner, space, I, markers, ::Val{Dim}, delta)

The stencil `inner_op` contributes `delta` points away in direction `Dim`, given `inner`, its
stencil already evaluated at `I`.

The one place the "shift by relabelling" assumption is made, so the one place a node that
cannot be relabelled has to be handled. The default dispatches on
`stencil_shift_trait`: [`TranslationInvariantStencil`](@ref) relabels `inner`'s
offsets and never touches `inner_op` again; [`PointDependentStencil`](@ref) discards `inner`
and evaluates `inner_op` at the shifted point instead.

Two node types override this default outright rather than answering through the trait alone,
because the trait's two stock branches cannot express what they need: re-evaluating the
whole subtree at the shifted point is exact for a source, which carries only a value, but
wrong for anything that also carries a trial or test column, since that column has to move
by relabelling while whatever multiplies it has to be read at the new point:

  - `GridFunctionScale` shifts its operand by the operand's own rule (a recursive call to
    this same function), then reads its coefficient at the shifted point and scales by it,
    so `D₋ₓ(cₕ * u)` moves the trial column *and* re-reads `cₕ` at the tap
    (gpena/Bramble.jl#271).
  - `OperatorAdd` recurses into each summand and `concatenate_stencils` the two results,
    since a sum can mix a point-dependent summand (one carrying a `GridFunctionScale`) with
    a translation-invariant one, and the generic point-dependent branch would re-evaluate
    the translation-invariant summand at the shifted point too, discarding its trial column.

Every branch -- the two default ones and these two overrides -- produces a tuple of the same
static length, since it is the same operator either way, so the callers' `concatenate_stencils`
sees exactly the shape it always did.
"""
@inline function shifted_inner_stencil(
        inner_op, inner, space, I::CartesianIndex{D}, markers, ::Val{Dim}, delta
) where {D, Dim}
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
    delta
) where {D, Dim} = shift_stencil(inner, Val(Dim), delta)

@inline function _shifted_inner_stencil(
        ::PointDependentStencil,
        inner_op,
        inner,
        space,
        I::CartesianIndex{D},
        markers,
        ::Val{Dim},
        delta
) where {D, Dim}
    m = mesh(space)
    Ishift = _clamped_shift(m, I, Val(Dim), _shift_delta(delta))
    return local_stencil(
        inner_op, space, Ishift, markers, LinearIndices(indices(m))[Ishift]
    )
end

"""
    _grid_function_value(grid_function, lin_idx::Int)

The value a [`GridFunctionScale`](@ref) coefficient contributes at linear index `lin_idx`.

`grid_function` is whatever `GridFunctionScale.grid_function` may hold: a zero-argument
`Function` thunk deferred until `resolve_ast` calls it (see [`SourceVector`](@ref)'s
docstring for why a thunk, not a plain vector, is the wire format), a bare `Number` for a
uniform scale, or an array of per-point values. Shared by `local_stencil(::GridFunctionScale,
...)` (`form/stencil_eval.jl`), which reads the coefficient at the point being visited, and
the `shifted_inner_stencil` override below, which reads it at the point a tap shifts to --
previously the same three-way branch, written out twice.
"""
@inline function _grid_function_value(grid_function, lin_idx::Int)
    if grid_function isa Function
        val = grid_function()
        return val isa Number ? val : val[lin_idx]
    else
        return grid_function isa Number ? grid_function : grid_function[lin_idx]
    end
end

# `GridFunctionScale` is marked `PointDependentStencil` in form/operators/interpolation.jl,
# but the generic `PointDependentStencil` branch above is wrong for it: discarding `inner`
# and re-evaluating the whole node at `Ishift` would lose the trial or test column the
# operand contributes (`GridFunctionScale(c, u)` evaluated fresh at `Ishift` reduces to `c`
# read there times `u`'s own stencil *at* `Ishift` -- offset zero, not `delta`). What is
# needed instead is the operand shifted by its own rule, with the coefficient read
# separately at the shifted point, which is what this override gives it ahead of the
# generic trait dispatch (gpena/Bramble.jl#271). `inner` is discarded, the same as the
# generic point-dependent branch discards it: it was evaluated at `I`, and the coefficient
# this tap needs is the one at `Ishift`.
@inline function shifted_inner_stencil(
        inner_op::GridFunctionScale, inner, space, I::CartesianIndex{D}, markers, ::Val{Dim}, delta
) where {D, Dim}
    m = mesh(space)
    lins = LinearIndices(indices(m))
    Ishift = _clamped_shift(m, I, Val(Dim), _shift_delta(delta))
    sub = local_stencil(inner_op.inner_op, space, I, markers, lins[I])
    shifted = shifted_inner_stencil(inner_op.inner_op, sub, space, I, markers, Val(Dim), delta)
    return scale_stencil(shifted, _grid_function_value(inner_op.grid_function, lins[Ishift]))
end

# A sum inherits `PointDependentStencil` (`_combine_shift_traits`,
# form/operators/interpolation.jl) the moment either side does, which now includes any side
# holding a `GridFunctionScale`. Without this override the generic branch above would
# re-evaluate *both* summands at `Ishift`, including a translation-invariant one, losing its
# trial column exactly as the unwrapped `GridFunctionScale` case above would; recursing into
# each summand instead lets the translation-invariant side keep relabelling while the
# point-dependent side re-reads its coefficient. The tuple length is unchanged from the
# generic branch: that one already returned `local_stencil(::OperatorAdd, Ishift)`, itself a
# concatenation of the same two halves.
@inline function shifted_inner_stencil(
        inner_op::OperatorAdd, inner, space, I::CartesianIndex{D}, markers, ::Val{Dim}, delta
) where {D, Dim}
    lin_idx = LinearIndices(indices(mesh(space)))[I]
    left = shifted_inner_stencil(
        inner_op.left_op,
        local_stencil(inner_op.left_op, space, I, markers, lin_idx),
        space, I, markers, Val(Dim), delta
    )
    right = shifted_inner_stencil(
        inner_op.right_op,
        local_stencil(inner_op.right_op, space, I, markers, lin_idx),
        space, I, markers, Val(Dim), delta
    )
    return concatenate_stencils(left, right)
end

# ==============================================================================
# 3. Form API & Bramble Standard Mapping
# ==============================================================================

# Indexing a node by component (`v(1)`, and distribution through whatever is built on
# top of it) lives in form/component.jl, included after the operator files because it needs
# every node type in its signatures.

"""
    trial_function(::Val{D}) -> TrialFunction{D, nothing}
    trial_function(space::AbstractSpaceType) -> TrialFunction{dim(space), leaf_count(space)}

Constructs a `TrialFunction` of dimension `D` or for a specific grid `space`.
"""
@inline trial_function(::Val{D}) where {D} = TrialFunction{D, nothing}()
@inline trial_function(space::AbstractSpaceType) = TrialFunction{dim(space), leaf_count(space)}()

"""
    test_function(::Val{D}) -> TestFunction{D, nothing}
    test_function(space::AbstractSpaceType) -> TestFunction{dim(space), leaf_count(space)}

Constructs a `TestFunction` of dimension `D` or for a specific grid `space`.
"""
@inline test_function(::Val{D}) where {D} = TestFunction{D, nothing}()
@inline test_function(space::AbstractSpaceType) = TestFunction{dim(space), leaf_count(space)}()

"""
    source_function(f, ::Val{D}) -> SourceFunction{D, typeof(f)}

Constructs a `SourceFunction` wrapping function `f`.
"""
source_function(f, ::Val{D}) where {D} = SourceFunction{D, typeof(f)}(f)

# --- Eager spatial lowering of source functions (gpena/Bramble.jl#197) ------------- #
#
# `local_stencil(op::SourceFunction, space, I, markers, lin_idx)` calls `op.func(point(mesh(
# space), I))` fresh at every grid point of every assembly -- which is what lets a closure
# capturing a `Ref` or other mutable state stay live across repeated `assemble!` calls, but
# also means Julia treats each syntactically distinct closure as its own type, forcing full
# recompilation of the whole assembly pipeline the first time any new closure is used as a
# source (measured ~11ms per distinct closure).
#
# `_lower_sources(op, space) -> LazyOp` rewrites every `SourceFunction` reachable from `op`
# into a `SourceVector` sampled once, now, against `space` (`Rₕ(space, func)`, `parent(...)`
# of the result) -- so a form built from a brand-new closure pays that cost once, at
# construction, and every later `assemble!`/`assemble` call is a flat array read like any
# other `SourceVector`, regardless of how novel the closure's type is.
#
# This is a real behaviour change, not merely an optimisation: a `SourceFunction` closure
# that captures mutable state is no longer re-evaluated on later assemblies -- it is
# sampled once, here. `VectorElement` and `Ref` coefficients are unaffected (neither is ever
# wrapped in a `SourceFunction`; see `GridFunctionScale`/`OperatorScale` below, both left as
# pass-through, matching `simplify_ast`'s own shape in `simplifier.jl`), and no existing test
# relied on a raw closure being re-invoked outside that documented pattern. A source meant to
# vary after construction (a time-dependent PDE source, say) must use the already-documented
# `update_coefficients!`/`VectorElement` pattern (`semidiscretize`'s docstring) instead of a
# raw closure captured directly in the form.
#
# Every node type not named here is a leaf as far as this pass is concerned (it can never
# contain a `SourceFunction`, or lowering inside it is handled by its own caller -- see
# `_lower_sources_for_space` in `form/linear.jl` for `LinearProduct`/`OperatorAdd` routing
# across a `CompositeGridSpace`'s leaves) and is returned unchanged.
@inline _lower_sources(op::LazyOp, space) = op

@inline function _lower_sources(op::SourceFunction{D}, space) where {D}
    vec = parent(Rₕ(space, op.func))
    return SourceVector{D, typeof(vec)}(vec)
end

function _lower_sources(op::OperatorScale, space)
    inner = _lower_sources(op.inner_op, space)
    return inner === op.inner_op ? op : OperatorScale(op.scalar, inner)
end

function _lower_sources(op::GridFunctionScale, space)
    inner = _lower_sources(op.inner_op, space)
    return inner === op.inner_op ? op : GridFunctionScale(op.grid_function, inner)
end

function _lower_sources(op::OperatorAdd, space)
    left = _lower_sources(op.left_op, space)
    right = _lower_sources(op.right_op, space)
    return left === op.left_op && right === op.right_op ? op : OperatorAdd(left, right)
end

# `ShiftNode` (form/operators/average.jl) and `LinearProduct` (form/operators/inner.jl) are
# both included after this file, so their `_lower_sources` methods -- naming the type in the
# signature, unlike everything above -- live in form/linear.jl instead (bramble-performance
# skill, "include-order rule").

# ==============================================================================
# 4. Deprecated `ast` keyword (gpena/Bramble.jl#105)
# ==============================================================================

# Measured to buy nothing: `resolve_form_ast(form)` is a field read, not a resolution, so
# there is no per-call cost left to hoist by passing it back in. Its one real capability --
# assembling a different form's AST into a matrix built for another form's pattern -- is
# redundant with assembling that other form directly. Shared by `assemble!`/`assemble`/
# `evaluate!` (linear.jl, bilinear.jl) and `assemble_parallel!`'s positional `ast` argument,
# so the message and the removal version stay in one place.
@noinline function _warn_ast_keyword(funcsym::Symbol)
    Base.depwarn(
        "the `ast` keyword measures no benefit over the form's own resolved AST, and its " *
        "only real use -- swapping in another form's AST -- is redundant with assembling " *
        "that form directly. It will be removed in v3.0.0 without replacement.",
        funcsym
    )
    return nothing
end
