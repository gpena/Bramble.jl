# matrix_free.jl: `MatrixFreeOperator`, a `BilinearForm` applied to a vector without its
# matrix. `mul!` walks the form's (term, block) units exactly as the
# serial replay does (`_replay_summands!`/`_replay_blocks!`, bilinear_execution.jl), through
# the same `visit_bilinear_stencil`, and hands each entry to an `ActionSink` that adds
# `α * weight * x[col]` into `y[row]` instead of into a stored `nzval`. Every AST node the
# assembly supports is therefore supported here, with no second stencil evaluator to keep in
# step with the first.

"""
    ActionSink(y::AbstractVector, x::AbstractVector, α, mask)

The matrix-free sink: an entry `(row, col, weight)` of the form's stencil adds
`α * weight * x[col]` to `y[row]`, so one walk of [`visit_bilinear_stencil`](@ref) adds
`α * A * x` to `y` without `A` ever being stored.

`mask` is `nothing` for a form with no Dirichlet rows, or a `BitVector` over the rows whose
set entries are skipped: those rows are identity rows, written after the walk (the
[`dirichlet_bc!`](@ref) contract).
"""
struct ActionSink{Y <: AbstractVector, X <: AbstractVector, S, M}
    y::Y
    x::X
    α::S
    mask::M
end

# Whether row `row` takes stencil entries: always without a mask, off Γ_D with one.
@inline _mf_live(::Nothing, ::Int) = true
@inline _mf_live(mask::BitVector, row::Int) = !@inbounds(mask[row])

# `row` and `col` land inside the matrix (the walk's guard), and `mul!` checked `y` and `x`
# against the matrix's size, so both reads are in bounds.
@inline function _sink_entry!(s::ActionSink, row::Int, col::Int, weight, ::Int)
    _mf_live(s.mask, row) && @inbounds(s.y[row] += s.α * weight * s.x[col])
    return nothing
end

# A transposed pair ⟨Au, Bv⟩ + ⟨Bu, Av⟩ walked once, as `_PairReplaySink` does: an entry
# `(row, col)` of the first term also stands for the second term's entry `(col + dr, row + dc)`.
# `half` selects both (`0`), the first term's only (`1`) or the transposed only (`2`).
struct _PairActionSink{Y <: AbstractVector, X <: AbstractVector, S1, S2, M}
    y::Y
    x::X
    α1::S1
    α2::S2
    mask::M
    dr::Int
    dc::Int
    half::Int
end

@inline function _sink_entry!(s::_PairActionSink, row::Int, col::Int, weight, ::Int)
    if s.half != 2 && _mf_live(s.mask, row)
        @inbounds s.y[row] += s.α1 * weight * s.x[col]
    end
    if s.half != 1
        r2 = col + s.dr
        _mf_live(s.mask, r2) && @inbounds(s.y[r2] += s.α2 * weight * s.x[row + s.dc])
    end
    return nothing
end

@inline _pair_action_sink(s::ActionSink, α1, α2, dr::Int, dc::Int, half::Int) = _PairActionSink(
    s.y, s.x, α1, α2, s.mask, dr, dc, half)

"""
    ContractionSink(v::AbstractVector, u::AbstractVector, acc::Base.RefValue)

The sink of the form call `a(u, v)`: an entry `(row, col, weight)` of the form's stencil adds
`conj(v[row]) * weight * u[col]` to `acc[]`, so one walk of [`visit_bilinear_stencil`](@ref)
sums `vᴴ A u` without `A` or `A * u` ever being stored. `acc` is the call's own cell, so
concurrent calls on one form do not share it.
"""
struct ContractionSink{V <: AbstractVector, U <: AbstractVector, T}
    v::V
    u::U
    acc::Base.RefValue{T}
end

# The form call checked `v` and `u` against the matrix's size, so both reads are in bounds.
@inline function _sink_entry!(s::ContractionSink, row::Int, col::Int, weight, ::Int)
    @inbounds s.acc[] += conj(s.v[row]) * weight * s.u[col]
    return nothing
end

# `vᴴ A u` for `A = assemble(a)`, through the serial unit walk below, in the element type
# promoted from `A`'s, `u`'s and `v`'s (`(a::BilinearForm)(u, v)`, bilinear.jl).
function _contract(a::BilinearForm, ud::AbstractVector, vd::AbstractVector)
    (length(ud) == ndofs(trial_space(a)) && length(vd) == ndofs(test_space(a))) ||
        _throw_contraction_dimmismatch(a, ud, vd)
    Base.require_one_based_indexing(ud, vd)
    T = promote_type(_matrix_eltype(a, a.ast), eltype(ud), eltype(vd))
    s = ContractionSink(vd, ud, Ref(zero(T)))
    _mf_apply!(CpuSerial(), s, a)
    return s.acc[]
end

@noinline function _throw_contraction_dimmismatch(a::BilinearForm, ud, vd)
    throw(
        DimensionMismatch(
        "a bilinear form of size $(ndofs(test_space(a)))×$(ndofs(trial_space(a))) " *
        "cannot contract a trial vector of length $(length(ud)) with a test vector of " *
        "length $(length(vd))",
    ),
    )
end

"""
    MatrixFreeOperator{T, Form, DLabels, EP, Plan} <: AbstractMatrix{T}

A [`BilinearForm`](@ref) as a linear operator: `mul!(y, op, x)` computes `A * x`, and
`mul!(y, op, x, α, β)` computes `α * A * x + β * y`, for `A = assemble(a; dirichlet)`,
without ever storing `A`. Each product walks the form's stencil once, evaluated afresh, so a
coefficient updated in place (`Rₕ!`, a `Ref`) is seen by the next product. Build one with
[`matrix_free_operator`](@ref).

`x` and `y` are plain vectors or [`VectorElement`](@ref)s: `op * x` returns a new vector,
`op * uₕ` a new element of the form's test space. `mul!` allocates nothing on a serial
policy, except on a form with a region restriction (`restrict_to`), whose stencil
evaluation allocates as it does in `assemble!`.

On a [`CpuThreaded`](@ref) or [`CpuPolyester`](@ref) policy the grid is cut into one band
per thread along its last axis, and each thread walks every term over its band and adds only
into the rows of its band's points, so no two threads add into the same entry of `y` and the
result does not depend on scheduling. A product is one parallel region however many terms
the form has; the terms' reach, which widens each band by the points writing into it, is
computed when the operator is built. Each row receives its entries in the serial order, so
the threaded product is bitwise the serial one (up to rounding when a term runs serially,
see below). The threads are those of the leaves' own mechanism: `Polyester.@batch` on a
`CpuPolyester` leaf, one `Threads.@spawn`ed task per band otherwise. A
term with a test-side interpolation runs serially. A form whose leaves walk grids of
different sizes, or carry different policies, is swept one term at a time instead, each in
the colour bands the threaded [`assemble!`](@ref) uses. A threaded product allocates its task
launches, a fixed cost that does not grow with the grid.

Dirichlet rows follow [`dirichlet_bc!`](@ref): a row in Γ_D is the identity row, so
`(A * x)[i] = x[i]` there and its columns are untouched. A row in Γ_D past the last column
(a rectangular form) is zero, as in the assembled matrix.

Subtypes `AbstractMatrix{T}` so it can stand in for the assembled matrix in a Krylov solve
(`LinearProblem` with `KrylovJL_CG`, say). `getindex` is supported for inspection and costs
one product per entry.

# Type parameters
- `T`: The element type of `assemble(a)`.
- `Form`: The form's type.
- `DLabels`: `Nothing`, or the `Tuple` of Dirichlet labels.
- `EP`: The [`ExecutionPolicy`](@ref) the operator was built with.
- `Plan`: `Nothing`, or the type of the threaded sweep's plan (the grid its bands cut and
  the terms' reach), fixed when the operator is built.

See also: [`matrix_free_operator`](@ref), [`assemble`](@ref), [`KroneckerLinearOperator`](@ref).
"""
struct MatrixFreeOperator{T, Form <: BilinearForm, DLabels, EP <: ExecutionPolicy, Plan} <:
       AbstractMatrix{T}
    form::Form
    labels::DLabels
    mask::BitVector
    policy::EP
    plan::Plan
    nrows::Int
    ncols::Int
end

@noinline function _throw_matrix_free_gpu(policy)
    throw(
        ArgumentError(
        "matrix_free_operator does not run on a GpuPolicy ($(typeof(policy))): a device " *
        "matrix-free mul! is tracked on milestone v4.4.0. Use a CPU policy, or assemble " *
        "the matrix with `assemble`.",
    ),
    )
end

"""
    matrix_free_operator(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing, policy = execution_policy(trial_space(a))) -> MatrixFreeOperator

The linear operator of `a`, applied without assembling its matrix: `op * x` agrees with
`assemble(a; dirichlet, dirichlet_components) * x`.

# Arguments
- `a`: The bilinear form. Any form [`assemble`](@ref) accepts, on scalar or composite spaces.

# Keywords
- `dirichlet`: The Dirichlet labels whose rows become identity rows, in any form `assemble`
  accepts (a symbol, a tuple of labels, a pair with its value, [`dirichlet_constraints`](@ref));
  boundary values are ignored, as they are by the matrix (default: `nothing`).

- `dirichlet_components`: The leaves of a composite test space the labels bind to, as in
  [`dirichlet_bc!`](@ref). It is an `Int`, a `Tuple` of `Int`s, or `nothing` for every leaf
  (default: `nothing`).

- `policy`: The [`ExecutionPolicy`](@ref) of `mul!` (default: the trial space's).
  [`CpuSerial`](@ref) walks the form on the calling task; any other CPU policy threads it,
  with each leaf's own mechanism (see [`MatrixFreeOperator`](@ref)).

# Returns
- [`MatrixFreeOperator`](@ref): of size `(ndofs(test_space(a)), ndofs(trial_space(a)))` and
  the element type of `assemble(a)`.

# Throws
- `ArgumentError`: `policy` is a [`GpuPolicy`](@ref); device execution is tracked on
  milestone v4.4.0.
- `ArgumentError`: `dirichlet_components` names a leaf the test space does not have.

# Examples
```jldoctest
using Bramble, LinearAlgebra
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11, false))
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
op = matrix_free_operator(a; dirichlet = :boundary)
x = rand(ndofs(Wₕ))
y = similar(x)
mul!(y, op, x)
y ≈ assemble(a; dirichlet = :boundary) * x

# output
true
```

See also: [`MatrixFreeOperator`](@ref), [`assemble`](@ref), [`kronecker_operator`](@ref).
"""
function matrix_free_operator(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
        policy = execution_policy(trial_space(a))
)
    policy isa GpuPolicy && _throw_matrix_free_gpu(policy)
    labels, _ = _normalize_dirichlet(dirichlet)
    Wv = test_space(a)
    nrows = ndofs(Wv)
    mask = falses(labels === nothing ? 0 : nrows)
    if labels !== nothing && !isempty(labels)
        _mf_fill_mask!(mask, Wv, labels, dirichlet_components)
    end
    T = _matrix_eltype(a, a.ast)
    plan = _mf_plan(policy, a)
    return MatrixFreeOperator{T, typeof(a), typeof(labels), typeof(policy), typeof(plan)}(
        a, labels, mask, policy, plan, nrows, ndofs(trial_space(a))
    )
end

# The rows `dirichlet_bc!(A, Wv, labels...; components)` would constrain, validated the same way.
function _mf_fill_mask!(mask::BitVector, Wv, labels, components)
    leaves = leaf_spaces_offsets(Wv)
    if Wv isa ScalarGridSpace
        _validate_scalar_components(components)
    else
        _validate_dirichlet_components(components, length(leaves))
    end
    for (i, (sp, offset)) in enumerate(leaves)
        _leaf_selected(components, i) || continue
        mask[(offset + 1):(offset + ndofs(sp))] .= _combined_mask(mesh(sp), labels)
    end
    return mask
end

Base.size(op::MatrixFreeOperator) = (op.nrows, op.ncols)

@inline _mf_mask(::MatrixFreeOperator{<:Any, <:BilinearForm, Nothing}) = nothing
@inline _mf_mask(op::MatrixFreeOperator) = op.mask

# The raw storage of a grid function, a plain vector as it is: the walk indexes it directly.
@inline _mf_data(x::VectorElement) = parent(x)
@inline _mf_data(x::AbstractVector) = x

@noinline function _throw_matrix_free_dimmismatch(op::MatrixFreeOperator, x, y)
    throw(
        DimensionMismatch(
        "MatrixFreeOperator of size $(size(op)) cannot multiply a vector of length " *
        "$(length(x)) into one of length $(length(y))",
    ),
    )
end

# Only the five-argument method: `LinearAlgebra`'s three-argument `mul!` and `*` both reach
# it (with `α = true, β = false`), and a three-argument method here would be ambiguous
# against `ReverseDiff`'s `mul!(::TrackedArray, ::AbstractMatrix, ::TrackedArray)`, as
# `KroneckerLinearOperator`'s is. `β == 0` overwrites `y`, so a `NaN` in it does not survive.
function mul!(
        y::AbstractVector, op::MatrixFreeOperator, x::AbstractVector, α::Number, β::Number
)
    (length(x) == op.ncols && length(y) == op.nrows) ||
        _throw_matrix_free_dimmismatch(op, x, y)
    yd = _mf_data(y)
    xd = _mf_data(x)
    Base.require_one_based_indexing(yd, xd)
    if iszero(β)
        fill!(yd, zero(eltype(yd)))
    elseif !isone(β)
        yd .*= β
    end
    mask = _mf_mask(op)
    _mf_product!(op.plan, op.policy, ActionSink(yd, xd, α, mask), op.form)
    _mf_identity_rows!(yd, xd, α, mask)
    return y
end

# A Dirichlet row of `α * A * x + β * y` is `α * x[i] + β * y[i]`; `β * y[i]` is already in
# place, since the walk skipped the row. A row past the last column (a rectangular form, say
# `πₕ` onto a finer test mesh) has no diagonal, and `dirichlet_bc!` leaves it all zero, so
# only `β * y[i]` remains there.
@inline _mf_identity_rows!(_, _, _, ::Nothing) = nothing
@inline function _mf_identity_rows!(yd, xd, α, mask::BitVector)
    n = length(xd)
    _each_marked(mask, 0) do i
        i <= n && @inbounds(yd[i] += α * xd[i])
    end
    return nothing
end

# `op` applied to a grid function: a new element of the form's test space, in the promoted
# element type of `op` and `uₕ`. A plain vector takes `LinearAlgebra`'s `*`, which allocates
# the result and calls the `mul!` above.
function Base.:*(op::MatrixFreeOperator{T}, uₕ::VectorElement) where {T}
    xd = parent(uₕ)
    y = similar(xd, promote_type(T, eltype(xd)), op.nrows)
    mul!(y, op, xd, true, false)
    return VectorElement(y, test_space(op.form))
end

"""
    getindex(op::MatrixFreeOperator, i::Int, j::Int) -> Number

The `(i, j)` entry of the matrix `op` stands for, read off one product with the `j`-th unit
vector. For inspection: each call walks the whole form.
"""
function Base.getindex(op::MatrixFreeOperator{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(op, i, j)
    x = zeros(T, op.ncols)
    x[j] = one(T)
    y = zeros(T, op.nrows)
    mul!(y, op, x, true, false)
    return y[i]
end

# A one-line summary in both forms: the `AbstractMatrix` printers read every shown entry
# through `getindex`, one product each.
function Base.show(io::IO, op::MatrixFreeOperator{T}) where {T}
    print(io, op.nrows, "×", op.ncols, " MatrixFreeOperator{", T, "}")
    op.labels === nothing || print(io, " with Dirichlet rows on ", op.labels)
    return nothing
end
Base.show(io::IO, ::MIME"text/plain", op::MatrixFreeOperator) = show(io, op)

# --- The (term, block) units, walked as the serial replay walks them ----------------- #
#
# The same unit walk as `_replay_bilinear_core!` (bilinear_execution.jl), sink for segment:
# a scalar form one summand at a time, a composite one block by block, a transposed pair in
# one walk. Every method `@noinline` for the reason given there: each term's walk stays its
# own method instance instead of inlining into one grown with the term count. `policy` is the
# operator's, and decides only how each unit is walked (`_mf_visit!`, `_mf_visit_pair!`).
#
# The walk is generic over the sink: any sink with a `_sink_entry!` method (`ActionSink`,
# `ContractionSink`, `DiagonalSink`) walks the same units. Only an `ActionSink` fuses a
# transposed pair into one walk (`_PairActionSink`); any other sink walks the pair's two terms
# one after the other, which gives the same entries. A sink other than `ActionSink` walks
# under `CpuSerial` only: the threaded sweep (`_sweep_point!` and the Polyester hooks below)
# has methods for an action target (`_ActionTarget`) alone.

function _mf_apply!(policy, s, a::BilinearForm)
    Wu, Wv, ast = trial_space(a), test_space(a), a.ast
    if _is_block_pair(Wu, Wv)
        _mf_blocks!(policy, s, ast, leaf_spaces_offsets(Wu), leaf_spaces_offsets(Wv))
        return nothing
    end
    bound = _bind_interp_spaces(ast, Wu, Wv)
    _check_block_meshes(bound, Wu, Wv)
    sp = host_weights(_walked_leaf(bound, Wu, Wv))
    _mf_summands!(policy, s, bound, sp)
    return nothing
end

@noinline function _mf_summands!(policy, s, op::OperatorAdd, sp)
    _foldl_pairs(
        (_, t) -> _mf_summands!(policy, s, t, sp),
        (_, t1, t2) -> _mf_pair!(policy, s, t1, t2, sp),
        nothing,
        _summands(op)
    )
    return nothing
end

@noinline function _mf_summands!(policy, s, term::TERM, sp) where {TERM}
    _mf_visit!(policy, s, term, sp, 0, 0)
    return nothing
end

@noinline function _mf_pair!(policy, s::ActionSink, t1, t2, sp)
    sink = _pair_action_sink(s, s.α * _term_scale(t1), s.α * _term_scale(t2), 0, 0, 0)
    _mf_visit_pair!(policy, sink, _bare_product(t1), _bare_product(t2), sp, 0, 0)
    return nothing
end

@noinline function _mf_pair!(policy, s, t1, t2, sp)
    _mf_summands!(policy, s, t1, sp)
    _mf_summands!(policy, s, t2, sp)
    return nothing
end

@noinline function _mf_blocks!(
        policy, s, op::OperatorAdd, trial_leaves, test_leaves
)
    _foldl_pairs(
        (_, t) -> _mf_blocks!(policy, s, t, trial_leaves, test_leaves),
        (_, t1, t2) -> _mf_pair_blocks!(policy, s, t1, t2, trial_leaves, test_leaves),
        nothing,
        _summands(op)
    )
    return nothing
end

@noinline function _mf_blocks!(
        policy, s, term::TERM, trial_leaves, test_leaves
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        bound = _bind_interp_spaces(term, blk.trial_leaf, blk.test_leaf)
        _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
        sp = host_weights(_walked_leaf(bound, blk.trial_leaf, blk.test_leaf))
        _mf_visit!(policy, s, bound, sp, blk.row_offset, blk.col_offset)
    end
    return nothing
end

# `_replay_pair_blocks!`'s walk: block `k` of the second term holds the transposes of block
# `k` of the first, walked on one leaf when both blocks share it and once per leaf, one half
# each, when they do not.
@noinline function _mf_pair_blocks!(policy, s::ActionSink, t1, t2, trial_leaves, test_leaves)
    p1 = _bare_product(t1)
    p2 = _bare_product(t2)
    b1 = blocks(p1, trial_leaves, test_leaves)
    b2 = blocks(p2, trial_leaves, test_leaves)
    if _pair_blocks_ok(b1, b2)
        α1 = s.α * _term_scale(t1)
        α2 = s.α * _term_scale(t2)
        for (blk, blk2) in map(tuple, b1, b2)
            bound = _bind_interp_spaces(p1, blk.trial_leaf, blk.test_leaf)
            _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
            _check_block_meshes(p2, blk2.trial_leaf, blk2.test_leaf)
            sp = _walked_leaf(bound, blk.trial_leaf, blk.test_leaf)
            dr, dc = _pair_shift(blk, blk2)
            ro, co = blk.row_offset, blk.col_offset
            if sp === blk2.test_leaf
                _mf_visit_pair!(
                    policy, _pair_action_sink(s, α1, α2, dr, dc, 0), bound, p2,
                    host_weights(sp), ro, co
                )
            else
                _mf_visit_pair!(
                    policy, _pair_action_sink(s, α1, α2, dr, dc, 1), bound, p2,
                    host_weights(sp), ro, co
                )
                _mf_visit_pair!(
                    policy, _pair_action_sink(s, α1, α2, dr, dc, 2), bound, p2,
                    host_weights(blk2.test_leaf), ro, co
                )
            end
        end
        return nothing
    end
    Base.inferencebarrier(_mf_blocks!)(policy, s, t1, trial_leaves, test_leaves)
    Base.inferencebarrier(_mf_blocks!)(policy, s, t2, trial_leaves, test_leaves)
    return nothing
end

@noinline function _mf_pair_blocks!(policy, s, t1, t2, trial_leaves, test_leaves)
    _mf_blocks!(policy, s, t1, trial_leaves, test_leaves)
    _mf_blocks!(policy, s, t2, trial_leaves, test_leaves)
    return nothing
end

# --- One unit, serial or threaded --------------------------------------------------- #
#
# Serially, one unit is one `visit_bilinear_stencil` walk, with its unguarded interior. Under
# a threaded policy whose form the fused sweep below cannot take (`_mf_plan` answered
# `nothing`), it is the colour-banded sweep the threaded assembly runs
# (`_sweep_bilinear!`, bilinear_execution.jl), with the action sink in place of a replay
# target: two points swept at once never add into the same `y[row]`, because each colour
# keeps them farther apart than their rows reach. `_sweep_bilinear!` takes its mechanism from
# the walked leaf, as assembly does: `Threads.@threads` for a `CpuSerial` or `CpuThreaded`
# leaf, the Polyester hooks for a `CpuPolyester` one. An operator whose own policy is
# `CpuPolyester` takes the Polyester hooks for every leaf (`_mf_target`). A unit whose rows
# are not a fixed reach from the point (a test-side interpolation, which names rows through
# `locate_cell`) walks serially, as the threaded assembly's does (`_sweep_bilinear_serial!`).
#
# The colours are assembly's own, `_term_colour_strides(term)` (`_colour_strides` over the
# term's row reach, computed without a heap vector, linear.jl), and a pair's take both terms'
# row reach, since its transposed entries land on the first term's columns.

const _ActionTarget = Union{ActionSink, _PairActionSink}

# An action target the sweep walks with the Polyester hooks whatever its leaf's backend:
# `_sweep_bilinear!` picks the mechanism from the leaf, and the two colour methods below hand
# the unwrapped target to the `CpuPolyester` ones instead. Built only by `_mf_target`.
struct _BatchedAction{S <: _ActionTarget}
    s::S
end

# The target a unit's threaded sweep writes through under the operator's policy.
@inline _mf_target(::CpuPolicy, s) = s
@inline _mf_target(::CpuPolyester, s::_ActionTarget) = _BatchedAction(s)

for P in (CpuThreaded, CpuPolyester)
    @eval begin
        @inline _sweep_band_colour!(
            ::$P, t::_BatchedAction, sp, term, ax, bidx, nbands::Int, rest, lin_indices,
            mesh_markers, row_offset::Int, col_offset::Int, α) = _sweep_band_colour!(
            CpuPolyester(), t.s, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers,
            row_offset, col_offset, α)
        @inline _sweep_bilinear_colour!(
            ::$P, t::_BatchedAction, sp, term, idxs, lin_indices, mesh_markers,
            row_offset::Int, col_offset::Int, α) = _sweep_bilinear_colour!(
            CpuPolyester(), t.s, sp, term, idxs, lin_indices, mesh_markers, row_offset,
            col_offset, α)
    end
end

@inline _mf_visit!(::CpuSerial, s, term, sp, ro::Int, co::Int) = (
    visit_bilinear_stencil(s, term, sp, ro, co); nothing)
@inline function _mf_visit!(p::CpuPolicy, s, term::TERM, sp, ro::Int, co::Int) where {TERM}
    if _has_test_interp(term)
        visit_bilinear_stencil(s, term, sp, ro, co)
    else
        _sweep_bilinear!(_mf_target(p, s), sp, term, _term_colour_strides(term), ro, co)
    end
    return nothing
end

# A pair's entry writes its transpose too, so the serial fallback also takes an absolute
# column of `p1` (a trial-side interpolation, which becomes a transposed row), as
# `_replay_pair_unit!` does.
@inline _mf_visit_pair!(::CpuSerial, s, p1, _p2, sp, ro::Int, co::Int) = (
    visit_bilinear_stencil(s, p1, sp, ro, co); nothing)
@inline function _mf_visit_pair!(
        p::CpuPolicy, s, p1::P1, p2::P2, sp, ro::Int, co::Int
) where {P1, P2}
    if _has_test_interp(p1) || _has_trial_interp(p1) || _has_test_interp(p2)
        visit_bilinear_stencil(s, p1, sp, ro, co)
    else
        _sweep_bilinear!(_mf_target(p, s), sp, p1, _term_colour_strides(p1, p2), ro, co)
    end
    return nothing
end

# One point of a threaded sweep (`_sweep_point!`, bilinear_execution.jl): the stencil at `I`,
# through the guarded entry walk. The sink carries its own `α`.
@inline _sweep_point!(
    s::_ActionTarget, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, _) = _replay_point!(
    s, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset)

# `CpuPolyester`'s bands and colours, through the same hooks the threaded replay uses
# (`_batch_bilinear_band_replay!`, `_batch_bilinear_colour_replay!`), filled by
# `BramblePolyesterExt` for an action target too.
@noinline function _sweep_band_colour!(
        ::CpuPolyester, s::_ActionTarget, sp, term::TERM, ax, bidx, nbands::Int, rest,
        lin_indices, mesh_markers, row_offset::Int, col_offset::Int, _
) where {TERM}
    return _late(
        _batch_bilinear_band_replay!,
        s, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, row_offset, col_offset
    )
end

@noinline function _sweep_bilinear_colour!(
        ::CpuPolyester, s::_ActionTarget, sp, term::TERM, idxs, lin_indices, mesh_markers,
        row_offset::Int, col_offset::Int, _
) where {TERM}
    return _late(
        _batch_bilinear_colour_replay!,
        s, sp, term, idxs, lin_indices, mesh_markers, row_offset, col_offset
    )
end

# --- The fused threaded sweep ------------------------------ #
#
# Sweeping each unit in its own colour bands, as above, costs two parallel regions per unit
# and walks every point through the guarded entry walk: on 4 threads the product was at most
# 1.8x serial, and slower than serial below about 65k unknowns, where the regions' launches
# alone outweighed the walk. The fused sweep turns the loops inside out: the grid is cut into
# one band per thread along its last axis, and each band's task walks every unit over its
# own slab of the grid, the interior of the slab unguarded and its share of the boundary shell
# guarded, exactly as `visit_bilinear_stencil` splits the whole grid. A product is one
# parallel region whatever the unit count, and a point costs what it costs serially.
#
# One region, not two colours: each task owns the rows of its band's points and adds only
# into those, so no two tasks write one entry of `y` and no colour has to wait for another.
# A point near a band's edge also writes rows of the next band (the walk scatters to
# test-offset rows), so a task walks its band widened by the rows' reach, `[a - omax,
# b - omin]` along the last axis for owned slices `a:b`, and keeps, on the widened rim only,
# the entries whose row it owns (`_OwnedAction`). Every entry is then added exactly once, by
# its row's owner. Its core, the points all of whose rows fall in the band, is walked with
# the plain sink. Two colours of alternate bands cost a second region instead, and on this
# host a `Threads.@threads :static` region launched right after another costs about 30 us:
# the two alone were slower than the serial product at 4096 unknowns (see `_mf_run_bands!`
# for the one region's own mechanism).
#
# `omin`/`omax` bound every unit's row offsets along the last axis (`stencil_offsets`, a
# pair's two terms together, since its transposed entries land on the first term's columns).
# `stencil_offsets` allocates, so they are collected once, when the operator is built
# (`_mf_plan`), by the same unit walk the product runs. Units writing different row blocks
# (a composite's leaves) own the same slices of their own block.
#
# The fused sweep needs one band cut every unit agrees on: every unit it takes walks a grid of
# the same size, under the same effective policy (the mechanism, `_run_bands!`, is chosen
# once). A form that fails this (a composite on leaves of different sizes, or mixed backends)
# keeps the per-unit sweep above. Units with a test-side interpolation name their rows through
# `locate_cell`, which no offset bounds; they are walked serially after the bands, as in the
# per-unit sweep.

"""
    _MFFusedPlan{D, EP, F}

The fused threaded sweep's plan, fixed when a [`MatrixFreeOperator`](@ref) is built: the
grid size `dims` every fused unit walks, the least and greatest row offset `omin`, `omax`
of any fused unit along the last axis, the effective parallel policy `policy` every fused
unit's leaf carries (or the operator's own, when that is [`CpuPolyester`](@ref)), whether
some unit walks serially (`interp`, a test-side interpolation), and the operator's own form
`form`, boxed once here.

The band tasks read the form through `form` rather than taking it as an argument:
`Polyester.@batch` copies its arguments into a heap box on every call, and a form stored
inline (its spaces' weight vectors, a coefficient's grid function) made that box 400-740 B
per product. The box now holds one pointer to `form`. The box is written once, here, and
only read afterwards, so concurrent products on one operator share it safely, and it keeps
alive nothing the operator does not already hold.
"""
struct _MFFusedPlan{D, EP <: CpuPolicy, F <: BilinearForm}
    dims::NTuple{D, Int}
    omin::Int
    omax::Int
    policy::EP
    interp::Bool
    form::Base.RefValue{F}
end

# What the build-time walk collects: every fused unit's row offsets, the grid size and
# effective policy of the first fused unit's leaf, whether every other one agrees, and
# whether some unit walks serially. `forced`, when not `nothing`, is the effective policy of
# every leaf instead (the operator's `CpuPolyester`). Build time only, so untyped.
mutable struct _MFCollected
    offsets::Vector{Any}
    dims::Any
    policy::Any
    agree::Bool
    interp::Bool
    forced::Any
end
_MFCollected(forced = nothing) = _MFCollected(Any[], nothing, nothing, true, false, forced)

# The unit walk's `policy` inside the fused sweep: `_mf_apply!` runs its usual recursion, and
# each unit is collected (`_MF_COLLECT`, build time), walked over one band (`_MF_BAND`: owned
# slices `own`, row reach `omin:omax`) or, if it walks serially, walked whole (`_MF_SERIAL`).
# One type for the three, so the unit walk compiles once for them.
const _MF_COLLECT = 0
const _MF_BAND = 1
const _MF_SERIAL = 2

struct _MFPass
    mode::Int
    own::UnitRange{Int}
    omin::Int
    omax::Int
    acc::_MFCollected
end

# Read by no band or serial pass: those never collect. It is mutable and shared by every
# task, so a pass that writes to it would race; only the collect pass, which runs once and
# alone in `_mf_plan`, may write to its own `_MFCollected`.
const _MF_NO_COLLECT = _MFCollected()

# The plan for `a` under `policy`: `nothing` for a serial policy, or a form the fused sweep
# cannot take (no fused unit, or units that disagree on the grid or policy). Under
# `CpuPolyester` every leaf's effective policy is `CpuPolyester`, whatever its backend, so the
# bands reach `_run_bands!(::CpuPolyester)`; any other policy takes each leaf's own.
_mf_plan(::CpuSerial, ::BilinearForm) = nothing
function _mf_plan(policy::CpuPolicy, a::BilinearForm)
    acc = _MFCollected(policy isa CpuPolyester ? policy : nothing)
    T = _matrix_eltype(a, a.ast)
    _mf_apply!(_MFPass(_MF_COLLECT, 1:0, 0, 0, acc), ActionSink(T[], T[], true, nothing), a)
    (acc.agree && acc.dims !== nothing) || return nothing
    dims = acc.dims::NTuple
    D = length(dims)
    last_offsets = Int[o[D] for o in acc.offsets]
    return _MFFusedPlan(
        dims, minimum(last_offsets; init = 0), maximum(last_offsets; init = 0), acc.policy,
        acc.interp, Ref(a)
    )
end

function _mf_collect!(acc::_MFCollected, offsets, sp)
    grid_inds = indices(mesh(sp))
    dims = size(grid_inds)
    policy = execution_policy(sp)
    # The bands cut `1:dims[D]`, and `_run_bands!` threads a CPU policy.
    if all(r -> first(r) == 1, axes(grid_inds)) && policy isa CpuPolicy
        policy = acc.forced === nothing ? _coerce_serial_to_threaded(policy) : acc.forced
    else
        acc.agree = false
    end
    if acc.dims === nothing
        acc.dims = dims
        acc.policy = policy
    elseif acc.dims != dims || acc.policy != policy
        acc.agree = false
    end
    append!(acc.offsets, offsets)
    return nothing
end

@inline function _mf_visit!(p::_MFPass, s, term::TERM, sp, ro::Int, co::Int) where {TERM}
    if _has_test_interp(term)
        p.mode == _MF_SERIAL && visit_bilinear_stencil(s, term, sp, ro, co)
        p.mode == _MF_COLLECT && (p.acc.interp = true)
    elseif p.mode == _MF_BAND
        _mf_visit_band!(p, s, term, sp, ro, co)
    elseif p.mode == _MF_COLLECT
        _mf_collect!(p.acc, stencil_offsets(term), sp)
    end
    return nothing
end

@inline function _mf_visit_pair!(
        p::_MFPass, s, p1::P1, p2::P2, sp, ro::Int, co::Int
) where {P1, P2}
    if _has_test_interp(p1) || _has_trial_interp(p1) || _has_test_interp(p2)
        p.mode == _MF_SERIAL && visit_bilinear_stencil(s, p1, sp, ro, co)
        p.mode == _MF_COLLECT && (p.acc.interp = true)
    elseif p.mode == _MF_BAND
        _mf_visit_band!(p, s, p1, sp, ro, co)
    elseif p.mode == _MF_COLLECT
        _mf_collect!(p.acc, union(stencil_offsets(p1), stencil_offsets(p2)), sp)
    end
    return nothing
end

"""
    _OwnedAction(s, lo, hi, lo2, hi2)

An action sink (`ActionSink` or `_PairActionSink`) that keeps only the entries whose row
the calling task owns: `lo:hi` for an entry's own row, `lo2:hi2` for a pair's transposed
row. The fused sweep's widened band rim walks through it (see the section comment).
"""
struct _OwnedAction{S}
    s::S
    lo::Int
    hi::Int
    lo2::Int
    hi2::Int
end

# The rows of block-local linear indices `lo:hi`: at `ro` for an entry's own row, at the
# transposed block's origin `co + dr` for a pair's transposed one.
@inline _mf_owned(s::ActionSink, lo::Int, hi::Int, ro::Int, ::Int) = _OwnedAction(
    s, lo + ro, hi + ro, 0, -1)
@inline _mf_owned(s::_PairActionSink, lo::Int, hi::Int, ro::Int, co::Int) = _OwnedAction(
    s, lo + ro, hi + ro, lo + co + s.dr, hi + co + s.dr)

@inline function _sink_entry!(o::_OwnedAction{<:ActionSink}, row::Int, col::Int, weight, slot::Int)
    o.lo <= row <= o.hi && _sink_entry!(o.s, row, col, weight, slot)
    return nothing
end

@inline function _sink_entry!(
        o::_OwnedAction{<:_PairActionSink}, row::Int, col::Int, weight, ::Int
)
    s = o.s
    if s.half != 2 && o.lo <= row <= o.hi && _mf_live(s.mask, row)
        @inbounds s.y[row] += s.α1 * weight * s.x[col]
    end
    if s.half != 1
        r2 = col + s.dr
        if o.lo2 <= r2 <= o.hi2 && _mf_live(s.mask, r2)
            @inbounds s.y[r2] += s.α2 * weight * s.x[row + s.dc]
        end
    end
    return nothing
end

# One unit over one band: the owned slices `a:b` widened to every point that writes one of
# their rows, `[a - omax, b - omin]`. Its core `[a - omin, b - omax]`, whose rows all fall in
# `a:b`, goes through the plain sink, and the rim on either side through `_OwnedAction`.
# `lo`, `core` and `hi` are the widened range cut in three, in increasing order (the whole
# range is `lo` when the band is too narrow to have a core).
@inline function _mf_visit_band!(p::_MFPass, s, term::TERM, sp, ro::Int, co::Int) where {TERM}
    dims = size(indices(mesh(sp)))
    len = last(dims)
    stride = prod(Base.front(dims))
    a, b = first(p.own), last(p.own)
    vlo, vhi = max(1, a - p.omax), min(len, b - p.omin)
    clo, chi = max(vlo, a - p.omin), min(vhi, b - p.omax)
    owned = _mf_owned(s, (a - 1) * stride + 1, b * stride, ro, co)
    if clo > chi
        _mf_visit_slab!(s, owned, term, sp, ro, co, vlo:vhi, 1:0, 1:0)
    else
        _mf_visit_slab!(s, owned, term, sp, ro, co, vlo:(clo - 1), clo:chi, (chi + 1):vhi)
    end
    return nothing
end

# `visit_bilinear_stencil` restricted to the points of `sp`'s grid whose last index lies in
# `lo`, `core` or `hi`: the interior box and each boundary slab of the whole grid, in the
# whole-grid walk's order, each cut to the three ranges in turn, the interior unguarded.
# Every row a band owns therefore receives its entries in the order the serial walk adds
# them, since within a piece the points are visited column-major, the last axis slowest, as
# serially: the threaded product is bitwise the serial one. The regions are the same
# `CartesianIndices` of `UnitRange{Int}`s the whole-grid walk passes, so the two share their
# compiled walks.
@noinline function _mf_visit_slab!(
        s::SINK, owned::OWNED, term::TERM, sp, ro::Int, co::Int, lo::UnitRange{Int},
        core::UnitRange{Int}, hi::UnitRange{Int}
) where {SINK, OWNED, TERM}
    Ωₕ = mesh(sp)
    mesh_markers = markers(Ωₕ)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    margin = _stencil_margin(term)
    ax = axes(grid_inds)
    if _peelable(ax, margin)
        interior = map(r -> _interior_range(r, margin), ax)
        front, r = Base.front(interior), last(interior)
        _mf_piece!(owned, term, sp, mesh_markers, lin_indices, front, r, lo, ro, co, true)
        _mf_piece!(s, term, sp, mesh_markers, lin_indices, front, r, core, ro, co, true)
        _mf_piece!(owned, term, sp, mesh_markers, lin_indices, front, r, hi, ro, co, true)
        @inbounds for shell in _boundary_shell_slabs(ax, margin)
            front, r = Base.front(shell.indices), last(shell.indices)
            _mf_piece!(owned, term, sp, mesh_markers, lin_indices, front, r, lo, ro, co, false)
            _mf_piece!(s, term, sp, mesh_markers, lin_indices, front, r, core, ro, co, false)
            _mf_piece!(owned, term, sp, mesh_markers, lin_indices, front, r, hi, ro, co, false)
        end
    else
        front, r = map(_full_range, Base.front(ax)), _full_range(last(ax))
        _mf_piece!(owned, term, sp, mesh_markers, lin_indices, front, r, lo, ro, co, false)
        _mf_piece!(s, term, sp, mesh_markers, lin_indices, front, r, core, ro, co, false)
        _mf_piece!(owned, term, sp, mesh_markers, lin_indices, front, r, hi, ro, co, false)
    end
    return nothing
end

# One piece of the whole-grid walk (`front` by `r`), cut to the last-axis range `cut`.
@inline function _mf_piece!(
        sink, term, sp, mesh_markers, lin_indices, front, r, cut, ro, co, unguarded::Bool
)
    region = CartesianIndices((front..., intersect(r, cut)))
    isempty(region) && return nothing
    if unguarded
        _visit_interior!(sink, term, sp, mesh_markers, lin_indices, region, ro, co)
    else
        _visit_guarded_region!(sink, term, sp, mesh_markers, lin_indices, region, ro, co)
    end
    return nothing
end

# One product. Without a plan, the unit walk under the operator's own policy: serial, or the
# per-unit threaded sweep. A plan whose boxed form is not `a` (an operator rebuilt around
# another form) walks per unit too, since its bands would read the wrong form.
@inline _mf_product!(::Nothing, policy, s, a::BilinearForm) = _mf_apply!(policy, s, a)

function _mf_product!(plan::_MFFusedPlan{D}, policy, s::ActionSink, a::BilinearForm) where {D}
    plan.form[] === a || return _mf_apply!(policy, s, a)
    len = plan.dims[D]
    nbands = min(Threads.nthreads(), len)
    _mf_run_bands!(plan.policy, s, a, plan, nbands)
    plan.interp && _mf_apply!(_MFPass(_MF_SERIAL, 1:0, 0, 0, _MF_NO_COLLECT), s, a)
    return nothing
end

# Task `k` of `ntasks` (`_run_bands!`'s trailing pair): bands `k`, `k + ntasks`, ... of
# `nbands`, every unit walked over each. `y` is `s.y`, first for `_run_bands!`'s sake. The
# form is the plan's (`_MFFusedPlan`); the third argument is ignored, `nothing` under
# `_run_bands!` so that no `@batch` box carries the form inline.
@noinline function _mf_band_task!(
        _y, s, _, plan::_MFFusedPlan{D}, nbands::Int, ntasks::Int, k::Int
) where {D}
    a = plan.form[]
    len = plan.dims[D]
    for b in k:ntasks:nbands
        own = _band_range(1:len, nbands, b)
        _mf_apply!(_MFPass(_MF_BAND, own, plan.omin, plan.omax, _MF_NO_COLLECT), s, a)
    end
    return nothing
end

# The region's mechanism. `CpuPolyester` (and any other CPU policy) takes `_run_bands!`, as
# the engines do. `CpuThreaded` spawns one task per band and walks the first band on the
# calling task, rather than `_run_bands!`'s `Threads.@threads :static`. On this host a
# `:static` region costs 25-30 us when it follows another closely (a product repeated in a
# solver), `:dynamic` about 20 us with the walk, a spawn about 5 us, and at 4096 unknowns the
# whole serial product is 30 us (2D 64²: 32 us `:static`, 20 us `:dynamic`, 11 us spawned).
# Nothing on the path reads `threadid()`, so a task may run on, or move to, any thread. A
# spawn nests inside a user's `Threads.@threads` loop, which `:static` cannot
# (`_static_or_serial`), so no fallback is needed. `@sync` waits by yielding, and the waiting
# thread runs the bands itself if every other one is busy. Each band writes only its own
# rows, in the serial order, so the result does not depend on where a band runs.
@inline _mf_run_bands!(policy::CpuPolicy, s, _, plan, nbands::Int) = _run_bands!(
    policy, _mf_band_task!, s.y, s, nothing, plan, nbands)

@noinline function _mf_run_bands!(::CpuThreaded, s, a, plan, nbands::Int)
    @sync begin
        for b in 2:nbands
            Threads.@spawn _mf_band_task!(s.y, s, a, plan, nbands, nbands, b)
        end
        _mf_band_task!(s.y, s, a, plan, nbands, nbands, 1)
    end
    return nothing
end
