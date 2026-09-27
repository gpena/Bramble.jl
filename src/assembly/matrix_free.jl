# matrix_free.jl: `MatrixFreeOperator`, a `BilinearForm` applied to a vector without its
# matrix (gpena/Bramble.jl#326). `mul!` walks the form's (term, block) units exactly as the
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
    MatrixFreeOperator{T, Form, DLabels, EP} <: AbstractMatrix{T}

A [`BilinearForm`](@ref) as a linear operator: `mul!(y, op, x)` computes `A * x`, and
`mul!(y, op, x, α, β)` computes `α * A * x + β * y`, for `A = assemble(a; dirichlet)`,
without ever storing `A`. Each product walks the form's stencil once, evaluated afresh, so a
coefficient updated in place (`Rₕ!`, a `Ref`) is seen by the next product. Build one with
[`matrix_free_operator`](@ref).

`x` and `y` are plain vectors or [`VectorElement`](@ref)s: `op * x` returns a new vector,
`op * uₕ` a new element of the form's test space. `mul!` allocates nothing on a serial
policy, except on a form with a region restriction (`restrict_to`), whose stencil
evaluation allocates as it does in `assemble!`.

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

See also: [`matrix_free_operator`](@ref), [`assemble`](@ref), [`KroneckerLinearOperator`](@ref).
"""
struct MatrixFreeOperator{T, Form <: BilinearForm, DLabels, EP <: ExecutionPolicy} <:
       AbstractMatrix{T}
    form::Form
    labels::DLabels
    mask::BitVector
    policy::EP
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
  accepts (`:boundary`, a tuple of labels, `:left => g`, [`dirichlet_constraints`](@ref));
  boundary values are ignored, as they are by the matrix (default: `nothing`).
- `dirichlet_components`: The leaves of a composite test space the labels bind to, as in
  [`dirichlet_bc!`](@ref): an `Int`, a `Tuple` of `Int`s, or `nothing` for every leaf
  (default: `nothing`).
- `policy`: The [`ExecutionPolicy`](@ref) of `mul!` (default: the trial space's). A CPU
  policy's product currently runs serially.

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
    return MatrixFreeOperator{T, typeof(a), typeof(labels), typeof(policy)}(
        a, labels, mask, policy, nrows, ndofs(trial_space(a))
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
    if iszero(β)
        fill!(yd, zero(eltype(yd)))
    elseif !isone(β)
        yd .*= β
    end
    mask = _mf_mask(op)
    _mf_apply!(ActionSink(yd, xd, α, mask), op.form)
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
# own method instance instead of inlining into one grown with the term count.

function _mf_apply!(s::ActionSink, a::BilinearForm)
    Wu, Wv, ast = trial_space(a), test_space(a), a.ast
    if _is_block_pair(Wu, Wv)
        _mf_blocks!(s, ast, leaf_spaces_offsets(Wu), leaf_spaces_offsets(Wv))
        return nothing
    end
    bound = _bind_interp_spaces(ast, Wu, Wv)
    _check_block_meshes(bound, Wu, Wv)
    sp = host_weights(_walked_leaf(bound, Wu, Wv))
    _mf_summands!(s, bound, sp)
    return nothing
end

@noinline function _mf_summands!(s::ActionSink, op::OperatorAdd, sp)
    _foldl_pairs(
        (_, t) -> _mf_summands!(s, t, sp),
        (_, t1, t2) -> _mf_pair!(s, t1, t2, sp),
        nothing,
        _summands(op)
    )
    return nothing
end

@noinline function _mf_summands!(s::ActionSink, term::TERM, sp) where {TERM}
    visit_bilinear_stencil(s, term, sp, 0, 0)
    return nothing
end

@noinline function _mf_pair!(s::ActionSink, t1, t2, sp)
    sink = _pair_action_sink(s, s.α * _term_scale(t1), s.α * _term_scale(t2), 0, 0, 0)
    visit_bilinear_stencil(sink, _bare_product(t1), sp, 0, 0)
    return nothing
end

@noinline function _mf_blocks!(s::ActionSink, op::OperatorAdd, trial_leaves, test_leaves)
    _foldl_pairs(
        (_, t) -> _mf_blocks!(s, t, trial_leaves, test_leaves),
        (_, t1, t2) -> _mf_pair_blocks!(s, t1, t2, trial_leaves, test_leaves),
        nothing,
        _summands(op)
    )
    return nothing
end

@noinline function _mf_blocks!(
        s::ActionSink, term::TERM, trial_leaves, test_leaves
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        bound = _bind_interp_spaces(term, blk.trial_leaf, blk.test_leaf)
        _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
        sp = host_weights(_walked_leaf(bound, blk.trial_leaf, blk.test_leaf))
        visit_bilinear_stencil(s, bound, sp, blk.row_offset, blk.col_offset)
    end
    return nothing
end

# `_replay_pair_blocks!`'s walk: block `k` of the second term holds the transposes of block
# `k` of the first, walked on one leaf when both blocks share it and once per leaf, one half
# each, when they do not.
@noinline function _mf_pair_blocks!(s::ActionSink, t1, t2, trial_leaves, test_leaves)
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
                visit_bilinear_stencil(
                    _pair_action_sink(s, α1, α2, dr, dc, 0), bound, host_weights(sp), ro, co
                )
            else
                visit_bilinear_stencil(
                    _pair_action_sink(s, α1, α2, dr, dc, 1), bound, host_weights(sp), ro, co
                )
                visit_bilinear_stencil(
                    _pair_action_sink(s, α1, α2, dr, dc, 2), bound,
                    host_weights(blk2.test_leaf), ro, co
                )
            end
        end
        return nothing
    end
    Base.inferencebarrier(_mf_blocks!)(s, t1, trial_leaves, test_leaves)
    Base.inferencebarrier(_mf_blocks!)(s, t2, trial_leaves, test_leaves)
    return nothing
end
