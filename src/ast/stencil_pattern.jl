# stencil_pattern.jl
#
# Grid offsets an operator reaches, read off its AST before assembling it.
#
# Every node here reaches a fixed set of neighbours: a backward difference reads the point
# and the one before it, a cross-weighted difference reads three in a row, an average reads
# two. Those offsets are a property of the tree, not of the grid point: truncation at a
# boundary zeroes the coefficients and keeps the offsets, which is what makes the set well
# defined. So the sparsity pattern of the assembled matrix is known before a single entry
# is computed.
#
# The pattern is used for preallocation. The sparsity pattern of an assembled operator
# matches the stencil's reach, and building the backend matrix with this pattern ensures
# assembly only updates stored values without structural inserts. The offsets inform how
# the backend matrix is preallocated; they do not dictate the concrete matrix type, which
# is configured by `Backend{VT, MT}`.

"""
    stencil_offsets(op) -> Vector{NTuple{D, Int}}

The grid offsets the operator `op` reaches, sorted and without repeats.

Read from the AST rather than from an evaluated stencil, and the two agree: a truncated
point keeps its offsets and zeroes its coefficients, so the set does not vary over the grid.
The one node whose reach is not fixed by its type is [`ShiftNode`](@ref), which carries its
step as a field; the value is available here because this walks the built tree.

For a `BilinearProduct` this is the row (test-side) reach only, not the full row/column
pattern -- see the note on that method.
"""
function stencil_offsets end

# --- the leaves: a single entry on the diagonal ------------------------------------ #
_origin(::LazyOp{D}) where {D} = [ntuple(_ -> 0, D)]

stencil_offsets(op::TrialFunction) = _origin(op)
stencil_offsets(op::TestFunction) = _origin(op)
stencil_offsets(op::IndexedTrialFunction) = _origin(op)
stencil_offsets(op::IndexedTestFunction) = _origin(op)
stencil_offsets(op::SourceFunction) = _origin(op)
stencil_offsets(op::SourceVector) = _origin(op)
stencil_offsets(op::SourceConstant) = _origin(op)
stencil_offsets(op::DiracSource) = _origin(op)
stencil_offsets(op::IdentityOperator) = _origin(op)
stencil_offsets(op::ZeroOperator) = _origin(op)

# --- combining what a node's child reaches ----------------------------------------- #

# `op` reaches everything its child does, each moved by every step in `steps`.
@inline function _reach(inner::Vector, ::Val{Dim}, steps::Tuple) where {Dim}
    out = eltype(inner)[]
    for s in steps, o in inner

        p = shift_offset(o, Dim, s)
        p in out || push!(out, p)
    end
    return sort!(out)
end

# --- the operators ------------------------------------------------------------------ #

# A difference, an average or a jump reaches the point and one or two neighbours, and which
# ones is precisely the tap set its stencil is built from. That used to be written out a
# second time here, once per node, so the reach and the stencil agreed only by both being
# maintained: it now reads the node's own `_stencil_taps` (gpena/Bramble.jl#70). `_reach`
# sorts and de-duplicates, so the taps' evaluation order does not matter to it -- only the
# set does, which is why the same declaration serves both.
function stencil_offsets(op::TappedNode{D, Dim}) where {D, Dim}
    return _reach(
        stencil_offsets(op.inner_op), Val(Dim), map(_shift_delta, _stencil_taps(op))
    )
end

# a shift moves the whole reach and widens nothing
function stencil_offsets(op::ShiftNode{D, Dim}) where {D, Dim}
    return _reach(stencil_offsets(op.inner_op), Val(Dim), (op.shift_amount,))
end

# scaling changes coefficients, not reach
stencil_offsets(op::OperatorScale) = stencil_offsets(op.inner_op)
stencil_offsets(op::GridFunctionScale) = stencil_offsets(op.inner_op)

# a restriction empties the stencil outside its region and leaves it alone inside, so the
# reach is its child's wherever the operator contributes at all
stencil_offsets(op::RegionRestriction) = stencil_offsets(op.inner_op)

# A linear product contracts the left factor away (`multiply_stencils_linear` keeps only
# the right offsets), so its reach is the test side's.
stencil_offsets(op::LinearProduct) = stencil_offsets(op.right_op)

# A bilinear product's stencil entries are (row offset, column offset, value) triples, so
# the general "what does this reach" question is genuinely ambiguous between the two sets.
# The one caller that matters (`_colour_strides`, form/linear.jl, used for both the linear
# and bilinear parallel-assembly paths) only ever needs the row side, though: `add_to_sparse!`
# (form/bilinear.jl) finds a fixed nzval slot for each stored (row, col) pair, so two writes
# can only collide if both their row AND column coincide. If colouring already keeps rows
# from coinciding, the pair can't either, whatever the columns are -- so this reduces to the
# test factor's reach, the same way `stencil_offsets(::LinearProduct)` above already does.
# Building the full sparsity pattern (both sets) is a different question, answered
# separately by walking `local_stencil` directly (`_coord_walk!`, form/bilinear_pattern.jl).
stencil_offsets(op::BilinearProduct) = stencil_offsets(op.right_op)

function stencil_offsets(op::OperatorAdd)
    return sort!(union(stencil_offsets(op.left_op), stencil_offsets(op.right_op)))
end

"""
    _stencil_margin(op) -> Int

The largest `|offset|` component anywhere in `op`'s stencil, trial and test sides combined.

Unlike [`stencil_offsets`](@ref), which keeps only the test-side (row) reach at a
`BilinearProduct` because that is all its one caller (`_colour_strides`) needs, this keeps
both. It exists for `visit_bilinear_stencil`'s interior/boundary split
(`form/bilinear_traversal.jl`, gpena/Bramble.jl#160), which has to guard every offset an
entry can carry -- trial and test alike, not just the row's -- before it can skip the guard
anywhere.

Answered as a scalar rather than the offset set `stencil_offsets` returns: the traversal
only needs a margin to build a rectangular interior box from, and a scalar costs no
allocation to compute, where reusing `stencil_offsets` (built on `union`-ing `Vector`s)
would mean paying that cost inside `visit_bilinear_stencil` itself on the `ReplaySink` path,
once per `assemble!` call rather than once per form.

A composed reach past magnitude 1 is not a hypothetical this ignores: `D₋ₓ(D₋ₓ(u))` and
`Shift(u, dim, 2)` both carry offsets wider than the one step a single difference or a unit
shift would suggest, which is why this walks the same tree `stencil_offsets` does rather
than assuming every node caps out at 1.
"""
_stencil_margin(op::TrialFunction) = 0
_stencil_margin(op::TestFunction) = 0
_stencil_margin(op::IndexedTrialFunction) = 0
_stencil_margin(op::IndexedTestFunction) = 0
_stencil_margin(op::SourceFunction) = 0
_stencil_margin(op::SourceVector) = 0
_stencil_margin(op::SourceConstant) = 0
_stencil_margin(op::DiracSource) = 0
_stencil_margin(op::IdentityOperator) = 0
_stencil_margin(op::ZeroOperator) = 0

# A tapped node's own step is always exactly 1: every `_stencil_taps` entry across every
# difference, average and jump is drawn from `{-1, 0, 1}` (`stencil_eval.jl`), so composing
# margin here is `+1` regardless of which node it is -- the same fact `_reach` relies on to
# shift the offset set by one tap.
_stencil_margin(op::TappedNode) = _stencil_margin(op.inner_op) + 1

# A shift's step is its own field, unlike a tap's, and unbounded: `simplify_ast` folds nested
# shifts along the same dimension into one (`Shift_a(Shift_b(u)) -> Shift_{a+b}(u)`), so this
# has to read the value rather than assume magnitude 1.
_stencil_margin(op::ShiftNode) = _stencil_margin(op.inner_op) + abs(op.shift_amount)

_stencil_margin(op::OperatorScale) = _stencil_margin(op.inner_op)
_stencil_margin(op::GridFunctionScale) = _stencil_margin(op.inner_op)
_stencil_margin(op::RegionRestriction) = _stencil_margin(op.inner_op)

# Names an absolute column on another mesh via `locate_cell`, never a grid-relative offset on
# the mesh being walked -- see `stencil_offsets(::InterpolationNode)` just above.
_stencil_margin(op::InterpolationNode) = 0

# The one place trial and test reach actually differ: `left_op` is what `off_u` (trial/column)
# ranges over, `right_op` what `off_v` (test/row) does (`multiply_stencils_bilinear`,
# `form/common.jl`), so the margin has to cover whichever side reaches further.
function _stencil_margin(op::BilinearProduct)
    return max(_stencil_margin(op.left_op), _stencil_margin(op.right_op))
end

function _stencil_margin(op::OperatorAdd)
    return max(_stencil_margin(op.left_op), _stencil_margin(op.right_op))
end

# --- bandwidths and blockbandwidths, read from the AST alone ----------------------- #
#
# `bandwidths`/`blockbandwidths` take a `BilinearForm`, but that type is defined later
# (`form/bilinear.jl`, included after this file): a type annotation naming it here would
# need it to already exist at `include` time, which it does not. The argument is left
# untyped instead -- `resolve_form_ast`, `trial_space`, `test_space` and
# `CompositeGridSpace` below are ordinary calls, looked up when the function actually
# runs rather than when this file loads, so the forward reference costs nothing once the
# package has finished loading (gpena/Bramble.jl#175).

@noinline function _throw_composite_not_banded()
    throw(
        ArgumentError(
        "bandwidths and blockbandwidths are not defined for a composite trial or test " *
        "space. A composite system matrix is assembled block by block (leaf_spaces_offsets), " *
        "and a leaf's own mesh can differ in point count from another leaf's, so a block's " *
        "offset is not, in general, one lexicographic distance in a single shared ordering. " *
        "Compute the bandwidth of one leaf pair's block instead, on the non-composite form " *
        "that block alone would be.",
    ),
    )
end

@noinline function _throw_interpolation_not_banded()
    throw(
        ArgumentError(
        "bandwidths and blockbandwidths are not defined for a form containing an " *
        "interpolation operator (πₕ). Interpolation addresses an absolute column or row " *
        "on another mesh, which has no grid-relative offset and so no lexicographic " *
        "distance to report.",
    ),
    )
end

# Composite spaces, any interpolation anywhere, and a genuine cross-mesh pairing (two
# leaves that share neither a mesh nor an interpolation between them, `_throw_cross_mesh_block`
# in `form/bilinear.jl`) are the three ways a form has no single lexicographic bandwidth.
# The first two are refused outright; the third reuses the same guard `allocate_system_matrix`
# already runs, rather than duplicating its mesh-compatibility check.
function _check_bandable(a)
    (trial_space(a) isa CompositeGridSpace || test_space(a) isa CompositeGridSpace) &&
        _throw_composite_not_banded()
    ast = resolve_form_ast(a)
    (_has_trial_interp(ast) || _has_test_interp(ast)) && _throw_interpolation_not_banded()
    _check_block_meshes(ast, trial_space(a), test_space(a))
    return nothing
end

# The strides a column-major, first-axis-fastest lexicographic index uses: moving one step
# along axis `d` moves the linear index by `stride[d] = n₁⋯n_{d-1}` (`stride[1] = 1`, the
# empty product).
@inline _lex_strides(n::NTuple{D, Int}) where {D} = ntuple(k -> prod(n[1:(k - 1)]), D)

# The column-minus-row distance a trial offset `ou` paired with a test offset `ov`
# contributes, in the ordering `strides` describes.
@inline function _lex_distance(
        ou::NTuple{D, Int}, ov::NTuple{D, Int}, strides::NTuple{D, Int}
) where {D}
    return sum(ntuple(d -> (ou[d] - ov[d]) * strides[d], D))
end

# Every `BilinearProduct` a form's AST sums, as its (trial, test) offset reach -- the one
# pairing `stencil_offsets(::BilinearProduct)` itself does not keep, since its one caller
# (`_colour_strides`) only ever needs the test side. A sum reaches every term either side
# has; scaling, a grid-function coefficient or a restriction changes no term's reach, only
# its coefficients, so the walk passes straight through them to their own `inner_op`.
function _collect_bilinear_terms!(terms, op::OperatorAdd)
    _collect_bilinear_terms!(terms, op.left_op)
    _collect_bilinear_terms!(terms, op.right_op)
    return terms
end

function _collect_bilinear_terms!(terms, op::BilinearProduct)
    push!(terms, (stencil_offsets(op.left_op), stencil_offsets(op.right_op)))
    return terms
end

_collect_bilinear_terms!(terms, op::UnaryWrapper) = _collect_bilinear_terms!(terms, op.inner_op)
_collect_bilinear_terms!(terms, ::LazyOp) = terms

_bilinear_terms(ast) = _collect_bilinear_terms!(Tuple{Vector, Vector}[], ast)

"""
    bandwidths(a::BilinearForm) -> Tuple{Int, Int}

The lower and upper bandwidth `(l, u)` of the matrix `a` assembles into, in the
lexicographic ordering [`indices`](@ref)`(Ωₕ)` walks: column-major, first axis fastest.
Row `i` can only carry a stored entry in column `j` when `-l <= j - i <= u`.

Read entirely from the resolved AST, without assembling anything. `a` is a sum of
`BilinearProduct` terms; for each, [`stencil_offsets`](@ref) gives the trial (column) and
test (row) reach along every axis. An offset `(o₁, …, o_D)` moves the lexicographic index
by `o₁ + o₂n₁ + o₃n₁n₂ + …`, where `(n₁, …, n_D)` are the mesh's per-axis point counts
([`npoints`](@ref)`(Ωₕ, Tuple)`). `l` and `u` are the largest such distance, negated and
as-is, taken over every pairing of a trial offset with a test offset with the
corresponding row's offset, over every term of the sum.

# Throws
- `ArgumentError`: the trial or test space is a `CompositeGridSpace`; the form contains
  an interpolation operator (`πₕ`); or the trial and test spaces couple two leaves that
  share neither a mesh nor an interpolation between them.

This is internal: it is not exported, and has had no active consumer since the banded
backends were dropped in v3.3.0. Reach it as `Bramble.bandwidths` or with
`using Bramble: bandwidths`.

See also: [`blockbandwidths`](@ref), [`stencil_offsets`](@ref).
"""
function bandwidths(a)
    _check_bandable(a)
    ast = resolve_form_ast(a)
    strides = _lex_strides(npoints(mesh(test_space(a)), Tuple))
    l = 0
    u = 0
    for (trial_offsets, test_offsets) in _bilinear_terms(ast)
        for ou in trial_offsets, ov in test_offsets

            d = _lex_distance(ou, ov, strides)
            l = max(l, -d)
            u = max(u, d)
        end
    end
    return (l, u)
end

"""
    blockbandwidths(a::BilinearForm) -> Tuple{Tuple{Int, Int}, Tuple{Int, Int}}

The block bandwidth and sub-block bandwidth of the matrix `a` assembles into, reading its
lexicographic index as an outer block along the last axis and an inner lexicographic
index over the remaining `D - 1` axes -- the layout `BlockBandedMatrices.jl`'s
`BandedBlockBandedMatrix` expects, `(blockbandwidths, subblockbandwidths)`.

For `D == 1` every block is a single point, so this is `((0, 0), bandwidths(a))`.

For `D >= 2`, blocks are the `n₁⋯n_{D-1}`-point rectangles indexed by the `D`-th axis:
`l_blk`/`u_blk` are the largest reach a term's `D`-th-axis offsets carry between a trial
and a test point, `l_sub`/`u_sub` the largest lexicographic distance the remaining
`D - 1` axes carry, both computed the same way [`bandwidths`](@ref) computes its one
pair, just restricted to their own axes.

# Throws
- `ArgumentError`: same conditions as [`bandwidths`](@ref).

This is internal: it is not exported, and has had no active consumer since the banded
backends were dropped in v3.3.0. Reach it as `Bramble.blockbandwidths` or with
`using Bramble: blockbandwidths`.

See also: [`bandwidths`](@ref).
"""
function blockbandwidths(a)
    D = dim(test_space(a))
    D == 1 && return ((0, 0), bandwidths(a))

    _check_bandable(a)
    ast = resolve_form_ast(a)
    n = npoints(mesh(test_space(a)), Tuple)
    sub_strides = _lex_strides(Base.front(n))

    l_blk = 0
    u_blk = 0
    l_sub = 0
    u_sub = 0
    for (trial_offsets, test_offsets) in _bilinear_terms(ast)
        for ou in trial_offsets, ov in test_offsets

            db = last(ou) - last(ov)
            l_blk = max(l_blk, -db)
            u_blk = max(u_blk, db)
            # `ou`/`ov` come from `stencil_offsets`, documented to return `Vector{NTuple{D, Int}}`
            # -- always a `Tuple`, never a `NamedTuple` -- but a form built through enough
            # generic wrapper nodes (`OperatorAdd`, `RegionRestriction`, ...) infers that Vector's
            # element type no more precisely than `Any`. `Base.front` has methods for both `Tuple`
            # and `NamedTuple`, so calling it on an `Any` union-splits into both, and only the
            # `Tuple` branch has a matching `_lex_distance` method. The assertions state what is
            # already true of every `stencil_offsets` result, not a new constraint on it.
            ds = _lex_distance(Base.front(ou)::Tuple, Base.front(ov)::Tuple, sub_strides)
            l_sub = max(l_sub, -ds)
            u_sub = max(u_sub, ds)
        end
    end
    return ((l_blk, u_blk), (l_sub, u_sub))
end
