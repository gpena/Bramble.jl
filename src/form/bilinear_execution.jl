# bilinear_execution.jl: the two strategies that scatter a `BilinearForm`'s values into an
# already-allocated matrix -- the serial record/replay cache (`_assemble_bilinear_core_cached!`)
# and the threaded, band-coloured sweep (`_assemble_bilinear_parallel_core!`). The parallel
# path never uses the sinks in `bilinear_traversal.jl`: it owns its own grid loop so threads
# can be handed disjoint bands, sharing only the entry rule (`_entry_target`) and
# `add_to_sparse!`.

# --- Serial: record once, replay thereafter ---------------------------------------- #
#
# Two ways to fill a term's block, serially: `_record_segment!` searches for each entry's
# nzval position (as every call used to) and also records it; `_replay_segment!` reads a
# previously recorded position back instead of searching. `_assemble_bilinear_core_cached!`
# picks between them by whether `A` is the exact matrix object the form's cache was last
# built against (gpena/Bramble.jl#26).
#
# The parallel path below is untouched by this cache: it always threads, and recording is an
# inherently serial, one-time pass (`push!` from multiple threads would race), so caching it
# would mean its first call silently stopped threading -- breaking `assemble_parallel!`'s own
# documented contract ("always threads").

# One term into one block, serially, searching for each entry's nzval position (once) and
# recording it into a fresh `AnySegment` alongside performing the (first) real scatter.
# `row_offset` comes from the test leaf and `col_offset` from the trial leaf: a matrix row
# is indexed by the test function.
function _record_segment!(
        A::SparseMatrixCSC, term::TERM, sp, row_offset::Int, col_offset::Int
) where {TERM}
    n = length(indices(mesh(sp)))
    sink = RecordSink(A, term, Vector{Int}(undef, n + 1), Int[])
    visit_bilinear_stencil(sink, term, sp, row_offset, col_offset)
    @inbounds sink.point_ptr[n + 1] = length(sink.positions) + 1
    # No `::AnySegment{D}` type assertion here: `AnySegment` alone (no `D`) is an abstract
    # `UnionAll`, and asserting a value into it would *widen* what the compiler tracks the
    # result as, discarding the concrete two-member union `_try_diagonal_segment` actually
    # returns for this specialization -- exactly the boxing `DiagonalSegment{D}`'s own
    # docstring warns about, just moved one call up.
    return _try_diagonal_segment(term, sp, sink.point_ptr, sink.positions)
end

# Attempts to repackage a freshly recorded flat segment as a `DiagonalSegment`, restricted to
# `term`'s own interior/boundary split (gpena/Bramble.jl#160): `_visit_interior!` -- called by
# `visit_bilinear_stencil` before any boundary slab -- pushes every interior point's `P`
# entries, in `LinearIndices(interior)` order, so `positions[1:n_interior*P]` already *is*
# the interior in that order, with no need to touch `point_ptr` to find it.
#
# Validated against the actual recorded positions, not assumed from `term`'s own margin
# (gpena/Bramble.jl#161): a form summing terms of different margins into the same block
# (`_record_blocks!(::OperatorAdd, ...)`) records one segment per summand, and a narrower
# term's own "interior" can still include columns whose true `nzval` footprint -- set by
# every segment sharing that column, not just this one -- varies where a wider-margin
# sibling still adds taps. That shows up here as a non-constant per-tap stride; any mismatch,
# or too little interior to bother, falls back to the plain `NzvalSegment` unchanged, which
# `ReplaySink` already handles.
function _try_diagonal_segment(
        term::TERM, sp, point_ptr::Vector{Int}, positions::Vector{Int}
) where {TERM}
    flat = (point_ptr, positions)::NzvalSegment
    grid_inds = indices(mesh(sp))
    ax = axes(grid_inds)
    margin = _stencil_margin(term)
    _peelable(ax, margin) || return flat

    interior = CartesianIndices(map(r -> _interior_range(r, margin), ax))
    n_interior = length(interior)
    n_interior < 2 && return flat

    first_lin = LinearIndices(grid_inds)[first(interior)]
    @inbounds point_ptr[first_lin] == 1 || return flat
    @inbounds P = point_ptr[first_lin + 1] - point_ptr[first_lin]
    P == 0 && return flat

    total_interior = n_interior * P
    total_interior > length(positions) && return flat

    base = positions[1:P]
    stride = Vector{Int}(undef, P)
    @inbounds for k in 1:P
        stride[k] = positions[P + k] - base[k]
    end
    @inbounds for n in 2:(n_interior - 1)
        off = n * P
        prev = off - P
        for k in 1:P
            positions[off + k] - positions[prev + k] == stride[k] || return flat
        end
    end

    boundary_positions = positions[(total_interior + 1):end]
    boundary_point_ptr = point_ptr .- total_interior
    return DiagonalSegment(
        base, stride, P, interior, (boundary_point_ptr, boundary_positions)
    )
end

# The replay counterpart: same walk, same fresh stencil evaluation (weights may be live --
# only positions are fixed), but each entry's nzval index comes from `segment` instead of a
# search. `point_ptr[lin_idx]` addresses each point's own slice of `positions` directly, so
# this stays correct regardless of what order grid points are visited in.
function _replay_segment!(
        A::SparseMatrixCSC,
        term::TERM,
        sp,
        row_offset::Int,
        col_offset::Int,
        segment::NzvalSegment
) where {TERM}
    point_ptr, positions = segment
    visit_bilinear_stencil(
        ReplaySink(A, point_ptr, positions), term, sp, row_offset, col_offset
    )
    return nothing
end

# The diagonal counterpart: the interior core replays through `DiagonalReplaySink`'s stride
# arithmetic, and the boundary shell replays exactly as `NzvalSegment` always has, through an
# ordinary `ReplaySink` over `segment.boundary` -- the two-sink form of
# `visit_bilinear_stencil` runs both in the one walk.
function _replay_segment!(
        A::SparseMatrixCSC,
        term::TERM,
        sp,
        row_offset::Int,
        col_offset::Int,
        segment::DiagonalSegment
) where {TERM}
    boundary_point_ptr, boundary_positions = segment.boundary
    visit_bilinear_stencil(
        DiagonalReplaySink(A, segment.interior, segment.base, segment.stride, segment.P),
        ReplaySink(A, boundary_point_ptr, boundary_positions),
        term,
        sp,
        row_offset,
        col_offset
    )
    return nothing
end

# The scalar case: one block, no offsets, so exactly one segment either way.
#
# `segments::Vector{T}) where {...,T<:AnySegment{D}}` throughout this file, rather than the
# nominally simpler `segments::Vector{AnySegment{D}}`: `D` is fixed per call (the form's own
# dimension), but which of `NzvalSegment`/`DiagonalSegment{D}` each *element* is stays a
# per-element concrete choice `_record_segment!` makes -- `T` is that two-member union,
# inferred, not `AnySegment{D}` the abstract alias itself.
function _record_bilinear_core!(
        A::SparseMatrixCSC, trial_space, test_space, ast::AST_TYPE, segments::Vector{T}
) where {AST_TYPE, T <: AnySegment}
    _check_block_meshes(ast, trial_space, test_space)
    push!(segments, _record_segment!(A, ast, test_space, 0, 0))
    return nothing
end

function _replay_bilinear_core!(
        A::SparseMatrixCSC, trial_space, test_space, ast::AST_TYPE, segments::Vector{T}
) where {AST_TYPE, T <: AnySegment}
    _check_block_meshes(ast, trial_space, test_space)
    _replay_segment!(A, ast, test_space, 0, 0, segments[1])
    return nothing
end

function _record_blocks!(
        A::SparseMatrixCSC, op::OperatorAdd, trial_leaves, test_leaves, segments::Vector{T}
) where {T <: AnySegment}
    _record_blocks!(A, op.left_op, trial_leaves, test_leaves, segments)
    _record_blocks!(A, op.right_op, trial_leaves, test_leaves, segments)
    return nothing
end

function _record_blocks!(
        A::SparseMatrixCSC, term::TERM, trial_leaves, test_leaves, segments::Vector{T}
) where {TERM, T <: AnySegment}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        push!(
            segments,
            _record_segment!(A, term, blk.test_leaf, blk.row_offset, blk.col_offset)
        )
    end
    return nothing
end

# `next` is threaded through by value and returned, rather than via a mutable `Ref`, so
# this stays allocation-free: the segment index the *next* leaf-term/block should consume,
# in the same left-then-right order `_record_blocks!` built `segments` in.
function _replay_blocks!(
        A::SparseMatrixCSC,
        op::OperatorAdd,
        trial_leaves,
        test_leaves,
        segments::Vector{T},
        next::Int
) where {T <: AnySegment}
    next = _replay_blocks!(A, op.left_op, trial_leaves, test_leaves, segments, next)
    next = _replay_blocks!(A, op.right_op, trial_leaves, test_leaves, segments, next)
    return next
end

function _replay_blocks!(
        A::SparseMatrixCSC,
        term::TERM,
        trial_leaves,
        test_leaves,
        segments::Vector{T},
        next::Int
) where {TERM, T <: AnySegment}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        next += 1
        _replay_segment!(
            A, term, blk.test_leaf, blk.row_offset, blk.col_offset, segments[next]
        )
    end
    return next
end

function _record_bilinear_core!(
        A::SparseMatrixCSC,
        trial_space::CompositeGridSpace,
        test_space::CompositeGridSpace,
        ast::AST_TYPE,
        segments::Vector{T}
) where {AST_TYPE, T <: AnySegment}
    _record_blocks!(
        A, ast, leaf_spaces_offsets(trial_space), leaf_spaces_offsets(test_space), segments
    )
    return nothing
end

function _replay_bilinear_core!(
        A::SparseMatrixCSC,
        trial_space::CompositeGridSpace,
        test_space::CompositeGridSpace,
        ast::AST_TYPE,
        segments::Vector{T}
) where {AST_TYPE, T <: AnySegment}
    _replay_blocks!(
        A,
        ast,
        leaf_spaces_offsets(trial_space),
        leaf_spaces_offsets(test_space),
        segments,
        0
    )
    return nothing
end

# Picks recording (cache miss: a fresh `A`, a changed `ast` -- e.g. `assemble!(A, form;
# ast = ...)` given something other than `form.ast` -- or the first call ever) or replay
# (cache hit) and keeps `cache` in step with whichever one ran. `ast` is checked as well as
# `A`: the cache's positions are only valid for the exact stencil shape they were recorded
# against, and a different `ast` can visit a different number of entries per point.
function _assemble_bilinear_core_cached!(
        A::SparseMatrixCSC, trial_space, test_space, ast::AST_TYPE, cache::_AssemblyCache{D}
) where {AST_TYPE, D}
    if cache.A === A && cache.ast === ast
        _replay_bilinear_core!(A, trial_space, test_space, ast, cache.segments)
    else
        # A fresh vector, not `empty!` on whatever `cache.segments` currently references,
        # in case that reference is ever shared (it never is, today, but nothing here
        # relies on `cache.segments` being exclusively owned).
        segments = AnySegment{D}[]
        _record_bilinear_core!(A, trial_space, test_space, ast, segments)
        cache.segments = segments
        cache.A = A
        cache.ast = ast
    end
    return A
end

# --- Threaded: band-coloured sweeps ------------------------------------------------ #

# One grid point's stencil, scattered into the matrix. Used only by the parallel path
# below: it always searches (never caches), so a serial recording pass is never required
# before a `Parallel()`-backend form's first assembly.
@inline function _scatter_point!(
        A::SparseMatrixCSC,
        term::TERM,
        sp,
        I::CartesianIndex,
        lin_indices,
        mesh_markers,
        row_offset::Int,
        col_offset::Int
) where {TERM}
    stencil = local_stencil(term, sp, I, mesh_markers, lin_indices[I])

    # One point rather than the whole grid, so this cannot call `visit_bilinear_stencil`
    # itself -- the threaded sweep owns the grid loop. It shares the entry rule instead, so
    # the guard and the `AbsoluteColumn` case are still stated in exactly one place.
    for (off_u, off_v, weight) in stencil
        row, col = _entry_target(lin_indices, I, off_u, off_v, row_offset, col_offset)
        row == 0 && continue
        add_to_sparse!(A, row, col, weight, term)
    end
    return nothing
end

# One colour, threaded, writing directly into the matrix.
@noinline function _sweep_bilinear_colour!(
        A::SparseMatrixCSC,
        sp,
        term::TERM,
        idxs,
        lin_indices,
        mesh_markers,
        row_offset::Int,
        col_offset::Int
) where {TERM}
    Threads.@threads for I in idxs
        _scatter_point!(A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset)
    end
    return nothing
end

"""
    _sweep_band_colour!(A, sp, term, ax, parity, nbands, rest, lin_indices, mesh_markers, row_offset, col_offset) -> Nothing

Scatter one band colour of `term` into `A` across threads.

Each thread takes one slab of the last axis and walks it whole. Two grid points can only
reach the same matrix entry when they are closer than `strides[D]` along that axis -- their
stencil footprints cannot meet otherwise -- so slabs of at least that width, taken every
other one, never write the same entry concurrently, whatever the remaining axes do. A term
that reaches only its own point cannot collide at all, and then `bidx` is every band at
once.
"""
@noinline function _sweep_band_colour!(
        A::SparseMatrixCSC,
        sp,
        term::TERM,
        ax,
        bidx,
        nbands::Int,
        rest,
        lin_indices,
        mesh_markers,
        row_offset::Int,
        col_offset::Int
) where {TERM}
    Threads.@threads for b in bidx
        for I in CartesianIndices((rest..., _band_range(ax, nbands, b)))
            _scatter_point!(
                A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset
            )
        end
    end
    return nothing
end

# Every colour in turn, using strided subgrids.
function _sweep_bilinear!(
        A::SparseMatrixCSC, sp, term::TERM, strides, row_offset::Int, col_offset::Int
) where {TERM}
    Ωₕ = mesh(sp)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    mesh_markers = markers(Ωₕ)

    # Bands first: a colour is then a slab of whole rows, walked contiguously, and there
    # are two of them however wide the stencil is. Point colouring strides every axis and
    # costs `prod(strides)` barriers, which on a nine-colour form was slower than not
    # threading at all (gpena/Bramble.jl, `Dcₓ + Dcᵧ` at 1000x1000: 16.6 ms threaded
    # against 16.4 ms serial, and 8.1 ms banded).
    inds = grid_inds.indices
    D = length(strides)
    ax = inds[D]
    nbands = _band_count(length(ax), strides[D], Threads.nthreads())

    if nbands != 0
        rest = Base.front(inds)
        # A term reaching only its own point cannot collide, so one pass takes every band:
        # contiguous slabs without the second barrier, which such a term would otherwise
        # pay for nothing (it doubled `assemble_parallel!`'s task allocations).
        bands = prod(strides) == 1 ? (1:1:nbands,) : (1:2:nbands, 2:2:nbands)
        for bidx in bands
            _sweep_band_colour!(
                A,
                sp,
                term,
                ax,
                bidx,
                nbands,
                rest,
                lin_indices,
                mesh_markers,
                row_offset,
                col_offset
            )
        end
        return A
    end

    # Too short to band (a small grid, or a stencil reaching most of the axis): point
    # colouring has no width requirement.
    if prod(strides) == 1
        _sweep_bilinear_colour!(
            A, sp, term, grid_inds, lin_indices, mesh_markers, row_offset, col_offset
        )
        return A
    end

    for c in CartesianIndices(strides)
        _sweep_bilinear_colour!(
            A,
            sp,
            term,
            _colour_subgrid(grid_inds, c, strides),
            lin_indices,
            mesh_markers,
            row_offset,
            col_offset
        )
    end
    return A
end

function _assemble_blocks_parallel!(
        A::SparseMatrixCSC, op::OperatorAdd, trial_leaves, test_leaves
)
    return _visit_operator_add2(
        _assemble_blocks_parallel!, A, op, trial_leaves, test_leaves
    )
end

function _assemble_blocks_parallel!(
        A::SparseMatrixCSC, term::TERM, trial_leaves, test_leaves
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        _sweep_bilinear!(
            A,
            blk.test_leaf,
            term,
            _colour_strides(stencil_offsets(term)),
            blk.row_offset,
            blk.col_offset
        )
    end
    return A
end

function _assemble_bilinear_parallel_core!(
        A::SparseMatrixCSC, trial_space, test_space, ast::AST_TYPE
) where {AST_TYPE}
    _check_block_meshes(ast, trial_space, test_space)
    _sweep_bilinear!(A, test_space, ast, _colour_strides(stencil_offsets(ast)), 0, 0)
    return A
end

function _assemble_bilinear_parallel_core!(
        A::SparseMatrixCSC,
        trial_space::CompositeGridSpace,
        test_space::CompositeGridSpace,
        ast::AST_TYPE
) where {AST_TYPE}
    _assemble_blocks_parallel!(
        A, ast, leaf_spaces_offsets(trial_space), leaf_spaces_offsets(test_space)
    )
    return A
end
