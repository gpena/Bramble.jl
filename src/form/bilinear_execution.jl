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
# inherently serial, one-time pass (concurrent writes into a shared cache would race), so
# caching it would mean its first call silently stopped threading -- breaking
# `assemble_parallel!`'s own documented contract ("always threads").

# One term into one block, serially, searching for each entry's nzval position (once) and
# recording it into a fresh `Segment` alongside performing the (first) real scatter.
# `row_offset` comes from the test leaf and `col_offset` from the trial leaf: a matrix row
# is indexed by the test function.
#
# Two passes, not one: `_SegmentCountSink` walks the term first to learn exactly how many
# entries it has, so `RecordSink`'s own `positions` can be preallocated to that size and
# filled with `setindex!` -- never grown with `push!` (gpena/Bramble.jl#240). See
# `_SegmentCountSink`'s own docstring for why: `push!` here is what made `Enzyme` unable to
# compile this function at all when the term's own coefficient is what is being
# differentiated, independent of (and found only after fixing) the `Union` gpena/Bramble.jl#240
# was originally filed against.
function _record_segment!(
        A::SparseMatrixCSC, term::TERM, sp, row_offset::Int, col_offset::Int, α
) where {TERM}
    n = length(indices(mesh(sp)))
    point_ptr = Vector{Int}(undef, n + 1)
    count_sink = _SegmentCountSink(point_ptr, 0)
    visit_bilinear_stencil(count_sink, term, sp, row_offset, col_offset)
    total = count_sink.n
    @inbounds point_ptr[n + 1] = total + 1

    sink = RecordSink(A, term, point_ptr, Vector{Int}(undef, total), α, 0)
    visit_bilinear_stencil(sink, term, sp, row_offset, col_offset)
    # No `::Segment{D}` type assertion here: `Segment` alone (no `D`) is an abstract
    # `UnionAll`, and asserting a value into it would *widen* what the compiler tracks the
    # result as, discarding the concrete `Segment{D}` `_try_diagonal_segment` actually
    # returns for this specialization -- exactly the boxing `Segment{D}`'s own docstring
    # warns about, just moved one call up.
    return _try_diagonal_segment(term, sp, sink.point_ptr, sink.positions)
end

# Attempts to repackage a freshly recorded flat segment as a diagonal `Segment`, restricted
# to `term`'s own interior/boundary split (gpena/Bramble.jl#160): `_visit_interior!` --
# called by `visit_bilinear_stencil` before any boundary slab -- pushes every interior
# point's `P` entries, in `LinearIndices(interior)` order, so `positions[1:n_interior*P]`
# already *is* the interior in that order, with no need to touch `point_ptr` to find it.
#
# Validated against the actual recorded positions, not assumed from `term`'s own margin
# (gpena/Bramble.jl#161): a form summing terms of different margins into the same block
# (`_record_blocks!(::OperatorAdd, ...)`) records one segment per summand, and a narrower
# term's own "interior" can still include columns whose true `nzval` footprint -- set by
# every segment sharing that column, not just this one -- varies where a wider-margin
# sibling still adds taps. That shows up here as a non-constant per-tap stride; any mismatch,
# or too little interior to bother, falls back to the plain flat segment unchanged, which
# `ReplaySink` already handles.
#
# Every early return builds a `Segment{D}` via `_flat_segment`, never a bare tuple: the
# return type must stay the single concrete `Segment{D}` on every path, not a `Union` with
# some other flat representation -- that Union is exactly what gpena/Bramble.jl#240 removed.
function _try_diagonal_segment(
        term::TERM, sp, point_ptr::Vector{Int}, positions::Vector{Int}
) where {TERM}
    grid_inds = indices(mesh(sp))
    ax = axes(grid_inds)
    D = length(ax)
    margin = _stencil_margin(term)
    _peelable(ax, margin) || return _flat_segment(Val(D), point_ptr, positions)

    interior = CartesianIndices(map(r -> _interior_range(r, margin), ax))
    n_interior = length(interior)
    n_interior < 2 && return _flat_segment(Val(D), point_ptr, positions)

    first_lin = LinearIndices(grid_inds)[first(interior)]
    @inbounds point_ptr[first_lin] == 1 || return _flat_segment(Val(D), point_ptr, positions)
    @inbounds P = point_ptr[first_lin + 1] - point_ptr[first_lin]
    P == 0 && return _flat_segment(Val(D), point_ptr, positions)

    total_interior = n_interior * P
    total_interior > length(positions) && return _flat_segment(Val(D), point_ptr, positions)

    base = positions[1:P]
    stride = Vector{Int}(undef, P)
    @inbounds for k in 1:P
        stride[k] = positions[P + k] - base[k]
    end
    @inbounds for n in 2:(n_interior - 1)
        off = n * P
        prev = off - P
        for k in 1:P
            positions[off + k] - positions[prev + k] == stride[k] ||
                return _flat_segment(Val(D), point_ptr, positions)
        end
    end

    boundary_positions = positions[(total_interior + 1):end]
    boundary_point_ptr = point_ptr .- total_interior
    return Segment{D}(true, boundary_point_ptr, boundary_positions, base, stride, P, interior)
end

# The replay counterpart: same walk, same fresh stencil evaluation (weights may be live --
# only positions are fixed), but each entry's nzval index comes from `segment` instead of a
# search. `point_ptr[lin_idx]` addresses each point's own slice of `positions` directly, so
# this stays correct regardless of what order grid points are visited in.
#
# One method branching on `segment.is_diagonal`, not two dispatching on `NzvalSegment` versus
# `DiagonalSegment` (gpena/Bramble.jl#240): both branches are fully concrete (`Segment{D}`
# has one shape, not a Union), so the runtime `if` costs nothing extra over the dispatch it
# replaced -- flat replays exactly as before through `ReplaySink`; diagonal replays the
# interior core through `DiagonalReplaySink`'s stride arithmetic and the boundary shell
# through an ordinary `ReplaySink`, the two-sink form of `visit_bilinear_stencil` running
# both in the one walk.
function _replay_segment!(
        A::SparseMatrixCSC,
        term::TERM,
        sp,
        row_offset::Int,
        col_offset::Int,
        segment::Segment,
        α
) where {TERM}
    if segment.is_diagonal
        visit_bilinear_stencil(
            DiagonalReplaySink(
                A, segment.interior, segment.base, segment.stride, segment.P, α
            ),
            ReplaySink(A, segment.point_ptr, segment.positions, α),
            term,
            sp,
            row_offset,
            col_offset
        )
    else
        visit_bilinear_stencil(
            ReplaySink(A, segment.point_ptr, segment.positions, α),
            term,
            sp,
            row_offset,
            col_offset
        )
    end
    return nothing
end

# The scalar case: one block, no offsets, so exactly one segment either way.
#
# `segments::Vector{Segment{D}}` throughout this file: `D` is fixed per call (the form's own
# dimension) and, since gpena/Bramble.jl#240, `Segment{D}` is the single concrete element
# type regardless of whether a given element is flat or diagonal (`is_diagonal` selects the
# shape at the value level, not the type level) -- no `where {T <: ...}` indirection needed
# to keep the vector unboxed, unlike the `AnySegment{D}` union this replaced.
function _record_bilinear_core!(
        A::SparseMatrixCSC, trial_space, test_space, ast::AST_TYPE, segments::Vector{Segment{D}}, α
) where {AST_TYPE, D}
    bound = _bind_interp_spaces(ast, trial_space, test_space)
    _check_block_meshes(bound, trial_space, test_space)
    sp = _walked_leaf(bound, trial_space, test_space)
    push!(segments, _record_segment!(A, bound, sp, 0, 0, α))
    return nothing
end

function _replay_bilinear_core!(
        A::SparseMatrixCSC, trial_space, test_space, ast::AST_TYPE, segments::Vector{Segment{D}}, α
) where {AST_TYPE, D}
    bound = _bind_interp_spaces(ast, trial_space, test_space)
    _check_block_meshes(bound, trial_space, test_space)
    sp = _walked_leaf(bound, trial_space, test_space)
    _replay_segment!(A, bound, sp, 0, 0, segments[1], α)
    return nothing
end

function _record_blocks!(
        A::SparseMatrixCSC, op::OperatorAdd, trial_leaves, test_leaves, segments::Vector{Segment{D}}, α
) where {D}
    _record_blocks!(A, op.left_op, trial_leaves, test_leaves, segments, α)
    _record_blocks!(A, op.right_op, trial_leaves, test_leaves, segments, α)
    return nothing
end

function _record_blocks!(
        A::SparseMatrixCSC, term::TERM, trial_leaves, test_leaves, segments::Vector{Segment{D}}, α
) where {TERM, D}
    for blk in blocks(term, trial_leaves, test_leaves)
        bound = _bind_interp_spaces(term, blk.trial_leaf, blk.test_leaf)
        _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
        sp = _walked_leaf(bound, blk.trial_leaf, blk.test_leaf)
        push!(
            segments, _record_segment!(A, bound, sp, blk.row_offset, blk.col_offset, α)
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
        segments::Vector{Segment{D}},
        next::Int,
        α
) where {D}
    next = _replay_blocks!(A, op.left_op, trial_leaves, test_leaves, segments, next, α)
    next = _replay_blocks!(A, op.right_op, trial_leaves, test_leaves, segments, next, α)
    return next
end

function _replay_blocks!(
        A::SparseMatrixCSC,
        term::TERM,
        trial_leaves,
        test_leaves,
        segments::Vector{Segment{D}},
        next::Int,
        α
) where {TERM, D}
    for blk in blocks(term, trial_leaves, test_leaves)
        bound = _bind_interp_spaces(term, blk.trial_leaf, blk.test_leaf)
        _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
        sp = _walked_leaf(bound, blk.trial_leaf, blk.test_leaf)
        next += 1
        _replay_segment!(
            A, bound, sp, blk.row_offset, blk.col_offset, segments[next], α
        )
    end
    return next
end

function _record_bilinear_core!(
        A::SparseMatrixCSC,
        trial_space::CompositeGridSpace,
        test_space::CompositeGridSpace,
        ast::AST_TYPE,
        segments::Vector{Segment{D}},
        α
) where {AST_TYPE, D}
    _record_blocks!(
        A, ast, leaf_spaces_offsets(trial_space), leaf_spaces_offsets(test_space), segments, α
    )
    return nothing
end

function _replay_bilinear_core!(
        A::SparseMatrixCSC,
        trial_space::CompositeGridSpace,
        test_space::CompositeGridSpace,
        ast::AST_TYPE,
        segments::Vector{Segment{D}},
        α
) where {AST_TYPE, D}
    _replay_blocks!(
        A,
        ast,
        leaf_spaces_offsets(trial_space),
        leaf_spaces_offsets(test_space),
        segments,
        0,
        α
    )
    return nothing
end

# Picks recording (cache miss: a fresh `A`, a changed `ast` -- e.g. `assemble!(A, form;
# ast = ...)` given something other than `form.ast` -- or the first call ever) or replay
# (cache hit) and keeps `cache` in step with whichever one ran. `ast` is checked as well as
# `A`: the cache's positions are only valid for the exact stencil shape they were recorded
# against, and a different `ast` can visit a different number of entries per point.
function _assemble_bilinear_core_cached!(
        A::SparseMatrixCSC,
        trial_space,
        test_space,
        ast::AST_TYPE,
        cache::_AssemblyCache{D, CACHED_AST},
        α = true
) where {AST_TYPE, D, CACHED_AST}
    if cache.valid && cache.A_id === objectid(A) && cache.ast === ast
        _replay_bilinear_core!(A, trial_space, test_space, ast, cache.segments, α)
    else
        # A fresh vector, not `empty!` on whatever `cache.segments` currently references,
        # in case that reference is ever shared (it never is, today, but nothing here
        # relies on `cache.segments` being exclusively owned).
        #
        # Recorded with `α`: recording performs the (first) real scatter as well as
        # learning the `nzval` positions (`_record_segment!`'s own docstring), so the very
        # first `assemble_add!(A, a, α)` call must scale that scatter too, not only the
        # replays after it -- the cache is keyed on `(A, ast)` alone, never on `α`, since
        # nzval *positions* never depend on it: an `assemble_add!` caller is free to change
        # `α` (a `Ref`'s current value, say) on every call and still replay from cache.
        segments = Segment{D}[]
        _record_bilinear_core!(A, trial_space, test_space, ast, segments, α)
        _store_recording!(cache, ast, segments, A)
    end
    return A
end

# Keeps `cache` in step with a recording that just ran, but only for an `ast` of the type the
# cache was built around -- `form.ast`'s type, since `form` constructs the two together.
@inline function _store_recording!(
        cache::_AssemblyCache{D, AST}, ast::AST, segments::Vector{Segment{D}}, A
) where {D, AST}
    cache.segments = segments
    cache.A_id = objectid(A)
    cache.ast = ast
    cache.valid = true
    return nothing
end

# A differently-typed `ast` records and scatters, then stores nothing. This is the deprecated
# `assemble!(A, form; ast = <another form's tree>)` keyword (gpena/Bramble.jl#105, removed in
# v3.0.0): the cache's `AST` parameter belongs to `form.ast`, so there is nothing to store it
# in. Dropping the write also leaves whatever `form.ast` recording the cache already holds
# intact and still correct -- `nzval` positions depend on `A`'s sparsity pattern, which a
# scatter does not change -- so the next ordinary `assemble!` still replays instead of paying
# for a re-record. Before the cache carried its AST as a type parameter this case overwrote
# the cache with the substitute tree; now it cannot, and need not.
@inline _store_recording!(::_AssemblyCache, _ast, _segments, _A) = nothing

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
        col_offset::Int,
        α
) where {TERM}
    stencil = local_stencil(term, sp, I, mesh_markers, lin_indices[I])

    # One point rather than the whole grid, so this cannot call `visit_bilinear_stencil`
    # itself -- the threaded sweep owns the grid loop. It shares the entry rule instead, so
    # the guard and the `AbsoluteColumn` case are still stated in exactly one place.
    for (off_u, off_v, weight) in stencil
        row, col = _entry_target(lin_indices, I, off_u, off_v, row_offset, col_offset)
        row == 0 && continue
        add_to_sparse!(A, row, col, α * weight, term)
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
        col_offset::Int,
        α
) where {TERM}
    Threads.@threads for I in idxs
        _scatter_point!(A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α)
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
        col_offset::Int,
        α
) where {TERM}
    Threads.@threads for b in bidx
        for I in CartesianIndices((rest..., _band_range(ax, nbands, b)))
            _scatter_point!(
                A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α
            )
        end
    end
    return nothing
end

# The serial fallback for a term whose rows are not a fixed reach from the point being
# visited.
#
# Colouring keeps two concurrently-swept points from writing the same row, and it decides
# which points may run together from the term's own offsets (`_colour_strides` over
# `stencil_offsets`). A test-side interpolation (gpena/Bramble.jl#263) names its rows
# absolutely, through `locate_cell`, so no offset set describes them and two points in
# different colours can land on the same row. The term is swept on one thread instead, which
# is correct at the cost of the threading, and `_has_test_interp` decides it from the type,
# so an ordinary term pays nothing for the choice.
function _sweep_bilinear_serial!(
        A::SparseMatrixCSC, sp, term::TERM, row_offset::Int, col_offset::Int, α = true
) where {TERM}
    Ωₕ = mesh(sp)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    mesh_markers = markers(Ωₕ)

    @inbounds for I in grid_inds
        _scatter_point!(
            A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α
        )
    end
    return nothing
end

# Every colour in turn, using strided subgrids.
function _sweep_bilinear!(
        A::SparseMatrixCSC, sp, term::TERM, strides, row_offset::Int, col_offset::Int, α = true
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
                col_offset,
                α
            )
        end
        return A
    end

    # Too short to band (a small grid, or a stencil reaching most of the axis): point
    # colouring has no width requirement.
    if prod(strides) == 1
        _sweep_bilinear_colour!(
            A, sp, term, grid_inds, lin_indices, mesh_markers, row_offset, col_offset, α
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
            col_offset,
            α
        )
    end
    return A
end

function _assemble_blocks_parallel!(
        A::SparseMatrixCSC, op::OperatorAdd, trial_leaves, test_leaves, α = true
)
    return _visit_operator_add2(
        _assemble_blocks_parallel!, A, op, trial_leaves, test_leaves, α
    )
end

function _assemble_blocks_parallel!(
        A::SparseMatrixCSC, term::TERM, trial_leaves, test_leaves, α = true
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        bound = _bind_interp_spaces(term, blk.trial_leaf, blk.test_leaf)
        _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
        sp = _walked_leaf(bound, blk.trial_leaf, blk.test_leaf)
        if _has_test_interp(bound)
            _sweep_bilinear_serial!(A, sp, bound, blk.row_offset, blk.col_offset, α)
        else
            _sweep_bilinear!(
                A,
                sp,
                bound,
                _colour_strides(stencil_offsets(bound)),
                blk.row_offset,
                blk.col_offset,
                α
            )
        end
    end
    return A
end

function _assemble_bilinear_parallel_core!(
        A::SparseMatrixCSC, trial_space, test_space, ast::AST_TYPE, α = true
) where {AST_TYPE}
    bound = _bind_interp_spaces(ast, trial_space, test_space)
    _check_block_meshes(bound, trial_space, test_space)
    sp = _walked_leaf(bound, trial_space, test_space)
    if _has_test_interp(bound)
        _sweep_bilinear_serial!(A, sp, bound, 0, 0, α)
    else
        _sweep_bilinear!(A, sp, bound, _colour_strides(stencil_offsets(bound)), 0, 0, α)
    end
    return A
end

function _assemble_bilinear_parallel_core!(
        A::SparseMatrixCSC,
        trial_space::CompositeGridSpace,
        test_space::CompositeGridSpace,
        ast::AST_TYPE,
        α = true
) where {AST_TYPE}
    _assemble_blocks_parallel!(
        A, ast, leaf_spaces_offsets(trial_space), leaf_spaces_offsets(test_space), α
    )
    return A
end
