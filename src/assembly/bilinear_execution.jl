# bilinear_execution.jl: the two strategies that scatter a `BilinearForm`'s values into an
# already-allocated matrix -- the serial record/replay cache (`_assemble_bilinear_core_cached!`)
# and the threaded, band-coloured sweep (`_assemble_bilinear_parallel_core!`). The parallel
# path never uses the sinks in `bilinear_traversal.jl`: it owns its own grid loop so threads
# can be handed disjoint bands, sharing only the entry rule (`_entry_target`) and
# `add_to_sparse!`.

# --- Serial: record once, replay thereafter ---------------------------------------- #
#
# A form's first serial fill is a replay: the coordinate walk (`_form_coordinates`,
# bilinear_pattern.jl) gives every entry's `(row, col)`, which searched in `A` become the
# recorded `nzval` positions, one `Segment` per (term, block) unit, and `_replay_segment!`
# reads them back instead of searching. `allocate_system_matrix` stores that recording for
# the matrix it builds; `_assemble_bilinear_core_cached!` records afresh when `A` is not the
# exact matrix object the form's cache was last built against (gpena/Bramble.jl#26).
#
# The parallel path below is untouched by this cache: it always threads, and recording is an
# inherently serial, one-time pass (concurrent writes into a shared cache would race), so
# caching it would mean its first call silently stopped threading -- breaking
# `assemble_parallel!`'s own documented contract ("always threads").

# Whether a `D`-dimensional form records diagonal segments at all. Only in 1D: from 2D up a
# difference term's interior has no constant per-tap stride (a boundary column between two
# interior rows holds fewer entries), so every such segment came back flat and the diagonal
# replay was compiled per term for nothing. Decided from `D` alone, so the branch folds away.
@inline _diagonal_replay(::Val{1}) = true
@inline _diagonal_replay(::Val) = false

# Attempts to repackage a unit's flat positions as a diagonal `Segment`, restricted to the
# unit's own interior/boundary split (gpena/Bramble.jl#160): the coordinate walk visits the
# interior box before any boundary slab, pushing every interior point's `P` entries in
# `LinearIndices(interior)` order, so `positions[1:n_interior*P]` already *is* the interior
# in that order, with no need to touch `point_ptr` to find it.
#
# Validated against the actual positions, not assumed from the term's own margin
# (gpena/Bramble.jl#161): a form summing terms of different margins into the same block
# records one segment per summand, and a narrower term's own "interior" can still include
# columns whose true `nzval` footprint -- set by every segment sharing that column, not just
# this one -- varies where a wider-margin sibling still adds taps. That shows up here as a
# non-constant per-tap stride; any mismatch, or too little interior to bother, falls back to
# the plain flat segment unchanged, which `ReplaySink` already handles.
#
# Every early return builds a `Segment{D}` via `_flat_segment`, never a bare tuple: the
# return type must stay the single concrete `Segment{D}` on every path, not a `Union` with
# some other flat representation -- that Union is exactly what gpena/Bramble.jl#240 removed.
function _try_diagonal_segment(
        margin::Int, grid_inds::CartesianIndices, point_ptr::Vector{Int}, positions::Vector{Int}
)
    ax = axes(grid_inds)
    D = length(ax)
    _diagonal_replay(Val(D)) || return _flat_segment(Val(D), point_ptr, positions)
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
    return Segment{D}(
        true, boundary_point_ptr, boundary_positions, base, stride, P, interior, Int[]
    )
end

# One term into one block: the walk, a fresh stencil evaluation (weights may be live -- only
# positions are fixed), and each entry's nzval index read from `segment` instead of searched.
# `point_ptr[lin_idx]` addresses each point's own slice of `positions` directly, so this
# stays correct regardless of what order grid points are visited in.
#
# One method branching on `segment.is_diagonal`, not two dispatching on `NzvalSegment` versus
# `DiagonalSegment` (gpena/Bramble.jl#240): both branches are fully concrete (`Segment{D}`
# has one shape, not a Union). Flat replays through `ReplaySink`; diagonal (1D only,
# `_diagonal_replay`) replays the interior core through `DiagonalReplaySink`'s stride
# arithmetic and the boundary shell through an ordinary `ReplaySink`, the two-sink form of
# `visit_bilinear_stencil` running both in the one walk.
function _replay_segment!(
        A::AbstractMatrix,
        term::TERM,
        sp,
        row_offset::Int,
        col_offset::Int,
        segment::Segment{D},
        α
) where {TERM, D}
    if _diagonal_replay(Val(D)) && segment.is_diagonal
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

# `segments::Vector{Segment{D}}` throughout this file: `D` is fixed per call (the form's own
# dimension) and, since gpena/Bramble.jl#240, `Segment{D}` is the single concrete element
# type regardless of whether a given element is flat or diagonal (`is_diagonal` selects the
# shape at the value level, not the type level).
#
# A composite space on either side is assembled block by block, the scalar side (if any) as
# a one-leaf composite. `_walked_leaf` on a mixed pair would pick one whole space and drop the
# component the term names on the composite side. Decided from the types alone, so the branch
# folds away.
@inline _is_block_pair(::Any, ::Any) = false
@inline _is_block_pair(::CompositeGridSpace, ::Any) = true
@inline _is_block_pair(::Any, ::CompositeGridSpace) = true
@inline _is_block_pair(::CompositeGridSpace, ::CompositeGridSpace) = true

# A cache miss: the recording `allocate_system_matrix` would have stored, built against `A`,
# then replayed. Its positions search reports an entry `A`'s pattern cannot hold.
function _record_bilinear_core!(
        A::AbstractMatrix, trial_space, test_space, ast::AST_TYPE, ::Val{D}, α
) where {AST_TYPE, D}
    p = _form_coordinates(trial_space, test_space, ast)
    _coordinates_to_positions!(A, p, ast)
    segments = _segments_from_positions(Val(D), p)
    _replay_bilinear_core!(A, trial_space, test_space, ast, segments, α)
    return segments
end

function _replay_bilinear_core!(
        A::AbstractMatrix, trial_space, test_space, ast::AST_TYPE, segments::Vector{Segment{D}}, α
) where {AST_TYPE, D}
    if _is_block_pair(trial_space, test_space)
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
    bound = _bind_interp_spaces(ast, trial_space, test_space)
    _check_block_meshes(bound, trial_space, test_space)
    sp = _walked_leaf(bound, trial_space, test_space)
    _replay_summands!(A, bound, sp, segments, 0, α)
    return nothing
end

# --- The (term, block) units, in segment order ---------------------------------------- #
#
# `_foreach_unit` (setup: the coordinate walk) and `_replay_summands!`/`_replay_blocks!`
# (every fill) visit the same units in the same order, so segment `k` of the one is unit `k`
# of the other. A sum's terms are walked as `_summands(op)` (`stencil_eval.jl`), not by
# recursing left/right, for the reason given there. Every method is `@noinline`, so each
# term's block loop stays its own method instance instead of inlining into the walk (3D
# `innerₕ(εcₕ(u), εcₕ(v))`, first assemble: 13.0–13.5 s with the barrier, 14.2–14.4 s
# without). One call per term per assembly is what it costs at run time.
#
# A scalar form's top-level sum is one unit per summand, as a composite form's blocks
# already are, never one fused stencil: fused, every grid-point walk inlines the whole sum's
# stencil, and optimising that body grows superlinearly with the term count (3D scalar form
# of N distinct `innerₕ` difference terms: recording inference 0.64 s at N = 9, 11.0 s at
# N = 27; first assemble at N = 27 went from 44.4 s to 16.0 s).
#
# A transposed pair ⟨Au, Bv⟩ + ⟨Bu, Av⟩ among the summands (`_pair_plan`, form/symmetry.jl)
# is one unit, not two: ⟨Au, Bv⟩ is walked once and each entry is also added, with the second
# term's scaling, at its transpose (`_PairReplaySink`). First assemble of a 3D scalar form of
# 9 such pairs: 11.9 s unpaired, 4.9 s paired; 3D `innerₕ(εcₕ(u), εcₕ(v))` (3 pairs among 15
# summands) 14.8 s to 13.0 s, and its warm `assemble!` on 24³ 5.0 ms to 3.65 ms.
#
# `f(term, sp, row_offset, col_offset, dr, dc, half)` is called once per unit: `half` is `-1`
# for a term walked alone, and for a pair's unit `0`, `1` or `2` as `_PairReplaySink` reads
# it, `dr`/`dc` moving an entry from the first term's block to the second's.
function _foreach_unit(f::F, trial_space, test_space, ast) where {F}
    if _is_block_pair(trial_space, test_space)
        _foreach_block_unit(
            f, ast, leaf_spaces_offsets(trial_space), leaf_spaces_offsets(test_space)
        )
        return nothing
    end
    bound = _bind_interp_spaces(ast, trial_space, test_space)
    _check_block_meshes(bound, trial_space, test_space)
    sp = _walked_leaf(bound, trial_space, test_space)
    _foreach_summand_unit(f, bound, sp)
    return nothing
end

@noinline function _foreach_summand_unit(f::F, op::OperatorAdd, sp) where {F}
    _foldl_pairs(
        (_, t) -> _foreach_summand_unit(f, t, sp),
        (_, t1, t2) -> f(_bare_product(t1), sp, 0, 0, 0, 0, 0),
        nothing,
        _summands(op)
    )
    return nothing
end

@noinline function _foreach_summand_unit(f::F, term::TERM, sp) where {F, TERM}
    f(term, sp, 0, 0, 0, 0, -1)
    return nothing
end

@noinline function _foreach_block_unit(
        f::F, op::OperatorAdd, trial_leaves, test_leaves
) where {F}
    _foldl_pairs(
        (_, t) -> _foreach_block_unit(f, t, trial_leaves, test_leaves),
        (_, t1, t2) -> _foreach_pair_block_unit(f, t1, t2, trial_leaves, test_leaves),
        nothing,
        _summands(op)
    )
    return nothing
end

@noinline function _foreach_block_unit(
        f::F, term::TERM, trial_leaves, test_leaves
) where {F, TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        bound = _bind_interp_spaces(term, blk.trial_leaf, blk.test_leaf)
        _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
        sp = _walked_leaf(bound, blk.trial_leaf, blk.test_leaf)
        f(bound, sp, blk.row_offset, blk.col_offset, 0, 0, -1)
    end
    return nothing
end

# The replay counterpart, `next` threaded by value and returned, rather than via a mutable
# `Ref`, so this stays allocation-free: the segment index the *next* unit should consume.
@noinline function _replay_summands!(
        A::AbstractMatrix, op::OperatorAdd, sp, segments::Vector{Segment{D}}, next::Int, α
) where {D}
    return _foldl_pairs(
        (n, t) -> _replay_summands!(A, t, sp, segments, n, α),
        (n, t1, t2) -> _replay_pair!(A, t1, t2, sp, segments, n, α),
        next,
        _summands(op)
    )
end

@noinline function _replay_summands!(
        A::AbstractMatrix, term::TERM, sp, segments::Vector{Segment{D}}, next::Int, α
) where {TERM, D}
    next += 1
    _replay_segment!(A, term, sp, 0, 0, segments[next], α)
    return next
end

@noinline function _replay_blocks!(
        A::AbstractMatrix,
        op::OperatorAdd,
        trial_leaves,
        test_leaves,
        segments::Vector{Segment{D}},
        next::Int,
        α
) where {D}
    return _foldl_pairs(
        (n, t) -> _replay_blocks!(A, t, trial_leaves, test_leaves, segments, n, α),
        (n, t1, t2) -> _replay_pair_blocks!(A, t1, t2, trial_leaves, test_leaves, segments, n, α),
        next,
        _summands(op)
    )
end

@noinline function _replay_blocks!(
        A::AbstractMatrix,
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

# --- Transposed pairs: one kernel for ⟨Au, Bv⟩ + ⟨Bu, Av⟩ ---------------------------- #
#
# `_pair_plan` pairs two summands by type, and only product types whose values carry
# nothing but component indices (`_pairable_type`), so at every grid point of one leaf the
# second term's stencil is the first's with trial and test offsets exchanged. Component
# indices only route a term to its block, and the second term's own blocks give the offsets
# of its entries. On a composite space whose two blocks sit on different leaf objects,
# ⟨Au, Bv⟩ is walked once on each leaf, the first term's half on its own leaf and the
# transposed half on the second's: the same kernel, so nothing new is compiled when the
# leaves share a type, and nothing is dispatched at run time.

@inline function _replay_pair_segment!(
        A::AbstractMatrix, term::TERM, sp, row_offset::Int, col_offset::Int,
        segment::Segment, α1, α2, half::Int
) where {TERM}
    visit_bilinear_stencil(
        _PairReplaySink(
            A, segment.point_ptr, segment.positions, segment.positions_t, α1, α2, half
        ),
        term,
        sp,
        row_offset,
        col_offset
    )
    return nothing
end

@noinline function _replay_pair!(
        A::AbstractMatrix, t1, t2, sp, segments::Vector{Segment{D}}, next::Int, α
) where {D}
    next += 1
    _replay_pair_segment!(
        A, _bare_product(t1), sp, 0, 0, segments[next], α * _term_scale(t1),
        α * _term_scale(t2), 0
    )
    return next
end

# Block `k` of the second term holds the transposes of block `k` of the first. Decided by the
# tuple types, so the fallback folds away.
@inline _pair_blocks_ok(b1::Tuple, b2::Tuple) = length(b1) == length(b2)

# Entry `(row, col)` of the first term's block moves to `(col + dr, row + dc)` in the second's.
@inline _pair_shift(blk, blk2) = (blk2.row_offset - blk.col_offset, blk2.col_offset - blk.row_offset)

@noinline function _foreach_pair_block_unit(
        f::F, t1, t2, trial_leaves, test_leaves
) where {F}
    p1 = _bare_product(t1)
    p2 = _bare_product(t2)
    b1 = blocks(p1, trial_leaves, test_leaves)
    b2 = blocks(p2, trial_leaves, test_leaves)
    if _pair_blocks_ok(b1, b2)
        for (blk, blk2) in map(tuple, b1, b2)
            bound = _bind_interp_spaces(p1, blk.trial_leaf, blk.test_leaf)
            _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
            _check_block_meshes(p2, blk2.trial_leaf, blk2.test_leaf)
            sp = _walked_leaf(bound, blk.trial_leaf, blk.test_leaf)
            dr, dc = _pair_shift(blk, blk2)
            ro, co = blk.row_offset, blk.col_offset
            if sp === blk2.test_leaf
                f(bound, sp, ro, co, dr, dc, 0)
            else
                f(bound, sp, ro, co, dr, dc, 1)
                f(bound, blk2.test_leaf, ro, co, dr, dc, 2)
            end
        end
    else
        Base.inferencebarrier(_foreach_block_unit)(f, t1, trial_leaves, test_leaves)
        Base.inferencebarrier(_foreach_block_unit)(f, t2, trial_leaves, test_leaves)
    end
    return nothing
end

@noinline function _replay_pair_blocks!(
        A::AbstractMatrix,
        t1,
        t2,
        trial_leaves,
        test_leaves,
        segments::Vector{Segment{D}},
        next::Int,
        α
) where {D}
    p1 = _bare_product(t1)
    p2 = _bare_product(t2)
    b1 = blocks(p1, trial_leaves, test_leaves)
    b2 = blocks(p2, trial_leaves, test_leaves)
    if _pair_blocks_ok(b1, b2)
        α1 = α * _term_scale(t1)
        α2 = α * _term_scale(t2)
        for (blk, blk2) in map(tuple, b1, b2)
            bound = _bind_interp_spaces(p1, blk.trial_leaf, blk.test_leaf)
            _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
            sp = _walked_leaf(bound, blk.trial_leaf, blk.test_leaf)
            ro, co = blk.row_offset, blk.col_offset
            if sp === blk2.test_leaf
                next += 1
                _replay_pair_segment!(A, bound, sp, ro, co, segments[next], α1, α2, 0)
            else
                _replay_pair_segment!(A, bound, sp, ro, co, segments[next + 1], α1, α2, 1)
                _replay_pair_segment!(
                    A, bound, blk2.test_leaf, ro, co, segments[next + 2], α1, α2, 2
                )
                next += 2
            end
        end
        return next
    end
    next = Base.inferencebarrier(_replay_blocks!)(
        A, t1, trial_leaves, test_leaves, segments, next, α
    )::Int
    return Base.inferencebarrier(_replay_blocks!)(
        A, t2, trial_leaves, test_leaves, segments, next, α
    )::Int
end

# Picks recording (cache miss: a fresh `A`, a changed `ast` -- e.g. `assemble!(A, form;
# ast = ...)` given something other than `form.ast` -- or the first call ever) or replay
# (cache hit) and keeps `cache` in step with whichever one ran. `ast` is checked as well as
# `A`: the cache's positions are only valid for the exact stencil shape they were recorded
# against, and a different `ast` can visit a different number of entries per point.
function _assemble_bilinear_core_cached!(
        A::AbstractMatrix,
        trial_space,
        test_space,
        ast::AST_TYPE,
        cache::_AssemblyCache{D, CACHED_AST},
        α = true
) where {AST_TYPE, D, CACHED_AST}
    if cache.valid && cache.A_id === objectid(A) && cache.ast === ast
        _replay_bilinear_core!(A, trial_space, test_space, ast, cache.segments, α)
    else
        # A fresh vector, never `empty!` on whatever `cache.segments` currently references.
        #
        # Replayed with `α`: the cache is keyed on `(A, ast)` alone, never on `α`, since
        # nzval *positions* never depend on it: an `assemble_add!` caller is free to change
        # `α` (a `Ref`'s current value, say) on every call and still replay from cache.
        segments = _record_bilinear_core!(A, trial_space, test_space, ast, Val(D), α)
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
#
# No device-specific parameter here (gpena/Bramble.jl#313): `add_to_sparse!` reads a device
# matrix's own `mirror` field off `A` directly, so this function -- and every sweep function
# between it and `_assemble_bilinear_parallel_core!` -- carries the same signature whether
# `A` is host- or device-resident.
@inline function _scatter_point!(
        A::AbstractMatrix,
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

# One colour, threaded, writing directly into the matrix. Dispatches on the *effective*
# execution policy (`_sweep_bilinear!` computes it): `CpuThreaded` keeps `Threads.@threads`
# exactly as before; `CpuPolyester` reaches its own hook instead, so it never silently threads
# with the wrong mechanism (gpena/Bramble.jl#190). `CpuSerial` never reaches this function --
# `_effective_parallel_policy` only ever hands it `CpuThreaded` or `CpuPolyester`.
@noinline function _sweep_bilinear_colour!(
        ::CpuThreaded,
        A::AbstractMatrix,
        sp,
        term::TERM,
        idxs,
        lin_indices,
        mesh_markers,
        row_offset::Int,
        col_offset::Int,
        α
) where {TERM}
    Threads.@threads :static for I in idxs
        _scatter_point!(A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α)
    end
    return nothing
end

@noinline function _sweep_bilinear_colour!(
        ::CpuPolyester,
        A::AbstractMatrix,
        sp,
        term::TERM,
        idxs,
        lin_indices,
        mesh_markers,
        row_offset::Int,
        col_offset::Int,
        α
) where {TERM}
    return _batch_bilinear_colour_sweep!(
        A, sp, term, idxs, lin_indices, mesh_markers, row_offset, col_offset, α
    )
end

"""
    _batch_bilinear_colour_sweep!(A, sp, term, idxs, lin_indices, mesh_markers, row_offset, col_offset, α) -> Nothing

[`CpuPolyester`](@ref)'s counterpart of the `Threads.@threads` body in
`_sweep_bilinear_colour!`, filled by `BramblePolyesterExt` (gpena/Bramble.jl#190).
The only `src/` method errors naming Polyester.
"""
@noinline function _batch_bilinear_colour_sweep!(
        A, sp, term, idxs, lin_indices, mesh_markers, row_offset, col_offset, α
)
    return _throw_cpubatch_without_polyester(:_batch_bilinear_colour_sweep!)
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
        ::CpuThreaded,
        A::AbstractMatrix,
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
    Threads.@threads :static for b in bidx
        for I in CartesianIndices((rest..., _band_range(ax, nbands, b)))
            _scatter_point!(
                A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α
            )
        end
    end
    return nothing
end

@noinline function _sweep_band_colour!(
        ::CpuPolyester,
        A::AbstractMatrix,
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
    return _batch_bilinear_band_sweep!(
        A, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, row_offset, col_offset, α
    )
end

"""
    _batch_bilinear_band_sweep!(A, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, row_offset, col_offset, α) -> Nothing

[`CpuPolyester`](@ref)'s counterpart of the `Threads.@threads` body in
[`_sweep_band_colour!`](@ref), filled by `BramblePolyesterExt` (gpena/Bramble.jl#190). The
only `src/` method errors naming Polyester.
"""
@noinline function _batch_bilinear_band_sweep!(
        A, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, row_offset, col_offset, α
)
    return _throw_cpubatch_without_polyester(:_batch_bilinear_band_sweep!)
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
        A::AbstractMatrix, sp, term::TERM, row_offset::Int, col_offset::Int, α = true
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

# Every colour in turn, using strided subgrids. `policy` is the *effective* policy
# (`_effective_parallel_policy(sp)`, computed once here): `CpuSerial` is coerced to
# `CpuThreaded` since every call into this function is already on the forced-threaded path
# (`_assemble_bilinear_parallel_core!`, always entered from a non-`CpuSerial` branch, or from
# `assemble_parallel!`'s own "regardless of policy" contract); `CpuPolyester` passes through
# unchanged so the colour/band sweeps below reach their own hook instead of `Threads.@threads`
# (gpena/Bramble.jl#190).
function _sweep_bilinear!(
        A::AbstractMatrix, sp, term::TERM, strides, row_offset::Int, col_offset::Int, α = true
) where {TERM}
    Ωₕ = mesh(sp)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    mesh_markers = markers(Ωₕ)
    policy = _effective_parallel_policy(sp)

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
                policy,
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
            policy, A, sp, term, grid_inds, lin_indices, mesh_markers, row_offset, col_offset, α
        )
        return A
    end

    for c in CartesianIndices(strides)
        _sweep_bilinear_colour!(
            policy,
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
        A::AbstractMatrix, op::OperatorAdd, trial_leaves, test_leaves, α = true
)
    return _visit_operator_add2(
        _assemble_blocks_parallel!, A, op, trial_leaves, test_leaves, α
    )
end

function _assemble_blocks_parallel!(
        A::AbstractMatrix, term::TERM, trial_leaves, test_leaves, α = true
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        bound = _bind_interp_spaces(term, blk.trial_leaf, blk.test_leaf)
        _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
        # `host_weights` (gpena/Bramble.jl#94, S4.2): the mirror of S4.0's own fix to the
        # *pattern* walk (`bilinear_pattern.jl`), now applied to the walk that *refills* the
        # matrix. `local_stencil` below reads this leaf's weights and its mesh's spacings one
        # grid point at a time (`compute_weight`, `ast/operators/inner.jl`); on a device
        # backend those are `MtlVector`s, and reading them element-by-element is exactly the
        # `Scalar indexing is disallowed` this milestone has hit three times already
        # (`_probe_point` -> S2.3, `spacing`/`forward_spacing` -> S2.10, the pattern walk's own
        # `SeparableWeights.__prod` -> S4.0). A no-op on a host leaf (`host_weights(Wc) === Wc`).
        sp = host_weights(_walked_leaf(bound, blk.trial_leaf, blk.test_leaf))
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

# Matrix-type generic since S7.1 (gpena/Bramble.jl#190): the band-coloured sweep above
# (`_scatter_point!`/`_sweep_bilinear!`/`_assemble_blocks_parallel!`) only ever calls
# `add_to_sparse!`, itself matrix-type generic since S1.1 (`bilinear_traversal.jl`), so
# nothing here races any differently for a dense `Matrix` than for `SparseMatrixCSC` --
# verified equal to the serial record/replay pass on a dense backend (S7.1's own check).
# `_sweep_bilinear!` reads the space's effective policy itself (`_effective_parallel_policy`)
# and only `Threads.@threads`es under `CpuThreaded`; a `CpuPolyester` backend reaches its own
# hook instead of silently threading with the wrong mechanism, so this function no longer
# needs a separate non-threading fallback for a matrix type it cannot thread.
function _assemble_bilinear_parallel_core!(
        A::AbstractMatrix, trial_space, test_space, ast::AST_TYPE, α = true
) where {AST_TYPE}
    if _is_block_pair(trial_space, test_space)
        # Every leaf block `_assemble_blocks_parallel!` walks below gets its own
        # `host_weights` leaf (different leaves can sit on different meshes), but they all
        # scatter into the same `A`, and therefore the same `A.mirror` -- `add_to_sparse!`
        # reads it off `A` directly, so nothing needs resolving or threading through here
        # (gpena/Bramble.jl#313).
        _assemble_blocks_parallel!(
            A, ast, leaf_spaces_offsets(trial_space), leaf_spaces_offsets(test_space), α
        )
        _flush_device_scatter!(A)
        return A
    end
    bound = _bind_interp_spaces(ast, trial_space, test_space)
    _check_block_meshes(bound, trial_space, test_space)
    # `host_weights` (gpena/Bramble.jl#94, S4.2) -- see `_assemble_blocks_parallel!` above for why.
    sp = host_weights(_walked_leaf(bound, trial_space, test_space))
    if _has_test_interp(bound)
        _sweep_bilinear_serial!(A, sp, bound, 0, 0, α)
    else
        _sweep_bilinear!(A, sp, bound, _colour_strides(stencil_offsets(bound)), 0, 0, α)
    end
    # A no-op for a host matrix; for a device one, copies `A`'s own mirror (gpena/Bramble.jl#313)
    # back across in one bulk `copyto!` -- see `_flush_device_scatter!`'s own docstring
    # (`bilinear_traversal.jl`).
    _flush_device_scatter!(A)
    return A
end
