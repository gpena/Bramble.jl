# bilinear_traversal.jl: one shared walk over a term's local stencils
# (`visit_bilinear_stencil`), and the sinks that plug into it (`PatternSink`, `RecordSink`,
# `ReplaySink`). Kept as a single file deliberately: the walk and its sinks are a tightly
# coupled design for guaranteeing allocation-free inner loops (gpena/Bramble.jl#50), and
# splitting them further would scatter that coupling across files without a readability
# gain (gpena/Bramble.jl#116).

# --- Utility helpers -------------------------------------------------------------- #

# The nzval index storing (row, col) in A, or 0 if it names no stored entry. A linear scan
# of the column when it holds few entries, a binary search otherwise -- both rely on
# `SparseMatrixCSC`'s own invariant that `rowval` is sorted within each column.
@inline function _find_nzval_position(A::SparseMatrixCSC, row::Int, col::Int)
    p1 = A.colptr[col]
    p2 = A.colptr[col + 1] - 1

    if (p2 - p1) < 32
        idx = p1
        @inbounds while idx <= p2
            A.rowval[idx] == row && return idx
            idx += 1
        end
    else
        lo = p1
        hi = p2
        @inbounds while lo <= hi
            mid = (lo + hi) >>> 1
            mid_row = A.rowval[mid]
            if mid_row < row
                lo = mid + 1
            elseif mid_row > row
                hi = mid - 1
            else
                return mid
            end
        end
    end
    return 0
end

"""
    add_to_sparse!(A::SparseMatrixCSC, row::Int, col::Int, val::Number, term) -> Nothing

Add `val` to `A[row, col]`, which the preallocated sparsity pattern is required to contain.

# Throws
- `ArgumentError`: `(row, col)` is not a stored entry of `A`.

This used to return quietly on a missing entry, which let a matrix whose pattern cannot hold
the form assemble to a plausible wrong answer instead of failing. That is how a term naming
both components of a composite space once vanished without a word: the pattern held the
diagonal blocks only, so every off-diagonal contribution was discarded. The regression test
for that is `test/form/bilinear.jl`, "Composite blocks".

`term` is carried only to name the offending node in the message, and read only on the
branch that throws.

See also: [`allocate_system_matrix`](@ref) and [`RecordSink`](@ref), which raises the same
way on the serial recording pass.
"""
@inline function add_to_sparse!(A::SparseMatrixCSC, row::Int, col::Int, val::Number, term)
    pos = _find_nzval_position(A, row, col)
    pos == 0 && _throw_missing_pattern_entry(term)
    @inbounds A.nzval[pos] += val
    return nothing
end

# Whether an earlier entry of this stencil already named the same pair of offsets.
@inline function _offsets_seen_before(stencil, k::Int, off_u, off_v)
    @inbounds for l in 1:(k - 1)
        stencil[l][1] == off_u && stencil[l][2] == off_v && return true
    end
    return false
end

"""
    _trial_column(lin_indices, I::CartesianIndex, off_u) -> Int

Which column of the trial block a stencil entry's trial slot names, or `0` for none.

An ordinary offset is bounds-checked against the grid and answers `0` on a boundary, so the
entry is dropped. An interpolation entry carries an [`AbsoluteColumn`](@ref) instead, naming
a source column outright, because the trial degrees of freedom it reaches live on a
different mesh and which ones depends on where the point falls.

See also: [`_entry_target`](@ref).
"""
Base.@propagate_inbounds function _trial_column(lin_indices, I::CartesianIndex, off_u)
    Iu = I + CartesianIndex(off_u)
    return checkbounds(Bool, lin_indices, Iu) ? lin_indices[Iu] : 0
end

@inline _trial_column(lin_indices, I::CartesianIndex, off_u::AbsoluteColumn) = off_u.col

"""
    _trial_inbounds(lin_indices, I::CartesianIndex, off_u) -> Bool

Whether `off_u`'s trial index survives, without computing the linear index
[`_trial_column`](@ref) would return -- for sinks that only need to know whether the entry
lands, not where ([`_sink_needs_coordinates`](@ref)). An [`AbsoluteColumn`](@ref) always
survives, matching `_trial_column`: it names a source column directly and is never `0`.
"""
Base.@propagate_inbounds function _trial_inbounds(lin_indices, I::CartesianIndex, off_u)
    Iu = I + CartesianIndex(off_u)
    return checkbounds(Bool, lin_indices, Iu)
end
@inline _trial_inbounds(lin_indices, I::CartesianIndex, off_u::AbsoluteColumn) = true

"""
    _trial_column_unguarded(lin_indices, I::CartesianIndex, off_u) -> Int

[`_trial_column`](@ref), without the bounds check.

Only called from the interior core of [`visit_bilinear_stencil`](@ref), where
[`_stencil_margin`](@ref) has already guaranteed `I + CartesianIndex(off_u)` lands inside
`lin_indices` for every entry the term's stencil can produce. An `AbsoluteColumn` names a
source column outright either way, matching `_trial_column`.
"""
Base.@propagate_inbounds _trial_column_unguarded(lin_indices, I::CartesianIndex, off_u) =
    lin_indices[I + CartesianIndex(off_u)]
@inline _trial_column_unguarded(lin_indices, I::CartesianIndex, off_u::AbsoluteColumn) =
    off_u.col

# --- one traversal, pluggable sinks (gpena/Bramble.jl#50) -------------------------- #

#=
Five sweeps used to re-derive this same walk: `mesh` -> `markers` -> `LinearIndices`, loop
the grid, evaluate `local_stencil`, decide each entry's (row, col), guard it, act. Two of
them recorded the sparsity pattern and three wrote values into it, and the property that
matters -- *every (row, col) the scatter touches is present in the pattern* -- held only
because two independently written traversals agreed.

The walk is now written once. A sink says what to do per entry, and declares by dispatch
whether it wants the de-duplication the pattern passes need and the value passes must not
have (repeated offsets accumulate).

Every rule that used to live in five places lives here: which entries are dropped
(out-of-range rows, and `_trial_column` answering 0), that `off_u` may be an
`AbsoluteColumn` while `off_v` never is, and that the fifth argument to `local_stencil` is
the *leaf's* linear index -- the wrong leaf reads the wrong `SourceVector`, silently.
=#

"""
    _sink_dedups(sink) -> Bool

Whether `sink` wants repeated `(off_u, off_v)` pairs within one point's stencil collapsed
to a single entry.

The sparsity pattern wants each `(row, col)` once however many stencil entries name it, and
a value sink wants every one of them, because repeated entries accumulate. Getting this
backwards gives a wrong matrix rather than an error, so it is answered by dispatch on the
sink type: the default is `false`, and the branch folds away at compile time, leaving the
value sweeps with no de-duplication scan at all.

See also: [`visit_bilinear_stencil`](@ref), [`PatternSink`](@ref).
"""
@inline _sink_dedups(::Any) = false

"""
    _sink_needs_coordinates(sink) -> Bool

Whether `sink` reads the `(row, col)` an entry computes, rather than discarding it.

The default is `true`. [`ReplaySink`](@ref) answers `false`: its [`_sink_entry!`](@ref)
ignores `row` and `col` entirely, since the `nzval` position is already recorded in
`sink.positions`. `false` lets [`_step_entry!`](@ref) skip the linear-index lookups and
offset additions that would only be discarded, while still running the bounds checks that
decide whether the entry survives at all -- dropped entries must match the record pass
exactly, or `slot` drifts out of step with `positions`.

See also: [`_step_entry!`](@ref), [`_trial_inbounds`](@ref).
"""
@inline _sink_needs_coordinates(::Any) = true

"""
    _sink_point!(sink, lin_idx::Int) -> Int

Announce grid point `lin_idx` to `sink`, and answer the base slot for its entries.

The default answers `0`. [`RecordSink`](@ref) uses the call to open that point's slice of
the position list; [`ReplaySink`](@ref) answers the start of that slice, which the traversal
then adds the entry ordinal to. Addressing each point from its own base is what lets a
replay stay correct regardless of the order grid points are visited in, without any sink
having to carry a mutable cursor.

See also: [`visit_bilinear_stencil`](@ref), [`_sink_entry!`](@ref).
"""
@inline _sink_point!(::Any, ::Int) = 0

"""
    _entry_target(lin_indices, I::CartesianIndex, off_u, off_v,
                  row_offset::Int, col_offset::Int) -> Tuple{Int,Int}

The matrix position a stencil entry writes to, or `(0, 0)` when it writes nowhere.

The row comes from the test offset `off_v` and is dropped when it leaves the grid. The
column comes from [`_trial_column`](@ref), which answers `0` for a trial offset outside the
grid and reads an [`AbsoluteColumn`](@ref) directly, since an interpolation entry names its
source column rather than an offset from `I`. Both are then shifted into the block by
`row_offset` and `col_offset`.

`(0, 0)` is a sentinel rather than `nothing` so the return type stays concrete on the
assembly hot path.

# Returns
- `Tuple{Int,Int}`: The `(row, col)` to write, or `(0, 0)` to skip the entry.
"""
Base.@propagate_inbounds function _entry_target(
    lin_indices, I::CartesianIndex, off_u, off_v, row_offset::Int, col_offset::Int
)
    Iv = I + CartesianIndex(off_v)
    checkbounds(Bool, lin_indices, Iv) || return (0, 0)
    col = _trial_column(lin_indices, I, off_u)
    col == 0 && return (0, 0)
    return (lin_indices[Iv] + row_offset, col + col_offset)
end

"""
    _step_entry!(sink, lin_indices, I, off_u, off_v, weight, row_offset::Int, col_offset::Int, slot::Int) -> Bool

One entry: guard it, and hand it to the sink if it lands inside. Answers whether it did, so
the caller can advance the slot.

Fused rather than "compute the target, then act on it" ([`_entry_target`](@ref), which the
tests use and `_scatter_point!` shares) because returning a `(row, col)` sentinel tuple has to
be merged from three return points: measured against the hand-written loop it cost 5 extra
phi nodes, 4 integer adds and 3 comparisons per entry, with identical loads, stores and calls.
"""
Base.@propagate_inbounds function _step_entry!(
    sink::SINK,
    lin_indices,
    I,
    off_u,
    off_v,
    weight,
    row_offset::Int,
    col_offset::Int,
    slot::Int,
) where {SINK}
    Iv = I + CartesianIndex(off_v)
    checkbounds(Bool, lin_indices, Iv) || return false
    if _sink_needs_coordinates(sink)
        col = _trial_column(lin_indices, I, off_u)
        col == 0 && return false
        _sink_entry!(sink, lin_indices[Iv] + row_offset, col + col_offset, weight, slot)
    else
        _trial_inbounds(lin_indices, I, off_u) || return false
        _sink_entry!(sink, 0, 0, weight, slot)
    end
    return true
end

"""
    _step_entry_unguarded!(sink, lin_indices, I, off_u, off_v, weight, row_offset::Int, col_offset::Int, slot::Int) -> Nothing

[`_step_entry!`](@ref), without the guard: the interior-core counterpart, called only where
[`_stencil_margin`](@ref) already guarantees every offset lands inside the grid. Always
accepts its entry, so it has nothing to answer back and the caller advances `slot`
unconditionally rather than being told to.
"""
Base.@propagate_inbounds function _step_entry_unguarded!(
    sink::SINK,
    lin_indices,
    I,
    off_u,
    off_v,
    weight,
    row_offset::Int,
    col_offset::Int,
    slot::Int,
) where {SINK}
    if _sink_needs_coordinates(sink)
        Iv = I + CartesianIndex(off_v)
        col = _trial_column_unguarded(lin_indices, I, off_u)
        _sink_entry!(sink, lin_indices[Iv] + row_offset, col + col_offset, weight, slot)
    else
        _sink_entry!(sink, 0, 0, weight, slot)
    end
    return nothing
end

# One point's entries, in two forms chosen by the de-duplication trait.
#
# A de-duplicating sink needs each entry's index, to ask `_offsets_seen_before` about the
# ones before it, so its loop runs over `eachindex`. A value sink does not, and gets plain
# iteration over the stencil tuple instead: measured, indexing a heterogeneous tuple by a
# loop variable costs 8% on the stiffness replay and 17% on the mass one, at every problem
# size from 64^2 to 1024^2, because it defeats the unrolling direct iteration gets. Both
# forms are selected at compile time, so neither carries the other's cost.
Base.@propagate_inbounds function _visit_entries(
    sink::SINK, stencil, lin_indices, I, row_offset::Int, col_offset::Int, slot::Int
) where {SINK}
    if _sink_dedups(sink)
        for k in eachindex(stencil)
            off_u, off_v, weight = stencil[k]
            _offsets_seen_before(stencil, k, off_u, off_v) && continue
            _step_entry!(
                sink, lin_indices, I, off_u, off_v, weight, row_offset, col_offset, slot
            ) && (slot += 1)
        end
    else
        for (off_u, off_v, weight) in stencil
            _step_entry!(
                sink, lin_indices, I, off_u, off_v, weight, row_offset, col_offset, slot
            ) && (slot += 1)
        end
    end
    return nothing
end

# The interior-core counterpart of `_visit_entries`: every entry survives, so `slot`
# advances unconditionally instead of on `_step_entry_unguarded!`'s answer -- it has none to
# give. Kept as its own loop rather than branching inside `_visit_entries` per entry, which
# would reintroduce exactly the per-entry branch peeling exists to remove.
Base.@propagate_inbounds function _visit_entries_unguarded(
    sink::SINK, stencil, lin_indices, I, row_offset::Int, col_offset::Int, slot::Int
) where {SINK}
    if _sink_dedups(sink)
        for k in eachindex(stencil)
            off_u, off_v, weight = stencil[k]
            _offsets_seen_before(stencil, k, off_u, off_v) && continue
            _step_entry_unguarded!(
                sink, lin_indices, I, off_u, off_v, weight, row_offset, col_offset, slot
            )
            slot += 1
        end
    else
        for (off_u, off_v, weight) in stencil
            _step_entry_unguarded!(
                sink, lin_indices, I, off_u, off_v, weight, row_offset, col_offset, slot
            )
            slot += 1
        end
    end
    return nothing
end

# --- interior/boundary geometry (gpena/Bramble.jl#160) ----------------------------- #
#
# `visit_bilinear_stencil` used to guard every point alike: a `checkbounds` and a
# `_trial_column` lookup per entry, everywhere, even though on a 200x200 grid 98% of points
# (every one at least `_stencil_margin(term)` cells from every face) can never fail either
# guard. That thrashes the branch predictor and stops LLVM from unrolling the inner
# accumulation, for a check whose answer is "yes" almost everywhere it runs.
#
# The fix is geometric, not a per-point shortcut: split `indices(Ωₕ)` into a rectangular
# interior box, `margin` cells in from every face, where every offset the term can produce is
# statically guaranteed in bounds, and a boundary shell outside it, where the guarded path
# from before still runs unchanged. `margin` is `_stencil_margin(term)` (`stencil_pattern.jl`)
# -- the widest offset the term's whole AST can reach, trial and test sides combined, not a
# hardcoded 1: a nested difference or a multi-cell `Shift` reaches further than a single tap,
# and treating a 1-cell rim as always safe for those would silently corrupt the boundary rows
# instead of merely running slower.
#
# The shell is walked as `2D` axis-aligned slabs, one per `(dimension, low/high face)` pair.
# Slab `2d-1`/`2d` covers dimension `d`'s low/high margin, at the box's full extent in every
# dimension not yet peeled and the box's own interior range in every dimension an
# earlier-numbered slab already claimed. That is what keeps a corner from being visited
# twice: dimension 1's slabs claim the corner outright, dimension 2's slabs skip the range
# dimension 1 already spoke for, and so on -- together with the interior box, the slabs cover
# `indices(Ωₕ)` exactly once.
#
# Peeling only happens when it is geometrically sound: `_peelable` requires every axis to be
# at least `2 * margin` points wide, or a term's own margin could make its low and high rim
# overlap (a 3-point axis with a margin-2 term has no way to keep them apart) and a shared
# point would be scattered into twice. A term or grid that fails this falls back to the one
# guarded loop over the whole grid this file always ran before -- unchanged behaviour, not a
# regression, for exactly the sizes where peeling would not be safe.

@inline _interior_range(r::AbstractUnitRange{Int}, margin::Int) =
    (first(r) + margin):(last(r) - margin)
@inline _low_rim(r::AbstractUnitRange{Int}, margin::Int) = first(r):(first(r) + margin - 1)
@inline _high_rim(r::AbstractUnitRange{Int}, margin::Int) = (last(r) - margin + 1):last(r)
# The unrestricted case in `_boundary_shell_slabs` below, normalized to the same
# `UnitRange{Int}` the three range-builders above already return: `ax[k]` on its own is
# whatever concrete range type the mesh's `CartesianIndices` axes happen to carry (typically
# `Base.OneTo{Int}`), and mixing that with `UnitRange{Int}` across an `ntuple`'s branches
# makes the tuple's element type a `Union` -- which is exactly what made the first version of
# this function allocate 672 B per call instead of 0.
@inline _full_range(r::AbstractUnitRange{Int}) = first(r):last(r)

@inline _peelable(ax::NTuple{D,AbstractUnitRange{Int}}, margin::Int) where {D} =
    all(r -> 2 * margin <= length(r), ax)

# The `2D` non-overlapping slabs partitioning `indices(Ωₕ)`'s rim, described above. `Val(2D)`
# and the inner `Val(D)` are both resolved from `term`/`sp`'s own type parameters, so both
# `ntuple`s unroll at compile time -- no closure captures a runtime dimension count the way
# `_define_vectorial_alias`'s comment (`space/operators/stencil.jl`) warns a `Val(i)` built
# from a loop variable would.
@inline function _boundary_shell_slabs(
    ax::NTuple{D,AbstractUnitRange{Int}}, margin::Int
) where {D}
    return ntuple(Val(2D)) do s
        d = (s + 1) >>> 1
        ranges = ntuple(Val(D)) do k
            if k < d
                _interior_range(ax[k], margin)
            elseif k > d
                _full_range(ax[k])
            else
                isodd(s) ? _low_rim(ax[k], margin) : _high_rim(ax[k], margin)
            end
        end
        CartesianIndices(ranges)
    end
end

# One point's stencil, guarded -- shared by the whole-grid fallback and every boundary slab.
@inline function _visit_guarded_region!(
    sink::SINK,
    term::TERM,
    sp,
    mesh_markers,
    lin_indices,
    region,
    row_offset::Int,
    col_offset::Int,
) where {SINK,TERM}
    @inbounds for I in region
        lin_idx = lin_indices[I]
        stencil = local_stencil(term, sp, I, mesh_markers, lin_idx)
        slot = _sink_point!(sink, lin_idx)
        _visit_entries(sink, stencil, lin_indices, I, row_offset, col_offset, slot)
    end
    return nothing
end

# One point's stencil, unguarded -- the interior core, safe only where `_peelable` held.
@inline function _visit_interior!(
    sink::SINK,
    term::TERM,
    sp,
    mesh_markers,
    lin_indices,
    interior,
    row_offset::Int,
    col_offset::Int,
) where {SINK,TERM}
    @inbounds for I in interior
        lin_idx = lin_indices[I]
        stencil = local_stencil(term, sp, I, mesh_markers, lin_idx)
        slot = _sink_point!(sink, lin_idx)
        _visit_entries_unguarded(
            sink, stencil, lin_indices, I, row_offset, col_offset, slot
        )
    end
    return nothing
end

"""
    visit_bilinear_stencil(sink, term, sp, row_offset::Int, col_offset::Int) -> sink

Walk every grid point of `sp`, evaluate `term`'s local stencil there, and hand each entry
that lands inside the matrix to `sink`.

The walk that five separate sweeps used to re-derive: mesh, markers, linear indices, the
loop over grid points, the stencil evaluation, and the decision of where each entry lands
([`_entry_target`](@ref)). A sink supplies only what to do with an entry, so the pattern
passes and the value passes share one traversal instead of agreeing by coincidence.

The fifth argument handed to `local_stencil` is the *leaf's* linear index. A term routed to
the wrong leaf reads the wrong `SourceVector` without complaint, which is why the index is
derived here rather than by each caller.

Splits into an interior core and a boundary shell when [`_stencil_margin`](@ref) and the
grid size allow it (see the comment above `_peelable`), so most points skip the bounds
guard entirely; every point still gets exactly one visit either way, so `sink` sees the same
set of entries regardless of which path ran, in a possibly different order.
[`RecordSink`](@ref)/[`ReplaySink`](@ref) are unaffected by that: they address each point by
its own linear index, not by visit order (see their docstrings).

# Arguments
- `sink`: What to do per entry. See [`PatternSink`](@ref), [`RecordSink`](@ref) and
  [`ReplaySink`](@ref), and the contract in [`_sink_entry!`](@ref),
  [`_sink_point!`](@ref) and [`_sink_dedups`](@ref).
- `term`: The AST node whose stencil is evaluated at each point.
- `sp`: The test leaf whose grid is walked and whose markers the stencil sees.
- `row_offset`, `col_offset`: The block's origin in the assembled matrix, `0` for a scalar
  space.

# Returns
- `sink`: The same sink, so a caller can read what it collected.

See also: [`allocate_system_matrix`](@ref), [`add_to_sparse!`](@ref).
"""
@inline function visit_bilinear_stencil(
    sink::SINK, term::TERM, sp, row_offset::Int, col_offset::Int
) where {SINK,TERM}
    Ωₕ = mesh(sp)
    mesh_markers = markers(Ωₕ)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    margin = _stencil_margin(term)
    ax = axes(grid_inds)

    # `@inbounds` here (and inside `_visit_guarded_region!`/`_visit_interior!`) is what the
    # hand-written sweeps had wrapping their whole nested loop. It does not cross a function
    # call on its own, so `_visit_entries`, `_visit_entries_unguarded`, `_entry_target` and
    # `_trial_column` are `Base.@propagate_inbounds` to inherit it -- without that, the bounds
    # checks come back and the replay path costs 8-17% more.
    if _peelable(ax, margin)
        interior = CartesianIndices(map(r -> _interior_range(r, margin), ax))
        _visit_interior!(
            sink, term, sp, mesh_markers, lin_indices, interior, row_offset, col_offset
        )
        @inbounds for slab in _boundary_shell_slabs(ax, margin)
            _visit_guarded_region!(
                sink, term, sp, mesh_markers, lin_indices, slab, row_offset, col_offset
            )
        end
    else
        _visit_guarded_region!(
            sink, term, sp, mesh_markers, lin_indices, grid_inds, row_offset, col_offset
        )
    end
    return sink
end

# --- the sinks --------------------------------------------------------------------- #

# What sharing the traversal costs here, measured rather than assumed (gpena/Bramble.jl#50).
# Against the hand-written walk this replaced, on a 300x300 grid with
# `innerₕ(D₋ₓ(u), D₋ₓ(v))` -- the form `benchmark/benchmarks.jl`'s
# `forms / allocate_system_matrix 2D` uses:
#
#   entries pushed  359,100 either way        resulting nnz  269,400 either way
#   the walk        1268 us -> 1398 us        the whole call 2879 us -> 3286 us
#
# So `sparse!` does identical work and the traversal itself is about 10% slower, on forms
# whose stencils carry several distinct offset pairs per point; `innerₕ(u, v)` came out
# faster and `inner₊(∇₋ₕ(u), ∇₋ₕ(v))` unchanged. The enclosing function's LLVM is
# near-identical, so the difference is a codegen subtlety that was not localised further.
#
# Accepted deliberately. `allocate_system_matrix` runs once per form, and the per-refill
# path (`RecordSink`/`ReplaySink`) is neutral, so this is a one-time setup cost rather than
# something a time loop pays -- a cheap place to buy pattern and scatter sharing one walk,
# which is what makes "every scattered entry is in the pattern" assertable at all. Reverting
# just this sink would recover it and cost that.
"""
    PatternSink(I_vec::Vector{Int}, J_vec::Vector{Int})

Collect the `(row, col)` coordinates a term can reach, for building a sparsity pattern.

Appends each coordinate to `I_vec` and `J_vec`, which [`allocate_system_matrix`](@ref) then
hands to `sparse!`. The only sink that de-duplicates ([`_sink_dedups`](@ref)): a coordinate
named twice by one point's stencil is one entry of the pattern, and the weights it carries
are not read here at all.

See also: [`visit_bilinear_stencil`](@ref), [`RecordSink`](@ref).
"""
struct PatternSink
    I_vec::Vector{Int}
    J_vec::Vector{Int}
end
@inline _sink_dedups(::PatternSink) = true

"""
    _sink_entry!(sink, row::Int, col::Int, weight, slot::Int) -> Nothing

Act on one stencil entry landing at `(row, col)` with coefficient `weight`.

The one method each sink has to supply. Called by [`visit_bilinear_stencil`](@ref) only for
entries that land inside the matrix, so a sink never has to guard the index itself. `slot`
counts accepted entries from this point's base ([`_sink_point!`](@ref)) and matters only to
[`ReplaySink`](@ref); the others ignore it.
"""
@inline function _sink_entry!(sink::PatternSink, row::Int, col::Int, _, ::Int)
    push!(sink.I_vec, row)
    push!(sink.J_vec, col)
    return nothing
end

# Kept beside `RecordSink` below, its only caller: this pass builds the replay cache, so a
# pattern that cannot hold the term is reported rather than skipped, as `add_to_sparse!`
# (used by the threaded path in `bilinear_execution.jl`) already does.
@noinline function _throw_missing_pattern_entry(term)
    throw(
        ArgumentError(
            "assembling $(typeof(term)) reached a matrix entry outside its preallocated " *
            "sparsity pattern. `A` was not built by `allocate_system_matrix` for this exact " *
            "form, or the form's `ast` changed after `A` was built.",
        ),
    )
end

"""
    RecordSink(A::SparseMatrixCSC, term, point_ptr::Vector{Int}, positions::Vector{Int})

Add a term's values to `A` and record where each entry landed, building the replay cache.

For each entry it searches `A` for the `(row, col)`'s slot in `nzval`, adds the weight
there, and appends the slot to `positions`. [`_sink_point!`](@ref) opens each grid point's
own slice of that list in `point_ptr`, so a later replay can address a point directly
instead of relying on the walk order.

The search is the expensive half of assembly, which is why it is done once and replayed by
[`ReplaySink`](@ref) afterwards.

# Throws
- `ArgumentError`: A `(row, col)` the pattern does not contain. This pass builds the cache,
  so a pattern that cannot hold the term is reported rather than skipped, as
  [`add_to_sparse!`](@ref) now does on the threaded path too.

See also: [`visit_bilinear_stencil`](@ref), [`NzvalSegment`](@ref).
"""
struct RecordSink{M<:SparseMatrixCSC,TERM}
    A::M
    term::TERM
    point_ptr::Vector{Int}
    positions::Vector{Int}
end
@inline _sink_point!(sink::RecordSink, lin_idx::Int) =
    (@inbounds sink.point_ptr[lin_idx] = length(sink.positions) + 1; 0)
@inline function _sink_entry!(sink::RecordSink, row::Int, col::Int, weight, ::Int)
    pos = _find_nzval_position(sink.A, row, col)
    pos == 0 && _throw_missing_pattern_entry(sink.term)
    @inbounds sink.A.nzval[pos] += weight
    push!(sink.positions, pos)
    return nothing
end

"""
    ReplaySink(A::SparseMatrixCSC, point_ptr::Vector{Int}, positions::Vector{Int})

Add a term's values to `A` using slots recorded earlier by [`RecordSink`](@ref).

The same walk and the same fresh stencil evaluation, because weights may be live: a
coefficient grid function updated in place through `Rₕ!` is seen by the next assembly. Only
the slot lookup is skipped, taken from `positions` rather than searched for, which is what
the cache buys. `row` and `col` are ignored for that reason, and
[`_sink_needs_coordinates`](@ref) tells [`_step_entry!`](@ref) as much, so it skips computing
them at all rather than computing and discarding them.

Immutable: the walk's position advances as a loop-local in
[`visit_bilinear_stencil`](@ref), handed back through `slot`, rather than as a field of the
sink. Carrying it as a mutable field measured about 20% slower on the cheapest replays.

See also: [`visit_bilinear_stencil`](@ref).
"""
struct ReplaySink{M<:SparseMatrixCSC}
    A::M
    point_ptr::Vector{Int}
    positions::Vector{Int}
end
@inline _sink_point!(sink::ReplaySink, lin_idx::Int) = @inbounds(sink.point_ptr[lin_idx])
@inline _sink_needs_coordinates(::ReplaySink) = false
Base.@propagate_inbounds function _sink_entry!(
    sink::ReplaySink, ::Int, ::Int, weight, slot::Int
)
    @inbounds sink.A.nzval[sink.positions[slot]] += weight
    return nothing
end
