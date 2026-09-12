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

# Whether `off_u`'s trial index survives, without computing the linear index `_trial_column`
# would return -- for sinks that only need to know whether the entry lands, not where
# ([`_sink_needs_coordinates`](@ref)). An `AbsoluteColumn` always survives, matching
# `_trial_column`: it names a source column directly and is never `0`.
Base.@propagate_inbounds function _trial_inbounds(lin_indices, I::CartesianIndex, off_u)
    Iu = I + CartesianIndex(off_u)
    return checkbounds(Bool, lin_indices, Iu)
end
@inline _trial_inbounds(lin_indices, I::CartesianIndex, off_u::AbsoluteColumn) = true

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

# One entry: guard it, and hand it to the sink if it lands inside. Answers whether it did,
# so the caller can advance the slot.
#
# Fused rather than "compute the target, then act on it" (`_entry_target`, which the tests
# use and `_scatter_point!` shares) because returning a `(row, col)` sentinel tuple has to be
# merged from three return points: measured against the hand-written loop it cost 5 extra phi
# nodes, 4 integer adds and 3 comparisons per entry, with identical loads, stores and calls.
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
    lin_indices = LinearIndices(indices(Ωₕ))

    # `@inbounds` here is what the hand-written sweeps had wrapping their whole nested
    # loop. It does not cross a function call on its own, so `_visit_entries`,
    # `_entry_target` and `_trial_column` are `Base.@propagate_inbounds` to inherit it --
    # without that, the bounds checks come back and the replay path costs 8-17% more.
    @inbounds for I in indices(Ωₕ)
        lin_idx = lin_indices[I]
        stencil = local_stencil(term, sp, I, mesh_markers, lin_idx)
        # `slot` stays a loop-local so it lives in a register rather than a sink field.
        slot = _sink_point!(sink, lin_idx)
        _visit_entries(sink, stencil, lin_indices, I, row_offset, col_offset, slot)
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
