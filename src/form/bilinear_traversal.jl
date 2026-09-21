# bilinear_traversal.jl: one shared walk over a term's local stencils
# (`visit_bilinear_stencil`), and the sinks that plug into it (`PatternSink`, `RecordSink`,
# `ReplaySink`). Kept as a single file deliberately: the walk and its sinks are a tightly
# coupled design for guaranteeing allocation-free inner loops (gpena/Bramble.jl#50), and
# splitting them further would scatter that coupling across files without a readability
# gain (gpena/Bramble.jl#116).

# --- Utility helpers -------------------------------------------------------------- #

"""
    _scatter_position(A::AbstractMatrix, row::Int, col::Int) -> Int

Where entry `(row, col)` lives in `A`'s own storage, or `0` for a matrix type that can
answer "not stored" (only `SparseMatrixCSC` does; every dense fallback below always
answers a valid position).

The seam a new backend's matrix type implements to plug into assembly (S1.1,
gpena/Bramble.jl#12): `PatternSink`, `RecordSink`, `ReplaySink`, `DiagonalReplaySink` and
[`add_to_sparse!`](@ref) reduce to this and [`_scatter_add!`](@ref) once the raw `nzval`/
linear-index field access each used to do directly is factored out here.

The `SparseMatrixCSC` method is exactly the position search this file always ran: a linear
scan of the column when it holds few entries, a binary search otherwise -- both rely on
`SparseMatrixCSC`'s own invariant that `rowval` is sorted within each column. The generic
`AbstractMatrix` fallback needs no search at all: every `(row, col)` inside the matrix's
bounds is "stored" for a dense backend, at `LinearIndices(A)[row, col]`.
"""
@inline function _scatter_position(A::SparseMatrixCSC, row::Int, col::Int)
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

@inline _scatter_position(A::AbstractMatrix, row::Int, col::Int) = _scatter_position(locality(typeof(A)), A, row, col)
@inline _scatter_position(::HostLocality, A::AbstractMatrix, row::Int, col::Int) = LinearIndices(A)[row, col]

"""
    _scatter_add!(A::AbstractMatrix, pos::Int, val) -> Nothing

Add `val` at the position [`_scatter_position`](@ref) named in `A`'s own storage.

`SparseMatrixCSC`'s method writes `nzval[pos]`; the generic `AbstractMatrix` fallback writes
`A[pos]` directly (linear indexing into the backing array), the other half of the seam
[`_scatter_position`](@ref) documents.
"""
@inline function _scatter_add!(A::SparseMatrixCSC, pos::Int, val)
    @inbounds A.nzval[pos] += val
    return nothing
end
@inline _scatter_add!(A::AbstractMatrix, pos::Int, val) = _scatter_add!(locality(typeof(A)), A, pos, val)
@inline function _scatter_add!(::HostLocality, A::AbstractMatrix, pos::Int, val)
    @inbounds A[pos] += val
    return nothing
end

# --- device-resident CSR: search and scatter without scalar indexing (gpena/Bramble.jl#94,
# S4.2) --------------------------------------------------------------------------------- #
#
# A Bramble-owned device CSR (`BrambleMetalExt.MetalSparseMatrixCSR` today, a future
# `CuSparseMatrixCSR`/etc. tomorrow) is not a `SparseMatrixCSC`, so unmodified it fell into
# the generic `AbstractMatrix` fallbacks above, which scalar-index the backing array and
# throw `Scalar indexing is disallowed` the moment `assemble!`/`assemble_parallel!` reach it.
#
# This file has no dependency on Metal (or on `GPUArrays`), so it cannot name
# `MetalSparseMatrixCSR` and dispatch on it directly -- exactly the reason every new device
# method in this milestone keys on `::DeviceLocality` (derived from storage,
# `Bramble.locality`), never on a concrete backend type. What *is* named here is a field
# layout, not a type: `rowPtr`/`colVal`/`nzVal`/`dims`, the row-major CSR convention
# `BrambleMetalExt` already documents and that CUDA.jl's own `CuSparseMatrixCSR` shares --
# the whole point of `ka_device`'s GPU-agnostic design (S0.1) is that a second backend
# inherits this for free by matching the same field names.
#
# The search itself needs `rowPtr`/`colVal` off the device, and per-entry scatter needs
# somewhere to accumulate that is not a scalar `setindex!` into device memory. Both host
# mirrors are cached per matrix object (`_device_csr_mirror`), built once with two bulk
# transfers (`Array(A.rowPtr)`, `Array(A.colVal)`) rather than one scalar read per entry --
# the same "one bulk transfer, never one per element" rule `host_weights` (S4.0) follows.
# `_zero_stored!` resets the cached `nzval` mirror itself (below) -- never `A.nzVal` on a
# device matrix, which would race the flush (see that method's own comment) -- so a second
# `assemble!` on the same `A` does not accumulate on top of the first.
#
# `visit_bilinear_stencil`'s own sinks (`RecordSink`/`ReplaySink`) reduce to exactly
# `_scatter_position`/`_scatter_add!` too, so they pick up this method for free; the only
# path that actually reaches a device matrix today is the band-coloured sweep
# (`_assemble_bilinear_parallel_core!`, `bilinear_execution.jl`) that a `GpuPolicy` backend's
# assembly is forced into (see `_coerce_serial_to_threaded` below).
#
# The cache exists for exactly one reason: so a *second* `assemble!` on the same long-lived
# `A` reuses the already-downloaded `rowPtr`/`colVal` instead of re-fetching them every call.
# It is **not** consulted per scattered entry -- `bilinear_execution.jl` resolves a matrix's
# mirror once, before its sweep starts (`_resolve_device_mirror`), and threads that one object
# through the whole sweep and the flush as a plain argument. That distinction is the result of
# S4.2's actual bug (round 7), not a design chosen up front: a per-call re-lookup keyed on `A`
# -- what every earlier round of this cache still did, however it was guarded -- lost the odd
# near-boundary matrix entry in a repeated-assembly stress test (`n = 513`/`1025`/`2049`, 40+
# assemblies each, every one compared against the CPU result), and neither `GC.@preserve A`
# around the sweep, nor `ka_synchronize` at several points, nor a `ReentrantLock` around every
# access, nor swapping the container from `IdDict` to `Dict`, closed it alone. Bypassing the
# lookup during the sweep entirely did, immediately and completely -- confirmed first with a
# plain `Ref` standing in for this whole cache, then written the "real" way below. See
# [`_resolve_device_mirror`](@ref)'s own docstring for the full account.
#
# The cache is keyed on `objectid(A)` -- a plain `UInt`, not `A` itself -- with a `WeakRef`
# carried alongside to confirm the key was not reused by an unrelated later object once `A`
# is collected. `IdDict{Any,_DeviceSparseMirror}` keyed directly on `A` was the first version
# of this and leaked: an `IdDict` holds its keys strongly, so `A` -- and through it, its
# device `rowPtr`/`colVal`/`nzVal` -- stayed reachable, and therefore unfreed, for the rest
# of the session, however many forms were assembled and discarded after it (review finding,
# S4.2, round 2). Neither `A` itself nor the host mirror is ever kept alive by anything other
# than the `WeakRef`, so once nothing else references `A`, this entry decays to a dead
# `WeakRef` that the next `_device_csr_mirror` call on a colliding `objectid` replaces
# outright. `Dict`, not `IdDict`, backs the cache itself: the key here is a plain `UInt`
# value, not an object with a meaningful notion of identity beyond its value, which is
# exactly what ordinary `==`/`hash` comparison is for and `IdDict`'s `===` semantics are not
# -- keeping `IdDict` for this key type was one of the things tried and measurably wrong.
#
# A dead entry is still a second leak on its own, smaller but real (review finding, round 2):
# the dict keeps the *tuple* -- `WeakRef` and `_DeviceSparseMirror` alike -- reachable forever
# once inserted, so a dead `WeakRef` leaves its three host vectors (`rowptr`/`colval`/`nzval`,
# each `nnz` long) stranded rather than freed. `_device_csr_mirror` prunes every dead entry
# each time it builds a new one: pruning is a linear scan of the whole cache, and a build
# already pays for two device-to-host transfers, so amortising the scan there costs nothing
# extra that matters -- and, now that a build happens once per assembly rather than once per
# scattered entry, the scan runs far less often than it once would have.
mutable struct _DeviceSparseMirror{Tv}
    const rowptr::Vector{Int}
    const colval::Vector{Int}
    const nzval::Vector{Tv}
end

const _device_sparse_mirrors = Dict{UInt, Tuple{WeakRef, _DeviceSparseMirror}}()

# Guards every access now that a build can, in principle, run concurrently with another
# thread's read of a *different* matrix's entry (`Threads.@threads`-driven assembly is not
# ruled out anywhere in this package): `Dict` inserts are not thread-safe, and this is cheap
# insurance against it now that it only runs once per assembly rather than once per scattered
# entry.
const _device_sparse_mirrors_lock = ReentrantLock()

@inline function _existing_device_mirror(A)
    entry = lock(_device_sparse_mirrors_lock) do
        get(_device_sparse_mirrors, objectid(A), nothing)
    end
    entry === nothing && return nothing
    wr, mirror = entry
    return wr.value === A ? mirror : nothing
end

@inline function _device_csr_mirror(A)
    mirror = _existing_device_mirror(A)
    mirror === nothing || return mirror
    ka_synchronize(A.rowPtr)
    rowptr = Vector{Int}(Array(A.rowPtr))
    colval = Vector{Int}(Array(A.colVal))
    built = _DeviceSparseMirror{eltype(A)}(rowptr, colval, zeros(eltype(A), length(colval)))
    lock(_device_sparse_mirrors_lock) do
        # Prune dead entries before inserting the new one -- see the comment above the cache
        # itself for why this is here and not in `_existing_device_mirror`.
        filter!(kv -> kv.second[1].value !== nothing, _device_sparse_mirrors)
        _device_sparse_mirrors[objectid(A)] = (WeakRef(A), built)
    end
    return built
end

"""
    _resolve_device_mirror(A::AbstractMatrix) -> Union{Nothing, _DeviceSparseMirror}

Build (or confirm cached) the host mirror [`_scatter_position`](@ref)/[`_scatter_add!`](@ref)
need for a `DeviceLocality` matrix, before any threaded sweep starts, and return it.

`nothing` for a host matrix. `bilinear_execution.jl` calls this exactly once per assembly,
ahead of the band-coloured sweep, and threads the returned object through the sweep and the
flush as a plain argument from then on -- **not** by handing every caller `A` and letting
`_scatter_position`/`_scatter_add!` look the mirror up again via `_device_csr_mirror`
each time.

That per-call lookup was S4.2's actual bug, round 7, found only by an instrumented stress
test and not by reasoning about it: a 40+-assembly repeated-assembly check at `n = 513`,
`n = 1025` and `n = 2049` lost the odd near-boundary matrix entry (`_zero_stored!`'s device
`fill!` race, `A`'s GC liveness during the sweep, and the cache's `IdDict` vs `Dict` key
semantics were each tried, each measurably helped, and none alone reached zero failures) --
until the lookup itself was replaced with a single value resolved once and passed by hand,
which reached zero failures outright and stayed there. Once the mechanism was identified,
what varied between rounds stopped mattering: a *global, keyed* re-lookup on every scattered
entry is the shape of bug this file no longer has, regardless of which container backs it.
`_device_csr_mirror`'s cache (below) still exists, and is still worth keeping, for the one
thing a keyed lookup is actually needed for: recognising the *same* long-lived `A` across
repeated `assemble!` calls, so its `rowPtr`/`colVal` are not re-downloaded every time.
"""
@inline _resolve_device_mirror(A::AbstractMatrix) = _resolve_device_mirror(locality(typeof(A)), A)
@inline _resolve_device_mirror(::HostLocality, A::AbstractMatrix) = nothing
@inline _resolve_device_mirror(::DeviceLocality, A::AbstractMatrix) = _device_csr_mirror(A)

"""
    _flush_device_scatter!(A::AbstractMatrix, mirror) -> Nothing

Copy the host-staged `nzval` mirror back to `A`'s device storage in one bulk `copyto!`, and
block until that copy has actually landed.

A no-op for a host matrix (`mirror === nothing` always, there), and for a device matrix with
no mirror yet (nothing was ever scattered into it). `bilinear_execution.jl` calls this
exactly once, after its band-coloured sweep returns, passing the *same* mirror object
[`_resolve_device_mirror`](@ref) gave it before the sweep started -- never re-fetched here --
for the reason [`_resolve_device_mirror`](@ref)'s own docstring gives in full. One `copyto!`
of the whole array per assembly, not one per entry, now that owning `bilinear_execution.jl`
gives this seam a place to run from (S4.2's first round scattered each entry straight to
`A.nzVal` piecemeal; correct, but hundreds of thousands of 4-byte device transfers on a real
matrix).

The `copyto!` alone was not enough either (S4.2, round 4): a device write queues
asynchronously exactly like a `@kernel` launch does, and every kernel launch in this
milestone is followed by a `synchronize` -- this one was not, so `assemble`/`assemble!` could
return before the copy had actually landed, and a caller reading `A` right after got
whatever was there so far. [`ka_synchronize`](@ref) is the fix, following that same
kernel-launch idiom for a caller that has a device array but no kernel of its own.
"""
@inline _flush_device_scatter!(A::AbstractMatrix, mirror) = _flush_device_scatter!(locality(typeof(A)), A, mirror)
@inline _flush_device_scatter!(::HostLocality, A::AbstractMatrix, mirror) = nothing
@inline function _flush_device_scatter!(::DeviceLocality, A::AbstractMatrix, mirror)
    mirror === nothing && return nothing
    copyto!(A.nzVal, mirror.nzval)
    ka_synchronize(A.nzVal)
    return nothing
end

# The CSR mirror image of the `SparseMatrixCSC` search above: `row` picks the slice of
# `colVal` (sorted ascending within it, `BrambleMetalExt.metal_sparse_csr`'s own invariant,
# built from `SparseMatrixCSC(transpose(A))`) and the search is for `col` inside it.
#
# Takes the mirror directly rather than looking it up from `A` -- see
# [`_resolve_device_mirror`](@ref)'s docstring for why a per-call lookup keyed on `A` is
# exactly the bug this shape avoids. [`_scatter_position`](@ref)'s own `::DeviceLocality`
# method below still does that lookup, for the sinks in this file (`RecordSink`/`ReplaySink`)
# that only ever carry `A`, not a resolved mirror; it is unreachable for a device matrix
# today (a `GpuPolicy` backend's assembly never takes the sink-based path,
# `bilinear_execution.jl`'s own header comment), kept only so the seam still behaves if that
# changes, not as the fast path.
@inline function _scatter_position_mirror(mirror::_DeviceSparseMirror, row::Int, col::Int)
    rowptr = mirror.rowptr
    colval = mirror.colval
    p1 = rowptr[row]
    p2 = rowptr[row + 1] - 1

    if (p2 - p1) < 32
        idx = p1
        @inbounds while idx <= p2
            colval[idx] == col && return idx
            idx += 1
        end
    else
        lo = p1
        hi = p2
        @inbounds while lo <= hi
            mid = (lo + hi) >>> 1
            mid_col = colval[mid]
            if mid_col < col
                lo = mid + 1
            elseif mid_col > col
                hi = mid - 1
            else
                return mid
            end
        end
    end
    return 0
end

@inline _scatter_position(::DeviceLocality, A::AbstractMatrix, row::Int, col::Int) = _scatter_position_mirror(
    _device_csr_mirror(A), row, col)

# Accumulates into the host mirror only -- never a device scalar read or write: `+=` would
# need the former, and `A.nzVal` is left untouched until `_flush_device_scatter!` copies the
# whole mirror across once, after the sweep that calls this returns. The device array is
# stale for the duration of that sweep, but nothing reads it before then: the pattern this
# file guarantees ("every scattered entry is in the pattern") says where an entry lands, not
# when `A` itself becomes readable, and `bilinear_execution.jl` never hands `A` back to a
# caller mid-sweep.
#
# Takes the mirror directly -- see `_scatter_position_mirror` just above for why.
@inline function _scatter_add_mirror!(mirror::_DeviceSparseMirror, pos::Int, val)
    @inbounds mirror.nzval[pos] += val
    return nothing
end

@inline _scatter_add!(::DeviceLocality, A::AbstractMatrix, pos::Int, val) = _scatter_add_mirror!(_device_csr_mirror(A), pos, val)

# The other half of `_zero_stored!`'s matrix-type seam (`bilinear.jl`): any
# `AbstractSparseMatrix` that is not the concrete `SparseMatrixCSC` above -- a device CSR
# included, since `MetalSparseMatrixCSR <: ... <: AbstractSparseMatrix` through
# `GPUArrays.AbstractGPUSparseArray` -- zeros its stored values, generically. The plain
# `AbstractMatrix` fallback in `bilinear.jl` would otherwise reach `fill!(A, ...)`, which
# tries `setindex!` at every `(i, j)` including positions no sparse format stores at all --
# wrong for any sparse matrix, not only a device one, and the reason this needs to be its own
# method.
#
# This DOES need a `DeviceLocality` split, unlike the first two rounds assumed (S4.2, round
# 5): `nonzeros(A)` resolves to `A.nzVal` on a device matrix, and `fill!` on an `MtlVector` is
# an asynchronous device kernel -- a second, unordered writer racing
# `_flush_device_scatter!`'s `copyto!` at the end of the same assembly. Nothing orders "queue
# the zeroing kernel" against "sweep the host mirror, then blit it over `A.nzVal`", so the
# zero-fill could land *after* the flush and silently wipe entries the sweep had just written
# correctly -- reproduced at `n = 1025` (2-3 wrong matrices per 40, worst element error
# `2048.0`, matching `2/h` at `h = 1/1024`), invisible at small `n` only because the fill
# kernel happens to finish before the host-side sweep does. `ka_synchronize` after the flush
# does not touch this: the corruption is already committed to `A.nzVal` before anything reads
# it, so no amount of synchronising a *read* afterwards helps.
#
# The fix removes the second writer instead of ordering the two: `_flush_device_scatter!`
# unconditionally overwrites every stored entry of `A.nzVal` from `mirror.nzval`, so zeroing
# the device array first is redundant work as well as the race's other half. Zeroing the
# *mirror* -- host memory, no device kernel, no timing to get wrong -- is what the sweep
# actually reads and writes, so that alone is both correct and one fewer device kernel per
# assembly. `_scatter_add!` accumulates into it with `+=`, so a second `assemble!` on the
# same `A` would double the first assembly's contribution at every entry the new one also
# touches, without this reset.
@inline _zero_stored!(A::SparseArrays.AbstractSparseMatrix) = _zero_stored!(locality(typeof(A)), A)
@inline function _zero_stored!(::HostLocality, A::SparseArrays.AbstractSparseMatrix)
    fill!(nonzeros(A), zero(eltype(A)))
    return A
end
@inline function _zero_stored!(::DeviceLocality, A::SparseArrays.AbstractSparseMatrix)
    ka_synchronize(nonzeros(A))
    mirror = _existing_device_mirror(A)
    mirror === nothing || fill!(mirror.nzval, zero(eltype(mirror.nzval)))
    return A
end

# `_effective_parallel_policy` (`linear_algebra.jl`) coerces `CpuSerial` to `CpuThreaded` so
# a forced-threaded sweep always threads; it has no answer for a `GpuPolicy` backend, which
# reaches it because `_assemble_bilinear!` (`bilinear.jl`) sends anything that is not
# `CpuSerial` down the same forced-parallel path. The band-coloured sweep itself is already
# matrix-type generic (`_sweep_bilinear_colour!`/`_sweep_band_colour!`, `bilinear_execution.jl`
# -- both dispatch on `CpuThreaded`/`CpuBatch` and reach storage only through
# `_scatter_position`/`_scatter_add!` above), so a `GpuPolicy` backend can run the exact same
# `Threads.@threads` sweep as any CPU one: coloring already keeps two concurrently-swept
# points from writing the same row, which is the only safety property either side of this
# seam needs.
@inline _coerce_serial_to_threaded(::GpuPolicy) = CpuThreaded()

"""
    add_to_sparse!(A::AbstractMatrix, row::Int, col::Int, val::Number, term, mirror = nothing) -> Nothing

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

`mirror`, when given, is a `_DeviceSparseMirror` [`_resolve_device_mirror`](@ref) already
built for `A`: `bilinear_execution.jl`'s band-coloured sweep passes the *same* one on every
call for one assembly, so the search and the add below go straight to it
(`_scatter_position_mirror`/`_scatter_add_mirror!`) instead of resolving it
from `A` again each time -- the per-call re-lookup was S4.2's actual bug (see
[`_resolve_device_mirror`](@ref)'s docstring). Left as `nothing` (the default), this falls
back to the ordinary `_scatter_position(A, ...)`/`_scatter_add!(A, ...)` dispatch, correct
for a host matrix and for a device one reached some other way.

See also: [`allocate_system_matrix`](@ref) and [`RecordSink`](@ref), which raises the same
way on the serial recording pass.
"""
@inline function add_to_sparse!(A::AbstractMatrix, row::Int, col::Int, val::Number, term, mirror = nothing)
    if mirror === nothing
        pos = _scatter_position(A, row, col)
        pos == 0 && _throw_missing_pattern_entry(term)
        _scatter_add!(A, pos, val)
    else
        pos = _scatter_position_mirror(mirror, row, col)
        pos == 0 && _throw_missing_pattern_entry(term)
        _scatter_add_mirror!(mirror, pos, val)
    end
    return nothing
end

# Whether an earlier entry already named the same pair of offsets. `entries` is anything
# whose `k`-th element starts with `(off_u, off_v)`: `entry_offsets(stencil)`
# (form/common.jl) from `_visit_entries`, which keeps offsets and weights in separate
# containers for Enzyme's sake, or the stencil itself from the pattern builders in
# form/jacobian_pattern.jl, which never read a weight at all.
@inline function _offsets_seen_before(entries, k::Int, off_u, off_v)
    @inbounds for l in 1:(k - 1)
        entries[l][1] == off_u && entries[l][2] == off_v && return true
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
    _test_row(lin_indices, I::CartesianIndex, off_v) -> Int

Which row of the test block a stencil entry's test slot names, or `0` for none.

The mirror of [`_trial_column`](@ref). An ordinary offset is bounds-checked against the grid
and answers `0` outside it; a test-side interpolation carries an [`AbsoluteRow`](@ref)
instead, naming a row of the other mesh outright (gpena/Bramble.jl#263).
"""
Base.@propagate_inbounds function _test_row(lin_indices, I::CartesianIndex, off_v)
    Iv = I + CartesianIndex(off_v)
    return checkbounds(Bool, lin_indices, Iv) ? lin_indices[Iv] : 0
end

@inline _test_row(lin_indices, I::CartesianIndex, off_v::AbsoluteRow) = off_v.row

"""
    _test_inbounds(lin_indices, I::CartesianIndex, off_v) -> Bool

Whether `off_v`'s test index survives, without computing the row [`_test_row`](@ref) would
return. An [`AbsoluteRow`](@ref) always survives, matching `_test_row`.
"""
Base.@propagate_inbounds function _test_inbounds(lin_indices, I::CartesianIndex, off_v)
    return checkbounds(Bool, lin_indices, I + CartesianIndex(off_v))
end
@inline _test_inbounds(lin_indices, I::CartesianIndex, off_v::AbsoluteRow) = true

"""
    _test_row_unguarded(lin_indices, I::CartesianIndex, off_v) -> Int

[`_test_row`](@ref), without the bounds check: the interior-core counterpart, matching
[`_trial_column_unguarded`](@ref).
"""
Base.@propagate_inbounds _test_row_unguarded(lin_indices, I::CartesianIndex, off_v) = lin_indices[I + CartesianIndex(off_v)]
@inline _test_row_unguarded(lin_indices, I::CartesianIndex, off_v::AbsoluteRow) = off_v.row

"""
    _trial_column_unguarded(lin_indices, I::CartesianIndex, off_u) -> Int

[`_trial_column`](@ref), without the bounds check.

Only called from the interior core of [`visit_bilinear_stencil`](@ref), where
[`_stencil_margin`](@ref) has already guaranteed `I + CartesianIndex(off_u)` lands inside
`lin_indices` for every entry the term's stencil can produce. An `AbsoluteColumn` names a
source column outright either way, matching `_trial_column`.
"""
Base.@propagate_inbounds _trial_column_unguarded(lin_indices, I::CartesianIndex, off_u) = lin_indices[I + CartesianIndex(off_u)]
@inline _trial_column_unguarded(lin_indices, I::CartesianIndex, off_u::AbsoluteColumn) = off_u.col

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
    _sink_point!(sink, lin_idx::Int, I::CartesianIndex) -> Int

Announce grid point `lin_idx` (at cartesian index `I`) to `sink`, and answer the base slot
for its entries.

The default answers `0`. [`RecordSink`](@ref) uses the call to open that point's slice of
the position list; [`ReplaySink`](@ref) answers the start of that slice, which the traversal
then adds the entry ordinal to. Addressing each point from its own base is what lets a
replay stay correct regardless of the order grid points are visited in, without any sink
having to carry a mutable cursor. `I` is passed alongside `lin_idx` for
[`DiagonalReplaySink`](@ref), which addresses a point by its rank in `LinearIndices(interior)`
rather than by `lin_idx` -- every other sink ignores it.

See also: [`visit_bilinear_stencil`](@ref), [`_sink_entry!`](@ref).
"""
@inline _sink_point!(::Any, ::Int, ::CartesianIndex) = 0

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
    row = _test_row(lin_indices, I, off_v)
    row == 0 && return (0, 0)
    col = _trial_column(lin_indices, I, off_u)
    col == 0 && return (0, 0)
    return (row + row_offset, col + col_offset)
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
        slot::Int
) where {SINK}
    if _sink_needs_coordinates(sink)
        row = _test_row(lin_indices, I, off_v)
        row == 0 && return false
        col = _trial_column(lin_indices, I, off_u)
        col == 0 && return false
        _sink_entry!(sink, row + row_offset, col + col_offset, weight, slot)
    else
        _test_inbounds(lin_indices, I, off_v) || return false
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
        slot::Int
) where {SINK}
    if _sink_needs_coordinates(sink)
        row = _test_row_unguarded(lin_indices, I, off_v)
        col = _trial_column_unguarded(lin_indices, I, off_u)
        _sink_entry!(sink, row + row_offset, col + col_offset, weight, slot)
    else
        _sink_entry!(sink, 0, 0, weight, slot)
    end
    return nothing
end

# One point's entries.
#
# The offsets and the weights are read from two separate containers ([`entry_offsets`](@ref)
# /[`entry_weights`](@ref), form/common.jl) rather than from the stencil's own mixed
# `Int`/`Float64` tuples. That is what lets `Enzyme` differentiate assembly with respect to
# an operator's own coefficient (gpena/Bramble.jl#249): this function reads both halves of
# every entry and is inlined into `_visit_guarded_region!` rather than into whatever closure
# is being differentiated, which is exactly the combination Enzyme's type analysis cannot
# handle on the entries as they come out of `local_stencil`.
#
# Both loops now index by `k`, where the value-sink branch used to iterate the stencil tuple
# directly: indexing a *heterogeneous* tuple by a loop variable cost 8% on the stiffness
# replay and 17% on the mass one (measured, gpena/Bramble.jl#50), because it defeats the
# unrolling direct iteration gets. `entry_offsets`/`entry_weights` are homogeneous, which is
# what makes indexing them affordable -- re-measured on the same replay benchmarks, not
# assumed.
#
# The de-duplication branch is still chosen at compile time, so a value sink pays nothing
# for `_offsets_seen_before`.
Base.@propagate_inbounds function _visit_entries(
        sink::SINK, stencil, lin_indices, I, row_offset::Int, col_offset::Int, slot::Int
) where {SINK}
    offs = entry_offsets(stencil)
    wts = entry_weights(stencil)
    if _sink_dedups(sink)
        for k in eachindex(wts)
            off_u, off_v = offs[k]
            _offsets_seen_before(offs, k, off_u, off_v) && continue
            _step_entry!(
                sink, lin_indices, I, off_u, off_v, wts[k], row_offset, col_offset, slot
            ) && (slot += 1)
        end
    else
        for k in eachindex(wts)
            off_u, off_v = offs[k]
            _step_entry!(
                sink, lin_indices, I, off_u, off_v, wts[k], row_offset, col_offset, slot
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
    offs = entry_offsets(stencil)
    wts = entry_weights(stencil)
    if _sink_dedups(sink)
        for k in eachindex(wts)
            off_u, off_v = offs[k]
            _offsets_seen_before(offs, k, off_u, off_v) && continue
            _step_entry_unguarded!(
                sink, lin_indices, I, off_u, off_v, wts[k], row_offset, col_offset, slot
            )
            slot += 1
        end
    else
        for k in eachindex(wts)
            off_u, off_v = offs[k]
            _step_entry_unguarded!(
                sink, lin_indices, I, off_u, off_v, wts[k], row_offset, col_offset, slot
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

@inline _interior_range(r::AbstractUnitRange{Int}, margin::Int) = (first(r) + margin):(last(r) - margin)
@inline _low_rim(r::AbstractUnitRange{Int}, margin::Int) = first(r):(first(r) + margin - 1)
@inline _high_rim(r::AbstractUnitRange{Int}, margin::Int) = (last(r) - margin + 1):last(r)
# The unrestricted case in `_boundary_shell_slabs` below, normalized to the same
# `UnitRange{Int}` the three range-builders above already return: `ax[k]` on its own is
# whatever concrete range type the mesh's `CartesianIndices` axes happen to carry (typically
# `Base.OneTo{Int}`), and mixing that with `UnitRange{Int}` across an `ntuple`'s branches
# makes the tuple's element type a `Union` -- which is exactly what made the first version of
# this function allocate 672 B per call instead of 0.
@inline _full_range(r::AbstractUnitRange{Int}) = first(r):last(r)

@inline _peelable(ax::NTuple{D, AbstractUnitRange{Int}}, margin::Int) where {D} = all(r -> 2 * margin <= length(r), ax)

# The `2D` non-overlapping slabs partitioning `indices(Ωₕ)`'s rim, described above. `Val(2D)`
# and the inner `Val(D)` are both resolved from `term`/`sp`'s own type parameters, so both
# `ntuple`s unroll at compile time -- no closure captures a runtime dimension count the way
# `_vectorial_expr`'s comment (`space/operators/stencil.jl`) warns a `Val(i)` built
# from a loop variable would.
@inline function _boundary_shell_slabs(
        ax::NTuple{D, AbstractUnitRange{Int}}, margin::Int
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
        col_offset::Int
) where {SINK, TERM}
    @inbounds for I in region
        lin_idx = lin_indices[I]
        stencil = local_stencil(term, sp, I, mesh_markers, lin_idx)
        slot = _sink_point!(sink, lin_idx, I)
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
        col_offset::Int
) where {SINK, TERM}
    @inbounds for I in interior
        lin_idx = lin_indices[I]
        stencil = local_stencil(term, sp, I, mesh_markers, lin_idx)
        slot = _sink_point!(sink, lin_idx, I)
        _visit_entries_unguarded(
            sink, stencil, lin_indices, I, row_offset, col_offset, slot
        )
    end
    return nothing
end

"""
    visit_bilinear_stencil(sink, term, sp, row_offset::Int, col_offset::Int) -> sink
    visit_bilinear_stencil(interior_sink, boundary_sink, term, sp, row_offset::Int, col_offset::Int) -> boundary_sink

Walk every grid point of `sp`, evaluate `term`'s local stencil there, and hand each entry
that lands inside the matrix to a sink.

The walk that five separate sweeps used to re-derive: mesh, markers, linear indices, the
loop over grid points, the stencil evaluation, and the decision of where each entry lands
([`_entry_target`](@ref)). A sink supplies only what to do with an entry, so the pattern
passes and the value passes share one traversal instead of agreeing by coincidence.

The fifth argument handed to `local_stencil` is the *leaf's* linear index. A term routed to
the wrong leaf reads the wrong `SourceVector` without complaint, which is why the index is
derived here rather than by each caller.

Splits into an interior core and a boundary shell when [`_stencil_margin`](@ref) and the
grid size allow it (see the comment above `_peelable`), so most points skip the bounds
guard entirely; every point still gets exactly one visit either way, so a sink sees the same
set of entries regardless of which path ran, in a possibly different order.
[`RecordSink`](@ref)/[`ReplaySink`](@ref) are unaffected by that: they address each point by
its own linear index, not by visit order (see their docstrings).

The two-sink form lets the interior and the boundary shell be handled by different sinks --
only [`DiagonalReplaySink`](@ref) needs this, pairing itself (interior) with an ordinary
[`ReplaySink`](@ref) (boundary shell), so the one-sink form below is the thin, common case.

# Arguments
- `sink` (or `interior_sink`/`boundary_sink`): What to do per entry. See [`PatternSink`](@ref),
  [`RecordSink`](@ref), [`ReplaySink`](@ref) and [`DiagonalReplaySink`](@ref), and the
  contract in [`_sink_entry!`](@ref), [`_sink_point!`](@ref) and [`_sink_dedups`](@ref).
- `term`: The AST node whose stencil is evaluated at each point.
- `sp`: The test leaf whose grid is walked and whose markers the stencil sees.
- `row_offset`, `col_offset`: The block's origin in the assembled matrix, `0` for a scalar
  space.

# Returns
- The boundary-shell sink (`sink` itself, in the one-sink form), so a caller can read what
  it collected.

See also: [`allocate_system_matrix`](@ref), [`add_to_sparse!`](@ref).
"""
@inline function visit_bilinear_stencil(
        interior_sink::SINK1,
        boundary_sink::SINK2,
        term::TERM,
        sp,
        row_offset::Int,
        col_offset::Int
) where {SINK1, SINK2, TERM}
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
            interior_sink,
            term,
            sp,
            mesh_markers,
            lin_indices,
            interior,
            row_offset,
            col_offset
        )
        @inbounds for slab in _boundary_shell_slabs(ax, margin)
            _visit_guarded_region!(
                boundary_sink,
                term,
                sp,
                mesh_markers,
                lin_indices,
                slab,
                row_offset,
                col_offset
            )
        end
    else
        _visit_guarded_region!(
            boundary_sink,
            term,
            sp,
            mesh_markers,
            lin_indices,
            grid_inds,
            row_offset,
            col_offset
        )
    end
    return boundary_sink
end

# The ordinary one-sink call every caller but the diagonal replay path uses: the same sink
# plays both roles, so `visit_bilinear_stencil(sink, ...)` behaves exactly as it always has.
@inline visit_bilinear_stencil(sink, term, sp, row_offset::Int, col_offset::Int) = visit_bilinear_stencil(
    sink, sink, term, sp, row_offset, col_offset)

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
# faster and `inner₊(∇ₕ(u), ∇ₕ(v))` unchanged. The enclosing function's LLVM is
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
    _SegmentCountSink(point_ptr::Vector{Int})

Count-only pass over a term's stencil: how many entries each grid point contributes and how
many the whole block has, with `A` never touched at all (weights are still evaluated -- the
shared traversal computes them before any sink sees an entry -- just discarded).

Exists solely so [`RecordSink`](@ref) can preallocate its own `positions` to the exact right
size instead of growing it with `push!` (gpena/Bramble.jl#240): `push!`, called from inside
`_record_segment!` while a term's own coefficient is what is being differentiated, is what
made `Enzyme` fail to compile the recording pass at all (`EnzymeNoTypeError`) -- confirmed
directly by isolating it, not assumed: a hand-rolled sink identical to `RecordSink` except
for pre-sized, `setindex!`-only `positions` compiled and differentiated correctly, matching
finite differences, with no `Enzyme.API` flag of any kind. The one-time cost is walking a
term's stencil twice during recording (once to count, once to search and scatter) rather than
once -- recording is already the expensive half of assembly and happens once per matrix
(replayed thereafter by [`ReplaySink`](@ref)), so this doubles a cost paid once, never the
per-replay cost `assemble!`'s zero-allocation guarantee actually protects.

Uses the same coordinate-computing path `RecordSink` does (`_sink_needs_coordinates` left at
its default `true`), deliberately not the cheaper coordinate-free path `ReplaySink`/
`DiagonalReplaySink` use: this pass's whole point is to agree with `RecordSink`'s own entry
count exactly, and sharing its guard branch removes any risk of the two disagreeing.

See also: [`RecordSink`](@ref), [`visit_bilinear_stencil`](@ref).
"""
mutable struct _SegmentCountSink
    const point_ptr::Vector{Int}
    n::Int
end
# Not `@inline` -- deliberately: `@inline` here is what made `Enzyme` unable to compile
# `_record_segment!` at all when the term's coefficient is what is being differentiated
# (`EnzymeNoTypeError`), confirmed directly by isolating it (`@inline` alone reproduces the
# failure, `const` fields do not). One-time recording cost either way; no measurable effect
# on `assemble!`'s own zero-allocation replay, which never touches this sink.
function _sink_point!(sink::_SegmentCountSink, lin_idx::Int, ::CartesianIndex)
    (
        @inbounds sink.point_ptr[lin_idx] = sink.n + 1; 0)
end
function _sink_entry!(sink::_SegmentCountSink, ::Int, ::Int, weight, ::Int)
    sink.n += 1
    return nothing
end

"""
    RecordSink(A::AbstractMatrix, term, point_ptr::Vector{Int}, positions::Vector{Int}, α, n::Int)

Add a term's values to `A` and record where each entry landed, building the replay cache.

For each entry it finds the `(row, col)`'s slot via [`_scatter_position`](@ref), adds the
weight there with [`_scatter_add!`](@ref), and writes the slot into `positions[n]` for a
running `n` (`positions` is preallocated to its final size by [`_SegmentCountSink`](@ref)
before `RecordSink` ever runs -- see its docstring for why this is `setindex!`, not `push!`).
[`_sink_point!`](@ref) opens each grid point's own slice of that list in `point_ptr`, so a
later replay can address a point directly instead of relying on the walk order.

The search is the expensive half of assembly, which is why it is done once and replayed by
[`ReplaySink`](@ref) afterwards.

# Throws
- `ArgumentError`: A `(row, col)` the pattern does not contain. This pass builds the cache,
  so a pattern that cannot hold the term is reported rather than skipped, as
  [`add_to_sparse!`](@ref) now does on the threaded path too.

See also: [`visit_bilinear_stencil`](@ref), [`NzvalSegment`](@ref).
"""
mutable struct RecordSink{M <: AbstractMatrix, TERM, S}
    const A::M
    const term::TERM
    const point_ptr::Vector{Int}
    const positions::Vector{Int}
    const α::S
    n::Int
end
# Not `@inline` -- see `_SegmentCountSink`'s comment just above: the same reason, verified
# the same way.
_sink_point!(sink::RecordSink, lin_idx::Int, ::CartesianIndex) = (
    @inbounds sink.point_ptr[lin_idx] = sink.n + 1; 0)
function _sink_entry!(sink::RecordSink, row::Int, col::Int, weight, ::Int)
    pos = _scatter_position(sink.A, row, col)
    pos == 0 && _throw_missing_pattern_entry(sink.term)
    _scatter_add!(sink.A, pos, sink.α * weight)
    sink.n += 1
    @inbounds sink.positions[sink.n] = pos
    return nothing
end

"""
    ReplaySink(A::AbstractMatrix, point_ptr::Vector{Int}, positions::Vector{Int}, α)

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
struct ReplaySink{M <: AbstractMatrix, S}
    A::M
    point_ptr::Vector{Int}
    positions::Vector{Int}
    α::S
end
@inline _sink_point!(sink::ReplaySink, lin_idx::Int, ::CartesianIndex) = @inbounds(sink.point_ptr[lin_idx])
@inline _sink_needs_coordinates(::ReplaySink) = false
Base.@propagate_inbounds function _sink_entry!(
        sink::ReplaySink, ::Int, ::Int, weight, slot::Int
)
    @inbounds _scatter_add!(sink.A, sink.positions[slot], sink.α * weight)
    return nothing
end

# `n` is mutable, unlike every other field: `_sink_point!` sets it once per point and
# `_sink_entry!` reads it for every one of that point's `P` taps, recovering the tap number
# as `slot - n * P` (a multiply and a subtract). The alternative -- reconstructing `n` from
# `slot` alone via `divrem(slot, P)` -- needs a genuine integer division every entry, because
# `P` is a runtime field (it varies per term/segment, so it cannot be a type parameter
# without reopening the boxing `Segment`'s docstring already describes for a
# per-element-varying parameter). Measured: the `divrem` version replayed a 1D interior
# 30-40% *slower* than the flat `ReplaySink` it was meant to beat, even though it read no
# `positions` array at all -- division dominated the saving.
#
# A comment block here, between the docstring below and the struct it documents, silently
# detaches the docstring from `DiagonalReplaySink` entirely -- `Docs.doc` and every `@ref`
# to it then resolve to nothing, with no error at load time to catch it (found only by
# building the docs: `docs/make.jl` reported four unresolved `@ref`s this struct's own
# docstring and others' made to it and to its neighbours, all silently broken the same way).
# The docstring must be the last thing before the struct, so this note moved above it.
"""
    DiagonalReplaySink(A::AbstractMatrix, interior::CartesianIndices, base::Vector{Int}, stride::Vector{Int}, P::Int, α)

Add a term's values to `A`'s interior core using [`Segment`](@ref)'s per-tap stride
instead of a stored position per entry.

Paired with an ordinary [`ReplaySink`](@ref) for the boundary shell through the two-sink
form of [`visit_bilinear_stencil`](@ref): this sink is only ever handed the interior region,
so `_sink_point!` addresses a point by its rank `n` in `interior`'s own iteration order
(zero-based, via `_interior_rank`) rather than by `lin_idx`, matching the order
`_record_segment!` validated the stride against. The `k`-th tap of that point
(`1`-based, `k in 1:P`) then lands at `base[k] + stride[k] * n`, recovered from the running
`slot` the shared walk already threads (`slot = n * P + (k - 1)`), so no per-entry lookup
runs at all.

See also: [`visit_bilinear_stencil`](@ref), `_replay_segment!`.
"""
mutable struct DiagonalReplaySink{M <: AbstractMatrix, D, R, S}
    const A::M
    const interior::CartesianIndices{D, R}
    const base::Vector{Int}
    const stride::Vector{Int}
    const P::Int
    const α::S
    n::Int
end
function DiagonalReplaySink(A, interior, base, stride, P, α)
    return DiagonalReplaySink(A, interior, base, stride, P, α, 0)
end

# `interior`'s axes are `_interior_range`'s output -- typically not 1-based (a margin-1
# interior on a `OneTo(n)` grid starts at 2) -- so `LinearIndices(interior)` cannot be used
# directly: it normalizes to a 1-based range over the same *length*, not the same *values*,
# and indexing it with `I` unchanged throws (or silently answers a different point). This
# computes the 0-based rank `I` holds in `interior`'s own column-major iteration order --
# first axis fastest, exactly how `for I in interior` visits it -- from first principles.
#
# `CartesianIndices{D,R}` names *both* type parameters deliberately: `CartesianIndices{D}`
# alone is still a `UnionAll` over the ranges-tuple type `R`, not a concrete type, and a
# struct field or argument declared that way is stored boxed -- this is what made every
# `DiagonalReplaySink`/`Segment` built from it allocate (measured 144-384 B per
# replay before this was named).
@inline function _interior_rank(
        interior::CartesianIndices{D, R}, I::CartesianIndex{D}
) where {D, R}
    ax = interior.indices
    n = 0
    stride = 1
    @inbounds for d in 1:D
        n += (I[d] - first(ax[d])) * stride
        stride *= length(ax[d])
    end
    return n
end

@inline _sink_needs_coordinates(::DiagonalReplaySink) = false
@inline function _sink_point!(sink::DiagonalReplaySink, ::Int, I::CartesianIndex)
    n = _interior_rank(sink.interior, I)
    sink.n = n
    return n * sink.P
end
Base.@propagate_inbounds function _sink_entry!(
        sink::DiagonalReplaySink, ::Int, ::Int, weight, slot::Int
)
    n = sink.n
    k0 = slot - n * sink.P
    @inbounds _scatter_add!(sink.A, sink.base[k0 + 1] + sink.stride[k0 + 1] * n, sink.α * weight)
    return nothing
end
