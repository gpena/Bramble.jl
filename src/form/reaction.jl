#=
# reaction.jl

Boundary flux/reaction extraction from a solved constrained problem (gpena/Bramble.jl#227).

## Mathematical background

`dirichlet_bc!` overwrites the constrained rows of `A` and `F` with identity rows, so by
the time a solution `uₕ` exists, the rows that encoded the flux are gone. But the
**unconstrained** operator and load are still recoverable -- `assemble(a)` and
`assemble(l)` with no `dirichlet` keyword give exactly those -- and the residual

```math
r = A u_h - F
```

is (to truncation error) zero on every row the mesh did not constrain, and on a
constrained row is exactly the discrete flux the imposed value had to supply there. This
is the discrete form of the divergence theorem: for `-Δu = f`, `u = g` on `Γ`,
`∫_Ω f = -∮_Γ ∂u/∂n`, so the physical outward flux `q·n = -∂u/∂n` (Fourier's law, unit
conductivity) satisfies `∮_Γ q·n = ∫_Ω f`.

## Sign convention

`reaction` returns `-r` summed over the marker: positive means flux leaving the domain
along the outward normal, so that summing `reaction` over every boundary marker recovers
the net source, `∫_Ω f`, to round-off -- not its negative. This was checked, not assumed:
see the note above `reaction` itself.

## Scaling

`r` already carries the cell-measure weighting `innerₕ`/`inner₊` impose (the same weights
`weights(space, Innerh())` returns), so `reaction`'s net (already-integrated) quantity is
`-r` summed directly, with **no** further rescaling. The pointwise density
([`reaction_density`](@ref)) is the opposite: dividing each marked entry by its own cell
measure recovers a flux **density**, comparable point to point and exportable as one.

## Overlap and components

Marker overlap reuses `_combined_mask` (space/inner_product.jl), the same union used by
`dirichlet_bc!`, so a point shared by two markers is counted once. `components` restricts
a composite space's leaves exactly as `dirichlet_components` does for `assemble`/
`dirichlet_bc!` -- 1-based positions in `leaf_spaces_offsets`, the same order `u(1)`,
`u(2)`, ... addresses.

See also: [`dirichlet_bc!`](@ref), [`assemble`](@ref), [`normal_vector`](@ref).
=#

@inline _reaction_markers(marker::Symbol) = (marker,)
@inline _reaction_markers(marker::NTuple{N, Symbol}) where {N} = marker

function _validate_reaction_spaces(a::BilinearForm, l::LinearForm, uₕ::VectorElement)
    n = ndofs(test_space(l))
    if ndofs(test_space(a)) != n || ndofs(trial_space(a)) != n
        throw(
            ArgumentError(
            "reaction: `a` and `l` must be posed on the same space; got " *
            "$(ndofs(trial_space(a)))×$(ndofs(test_space(a))) for `a` and $n for `l`.",
        ),
        )
    end
    if ndofs(space(uₕ)) != n
        throw(
            ArgumentError(
            "reaction: `uₕ` must be posed on the same space as `a`/`l`; got " *
            "$(ndofs(space(uₕ))) degrees of freedom, expected $n.",
        ),
        )
    end
    return nothing
end

# Offset-aware, allocation-free walk over the union of `markers`' marked points, for every
# reader below (`_reaction_over_leaves`, `_reaction_density_over_leaves!`, and the `!`
# variants' own `_zero_marked!`/`_subtract_marked!`/`_copy_marked!`). A single marker is just
# `MarkedIndices` (`index_in_marker` hands back the mesh's own stored mask, no copy); several
# reuse the lazy `MarkedIndicesUnion` `_combined_marked_indices` (space/inner_product.jl)
# already builds instead of `_combined_mask`'s eager one (a fresh `copy` + `.|=` per call,
# gpena/Bramble.jl#149) -- `reaction!`/`reaction_density!` must allocate zero bytes even with
# several marker labels, which that eager union can't give. `MarkedIndicesUnion` has no
# `offset` field of its own (its only prior caller, `innerₕ`, never needed one), so
# `_ReactionMarked` adds it here rather than in `linear_algebra.jl`. Both walks visit a point
# shared by two markers exactly once, same as `_combined_mask`'s union.
@inline _reaction_marked(Ωₕ, markers::NTuple{1, Symbol}, offset::Int) =
    MarkedIndices(index_in_marker(Ωₕ, markers[1]), offset)
@inline _reaction_marked(Ωₕ, markers::NTuple{N, Symbol}, offset::Int) where {N} =
    _ReactionMarked(_combined_marked_indices(Ωₕ, markers), offset)

struct _ReactionMarked{U}
    union::U
    offset::Int
end

@inline function Base.iterate(m::_ReactionMarked)
    it = iterate(m.union)
    it === nothing && return nothing
    i, st = it
    return i + m.offset, st
end
@inline function Base.iterate(m::_ReactionMarked, st)
    it = iterate(m.union, st)
    it === nothing && return nothing
    i, st = it
    return i + m.offset, st
end
Base.IteratorSize(::Type{<:_ReactionMarked}) = Base.SizeUnknown()
Base.eltype(::Type{<:_ReactionMarked}) = Int

# Walked once per selected leaf, unrolled by recursion over the static `leaf_spaces_offsets`
# tuple -- same idiom as `_apply_conditions!`/`_leaf_entries_impl` (dirichlet_constraints.jl):
# explicit recursion keeps leaf types concrete and avoids a closure allocation for the
# accumulator a `do`-block callback would need.
@inline _reaction_over_leaves(::Tuple{}, r, markers, components, i, acc) = acc
@inline function _reaction_over_leaves(leaves::Tuple, r, markers, components, i, acc)
    sp, offset = first(leaves)
    if _leaf_selected(components, i)
        acc -= _sum_masked(r, _reaction_marked(mesh(sp), markers, offset))
    end
    return _reaction_over_leaves(Base.tail(leaves), r, markers, components, i + 1, acc)
end

@inline function _sum_masked(r, idxs)
    s = zero(eltype(r))
    @inbounds for i in idxs
        s += r[i]
    end
    return s
end

# --- Restricted residual (gpena/Bramble.jl#289) ------------------------------------ #
#
# `_reaction_over_leaves`/`_reaction_density_over_leaves!` above only ever read `r` at
# `MarkedIndices(mask, offset)` -- a small fraction of `ndofs` on a typical grid. Materializing
# `r = A * parent(uₕ) .- F` over every row costs a full sparse matvec plus a full-length
# allocation for that. `SparseMatrixCSC` has no row index, so pulling out even one row still
# costs `nnz(A)`; testing every stored entry once against every *selected* leaf keeps a call
# with several markers/components at one sweep total, not one per row or per leaf.

# One entry per leaf, always (mirrors `_leaf_entries`/`_leaf_entries_impl`,
# dirichlet_constraints.jl): keeps the tuple's shape independent of `components`, so
# `_restricted_matvec`/`_row_marked`'s recursion below stays fully unrolled. `markers` here
# can name several labels, so each entry's mask is `_combined_mask`'s union rather than a
# single `index_in_marker` -- everything else matches.
@inline _reaction_leaf_entries(leaves::Tuple, markers, components) =
    _reaction_leaf_entries_impl(leaves, markers, components, 1)
@inline _reaction_leaf_entries_impl(::Tuple{}, markers, components, i::Int) = ()
@inline function _reaction_leaf_entries_impl(leaves::Tuple, markers, components, i::Int)
    sp, offset = first(leaves)
    entry = (
        _combined_mask(mesh(sp), markers), offset, ndofs(sp), _leaf_selected(components, i)
    )
    return (entry, _reaction_leaf_entries_impl(Base.tail(leaves), markers, components, i + 1)...)
end

# Sparse: a single sweep of the stored values, testing each row against every *selected*
# leaf via `_row_marked` (dirichlet_constraints.jl, same `(mask, offset, n, active)` entry
# shape as `_dirichlet_bc_rows!` uses) -- a `BitVector` test that costs nothing on the ~99%
# of entries that miss, unlike a sorted-row binary search tried and measured slower here
# (paid `O(log m)` on every entry, hit or miss). Accumulates column-by-column with `muladd`,
# the same order `SparseArrays`'s own `A * u` uses (`_spmatmul!`, SparseArrays/linalg.jl), so
# every row kept here comes out bit-identical to the corresponding entry of the full matvec
# it replaces.
function _restricted_matvec(A::SparseMatrixCSC, u::AbstractVector, entries::Tuple)
    T = promote_type(eltype(A), eltype(u))
    acc = Dict{Int, T}()
    rows = rowvals(A)
    vals = nonzeros(A)

    @inbounds for j in axes(A, 2)
        uj = u[j]
        for k in nzrange(A, j)
            row = rows[k]
            if _row_marked(entries, row)
                acc[row] = muladd(vals[k], uj, get(acc, row, zero(T)))
            end
        end
    end
    return acc
end

# Answers `r[j] = (A * u)[j] - F[j]` at just the rows `_restricted_matvec` kept -- every `j`
# `_reaction_over_leaves`/`_reaction_density_over_leaves!` ever ask for is one of them, since
# both walk the same `MarkedIndices` `_reaction_leaf_entries` was built from.
struct _RestrictedResidual{T, Fv <: AbstractVector}
    acc::Dict{Int, T}
    F::Fv
end

@inline Base.eltype(::_RestrictedResidual{T}) where {T} = T
@inline function Base.getindex(r::_RestrictedResidual{T}, j::Int) where {T}
    return get(r.acc, j, zero(T)) - @inbounds r.F[j]
end

# `A::SparseMatrixCSC`: restricted to the rows `entries` marks, per the note above -- no
# full-length residual, no full matvec.
_reaction_residual(A::SparseMatrixCSC, F::AbstractVector, u::AbstractVector, entries::Tuple) =
    _RestrictedResidual(_restricted_matvec(A, u, entries), F)

# Any other matrix type (dense, or an unrecognised backend): unchanged full computation --
# the restricted path above only pays off against `SparseMatrixCSC`'s stored-entry sweep.
function _reaction_residual(A::AbstractMatrix, F::AbstractVector, u::AbstractVector, ::Tuple)
    r = A * u
    r .-= F
    return r
end

# --- In-place variants: caller-owned scratch, zero allocation (gpena/Bramble.jl#289 item 3) --
#
# `reaction!`/`reaction_density!` need literally zero bytes per call, steady state -- not just
# "small", which rules out `_reaction_leaf_entries`/`_restricted_matvec` above even though that
# path is already far smaller than the original full matvec: `_combined_mask` (used to build
# each leaf's entry there) allocates a fresh unioned `BitVector` whenever `markers` names more
# than one label. Below recurses over `leaves`/`markers`/`components` directly instead (the
# same shape `_reaction_over_leaves` uses) and reads/writes only through `_reaction_marked`,
# so it is allocation-free regardless of how many labels `marker` names.
@inline _zero_marked!(scratch::AbstractVector, ::Tuple{}, markers, components, i) = nothing
@inline function _zero_marked!(
        scratch::AbstractVector{T}, leaves::Tuple, markers, components, i
) where {T}
    sp, offset = first(leaves)
    if _leaf_selected(components, i)
        @inbounds for j in _reaction_marked(mesh(sp), markers, offset)
            scratch[j] = zero(T)
        end
    end
    return _zero_marked!(scratch, Base.tail(leaves), markers, components, i + 1)
end

@inline _subtract_marked!(
    scratch::AbstractVector, F::AbstractVector, ::Tuple{}, markers, components, i
) = nothing
@inline function _subtract_marked!(
        scratch::AbstractVector, F::AbstractVector, leaves::Tuple, markers, components, i
)
    sp, offset = first(leaves)
    if _leaf_selected(components, i)
        @inbounds for j in _reaction_marked(mesh(sp), markers, offset)
            scratch[j] -= F[j]
        end
    end
    return _subtract_marked!(scratch, F, Base.tail(leaves), markers, components, i + 1)
end

@inline _copy_marked!(
    scratch::AbstractVector, r::AbstractVector, ::Tuple{}, markers, components, i
) = nothing
@inline function _copy_marked!(
        scratch::AbstractVector, r::AbstractVector, leaves::Tuple, markers, components, i
)
    sp, offset = first(leaves)
    if _leaf_selected(components, i)
        @inbounds for j in _reaction_marked(mesh(sp), markers, offset)
            scratch[j] = r[j]
        end
    end
    return _copy_marked!(scratch, r, Base.tail(leaves), markers, components, i + 1)
end

# The hot sparse sweep below still needs an O(1) indexed row test (a `BitVector` per stored
# entry, `nnz(A)` times), which the allocation-free walk above can't give -- iterating it is
# `O(1)` per step but has no `getindex`. So this keeps its own, separate entries: one
# `NTuple{N,BitVector}` of `markers`' individual, per-label masks per leaf (each
# `index_in_marker` call is itself allocation-free), ORed bit-by-bit on the fly in
# `_reaction_row_marked` -- unlike `_reaction_leaf_entries`/`_row_marked`
# (dirichlet_constraints.jl) above, no label ever gets unioned into a fresh `BitVector`.
@inline _reaction_leaf_entries!(leaves::Tuple, markers, components) =
    _reaction_leaf_entries_impl!(leaves, markers, components, 1)
@inline _reaction_leaf_entries_impl!(
    ::Tuple{}, markers::NTuple{N, Symbol}, components, i::Int
) where {N} = ()
@inline function _reaction_leaf_entries_impl!(
        leaves::Tuple, markers::NTuple{N, Symbol}, components, i::Int
) where {N}
    sp, offset = first(leaves)
    masks = ntuple(k -> index_in_marker(mesh(sp), markers[k]), Val(N))
    entry = (masks, offset, ndofs(sp), _leaf_selected(components, i))
    return (
        entry, _reaction_leaf_entries_impl!(Base.tail(leaves), markers, components, i + 1)...
    )
end

@inline _reaction_row_marked(::Tuple{}, row::Int) = false
@inline function _reaction_row_marked(entries::Tuple, row::Int)
    masks, offset, n, active = first(entries)
    if active
        i = row - offset
        if 1 <= i <= n
            for mask in masks
                @inbounds(mask[i]) && return true
            end
        end
    end
    return _reaction_row_marked(Base.tail(entries), row)
end

# Sparse: no `Dict` at all -- `scratch` (caller-owned, reused across calls) is indexed
# directly by row, so this is zero allocation once `scratch` exists. Zero the marked entries
# first (`_zero_marked!`, proportional to the marker's own size, not `ndofs`), accumulate with
# the same `muladd`/column order `_restricted_matvec`/`SparseArrays`'s own `A * u` use (so
# results are bit-identical to both), then subtract `F` at just those entries as a separate
# final step -- not folded into the initial accumulator, which would reorder the floating-point
# sum and break that bit-identity.
function _reaction_residual!(
        scratch::AbstractVector, A::SparseMatrixCSC, F::AbstractVector, u::AbstractVector,
        leaves::Tuple, markers, components
)
    _zero_marked!(scratch, leaves, markers, components, 1)
    entries = _reaction_leaf_entries!(leaves, markers, components)
    rows = rowvals(A)
    vals = nonzeros(A)

    @inbounds for j in axes(A, 2)
        uj = u[j]
        for k in nzrange(A, j)
            row = rows[k]
            if _reaction_row_marked(entries, row)
                scratch[row] = muladd(vals[k], uj, scratch[row])
            end
        end
    end

    _subtract_marked!(scratch, F, leaves, markers, components, 1)
    return scratch
end

# Any other matrix type: the same full `A * u` the non-`!` fallback uses (bit-identical to
# it), but only the marked entries are copied into `scratch` -- matching the sparse path's
# "only marked entries touched" contract, which `reaction_density!` needs to stay exactly
# zero away from the marker rather than merely small. Still allocates the temporary `r`
# internally: unlike `SparseMatrixCSC`, a dense matrix has no cheaper way to get `A * u` at
# selected rows only, so `!` doesn't buy zero allocation here, only the non-`!` path's own.
function _reaction_residual!(
        scratch::AbstractVector, A::AbstractMatrix, F::AbstractVector, u::AbstractVector,
        leaves::Tuple, markers, components
)
    r = A * u
    r .-= F
    _copy_marked!(scratch, r, leaves, markers, components, 1)
    return scratch
end

"""
    reaction(A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement; marker, components = nothing) -> Real

Net flux through the region(s) named by `marker`, from the **unconstrained** operator `A`
and load `F` (`assemble(a)`/`assemble(l)` with no `dirichlet` keyword -- `dirichlet_bc!`
would have already overwritten the rows this needs).

`marker` is a `Symbol` or a `Tuple` of them, exactly as `dirichlet` accepts one or several
labels; overlap between markers is counted once (`_combined_mask`, the same union
`dirichlet_bc!` uses). `components` restricts a composite space's leaves (1-based
positions in `leaf_spaces_offsets`, `nothing` meaning every leaf).

# Sign and scaling

Positive means flux leaving the domain along the outward normal: for `-Δu = f`,
`∮_Γ reaction = ∫_Ω f` to round-off. The residual already carries the cell-measure
weighting the form's inner product imposes, so no further rescaling is applied here --
see [`reaction_density`](@ref) for the pointwise quantity that does.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 41)
Wₕ = gridspace(Ωₕ)
sol(x) = sin(pi * x[1])
src(x) = pi^2 * sin(pi * x[1])

a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
A, F = assemble(a, l; dirichlet = :boundary => sol)

uₕ = element(Wₕ)
uₕ .= A \\ F

reaction(a, l, uₕ; marker = :left)    # ≈ π
reaction(a, l, uₕ; marker = :right)   # ≈ π
```

See also: [`reaction_density`](@ref), [`assemble`](@ref), [`normal_vector`](@ref).
"""
function reaction(
        A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement;
        marker, components = nothing
)
    sp = space(uₕ)
    leaves = leaf_spaces_offsets(sp)
    _validate_dirichlet_components(components, length(leaves))

    markers = _reaction_markers(marker)
    entries = _reaction_leaf_entries(leaves, markers, components)
    r = _reaction_residual(A, F, parent(uₕ), entries)
    return _reaction_over_leaves(leaves, r, markers, components, 1, zero(eltype(r)))
end

"""
    reaction(a::BilinearForm, l::LinearForm, uₕ::VectorElement; marker, dirichlet_components = nothing) -> Real

Reassembles the unconstrained `a`/`l` (`assemble` with no `dirichlet` keyword) and calls
[`reaction`](@ref)`(A, F, uₕ; marker, components = dirichlet_components)`. `a` and `l` must
be posed on the same space as `uₕ`.
"""
function reaction(
        a::BilinearForm, l::LinearForm, uₕ::VectorElement;
        marker, dirichlet_components = nothing
)
    _validate_reaction_spaces(a, l, uₕ)
    A = assemble(a)
    F = assemble(l)
    return reaction(A, F, uₕ; marker = marker, components = dirichlet_components)
end

"""
    reaction!(scratch::AbstractVector, A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement; marker, components = nothing) -> Real

In-place counterpart of [`reaction`](@ref): the same net flux, computed with zero allocation
using a caller-owned `scratch` (sized `ndofs(space(uₕ))`) instead of an internal one -- meant
for a loop that calls this every step (e.g. a transient solve) and reuses `scratch` across
calls. `scratch`'s ownership/lifetime is the caller's; only the entries `marker`/`components`
select are written, so reusing `scratch` across calls with a *different* `marker`/`components`
reads stale values outside the new selection -- calling with the same selection every time
(the intended use) is always safe.
"""
function reaction!(
        scratch::AbstractVector, A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement;
        marker, components = nothing
)
    sp = space(uₕ)
    leaves = leaf_spaces_offsets(sp)
    _validate_dirichlet_components(components, length(leaves))

    markers = _reaction_markers(marker)
    r = _reaction_residual!(scratch, A, F, parent(uₕ), leaves, markers, components)
    return _reaction_over_leaves(leaves, r, markers, components, 1, zero(eltype(r)))
end

# Same recursion shape as `_reaction_over_leaves`, writing the per-point density into a
# preallocated `dens` (zero outside the marker) instead of accumulating a scalar.
@inline _reaction_density_over_leaves!(dens, ::Tuple{}, r, markers, components, i) = dens
@inline function _reaction_density_over_leaves!(dens, leaves::Tuple, r, markers, components, i)
    sp, offset = first(leaves)
    if _leaf_selected(components, i)
        w = weights(sp, Innerh())
        @inbounds for j in _reaction_marked(mesh(sp), markers, offset)
            dens[j] = -r[j] / w[j - offset]
        end
    end
    return _reaction_density_over_leaves!(dens, Base.tail(leaves), r, markers, components, i + 1)
end

"""
    reaction_density(A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement; marker, components = nothing) -> VectorElement

The pointwise counterpart of [`reaction`](@ref): a grid function, zero away from `marker`,
equal at each marked point to the physical flux **density** there -- `reaction`'s summand
divided by that point's own cell measure (`weights(space, Innerh())`), so that summing
`density .* weights` over the marker recovers [`reaction`](@ref)'s scalar exactly. Meant to
be exported and plotted, e.g. `export_vtk(filename, Ωₕ, "reaction" => reaction_density(...))`.
"""
function reaction_density(
        A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement;
        marker, components = nothing
)
    sp = space(uₕ)
    leaves = leaf_spaces_offsets(sp)
    _validate_dirichlet_components(components, length(leaves))

    markers = _reaction_markers(marker)
    entries = _reaction_leaf_entries(leaves, markers, components)
    r = _reaction_residual(A, F, parent(uₕ), entries)
    dens = zeros(eltype(r), ndofs(sp))
    _reaction_density_over_leaves!(dens, leaves, r, markers, components, 1)
    return element(sp, dens)
end

"""
    reaction_density(a::BilinearForm, l::LinearForm, uₕ::VectorElement; marker, dirichlet_components = nothing) -> VectorElement

Reassembles the unconstrained `a`/`l` and calls [`reaction_density`](@ref)`(A, F, uₕ;
marker, components = dirichlet_components)`.
"""
function reaction_density(
        a::BilinearForm, l::LinearForm, uₕ::VectorElement;
        marker, dirichlet_components = nothing
)
    _validate_reaction_spaces(a, l, uₕ)
    A = assemble(a)
    F = assemble(l)
    return reaction_density(A, F, uₕ; marker = marker, components = dirichlet_components)
end

"""
    reaction_density!(dens::AbstractVector, A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement; marker, components = nothing) -> AbstractVector

In-place counterpart of [`reaction_density`](@ref): writes the flux density into a
caller-owned `dens` (sized `ndofs(space(uₕ))`) with zero allocation instead of allocating a
fresh one -- meant for a loop that calls this every step and reuses `dens` across calls. As
with [`reaction!`](@ref), only the entries `marker`/`components` select are written, so `dens`
stays exactly zero away from the marker only if it started there (a fresh `zeros(...)`, or a
previous call with the same `marker`/`components`). Returns `dens` itself, not a
`VectorElement`; wrap with `element(space(uₕ), dens)` for a grid function.
"""
function reaction_density!(
        dens::AbstractVector, A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement;
        marker, components = nothing
)
    sp = space(uₕ)
    leaves = leaf_spaces_offsets(sp)
    _validate_dirichlet_components(components, length(leaves))

    markers = _reaction_markers(marker)
    r = _reaction_residual!(dens, A, F, parent(uₕ), leaves, markers, components)
    _reaction_density_over_leaves!(dens, leaves, r, markers, components, 1)
    return dens
end
