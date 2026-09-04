#=
# dirichlet_constraints.jl

This file implements Dirichlet boundary condition handling for finite element assembly.

## Mathematical background

Dirichlet boundary conditions impose constraints of the form:
```math
u(x) = g(x) \\quad \\text{for } x \\in \\Gamma_D
```

where Γ_D is the Dirichlet boundary and g is the prescribed function.

## Usage pattern

```julia
# Define boundary conditions
bc = dirichlet_constraints(Ωₕ, :left => x -> 0.0, :right => x -> 1.0)

# Apply to matrix and vector
dirichlet_bc!(A, mesh(Wₕ), :left, :right)
dirichlet_bc!(F, mesh(Wₕ), bc, :left, :right)

# Symmetrize system after BCs
symmetrize!(A, F, mesh(Wₕ), :left, :right)
```

## Performance optimizations

- BitVector chunk-based processing (64 bits at a time)
- Direct sparse matrix CSC structure manipulation
- SIMD-friendly loops for cache efficiency

See also: [`dirichlet_constraints`](@ref), [`dirichlet_bc!`](@ref), [`symmetrize!`](@ref)
=#

"""
	$(TYPEDEF)

Alias for storage of Dirichlet constraints.
"""
const DirichletConstraint{CT} = DomainMarkers{CT}

"""
	dirichlet_constraints(_set, [I::CartesianProduct{1}], pairs...)

Creates Dirichlet boundary constraints.

Each `pair` is of the form `:label => func`, where `:label` identifies the boundary region and `func` defines the Dirichlet values. If the optional time domain `I` is provided, `func` should be a time-dependent function `func(x, t)`.

The `cartesian_product` can be a `CartesianProduct` mesh domain or an `ScalarGridSpace` from which the mesh can be extracted. The `:label` must match a label in the mesh definition.
"""
function dirichlet_constraints(input, pairs::Pair...)
    _constraint_domain(input)      # validates `input`; the domain itself is never stored
    return _create_generic_markers(pairs...)
end

function dirichlet_constraints(input, I::CartesianProduct{1}, pairs::Pair...)
    _constraint_domain(input)      # validates `input`; the domain itself is never stored
    return _create_generic_markers(pairs...)
end

@inline _constraint_domain(input::ScalarGridSpace) = set(mesh(input))
# recursive: the first leaf space. Every leaf of a composite space shares the domain, so
# which one is asked does not matter.
@inline _constraint_domain(input::CompositeGridSpace) = set(mesh(first_space(input)))
@inline _constraint_domain(input) = set(input)

#===========================================================================#
# The element type a constraint stores its values in
#
# Used to need settling ahead of time (point 48's history): the conditions lived in a
# `Set{Marker{BrambleFunction{…, CoType, …}}}`, and a `FunctionWrapper`'s whole point is a
# *fixed* concrete return type declared upfront, which meant probing every condition at
# construction time to promote a `Dual`-returning one correctly. `conditions` is now a
# `Tuple`, one `Marker{F}` per condition's own raw closure — nothing is erased, so nothing
# needs a return type settled ahead of the call: `v[idx] = func(point(...))` converts to
# `v`'s own eltype exactly the way any other assignment does, the same rule `Rₕ`/`avgₕ`
# already follow. `_probe_at`/`_probed_type`/`_constraint_value_type`/
# `_get_eltype_and_domain` accordingly no longer exist.
#===========================================================================#

"""
	dirichlet_constraints(X::CartesianProduct, f::Function)

	Creates a single Dirichlet boundary constraint with function `f` with the label `:dirichlet`.
"""
@inline dirichlet_constraints(X::CartesianProduct, f::F) where {F <:
                                                                Function} = dirichlet_constraints(
    X, :boundary => f)

"""
	_validate_dirichlet_labels(labels)

Internal helper to validate dirichlet_labels parameter.

Ensures that `labels` is either `nothing`, a `Symbol`, or a `Tuple` of `Symbol`s.
Throws an error if the validation fails.

This function is used by both `bilinear_form.jl` and `linear_form.jl` to validate
the `dirichlet_labels` keyword argument before applying boundary conditions.
"""
function _validate_dirichlet_labels(labels)
    if labels !== nothing && !(labels isa Symbol || labels isa Tuple)
        error("dirichlet_labels must be nothing, a Symbol, or a Tuple of Symbols")
    end
end

#===========================================================================#
# Restricting which leaf(-ves) of a composite space `dirichlet_labels` binds to
#
# `components` names leaves by their 1-based position in `leaf_spaces_offsets`, the same
# depth-first, left-to-right order `u(1)`/`u(2)` addressing already uses elsewhere in the
# form layer — so a Stokes-style `Wₕ = vector_gridspace(Ωₕ, Val(2))` (velocity, pressure)
# constrains only velocity with `components = 1`, leaving the pressure block untouched.
# `nothing` (the default) means every leaf, exactly today's unrestricted behaviour.
#===========================================================================#

function _validate_dirichlet_components(components, n_leaves::Int)
    components === nothing && return nothing
    if !(components isa Int || components isa Tuple)
        error("dirichlet_components must be nothing, an Int, or a Tuple of Ints")
    end
    for c in (components isa Int ? (components,) : components)
        c isa Int ||
            error("dirichlet_components must be nothing, an Int, or a Tuple of Ints")
        (1 <= c <= n_leaves) ||
            _throw_dirichlet_component_out_of_range(c, n_leaves)
    end
    return nothing
end

@noinline function _throw_dirichlet_component_out_of_range(c::Int, n_leaves::Int)
    throw(ArgumentError(
        "dirichlet_components names leaf $c, but this space only has $n_leaves leaf " *
        "space(s) — leaves are numbered 1 to $n_leaves, the same order u(1), u(2), ... " *
        "addresses."))
end

# A scalar space has exactly one implicit leaf: `components` may only ask for it or ask for
# nothing.
@inline _validate_scalar_components(::Nothing) = nothing
@inline _validate_scalar_components(components) = _validate_dirichlet_components(
    components, 1)

# Whether the leaf at 1-based position `i` is one `components` names — `nothing` means
# every leaf, matching the unrestricted default.
@inline _leaf_selected(::Nothing, i::Int) = true
@inline _leaf_selected(components::Int, i::Int) = components == i
@inline _leaf_selected(components::Tuple, i::Int) = i in components

# Calls `f(sp, offset)` for each leaf `components` selects, walking the *full* tuple from
# `leaf_spaces_offsets` `Base.tail`-recursively rather than building a filtered sub-tuple
# first. A sub-tuple's length would depend on `components`, a runtime value, so the compiler
# cannot give it one concrete type — precisely the "leaves in a `Vector{Tuple{Any,Int}}`"
# boxing this file already avoids elsewhere; walking the untouched, statically-shaped tuple
# and skipping unselected leaves keeps every leaf's own type and stays allocation free.
# Same `f::F where {F}` pattern `_each_marked` already uses below for the same reason.
@inline _each_selected_leaf(f::F, ::Tuple{}, components, i::Int = 1) where {F} = nothing
@inline function _each_selected_leaf(f::F, leaves::Tuple, components, i::Int = 1) where {F}
    sp, offset = first(leaves)
    _leaf_selected(components, i) && f(sp, offset)
    _each_selected_leaf(f, Base.tail(leaves), components, i + 1)
    return nothing
end

#==============================================================================
						APPLYING DIRICHLET BOUNDARY CONDITIONS
==============================================================================#

"""
	dirichlet_bc!(A::AbstractMatrix, Ωₕ::AbstractMeshType, labels::Symbol...)

Applies Dirichlet boundary conditions to matrix `A` based on marked regions in the mesh `Ωₕ`.

For each index `i` associated with the given Dirichlet `labels`, this function:

 1. Sets all elements in the `i`-th row of `A` to zero.
 2. Sets the diagonal element `A[i, i]` to one.
"""
function dirichlet_bc!(A::AbstractMatrix, Ωₕ::AbstractMeshType, labels::Symbol...)
    for p in labels
        vec_bool = index_in_marker(Ωₕ, p)
        _dirichlet_bc_indices!(A, vec_bool)
    end
    return A
end

# Overloads for ScalarGridSpace / AbstractSpaceType (single component)
@inline function dirichlet_bc!(
        A::AbstractMatrix, space::ScalarGridSpace, labels::Symbol...;
        components = nothing)
    _validate_scalar_components(components)
    return dirichlet_bc!(A, mesh(space), labels...)
end

@inline function dirichlet_bc!(v::AbstractVector, space::ScalarGridSpace, bcs,
        labels::Symbol...; components = nothing)
    _validate_scalar_components(components)
    return dirichlet_bc!(v, mesh(space), bcs, labels...)
end

# Overloads for CompositeGridSpace — handles both flat and hierarchical spaces.
#
# `leaf_spaces_offsets` (space/vector_gridspace.jl) answers with a tuple, so the leaves
# keep their concrete types and the loops below unroll rather than dispatching per leaf.
# It matters: with the leaves in a `Vector{Tuple{Any, Int}}` the innermost assignment
# boxed a Bool on every degree of freedom, 39,646 allocations and 809 KB for one call on a
# 60x60 grid with three components.
#
# The masks are per leaf and the rows are global, so a leaf's mask is consulted through
# its offset rather than copied into a mask over the whole system. That is what keeps the
# call allocation free: `index_in_marker` hands back the mesh's stored BitVector, and
# nothing else is built.

# Whether global row `r` falls in any *selected* leaf's marked set. Recursive over the
# tuple of leaves, so it unrolls to a handful of comparisons. An unselected leaf's own
# `active` flag short-circuits it without touching its mask.
@inline _row_marked(::Tuple{}, r::Int) = false

@inline function _row_marked(entries::Tuple, r::Int)
    mask, offset, n, active = first(entries)
    if active
        i = r - offset
        (1 <= i <= n) && @inbounds(mask[i]) && return true
    end
    return _row_marked(Base.tail(entries), r)
end

# One entry per leaf, always — `components` never changes how many entries there are, only
# each one's `active` flag, so this stays the same fully-unrolled shape whether or not a
# caller restricts `components` (see `_each_selected_leaf` for why filtering the tuple
# itself is the wrong move).
@inline _leaf_entries(leaves::Tuple, label::Symbol, components) = _leaf_entries_impl(
    leaves, label, components, 1)
@inline _leaf_entries_impl(::Tuple{}, label::Symbol, components, i::Int) = ()
@inline function _leaf_entries_impl(leaves::Tuple, label::Symbol, components, i::Int)
    sp, offset = first(leaves)
    entry = (index_in_marker(mesh(sp), label), offset, ndofs(sp),
        _leaf_selected(components, i))
    return (entry, _leaf_entries_impl(Base.tail(leaves), label, components, i + 1)...)
end

"""
	dirichlet_bc!(A::AbstractMatrix, space::CompositeGridSpace, labels::Symbol...; components = nothing)

Applies Dirichlet boundary conditions to matrix `A` on the regions `labels` name, restricted
to the leaf(-ves) `components` names — 1-based positions in `leaf_spaces_offsets(space)`, the
same depth-first order `u(1)`/`u(2)` addressing already uses. `components = nothing` (the
default) applies to every leaf, unchanged from before this keyword existed.

This is what lets a coupled system constrain one field and leave another free — a Stokes
problem prescribing velocity while leaving pressure unconstrained, say:

```julia
Wₕ = vector_gridspace(Ωₕ, Val(2))   # 1: velocity, 2: pressure
dirichlet_bc!(A, Wₕ, :left, :right; components = 1)   # velocity only
```

Call again with a different `labels`/`components` pair for another field; each call only
ever restricts which leaves it touches, so several calls compose.
"""
function dirichlet_bc!(A::AbstractMatrix, space::CompositeGridSpace, labels::Symbol...;
        components = nothing)
    leaves = leaf_spaces_offsets(space)
    _validate_dirichlet_components(components, length(leaves))
    for p in labels
        _dirichlet_bc_rows!(A, _leaf_entries(leaves, p, components))
    end
    return A
end

# Dense: one pass over the marked rows of each leaf, skipping unselected ones.
function _dirichlet_bc_rows!(A::AbstractMatrix, entries::Tuple)
    T = eltype(A)
    for (mask, offset, n, active) in entries
        active || continue
        @inbounds for i in 1:n
            if mask[i]
                r = offset + i
                @views A[r, :] .= zero(T)
                A[r, r] = one(T)
            end
        end
    end
    return A
end

# Sparse: a single sweep of the stored values, testing each row against every *selected*
# leaf, then the diagonals. Sweeping once per leaf instead would cost `nnz` per component.
function _dirichlet_bc_rows!(A::SparseMatrixCSC, entries::Tuple)
    T = eltype(A)
    rows = rowvals(A)
    vals = nonzeros(A)

    @inbounds for j in axes(A, 2)
        for k in nzrange(A, j)
            _row_marked(entries, rows[k]) && (vals[k] = zero(T))
        end
    end

    for (mask, offset, n, active) in entries
        active || continue
        @inbounds for i in 1:n
            if mask[i]
                r = offset + i
                A[r, r] = one(T)
            end
        end
    end
    return A
end

@inline function dirichlet_bc!(v::AbstractVector, space::CompositeGridSpace, bcs,
        labels::Symbol...; components = nothing)
    leaves = leaf_spaces_offsets(space)
    _validate_dirichlet_components(components, length(leaves))
    _each_selected_leaf(leaves, components) do sp, offset
        dirichlet_bc!(v, mesh(sp), bcs, labels, offset)
    end
    return v
end

"""
	ConstraintMarkers

Either shape a set of Dirichlet conditions arrives in: the constraints themselves, or the
same constraints already evaluated at a point in time.

The two are distinct types — `EvaluatedDomainMarkers` holds the original alongside the
timestamp — but they answer `conditions`, `label` and `identifier` identically, and nothing
that applies a condition needs to tell them apart. There used to be two byte-identical
methods, one per type.
"""
const ConstraintMarkers = Union{DomainMarkers, EvaluatedDomainMarkers}

"""
	dirichlet_bc!(v, Ωₕ, bcs, labels...)

Writes the Dirichlet values into `v` at the points `labels` marks.

`bcs` may be the constraints or a time-evaluated form of them; see [`ConstraintMarkers`](@ref).
Only the marked entries are touched, and the work is proportional to how many there are
rather than to the size of the grid — a boundary in a volume is a small fraction of it.
"""
@inline function dirichlet_bc!(
        v::AbstractVector, Ωₕ::AbstractMeshType, bcs::ConstraintMarkers,
        labels::NTuple{N, Symbol}, offset::Int = 0) where {N}
    isempty(labels) && return v
    _apply_conditions!(conditions(bcs), v, Ωₕ, labels, offset)
    return v
end

# Unrolled by recursion on the conditions tuple, same idiom as `_write_components!`
# (utils/linear_algebra.jl) — a plain `for marker in conditions(bcs)` measured 7% slower
# and non-allocation-free (208 B) against this, point 48: the compiler does not always
# fully unroll a `for` over a small heterogeneous tuple the way explicit recursion does.
@inline _apply_conditions!(::Tuple{}, v, Ωₕ, labels, offset) = nothing
@inline function _apply_conditions!(markers::Tuple, v, Ωₕ, labels, offset)
    marker = first(markers)
    if label(marker) in labels
        _dirichlet_bc_indices!(v, Ωₕ, index_in_marker(Ωₕ, label(marker)),
            identifier(marker), offset)
    end
    return _apply_conditions!(Base.tail(markers), v, Ωₕ, labels, offset)
end

@inline dirichlet_bc!(v::AbstractVector, Ωₕ::AbstractMeshType, bcs::ConstraintMarkers,
    labels::Symbol...) = dirichlet_bc!(v, Ωₕ, bcs, labels, 0)

"""
	_dirichlet_bc_indices!(A, marker_indices)

Internal helper to apply Dirichlet boundary conditions to matrix `A` for a given set of indices.
"""
function _dirichlet_bc_indices!(A::AbstractMatrix, index_in_marker::BitVector)
    T = eltype(A)

    chunks = index_in_marker.chunks
    @inbounds for (chunk_idx, chunk) in enumerate(chunks)
        chunk == zero(UInt64) && continue # Skip chunks with no Dirichlet nodes

        offset = (chunk_idx - 1) * 64
        temp_chunk = chunk
        while temp_chunk != zero(UInt64)
            bit_pos = trailing_zeros(temp_chunk)
            i = offset + bit_pos + 1

            # Zero out the i-th row and set diagonal to one
            @views A[i, :] .= zero(T)
            A[i, i] = one(T)

            temp_chunk &= temp_chunk - 1 # Clear the processed bit
        end
    end
    return A
end

"""
_dirichlet_bc_indices!(A::SparseMatrixCSC, index_in_marker::BitVector)

Applies Dirichlet boundary conditions to a sparse matrix `A` by directly manipulating
its CSC data structure for high performance.
"""
function _dirichlet_bc_indices!(A::SparseMatrixCSC, index_in_marker::BitVector)
    T = eltype(A)
    rows = rowvals(A)
    vals = nonzeros(A)

    # 1. Zero out non-zero values in Dirichlet rows
    @inbounds for j in axes(A, 2)
        @simd for i in nzrange(A, j)
            if index_in_marker[rows[i]]
                vals[i] = zero(T)
            end
        end
    end

    # 2. Set diagonal elements to one for all Dirichlet rows
    chunks = index_in_marker.chunks
    @inbounds for (chunk_idx, chunk) in enumerate(chunks)
        chunk == zero(UInt64) && continue

        offset = (chunk_idx - 1) * 64
        temp_chunk = chunk
        while temp_chunk != zero(UInt64)
            bit_pos = trailing_zeros(temp_chunk)
            i = offset + bit_pos + 1
            A[i, i] = one(T)
            temp_chunk &= temp_chunk - 1
        end
    end
    return A
end

"""
	_function_in_linear_indices(func, Ωₕ, i)

Internal helper to evaluate a function at a grid point given its linear index.

Converts linear index `i` to Cartesian indices and evaluates `func` at the
corresponding physical point in mesh `Ωₕ`.

# Arguments

  - `func`: Function to evaluate (typically a boundary condition function)
  - `Ωₕ`: The mesh
  - `i`: Linear index into the mesh points

# Returns

The value of `func` at the `i`-th mesh point.
"""
_function_in_linear_indices(func, Ωₕ, i) = func(point(Ωₕ, indices(Ωₕ)[i]))

@inline function _dirichlet_bc_indices!(v::AbstractVector, Ωₕ::AbstractMeshType,
        index_in_marker::BitVector, func::F, offset::Int = 0) where {F}
    cart_indices = indices(Ωₕ)

    chunks = index_in_marker.chunks
    @inbounds for (chunk_idx, chunk) in enumerate(chunks)
        chunk == zero(UInt64) && continue

        chunk_offset = (chunk_idx - 1) * 64
        temp_chunk = chunk
        while temp_chunk != zero(UInt64)
            bit_pos = trailing_zeros(temp_chunk)
            idx = chunk_offset + bit_pos + 1
            v[idx + offset] = func(point(Ωₕ, cart_indices[idx]))
            temp_chunk &= temp_chunk - 1
        end
    end

    return v
end

#==============================================================================
					SYMMETRIZATION OF THE LINEAR SYSTEM
==============================================================================#

"""
	dirichlet_bc_symmetrize!(A, F, Ωₕ, labels...)

Imposes the Dirichlet conditions on `A` and then symmetrizes the system, in that order.

The stored zeros this leaves behind stay stored. There used to be a `dropzeros` option to
strip them, and it is gone rather than merely defaulted off: the sparsity pattern of an
assembled operator is the stencil's, known ahead of time and shared with every matrix
assembled the same way. Dropping entries makes the pattern depend on the boundary data, and
every later write to a dropped position stops being a value update and becomes a structural
insert, which rebuilds the column. Keeping explicit zeros costs one stored value each and
keeps the pattern fixed, which is the cheaper side of that trade by a wide margin.
"""
function dirichlet_bc_symmetrize!(
        A::AbstractMatrix, F::AbstractVector, Ωₕ::AbstractMeshType, labels::Symbol...)
    dirichlet_bc!(A, Ωₕ, labels...)
    symmetrize!(A, F, Ωₕ, labels...)
    return nothing
end

"""
	symmetrize!(A, F, Ωₕ, labels)

Modifies the linear system `Ax = F` to make `A` symmetric after applying Dirichlet
conditions. It updates the vector `F` and zeros out the columns of `A` corresponding
to Dirichlet nodes.

The algorithm goes as follows: for any given row `i` where Dirichlet boundary conditions have been applied

	- calculate `dᵢ = cᵢ .* F`, where `cᵢ` is the `i`-th column of `A`;
	- replace `F` by subtracting `dᵢ` to `F` (except for the `i`-th component)
	- replace all elements in the `i`-th column of `A` (except the `i`-th by zero).
"""
# A scalar space carries its mesh, and every other entry point here takes one. `symmetrize!`
# did not, so `dirichlet_bc!(A, Wₕ, :bottom)` worked while `symmetrize!(A, F, Wₕ, :bottom)`
# was a MethodError — for a pair of calls that are almost always written together.
@inline function symmetrize!(A::AbstractMatrix, F::AbstractVector, Wₕ::ScalarGridSpace,
        labels::Symbol...; components = nothing)
    _validate_scalar_components(components)
    return symmetrize!(A, F, mesh(Wₕ), labels...)
end

function symmetrize!(A::AbstractMatrix, F::AbstractVector, Ωₕ::AbstractMeshType,
        labels::Symbol...)
    for p in labels
        symmetrize!(A, F, index_in_marker(Ωₕ, p), 0)
    end
end

"""
	symmetrize!(A, F, Wₕ::CompositeGridSpace, labels...)

Symmetrizes a coupled system, one leaf space at a time.

The counterpart of the composite `dirichlet_bc!`, and it works the same way: each leaf's
marker mask is read at that leaf's offset into the global system rather than gathered into
a mask over the whole of it, so the call allocates nothing. `leaf_spaces_offsets` answers
with a tuple, so the loop unrolls and every read through a leaf keeps its concrete type.

Without this method a composite space met a `MethodError` here while `dirichlet_bc!`
accepted it — the two have to agree, since a system is rarely constrained by one and not
the other.

Takes the same `components` keyword as the composite `dirichlet_bc!`, restricting which
leaf(-ves) `labels` binds to — 1-based positions in `leaf_spaces_offsets(Wₕ)`.
`components = nothing` (the default) is every leaf, unchanged from before this keyword
existed.
"""
function symmetrize!(A::AbstractMatrix, F::AbstractVector, Wₕ::CompositeGridSpace,
        labels::Symbol...; components = nothing)
    leaves = leaf_spaces_offsets(Wₕ)
    _validate_dirichlet_components(components, length(leaves))
    for p in labels
        _each_selected_leaf(leaves, components) do sp, offset
            symmetrize!(A, F, index_in_marker(mesh(sp), p), offset)
        end
    end
end

# Walking the set bits of the mask, rather than the mask itself. The marked set is a
# boundary and the grid is a volume, so it is sparse in the extreme — 60 of 3,600 degrees
# of freedom on a 60x60 grid marked `:bottom`. Testing every bit would do 3,600 tests to
# find 60; skipping whole zero chunks and then walking set bits with `trailing_zeros` does
# work proportional to what is marked.
#
# `offset` is where this mask's leaf starts in the global system: the mask is per leaf and
# the matrix is the whole coupled system. Zero for a scalar space.
@inline function _each_marked(f::F, mask::BitVector, offset::Int) where {F}
    @inbounds for (chunk_idx, chunk) in enumerate(mask.chunks)
        chunk == zero(UInt64) && continue
        base = offset + (chunk_idx - 1) * 64
        rest = chunk
        while rest != zero(UInt64)
            f(base + trailing_zeros(rest) + 1)
            rest &= rest - 1
        end
    end
end

# Generic implementation for dense matrices
function symmetrize!(A::AbstractMatrix, F::AbstractVector, mask::BitVector, offset::Int = 0)
    T = eltype(A)
    # `findall(mask)` used to build the index vector here — 576 B on a 60x60 grid, and
    # growing with the boundary. The bit walk needs none.
    _each_marked(mask, offset) do i
        dirichlet_val = F[i]
        @inbounds for k in axes(A, 1)
            if i != k
                F[k] -= A[k, i] * dirichlet_val
                A[k, i] = zero(T)
            end
        end
    end
end

# Implementation for sparse matrices
#
# The diagonal is written where the sweep finds it, not through `A[i, i] = one(T)`
# afterwards. That assignment looks free and is not: `setindex!` on a CSC matrix binary
# searches the column's row indices for `i`, once per marked degree of freedom, immediately
# after a loop that has just walked past that exact entry. Writing it in place makes
# `symmetrize!` about twice as fast — 2.58 µs to 1.38 µs on a 200x200 grid — and the whole
# of that gain is this one change.
#
# The fallback stays for the case the sweep does not find a diagonal, which is a matrix
# that does not store one in a marked column. `dirichlet_bc!` leaves one behind, so the
# usual path never needs it, but `symmetrize!` can be called on its own.
#
# Skipping `F` when the boundary value is zero is worth about 12% on homogeneous
# conditions, which are the common ones, and nothing on inhomogeneous. It is safe under
# automatic differentiation: `iszero` on a `ForwardDiff.Dual` tests the partials as well as
# the value, so a value that is zero here but still varying is not skipped.
#
# No `@simd`: the branch rules it out. It bought nothing anyway — `F[rows[k]]` is an
# indirect scatter.
function symmetrize!(A::SparseMatrixCSC, F::AbstractVector, mask::BitVector,
        offset::Int = 0)
    T = eltype(A)
    rows = rowvals(A)
    vals = nonzeros(A)

    _each_marked(mask, offset) do i
        dirichlet_val = F[i]
        value_is_zero = iszero(dirichlet_val)
        diagonal_found = false

        @inbounds for k_ptr in nzrange(A, i)
            row_k = rows[k_ptr]
            if row_k == i
                # the diagonal: set rather than eliminated, and `F[i]` left alone, so
                # there is nothing to restore afterwards
                vals[k_ptr] = one(T)
                diagonal_found = true
            else
                value_is_zero || (F[row_k] -= vals[k_ptr] * dirichlet_val)
                vals[k_ptr] = zero(T)
            end
        end

        diagonal_found || (A[i, i] = one(T))
    end
end
