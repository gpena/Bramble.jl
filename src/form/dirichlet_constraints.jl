#=
# dirichlet_constraints.jl

This file implements Dirichlet boundary condition handling for finite element assembly.

## Mathematical Background

Dirichlet boundary conditions impose constraints of the form:
```math
u(x) = g(x) \\quad \\text{for } x \\in \\Gamma_D
```

where Γ_D is the Dirichlet boundary and g is the prescribed function.

## Usage Pattern

```julia
# Define boundary conditions
bc = dirichlet_constraints(Ωₕ, :left => x -> 0.0, :right => x -> 1.0)

# Apply to matrix and vector
dirichlet_bc!(A, mesh(Wₕ), :left, :right)
dirichlet_bc!(F, mesh(Wₕ), bc, :left, :right)

# Symmetrize system after BCs
symmetrize!(A, F, mesh(Wₕ), :left, :right)
```

## Performance Optimizations

- BitVector chunk-based processing (64 bits at a time)
- Direct sparse matrix CSC structure manipulation
- SIMD-friendly loops for cache efficiency

See also: [`dirichlet_constraints`](@ref), [`dirichlet_bc!`](@ref), [`symmetrize!`](@ref)
=#

"""
	$(TYPEDEF)

Alias for storage of Dirichlet constraints.
"""
const DirichletConstraint{FType} = DomainMarkers{FType}

"""
	dirichlet_constraints(_set, [I::CartesianProduct{1}], pairs...)

Creates Dirichlet boundary constraints.

Each `pair` is of the form `:label => func`, where `:label` identifies the boundary region and `func` defines the Dirichlet values. If the optional time domain `I` is provided, `func` should be a time-dependent function `func(x, t)`.

The `cartesian_product` can be a `CartesianProduct` mesh domain or an `ScalarGridSpace` from which the mesh can be extracted. The `:label` must match a label in the mesh definition.
"""
function dirichlet_constraints(input, pairs::Pair...)
    cartesian_product = if input isa ScalarGridSpace
        set(mesh(input))
    elseif input isa CompositeGridSpace
        set(mesh(first_space(input)))  # recursive: first leaf space
    else
        set(input)
    end
    T, domain = _get_eltype_and_domain(cartesian_product)
    _create_generic_markers(T, domain, pairs...)
end

function dirichlet_constraints(input, I::CartesianProduct{1}, pairs::Pair...)
    cartesian_product = if input isa ScalarGridSpace
        set(mesh(input))
    elseif input isa CompositeGridSpace
        set(mesh(first_space(input)))
    else
        set(input)
    end
    T, domain = _get_eltype_and_domain(cartesian_product)
    _create_generic_markers(T, domain, I, pairs...)
end

"""
	_get_eltype_and_domain(cartesian_product)

Internal helper to extract element type and spatial domain from either a
`CartesianProduct` or `ScalarGridSpace`.

# Arguments

  - `cartesian_product`: Either a `CartesianProduct` domain or a `ScalarGridSpace`

# Returns

  - `(T, domain)` where `T` is the element type and `domain` is the `CartesianProduct`

This helper enables a unified interface for `dirichlet_constraints` that accepts
both mesh domains and grid spaces.
"""
_get_eltype_and_domain(X::CartesianProduct{D, T}) where {D, T} = (T, X)
_get_eltype_and_domain(Wₕ::ScalarGridSpace) = (eltype(Wₕ), set(mesh(Wₕ)))

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
end

# Overloads for ScalarGridSpace / AbstractSpaceType (single component)
@inline function dirichlet_bc!(A::AbstractMatrix, space::ScalarGridSpace, labels::Symbol...)
    return dirichlet_bc!(A, mesh(space), labels...)
end

@inline function dirichlet_bc!(v::AbstractVector, space::ScalarGridSpace, bcs, labels::Symbol...)
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

# Whether global row `r` falls in any leaf's marked set. Recursive over the tuple of
# leaves, so it unrolls to a handful of comparisons.
@inline _row_marked(::Tuple{}, r::Int) = false

@inline function _row_marked(entries::Tuple, r::Int)
    mask, offset, n = first(entries)
    i = r - offset
    (1 <= i <= n) && @inbounds(mask[i]) && return true
    return _row_marked(Base.tail(entries), r)
end

@inline _leaf_entries(leaves::Tuple, label::Symbol) = map(
    e -> (index_in_marker(mesh(first(e)), label), last(e), ndofs(first(e))), leaves)

function dirichlet_bc!(A::AbstractMatrix, space::CompositeGridSpace, labels::Symbol...)
    leaves = leaf_spaces_offsets(space)
    for p in labels
        _dirichlet_bc_rows!(A, _leaf_entries(leaves, p))
    end
    return nothing
end

# Dense: one pass over the marked rows of each leaf.
function _dirichlet_bc_rows!(A::AbstractMatrix, entries::Tuple)
    T = eltype(A)
    for (mask, offset, n) in entries
        @inbounds for i in 1:n
            if mask[i]
                r = offset + i
                @views A[r, :] .= zero(T)
                A[r, r] = one(T)
            end
        end
    end
    return nothing
end

# Sparse: a single sweep of the stored values, testing each row against every leaf, then
# the diagonals. Sweeping once per leaf instead would cost `nnz` per component.
function _dirichlet_bc_rows!(A::SparseMatrixCSC, entries::Tuple)
    T = eltype(A)
    rows = rowvals(A)
    vals = nonzeros(A)

    @inbounds for j in axes(A, 2)
        for k in nzrange(A, j)
            _row_marked(entries, rows[k]) && (vals[k] = zero(T))
        end
    end

    for (mask, offset, n) in entries
        @inbounds for i in 1:n
            if mask[i]
                r = offset + i
                A[r, r] = one(T)
            end
        end
    end
    return nothing
end

function dirichlet_bc!(v::AbstractVector, space::CompositeGridSpace, bcs, labels::Symbol...)
    for (sp, offset) in leaf_spaces_offsets(space)
        dirichlet_bc!(view(v, (offset + 1):(offset + ndofs(sp))), mesh(sp), bcs, labels...)
    end
    return nothing
end

"""
	dirichlet_bc!(v::AbstractVector, Ωₕ::AbstractMeshType, bcs::DirichletConstraint, labels::Symbol...)

Apply Dirichlet boundary conditions to vector `v` using the [DirichletConstraint](@ref) object `bcs` and the mesh `Ωₕ`.
"""
function dirichlet_bc!(v::AbstractVector, Ωₕ::AbstractMeshType, bcs::DirichletConstraint, labels::Symbol...)
    isempty(labels) && return

    for marker in conditions(bcs)
        current_label = label(marker)
        if current_label in labels
            func = identifier(marker)
            marker_indices = index_in_marker(Ωₕ, current_label)
            _dirichlet_bc_indices!(v, Ωₕ, marker_indices, func)
        end
    end

    return
end

function dirichlet_bc!(v::AbstractVector, Ωₕ::AbstractMeshType, bcs::EvaluatedDomainMarkers, labels::Symbol...)
    isempty(labels) && return

    for marker in conditions(bcs)
        current_label = label(marker)
        if current_label in labels
            func = identifier(marker)
            marker_indices = index_in_marker(Ωₕ, current_label)
            _dirichlet_bc_indices!(v, Ωₕ, marker_indices, func)
        end
    end

    return
end

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

function _dirichlet_bc_indices!(v::AbstractVector, Ωₕ::AbstractMeshType,
        index_in_marker::BitVector, func::BrambleFunction)
    g = PointwiseEvaluator(func, Ωₕ)
    cart_indices = indices(Ωₕ)

    chunks = index_in_marker.chunks
    @inbounds for (chunk_idx, chunk) in enumerate(chunks)
        chunk == zero(UInt64) && continue

        offset = (chunk_idx - 1) * 64
        temp_chunk = chunk
        while temp_chunk != zero(UInt64)
            bit_pos = trailing_zeros(temp_chunk)
            idx = offset + bit_pos + 1
            v[idx] = g(cart_indices[idx])
            temp_chunk &= temp_chunk - 1
        end
    end

    return
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
function symmetrize!(A::AbstractMatrix, F::AbstractVector, Ωₕ::AbstractMeshType,
        labels::Symbol...)
    for p in labels
        symmetrize!(A, F, index_in_marker(Ωₕ, p), 0)
    end
    return nothing
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
"""
function symmetrize!(A::AbstractMatrix, F::AbstractVector, Wₕ::CompositeGridSpace,
        labels::Symbol...)
    for p in labels
        for (sp, offset) in leaf_spaces_offsets(Wₕ)
            symmetrize!(A, F, index_in_marker(mesh(sp), p), offset)
        end
    end
    return nothing
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
    return nothing
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
    return nothing
end

# Implementation for sparse matrices
function symmetrize!(A::SparseMatrixCSC, F::AbstractVector, mask::BitVector,
        offset::Int = 0)
    T = eltype(A)
    rows = rowvals(A)
    vals = nonzeros(A)

    _each_marked(mask, offset) do i
        dirichlet_val = F[i]

        # Update F and zero out column `i` using sparse structure
        @inbounds @simd for k_ptr in nzrange(A, i)
            row_k = rows[k_ptr]
            F[row_k] -= vals[k_ptr] * dirichlet_val
            vals[k_ptr] = zero(T)
        end

        # Restore diagonal and RHS vector value
        A[i, i] = one(T)
        F[i] = dirichlet_val
    end
    return nothing
end
