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

- Constrained indices are walked, never scanned for: `_each_marked` calls
  `MarkedIndices` (utils/linear_algebra.jl), which skips empty `BitVector` chunks and
  steps set bits with `trailing_zeros`, so every routine here costs the boundary
  cardinality rather than `ndofs`. It is the one iterator; nothing else in this file walks a
  mask by hand.
- Sparse matrices are modified through their CSC arrays directly, in a single sweep that
  writes the diagonal where it meets it. Calling `A[i, i] = one(T)` afterwards would
  binary-search the column, so the sweeps that can avoid that do.
- No `@simd` anywhere: every one of these loops is branch-driven, which rules it out.

See also: [`dirichlet_constraints`](@ref), [`dirichlet_bc!`](@ref), [`symmetrize!`](@ref)
=#

"""
    DirichletConstraint{CT} = DomainMarkers{CT}

Type alias for Dirichlet boundary constraint storage.
"""
const DirichletConstraint{CT} = DomainMarkers{CT}

"""
    dirichlet_constraints(input, [I::CartesianProduct{1}], pairs::Pair...) -> DomainMarkers

Create Dirichlet boundary constraints.

Each `pair` is of the form `:label => func`, where `:label` identifies the boundary region and `func` defines the Dirichlet values. If the optional time domain `I` is provided, `func` must be a time-dependent function `func(x, t)`; this is checked by arity, since nothing about `func` itself can be evaluated at construction time.

`input` can be a `CartesianProduct`, a `Domain`, an `AbstractMeshType`, a `ScalarGridSpace`, or a `CompositeGridSpace` from which the mesh is extracted. The `:label` must match a label in the mesh definition.
"""
function dirichlet_constraints(input, pairs::Pair...)
    _constraint_domain(input)      # validates `input`; the domain itself is never stored
    return _create_generic_markers(pairs...)
end

function dirichlet_constraints(input, I::CartesianProduct{1}, pairs::Pair...)
    _constraint_domain(input)      # validates `input`; the domain itself is never stored
    _validate_time_dependent_arity(pairs)
    return _create_generic_markers(pairs...)
end

# A time domain `I` promises the evaluation path (`(dm::DomainMarkers)(t)`, which does
# `Base.Fix2(func, t)`) that every `func` here accepts `(x, t)`. Nothing downstream checks
# this: `Fix2` builds regardless of arity and only fails once the resulting closure is
# called during assembly, far from the mistake. Caught here by arity alone, not by calling
# `func`, since a condition's closure is otherwise never evaluated before assembly.
function _validate_time_dependent_arity(pairs::Tuple{Vararg{Pair}})
    for (lbl, func) in pairs
        hasmethod(func, Tuple{Any, Any}) ||
            error("dirichlet_constraints: condition for label `:$lbl` must accept (x, t) since a time domain was given, got $(func)")
    end
end

@inline _constraint_domain(input::ScalarGridSpace) = set(mesh(input))
# recursive: the first leaf space. Every leaf of a composite space shares the domain, so
# which one is asked does not matter.
@inline _constraint_domain(input::CompositeGridSpace) = set(mesh(first_space(input)))
@inline _constraint_domain(input::Union{
    CartesianProduct, Domain, AbstractMeshType}) = set(input)
@inline _constraint_domain(input) = _throw_bad_dirichlet_input(input)

@noinline function _throw_bad_dirichlet_input(input)
    throw(ArgumentError(
        "dirichlet_constraints: `input` must be a CartesianProduct, Domain, " *
        "AbstractMeshType, ScalarGridSpace, or CompositeGridSpace, got a $(typeof(input))"))
end

#===========================================================================#
# Element type handling in boundary constraints
#
# `conditions` is stored as a `Tuple`, containing one `Marker{F}` per condition's
# raw closure. Since closures are preserved without type erasure, no return type
# probe is required at construction time: `v[idx] = func(point(...))` converts
# directly to `v`'s element type upon assignment, accommodating both standard floats
# and `ForwardDiff.Dual` numbers without allocation.
#===========================================================================#

"""
    dirichlet_constraints(X::CartesianProduct, f::Function) -> DomainMarkers

Create a single Dirichlet boundary constraint with function `f` under the `:boundary` label.
"""
@inline dirichlet_constraints(X::CartesianProduct, f::F) where {F <:
                                                                Function} = dirichlet_constraints(
    X, :boundary => f)

"""
    _validate_dirichlet_labels(labels)

Internal helper to validate the `dirichlet_labels` parameter.

Ensures that `labels` is either `nothing`, a `Symbol`, or a `Tuple` of `Symbol`s.
Throws an error if the validation fails.

Used by `bilinear_form.jl` and `linear_form.jl` to validate the `dirichlet_labels`
keyword argument before applying boundary conditions.
"""
function _validate_dirichlet_labels(labels)
    if labels !== nothing && !(labels isa Symbol || labels isa Tuple)
        error("dirichlet_labels must be nothing, a Symbol, or a Tuple of Symbols")
    end
end

#===========================================================================#
# Restricting which leaf components of a composite space `dirichlet_labels` binds to
#
# `components` names leaves by their 1-based position in `leaf_spaces_offsets`, following the
# depth-first, left-to-right ordering used by `u(1)`/`u(2)` addressing elsewhere in the
# form layer: for a Stokes-style `Wₕ = vector_gridspace(Ωₕ, Val(2))` (velocity, pressure),
# setting `components = 1` constrains velocity only, leaving pressure unconstrained.
# `nothing` (the default) targets all leaves.
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
        "space(s); leaves are numbered 1 to $n_leaves, the same order u(1), u(2), ... " *
        "addresses."))
end

# A scalar space has exactly one implicit leaf: `components` may only ask for it or ask for
# nothing.
@inline _validate_scalar_components(::Nothing) = nothing
@inline _validate_scalar_components(components) = _validate_dirichlet_components(
    components, 1)

# Whether the leaf at 1-based position `i` is selected by `components` (`nothing` means
# all leaves, matching the unrestricted default).
@inline _leaf_selected(::Nothing, i::Int) = true
@inline _leaf_selected(components::Int, i::Int) = components == i
@inline _leaf_selected(components::Tuple, i::Int) = i in components

# Calls `f(sp, offset)` for each leaf selected by `components`, walking the full tuple from
# `leaf_spaces_offsets` using `Base.tail` recursion rather than creating a dynamic sub-tuple.
# Walking the statically-shaped tuple keeps leaf types concrete and avoids heap allocation.
@inline _each_selected_leaf(f::F, ::Tuple{}, components, i::Int = 1) where {F} = nothing
@inline function _each_selected_leaf(f::F, leaves::Tuple, components, i::Int = 1) where {F}
    sp, offset = first(leaves)
    _leaf_selected(components, i) && f(sp, offset)
    _each_selected_leaf(f, Base.tail(leaves), components, i + 1)
    return nothing
end

# --- Walking a boundary mask -------------------------------------------------------- #

# The one way this file iterates constrained indices. A thin caller over `MarkedIndices`
# (utils/linear_algebra.jl): the chunk-skipping bit-walk itself is a linear-algebra utility,
# not something specific to Dirichlet boundary conditions, so it lives there and this file
# just calls it (gpena/Bramble.jl#71). `_dot_masked` walks the same set of indices for the
# same reason.
#
# `offset` is where this mask's leaf starts in the global system (zero for a scalar space),
# so `f` always receives a global index.
@inline function _each_marked(f::F, mask::BitVector, offset::Int) where {F}
    @inbounds for i in MarkedIndices(mask, offset)
        f(i)
    end
end

# --- Applying Dirichlet boundary conditions ---------------------------------------- #

"""
    dirichlet_bc!(A::AbstractMatrix, Ωₕ::AbstractMeshType, labels::Symbol...) -> AbstractMatrix

Apply Dirichlet boundary conditions to matrix `A` based on marked regions in the mesh `Ωₕ`.

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

# Overloads for CompositeGridSpace: handles both flat and hierarchical spaces.
#
# `leaf_spaces_offsets` (space/vector_gridspace.jl) answers with a tuple, so the leaves
# keep their concrete types and the loops below unroll rather than dispatching per leaf.
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

# One entry per leaf, always: `components` never changes how many entries there are, only
# each one's `active` flag, so this stays the same fully-unrolled shape whether or not a
# caller restricts `components` (see `_each_selected_leaf` for why filtering the tuple
# itself is avoided).
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
    dirichlet_bc!(A::AbstractMatrix, space::CompositeGridSpace, labels::Symbol...; components = nothing) -> AbstractMatrix

Apply Dirichlet boundary conditions to matrix `A` on the regions named by `labels`, restricted
to the leaf components specified in `components` (1-based positions in `leaf_spaces_offsets(space)`,
following the depth-first ordering used by `u(1)`/`u(2)` addressing). `components = nothing` (the
default) applies to every leaf component.

This allows coupled systems to constrain selected fields while leaving others unconstrained
(for example, prescribing velocity while leaving pressure free in a Stokes problem):

```julia
Wₕ = vector_gridspace(Ωₕ, Val(2))   # 1: velocity, 2: pressure
dirichlet_bc!(A, Wₕ, :left, :right; components = 1)   # velocity only
```

Successive calls with different `labels`/`components` pairs compose cleanly.
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
    for (mask, offset, _, active) in entries
        active || continue
        _each_marked(mask, offset) do r
            @views A[r, :] .= zero(T)
            A[r, r] = one(T)
        end
    end
    return A
end

# Sparse: a single sweep of the stored values, testing each row against every *selected*
# leaf. Sweeping once per leaf instead would cost `nnz` per component.
#
# The diagonals are written by that same sweep rather than by a second pass afterwards --
# see the longer note on `_dirichlet_bc_indices!(::SparseMatrixCSC, ...)`, which does the
# same thing against a flat mask.
function _dirichlet_bc_rows!(A::SparseMatrixCSC, entries::Tuple)
    T = eltype(A)
    rows = rowvals(A)
    vals = nonzeros(A)

    @inbounds for j in axes(A, 2)
        column_is_constrained = _row_marked(entries, j)
        diagonal_found = false

        for k in nzrange(A, j)
            row = rows[k]
            if column_is_constrained && row == j
                vals[k] = one(T)
                diagonal_found = true
            elseif _row_marked(entries, row)
                vals[k] = zero(T)
            end
        end

        column_is_constrained && !diagonal_found && (A[j, j] = one(T))
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

Union representing either unevaluated Dirichlet constraints (`DomainMarkers`) or time-evaluated
constraints (`EvaluatedDomainMarkers`).
"""
const ConstraintMarkers = Union{DomainMarkers, EvaluatedDomainMarkers}

"""
    dirichlet_bc!(v::AbstractVector, Ωₕ::AbstractMeshType, bcs::ConstraintMarkers, labels::Symbol...) -> AbstractVector

Write Dirichlet values into `v` at the points marked by `labels`.

`bcs` may be unevaluated constraints or time-evaluated constraints (see [`ConstraintMarkers`](@ref)).
Only the marked entries are modified, with complexity proportional to the boundary cardinality.
"""
@inline function dirichlet_bc!(
        v::AbstractVector, Ωₕ::AbstractMeshType, bcs::ConstraintMarkers,
        labels::NTuple{N, Symbol}, offset::Int = 0) where {N}
    isempty(labels) && return v
    _apply_conditions!(conditions(bcs), v, Ωₕ, labels, offset)
    return v
end

# Unrolled by recursion on the conditions tuple, same idiom as `_write_components!`
# (utils/linear_algebra.jl): explicit recursion avoids heap allocations from small heterogeneous tuple iterations.
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
    _dirichlet_bc_indices!(A::AbstractMatrix, index_in_marker::BitVector)

Internal helper to apply Dirichlet boundary conditions to matrix `A` at the indices marked
in `index_in_marker`: each marked row is zeroed and its diagonal set to one.

Costs the boundary cardinality, not `ndofs` — the marked indices are walked with
`_each_marked` rather than scanned for.
"""
function _dirichlet_bc_indices!(A::AbstractMatrix, index_in_marker::BitVector)
    T = eltype(A)
    _each_marked(index_in_marker, 0) do i
        @views A[i, :] .= zero(T)
        A[i, i] = one(T)
    end
    return A
end

"""
    _dirichlet_bc_indices!(A::SparseMatrixCSC, index_in_marker::BitVector)

Apply Dirichlet boundary conditions to a sparse matrix `A` by directly manipulating
its CSC data structure.

A single sweep of the stored values does both halves of the job: entries in a constrained
row are zeroed, and the diagonal of such a row is set to one where the sweep meets it,
rather than by a second pass afterwards. Explicit zeros are left in place, so the sparsity
pattern is unchanged and the matrix can be refilled without reallocating its columns.
"""
function _dirichlet_bc_indices!(A::SparseMatrixCSC, index_in_marker::BitVector)
    T = eltype(A)
    rows = rowvals(A)
    vals = nonzeros(A)

    # One sweep, not two: the diagonal of a constrained row is a stored entry like any
    # other, so it is written where this sweep meets it rather than searched for afterwards
    # -- `A[i, i] = one(T)` binary-searches the column every call. Same reasoning, and the
    # same `diagonal_found` fallback, as `symmetrize!(::SparseMatrixCSC, ...)` below.
    #
    # No `@simd`: the branches rule it out, as they already did before the diagonal write
    # moved in here.
    @inbounds for j in axes(A, 2)
        column_is_constrained = index_in_marker[j]
        diagonal_found = false

        for k in nzrange(A, j)
            row = rows[k]
            if column_is_constrained && row == j
                vals[k] = one(T)
                diagonal_found = true
            elseif index_in_marker[row]
                vals[k] = zero(T)
            end
        end

        # A constrained row whose diagonal is not stored: rare, since the pattern comes
        # from `allocate_system_matrix`, but this is the one write that has to grow it.
        column_is_constrained && !diagonal_found && (A[j, j] = one(T))
    end
    return A
end

"""
    _function_in_linear_indices(func, Ωₕ, i)

Internal helper to evaluate a boundary function at a grid point given its linear index.

Converts linear index `i` to Cartesian indices and evaluates `func` at the
corresponding physical coordinates in mesh `Ωₕ`.

# Arguments
- `func`: Boundary condition function
- `Ωₕ`: Mesh
- `i`: Linear index into mesh points

# Returns
The value of `func` at the `i`-th mesh point.
"""
_function_in_linear_indices(func, Ωₕ, i) = func(point(Ωₕ, indices(Ωₕ)[i]))

@inline function _dirichlet_bc_indices!(v::AbstractVector, Ωₕ::AbstractMeshType,
        index_in_marker::BitVector, func::F, offset::Int = 0) where {F}
    cart_indices = indices(Ωₕ)

    # Walked at offset zero, not at `offset`: the index is needed twice over, once against
    # the leaf's own mesh points and once against the global vector, and only the latter is
    # shifted.
    _each_marked(index_in_marker, 0) do idx
        @inbounds v[idx + offset] = func(point(Ωₕ, cart_indices[idx]))
    end

    return v
end

# --- Symmetrization of the linear system ------------------------------------------- #

"""
    dirichlet_bc_symmetrize!(A::AbstractMatrix, F::AbstractVector, Ωₕ::AbstractMeshType, labels::Symbol...)

Impose Dirichlet conditions on `A` and symmetrize the linear system `Ax = F`.

The stored zeros this leaves behind remain in the sparse structure. Preserving explicit zeros
keeps the sparsity pattern fixed across assemblies, avoiding costly CSC column reallocations.
"""
function dirichlet_bc_symmetrize!(
        A::AbstractMatrix, F::AbstractVector, Ωₕ::AbstractMeshType, labels::Symbol...)
    dirichlet_bc!(A, Ωₕ, labels...)
    symmetrize!(A, F, Ωₕ, labels...)
    return nothing
end

"""
    symmetrize!(A::AbstractMatrix, F::AbstractVector, Ωₕ::AbstractMeshType, labels::Symbol...)

Modify the linear system `Ax = F` to restore symmetry in `A` after applying Dirichlet conditions.
Updates `F` to account for prescribed boundary data and zeroes the corresponding columns in `A`.

For each index `i` with prescribed Dirichlet boundary conditions:
- Calculate `dᵢ = cᵢ .* F[i]`, where `cᵢ` is the `i`-th column of `A`;
- Update `F` by subtracting `dᵢ` from `F` (preserving the `i`-th component);
- Zero out off-diagonal elements in the `i`-th column of `A`.
"""
# A scalar space carries its mesh, and every other entry point here takes one.
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
    symmetrize!(A::AbstractMatrix, F::AbstractVector, Wₕ::CompositeGridSpace, labels::Symbol...; components = nothing)

Symmetrize a coupled linear system leaf space by leaf space.

The counterpart of the composite `dirichlet_bc!`: each leaf's marker mask is read at that
leaf's offset into the global system without allocating full-system masks. `leaf_spaces_offsets`
returns a tuple, enabling loop unrolling and type stability.

Takes the same `components` keyword as composite `dirichlet_bc!`, restricting which leaf
components `labels` binds to (1-based positions in `leaf_spaces_offsets(Wₕ)`). `components = nothing`
(the default) applies to every leaf.
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

# Generic implementation for dense matrices
function symmetrize!(A::AbstractMatrix, F::AbstractVector, mask::BitVector, offset::Int = 0)
    T = eltype(A)
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
# The diagonal is written directly where the sweep encounters it rather than calling
# `A[i, i] = one(T)` afterwards (which would perform a binary search for the row index).
#
# Skipping `F` when the boundary value is zero benefits homogeneous conditions.
# It is safe under automatic differentiation: `iszero` on a `ForwardDiff.Dual` checks partials
# as well as the value.
#
# No `@simd`: the branch rules it out, and `F[rows[k]]` is an indirect scatter.
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
