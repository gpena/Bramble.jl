# shift.jl
#
# The file holds both halves of the shift family: the numerical operators, which act on grid
# functions and build matrices, and then the AST nodes that the same names build when handed
# a `LazyOp`. It is included after the AST core (`ast/ast.jl`, `common.jl`,
# `expression.jl`, `operators/node_family.jl`), which the node half needs.

"""
    ⊗(A, B)

Kronecker product operator (alias for `kron`).

Computes the Kronecker product (tensor product) of matrices `A` and `B`.
This operator is used extensively in constructing multidimensional shift operators.

# Examples

```julia
I₂ = I(2)
I₃ = I(3)
result = I₂ ⊗ I₃  # 6×6 identity matrix
```

See also: [`shift`](@ref)
"""
@inline ⊗(A, B) = kron(A, B)

"""
    _Eye(be::Backend, npts, ::Val{i})

Internal helper to create identity or shifted diagonal matrices, in the matrix type the
backend `be` chose, routed through [`backend_eye`](@ref)/`matrix_type(be)` rather than a
package-specific lazy type, so the result always matches whatever `matrix_type` the caller's
backend picked.

# Arguments

  - `be::Backend`: the backend whose `matrix_type` the result is built in.
  - `npts::Int`: Size of the square matrix.
  - `::Val{i}`: Diagonal offset (0 = main diagonal, 1 = superdiagonal, -1 = subdiagonal).

# Returns

  - For `i=0`: Identity matrix of size `npts × npts`.
  - For `i≠0`: Matrix with ones on the `i`-th diagonal, zeros elsewhere.

See also: [`shift`](@ref)
"""
@inline _Eye(be, npts::Int, ::Val{0}) = backend_eye(be, npts)
@inline _Eye(be, npts::Int, ::Val{i}) where {i} = _shift_ones(matrix_type(be), npts, i, npts - abs(i))

# The three-way dispatch mirrors `_backend_eye` (backend.jl): a fast path for
# `SparseMatrixCSC` and for dense `Matrix`, and a generic fallback for anything else.
# `spdiagm` alone carries the offset arithmetic, so the two fast paths only differ in
# whether the sparse result is converted afterwards.
@inline _shift_ones(
    ::Type{<:SparseMatrixCSC{T, Ti}}, npts::Int, i::Int, nz::Int
) where {T, Ti} = spdiagm(npts, npts, i => fill(one(T), nz))
@inline _shift_ones(::Type{<:Matrix{T}}, npts::Int, i::Int, nz::Int) where {T} = Matrix{T}(spdiagm(npts, npts, i =>
    fill(one(T), nz)))
# The generic fallback (gpena/Bramble.jl#94): `nz` scalar `setindex!` calls straight into a
# device array error outright under Metal.jl's scalar-indexing guard, rather than merely
# running slowly -- measured, not assumed, in the plan's S2.6 subplan. Built on the host,
# where `setindex!` is a plain memory write, and handed to `MT` in one `copyto!` instead.
function _shift_ones(::Type{MT}, npts::Int, i::Int, nz::Int) where {T, MT <: AbstractMatrix{T}}
    host = zeros(T, npts, npts)
    r0, c0 = i >= 0 ? (0, i) : (-i, 0)
    for k in 1:nz
        host[r0 + k, c0 + k] = one(T)
    end
    A = MT(undef, npts, npts)
    copyto!(A, host)
    return A
end

@inline function _recursive_shift(
        Ωₕ::AbstractMeshType, ::Val{1}, ::Val{DIFF_DIM}, ::Val{i}
) where {DIFF_DIM, i}
    dims = npoints(Ωₕ, Tuple)
    be = backend(Ωₕ)

    if DIFF_DIM == 1
        return _Eye(be, dims[1], Val(i))
    else
        return backend_eye(be, dims[1])
    end
end

@inline function _recursive_shift(
        Ωₕ::AbstractMeshType, ::Val{D}, ::Val{DIFF_DIM}, ::Val{i}
) where {D, DIFF_DIM, i}
    dims = npoints(Ωₕ, Tuple)
    be = backend(Ωₕ)

    # Determine the operator for the current (outermost) dimension D.
    if DIFF_DIM == D
        op_current = _Eye(be, dims[D], Val(i))
    else
        op_current = backend_eye(be, dims[D])
    end

    # Recurse on the inner dimensions (from D-1 down to 1).
    op_lower_dims = _recursive_shift(Ωₕ, Val(D - 1), Val(DIFF_DIM), Val(i))

    # Combine them: M_D ⊗ (M_{D-1} ⊗ ...)
    return op_current ⊗ op_lower_dims
end

"""
    shift(Ωₕ::AbstractMeshType, ::Val{SHIFT_DIM}, ::Val{i})

Returns the matrix that shifts a grid function by `i` points along direction
`SHIFT_DIM`, as a sparse operator over the flattened degrees of freedom of `Ωₕ`.

# Arguments

  - `SHIFT_DIM`: the direction to shift along, `1` for ``x``, `2` for ``y``, `3` for ``z``.
  - `i`: how far to shift. `1` is the superdiagonal, `-1` the subdiagonal, `0` the
    identity. The stencil is truncated at the boundary rather than wrapped, so the
    matrix has `n - |i|` nonzeros per direction rather than `n`.

# Tensor-product structure

A mesh is a tensor product of its one-dimensional meshes, and its degrees of freedom are
flattened in column-major order, so a shift along one direction is the identity in every
other direction. Writing ``E_k`` for the identity of size ``n_k`` and ``S_k(i)`` for the
one-dimensional shift by `i`, the operator is a Kronecker product with ``S`` in one slot:

```math
\\begin{aligned}
\\text{along } x: &\\quad E_z \\otimes E_y \\otimes S_x(i) \\\\
\\text{along } y: &\\quad E_z \\otimes S_y(i) \\otimes E_x \\\\
\\text{along } z: &\\quad S_z(i) \\otimes E_y \\otimes E_x
\\end{aligned}
```

`_recursive_shift` builds exactly this, recursing from the outermost dimension inwards
and placing ``S`` when it reaches `SHIFT_DIM`. The per-direction forms it generalises,
each of which the test suite checks against `shift`:

```julia
# 1D
shift(Ωₕ, Val(1), Val(i))  ==  _Eye(be, nₓ, Val(i))

# 2D, on an nₓ × n_y grid
shift(Ωₕ, Val(1), Val(i))  ==  backend_eye(be, n_y) ⊗ _Eye(be, nₓ, Val(i))
shift(Ωₕ, Val(2), Val(i))  ==  _Eye(be, n_y, Val(i)) ⊗ backend_eye(be, nₓ)

# 3D, on an nₓ × n_y × n_z grid
shift(Ωₕ, Val(3), Val(i))  ==  _Eye(be, n_z, Val(i)) ⊗ backend_eye(be, nₓ * n_y)
```

`i == 0` short-circuits to the identity of the whole grid without building any Kronecker
product.

This is the building block of the difference, jump and average matrices: a backward
difference is `shift(Ωₕ, dim, Val(0)) - shift(Ωₕ, dim, Val(-1))`, and the other families
differ only in which pair of shifts they subtract or average.

See also: [`⊗`](@ref), [`diff₋ₓ`](@ref).
"""
function shift(Ωₕ::AbstractMeshType, ::Val{SHIFT_DIM}, ::Val{i}) where {SHIFT_DIM, i}
    if i == 0
        return backend_eye(backend(Ωₕ), npoints(Ωₕ))
    end

    return _recursive_shift(Ωₕ, Val(dim(Ωₕ)), Val(SHIFT_DIM), Val(i))
end

"""
    kronecker_operator_matrix(Ωₕ, op)

The Kronecker-product construction [`stencil_matrix`](@ref) replaces in every operator
family (gpena/Bramble.jl#185): `op` is one of the public per-axis aliases (`D₋ₓ`, `jumpᵧ`,
`M₂`, ...) and this returns the same matrix built the old way, out of [`shift`](@ref) and
its combinations (`difference_shift`, `add_half_shift`).

Kept as the retained oracle [`stencil_matrix`](@ref) is checked against, per the plan's own
departure note on gpena/Bramble.jl#185: the equality test needs an independent
construction, not a deprecated one. Declared here; a dispatch method is added next to each
family's own implementation (`difference.jl`, `average.jl`, `jump.jl`), mapping its public
aliases to the `_kron_*` construction that family kept.
"""
function kronecker_operator_matrix end

# ==============================================================================
# The index shifts `S₊` and `S₋` (gpena/Bramble.jl#352)
# ==============================================================================
#
# `S₊ₓ(uₕ)` is `u_{i+1}` and `S₋ₓ(uₕ)` is `u_{i-1}`, one grid point along the direction and
# no spacing: they relabel values, they do not measure anything. The one boundary slice
# with no neighbour reads 0, the convention `jump` uses (`D₊` instead writes 0 on that
# slice), so the grid function agrees with `shift(Ωₕ, Val(dim), Val(±1))`, whose missing
# diagonal entry is exactly that 0, and `S₊ₓ(uₕ) - uₕ == jumpₓ(uₕ)` holds on every point,
# the last one included.
#
# The traversal is the average engine's (`average.jl`): an interior pass reading the
# neighbour, a boundary pass writing 0, both banded along the grid's last axis under
# `CpuThreaded`/`CpuPolyester`. Only the per-point kernel differs. The serial engine is the
# single band of one, so every policy runs the same loop body and agrees bitwise.

# The neighbour, unchanged. `zero(cur)` on the boundary keeps the grid's element type.
@inline @propagate_inbounds _compute_shift(::GridDirection, ::Val{false}, cur, other) = other
@inline @propagate_inbounds _compute_shift(::GridDirection, ::Val{true}, cur) = zero(cur)

"""
    _shift_band!(out, in_ref, dims::NTuple{D, Int}, dir::GridDirection, dim_val::Val, nbands::Int, b::Int) -> Nothing

Writes the index shift of `in_ref` along `dim_val` in direction `dir` (`Forward` reads
`u_{i+1}`, `Backward` reads `u_{i-1}`) into `out`, on the `b`-th of `nbands` slabs of the grid
cut along its last axis, as [`_average_band!`](@ref) does for the average. The slice with no
neighbour gets 0. `nbands == 1` is the whole grid, which is the serial engine.
"""
@inline function _shift_band!(
        out, in_ref, dims::NTuple{D, Int}, dir::GridDirection, ::Val{DIM}, nbands::Int, b::Int
) where {D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    band = _band_range(axes(li, D), nbands, b)
    interior, boundary = _stencil_ranges(axes(li), Val(DIM), dir)
    interior, boundary = _band_slab(interior, band), _band_slab(boundary, band)

    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        out[idx] = _compute_shift(
            dir, Val(false), in_ref[idx], in_ref[li[_neighbour(dir, I, step)]]
        )
    end

    @inbounds @simd for I in CartesianIndices(boundary)
        idx = li[I]
        out[idx] = _compute_shift(dir, Val(true), in_ref[idx])
    end

    return nothing
end

@noinline function _throw_no_device_shift()
    throw(
        ArgumentError(
        "the index shifts (S₊ₓ, S₊ᵧ, S₊₂, S₋ₓ, S₋ᵧ, S₋₂, S₊ₕ, S₋ₕ) have no device kernel: " *
        "one is tracked on milestone v4.4.0. Apply them to a host-backed VectorElement.",
    ),
    )
end

# Every CPU policy goes through `_run_bands!` (`vector_calculus.jl`): the single band
# serially, one band per thread under `CpuThreaded`, one per `@batch` task under
# `CpuPolyester` through the generic `_batch_run_bands!` hook `BramblePolyesterExt` already
# fills, so the shifts need no extension hook of their own.
@inline _shift_engine!(policy::CpuPolicy, out, in_ref, dims, dir, dim_val) = _run_bands!(
    policy, _shift_band!, out, in_ref, dims, dir, dim_val)
@noinline _shift_engine!(::GpuPolicy, out, in_ref, dims, dir, dim_val) = _throw_no_device_shift()

# The applicator `@operator_family` calls. A device-backed element on a host policy is
# refused here too, before the host loop scalar-indexes it; a `GpuPolicy` space reaches
# `_shift_engine!`'s refusal. The direction check matters because a `Val` past the mesh
# dimension would otherwise make `_stencil_ranges` treat every point as both interior and
# boundary and return all zeros.
@inline function _apply_shifted!(
        vₕ::VectorElement{<:ScalarGridSpace},
        uₕ::VectorElement{<:ScalarGridSpace},
        dir::GridDirection,
        ::Val{DIM}
) where {DIM}
    _check_no_alias(vₕ, uₕ)
    _check_same_grid(vₕ, uₕ)
    (locality(typeof(vₕ.data)) isa DeviceLocality ||
     locality(typeof(uₕ.data)) isa DeviceLocality) && _throw_no_device_shift()
    dims = _grid_dims(uₕ)
    1 <= DIM <= length(dims) || _throw_stencil_dim_error(DIM, length(dims))
    _shift_engine!(execution_policy(space(uₕ)), vₕ.data, uₕ.data, dims, dir, Val(DIM))
    return vₕ
end

# Componentwise, re-deriving each leaf's grid, as `_apply_averaged!` does.
@inline function _apply_shifted!(
        vₕ::VectorElement{<:CompositeGridSpace},
        uₕ::VectorElement{<:CompositeGridSpace},
        dir::GridDirection,
        dim_val::Val
)
    _apply_componentwise!((v, u) -> _apply_shifted!(v, u, dir, dim_val), vₕ, uₕ)
    return vₕ
end

@inline function _shift_matrix(Ωₕ::AbstractMeshType, ::Val{DIM}, amount::Val) where {DIM}
    1 <= DIM <= dim(Ωₕ) || _throw_stencil_dim_error(DIM, dim(Ωₕ))
    return shift(Ωₕ, Val(DIM), amount)
end

"""
    forward_shift(arg, dim_val::Val) -> AbstractMatrix or VectorElement

The forward index shift along direction `dim_val`, ``(S_+ u)_i = u_{i+1}``.

It relabels values one grid point along the direction and involves no spacing, so it is the
same on uniform and non-uniform meshes. The last point along the direction has no forward
neighbour and reads 0, the convention of [`jumpₓ`](@ref) (unlike [`D₊ₓ`](@ref), which is
0 at that point); hence
``S_+ u - u`` is the jump ``u_{i+1} - u_i`` at every point, the last one included, and the
matrix of `forward_shift` is the transpose of that of [`backward_shift`](@ref).

# Arguments
- `arg`: A mesh `Ωₕ`, a grid space `Wₕ` or a [`VectorElement`](@ref) `uₕ`, scalar or composite
  (componentwise on the latter), or a symbolic operand of a form (a `LazyOp` such as the
  trial function `u` or `D₋ₓ(u)`).
- `dim_val`: The direction, `Val(1)`, `Val(2)` or `Val(3)`. On a form operand the entry
  points `S₊`/`S₋` take this `Val` form only; their `Int` and `Symbol` directions are for a
  mesh, a space or a grid function, since the direction is part of the node's type.

# Returns
- For a mesh or a grid space, the `npoints(Ωₕ) × npoints(Ωₕ)` matrix `shift(Ωₕ, dim_val,
  Val(1))`, in the backend's `matrix_type`, with ones on the superdiagonal of that direction.
- For a `VectorElement`, a new `VectorElement` of the same space holding ``u_{i+1}``.
- For a `LazyOp`, the `ShiftNode` that assembles ``u_{i+1}`` inside a form.

# Throws
- `ArgumentError`: `dim_val` is not between 1 and the mesh (or operand) dimension; or `uₕ` is
  device-backed or its space has a [`GpuPolicy`](@ref) (no device kernel yet, tracked on
  milestone v4.4.0).

# Examples
```jldoctest
using Bramble
using Bramble: forward_shift
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 5, true))
uₕ = Rₕ(Wₕ, x -> 4x)
parent(forward_shift(uₕ, Val(1)))

# output
5-element Vector{Float64}:
 1.0
 2.0
 3.0
 4.0
 0.0
```

See also: [`backward_shift`](@ref), [`S₊ₕ`](@ref), [`jump`](@ref).
"""
@inline forward_shift(Ωₕ::AbstractMeshType, dim_val::Val) = _shift_matrix(Ωₕ, dim_val, Val(1))
@inline forward_shift(Wₕ::AbstractSpaceType, dim_val::Val) = forward_shift(mesh(Wₕ), dim_val)

"""
    backward_shift(arg, dim_val::Val) -> AbstractMatrix or VectorElement

The backward index shift along direction `dim_val`, ``(S_- u)_i = u_{i-1}``.

It involves no spacing. The first point along the direction has no backward neighbour and
reads 0, so ``u - S_- u`` is the unscaled backward difference ``u_i - u_{i-1}`` at every
point, the first one reading ``u_1``, and the matrix of `backward_shift` is the transpose of
that of [`forward_shift`](@ref).

# Arguments
- `arg`: A mesh `Ωₕ`, a grid space `Wₕ` or a [`VectorElement`](@ref) `uₕ`, scalar or composite
  (componentwise on the latter), or a symbolic operand of a form (a `LazyOp` such as the
  trial function `u` or `D₋ₓ(u)`).
- `dim_val`: The direction, `Val(1)`, `Val(2)` or `Val(3)`. On a form operand the entry
  points `S₊`/`S₋` take this `Val` form only; their `Int` and `Symbol` directions are for a
  mesh, a space or a grid function, since the direction is part of the node's type.

# Returns
- For a mesh or a grid space, the `npoints(Ωₕ) × npoints(Ωₕ)` matrix `shift(Ωₕ, dim_val,
  Val(-1))`, in the backend's `matrix_type`, with ones on the subdiagonal of that direction.
- For a `VectorElement`, a new `VectorElement` of the same space holding ``u_{i-1}``.
- For a `LazyOp`, the `ShiftNode` that assembles ``u_{i-1}`` inside a form.

# Throws
- `ArgumentError`: `dim_val` is not between 1 and the mesh (or operand) dimension; or `uₕ` is
  device-backed or its space has a [`GpuPolicy`](@ref) (no device kernel yet, tracked on
  milestone v4.4.0).

# Examples
```jldoctest
using Bramble
using Bramble: backward_shift
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 5, true))
uₕ = Rₕ(Wₕ, x -> 4x)
parent(backward_shift(uₕ, Val(1)))

# output
5-element Vector{Float64}:
 0.0
 0.0
 1.0
 2.0
 3.0
```

See also: [`forward_shift`](@ref), [`S₋ₕ`](@ref).
"""
@inline backward_shift(Ωₕ::AbstractMeshType, dim_val::Val) = _shift_matrix(Ωₕ, dim_val, Val(-1))
@inline backward_shift(Wₕ::AbstractSpaceType, dim_val::Val) = backward_shift(mesh(Wₕ), dim_val)

@operator_family(base=forward_shift,
    stem=S₊,
    apply_fn=_apply_shifted!,
    direction=Forward(),
    opening_sentence="The forward index shift along the `{direction}` direction, "*
    "``(S_+ u)_i = u_{i+1}``.",
    trailing_note="The last point along `{direction}` has no forward neighbour and reads 0, "*
    "so `S₊{suffix}(uₕ) - uₕ` equals [`jump{suffix}`](@ref)`(uₕ)` everywhere.",
    bang_opening_sentence="The forward index shift along the `{direction}` direction, "*
    "``(S_+ u)_i = u_{i+1}``, written into `vₕ`.",
    vectorial_alias=S₊ₕ,
    vectorial_dir_string="forward",
    vectorial_what="index shift")

@operator_family(base=backward_shift,
    stem=S₋,
    apply_fn=_apply_shifted!,
    direction=Backward(),
    opening_sentence="The backward index shift along the `{direction}` direction, "*
    "``(S_- u)_i = u_{i-1}``.",
    trailing_note="The first point along `{direction}` has no backward neighbour and reads "*
    "0; the matrix is the transpose of [`S₊{suffix}`](@ref)'s.",
    bang_opening_sentence="The backward index shift along the `{direction}` direction, "*
    "``(S_- u)_i = u_{i-1}``, written into `vₕ`.",
    vectorial_alias=S₋ₕ,
    vectorial_dir_string="backward",
    vectorial_what="index shift")

# ==============================================================================
# ==============================================================================
# The AST nodes: the stencil shift `ShiftNode` and its constructor `shift_op`
# ==============================================================================
# ==============================================================================

"""
    ShiftNode{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a stencil shift operation by `shift_amount` grid points in dimension `Dim`.

Built by `shift_op` and, with `shift_amount` ``\\pm 1``, by the public shifts
`S₊ₓ(op)`, `S₋ᵧ(op)`, ... on a symbolic operand. A neighbour off the grid reads 0, as for a
grid function.
"""
struct ShiftNode{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    shift_amount::Int
    inner_op::OpType
end

"""
    shift_op(op::LazyOp{D}, dim::Int, amount::Int) where D

Shifts the stencil of `op` by `amount` grid points in dimension `dim`.
"""
function shift_op(op::LazyOp{D}, dim::Int, amount::Int) where {D}
    return ShiftNode{D, dim, typeof(op)}(amount, op)
end

# The public shifts on a symbolic operand (gpena/Bramble.jl#352): `S₊ₓ(u)` is
# `forward_shift(u, Val(1))`, the node `shift_op(u, 1, 1)` builds but with the direction read
# off a `Val`, so the node type is known to the compiler rather than chosen from a runtime
# `Int`. The entry points `S₊`/`S₋` take the `Val` form only, as every node family's does
# (`node_family.jl`), and the vectorial `S₊ₕ`/`S₋ₕ` give one node per direction, the node
# itself in one dimension, as `∇ₕ` does.
# A direction past `D` is refused here, as `_apply_shifted!` refuses it for a grid function:
# the node would otherwise assemble as the identity.
@inline function forward_shift(op::LazyOp{D}, ::Val{Dim}) where {D, Dim}
    1 <= Dim <= D || _throw_stencil_dim_error(Dim, D)
    return ShiftNode{D, Dim, typeof(op)}(1, op)
end
@inline function backward_shift(op::LazyOp{D}, ::Val{Dim}) where {D, Dim}
    1 <= Dim <= D || _throw_stencil_dim_error(Dim, D)
    return ShiftNode{D, Dim, typeof(op)}(-1, op)
end

@inline S₊(op::LazyOp, dim_val::Val) = forward_shift(op, dim_val)
@inline S₋(op::LazyOp, dim_val::Val) = backward_shift(op, dim_val)

@inline S₊ₕ(op::LazyOp{1}) = forward_shift(op, Val(1))
@inline S₊ₕ(op::LazyOp{D}) where {D} = ntuple(d -> forward_shift(op, Val(d)), Val(D))
@inline S₋ₕ(op::LazyOp{1}) = backward_shift(op, Val(1))
@inline S₋ₕ(op::LazyOp{D}) where {D} = ntuple(d -> backward_shift(op, Val(d)), Val(D))

@inline function local_stencil(
        op::ShiftNode{D, Dim}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return _shift_node_stencil(
        stencil_shift_trait(op.inner_op), op, inner, space, I, markers
    )
end

@inline _shift_node_stencil(
    ::TranslationInvariantStencil,
    op::ShiftNode{D, Dim},
    inner,
    space,
    I::CartesianIndex{D},
    markers
) where {D, Dim} = shifted_inner_stencil(
    op.inner_op, inner, space, I, markers, Val(Dim), op.shift_amount
)

# `ShiftNode` has no mask of its own: every other wrapper that reaches a neighbour
# (differences, averages, jumps) computes one first and multiplies a clamped boundary read by
# it, which is what makes `_clamped_shift`'s "clamp now, a zero mask absorbs it" contract safe
# for them. For an operand with offsets, `_reevaluated_shift` (`ast/common.jl`) supplies that
# zero itself where the clamp bites. Nothing there would absorb it for a source: a source has
# already been reduced to a value by the time this runs, with no offset left to relabel. A
# source shifted off the grid therefore reads as zero here, an empty stencil, the same
# "missing neighbour is zero" convention the masked stencils use, mirroring how
# `RegionRestriction` already spells "contributes nothing here".
#
# An interpolation is not a source: it is re-evaluated at the clamped point like any other
# operand, which is harmless inside a masked tap. A shift has no mask, so off the grid it
# zeroes the re-evaluated stencil itself, keeping the tuple length; the shift then reads 0
# there over an interpolation too, as it does over a grid function (gpena/Bramble.jl#352).
# `locate_cell`'s own clamp is a different matter, the interpolant's extrapolation at a
# point that is on the target grid, and is unchanged.
@inline function _shift_node_stencil(
        ::PointDependentStencil,
        op::ShiftNode{D, Dim},
        inner,
        space,
        I::CartesianIndex{D},
        markers
) where {D, Dim}
    if _is_source_only(op.inner_op)
        Ishift = I + _stencil_step(Val(Dim), Val(D)) * op.shift_amount
        _in_grid(space, Ishift) || return ()
        return local_stencil(
            op.inner_op, space, Ishift, markers, LinearIndices(indices(mesh(space)))[Ishift]
        )
    else
        Ishift = I + _stencil_step(Val(Dim), Val(D)) * op.shift_amount
        T = eltype(space)
        return scale_stencil(
            shifted_inner_stencil(
                op.inner_op, inner, space, I, markers, Val(Dim), op.shift_amount
            ),
            _in_grid(space, Ishift) ? one(T) : zero(T)
        )
    end
end

function resolve_ast(op::ShiftNode{D, Dim}) where {D, Dim}
    inner = resolve_ast(op.inner_op)
    return ShiftNode{D, Dim, typeof(inner)}(op.shift_amount, inner)
end

# `ShiftNode` carries a second field, so it writes its own binder rather than taking the one
# `@node_family` generates (operators/interpolation.jl explains the pass).
function _bind_interp_spaces(
        op::ShiftNode{D, Dim}, trial_leaf, test_leaf
) where {D, Dim}
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    return ShiftNode{D, Dim, typeof(inner)}(op.shift_amount, inner)
end

function expression(op::ShiftNode{D, Dim}) where {D, Dim}
    "shift($(expression(op.inner_op)), $(_BRAMBLE_var2symbol[Dim]), $(op.shift_amount))"
end
