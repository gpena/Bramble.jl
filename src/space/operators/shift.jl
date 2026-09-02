"""
	⊗(A, B)

Kronecker product operator (alias for `kron`).

Computes the Kronecker product (tensor product) of matrices `A` and `B`.
This operator is used extensively in constructing multidimensional shift operators.

# Example

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
backend `be` chose -- routed through [`backend_eye`](@ref)/`matrix_type(be)` rather than a
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
@inline _Eye(be, npts::Int, ::Val{i}) where {i} = _shift_ones(
    matrix_type(be), npts, i, npts - abs(i))

# The three-way dispatch mirrors `_backend_eye` (backend.jl): a fast path for
# `SparseMatrixCSC` and for dense `Matrix`, and a generic scalar-indexing fallback for
# anything else. `spdiagm` alone carries the offset arithmetic, so the two fast paths only
# differ in whether the sparse result is converted afterwards.
@inline _shift_ones(::Type{<:SparseMatrixCSC{T, Ti}}, npts::Int, i::Int,
    nz::Int) where {T, Ti} = spdiagm(npts, npts, i => fill(one(T), nz))
@inline _shift_ones(::Type{<:Matrix{T}}, npts::Int, i::Int, nz::Int) where {T} = Matrix{T}(
    spdiagm(npts, npts, i => fill(one(T), nz)))
function _shift_ones(::Type{MT}, npts::Int, i::Int, nz::Int) where {
        T, MT <: AbstractMatrix{T}}
    A = MT(undef, npts, npts)
    fill!(A, zero(T))
    r0, c0 = i >= 0 ? (0, i) : (-i, 0)
    for k in 1:nz
        A[r0 + k, c0 + k] = one(T)
    end
    return A
end

@inline function _recursive_shift(
        Ωₕ::AbstractMeshType, ::Val{1}, ::Val{DIFF_DIM}, ::Val{i}) where {DIFF_DIM, i}
    dims = npoints(Ωₕ, Tuple)
    be = backend(Ωₕ)

    if DIFF_DIM == 1
        return _Eye(be, dims[1], Val(i))
    else
        return backend_eye(be, dims[1])
    end
end

@inline function _recursive_shift(
        Ωₕ::AbstractMeshType, ::Val{D}, ::Val{DIFF_DIM}, ::Val{i}) where {D, DIFF_DIM, i}
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

# The tensor-product structure

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
