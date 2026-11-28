# kronecker_oracle.jl
#
# The Kronecker construction of the operator matrices (gpena/Bramble.jl#185): each family's
# matrix built out of `Bramble.shift` and a diagonal scaling, independent of
# `stencil_matrix`, so the tests can check the one against the other entrywise. Only the
# tests and `benchmark/operator_matrices.jl` read it, which is why it lives here and not
# in `src/`. A test file or a benchmark script includes it; it defines `KroneckerOracle`
# and exports `kronecker_operator_matrix`.
#
# A new operator family needs a `_kron_*` builder here and a dispatch method at the end,
# or its `stencil_matrix` has nothing to be checked against.

module KroneckerOracle

using Bramble
using Bramble: shift, AbstractMeshType, Forward, Backward, Centered, GridDirection,
               npoints, spacings, dim, backend, _check_stencil_dim,
               _throw_centered_too_few_points, forward_spacing_for_derivative,
               spacing_for_derivative, _BRAMBLE_var2symbol, __vector
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, findnz, sparse

export kronecker_operator_matrix

"""
    kronecker_operator_matrix(Ωₕ, op)

The Kronecker-product construction `stencil_matrix` replaces in every operator family:
`op` is one of the public per-axis aliases (`D₋ₓ`, `jumpᵧ`, `M₂`, ...) and this returns
the same matrix built out of `Bramble.shift` and its combinations (`difference_shift`,
`add_half_shift`). The equality test needs an independent construction, which is why it is
kept. A dispatch method at the end of this file maps each public alias to its `_kron_*`
builder.
"""
function kronecker_operator_matrix end

# --- Diagonal scaling of an operator matrix ----------------------------------------- #
# Every weighted operator in this subsystem is a diagonal scaling of an unscaled one: build
# the matrix out of shifts, then multiply row `i` by a weight that depends on `i` alone.
#
# `w .* A` is the obvious spelling of that and the wrong one. Broadcasting a dense vector
# against a `SparseMatrixCSC` sizes the result's `rowval` and `nzval` buffers for the dense
# `n x n` case and shrinks them to the sparse result afterwards, and shrinking an array does
# not release the `Memory` behind it. What comes back is numerically right and reports the
# right `nnz`, `length(nzval)` and `sizeof(nzval)`, but it carries two buffers of `n^2`
# elements for as long as the operator lives. Measured on a 100 x 100 mesh: `D₋ₓ` then had
# 19800 stored entries and a `Base.summarysize` of 1.51 GiB, against 400 KB for the unscaled
# `backward_difference` it scales. `Base.summarysize` is the only size that shows it.
#
# Scaling the stored entries in place touches the nonzeros alone. `rowvals(A)[k]` is the row
# owning stored value `k`, whichever column it sits in, and that row is exactly the index
# into the weight vector. Every caller passes a matrix `shift` has just built for it, so
# mutating that matrix in place is safe.
#
# A truncated boundary slice is expressed as a zero weight, and the entries that weight
# annihilates stay stored as explicit zeros. The pattern belongs to the stencil, not to its
# values: pruning them would make it depend on the mesh, and a pattern that changes changes
# identity behind any cache keyed on it (#534's replay). `stencil_matrix` stores the same
# entries, so the two constructions agree on `nnz` as well as on values.
#
# One method with an `isa` branch, rather than the pair of methods this subsystem would
# otherwise reach for. The branch is on the argument's *type*, so it folds away wherever the
# backend's matrix type is known, and the method returns `A` itself, which keeps the return
# type equal to the argument type. A `SparseMatrixCSC`/`AbstractMatrix` pair instead joins
# the two returns to `AbstractArray` wherever the matrix type is not known statically, and
# JET reads that back as a missing `innerₕ` method in `_pc_form_stencils`, whose operand is
# untyped.
@inline function _scale_rows!(A::AbstractMatrix, w::AbstractVector)
    @boundscheck length(w) == size(A, 1) ||
                 throw(DimensionMismatch("the weight vector has $(length(w)) entries and the operator has $(size(A, 1)) rows"))

    if A isa SparseMatrixCSC
        rows = rowvals(A)
        nz = nonzeros(A)

        @inbounds for k in eachindex(nz)
            nz[k] *= w[rows[k]]
        end
    else
        # A dense or GPU backend has no stored-entry list to walk, and nothing is saved by
        # avoiding the full matrix: the broadcast is in place and allocates nothing.
        A .= w .* A
    end

    return A
end

# --- Differences --------------------------------------------------------------------- #

function difference_shift(
        Ωₕ::AbstractMeshType, ::Val{DIFF_DIM}, ::Val{first}, ::Val{second}
) where {DIFF_DIM, first, second}
    return shift(Ωₕ, Val(DIFF_DIM), Val(first)) - shift(Ωₕ, Val(DIFF_DIM), Val(second))
end

function _difference_operator(
        Ωₕ::AbstractMeshType, ::Forward, ::Val{DIFF_DIM}
) where {DIFF_DIM}
    return difference_shift(Ωₕ, Val(DIFF_DIM), Val(1), Val(0))
end

function _difference_operator(
        Ωₕ::AbstractMeshType, ::Backward, ::Val{DIFF_DIM}
) where {DIFF_DIM}
    return difference_shift(Ωₕ, Val(DIFF_DIM), Val(0), Val(-1))
end

# `spacing_func::F` is a type parameter so the per-point call specialises; untyped, every
# grid point pays a dynamic dispatch.
#
# It is the per-index function rather than the mesh's cached spacing vector on purpose.
# The vector's truncated entry is not meaningful, whereas `spacing_for_derivative` returns
# zero there, and this loop covers every index and turns that zero into a zero weight.
function _derivative_weights!(
        v::AbstractVector, Ωₕ::AbstractMeshType, spacing_func::F, ::Val{DIFF_DIM}
) where {F, DIFF_DIM}
    dims = npoints(Ωₕ, Tuple)

    _check_stencil_dim(DIFF_DIM, dim(Ωₕ))

    sub = Ωₕ(DIFF_DIM)
    li = LinearIndices(dims)

    @inbounds @simd for I in CartesianIndices(dims)
        x = spacing_func(sub, I[DIFF_DIM])
        v[li[I]] = iszero(x) ? zero(eltype(v)) : inv(x)
    end
    return nothing
end

@inline function forward_derivative_weights!(
        v::AbstractVector, Ωₕ::AbstractMeshType, diff_dim::Val
)
    _derivative_weights!(v, Ωₕ, forward_spacing_for_derivative, diff_dim)
end

@inline function backward_derivative_weights!(
        v::AbstractVector, Ωₕ::AbstractMeshType, diff_dim::Val
)
    _derivative_weights!(v, Ωₕ, spacing_for_derivative, diff_dim)
end

for (dir_instance, diff_name, finite_diff_name, weights_func!) in (
    (Forward(), :forward_difference, :forward_finite_difference, :forward_derivative_weights!),
    (Backward(), :backward_difference, :backward_finite_difference, :backward_derivative_weights!)
)
    @eval begin
        @inline $(Symbol(:_kron_, diff_name))(Ωₕ::AbstractMeshType, dim_val::Val) = _difference_operator(Ωₕ, $dir_instance, dim_val)

        function $(Symbol(:_kron_, finite_diff_name))(
                Ωₕ::AbstractMeshType, dim_val::Val; vector_cache = __vector(Ωₕ)
        )
            diff_matrix = $(Symbol(:_kron_, diff_name))(Ωₕ, dim_val)
            $weights_func!(vector_cache, Ωₕ, dim_val)
            return _scale_rows!(diff_matrix, vector_cache)
        end
    end
end

# The weights of the three centred families, each a diagonal scaling of unscaled
# difference matrices:
#
#     D̃ = diag(2/(hᵢ + hᵢ₊₁))                  · (shift₊₁ - shift₀)
#     Dc     = diag(1/(hᵢ + hᵢ₊₁))                  · (shift₊₁ - shift₋₁)
#     D̽ₕ     = diag(hᵢ/((hᵢ+hᵢ₊₁)hᵢ₊₁))             · diff₊
#            + diag(hᵢ₊₁/((hᵢ+hᵢ₊₁)hᵢ))             · diff₋
#
# The cross-weighted one falls out of its own definition: it is D₋ at xᵢ₊₁ weighted by hᵢ
# and D₋ at xᵢ weighted by hᵢ₊₁, over their sum, and D₋(u)ᵢ₊₁ is diff₊(u)ᵢ/hᵢ₊₁ while
# D₋(u)ᵢ is diff₋(u)ᵢ/hᵢ. So the two weights above are what is left after dividing through.
#
# The weights read the mesh's cached `spacings` rather than `spacing_for_derivative`. The
# cached vector repeats the first interval in h₁ instead of zeroing it, and that repeated
# value is the one the grid-function kernels use, so reading it is what makes the matrix
# agree with them at the first point. Truncation is applied by index here instead, which
# is also what the form layer's stencils do.

# Returns `w`, as a mutating function with a single destination does, so that the builders
# below can write `_scale_rows!(matrix, _extended_weights!(cache, …))` rather than filling
# the cache on one line and reaching for it on the next.
@inline function _extended_weights!(
        w::AbstractVector, Ωₕ::AbstractMeshType, ::Val{DIFF_DIM}, weight::F
) where {F, DIFF_DIM}
    _check_stencil_dim(DIFF_DIM, dim(Ωₕ))

    dims = npoints(Ωₕ, Tuple)
    h = spacings(Ωₕ(DIFF_DIM))
    n = dims[DIFF_DIM]
    li = LinearIndices(dims)

    @inbounds for I in CartesianIndices(dims)
        w[li[I]] = weight(h, I[DIFF_DIM], n)
    end
    return w
end

# `_star_weight`/`_centered_weight` return zero wherever their stencil would need a
# neighbour the grid does not have, truncating that slice of the matrix to a zero row.
@inline _star_weight(h, i, n) = i == n ? zero(eltype(h)) : 2 / (h[i] + h[i + 1])
@inline _centered_weight(h, i, n) = (i == 1 || i == n) ? zero(eltype(h)) : inv(h[i] + h[i + 1])

# The cross-weighted pair instead falls back to the one-sided difference on whichever
# side is still on the grid at each end: `diff₊`'s row 1 is already `u_2 - u_1` and
# `diff₋`'s row `n` is already `u_n - u_{n-1}` (the unscaled matrices' own boundary
# convention, `_difference_operator`), so weighting those rows by `1/h[1]` and `1/h[n]`
# respectively -- with the other family's row zeroed there -- reproduces `D₊`/`D₋`
# exactly, matching the grid-function engine's boundary case (gpena/Bramble.jl#183).
@inline _cross_forward_weight(h, i, n) = i == n ? zero(eltype(h)) :
                                         (i == 1 ? inv(h[1]) : h[i] / ((h[i] + h[i + 1]) * h[i + 1]))
@inline _cross_backward_weight(h, i, n) = i == 1 ? zero(eltype(h)) :
                                          (i == n ? inv(h[n]) : h[i + 1] / ((h[i] + h[i + 1]) * h[i]))

function _kron_forward_star_difference(
        Ωₕ::AbstractMeshType, dim_val::Val; vector_cache = __vector(Ωₕ)
)
    w = _extended_weights!(vector_cache, Ωₕ, dim_val, _star_weight)
    return _scale_rows!(_difference_operator(Ωₕ, Forward(), dim_val), w)
end

function _kron_centered_difference(
        Ωₕ::AbstractMeshType, dim_val::Val{DIM}; vector_cache = __vector(Ωₕ)
) where {DIM}
    n = npoints(Ωₕ(DIM))
    n >= 3 || _throw_centered_too_few_points(DIM, n)

    w = _extended_weights!(vector_cache, Ωₕ, dim_val, _centered_weight)
    return _scale_rows!(difference_shift(Ωₕ, dim_val, Val(1), Val(-1)), w)
end

function _kron_cross_weighted_difference(
        Ωₕ::AbstractMeshType, dim_val::Val{DIM}; vector_cache = __vector(Ωₕ)
) where {DIM}
    n = npoints(Ωₕ(DIM))
    n >= 3 || _throw_centered_too_few_points(DIM, n)

    forward = _scale_rows!(_difference_operator(Ωₕ, Forward(), dim_val),
        _extended_weights!(vector_cache, Ωₕ, dim_val, _cross_forward_weight))

    # the scaling above is already applied, so the cache is free to be rewritten
    backward = _scale_rows!(_difference_operator(Ωₕ, Backward(), dim_val),
        _extended_weights!(vector_cache, Ωₕ, dim_val, _cross_backward_weight))

    # A dense backend has no pattern to keep. On a sparse one `+` would prune every entry
    # where the two cancel (all of the diagonal on a uniform mesh); one `sparse` call over
    # both triplet lists sums them on the union pattern instead.
    forward isa SparseMatrixCSC || return forward + backward
    i₊, j₊, v₊ = findnz(forward)
    i₋, j₋, v₋ = findnz(backward)
    return sparse(vcat(i₊, i₋), vcat(j₊, j₋), vcat(v₊, v₋), size(forward)...)
end

# --- Averages ------------------------------------------------------------------------ #

# Divided by 2 rather than multiplied by 0.5: the literal is a Float64 and promotes the
# whole matrix. On a Float32 backend everything else in the library stayed Float32 and
# only the averaging matrices came back Float64.
function add_half_shift(
        Ωₕ::AbstractMeshType, ::Val{DIFF_DIM}, ::Val{first}, ::Val{second}
) where {DIFF_DIM, first, second}
    return (shift(Ωₕ, Val(DIFF_DIM), Val(first)) + shift(Ωₕ, Val(DIFF_DIM), Val(second))) /
           2
end

# Forward and Backward differ in the offset of the second shift alone.
function _average_operator(
        Ωₕ::AbstractMeshType, dir::Union{Forward, Backward}, ::Val{AVG_DIM}
) where {AVG_DIM}
    return add_half_shift(Ωₕ, Val(AVG_DIM), Val(0), Val(dir isa Forward ? 1 : -1))
end

function _average_operator(Ωₕ::AbstractMeshType, ::Centered, ::Val{AVG_DIM}) where {AVG_DIM}
    return (shift(Ωₕ, Val(AVG_DIM), Val(-1)) + 2 * shift(Ωₕ, Val(AVG_DIM), Val(0)) +
            shift(Ωₕ, Val(AVG_DIM), Val(1))) / 4
end

# The centered average's rows are one away from both end slices and zero on them, the
# weight vector of the centered average.
function _average_weights!(
        v::AbstractVector, Ωₕ::AbstractMeshType, ::Centered, ::Val{DIFF_DIM}
) where {DIFF_DIM}
    dims = npoints(Ωₕ, Tuple)
    _check_stencil_dim(DIFF_DIM, dim(Ωₕ))
    li = LinearIndices(dims)
    n = dims[DIFF_DIM]

    @inbounds for I in CartesianIndices(dims)
        v[li[I]] = (I[DIFF_DIM] == 1 || I[DIFF_DIM] == n) ? zero(eltype(v)) : one(eltype(v))
    end
    return nothing
end

function _average_weights!(
        v::AbstractVector, Ωₕ::AbstractMeshType, dir::GridDirection, dim_val::Val{DIFF_DIM}
) where {DIFF_DIM}
    dims = npoints(Ωₕ, Tuple)

    _check_stencil_dim(DIFF_DIM, dim(Ωₕ))

    li = LinearIndices(dims)

    @inbounds @simd for I in CartesianIndices(dims)
        idx = I[DIFF_DIM]
        if dir isa Forward
            v[li[I]] = idx == dims[DIFF_DIM] ? zero(eltype(v)) : one(eltype(v))
        else # Backward
            v[li[I]] = idx == 1 ? zero(eltype(v)) : one(eltype(v))
        end
    end
    return nothing
end

for (dir_instance, average_name) in (
    (Forward(), :forward_average), (Backward(), :backward_average),
    (Centered(), :centered_average)
)
    @eval function $(Symbol(:_kron_, average_name))(
            Ωₕ::AbstractMeshType, dim_val::Val; vector_cache = __vector(Ωₕ)
    )
        avg_matrix = _average_operator(Ωₕ, $dir_instance, dim_val)
        _average_weights!(vector_cache, Ωₕ, $dir_instance, dim_val)
        return _scale_rows!(avg_matrix, vector_cache)
    end
end

# --- Dispatch ------------------------------------------------------------------------ #
#
# Each public per-axis alias maps to its `_kron_*` construction. The jump forwards to the
# unscaled forward difference, as `jump` itself does at runtime.
for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
    for (stem, kron_fn) in (
        (:D₋, :_kron_backward_finite_difference),
        (:D₊, :_kron_forward_finite_difference),
        (:D̃, :_kron_forward_star_difference),
        (:Dc, :_kron_centered_difference),
        (:D̽, :_kron_cross_weighted_difference),
        (:M, :_kron_backward_average),
        (:M₊, :_kron_forward_average),
        (:Mc, :_kron_centered_average),
        (:jump, :_kron_forward_difference)
    )
        alias = Symbol(stem, suffix)
        @eval kronecker_operator_matrix(Ωₕ::AbstractMeshType, ::typeof(Bramble.$alias)) = $kron_fn(Ωₕ, Val($i))
    end
end

end # module KroneckerOracle
