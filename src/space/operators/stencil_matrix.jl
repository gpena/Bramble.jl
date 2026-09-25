##############################################################################
#                                                                            #
#                       Single-pass operator matrix synthesis               #
#                                                                            #
##############################################################################

#=
# stencil_matrix.jl

`stencil_matrix` and its two `_stencil_matrix` backends (the CSC two-pass writer and the
dense/device fallback), `_scale_rows!` (the diagonal row-scaling helper they and the weighted
operators in `difference.jl`/`average.jl` share), and `_stencil_taps`/`_stencil_weights`/
`_HostAxisSpacings`, the per-family tap and weight helpers only `_stencil_matrix` calls.
Moved out of `stencil.jl` (gpena/Bramble.jl#280) to keep that file to the shared traversal
and alias framework alone, plus the `StencilOp` type hierarchy other operator files still
tag their calls to `stencil_matrix` with; this is a pure move -- Julia does not care which
file a definition lives in within one module -- and `Bramble.jl` includes this file directly
after `stencil.jl`, still ahead of every file that calls into either.
=#

"""
    _stencil_taps(op::StencilOp) -> NTuple{K,Int}

The fixed neighbour offsets along `op`'s axis that its stencil reads, in the same order
[`_stencil_weights`](@ref) returns their coefficients. A tap whose neighbour falls outside
the grid contributes no entry at all (checked in [`stencil_matrix`](@ref)), rather than a
stored zero.
"""
@inline _stencil_taps(::BackwardFiniteDiffOp) = (0, -1)
@inline _stencil_taps(::ForwardFiniteDiffOp) = (1, 0)
@inline _stencil_taps(::UnscaledBackwardDiffOp) = (0, -1)
@inline _stencil_taps(::UnscaledForwardDiffOp) = (1, 0)
@inline _stencil_taps(::StarDiffOp) = (1, 0)
@inline _stencil_taps(::CenteredDiffOp) = (1, -1)
@inline _stencil_taps(::CrossWeightedDiffOp) = (1, 0, -1)
@inline _stencil_taps(::BackwardAvgOp) = (0, -1)
@inline _stencil_taps(::ForwardAvgOp) = (1, 0)
@inline _stencil_taps(::CenteredAvgOp) = (-1, 0, 1)

# --- Host mirror for the dense fallback's per-point weight reads --------------------- #
#
# `_stencil_weights` reads its mesh through four queries alone (checked against all nine
# methods below): the spacing at a point, the spacing to its forward neighbour, the point
# count along one axis, and the element type. For a device-backed mesh (Metal.jl), reading
# spacing straight off the mesh scalar-indexes a device array and is refused outright by the
# scalar-indexing guard -- not merely slow, an error before the dense fallback (below,
# `_stencil_matrix`) ever gets to write anything (gpena/Bramble.jl#94, measured in S2.6). The
# dense fallback only ever needs `op`'s own axis, so this mirrors *that axis alone* to the
# host once per `_stencil_matrix` call, via `host_spacings` (`mesh1d.jl`, gpena/Bramble.jl#94
# S2.10) -- one bulk transfer, not one scalar read per point, and free on a host-backed mesh
# since `host_spacings` returns the existing array there instead of copying it.
#
# It neither subtypes `AbstractMeshType` nor overloads `spacing`/`forward_spacing`/
# `npoints`/`eltype` (gpena/Bramble.jl#94, JET cleanup, S2.6): a first attempt did both, and
# `report_package` flagged it twice over, in two different ways. Subtyping `AbstractMeshType`
# while answering only four of its methods made every `AbstractMeshType`-typed function in
# the package -- not just `_stencil_weights` below -- pick this mirror up in a union split
# and report the methods it does not implement (`point`, `_mesh_version`, calling a mesh as
# `Ωₕ(dim)`) as missing. Dropping the supertype but keeping the four overloads traded that
# for a subtler version of the same problem: `report_package` widens an untyped argument's
# candidate type set to *every* concrete type answering the same call shape anywhere in the
# package, not only the ones related through `AbstractMeshType` -- so this mirror's own
# `npoints(m, ::Type{Tuple})` matched unrelated `npoints(Ωₕ, Tuple)` calls in
# `src/mesh/marker.jl`, `src/space/operators/restriction.jl` and `src/space/inner_product.jl`,
# widened those functions' inferred argument type to include it, and every *other* call on
# that same variable then flagged it missing too -- a new report in a different file for
# every fix, each patchable only by adding this mirror to an interface it has no business
# answering. `_axis_spacing`/`_axis_forward_spacing`/`_axis_npoints`/`_axis_eltype` below give
# `_stencil_weights` the same four queries under names nothing else in the package calls, so
# this mirror never becomes a candidate for a dispatch it was not built to answer.
struct _HostAxisSpacings{T, Dim}
    h::Vector{T}
end

@inline _axis_eltype(Ωₕ::AbstractMeshType) = eltype(Ωₕ)
@inline _axis_eltype(::_HostAxisSpacings{T, Dim}) where {T, Dim} = T

@inline _axis_npoints(Ωₕ::AbstractMeshType, dim::Int) = npoints(Ωₕ, Tuple)[dim]
@inline _axis_npoints(m::_HostAxisSpacings, ::Int) = length(m.h)

@inline _axis_spacing(Ωₕ::AbstractMeshType, I::CartesianIndex, dim::Int) = spacing(Ωₕ, I, dim)
@inline _axis_spacing(m::_HostAxisSpacings, I::CartesianIndex, dim::Int) = @inbounds m.h[I[dim]]

@inline _axis_forward_spacing(Ωₕ::AbstractMeshType, I::CartesianIndex, dim::Int) = forward_spacing(
    Ωₕ, I, dim
)
@inline function _axis_forward_spacing(m::_HostAxisSpacings, I::CartesianIndex, dim::Int)
    n = length(m.h)
    i = I[dim]
    return @inbounds m.h[i == n ? n : i + 1]
end

"""
    _stencil_weights(op::StencilOp{Dim}, Ωₕ::AbstractMeshType, I::CartesianIndex) -> NTuple{K}

The coefficients matching [`_stencil_taps`](@ref)`(op)` at `I`. The unscaled families'
self-weight is always ±1 regardless of boundary -- their truncation is entirely the "tap
falls outside the grid" exclusion `stencil_matrix` applies -- while the scaled and averaged
families additionally zero their own weight at a boundary that a Kronecker construction
would reach through a since-zeroed weight vector rather than exclude, which this reproduces
explicitly.

`Ωₕ` is typed as `Union{AbstractMeshType, _HostAxisSpacings}` rather than `AbstractMeshType`
alone: every method below reads its mesh through `_axis_spacing`/`_axis_forward_spacing`/
`_axis_npoints`/`_axis_eltype` only, and `_HostAxisSpacings` answers exactly those
four without claiming to be a mesh (see its own docstring).
"""
@inline _stencil_weights(
    ::UnscaledBackwardDiffOp, Ωₕ::Union{AbstractMeshType, _HostAxisSpacings}, I::CartesianIndex
) = (1, -1)
@inline _stencil_weights(
    ::UnscaledForwardDiffOp, Ωₕ::Union{AbstractMeshType, _HostAxisSpacings}, I::CartesianIndex
) = (1, -1)

@inline function _stencil_weights(
        ::BackwardFiniteDiffOp{Dim}, Ωₕ::Union{AbstractMeshType, _HostAxisSpacings}, I::CartesianIndex
) where {Dim}
    h = _axis_spacing(Ωₕ, I, Dim)
    mask = I[Dim] == 1 ? 0 : 1
    return (mask / h, -mask / h)
end

@inline function _stencil_weights(
        ::ForwardFiniteDiffOp{Dim}, Ωₕ::Union{AbstractMeshType, _HostAxisSpacings}, I::CartesianIndex
) where {Dim}
    n = _axis_npoints(Ωₕ, Dim)
    h = _axis_forward_spacing(Ωₕ, I, Dim)
    mask = I[Dim] == n ? 0 : 1
    return (mask / h, -mask / h)
end

@inline function _stencil_weights(
        ::StarDiffOp{Dim}, Ωₕ::Union{AbstractMeshType, _HostAxisSpacings}, I::CartesianIndex
) where {Dim}
    n = _axis_npoints(Ωₕ, Dim)
    mask = I[Dim] == n ? 0 : 1
    c = 2 * mask / (_axis_spacing(Ωₕ, I, Dim) + _axis_forward_spacing(Ωₕ, I, Dim))
    return (c, -c)
end

@inline function _stencil_weights(
        ::CenteredDiffOp{Dim}, Ωₕ::Union{AbstractMeshType, _HostAxisSpacings}, I::CartesianIndex
) where {Dim}
    n = _axis_npoints(Ωₕ, Dim)
    mask = (I[Dim] == 1 || I[Dim] == n) ? 0 : 1
    c = mask / (_axis_spacing(Ωₕ, I, Dim) + _axis_forward_spacing(Ωₕ, I, Dim))
    return (c, -c)
end

@inline function _stencil_weights(
        ::CrossWeightedDiffOp{Dim}, Ωₕ::Union{AbstractMeshType, _HostAxisSpacings}, I::CartesianIndex
) where {Dim}
    n = _axis_npoints(Ωₕ, Dim)
    if I[Dim] == 1
        a = inv(_axis_spacing(Ωₕ, I, Dim))
        return (a, -a, zero(a))
    elseif I[Dim] == n
        b = inv(_axis_spacing(Ωₕ, I, Dim))
        return (zero(b), b, -b)
    else
        h = _axis_spacing(Ωₕ, I, Dim)
        hf = _axis_forward_spacing(Ωₕ, I, Dim)
        total = h + hf
        a = h / (total * hf)
        b = hf / (total * h)
        return (a, b - a, -b)
    end
end

@inline function _stencil_weights(
        ::BackwardAvgOp{Dim}, Ωₕ::Union{AbstractMeshType, _HostAxisSpacings}, I::CartesianIndex
) where {Dim}
    T = _axis_eltype(Ωₕ)
    mask = I[Dim] == 1 ? zero(T) : T(1) / 2
    return (mask, mask)
end

@inline function _stencil_weights(
        ::ForwardAvgOp{Dim}, Ωₕ::Union{AbstractMeshType, _HostAxisSpacings}, I::CartesianIndex
) where {Dim}
    T = _axis_eltype(Ωₕ)
    n = _axis_npoints(Ωₕ, Dim)
    mask = I[Dim] == n ? zero(T) : T(1) / 2
    return (mask, mask)
end

# Both end slices lack a neighbour on one side, so both rows are zeroed, as `CenteredDiffOp`
# zeroes them.
@inline function _stencil_weights(
        ::CenteredAvgOp{Dim}, Ωₕ::Union{AbstractMeshType, _HostAxisSpacings}, I::CartesianIndex
) where {Dim}
    T = _axis_eltype(Ωₕ)
    n = _axis_npoints(Ωₕ, Dim)
    q = (I[Dim] == 1 || I[Dim] == n) ? zero(T) : T(1) / 4
    return (q, 2q, q)
end

# --- Single-pass operator matrices: stencil_matrix ----------------------------------- #
#
# gpena/Bramble.jl#185: every matrix form in difference.jl/average.jl/jump.jl built its
# result as a Kronecker product of 1D shift matrices (`shift`, `_recursive_shift`,
# `_difference_operator`, `_average_operator`, `add_half_shift`), each product and
# subtraction/sum along the way an intermediate `SparseMatrixCSC` of its own, built and
# thrown away. That is a second, independent implementation of every family's arithmetic
# next to the one `local_stencil` already carries for form assembly (issue #185's own
# complaint: "any change to operator boundary conventions requires manually keeping three
# independent codebases in sync"), and it is not the cheapest way to fill a matrix whose
# sparsity pattern is known ahead of a single stencil's reach.
#
# `stencil_matrix(Ωₕ, op)` replaces the Kronecker construction with one sweep over
# `CartesianIndices(Ωₕ)` that writes `colptr`/`rowval`/`nzval` directly: two passes (count,
# then fill), no intermediate matrix. `op` names one operator family along one axis --
# `_stencil_taps(op)` gives its fixed neighbour offsets along that axis, `_stencil_weights`
# the coefficients matching those offsets at a grid point.
#
# This mirrors, rather than calls, the offsets and arithmetic `local_stencil` uses for the
# same families in `src/ast/operators/{difference,average,jump}.jl` (their own
# `_stencil_taps`/`_stencil_weights`, keyed on AST node types such as `BackwardDifference`
# and `JumpNode`). Calling those directly would mean constructing a `LazyOp` tree from this
# file to stand in for the node's `inner_op` field, which `src/space/` has no business
# doing: forms are built on top of the space layer's operators, not the other way around,
# and `src/ast/` is out of scope for this subplan besides. The two are proved equal by the
# equality test against `kronecker_operator_matrix` below instead of by sharing code --
# reported as the duplication the plan anticipated rather than resolved.

# --- Diagonal scaling of an operator matrix ----------------------------------------- #
# Every weighted operator in this subsystem is a diagonal scaling of an unscaled one: build
# the matrix out of shifts, then multiply row `i` by a weight that depends on `i` alone.
#
# `w .* A` is the obvious spelling of that and the wrong one. Broadcasting a dense vector
# against a `SparseMatrixCSC` sizes the result's `rowval` and `nzval` buffers for the dense
# `n x n` case and shrinks them to the sparse result afterwards, and shrinking an array does
# not release the `Memory` behind it. What comes back is numerically right and reports the
# right `nnz`, `length(nzval)` and `sizeof(nzval)`, but it carries two buffers of `n^2`
# elements for as long as the operator lives. Measured on a 100 x 100 mesh: `D₋ₓ` has 19800
# stored entries and a `Base.summarysize` of 1.51 GiB, against 400 KB for the unscaled
# `backward_difference` it scales. `Base.summarysize` is the only size that shows it.
#
# Scaling the stored entries in place touches the nonzeros alone. `rowvals(A)[k]` is the row
# owning stored value `k`, whichever column it sits in, and that row is exactly the index
# into the weight vector. Every caller passes a matrix `shift` has just built for it, so
# mutating that matrix in place is safe.
#
# `dropzeros!` is not tidying: a truncated boundary slice is expressed as a zero weight, and
# the broadcast this replaces pruned the entries that weight annihilated. Keeping them would
# leave the operators with a wider stored pattern than they have always had.
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

        dropzeros!(A)
    else
        # A dense or GPU backend has no stored-entry list to walk, and nothing is saved by
        # avoiding the full matrix: the broadcast is in place and allocates nothing.
        A .= w .* A
    end

    return A
end

"""
    stencil_matrix(Ωₕ::AbstractMeshType, op::StencilOp)
    stencil_matrix(Wₕ::AbstractSpaceType, op::StencilOp)

Builds `op`'s operator matrix in one pass over `Ωₕ`'s grid points, in the matrix type
`matrix_type(backend(Ωₕ))` picked: a `SparseMatrixCSC` backend writes `colptr`/`rowval`/
`nzval` directly (two passes, count then fill; no intermediate matrix, no broadcast), any
other `AbstractMatrix` backend falls back to a plain dense fill, built on the host and
handed to the backend's matrix type in one `copyto!` -- never one scalar write per stored
entry into device memory (gpena/Bramble.jl#94). `Ωₕ` may be a mesh or a grid space, taken
as `mesh(Wₕ)`.

Checked entrywise, `nnz` included, against [`kronecker_operator_matrix`](@ref), the
Kronecker-product construction every operator family used before (gpena/Bramble.jl#185).
"""
@inline function stencil_matrix(Ωₕ::AbstractMeshType, op::StencilOp)
    return _stencil_matrix(matrix_type(backend(Ωₕ)), Ωₕ, op)
end

@inline stencil_matrix(Wₕ::AbstractSpaceType, op::StencilOp) = stencil_matrix(mesh(Wₕ), op)

function _stencil_matrix(
        ::Type{<:SparseMatrixCSC{Tv, Ti}}, Ωₕ::AbstractMeshType, op::StencilOp{Dim}
) where {Tv, Ti, Dim}
    dims = npoints(Ωₕ, Tuple)
    D = length(dims)
    n = dims[Dim]
    N = prod(dims)
    li = LinearIndices(dims)
    step = _stencil_step(Val(Dim), Val(D))
    taps = _stencil_taps(op)
    K = length(taps)

    # Pass 1: how many stored entries land in each column -- one per (row, tap) whose
    # neighbour stays on the grid *and* whose weight is not exactly zero, tallied at
    # `colptr[col + 1]` and turned into the standard CSC column-start array by the running
    # sum below.
    #
    # The weight check matches what the Kronecker oracle does: its own weighting step,
    # `_scale_rows!` (above), drops a stored entry outright when scaling takes it to exactly
    # zero, rather than keeping it as an explicit zero. That is what happens at a family's
    # zeroed boundary weight (a `BackwardFiniteDiffOp`'s row 1, an average's row 1 or `n`):
    # the entry is column-absent in the oracle, not merely zero-valued, so `stencil_matrix`
    # drops it too rather than storing a value the equality check's `nnz` comparison would
    # then disagree on.
    colptr = zeros(Ti, N + 1)
    @inbounds for I in CartesianIndices(dims)
        idim = I[Dim]
        w = _stencil_weights(op, Ωₕ, I)
        for k in 1:K
            j = idim + taps[k]
            (1 <= j <= n) || continue
            iszero(w[k]) && continue
            col = li[I + taps[k] * step]
            colptr[col + 1] += 1
        end
    end
    colptr[1] = 1
    @inbounds for col in 1:N
        colptr[col + 1] += colptr[col]
    end

    # Pass 2: fill, walking the same rows in the same order, so a column's entries land
    # sorted by row (row increases monotonically across the whole sweep, and each column's
    # entries are the subsequence of rows that happen to reach it). `colptr` itself doubles
    # as the fill cursor -- `colptr[col]` is incremented as each of the column's entries is
    # placed -- rather than a second copy of it, so this needs one array fewer than the
    # count-then-fill idiom usually does; the loop below it undoes the shift that leaves it
    # in afterwards.
    nz = colptr[N + 1] - 1
    rowval = Vector{Ti}(undef, nz)
    nzval = Vector{Tv}(undef, nz)

    @inbounds for I in CartesianIndices(dims)
        idim = I[Dim]
        w = _stencil_weights(op, Ωₕ, I)
        row = li[I]
        for k in 1:K
            j = idim + taps[k]
            (1 <= j <= n) || continue
            iszero(w[k]) && continue
            col = li[I + taps[k] * step]
            pos = colptr[col]
            rowval[pos] = row
            nzval[pos] = Tv(w[k])
            colptr[col] = pos + 1
        end
    end

    # `colptr[col]` (1 <= col <= N) now holds what `colptr[col + 1]` held before the fill
    # pass (each column's cursor walked from its start to its end, which is the next
    # column's start); shift it back into the column-start array `SparseMatrixCSC` expects.
    @inbounds for col in N:-1:2
        colptr[col] = colptr[col - 1]
    end
    colptr[1] = 1

    return SparseMatrixCSC{Tv, Ti}(N, N, colptr, rowval, nzval)
end

# --- Dense fallback: mirror the axis to the host, then copy in one shot --------------- #
#
# `_HostAxisSpacings`, defined above (before `_stencil_weights`, which is typed to accept
# it) alongside the other `StencilOp` scaffolding, is what the dense fallback below mirrors
# `op`'s axis onto.

function _stencil_matrix(
        ::Type{MT}, Ωₕ::AbstractMeshType, op::StencilOp{Dim}
) where {T, MT <: AbstractMatrix{T}, Dim}
    dims = npoints(Ωₕ, Tuple)
    D = length(dims)
    n = dims[Dim]
    N = prod(dims)
    li = LinearIndices(dims)
    step = _stencil_step(Val(Dim), Val(D))
    taps = _stencil_taps(op)
    K = length(taps)

    # Built on the host, where `setindex!` is a plain memory write, and handed to `MT` in
    # one `copyto!` -- not one device-side scalar write per stored entry (gpena/Bramble.jl#94;
    # see `_shift_ones` for the same trade). `mirror` (above) is what makes `_stencil_weights`
    # itself safe to call in this loop despite `Ωₕ` possibly being device-backed.
    mirror = _HostAxisSpacings{T, Dim}(host_spacings(Ωₕ(Dim)))
    host = zeros(T, N, N)
    @inbounds for I in CartesianIndices(dims)
        idim = I[Dim]
        w = _stencil_weights(op, mirror, I)
        row = li[I]
        for k in 1:K
            j = idim + taps[k]
            (1 <= j <= n) || continue
            iszero(w[k]) && continue
            host[row, li[I + taps[k] * step]] = T(w[k])
        end
    end
    A = MT(undef, N, N)
    copyto!(A, host)
    return A
end
