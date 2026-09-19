##############################################################################
#                                                                            #
#                  Shared stencil traversal and alias framework              #
#                                                                            #
##############################################################################

#=
# stencil.jl

The one-sided grid traversal, argument validation and alias/docstring generators that
`difference.jl`, `average.jl`, `jump.jl` and `inner_product.jl` all build on.

Extracted from `difference.jl` (gpena/Bramble.jl#42): none of this is about differences in
particular, but until now `average.jl` and `jump.jl` had to depend on the largest file in
`src/`, about a different operator family, because there was nowhere better to point.
Included before all four consumers; Julia does not care which file a definition lives in
within one module, so this is a pure move with no runtime change.

What stays behind in `difference.jl` instead: everything actually specific to differencing
-- `CenteredStencil`/`Centered`/`CrossWeighted` (only the centered/cross-weighted families
use them), `_difference_engine!` and `_centered_stencil_ranges`, `_apply_stencil!` and
`_apply_spaced!` (the difference-family applicator, distinct from the generic
`_apply_componentwise!` below), and `_throw_centered_too_few_points` (the three-point
precondition, specific to centered stencils).
=#

# --- Direction traits -------------------------------------------------------------- #
abstract type GridDirection end
struct Forward <: GridDirection end
struct Backward <: GridDirection end

# --- Argument validation shared by every operator ----------------------------------- #
# Thrown rather than asserted: these check caller arguments, and an @assert reports a
# size mismatch as an AssertionError, which is not what a caller should have to catch.
@noinline _throw_stencil_dim_error(dim::Int, D::Int) = throw(ArgumentError("the stencil direction must be between 1 and $D, got $dim"))

@noinline _throw_stencil_symbol_error(d::Symbol) = throw(ArgumentError("the stencil direction must be :x, :y or :z, got :$d"))

@noinline function _throw_stencil_size_error(lout::Int, lin::Int, dims)
    throw(
        DimensionMismatch(
        "out has $lout entries and in has $lin, but the grid $(dims) has $(prod(dims))"
    ),
    )
end

# Every stencil reads a neighbour of the coordinate it writes, and the traversal is a
# single contiguous pass: aliased destination and source would overwrite an entry before
# the interior point that still needs it as a neighbour has been computed, corrupting every
# value downstream of the first write (see `D₋ₓ!` in the docs). Checked with `mightalias`
# rather than `===` so that two distinct `VectorElement`s sharing the same backing array (a
# view, or one built directly on the other's data) are caught too.
@noinline _throw_alias_error() = throw(ArgumentError("destination and source must not alias"))

@inline _check_no_alias(vₕ::VectorElement, uₕ::VectorElement) = Base.mightalias(parent(vₕ), parent(uₕ)) &&
                                                                _throw_alias_error()

# --- Diagonal scaling of an operator matrix (kept for the retained Kronecker oracle) --- #
#
# Every weighted family in this subsystem used to compute its matrix as a diagonal scaling
# of an unscaled one, `w .* A` -- a dense vector broadcast against a `SparseMatrixCSC`,
# which is a separately measured `SparseArrays` memory bug (notes on gpena/Bramble.jl#185
# §1.9) fixed independently of this subplan for every *production* weighted matrix.
# `stencil_matrix` never had that step to begin with, weights being written directly into
# `nzval` at scatter time (below); `_scale_rows!` is kept here only so
# `kronecker_operator_matrix`'s retained `_kron_*` bodies (`difference.jl`, `average.jl`)
# scale rows the same way that fix does, rather than reintroducing the broadcast in the one
# place still doing the old Kronecker-then-scale construction.
#
# `SparseArrays.dropzeros!` is not reachable from this file without adding it to the
# `using SparseArrays: ...` list in `src/Bramble.jl`, out of scope for this subplan (owned
# by the fix above); the drop is inlined as a compaction over the same three CSC arrays
# instead of calling it by name. `_drop_zero_entries!` is the same operation `dropzeros!`
# performs, so a later merge of that fix needs at most a name swap here, not new logic.
@inline function _scale_rows!(A::SparseMatrixCSC, w::AbstractVector)
    @boundscheck length(w) == size(A, 1) ||
                 throw(DimensionMismatch("the weight vector has $(length(w)) entries and the operator has $(size(A, 1)) rows"))

    rows = rowvals(A)
    nz = nonzeros(A)
    @inbounds for k in eachindex(nz)
        nz[k] *= w[rows[k]]
    end

    return _drop_zero_entries!(A)
end

# A dense or GPU backend has no stored-entry list to walk, and nothing is saved by avoiding
# the full matrix: the broadcast is in place and allocates nothing.
@inline function _scale_rows!(A::AbstractMatrix, w::AbstractVector)
    A .= w .* A
    return A
end

function _drop_zero_entries!(A::SparseMatrixCSC)
    colptr = A.colptr
    rows = A.rowval
    nz = A.nzval
    dest = 1
    @inbounds for col in 1:size(A, 2)
        stop = colptr[col + 1] - 1
        start = colptr[col]
        colptr[col] = dest
        for k in start:stop
            if !iszero(nz[k])
                rows[dest] = rows[k]
                nz[dest] = nz[k]
                dest += 1
            end
        end
    end
    colptr[size(A, 2) + 1] = dest
    resize!(rows, dest - 1)
    resize!(nz, dest - 1)
    return A
end

# --- Argument handling shared by every operator ------------------------------------- #
# The operators accept a mesh, a grid space or a grid function, and the vectorial aliases
# need the spatial dimension of whichever was passed. Going through `space` alone would
# reject a mesh, which the scalar aliases do accept.
@inline _op_mesh(Ωₕ::AbstractMeshType) = Ωₕ
@inline _op_mesh(Wₕ::AbstractSpaceType) = mesh(Wₕ)
@inline _op_mesh(uₕ::VectorElement) = mesh(space(uₕ))

"""
    OperatorArgument

The three things a discrete operator accepts: a mesh, a grid space or a
[`VectorElement`](@ref). Exactly the three `_op_mesh` has a method for.

Named so that the dimensional entry points can say it. The per-coordinate aliases take an
untyped `arg`, which is harmless at one argument, but `D₋(arg, d)` shares its name with the
form layer's `D₋(op::LazyOp, ::Val)`: an untyped two-argument method would swallow a
`LazyOp` and hand it to the grid engines, and an `Int` direction on a symbolic node cannot
be type-stable anyway (`BackwardDifference{D, 1, …}` and `{D, 2, …}` are different types).
Spelling the union keeps that a `MethodError` (gpena/Bramble.jl#74).
"""
const OperatorArgument = Union{AbstractMeshType, AbstractSpaceType, VectorElement}

# --- Runtime direction selection ----------------------------------------------------- #
#
# `Val(d)` built from a non-constant `d` boxes it, and nothing downstream of the boxed
# `Val` can constant-fold: that is the regression gpena/Bramble.jl#146 measured through
# `ntuple(i -> …, Val(D))` closures, and the reason the vectorial aliases are unrolled
# rather than looped. A runtime `Int` therefore never reaches `Val` directly. It selects
# between literal `Val`s instead, one arm per direction the mesh actually has, so every
# call the compiler sees carries a `Val` it can read.
#
# Branching on the mesh dimension first is what keeps the arms exhaustive *and* correct:
# a 1D mesh gets one arm and an honest "must be between 1 and 1" for anything else, rather
# than a `Val(3)` specialisation of the engines that no 1D grid can serve.

@inline function _dim_index(d::Symbol)
    d === :x && return 1
    d === :y && return 2
    d === :z && return 3
    return _throw_stencil_symbol_error(d)
end

@inline _dispatch_dim(f::F, arg, d::Int) where {F} = _dispatch_dim(
    f, arg, d, Val(dim(_op_mesh(arg)))
)

@inline _dispatch_dim(f::F, arg, d::Int, ::Val{1}) where {F} = d == 1 ? f(arg, Val(1)) :
                                                               _throw_stencil_dim_error(
    d, 1
)

@inline function _dispatch_dim(f::F, arg, d::Int, ::Val{2}) where {F}
    d == 1 && return f(arg, Val(1))
    d == 2 && return f(arg, Val(2))
    return _throw_stencil_dim_error(d, 2)
end

@inline function _dispatch_dim(f::F, arg, d::Int, ::Val{3}) where {F}
    d == 1 && return f(arg, Val(1))
    d == 2 && return f(arg, Val(2))
    d == 3 && return f(arg, Val(3))
    return _throw_stencil_dim_error(d, 3)
end

# The tuple-valued aliases' unrolling, written once rather than generated per family.
#
# It lived in `_vectorial_expr` as three `alias(arg, ::Val{D})` methods until
# gpena/Bramble.jl#74. That put the *mesh dimension* on a public two-argument signature,
# which is the signature #74 needs for the *direction*: on a 2D mesh `Dₕ(u, Val(2))` meant
# the pair, and has to mean the `y` difference. `Dₕ` is the one family whose stem and
# vectorial alias are the same name, so only it collided -- but moving the unrolling here
# removes the shape rather than special-casing the one name, and drops three generated
# methods per family while doing it.
@inline _vectorial_apply(f::F, arg, ::Val{1}) where {F} = f(arg, Val(1))
@inline _vectorial_apply(f::F, arg, ::Val{2}) where {F} = (f(arg, Val(1)), f(arg, Val(2)))
@inline _vectorial_apply(f::F, arg, ::Val{3}) where {F} = (
    f(arg, Val(1)), f(arg, Val(2)), f(arg, Val(3))
)

# A composite grid function is a stack of scalar ones, so an operator applies to each
# component in turn. Their `components` are views onto the parent, so writing into the
# components of `similar(uₕ)` fills it.
#
# The grid shape has to come from the mesh. `ndofs(space, Tuple)` gives it for a scalar
# space but gives the per-component dof counts for a composite one, so using it here
# addressed prod(ndofs) slots into a vector holding ndofs of them: a 3-component 4x6
# space addressed 13824 slots into 72, which segfaulted under the engines' @inbounds.
@inline _grid_dims(uₕ::VectorElement) = npoints(_op_mesh(uₕ), Tuple)

# `f!` is the single-component applicator; it is called once per *leaf* (`components`
# flattens any nesting), so this needs no component count of its own: `map` over the two
# tuples `components` returns unrolls exactly as the old `ntuple(…, Val(NC))` did, and stays
# correct regardless of how deeply either space nests.
@inline function _apply_componentwise!(
        f!, vₕ::VectorElement{<:CompositeGridSpace}, uₕ::VectorElement{<:CompositeGridSpace}
)
    map(f!, components(vₕ), components(uₕ))
    return nothing
end

# --- Shared one-sided stencil traversal ---------------------------------------------- #
# The difference and the average walk the grid identically: one pass over the interior,
# where every point has a neighbour along the stencil direction, and one over the single
# boundary slice, where it does not. Only the per-point kernel differs, so the traversal
# is written once here and both engines using it (difference.jl, average.jl) share it.

# The unit step along `DIM`.
@inline _stencil_step(::Val{DIM}, ::Val{D}) where {DIM, D} = CartesianIndex(ntuple(i -> i == DIM ? 1 : 0, Val(D)))

# The neighbour of `I`: ahead of it for a forward stencil, behind it for a backward one.
@inline _neighbour(::Forward, I, step) = I + step
@inline _neighbour(::Backward, I, step) = I - step

# The interior and boundary index ranges, as tuples of ranges to build
# `CartesianIndices` from. A forward stencil reaches past the last slice along `DIM`, a
# backward one past the first.
@inline function _stencil_ranges(
        full_axes::NTuple{D, Any}, ::Val{DIM}, ::Forward
) where {D, DIM}
    interior = ntuple(
        d -> d == DIM ? (first(full_axes[d]):(last(full_axes[d]) - 1)) : full_axes[d],
        Val(D)
    )
    boundary = ntuple(
        d -> d == DIM ? (last(full_axes[d]):last(full_axes[d])) : full_axes[d], Val(D)
    )
    return interior, boundary
end

@inline function _stencil_ranges(
        full_axes::NTuple{D, Any}, ::Val{DIM}, ::Backward
) where {D, DIM}
    interior = ntuple(
        d -> d == DIM ? ((first(full_axes[d]) + 1):last(full_axes[d])) : full_axes[d],
        Val(D)
    )
    boundary = ntuple(
        d -> d == DIM ? (first(full_axes[d]):first(full_axes[d])) : full_axes[d], Val(D)
    )
    return interior, boundary
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
# same families in `src/form/operators/{difference,average,jump}.jl` (their own
# `_stencil_taps`/`_stencil_weights`, keyed on AST node types such as `BackwardDifference`
# and `JumpNode`). Calling those directly would mean constructing a `LazyOp` tree from this
# file to stand in for the node's `inner_op` field, which `src/space/` has no business
# doing: forms are built on top of the space layer's operators, not the other way around,
# and `src/form/` is out of scope for this subplan besides. The two are proved equal by the
# equality test against `kronecker_operator_matrix` below instead of by sharing code --
# reported as the duplication the plan anticipated rather than resolved.

"""
    StencilOp{Dim}

One operator family along axis `Dim`, as [`stencil_matrix`](@ref) takes it. Each concrete
subtype answers [`_stencil_taps`](@ref) (its fixed neighbour offsets along `Dim`) and
[`_stencil_weights`](@ref) (the coefficients matching those offsets at a grid point).
"""
abstract type StencilOp{Dim} end

struct BackwardFiniteDiffOp{Dim} <: StencilOp{Dim} end
struct ForwardFiniteDiffOp{Dim} <: StencilOp{Dim} end
struct UnscaledBackwardDiffOp{Dim} <: StencilOp{Dim} end
struct UnscaledForwardDiffOp{Dim} <: StencilOp{Dim} end
struct StarDiffOp{Dim} <: StencilOp{Dim} end
struct CenteredDiffOp{Dim} <: StencilOp{Dim} end
struct CrossWeightedDiffOp{Dim} <: StencilOp{Dim} end
struct BackwardAvgOp{Dim} <: StencilOp{Dim} end
struct ForwardAvgOp{Dim} <: StencilOp{Dim} end

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

"""
    _stencil_weights(op::StencilOp{Dim}, Ωₕ::AbstractMeshType, I::CartesianIndex) -> NTuple{K}

The coefficients matching [`_stencil_taps`](@ref)`(op)` at `I`. The unscaled families'
self-weight is always ±1 regardless of boundary -- their truncation is entirely the "tap
falls outside the grid" exclusion `stencil_matrix` applies -- while the scaled and averaged
families additionally zero their own weight at a boundary that a Kronecker construction
would reach through a since-zeroed weight vector rather than exclude, which this reproduces
explicitly.
"""
@inline _stencil_weights(::UnscaledBackwardDiffOp, Ωₕ::AbstractMeshType, I::CartesianIndex) = (1, -1)
@inline _stencil_weights(::UnscaledForwardDiffOp, Ωₕ::AbstractMeshType, I::CartesianIndex) = (1, -1)

@inline function _stencil_weights(
        ::BackwardFiniteDiffOp{Dim}, Ωₕ::AbstractMeshType, I::CartesianIndex
) where {Dim}
    h = spacing(Ωₕ, I, Dim)
    mask = I[Dim] == 1 ? 0 : 1
    return (mask / h, -mask / h)
end

@inline function _stencil_weights(
        ::ForwardFiniteDiffOp{Dim}, Ωₕ::AbstractMeshType, I::CartesianIndex
) where {Dim}
    n = npoints(Ωₕ, Tuple)[Dim]
    h = forward_spacing(Ωₕ, I, Dim)
    mask = I[Dim] == n ? 0 : 1
    return (mask / h, -mask / h)
end

@inline function _stencil_weights(::StarDiffOp{Dim}, Ωₕ::AbstractMeshType, I::CartesianIndex) where {Dim}
    n = npoints(Ωₕ, Tuple)[Dim]
    mask = I[Dim] == n ? 0 : 1
    c = 2 * mask / (spacing(Ωₕ, I, Dim) + forward_spacing(Ωₕ, I, Dim))
    return (c, -c)
end

@inline function _stencil_weights(::CenteredDiffOp{Dim}, Ωₕ::AbstractMeshType, I::CartesianIndex) where {Dim}
    n = npoints(Ωₕ, Tuple)[Dim]
    mask = (I[Dim] == 1 || I[Dim] == n) ? 0 : 1
    c = mask / (spacing(Ωₕ, I, Dim) + forward_spacing(Ωₕ, I, Dim))
    return (c, -c)
end

@inline function _stencil_weights(
        ::CrossWeightedDiffOp{Dim}, Ωₕ::AbstractMeshType, I::CartesianIndex
) where {Dim}
    n = npoints(Ωₕ, Tuple)[Dim]
    if I[Dim] == 1
        a = inv(spacing(Ωₕ, I, Dim))
        return (a, -a, zero(a))
    elseif I[Dim] == n
        b = inv(spacing(Ωₕ, I, Dim))
        return (zero(b), b, -b)
    else
        h = spacing(Ωₕ, I, Dim)
        hf = forward_spacing(Ωₕ, I, Dim)
        total = h + hf
        a = h / (total * hf)
        b = hf / (total * h)
        return (a, b - a, -b)
    end
end

@inline function _stencil_weights(::BackwardAvgOp{Dim}, Ωₕ::AbstractMeshType, I::CartesianIndex) where {Dim}
    T = eltype(Ωₕ)
    mask = I[Dim] == 1 ? zero(T) : T(1) / 2
    return (mask, mask)
end

@inline function _stencil_weights(::ForwardAvgOp{Dim}, Ωₕ::AbstractMeshType, I::CartesianIndex) where {Dim}
    T = eltype(Ωₕ)
    n = npoints(Ωₕ, Tuple)[Dim]
    mask = I[Dim] == n ? zero(T) : T(1) / 2
    return (mask, mask)
end

"""
    stencil_matrix(Ωₕ::AbstractMeshType, op::StencilOp)
    stencil_matrix(Wₕ::AbstractSpaceType, op::StencilOp)

Builds `op`'s operator matrix in one pass over `Ωₕ`'s grid points, in the matrix type
`matrix_type(backend(Ωₕ))` picked: a `SparseMatrixCSC` backend writes `colptr`/`rowval`/
`nzval` directly (two passes, count then fill; no intermediate matrix, no broadcast), any
other `AbstractMatrix` backend falls back to a plain dense fill. `Ωₕ` may be a mesh or a
grid space, taken as `mesh(Wₕ)`.

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

    A = MT(undef, N, N)
    fill!(A, zero(T))
    @inbounds for I in CartesianIndices(dims)
        idim = I[Dim]
        w = _stencil_weights(op, Ωₕ, I)
        row = li[I]
        for k in 1:K
            j = idim + taps[k]
            (1 <= j <= n) || continue
            iszero(w[k]) && continue
            A[row, li[I + taps[k] * step]] = T(w[k])
        end
    end
    return A
end

# --- Alias and docstring generators -------------------------------------------------- #
#
# Every generator below *returns* the expression that defines its methods; nothing is
# evaluated here (gpena/Bramble.jl#258). The `@operator_family` macro at the end of the
# file splices those expressions into its own expansion, so each operator method reaches
# the compiler as an ordinary declaration the parser already saw, rather than as something
# `Core.eval`ed into the module while it loads.
#
# The prose a family needs per alias used to arrive as a closure over `direction`/`suffix`.
# A macro cannot call a closure -- it only ever sees the expression that would build one --
# so the notes are string literals carrying `{direction}` and `{suffix}` placeholders
# instead, substituted here at expansion time.

"""
    _relocate!(ex, source)

Rewrites every `LineNumberNode` in `ex` to `source`, and returns `ex`.

A generated method otherwise reports the `:(...)` quote it was built from -- one shared line
in this file for every family -- because that is the line info the quote carries. Rewriting
it to the `__source__` the macro was expanded at makes `methods(D₋ₓ)`, a stacktrace and an
editor's "go to definition" all land on the family's own `@operator_family` call.

`source` given as `nothing` leaves `ex` untouched, which is what a direct call outside the
macro gets.
"""
_relocate!(ex, ::Nothing) = ex
_relocate!(ex, ::LineNumberNode) = ex

function _relocate!(ex::Expr, source::LineNumberNode)
    for (i, arg) in enumerate(ex.args)
        if arg isa LineNumberNode
            ex.args[i] = source
        elseif arg isa Expr
            _relocate!(arg, source)
        end
    end
    return ex
end

"""
    _subst(template, direction, suffix)

Substitutes the `{direction}` and `{suffix}` placeholders in a family's prose template.

Both placeholders are replaced in a single pass, so neither can be rewritten by the other's
replacement text. An empty template stays empty, which is how a family says it needs no note
at that insertion point.
"""
@inline function _subst(template::AbstractString, direction, suffix)
    isempty(template) && return ""
    return replace(String(template), "{direction}" => direction, "{suffix}" => suffix)
end

"""
    _alias_bang_expr(base_op_name, alias_name, dir_string, suffix, direction_index, what,
                     formula; opening_sentence = "", source = nothing)

Returns the expression defining `alias_name(vₕ, uₕ)` as
`base_op_name(vₕ, uₕ, Val(direction_index))`, with its docstring attached.

The in-place sibling of `_alias_expr`. Two generators rather than one because the shapes
differ: the allocating alias takes a single argument that may be a mesh, a space or a grid
function, while this one takes a destination and a source and is only ever about grid
functions.

`opening_sentence`, given non-empty, replaces the generic "The `\$dir_string` `\$what` of
`uₕ` along the `\$suffix` direction, ``\$formula``, written into `vₕ`." the same way it does
for `_alias_expr` -- `Dc!`/`Dₕ!` have no backward/forward adjective to put in `dir_string`
either.

`source` is the `LineNumberNode` the generated method is attributed to; the macro passes its
own `__source__`, so a generated method reports the family's call site rather than the quoted
line in this file.
"""
function _alias_bang_expr(
        base_op_name,
        alias_name,
        dir_string,
        suffix,
        direction_index,
        what,
        formula;
        opening_sentence::String = "",
        source = nothing
)
    opening = if isempty(opening_sentence)
        "The `$dir_string` $what of `uₕ` along the `$suffix` direction, " *
        "``$formula``, written into `vₕ`."
    else
        opening_sentence
    end
    doc_string = """
        $alias_name(vₕ, uₕ)

    $opening

    The in-place form of [`$(replace(String(alias_name), "!" => ""))`](@ref): it allocates
    nothing, where the allocating form allocates its result. Returns `vₕ`, so it composes:
    `normₕ($alias_name(vₕ, uₕ))`.

    `vₕ` and `uₕ` must be grid functions of the same space, and must not be the same
    object, as every stencil reads neighbours of the target coordinate; aliasing them
    would read values that have already been overwritten.

    Alias for `$base_op_name(vₕ, uₕ, Val($direction_index))`. Accepts a grid function of a
    scalar or of a composite grid space, componentwise on the latter.
    """

    func_def_expr = _relocate!(
        :(@inline $(alias_name)(vₕ, uₕ) = $(base_op_name)(vₕ, uₕ, Val($(direction_index)))), source
    )
    return Expr(
        :macrocall, GlobalRef(Core, Symbol("@doc")), source, doc_string, func_def_expr
    )
end

"""
    _alias_expr(base_op_name, alias_name, dir_string, suffix, direction_index, what,
                formula; opening_sentence = "", formula_note = "", alias_note = "",
                trailing_note = "", source = nothing)

Returns the expression defining `alias_name(arg)` as `base_op_name(arg, Val(direction_index))`,
with its docstring attached.

`what` names the quantity, such as `"finite difference"`, and `formula` is the LaTeX for
it. Both are needed because the four operator families share this generator: describing
every alias as a "difference" would be wrong for the averages, and would
not separate the unscaled difference from the finite difference.

`opening_sentence`, given non-empty, replaces the generic "The `\$dir_string` `\$what`
along the `\$suffix` direction, ``\$formula``." with the caller's own wording: `Dc` and
`Dₕ` have no backward/forward adjective to put in `dir_string` at all.

The three remaining keyword notes are each a sentence the docstring includes only when
given (non-empty), one per insertion point a family may need: `formula_note` follows the
opening sentence (the diff/finite-difference families use this to contrast the two, which
does not apply to an average); `alias_note` follows the `Alias for ...` sentence, before
`arg` is described (`Dₕ` uses this to compare itself with `Dc`); `trailing_note` follows
the description of `arg`, before the closing "Accepts a grid function..." paragraph
(`D̽`, `Dc` and `Dₕ` use this for their boundary-behaviour and precondition caveats,
which differ both in what happens at the ends -- `D̽`/`Dc` truncate, `Dₕ` falls back
to a one-sided difference (gpena/Bramble.jl#183) -- and in whether a mesh needs at least
three points along the direction).

`source` attributes the generated method to the family's own call site, as in
[`_alias_bang_expr`](@ref).
"""
function _alias_expr(
        base_op_name,
        alias_name,
        dir_string,
        suffix,
        direction_index,
        what,
        formula;
        opening_sentence::String = "",
        formula_note::String = "",
        alias_note::String = "",
        trailing_note::String = "",
        source = nothing
)
    fn = isempty(formula_note) ? "" : " " * formula_note
    an = isempty(alias_note) ? "" : " " * alias_note
    tn = isempty(trailing_note) ? "" : " " * trailing_note
    opening = if isempty(opening_sentence)
        "The `$dir_string` $what along the `$suffix` direction, ``$formula``.$fn"
    else
        opening_sentence
    end

    doc_string = """
        $alias_name(arg)

    $opening

    Alias for `$base_op_name(arg, Val($direction_index))`.$an `arg` is a mesh, a grid space
    or a [`VectorElement`](@ref): the first two give the operator as a sparse matrix, the
    third applies it and returns a `VectorElement`.$tn

    Accepts a grid function of a scalar or of a composite grid space. On a composite one
    the operator is applied to each component in turn, and the result is the composite
    grid function whose components are those results.
    """

    func_def_expr = _relocate!(
        :(@inline $(alias_name)(arg) = $(base_op_name)(arg, Val($(direction_index)))), source
    )

    return Expr(
        :macrocall, GlobalRef(Core, Symbol("@doc")), source, doc_string, func_def_expr
    )
end

"""
    _vectorial_expr(base_op_name, alias_name, dir_string, what; note = "", source = nothing)

Returns the expression defining the `ₕ` alias that applies `base_op_name` along every
coordinate and returns a tuple, one entry per spatial dimension, as a `Vector{Expr}`. On a
one-dimensional mesh the alias returns that single entry rather than a one-tuple.

The counterpart of `_alias_expr` for the tuple-valued aliases (`∇ₕ`, `diff₋ₕ`, `Mₕ`). The
operator families generated the same three methods independently before this existed.

`note`, given non-empty, is an extra sentence appended after the worked 2D example --
`Dₕ` uses this to place itself relative to `∇ₕ`/`∇₊ₕ`, a comparison none of the other
vectorial aliases need.
"""
function _vectorial_expr(
        base_op_name, alias_name, dir_string, what; note::String = "", source = nothing
)
    n = isempty(note) ? "" : " " * note
    # A family with no direction word to give -- the jump belongs to an interface, not to a
    # direction of travel -- passes `dir_string` empty, and the qualifier collapses to
    # `what` alone rather than leaving a double space in the sentence.
    qualifier = isempty(dir_string) ? what : "$dir_string $what"
    doc_string = """
        $alias_name(arg)

    The $qualifier of `arg` along every coordinate, as a tuple with one entry per
    spatial dimension. On a one-dimensional mesh it returns that single entry rather than
    a one-tuple.

    For a 2D space, `$alias_name(uₕ)` is
    `($base_op_name(uₕ, Val(1)), $base_op_name(uₕ, Val(2)))`. `arg` is a mesh, a grid
    space or a [`VectorElement`](@ref), as for `$base_op_name`.$n

    Accepts a grid function of a scalar or of a composite grid space, componentwise on
    the latter: each entry of the tuple is then itself a composite grid function.
    """

    # One method, not four. The per-dimension unrolling is `_vectorial_apply` above,
    # shared by every family: it used to be generated here as three `alias(arg, ::Val{D})`
    # methods, and that two-argument signature is the one the dimensional entry point needs
    # for the *direction* (gpena/Bramble.jl#74).
    #
    # `Val(dim(_op_mesh(arg)))` is read off the mesh *type*, so it is a literal by the time
    # `_vectorial_apply` dispatches on it; no `Val` here is ever built from a runtime value
    # (gpena/Bramble.jl#146).
    entry = :(@inline $(alias_name)(arg) = _vectorial_apply(
        $(base_op_name), arg, Val(dim(_op_mesh(arg)))
    ))

    documented_entry = Expr(
        :macrocall, GlobalRef(Core, Symbol("@doc")), source, doc_string, entry
    )
    return Expr[_relocate!(documented_entry, source)]
end

"""
    _dimensional_expr(base_op_name, alias_name, alias_stem, dir_string, what, formula;
                      opening_sentence = "", source = nothing)

Returns the expressions defining one family's dimensional entry point, as a `Vector{Expr}`:
`alias_name(arg, d)` for `d` a `Val`, an `Int` or a `Symbol`.

The name gpena/Bramble.jl#74 adds over machinery that was already there. Every family
bottoms out in a `Val`-parameterised base function, and the 63 subscript aliases are
one-line `Val` bindings over it; what was missing was a way to say "this operator, along
*that* direction" with the direction a value. `sum(D₋(uₕ, d) for d in 1:D)` is the first
dimension-agnostic form the package can express, and the subscripts stop being the only way
in.

The `Val` method is omitted when `alias_name` is `base_op_name` itself, as it is for
`jump`: the family's own file already wrote that method, and generating it again would
define `jump(arg, ::Val) = jump(arg, ::Val)`.

`alias_stem` is the family's subscript stem, which the docstring cross-references: it is
`alias_name` for most families, but the averages put their entry point on the tuple-valued
`Mₕ`/`M₊ₕ` while the per-coordinate aliases are still `Mₓ`/`M₊ₓ`, and `Mₕₓ` is not a name.

`opening_sentence` is the family's own prose template, substituted with `d` as the
direction; the families that do not override it get the generic
"The `\$dir_string` `\$what` along the `d` direction, ``\$formula``." instead.
"""
function _dimensional_expr(
        base_op_name,
        alias_name,
        alias_stem,
        dir_string,
        what,
        formula;
        opening_sentence::String = "",
        source = nothing
)
    opening = if isempty(opening_sentence)
        "The `$dir_string` $what along the `d` direction, ``$formula``."
    else
        _subst(opening_sentence, "d", "")
    end

    doc_string = """
        $alias_name(arg, d)

    $opening

    `d` names the direction three ways: a `Val` (`Val(1)`), an `Int` (`1`, `2`, `3`) or a
    `Symbol` (`:x`, `:y`, `:z`). The `Val` form is the one
    [`$(alias_stem)ₓ`](@ref) and its siblings forward through; the other two select between
    literal `Val`s, one arm per direction the mesh has, so the direction still reaches the
    stencil engine as a compile-time constant and a loop over `d` allocates nothing beyond
    its results. An `Int` outside `1:dim(mesh)`, or a `Symbol` that is not `:x`/`:y`/`:z`,
    throws an `ArgumentError`.

    Alias for `$base_op_name(arg, Val(d))`. `arg` is a mesh, a grid space or a
    [`VectorElement`](@ref): the first two give the operator as a sparse matrix, the third
    applies it and returns a `VectorElement`.

    Accepts a grid function of a scalar or of a composite grid space. On a composite one
    the operator is applied to each component in turn, and the result is the composite
    grid function whose components are those results.
    """

    int_method = :(@inline $(alias_name)(arg::OperatorArgument, d::Int) = _dispatch_dim(
        $(base_op_name), arg, d
    ))
    sym_method = :(@inline $(alias_name)(arg::OperatorArgument, d::Symbol) = _dispatch_dim(
        $(base_op_name), arg, _dim_index(d)
    ))

    documented_int = Expr(
        :macrocall, GlobalRef(Core, Symbol("@doc")), source, doc_string, int_method
    )

    exprs = Expr[documented_int, sym_method]
    if alias_name !== base_op_name
        pushfirst!(
            exprs,
            :(@inline $(alias_name)(arg::OperatorArgument, dim_val::Val) = $(base_op_name)(
                arg, dim_val
            ))
        )
    end
    return Expr[_relocate!(e, source) for e in exprs]
end

"""
    _grid_function_forms_expr(base_name, apply_fn, extra_args, dir_instance;
                              docstring = "", source = nothing)

Returns the expressions defining the three methods every directional operator family
applies a grid function through, as a `Vector{Expr}`: `base_name!` on a scalar grid
function, `base_name!` on a composite one, and the allocating `base_name` built on top of
them.

The three are byte-identical across the families apart from which applicator they call and
what it takes before the direction (gpena/Bramble.jl#101); `difference.jl` and
`average.jl` each generated them from their own `@eval` loop before this existed:

| Family | `apply_fn` | `extra_args` |
|:--|:--|:--|
| unscaled difference | `_apply_spaced!` | `(_no_spacing, _no_precheck)` |
| finite difference | `_apply_spaced!` | `(spacings_func, _no_precheck)` |
| `D̽`/`Dc`/`Dₕ` | `_apply_spaced!` | `(spacing_func, precheck)` |
| average | `_apply_averaged!` | `()` |

`extra_args` are spliced as bare identifiers, so each generated method names an ordinary
top-level function rather than closing over one: that is what keeps every call site
specialising to its own zero-allocation method, as the hand-written versions did.

The composite method recurses into the scalar one through `apply_fn`'s own composite
method rather than repeating the walk, so a leaf's own submesh is what each leaf is
measured against (gpena/Bramble.jl#79).

`docstring`, given non-empty, is attached to the scalar `base_name!` method; the families
whose prose lives here rather than on a separately hand-written matrix form use it.
"""
function _grid_function_forms_expr(
        base_name, apply_fn, extra_args, dir_instance;
        docstring::String = "", source = nothing
)
    bang_name = Symbol(base_name, :!)
    extra = Any[extra_args...]

    scalar = :(@inline $(bang_name)(
        vₕ::VectorElement{<:ScalarGridSpace},
        uₕ::VectorElement{<:ScalarGridSpace},
        dim_val::Val
    ) = $(apply_fn)(vₕ, uₕ, $(extra...), $(dir_instance), dim_val))

    composite = :(@inline $(bang_name)(
        vₕ::VectorElement{<:CompositeGridSpace},
        uₕ::VectorElement{<:CompositeGridSpace},
        dim_val::Val
    ) = $(apply_fn)(vₕ, uₕ, $(extra...), $(dir_instance), dim_val))

    allocating = :(@inline $(base_name)(uₕ::VectorElement, dim_val::Val) = $(bang_name)(similar(uₕ), uₕ, dim_val))

    documented_scalar = if isempty(docstring)
        scalar
    else
        Expr(:macrocall, GlobalRef(Core, Symbol("@doc")), source, docstring, scalar)
    end
    return Expr[_relocate!(e, source) for e in (documented_scalar, composite, allocating)]
end

"""
    _operator_aliases_expr(base_name, alias_stem, dir_string, what, formula;
                           vectorial_alias = nothing, vectorial_note = "",
                           opening_sentence = "", formula_note = "", alias_note = "",
                           trailing_note = "", bang_opening_sentence = "", source = nothing)

Returns the expressions defining one family's whole alias surface, as a `Vector{Expr}`: the
per-coordinate `alias_stem` pair for every direction (`Dcₓ`/`Dcₓ!`, `Mᵧ`/`Mᵧ!`, …), the
dimensional entry point `dispatch_alias(arg, d)`, and, when `vectorial_alias` is given, the
`ₕ` alias over every coordinate at once.

`dispatch_alias` defaults to `alias_stem`, which is what every difference family wants:
`D₋ₓ`, `D₋ᵧ`, `D₋₂` and `D₋(uₕ, d)`. The averages pass their tuple-valued alias instead, so
that the family gains a direction argument without minting a bare `M` -- the single most
common local name in finite-element code, and one `using Bramble` would then reserve.
Passing `nothing` skips the entry point altogether.

[`_alias_expr`](@ref)/[`_alias_bang_expr`](@ref) and [`_vectorial_expr`](@ref) were already
shared; the *loop* calling them was written out once per family in `difference.jl` and once
more in `average.jl` (gpena/Bramble.jl#101). This is that loop.

The five note keywords are prose templates rather than finished sentences: each may carry
`{direction}` and `{suffix}`, substituted per alias by [`_subst`](@ref). They were closures
taking `(direction, suffix)` until gpena/Bramble.jl#258 -- a macro sees the expression that
would build a closure, never the closure itself, so the notes a family needs
("over the averaged spacing", "second order on a non-uniform grid, where `Dc{suffix}` is
first") travel as string literals instead.

`vectorial_dir_string`/`vectorial_what` default to `dir_string`/`what` and exist for the
families that describe the tuple-valued alias differently from the per-coordinate ones:
`D̽`/`Dc`/`Dₕ` override every directional `opening_sentence` and so pass `dir_string`
and `what` empty, while `D̽ₕ`/`Dcₕ`/`Dₕ` still want "the centered difference of `arg`
along every coordinate".
"""
function _operator_aliases_expr(
        base_name,
        alias_stem,
        dir_string,
        what,
        formula;
        dispatch_alias = alias_stem,
        vectorial_alias = nothing,
        vectorial_dir_string = dir_string,
        vectorial_what = what,
        vectorial_note::String = "",
        opening_sentence::String = "",
        formula_note::String = "",
        alias_note::String = "",
        trailing_note::String = "",
        bang_opening_sentence::String = "",
        source = nothing
)
    bang_name = Symbol(base_name, :!)
    exprs = Expr[]

    for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
        # The human-readable label ("x"), not the subscript ("ₓ"): this lands in the
        # generic "along the `x` direction" sentence, which a family overriding
        # `opening_sentence` never reaches but the templated ones do.
        direction = _BRAMBLE_var2label[i]
        push!(
            exprs,
            _alias_expr(
                base_name,
                Symbol(alias_stem, suffix),
                dir_string,
                direction,
                i,
                what,
                formula;
                opening_sentence = _subst(opening_sentence, direction, suffix),
                formula_note = _subst(formula_note, direction, suffix),
                alias_note = _subst(alias_note, direction, suffix),
                trailing_note = _subst(trailing_note, direction, suffix),
                source
            )
        )
        push!(
            exprs,
            _alias_bang_expr(
                bang_name,
                Symbol(alias_stem, suffix, :!),
                dir_string,
                direction,
                i,
                what,
                formula;
                opening_sentence = _subst(bang_opening_sentence, direction, suffix),
                source
            )
        )
    end

    if dispatch_alias !== nothing
        append!(
            exprs,
            _dimensional_expr(
                base_name,
                dispatch_alias,
                alias_stem,
                dir_string,
                what,
                formula;
                opening_sentence,
                source
            )
        )
    end

    vectorial_alias === nothing && return exprs
    append!(
        exprs,
        _vectorial_expr(
            base_name,
            vectorial_alias,
            vectorial_dir_string,
            vectorial_what;
            note = vectorial_note,
            source
        )
    )
    return exprs
end

# --- The family macro ---------------------------------------------------------------- #

"""
    @operator_family(kwargs...)

Defines one operator family's grid-function forms and its whole alias surface, declaratively.

Replaces the `Core.eval` generators this file used to carry (gpena/Bramble.jl#258): the
methods now arrive through macro expansion, so the parser sees each definition where it is
written instead of the module growing methods while it loads.

Keywords, all optional except `base` and `stem`:

| Keyword | Meaning |
|:--|:--|
| `base` | the base operator name the aliases forward to, e.g. `backward_finite_difference` |
| `stem` | the alias stem, e.g. `D₋`, giving `D₋ₓ`, `D₋ᵧ`, `D₋₂` and their `!` forms |
| `apply_fn`, `extra_args`, `direction` | passed to [`_grid_function_forms_expr`](@ref); omitting `apply_fn` skips the grid-function forms, for a family that writes them itself |
| `docstring` | attached to the scalar `base!` method |
| `dir_string`, `what`, `formula` | fill the generic docstring template |
| `dispatch_alias` | the name carrying the dimensional entry point `f(arg, d)`; defaults to `stem` (gpena/Bramble.jl#74) |
| `vectorial_alias`, `vectorial_dir_string`, `vectorial_what`, `vectorial_note` | the `ₕ` alias over every coordinate |
| `opening_sentence`, `formula_note`, `alias_note`, `trailing_note`, `bang_opening_sentence` | prose templates; see [`_operator_aliases_expr`](@ref) for where each lands and for the `{direction}`/`{suffix}` placeholders |

Every value is a literal: a symbol for a name, a string for prose, a call such as
`Backward()` for the direction tag. Nothing is evaluated at expansion time beyond assembling
the docstrings, which is ordinary string building over those literals.
"""
macro operator_family(kwargs...)
    opts = Dict{Symbol, Any}()
    for kw in kwargs
        (kw isa Expr && kw.head === :(=)) ||
            error("@operator_family takes `key = value` arguments, got $(kw)")
        opts[kw.args[1]] = kw.args[2]
    end

    get_str(key) = _macro_string(get(opts, key, ""))
    exprs = Expr[]

    if haskey(opts, :apply_fn)
        append!(
            exprs,
            _grid_function_forms_expr(
                opts[:base],
                opts[:apply_fn],
                Tuple(_tuple_args(get(opts, :extra_args, Expr(:tuple)))),
                opts[:direction];
                docstring = get_str(:docstring),
                source = __source__
            )
        )
    end

    append!(
        exprs,
        _operator_aliases_expr(
            opts[:base],
            opts[:stem],
            get_str(:dir_string),
            get_str(:what),
            get_str(:formula);
            dispatch_alias = get(opts, :dispatch_alias, opts[:stem]),
            vectorial_alias = get(opts, :vectorial_alias, nothing),
            vectorial_dir_string = _get_str_or(opts, :vectorial_dir_string, get_str(:dir_string)),
            vectorial_what = _get_str_or(opts, :vectorial_what, get_str(:what)),
            vectorial_note = get_str(:vectorial_note),
            opening_sentence = get_str(:opening_sentence),
            formula_note = get_str(:formula_note),
            alias_note = get_str(:alias_note),
            trailing_note = get_str(:trailing_note),
            bang_opening_sentence = get_str(:bang_opening_sentence),
            source = __source__
        )
    )

    return esc(Expr(:block, exprs...))
end

# `extra_args = (spacings_func, _no_precheck)` reaches the macro as a tuple expression; a
# family that needs none writes nothing and gets the empty tuple.
_tuple_args(ex::Expr) = ex.head === :tuple ? ex.args : Any[ex]
_tuple_args(s::Symbol) = Any[s]

# A keyword that defaults to another keyword's value rather than to the empty string.
_get_str_or(opts, key, fallback) = haskey(opts, key) ? _macro_string(opts[key]) : fallback

# Prose long enough to wrap is written as `"..." * "..."` in the family's call, so what
# reaches the macro is the concatenation expression rather than one literal. Folding it here
# keeps the source readable without making the macro evaluate anything: only `*` chains of
# string literals are accepted, and anything else is a mistake worth reporting as one.
_macro_string(s::AbstractString) = String(s)

function _macro_string(ex::Expr)
    (ex.head === :call && ex.args[1] === :*) ||
        error("expected a string literal or a `*` concatenation of string literals, got $(ex)")
    return join(_macro_string(a) for a in ex.args[2:end])
end
