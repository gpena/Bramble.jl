# Matrix-free application of a ghost-space form (#781): `y = E_vᵀ A_pad E_u x`, the product
# of the reduced matrix `assemble(a)` returns, without assembling `A_pad` or the reduction.
#
# The padded form `a.padded` is applied by a `MatrixFreeOperator` on the padded spaces,
# between the closure of the trial space without its affine part (`E_u x`, read from the
# stored `E_uᵀ` by `_csc_columns_band!`) and the transpose of the test space's closure
# (`E_vᵀ`, read from the columns of `E_v`). The affine part `d` belongs to the load, as in
# the assembled path. Both padded vectors are buffers built with the operator, so the
# closure passes allocate nothing. A product holds them through the `busy` flag, taken by
# compare-and-swap, so a product entered on the same operator while another runs throws
# an `ArgumentError` rather than overwrite them. The closure passes run in the bands of
# the operator's policy (`_run_bands!`). A serial product computes the padded rows the
# test closure reads only, over the points writing them (the form's `walk`, #884).
#
# `dirichlet` replaces reduced rows by identity rows, through the masks of the physical
# mesh, on the spaces `assemble(a; dirichlet)` allows it on (`_check_rows`), as that does.
#
# See also: `GhostBilinearForm`, `MatrixFreeOperator`

"""
    GhostMatrixFreeOperator{T} <: AbstractMatrix{T}

The operator [`matrix_free_operator`](@ref) returns for a form on ghost spaces: the
reduced matrix `E_vᵀ A_pad E_u` of `assemble(a)`, applied without assembling it. It holds
the [`MatrixFreeOperator`](@ref) of the form on the padded spaces, the two closures, and
the padded buffers a product works in. One operator applies one product at a time: a
product entered while another runs on it throws an `ArgumentError`, so build one operator
per task. `getindex` costs one product per entry.
"""
struct GhostMatrixFreeOperator{T, Op <: MatrixFreeOperator, WT <: GhostGridSpace,
    VT <: GhostGridSpace, EP <: ExecutionPolicy, BT <: AbstractVector, RW <: _WalkRows} <:
       AbstractMatrix{T}
    "the matrix-free operator of the form on the padded spaces."
    padded::Op
    trial_space::WT
    test_space::VT
    "the reduced rows set to identity rows by `dirichlet`, or an empty mask."
    mask::BitVector
    policy::EP
    "the padded vectors `E_u x` and `A_pad E_u x`."
    xpad::BT
    ypad::BT
    "whether a product holds `xpad` and `ypad`."
    busy::Threads.Atomic{Bool}
    "the padded points and rows the product walks, the form's."
    walk::RW
end

function matrix_free_operator(a::GhostBilinearForm; dirichlet = nothing,
        dirichlet_components = nothing, policy = _ghost_policy(a.trial_space))
    labels, _ = _normalize_dirichlet(dirichlet)
    _check_rows(a, labels)
    op = matrix_free_operator(a.padded; policy)
    nrows = ndofs(a.test_space)
    mask = falses(labels === nothing ? 0 : nrows)
    if labels !== nothing
        _validate_scalar_components(dirichlet_components)
        Ωₕ = physical_mesh(mesh(a.test_space))
        for lbl in labels
            mask .|= _label_mask(Ωₕ, a.rows, lbl)
        end
    end
    return _ghost_mf_operator(op, a, mask, policy)
end

# The operator from its parts, each type parameter read off the value passed. The padded
# `op` infers as a union (whether it caches its geometry is a runtime question, see
# `_mf_operator`): this call takes each member on its own, so `Op` is the type of `op`.
function _ghost_mf_operator(op::MatrixFreeOperator, a::GhostBilinearForm, mask::BitVector,
        policy::ExecutionPolicy)
    T = promote_type(eltype(op), eltype(a.test_space.E), eltype(a.trial_space.E))
    xpad = similar(a.trial_space.d, T, size(op, 2))
    ypad = similar(a.trial_space.d, T, size(op, 1))
    return GhostMatrixFreeOperator{T, typeof(op), typeof(a.trial_space),
        typeof(a.test_space), typeof(policy), typeof(xpad), typeof(a.walk)}(
        op, a.trial_space, a.test_space, mask, policy, xpad, ypad,
        Threads.Atomic{Bool}(false), a.walk)
end

Base.size(op::GhostMatrixFreeOperator) = (ndofs(op.test_space), ndofs(op.trial_space))

@noinline function _throw_ghost_mf_dimmismatch(op::GhostMatrixFreeOperator, x, y)
    throw(
        DimensionMismatch(
        "the matrix-free operator of size $(size(op)) cannot multiply a vector of length " *
        "$(length(x)) into one of length $(length(y))",
    ),
    )
end

# The five-argument method only, as for `MatrixFreeOperator`: `LinearAlgebra`'s `mul!` and
# `*` reach it. Entries of a type the buffers cannot hold (a complex `x` on a real form)
# take padded vectors of their own; any other product takes the stored ones through `busy`,
# released however the product ends.
function mul!(y::AbstractVector, op::GhostMatrixFreeOperator{T}, x::AbstractVector,
        α::Number, β::Number) where {T}
    (length(x) == size(op, 2) && length(y) == size(op, 1)) ||
        _throw_ghost_mf_dimmismatch(op, x, y)
    yd = _mf_data(y)
    xd = _mf_data(x)
    Base.require_one_based_indexing(yd, xd)
    S = promote_type(T, eltype(xd))
    if S === T
        Threads.atomic_cas!(op.busy, false, true) && _throw_ghost_mf_busy()
        try
            _ghost_mf_apply!(yd, op, xd, α, β, op.xpad, op.ypad)
        finally
            op.busy[] = false
        end
    else
        _ghost_mf_apply!(yd, op, xd, α, β, similar(op.xpad, S), similar(op.ypad, S))
    end
    return y
end

@noinline function _throw_ghost_mf_busy()
    throw(
        ArgumentError(
        "this ghost-space matrix-free operator is already applying a product on another " *
        "task, and its padded buffers hold one product at a time. Build one operator per " *
        "task with matrix_free_operator(a).",
    ),
    )
end

function _ghost_mf_apply!(y, op::GhostMatrixFreeOperator, x, α, β, xpad, ypad)
    Et = op.trial_space.Et
    if op.walk.whole || !(op.policy isa CpuSerial)
        _run_bands!(op.policy, _csc_columns_band!, xpad, SparseArrays.getcolptr(Et),
            rowvals(Et), nonzeros(Et), x)
    else
        _closure_columns!(xpad, Et, x, op.walk.cols)
    end
    _mf_walked_mul!(ypad, op.padded, xpad, op.walk)
    E = op.test_space.E
    _run_bands!(op.policy, _ghost_mf_out_band!, y, SparseArrays.getcolptr(E), rowvals(E),
        nonzeros(E), ypad, op.mask.chunks, α, β)
    _mf_identity_rows!(y, x, α, isempty(op.mask) ? nothing : op.mask)
    return nothing
end

# `_csc_columns_band!` over the padded columns `cols` (`_GridLines`) alone, the ones the
# serial product's cut walk reads (`_mf_walked_mul!`); `y` keeps its other entries.
function _closure_columns!(y, Et::SparseMatrixCSC, x, cols::_GridLines)
    T = eltype(y)
    cp, rv, nz = SparseArrays.getcolptr(Et), rowvals(Et), nonzeros(Et)
    lin = LinearIndices(cols.grid)
    @inbounds for J in CartesianIndices(Base.tail(axes(cols.grid)))
        l = cols.lines[J]
        for t in cols.ptr[l]:(cols.ptr[l + 1] - 1), i in cols.lo[t]:cols.hi[t]

            r = lin[CartesianIndex(i, Tuple(J)...)]
            s = zero(T)
            for k in cp[r]:(cp[r + 1] - 1)
                s = convert(T, s + nz[k] * x[rv[k]])
            end
            y[r] = s
        end
    end
    return y
end

# Entries `r` of band `b` of `y = α E_vᵀ w + β y`, from column `r` of `E_v`; `β == 0`
# overwrites `y`, so a `NaN` in it does not survive. A row the nonempty `mask` marks keeps
# `β y[r]` only: `_mf_identity_rows!` adds its `α x[r]` after. The mask crosses as its
# words (`_MFMask`): a `BitVector` among the `@batch` arguments puts them on the heap.
@noinline function _ghost_mf_out_band!(y, colptr, rowval, nz, w, chunks, α, β, nbands::Int,
        b::Int)
    T = eltype(y)
    marked = !isempty(chunks)
    @inbounds for r in _band_range(1:length(y), nbands, b)
        s = zero(T)
        if !marked || _mf_live(_MFMask(chunks), r)
            for k in colptr[r]:(colptr[r + 1] - 1)
                s = convert(T, s + nz[k] * w[rowval[k]])
            end
        end
        y[r] = iszero(β) ? α * s : α * s + β * y[r]
    end
    return nothing
end

function Base.getindex(op::GhostMatrixFreeOperator{T}, i::Int, j::Int) where {T <: Number}
    @boundscheck checkbounds(op, i, j)
    x = zeros(T, size(op, 2))
    x[j] = one(T)
    y = zeros(T, size(op, 1))
    mul!(y, op, x, true, false)
    return y[i]
end

function Base.show(io::IO, op::GhostMatrixFreeOperator{T}) where {T}
    print(io, size(op, 1), "×", size(op, 2), " GhostMatrixFreeOperator{", T, "} on ",
        op.trial_space)
    isempty(op.mask) || print(io, " with Dirichlet rows")
    return nothing
end
Base.show(io::IO, ::MIME"text/plain", op::GhostMatrixFreeOperator) = show(io, op)
