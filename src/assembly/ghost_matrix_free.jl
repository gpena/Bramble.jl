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
# the operator's policy (`_run_bands!`). A product computes the padded rows the test
# closure reads only, over the points writing them (the form's `walk`, #884), serially or
# in its policy's bands.
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
    "the reduced rows each band of a `CpuThreaded` cut product sums (`_ghost_mf_bands!`)."
    outcut::Vector{Int}
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
        Threads.Atomic{Bool}(false), a.walk, _out_cuts(op, a))
end

# For a `CpuThreaded` product on a cut walk through the fused sweep: `cuts[b]:(cuts[b + 1]
# - 1)` are the reduced rows whose padded rows (the column of `E_v`) all lie in the slices
# band `b` owns (`_walk_own`), increasing with `b`. Empty for any other product, or when a
# reduced row reads rows of two bands, or the rows do not follow the bands in order.
function _out_cuts(op::MatrixFreeOperator, a::GhostBilinearForm)
    plan, w = op.plan, a.walk
    (w.whole || !(op.policy isa CpuThreaded) || !(plan isa _MFFusedPlan) ||
     _mf_mask(op) !== nothing) && return Int[]
    nb = _mf_nbands(plan)
    len = last(plan.dims)
    stride = prod(Base.front(plan.dims))
    band = zeros(Int, len)
    for b in 1:nb
        band[_walk_own(w, _band_range(1:len, nb, b))] .= b
    end
    E = a.test_space.E
    cp, rv = SparseArrays.getcolptr(E), rowvals(E)
    cuts = ones(Int, nb + 1)
    cur = 1
    for r in 1:size(E, 2)
        for k in cp[r]:(cp[r + 1] - 1)
            bk = band[(rv[k] - 1) ÷ stride + 1]
            k == cp[r] && bk > cur && (cuts[(cur + 1):bk] .= r; cur = bk)
            bk == cur || return Int[]
        end
    end
    cuts[(cur + 1):(nb + 1)] .= size(E, 2) + 1
    return cuts
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

# On a cut walk the closure computes the padded columns the walk reads only. Under
# `CpuThreaded` a `:static` region costs more than either closure pass (Figure 1 on 40×36
# points, 2 threads: 3.5 and 2.9 ns per unknown in bands, 1.7 and 1.1 serially), so the
# product runs in one region of spawned bands (`_ghost_mf_bands!`), or else serially; a
# `CpuPolyester` region costs about 1 µs.
function _ghost_mf_apply!(y, op::GhostMatrixFreeOperator, x, α, β, xpad, ypad)
    Et = op.trial_space.Et
    E = op.test_space.E
    # `outcut` is nonempty only with a fused plan (`_out_cuts`); the type test folds, so an
    # operator without one compiles no band.
    if op.padded.plan isa _MFFusedPlan && !isempty(op.outcut) && xpad === op.xpad
        _ghost_mf_bands!(y, op, x, α, β)
    elseif op.walk.whole
        _run_bands!(op.policy, _csc_columns_band!, xpad, SparseArrays.getcolptr(Et),
            rowvals(Et), nonzeros(Et), x)
        _mf_walked_mul!(ypad, op.padded, xpad, op.walk)
        _run_bands!(op.policy, _ghost_mf_out_band!, y, SparseArrays.getcolptr(E),
            rowvals(E), nonzeros(E), ypad, op.mask.chunks, α, β)
    else
        cols = op.walk.cols
        policy = _cut_closure_policy(op.policy)
        _run_bands!(policy, _closure_columns_band!, xpad, SparseArrays.getcolptr(Et),
            rowvals(Et), nonzeros(Et), x, cols.ptr, cols.lo, cols.hi, cols.grid, cols.lines)
        _mf_walked_mul!(ypad, op.padded, xpad, op.walk)
        _run_bands!(policy, _ghost_mf_out_band!, y, SparseArrays.getcolptr(E),
            rowvals(E), nonzeros(E), ypad, op.mask.chunks, α, β)
    end
    _mf_identity_rows!(y, x, α, isempty(op.mask) ? nothing : op.mask)
    return nothing
end

@inline _cut_closure_policy(policy) = policy
@inline _cut_closure_policy(::CpuThreaded) = CpuSerial()

# A `CpuThreaded` product on a cut walk: the closure columns and the zeroed padded rows on
# the calling task, then one region of spawned bands, band `b` of the fused sweep walking
# its band (`_mf_band_task!`) and summing the reduced rows that read its rows alone
# (`outcut`). The region ends in `@sync`, which blocks rather than spins, so a band runs on
# whichever thread is free, whatever task, sticky or not, made the product. A first region
# for the closure saved its 4.7 µs of work on Figure 1 (40×36 points, 2 threads) and cost
# about as much again, with 112 to 224 more bytes. Each value is summed as the serial
# product sums it. The spawned bands share one `Ref`, as the fused sweep's do.
function _ghost_mf_bands!(y, op::GhostMatrixFreeOperator, x, α, β)
    nb = length(op.outcut) - 1
    Et, cols = op.trial_space.Et, op.walk.cols
    _closure_columns_band!(op.xpad, SparseArrays.getcolptr(Et), rowvals(Et), nonzeros(Et),
        x, cols.ptr, cols.lo, cols.hi, cols.grid, cols.lines, 1, 1)
    fill!(op.ypad, zero(eltype(op.ypad)))
    s = ActionSink(op.ypad, op.xpad, true, nothing, nothing)
    shared = Ref((y, op, α, β, s))
    @sync begin
        for b in 2:nb
            Threads.@spawn _ghost_mf_walk_band!(shared, b)
        end
        _ghost_mf_walk_band!(shared, 1)
    end
    return nothing
end

@noinline function _ghost_mf_walk_band!(shared::Base.RefValue, b::Int)
    y, op, α, β, s = shared[]
    cuts = op.outcut
    nb = length(cuts) - 1
    _mf_band_task!(s.y, s, op.walk, op.padded.plan, nb, nb, b)
    E = op.test_space.E
    _ghost_mf_out_rows!(y, SparseArrays.getcolptr(E), rowvals(E), nonzeros(E), op.ypad,
        op.mask.chunks, α, β, cuts[b]:(cuts[b + 1] - 1))
    return nothing
end

# `_csc_columns_band!` over the padded columns of a cut walk alone (`_GridLines`, as its
# arrays and isbits fields, so that a `CpuPolyester` task boxes nothing), the ones its
# product reads (`_mf_walked_mul!`), band `b` of `nbands` taking a range of lines; `y`
# keeps its other entries.
@noinline function _closure_columns_band!(y, colptr, rowval, nz, x, ptr, lo, hi, grid,
        lines, nbands::Int, b::Int)
    _closure_columns_lines!(y, colptr, rowval, nz, x, ptr, lo, hi, grid,
        _band_range(1:length(lines), nbands, b))
    return nothing
end

# The same over the lines `ls` of the walk's columns.
@inline function _closure_columns_lines!(y, colptr, rowval, nz, x, ptr, lo, hi, grid, ls)
    T = eltype(y)
    lin = LinearIndices(grid)
    tail = CartesianIndices(Base.tail(axes(grid)))
    @inbounds for l in ls
        J = Tuple(tail[l])
        for t in ptr[l]:(ptr[l + 1] - 1)
            r0 = lin[CartesianIndex(lo[t], J...)] - lo[t]
            # The columns of a range are adjacent, so their entries are too.
            k = colptr[r0 + lo[t]]
            for r in (r0 + lo[t]):(r0 + hi[t])
                kend = colptr[r + 1]
                s = zero(T)
                while k < kend
                    s = convert(T, s + nz[k] * x[rowval[k]])
                    k += 1
                end
                y[r] = s
            end
        end
    end
    return nothing
end

# Entries `r` of band `b` of `y = α E_vᵀ w + β y`, from column `r` of `E_v`; `β == 0`
# overwrites `y`, so a `NaN` in it does not survive. A row the nonempty `mask` marks keeps
# `β y[r]` only: `_mf_identity_rows!` adds its `α x[r]` after. The mask crosses as its
# words (`_MFMask`): a `BitVector` among the `@batch` arguments puts them on the heap.
@noinline function _ghost_mf_out_band!(y, colptr, rowval, nz, w, chunks, α, β, nbands::Int,
        b::Int)
    _ghost_mf_out_rows!(y, colptr, rowval, nz, w, chunks, α, β,
        _band_range(1:length(y), nbands, b))
    return nothing
end

# The same over the entries `rs`.
@inline function _ghost_mf_out_rows!(y, colptr, rowval, nz, w, chunks, α, β, rs)
    T = eltype(y)
    marked = !isempty(chunks)
    isempty(rs) && return nothing
    # The columns `rs` are adjacent, so their entries are too.
    k = @inbounds colptr[first(rs)]
    @inbounds for r in rs
        kend = colptr[r + 1]
        s = zero(T)
        if !marked || _mf_live(_MFMask(chunks), r)
            while k < kend
                s = convert(T, s + nz[k] * w[rowval[k]])
                k += 1
            end
        end
        k = kend
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
