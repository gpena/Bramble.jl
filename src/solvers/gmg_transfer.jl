# gmg_transfer.jl: the grid transfers of geometric multigrid (gpena/Bramble.jl#329).
#
# `prolongate!` is multilinear interpolation from a level of a `GeometricMeshHierarchy` onto
# the next finer one, and `coarsen!` is its exact transpose. Neither stores the matrix: the
# levels nest, so every fine point is either a coarse point or the midpoint (by index) of two
# along each axis, and the weights are read off the fine level's own coordinates.
#
# `prolongate!` runs in place in the fine vector as a tensor-product sweep, one axis at a time:
# the coarse values are injected at the points odd along every axis, then sweep `d` fills the
# points even along axis `d` and odd along every later axis from their two neighbours along
# `d`, which earlier sweeps (or the injection) wrote. The sweeps partition the fine points by
# their last even axis, so each is written once, and no intermediate buffer is needed.
#
# `coarsen!` is not the reverse sweep: that would work on a fine-sized array, which is either
# the caller's input (which must not change) or a buffer, and a buffer has nowhere to live
# without making the hierarchy unsafe to use from two tasks at once. Each coarse value is
# instead gathered from its 3^D fine neighbours with the same per-axis weights, the same
# products `prolongate!` forms, so `coarsen!` is `Pᵀ` to rounding with no storage at all.
#
# Both sweep through `_run_bands!` (vector_calculus.jl), which bands the last axis of each
# sweep's index range: serial on `CpuSerial`, one band per thread under `Threads.@threads` on
# `CpuThreaded`, `Polyester.@batch` on `CpuPolyester`. Within one sweep the points written and
# the points read are disjoint, so the bands never race.

"""
    prolongate!(xf, H::GeometricMeshHierarchy, l::Integer, xc) -> xf

The prolongation `xf = P xc` from level `l - 1` of `H` to level `l`: the multilinear
interpolant of the coarse grid function `xc` at the points of `H[l]`. A fine point that is a
coarse point takes its value; one between two coarse points ``x_{j}, x_{j+1}`` along an axis
takes ``((x_{j+1} - x) u_j + (x - x_{j}) u_{j+1}) / (x_{j+1} - x_{j})``, per axis, from the
actual (non-uniform) coordinates. `P` equals
`Bramble.interpolation_matrix(gridspace(H[l]), gridspace(H[l - 1]))`, which is never built.

The product runs as one sweep per axis, in place in `xf`, under the execution policy of the
backend of `H[l]`. It allocates nothing on a [`CpuSerial`](@ref) backend; on
[`CpuThreaded`](@ref) only the task spawns, whatever the grid size. It keeps no state, so
transfers on one hierarchy may run from several tasks at once.

# Arguments
- `xf`: The fine grid function, overwritten: a vector of length `npoints(H[l])` or a
  [`VectorElement`](@ref) of a space on `H[l]`.
- `H`: The hierarchy.
- `l`: The index of the fine level, `2 ≤ l ≤ length(H)`.
- `xc`: The coarse grid function: a vector of length `npoints(H[l - 1])` or a
  [`VectorElement`](@ref) of a space on `H[l - 1]`. It must not alias `xf`.

# Returns
- `xf`, holding `P xc`.

# Throws
- `ArgumentError`: `l` is not in `2:length(H)`.
- `DimensionMismatch`: `xf` or `xc` has the wrong length.
- `ArgumentError`: `xf` or `xc` is not 1-based, or the two may alias.
- `ArgumentError`: the backend's policy is a [`GpuPolicy`](@ref); device transfers are
  tracked on milestone v4.4.0.

# Examples
Interpolation reproduces a linear function on a non-uniform mesh.
```jldoctest
using Bramble
H = GeometricMeshHierarchy(mesh(domain(interval(0.0, 1.0)), 9, false), 2)
xf = zeros(npoints(H[2]))
prolongate!(xf, H, 2, 2 .* points(H[1]) .+ 1)
xf ≈ 2 .* points(H[2]) .+ 1

# output
true
```

See also: [`coarsen!`](@ref), [`GeometricMeshHierarchy`](@ref).
"""
function prolongate!(
        xf::AbstractVector, H::GeometricMeshHierarchy, l::Integer, xc::AbstractVector
)
    Ωf, fd, cd, policy = _gmg_transfer_setup(:prolongate!, H, l, xf, xc)
    nf = npoints(Ωf, Tuple)
    _run_bands!(policy, _gmg_inject_band!, fd, cd, nf)
    _gmg_axis_sweeps!(policy, fd, _gmg_axis_points(Ωf), nf)
    return xf
end

"""
    coarsen!(xc, H::GeometricMeshHierarchy, l::Integer, xf) -> xc

The transpose of [`prolongate!`](@ref): `xc = Pᵀ xf`, from level `l` of `H` to level `l - 1`,
with `P` the prolongation from level `l - 1` to level `l`. There is no scaling: a coarse
value is the sum of the fine values around it, each weighted by what the coarse point
contributes to it under interpolation. Bramble's bilinear forms carry the discrete measure,
so `Pᵀ A_f P` has the scaling of the coarse form's matrix, not that matrix itself: they are
equal for the 1D stiffness, even on a non-uniform mesh, but in 2D and 3D, or for a mass form,
they differ entrywise by a quarter to a half. Full weighting (`Pᵀ / 2^D`), the choice for
strong-form difference equations, would be off by `2^D`.

Each coarse value is gathered from its ``3^D`` nearest fine points, with the weights
`prolongate!` uses, under the execution policy of the backend of `H[l]`. It allocates
nothing on a [`CpuSerial`](@ref) backend; on [`CpuThreaded`](@ref) only the task spawns,
whatever the grid size. It keeps no state, so transfers on one hierarchy may run from several
tasks at once.

# Arguments
- `xc`: The coarse grid function, overwritten: a vector of length `npoints(H[l - 1])` or a
  [`VectorElement`](@ref) of a space on `H[l - 1]`.
- `H`: The hierarchy.
- `l`: The index of the fine level, `2 ≤ l ≤ length(H)`.
- `xf`: The fine grid function, not modified: a vector of length `npoints(H[l])` or a
  [`VectorElement`](@ref) of a space on `H[l]`. It must not alias `xc`.

# Returns
- `xc`, holding `Pᵀ xf`.

# Throws
- `ArgumentError`: `l` is not in `2:length(H)`.
- `DimensionMismatch`: `xf` or `xc` has the wrong length.
- `ArgumentError`: `xf` or `xc` is not 1-based, or the two may alias.
- `ArgumentError`: the backend's policy is a [`GpuPolicy`](@ref); device transfers are
  tracked on milestone v4.4.0.

# Examples
The rows of `P` sum to one, so `Pᵀ` preserves the sum of a fine grid function.
```jldoctest
using Bramble
H = GeometricMeshHierarchy(mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (9, 5), false), 2)
xf = rand(npoints(H[2]))
xc = coarsen!(zeros(npoints(H[1])), H, 2, xf)
(length(xc), sum(xc) ≈ sum(xf))

# output
(15, true)
```

See also: [`prolongate!`](@ref), [`GeometricMeshHierarchy`](@ref).
"""
function coarsen!(
        xc::AbstractVector, H::GeometricMeshHierarchy, l::Integer, xf::AbstractVector
)
    Ωf, fd, cd, policy = _gmg_transfer_setup(:coarsen!, H, l, xf, xc)
    _run_bands!(policy, _gmg_coarsen_band!, cd, fd, _gmg_axis_points(Ωf), npoints(Ωf, Tuple))
    return xc
end

# The fine level, the raw storage of both vectors and the policy, once `l`, the lengths, the
# indexing and the policy are validated.
@inline function _gmg_transfer_setup(fname, H, l, xf, xc)
    2 <= l <= length(H) || _throw_gmg_level(fname, H, l)
    Ωf, Ωc = H[l], H[l - 1]
    fd, cd = _mf_data(xf), _mf_data(xc)
    (length(fd) == npoints(Ωf) && length(cd) == npoints(Ωc)) ||
        _throw_gmg_lengths(fname, Ωf, Ωc, length(fd), length(cd))
    Base.require_one_based_indexing(fd, cd)
    Base.mightalias(fd, cd) && throw(ArgumentError("$fname: xf and xc must not alias"))
    policy = execution_policy(Ωf)
    policy isa GpuPolicy && _throw_gmg_gpu(fname, policy)
    return Ωf, fd, cd, policy
end

@noinline function _throw_gmg_level(fname, H, l)
    throw(
        ArgumentError(
        "$fname needs the fine level index l in 2:$(length(H)) of a hierarchy with " *
        "$(length(H)) levels, got l = $l",
    ),
    )
end

@noinline function _throw_gmg_lengths(fname, Ωf, Ωc, nf, nc)
    throw(
        DimensionMismatch(
        "$fname between levels of $(npoints(Ωf)) and $(npoints(Ωc)) points got a fine " *
        "vector of length $nf and a coarse one of length $nc",
    ),
    )
end

@noinline function _throw_gmg_gpu(fname, policy)
    throw(
        ArgumentError(
        "$fname does not run on a GpuPolicy ($(typeof(policy))): device multigrid transfers " *
        "are tracked on milestone v4.4.0. Use a CPU backend.",
    ),
    )
end

@inline _gmg_axis_points(Ωₕ::AbstractMeshType{D}) where {D} = ntuple(d -> points(Ωₕ(d)), Val(D))

@inline _gmg_coarse_npoints(nf::NTuple{D, Int}) where {D} = map(k -> (k - 1) ÷ 2 + 1, nf)

# The `b`-th of `nbands` slabs of `R` along its last axis.
@inline _gmg_band_last(R::Tuple, nbands::Int, b::Int) = (
    Base.front(R)..., _band_range(last(R), nbands, b))

# The weights of an even (between-coarse) fine index `m` along one axis, towards the coarse
# points at `m - 1` and `m + 1`. `prolongate!` and `coarsen!` both use exactly these.
@inline function _gmg_weights(x::AbstractVector, m::Int)
    @inbounds x₋, x₀, x₊ = x[m - 1], x[m], x[m + 1]
    h = x₊ - x₋
    return (x₊ - x₀) / h, (x₀ - x₋) / h
end

# The coarse values at the fine points odd along every axis.
@inline function _gmg_inject_band!(fd, cd, nf::NTuple{D, Int}, nbands::Int, b::Int) where {D}
    nc = _gmg_coarse_npoints(nf)
    lf, lc = LinearIndices(nf), LinearIndices(nc)
    R = _gmg_band_last(ntuple(k -> 1:1:nc[k], Val(D)), nbands, b)
    @inbounds for J in CartesianIndices(R)
        fd[lf[CartesianIndex(map(j -> 2j - 1, Tuple(J)))]] = cd[lc[J]]
    end
    return nothing
end

function _gmg_axis_sweeps!(policy, fd, xs::NTuple{D, Any}, nf::NTuple{D, Int}) where {D}
    for d in 1:D
        _run_bands!(policy, _gmg_axis_band!, fd, xs[d], nf, d)
    end
    return nothing
end

# Sweep `d`: the points even along `d` and odd along every later axis, from their neighbours
# along `d` (odd along `d`, so written by the injection or an earlier sweep).
@inline function _gmg_axis_band!(
        fd, x::AbstractVector, nf::NTuple{D, Int}, d::Int, nbands::Int, b::Int
) where {D}
    R = ntuple(k -> k < d ? (1:1:nf[k]) : k == d ? (2:2:(nf[k] - 1)) : (1:2:nf[k]), Val(D))
    R = _gmg_band_last(R, nbands, b)
    lf = LinearIndices(nf)
    s = prod(ntuple(k -> k < d ? nf[k] : 1, Val(D)))
    @inbounds for I in CartesianIndices(R)
        q = lf[I]
        w₋, w₊ = _gmg_weights(x, I[d])
        fd[q] = w₋ * fd[q - s] + w₊ * fd[q + s]
    end
    return nothing
end

# The weights with which the fine points `i - 1`, `i`, `i + 1` along one axis enter the coarse
# point at fine index `i` (odd): zero past either end of the axis.
@inline function _gmg_coarse_weights(x::AbstractVector{T}, i::Int, n::Int) where {T}
    w₋ = i > 1 ? _gmg_weights(x, i - 1)[2] : zero(T)
    w₊ = i < n ? _gmg_weights(x, i + 1)[1] : zero(T)
    return (w₋, one(T), w₊)
end

@inline function _gmg_coarsen_band!(
        cd, fd, xs::NTuple{D, Any}, nf::NTuple{D, Int}, nbands::Int, b::Int
) where {D}
    nc = _gmg_coarse_npoints(nf)
    lf, lc = LinearIndices(nf), LinearIndices(nc)
    R = _gmg_band_last(ntuple(k -> 1:1:nc[k], Val(D)), nbands, b)
    offsets = CartesianIndices(ntuple(_ -> -1:1, Val(D)))
    S = promote_type(eltype(fd), eltype(first(xs)))
    @inbounds for J in CartesianIndices(R)
        i = map(j -> 2j - 1, Tuple(J))
        w = ntuple(k -> _gmg_coarse_weights(xs[k], i[k], nf[k]), Val(D))
        acc = zero(S)
        for O in offsets
            o = Tuple(O)
            all(ntuple(k -> 1 <= i[k] + o[k] <= nf[k], Val(D))) || continue
            ω = prod(ntuple(k -> w[k][o[k] + 2], Val(D)))
            acc += ω * fd[lf[CartesianIndex(i .+ o)]]
        end
        cd[lc[J]] = acc
    end
    return nothing
end
