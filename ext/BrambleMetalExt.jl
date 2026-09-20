module BrambleMetalExt

using Bramble: Bramble, Backend, ExecutionPolicy
using Metal: Metal, MtlArray, MtlMatrix, MtlVector, MetalBackend, mtl
using LinearAlgebra: I
using SparseArrays: SparseArrays, SparseMatrixCSC

import Bramble: vector, matrix, _backend_eye, _backend_zeros, ka_device

# Deliberately no `@compile_workload` here (gpena/Bramble.jl#196): every method below
# allocates real Metal GPU arrays, which needs an actual Metal-capable device. Precompiling
# that on a headless CI runner or a non-Apple-Silicon machine would fail or hang, not just
# run slow -- unlike every other extension in `ext/`, this one is excluded on purpose.

# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

"""
	metal_backend(T::Type = Float32; policy = GpuAsync())

Returns a [`Backend`](@ref) that uses Apple Metal GPU arrays via
[Metal.jl](https://github.com/JuliaGPU/Metal.jl).

The default element type is `Float32` because Metal natively supports
32-bit floats. `Float16` is also supported. `Float64` is NOT supported
on Apple Silicon GPUs.

# Examples

```julia
using Bramble, Metal
b   = metal_backend()          # Backend{MtlVector{Float32}, MtlMatrix{Float32}, GpuAsync}
b32 = metal_backend(Float32)   # same
b16 = metal_backend(Float16)   # half-precision
```
"""
function Bramble._metal_backend(
        ::Type{T}, policy::ExecutionPolicy
) where {T <: Union{Float16, Float32}}
    return Backend{MtlVector{T}, MtlMatrix{T}, typeof(policy)}()
end

# ---------------------------------------------------------------------------
# ka_device — the device-kernel substrate seam (gpena/Bramble.jl#174)
# ---------------------------------------------------------------------------

ka_device(::Backend{<:MtlVector, MT, EP}) where {MT, EP} = MetalBackend()

# ---------------------------------------------------------------------------
# locality — the storage-locality trait (gpena/Bramble.jl#298)
# ---------------------------------------------------------------------------
#
# Locality is derived from storage, not from a caller's intent: an array living in Metal
# device memory answers `DeviceLocality()`, overriding the `HostLocality()` fallback in
# src/utils/backend.jl. This is the one method a GPU extension adds.

Bramble.locality(::Type{<:MtlVector}) = Bramble.DeviceLocality()
Bramble.locality(::Type{<:MtlMatrix}) = Bramble.DeviceLocality()

# ---------------------------------------------------------------------------
# _gpu_functional — the loaded-and-functional predicate gpu_backend needs
# (gpena/Bramble.jl#192)
# ---------------------------------------------------------------------------
#
# More specific than the `::Val` stub in `src/utils/backend.jl` (this one matches only
# `Val(:metal)`), so this is an added method, not an overwrite of the stub.

Bramble._gpu_functional(::Val{:metal}) = Metal.functional()

# ---------------------------------------------------------------------------
# vector / matrix allocation — GPU-side construction
# ---------------------------------------------------------------------------

@inline function vector(::Backend{VT, MT, EP}, n::Integer) where {T, VT <: MtlVector{T}, MT, EP}
    return MtlArray{T}(undef, n)
end

@inline function matrix(
        ::Backend{VT, MT, EP}, n::Integer, m::Integer
) where {T, VT, MT <: MtlMatrix{T}, EP}
    return MtlArray{T}(undef, n, m)
end

# ---------------------------------------------------------------------------
# backend_eye / backend_zeros — Metal-specific implementations
# ---------------------------------------------------------------------------

function _backend_eye(::Type{MtlMatrix{T}}, n::Integer) where {T}
    return mtl(Matrix{T}(I, n, n))
end

function _backend_zeros(::Type{MtlMatrix{T}}, n::Integer) where {T}
    out = MtlArray{T}(undef, n, n)
    Metal.fill!(out, zero(T))
    return out
end

# ---------------------------------------------------------------------------
# Sparse CSR/CSC types in device memory (gpena/Bramble.jl#250)
# ---------------------------------------------------------------------------
#
# Tagged Metal.jl (checked at v1.10.0 on this host) defines neither `MtlSparseMatrixCSR` nor
# `MtlSparseMatrixCSC` (JuliaGPU/Metal.jl#909 is open, not yet tagged). `Metal.GPUArrays`
# (checked at v11.5.14) already ships the abstract taxonomy -- `AbstractGPUSparseMatrixCSR`
# and `AbstractGPUSparseMatrixCSC` -- so these types subtype it directly instead of inventing
# a Bramble-owned hierarchy, and alias to the upstream concrete type the moment it ships, with
# no change to the names Bramble exposes.

if isdefined(Metal, :MtlSparseMatrixCSR)
    const MetalSparseMatrixCSR = Metal.MtlSparseMatrixCSR
else
    """
        MetalSparseMatrixCSR{Tv, Ti} <: Metal.GPUArrays.AbstractGPUSparseMatrixCSR{Tv, Ti}

    A sparse matrix in compressed sparse row (CSR) format, stored in Metal device memory as
    `MtlVector` fields `rowPtr`, `colVal`, `nzVal`, plus the matrix `dims`.

    A Bramble-owned placeholder for `Metal.MtlSparseMatrixCSR`, which tagged Metal.jl does not
    yet provide (JuliaGPU/Metal.jl#909). Once that type ships, this name aliases to it and no
    Bramble call site changes.
    """
    struct MetalSparseMatrixCSR{Tv, Ti} <: Metal.GPUArrays.AbstractGPUSparseMatrixCSR{Tv, Ti}
        rowPtr::MtlVector{Ti}
        colVal::MtlVector{Ti}
        nzVal::MtlVector{Tv}
        dims::NTuple{2, Int}
    end

    Base.size(A::MetalSparseMatrixCSR) = A.dims
    SparseArrays.nnz(A::MetalSparseMatrixCSR) = length(A.nzVal)

    # `Metal.GPUArrays` supplies `Array`/`collect` generically for any
    # `AbstractGPUSparseMatrixCSR` in terms of this method -- it has no generic
    # `SparseMatrixCSC(::AbstractGPUSparseMatrixCSR)` of its own (unlike the CSC case below),
    # so it is defined here.
    function SparseArrays.SparseMatrixCSC(A::MetalSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
        m, n = A.dims
        # `rowPtr`/`colVal`/`nzVal` store A row-major, which is exactly the CSC storage of
        # transpose(A) (an n x m matrix): column j of that CSC holds row j of A. Materialising
        # the transpose twice keeps every step a sparse-to-sparse conversion -- A is never
        # densified.
        Aᵀ = SparseMatrixCSC(n, m, Array(A.rowPtr), Array(A.colVal), Array(A.nzVal))
        return SparseMatrixCSC(transpose(Aᵀ))
    end

    function Metal.Adapt.adapt_structure(to, A::MetalSparseMatrixCSR)
        MetalSparseMatrixCSR(
            Metal.Adapt.adapt(to, A.rowPtr), Metal.Adapt.adapt(to, A.colVal),
            Metal.Adapt.adapt(to, A.nzVal), A.dims
        )
    end
end

if isdefined(Metal, :MtlSparseMatrixCSC)
    const MetalSparseMatrixCSC = Metal.MtlSparseMatrixCSC
else
    """
        MetalSparseMatrixCSC{Tv, Ti} <: Metal.GPUArrays.AbstractGPUSparseMatrixCSC{Tv, Ti}

    A sparse matrix in compressed sparse column (CSC) format, stored in Metal device memory as
    `MtlVector` fields `colPtr`, `rowVal`, `nzVal`, plus the matrix `dims`.

    Mirrors [`MetalSparseMatrixCSR`](@ref): a Bramble-owned placeholder for
    `Metal.MtlSparseMatrixCSC`, aliased away once tagged Metal.jl provides it
    (JuliaGPU/Metal.jl#909).
    """
    struct MetalSparseMatrixCSC{Tv, Ti} <: Metal.GPUArrays.AbstractGPUSparseMatrixCSC{Tv, Ti}
        colPtr::MtlVector{Ti}
        rowVal::MtlVector{Ti}
        nzVal::MtlVector{Tv}
        dims::NTuple{2, Int}
    end

    Base.size(A::MetalSparseMatrixCSC) = A.dims
    SparseArrays.nnz(A::MetalSparseMatrixCSC) = length(A.nzVal)

    # No `SparseMatrixCSC` method needed here: `Metal.GPUArrays` already supplies one
    # generically for any `AbstractGPUSparseMatrixCSC`, built from `size`, `getcolptr`,
    # `rowvals` and `nonzeros` -- all of which resolve from the field names above.

    function Metal.Adapt.adapt_structure(to, A::MetalSparseMatrixCSC)
        MetalSparseMatrixCSC(
            Metal.Adapt.adapt(to, A.colPtr), Metal.Adapt.adapt(to, A.rowVal),
            Metal.Adapt.adapt(to, A.nzVal), A.dims
        )
    end
end

# ---------------------------------------------------------------------------
# locality for the device sparse types (gpena/Bramble.jl#298)
# ---------------------------------------------------------------------------
#
# Same rule as MtlVector/MtlMatrix above -- locality is derived from storage, and these are
# device storage too, whether the name above resolved to the upstream Metal.jl type or to
# the Bramble-owned placeholder.

Bramble.locality(::Type{<:MetalSparseMatrixCSR}) = Bramble.DeviceLocality()
Bramble.locality(::Type{<:MetalSparseMatrixCSC}) = Bramble.DeviceLocality()

# ---------------------------------------------------------------------------
# Host -> device conversion (gpena/Bramble.jl#250)
# ---------------------------------------------------------------------------
#
# Docstrings live on the `metal_sparse_csr`/`metal_sparse_csc` stubs in `src/utils/backend.jl`
# -- that is the copy visible to a user without `using Metal`, matching this package's
# `_metal_backend` idiom.

function Bramble.metal_sparse_csr(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    # CSC(transpose(A)) is exactly A's row-major (CSR) storage: a sparse-to-sparse conversion,
    # not a densifying one.
    Aᵀ = SparseMatrixCSC(transpose(A))
    return MetalSparseMatrixCSR{Tv, Ti}(mtl(Aᵀ.colptr), mtl(Aᵀ.rowval), mtl(Aᵀ.nzval), (m, n))
end

function Bramble.metal_sparse_csc(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return MetalSparseMatrixCSC{Tv, Ti}(mtl(A.colptr), mtl(A.rowval), mtl(A.nzval), size(A))
end

end # module BrambleMetalExt
