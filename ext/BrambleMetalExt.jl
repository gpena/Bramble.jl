module BrambleMetalExt

using Bramble: Bramble, Backend, ExecutionPolicy
using Metal: Metal, MtlArray, MtlMatrix, MtlVector, MetalBackend, mtl
using LinearAlgebra: I
import LinearAlgebra: mul!
using SparseArrays: SparseArrays, SparseMatrixCSC

import Bramble: vector, matrix, _backend_eye, _backend_zeros, ka_device
using PrecompileTools: @setup_workload, @compile_workload

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

# ---------------------------------------------------------------------------
# _allocate_from_pattern -- born a form's system matrix in device memory
# (gpena/Bramble.jl#94)
# ---------------------------------------------------------------------------
#
# `matrix_type(backend(form.test_space))` for a Metal backend is `MtlMatrix{T}`
# (`metal_backend`'s own constructor), so an unmodified call falls into the generic
# `::Type{<:AbstractMatrix}` fallback (`bilinear_pattern.jl:35-53`) -- built with
# `Array{T}(undef, ...)` and scattered into with scalar `setindex!`, which fails on a device
# array. `MtlMatrix <: AbstractMatrix`, so a method keyed on `::Type{<:MtlMatrix}` here is
# strictly more specific and wins dispatch without touching that file. The pattern is still
# discovered on the host from `(I_vec, J_vec, V_vec)` -- unchanged, exactly as the
# `SparseMatrixCSC` method reads it -- and only the destination changes: build the CSR arrays
# host-side with `sparse!` (the same combiner `SparseMatrixCSC`'s own method uses) and
# transfer once with `metal_sparse_csr`, rather than allocating dense and scattering
# element-by-element on the device.
function Bramble._allocate_from_pattern(
        ::Type{MT}, nrows::Int, ncols::Int, I_vec::Vector{Int}, J_vec::Vector{Int},
        V_vec::AbstractVector
) where {MT <: MtlMatrix}
    host = SparseArrays.sparse!(I_vec, J_vec, V_vec, nrows, ncols, +)
    return Bramble.metal_sparse_csr(host)
end

# ---------------------------------------------------------------------------
# SpMV / SpMM -- mul!(y, A::CSR, x, α, β) and mul!(C, A::CSR, B, α, β) (gpena/Bramble.jl#250)
# ---------------------------------------------------------------------------
#
# The kernels themselves are `KernelAbstractions` kernels in `BrambleKernelAbstractionsExt`
# (gpena/Bramble.jl#174's departure from #250's `@metal` text -- see this plan's S3.2), so
# these methods do the type/dimension checks and forward the raw `rowPtr`/`colVal`/`nzVal`
# arrays -- never `A` itself, since a struct nesting a device array fails kernel compilation.

@noinline function _check_metal_sparse_eltype(::Type{Tv}) where {Tv}
    Tv <: Union{Float16, Float32} && return nothing
    throw(ArgumentError(
        "Metal sparse mul! supports Float16/Float32 only, got $Tv -- Apple Silicon GPUs " *
        "do not support Float64.",
    ))
end

function mul!(
        y::AbstractVector{Tv}, A::MetalSparseMatrixCSR{Tv, Ti}, x::AbstractVector{Tv},
        α::Number, β::Number
) where {Tv, Ti}
    _check_metal_sparse_eltype(Tv)
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch(
        "A has dimensions $(size(A)) but x has length $(length(x))"
    ))
    length(y) == m || throw(DimensionMismatch(
        "A has dimensions $(size(A)) but y has length $(length(y))"
    ))
    Bramble._launch_spmv_csr!(y, A.rowPtr, A.colVal, A.nzVal, x, Tv(α), Tv(β))
    return y
end

function mul!(
        C::AbstractMatrix{Tv}, A::MetalSparseMatrixCSR{Tv, Ti}, B::AbstractMatrix{Tv},
        α::Number, β::Number
) where {Tv, Ti}
    _check_metal_sparse_eltype(Tv)
    m, n = size(A)
    size(B, 1) == n || throw(DimensionMismatch(
        "A has dimensions $(size(A)) but B has dimensions $(size(B))"
    ))
    size(C) == (m, size(B, 2)) || throw(DimensionMismatch(
        "A has dimensions $(size(A)) but C has dimensions $(size(C))"
    ))
    Bramble._launch_spmm_csr!(C, A.rowPtr, A.colVal, A.nzVal, B, Tv(α), Tv(β))
    return C
end

# `MetalSparseMatrixCSC` has no `mul!` of its own -- matching JuliaGPU/Metal.jl#909's own
# convention (a row-major kernel needs row-major storage), the error names the fix rather
# than leaving a CSC matrix to fail some other, less legible way (a `MethodError`, or a
# silent fall-through to a dense generic fallback via `Metal.GPUArrays`).
#
# Split into a vector and a matrix method, rather than one `::AbstractVecOrMat` method,
# because `LinearAlgebra` itself ships a generic
# `mul!(y::AbstractVector, A::AbstractVecOrMat, x::AbstractVector, α, β)`
# (`stdlib/LinearAlgebra/src/matmul.jl`) that a single `AbstractVecOrMat` method here would
# tie with once `y`/`x` are concretely `AbstractVector` -- an `ArgumentError` needs to win
# outright, not raise a `MethodError: ... is ambiguous` instead.
function mul!(
        ::AbstractVector, A::MetalSparseMatrixCSC, ::AbstractVector, ::Number, ::Number
)
    throw(ArgumentError(_metal_sparse_csc_mul_message))
end

function mul!(
        ::AbstractMatrix, A::MetalSparseMatrixCSC, ::AbstractMatrix, ::Number, ::Number
)
    throw(ArgumentError(_metal_sparse_csc_mul_message))
end

const _metal_sparse_csc_mul_message = "mul! is not supported for MetalSparseMatrixCSC " *
                                      "(matching JuliaGPU/Metal.jl#909's own convention) " *
                                      "-- convert to CSR first, e.g. " *
                                      "`Bramble.metal_sparse_csr(SparseArrays.SparseMatrixCSC(A))`."

if Bramble.PRECOMPILE_WORKLOAD && Sys.isapple() && Metal.functional()
    @setup_workload begin
        @compile_workload begin
            b = Bramble.metal_backend(Float32)
            v = Bramble.vector(b, 4)
            M = Bramble.matrix(b, 4, 4)
            Bramble._backend_eye(b, 4)
            Bramble._backend_zeros(b, 4)
            Bramble.ka_device(b)

            S = SparseMatrixCSC{Float32, Int32}(LinearAlgebra.I, 4, 4)
            csr = Bramble.metal_sparse_csr(S)
            csc = Bramble.metal_sparse_csc(S)
            y = Bramble.vector(b, 4)
            x = Bramble.vector(b, 4)
            mul!(y, csr, x, 1.0f0, 0.0f0)

            I1 = Bramble.interval(0.0f0, 1.0f0)
            dom1 = Bramble.domain(I1, :left => :left, :right => :right)
            m1 = Bramble.mesh(dom1, 4, true; backend = b)
            w1 = Bramble.gridspace(m1)
            u1 = Bramble.Rₕ(w1, x -> sin(Float32(pi) * x[1]))
            Bramble.D₋ₓ(u1)
            Bramble.innerₕ(u1, u1)

            dom2 = Bramble.domain(I1 × I1)
            m2 = Bramble.mesh(dom2, 4, true; backend = b)
            w2 = Bramble.gridspace(m2)
            u2 = Bramble.Rₕ(w2, x -> sin(Float32(pi) * x[1]) * cos(Float32(pi) * x[2]))
            Bramble.D₋ₓ(u2)
            Bramble.innerₕ(u2, u2)
        end
    end
end

end # module BrambleMetalExt
