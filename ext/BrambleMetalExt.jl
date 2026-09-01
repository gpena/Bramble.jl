module BrambleMetalExt

using Bramble: Bramble, Backend, backend
using Metal: Metal, MtlArray, MtlMatrix, MtlVector, mtl
using LinearAlgebra: I

import Bramble: vector, matrix, _backend_eye, _backend_zeros

# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

"""
	metal_backend(T::Type = Float32)

Returns a [`Backend`](@ref) that uses Apple Metal GPU arrays via
[Metal.jl](https://github.com/JuliaGPU/Metal.jl).

The default element type is `Float32` because Metal natively supports
32-bit floats. `Float16` is also supported. `Float64` is NOT supported
on Apple Silicon GPUs.

# Examples

```julia
using Bramble, Metal
b   = metal_backend()          # Backend{MtlVector{Float32}, MtlMatrix{Float32}}
b32 = metal_backend(Float32)   # same
b16 = metal_backend(Float16)   # half-precision
```
"""
function Bramble._metal_backend(::Type{T}) where {T <: Union{Float16, Float32}}
    return Backend{MtlVector{T}, MtlMatrix{T}}()
end

# ---------------------------------------------------------------------------
# vector / matrix allocation — GPU-side construction
# ---------------------------------------------------------------------------

@inline function vector(::Backend{VT, MT}, n::Integer) where {T, VT <: MtlVector{T}, MT}
    return MtlArray{T}(undef, n)
end

@inline function matrix(::Backend{VT, MT}, n::Integer, m::Integer) where {
        T, VT, MT <: MtlMatrix{T}}
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

end # module BrambleMetalExt
