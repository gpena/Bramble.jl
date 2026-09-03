"""
	ExecutionPolicy

Abstract supertype for the execution intent a [`Backend`](@ref) carries: [`Serial`](@ref)
or [`Parallel`](@ref).

Named for intent rather than mechanism on purpose. `Parallel` is realised as
`Threads.@threads` on a CPU backend today, and is simply how a GPU backend already runs --
an earlier `Threaded` name would have meant nothing there.
"""
abstract type ExecutionPolicy end

"""
	Serial

Run every per-index and per-thread-capable operation (`Rₕ!`, `avgₕ!`, form assembly, ...) as
a plain loop, unconditionally. The default policy.
"""
struct Serial <: ExecutionPolicy end

"""
	Parallel

Run every per-index and per-thread-capable operation across threads, unconditionally.

There is no size threshold: choosing `Parallel()` on a backend means every eligible call
threads, however small, however many times a time-stepping loop repeats it. That is a
deliberate trade -- the alternative is an automatic per-call size heuristic, which would
have to be tuned separately for every operation (`Rₕ!`'s crossover is not `avgₕ!`'s), tuning
the caller cannot see or override. Choose [`Serial`](@ref) instead for small, frequently
repeated calls.
"""
struct Parallel <: ExecutionPolicy end

"""
	$(TYPEDEF)

A singleton structure containing type-level configuration for backend linear algebra objects.
This structure has no fields, only type parameters `VT`, `MT` and `EP`.

This allows specifying the desired concrete types for dense vectors (CPU or GPU)
and matrices (e.g., dense `Vector`, sparse or dense matrices like `SparseMatrixCSC` or `Matrix`,
different element types like `Float32`, `Float64`, or custom GPU arrays), and the
[`ExecutionPolicy`](@ref) -- [`Serial`](@ref) or [`Parallel`](@ref) -- every
threading-capable operation over this backend uses.
"""
struct Backend{VT <: DenseVector, MT <: AbstractMatrix, EP <: ExecutionPolicy} end

"""
	$(SIGNATURES)

Returns the vector type (`VT`) associated with the given [`Backend`](@ref) instance or type.
"""
@inline vector_type(::Backend{VT, MT, EP}) where {VT, MT, EP} = VT
@inline vector_type(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = VT

"""
	$(SIGNATURES)

Returns the matrix type (`MT`) associated with the given [`Backend`](@ref) instance or type.
"""
@inline matrix_type(::Backend{VT, MT, EP}) where {VT, MT, EP} = MT
@inline matrix_type(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = MT

"""
	$(SIGNATURES)

Returns the [`ExecutionPolicy`](@ref) instance ([`Serial`](@ref) or [`Parallel`](@ref))
associated with the given [`Backend`](@ref), mesh, or space.
"""
@inline execution_policy(::Backend{VT, MT, EP}) where {VT, MT, EP} = EP()
@inline execution_policy(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = EP()

"""
	$(SIGNATURES)

Creates a linear algebra [`Backend`](@ref) using keyword arguments.

Defaults to standard dense `Float64` vectors and `SparseMatrixCSC` matrices, run serially:
- `vector_type = Vector{Float64}`
- `matrix_type = SparseMatrixCSC{Float64,Int}`
- `policy = Serial()`

# Examples

```jldoctest
julia> dense_sparse = backend() # Default backend (Dense-Sparse Float64, Serial)

julia> dense_dense = backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})

julia> using SparseArrays;
       SMat{T} = SparseMatrixCSC{T,Int};
       T32 = Float32;
       dense32 = backend(vector_type = Vector{T32}, matrix_type = SMat{T32})

julia> threaded = backend(policy = Parallel())
```
"""
@inline backend(;
    vector_type = Vector{Float64}, matrix_type = SparseMatrixCSC{
        Float64, Int},
    policy::ExecutionPolicy = Serial()) = Backend{
    vector_type, matrix_type, typeof(policy)}()

"""
	backend(::Type{T}; policy = Serial())

The default dense-vector, sparse-matrix backend over the element type `T`, run serially
unless `policy = Parallel()` is given.

`backend(Float64)` is what `backend()` returns. The single-argument form exists so that a
mesh can take its element type from the domain it is built on rather than always being
`Float64`: `mesh` defaults its backend to `backend(eltype(Ω))`.
"""
@inline backend(::Type{T}; policy::ExecutionPolicy = Serial()) where {T} = Backend{
    Vector{T}, SparseMatrixCSC{T, Int}, typeof(policy)}()

"""
	$(SIGNATURES)

Returns a tuple with the backend associated types:
1. Element type of `VT`
2. Vector type `VT`
3. Matrix type `MT`
4. Concrete backend type `Backend{VT,MT,EP}`
"""
@inline backend_types(backend::Backend{VT, MT, EP}) where {VT, MT, EP} = eltype(VT), VT, MT,
typeof(backend)
@inline backend_types(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = eltype(VT),
VT, MT,
Backend{VT, MT, EP}

@noinline function _throw_vector_error(VT, n, e_undef, e_size)
    error("Cannot create vector of type $VT with size $n. Tried $VT(undef, n) (failed: $e_undef) and $VT(n) (failed: $e_size).")
end

@noinline function _throw_matrix_error(MT, n, m, e_undef, e_size)
    error("Cannot create matrix of type $MT with size ($n, $m). Tried $MT(undef, n, m) (failed: $e_undef) and $MT(n, m) (failed: $e_size).")
end

"""
	$(SIGNATURES)

Creates a vector of type `VT` associated with the given [`Backend`](@ref) instance with length `n`.
"""
function vector(::Backend{VT, MT, EP}, n::Integer) where {VT, MT, EP}
    try
        return VT(undef, n)
    catch e_undef
        try
            return VT(n)
        catch e_size
            _throw_vector_error(VT, n, e_undef, e_size)
        end
    end
end

# Specialized zero-overhead method for standard Vector{T}
@inline vector(::Backend{VT, MT, EP}, n::Integer) where {
    MT, T, VT <: Vector{T}, EP} = VT(
    undef, n)

"""
	$(SIGNATURES)

Creates a matrix of type `MT` associated with the given [`Backend`](@ref) instance with dimensions `n` × `m`.
"""
function matrix(backend::Backend{VT, MT, EP}, n::Integer, m::Integer) where {VT, MT, EP}
    try
        return MT(undef, n, m)
    catch e_undef
        try
            return MT(n, m)
        catch e_size
            _throw_matrix_error(MT, n, m, e_undef, e_size)
        end
    end
end

# Specialized zero-overhead methods for dense (CPU/GPU) and sparse matrix types
@inline matrix(::Backend{VT, MT, EP}, n::Integer, m::Integer) where {
    VT, T, MT <: DenseMatrix{T}, EP} = MT(undef, n, m)
@inline matrix(::Backend{VT, MT, EP}, n::Integer,
    m::Integer) where {VT, T, Ti, MT <: SparseMatrixCSC{T, Ti}, EP} = spzeros(T, Ti, n, m)

"""
	$(SIGNATURES)

Constructs an `n` × `n` identity matrix associated with the given [`Backend`](@ref) instance.
"""
@inline backend_eye(backend::Backend, n::Integer) = _backend_eye(matrix_type(backend), n)
@inline _backend_eye(::Type{<:SparseMatrixCSC{T, Ti}}, n::Integer) where {
    T, Ti} = SparseMatrixCSC{T, Ti}(I, n, n)
@inline _backend_eye(::Type{<:Matrix{T}}, n::Integer) where {T} = Matrix{T}(I, n, n)
function _backend_eye(::Type{MT}, n::Integer) where {T, MT <: AbstractMatrix{T}}
    A = MT(undef, n, n)
    fill!(A, zero(T))
    for i in 1:n
        A[i, i] = one(T)
    end
    return A
end

"""
	$(SIGNATURES)

Constructs an `n` × `n` zero matrix associated with the given [`Backend`](@ref) instance.
"""
@inline backend_zeros(backend::Backend, n::Integer) = _backend_zeros(matrix_type(backend), n)
@inline _backend_zeros(::Type{<:SparseMatrixCSC{T, Ti}}, n::Integer) where {
    T, Ti} = spzeros(T, Ti, n, n)
function _backend_zeros(::Type{MT}, n::Integer) where {T, MT <: AbstractMatrix{T}}
    try
        return fill!(MT(undef, n, n), zero(T))
    catch
        return fill!(MT(n, n), zero(T))
    end
end

"""
	$(SIGNATURES)

Returns the coordinate/element type of the vector type (`VT`) used in the given [`Backend`](@ref).
"""
@inline Base.eltype(backend::Backend{
    VT, MT, EP}) where {VT, MT, EP} = eltype(typeof(backend))
@inline Base.eltype(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = eltype(VT)

function Base.show(io::IO, be::Backend{VT, MT, EP}) where {VT, MT, EP}
    if get(io, :compact, false)
        print(io, "Backend{$(eltype(be))}")
    else
        print(io, "Backend(vector = $VT, matrix = $MT, policy = $EP)")
    end
end

"""
	metal_backend(T::Type = Float32; policy = Serial())

Returns a Metal GPU [`Backend`](@ref) using Apple Metal arrays via
[Metal.jl](https://github.com/JuliaGPU/Metal.jl).

Requires loading `Metal.jl` alongside `Bramble.jl`:

```julia
using Bramble, Metal
b = metal_backend()           # Float32 GPU backend
b = metal_backend(Float16)    # half-precision
```

!!! note
    `Float64` is not supported on Apple Silicon GPUs. Use `Float32` or `Float16`.
"""
function metal_backend(T::Type = Float32; policy::ExecutionPolicy = Serial())
    return _metal_backend(T, policy)
end
function _metal_backend(::Type, ::ExecutionPolicy)
    error("metal_backend requires Metal.jl. Add `using Metal` before calling this function.")
end
