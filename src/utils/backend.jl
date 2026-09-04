"""
    ExecutionPolicy

Abstract supertype for backend execution policies.

Directs grid operations and form assembly to execute either sequentially via
[`Serial`](@ref) or across threads via [`Parallel`](@ref).
"""
abstract type ExecutionPolicy end

"""
    Serial() <: ExecutionPolicy

Sequential execution policy.

Directs grid operations and form assembly to execute via single-threaded loops.
This is the default execution policy.

See also: [`Parallel`](@ref), [`ExecutionPolicy`](@ref).
"""
struct Serial <: ExecutionPolicy end

"""
    Parallel() <: ExecutionPolicy

Multithreaded execution policy.

Directs grid operations and form assembly to execute across CPU threads via static partitioning.
Execution is unconditional: no per-call size thresholds are imposed. For workloads dominated
by small, frequently repeated calls, use [`Serial`](@ref).

See also: [`Serial`](@ref), [`ExecutionPolicy`](@ref).
"""
struct Parallel <: ExecutionPolicy end

"""
    Backend{VT, MT, EP}()

Compile-time descriptor specifying vector type `VT`, matrix type `MT`, and execution policy `EP`.

# Type parameters
- `VT<:DenseVector`: Concrete dense vector type (for CPU or GPU).
- `MT<:AbstractMatrix`: Concrete matrix type (e.g. `SparseMatrixCSC{Float64, Int}` or `Matrix{Float64}`).
- `EP<:ExecutionPolicy`: Execution policy ([`Serial`](@ref) or [`Parallel`](@ref)).

See also: [`backend`](@ref), [`vector_type`](@ref), [`matrix_type`](@ref), [`execution_policy`](@ref).
"""
struct Backend{VT <: DenseVector, MT <: AbstractMatrix, EP <: ExecutionPolicy} end

"""
    vector_type(backend::Backend{VT}) -> Type{VT}
    vector_type(::Type{<:Backend{VT}}) -> Type{VT}

Return the vector type `VT` configured for `backend`.
"""
@inline vector_type(::Backend{VT, MT, EP}) where {VT, MT, EP} = VT
@inline vector_type(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = VT

"""
    matrix_type(backend::Backend{<:Any, MT}) -> Type{MT}
    matrix_type(::Type{<:Backend{<:Any, MT}}) -> Type{MT}

Return the matrix type `MT` configured for `backend`.
"""
@inline matrix_type(::Backend{VT, MT, EP}) where {VT, MT, EP} = MT
@inline matrix_type(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = MT

"""
    execution_policy(backend::Backend{<:Any, <:Any, EP}) -> EP
    execution_policy(::Type{<:Backend{<:Any, <:Any, EP}}) -> EP

Return the [`ExecutionPolicy`](@ref) instance ([`Serial`](@ref) or [`Parallel`](@ref)) configured for `backend`.
"""
@inline execution_policy(::Backend{VT, MT, EP}) where {VT, MT, EP} = EP()
@inline execution_policy(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = EP()

"""
    backend(; vector_type = Vector{Float64}, matrix_type = SparseMatrixCSC{Float64, Int}, policy::ExecutionPolicy = Serial()) -> Backend

Construct a [`Backend`](@ref) configuration.

# Keywords
- `vector_type`: Dense vector type (default: `Vector{Float64}`).
- `matrix_type`: Matrix type (default: `SparseMatrixCSC{Float64, Int}`).
- `policy`: Execution policy instance, [`Serial`](@ref) or [`Parallel`](@ref) (default: `Serial()`).

# Returns
- `Backend`: Singleton instance parameterized by `(vector_type, matrix_type, typeof(policy))`.

# Examples
```jldoctest
using Bramble, SparseArrays
b = backend()
b isa Backend{Vector{Float64}, SparseMatrixCSC{Float64, Int}, Serial}

# output
true
```
"""
@inline backend(;
    vector_type = Vector{Float64}, matrix_type = SparseMatrixCSC{
        Float64, Int},
    policy::ExecutionPolicy = Serial()) = Backend{
    vector_type, matrix_type, typeof(policy)}()

"""
    backend(::Type{T}; policy::ExecutionPolicy = Serial()) -> Backend{Vector{T}, SparseMatrixCSC{T, Int}, typeof(policy)}

Construct the default dense-vector, sparse-matrix backend over scalar coordinate type `T`.

Allows meshes to inherit their scalar coordinate type from the underlying geometric domain.

# Arguments
- `T`: Coordinate and scalar element type (e.g. `Float64`, `Float32`).

# Keywords
- `policy`: Execution policy instance ([`Serial`](@ref) or [`Parallel`](@ref), default: `Serial()`).

# Examples
```jldoctest
using Bramble, SparseArrays
b = backend(Float32)
vector_type(b) === Vector{Float32} && matrix_type(b) === SparseMatrixCSC{Float32, Int}

# output
true
```
"""
@inline backend(::Type{T}; policy::ExecutionPolicy = Serial()) where {T} = Backend{
    Vector{T}, SparseMatrixCSC{T, Int}, typeof(policy)}()

"""
    backend_types(backend::Backend{VT, MT, EP}) -> Tuple{Type, Type{VT}, Type{MT}, Type{Backend{VT, MT, EP}}}
    backend_types(::Type{<:Backend{VT, MT, EP}}) -> Tuple{Type, Type{VT}, Type{MT}, Type{Backend{VT, MT, EP}}}

Return a 4-tuple containing `(eltype(VT), VT, MT, Backend{VT, MT, EP})`.
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
    vector(backend::Backend{VT}, n::Integer) -> VT

Allocate an uninitialized vector of length `n` using vector type `VT` configured in `backend`.

# Arguments
- `backend`: Target backend instance.
- `n`: Number of vector elements.

# Throws
- `ErrorException`: If neither `VT(undef, n)` nor `VT(n)` succeeds.
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
    matrix(backend::Backend{<:Any, MT}, n::Integer, m::Integer) -> MT

Allocate a matrix of dimensions `n × m` using matrix type `MT` configured in `backend`.

For dense matrix types, allocates uninitialized storage via `MT(undef, n, m)`.
For sparse matrix types (`SparseMatrixCSC`), allocates an empty sparse matrix via `spzeros(T, Ti, n, m)`.

# Arguments
- `backend`: Target backend instance.
- `n`: Number of rows.
- `m`: Number of columns.

# Throws
- `ErrorException`: If `MT` cannot be constructed with dimensions `(n, m)`.
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
    backend_eye(backend::Backend, n::Integer) -> AbstractMatrix

Construct an ``n \\times n`` identity matrix matching the matrix type configured in `backend`.
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
    backend_zeros(backend::Backend, n::Integer) -> AbstractMatrix

Construct an ``n \\times n`` zero matrix matching the matrix type configured in `backend`.
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
    eltype(backend::Backend{VT}) -> Type
    eltype(::Type{<:Backend{VT}}) -> Type

Return the coordinate and scalar element type of vector type `VT` configured in `backend`.
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
    metal_backend(::Type{T} = Float32; policy::ExecutionPolicy = Serial()) -> Backend

Construct a Metal GPU [`Backend`](@ref) backed by `Metal.jl` arrays.

Requires `using Metal` in the caller environment. Apple Silicon GPUs support `Float32`
and `Float16`, but do not support 64-bit floating point arithmetic.

# Arguments
- `T`: Floating-point element type (`Float32` or `Float16`, default: `Float32`).

# Keywords
- `policy`: Execution policy instance ([`Serial`](@ref) or [`Parallel`](@ref), default: `Serial()`).

# Throws
- `ErrorException`: If `Metal.jl` is not loaded.
"""
function metal_backend(T::Type = Float32; policy::ExecutionPolicy = Serial())
    return _metal_backend(T, policy)
end
function _metal_backend(::Type, ::ExecutionPolicy)
    error("metal_backend requires Metal.jl. Add `using Metal` before calling this function.")
end
