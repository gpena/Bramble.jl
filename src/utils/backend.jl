"""
    ExecutionPolicy

Abstract supertype for backend execution policies.

Two regimes sit under it, and the split is the point (gpena/Bramble.jl#191): [`CpuPolicy`](@ref)
for work a CPU loop drives, [`GpuPolicy`](@ref) for work a device schedules. Before the split
there was only "serial or threaded", which had no way to say *where*: a GPU backend was
constructed with `Serial()`, a policy that says a single CPU thread walks the array element by
element -- the one thing a device array refuses.
"""
abstract type ExecutionPolicy end

"""
    CpuPolicy <: ExecutionPolicy

Abstract supertype for the policies a CPU loop executes: [`CpuSerial`](@ref) and
[`CpuThreaded`](@ref).

The CPU sweeps dispatch on this rather than on each concrete policy, so a `GpuPolicy` reaching
one is a method error with a message rather than a scalar-indexing failure several frames in.
"""
abstract type CpuPolicy <: ExecutionPolicy end

"""
    CpuSerial() <: CpuPolicy

Sequential execution policy.

Directs grid operations and form assembly to execute via single-threaded loops.
This is the default execution policy.

Spelled `Serial()` as often as not: `const Serial = CpuSerial`, kept because it is what every
call site, every test and every benchmark key in this repository already says.

See also: [`CpuThreaded`](@ref), [`ExecutionPolicy`](@ref).
"""
struct CpuSerial <: CpuPolicy end

"""
    CpuThreaded() <: CpuPolicy

Multithreaded execution policy.

Directs grid operations and form assembly to execute across CPU threads via static partitioning.
Execution is unconditional: no per-call size thresholds are imposed. For workloads dominated
by small, frequently repeated calls, use [`CpuSerial`](@ref).

Spelled `Parallel()` as often as not: `const Parallel = CpuThreaded`.

`Base.Threads.@threads` is the primitive, and naming it that way leaves room for the others
that are not this one -- Polyester's `@batch` (gpena/Bramble.jl#190) and MPI. [`CpuBatch`](@ref)
is that Polyester-backed sibling: the policy type ships here, but the sweeps it selects live
in the `BramblePolyesterExt` package extension, and requesting one without `using Polyester`
errors the way [`metal_backend`](@ref) does without `using Metal`.

See also: [`CpuSerial`](@ref), [`CpuBatch`](@ref), [`ExecutionPolicy`](@ref).
"""
struct CpuThreaded <: CpuPolicy end

"""
    CpuBatch() <: CpuPolicy

Polyester-batched execution policy.

Directs grid operations and form assembly through `Polyester.jl`'s `@batch`, a primitive
whose per-call overhead is low enough to pay off on grids where [`CpuThreaded`](@ref)'s
`Threads.@threads` does not (gpena/Bramble.jl#190). The sweeps this policy selects are
implemented in the `BramblePolyesterExt` package extension, not here: `using Polyester` must
be loaded before this policy reaches one of them, or the call errors naming the package,
matching [`metal_backend`](@ref)'s precedent. `Parallel()` is untouched and keeps meaning
`Threads.@threads`.

See also: [`CpuThreaded`](@ref), [`CpuSerial`](@ref), [`ExecutionPolicy`](@ref).
"""
struct CpuBatch <: CpuPolicy end

"""
    GpuPolicy <: ExecutionPolicy

Abstract supertype for the policies a device executes, currently [`GpuAsync`](@ref).

See also: [`CpuPolicy`](@ref), [`ExecutionPolicy`](@ref).
"""
abstract type GpuPolicy <: ExecutionPolicy end

"""
    GpuAsync() <: GpuPolicy

Device execution policy: kernels are launched on the accelerator and complete asynchronously.

What [`metal_backend`](@ref) carries by default. A GPU is massively parallel and cannot execute
serially, so the old default of `Serial()` was not a conservative choice but a false statement
about the hardware, and it sent CPU assembly loops at device arrays.

See also: [`GpuPolicy`](@ref), [`ExecutionPolicy`](@ref).
"""
struct GpuAsync <: GpuPolicy end

"""
    Serial

Alias for [`CpuSerial`](@ref). The spelling this package used before the CPU/GPU split
(gpena/Bramble.jl#191), and the one its call sites, tests and benchmark baseline keys use.
"""
const Serial = CpuSerial

"""
    Parallel

Alias for [`CpuThreaded`](@ref). The spelling this package used before the CPU/GPU split
(gpena/Bramble.jl#191).
"""
const Parallel = CpuThreaded

"""
    Backend{VT, MT, EP}()

Compile-time descriptor specifying vector type `VT`, matrix type `MT`, and execution policy `EP`.

# Type parameters
- `VT<:DenseVector`: Concrete dense vector type (for CPU or GPU).
- `MT<:AbstractMatrix`: Concrete matrix type (e.g. `SparseMatrixCSC{Float64, Int}` or `Matrix{Float64}`).
- `EP<:ExecutionPolicy`: Execution policy ([`Serial`](@ref), [`Parallel`](@ref), or [`CpuBatch`](@ref)).

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
    vector_type = Vector{Float64},
    matrix_type = SparseMatrixCSC{Float64, Int},
    policy::ExecutionPolicy = Serial()
) = Backend{vector_type, matrix_type, typeof(policy)}()

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
@inline backend_types(backend::Backend{VT, MT, EP}) where {VT, MT, EP} = eltype(VT), VT, MT, typeof(backend)
@inline backend_types(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = eltype(VT), VT, MT, Backend{VT, MT, EP}

"""
    supports_undef_construction(::Type{AT}) -> Bool

Whether an array type builds from an `undef` initializer, `AT(undef, dims...)`, rather than
from its dimensions alone, `AT(dims...)`.

The contract a custom array type opts into to be usable as a backend's `vector_type` or
`matrix_type`. `true` by default, which is what every `AbstractArray` in `Base` and in every
GPU package this repository has seen answers; a type constructed from its size alone
declares otherwise:

```julia
Bramble.supports_undef_construction(::Type{<:MySizedArray}) = false
```

Read at compile time from the type, so the branch it guards folds away
(gpena/Bramble.jl#100). This replaced a `try`/`catch` that called the `undef` constructor
and caught its failure to decide the same question: exceptions as a dispatch mechanism,
which blocks the compiler from seeing through the allocation path, and which reported a
genuinely broken array type as "tried both, both failed" rather than as the one thing that
was wrong. The three shipped backends never reached it either way -- `Vector`,
`DenseMatrix` and `SparseMatrixCSC` each have their own `@inline` method below -- so what
this buys is a stated contract for the types that do.
"""
supports_undef_construction(::Type{<:AbstractArray}) = true

@noinline function _throw_vector_error(VT, n, undef_form::Bool)
    tried = undef_form ? "$VT(undef, n)" : "$VT(n)"
    other = undef_form ? "supports_undef_construction($VT) = false" :
            "supports_undef_construction($VT) = true"
    error(
        "Cannot create vector of type $VT with size $n: $tried is not defined. Define it, or, if this type constructs the other way, declare Bramble.$other.",
    )
end

@noinline function _throw_matrix_error(MT, n, m, undef_form::Bool)
    tried = undef_form ? "$MT(undef, n, m)" : "$MT(n, m)"
    other = undef_form ? "supports_undef_construction($MT) = false" :
            "supports_undef_construction($MT) = true"
    error(
        "Cannot create matrix of type $MT with size ($n, $m): $tried is not defined. Define it, or, if this type constructs the other way, declare Bramble.$other.",
    )
end

# The one place either constructor is chosen. `hasmethod` catches a type that declares the
# trait it does not honour, which is the only failure left once the choice itself is static.
@inline function _undef_or_sized(::Type{AT}, n::Integer, thrower) where {AT}
    if supports_undef_construction(AT)
        hasmethod(AT, Tuple{UndefInitializer, Int}) || thrower(AT, n, true)
        return AT(undef, n)
    end
    hasmethod(AT, Tuple{Int}) || thrower(AT, n, false)
    return AT(n)
end

@inline function _undef_or_sized(::Type{AT}, n::Integer, m::Integer, thrower) where {AT}
    if supports_undef_construction(AT)
        hasmethod(AT, Tuple{UndefInitializer, Int, Int}) || thrower(AT, n, m, true)
        return AT(undef, n, m)
    end
    hasmethod(AT, Tuple{Int, Int}) || thrower(AT, n, m, false)
    return AT(n, m)
end

"""
    vector(backend::Backend{VT}, n::Integer) -> VT

Allocate an uninitialized vector of length `n` using vector type `VT` configured in `backend`.

# Arguments
- `backend`: Target backend instance.
- `n`: Number of vector elements.

Which constructor is used is decided by [`supports_undef_construction`](@ref)`(VT)`, from
the type alone.

# Throws
- `ErrorException`: If `VT` does not define the constructor its trait declares.
"""
function vector(::Backend{VT, MT, EP}, n::Integer) where {VT, MT, EP}
    return _undef_or_sized(VT, n, _throw_vector_error)
end

# Specialized zero-overhead method for standard Vector{T}
@inline vector(::Backend{VT, MT, EP}, n::Integer) where {MT, T, VT <: Vector{T}, EP} = VT(undef, n)

"""
    matrix(backend::Backend{<:Any, MT}, n::Integer, m::Integer) -> MT

Allocate a matrix of dimensions `n × m` using matrix type `MT` configured in `backend`.

For dense matrix types, allocates uninitialized storage via `MT(undef, n, m)`.
For sparse matrix types (`SparseMatrixCSC`), allocates an empty sparse matrix via `spzeros(T, Ti, n, m)`.

# Arguments
- `backend`: Target backend instance.
- `n`: Number of rows.
- `m`: Number of columns.

Which constructor is used is decided by [`supports_undef_construction`](@ref)`(MT)`, from
the type alone.

# Throws
- `ErrorException`: If `MT` does not define the constructor its trait declares.
"""
function matrix(::Backend{VT, MT, EP}, n::Integer, m::Integer) where {VT, MT, EP}
    return _undef_or_sized(MT, n, m, _throw_matrix_error)
end

# Specialized zero-overhead methods for dense (CPU/GPU) and sparse matrix types
@inline matrix(
    ::Backend{VT, MT, EP}, n::Integer, m::Integer
) where {VT, T, MT <: DenseMatrix{T}, EP} = MT(undef, n, m)
@inline matrix(
    ::Backend{VT, MT, EP}, n::Integer, m::Integer
) where {VT, T, Ti, MT <: SparseMatrixCSC{T, Ti}, EP} = spzeros(T, Ti, n, m)

"""
    backend_eye(backend::Backend, n::Integer) -> AbstractMatrix

Construct an ``n \\times n`` identity matrix matching the matrix type configured in `backend`.
"""
@inline backend_eye(backend::Backend, n::Integer) = _backend_eye(matrix_type(backend), n)
@inline _backend_eye(::Type{<:SparseMatrixCSC{T, Ti}}, n::Integer) where {T, Ti} = SparseMatrixCSC{T, Ti}(I, n, n)
@inline _backend_eye(::Type{<:Matrix{T}}, n::Integer) where {T} = Matrix{T}(I, n, n)
function _backend_eye(::Type{MT}, n::Integer) where {T, MT <: AbstractMatrix{T}}
    A = _undef_or_sized(MT, n, n, _throw_matrix_error)
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
@inline _backend_zeros(::Type{<:SparseMatrixCSC{T, Ti}}, n::Integer) where {T, Ti} = spzeros(T, Ti, n, n)
function _backend_zeros(::Type{MT}, n::Integer) where {T, MT <: AbstractMatrix{T}}
    return fill!(_undef_or_sized(MT, n, n, _throw_matrix_error), zero(T))
end

"""
    eltype(backend::Backend{VT}) -> Type
    eltype(::Type{<:Backend{VT}}) -> Type

Return the coordinate and scalar element type of vector type `VT` configured in `backend`.
"""
@inline Base.eltype(backend::Backend{VT, MT, EP}) where {VT, MT, EP} = eltype(typeof(backend))
@inline Base.eltype(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = eltype(VT)

function Base.show(io::IO, be::Backend{VT, MT, EP}) where {VT, MT, EP}
    if get(io, :compact, false)
        print(io, "Backend{$(eltype(be))}")
    else
        print(io, "Backend(vector = $VT, matrix = $MT, policy = $EP)")
    end
end

"""
    metal_backend(::Type{T} = Float32; policy::ExecutionPolicy = GpuAsync()) -> Backend

Construct a Metal GPU [`Backend`](@ref) backed by `Metal.jl` arrays.

Requires `using Metal` in the caller environment. Apple Silicon GPUs support `Float32`
and `Float16`, but do not support 64-bit floating point arithmetic.

# Arguments
- `T`: Floating-point element type (`Float32` or `Float16`, default: `Float32`).

# Keywords
- `policy`: Execution policy instance (default: [`GpuAsync`](@ref)). A [`CpuPolicy`](@ref) is
  accepted and means what it says -- CPU loops over device arrays -- which currently fails on
  scalar indexing; the default says where the work runs instead of understating it
  (gpena/Bramble.jl#191).

# Throws
- `ErrorException`: If `Metal.jl` is not loaded.
"""
function metal_backend(T::Type = Float32; policy::ExecutionPolicy = GpuAsync())
    return _metal_backend(T, policy)
end
function _metal_backend(::Type, ::ExecutionPolicy)
    return error(
        "metal_backend requires Metal.jl. Add `using Metal` before calling this function."
    )
end

"""
    csr_backend(::Type{T} = Float64; policy::ExecutionPolicy = Serial()) -> Backend

Construct a `SparseMatrixCSR` [`Backend`](@ref) backed by `SparseMatricesCSR.jl`.

Requires `using SparseMatricesCSR` in the caller environment. A finite-difference stencil
is assembled row by row, which compressed sparse row storage reaches without the column
scatter a `SparseMatrixCSC` assembly needs (gpena/Bramble.jl#214).

# Arguments
- `T`: Coordinate and scalar element type (default: `Float64`).

# Keywords
- `policy`: Execution policy instance (default: [`Serial`](@ref)).

# Throws
- `ErrorException`: If `SparseMatricesCSR.jl` is not loaded.
"""
function csr_backend(T::Type = Float64; policy::ExecutionPolicy = Serial())
    return _csr_backend(T, policy)
end
function _csr_backend(::Type, ::ExecutionPolicy)
    return error(
        "csr_backend requires SparseMatricesCSR.jl. Add `using SparseMatricesCSR` before calling this function."
    )
end

"""
    banded_backend(::Type{T} = Float64; policy::ExecutionPolicy = Serial()) -> Backend

Construct a `BandedMatrix` [`Backend`](@ref) backed by `BandedMatrices.jl`.

Requires `using BandedMatrices` in the caller environment. A high-order stencil on a
Cartesian mesh has a fixed, small bandwidth, which a banded LAPACK factorization
(`gbtrf!`/`gbtrs!`) solves without the fill-in a general sparse factorization pays for
(gpena/Bramble.jl#216).

# Arguments
- `T`: Coordinate and scalar element type (default: `Float64`).

# Keywords
- `policy`: Execution policy instance (default: [`Serial`](@ref)).

# Throws
- `ErrorException`: If `BandedMatrices.jl` is not loaded.
"""
function banded_backend(T::Type = Float64; policy::ExecutionPolicy = Serial())
    return _banded_backend(T, policy)
end
function _banded_backend(::Type, ::ExecutionPolicy)
    return error(
        "banded_backend requires BandedMatrices.jl. Add `using BandedMatrices` before calling this function."
    )
end

"""
    block_banded_backend(::Type{T} = Float64; policy::ExecutionPolicy = Serial()) -> Backend

Construct a `BlockBandedMatrix` [`Backend`](@ref) backed by `BlockBandedMatrices.jl`.

Requires `using BlockBandedMatrices` in the caller environment. In 2D and 3D, lexicographic
ordering of a Cartesian mesh turns a banded stencil into a matrix that is block-banded, with
banded sub-blocks, which `BlockBandedMatrices.jl` stores and factorizes without a general
sparse matrix's fill-in (gpena/Bramble.jl#216).

# Arguments
- `T`: Coordinate and scalar element type (default: `Float64`).

# Keywords
- `policy`: Execution policy instance (default: [`Serial`](@ref)).

# Throws
- `ErrorException`: If `BlockBandedMatrices.jl` is not loaded.
"""
function block_banded_backend(T::Type = Float64; policy::ExecutionPolicy = Serial())
    return _block_banded_backend(T, policy)
end
function _block_banded_backend(::Type, ::ExecutionPolicy)
    return error(
        "block_banded_backend requires BlockBandedMatrices.jl. Add `using BlockBandedMatrices` before calling this function."
    )
end
