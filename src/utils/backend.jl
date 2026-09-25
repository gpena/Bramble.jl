"""
    Locality

Abstract supertype for where an array's storage lives: [`HostLocality`](@ref) for memory a
CPU loop can index element by element, [`DeviceLocality`](@ref) for memory an accelerator
schedules against instead.

Never declared directly -- [`locality`](@ref) derives it from an array type, an
[`ExecutionPolicy`](@ref), or a [`Backend`](@ref). Introduced by gpena/Bramble.jl#298 so a
`Backend`'s storage and its execution policy answer the same question and can be checked
against each other, rather than each independently claiming a locality that need not agree.

See also: [`locality`](@ref), [`HostLocality`](@ref), [`DeviceLocality`](@ref).
"""
abstract type Locality end

"""
    HostLocality() <: Locality

Storage a CPU loop can index element by element: `Vector`, `Matrix`, `SparseMatrixCSC`, and,
by default, any array type [`locality`](@ref) has not been taught otherwise about.

See also: [`Locality`](@ref), [`DeviceLocality`](@ref), [`locality`](@ref).
"""
struct HostLocality <: Locality end

"""
    DeviceLocality() <: Locality

Storage an accelerator schedules against rather than a CPU loop indexing it directly:
`MtlVector`/`MtlMatrix` and any future GPU array type. Scalar indexing such storage from a
host loop is the failure this trait exists to catch before it happens (gpena/Bramble.jl#298).

See also: [`Locality`](@ref), [`HostLocality`](@ref), [`locality`](@ref).
"""
struct DeviceLocality <: Locality end

"""
    locality(x) -> Locality

Return the [`Locality`](@ref) of `x` -- an array type, an [`ExecutionPolicy`](@ref), or a
[`Backend`](@ref).

Locality is derived from storage, never declared. The fallback method,
`locality(::Type{<:AbstractArray}) = HostLocality()`, treats any array type this package has
not been taught otherwise about as host memory. Any array type subtyping
`GPUArraysCore.AbstractGPUArray` answers `DeviceLocality()` generically (gpena/Bramble.jl#321),
so a GPU package extension no longer has to add its own method for that to hold -- Metal's
`MtlVector`/`MtlMatrix` already subtype it, and any future CUDA/ROCm/oneAPI array type gets
the right answer the same way, for free. `BrambleMetalExt` keeps its own
`MtlVector`/`MtlMatrix`/sparse-matrix methods regardless: Julia dispatches to the more
specific method, and those predate this generic one -- the same idiom as [`ka_device`](@ref):
a method a package extension supplies for its own type, picked up automatically once loaded.

For an [`ExecutionPolicy`](@ref), `locality` answers what the policy claims: a
[`CpuPolicy`](@ref) claims [`HostLocality`](@ref), a [`GpuPolicy`](@ref) claims
[`DeviceLocality`](@ref). For a [`Backend`](@ref), it answers the locality of the backend's
vector type `VT` alone: `VT` is the storage a sweep actually indexes, so that is the
locality that matters for legality. The matrix type `MT` is not consulted
(gpena/Bramble.jl#298), so a device `MT` paired with a host `VT` goes unchecked.

# Examples
```jldoctest
using Bramble
const B = Bramble
B.locality(Vector{Float64}) === B.HostLocality() &&
    B.locality(B.CpuSerial()) === B.HostLocality() &&
    B.locality(B.GpuAsync()) === B.DeviceLocality()

# output
true
```

See also: [`Locality`](@ref), [`HostLocality`](@ref), [`DeviceLocality`](@ref),
[`ka_device`](@ref).
"""
function locality end

@inline locality(::Type{<:AbstractArray}) = HostLocality()

# GPUArraysCore is a direct dependency for exactly this (gpena/Bramble.jl#321): its
# `AbstractGPUArray` is the abstract supertype every GPU array type subtypes -- Metal.jl's
# `MtlArray` among them -- so this one generic method answers `DeviceLocality()` for any
# device array type without that backend's extension needing to redeclare it. Kept more
# specific than the `AbstractArray` fallback above but strictly less specific than
# `BrambleMetalExt`'s own `MtlVector`/`MtlMatrix`/sparse-matrix methods, which Julia still
# dispatches to and which this change does not touch.

@inline locality(::Type{<:GPUArraysCore.AbstractGPUArray}) = DeviceLocality()

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

# The policy half of the locality trait (gpena/Bramble.jl#298): every `CpuPolicy` claims
# host memory, checked against what the backend's vector type actually is.
@inline locality(::CpuPolicy) = HostLocality()

"""
    CpuSerial() <: CpuPolicy

Sequential execution policy.

Directs grid operations and form assembly to execute via single-threaded loops.
This is the default execution policy.

Spelled `Serial()` as often as not: `const Serial = CpuSerial`, kept because it is what every
call site, every test and every benchmark key in this repository already says.

The grid sizes at which [`CpuThreaded`](@ref) and [`CpuBatch`](@ref) start beating this policy
were measured per workload rather than assumed -- see their own docstrings for the numbers.

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

Where "small" ends was measured per workload rather than assumed (gpena/Bramble.jl#299,
`benchmark/polyester_crossover.jl`, commit 4b76d62b, on the Apple M2 host this milestone's
other measurements were taken on, `--threads=4`, AC power): the smallest grid size at which
this policy beats `CpuSerial` twice running is 64-96 points per axis for unmasked `Rₕ!`, 256
for masked `Rₕ!` (an O(perimeter) write against the mesh's `:boundary` marker, not O(n^D)),
and 24-32 for `avgₕ!` at `nq = 3`. Below those sizes `CpuSerial` is faster; the crossover
differs by an order of magnitude between workloads, so a number from one does not transfer to
another. `innerₕ`/`normₕ` have no entry here because they do not thread under this policy at
all: `_dot(::CpuThreaded, ...)` (`src/utils/linear_algebra.jl`) forwards to the identical
serial reduction, so switching to this policy leaves an inner product exactly as fast, or slow,
as [`CpuSerial`](@ref) (gpena/Bramble.jl#112, closed, superseded by #190).

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

The crossover against `CpuSerial` was measured per workload, closing the last open acceptance
criterion of gpena/Bramble.jl#190 (gpena/Bramble.jl#299, `benchmark/polyester_crossover.jl`,
commit 4b76d62b, same Apple M2 host, `--threads=4` and AC power as [`CpuThreaded`](@ref)'s
figures): the smallest grid size at which this policy beats `CpuSerial` twice running is 8-24
points per axis for unmasked `Rₕ!`, 16 for masked `Rₕ!`, 8 for `avgₕ!` at `nq = 3`, and 1,000
elements for `innerₕ`/`_dot` -- a real comparison here, unlike under [`CpuThreaded`](@ref),
whose `_dot` forwards to the serial reduction instead of threading. Every one of these
crossovers falls one to two orders of magnitude below [`CpuThreaded`](@ref)'s own crossover for
the same workload, and this policy beats [`CpuThreaded`](@ref) at every crossover measured.

See also: [`CpuThreaded`](@ref), [`CpuSerial`](@ref), [`ExecutionPolicy`](@ref).
"""
struct CpuBatch <: CpuPolicy end

"""
    GpuPolicy <: ExecutionPolicy

Abstract supertype for the policies a device executes, currently [`GpuAsync`](@ref).

See also: [`CpuPolicy`](@ref), [`ExecutionPolicy`](@ref).
"""
abstract type GpuPolicy <: ExecutionPolicy end

# The other half of the locality trait (gpena/Bramble.jl#298): every `GpuPolicy` claims
# device memory, checked against what the backend's vector type actually is.
@inline locality(::GpuPolicy) = DeviceLocality()

"""
    GpuAsync() <: GpuPolicy

Device execution policy: work is dispatched to the accelerator rather than to a host loop.

What [`metal_backend`](@ref) carries by default. A GPU is massively parallel and cannot execute
serially, so the old default of `Serial()` was not a conservative choice but a false statement
about the hardware, and it sent CPU assembly loops at device arrays.

**Despite the name, a call under this policy does not return before the device has finished.**
Every kernel launched from `BrambleKernelAbstractionsExt` is followed by a
`KernelAbstractions.synchronize`, so the launch is asynchronous but the Bramble-level call that
issued it is not: `Rₕ!`, `avgₕ!` and the difference and average operators have all completed on
the device by the time they return. Nothing in the package currently exposes a way to queue work
and synchronise later.

The name therefore describes the launch, not the call, and that is a wart rather than a design:
`Async` states a property of the API that the API does not have. Renaming it is breaking and
needs a deprecation cycle, so it is scheduled after the locality work rather than as part of it.
That work is gpena/Bramble.jl#298, which separates the memory locality this policy also encodes
-- already implied by the backend's array types -- from the execution strategy it selects.

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

# The one message a Backend gives when its storage locality and its policy locality disagree
# (gpena/Bramble.jl#298, #296): device storage under a CpuPolicy is memory a CPU loop cannot
# index element by element, and host storage under a GpuPolicy has no device to schedule
# against, so neither combination can execute anything. Lives in the inner constructor below
# rather than in `backend`'s keyword functions, so a direct `Backend{VT, MT, EP}()` spelling
# is caught too, not only the keyword paths. Kept distinct from `_throw_gpu_in_cpu_loop`
# (`src/utils/linear_algebra.jl`), which is about a host loop reached at call time, not a
# backend that should never have been constructed.
@noinline function _throw_backend_locality_mismatch(VT, EP)
    throw(
        ArgumentError(
        "Backend vector type $VT has locality $(locality(VT)), but execution policy $EP has " *
        "locality $(locality(EP())): a Backend's storage and its execution policy must agree " *
        "on locality, or the combination cannot execute anything.",
    ),
    )
end

# The one message a Backend gives when its vector and matrix storage disagree on locality
# (gpena/Bramble.jl#298, #296): a backend is meant to be wholly host or wholly device, and a
# device VT with a host MT (or the reverse) is not a configuration anyone means to build.
# Kept as its own helper rather than folded into `_throw_backend_locality_mismatch` above --
# that message is already VERIFIED and asserted on by a test, and this one names a different
# pair of arguments (VT, MT rather than VT, EP), so sharing a helper would mean branching on
# which pair to print rather than just calling the right one.
@noinline function _throw_backend_matrix_locality_mismatch(VT, MT)
    throw(
        ArgumentError(
        "Backend vector type $VT has locality $(locality(VT)), but matrix type $MT has " *
        "locality $(locality(MT)): a Backend's vector and matrix storage must agree on " *
        "locality, or the combination cannot execute anything.",
    ),
    )
end

"""
    Backend{VT, MT, EP}()

Compile-time descriptor specifying vector type `VT`, matrix type `MT`, and execution policy `EP`.

# Type parameters
- `VT<:DenseVector`: Concrete dense vector type (for CPU or GPU).
- `MT<:AbstractMatrix`: Concrete matrix type (e.g. `SparseMatrixCSC{Float64, Int}` or `Matrix{Float64}`).
- `EP<:ExecutionPolicy`: Execution policy ([`Serial`](@ref), [`Parallel`](@ref), or [`CpuBatch`](@ref)).

All three parameters must agree on [`locality`](@ref): a backend is meant to be wholly host --
a `Vector` alongside a CPU sparse or dense matrix, run under a [`CpuPolicy`](@ref) -- or wholly
device -- once a vendor is chosen, `VT`, `MT` and the policy are all that vendor's. Any other
combination (a device `VT` with a host `MT`, a device `VT` with a `CpuPolicy`, or a host `VT`
with a `GpuPolicy`) is rejected at construction rather than accepted and left to fail on first
use (gpena/Bramble.jl#298, #296).

# Throws
- `ArgumentError`: `locality(VT) != locality(EP())`.
- `ArgumentError`: `locality(MT) != locality(VT)`.

See also: [`backend`](@ref), [`vector_type`](@ref), [`matrix_type`](@ref), [`execution_policy`](@ref), [`locality`](@ref).
"""
struct Backend{VT <: DenseVector, MT <: AbstractMatrix, EP <: ExecutionPolicy}
    function Backend{VT, MT, EP}() where {
            VT <: DenseVector, MT <: AbstractMatrix, EP <: ExecutionPolicy}
        locality(VT) === locality(EP()) || _throw_backend_locality_mismatch(VT, EP)
        locality(MT) === locality(VT) || _throw_backend_matrix_locality_mismatch(VT, MT)
        return new{VT, MT, EP}()
    end
end

"""
    vector_type(backend::Backend{VT}) -> Type{VT}
    vector_type(::Type{<:Backend{VT}}) -> Type{VT}

Return the vector type `VT` configured for `backend`.
"""
@inline vector_type(::Backend{VT, MT, EP}) where {VT, MT, EP} = VT
@inline vector_type(::Type{<:Backend{VT, MT, EP}}) where {VT, MT, EP} = VT

# `Backend`'s half of the locality trait (gpena/Bramble.jl#298): reads `VT` alone, since `VT`
# is the storage a sweep actually indexes. `MT` is not consulted, so a device `MT` paired
# with a host `VT` is not caught by this method.
@inline locality(be::Backend) = locality(vector_type(be))
@inline locality(::Type{BT}) where {BT <: Backend} = locality(vector_type(BT))

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

# Throws
- `ArgumentError`: `vector_type`, `matrix_type` and `policy` disagree on [`locality`](@ref) --
  see [`Backend`](@ref).

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

**A function projected on this backend has to be GPU-compilable.** `Rₕ(W, f)` on a
`metal_backend` space compiles `f` into a device kernel rather than calling it from a host
loop, so `f` must be device-compilable: no `Float64` literals (a bare literal like `0.5`
forces double precision, which Apple Silicon GPUs do not support -- write `x[1] / 2`, not
`x[1] * 0.5`), no allocations, and no calls to a non-inlineable or host-only function.
`x -> sin(x[1])` compiles; anything capturing a boxed value or calling out to the host does
not, and fails at compile time with a dynamic-invocation error rather than at the call site.

# Arguments
- `T`: Floating-point element type (`Float32` or `Float16`, default: `Float32`).

# Keywords
- `policy`: Execution policy instance (default: [`GpuAsync`](@ref)). A [`CpuPolicy`](@ref) is
  rejected at construction, not accepted and left to fail on first use: Metal's device arrays
  answer [`DeviceLocality`](@ref) to [`locality`](@ref), a `CpuPolicy` answers
  [`HostLocality`](@ref), and [`Backend`](@ref)'s inner constructor requires the two to agree
  (gpena/Bramble.jl#298, #296).

# Throws
- `ErrorException`: If `Metal.jl` is not loaded.
- `ArgumentError`: If `policy` is a [`CpuPolicy`](@ref) (locality mismatch).
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
    metal_sparse_csr(A::SparseMatrixCSC) -> MetalSparseMatrixCSR

Convert a host `SparseMatrixCSC` to a sparse matrix in compressed sparse row (CSR) format,
stored in Metal device memory (gpena/Bramble.jl#250).

Requires `using Metal` in the caller environment. The conversion is sparse-to-sparse
throughout -- `A` is never densified.

# Arguments
- `A`: The host sparse matrix to convert.

# Throws
- `ErrorException`: If `Metal.jl` is not loaded.
"""
function metal_sparse_csr(A)
    return error(
        "metal_sparse_csr requires Metal.jl. Add `using Metal` before calling this function."
    )
end

"""
    metal_sparse_csc(A::SparseMatrixCSC) -> MetalSparseMatrixCSC

Convert a host `SparseMatrixCSC` to a sparse matrix in compressed sparse column (CSC) format,
stored in Metal device memory (gpena/Bramble.jl#250).

Requires `using Metal` in the caller environment. `SparseMatrixCSC`'s own storage already is
CSC, so no format conversion happens -- its fields are placed in device memory as-is.

# Arguments
- `A`: The host sparse matrix to convert.

# Throws
- `ErrorException`: If `Metal.jl` is not loaded.
"""
function metal_sparse_csc(A)
    return error(
        "metal_sparse_csc requires Metal.jl. Add `using Metal` before calling this function."
    )
end

"""
    _gpu_functional(::Val{name}) -> Bool

Whether the GPU extension named by `name` (a `Symbol`, e.g. `:metal`) has a functional
device, not merely that the corresponding package loaded.

Defaults to `false` for every name: loading a GPU package extension (`using Metal`, and
in future `using CUDA`/`using AMDGPU`) succeeds on any platform, whether or not that
host actually has the accelerator's hardware and drivers -- it degrades gracefully
rather than erroring. `ext/BrambleMetalExt.jl` overrides this for `Val(:metal)` with
`Metal.functional()`, the one call that actually probes the device
(gpena/Bramble.jl#192). [`gpu_backend`](@ref) requires both this and the extension being
Declared with the generic `::Val` signature (rather than `::Val{:metal}` itself) so that
`BrambleMetalExt`'s own `Val(:metal)` method is strictly more specific than this stub --
otherwise the extension's method would silently overwrite this one instead of adding to
it, which precompilation reports as method overwriting.
"""
# Test/mock hook: when set to a Bool, overrides _gpu_functional for testing device failure
# without method overwriting (which emits compiler warnings).
const _gpu_functional_override = Ref{Union{Nothing, Bool}}(nothing)

function _gpu_functional(::Val)
    override = _gpu_functional_override[]
    override !== nothing && return override
    return false
end

"""
    gpu_backend(::Type{T} = Float32; policy::ExecutionPolicy = GpuAsync()) -> Backend

Construct a GPU [`Backend`](@ref) for whichever accelerator extension is loaded **and**
has a functional device.

Detects a loaded GPU package extension with `Base.get_extension`, checks
`_gpu_functional` for that device, and forwards to the backend's own constructor:
currently [`metal_backend`](@ref), under `using Metal` with `Metal.functional()` true. A
CUDA or AMDGPU extension joins this dispatch once it exists (gpena/Bramble.jl#11, v3.5.0).

# Arguments
- `T`: Floating-point element type (default: `Float32`).

# Keywords
- `policy`: Execution policy instance (default: [`GpuAsync`](@ref)).

# Throws
- `ErrorException`: two distinct diagnoses, deliberately not sharing a message. If no GPU
  extension is loaded at all, the message names the package to load, chosen from the host
  architecture: Metal on Apple Silicon, CUDA on Linux/Windows, and a generic "no GPU
  hardware" message otherwise. If a GPU extension *is* loaded but its device is not
  functional (e.g. `using Metal` on a host without working Metal drivers), the message
  says so instead -- that is a driver, hardware or virtualisation problem, not a missing
  `using`.

See also: [`metal_backend`](@ref), [`backend`](@ref).
"""
function gpu_backend(T::Type = Float32; policy::ExecutionPolicy = GpuAsync())
    if Base.get_extension(Bramble, :BrambleMetalExt) !== nothing
        _gpu_functional(Val(:metal)) || return _throw_metal_not_functional()
        return metal_backend(T; policy)
    end
    return _throw_no_gpu_backend()
end

# Kept distinct from `_throw_no_gpu_backend` below on purpose (gpena/Bramble.jl#192, S1.4):
# a functional-device failure is a driver, hardware or virtualisation problem, which no
# `using` statement fixes, so it must not share a message with the "nothing loaded" case.
@noinline function _throw_metal_not_functional()
    return error(
        "gpu_backend found Metal.jl loaded, but Metal.functional() is false: no functional " *
        "Metal device was found. This is a driver, hardware or virtualisation problem, not " *
        "a missing `using Metal`.",
    )
end

# `BrambleMetalExt` is the only GPU extension this package has today; a `BrambleCUDAExt`/
# `BrambleAMDGPUExt` check joins the chain above the same way once those extensions and
# their own `cuda_backend`/`amdgpu_backend` constructors exist (gpena/Bramble.jl#11,
# v3.5.0). `Base.get_extension` on a name nothing declares simply never returns non-`nothing`,
# so there is nowhere to hook a check in today without calling a constructor this package
# does not yet define.
@noinline function _throw_no_gpu_backend()
    if Sys.isapple() && Sys.ARCH === :aarch64
        return error(
            "gpu_backend found no loaded GPU extension. Apple Silicon GPU detected: add " *
            "`using Metal` before calling this function to activate the Metal backend.",
        )
    elseif Sys.islinux() || Sys.iswindows()
        return error(
            "gpu_backend found no loaded GPU extension. Add `using CUDA` before calling " *
            "this function to activate the CUDA backend.",
        )
    else
        return error(
            "gpu_backend found no loaded GPU extension, and no supported GPU hardware was " *
            "found on this host.",
        )
    end
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
