# utils/backend_profile.jl: `profile_backends`, an explicit, once-per-session measurement of
# which execution policy pays off at which size on this host.
#
# It times one sweep kernel through `_sweep_for!`, the seam every threaded loop in this
# package goes through, so a policy that wins here wins at the same size in a weight build
# or a scatter. It is not an assembly benchmark, and it is not called by `backend` or
# `gridspace`: both are on hot internal paths and must stay measurement-free.
#
# Extension point, for a device extension (the Metal one adds a `GpuKernel` row):
#   - `_profile_eltype(policy)`: the widest element type the policy can sweep, `Float64`
#     by default. Every row is timed on the declared type of smallest `sizeof`, so the
#     rows compare (`Float32` for all of them once the Metal row is present).
#   - `_profile_time(policy, T, n)`: seconds for the best of three sweeps over `n` points
#     of a vector of `T`, after one warm run. The host method serves every `CpuPolicy`; a
#     device extension adds a method for its own policy type that allocates the device
#     vector, runs the sweep through `_sweep_for!` and synchronises before reading the
#     clock.
#   - `_profile_label(policy)`: the row label, e.g. `"Parallel()"`.
#   - `_profile_spelling(policy)`: the expression to type to get that policy.
#   - `_profile_available(policy)`: whether `_profile_candidates` lists the policy. `false`
#     by default; a device extension answers `true` only on a functional device, so a
#     loaded but unusable extension adds no row.

"""
    _ProfileKernel{T}

The kernel `profile_backends` sweeps: `i -> sqrt(T(i)) * a + b`. An isbits struct rather
than a closure, so the same value is a legal device kernel argument, and cheap enough per
point that the sweep is bound by the policy's memory traffic and scheduling, as the
weight and scatter loops are.
"""
struct _ProfileKernel{T <: AbstractFloat}
    a::T
    b::T
end

@inline (k::_ProfileKernel{T})(i) where {T} = sqrt(T(i)) * k.a + k.b

# The smallest speed-up over `Serial()` that counts as a crossover: below it, run-to-run
# noise on a loaded host can flip a row.
const _PROFILE_SPEEDUP = 1.2

# Sizes swept, from L1-resident to far past the last-level cache.
const _PROFILE_SIZES = 2 .^ (10:2:22)

"""
    _profile_eltype(policy::ExecutionPolicy) -> Type{<:AbstractFloat}

The widest element type `policy` can sweep. `Float64` unless a device extension says
otherwise; `profile_backends` times every row on the narrowest of the declared types.
"""
_profile_eltype(::ExecutionPolicy) = Float64

"""
    _profile_time(policy::ExecutionPolicy, ::Type{T}, n::Integer) -> Float64

Seconds taken by the fastest of three sweeps of `_ProfileKernel{T}` over `1:n` of a
`Vector{T}` under `policy`, after one warm run that compiles and faults the pages in.
The vector is allocated inside the call, so the timed region holds the sweep alone.

Device extensions add a method for their own policy type; see the note at the top of
`src/utils/backend_profile.jl`.
"""
function _profile_time(policy::CpuPolicy, ::Type{T}, n::Integer) where {T <: AbstractFloat}
    v = Vector{T}(undef, n)
    k = _ProfileKernel(T(1), T(0.5))
    idxs = 1:n
    _sweep_for!(policy, v, idxs, k)
    best = typemax(UInt64)
    for _ in 1:3
        t0 = time_ns()
        _sweep_for!(policy, v, idxs, k)
        best = min(best, time_ns() - t0)
    end
    return Float64(best) * 1.0e-9
end

_profile_label(::CpuSerial) = "Serial()"
_profile_label(::CpuThreaded) = "Parallel()"
_profile_label(::CpuPolyester) = "CpuPolyester()"

_profile_spelling(policy) = "backend(policy = $(_profile_label(policy)))"

# Device policies are listed only when their extension reports a usable device.
_profile_available(::ExecutionPolicy) = false

# `Serial()` is always the baseline, in the first column. `CpuPolyester()` is a candidate
# only once `using Polyester` has loaded its extension; without it the sweep throws.
function _profile_candidates()
    policies = Any[Serial(), Parallel()]
    Base.get_extension(@__MODULE__, :BramblePolyesterExt) !== nothing &&
        push!(policies, CpuPolyester())
    _profile_available(GpuKernel()) && push!(policies, GpuKernel())
    return policies
end

"""
    BackendProfile

Result of `profile_backends`: plain data, printed as a
table.

# Fields
- `eltype::DataType`: Element type every policy was timed on.
- `sizes::Vector{Int}`: Number of points of each sweep.
- `labels::Vector{String}`: One label per policy; the first is the `Serial()` baseline.
- `spellings::Vector{String}`: The expression that selects each policy.
- `times::Matrix{Float64}`: Seconds, `times[i, j]` for `sizes[i]` under policy `j`.
"""
struct BackendProfile
    eltype::DataType
    sizes::Vector{Int}
    labels::Vector{String}
    spellings::Vector{String}
    times::Matrix{Float64}
end

"""
    _crossover(p::BackendProfile, j::Integer) -> Union{Int, Nothing}

Smallest size from which policy `j` stays at least `_PROFILE_SPEEDUP` times faster than the
baseline (column 1) at every larger size of the sweep, or `nothing` if it never does.
"""
function _crossover(p::BackendProfile, j::Integer)
    first_size = nothing
    for i in reverse(eachindex(p.sizes))
        p.times[i, 1] >= _PROFILE_SPEEDUP * p.times[i, j] || break
        first_size = p.sizes[i]
    end
    return first_size
end

function show(io::IO, p::BackendProfile)
    return print(io, "BackendProfile(", join(p.labels, ", "), "; ", length(p.sizes), " sizes)")
end

function show(io::IO, ::MIME"text/plain", p::BackendProfile)
    cell(t) = string(round(t * 1.0e6; sigdigits = 3))
    cols = [vcat(p.labels[j], [cell(t) for t in view(p.times, :, j)]) for j in eachindex(p.labels)]
    ncol = vcat("n", string.(p.sizes))
    println(io, "Backend profile: one sweep kernel over n points of ", p.eltype, ", best of 3, ",
        Threads.nthreads(), " thread(s). Times in microseconds.")
    println(io)
    for r in eachindex(ncol)
        print(io, "  ", lpad(ncol[r], maximum(length, ncol)))
        for col in cols
            print(io, "  ", lpad(col[r], maximum(length, col)))
        end
        println(io)
    end
    println(io)
    println(io, "Crossover against ", p.labels[1], " (first n from which a policy stays at least ",
        _PROFILE_SPEEDUP, "x faster up to n = ", p.sizes[end], "):")
    for j in 2:length(p.labels)
        n = _crossover(p, j)
        println(io, "  ", p.labels[j], ": ", n === nothing ? "never" : "n = $n")
    end
    println(io)
    println(io, "Select a policy with:")
    for j in 2:length(p.labels)
        println(io, "  ", p.labels[j], ": ", p.spellings[j])
    end
    println(io)
    println(io, "The table times one sweep kernel, not assembly.")
    println(io, "Matrix storage is not profiled. To compare csr_backend() against the default CSC storage,")
    print(io, "assemble your own form under each backend and time it; see ?Bramble.profile_backends.")
    return nothing
end

"""
    profile_backends() -> BackendProfile

Time one sweep kernel under each execution policy this session can run on the host, for
sizes `2^10, 2^12, ..., 2^22`, and report where each policy overtakes `Serial()`.

Each policy sweeps a vector through the same loop the package's own weight and scatter
builds use, with an `isbits` kernel (`i -> sqrt(i) * a + b`): one warm run, then the best of
three runs. The candidates are `Serial()`, `Parallel()`, and `CpuPolyester()` once
`using Polyester` has loaded its extension. With `using Metal` on a functional device a
`GpuKernel()` row is added, timed with a device synchronisation. Every row is timed on the
same element type, so the rows compare: `Float64`, or `Float32` for all of them once the
Metal row is present (the only element type Metal offers), in which case the figures are
not drop-in figures for a `Float64` problem. The header of the table names the type. The
host policies take well under a second including compilation. With Metal loaded, the first
call is slow because it is often the session's first GPU operation, which compiles Metal's
own code as well as the sweep kernel; loading Polyester before Metal makes that first GPU
operation markedly slower (an upstream interaction), and loading it after Metal reduces it.
Later calls take well under a second. It is meant to be called by hand, once per session,
and is not run by [`backend`](@ref) or `gridspace`.

The result prints as a table of times, the crossover of each policy against `Serial()`
(the first size from which it stays at least 1.2 times faster to the end of the sweep, or
`never`) and the `backend(policy = ...)` spelling to use. The table times one sweep kernel,
not assembly: a policy that wins here is a good first choice, not a guarantee for your
form. The times are plain data in the `eltype`, `sizes`, `labels` and `times` fields.

# Returns
- `BackendProfile`: `eltype`, `sizes`, policy `labels`, `spellings` and the `times` matrix
  in seconds.

# Examples
```julia
using Bramble
Bramble.profile_backends()
```

# Matrix storage

`profile_backends` times execution policies only. It does not compare
[`csr_backend`](@ref) with the default `SparseMatrixCSC` storage: the matrix type only
matters during assembly and solves, and how much it matters depends on the form, the
boundary conditions and the solver. To compare the two on your own problem, load
`SparseMatricesCSR`, build the same mesh, space and form under each backend, and time
`assemble`, `assemble!` and the solve you use:

    using Bramble, SparseMatricesCSR, BenchmarkTools

    Ω = domain(box((0.0, 0.0), (1.0, 1.0)))
    for be in (backend(), csr_backend())
        Wₕ = gridspace(mesh(Ω, (257, 257); backend = be))
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        A = assemble(a; dirichlet = :boundary)
        @btime assemble!(\$A, \$a; dirichlet = :boundary)
    end

Use the policy `profile_backends` recommended in both backends, e.g.
`csr_backend(; policy = Parallel())`, so the comparison isolates the storage.

See also: [`backend`](@ref), [`csr_backend`](@ref), [`Parallel`](@ref).
"""
function profile_backends()
    policies = _profile_candidates()
    sizes = collect(_PROFILE_SIZES)
    T = argmin(sizeof, map(_profile_eltype, policies))
    times = Matrix{Float64}(undef, length(sizes), length(policies))
    for (j, policy) in enumerate(policies), (i, n) in enumerate(sizes)

        times[i, j] = _profile_time(policy, T, n)
    end
    return BackendProfile(
        T, sizes, String[_profile_label(p) for p in policies],
        String[_profile_spelling(p) for p in policies], times)
end
