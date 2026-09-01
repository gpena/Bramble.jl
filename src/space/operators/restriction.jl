#===========================================================================#
# The restriction operator Rₕ.
#
# Evaluates a function at the grid points. It lives with the other operators rather
# than with the element it produces, alongside the stencil families and avgₕ.
#===========================================================================#

########################
#                      #
# Restriction operator #
#                      #
########################


"""
	$(TYPEDEF)

Helper struct to bundle a function with its mesh for pointwise evaluation.

!!! note
    Not used by `Rₕ!` any more. Since the closure replaced it there, in commit `d2ab604`,
    the only caller is `form/dirichlet_constraints.jl`, which is not currently included.
    It is kept for that, not because the restriction operator needs it: routing `Rₕ!`
    through this struct cost 48.8 ms of compilation per previously unseen function
    against 6.3 ms for the equivalent closure, because the call operator is compiled as
    its own instance per function type rather than inlined into the index loop.

This allows passing both a function and its mesh as a single callable object. When called
with a grid index `idx`, it evaluates the function at the physical coordinates corresponding
to that index.

# Fields

$(FIELDS)

# Callable Interface

A `PointwiseEvaluator` is callable:

```julia
pe = PointwiseEvaluator(f, Ωₕ)
value = pe(idx)  # Equivalent to f(point(Ωₕ, idx))
```

This abstraction is used internally by the restriction operator [`Rₕ`](@ref) to evaluate
continuous functions at discrete grid points.

# Example

```julia
# Define a function in physical coordinates
f(x) = sin(x[1]) * cos(x[2])

# Create evaluator
pe = PointwiseEvaluator(f, Ωₕ)

# Evaluate at grid point (i, j)
value = pe(CartesianIndex(i, j))  # Computes f([xᵢ, yⱼ])
```

See also: [`Rₕ`](@ref), [`Rₕ!`](@ref), [`point`](@ref)
"""
struct PointwiseEvaluator{F, M}
    "the function to evaluate at grid points"
    func::F
    "the mesh providing the mapping from indices to physical coordinates"
    mesh::M
end

"""
	func(pe::PointwiseEvaluator)

Returns the function stored in the [`PointwiseEvaluator`](@ref).
"""
@inline func(pe::PointwiseEvaluator) = pe.func

"""
	mesh(pe::PointwiseEvaluator)

Returns the mesh stored in the [`PointwiseEvaluator`](@ref).
"""
@inline mesh(pe::PointwiseEvaluator) = pe.mesh

"""
	(pe::PointwiseEvaluator)(idx)

Evaluate the function at the physical coordinates of grid index `idx`.

# Arguments

  - `idx`: Grid index (e.g., `CartesianIndex(i, j)`)

# Returns

The function value at the physical point corresponding to `idx`.

# Example

```julia
pe = PointwiseEvaluator(f, Ωₕ)
value = pe(CartesianIndex(5, 10))  # Evaluates f at physical point (x₅, y₁₀)
```
"""
(pe::PointwiseEvaluator)(idx) = func(pe)(point(mesh(pe), idx))

@inline _scatter_comp!(::Tuple{}, ::Tuple, i::Int) = nothing
@inline function _scatter_comp!(raws::Tuple, vals::Tuple, i::Int)
    @inbounds raws[1][i] = vals[1]
    return _scatter_comp!(Base.tail(raws), Base.tail(vals), i)
end

"""
	Rₕ!(uₕ::VectorElement, f; markers = ())

In-place version of the restriction operator [`Rₕ`](@ref). Evaluates `f` at the
grid points and writes the result into `uₕ`. Returns `uₕ`.

# Arguments

  - `uₕ::VectorElement`: pre-allocated element to write into.
  - `f`: function of one grid point. It receives a scalar on a 1D mesh and an
    `NTuple{D}` on a `D`-dimensional one -- never an `SVector`.

# Keywords

  - `markers::NTuple{N,Symbol}`: restrict evaluation to the named marked
    regions, leaving every other entry zero. Several markers act as a union.

# Examples

```julia
Rₕ!(uₕ, x -> sin(x))                  # 1D: x is a Float64
Rₕ!(uₕ, x -> sin(x[1]) * cos(x[2]))   # 2D: x is a Tuple{Float64,Float64}

# only the points carrying the :left marker; the rest stay zero
Rₕ!(uₕ, x -> 1.0; markers = (:left,))
```

For an `N`-component element either shape of `f` works and both give the same
result; the single vector-valued function is evaluated once per grid point,
whereas the tuple evaluates each component function separately:

```julia
Rₕ!(uₕ, (f₁, f₂))                     # one function per component
Rₕ!(uₕ, x -> (f₁(x), f₂(x)))          # one function returning all components
```

See also: [`Rₕ`](@ref), [`avgₕ!`](@ref), [`element`](@ref)
"""
@inline function Rₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f;
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    if N > 0
        @debug "Using marker-based restriction" markers
    end

    (; space) = uₕ
    Ωₕ = mesh(space)
    raw = values(uₕ)
    idxs = indices(Ωₕ)
    n = length(idxs)

    if N == 0
        if Threads.nthreads() == 1 || n < PARALLEL_FOR_MIN
            @inbounds for i in 1:n
                raw[i] = f(point(Ωₕ, idxs[i]))
            end
            return uₕ
        end
        _threaded_Rₕ!(raw, Ωₕ, idxs, f)
        return uₕ
    end

    fill!(raw, zero(eltype(raw)))
    for m in markers
        mask = index_in_marker(Ωₕ, m)
        @inbounds for i in 1:n
            if mask[i]
                raw[i] = f(point(Ωₕ, idxs[i]))
            end
        end
    end
    return uₕ
end

@noinline function _threaded_Rₕ!(raw, Ωₕ, idxs, f)
    Threads.@threads :static for i in 1:length(idxs)
        @inbounds raw[i] = f(point(Ωₕ, idxs[i]))
    end
    return nothing
end

# A one-component space is a scalar space, so generic code that builds an
# NC-tuple of functions still works when NC == 1.
@inline Rₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::Tuple{Any};
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N} = Rₕ!(uₕ, f[1]; markers = markers)

# One function per component: each is already independent, so restrict each
# component with its own function.
@inline function Rₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::Tuple;
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {NC, N}
    comps = components(uₕ)
    ntuple(i -> Rₕ!(comps[i], f[i]; markers = markers), Val(NC))
    return uₕ
end

# A single function returning all components: evaluate it once per point and
# scatter, rather than once per component.
@inline function Rₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f;
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {NC, N}
    Ωₕ = mesh(space(uₕ))
    comps = components(uₕ)
    raws = ntuple(i -> values(comps[i]), Val(NC))
    idxs = indices(Ωₕ)
    n = length(idxs)

    if N == 0
        if Threads.nthreads() == 1 || n < PARALLEL_FOR_MIN
            @inbounds for i in 1:n
                vals = f(point(Ωₕ, idxs[i]))
                _scatter_comp!(raws, vals, i)
            end
            return uₕ
        end
        _threaded_scatter_Rₕ!(raws, Ωₕ, idxs, f)
        return uₕ
    end

    for raw in raws
        fill!(raw, zero(eltype(raw)))
    end
    for m in markers
        mask = index_in_marker(Ωₕ, m)
        @inbounds for i in 1:n
            if mask[i]
                vals = f(point(Ωₕ, idxs[i]))
                _scatter_comp!(raws, vals, i)
            end
        end
    end
    return uₕ
end

@noinline function _threaded_scatter_Rₕ!(raws, Ωₕ, idxs, f)
    Threads.@threads :static for i in 1:length(idxs)
        vals = f(point(Ωₕ, idxs[i]))
        _scatter_comp!(raws, vals, i)
    end
    return nothing
end

# The coefficient type of a restriction is the one `f` returns, promoted against the
# backend's. Promoted rather than taken outright so that an integer-valued `f` still gives
# a Float64 element on a Float64 backend, while a ForwardDiff.Dual-valued one gives a Dual
# element over the same, undifferentiated, Float64 mesh.
#
# The type is read from one evaluation at the first grid point. That is one extra call to
# `f` per restriction, against inferring it, which would have to guess at a return type
# the compiler may not know.
@inline _scalar_value_type(::Type{T}) where {T} = T
# `eltype` of a tuple type is the join of its fields, so `eltype(Tuple{Float64, Int})` is
# `Real` — abstract, which would make `element(Wₕ, Real)` allocate a `Vector{Real}` of boxed
# pointers with no contiguity and no SIMD. An integer literal among the components is enough
# to trigger it: `Rₕ(Vₕ, x -> (1.0, 2))` measured `eltype = Real`.
#
# `promote_type` over the field types gives what the arithmetic would give anyway — `Float64`
# there — and is unchanged for a homogeneous tuple.
@inline _scalar_value_type(::Type{T}) where {T <: Tuple} = promote_type(fieldtypes(T)...)

@inline _restricted_value_type(f, p) = _scalar_value_type(typeof(f(p)))
@inline _restricted_value_type(f::Tuple, p) = promote_type(map(
    g -> _scalar_value_type(typeof(g(p))), f)...)

# Where to sample `f` to learn its return type. With markers it has to be a point the
# caller actually selected: `f` need not be defined anywhere else, and probing the first
# grid point regardless turned working calls into errors — `Rₕ(Wₕ, x -> sqrt(x - 0.5);
# markers = (:right,))` threw a DomainError at x = 0 while `Rₕ!` with the same arguments
# succeeded, since the in-place form never probes.
#
# If no index is marked, nothing is written and the element type cannot matter, so the
# first grid point is as good as any.
@inline _probe_point(Ωₕ, ::NTuple{0, Symbol}) = point(Ωₕ, first(indices(Ωₕ)))

function _probe_point(Ωₕ, markers::NTuple{N, Symbol}) where {N}
    idxs = indices(Ωₕ)
    lin = LinearIndices(idxs)
    for m in markers
        mask = index_in_marker(Ωₕ, m)
        @inbounds for idx in idxs
            mask[lin[idx]] && return point(Ωₕ, idx)
        end
    end
    return point(Ωₕ, first(idxs))
end

@inline function _restriction_eltype(Wₕ::AbstractSpaceType, f,
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    Ωₕ = mesh(Wₕ)
    return promote_type(
        eltype(backend(Wₕ)), _restricted_value_type(f, _probe_point(Ωₕ, markers)))
end

"""
	Rₕ(Wₕ::AbstractSpaceType, f; markers = ())

Standard nodal restriction operator. It returns a [`VectorElement`](@ref) with the result of evaluating the function `f` at the points of `mesh(Wₕ)`.

# The shape of `f`

`f` is called with the coordinates of one grid point: a scalar for a 1D mesh and
an `NTuple{D}` for a `D`-dimensional one. It is never passed an `SVector`.

```julia
Rₕ(Wₕ, x -> sin(x))                # 1D:  x is a Float64
Rₕ(Wₕ, x -> sin(x[1]) * x[2])      # 2D:  x is a Tuple{Float64,Float64}
```

For an `N`-component space either form works and both give the same result:

```julia
Rₕ(Vₕ, (f₁, f₂))                   # one function per component
Rₕ(Vₕ, x -> (f₁(x), f₂(x)))        # one function returning all components
```

Prefer the second when the components share work: it is evaluated once per grid
point, while the first evaluates each component function separately.

`markers` restricts evaluation to the named marked regions, leaving every other
entry zero. [`avgₕ`](@ref) takes the same keyword; it additionally takes
`quad_points`, which has no meaning here because nodal restriction involves no
quadrature.

See also: [`Rₕ!`](@ref), [`avgₕ`](@ref).
"""
function Rₕ(Wₕ::AbstractSpaceType, f; markers::NTuple{N, Symbol} = NTuple{
        0, Symbol}()) where {N}
    uₕ = element(Wₕ, _restriction_eltype(Wₕ, f, markers))
    return Rₕ!(uₕ, f; markers = markers)
end
