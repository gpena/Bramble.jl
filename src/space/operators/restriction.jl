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

# Marker-restricted evaluation. `index_in_marker` hands back a BitVector mask
# over the linear indices, not a list of indices, so this must test the mask
# rather than iterate it -- iterating a BitVector yields `true`/`false`, which
# previously reached the kernel as an index.
@inline _func2array!(u::AbstractArray, g, masks::NTuple) = _masked_for!(u, masks, g)

# Whole-grid evaluation, which can run in parallel.
@inline _func2array!(u::AbstractArray, g, mesh_indices::CartesianIndices) = _parallel_for!(u, mesh_indices, g)
# Multi-component elements are handled by dispatching per component in `Rₕ!`,
# so a tuple should never reach this function.
@noinline _func2array!(::Tuple, f, mesh) = throw(ArgumentError(
    "_func2array! received a tuple; multi-component restriction dispatches per component in Rₕ!."))

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

Returns the function stored in the [PointwiseEvaluator](@ref).
"""
@inline func(pe::PointwiseEvaluator) = pe.func

"""
	mesh(pe::PointwiseEvaluator)

Returns the mesh stored in the [PointwiseEvaluator](@ref).
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

"""
	Rₕ!(uₕ::VectorElement, f; markers = ())

In-place version of the restriction operator [`Rₕ`](@ref). Evaluates `f` at the
grid points and writes the result into `uₕ`. Returns `nothing`.

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

    u = to_matrix(uₕ)

    # A `let` closure rather than `PointwiseEvaluator(f, Ωₕ)`. The two are
    # semantically identical, but the struct's call operator is compiled as its
    # own instance for every new `f` instead of being inlined into the index
    # loop, which measured 48.8 ms against 6.3 ms per previously unseen closure
    # on a 21-point 1D space. Run time is unchanged. Adding `@inline` to the
    # call operator does not recover it.
    g = let f = f, Ωₕ = Ωₕ
        idx -> f(point(Ωₕ, idx))
    end

    if N == 0
        _func2array!(u, g, indices(Ωₕ))
        return nothing
    end

    mesh_indices = ntuple(i -> index_in_marker(Ωₕ, markers[i]), Val(N))
    _func2array!(u, g, mesh_indices)
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
    return nothing
end

# A single function returning all components: evaluate it once per point and
# scatter, rather than once per component.
@inline function Rₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f;
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {NC, N}
    N == 0 || return _Rₕ_markers!(uₕ, f, markers)

    Ωₕ = mesh(space(uₕ))
    mats = ntuple(i -> to_matrix(components(uₕ)[i]), Val(NC))
    # See the note in the scalar `Rₕ!` above.
    g = let f = f, Ωₕ = Ωₕ
        idx -> f(point(Ωₕ, idx))
    end
    _scatter_for!(mats, indices(Ωₕ), g)
    return nothing
end

# Marker-restricted variant keeps the per-component path, which already handles
# the marker index sets.
@noinline function _Rₕ_markers!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f, markers) where {NC}
    comps = components(uₕ)
    ntuple(i -> Rₕ!(comps[i], x -> f(x)[i]; markers = markers), Val(NC))
    return nothing
end

"""
	Rₕ(Wₕ::AbstractSpaceType, f; markers = ())

Standard nodal restriction operator. It returns a [VectorElement](@ref) with the result of evaluating the function `f` at the points of `mesh(Wₕ)`.

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
# The coefficient type of a restriction is the one `f` returns, promoted against the
# backend's. Promoted rather than taken outright so that an integer-valued `f` still gives
# a Float64 element on a Float64 backend, while a ForwardDiff.Dual-valued one gives a Dual
# element over the same, undifferentiated, Float64 mesh.
#
# The type is read from one evaluation at the first grid point. That is one extra call to
# `f` per restriction, against inferring it, which would have to guess at a return type
# the compiler may not know.
@inline _scalar_value_type(::Type{T}) where {T} = T
@inline _scalar_value_type(::Type{T}) where {T <: Tuple} = eltype(T)

@inline _restricted_value_type(f, p) = _scalar_value_type(typeof(f(p)))
@inline _restricted_value_type(f::Tuple, p) = promote_type(map(
    g -> _scalar_value_type(typeof(g(p))), f)...)

@inline function _restriction_eltype(Wₕ::AbstractSpaceType, f)
    Ωₕ = mesh(Wₕ)
    p = point(Ωₕ, first(indices(Ωₕ)))
    return promote_type(eltype(backend(Wₕ)), _restricted_value_type(f, p))
end

function Rₕ(Wₕ::AbstractSpaceType, f; markers::NTuple{N, Symbol} = NTuple{
        0, Symbol}()) where {N}
    uₕ = element(Wₕ, _restriction_eltype(Wₕ, f))
    Rₕ!(uₕ, f; markers = markers)
    return uₕ
end
