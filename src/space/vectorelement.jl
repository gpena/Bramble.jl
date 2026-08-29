# Getters and setters for VectorElement
"""
	values(uₕ::VectorElement)

Returns the coefficients of the [VectorElement](@ref) `uₕ`.
"""
@inline values(uₕ::VectorElement) = uₕ.data

"""
	to_matrix(uₕ::VectorElement)

Reshapes the flat coefficient vector of `uₕ` into a multidimensional array that matches the logical layout of the grid points.

  - For a scalar space, this returns a D-dimensional array.
  - For an N-component vector space, it returns an N-tuple of arrays, one for each component.
"""
@inline to_matrix(uₕ::VectorElement{<:ScalarGridSpace}) = Base.ReshapedArray(
    values(uₕ), npoints(mesh(space(uₕ)), Tuple), ())
@inline to_matrix(uₕ::VectorElement{<:CompositeGridSpace{N}}) where {N} = ntuple(i -> to_matrix(uₕ(i)), Val(N))

"""
	values!(uₕ::VectorElement, s)

Copies the values of `s` into the coefficients of [VectorElement](@ref) `uₕ`. Returns `nothing`.
"""
@inline function values!(uₕ::VectorElement, s)
    copyto!(values(uₕ), s)
    return nothing
end

"""
	space(uₕ::VectorElement)

Returns the grid space associated with [VectorElement](@ref) `uₕ`.
"""
@inline space(uₕ::VectorElement) = uₕ.space
@inline space_type(::Type{<:VectorElement{S}}) where {S} = S

# Forward array-like methods to the `data` field. This allows a VectorElement
# to behave like a standard Julia vector (e.g., support `size`, `length`, `eltype`).
@forward VectorElement.data (Base.size, Bramble.show)

# A VectorElement wraps a vector, so indexing is linear; without this the
# AbstractArray default of IndexCartesian() is used.
@inline Base.IndexStyle(::Type{<:VectorElement}) = IndexLinear()
@forward VectorElement.space (Bramble.mesh,)

# ==============================================================================
# Component Indexing
# ==============================================================================

"""
	component_range(Wₕ::CompositeGridSpace{N}, i::Int) where N

Returns the degree-of-freedom index range for the `i`-th constituent space of composite space `Wₕ`.
"""
@inline function component_range(Wₕ::CompositeGridSpace{N}, i::Int) where {N}
    @boundscheck (1 <= i <= N) || throw(BoundsError(Wₕ, i))
    return @inbounds component_ranges(Wₕ)[i]
end

"""
	component_ranges(Wₕ::CompositeGridSpace{N}) where N

Returns an `NTuple{N, UnitRange{Int}}` containing the degree-of-freedom ranges for all `N` components.
"""
@inline function component_ranges(Wₕ::CompositeGridSpace{N}) where {N}
    subs = spaces(Wₕ)
    # `ntuple` over `Val(N)` and `cumsum` on a tuple both unroll, so this is a
    # handful of adds with no loop and no allocation. Sizes are summed rather
    # than assumed equal: subspaces of the same *type* can still hold different
    # numbers of degrees of freedom, so nothing here may be inferred from types.
    ns = ntuple(i -> ndofs(subs[i]), Val(N))
    stops = cumsum(ns)
    return ntuple(i -> (stops[i] - ns[i] + 1):stops[i], Val(N))
end

"""
	(uₕ::VectorElement)(i::Int)
	component(uₕ::VectorElement, i::Int)

Extracts a [`VectorElement`](@ref) view of the `i`-th field component of `uₕ`.

For a [`CompositeGridSpace`](@ref), this creates a lightweight, zero-copy view of the
`i`-th component's degrees of freedom. Mutating the returned component modifies `uₕ` in-place.

For a scalar [`ScalarGridSpace`](@ref), `uₕ(1)` returns `uₕ`.

# Examples
```julia
Vₕ = Wₕ^2
uₕ = element(Vₕ)
u_x = uₕ(1)
u_y = uₕ(2)

# In-place component assignment
u_x .= 1.0
```
"""
@inline function (uₕ::VectorElement{<:CompositeGridSpace})(i::Int)
    rng = component_range(space(uₕ), i)
    subspace = spaces(space(uₕ))[i]
    v_data = @views values(uₕ)[rng]
    return VectorElement(v_data, subspace)
end

@inline function (uₕ::VectorElement{<:ScalarGridSpace})(i::Int)
    @boundscheck i == 1 || throw(BoundsError(uₕ, i))
    return uₕ
end

"""
	component(uₕ::VectorElement, i::Int)

Extracts a [`VectorElement`](@ref) view of the `i`-th field component of `uₕ`. Alias for `uₕ(i)`.
"""
@inline component(uₕ::VectorElement, i::Int) = uₕ(i)

"""
	components(uₕ::VectorElement)

Returns an `NTuple` of [`VectorElement`](@ref) views for all components of `uₕ`.
"""
@inline function components(uₕ::VectorElement{<:CompositeGridSpace{N}}) where {N}
    ranges = component_ranges(space(uₕ))
    subs = spaces(space(uₕ))
    raw = values(uₕ)
    return ntuple(i -> VectorElement(@views(raw[ranges[i]]), subs[i]), Val(N))
end

@inline components(uₕ::VectorElement{<:ScalarGridSpace}) = (uₕ,)

# Constructor for VectorElement
"""
	element(Wₕ::AbstractSpaceType, [α::Number])

Returns a [VectorElement](@ref) for grid space `Wₕ` with uninitialized components. if ``\\alpha`` is provided, the components are initialized to ``\\alpha``.
"""
@inline function element(Wₕ::AbstractSpaceType)
    # Get the backend (e.g., CPU, GPU) from the space.
    b = backend(Wₕ)

    # Determine the types for the space, vector, and elements.
    ST = typeof(Wₕ)
    VT = vector_type(b)
    T = eltype(b)

    # Allocate a vector with the correct number of degrees of freedom (DoFs) and return the element.
    return VectorElement{ST, T, VT}(vector(b, ndofs(Wₕ)), Wₕ)
end

# Constructor with a fill value `α`.
function element(Wₕ::AbstractSpaceType, α::Number)
    uₕ = element(Wₕ)
    fill!(uₕ, α)
    return uₕ
end

"""
	element(Wₕ::AbstractSpaceType, v::AbstractVector)

Returns a [VectorElement](@ref) for a grid space `Wₕ` with the same coefficients of `v`.
"""
@inline function element(Wₕ::AbstractSpaceType, v::AbstractVector)
    # Ensure the provided vector has the correct number of DoFs.
    length(v) == ndofs(Wₕ) || throw(DimensionMismatch(
        "input vector has length $(length(v)), but the space has $(ndofs(Wₕ)) degrees of freedom."))
    elem = element(Wₕ)
    copyto!(elem, v)
    return elem
end

# Enable array-like indexing `uₕ[i]` for VectorElement.
@inline Base.@propagate_inbounds getindex(uₕ::VectorElement, i) = getindex(uₕ.data, i)
@inline Base.@propagate_inbounds setindex!(uₕ::VectorElement, val, i) = setindex!(uₕ.data, val, i)

# Create a new, uninitialized VectorElement with the same space as the input.
@inline Base.similar(uₕ::VectorElement) = element(space(uₕ))

# Broadcasting

# Enable broadcasting capabilities (e.g., uₕ .= vₕ .+ 1) for VectorElement.
Base.BroadcastStyle(::Type{<:VectorElement}) = Broadcast.ArrayStyle{VectorElement}()

# Define how to create a `similar` container for the broadcast result, preserving the space.
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}},
        ::Type{ElType}) where {ElType}
    vec_elem = _find_vec_in_broadcast(bc)
    vec_elem === nothing &&
        throw(ArgumentError("No VectorElement found in broadcast expression"))
    return VectorElement(similar(values(vec_elem), ElType), space(vec_elem))
end

# Version without a specified ElType.
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}})
    vec_elem = _find_vec_in_broadcast(bc)
    vec_elem === nothing &&
        throw(ArgumentError("No VectorElement found in broadcast expression"))
    return VectorElement(similar(values(vec_elem)), space(vec_elem))
end

"""
	_find_vec_in_broadcast(bc)

Internal helper to extract a [VectorElement](@ref) from a broadcast expression.

Recursively searches through the arguments of a broadcast expression tree to find
a `VectorElement` instance. This is used by the broadcasting machinery to determine
which function space should be used for the result.

# Arguments

  - `bc`: A broadcast expression, tuple of arguments, or individual value

# Returns

  - The first `VectorElement` found in the expression tree
  - `nothing` if no `VectorElement` is found

# Implementation Notes

Uses multiple dispatch to handle:

  - `Broadcasted` objects: Extract and search arguments
  - `Tuple`s: Recursively search each element
  - `VectorElement`: Return immediately (found!)
  - Other types: Return `nothing` and continue searching

This enables broadcasts like `uₕ .+ vₕ .* 2` to automatically preserve the space information.
"""
_find_vec_in_broadcast(bc::Broadcast.Broadcasted) = _find_vec_in_broadcast(bc.args)
function _find_vec_in_broadcast(args::Tuple)
    _find_vec_in_broadcast(_find_vec_in_broadcast(args[1]), Base.tail(args))
end
_find_vec_in_broadcast(x::VectorElement) = x
_find_vec_in_broadcast(::Any) = nothing # Not a VectorElement
_find_vec_in_broadcast(::Tuple{}) = nothing # End of recursion
_find_vec_in_broadcast(a::VectorElement, rest) = a # Found one
_find_vec_in_broadcast(::Any, rest) = _find_vec_in_broadcast(rest) # Keep searching

function Base.:*(uₕ::VectorElement, vₕ::NTuple{D, VectorElement}) where {D}
    zₕ = ntuple(i -> similar(vₕ[i]), D)
    for i in 1:D
        zₕ[i].data .= uₕ.data .* vₕ[i].data
    end
    return zₕ
end

function Base.:*(a::Number, vₕ::NTuple{D, VectorElement}) where {D}
    zₕ = ntuple(i -> similar(vₕ[i]), D)
    for i in 1:D
        zₕ[i].data .= a .* vₕ[i].data
    end
    return zₕ
end

@inline Base.:*(Vₕ::NTuple{D, VectorElement}, a::Number) where {D} = a * Vₕ
@inline Base.:*(Vₕ::NTuple{D, VectorElement}, uₕ::VectorElement) where {D} = uₕ * Vₕ

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
function Rₕ(Wₕ::AbstractSpaceType, f; markers::NTuple{N, Symbol} = NTuple{
        0, Symbol}()) where {N}
    uₕ = element(Wₕ)
    Rₕ!(uₕ, f; markers = markers)
    return uₕ
end

######################
#                    #
# Averaging operator #
#                    #
######################

"""
	avgₕ(Wₕ::AbstractSpaceType, f; quad_points = AVG_QUAD_POINTS, markers = ())

Returns a [VectorElement](@ref) with the average of function `f` with respect to the [cell_measure](@ref) of `mesh(Wₕ)` around each grid point.

Each cell average is a tensor-product Gauss-Legendre rule with `quad_points`
points per direction, exact for polynomials of degree `2 * quad_points - 1`.

# The shape of `f`

As for [`Rₕ`](@ref), `f` receives the coordinates of a point: a scalar on a 1D
mesh and an `NTuple{D}` on a `D`-dimensional one, never an `SVector`. For an
`N`-component space either a tuple of functions or a single function returning
all components works, and both give the same result.

# Keywords

  - `quad_points`: points per direction, per cell. Has no counterpart in
    [`Rₕ`](@ref), which involves no quadrature.
  - `markers`: restrict evaluation to the named marked regions, as in
    [`Rₕ`](@ref), leaving every other entry zero.
"""
Base.@constprop :aggressive function avgₕ(
        Wₕ::AbstractSpaceType, f; quad_points::Int = AVG_QUAD_POINTS,
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    uₕ = element(Wₕ)
    avgₕ!(uₕ, f; quad_points = quad_points, markers = markers)
    return uₕ
end

"""
	avgₕ!(uₕ::VectorElement, f; quad_points = AVG_QUAD_POINTS, markers = ())

In-place version of the averaging operator [`avgₕ`](@ref). Returns `nothing`.

Allocates only the task overhead of the parallel loop, independently of the
number of grid points, and `f` is called directly rather than wrapped so that it
specialises and inlines into the quadrature loop. This is the form to use inside
a time-stepping loop.

`f` and the keywords are as described for [`avgₕ`](@ref).

See also: [`avgₕ`](@ref), [`Rₕ!`](@ref)
"""
Base.@constprop :aggressive function avgₕ!(
        uₕ::VectorElement, f; quad_points::Int = AVG_QUAD_POINTS,
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    quad_points >= 1 || throw(ArgumentError("quad_points must be >= 1, got $quad_points"))
    Ωₕ = mesh(space(uₕ))

    if N > 0
        masks = ntuple(i -> index_in_marker(Ωₕ, markers[i]), Val(N))
        nodes, wts = _gauss_rule(Val(quad_points), eltype(uₕ))
        x = half_points(Ωₕ)
        _masked_for!(to_matrix(uₕ), masks, _cell_average_kernel(
            f, x, nodes, wts, Val(dim(Ωₕ))))
        return nothing
    end

    # `f` is passed through unwrapped on purpose. Embedding it in a
    # BrambleFunction gives a fixed compiled signature at the cost of an
    # indirect call, which stops `f` inlining into the quadrature loop and
    # measured 2.3x slower on the grids in the test suite. It also matches
    # `avgₕ`, which has always passed the raw function.
    _avgₕ!(uₕ, f, Val(dim(Ωₕ)), Val(quad_points))
    return nothing
end

# A one-component space is a scalar space, so an NC-tuple of functions with
# NC == 1 must still work.
@inline avgₕ!(
    uₕ::VectorElement{<:ScalarGridSpace}, f::Tuple{Any}; quad_points::Int = AVG_QUAD_POINTS,
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N} = avgₕ!(
    uₕ, f[1]; quad_points = quad_points, markers = markers)

# Evaluations per grid point for a tensor-product rule of `NQ` points per
# direction on a `D`-dimensional mesh. The threading threshold in
# `_parallel_for!` counts evaluations, so a kernel this expensive per index
# reaches the crossover at proportionally fewer indices.
@inline _avg_min_work(::Val{D}, ::Val{NQ}) where {D, NQ} = max(
    1, cld(PARALLEL_FOR_MIN, NQ^D))

function _avgₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f, ::Val{1}, nq::Val)
    Ωₕ = mesh(space(uₕ))
    x = half_points(Ωₕ)
    nodes, wts = _gauss_rule(nq, eltype(uₕ))

    _parallel_for!(values(uₕ), indices(Ωₕ), idx -> _cell_average(f, x, idx[1], nodes, wts);
        min_work = _avg_min_work(Val(1), nq))
    return nothing
end

function _avgₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f, ::Val{D}, nq::Val) where {D}
    Ωₕ = mesh(space(uₕ))
    x = half_points(Ωₕ)
    nodes, wts = _gauss_rule(nq, eltype(uₕ))

    _parallel_for!(to_matrix(uₕ), indices(Ωₕ), idx -> _cell_average(f, x, idx, nodes, wts);
        min_work = _avg_min_work(Val(D), nq))
    return nothing
end

function _avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::Tuple, val_dim::Val, nq::Val) where {NC}
    comps = components(uₕ)
    ntuple(i -> _avgₕ!(comps[i], f[i], val_dim, nq), Val(NC))
    return nothing
end

# A single function returning all components: average it in one pass over the
# grid rather than once per component.
function _avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f, ::Val{D}, nq::Val) where {
        NC, D}
    Ωₕ = mesh(space(uₕ))
    x = half_points(Ωₕ)
    nodes, wts = _gauss_rule(nq, eltype(uₕ))
    mats = ntuple(i -> to_matrix(components(uₕ)[i]), Val(NC))

    _scatter_for!(mats, indices(Ωₕ), idx -> _cell_average(f, x, idx, nodes, wts, Val(NC));
        min_work = _avg_min_work(Val(D), nq))
    return nothing
end

#=
Cell averages are computed with a fixed tensor-product Gauss-Legendre rule per
cell. Every cell integral is independent, low dimensional and over a smooth
integrand, so a small fixed rule is both cheaper and allocation free.

Writing the cell integral on the reference cube,

	1/|C| ∫_C f = ∫_{[0,1]^D} f(a + t ⊙ (b - a)) dt,

because |C| = ∏ₖ (bₖ - aₖ) exactly: the cell around a grid point spans
consecutive half points, whose spacing is the half spacing that
`cell_measure` returns. The quadrature weights below sum to one, so the
weighted sum *is* the average and no measure division is needed.
=#

"""
	AVG_QUAD_POINTS

Default number of Gauss-Legendre points per direction, per cell, used by
[`avgₕ`](@ref). Six points are exact for polynomials up to degree eleven.

Unlike an adaptive rule, a fixed one does not tighten itself on coarse cells, so
the default is chosen to be accurate on cells far coarser than any practical
grid. Measured on 4 points spanning [-1, 4] with a function varying by a factor
of e^5 across the domain -- deliberately harsher than a real mesh -- the worst
error over 30 random grids was

    points   1D        2D        3D        evaluations per cell (3D)
    3        6.1e-5    1.6e-4    3.1e-4     27
    4        2.8e-7    8.1e-7    3.4e-6     64
    5        1.9e-9    4.1e-9    7.8e-9    125
    6        5.5e-12   1.4e-11   1.6e-11   216

Cost is `quad_points^D` evaluations per cell. On a fine grid three points are
usually ample; lower it with the `quad_points` keyword when the integrand is
cheap to resolve and the cells are small.
"""
const AVG_QUAD_POINTS = 6

"""
	_gauss_rule(::Val{N}, ::Type{T})

Returns `(nodes, weights)` for the `N`-point Gauss-Legendre rule on `[0, 1]` as
`SVector{N,T}`, so the per-cell loop that consumes them does not allocate.

The rule is built by `QuadGK.gauss` in the requested element type, so `Float32`
and `BigFloat` grids get a rule at their own precision rather than a rounded
`Float64` one. Weights sum to one, which makes the weighted sum over a cell the
cell *average* directly.
"""
# When `T` is an isbits float its precision is fixed by the type, so the rule
# depends only on (N, T) and is folded into a compile-time constant: obtaining it
# then costs nothing at all. This covers Float16/32/64 and equally the stack
# allocated extended precision types such as Double64 or Float64x2.
#
# Anything else -- notably BigFloat, whose precision is a run-time setting that
# must not be baked in at compile time -- falls back to building the rule per
# call, at the precision in force at that moment. The same fallback catches an
# isbits type that QuadGK cannot construct a rule for.
@generated function _gauss_rule(::Val{N}, ::Type{T}) where {N, T}
    if isbitstype(T)
        try
            x, w = gauss(T, N, zero(T), one(T))
            nodes = Expr(:tuple, x...)
            wts = Expr(:tuple, w...)
            return :((SVector{$N, $T}($nodes), SVector{$N, $T}($wts)))
        catch
            # fall through to the run-time rule
        end
    end
    return :(_gauss_rule_runtime(Val($N), $T))
end

@inline function _gauss_rule_runtime(::Val{N}, ::Type{T}) where {N, T}
    x, w = gauss(T, N, zero(T), one(T))
    return SVector{N, T}(x), SVector{N, T}(w)
end

# Average of `f` over the 1D cell spanned by `x[i] .. x[i+1]`.
@inline function _cell_average(f, x::AbstractVector, i::Int, nodes::SVector{NQ, T},
        wts::SVector{NQ, T}) where {NQ, T}
    @inbounds a = T(x[i])
    @inbounds d = T(x[i + 1]) - a

    s = zero(T)
    @inbounds for q in 1:NQ
        s += wts[q] * f(a + nodes[q] * d)
    end
    return s
end

# Kernel form of the cell average: closes over the geometry and the rule so it
# can be handed to a generic index loop.
@inline _cell_average_kernel(f, x, nodes, wts, ::Val{1}) = idx -> _cell_average(
    f, x, idx[1], nodes, wts)
@inline _cell_average_kernel(f, x, nodes, wts, ::Val{D}) where {D} = idx -> _cell_average(
    f, x, idx, nodes, wts)

# Average of a vector valued `f` over the cell around `idx`, one value per
# component. `f` is evaluated once per quadrature node instead of once per node
# per component; the accumulator is a tuple and every operation on it broadcasts
# over `NC` isbits values, so nothing allocates.
@inline function _cell_average(
        f, x::NTuple{D}, idx::CartesianIndex{D}, nodes::SVector{NQ, T},
        wts::SVector{NQ, T}, ::Val{NC}) where {D, NQ, T, NC}
    a = ntuple(k -> @inbounds(T(x[k][idx[k]])), Val(D))
    b = ntuple(k -> @inbounds(T(x[k][idx[k] + 1])), Val(D))

    s = ntuple(_ -> zero(T), Val(NC))
    @inbounds for q in CartesianIndices(ntuple(_ -> NQ, Val(D)))
        w = one(T)
        for k in 1:D
            w *= wts[q[k]]
        end
        pt = ntuple(k -> a[k] + nodes[q[k]] * (b[k] - a[k]), Val(D))
        s = s .+ w .* f(pt)
    end
    return s
end

# Average of `f` over the D-dimensional cell around `idx`, whose corners are the
# half points `x[k][idx[k]]` and `x[k][idx[k] + 1]` along each axis.
@inline function _cell_average(f, x::NTuple{D}, idx::CartesianIndex{D},
        nodes::SVector{NQ, T}, wts::SVector{NQ, T}) where {D, NQ, T}
    a = ntuple(k -> @inbounds(T(x[k][idx[k]])), Val(D))
    b = ntuple(k -> @inbounds(T(x[k][idx[k] + 1])), Val(D))

    s = zero(T)
    @inbounds for q in CartesianIndices(ntuple(_ -> NQ, Val(D)))
        w = one(T)
        for k in 1:D
            w *= wts[q[k]]
        end
        pt = ntuple(k -> a[k] + nodes[q[k]] * (b[k] - a[k]), Val(D))
        s += w * f(pt)
    end
    return s
end
