# Getters and setters for VectorElement
"""
	values(uₕ::VectorElement)

Returns the coefficients of the [VectorElement](@ref) `uₕ`.
"""
@inline values(uₕ::VectorElement) = uₕ.data

"""
	to_matrix(uₕ::VectorElement, [::ComponentStyle])

Reshapes the flat coefficient vector of `uₕ` into a multidimensional array that matches the logical layout of the grid points.

  - For a `SingleComponent` space, this returns a D-dimensional array.
  - For a `MultiComponent` space, it returns a tuple of D arrays, one for each component.
"""
@inline to_matrix(uₕ::VectorElement{ST}) where ST = to_matrix(uₕ, ComponentStyle(ST))
@inline to_matrix(uₕ::VectorElement, ::SingleComponent) = Base.ReshapedArray(values(uₕ), npoints(mesh(space(uₕ)), Tuple), ())
@inline to_matrix(uₕ::VectorElement, ::MultiComponent{D}) where D = ntuple(i -> to_matrix(uₕ(i), SingleComponent()), Val(D))

"""
	values!(uₕ::VectorElement, s)

Copies the values of `s` into the coefficients of [VectorElement](@ref) `uₕ`.
"""
@inline values!(uₕ::VectorElement, s) = copyto!(values(uₕ), s)

"""
	space(uₕ::VectorElement)

Returns the grid space associated with [VectorElement](@ref) `uₕ`.
"""
@inline space(uₕ::VectorElement) = uₕ.space
@inline space_type(::Type{<:VectorElement{S}}) where S = S

# Forward array-like methods to the `data` field. This allows a VectorElement
# to behave like a standard Julia vector (e.g., support `size`, `length`, `eltype`).
@forward VectorElement.data (Base.size, Base.length, Base.firstindex, Base.lastindex, Base.iterate, Base.eltype, Base.axes, Base.ndims, Bramble.show)
@forward VectorElement.space (Bramble.mesh,)

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
	return VectorElement{ST,T,VT}(vector(b, ndofs(Wₕ)), Wₕ)
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
	@assert length(v) == ndofs(Wₕ) "Input vector length does not match the number of degrees of freedom in the space."
	elem = element(Wₕ)
	copyto!(elem, v)
	return elem
end

# Enable array-like indexing `uₕ[i]` for VectorElement.
@inline Base.@propagate_inbounds getindex(uₕ::VectorElement, i) = getindex(uₕ.data, i)
@inline Base.@propagate_inbounds setindex!(uₕ::VectorElement, val, i) = (setindex!(uₕ.data, val, i); return)

# Create a new, uninitialized VectorElement with the same space as the input.
@inline Base.similar(uₕ::VectorElement) = element(space(uₕ))#===============##===============#

# Broadcasting

# Enable broadcasting capabilities (e.g., uₕ .= vₕ .+ 1) for VectorElement.
Base.BroadcastStyle(::Type{<:VectorElement}) = Broadcast.ArrayStyle{VectorElement}()

# Define how to create a `similar` container for the broadcast result, preserving the space.
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}}, ::Type{ElType}) where ElType
	# Find a VectorElement within the broadcast expression to extract its space.
	vec_elem = _find_vec_in_broadcast(bc)
	# Create a new element on the same space with a similar data array of the correct element type.
	return element(space(vec_elem), similar(values(vec_elem), ElType))
end

# Version without a specified ElType.
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}})
	vec_elem = _find_vec_in_broadcast(bc)
	return element(space(vec_elem), similar(values(vec_elem)))
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
_find_vec_in_broadcast(args::Tuple) = _find_vec_in_broadcast(_find_vec_in_broadcast(args[1]), Base.tail(args))
_find_vec_in_broadcast(x::VectorElement) = x
_find_vec_in_broadcast(::Any) = nothing # Not a VectorElement
_find_vec_in_broadcast(::Tuple{}) = nothing # End of recursion
_find_vec_in_broadcast(a::VectorElement, rest) = a # Found one
_find_vec_in_broadcast(::Any, rest) = _find_vec_in_broadcast(rest) # Keep searching

#=
function *(uₕ::VectorElement, Vₕ::NTuple{D,VectorElement}) where D
	Zₕ = ntuple(i-> similar(Vₕ[i]), D)
	for i in 1:D
		Zₕ[i].data .= uₕ.data .* Vₕ[i].data
	end
	return Zₕ
end
=#

########################
#                      #
# Restriction operator #
#                      #
########################

# Workhorse function for the restriction operator: evaluates a function `g`
# point-by-point over the grid indices and stores the result in `u`.
function _func2array!(::SingleComponent, u, g, mesh_indices::NTuple)
	u .= zero(eltype(u))
	# Iterate over all specified indices in the mesh.
	@inbounds @simd for idxs in mesh_indices
		_apply!(u, g, idxs)
	end
end

# Optimized version for CartesianIndices, enabling parallel execution.
@inline _func2array!(::SingleComponent, u, g, mesh_indices::CartesianIndices) = (_parallel_for!(u, mesh_indices, g))
# Placeholder for multi-component version.
@inline _func2array!(::MultiComponent{D}, u, f, mesh) where D = nothing

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
struct PointwiseEvaluator{F,M}
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
	Rₕ!(uₕ::VectorElement, f; markers=())

In-place version of the restriction operator [Rₕ](@ref).

Evaluates function `f` at grid points and stores the result directly in `uₕ`,
avoiding allocation of a new vector.

# Arguments
- `uₕ::VectorElement`: Pre-allocated vector element to store the result
- `f`: Function to evaluate at grid points (signature: `f(x::SVector) -> Number`)
- `markers::NTuple{N,Symbol}`: Optional tuple of marker symbols to restrict evaluation to specific regions

# Example
```julia
Wₕ = gridspace(Ωₕ)
uₕ = element(Wₕ)

# Evaluate f at all grid points, storing result in uₕ
f(x) = sin(x[1]) * cos(x[2])
Rₕ!(uₕ, f)

# Equivalent to: uₕ = Rₕ(Wₕ, f)
```

!!! warning "Marker-based restriction"
    The `markers` keyword argument is experimental and may not work correctly in all cases.
    For production code, use full grid restriction (default) or verify behavior with tests.

See also: [`Rₕ`](@ref), [`element`](@ref)
"""
@inline function Rₕ!(uₕ::VectorElement{ST}, f; markers::NTuple{N,Symbol} = NTuple{0,Symbol}()) where {N,ST}
	if N > 0
		@debug "Using marker-based restriction" markers
	end

	@unpack data, space = uₕ
	Ωₕ = mesh(space)

	CStyle = ComponentStyle(ST)
	u = to_matrix(uₕ, CStyle)

	g = PointwiseEvaluator(f, Ωₕ)

	if N == 0
		_func2array!(CStyle, u, g, indices(Ωₕ))
		return
	end

	mesh_indices = ntuple(i->marker(Ωₕ, markers[i]), Val(N))
	_func2array!(CStyle, u, g, mesh_indices)
end

#@inline fuse_markers(_::Nothing, _::Nothing) = ()
#@inline fuse_markers(marker::Symbol, _::Nothing) = (marker,)
#@inline fuse_markers(_::Nothing, markers::NTuple{N,Symbol}) where N = markers
#@inline fuse_markers(marker::Symbol, markers::NTuple{N,Symbol}) where N = (marker, markers...)

"""
	Rₕ(Wₕ::AbstractSpaceType, f)

Standard nodal restriction operator. It returns a [VectorElement](@ref) with the result of evaluating the function `f` at the points of `mesh(Wₕ)`. It is defined as follows

  - 1D case

```math
\\textrm{R}ₕ f(x_i) = f(x_i), \\, i = 1,\\dots,N
```

  - 2D case

```math
\\textrm{R}ₕ f(x_i, y_j)= f(x_i, y_j), \\, i = 1,\\dots,N_x,  j = 1,\\dots,N_y
```

  - 3D case

```math
\\textrm{R}ₕ f(x_i, y_j, z_l)= f(x_i, y_j, z_l), \\, i = 1,\\dots,N_x,  j = 1,\\dots,N_y, l = 1,\\dots,N_z
```
"""
function Rₕ(Wₕ::AbstractSpaceType, f; markers::NTuple{N,Symbol} = NTuple{0,Symbol}()) where N
	uₕ = element(Wₕ)
	Rₕ!(uₕ, f, markers = markers)
	return uₕ
end

######################
#                    #
# Averaging operator #
#                    #
######################

"""
	avgₕ(Wₕ::AbstractSpaceType, f)

Returns a [VectorElement](@ref) with the average of function `f` with respect to the [cell_measure](@ref) of `mesh(Wₕ)` around each grid point. It is defined as follows

  - 1D case

```math
\\textrm{avg}ₕ f(x_i) = \\frac{1}{|\\square_i|} \\int_{\\square_i} f(x) dx, \\, i = 1,\\dots,N
```

  - 2D case

```math
\\textrm{avg}ₕ f(x_i, y_j) = \\frac{1}{|\\square_{i,j}|} \\iint_{\\square_{i,j}} f(x,y) dA, \\, i = 1,\\dots,N_x,  j = 1,\\dots,N_y
```

  - 3D case

```math
\\textrm{avg}ₕ f(x_i, y_j, z_l) = \\frac{1}{|\\square_{i,j,l}|} \\iiint_{\\square_{i,j,l}} f(x,y,z) dV, \\, i = 1,\\dots,N_x,  j = 1,\\dots,N_y, l = 1,\\dots,N_z
```

Please check the implementations of functions [cell_measure](@ref cell_measure(Ωₕ::Mesh1D, i)) (for the `1`-dimensional case) and [cell_measure](@ref cell_measure(Ωₕ::MeshnD, i)) (for the `n`-dimensional cases).
"""
@inline function avgₕ(Wₕ::AbstractSpaceType, f)
	uₕ = element(Wₕ)
	ST = typeof(space(uₕ))
	_avgₕ!(ComponentStyle(ST), uₕ, f, Val(dim(mesh(Wₕ))))
	return uₕ
end

"""
	avgₕ!(uₕ::VectorElement, f)

In-place version of averaging operator [avgₕ](@ref).
"""
@inline function avgₕ!(uₕ::VectorElement{ST}, f) where ST
	Ωₕ = mesh(space(uₕ))
	_f = embed_function(set(Ωₕ), f)

	_avgₕ!(ComponentStyle(ST), uₕ, _f, Val(dim(Ωₕ)))
	return
end

function _avgₕ!(::SingleComponent, uₕ::VectorElement{ST}, f, ::Val{1}) where ST
	Ωₕ = mesh(space(uₕ))

	x = half_points(Ωₕ)
	h = half_spacings(Ωₕ)

	idxs = eachindex(uₕ)
	param = (f, x, h, idxs)

	__quad!(ComponentStyle(ST), uₕ, (0, 1), param)
	return
end

function _avgₕ!(::MultiComponent{N}, uₕ::VectorElement{ST}, f, ::Val{1}) where {N,ST}
	return
end

function _avgₕ!(::SingleComponent, uₕ::VectorElement{ST}, f, ::Val{D}) where {D,ST}
	Ωₕ = mesh(space(uₕ))

	x = half_points(Ωₕ)
	meas = Base.Fix1(cell_measure, Ωₕ)
	param = (f, x, meas, indices(Ωₕ))

	_zeros = @SVector zeros(D)
	_ones = @SVector ones(D)

	__quadnd!(ComponentStyle(ST), uₕ, (_zeros, _ones), param)
	return
end

function _avgₕ!(::MultiComponent{N}, uₕ::VectorElement{ST}, f, ::Val{D}) where {N,D,ST}
	return
end

"""
	__integrand1d(y, t, p)

Implements the integrand function needed in the calculation of the averaging operator `avgₕ`. In this function, `y` denotes the return values, `t` denotes the integration variable and `p` denotes the parameters (integrand function `f`, points `x`, spacing `h` and indices `idxs`).

For efficiency, each integral in `avgₕ` is rewritten as an integral over `[0,1]` following

```math
\\int_{a}^{b} f(x) dx = (b-a) \\int_{0}^{1} f(a + t (b-a)) dt
```
"""
@inline @muladd function __integrand1d(y, t, p)
	f, x, h, idxs = p

	@inbounds @simd for idx in idxs
		i = idx[1]
		xip1 = x[i + 1]
		xi = x[i]
		diff = xip1 - xi
		y[i] = f(xi + t * diff) * diff / h[i]
	end
	return
end

function __quad!(::SingleComponent, uₕ::VectorElement, domain::NTuple{2}, p::ParamType) where ParamType
	domain_to_float = float.(domain)
	prototype = values(uₕ)
	sol = __quad_problem(prototype, domain_to_float, p)

	copyto!(uₕ, sol)
	return
end

function __quad!(::MultiComponent{N}, uₕ::VectorElement, domain::NTuple{2}, p::ParamType) where {N,ParamType}
	return
end

@inline @muladd function _point_tuple_and_volume(t::SVector{D}, x, idx::CartesianIndex{D}) where D
	lb_tuple = ntuple(i -> x[i][idx[i]], Val(D))
	ub_tuple = ntuple(i -> x[i][idx[i] + 1], Val(D))

	lb = SVector(lb_tuple)
	ub = SVector(ub_tuple)

	point_svector = lb .+ t .* (ub .- lb)

	volume_element = one(eltype(lb))
	for i in 1:D
		volume_element *= (ub[i] - lb[i])
	end

	return Tuple(point_svector), volume_element
end

"""
	__integrandnd(y, t, p)

Implements the integrand function needed in the calculation of `avgₕ`. In this function, `y` denotes the return values, `t` denotes the integration variable and `p` denotes the parameters (integrand function `f`, points `x`, measures `meas` and indices `idxs`).

For efficiency, each integral is calculated on ``[0,1]^D``, where ``D`` is the dimension of the integration domain. This is done through a similar change of variable as in [__integrand1d(y, t, p)](@ref).
"""
function __integrandnd(y, t, p)
	f, x, meas, idxs = p

	@inbounds @simd for idx in idxs
		point_tuple, volume_element = _point_tuple_and_volume(t, x, idx)
		y[idx] = f(point_tuple) * volume_element / meas(idx)
	end

	return
end

@inline int_function(::Val{D}) where D = __integrandnd
@inline int_function(::Val{1}) = __integrand1d

function __quad_problem(prototype, domain, p)
	D = length(size(prototype))
	T = eltype(domain[1])

	func = IntegralFunction(int_function(Val(D)), prototype)
	prob = IntegralProblem{true}(func, domain, p)
	sol = solve(prob, CubatureJLh()).u

	return sol::Array{T,D}
end

function __quadnd!(::SingleComponent, uₕ::VectorElement, domain::NTuple{S,SV}, p::ParamType) where {S,SV,ParamType}
	v = to_matrix(uₕ, SingleComponent())

	sol = __quad_problem(v, domain, p)
	copyto!(v, sol)
	return
end

function __quadnd!(::MultiComponent{N}, uₕ::VectorElement, domain::NTuple{D}, p::ParamType) where {N,D,ParamType}
	return
end