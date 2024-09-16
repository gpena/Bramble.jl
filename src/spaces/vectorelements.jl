"""
	VectorElement{S, T} <: AbstractVector{T}

Vector element of space `S` with coefficients of type `T`.
"""
struct VectorElement{S,T} <: AbstractVector{T}
	space::S
	values::Vector{T}
end

"""
	Element(S::SpaceType, v::AbstractVector)

Create a `VectorElement` for space `S` with coefficients in vector `v`.

Input:
S::SpaceType: a space
v::AbstractVector: a vector of coefficients

Output:
a `VectorElement` for space `S` with coefficients in vector `v`
"""
@inline Element(S::SpaceType, v::AbstractVector) = (@assert length(v) == ndofs(S); VectorElement(S, v))

"""
	Element(S::SpaceType)

Create a `VectorElement` for space `S`.

Input:
S::SpaceType: a space

Output:
a `VectorElement` for space `S`
"""
@inline Element(S::SpaceType) = Element(S, Vector{eltype(S)}(undef, ndofs(S)))

"""
	Element(S::SpaceType, v::Real)

Create a `VectorElement` for space `S` with coefficients set to `v`.

Input:
S::SpaceType: a space
v::Real: a value

Output:
a `VectorElement` for space `S` with coefficients set to `v`
"""
function Element(S::SpaceType, v::Real)
	v = Vector{eltype(S)}(convert(eltype(S), v) * Ones(ndofs(S)))
	return Element(S, v)
end

"""
	ndims(::Type{<:VectorElement})

Return the number of dimensions of the space the `VectorElement` belongs to.

Input:
T::Type{<:VectorElement}: a type of `VectorElement`

Output:
an integer representing the number of dimensions of the space the `VectorElement` belongs to
"""
@inline ndims(::Type{<:VectorElement}) = 1

Base.BroadcastStyle(::Type{<:VectorElement}) = Broadcast.DefaultArrayStyle{1}()
#=function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}}, ::Type{ElType}) where ElType
	# Scan the inputs for the VectorElement:
	println("similar")
	vec = find_aac(bc)
	# Use the char field of A to create the output
	Element(vec.space, Vector{ElType}(undef, ndofs(vec)))
end

"`A = find_aac(As)` returns the first ArrayAndChar among the arguments."
find_aac(bc::Base.Broadcast.Broadcasted) = find_aac(bc.args)
find_aac(args::Tuple) = find_aac(find_aac(args[1]), Base.tail(args))
find_aac(x) = x
find_aac(::Tuple{}) = nothing
find_aac(a::VectorElement, rest) = a
find_aac(::Any, rest) = find_aac(rest)
=#

#=@inline function copyto!(dest::VectorElement, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}})
	println("copyto!")
	copyto!(dest.values, convert(Broadcast.Broadcasted{Nothing}, bc))
end
=#

@inline function copyto!(dest::VectorElement, bc::Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{0}})
	# Typically, we must independently execute bc for every storage location in `dest`, but:
	println("here-1")
	# IF we're in the common no-op identity case with no nested args (like `dest .= val`),
	if bc.f === identity && bc.args isa Tuple{Any} && isflat(bc)
		# THEN we can just extract the argument and `fill!` the destination with it
		return fill!(dest.values, bc.args[1][])
	else
		# Otherwise, fall back to the default implementation like above
		return copyto!(dest.values, convert(Broadcasted{Nothing}, bc))
	end
end

@inline function copyto!(dest::VectorElement, bc::Broadcast.Broadcasted{Nothing})
	println("here0")
	axes(dest.values) == axes(bc) || throwdm(axes(dest.values), axes(bc))
	# Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
	if bc.f === identity && bc.args isa Tuple{VectorElement} # only a single input argument to broadcast!
		A = bc.args[1]
		if axes(dest.values) == axes(A)
			return copyto!(dest.values, A)
		end
	end
	bc′ = Broadcast.preprocess(dest.values, bc)
	# Performance may vary depending on whether `@inbounds` is placed outside the
	# for loop or not. (cf. https://github.com/JuliaLang/julia/issues/38086)
	@inbounds @simd for I in eachindex(bc′)
		dest.values[I] = bc′[I]
	end
	return dest
end

@inline function materialize!(dest::VectorElement, x)
	println("here1")
	return materialize!(dest.values, instantiate(Broadcasted(identity, (x,), axes(dest.values))))
end

@inline function materialize!(dest::VectorElement, bc::Broadcast.Broadcasted{<:Any})
	aux = Broadcast.instantiate(Broadcast.Broadcasted(bc.style, bc.f, bc.args, axes(dest.values)))

	for i in eachindex(aux)
		dest.values[i] = aux[i]
	end

	return dest
end

@inline function materialize!(::Broadcast.BroadcastStyle, dest::VectorElement, bc::Broadcast.Broadcasted{<:Any})
	println("here3")

	return copyto!(dest.values, instantiate(Broadcasted(bc.style, bc.f, bc.args, axes(dest.values))))
end

"""
	getindex(A::VectorElement, i)

Get the i-th element of a `VectorElement`.

Input:
A::VectorElement: a vector element
i: an integer

Output:
the i-th element of `A`
"""
@inline getindex(A::VectorElement, i) = getindex(A.values, i)

"""
	setindex!(A::VectorElement, val, i)

Set the i-th element of a `VectorElement` to `val`.

Input:
A::VectorElement: a vector element
val: a value
i: an integer

Output:
nothing
"""
@inline setindex!(A::VectorElement, val, i) = setindex!(A.values, val, i)

"""
	firstindex(u::VectorElement)

Get the first index of a `VectorElement`.

Input:
u::VectorElement: a vector element

Output:
an integer representing the first index of `u`
"""
@inline firstindex(u::VectorElement) = firstindex(u.values)

"""
	lastindex(u::VectorElement)

Get the last index of a `VectorElement`.

Input:
u::VectorElement: a vector element

Output:
an integer representing the last index of `u`
"""
@inline lastindex(u::VectorElement) = lastindex(u.values)

"""
	iterate(u::VectorElement)

Create an iterator for a `VectorElement`.

Input:
u::VectorElement: a vector element

Output:
an iterator over `u`
"""
@inline iterate(u::VectorElement) = iterate(u.values)

"""
	iterate(u::VectorElement, state)

Create an iterator for a `VectorElement` with a given state.

Input:
u::VectorElement: a vector element
state: an iterator state

Output:
an iterator over `u` with the given state
"""
@inline iterate(u::VectorElement, state) = iterate(u.values, state)

"""
	show(io::IO, u::VectorElement)

Print a `VectorElement` in a human-readable format.

Input:
io::IO: an IO stream
u::VectorElement: a vector element
"""
show(io::IO, u::VectorElement) = show(io, "text/plain", u.values)

"""
	eltype(::Type{<:VectorElement{S,T}})

Return the element type of a `VectorElement`.

Input:
T::Type{<:VectorElement{S,T}}: a type of `VectorElement`

Output:
the element type of `T`
"""
@inline eltype(::Type{<:VectorElement{S,T}}) where {S,T} = T

"""
	dim(u::VectorElement)

Return the number of dimensions of the space of a `VectorElement`.

Input:
u::VectorElement: a vector element

Output:
an integer representing the number of dimensions of the space of `u`
"""
@inline dim(u::VectorElement) = dim(typeof(u))
@inline dim(::Type{<:VectorElement{S}}) where {S} = dim(S)

"""
	length(u::VectorElement)

Return the length of a `VectorElement`.

Input:
u::VectorElement: a vector element

Output:
an integer representing the number of elements in `u`
"""
@inline length(u::VectorElement) = length(u.values)

"""
	ndofs(u::VectorElement)

Return the number of degrees of freedom of a `VectorElement`.

Input:
u::VectorElement: a vector element

Output:
an integer representing the number of degrees of freedom of `u`
"""
@inline ndofs(u::VectorElement) = ndofs(space(u))

"""
	npoints(u::VectorElement)

Return the number of points in the mesh of a `VectorElement`.

Input:
u::VectorElement: a vector element

Output:
an integer representing the number of points in the mesh of `u`
"""
@inline npoints(u::VectorElement) = npoints(mesh(space(u)))

"""
	space(u::VectorElement)

Return the space of a `VectorElement`.

Input:
u::VectorElement: a vector element

Output:
the space of `u`
"""
@inline space(u::VectorElement) = u.space

"""
	mesh(u::VectorElement)

Return the mesh of a `VectorElement`.

Input:
u::VectorElement: a vector element

Output:
the mesh of `u`
"""
@inline mesh(u::VectorElement) = mesh(space(u))

"""
	similar(uh::VectorElement)

Create a new `VectorElement` with the same space as `uh`.

Input:
uh::VectorElement: a vector element

Output:
a new `VectorElement` with the same space as `uh`
"""
@inline similar(uh::VectorElement) = VectorElement(space(uh), similar(uh.values))

"""
	size(u::VectorElement)

Return the size of a `VectorElement`.

Input:
u::VectorElement: a vector element

Output:
a tuple representing the size of `u`
"""
@inline size(u::VectorElement) = size(u.values)

"""
	copyto!(u::VectorElement, v::VectorElement)

Copy the coefficients of vector `v` into the coefficients of `u`.

Input:
u::VectorElement: a vector element
v::VectorElement: a vector element

Output:
nothing
"""
@inline copyto!(u::VectorElement, v::VectorElement) = (@assert length(u.values) == length(v.values); @.. u.values = v.values)

"""
	copyto!(u::VectorElement, v::AbstractVector)

Copy the coefficients of vector `v` into the coefficients of `u`.

Input:
u::VectorElement: a vector element
v::AbstractVector: a vector of coefficients

Output:
nothing
"""
@inline copyto!(u::VectorElement, v::AbstractVector) = (@assert length(u.values) == length(v); @.. u.values = v)

"""
	copyto!(u::VectorElement, v::Real)

Copy value `v` into the coefficients of `u`.

Input:
u::VectorElement: a vector element
v::Real: a value

Output:
nothing
"""
@inline copyto!(u::VectorElement, v::Real) = (@.. u.values = convert(eltype(u), v))

"""
	isequal(u::VectorElement, v::AbstractVector)

Tests if the coefficients of vector `u` are equal to the coefficients of vector `v`.

Input:
u::VectorElement: a vector element
v::AbstractVector: a vector of coefficients

Output:
true if the coefficients of vector `u` are equal to the coefficients of vector `v`, false otherwise
"""
@inline isequal(u::VectorElement, v::AbstractVector) = (isequal(u.values, v))

"""
	isequal(u::VectorElement, v::AbstractVector)

Tests if the coefficients of vector `u` are equal to `v`.

Input:
u::VectorElement: a vector element
v::Real: a value

Output:
true if the coefficients of vector `u` are equal to `v`, false otherwise
"""
@inline isequal(u::VectorElement, v::Real) = (isequal(u.values, v))

"""
	map(f, u::VectorElement)

Return a new `VectorElement` with coefficients obtained by applying `f` to the coefficients of `u`.

Input:
f::Function: a function
u::VectorElement: a vector element

Output:
a new `VectorElement` with coefficients obtained by applying `f` to the coefficients of `u`
"""
map(f, u::VectorElement) = Element(u.space, map(f, u.values))

for op in (:-, :*, :/, :^, :\, :+)
	@eval begin
		"""
			(op)(α::AbstractFloat, u::VectorElement)

		Return a new `VectorElement` with coefficients obtained by applying `(op)` to the coefficients of `u` and `α`.

		Input:
		α::AbstractFloat: a scalar
		u::VectorElement: a vector element

		Output:
		a new `VectorElement` with coefficients obtained by applying `(op)` to the coefficients of `u` and `α`
		"""
		function ($op)(α::AbstractFloat, u::VectorElement)
			r = similar(u)
			map!(Base.Fix1($op, α), r.values, u.values)
			return r
		end

		function ($op)(u::VectorElement, α::AbstractFloat)
			r = similar(u)
			map!(Base.Fix2($op, α), r.values, u.values)
			return r
		end

		function ($op)(u::VectorElement, v::VectorElement)
			r = similar(u)
			map!(($op), r.values, u.values, v.values)
			return r
		end
	end
end

########################
#                      #
# Restriction operator #
#                      #
########################

@inline function _func2array!(vector, f::F, mesh) where {F<:Function}
	pts = points(mesh)
	idxs = indices(mesh)

	g(idx) = _index2point(pts, idx)
	@.. vector = f(g(idxs))
end
#=
@generated function _func2array2!(A, f::F, mesh::MeshType{D}) where {D,F<:Function}
	return quote
		res = points(mesh)
		Base.Cartesian.@nloops $D i A begin
			#idx = CartesianIndex(Base.Cartesian.@ntuple $D i)
			x = Base.Cartesian.@ntuple $D (i -> res[i][i])
			(Base.Cartesian.@nref $D A i) = f(x)
		end
	end
end
=#
"""
	Rₕ!(u::VectorElement, f::F) where {F<:Function}

Evaluate the function `f` at the points of `mesh(u)` and store the result in `u`.

**Inputs:**

  - `u`: a vector element
  - `f`: a function of the form `f(x)`

**Output:** `u` with updated coefficients
"""
function Rₕ!(u::VectorElement, f::F) where {F<:Function}
	v = Base.ReshapedArray(u.values, npoints(mesh(u)), ())
	_func2array!(v, f, mesh(u))
	return u
end

"""
	Rₕ(S::SpaceType, f::F) where {F<:Function}

Evaluate the function `f` at the points of `mesh(S)` and return a new `VectorElement` with the result.

**Inputs:**

  - `S`: a space
  - `f`: a function of the form `f(x)`

**Output:** a new `VectorElement` with the result
"""
function Rₕ(S::SpaceType, f::F) where {F<:Function}
	u = Element(S)
	Rₕ!(u, f)
	return u
end

######################
#                    #
# Averaging operator #
#                    #
######################

"""
	avgₕ(S::SpaceType, f::F) where {F<:Function}

Compute the average of `f` with respect to the measure of `mesh(S)` and return a new `VectorElement` with the result.

**Inputs:**

  - `S`: a space
  - `f`: a function of the form `f(x)`

**Output:** a new `VectorElement` with the result
"""
function avgₕ(S::SpaceType, f::F) where {F<:Function}
	u = Element(S)
	avgₕ!(u, f, Val(dim(S)))
	return u
end

"""
	avgₕ!(u::VectorElement, f::F) where {F<:Function}

Compute the average of `f` with respect to the measure of `mesh(u)` and store the result in `u`.

**Inputs:**

  - `u`: a vector element
  - `f`: a function of the form `f(x)`

**Output:** `u` with updated coefficients
"""
@inline function avgₕ!(u::VectorElement, f::F) where {F<:Function}
	avgₕ!(u, f, Val(dim(u)))
	return u
end

"""
	__integrand1d(y, t, p)

Integrand for 1D quadrature.
"""
function __integrand1d(y, t, p)
	f, x, h, idxs = p

	@inbounds for i in idxs
		diff = (x(i + 1) - x(i))
		y[i] = f(x(i) + t * diff) * diff / h(i)
	end
end

"""
	__quad!(u::VectorElement, domain::NTuple{2,T}, p::ParamType) where {T, ParamType}

Evaluate the integral of `f` over the domain `domain` and store the result in `u`.
"""
function __quad!(u::VectorElement, domain::NTuple{2,T}, p::ParamType) where {T,ParamType}
	prototype = zeros(ndofs(space(u)))

	func = IntegralFunction(__integrand1d, prototype)
	prob = IntegralProblem(func, domain, p)
	sol = solve(prob, CubatureJLh())

	copyto!(u.values, sol.u)
	return nothing
end

"""
	avgₕ!(u::VectorElement, f::F, ::Val{1}) where {F<:Function}

Compute the average of `f` with respect to the measure of `mesh(u)` in 1D and store the result in `u`.

**Input:**

  - `u`: a vector element
  - `f`: a function of the form `f(x)`

**Output:** `u` with updated coefficients
"""
function avgₕ!(u::VectorElement, f::F, ::Val{1}) where {F<:Function}
	M = mesh(u)

	x = Base.Fix1(xmean, M)
	h = Base.Fix1(hmean, M)

	param = (f, x, h, 1:ndofs(M))
	__quad!(u, (0, 1), param)
end

"""
	avgₕ!(u::VectorElement, f::F, ::Val{D}) where {D, F<:Function}

Compute the average of `f` with respect to the measure of `mesh(u)` in D dimensions and store the result in `u`.

**Input:**

  - `u`: a vector element
  - `f`: a function of the form `f(x)`

**Output:** `u` with updated coefficients
"""
function avgₕ!(u::VectorElement, f::F, ::Val{D}) where {D,F<:Function}
	M = mesh(u)

	x = Base.Fix1(xmean, M)
	meas = Base.Fix1(meas_cell, M)

	param = (f, x, meas, indices(M))
	__quadnd!(u, (zeros(D), ones(D)), param)
end

"""
	__shift_index1(idx::CartesianIndex{D}) where D

Shift the index `idx` by one in each dimension.
"""
@inline @generated __shift_index1(idx::CartesianIndex{D}) where {D} = :(Base.Cartesian.@ntuple $D i->idx[i] + 1)

"""
	__idx2point(t, x, idx::CartesianIndex{D}) where D

Compute the point `x` at index `idx` in D dimensions and its difference from the point at the next index.
"""
function __idx2point(t, x, idx::CartesianIndex{D}) where {D}
	lb = x(Tuple(idx))
	ub = x(__shift_index1(idx))

	diff = ub .- lb
	point = ntuple(i -> lb[i] + t[i] * diff[i], D)

	return point, diff
end

"""
	__integrandnd(y, t, p)

Integrand for D dimensional quadrature.
"""
function __integrandnd(y, t, p)
	f, x, meas, idxs = p

	Threads.@threads for idx in idxs
		point, diff = __idx2point(t, x, idx)
		y[idx] = f(point) * prod(diff) / meas(idx)
	end
end

"""
	__quadnd!(u::VectorElement, domain::NTuple{D,T}, p::ParamType) where {D, T, ParamType}

Evaluate the integral of `f` over the domain `domain` in D dimensions and store the result in `u.values`.
"""
function __quadnd!(u::VectorElement, domain::NTuple{D,T}, p::ParamType) where {D,T,ParamType}
	npts = npoints(mesh(u))
	v = Base.ReshapedArray(u.values, npts, ())
	prototype = v

	func = IntegralFunction(__integrandnd, prototype)
	prob = IntegralProblem(func, domain, p)
	sol = solve(prob, CubatureJLh())

	copyto!(v, sol.u)
	return nothing
end