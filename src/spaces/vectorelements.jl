# Getters and setters for VectorElement
"""
	values(uₕ::VectorElement)

Returns the coefficients of the [VectorElement](@ref) `uₕ`.
"""
@inline values(uₕ::VectorElement) = uₕ.data

"""
	to_matrix(uₕ::VectorElement)

Reshapes the coefficients of the [VectorElement](@ref) `uₕ` to a matrix.
"""
@inline to_matrix(uₕ::VectorElement) = Base.ReshapedArray(values(uₕ), npoints(mesh(space(uₕ)), Tuple), ())

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

@forward VectorElement.data (Base.size, Base.length, Base.firstindex, Base.lastindex, Base.iterate, Base.eltype)
@forward VectorElement.space (Bramble.mesh,)

# Constructor for VectorElement
"""
	element(Wₕ::AbstractSpaceType)

Returns a [VectorElement](@ref) for grid space `Wₕ` with uninitialized components.
"""
@inline function element(Wₕ::AbstractSpaceType)
	b = backend(Wₕ)

	ST = typeof(Wₕ)
	VT = vector_type(b)
	T = eltype(b)
	VectorElement{ST,T,VT}(vector(b, ndofs(Wₕ)), Wₕ)
end

"""
	element(Wₕ::AbstractSpaceType, α::Number)

Returns a [VectorElement](@ref) for a grid space `Wₕ` with all components equal to `α`.
"""
function element(Wₕ::AbstractSpaceType, α::Number)
	uₕ = element(Wₕ)
	fill!(values(uₕ), α) # maybe fill!(uₕ, α) works?
	return uₕ
end

"""
	element(Wₕ::AbstractSpaceType, v::AbstractVector)

Returns a [VectorElement](@ref) for a grid space `Wₕ` with the same coefficients of `v`.
"""
@inline function element(Wₕ::AbstractSpaceType, v::AbstractVector)
	@assert length(v) == ndofs(Wₕ)
	elem = element(Wₕ)
	copyto!(values(elem), v)

	return elem
end

@inline ndims(::Type{<:VectorElement}) = 1

Base.BroadcastStyle(::Type{<:VectorElement}) = Broadcast.ArrayStyle{VectorElement}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}}, ::Type{ElType}) where ElType
	# Scan the inputs for the VectorElement:
	A = _find_vec_in_broadcast(bc)
	@unpack data, space = A
	return element(space, similar(data, ElType))
end

_find_vec_in_broadcast(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}}) = _find_vec_in_broadcast(bc.args)
_find_vec_in_broadcast(args::Tuple) = _find_vec_in_broadcast(_find_vec_in_broadcast(args[1]), Base.tail(args))
_find_vec_in_broadcast(x) = x
_find_vec_in_broadcast(::Tuple{}) = nothing
_find_vec_in_broadcast(a::VectorElement, rest) = a
_find_vec_in_broadcast(::Any, rest) = _find_vec_in_broadcast(rest)

@inline function copyto!(dest::VectorElement, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}})
	copyto!(values(dest), convert(Broadcast.Broadcasted{Nothing}, bc))

	return dest
end

@inline function materialize!(dest::VectorElement, bc::Broadcast.Broadcasted{<:Any})
	aux = Broadcast.instantiate(Broadcast.Broadcasted(bc.style, bc.f, bc.args, axes(dest)))
	@unpack data, space = dest
	@simd for i in eachindex(aux)
		data[i] = aux[i]
	end

	return dest
end

@inline Base.@propagate_inbounds function getindex(uₕ::VectorElement, i)
	@unpack data, space = uₕ
	@boundscheck checkbounds(data, i)
	return getindex(data, i)
end

@inline Base.@propagate_inbounds function setindex!(uₕ::VectorElement, val, i)
	@unpack data, space = uₕ
	@boundscheck checkbounds(data, i)
	setindex!(data, val, i)
end

@inline iterate(uₕ::VectorElement, state) = iterate(values(uₕ), state)

"""
	similar(uh::VectorElement)

Returns a new [VectorElement](@ref) belonging to the same space as `uh`, with uninitialized components.
"""
@inline similar(uₕ::VectorElement) = element(space(uₕ))

"""
	copyto!(uₕ::VectorElement, vₕ::VectorElement)
	copyto!(uₕ::VectorElement, v::AbstractVector)
	copyto!(uₕ::VectorElement, α::Number)

Copies the coefficients of [VectorElement](@ref) `vₕ` into [VectorElement](@ref) `uₕ`. The second argument can also be a regular `AbstractVector` or a `Number``.
"""
@inline copyto!(uₕ::VectorElement, vₕ::VectorElement) = (@assert length(values(uₕ)) == length(values(vₕ));
														 copyto!(values(uₕ), values(vₕ)))
@inline copyto!(uₕ::VectorElement, v::AbstractVector) = (@assert length(values(uₕ)) == length(v);
														 copyto!(values(uₕ), v))
@inline copyto!(uₕ::VectorElement, α::Number) = (s = eltype(uₕ)(α);
												 copyto!(values(uₕ), s))

for op in (:-, :*, :/, :+, :^)
	same_text = "\n\nReturns a new [VectorElement](@ref) with coefficients given by the elementwise evaluation of"
	docstr1 = "	" * string(op) * "(α::Number, uₕ::VectorElement)" * same_text * "`α`" * string(op) * "`uₕ`."
	docstr2 = "	" * string(op) * "(uₕ::VectorElement, α::Number)" * same_text * "`uₕ`" * string(op) * "`α`."
	docstr3 = "	" * string(op) * "(uₕ::VectorElement, vₕ::VectorElement)" * same_text * " `uₕ`" * string(op) * "`vₕ`."

	@eval begin
		@doc $docstr1 @inline function (Base.$op)(α::Number, uₕ::VectorElement)
			rₕ = similar(uₕ)
			tmap!(Base.Fix1(Base.$op, α), values(rₕ), values(uₕ))
			return rₕ
		end

		@doc $docstr2 @inline function (Base.$op)(uₕ::VectorElement, α::Number)
			rₕ = similar(uₕ)

			tmap!(Base.Fix2(Base.$op, α), values(rₕ), values(uₕ))
			return rₕ
		end

		@doc $docstr3 @inline function (Base.$op)(uₕ::VectorElement, vₕ::VectorElement)
			rₕ = similar(uₕ)
			tmap!((Base.$op), values(rₕ), values(uₕ), values(vₕ))
			return rₕ
		end
	end
end
#=
function *(uₕ::VectorElement, Vₕ::NTuple{D,VectorElement}) where D
	Zₕ = ntuple(i-> similar(Vₕ[i]), D)
	for i in 1:D
		Zₕ[i].data .= uₕ.data .* Vₕ[i].data
	end
	return Zₕ
end

########################
#                      #
# Restriction operator #
#                      #
########################

@inline function _func2array!(u, f, mesh)
	@assert length(u) === npoints(mesh) === length(indices(mesh))
	pts = points(mesh)
	idxs = indices(mesh)

	for i in eachindex(idxs)
		u[i] = f(_i2p(pts, idxs[i]))
	end
	return nothing
end

"""
	Rₕ!(uₕ::VectorElement, f)

In-place version of the restriction operator [Rₕ](@ref).
"""
@inline function Rₕ!(uₕ::VectorElement, f)
	u = Base.ReshapedArray(uₕ.data, npoints(mesh(space(uₕ)), Tuple), ())
	_func2array!(u, f, mesh(space(uₕ)))
	return nothing
end

"""
	Rₕ(Wₕ::AbstractSpaceType, f)

Standard nodal restriction operator. It returns a [VectorElement](@ref) with the result of evaluating the function `f` at the points of `mesh(Wₕ)`. It can accept any function (like `x->x[2]+x[1])`) or a [BrambleFunction](@ref). The latter is preferred.

  - 1D case

```math
\\textrm{R}ₕ(x_i) = f(x_i), \\, i = 1,\\dots,N
```

  - 2D case

```math
\\textrm{R}ₕ (x_i, y_j)= f(x_i, y_j), \\, i = 1,\\dots,N_x,  j = 1,\\dots,N_y
```

  - 3D case

```math
\\textrm{R}ₕ (x_i, y_j, z_l)= f(x_i, y_j, z_l), \\, i = 1,\\dots,N_x,  j = 1,\\dots,N_y, l = 1,\\dots,N_z
```
"""
function Rₕ(Wₕ::AbstractSpaceType, f)
	uₕ = element(Wₕ)
	Rₕ!(uₕ, f)
	return uₕ
end

######################
#                    #
# Averaging operator #
#                    #
######################

"""
	avgₕ(Wₕ::AbstractSpaceType, f)

Returns a [VectorElement](@ref) with the average of function `f` with respect to the [cell_measure](@ref) of `mesh(Wₕ)` around each grid point. It can accept any function (like `x->x[2]+x[1])`) or a [BrambleFunction](@ref). The latter is preferred. It is defined as follows

  - 1D case

```math
\\textrm{avg}ₕ(x_i) = \\frac{1}{|\\square_i|} \\int_{\\square_i} f(x) dx, \\, i = 1,\\dots,N
```

  - 2D case

```math
\\textrm{avg}ₕ(x_i, y_j) = \\frac{1}{|\\square_{i,j}|} \\iint_{\\square_{i,j}} f(x,y) dA, \\, i = 1,\\dots,N_x,  j = 1,\\dots,N_y
```

  - 3D case

```math
\\textrm{avg}ₕ(x_i, y_j, z_l) = \\frac{1}{|\\square_{i,j,l}|} \\iiint_{\\square_{i,j,l}} f(x,y,z) dV, \\, i = 1,\\dots,N_x,  j = 1,\\dots,N_y, l = 1,\\dots,N_z
```

Please check the implementations of functions [cell_measure](@ref cell_measure(Ωₕ::Mesh1D, i)) (for the `1`-dimensional case) and [cell_measure](@ref cell_measure(Ωₕ::MeshnD, i)) (for the `n`-dimensional cases).
"""
@inline function avgₕ(Wₕ::AbstractSpaceType, f)
	uₕ = element(Wₕ)
	_avgₕ!(uₕ, f, Val(dim(mesh(Wₕ))))
	return uₕ
end

"""
	avgₕ!(uₕ::VectorElement, f)

In-place version of averaging operator [avgₕ](@ref).
"""
@inline function avgₕ!(uₕ::VectorElement, f)
	_avgₕ!(uₕ, f, Val(dim(mesh(space(uₕ)))))
	return nothing
end

function _avgₕ!(uₕ::VectorElement, f, ::Val{1})
	Ωₕ = mesh(space(uₕ))

	x = Base.Fix1(half_points, Ωₕ)
	h = Base.Fix1(half_spacing, Ωₕ)

	param = (f, x, h, 1:npoints(Ωₕ)) ## indices(omegah)
	__quad!(uₕ, (0, 1), param)
	return nothing
end

function _avgₕ!(uₕ::VectorElement, f::BrambleFunction{1}, ::Val{1})
	Ωₕ = mesh(space(uₕ))

	x = Base.Fix1(half_points, Ωₕ)
	h = Base.Fix1(half_spacing, Ωₕ)

	param = (f, x, h, 1:npoints(Ωₕ))
	__quad!(uₕ, (0, 1), param)
	return nothing
end

function _avgₕ!(uₕ::VectorElement, f, ::Val{D}) where D
	Ωₕ = mesh(space(uₕ))

	x = Base.Fix1(half_points, Ωₕ)
	meas = Base.Fix1(cell_measure, Ωₕ)

	param = (f, x, meas, indices(Ωₕ))
	__quadnd!(uₕ, (zeros(D), ones(D)), param)
	return nothing
end

function _avgₕ!(uₕ::VectorElement, f::BrambleFunction{D}, ::Val{D}) where D
	Ωₕ = mesh(space(uₕ))

	x = Base.Fix1(half_points, Ωₕ)
	meas = Base.Fix1(cell_measure, Ωₕ)

	param = (f, x, meas, indices(Ωₕ))
	__quadnd!(uₕ, (zeros(D), ones(D)), param)
	return nothing
end

"""
	__integrand1d(y, t, p)

Implements the integrand function needed in the calculation of [avgₕ](@ref). In this function, `y` denotes the return values, `t` denotes the integration variable and `p` denotes the parameters (integrand function `f`, points `x`, spacing `h` and indices `idxs`).

For efficiency, each integral in [avgₕ](@ref) is rewritten as an integral over `[0,1]` following

```math
\\int_{a}^{b} f(x) dx = (b-a) \\int_{0}^{1} f(a + t (b-a)) dt
```
"""
function __integrand1d(y, t, p)
	f, x, h, idxs = p

	@inbounds for i in idxs
		diff = (x(i + 1) - x(i))
		y[i] = f(x(i) + t * diff) * diff / h(i)
	end
	return nothing
end

function __quad!(uₕ::VectorElement, domain::NTuple{2,T}, p::ParamType) where {T,ParamType}
	prototype = zeros(ndofs(space(uₕ)))

	func = IntegralFunction(__integrand1d, prototype)
	prob = IntegralProblem(func, domain, p)
	sol = solve(prob, CubatureJLh())

	copyto!(uₕ.data, sol.u)
	return nothing
end

@inline @generated __shift_index1(idx::CartesianIndex{D}) where D = :(Base.Cartesian.@ntuple $D i->idx[i] + 1)

function __idx2points(t, x, idx::CartesianIndex{D}) where D
	lb = x(Tuple(idx))
	ub = x(__shift_index1(idx))

	diff = ub .- lb
	point = ntuple(i -> lb[i] + t[i] * diff[i], D)

	return point, diff
end

"""
	__integrandnd(y, t, p)

Implements the integrand function needed in the calculation of [avgₕ](@ref). In this function, `y` denotes the return values, `t` denotes the integration variable and `p` denotes the parameters (integrand function `f`, points `x`, measures `meas` and indices `idxs`).

For efficiency, each integral is calculated on ``[0,1]^D``, where ``D`` is the dimension of the integration domain. This is done through a similar change of variable as in [__integrand1d(y, t, p)](@ref).
"""
function __integrandnd(y, t, p)
	f, x, meas, idxs = p

	for idx in idxs
		point, diff = __idx2points(t, x, idx)
		y[idx] = f(point) * prod(diff) / meas(idx)
	end

	return nothing
end

function __quadnd!(uₕ::VectorElement, domain::NTuple{D,T}, p::ParamType) where {D,T,ParamType}
	npts = npoints(mesh(space(uₕ)), Tuple)
	v = Base.ReshapedArray(uₕ.data, npts, ())
	prototype = v

	func = IntegralFunction(__integrandnd, prototype)
	prob = IntegralProblem(func, domain, p)
	sol = solve(prob, CubatureJLh())
	#@show sol.u
	copyto!(v, sol.u)
	return nothing
end
=#