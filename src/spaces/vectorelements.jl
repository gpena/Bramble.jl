"""
	element(Wₕ::SpaceType)

Returns a [VectorElement](@ref) for [GridSpace](@ref) `Wₕ` with uninitialized components.
"""
@inline element(Wₕ::SpaceType) = element(Wₕ, Vector{eltype(Wₕ)}(undef, ndofs(Wₕ)))

"""
	element(Wₕ::SpaceType, α::Number)

Returns a [VectorElement](@ref) for [GridSpace](@ref) `Wₕ` with all components equal to `α`.
"""
function element(Wₕ::SpaceType, α::Number)
	T = eltype(Wₕ)
	vₕ = Vector{T}(convert(T, α)::T * Ones(ndofs(Wₕ)))
	return element(Wₕ, vₕ)
end

"""
	element(Wₕ::SpaceType, v::AbstractVector)

Returns a [VectorElement](@ref) for [GridSpace](@ref) `Wₕ` with the same coefficients of `v`.
"""
@inline element(Wₕ::SpaceType, v::AbstractVector) = (@assert length(v) == ndofs(Wₕ);
													 VectorElement(Wₕ, v))

@inline ndims(::Type{<:VectorElement}) = 1

Base.BroadcastStyle(::Type{<:VectorElement}) = Broadcast.ArrayStyle{VectorElement}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}}, ::Type{ElType}) where ElType
	# Scan the inputs for the VectorElement:
	A = _find_vec_in_broadcast(bc)
	return element(space(A), similar(A.values, ElType))
end

_find_vec_in_broadcast(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}}) = _find_vec_in_broadcast(bc.args)
_find_vec_in_broadcast(args::Tuple) = _find_vec_in_broadcast(_find_vec_in_broadcast(args[1]), Base.tail(args))
_find_vec_in_broadcast(x) = x
_find_vec_in_broadcast(::Tuple{}) = nothing
_find_vec_in_broadcast(a::VectorElement, rest) = a
_find_vec_in_broadcast(::Any, rest) = _find_vec_in_broadcast(rest)

@inline axes(uₕ::VectorElement) = axes(uₕ.values)

@inline function copyto!(dest::VectorElement, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}})
	#println("copyto!")
	copyto!(dest.values, convert(Broadcast.Broadcasted{Nothing}, bc))

	return dest
end

function Base.copyto!(dest::DestType, bc::Broadcast.Broadcasted{Nothing}) where DestType
	"here1111"
	@show bc
	copyto!(dest, convert(Broadcast.Broadcasted{Nothing}, bc))
end

@inline function copyto!(dest::VectorElement, bc::Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{0}})
	println("here-1")

	if bc.f === identity && bc.args isa Tuple{Any} && isflat(bc)
		return fill!(dest.values, bc.args[1][])
	else
		return copyto!(dest.values, convert(Broadcasted{Nothing}, bc))
	end
end

@inline function copyto!(dest::VectorElement, bc::Broadcast.Broadcasted{Nothing})
	println("here0")
	axes(dest.values) == axes(bc) || throwdm(axes(dest.values), axes(bc))

	if bc.f === identity && bc.args isa Tuple{VectorElement} # only a single input argument to broadcast!
		A = bc.args[1]
		if axes(dest.values) == axes(A)
			return copyto!(dest.values, A)
		end
	end
	bc′ = Broadcast.preprocess(dest.values, bc)

	@inbounds @simd for I in eachindex(bc′)
		dest.values[I] = bc′[I]
	end
	return dest
end

function materialize!(dest::VectorElement, x)
	println("here1")
	return materialize!(dest.values, instantiate(Broadcasted(identity, (x,), axes(dest.values))))
end

@inline function materialize!(dest::VectorElement, bc::Broadcast.Broadcasted{<:Any})
	aux = Broadcast.instantiate(Broadcast.Broadcasted(bc.style, bc.f, bc.args, axes(dest)))
	@simd for i in eachindex(aux)
		dest.values[i] = aux[i]
	end

	return dest
end

function materialize(v::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement},Nothing,Any,Tuple{T1,VectorElement{S,T2}}}) where {T1,T2,S}
	println("materialize!")
end

@inline function materialize!(::Broadcast.BroadcastStyle, dest::VectorElement, bc::Broadcast.Broadcasted{<:Any})
	println("here3")

	return copyto!(dest.values, instantiate(Broadcasted(bc.style, bc.f, bc.args, axes(dest.values))))
end

@inline Base.@propagate_inbounds function getindex(uₕ::VectorElement, i)
	@boundscheck checkbounds(uₕ.values, i)
	return getindex(uₕ.values, i)
end
@inline Base.@propagate_inbounds function setindex!(uₕ::VectorElement, val, i)
	@boundscheck checkbounds(uₕ.values, i)
	setindex!(uₕ.values, val, i)
end
#=
@inline firstindex(uₕ::VectorElement) = firstindex(uₕ.values)
@inline lastindex(uₕ::VectorElement) = lastindex(uₕ.values)
=#
#@inline iterate(uₕ::VectorElement) = iterate(uₕ.values)
#@inline iterate(uₕ::VectorElement, state) = iterate(uₕ.values, state)

Base.show(io::IO, uₕ::VectorElement) = Base.show(io, "text/plain", uₕ.values)

"""
	eltype(uₕ::VectorElement{S,T})
	eltype(::Type{<:VectorElement{S,T}})

Returns the element type of a [VectorElement](@ref) `uₕ`, `T``.
"""
@inline eltype(uₕ::VectorElement{S,T}) where {S,T} = T
@inline eltype(::Type{<:VectorElement{S,T}}) where {S,T} = T

@inline length(uₕ::VectorElement) = length(uₕ.values)

"""
	space(uₕ::VectorElement)

Returns the space associated with [VectorElement](@ref) `uₕ`.
"""
@inline space(uₕ::VectorElement) = uₕ.space

"""
	similar(uh::VectorElement)

Returns a new [VectorElement](@ref) belonging to the same [GridSpace](@ref) as `uh`, with uninitialized components.
"""
@inline similar(uₕ::VectorElement) = VectorElement(space(uₕ), similar(uₕ.values))

@inline size(uₕ::VectorElement) = size(uₕ.values)

"""
	copyto!(uₕ::VectorElement, vₕ::VectorElement)
	copyto!(uₕ::VectorElement, v::AbstractVector)
	copyto!(uₕ::VectorElement, α::Number)

Copies the coefficients of [VectorElement](@ref) `vₕ` into [VectorElement](@ref) `uₕ`. The second argument can also be a regular `Vector` or a `Number``.
"""
@inline copyto!(uₕ::VectorElement, vₕ::VectorElement) = (@assert length(uₕ.values) == length(vₕ.values);
														 @.. uₕ.values = vₕ.values)
@inline copyto!(uₕ::VectorElement, v::AbstractVector) = (@assert length(uₕ.values) == length(v);
														 @.. uₕ.values = v)
@inline copyto!(uₕ::VectorElement, α::Number) = (s = convert(eltype(uₕ), α)::eltype(uₕ);
												 @.. uₕ.values = s)

"""
	map(f, uₕ::VectorElement)

Returns a new [VectorElement](@ref) with coefficients obtained by applying function `f` to each coefficient of `uₕ`

	yₕ[i] = f(uₕ[i])
"""
#@inline map(f, uₕ::VectorElement) = element(space(u), map(f, uₕ.values))

for op in (:-, :*, :/, :+, :^)
	same_text = "\n\nReturns a new [VectorElement](@ref) with coefficients given by the elementwise evaluation of"
	docstr1 = "	" * string(op) * "(α::Number, uₕ::VectorElement)" * same_text * "`α`" * string(op) * "`uₕ`."
	docstr2 = "	" * string(op) * "(uₕ::VectorElement, α::Number)" * same_text * "`uₕ`" * string(op) * "`α`."
	docstr3 = "	" * string(op) * "(uₕ::VectorElement, vₕ::VectorElement)" * same_text * " `uₕ`" * string(op) * "`vₕ`."

	@eval begin
		@doc $docstr1 @inline function (Base.$op)(α::Number, uₕ::VectorElement)
			rₕ = similar(uₕ)
			map!(Base.Fix1(Base.$op, α), rₕ.values, uₕ.values)
			return rₕ
		end

		@doc $docstr2 @inline function (Base.$op)(uₕ::VectorElement, α::Number)
			rₕ = similar(uₕ)

			map!(Base.Fix2(Base.$op, α), rₕ.values, uₕ.values)
			return rₕ
		end

		@doc $docstr3 @inline function (Base.$op)(uₕ::VectorElement, vₕ::VectorElement)
			rₕ = similar(uₕ)
			map!((Base.$op), rₕ.values, uₕ.values, vₕ.values)
			return rₕ
		end
	end
end

function *(uₕ::VectorElement, Vₕ::NTuple{D,VectorElement}) where D
	Zₕ = ntuple(i-> similar(Vₕ[i]), D)
	for i in 1:D
		Zₕ[i].values .= uₕ.values .* Vₕ[i].values
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
	u = Base.ReshapedArray(uₕ.values, npoints(mesh(space(uₕ)), Tuple), ())
	_func2array!(u, f, mesh(space(uₕ)))
	return nothing
end

"""
	Rₕ(Wₕ::SpaceType, f)

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
function Rₕ(Wₕ::SpaceType, f)
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
	avgₕ(Wₕ::SpaceType, f)

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
@inline function avgₕ(Wₕ::SpaceType, f)
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

	copyto!(uₕ.values, sol.u)
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
	v = Base.ReshapedArray(uₕ.values, npts, ())
	prototype = v

	func = IntegralFunction(__integrandnd, prototype)
	prob = IntegralProblem(func, domain, p)
	sol = solve(prob, CubatureJLh())
	#@show sol.u
	copyto!(v, sol.u)
	return nothing
end