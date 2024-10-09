"""
	struct CartesianProduct{D,T}
		data::NTuple{D,Tuple{T,T}}
	end

Type for storage of cartesian products of `D` intervals having elements of type `T`.
"""
struct CartesianProduct{D,T} <: BrambleType
	data::NTuple{D,Tuple{T,T}}
end

"""
	struct BrambleFunction{ArgsType,hastime,CoType}
		wrapped::FunctionWrapper{CoType,Tuple{ArgsType}}
	end

Structure to wrap around functions to make them more type agnostic. It uses `FunctionWrappers` to provide functions calculated on `ArgsType`. The type arguments are `hastime` to indicate if the function is time-dependent and `CoType`, the time of the codomain of the function.
"""
struct BrambleFunction{ArgsType,hastime,CoType}
	wrapped::FunctionWrapper{CoType,Tuple{ArgsType}}
end

"""
	interval(x, y)
	interval(x::CartesianProduct{1})

Returns a `1`-dimensional [CartesianProduct](@ref) representing the interval [`x`,`y`].

# Example

```@example
julia> interval(0, 1)
CartesianProduct{1,Float64}((0.0,1.0))
```
"""
@inline function interval(x, y)
	_x = float(x)
	_y = float(y)
	@assert _x <= _y

	return CartesianProduct{1,typeof(_x)}(((_x, _y),))
end

@inline interval(x::CartesianProduct{1}) = interval(x.data...)

"""
	cartesianproduct(x, y)
	cartesianproduct(data::NTuple)
	cartesianproduct(X::CartesianProduct)

Returns a `1`-dimensional [CartesianProduct](@ref) to represent the interval [`x`,`y`]. Alternatively, if `D` tuples are provided in `data`, it returns a `D`-dimensional [CartesianProduct](@ref).

# Example

```@example
julia> cartesianproduct(0, 1)
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]
```

```@example
julia> cartesianproduct(((0, 1), (4, 5)))
Type: Int64 
 Dim: 2 
 Set: [0, 1] × [4, 5]
```
"""
@inline cartesianproduct(x, y) = interval(x, y)
@inline cartesianproduct(X::CartesianProduct) = X

@inline function cartesianproduct(data::NTuple{D,Tuple{T,T}}) where {D,T}
	@assert all(x -> x[1] <= x[2], data)
	return CartesianProduct{D,T}(data)
end

"""
	(X::CartesianProduct)(i)

Returns the `i`-th interval in the [CartesianProduct](@ref).
"""
@inline function (X::CartesianProduct)(i)
	@assert i in eachindex(X.data)
	return X.data[i]
end

"""
	eltype(X::CartesianProduct)
	eltype(::Type{<:CartesianProduct})

Returns the element type of a [CartesianProduct](@ref).

# Example

```@example
julia> X = cartesianproduct(0, 1);
	   eltype(X);
Float64
```
"""
@inline eltype(_::CartesianProduct{D,T}) where {D,T} = T
@inline eltype(::Type{<:CartesianProduct{D,T}}) where {D,T} = T

"""
	dim(X::CartesianProduct)
	dim(::Type{<:CartesianProduct})

Returns the topological dimension of a [CartesianProduct](@ref).

# Example

```@example
julia> X = cartesianproduct(0, 1);
	   dim(X);
1
```

```@example
julia> Y = cartesianproduct(((0, 1), (4, 5)));
	   dim(Y);
2
```
"""
@inline dim(_::CartesianProduct{D}) where D = D
@inline dim(::Type{<:CartesianProduct{D}}) where D = D

"""
	tails(X::CartesianProduct, i)
	tails(X::CartesianProduct{D})

Returns `i`-th interval in [CartesianProduct](@ref) `X` as a Tuple. It can also be called on `X, returning a `D`-tuple with all intervals defining [CartesianProduct](@ref) `X`.

# Example

```@example
julia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5);
	   tails(X, 1);
(0.0, 1.0)
```

```@example
julia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5);
	   tails(X);
((0.0, 1.0), (4.0, 5.0))
```
"""
@inline function tails(X::CartesianProduct, i)
	@assert i in eachindex(X.data)
	return X(i)
end

@inline @generated tails(X::CartesianProduct{D}) where D = :(Base.Cartesian.@ntuple $D i->X(i))

@inline tails(X::CartesianProduct{1}) = X(1)

"""
	×(X::CartesianProduct, Y::CartesianProduct)

Returns the cartesian product of two [CartesianProduct](@ref)s `X` and `Y` as a new [CartesianProduct](@ref).

# Example

```@example
julia> X = interval(0, 1);
	   Y = interval(2, 3);
	   X × Y;
Type: Float64 
 Dim: 2 
 Set: [0.0, 1.0] × [2.0, 3.0]
```
"""
@inline function ×(X::CartesianProduct{D1,T}, Y::CartesianProduct{D2,T}) where {D1,D2,T}
	a = tails(X)
	b = tails(Y)
	c = tuple((a...)..., (b...)...)

	return CartesianProduct{D1 + D2,T}(ntuple(i -> (c[2 * i - 1], c[2 * i]), D1 + D2))
end

"""
	projection(X::CartesianProduct, i)

Returns the `i`-th interval in [CartesianProduct](@ref) `X` as a new `1`-dimensional [CartesianProduct](@ref).

# Example

```@example
julia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5);
	   projection(X, 1);
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]
```
"""
@inline projection(X::CartesianProduct, i) = interval(X(i)...)

function show(io::IO, X::CartesianProduct{D}) where D
	sets = ["[$(tails(X,i)[1]), $(tails(X,i)[2])]" for i in eachindex(X.data)]
	sets_string = join(sets, " × ")
	print(io, "Type: $(eltype(X)) \n Dim: $D \n Set: $sets_string")
end

function _embed_notime(X, f)
	D = dim(X)
	T = eltype(X)
	ArgsType = D == 1 ? T : NTuple{D,T}

	wrapped_f_tuple = FunctionWrapper{T,Tuple{ArgsType}}(f)

	return BrambleFunction{ArgsType,false,T}(wrapped_f_tuple)
end

"""
	@embed(X:CartesianProduct, f)
	@embed(X:Domain, f)
	@embed(X:MeshType, f)
	@embed(X:SpaceType, f)
	@embed(X × I, f)

Returns a new wrapped version of function `f`. If

  - `X` is not a [SpaceType](@ref), the topological dimension of `X` is used to caracterize the types in the returning [BrambleFunction](@ref); in this case, the new function can be applied to `Tuple`s (with point coordinates);

  - `X` is a [SpaceType](@ref), the returning function type, [BrambleFunction](@ref), is caracterized by the dimension of the [element](@ref)s of the space.

If the first argument is of the form `X × I`, where `I` is an interval, then `f` must be a time-dependent function defined as `f(x,t) = ...`.

# Example

```@example
julia> Ω = domain(interval(0, 1) × interval(0, 1));
	   f = @embed Ω x->x[1] * x[2] + 1;  # or f = @embed(Ω, x -> x[1]*x[2]+1)

```
"""
macro embed(domain_expr, func_expr)
	space_domain, time_domain = _get_domains(domain_expr)

	if func_expr isa Symbol || func_expr.head == :-> || func_expr.head == :.
		if time_domain isa Nothing
			# case of function only depending on space variables
			return esc(:(Bramble._embed_notime($space_domain, $func_expr)))
		end

		# case when the function also depends on a time variable
		return esc(:(Bramble._embed_withtime($space_domain, $time_domain, $func_expr)))
	end

	return :(@error "Don't know how to handle this expression")
end

@inline function _get_domains(expr::Expr)
	if expr.args isa Array && length(expr.args) == 3
		args = expr.args
		return args[2], args[3]
	end

	if expr.head == :call
		return :($expr), nothing
	end

	return :(@error "Invalid domain format. Please write the first input as Ω × I, where Ω is the space domain and I is the time domain.")
end

@inline _get_domains(s::Symbol) = :($s), nothing

function (f::BrambleFunction{NTuple{D,T},false})(x...) where {D,T}
	if x[1] isa Tuple
		@assert length(x[1]) == D
		y = Tuple(convert.(T, x[1])::NTuple{D,T})
		return f.wrapped(y)
	end

	@assert length(x) == D

	y = Tuple(convert.(T, x)::NTuple{D,T})
	return f.wrapped(y)
end

(f::BrambleFunction{ArgsType,false})(x) where {ArgsType<:Number} = f.wrapped(convert(ArgsType, x)::ArgsType)
(f::BrambleFunction{ArgsType,true})(t) where ArgsType = f.wrapped(t)

function _embed_withtime(X, I::CartesianProduct{1}, f)
	_f(t) = _embed_notime(X, Base.Fix2(f, t))

	ArgsType = eltype(X)
	CoType = typeof(_f(sum(tails(I)) * 0.5))

	wrapped_f_tuple = FunctionWrapper{CoType,Tuple{ArgsType}}(_f)

	return BrambleFunction{ArgsType,true,CoType}(wrapped_f_tuple)
end