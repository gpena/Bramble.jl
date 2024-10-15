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

@inline firstindex(X::CartesianProduct{1,T}) where T = firstindex(X(1))
@inline lastindex(X::CartesianProduct{1,T}) where T = lastindex(X(1))

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