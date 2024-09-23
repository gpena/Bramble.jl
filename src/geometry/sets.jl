"""
	interval(x, y)

Returns a 1D [CartesianProduct](@ref) set from two scalars `x` and `y`, where `x` and `y` are, respectively, the lower and upper bounds of the interval.

# Example

```
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
	cartesianproduct(data::NTuple)

Returns a [CartesianProduct](@ref) from a tuple of intervals.
"""
@inline function cartesianproduct(data::NTuple{D,Tuple{T,T}}) where {D,T}
	@assert all(x -> x[1] <= x[2], data)
	return CartesianProduct{D,T}(data)
end

"""
	cartesianproduct(x, y)

Returns a 1D [CartesianProduct](@ref) from two scalars `x` and `y`, where `x` and `y` are, respectively, the lower and upper bounds of the interval.

# Example

```
julia> cartesianproduct(0, 1)
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]
```
"""
@inline cartesianproduct(x, y) = interval(x, y)

@inline function (X::CartesianProduct)(i)
	@assert i in eachindex(X.data)
	return X.data[i]
end

"""
	eltype(X::CartesianProduct)

Returns the element type of a [CartesianProduct](@ref).

# Example

```
julia> X = cartesianproduct(0, 1); eltype(X)
Float64
```
"""
@inline eltype(_::CartesianProduct{D,T}) where {D,T} = T
@inline eltype(::Type{<:CartesianProduct{D,T}}) where {D,T} = T

"""
	dim(X::CartesianProduct)

Returns the topological dimension of a [CartesianProduct](@ref).

# Example

```
julia> X = cartesianproduct(0, 1); dim(X)
1
```
"""
@inline dim(_::CartesianProduct{D}) where D = D
@inline dim(::Type{<:CartesianProduct{D}}) where D = D

@inline interval(x::CartesianProduct{1}) = interval(x.data...)

@inline cartesianproduct(X::CartesianProduct) = X

"""
	tails(X::CartesianProduct, i)

Returns a tuple with the 1D [CartesianProduct](@ref) of the i-th interval of the [CartesianProduct](@ref) `X`.

# Example

```
julia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5); tails(X,1)
(0.0, 1.0)
```
"""
@inline function tails(X::CartesianProduct, i)
	@assert i in eachindex(X.data)
	return X(i)
end

"""
	tails(X::CartesianProduct)

Returns a tuple of tuples with 1D [CartesianProduct](@ref)s that make up the [CartesianProduct](@ref) `X`.

# Example

```
julia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5); tails(X)
((0.0, 1.0), (4.0, 5.0))
```
"""
@inline @generated tails(X::CartesianProduct{D}) where D = :(Base.Cartesian.@ntuple $D i->X(i))

@inline tails(X::CartesianProduct{1}) = X(1)

"""
	×(X::CartesianProduct, Y::CartesianProduct)

Returns the cartesian product of two [CartesianProduct](@ref) `X` and `Y` as a [CartesianProduct](@ref).

# Example

```
julia> X = cartesianproduct(0, 1); Y = cartesianproduct(2, 3);
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

Returns the i-th 1D [CartesianProduct](@ref) of the [CartesianProduct](@ref) `X`.

# Example

```
julia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5); projection(X, 1)
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