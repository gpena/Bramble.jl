# Defines the [`Interval`](@ref) and [`CartesianProduct`](@ref) types
# 
# @author: Gonçalo Pena
#

"""
$(SIGNATURES)
	struct CartesianProduct{D,T}
		data::NTuple{D,Tuple{T,T}}
	end

Type for storage of cartesian products of `D` intervals having elements of type `T`.
"""
struct CartesianProduct{D,T} <: BrambleType
	data::NTuple{D,Tuple{T,T}}
end

"""
$(SIGNATURES)
Returns a 1D [CartesianProduct](@ref) set from two scalars `x` and `y`, where `x` and `y` are, respectively, the lower and upper bounds of the interval.

# Example
```
julia> Interval(0, 1)
CartesianProduct{1,Float64}((0.0,1.0))
```
"""
@inline function Interval(x, y)
	_x = float(x)
	_y = float(y)
	@assert _x <= _y

	return CartesianProduct{1,typeof(_x)}(((_x, _y),))
end

"""
$(SIGNATURES)
Returns a 1D [CartesianProduct](@ref) from two scalars `x` and `y`, where `x` and `y` are, respectively, the lower and upper bounds of the interval.

# Example
```
julia> CartesianProduct(0, 1)
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]
```
"""
@inline CartesianProduct(x, y) = Interval(x, y)

@inline (X::CartesianProduct)(i) = X.data[i]

"""
$(SIGNATURES)
Returns the element type of a [CartesianProduct](@ref).

# Example
```
julia> X = CartesianProduct(0, 1); eltype(X)
Float64
```
"""
@inline eltype(X::CartesianProduct{D,T}) where {D,T} = T
@inline eltype(::Type{<:CartesianProduct{D,T}}) where {D,T} = T

"""
$(SIGNATURES)
Returns the topological dimension of a [CartesianProduct](@ref).

# Example
```
julia> X = CartesianProduct(0, 1); dim(X)
1
```
"""
@inline dim(X::CartesianProduct{D}) where D = D
@inline dim(::Type{CartesianProduct{D}}) where D = D

@inline Interval(x::CartesianProduct{1}) = Interval(x.data...)

@inline CartesianProduct(X::CartesianProduct) = X

"""
$(SIGNATURES)

Returns a tuple with the 1D [CartesianProduct](@ref) of the i-th interval of the [CartesianProduct](@ref) `X`.

# Example
```
julia> X = CartesianProduct(0, 1) × CartesianProduct(4, 5); tails(X,1)
(0.0, 1.0)
```
"""
@inline tails(X::CartesianProduct, i) = X(i)

"""
$(SIGNATURES)

Returns a tuple of tuples with 1D [CartesianProduct](@ref)s that make up the [CartesianProduct](@ref) `X`.
# Example
```
julia> X = CartesianProduct(0, 1) × CartesianProduct(4, 5); tails(X)
((0.0, 1.0), (4.0, 5.0))
```
"""
@inline @generated tails(X::CartesianProduct{D}) where D = :(Base.Cartesian.@ntuple $D i->X(i))

@inline tails(X::CartesianProduct{1}) = X(1)

"""
$(SIGNATURES)
Returns the cartesian product of two [CartesianProduct](@ref) `X` and `Y` as a [CartesianProduct](@ref).

# Example
```
julia> X = CartesianProduct(0, 1); Y = CartesianProduct(2, 3);
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
$(SIGNATURES)
Returns the i-th 1D [CartesianProduct](@ref) of the [CartesianProduct](@ref) `X`.

# Example
```
julia> X = CartesianProduct(0, 1) × CartesianProduct(4, 5); projection(X, 1)
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]
```
"""
@inline projection(X::CartesianProduct, i) = Interval(X(i)...)

function show(io::IO, X::CartesianProduct{D}) where D
	sets = ["[$(tails(X,i)[1]), $(tails(X,i)[2])]" for i in 1:D]
	sets_string = join(sets, " × ")
	print(io, "Type: $(eltype(X)) \n Dim: $D \n Set: $sets_string")
end