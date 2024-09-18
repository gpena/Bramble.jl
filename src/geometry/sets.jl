# Defines the [`Interval`](@ref) and [`CartesianProduct`](@ref) types
# 
# @author: Gonçalo Pena
#

"""
$(SIGNATURES)
Creates a Cartesian product of `D` intervals with elements of type `T`.

# Fields
  - `data`, a D-tuple containing the intervals defining the coordinate projections as 2-tuples.
"""
struct CartesianProduct{D,T} <: BrambleType
	data::NTuple{D,Tuple{T,T}}
end

"""
$(SIGNATURES)
Creates an interval set from two scalars x and y.

# Fields
  - `x`, the lower bound
  - `y`, the upper bound

# Example
```
julia> Interval(0.0, 1.0)
CartesianProduct{1,Float64}((0.0,1.0))
```
"""
@inline function Interval(x, y)
	_x = float(x)
	_y = float(y)
	return CartesianProduct{1,typeof(_x)}(((_x, _y),))
end


@inline CartesianProduct(x, y) = Interval(x, y)

@inline (X::CartesianProduct)(i) = X.data[i]

"""
$(SIGNATURES)

Get the element type of a Cartesian product.

# Fields
  - `X` -- the Cartesian product

# Example
```
julia> eltype(CartesianProduct(0.0, 1.0))
Float64
```
"""
@inline eltype(_::CartesianProduct{D,T}) where {D,T} = T
@inline eltype(_::Type{<:CartesianProduct{D,T}}) where {D,T} = T

"""
$(SIGNATURES)

Get the topological dimension of a Cartesian product.

# Fields
  - `X` -- the Cartesian product

# Example
```
julia> dim(CartesianProduct(0.0, 1.0))
1
```
"""
@inline dim(_::CartesianProduct{D}) where D = D
@inline dim(_::Type{CartesianProduct{D}}) where D = D

@inline Interval(x::CartesianProduct{1}) = Interval(x.data...)

@inline CartesianProduct(X::CartesianProduct) = X

@inline tails(X::CartesianProduct, i) = X(i)

@inline @generated tails(X::CartesianProduct{D}) where D = :(Base.Cartesian.@ntuple $D i->X(i))
#ntuple(i -> X(i), D)

@inline tails(X::CartesianProduct{1}) = X(1)

"""
$(SIGNATURES)

Compute the Cartesian product of two Cartesian products X and Y.

# Fields
  - `X` -- the first Cartesian product
  - `Y` -- the second Cartesian product

# Example
```
julia> X = CartesianProduct(0.0, 1.0);
	   Y = CartesianProduct(2.0, 3.0);
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

Get the i-th set in the Cartesian product X as an Interval.

# Fields
  - `X` -- the Cartesian product
  - `i` -- the index of the set
"""
@inline projection(X::CartesianProduct, i) = Interval(X(i)...)

function show(io::IO, X::CartesianProduct{D}) where D
	sets = ["[$(tails(X,i)[1]), $(tails(X,i)[2])]" for i in 1:D]
	sets_string = join(sets, " × ")
	print(io, "Type: $(eltype(X)) \n Dim: $D \n Set: $sets_string")
end