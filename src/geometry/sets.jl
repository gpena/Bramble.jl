# Defines the [`Interval`](@ref) and [`CartesianProduct`](@ref) types
# 
# @author: Gonçalo Pena
#

"""
$(SIGNATURES)
Creates a cartesian product of `D` intervals with elements of type `T`.

# Fields
  - `data`, a D-tuple containing the intervals defining the coordinate projections as 2-tuples.
```@docs; canonical=false
```
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
```@docs; canonical=false
julia> Interval(0.0, 1.0)
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
Creates an interval set from two scalars x and y.

# Fields
  - `x`, the lower bound
  - `y`, the upper bound

# Example
```@docs; canonical=false
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
Get the element type of a cartesian product set.

# Fields
  - `X` -- the set

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
Get the topological dimension of a cartesian product set.

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

@inline tails(X::CartesianProduct{1}) = X(1)

"""
$(SIGNATURES)
Compute the cartesian product of two cartesian products X and Y.

# Fields
  - `X` -- the first set
  - `Y` -- the second set

# Example
```@docs; canonical=false
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
Get the i-th set in the cartesian product set X (as a 1D set).

# Fields
  - `X` -- the set
  - `i` -- the index of the set
"""
@inline projection(X::CartesianProduct, i) = Interval(X(i)...)

function show(io::IO, X::CartesianProduct{D}) where D
	sets = ["[$(tails(X,i)[1]), $(tails(X,i)[2])]" for i in 1:D]
	sets_string = join(sets, " × ")
	print(io, "Type: $(eltype(X)) \n Dim: $D \n Set: $sets_string")
end