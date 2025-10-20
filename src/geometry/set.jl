
"""
	struct CartesianProduct{D,T}
		box::NTuple{D,Tuple{T,T}}
		collapsed::NTuple{D,Bool}
	end

A type representing the Cartesian product of `D` closed intervals in a space with element type `T`.

# Fields

  - `box::NTuple{D,Tuple{T,T}}`: A tuple of `D` pairs, where each pair represents the bounds `(min, max)` of an interval
  - `collapsed::NTuple{D,Bool}`: A tuple of `D` boolean values indicating whether each dimension is collapsed (i.e., min = max)
"""
struct CartesianProduct{D,T} <: BrambleType
	box::NTuple{D,Tuple{T,T}}
	collapsed::NTuple{D,Bool}
end

"""
	is_collapsed(a::T, b::T) where T<:Number
	is_collapsed(X::CartesianProduct{1})

Check if values or a set are collapsed (i.e. degenerate).

For two numbers `a` and `b`, returns `true` if they are approximately equal using `isapprox`.
For a 1-dimensional `CartesianProduct`, returns the collapse status stored in `X.collapsed[1]`.
"""
@inline is_collapsed(a::T, b::T) where T<:Number = isapprox(a, b)
@inline is_collapsed(X::CartesianProduct{1}) = X.collapsed[1]

@inline set(X::CartesianProduct) = X

"""
	interval(x, y)
	interval(x::CartesianProduct{1})

Returns a `1`-dimensional [CartesianProduct](@ref) representing the interval [`x`,`y`].
"""
@inline function interval(x, y)
	_x = float(x)
	_y = float(y)
	@assert _x <= _y

	_is_collapsed = is_collapsed(_x, _y)
	return CartesianProduct{1,typeof(_x)}(((_x, _y),), (_is_collapsed,))
end

@inline interval(x::CartesianProduct{1}) = interval(x(1)...)

"""
	point(x)

Returns the interval `[x,x]`.
"""
@inline function point(x)
	_x = float(x)

	return CartesianProduct{1,typeof(_x)}(((_x, _x),), (true,))
end

"""
	cartesian_product(x, y)
	cartesian_product(box::NTuple)
	cartesian_product(X::CartesianProduct)

Returns a [CartesianProduct](@ref).

  - If `D` tuples are provided in `box`, it returns a `D`-dimensional [CartesianProduct](@ref).
  - If two values `x` and `y` are provided, it returns a `1`-dimensional [CartesianProduct](@ref) to represent the interval [`x`,`y`].
"""
@inline cartesian_product(x, y) = interval(x, y)
@inline cartesian_product(X::CartesianProduct) = X

@inline function cartesian_product(box::NTuple{D,Tuple{T,T}}) where {D,T}
	predicate_result = all(x -> x[1] <= x[2], box)

	@assert predicate_result===true "Invalid box: Each tuple must satisfy x[1] <= x[2]. Check for non-compliant pairs or unexpected values. Found box: $box"

	_box = ntuple(i -> float.(box[i]), D)
	_falses = ntuple(i -> is_collapsed(_box[i]...) ? true : false, D)
	return CartesianProduct{D,eltype(_box[1])}(_box, _falses)
end

"""
	box(a::NTuple{D,T}, b::NTuple{D,T})
	box(a::Number, b::Number)

Creates a [CartesianProduct](@ref) from the coordinates in `a` and `b`. It accepts `Number` or `NTuple{D}`. The subintervals of the new set are defined as `interval(a[1],b[1]) × ... × interval(a[D],b[D])`.
"""
@inline box(a::Number, b::Number) = interval(a, b)

@inline function box(a::NTuple{D}, b::NTuple{D}) where D
	box_coords = Base.ntuple(Val(D)) do i
		return float.((a[i], b[i]))
	end

	collapsed_flags = Base.ntuple(Val(D)) do i
		return is_collapsed(float.(box_coords[i])...)
	end

	return CartesianProduct(box_coords, collapsed_flags)
end

"""
	(X::CartesianProduct)(i)

Returns the `i`-th [interval](@ref)) or [point](@ref) in the [CartesianProduct](@ref).
"""
@inline function (X::CartesianProduct)(i)
	@unpack box = X
	@assert i in eachindex(box)
	return box[i]
end

"""
	eltype(::CartesianProduct)
	eltype(::Type{<:CartesianProduct})

Returns the element type of a [CartesianProduct](@ref).

# Example

```julia
X = cartesian_product(0, 1);
eltype(X)
Float64
```
"""
@inline eltype(::CartesianProduct{D,T}) where {D,T} = T
@inline eltype(::Type{<:CartesianProduct{D,T}}) where {D,T} = T

"""
	dim(X::CartesianProduct)
	dim(::Type{<:CartesianProduct})

Returns the dimension of the space where a [CartesianProduct](@ref) is embedded.

# Examples

```julia
X = cartesian_product(0, 1);
dim(X)
1
```

```julia
Y = cartesian_product(((0, 1), (4, 5)));
dim(Y)
2
```
"""
@inline dim(::CartesianProduct{D}) where D = D
@inline dim(::Type{<:CartesianProduct{D}}) where D = D

"""
	topo_dim(X::CartesianProduct{D})

Returns the topological dimension of a [CartesianProduct](@ref).
"""
@inline topo_dim(X::CartesianProduct{D}) where D = (D - sum(X.collapsed))

"""
	tails(X::CartesianProduct{D})
	tails(X::CartesianProduct, i)

Returns `i`-th [interval](@ref) or [point](@ref) in [CartesianProduct](@ref) `X` as a Tuple. It can also be called on `X`, returning a `D`-tuple with all intervals defining [CartesianProduct](@ref) `X`.
"""
@inline function tails(X::CartesianProduct, i)
	@unpack box = X
	@assert i in eachindex(box)
	return X(i)
end

"""
	tails(X::CartesianProduct{1})
	tails(X::CartesianProduct{D}) where D

Returns the component sets of a Cartesian product.

For a one-dimensional [CartesianProduct](@ref) (`D=1`), returns the single component set.
For a D-dimensional [CartesianProduct](@ref), returns a tuple containing all component sets.
"""
@inline tails(X::CartesianProduct{1}) = X(1)
@inline tails(X::CartesianProduct{D}) where D = ntuple(i -> X(i), Val(D))

"""
	first(X::CartesianProduct{1})
	last(X::CartesianProduct{1})

Return the first and last elements of a one-dimensional [CartesianProduct](@ref), respectively.
These methods extend `Base.first` and `Base.last` for `CartesianProduct{1}`` types.
"""
@inline first(X::CartesianProduct{1}) = first(X(1))
@inline last(X::CartesianProduct{1}) = last(X(1))

"""
	×(X::CartesianProduct, Y::CartesianProduct)

Returns the cartesian product of two [CartesianProduct](@ref)s `X` and `Y` as a new [CartesianProduct](@ref).
"""
@generated function ×(X::CartesianProduct{D1,T}, Y::CartesianProduct{D2,T}) where {D1,D2,T}
	new_box_expr = :(tuple(X.box..., Y.box...))
	new_falses_expr = :(tuple(X.collapsed..., Y.collapsed...))
	ResultType = CartesianProduct{D1 + D2,T}
	final_expr = :($ResultType($new_box_expr, $new_falses_expr))

	return final_expr
end

"""
	projection(X::CartesianProduct, i)

Returns the `i`-th [interval](@ref) or [point](@ref)) in [CartesianProduct](@ref) `X` as a new [CartesianProduct](@ref).
"""
@inline projection(X::CartesianProduct, i) = interval(X(i)...)