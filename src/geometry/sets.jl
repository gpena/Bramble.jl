"""
	struct CartesianProduct{D,T}
		box::NTuple{D,Tuple{T,T}}
	end

Type for storage of cartesian products of `D` intervals having elements of type `T`.
"""
struct CartesianProduct{D,T} <: BrambleType
	box::NTuple{D,Tuple{T,T}}
	collapsed::NTuple{D,Bool}
end

@inline is_collapsed(a::T, b::T) where T<:Number = isapprox(a, b)
@inline is_collapsed(X::CartesianProduct{1}) = X.collapsed[1]

@inline set(X::CartesianProduct) = X
"""
	interval(x, y)
	interval(x::CartesianProduct{1})

Returns a `1`-dimensional [CartesianProduct](@ref) representing the interval [`x`,`y`].

# Example

```@example
julia> interval(0, 1)
	Set
   Type: Float64
  Space: ℝ
	Dim: 1
	Set: [0.0, 1.0]
```
"""
@inline function interval(_x, _y)
	x = float(_x)
	y = float(_y)
	@assert x <= y

	_is_collapsed = is_collapsed(x, y)
	return CartesianProduct{1,typeof(x)}(((x, y),), (_is_collapsed,))
end

@inline interval(x::CartesianProduct{1}) = interval(x(1)...)

"""
	point(x)

Returns the interval `[x,x]`.

```@example
julia> point(0)
	Set
   Type: Float64
  Space: ℝ
	Dim: 0
	Set: {0.0}
```
"""
@inline function point(_x)
	x = float(_x)

	return CartesianProduct{1,typeof(x)}(((x, x),), (true,))
end

"""
	cartesianproduct(x, y)
	cartesianproduct(box::NTuple)
	cartesianproduct(X::CartesianProduct)

Returns a [CartesianProduct](@ref).

  - If `D` tuples are provided in `box`, it returns a `D`-dimensional [CartesianProduct](@ref).
  - If two values `x` and `y` are provided, it returns a `1`-dimensional [CartesianProduct](@ref) to represent the interval [`x`,`y`].

# Example

```@example
julia> cartesianproduct(0, 1)
	Set
   Type: Float64
  Space: ℝ
	Dim: 1
	Set: [0.0, 1.0]
```

```@example
julia> cartesianproduct(((0, 1), (4, 5)))
	Set
   Type: Float64
  Space: ℝ²
	Dim: 2
	Set: [0.0, 1.0] × [4.0, 5.0]
```
"""
@inline cartesianproduct(x, y) = interval(x, y)
@inline cartesianproduct(X::CartesianProduct) = X

@inline function cartesianproduct(box::NTuple{D,Tuple{T,T}}) where {D,T}
	predicate_result = all(x -> x[1] <= x[2], box)

	@assert predicate_result===true "Invalid box: Each tuple must satisfy x[1] <= x[2]. Check for non-compliant pairs or unexpected values. Found box: $box"

	_box = ntuple(i -> float.(box[i]), D)
	_falses = ntuple(i -> is_collapsed(_box[i]...) ? true : false, D)
	return CartesianProduct{D,eltype(_box[1])}(_box, _falses)
end

"""
	box(a,b)

Creates a [CartesianProduct](@ref) from the coordinates in `a` and `b`. It accepts `Number` or `NTuple{D}`. The subintervals of the new set are defined as `interval(a[1],b[1]) × ... × interval(a[D],b[D])`.
"""
@inline box(a, b) = interval(a, b)

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
	#return interval(box[i]...)
	return box[i]
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
@inline Base.eltype(_::CartesianProduct{D,T}) where {D,T} = T
@inline Base.eltype(::Type{<:CartesianProduct{D,T}}) where {D,T} = T

"""
	dim(X::CartesianProduct)
	dim(::Type{<:CartesianProduct})

Returns the dimension of the space where a [CartesianProduct](@ref) is embedded.

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
	topo_dim(X::CartesianProduct)

Returns the topological dimension of a [CartesianProduct](@ref).
"""
@inline topo_dim(X::CartesianProduct{D}) where D = (D - sum(X.collapsed))

"""
	tails(X::CartesianProduct, i)
	tails(X::CartesianProduct{D})

Returns `i`-th [interval](@ref) or [point](@ref) in [CartesianProduct](@ref) `X` as a Tuple. It can also be called on `X, returning a `D`-tuple with all intervals defining [CartesianProduct](@ref) `X`.

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
	@unpack box = X
	@assert i in eachindex(box)
	return X(i)
end

@inline tails(X::CartesianProduct{1}) = X(1)
@inline tails(X::CartesianProduct{D}) where D = ntuple(i -> X(i), Val(D))

@inline first(X::CartesianProduct{1}) = first(X(1))
@inline last(X::CartesianProduct{1}) = last(X(1))

"""
	×(X::CartesianProduct, Y::CartesianProduct)

Returns the cartesian product of two [CartesianProduct](@ref)s `X` and `Y` as a new [CartesianProduct](@ref).

# Example

```@example
julia> X = interval(0, 1);
	   Y = interval(2, 3);
	   X × Y;
	Set
   Type: Float64
  Space: ℝ²
	Dim: 2
	Set: [0.0, 1.0] × [2.0, 3.0]
```
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

# Example

```@example
julia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5);
	   projection(X, 1);
	Set
   Type: Float64
  Space: ℝ
	Dim: 1
	Set: [0.0, 1.0]
```
"""
@inline projection(X::CartesianProduct, i) = interval(X(i)...)
#=
function Base.show(io::IO, X::CartesianProduct{D}) where D
	@unpack box = X

	fields = ("Type", "Dim", "Set", "Markers")
	mlength = max_length_fields(fields)

	title_info = style_title("Set", max_length = mlength)
	output = style_join(title_info, set_info_only(X, mlength))
	print(io, output)
	return nothing
end

function set_info_only(X::CartesianProduct{D}, mlength) where D
	@unpack box = X

	colors = style_color_sets()
	num_colors = length(colors)

	styled_sets = [let
					   a, b = tails(X, i)
					   set_str = is_collapsed(a, b) ? "{$a}" : "[$a, $b]"
					   color_sym = colors[mod1(i, num_colors)]
					   styled"{$color_sym:$(set_str)}"
				   end
				   for i in eachindex(box)]

	sets_styled_combined = join(styled_sets, " × ")

	type_info = style_field("Type", eltype(X), max_length = mlength)
	dim_info = style_field("Space", style_real_space(Val(D)), max_length = mlength)
	topological_info = style_field("Dim", topo_dim(X), max_length = mlength)
	set_info = style_field("Set", sets_styled_combined, max_length = mlength)

	output = style_join(type_info, dim_info, topological_info, set_info)
	return output
end
=#