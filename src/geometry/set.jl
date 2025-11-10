
"""
	$(TYPEDEF)

A type representing the cartesian product of `D` closed intervals in a space with element type `T`.

# Fields

$(FIELDS)
"""
struct CartesianProduct{D,T}
	"a container of `D` pairs, where each pair represents the bounds `(min, max)` of an interval"
	box::SVector{D,Tuple{T,T}}
	"a container of `D` boolean values indicating whether each dimension is collapsed (i.e., min = max)"
	collapsed::SVector{D,Bool}
end

"""
	is_collapsed(a::T, b::T)
	is_collapsed(X::CartesianProduct{1})

Checks if a 1D [CartesianProduct](@ref) or two numbers are "collapsed" (i.e., degenerate).

  - For two numbers `a` and `b`, it returns `true` if they are approximately equal using `isapprox`.
  - For a 1-dimensional [CartesianProduct](@ref), it returns the pre-computed collapse status stored in `X.collapsed[1]`.
"""
@inline is_collapsed(a::T, b::T) where T<:Number = isapprox(a, b)
@inline is_collapsed(X::CartesianProduct{1}) = X.collapsed[1]

"""
	set(X::CartesianProduct)

Returns the [CartesianProduct](@ref) itself. This can be useful for functions that expect a domain object and might receive either the object or a wrapper.
"""
@inline set(X::CartesianProduct) = X

"""
	$(SIGNATURES)

Constructs a 1-dimensional [CartesianProduct](@ref) representing the closed interval ``[x, y]``.
The inputs are converted to floating-point numbers. It also accepts an existing 1D [CartesianProduct](@ref) as a single argument.
"""
@inline function interval(x, y)
	_x = float(x)
	_y = float(y)
	@assert _x <= _y

	_is_collapsed = is_collapsed(_x, _y)
	box = SVector{1}(((_x, _y),))
	collapsed = SVector{1}((_is_collapsed,))
	return CartesianProduct{1,typeof(_x)}(box, collapsed)
end

@inline interval(x::CartesianProduct{1}) = interval(x(1)...)

"""
	$(SIGNATURES)

Creates a collapsed 1D [CartesianProduct](@ref) representing the point ``[x, x]``.
"""
@inline function point(x)
	_x = float(x)
	box = SVector{1}(((_x, _x),))
	collapsed = SVector{1}((true,))
	return CartesianProduct{1,typeof(_x)}(box, collapsed)
end

"""
	cartesian_product(x, y)
	cartesian_product(box)

Returns a [CartesianProduct](@ref).

  - If `D` tuples are provided in `box`, it returns a `D`-dimensional [CartesianProduct](@ref).
  - If two values `x` and `y` are provided, it returns a `1`-dimensional [CartesianProduct](@ref) to represent the interval [`x`,`y`].
"""
@inline cartesian_product(x, y) = interval(x, y)

@inline function cartesian_product(box::NTuple{D,Tuple{T,T}}) where {D,T}
	# Check that all intervals are valid (min <= max).
	predicate_result = all(x -> x[1] <= x[2], box)
	@assert predicate_result===true "Invalid box: Each tuple must satisfy x[1] <= x[2]. Check for non-compliant pairs or unexpected values. Found box: $box"

	# Convert all coordinates to floating-point numbers.
	_box = ntuple(i -> float.(box[i]), D)
	# Pre-compute whether each dimension is collapsed.
	_collapsed_flags = ntuple(i -> is_collapsed(_box[i]...), D)

	return CartesianProduct{D,eltype(_box[1])}(SVector(_box), SVector(_collapsed_flags))
end

@inline cartesian_product(X::CartesianProduct) = X

"""
	box(a::Number, b::Number)
	box(a::NTuple, b::NTuple)

Creates a [CartesianProduct](@ref) from two points `a` and `b`, which define the corners of the box.
The component intervals are defined as ``[\\min(a_i, bᵢ), \\max(a_i, b_i)]``. It accepts both numbers (for 1D)
and `NTuple` (for D-dimensions).
"""
@inline box(a::Number, b::Number) = interval(a, b)

@inline function box(a::NTuple{D}, b::NTuple{D}) where D
	# Create the box coordinates by taking the min and max for each dimension.
	box_coords = ntuple(i -> float.((min(a[i], b[i]), max(a[i], b[i]))), Val(D))

	# Determine which dimensions are collapsed.
	collapsed_flags = ntuple(i -> is_collapsed(box_coords[i]...), Val(D))

	return CartesianProduct{D,eltype(box_coords[1])}(SVector(box_coords), SVector(collapsed_flags))
end

"""
	center(cp::CartesianProduct{D,T}) -> SVector{D,T}

Returns the center point of a CartesianProduct domain.
"""
@inline function center(cp::CartesianProduct{D,T}) where {D,T}
	return SVector{D,T}(ntuple(i -> muladd(T(0.5), cp.box[i][2], T(0.5) * cp.box[i][1]), Val(D)))
end

"""
	(X::CartesianProduct)(i)

Returns the `i`-th [interval](@ref) (or [point](@ref)) in the [CartesianProduct](@ref).

# Example

```jldoctest
julia> Y = cartesian_product(((0.0, 1.0), (4.0, 5.0)));
	   Y(2)
(4.0, 5.0)
```
"""
@inline function (X::CartesianProduct)(i)
	@unpack box = X
	@boundscheck @assert i in eachindex(box) "Index $i is out of bounds for a CartesianProduct of dimension $(dim(X))."
	return @inbounds box[i]
end

"""
	eltype(X::CartesianProduct)

Returns the element type of a [CartesianProduct](@ref). Can be applied directly to the type of the [CartesianProduct](@ref).

# Example

```jldoctest
julia> X = cartesian_product(0, 1);
	   eltype(X)
Float64
```
"""
@inline eltype(::CartesianProduct{D,T}) where {D,T} = T
@inline eltype(::Type{<:CartesianProduct{D,T}}) where {D,T} = T

"""
	dim(X::CartesianProduct)

Returns the dimension of the space where the [CartesianProduct](@ref) is embedded. Can be applied directly to the type of the [CartesianProduct](@ref).

# Examples

```jldoctest
julia> X = cartesian_product(0, 1);
	   dim(X)
1
```

```jldoctest
julia> Y = cartesian_product(((0, 1), (4, 5)));
	   dim(Y)
2
```
"""
@inline dim(::CartesianProduct{D}) where D = D
@inline dim(::Type{<:CartesianProduct{D}}) where D = D

"""
	topo_dim(X::CartesianProduct)

Returns the topological dimension of a [CartesianProduct](@ref). The depends on the dimension of the [CartesianProduct](@ref) and the number of collapsed dimensions.
"""
@inline @fastmath topo_dim(X::CartesianProduct{D}) where D = (D - sum(X.collapsed))

"""
	tails(X::CartesianProduct, i)

Returns the `i`-th component interval of the [CartesianProduct](@ref) `X` as a `Tuple`. This is an alias for `X(i)`.
"""
@inline function tails(X::CartesianProduct, i)
	@unpack box = X
	@assert i in eachindex(box)
	return X(i)
end

"""
	tails(X::CartesianProduct)

Returns the component sets of a [CartesianProduct](@ref):

  - for a one-dimensional [CartesianProduct](@ref) (`D=1`), returns the single component set.
  - for a `D`-dimensional [CartesianProduct](@ref), returns a tuple containing all component sets.
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

Computes the [CartesianProduct](@ref) of two [CartesianProduct](@ref)s X and Y. The new dimension will be the sum of the dimensions of `X` and `Y`. Can be used as

```julia
X × Y
```
"""
@inline function ×(X::CartesianProduct{D1,T}, Y::CartesianProduct{D2,T}) where {D1,D2,T}
	D = D1 + D2
	# Concatenate boxes and collapsed flags using vcat (works for both SVector and NTuple)
	new_box = vcat(X.box, Y.box)
	new_collapsed = vcat(X.collapsed, Y.collapsed)

	return CartesianProduct{D,T}(new_box, new_collapsed)
end

"""
	projection(X::CartesianProduct, i)

Returns the `i`-th component interval of `X`` as a new 1D [CartesianProduct](@ref).
"""
@inline projection(X::CartesianProduct, i) = interval(X(i)...)

"""
	point_type(X::CartesianProduct)

Determines the type of a coordinate point within a `CartesianProduct` space. Returns `T` for 1D spaces and `NTuple{D,T}` for D-dimensional spaces.
"""
@inline point_type(X::CartesianProduct{1,T}) where T = T
@inline point_type(X::CartesianProduct{D,T}) where {D,T} = NTuple{D,T}

"""
	Base.show(io::IO, X::CartesianProduct)

Custom display for CartesianProduct objects, showing dimension info and collapsed status with colors.
"""
function Base.show(io::IO, X::CartesianProduct{D,T}) where {D,T}
	pp = PrettyPrinter(io)

	if pp.compact
		# Compact display for arrays/collections
		if D == 1
			collapsed = X.collapsed[1]
			if collapsed
				print(io, "Point(", X.box[1][1], ")")
			else
				print(io, "[", X.box[1][1], ", ", X.box[1][2], "]")
			end
		else
			for i in 1:D
				i > 1 && print(io, " × ")
				if X.collapsed[i]
					print(io, X.box[i][1])
				else
					print(io, "[", X.box[i][1], ", ", X.box[i][2], "]")
				end
			end
		end
	else
		# Detailed display
		topodim = topo_dim(X)

		if D == 1
			collapsed = X.collapsed[1]
			print_colored(pp, "CartesianProduct{$D,$T}"; bold = true, color = :cyan)
			print(io, ": ")
			if collapsed
				print_colored(pp, "Point"; color = :yellow)
				print(io, " at ")
				print_value(pp, X.box[1][1])
			else
				print_colored(pp, "Interval"; color = :yellow)
				print(io, " ")
				print_interval(pp, X.box[1][1], X.box[1][2])
			end
		else
			# Header with topological dimension info
			print_colored(pp, "CartesianProduct{$D,$T}"; bold = true, color = :cyan)
			if topodim < D
				print_colored(pp, " (topological dim $topodim)"; color = :yellow)
			end
			println(io, ":")

			# Show each dimension
			pp_indented = with_indent(pp, 1)
			for i in 1:D
				label = get_dimension_label(i)
				print_dimension_info(pp_indented, label, X.box[i][1], X.box[i][2], X.collapsed[i])
			end

			# Remove trailing newline
			remove_trailing_newline(io)
		end
	end
end