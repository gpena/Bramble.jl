abstract type DomainBaseType <: BrambleType end

"""
	struct Domain{SetType, MarkersType}
		set::SetType
		markers::MarkersType
	end

Structure to represent a domain composed of a [CartesianProduct](@ref) and a [DomainMarkers](@ref).
"""
struct Domain{SetType,MarkersType} <: DomainBaseType
	set::SetType
	markers::MarkersType
end

"""
	markers(Ω::Domain)

Returns a generator with the [DomainMarkers](@ref) associated with [Domain](@ref) `Ω`.
"""
@inline markers(Ω::Domain) = Ω.markers

"""
	labels(Ω::Domain)

Returns a generator with the labels of the [DomainMarkers](@ref] associated with [Domain](@ref) `Ω`.
"""
@inline function labels(Ω::Domain)
	@unpack symbols, tuples, conditions = markers(Ω)
	return (label(marker)::Symbol for marker in Iterators.flatten((symbols, tuples, conditions)))
end

"""
	marker_identifiers(Ω::Domain)

Returns a generator with the [DomainMarkers](@ref)'s identifiers ([BrambleFunction](@ref), `Symbol` or `Tuple` of `Symbol`) associated with [Domain](@ref) `Ω`.
"""
@inline function marker_identifiers(Ω::Domain)
	@unpack symbols, tuples, conditions = markers(Ω)
	return (identifier(marker) for marker in Iterators.flatten((symbols, tuples, conditions)))
end

@inline function marker_symbols(Ω::Domain)
	@unpack symbols = markers(Ω)
	return (identifier(marker) for marker in symbols)
end

@inline function marker_tuples(Ω::Domain)
	@unpack symbols, tuples = markers(Ω)
	return (identifier(marker) for marker in tuples)
end

@inline function marker_conditions(Ω::Domain)
	@unpack symbols, tuples, conditions = markers(Ω)
	return (identifier(marker) for marker in conditions)
end

"""
	label_identifiers(Ω::Domain)

Returns a generator with all labels on [DomainMarkers](@ref).
"""
@inline label_identifiers(Ω::Domain) = label_identifiers(markers(Ω))

"""
	label_symbols(Ω::Domain)

Returns a generator with the labels of the symbols on [DomainMarkers](@ref).
"""
@inline label_symbols(Ω::Domain) = label_symbols(markers(Ω))

"""
	label_tuples(Ω::Domain)

Returns a generator with the labels of the tuples on [DomainMarkers](@ref).
"""
@inline label_tuples(Ω::Domain) = label_tuples(markers(Ω))

"""
	label_conditions(Ω::Domain)

Returns a generator with the labels of the conditions on [DomainMarkers](@ref).
"""
@inline label_conditions(Ω::Domain) = label_conditions(markers(Ω))

"""
	domain(X::CartesianProduct, [markers...])

Returns a [Domain](@ref) from a [CartesianProduct](@ref), assuming a single [Marker](@ref) with the label `:boundary` that marks the whole boundary of X. Alternatively, a list of [Marker](@ref) can be passed as argument in the form of `:symbol => key` (see examples and [markers](@ref)).
"""
@inline domain(X::CartesianProduct) = Domain(X, markers(X, :boundary => get_boundary_symbols(X)))
@inline domain(X::CartesianProduct, markers::DomainMarkers) = Domain(X, markers)
@inline domain(X::CartesianProduct, pairs...) = domain(X, markers(X, pairs...))

"""
	set(Ω::Domain)

Returns the [CartesianProduct](@ref) associated with the [Domain](@ref) `Ω`.
"""
@inline set(Ω::Domain) = Ω.set

"""
	dim(Ω::Domain)
	dim(::Type{<:Domain})

Returns the dimension of the space where [Domain](@ref) `Ω` is embedded.

# Example

```julia
I = interval(0.0, 1.0);
dim(domain(I × I))
2
```
"""
@inline dim(Ω::Domain) = dim(set(Ω))
@inline dim(::Type{<:Domain{SetType}}) where SetType = dim(SetType)

"""
	topo_dim(Ω::Domain)

Returns the topological dimension [Domain](@ref) `Ω`.
"""
@inline topo_dim(Ω::Domain) = topo_dim(set(Ω))

"""
	eltype(Ω::Domain)
	eltype(::Type{<:Domain{SetType}})

Returns the type of the bounds defining [Domain](@ref) `Ω`.

# Example

```julia
I = interval(0.0, 1.0);
eltype(domain(I × I))
Float64
```
"""
@inline eltype(Ω::Domain) = eltype(set(Ω))
@inline eltype(::Type{<:Domain{SetType}}) where SetType = eltype(SetType)

"""
	projection(Ω::Domain, i)

Returns the `i`-th [CartesianProduct](@ref) of the set associated with [Domain](@ref) `Ω`.

For example, `projection(domain(I × I), 1)` will return `I`.
"""
@inline function projection(Ω::Domain, i)
	@unpack box = set(Ω)
	return cartesian_product(box[i]...)
end

"""
	get_boundary_symbols(X::CartesianProduct)

Returns a tuple of default boundary symbols for a [CartesianProduct](@ref).

  - in 1D `[x₁,x₂]`, :left (x=x₁), :right (x=x₂)
  - in 2D `[x₁,x₂] \\times [y₁,y₂]`, :left (x=x₁), :right (x=x₂), :top (y=y₂), :bottom (y=y₁)
  - in 3D `[x₁,x₂] \\times [y₁,y₂] \\times [z₁,z₂]`, :front (x=x₂), :back (x=x₁), :left (y=y₁), :right (y=y₂), :top (z=z₃), :bottom (z=z₁)
"""
@inline get_boundary_symbols(::CartesianProduct{1}) = (:left, :right)
@inline get_boundary_symbols(::CartesianProduct{2}) = (:bottom, :top, :left, :right)
@inline get_boundary_symbols(::CartesianProduct{3}) = (:bottom, :top, :back, :front, :left, :right)