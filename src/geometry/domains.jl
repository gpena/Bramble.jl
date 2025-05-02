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

Returns a generator with the [DomainMarkers](@ref] associated with [Domain](@ref) `Ω`.
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
	@unpack symbols, tuples, conditions = markers(Ω)
	return (identifier(marker) for marker in symbols)
end

@inline function marker_tuples(Ω::Domain)
	@unpack symbols, tuples, conditions = markers(Ω)
	return (identifier(marker) for marker in tuples)
end

@inline function marker_conditions(Ω::Domain)
	@unpack symbols, tuples, conditions = markers(Ω)
	return (identifier(marker) for marker in conditions)
end

"""
	marker_identifiers(Ω::Domain)

Returns a generator with the labels of the [DomainMarkers](@ref)'s identifiers ([BrambleFunction](@ref), `Symbol` or `Tuple` of `Symbol`) associated with [Domain](@ref) `Ω`.
"""
@inline label_identifiers(Ω::Domain) = label_identifiers(markers(Ω))
@inline label_symbols(Ω::Domain) = label_symbols(markers(Ω))
@inline label_tuples(Ω::Domain) = label_tuples(markers(Ω))
@inline label_conditions(Ω::Domain) = label_conditions(markers(Ω))

"""
	domain(X::CartesianProduct)
	domain(X::CartesianProduct, markers...)

Returns a [Domain](@ref) from a [CartesianProduct](@ref), assuming a single [Marker](@ref) with the label `:dirichlet` that marks the whole boundary of X. Alternatively, a list of [Marker](@ref) can be passed as argument in the form of `:symbol => key` (see examples and [create_markers](@ref)).

# Example

```@example
julia> domain(interval(0, 1))
 Domain
   Type: Float64
  Space: ℝ
	Dim: 1
	Set: [0.0, 1.0]
Markers: dirichlet
```

```@example
julia> I = interval(0, 1);
	   m = create_markers(I, :dirichlet => x -> x[1] - 1 == 0, :dirichlet_alternative => :right);
	   domain(I, m);
 Domain
   Type: Float64
  Space: ℝ
	Dim: 1
	Set: [0.0, 1.0]
Markers: dirichlet_alternative, dirichlet
```

```@example
julia> I = interval(0, 1) × interval(2, 3);
	   m = create_markers(I, :dirichlet => x -> x[1] - 1 == 0, :dirichlet_alternative => :right, :neuman => (:bottom, :top));
	   domain(I, m);
 Domain
   Type: Float64
  Space: ℝ²
	Dim: 2
	Set: [0.0, 1.0] × [2.0, 3.0]
Markers: dirichlet_alternative, neuman, dirichlet
```
"""
@inline domain(X::CartesianProduct) = Domain(X, create_markers(X, :dirichlet => get_boundary_symbols(X)))
@inline domain(X::CartesianProduct, markers::DomainMarkers) = Domain(X, markers)
@inline domain(X::CartesianProduct, pairs...) = domain(X, create_markers(X, pairs...))

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

```@example
julia> I = interval(0.0, 1.0);
	   dim(domain(I × I));
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

```@example
julia> I = interval(0.0, 1.0);
	   eltype(domain(I × I));
Float64
```
"""
@inline Base.eltype(Ω::Domain) = eltype(set(Ω))
@inline Base.eltype(::Type{<:Domain{SetType}}) where SetType = eltype(SetType)

"""
	projection(Ω::Domain, i)

Returns the `i`-th [CartesianProduct](@ref) of the [set](@ref set(Ω::Domain)) associated with [Domain](@ref) `Ω`.

For example, `projection(domain(I × I), 1)` will return `I`.
"""
@inline function projection(Ω::Domain, i)
	@unpack box = set(Ω)
	@assert i in eachindex(box)
	return cartesianproduct(box[i]...)
end

function Base.show(io::IO, Ω::Domain)
	fields = ("Type", "Dim", "Set", "Markers")
	mlength = max_length_fields(fields)

	title_info = style_title("Domain", max_length = mlength)
	output = style_join(title_info, set_info_only(set(Ω), mlength))
	print(io, output * "\n")

	show(io, markers(Ω))
end

"""
	get_boundary_symbols(X::CartesianProduct)

Returns a tuple of default boundary symbols for a [CartesianProduct](@ref).

  - in 1D `[x₁,x₂]`, :left (x=x₁), :right (x=x₂)
  - in 2D `[x₁,x₂] \\times [y₁,y₂]`, :left (x=x₁), :right (x=x₂), :top (y=y₂), :bottom (y=y₁)
  - in 3D `[x₁,x₂] \\times [y₁,y₂] \\times [z₁,z₂]`, :front (x=x₂), :back (x=x₁), :left (y=y₁), :right (y=y₂), :top (z=z₃), :bottom (z=z₁)
"""
@inline get_boundary_symbols(_::CartesianProduct{1}) = (:left, :right)
@inline get_boundary_symbols(_::CartesianProduct{2}) = (:bottom, :top, :left, :right)
@inline get_boundary_symbols(_::CartesianProduct{3}) = (:bottom, :top, :back, :front, :left, :right)