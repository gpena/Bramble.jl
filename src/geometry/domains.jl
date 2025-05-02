"""
	struct Marker{F}
		label::Symbol
		identifier::F
	end

Structure to implement markers for a portion of a domain or even boundary conditions. Each [Marker](@ref) is composed of a label represented by a Symbol, a [BrambleFunction](@ref) or a Symbol representing an identified part of the boundary (see [get_boundary_symbols](@ref)). The function works as a levelset function, returning True if the point verifies the conditions and false otherwise.
"""
struct Marker{F}
	label::Symbol
	identifier::F
end

MarkerPair{F} = Pair{Symbol,F}

@inline label(m::Marker{F}) where F = m.label
@inline identifier(m::Marker{F}) where F = m.identifier

"""
	struct DomainMarkers{BFType}
		symbols::Set{Marker{Symbol}}
		tuples::Set{Marker{Set{Symbol}}}
		conditions::Set{Marker{BFType}}
	end

The `DomainMarkers` struct represents a collection of markers associated with a domain, categorizing them based on how they identify parts of the domain or its boundary. Each marker is identified as a `Symbol` and a set of [Marker](@ref).
"""
struct DomainMarkers{BFType} <: BrambleType
	symbols::Set{Marker{Symbol}}
	tuples::Set{Marker{Set{Symbol}}}
	conditions::Set{Marker{BFType}}
end

@inline symbols(domain_markers::DomainMarkers) = domain_markers.symbols
@inline tuples(domain_markers::DomainMarkers) = domain_markers.tuples
@inline conditions(domain_markers::DomainMarkers) = domain_markers.conditions

@inline function label_identifiers(domain_markers::DomainMarkers)
	@unpack symbols, tuples, conditions = domain_markers
	return (label(marker)::Symbol for marker in Iterators.flatten((symbols, tuples, conditions)))
end

@inline label_symbols(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in symbols(domain_markers))
@inline label_tuples(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in tuples(domain_markers))
@inline label_conditions(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in conditions(domain_markers))

"""
	create_markers(X::CartesianProduct, pairs...)

Converts several `Pair{Symbol,F}` (:symbol => key), where key is Symbol (of a Tuple of these) or a Function, to a [DomainMarkers](@ref). This is to be passed in the construction of a [Domain](@ref).

# Example

```@example
julia> create_markers(I,
					  :left_boundary => :left,                         # Symbol ID
					  :corners => (:top, :right),                      # Tuple ID
					  :all_boundary => (:top, :right, :left, :bottom), # Tuple ID
					  :internal => x -> 0 <= x[1] - 0.5 <= 1)          # Function ID
Markers: left_boundary, corners, all_boundary, internal
```
"""
@inline create_markers(set::CartesianProduct{1,T}, pairs...) where T = create_markers(T, set, pairs...)
@inline create_markers(set::CartesianProduct{D,T}, pairs...) where {D,T} = create_markers(NTuple{D,T}, set, pairs...)

@inline function create_markers(::Type{CoType}, embedder::CartesianProduct{D,T}, pairs...) where {D,T,CoType}
	symbols = pairs_to_set(Symbol, Symbol, embedder, pairs...)
	tuples = pairs_to_set(Tuple, Set{Symbol}, embedder, pairs...)
	conditions = pairs_to_set(Function, BrambleFunction{CoType,false,Bool}, embedder, pairs...)

	return DomainMarkers(symbols, tuples, conditions)
end

function pairs_to_set(::Type{InputType}, ::Type{OutputType}, embedder, pairs...) where {InputType,OutputType}
	MarkerOutput = Marker{OutputType}
	generator = (Marker(first(pair), process_identifier(embedder, last(pair)))::MarkerOutput for pair in pairs if last(pair) isa InputType)
	return Set{MarkerOutput}(generator)
end

@inline process_identifier(embedder::CartesianProduct, identifier::F) where F<:Function = _embed_notime(embedder, identifier, CoType = Bool)
@inline process_identifier(_::CartesianProduct, identifier::Symbol) = identifier
@inline process_identifier(_::CartesianProduct, identifier::NTuple{N,Symbol}) where N = Set(identifier)
@inline function process_identifier(_, identifier)
	error("Invalid identifier type for create_markers: $(typeof(identifier)). Expected Symbol, Function, or Tuple containing only Symbols.")
end

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

function Base.show(io::IO, markers::DomainMarkers)
	fields = ("Markers")
	mlength = max_length_fields(fields)

	labels = collect(label_identifiers(markers))
	labels_styled_combined = color_markers(labels)

	final_output = style_field("Markers", labels_styled_combined, max_length = mlength)
	print(io, final_output)
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