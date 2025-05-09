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
#=
function Base.show(io::IO, markers::DomainMarkers)
	fields = ("Markers")
	mlength = max_length_fields(fields)

	labels = collect(label_identifiers(markers))
	labels_styled_combined = color_markers(labels)

	final_output = style_field("Markers", labels_styled_combined, max_length = mlength)
	print(io, final_output)
	return nothing
end=#