abstract type DomainBaseType <: BrambleType end

"""
	struct Domain{SetType, MarkersType}
		set::SetType
		markers::MarkersType
	end

Structure to represent a domain composed of a [CartesianProduct](@ref) and a tuple of [Marker](@ref)s.
"""
struct Domain{SetType,MarkersType} <: DomainBaseType
	set::SetType
	markers::MarkersType
end

"""
	struct Marker{F}
		label::Symbol
		identifier::F
		boundary_part::Symbol
	end

Structure to implement markers for a portion of a domain or even boundary conditions. Each [Marker](@ref) is composed of a label represented by a Symbol, a [BrambleFunction](@ref) or a Symbol representing an identified part of the boundary (see [get_boundary_symbols](@ref)). The function works as a levelset function, returning True if the point verifies the conditions and false otherwise.
"""
struct Marker{F}
	label::Symbol
	identifier::F
end

MarkerType{F} = Pair{Symbol,F}

@inline label(m::Marker{F}) where F = m.label
@inline identifier(m::Marker{F}) where F = m.identifier

"""
	get_boundary_symbols(X::CartesianProduct)

Returns a tuple of standard boundary symbols for a [CartesianProduct](@ref).
"""
@inline get_boundary_symbols(X::CartesianProduct) = get_boundary_symbols(Val(dim(X)))

@inline get_boundary_symbols(::Val{1}) = (:left, :right)
@inline get_boundary_symbols(::Val{2}) = (:bottom, :top, :left, :right)
@inline get_boundary_symbols(::Val{3}) = (:bottom, :top, :back, :front, :left, :right)

"""
	domain(X::CartesianProduct)
	domain(X::CartesianProduct, markers::MarkersType)

Returns a [Domain](@ref) from a [CartesianProduct](@ref), assuming a single [Marker](@ref) with the label `:dirichlet` that marks the whole boundary of X. Alternatively, a set of [Marker](@ref) can be passed as an argument.

# Example

```@example
julia> domain(interval(0, 1))
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]

Boundary markers: :dirichlet
```

```@example
julia> I = interval(0, 1);
	   m = create_markers(I, :dirichlet => x -> x[1] - 1 == 0, :dirichlet_alternative => :right);
	   domain(I, m);
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]

Boundary markers: :dirichlet, :dirichlet_alternative
```

```@example
julia> I = interval(0, 1) × interval(2, 3);
	   m = create_markers(I, :dirichlet => x -> x[1] - 1 == 0, :dirichlet_alternative => :right, :neuman => (:bottom, :top));
	   domain(I, m);
Type: Float64 
 Dim: 2 
 Set: [0.0, 1.0] × [2.0, 3.0]

Boundary markers: :dirichlet, :dirichlet_alternative, :neuman
```
"""
function domain(X::CartesianProduct{D}) where D
	markers = create_markers(X, :dirichlet => get_boundary_symbols(X))
	return Domain(X, markers)
end

@inline domain(X::CartesianProduct, markers::MType) where MType = Domain{typeof(X),MType}(X, markers)

"""
	domain(X::CartesianProduct, marker_pairs::Vararg{Pair})

Constructs a [Domain](@ref) from a [CartesianProduct](@ref) and a variable number of marker pairs.
This is a convenience method that calls create_markers internally.

# Example

```@example
julia> dom = domain(I, :dirichlet => :left, :source => f1)

```
"""
function domain(X::CartesianProduct, marker_pairs::Vararg{Pair})
	markers_tuple = create_markers(X, marker_pairs...)
	return domain(X, markers_tuple)
end

"""
	set(Ω::Domain)

Returns the [CartesianProduct](@ref) associated with the [Domain](@ref) `Ω`.
"""
@inline set(Ω::Domain) = Ω.set

"""
	dim(Ω::Domain)
	dim(::Type{<:Domain})

Returns the topological dimension of the [Domain](@ref) `Ω`.

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
@inline eltype(Ω::Domain) = eltype(set(Ω))
@inline eltype(::Type{<:Domain{SetType}}) where SetType = eltype(SetType)

"""
	projection(Ω::Domain, i)

Returns the `i`-th `1`-dimensional [CartesianProduct](@ref) of the [set](@ref set(Ω::Domain)) associated with [Domain](@ref) `Ω`.

For example, `projection(domain(I × I), 1)` will return `I`.
"""
@inline function projection(Ω::Domain, i)
	_set = set(Ω)

	@assert i in eachindex(bounds(_set))
	return cartesianproduct(_set(i)...)
end

function show(io::IO, Ω::Domain)
	l = join(labels(Ω), ", ")

	show(io, set(Ω))
	print(io, "\n\nMarkers: $l")
end

"""
	create_markers(X::CartesianProduct, marker_pairs::Vararg{Pair})

Converts several `Pair{Symbol,F}` (:symbol => arg), where arg is Symbol, Function or a tuple of these types, to [Marker](@ref)s. These are to be passed in the construction of a [Domain](@ref)..

# Example

```@example
julia> create_markers(I,
					  :boundary => :left,                        # Symbol ID
					  :corners => (:top, :right),                # Tuple{Symbol, Symbol} ID
					  :internal => x -> 0 <= x[1] - 0.5 <= 1)    # Function ID

```
"""
@inline function create_markers(embedder::CartesianProduct, marker_pairs::Vararg{Pair})
	return tuple((Marker(p.first, process_identifier(embedder, p.second)) for p in marker_pairs)...)
end

@inline process_identifier(embedder::CartesianProduct, identifier::F) where F<:Function = embed_function(embedder, identifier)
@inline process_identifier(_::CartesianProduct, identifier::Symbol) = identifier
@inline process_identifier(_::CartesianProduct, identifier::Tuple{Vararg{Symbol}}) = identifier

# Method to catch invalid identifier types
function process_identifier(_, identifier)
	error("Invalid identifier type for create_markers: $(typeof(identifier)). Expected Symbol, Function, or Tuple containing only Symbols.")
end

"""
	markers(Ω::Domain)

Returns a generator with the [Marker](@ref)s associated with [Domain](@ref) `Ω`.
"""
@inline markers(Ω::Domain) = Ω.markers

"""
	labels(Ω::Domain)

Returns a generator with the labels of the [Marker](@ref)s associated with [Domain](@ref) `Ω`.
"""
@inline labels(Ω::Domain) = (p.label for p in Ω.markers)

"""
	marker_identifiers(Ω::Domain)

Returns a generator with the [Marker](@ref)'s identifiers ([BrambleFunction](@ref), Symbol or Tuples of Symbols) associated with [Domain](@ref) `Ω`.
"""
@inline marker_identifiers(Ω::Domain) = (p.identifier for p in Ω.markers)

@inline marker_symbols(Ω::Domain) = (p.identifier for p in Ω.markers if p.identifier isa Symbol)
@inline marker_tuples(Ω::Domain) = (p.identifier for p in Ω.markers if p.identifier isa Tuple)
@inline marker_conditions(Ω::Domain) = (p.identifier for p in Ω.markers if p.identifier isa BrambleFunction)