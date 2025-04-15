abstract type DomainBaseType <: BrambleType end

"""
	struct Domain{SetType, MarkersType}
		set::SetType
		markers::MarkersType
	end

Structure to represent a domain composed of a [CartesianProduct](@ref) and a set of [Marker](@ref)s.
"""
struct Domain{SetType,MarkersType} <: DomainBaseType
	set::SetType
	markers::MarkersType
end

"""
	struct Marker{F}
		label::Symbol
		identifier::F
	end

Structure to implement markers for a portion of a domain or even boundary conditions. Each [Marker](@ref) is composed of a label represented by a symbol and a [BrambleFunction](@ref) or a symbol representing the identified part of the domain. The function will work as a levelset function and the symbol will correspond to part of the boundary:

  - in 1D `[x₁,x₂]`, :left (x[1]=x₁), :right (x[1]=x₂)
  - in 2D `[x₁,x₂] \\times [y₁,y₂]`, :left (x[1]=x₁), :right (x[1]=x₂), :top (x[2]=y₂), :bottom (x[2]=y₁)
  - in 3D `[x₁,x₂] \\times [y₁,y₂] \\times [z₁,z₂]`, :front (x[1]=x₂), :back (x[1]=x₁, :left (x[2]=y₁), :right (x[2]=y₂), :top (x[3]=z₃), :bottom (x[3]=z₁)
"""
struct Marker{F}
	label::Symbol
	identifier::F
end

MarkerType{F} = Pair{Symbol,F}

@inline label(m::Marker{F}) where F = m.label
@inline identifier(m::Marker{F}) where F = m.identifier

"""
	domain(X::CartesianProduct)
	domain(Ω::CartesianProduct, markers::MarkersType)

Returns a [Domain](@ref) from a [CartesianProduct](@ref), assuming the single [Marker](@ref) `":dirichlet" => x -> 0`. Alternatively, a set of [Marker](@ref) can be passed as an argument.

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
	   m = create_markers(I, :dirichlet => x -> x[1] - 1, :neumann => x -> x[1] - 0);
	   domain(I, m);
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]

Boundary markers: :dirichlet, :neumann
```
"""
function domain(X::CartesianProduct)
	f(x) = zero(eltype(x))
	markers = create_markers(:dirichlet => f)
	return Domain(X, markers)
end

@inline domain(X::CartesianProduct, markers::MType) where MType = Domain{typeof(X),MType}(X, markers)

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
	@assert i in eachindex(set(Ω).data)
	return cartesianproduct(set(Ω).data[i]...)
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
julia> create_markers(I, :dirichlet => x -> x[1] - 1, :neumann => x -> x[2] - 0)

```
"""
@inline function create_markers(embedder, marker_pairs::Vararg{Pair})
	return tuple((Marker(p.first, process_identifier(embedder, p.second)) for p in marker_pairs)...)
end

@inline process_identifier(embedder, identifier::Function) = embed_function(embedder, identifier)
@inline process_identifier(_, identifier::Symbol) = identifier

_process_tuple_recursive(embedder, current_tuple::Tuple{}) = ()

function _process_tuple_recursive(embedder, current_tuple::Tuple)
	first_element = first(current_tuple)
	rest_of_tuple = Base.tail(current_tuple)

	processed_first = process_identifier(embedder, first_element)
	processed_rest = _process_tuple_recursive(embedder, rest_of_tuple)

	return (processed_first, processed_rest...)
end

function process_identifier(embedder, identifier::Tuple)
	return _process_tuple_recursive(embedder, identifier)
end

function process_identifier(embedder, identifier::Any)
	@warn "Unhandled identifier type `$(typeof(identifier))` for embedder `$embedder`. Storing as-is."
	return identifier
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

Returns a generator with the [Marker](@ref)'s identifiers ([BrambleFunction](@ref), Symbol or Tuples of the previous) associated with [Domain](@ref) `Ω`.
"""
@inline marker_identifiers(Ω::Domain) = (p.f for p in Ω.markers)