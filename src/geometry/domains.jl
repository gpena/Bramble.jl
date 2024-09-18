"""
	struct Marker{F<:Function}
		label::String
		f::F
	end

Structure to implement markers for the boundary or interior of a domain.
Each marker is composed of a label and a levelset function that identifies a portion of the domain.

# Fields
  - `label::String`, label associated with the marker.
  - `f::F``, function associated with the marker.
"""
struct Marker{F<:Function}
	label::String
	f::F
end

MarkerType{F} = Pair{String,F}

"""
	DomainBaseType

An abstract type for representing a domain.
"""
abstract type DomainBaseType <: BrambleType end

"""
	struct Domain{SetType, MarkersType}
		set::SetType
		markers::MarkersType
	end

Structure to represent a domain composed of cartesian set and a set of markers.
"""
struct Domain{SetType,MarkersType} <: DomainBaseType
	set::SetType
	markers::MarkersType
end

#const __zerofunc(x) = zero(eltype(x))

"""
$(SIGNATURES)
	Domain(X::CartesianProduct)

Creates a domain from a cartesian product. The default marker used 
has label "Dirichlet" and the zero function.

# Example
```
julia> Domain(Interval(0,1))
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]

Boundary markers: Dirichlet
```
"""
Domain(X::CartesianProduct) = Domain(X, (Marker("Dirichlet", x -> zero(eltype(x))),))



@inline Domain(X::CartesianProduct, markers::MarkersType) where MarkersType = Domain{typeof(X), MarkersType}(X, markers)

"""
$(SIGNATURES)
	set(domain)

Returns the set associated with a domain.
"""
@inline set(domain::Domain) = domain.set

"""
	dim(domain)

Returns the dimension of a domain.

# Example
```
julia> I = Interval(0.0, 1.0);
	   dim(Domain(I × I));
2
```
"""
@inline dim(domain::DomainBaseType) = dim(set(domain))
@inline dim(_::Type{<:Domain{SetType}}) where SetType = dim(SetType)

"""
	eltype(domain)

Returns the element type of a domain.

# Example
```
julia> eltype(Domain(I × I))
Float64```
```
"""
@inline eltype(domain::Domain) = eltype(set(domain))
@inline eltype(_::Type{<:Domain{SetType}}) where SetType = eltype(SetType)

"""
	projection(domain, i)

Returns the `i`-th projection of a domain. For example, `projection(Domain(I × I), 1)`
will return `I`.
"""
@inline projection(domain::Domain, i::Int) = CartesianProduct(set(domain).data[i]...)

function show(io::IO, domain::Domain)
	l = join(labels(domain), ", ")

	show(io, set(domain))
	print(io, "\n\nBoundary markers: $l")
end

"""
	markers(p...)

Converts pairs of "label" => func to domain markers to be accepted in the Domain constructor.

# Example
```
julia> markers( "Dirichlet" => (x -> x-1), "Neumann" => (x -> x-0) )
```
"""
@inline @generated function markers(ps::MarkerType...)
	D = length(ps) 

	tuple_expr = Expr(:tuple)
	for i in 1:D
		push!(tuple_expr.args, :(Marker(ps[$i]...)))
	end

	return tuple_expr
end

"""
	markers(domain)

Returns a generator with the markers associated with a domain.
"""
@inline markers(domain::Domain) = domain.markers

"""
	labels(domain)

Returns a generator with the labels of the markers associated with a domain.

"""
@inline labels(domain::Domain) = (p.label for p in domain.markers)

"""
	markerfuncs(domain)

Returns a generator with the marker levelset functions associated with a domain.
"""
@inline markerfuncs(domain::Domain) = (p.f for p in domain.markers)