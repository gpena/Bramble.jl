"""
	DomainBaseType

An abstract type for representing domains.
"""
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
	struct Marker{F<:Function}
		label::String
		f::F
	end

Structure to implement markers for a portion of a domain. Each [Marker](@ref) is composed of a label and a levelset function that identifies a portion of the domain.
"""
struct Marker{F<:Function}
	label::String
	f::F
end

MarkerType{F} = Pair{String,F}

"""
$(SIGNATURES)
Creates a [Domain](@ref) from a [CartesianProduct](@ref) assuming the single [Marker](@ref) `"Dirichlet" => x -> zero(eltype(x))`.

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

"""
$(SIGNATURES)
Creates a [Domain](@ref) from a [CartesianProduct](@ref) assuming the single [Marker](@ref) `"Dirichlet" => x -> zero(eltype(x))`.

# Example
```
julia> m = markers( "Dirichlet" => (x -> x-1), "Neumann" => (x -> x-0) ); Domain(Interval(0,1), m)
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]

Boundary markers: Dirichlet, Neumann
```
"""
@inline Domain(X::CartesianProduct, markers::MarkersType) where MarkersType = Domain{typeof(X), MarkersType}(X, markers)

"""
$(SIGNATURES)
Returns the [CartesianProduct](@ref) associated with a [Domain](@ref) `X`.
"""
@inline set(X::Domain) = X.set

"""
$(SIGNATURES)

Returns the topological dimension of a [Domain](@ref) `X`.

# Example
```
julia> I = Interval(0.0, 1.0); dim(Domain(I × I))
2
```
"""
@inline dim(X::DomainBaseType) = dim(set(X))
@inline dim(X::Type{<:Domain{SetType}}) where SetType = dim(SetType)

"""
$(SIGNATURES)

Returns the element type of a [Domain](@ref) `X`.

# Example
```
julia> eltype(Domain(I × I))
Float64
```
"""
@inline eltype(X::Domain) = eltype(set(X))
@inline eltype(X::Type{<:Domain{SetType}}) where SetType = eltype(SetType)

"""
$(SIGNATURES)
Returns the [CartesianProduct](@ref) of the `i`-th projection of the set of the [Domain](@ref) `X`. 

For example, `projection(Domain(I × I), 1)` will return `I`.
"""
@inline projection(X::Domain, i::Int) = CartesianProduct(set(X).data[i]...)

function show(io::IO, X::Domain)
	l = join(labels(X), ", ")

	show(io, set(X))
	print(io, "\n\nMarkers: $l")
end

"""
$(SIGNATURES)
Converts several `Pair{String,F}` ("label" => func) to domain [Marker](@ref)s to be passed in the construction of a [Domain](@ref) `X`.

# Example
```
julia> create_markers( "Dirichlet" => (x -> x-1), "Neumann" => (x -> x-0) )
```
"""
@inline @generated function create_markers(m::MarkerType...)
	D = length(m) 

	tuple_expr = Expr(:tuple)
	for i in 1:D
		push!(tuple_expr.args, :(Marker(m[$i]...)))
	end

	return tuple_expr
end

"""
$(SIGNATURES)
Returns a generator with the [Marker](@ref)s associated with a [Domain](@ref) `X`.
"""
@inline markers(X::Domain) = X.markers

"""
	labels(X)

Returns a generator with the labels of the [Marker](@ref)s associated with a [Domain](@ref) `X`.

"""
@inline labels(X::Domain) = (p.label for p in X.markers)

"""
$(SIGNATURES)
Returns a generator with the [Marker](@ref)s levelset functions associated with a [Domain](@ref) `X`.
"""
@inline marker_funcs(X::Domain) = (p.f for p in X.markers)