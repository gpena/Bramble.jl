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
	Domain(Ω::CartesianProduct)

Creates a [Domain](@ref) from a [CartesianProduct](@ref) assuming the single [Marker](@ref) `"Dirichlet" => x -> zero(eltype(x))`.

# Example

```
julia> domain(Interval(0,1))
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]

Boundary markers: Dirichlet
```
"""
@inline domain(Ω::CartesianProduct) = Domain(Ω, (Marker("Dirichlet", x -> zero(eltype(x))),))

"""
	Domain(Ω::CartesianProduct, markers::MarkersType)

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
@inline domain(Ω::CartesianProduct, markers::MarkersType) where MarkersType = Domain{typeof(Ω),MarkersType}(Ω, markers)

"""
	set(Ω::Domain)

Returns the [CartesianProduct](@ref) associated with a [Domain](@ref) `Ω`.
"""
@inline set(Ω::Domain) = Ω.set

"""
	dim(Ω::DomainBaseType)

Returns the topological dimension of a [Domain](@ref) `Ω`.

# Example

```
julia> I = Interval(0.0, 1.0); dim(Domain(I × I))
2
```
"""
@inline dim(Ω::Domain) = dim(set(Ω))
@inline dim(::Type{<:Domain{SetType}}) where SetType = dim(SetType)

"""
	eltype(Ω::Domain)

Returns the element type of a [Domain](@ref) `Ω`.

# Example

```
julia> eltype(Domain(I × I))
Float64
```
"""
@inline eltype(Ω::Domain) = eltype(set(Ω))
@inline eltype(::Type{<:Domain{SetType}}) where SetType = eltype(SetType)

"""
	projection(Ω::Domain, i::Int)

Returns the [CartesianProduct](@ref) of the `i`-th projection of the set of the [Domain](@ref) `Ω`.

For example, `projection(Domain(I × I), 1)` will return `I`.
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
	create_markers(m::MarkerType...)

Converts several `Pair{String,F}` ("label" => func) to domain [Marker](@ref)s to be passed in the construction of a [Domain](@ref) `Ω`.

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
	markers(Ω::Domain)

Returns a generator with the [Marker](@ref)s associated with a [Domain](@ref) `Ω`.
"""
@inline markers(Ω::Domain) = Ω.markers

"""
	labels(Ω::Domain)

Returns a generator with the labels of the [Marker](@ref)s associated with a [Domain](@ref) `Ω`.
"""
@inline labels(Ω::Domain) = (p.label for p in Ω.markers)

"""
	marker_funcs(Ω::Domain)

Returns a generator with the [Marker](@ref)s levelset functions associated with a [Domain](@ref) `Ω`.
"""
@inline marker_funcs(Ω::Domain) = (p.f for p in Ω.markers)
