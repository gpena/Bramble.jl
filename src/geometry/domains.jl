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
		f::F
	end

Structure to implement markers for a portion of a domain or even boundary conditions. Each [Marker](@ref) is composed of a symbol and a [BrambleFunction](@ref).
"""
struct Marker{F}
	label::Symbol
	f::F
end

MarkerType{F} = Pair{Symbol,F}

"""
	domain(X::CartesianProduct)
	domain(Ω::CartesianProduct, markers::MarkersType)

Returns a [Domain](@ref) from a [CartesianProduct](@ref), assuming the single [Marker](@ref) `":dirichlet" => x -> 0`. Alternatively, a set of [Marker](@ref) can be passed as an argument.

# Example

```
julia> domain(interval(0,1))
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]

Boundary markers: :dirichlet
```

```
julia> I = interval(0,1); m = markers( :dirichlet => ↪(I, x->x[1]-1)), :neumann => ↪(I, x->x[1]-0)); domain(interval(0,1), m)
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]

Boundary markers: :dirichlet, :neumann
```
"""
@inline domain(X::CartesianProduct) = Domain(X, (Marker(:Dirichlet, x -> zero(eltype(x))),))
@inline domain(X::CartesianProduct, markers::MType) where MType = Domain{typeof(X),MType}(X, markers)

"""
	set(Ω::Domain)

Returns the [CartesianProduct](@ref) associated with the [Domain](@ref) `Ω`.
"""
@inline set(Ω::Domain) = Ω.set

"""
	dim(Ω::Domain)
	dim(::Type{<:Domain{SetType}})

Returns the topological dimension of the [Domain](@ref) `Ω`.

# Example

```
julia> I = interval(0.0, 1.0); dim(domain(I × I))
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

```
julia> I = interval(0.0, 1.0); eltype(domain(I × I))
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
	create_markers(m::MarkerType...)

Converts several `Pair{Symbol,F}` (:symbol => func) to [Marker](@ref)s. These are to be passed in the construction of a [Domain](@ref). The functions need to be defined as [BrambleFunction](@ref)s.

# Example

```
julia> create_markers( :dirichlet => ↪(X, x -> x[1]-1), :neumann => ↪(X, x -> x[2]-0) )
```
"""
@inline @generated function create_markers(m::MarkerType{BrambleFunction{D,T,H}}...) where {D,T,H}
	nPairs = length(m)

	tuple_expr = Expr(:tuple)
	for i in 1:nPairs
		push!(tuple_expr.args, :(Marker(m[$i]...)))
	end

	return tuple_expr
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
	marker_funcs(Ω::Domain)

Returns a generator with the [Marker](@ref)'s [BrambleFunction](@ref)s associated with [Domain](@ref) `Ω`.
"""
@inline marker_funcs(Ω::Domain) = (p.f for p in Ω.markers)

"""
	embed(f, Ω::Domain)
	embed(f::BrambleFunction, Ω::Domain)

Returns a [BrambleFunction](@ref). The dimension of [Domain](@ref) `Ω` is used to caracterize the types in the returning [BrambleFunction](@ref). The notation `↪` can also be used for this operation.

# Example

julia> Ω = domain(interval(0,1) × interval(0,1)); f = embed(Ω, x -> x[1]*x[2]+1);
"""
function embed(Ω::Domain, f)
	D = dim(Ω)
	T = eltype(Ω)
	wrapped_f_tuple = FunctionWrapper{T,Tuple{NTuple{D,T}}}(f)
	wrapped_f_cartesian = FunctionWrapper{T,Tuple{CartesianIndex{D}}}(zero)

	return BrambleFunction{D,T,false}(wrapped_f_tuple, wrapped_f_cartesian)
end

@inline embed(Ω::Domain, f::BrambleFunction) = f

"""
	↪(Ω::Domain, f)

Alias for [embed](@ref embed(Ω::Domain, f)).
"""
@inline ↪(Ω::Domain, f) = embed(Ω, f)