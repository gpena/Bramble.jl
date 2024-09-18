"""
	struct Marker{F<:Function}

Represents a marker function.

# Fields
  - `label::String`, the label associated with the marker.
  -`f::F``, The function associated with the marker.
"""
struct Marker{F<:Function}
	label::String
	f::F
end

"""
	MarkerType{F}

Represents a type that can be used to represent a marker function. This is
a `Pair{String, F}` where the first element of the pair is the label
associated with the marker and the second is the function associated
with the marker.
"""
MarkerType{F} = Pair{String,F}

"""
# DomainBaseType

An abstract type representing a domain with a set and a set of markers.
"""
abstract type DomainBaseType <: BrambleType end

"""
	struct Domain{SetType, MarkersType}

Represents a domain with a set and a set of markers.

	Inputs:
	set::SetType, the set represented by the domain.
	markers::MarkersType, the markers associated with the domain.
"""
struct Domain{SetType,MarkersType} <: DomainBaseType
	set::SetType
	markers::MarkersType
end

#const __zerofunc(x) = zero(eltype(x))

"""
The Domain type represents a domain with a set and a set of markers.

The default marker for a domain is a "Dirichlet" marker with a function `f(x) = 0`.
"""
Domain(X::CartesianProduct) = Domain(X, (Marker("Dirichlet", x -> zero(eltype(x))),))

"""
	set(domain)

Returns the set associated with a domain.
"""
@inline set(domain::Domain) = domain.set

"""
	dim(domain)

Returns the dimension of a domain.

# Example

```jldoctest
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

```jldoctest
julia> eltype(Domain(I × I))
Float64```
```
"""
@inline eltype(domain::Domain) = eltype(set(domain))
@inline eltype(_::Type{<:Domain{SetType}}) where SetType = eltype(SetType)

"""
	projection(domain, i)

Returns the `i`th projection of a domain. For example, `projection(Domain(I × I), 1)`
will return `I`.
"""
@inline projection(domain::Domain, i::Int) = CartesianProduct(set(domain).data[i]...)

"""
	show(io, domain)

Prints a domain to `io`. The output will show the set of the domain, and
the labels of the markers.
"""
function show(io::IO, domain::Domain)
	l = join(labels(domain), ", ")

	show(io, set(domain))
	print(io, "\n\nBoundary markers: $l")
end

"""
	markers(p...)

Converts pairs of "label" => func to domain markers.

# Example
```
markers( "Dirichlet" => (x -> x-1), "Neumann" => (x -> x-0) )
```
"""
@inline @generated function markers(ps::MarkerType...)
	D = length(ps) # Get the number of arguments

	# Generate an expression to create a tuple
	tuple_expr = Expr(:tuple)
	for i in 1:D
		push!(tuple_expr.args, :(Marker(ps[$i]...)))
	end

	return tuple_expr
end

"""
	markers(domain)

Returns the markers associated with a domain.
"""
@inline markers(domain::Domain) = domain.markers

"""
	labels(domain)

Returns the labels of the markers associated with a domain.
"""
@inline labels(domain::Domain) = (p.label for p in domain.markers)

"""
	markerfuncs(domain)

Returns the marker functions associated with a domain.
"""
@inline markerfuncs(domain::Domain) = (p.f for p in domain.markers)

#=
function make_marker(symb::Expr, ex)
	return :(DomainMarker(Symbol($symb), $ex))
end

function make_marker(symb::Symbol, ex)
	return :(DomainMarker($symb, $ex))
end

macro markers(ex::Expr)
	@show ex
	res = Expr(:tuple)
	if ex.head == :tuple
		for i in 1:length(ex.args)
			str = Expr(:call, getindex, ex.args[i], 1)
			symb = :(Symbol($str))
			_func = Expr(:call, getindex, ex.args[i], 2)

			marker = make_marker(symb, esc(_func))
			push!(res.args, :($marker))
		end
	else
		_marker = ex.args[2]
		_func = ex.args[3]
		symb = :(Symbol($_marker))
		marker = make_marker((symb), (esc(_func)))
		push!(res.args, :($marker))
	end

	return :(Set($res))
end
=#
#=
macro Marker(ex::Expr)
	@show dump(ex)
	@show ex.args[1]
	_marker = Expr(:call, getindex, ex, 1)
	_func = Expr(:call, getindex, ex, 2)
	symb = :(Symbol($_marker))
	#res = Expr(:tuple)
	#marker = make_marker2((symb), (esc(_func)))

	#push!(res, :($marker))

	return make_marker2((symb), (esc(_func)))
end
=#

#label2marker(str::String) = str, Symbol(str_domain_marker*str)
#label2marker(symb::Symbol) = label2marker(String(symb))
#=
function marker2label(symb::Symbol) 
	str = String(symb)
	range = findfirst(str_domain_marker, str)
	return str[(range[end]+1):end]
end
=#
