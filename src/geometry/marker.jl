"""
	struct Marker{F}
		label::Symbol
		identifier::F
	end

Structure to implement markers for a portion of a domain or even boundary conditions. Each [Marker](@ref) is composed of a label represented by a Symbol, a [BrambleFunction](@ref) or a Symbol representing an identified part of the boundary. The function works as a levelset function, returning `True` if the point verifies the conditions and false otherwise.
"""
struct Marker{F}
	label::Symbol
	identifier::F
end

"""
	MarkerPair{F}

A type alias representing a pair between a `Symbol` and a value of type `F`.
Used for marking geometric features with symbolic labels.
"""
const MarkerPair{F} = Pair{Symbol,F}

"""
	label(m::Marker)
	label(m::MarkerPair)

Returns the label of a [Marker](@ref) or the first element (label) of a [MarkerPair](@ref).
"""
@inline label(m::Marker) = m.label
@inline label(m::MarkerPair) = first(m)

"""
	identifier(m::Marker)
	identifier(m::MarkerPair)

Returns the identifier of a [Marker](@ref) or the last element (identifier) of a [MarkerPair](@ref).
"""
@inline identifier(m::Marker) = m.identifier
@inline identifier(m::MarkerPair) = last(m)

"""
	struct DomainMarkers{BFType}
		symbols::Set{Marker{Symbol}}
		tuples::Set{Marker{Set{Symbol}}}
		conditions::Set{Marker{BFType}}
	end

This struct struct represents a collection of markers associated with a domain, categorizing them based on how they identify parts of the domain or its boundary. See [markers](@ref) for more details on building these structs and [get\\_boundary\\_symbols](@ref) for the available predefined symbols.
"""
struct DomainMarkers{BFType} <: BrambleType
	symbols::Set{Marker{Symbol}}
	tuples::Set{Marker{Set{Symbol}}}
	conditions::Set{Marker{BFType}}
end

"""
	symbols(domain_markers::DomainMarkers)

Accessor function for the [DomainMarkers](@ref) type. Returns the `symbols` field of the DomainMarkers.
"""
@inline symbols(domain_markers::DomainMarkers) = domain_markers.symbols

"""
	tuples(domain_markers::DomainMarkers)

Accessor function for the [DomainMarkers](@ref) type. Returns the `tuples` field of the DomainMarkers.
"""
@inline tuples(domain_markers::DomainMarkers) = domain_markers.tuples

"""
	conditions(domain_markers::DomainMarkers)

Accessor function for the [DomainMarkers](@ref) type. Returns the `conditions` field of the DomainMarkers.
"""
@inline conditions(domain_markers::DomainMarkers) = domain_markers.conditions

@inline function label_identifiers(domain_markers::DomainMarkers)
	@unpack symbols, tuples, conditions = domain_markers
	return (label(marker)::Symbol for marker in Iterators.flatten((symbols, tuples, conditions)))
end

"""
	label_symbols(domain_markers::DomainMarkers)

Extract labels associated with the `symbols` in a [DomainMarkers](@ref) collection. Returns a generator of `Symbol`s representing the labels of the markers.
"""
@inline label_symbols(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in symbols(domain_markers))

"""
	label_tuples(domain_markers::DomainMarkers) 

Extract labels associated with the `tuples` in a [DomainMarkers](@ref) collection. Returns a generator of `Symbol`s representing the labels of the markers.
"""
@inline label_tuples(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in tuples(domain_markers))

"""
	label_conditions(domain_markers::DomainMarkers)

Extract labels associated with the `conditions` in a [DomainMarkers](@ref) collection. Returns a generator of `Symbol`s representing the labels of the markers.
"""
@inline label_conditions(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in conditions(domain_markers))

"""
	point_type(::CartesianProduct{1,T})
	point_type(::CartesianProduct{D,T})

Extract the type of points in [CartesianProduct](@ref) space. Returns the element type `T` for 1D spaces, and an `NTuple{D,T}` for D-dimensional spaces.
"""
@inline point_type(::CartesianProduct{1,T}) where T = T
@inline point_type(::CartesianProduct{D,T}) where {D,T} = NTuple{D,T}

"""
	markers(space_set::CartesianProduct, pairs...)
	markers(space_set::CartesianProduct, [time_set::CartesianProduct{1}], pairs...)

Converts several `Pair{Symbol,F}` (:symbol => key), where key is Symbol (of a Tuple of these) or a Function, to a [DomainMarkers](@ref). This is to be passed in the construction of a [Domain](@ref).

# Example

```julia
tuples = (:corners => (:top, :right), :all_boundary => (:top, :right, :left, :bottom));
ids = (:left_boundary => :left, tuples..., :internal => x -> 0 <= x[1] - 0.5 <= 1);
markers(I, ids...)
```

The full list of boundary symbols that can be used can be found in [get\\_boundary\\_symbols](@ref).
"""
@inline markers(space_set::CartesianProduct, pairs::Pair...) = _create_generic_markers(Bool, space_set, pairs...)
@inline markers(space_set::CartesianProduct, time_set::CartesianProduct{1}, pairs::Pair...) = _create_generic_markers(Bool, space_set, time_set, pairs...)

#=========================================================================
Internal helper to parse identifier-based markers (Symbols and Tuples of Symbols)
from a collection of pairs. Returns a tuple containing the set of symbol markers
and the set of tuple markers.
=========================================================================#
function _extract_identifier_markers(pairs::Tuple)
	symbols = Set{Marker{Symbol}}(Marker(p.first, p.second) for p in pairs if p.second isa Symbol)
	tuples  = Set{Marker{Set{Symbol}}}(Marker(p.first, Set(p.second)) for p in pairs if p.second isa NTuple{N,Symbol} where N)
	return symbols, tuples
end

#=========================================================================
Creates DomainMarkers from pairs, handling spatial domains.
=========================================================================#
function _create_generic_markers(FinalType::Type, space_domain::CartesianProduct, pairs::Pair...)
	symbols, tuples = _extract_identifier_markers(pairs)
	conditions = _pairs_to_set_conditions(FinalType, space_domain, pairs)

	return DomainMarkers(symbols, tuples, conditions)
end

#=========================================================================
Creates DomainMarkers from pairs, handling spatio-temporal domains.
=========================================================================#
function _create_generic_markers(FinalType::Type, space_domain::CartesianProduct, time_domain::CartesianProduct{1}, pairs::Pair...)
	symbols, tuples = _extract_identifier_markers(pairs)
	conditions = _pairs_to_set_conditions(FinalType, space_domain, time_domain, pairs)

	return DomainMarkers(symbols, tuples, conditions)
end

function _pairs_to_set_conditions(FinalType::Type, space_domain::CartesianProduct{D,T}, pairs) where {D,T}
	ArgsT = point_type(space_domain)
	BrambleFuncType = BrambleFunction{ArgsT,false,FinalType,typeof(space_domain)}

	generator = (Marker(p.first, process_identifier(space_domain, p.second; FinalType))
				 for p in pairs if p.second isa Function)

	return Set{Marker{BrambleFuncType}}(generator)
end

function _pairs_to_set_conditions(FinalType::Type, space_domain::CartesianProduct{D,T}, time_domain::CartesianProduct{1,T}, pairs) where {D,T}
	SpaceArgsT = point_type(space_domain)
	SpaceFuncType = BrambleFunction{SpaceArgsT,false,FinalType,typeof(space_domain)}
	BrambleFuncType = BrambleFunction{T,true,SpaceFuncType,typeof(time_domain)}

	generator = (Marker(p.first, process_identifier(space_domain, time_domain, p.second; FinalType))
				 for p in pairs if p.second isa Function)

	return Set{Marker{BrambleFuncType}}(generator)
end

@inline function process_identifier(space_domain::CartesianProduct, identifier::F; FinalType = Bool) where {F<:Function}
	return _embed_notime(space_domain, identifier, CoType = FinalType)
end

@inline function process_identifier(space_domain::CartesianProduct, time_domain::CartesianProduct{1}, identifier::F; FinalType = Bool) where {F<:Function}
	return _embed_withtime(space_domain, time_domain, identifier, FinalCoType = FinalType)
end

@inline process_identifier(_::CartesianProduct, identifier::Symbol) = identifier
@inline process_identifier(_::CartesianProduct, identifier::NTuple{N,Symbol}) where N = Set(identifier)

"""
	EvaluatedDomainMarkers{M, T}

A lazy, view-like wrapper that represents a [DomainMarkers](@ref) object evaluated
at a specific time `t`.

This struct avoids allocating a new collection for time-evaluated functions.
Instead, it generates the time-independent functions on-the-fly when the
`conditions` are iterated over. It shares the `symbols` and `tuples` directly
from the original object.
"""
struct EvaluatedDomainMarkers{M<:DomainMarkers,T<:Number}
	original_markers::M
	evaluation_time::T
end

symbols(edm::EvaluatedDomainMarkers) = symbols(edm.original_markers)
tuples(edm::EvaluatedDomainMarkers) = tuples(edm.original_markers)

"""
	conditions(edm::EvaluatedDomainMarkers)

Returns a lazy generator that yields time-evaluated markers.

This is the core of the lazy evaluation. It iterates over the original
conditions and yields new [Marker](@ref) objects with their functions evaluated at
`edm.evaluation_time`, but only when requested.
"""
function conditions(edm::EvaluatedDomainMarkers)
	return (begin
				bramble_func = identifier(marker)
				t = edm.evaluation_time

				if applicable(bramble_func, t)
					Marker(label(marker), bramble_func(t))
				else
					marker
				end
			end
			for marker in conditions(edm.original_markers))
end

"""
	(dm::DomainMarkers)(t::Number)

Evaluates a time-dependent [DomainMarkers](@ref) object at a specific time `t`. This function returns a new, time-independent [DomainMarkers](@ref) object.

# Example

```julia
# Given a time-dependent set of markers
time_dep_markers = markers(space_domain, time_domain, :moving_front => (x, t) -> x[1] > t)

# Get a snapshot at t = 0.5
markers_at_0_5 = time_dep_markers(0.5)# Symbol and Tuple markers are time-agnostic, so they are preserved.
```
"""
(dm::DomainMarkers)(t::Number) = EvaluatedDomainMarkers(dm, t)