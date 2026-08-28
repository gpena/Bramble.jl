"""
	$(TYPEDEF)

Represents a labeled region or boundary of a computational domain.

Each `Marker` consists of a `label` (a `Symbol`) and an `identifier`. The `identifier` specifies how to locate the marked region:
- A `Symbol` for predefined boundaries (e.g., `:left`, `:top`).
- A `Set{Symbol}` for collections of predefined boundaries (e.g., `Set([:top, :right])`).
- A function (wrapped in a `BrambleFunction`) that acts as a characteristic or level-set function returning `true` for points in the marked region.

# Fields

$(FIELDS)
"""
struct Marker{F}
	"A `Symbol` naming the marked region (e.g., `:inlet`, `:wall`, `:boundary`)."
	label::Symbol
	"The object identifying the region (`Symbol`, `Set{Symbol}`, or `BrambleFunction`)."
	identifier::F
end

"""
	MarkerPair{F}

A type alias for `Pair{Symbol, F}`, representing a convenient way to define a marker (e.g., `:boundary => :left`).
"""
const MarkerPair{F} = Pair{Symbol,F}

"""
	$(SIGNATURES)

Returns the `Symbol` label of a `Marker` or `MarkerPair`.
"""
@inline label(m::Marker) = m.label
@inline label(m::MarkerPair) = first(m)

"""
	$(SIGNATURES)

Returns the identifier (`Symbol`, `Set{Symbol}`, or function) of a `Marker` or `MarkerPair`.
"""
@inline identifier(m::Marker) = m.identifier
@inline identifier(m::MarkerPair) = last(m)

"""
	$(TYPEDEF)

A container that categorizes and stores all markers for a given computational domain.

# Fields

$(FIELDS)
"""
struct DomainMarkers{BFType}
	"markers identified by a single predefined `Symbol` (e.g., `:left`)."
	symbols::Set{Marker{Symbol}}
	"markers identified by a collection of predefined `Symbol`s (e.g., `(:top, :right)`)."
	tuples::Set{Marker{Set{Symbol}}}
	"markers identified by a boolean function `f(x)` or `f(x, t)`."
	conditions::Set{Marker{BFType}}
end

"""
	$(SIGNATURES)

Returns the set of single-symbol markers from a [`DomainMarkers`](@ref) object.
"""
@inline symbols(domain_markers::DomainMarkers) = domain_markers.symbols

"""
	$(SIGNATURES)

Returns the set of symbol-tuple markers from a [`DomainMarkers`](@ref) object.
"""
@inline tuples(domain_markers::DomainMarkers) = domain_markers.tuples

"""
	$(SIGNATURES)

Returns the set of function condition markers from a [`DomainMarkers`](@ref) object.
"""
@inline conditions(domain_markers::DomainMarkers) = domain_markers.conditions

"""
	$(SIGNATURES)

Returns a generator that yields the label (`Symbol`) of every marker in the [`DomainMarkers`](@ref) collection.
"""
@inline function label_identifiers(domain_markers::DomainMarkers)
	(; symbols, tuples, conditions) = domain_markers
	return (label(marker)::Symbol for marker in Iterators.flatten((symbols, tuples, conditions)))
end

@inline labels(domain_markers::DomainMarkers) = label_identifiers(domain_markers)

"""
	$(SIGNATURES)

Returns a generator that yields the labels from the single-symbol markers.
"""
@inline label_symbols(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in symbols(domain_markers))

"""
	$(SIGNATURES)

Returns a generator that yields the labels from the symbol-tuple markers.
"""
@inline label_tuples(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in tuples(domain_markers))

"""
	$(SIGNATURES)

Returns a generator that yields the labels from the function-based condition markers.
"""
@inline label_conditions(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in conditions(domain_markers))

"""
	$(SIGNATURES)

Constructs a [`DomainMarkers`](@ref) object from a series of `label => identifier` pairs.

The `identifier` can be a `Symbol`, a `Tuple` of `Symbol`s, or a `Function`.

# Example

```jldoctest
julia> I = cartesian_product(0.0, 1.0);
julia> m = markers(I, :left_boundary => :left, :internal => x -> 0.2 < x < 0.8);
julia> length(m.symbols)
1
julia> length(m.conditions)
1
```
"""
@inline markers(space_set::CartesianProduct, pairs::Pair...) = _create_generic_markers(Bool, space_set, pairs...)
@inline markers(space_set::CartesianProduct, time_set::CartesianProduct{1}, pairs::Pair...) = _create_generic_markers(Bool, space_set, time_set, pairs...)

#=========================================================================
Internal helper to parse identifier-based markers (Symbols and Tuples of Symbols)
from a collection of pairs. Returns a tuple containing the set of symbol markers
and the set of tuple markers.
=========================================================================#
function _extract_identifier_markers(pairs::Tuple)
	symbols = Set{Marker{Symbol}}()
	tuples = Set{Marker{Set{Symbol}}}()
	sizehint!(symbols, length(pairs))
	sizehint!(tuples, length(pairs))

	for p in pairs
		if p.second isa Symbol
			push!(symbols, Marker(p.first, p.second))
		elseif p.second isa NTuple{N,Symbol} where {N}
			push!(tuples, Marker(p.first, Set(p.second)))
		end
	end

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

	result = Set{Marker{BrambleFuncType}}()
	sizehint!(result, length(pairs))

	@inbounds for p in pairs
		if p.second isa Function
			push!(result, Marker(p.first, process_identifier(space_domain, p.second; FinalType)))
		end
	end

	return result
end

function _pairs_to_set_conditions(FinalType::Type, space_domain::CartesianProduct{D,T}, time_domain::CartesianProduct{1,T}, pairs) where {D,T}
	SpaceArgsT = point_type(space_domain)
	SpaceFuncType = BrambleFunction{SpaceArgsT,false,FinalType,typeof(space_domain)}
	BrambleFuncType = BrambleFunction{T,true,SpaceFuncType,typeof(time_domain)}

	result = Set{Marker{BrambleFuncType}}()
	sizehint!(result, length(pairs))

	@inbounds for p in pairs
		if p.second isa Function
			push!(result, Marker(p.first, process_identifier(space_domain, time_domain, p.second; FinalType)))
		end
	end

	return result
end

@inline function process_identifier(space_domain::CartesianProduct, identifier::F; FinalType = Bool) where {F<:Function}
	return _embed_notime(space_domain, identifier, CoType = FinalType)
end

@inline function process_identifier(space_domain::CartesianProduct, time_domain::CartesianProduct{1}, identifier::F; FinalType = Bool) where {F<:Function}
	return _embed_withtime(space_domain, time_domain, identifier, FinalCoType = FinalType)
end

@inline process_identifier(::CartesianProduct, identifier::Symbol) = identifier
@inline process_identifier(::CartesianProduct, identifier::NTuple{N,Symbol}) where {N} = Set(identifier)

"""
	$(TYPEDEF)

A lazy, view-like wrapper representing a [`DomainMarkers`](@ref) object evaluated at a specific time `t`.

# Fields

$(FIELDS)
"""
struct EvaluatedDomainMarkers{M<:DomainMarkers,T<:Number}
	"the original time-dependent DomainMarkers instance."
	original_markers::M
	"the evaluation timestamp."
	evaluation_time::T
end

symbols(edm::EvaluatedDomainMarkers) = symbols(edm.original_markers)
tuples(edm::EvaluatedDomainMarkers) = tuples(edm.original_markers)

"""
	$(SIGNATURES)

Returns a lazy generator that yields time-evaluated markers at `edm.evaluation_time`.
"""
function conditions(edm::EvaluatedDomainMarkers)
	t = edm.evaluation_time
	return (_evaluate_marker_at_time(marker, t) for marker in conditions(edm.original_markers))
end

@inline function _evaluate_marker_at_time(marker, t)
	bramble_func = identifier(marker)
	if applicable(bramble_func, t)
		return Marker(label(marker), bramble_func(t))
	else
		return marker
	end
end

label_identifiers(edm::EvaluatedDomainMarkers) = (label(m)::Symbol for m in Iterators.flatten((symbols(edm), tuples(edm), conditions(edm))))
labels(edm::EvaluatedDomainMarkers) = label_identifiers(edm)
label_symbols(edm::EvaluatedDomainMarkers) = (label(m)::Symbol for m in symbols(edm))
label_tuples(edm::EvaluatedDomainMarkers) = (label(m)::Symbol for m in tuples(edm))
label_conditions(edm::EvaluatedDomainMarkers) = (label(m)::Symbol for m in conditions(edm))
Base.length(edm::EvaluatedDomainMarkers) = length(edm.original_markers)
Base.isempty(edm::EvaluatedDomainMarkers) = isempty(edm.original_markers)

"""
	(dm::DomainMarkers)(t::Number)

Evaluates a time-dependent [`DomainMarkers`](@ref) object at a specific time `t`.
"""
(dm::DomainMarkers)(t::Number) = EvaluatedDomainMarkers(dm, t)

"""
	Base.show(io::IO, m::Marker)

Custom display for [`Marker`](@ref) objects.
"""
function Base.show(io::IO, m::Marker{F}) where {F}
	if F <: Symbol
		print(io, "Marker(:$(m.label) => :$(m.identifier))")
	elseif F <: Set{Symbol}
		syms = join(m.identifier, ", ")
		print(io, "Marker(:$(m.label) => ($syms))")
	else
		print(io, "Marker(:$(m.label) => <function>)")
	end
end

"""
	Base.show(io::IO, dm::DomainMarkers)

Custom display for [`DomainMarkers`](@ref) objects with colored formatting.
"""
function Base.show(io::IO, dm::DomainMarkers)
	pp = PrettyPrinter(io)

	n_sym = length(dm.symbols)
	n_tup = length(dm.tuples)
	n_cond = length(dm.conditions)
	total = n_sym + n_tup + n_cond

	if pp.compact
		# Compact mode for arrays/collections
		print(io, "DomainMarkers($total total)")
	else
		# Detailed mode
		if total == 0
			print_empty_message(pp, "DomainMarkers: (empty)")
			return
		end

		print_header(pp, "DomainMarkers:")
		println(io, " with $total marker$(total == 1 ? "" : "s"):")

		pp_indented = with_indent(pp, 1)

		# Show symbol markers
		if n_sym > 0
			print_section_header(pp_indented, "Symbol markers ($n_sym):")
			pp_double_indent = with_indent(pp, 2)
			for m in dm.symbols
				print_key_value(pp_double_indent, ":$(label(m))", ":$(identifier(m))")
			end
		end

		# Show tuple markers
		if n_tup > 0
			print_section_header(pp_indented, "Tuple markers ($n_tup):")
			pp_double_indent = with_indent(pp, 2)
			for m in dm.tuples
				print_indent(pp_double_indent)
				printstyled(io, ":$(label(m))"; color = :green)
				print(io, " => (")
				syms = sort!(collect(identifier(m)))
				for (i, s) in enumerate(syms)
					printstyled(io, ":$s"; color = :blue)
					i < length(syms) && print(io, ", ")
				end
				println(io, ")")
			end
		end

		# Show condition markers
		if n_cond > 0
			print_section_header(pp_indented, "Function markers ($n_cond):")
			pp_double_indent = with_indent(pp, 2)
			for m in dm.conditions
				bf = identifier(m)
				print_indent(pp_double_indent)
				printstyled(io, ":$(label(m))"; color = :green)
				print(io, " => ")
				printstyled(io, "<function>"; color = :magenta)
				has_time(bf) && printstyled(io, " (time-dependent)"; color = :red)
				println(io)
			end
		end

		# Remove trailing newline
		remove_trailing_newline(io)
	end
end

"""
	Base.length(dm::DomainMarkers)

Returns the total number of markers in a [`DomainMarkers`](@ref) object.
"""
Base.length(dm::DomainMarkers) = length(dm.symbols) + length(dm.tuples) + length(dm.conditions)

"""
	Base.isempty(dm::DomainMarkers)

Checks if a [`DomainMarkers`](@ref) object contains zero markers.
"""
Base.isempty(dm::DomainMarkers) = isempty(dm.symbols) && isempty(dm.tuples) && isempty(dm.conditions)
