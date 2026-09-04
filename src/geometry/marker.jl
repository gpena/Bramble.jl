"""
	$(TYPEDEF)

Represents a labeled region or boundary of a computational domain.

Each `Marker` consists of a `label` (a `Symbol`) and an `identifier`. The `identifier` specifies how to locate the marked region:
- A `Symbol` for predefined boundaries (e.g., `:left`, `:top`).
- A `Set{Symbol}` for collections of predefined boundaries (e.g., `Set([:top, :right])`).
- A function that acts as a characteristic or level-set function returning `true` for points in the marked region.

# Fields

$(FIELDS)
"""
struct Marker{F}
    "A `Symbol` naming the marked region (e.g., `:inlet`, `:wall`, `:boundary`)."
    label::Symbol
    "The object identifying the region (`Symbol`, `Set{Symbol}`, or a raw function)."
    identifier::F
end

"""
	MarkerPair{F}

A type alias for `Pair{Symbol, F}`, representing a convenient way to define a marker (e.g., `:boundary => :left`).
"""
const MarkerPair{F} = Pair{Symbol, F}

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

`conditions` is a `Tuple` rather than a `Set`: each marker keeps its own condition's
concrete closure type as one of the tuple's element types, rather than erasing every
condition into one shared wrapper type the way a homogeneous `Set` would force. Point 48
found this genuinely faster (not just simpler) for the one place a condition is called in
a real hot loop, per-point: ~1.84× on Dirichlet value application, allocation-free either
way, and the previous `BrambleFunction`-wrapped path wasn't even allocation-free once time
dependence was involved (evaluating the outer time-closure built a new wrapper every call).
A tuple pays for this with a distinct `DomainMarkers` type per distinct set of condition
closures, rather than one type covering any combination — the same tradeoff `LinearForm`/
`BilinearForm` already accepted for a form's AST.

# Fields

$(FIELDS)
"""
struct DomainMarkers{CT <: Tuple}
    "markers identified by a single predefined `Symbol` (e.g., `:left`)."
    symbols::Set{Marker{Symbol}}
    "markers identified by a collection of predefined `Symbol`s (e.g., `(:top, :right)`)."
    tuples::Set{Marker{Set{Symbol}}}
    "markers identified by a boolean function `f(x)` or `f(x, t)`, one `Marker{F}` per condition's own closure type."
    conditions::CT
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
    return (label(marker)::Symbol
    for marker in Iterators.flatten((symbols, tuples, conditions)))
end

@inline labels(domain_markers::DomainMarkers) = label_identifiers(domain_markers)

"""
	$(SIGNATURES)

Returns a generator that yields the labels from the single-symbol markers.
"""
@inline label_symbols(domain_markers::DomainMarkers) = (label(marker)::Symbol
for marker in symbols(domain_markers))

"""
	$(SIGNATURES)

Returns a generator that yields the labels from the symbol-tuple markers.
"""
@inline label_tuples(domain_markers::DomainMarkers) = (label(marker)::Symbol
for marker in tuples(domain_markers))

"""
	$(SIGNATURES)

Returns a generator that yields the labels from the function-based condition markers.
"""
@inline label_conditions(domain_markers::DomainMarkers) = (label(marker)::Symbol
for marker in conditions(domain_markers))

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
@inline markers(space_set::CartesianProduct, pairs::Pair...) = _create_generic_markers(pairs...)
@inline markers(space_set::CartesianProduct, time_set::CartesianProduct{1},
    pairs::Pair...) = _create_generic_markers(pairs...)

#=========================================================================
Internal helper to parse identifier-based markers (Symbols and Tuples of Symbols)
from a collection of pairs. Returns a tuple containing the set of symbol markers
and the set of tuple markers.
=========================================================================#
function _extract_identifier_markers(pairs::Tuple)
    symbols = Set{Marker{Symbol}}()
    tuples = Set{Marker{Set{Symbol}}}()
    n = length(pairs)
    sizehint!(symbols, n ÷ 2 + 1)
    sizehint!(tuples, n ÷ 2 + 1)

    for p in pairs
        if p.second isa Symbol
            push!(symbols, Marker(p.first, p.second))
        elseif p.second isa NTuple{N, Symbol} where {N}
            push!(tuples, Marker(p.first, Set(p.second)))
        end
    end

    return symbols, tuples
end

#=========================================================================
Creates DomainMarkers from pairs. The space/time domain used to matter here only to
settle a BrambleFunction's CoType ahead of time (point 48) -- a raw closure needs no
such settling, so the same builder now covers both the spatial and spatio-temporal
constructors; what distinguishes them is only whether the caller later calls the result
as `dm(t)`, which `EvaluatedDomainMarkers` handles.
=========================================================================#
function _create_generic_markers(pairs::Pair...)
    symbols, tuples = _extract_identifier_markers(pairs)
    conditions = _pairs_to_tuple_conditions(pairs)

    return DomainMarkers(symbols, tuples, conditions)
end

# One `Marker{F}` per function-valued pair, each keeping its own closure's concrete type
# rather than erasing every condition into one shared wrapper type (point 48). `filter`/`map`
# over a `Tuple` are themselves recursive/generated in Base, so this stays allocation-free
# and fully specialised -- the same guarantee `_write_components!` relies on elsewhere for a
# differently-shaped heterogeneous-tuple problem.
@inline function _pairs_to_tuple_conditions(pairs::Tuple)
    fn_pairs = filter(p -> p.second isa Function, pairs)
    return map(p -> Marker(p.first, p.second), fn_pairs)
end

@inline process_identifier(::CartesianProduct, identifier::Symbol) = identifier
@inline process_identifier(::CartesianProduct, identifier::NTuple{
    N, Symbol}) where {N} = Set(identifier)
@inline process_identifier(::CartesianProduct, identifier::AbstractVector{Symbol}) = Set(identifier)

"""
	$(TYPEDEF)

A lazy, view-like wrapper representing a [`DomainMarkers`](@ref) object evaluated at a specific time `t`.

# Fields

$(FIELDS)
"""
struct EvaluatedDomainMarkers{M <: DomainMarkers, T <: Number}
    "the original time-dependent DomainMarkers instance."
    original_markers::M
    "the evaluation timestamp."
    evaluation_time::T
end

"""
	$(SIGNATURES)

Returns the set of single-symbol markers from an [`EvaluatedDomainMarkers`](@ref) object.
"""
@inline symbols(edm::EvaluatedDomainMarkers) = symbols(edm.original_markers)

"""
	$(SIGNATURES)

Returns the set of symbol-tuple markers from an [`EvaluatedDomainMarkers`](@ref) object.
"""
@inline tuples(edm::EvaluatedDomainMarkers) = tuples(edm.original_markers)

"""
	$(SIGNATURES)

Returns a tuple of condition markers evaluated at `edm.evaluation_time`: each raw `f(x, t)`
closure becomes an `f(x)` one, via `Base.Fix2(f, t)` rather than a call through a wrapper --
every condition built through the time-dependent constructor is uniformly two-argument, so
this needs no per-marker check the way the old `BrambleFunction`-wrapped path did. `map`
over a `Tuple` preserves the tuple (and each closure's own concrete type), the same
guarantee [`conditions`](@ref)`(::DomainMarkers)` itself relies on.
"""
function conditions(edm::EvaluatedDomainMarkers)
    t = edm.evaluation_time
    return map(m -> Marker(label(m), Base.Fix2(identifier(m), t)),
        conditions(edm.original_markers))
end

"""
	$(SIGNATURES)

Returns a generator that yields the label (`Symbol`) of every marker in an [`EvaluatedDomainMarkers`](@ref) collection.
"""
@inline label_identifiers(edm::EvaluatedDomainMarkers) = (label(m)::Symbol
for m in Iterators.flatten((symbols(edm), tuples(edm), conditions(edm))))

"""
	$(SIGNATURES)

Returns a generator that yields the labels of all markers in an [`EvaluatedDomainMarkers`](@ref) collection.
"""
@inline labels(edm::EvaluatedDomainMarkers) = label_identifiers(edm)

"""
	$(SIGNATURES)

Returns a generator that yields the labels of the single-symbol markers in an [`EvaluatedDomainMarkers`](@ref) collection.
"""
@inline label_symbols(edm::EvaluatedDomainMarkers) = (label(m)::Symbol
for m in symbols(edm))

"""
	$(SIGNATURES)

Returns a generator that yields the labels of the symbol-tuple markers in an [`EvaluatedDomainMarkers`](@ref) collection.
"""
@inline label_tuples(edm::EvaluatedDomainMarkers) = (label(m)::Symbol for m in tuples(edm))

"""
	$(SIGNATURES)

Returns a generator that yields the labels of the function-based condition markers in an [`EvaluatedDomainMarkers`](@ref) collection.
"""
@inline label_conditions(edm::EvaluatedDomainMarkers) = (label(m)::Symbol
for m in conditions(edm))

"""
	$(SIGNATURES)

Returns the total number of markers in an [`EvaluatedDomainMarkers`](@ref) object.
"""
@inline Base.length(edm::EvaluatedDomainMarkers) = length(edm.original_markers)

"""
	$(SIGNATURES)

Returns `true` if an [`EvaluatedDomainMarkers`](@ref) object contains zero markers.
"""
@inline Base.isempty(edm::EvaluatedDomainMarkers) = isempty(edm.original_markers)

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

        # Show condition markers. Whether a given condition is time-dependent is no longer
        # a per-marker trait to query -- every condition built through the time-dependent
        # constructor is uniformly two-argument -- so this only prints the label.
        if n_cond > 0
            print_section_header(pp_indented, "Function markers ($n_cond):")
            pp_double_indent = with_indent(pp, 2)
            for m in dm.conditions
                print_indent(pp_double_indent)
                printstyled(io, ":$(label(m))"; color = :green)
                print(io, " => ")
                printstyled(io, "<function>"; color = :magenta)
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
function Base.length(dm::DomainMarkers)
    length(dm.symbols) + length(dm.tuples) + length(dm.conditions)
end

"""
	Base.isempty(dm::DomainMarkers)

Checks if a [`DomainMarkers`](@ref) object contains zero markers.
"""
function Base.isempty(dm::DomainMarkers)
    isempty(dm.symbols) && isempty(dm.tuples) && isempty(dm.conditions)
end
