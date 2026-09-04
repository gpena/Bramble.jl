"""
    Marker(label::Symbol, identifier::F)

Labeled geometric region or boundary marker on a computational domain.

# Arguments
- `label`: Region identifier (e.g. `:inlet`, `:wall`, `:boundary`).
- `identifier`: Location specification, either a predefined boundary `Symbol` (e.g. `:left`),
  a `Set{Symbol}` of boundary names, or a boolean spatial predicate function `f(x)`.
"""
struct Marker{F}
    label::Symbol
    identifier::F
end

"""
    MarkerPair{F}

Type alias for `Pair{Symbol, F}` used to specify region markers (e.g. `:boundary => :left`).
"""
const MarkerPair{F} = Pair{Symbol, F}

"""
    label(m::Marker) -> Symbol
    label(m::MarkerPair) -> Symbol

Return the `Symbol` label of marker `m`.
"""
@inline label(m::Marker) = m.label
@inline label(m::MarkerPair) = first(m)

"""
    identifier(m::Marker{F}) -> F
    identifier(m::MarkerPair{F}) -> F

Return the identifying symbol, set of symbols, or predicate function of marker `m`.
"""
@inline identifier(m::Marker) = m.identifier
@inline identifier(m::MarkerPair) = last(m)

"""
    DomainMarkers(symbols::Set{Marker{Symbol}}, tuples::Set{Marker{Set{Symbol}}}, conditions::Tuple)

Container categorizing and indexing boundary and interior markers for a computational domain.

Condition predicates are stored in a tuple to preserve closure specialization at the type level.

# Fields
- `symbols`: Markers identified by a single predefined boundary `Symbol` (e.g. `:left`).
- `tuples`: Markers identified by collections of predefined boundary symbols (e.g. `Set([:top, :right])`).
- `conditions`: Statically typed tuple of predicate function markers `f(x)` or `f(x, t)`.

See also: [`markers`](@ref), [`symbols`](@ref), [`tuples`](@ref), [`conditions`](@ref).
"""
struct DomainMarkers{CT <: Tuple}
    symbols::Set{Marker{Symbol}}
    tuples::Set{Marker{Set{Symbol}}}
    conditions::CT
end

"""
    symbols(domain_markers::DomainMarkers) -> Set{Marker{Symbol}}

Return the set of single-symbol markers configured in `domain_markers`.
"""
@inline symbols(domain_markers::DomainMarkers) = domain_markers.symbols

"""
    tuples(domain_markers::DomainMarkers) -> Set{Marker{Set{Symbol}}}

Return the set of multi-symbol markers configured in `domain_markers`.
"""
@inline tuples(domain_markers::DomainMarkers) = domain_markers.tuples

"""
    conditions(domain_markers::DomainMarkers) -> Tuple

Return the tuple of predicate condition markers configured in `domain_markers`.
"""
@inline conditions(domain_markers::DomainMarkers) = domain_markers.conditions

"""
    label_identifiers(domain_markers::DomainMarkers)
    labels(domain_markers::DomainMarkers)

Return an iterator yielding the `Symbol` label of every marker in `domain_markers`.

!!! note
    Flattening across heterogeneous marker types (`symbols`, `tuples`, and `conditions`)
    allocates ~224 bytes for the union iterator state. For zero allocations in performance-critical
    paths, iterate directly over [`label_symbols`](@ref), [`label_tuples`](@ref), or
    [`label_conditions`](@ref), which allocate 0 bytes.
"""
@inline function label_identifiers(domain_markers::DomainMarkers)
    (; symbols, tuples, conditions) = domain_markers
    return (label(marker)::Symbol
    for marker in Iterators.flatten((symbols, tuples, conditions)))
end

@inline labels(domain_markers::DomainMarkers) = label_identifiers(domain_markers)

"""
    label_symbols(domain_markers::DomainMarkers)

Return an iterator yielding labels of all single-symbol markers.
"""
@inline label_symbols(domain_markers::DomainMarkers) = (label(marker)::Symbol
for marker in symbols(domain_markers))

"""
    label_tuples(domain_markers::DomainMarkers)

Return an iterator yielding labels of all multi-symbol markers.
"""
@inline label_tuples(domain_markers::DomainMarkers) = (label(marker)::Symbol
for marker in tuples(domain_markers))

"""
    label_conditions(domain_markers::DomainMarkers)

Return an iterator yielding labels of all condition predicate markers.
"""
@inline label_conditions(domain_markers::DomainMarkers) = (label(marker)::Symbol
for marker in conditions(domain_markers))

"""
    markers(space_set::CartesianProduct, pairs::Pair...) -> DomainMarkers
    markers(space_set::CartesianProduct, time_set::CartesianProduct{1}, pairs::Pair...) -> DomainMarkers

Construct a [`DomainMarkers`](@ref) collection from `label => identifier` pairs.

# Arguments
- `space_set`: Geometric spatial set.
- `time_set`: Optional 1D temporal interval for time-dependent boundary conditions.
- `pairs`: Vararg sequence of `label => identifier` pairs where identifier is a `Symbol`,
  `NTuple{N, Symbol}`, or predicate `Function`.

# Examples
```jldoctest
using Bramble
I = cartesian_product(0.0, 1.0)
m = markers(I, :left_boundary => :left, :internal => x -> 0.2 < x < 0.8)
length(symbols(m)) == 1 && length(conditions(m)) == 1

# output
true
```
"""
@inline markers(space_set::CartesianProduct, pairs::Pair...) = _create_generic_markers(pairs...)
@inline markers(space_set::CartesianProduct, time_set::CartesianProduct{1},
    pairs::Pair...) = _create_generic_markers(pairs...)

# Parse identifier-based markers (Symbols and Tuples of Symbols) from input pairs
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

# Construct DomainMarkers from label-identifier pairs, extracting symbol and tuple sets
# while preserving concrete closure types in a specialized conditions tuple.
function _create_generic_markers(pairs::Pair...)
    symbols, tuples = _extract_identifier_markers(pairs)
    conditions = _pairs_to_tuple_conditions(pairs)

    return DomainMarkers(symbols, tuples, conditions)
end

# Extract function-valued pairs into a tuple of Marker{F} instances, keeping each closure's
# concrete type specialized for zero heap allocations during evaluation.
@inline function _pairs_to_tuple_conditions(pairs::Tuple)
    fn_pairs = filter(p -> p.second isa Function, pairs)
    return map(p -> Marker(p.first, p.second), fn_pairs)
end

@inline process_identifier(::CartesianProduct, identifier::Symbol) = identifier
@inline process_identifier(::CartesianProduct, identifier::NTuple{
    N, Symbol}) where {N} = Set(identifier)
@inline process_identifier(::CartesianProduct, identifier::AbstractVector{Symbol}) = Set(identifier)

"""
    EvaluatedDomainMarkers(original_markers::DomainMarkers, evaluation_time::Number)

Time-evaluated wrapper representing a time-dependent [`DomainMarkers`](@ref) collection evaluated at timestamp `t`.

# Fields
- `original_markers`: Underlying [`DomainMarkers`](@ref) object.
- `evaluation_time`: Evaluation timestamp `t`.
"""
struct EvaluatedDomainMarkers{M <: DomainMarkers, T <: Number}
    original_markers::M
    evaluation_time::T
end

"""
    symbols(edm::EvaluatedDomainMarkers) -> Set{Marker{Symbol}}

Return single-symbol markers from the underlying domain markers.
"""
@inline symbols(edm::EvaluatedDomainMarkers) = symbols(edm.original_markers)

"""
    tuples(edm::EvaluatedDomainMarkers) -> Set{Marker{Set{Symbol}}}

Return multi-symbol markers from the underlying domain markers.
"""
@inline tuples(edm::EvaluatedDomainMarkers) = tuples(edm.original_markers)

"""
    conditions(edm::EvaluatedDomainMarkers) -> Tuple

Return condition markers evaluated at timestamp `edm.evaluation_time`, converting `f(x, t)`
closures into unary spatial predicates `f(x)` via `Base.Fix2(f, t)`.
"""
function conditions(edm::EvaluatedDomainMarkers)
    t = edm.evaluation_time
    return map(m -> Marker(label(m), Base.Fix2(identifier(m), t)),
        conditions(edm.original_markers))
end

"""
    label_identifiers(edm::EvaluatedDomainMarkers)
    labels(edm::EvaluatedDomainMarkers)

Return an iterator yielding the `Symbol` label of every marker in evaluated marker collection `edm`.

!!! note
    Iterating directly over [`label_symbols`](@ref), [`label_tuples`](@ref), or
    [`label_conditions`](@ref) allocates 0 bytes.
"""
@inline label_identifiers(edm::EvaluatedDomainMarkers) = (label(m)::Symbol
for m in Iterators.flatten((symbols(edm), tuples(edm), conditions(edm))))

@inline labels(edm::EvaluatedDomainMarkers) = label_identifiers(edm)

@inline label_symbols(edm::EvaluatedDomainMarkers) = (label(m)::Symbol
for m in symbols(edm))

@inline label_tuples(edm::EvaluatedDomainMarkers) = (label(m)::Symbol for m in tuples(edm))

@inline label_conditions(edm::EvaluatedDomainMarkers) = (label(m)::Symbol
for m in conditions(edm))

@inline Base.length(edm::EvaluatedDomainMarkers) = length(edm.original_markers)
@inline Base.isempty(edm::EvaluatedDomainMarkers) = isempty(edm.original_markers)

"""
    (dm::DomainMarkers)(t::Number) -> EvaluatedDomainMarkers

Evaluate time-dependent condition markers at timestamp `t`.
"""
(dm::DomainMarkers)(t::Number) = EvaluatedDomainMarkers(dm, t)

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

function Base.show(io::IO, dm::DomainMarkers)
    pp = PrettyPrinter(io)

    n_sym = length(dm.symbols)
    n_tup = length(dm.tuples)
    n_cond = length(dm.conditions)
    total = n_sym + n_tup + n_cond

    if pp.compact
        print(io, "DomainMarkers($total total)")
    else
        if total == 0
            print_empty_message(pp, "DomainMarkers: (empty)")
            return
        end

        print_header(pp, "DomainMarkers:")
        println(io, " with $total marker$(total == 1 ? "" : "s"):")

        pp_indented = with_indent(pp, 1)

        if n_sym > 0
            print_section_header(pp_indented, "Symbol markers ($n_sym):")
            pp_double_indent = with_indent(pp, 2)
            for m in dm.symbols
                print_key_value(pp_double_indent, ":$(label(m))", ":$(identifier(m))")
            end
        end

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

        remove_trailing_newline(io)
    end
end

@inline function Base.length(dm::DomainMarkers)
    length(dm.symbols) + length(dm.tuples) + length(dm.conditions)
end

@inline function Base.isempty(dm::DomainMarkers)
    isempty(dm.symbols) && isempty(dm.tuples) && isempty(dm.conditions)
end
