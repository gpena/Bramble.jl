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
    DomainMarkers(symbols::Tuple, tuples::Tuple, conditions::Tuple)

Container categorizing and indexing boundary and interior markers for a computational domain.

Every field is a statically typed tuple, so `symbols`, `tuples`, and `conditions` markers
are each an unrolled, zero-allocation sweep to iterate.

# Fields
- `symbols`: Tuple of markers identified by a single predefined boundary `Symbol` (e.g. `:left`).
- `tuples`: Tuple of markers identified by collections of predefined boundary symbols (e.g. `Set([:top, :right])`).
- `conditions`: Statically typed tuple of predicate function markers `f(x)` or `f(x, t)`.

See also: [`markers`](@ref), [`symbols`](@ref), [`tuples`](@ref), [`conditions`](@ref).
"""
struct DomainMarkers{ST <: Tuple, TT <: Tuple, CT <: Tuple}
    symbols::ST
    tuples::TT
    conditions::CT
end

"""
    symbols(domain_markers::DomainMarkers) -> Tuple

Return the tuple of single-symbol markers configured in `domain_markers`.
"""
@inline symbols(domain_markers::DomainMarkers) = domain_markers.symbols

"""
    tuples(domain_markers::DomainMarkers) -> Tuple

Return the tuple of multi-symbol markers configured in `domain_markers`.
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

Return a tuple of the `Symbol` label of every marker in `domain_markers`.

Concatenates the per-category labels of `symbols`, `tuples`, and `conditions` as a single
compile-time-unrolled tuple build, so this allocates 0 bytes. For the labels of one
category alone, [`label_symbols`](@ref), [`label_tuples`](@ref), or
[`label_conditions`](@ref) are equally zero-allocation.
"""
@inline function label_identifiers(domain_markers::DomainMarkers)
    (; symbols, tuples, conditions) = domain_markers
    return (map(label, symbols)..., map(label, tuples)..., map(label, conditions)...)
end

@inline labels(domain_markers::DomainMarkers) = label_identifiers(domain_markers)

"""
    label_symbols(domain_markers::DomainMarkers)

Return an iterator yielding labels of all single-symbol markers.
"""
@inline label_symbols(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in symbols(domain_markers))

"""
    label_tuples(domain_markers::DomainMarkers)

Return an iterator yielding labels of all multi-symbol markers.
"""
@inline label_tuples(domain_markers::DomainMarkers) = (label(marker)::Symbol for marker in tuples(domain_markers))

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
I = interval(0.0, 1.0)
m = markers(I, :left_boundary => :left, :internal => x -> 0.2 < x < 0.8)
length(symbols(m)) == 1 && length(conditions(m)) == 1

# output
true
```
"""
@inline function markers(space_set::CartesianProduct, pairs::Pair...)
    for p in pairs
        _validate_marker_pair(space_set, p)
    end
    return _create_generic_markers(pairs...)
end

@inline function markers(
        space_set::CartesianProduct, time_set::CartesianProduct{1}, pairs::Pair...
)
    for p in pairs
        _validate_marker_pair(space_set, time_set, p)
    end
    return _create_generic_markers(pairs...)
end

function _validate_marker_pair(space_set::CartesianProduct{D}, p::Pair) where {D}
    lbl = p.first
    ident = p.second
    if ident isa Symbol
        if !(ident in _all_boundary_symbols(space_set))
            _throw_unknown_boundary_symbol_for_domain(lbl, ident, space_set)
        end
    elseif ident isa NTuple{N, Symbol} where {N}
        for s in ident
            if !(s in _all_boundary_symbols(space_set))
                _throw_unknown_boundary_symbol_for_domain(lbl, s, space_set)
            end
        end
    elseif ident isa Function
        probe = D == 1 ? center(space_set)[1] : center(space_set)
        res = try
            ident(probe)
        catch err
            if D == 1
                try
                    ident((probe,))
                catch
                    _throw_invalid_marker_predicate_call(lbl, D, err)
                end
            else
                _throw_invalid_marker_predicate_call(lbl, D, err)
            end
        end
        if !(res isa Bool)
            _throw_non_bool_marker_predicate(lbl, res)
        end
    end
    return nothing
end

function _validate_marker_pair(
        space_set::CartesianProduct{D}, time_set::CartesianProduct{1}, p::Pair
) where {D}
    lbl = p.first
    ident = p.second
    if ident isa Symbol || ident isa NTuple{N, Symbol} where {N}
        _validate_marker_pair(space_set, p)
    elseif ident isa Function
        probe_x = D == 1 ? center(space_set)[1] : center(space_set)
        probe_t = first(extrema(time_set))
        res = try
            if hasmethod(ident, Tuple{typeof(probe_x), typeof(probe_t)})
                ident(probe_x, probe_t)
            else
                ident(probe_x)
            end
        catch err
            _throw_invalid_marker_predicate_call(lbl, D, err)
        end
        if !(res isa Bool)
            _throw_non_bool_marker_predicate(lbl, res)
        end
    end
    return nothing
end

@noinline function _throw_non_bool_marker_predicate(lbl::Symbol, res)
    throw(
        ArgumentError(
        "Marker predicate for label :$lbl returned a $(typeof(res)) ($res), " *
        "but expected a Bool. For level-set or geometric expressions, write a boolean condition " *
        "(e.g. `x -> predicate(x) <= 0`).",
    ),
    )
end

@noinline function _throw_invalid_marker_predicate_call(lbl::Symbol, D::Int, err)
    throw(
        ArgumentError(
        "Marker predicate for label :$lbl failed when evaluated on sample domain point: " *
        "expected a function accepting a $(D == 1 ? "1D coordinate (scalar or 1-tuple)" : "$D-element coordinate tuple"). " *
        "Underlying error: $err",
    ),
    )
end

@noinline function _throw_unknown_boundary_symbol_for_domain(lbl::Symbol, sym::Symbol, space_set)
    known = sort!(collect(_all_boundary_symbols(space_set)))
    avail = join(map(s -> ":$s", known), ", ")
    throw(
        ArgumentError(
        "Unknown boundary symbol :$sym for marker :$lbl. " *
        "Valid boundary symbols for this $(dim(space_set))D domain are: $avail.",
    ),
    )
end

# Parse identifier-based markers (Symbols and Tuples of Symbols) from input pairs.
# Deduplication needs Set semantics (a `:label => :left` pair repeated verbatim collapses
# to one marker), but that is a one-time construction cost, not a per-query one -- the
# result is converted to a Tuple so every later read of DomainMarkers is zero-allocation.
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

    return Tuple(symbols), Tuple(tuples)
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
@inline process_identifier(::CartesianProduct, identifier::NTuple{N, Symbol}) where {N} = Set(identifier)
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
    return map(
        m -> Marker(label(m), Base.Fix2(identifier(m), t)), conditions(edm.original_markers)
    )
end

"""
    label_identifiers(edm::EvaluatedDomainMarkers)
    labels(edm::EvaluatedDomainMarkers)

Return a tuple of the `Symbol` label of every marker in evaluated marker collection `edm`.

Zero-allocation, like [`label_identifiers`](@ref label_identifiers(::DomainMarkers)).
"""
@inline label_identifiers(edm::EvaluatedDomainMarkers) = (
    map(label, symbols(edm))...,
    map(label, tuples(edm))...,
    map(label, conditions(edm))...
)

@inline labels(edm::EvaluatedDomainMarkers) = label_identifiers(edm)

@inline label_symbols(edm::EvaluatedDomainMarkers) = (label(m)::Symbol for m in symbols(edm))

@inline label_tuples(edm::EvaluatedDomainMarkers) = (label(m)::Symbol for m in tuples(edm))

@inline label_conditions(edm::EvaluatedDomainMarkers) = (label(m)::Symbol for m in conditions(edm))

@inline Base.length(edm::EvaluatedDomainMarkers) = length(edm.original_markers)
@inline Base.isempty(edm::EvaluatedDomainMarkers) = isempty(edm.original_markers)

"""
    (dm::DomainMarkers)(t::Number) -> EvaluatedDomainMarkers

Evaluate time-dependent condition markers at timestamp `t`.
"""
(dm::DomainMarkers)(t::Number) = EvaluatedDomainMarkers(dm, t)

"""
    EvaluatedParametricDomainMarkers(original_markers::DomainMarkers, evaluation_time::Number, p)

Time- and parameter-evaluated wrapper for a [`DomainMarkers`](@ref) collection whose
conditions accept `(x, t, p)` rather than `(x, t)`, evaluated at timestamp `t` and parameter
`p`.

Kept separate from [`EvaluatedDomainMarkers`](@ref) rather than adding a `p` field to it: the
two-argument `Base.Fix2(identifier(m), t)` this file already builds for the `(x, t)` case
would otherwise need a runtime branch on whether `p` is present, and this exact kind of
branch-carrying `Union`/optional field is what made Enzyme's strict-aliasing type analysis
reject `AnySegment{D}` (gpena/Bramble.jl#240) -- a fresh concrete type costs a few duplicated
one-line accessors instead, and keeps every existing `EvaluatedDomainMarkers` call site
untouched.

# Fields
- `original_markers`: Underlying [`DomainMarkers`](@ref) object.
- `evaluation_time`: Evaluation timestamp `t`.
- `p`: Evaluation parameter, passed through unmodified (e.g. a `Vector` or a scalar).
"""
struct EvaluatedParametricDomainMarkers{M <: DomainMarkers, T <: Number, P}
    original_markers::M
    evaluation_time::T
    p::P
end

"""
    (dm::DomainMarkers)(t::Number, p) -> EvaluatedParametricDomainMarkers

Evaluate parameter- and time-dependent condition markers at timestamp `t` and parameter `p`.
"""
(dm::DomainMarkers)(t::Number, p) = EvaluatedParametricDomainMarkers(dm, t, p)

@inline symbols(edm::EvaluatedParametricDomainMarkers) = symbols(edm.original_markers)
@inline tuples(edm::EvaluatedParametricDomainMarkers) = tuples(edm.original_markers)

"""
    conditions(edm::EvaluatedParametricDomainMarkers) -> Tuple

Return condition markers evaluated at `edm.evaluation_time` and `edm.p`, converting `f(x, t,
p)` closures into unary spatial predicates `f(x)`.

`Base.Fix{N}` generalizes `Base.Fix1`/`Base.Fix2` to insert its fixed value at position `N`
of *whatever args a given call supplies*, not at position `N` of `f`'s own argument list
(`Base.Fix`'s own docstring example: `Fix{1}(Fix{2}(f, 4), 4)` fixes the first and second
arg, not the first and fourth). So `Base.Fix{3}(f, p)` first, fixing `f`'s own third
argument to `p` and leaving a 2-argument `(x, t) -> f(x, t, p)`; `Base.Fix2` on *that* then
fixes its second argument to `t`, landing on `x -> f(x, t, p)`. Reversing the order --
`Base.Fix2(Base.Fix2(f, p), t)`, the tempting reading of "fix p, then fix t" -- instead
inserts `p` at position 2 and shoves `t` to position 3, calling `f(x, p, t)` (caught by the
test that checks a `(1.0, 3.0)` vs a swapped-order failure, not by inspection).
"""
function conditions(edm::EvaluatedParametricDomainMarkers)
    t = edm.evaluation_time
    p = edm.p
    return map(
        m -> Marker(label(m), Base.Fix2(Base.Fix{3}(identifier(m), p), t)),
        conditions(edm.original_markers)
    )
end

@inline label_identifiers(edm::EvaluatedParametricDomainMarkers) = (
    map(label, symbols(edm))...,
    map(label, tuples(edm))...,
    map(label, conditions(edm))...
)

@inline labels(edm::EvaluatedParametricDomainMarkers) = label_identifiers(edm)

@inline label_symbols(edm::EvaluatedParametricDomainMarkers) = (label(m)::Symbol for m in symbols(edm))
@inline label_tuples(edm::EvaluatedParametricDomainMarkers) = (label(m)::Symbol for m in tuples(edm))
@inline label_conditions(edm::EvaluatedParametricDomainMarkers) = (label(m)::Symbol for m in conditions(edm))

@inline Base.length(edm::EvaluatedParametricDomainMarkers) = length(edm.original_markers)
@inline Base.isempty(edm::EvaluatedParametricDomainMarkers) = isempty(edm.original_markers)

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
    total = length(dm.symbols) + length(dm.tuples) + length(dm.conditions)
    print(io, "DomainMarkers($total total)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", dm::DomainMarkers)
    return show_block(io) do io
        pp = PrettyPrinter(io)

        n_sym = length(dm.symbols)
        n_tup = length(dm.tuples)
        n_cond = length(dm.conditions)
        total = n_sym + n_tup + n_cond

        if total == 0
            print_empty_message(pp, "DomainMarkers: (empty)")
            return nothing
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
                print_joined(pp, sort!(collect(identifier(m)))) do s
                    return printstyled(io, ":$s"; color = :blue)
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
    end
end

@inline function Base.length(dm::DomainMarkers)
    length(dm.symbols) + length(dm.tuples) + length(dm.conditions)
end

@inline function Base.isempty(dm::DomainMarkers)
    isempty(dm.symbols) && isempty(dm.tuples) && isempty(dm.conditions)
end
