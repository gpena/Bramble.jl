# Abstract supertype for all domain-related types
abstract type DomainBaseType end

"""
    Domain(set::SetType, markers::MarkersType)

Computational domain pairing a geometric set (e.g. [`CartesianProduct`](@ref)) with labeled [`DomainMarkers`](@ref).

# Fields
- `set`: Geometric set defining the spatial bounding box or interval.
- `markers`: [`DomainMarkers`](@ref) collection indexing labeled boundaries and subregions.

See also: [`domain`](@ref), [`CartesianProduct`](@ref), [`DomainMarkers`](@ref).
"""
struct Domain{SetType,MarkersType} <: DomainBaseType
    set::SetType
    markers::MarkersType
end

"""
    markers(Ω::Domain) -> DomainMarkers

Return the [`DomainMarkers`](@ref) collection associated with domain `Ω`.
"""
@inline markers(Ω::Domain) = Ω.markers

"""
    symbols(Ω::Domain) -> Tuple

Return the tuple of single-symbol markers associated with domain `Ω`.
"""
@inline symbols(Ω::Domain) = symbols(markers(Ω))

"""
    tuples(Ω::Domain) -> Tuple

Return the tuple of multi-symbol markers associated with domain `Ω`.
"""
@inline tuples(Ω::Domain) = tuples(markers(Ω))

"""
    conditions(Ω::Domain) -> Tuple

Return the tuple of condition predicate markers associated with domain `Ω`.
"""
@inline conditions(Ω::Domain) = conditions(markers(Ω))

"""
    labels(Ω::Domain)

Return a tuple of the `Symbol` label of every marker in domain `Ω`.

Zero-allocation; see [`label_identifiers`](@ref).
"""
@inline labels(Ω::Domain) = labels(markers(Ω))

"""
    marker_identifiers(Ω::Domain)

Return an iterator yielding the identifying symbols, symbol sets, or predicate functions of all markers in domain `Ω`.

!!! note
    Iterating directly over [`marker_symbols`](@ref), [`marker_tuples`](@ref), or
    [`marker_conditions`](@ref) allocates 0 bytes.
"""
@inline function marker_identifiers(Ω::Domain)
    return (
        identifier(marker) for
        marker in Iterators.flatten((symbols(Ω), tuples(Ω), conditions(Ω)))
    )
end

"""
    marker_symbols(Ω::Domain)

Return an iterator yielding identifiers of single-symbol markers on domain `Ω`.
"""
@inline function marker_symbols(Ω::Domain)
    return (identifier(marker) for marker in symbols(Ω))
end

"""
    marker_tuples(Ω::Domain)

Return an iterator yielding identifiers of multi-symbol markers on domain `Ω`.
"""
@inline function marker_tuples(Ω::Domain)
    return (identifier(marker) for marker in tuples(Ω))
end

"""
    marker_conditions(Ω::Domain)

Return an iterator yielding predicate functions of condition markers on domain `Ω`.
"""
@inline function marker_conditions(Ω::Domain)
    return (identifier(marker) for marker in conditions(Ω))
end

@inline label_identifiers(Ω::Domain) = label_identifiers(markers(Ω))
@inline label_symbols(Ω::Domain) = label_symbols(markers(Ω))
@inline label_tuples(Ω::Domain) = label_tuples(markers(Ω))
@inline label_conditions(Ω::Domain) = label_conditions(markers(Ω))

"""
    domain(X::CartesianProduct) -> Domain
    domain(X::CartesianProduct, markers::DomainMarkers) -> Domain
    domain(X::CartesianProduct, pairs::Pair...) -> Domain
    domain(space_set::CartesianProduct, time_set::CartesianProduct{1}, pairs::Pair...) -> Domain

Construct a computational [`Domain`](@ref) from a [`CartesianProduct`](@ref) set and optional markers.

When no markers are supplied, defaults to a `:boundary` marker covering all boundaries of `X`.

# Arguments
- `X`: Underlying geometric set.
- `markers`: Explicit [`DomainMarkers`](@ref) container.
- `pairs`: Variable sequence of `label => identifier` pairs.
- `space_set`: Spatial bounding set.
- `time_set`: 1D temporal interval for time-dependent boundary conditions.

# Examples
```jldoctest
using Bramble
Ω = domain(interval(0.0, 1.0))
dim(Ω) == 1 && eltype(Ω) === Float64

# output
true
```
"""
@inline domain(X::CartesianProduct) =
    Domain(X, markers(X, :boundary => boundary_symbols(X)))
@inline domain(X::CartesianProduct, markers::DomainMarkers) = Domain(X, markers)
@inline domain(X::CartesianProduct, pairs::Pair...) = domain(X, markers(X, pairs...))
@inline domain(space_set::CartesianProduct, time_set::CartesianProduct{1}, pairs::Pair...) =
    domain(space_set, markers(space_set, time_set, pairs...))

"""
    (Ω::Domain)(t::Number) -> Domain

Evaluate a time-dependent [`Domain`](@ref) at timestamp `t`.
"""
@inline (Ω::Domain)(t::Number) = Domain(set(Ω), markers(Ω)(t))

"""
    set(Ω::Domain) -> CartesianProduct

Return the geometric set defining [`Domain`](@ref) `Ω`.
"""
@inline set(Ω::Domain) = Ω.set

"""
    dim(Ω::Domain) -> Int
    dim(::Type{<:Domain{SetType}}) -> Int

Return the spatial embedding dimension of [`Domain`](@ref) `Ω`.
"""
@inline dim(Ω::Domain) = dim(set(Ω))
@inline dim(::Type{<:Domain{SetType}}) where {SetType} = dim(SetType)

"""
    topo_dim(Ω::Domain) -> Int

Return the topological dimension of [`Domain`](@ref) `Ω`.
"""
@inline topo_dim(Ω::Domain) = topo_dim(set(Ω))

"""
    eltype(Ω::Domain) -> Type
    eltype(::Type{<:Domain{SetType}}) -> Type

Return the coordinate element type of [`Domain`](@ref) `Ω`.
"""
@inline eltype(Ω::Domain) = eltype(set(Ω))
@inline eltype(::Type{<:Domain{SetType}}) where {SetType} = eltype(SetType)

"""
    point_type(Ω::Domain) -> Type
    point_type(::Type{<:Domain{SetType}}) -> Type

Return the coordinate point representation type of [`Domain`](@ref) `Ω`.
"""
@inline point_type(Ω::Domain) = point_type(set(Ω))
@inline point_type(::Type{<:Domain{SetType}}) where {SetType} = point_type(SetType)

@inline Base.length(Ω::Domain) = length(markers(Ω))
@inline Base.isempty(Ω::Domain) = isempty(markers(Ω))

"""
    center(Ω::Domain) -> NTuple

Compute the geometric center point of [`Domain`](@ref) `Ω`.
"""
@inline center(Ω::Domain) = center(set(Ω))

"""
    in(x, Ω::Domain) -> Bool

Query whether point `x` lies within the closed domain `Ω`.
"""
@inline Base.in(x, Ω::Domain) = x ∈ set(Ω)

"""
    extrema(Ω::Domain) -> Tuple
    extrema(Ω::Domain, i::Integer) -> Tuple{T, T}

Return the coordinate interval endpoints of domain `Ω`.
"""
@inline Base.extrema(Ω::Domain) = extrema(set(Ω))
@inline Base.extrema(Ω::Domain, i::Integer) = extrema(set(Ω), i)

"""
    is_collapsed(Ω::Domain) -> Bool
    is_collapsed(Ω::Domain, i::Integer) -> Bool

Return whether the underlying geometric set of domain `Ω` is collapsed across any
dimension, or along coordinate dimension `i`.
"""
@inline is_collapsed(Ω::Domain) = is_collapsed(set(Ω))
@inline is_collapsed(Ω::Domain, i::Integer) = is_collapsed(set(Ω), i)

@inline (Ω::Domain)(i::Integer) = set(Ω)(i)

"""
    projection(Ω::Domain, i::Integer) -> CartesianProduct{1}

Extract the `i`-th coordinate dimension of domain `Ω` as a 1D [`CartesianProduct`](@ref).
"""
@inline projection(Ω::Domain, i::Integer) = projection(set(Ω), i)

"""
    boundary_symbols(Ω::Domain) -> Tuple{Vararg{Symbol}}
    boundary_symbols(X::CartesianProduct) -> Tuple{Vararg{Symbol}}
    boundary_symbols(D::Integer) -> Tuple{Vararg{Symbol}}

Return the default boundary symbols for dimension `D` or domain `Ω`:
- 1D ``[x_1, x_2]``: `(:left, :right)`
- 2D ``[x_1, x_2] \\times [y_1, y_2]``: `(:bottom, :top, :left, :right)`
- 3D ``[x_1, x_2] \\times [y_1, y_2] \\times [z_1, z_2]``: `(:bottom, :top, :back, :front, :left, :right)`

# Throws
- `ErrorException`: If dimension `D > 3`.
"""
@inline boundary_symbols(Ω::Domain) = boundary_symbols(set(Ω))
@inline boundary_symbols(::CartesianProduct{1}) = (:left, :right)
@inline boundary_symbols(::CartesianProduct{2}) = (:bottom, :top, :left, :right)
@inline boundary_symbols(::CartesianProduct{3}) =
    (:bottom, :top, :back, :front, :left, :right)
@inline boundary_symbols(::Type{<:CartesianProduct{1}}) = (:left, :right)
@inline boundary_symbols(::Type{<:CartesianProduct{2}}) = (:bottom, :top, :left, :right)
@inline boundary_symbols(::Type{<:CartesianProduct{3}}) =
    (:bottom, :top, :back, :front, :left, :right)
function boundary_symbols(D::Integer)
    D == 1 && return (:left, :right)
    D == 2 && return (:bottom, :top, :left, :right)
    D == 3 && return (:bottom, :top, :back, :front, :left, :right)
    return error(
        "boundary_symbols is not defined for $(D)D domains. " *
        "Provide explicit boundary names via the markers() interface.",
    )
end
@noinline function boundary_symbols(::Type{<:CartesianProduct{D}}) where {D}
    error(
        "boundary_symbols is not defined for $(D)D domains. " *
        "Provide explicit boundary names via the markers() interface.",
    )
end
@inline boundary_symbols(::Type{<:Domain{SetType}}) where {SetType} =
    boundary_symbols(SetType)

# The compact, embeddable form (gpena/Bramble.jl#45). It used to read
# `Domain{2D, Float64}:` -- a trailing colon promising content that never followed it,
# which is what `"$Ω"` and array display actually showed. The colon is gone and the marker
# count, the thing a `Domain` carries that its underlying set does not, is named instead.
function Base.show(io::IO, Ω::Domain)
    # Counted from the three marker collections rather than `length(labels(Ω))`: `labels`
    # returns a lazy `Iterators.Flatten`, which has no length.
    n = length(symbols(Ω)) + length(tuples(Ω)) + length(conditions(Ω))
    print(io, "Domain{$(dim(Ω))D, $(eltype(Ω))}(")
    show(io, set(Ω))
    print(io, ", ", n, " marker", n == 1 ? "" : "s", ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", Ω::Domain)
    return show_block(io) do io
        return _show_domain_detailed(io, Ω)
    end
end

function _show_domain_detailed(io::IO, Ω::Domain)
    pp = PrettyPrinter(io)
    X = set(Ω)

    printstyled(io, "Domain"; bold=true, color=:cyan)
    print(io, " {")
    printstyled(io, "$(dim(Ω))D"; color=:yellow)
    print(io, ", ")
    printstyled(io, "$(eltype(Ω))"; color=:yellow)
    println(io, "}:")

    println(io)
    pp_indented = with_indent(pp, 1)
    print_section_header(pp_indented, "Set:")

    D = dim(X)
    topodim = topo_dim(X)
    pp_double_indent = with_indent(pp, 2)

    # The set body itself is `set.jl`'s to render (gpena/Bramble.jl#47). Only the
    # surrounding indentation and the topological-dimension wording are this layout's
    # own: there is no `CartesianProduct{…}` header here to append "(topological dim N)"
    # to, so it gets its own line.
    if D == 1
        print(io, "    ")
        print_set_extent(pp, X)
        println(io)
    else
        if topodim < D
            print(io, "    ")
            print_colored(pp, "Topological dimension: $topodim"; color=:yellow)
            println(io)
        end
        print_set_axes(pp_double_indent, X)
    end

    println(io)
    print_section_header(pp_indented, "Markers:")

    n_sym = length(symbols(Ω))
    n_tup = length(tuples(Ω))
    n_cond = length(conditions(Ω))

    if n_sym + n_tup + n_cond == 0
        print(io, "    ")
        print_empty_message(pp, "(none)")
    else
        print_marker_summary(with_indent(pp, 2), n_sym, n_tup, n_cond)

        print(io, "    ")
        print_labels_list(pp, collect(labels(Ω)))
    end
    return nothing
end
