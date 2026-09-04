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
struct Domain{SetType, MarkersType} <: DomainBaseType
    set::SetType
    markers::MarkersType
end

"""
    markers(Ω::Domain) -> DomainMarkers

Return the [`DomainMarkers`](@ref) collection associated with domain `Ω`.
"""
@inline markers(Ω::Domain) = Ω.markers

"""
    symbols(Ω::Domain) -> Set{Marker{Symbol}}

Return the set of single-symbol markers associated with domain `Ω`.
"""
@inline symbols(Ω::Domain) = symbols(markers(Ω))

"""
    tuples(Ω::Domain) -> Set{Marker{Set{Symbol}}}

Return the set of multi-symbol markers associated with domain `Ω`.
"""
@inline tuples(Ω::Domain) = tuples(markers(Ω))

"""
    conditions(Ω::Domain) -> Tuple

Return the tuple of condition predicate markers associated with domain `Ω`.
"""
@inline conditions(Ω::Domain) = conditions(markers(Ω))

"""
    labels(Ω::Domain)

Return an iterator yielding the `Symbol` label of every marker in domain `Ω`.

!!! note
    Flattening across heterogeneous marker types (`symbols`, `tuples`, and `conditions`)
    allocates ~224 bytes for the union iterator state. For zero allocations in performance-critical
    paths, iterate directly over [`label_symbols`](@ref), [`label_tuples`](@ref), or
    [`label_conditions`](@ref), which allocate 0 bytes.
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
    return (identifier(marker)
    for marker in Iterators.flatten((symbols(Ω), tuples(Ω), conditions(Ω))))
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
@inline domain(X::CartesianProduct) = Domain(X, markers(X, :boundary =>
    get_boundary_symbols(X)))
@inline domain(X::CartesianProduct, markers::DomainMarkers) = Domain(X, markers)
@inline domain(X::CartesianProduct, pairs::Pair...) = domain(X, markers(X, pairs...))
@inline domain(space_set::CartesianProduct, time_set::CartesianProduct{1}, pairs::Pair...) = domain(
    space_set, markers(space_set, time_set, pairs...))

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
    center(Ω::Domain) -> SVector

Compute the geometric center point of [`Domain`](@ref) `Ω`.
"""
@inline center(Ω::Domain) = center(set(Ω))

"""
    in(x, Ω::Domain) -> Bool

Query whether point `x` lies within the closed domain `Ω`.
"""
@inline Base.in(x, Ω::Domain) = x ∈ set(Ω)

"""
    tails(Ω::Domain) -> Tuple
    tails(Ω::Domain, i::Integer) -> Tuple{T, T}

Return the coordinate interval endpoints of domain `Ω`.
"""
@inline tails(Ω::Domain) = tails(set(Ω))
@inline tails(Ω::Domain, i::Integer) = tails(set(Ω), i)

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
    get_boundary_symbols(Ω::Domain) -> Tuple{Vararg{Symbol}}
    get_boundary_symbols(X::CartesianProduct) -> Tuple{Vararg{Symbol}}
    get_boundary_symbols(D::Integer) -> Tuple{Vararg{Symbol}}

Return the default boundary symbols for dimension `D` or domain `Ω`:
- 1D ``[x_1, x_2]``: `(:left, :right)`
- 2D ``[x_1, x_2] \\times [y_1, y_2]``: `(:bottom, :top, :left, :right)`
- 3D ``[x_1, x_2] \\times [y_1, y_2] \\times [z_1, z_2]``: `(:bottom, :top, :back, :front, :left, :right)`

# Throws
- `ErrorException`: If dimension `D > 3`.
"""
@inline get_boundary_symbols(Ω::Domain) = get_boundary_symbols(set(Ω))
@inline get_boundary_symbols(::CartesianProduct{1}) = (:left, :right)
@inline get_boundary_symbols(::CartesianProduct{2}) = (:bottom, :top, :left, :right)
@inline get_boundary_symbols(::CartesianProduct{3}) = (
    :bottom, :top, :back, :front, :left, :right)
@inline get_boundary_symbols(::Type{<:CartesianProduct{1}}) = (:left, :right)
@inline get_boundary_symbols(::Type{<:CartesianProduct{2}}) = (:bottom, :top, :left, :right)
@inline get_boundary_symbols(::Type{<:CartesianProduct{3}}) = (
    :bottom, :top, :back, :front, :left, :right)
function get_boundary_symbols(D::Integer)
    D == 1 && return (:left, :right)
    D == 2 && return (:bottom, :top, :left, :right)
    D == 3 && return (:bottom, :top, :back, :front, :left, :right)
    error("get_boundary_symbols is not defined for $(D)D domains. " *
          "Provide explicit boundary names via the markers() interface.")
end
@noinline function get_boundary_symbols(::Type{<:CartesianProduct{D}}) where {D}
    error("get_boundary_symbols is not defined for $(D)D domains. " *
          "Provide explicit boundary names via the markers() interface.")
end
@inline get_boundary_symbols(::Type{<:Domain{SetType}}) where {SetType} = get_boundary_symbols(SetType)

function Base.show(io::IO, Ω::Domain)
    pp = PrettyPrinter(io)

    if pp.compact
        print(io, "Domain{$(dim(Ω))D, $(eltype(Ω))}:")
    else
        X = set(Ω)
        dm = markers(Ω)

        printstyled(io, "Domain"; bold = true, color = :cyan)
        print(io, " {")
        printstyled(io, "$(dim(Ω))D", color = :yellow)
        print(io, ", ")
        printstyled(io, "$(eltype(Ω))", color = :yellow)
        println(io, "}:")

        println(io)
        pp_indented = with_indent(pp, 1)
        print_section_header(pp_indented, "Set:")

        D = dim(X)
        topodim = topo_dim(X)
        pp_double_indent = with_indent(pp, 2)

        if D == 1
            collapsed = X.collapsed[1]
            print(io, "    ")
            if collapsed
                print_colored(pp, "Point", color = :yellow)
                print(io, " at ")
                print_value(pp, X.box[1][1])
            else
                print_colored(pp, "Interval", color = :yellow)
                print(io, " ")
                print_interval(pp, X.box[1][1], X.box[1][2])
            end
            println(io)
        else
            if topodim < D
                print(io, "    ")
                print_colored(pp, "Topological dimension: $topodim", color = :yellow)
                println(io)
            end

            for i in 1:D
                label = get_dimension_label(i)
                print_dimension_info(
                    pp_double_indent, label, X.box[i][1], X.box[i][2], X.collapsed[i])
            end
        end

        println(io)
        print_section_header(pp_indented, "Markers:")

        n_sym = length(symbols(Ω))
        n_tup = length(tuples(Ω))
        n_cond = length(conditions(Ω))
        total = n_sym + n_tup + n_cond

        if total == 0
            print(io, "    ")
            print_empty_message(pp, "(none)")
            println(io)
        else
            print_marker_summary(with_indent(pp, 2), n_sym, n_tup, n_cond)

            print(io, "    ")
            print_labels_list(pp, collect(labels(Ω)))
            println(io)
        end

        remove_trailing_newline(io)
    end
end
