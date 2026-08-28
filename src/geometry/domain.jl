# Abstract supertype for all domain-related types.
abstract type DomainBaseType end

"""
	$(TYPEDEF)

Represents a computational domain, combining a geometric set (e.g., [`CartesianProduct`](@ref)) with a collection of labeled [`DomainMarkers`](@ref).

# Fields

$(FIELDS)

# Related Types

- Use `mesh` to discretize a [`Domain`](@ref) into a computational mesh.
- See [`CartesianProduct`](@ref) for the underlying geometric representation.
- See [`DomainMarkers`](@ref) for marker management.
"""
struct Domain{SetType,MarkersType} <: DomainBaseType
	"the geometric set defining the domain's extent (e.g., a [`CartesianProduct`](@ref))."
	set::SetType
	"a [`DomainMarkers`](@ref) object containing all labeled regions for this domain."
	markers::MarkersType
end

"""
	$(SIGNATURES)

Returns the [`DomainMarkers`](@ref) object associated with the [`Domain`](@ref) `Ω`.
"""
@inline markers(Ω::Domain) = Ω.markers

"""
	$(SIGNATURES)

Returns the set of single-symbol markers associated with the [`Domain`](@ref) `Ω`.
"""
@inline symbols(Ω::Domain) = symbols(markers(Ω))

"""
	$(SIGNATURES)

Returns the set of symbol-tuple markers associated with the [`Domain`](@ref) `Ω`.
"""
@inline tuples(Ω::Domain) = tuples(markers(Ω))

"""
	$(SIGNATURES)

Returns the set of condition-based markers associated with the [`Domain`](@ref) `Ω`.
"""
@inline conditions(Ω::Domain) = conditions(markers(Ω))

"""
	$(SIGNATURES)

Returns a generator that yields the labels (`Symbol`) of all markers associated with the [`Domain`](@ref) `Ω`.
"""
@inline labels(Ω::Domain) = labels(markers(Ω))

"""
	$(SIGNATURES)

Returns a generator that yields the identifiers (`Symbol`, `Set{Symbol}`, or `BrambleFunction`) of all markers in the [`Domain`](@ref) `Ω`.
"""
@inline function marker_identifiers(Ω::Domain)
	return (identifier(marker) for marker in Iterators.flatten((symbols(Ω), tuples(Ω), conditions(Ω))))
end

"""
	$(SIGNATURES)

Returns a generator yielding the identifiers of single-symbol markers.
"""
@inline function marker_symbols(Ω::Domain)
	return (identifier(marker) for marker in symbols(Ω))
end

"""
	$(SIGNATURES)

Returns a generator yielding the identifiers of symbol-tuple markers.
"""
@inline function marker_tuples(Ω::Domain)
	return (identifier(marker) for marker in tuples(Ω))
end

"""
	$(SIGNATURES)

Returns a generator yielding the identifiers (functions) of condition-based markers.
"""
@inline function marker_conditions(Ω::Domain)
	return (identifier(marker) for marker in conditions(Ω))
end

"""
	$(SIGNATURES)

Returns a generator with all marker labels on [`Domain`](@ref) `Ω`.
"""
@inline label_identifiers(Ω::Domain) = label_identifiers(markers(Ω))

"""
	$(SIGNATURES)

Returns a generator with the labels of symbol markers on [`Domain`](@ref) `Ω`.
"""
@inline label_symbols(Ω::Domain) = label_symbols(markers(Ω))

"""
	$(SIGNATURES)

Returns a generator with the labels of symbol-tuple markers on [`Domain`](@ref) `Ω`.
"""
@inline label_tuples(Ω::Domain) = label_tuples(markers(Ω))

"""
	$(SIGNATURES)

Returns a generator with the labels of function condition markers on [`Domain`](@ref) `Ω`.
"""
@inline label_conditions(Ω::Domain) = label_conditions(markers(Ω))

"""
	$(SIGNATURES)

Returns a [`Domain`](@ref) from a [`CartesianProduct`](@ref).

- `domain(X)`: Assumes a default `:boundary` marker covering all boundaries of `X`.
- `domain(X, markers::DomainMarkers)`: Constructs a domain with explicit markers.
- `domain(X, pairs...)`: Constructs a domain with markers defined by label-identifier pairs.
- `domain(space_set, time_set, pairs...)`: Constructs a spatio-temporal domain.
"""
@inline domain(X::CartesianProduct) = Domain(X, markers(X, :boundary => get_boundary_symbols(X)))
@inline domain(X::CartesianProduct, markers::DomainMarkers) = Domain(X, markers)
@inline domain(X::CartesianProduct, pairs::Pair...) = domain(X, markers(X, pairs...))
@inline domain(space_set::CartesianProduct, time_set::CartesianProduct{1}, pairs::Pair...) = domain(space_set, markers(space_set, time_set, pairs...))

"""
	(Ω::Domain)(t::Number)

Evaluates a time-dependent [`Domain`](@ref) at time `t`, returning a time-evaluated [`Domain`](@ref).
"""
@inline (Ω::Domain)(t::Number) = Domain(set(Ω), markers(Ω)(t))

"""
	$(SIGNATURES)

Returns the [`CartesianProduct`](@ref) geometric set associated with the [`Domain`](@ref) `Ω`.
"""
@inline set(Ω::Domain) = Ω.set

"""
	$(SIGNATURES)

Returns the dimension of the space where the [`Domain`](@ref) `Ω` is embedded.
"""
@inline dim(Ω::Domain) = dim(set(Ω))
@inline dim(::Type{<:Domain{SetType}}) where {SetType} = dim(SetType)

"""
	$(SIGNATURES)

Returns the topological dimension of [`Domain`](@ref) `Ω`.
"""
@inline topo_dim(Ω::Domain) = topo_dim(set(Ω))

"""
	$(SIGNATURES)

Returns the element type of the bounds defining [`Domain`](@ref) `Ω`.
"""
@inline eltype(Ω::Domain) = eltype(set(Ω))
@inline eltype(::Type{<:Domain{SetType}}) where {SetType} = eltype(SetType)

"""
	$(SIGNATURES)

Determines the coordinate point type within a [`Domain`](@ref) space.
"""
@inline point_type(Ω::Domain) = point_type(set(Ω))
@inline point_type(::Type{<:Domain{SetType}}) where {SetType} = point_type(SetType)

"""
	$(SIGNATURES)

Returns the total number of markers defined on [`Domain`](@ref) `Ω`.
"""
@inline Base.length(Ω::Domain) = length(markers(Ω))

"""
	$(SIGNATURES)

Returns `true` if [`Domain`](@ref) `Ω` has no markers attached.
"""
@inline Base.isempty(Ω::Domain) = isempty(markers(Ω))

"""
	$(SIGNATURES)

Returns the center point of [`Domain`](@ref) `Ω` as an `SVector{D,T}`.
"""
@inline center(Ω::Domain) = center(set(Ω))

"""
	Base.in(x, Ω::Domain)

Returns `true` if point `x` is contained in the closed [`Domain`](@ref) `Ω`.
"""
@inline Base.in(x, Ω::Domain) = x ∈ set(Ω)

"""
	$(SIGNATURES)

Returns the boundary tails/intervals of the underlying set of [`Domain`](@ref) `Ω`.
"""
@inline tails(Ω::Domain) = tails(set(Ω))
@inline tails(Ω::Domain, i::Integer) = tails(set(Ω), i)

"""
	$(SIGNATURES)

Returns whether the underlying set of [`Domain`](@ref) `Ω` is collapsed.
"""
@inline is_collapsed(Ω::Domain) = is_collapsed(set(Ω))

@inline (Ω::Domain)(i::Integer) = set(Ω)(i)

"""
	$(SIGNATURES)

Returns the `i`-th 1D [`CartesianProduct`](@ref) component of the set associated with [`Domain`](@ref) `Ω`.
"""
@inline projection(Ω::Domain, i::Integer) = projection(set(Ω), i)

"""
	$(SIGNATURES)

Returns a tuple of default boundary symbols for a [`CartesianProduct`](@ref) or [`Domain`](@ref).

- 1D ``[x_1, x_2]``: `(:left, :right)`
- 2D ``[x_1, x_2] \\times [y_1, y_2]``: `(:bottom, :top, :left, :right)`
- 3D ``[x_1, x_2] \\times [y_1, y_2] \\times [z_1, z_2]``: `(:bottom, :top, :back, :front, :left, :right)`
"""
@inline get_boundary_symbols(Ω::Domain) = get_boundary_symbols(set(Ω))
@inline get_boundary_symbols(::CartesianProduct{1}) = (:left, :right)
@inline get_boundary_symbols(::CartesianProduct{2}) = (:bottom, :top, :left, :right)
@inline get_boundary_symbols(::CartesianProduct{3}) = (:bottom, :top, :back, :front, :left, :right)
@inline get_boundary_symbols(::Type{<:CartesianProduct{1}}) = (:left, :right)
@inline get_boundary_symbols(::Type{<:CartesianProduct{2}}) = (:bottom, :top, :left, :right)
@inline get_boundary_symbols(::Type{<:CartesianProduct{3}}) = (:bottom, :top, :back, :front, :left, :right)
@inline get_boundary_symbols(D::Integer) = get_boundary_symbols(CartesianProduct{D})
@noinline function get_boundary_symbols(::Type{<:CartesianProduct{D}}) where D
	error("get_boundary_symbols is not defined for $(D)D domains. " *
		  "Provide explicit boundary names via the markers() interface.")
end
@inline get_boundary_symbols(::Type{<:Domain{SetType}}) where {SetType} = get_boundary_symbols(SetType)

"""
	Base.show(io::IO, Ω::Domain)

Custom display for [`Domain`](@ref) objects, combining set geometry and marker information with colors.
"""
function Base.show(io::IO, Ω::Domain)
	pp = PrettyPrinter(io)

	if pp.compact
		# Compact mode for arrays/collections
		print(io, "Domain{$(dim(Ω))D, $(eltype(Ω))}:")
	else
		# Detailed mode
		X = set(Ω)
		dm = markers(Ω)

		# Header
		printstyled(io, "Domain"; bold = true, color = :cyan)
		print(io, " {")
		printstyled(io, "$(dim(Ω))D", color = :yellow)
		print(io, ", ")
		printstyled(io, "$(eltype(Ω))", color = :yellow)
		println(io, "}:")

		# Set information
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
				print_dimension_info(pp_double_indent, label, X.box[i][1], X.box[i][2], X.collapsed[i])
			end
		end

		# Markers information
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
			# Show summary counts
			print(io, "    ")
			print_colored(pp, "$total marker$(total == 1 ? "" : "s")", color = :yellow)
			print(io, " (")
			first = true
			if n_sym > 0
				print(io, "$n_sym symbol$(n_sym == 1 ? "" : "s")")
				first = false
			end
			if n_tup > 0
				first || print(io, ", ")
				print(io, "$n_tup tuple$(n_tup == 1 ? "" : "s")")
				first = false
			end
			if n_cond > 0
				first || print(io, ", ")
				print(io, "$n_cond function$(n_cond == 1 ? "" : "s")")
			end
			println(io, ")")

			# Show marker labels
			print(io, "    ")
			print_labels_list(pp, collect(labels(Ω)))
			println(io)
		end

		# Remove trailing newline
		remove_trailing_newline(io)
	end
end
