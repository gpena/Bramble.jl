# Pretty printing utilities for mesh types

"""
	print_mesh_header(pp::PrettyPrinter, mesh_type::String, D::Int, T::Type, npts)

Prints a colored header for mesh objects.
"""
function print_mesh_header(pp::PrettyPrinter, mesh_type::String, D::Int, T::Type, npts)
	printstyled(pp.io, mesh_type; bold = true, color = :cyan)
	print(pp.io, " {")
	printstyled(pp.io, "$(D)D"; color = :yellow)
	print(pp.io, ", ")
	printstyled(pp.io, "$T"; color = :yellow)
	print(pp.io, "}")
end

"""
	print_mesh_summary(pp::PrettyPrinter, npts, topodim::Int, collapsed::Bool)

Prints a summary line for mesh properties (number of points, topology).
"""
function print_mesh_summary(pp::PrettyPrinter, npts, topodim::Int, collapsed::Bool)
	print_indent(pp)

	# Print number of points
	if npts isa Tuple
		total_pts = prod(npts)
		printstyled(pp.io, "$total_pts points"; color = :blue)
		print(pp.io, " (")
		for (i, n) in enumerate(npts)
			print(pp.io, n)
			i < length(npts) && print(pp.io, " × ")
		end
		print(pp.io, ")")
	else
		printstyled(pp.io, "$npts points"; color = :blue)
	end

	# Print topological dimension if relevant
	if collapsed
		print(pp.io, " • ")
		printstyled(pp.io, "collapsed"; color = :light_black)
	elseif topodim < (npts isa Tuple ? length(npts) : 1)
		print(pp.io, " • ")
		printstyled(pp.io, "topological dim $topodim"; color = :yellow)
	end

	println(pp.io)
end

"""
	print_mesh_domain_info(pp::PrettyPrinter, set::CartesianProduct)

Prints the domain information for a mesh.
"""
function print_mesh_domain_info(pp::PrettyPrinter, set::CartesianProduct{D,T}) where {D,T}
	print_indent(pp)
	printstyled(pp.io, "Domain: "; color = :light_black)

	if D == 1
		a, b = set.box[1]
		collapsed = set.collapsed[1]
		if collapsed
			printstyled(pp.io, "Point($a)"; color = :green)
		else
			print(pp.io, "[")
			printstyled(pp.io, "$a, $b"; color = :green)
			print(pp.io, "]")
		end
	else
		for i in 1:D
			i > 1 && print(pp.io, " × ")
			a, b = set.box[i]
			collapsed = set.collapsed[i]
			if collapsed
				printstyled(pp.io, "$a"; color = :green)
			else
				print(pp.io, "[")
				printstyled(pp.io, "$a, $b"; color = :green)
				print(pp.io, "]")
			end
		end
	end
	println(pp.io)
end

"""
	print_mesh_spacing_info(pp::PrettyPrinter, uniform::Bool, hmax)

Prints mesh spacing information.
"""
function print_mesh_spacing_info(pp::PrettyPrinter, uniform::Union{Bool,Tuple{Vararg{Bool}}}, hmax)
	print_indent(pp)
	printstyled(pp.io, "Spacing: "; color = :light_black)

	if uniform isa Bool
		print(pp.io, uniform ? "uniform" : "non-uniform")
	else
		# For multidimensional meshes
		all_uniform = all(uniform)
		if all_uniform
			print(pp.io, "uniform")
		else
			print(pp.io, "mixed (")
			for (i, u) in enumerate(uniform)
				label = get_dimension_label(i)
				print(pp.io, "$label: ", u ? "uniform" : "non-uniform")
				i < length(uniform) && print(pp.io, ", ")
			end
			print(pp.io, ")")
		end
	end

	print(pp.io, " • ")
	printstyled(pp.io, "h"; color = :magenta)
	print(pp.io, "ₘₐₓ = ")
	printstyled(pp.io, "$(round(hmax, digits=6))"; color = :blue)
	println(pp.io)
end

"""
	print_mesh_markers(pp::PrettyPrinter, mesh_markers::MeshMarkers)

Prints marker information for a mesh.
"""
function print_mesh_markers(pp::PrettyPrinter, mesh_markers::MeshMarkers)
	n_markers = length(mesh_markers)

	if n_markers == 0
		print_indent(pp)
		printstyled(pp.io, "Markers: "; color = :light_black)
		printstyled(pp.io, "(none)"; color = :light_black)
		println(pp.io)
		return
	end

	print_indent(pp)
	printstyled(pp.io, "Markers: "; color = :light_black)
	printstyled(pp.io, "$n_markers label$(n_markers == 1 ? "" : "s")"; color = :yellow)
	print(pp.io, " • ")

	# Print labels
	labels_list = collect(keys(mesh_markers))
	for (i, label) in enumerate(labels_list)
		printstyled(pp.io, ":$label"; color = :green)

		# Count marked points
		marked_count = count(mesh_markers[label])
		if marked_count > 0
			print(pp.io, " (")
			printstyled(pp.io, "$marked_count"; color = :blue)
			print(pp.io, ")")
		end

		i < length(labels_list) && print(pp.io, ", ")
	end
	println(pp.io)
end

"""
	print_submesh_info(pp::PrettyPrinter, submeshes::NTuple{D}, show_details::Bool=false) where D

Prints information about submeshes in an nD mesh.
"""
function print_submesh_info(pp::PrettyPrinter, submeshes::NTuple{D}, show_details::Bool = false) where D
	if !show_details
		return
	end

	println(pp.io)
	print_section_header(with_indent(pp, 1), "Submeshes:")
	println(pp.io)

	pp_indented = with_indent(pp, 2)
	for i in 1:D
		label = get_dimension_label(i)
		submesh = submeshes[i]
		npts = npoints(submesh)
		a, b = tails(set(submesh))

		print_indent(pp_indented)
		printstyled(pp.io, "$label"; color = :green)
		print(pp.io, ": ")
		printstyled(pp.io, "$npts points"; color = :blue)
		print(pp.io, " on [")
		printstyled(pp.io, "$a, $b"; color = :magenta)
		print(pp.io, "]")
		println(pp.io)
	end
end
