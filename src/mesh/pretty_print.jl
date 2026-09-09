# Pretty printing utilities for mesh types

"""
    print_mesh_header(pp::PrettyPrinter, mesh_type::String, D::Int, T::Type, npts) -> Nothing

Print a styled header for mesh objects.
"""
function print_mesh_header(pp::PrettyPrinter, mesh_type::String, D::Int, T::Type, npts)
    printstyled(pp.io, mesh_type; bold=true, color=:cyan)
    print(pp.io, " {")
    printstyled(pp.io, "$(D)D"; color=:yellow)
    print(pp.io, ", ")
    printstyled(pp.io, "$T"; color=:yellow)
    return print(pp.io, "}")
end

"""
    print_mesh_summary(pp::PrettyPrinter, npts, topodim::Int, collapsed::Bool) -> Nothing

Print a summary line for mesh properties (number of points, topology).
"""
function print_mesh_summary(pp::PrettyPrinter, npts, topodim::Int, collapsed::Bool)
    print_indent(pp)

    # Print number of points
    if npts isa Tuple
        total_pts = prod(npts)
        printstyled(pp.io, "$total_pts points"; color=:blue)
        print(pp.io, " (")
        print_joined(pp, npts; sep=" × ") do n
            return print(pp.io, n)
        end
        print(pp.io, ")")
    else
        printstyled(pp.io, "$npts points"; color=:blue)
    end

    # Print topological dimension if relevant
    if collapsed
        print(pp.io, " • ")
        printstyled(pp.io, "collapsed"; color=:light_black)
    elseif topodim < (npts isa Tuple ? length(npts) : 1)
        print(pp.io, " • ")
        printstyled(pp.io, "topological dim $topodim"; color=:yellow)
    end

    return println(pp.io)
end

"""
    print_mesh_domain_info(pp::PrettyPrinter, set::CartesianProduct) -> Nothing

Print the domain information for a mesh.
"""
function print_mesh_domain_info(pp::PrettyPrinter, set::CartesianProduct)
    print_indent(pp)
    printstyled(pp.io, "Domain: "; color=:light_black)
    # The two-argument `show` is the embeddable one-liner, which already renders
    # `[a, b] × [c, d]` and collapses degenerate axes to a single value. No `:compact`
    # context needed to ask for it any more: it is the only thing this method does
    # (gpena/Bramble.jl#45).
    show(pp.io, set)
    return println(pp.io)
end

"""
    print_mesh_spacing_info(pp::PrettyPrinter, uniform::Bool, hmax) -> Nothing
    print_mesh_spacing_info(pp::PrettyPrinter, uniform::Tuple{Vararg{Bool}}, hmax) -> Nothing

Print mesh spacing information and maximum cell diagonal.
"""
# One method per shape of `uniform` rather than one branching on it: a 1D mesh answers
# with a `Bool` and an nD one with a tuple, which is a dispatch decision
# (gpena/Bramble.jl#47).
function print_mesh_spacing_info(pp::PrettyPrinter, uniform::Bool, hmax)
    _print_spacing_prefix(pp)
    print(pp.io, uniform ? "uniform" : "non-uniform")
    return _print_spacing_suffix(pp, hmax)
end

function print_mesh_spacing_info(pp::PrettyPrinter, uniform::Tuple{Vararg{Bool}}, hmax)
    _print_spacing_prefix(pp)
    if all(uniform)
        print(pp.io, "uniform")
    else
        print(pp.io, "mixed (")
        print_joined(pp, enumerate(uniform)) do (i, u)
            return print(pp.io, get_dimension_label(i), ": ", u ? "uniform" : "non-uniform")
        end
        print(pp.io, ")")
    end
    return _print_spacing_suffix(pp, hmax)
end

@inline function _print_spacing_prefix(pp::PrettyPrinter)
    print_indent(pp)
    printstyled(pp.io, "Spacing: "; color=:light_black)
    return nothing
end

@inline function _print_spacing_suffix(pp::PrettyPrinter, hmax)
    print(pp.io, " • ")
    printstyled(pp.io, "h"; color=:magenta)
    print(pp.io, "ₘₐₓ = ")
    printstyled(pp.io, "$(round(hmax, digits=6))"; color=:blue)
    return println(pp.io)
end

"""
    print_mesh_markers(pp::PrettyPrinter, mesh_markers::MeshMarkers) -> Nothing

Print marker information and labeled point counts for a mesh.
"""
function print_mesh_markers(pp::PrettyPrinter, mesh_markers::MeshMarkers)
    n_markers = length(mesh_markers)

    if n_markers == 0
        print_indent(pp)
        printstyled(pp.io, "Markers: "; color=:light_black)
        printstyled(pp.io, "(none)"; color=:light_black)
        println(pp.io)
        return nothing
    end

    print_indent(pp)
    printstyled(pp.io, "Markers: "; color=:light_black)
    printstyled(pp.io, "$n_markers label$(n_markers == 1 ? "" : "s")"; color=:yellow)
    print(pp.io, " • ")

    # Print labels
    print_joined(pp, keys(mesh_markers)) do label
        printstyled(pp.io, ":$label"; color=:green)
        marked_count = count(mesh_markers[label])
        if marked_count > 0
            print(pp.io, " (")
            printstyled(pp.io, "$marked_count"; color=:blue)
            print(pp.io, ")")
        end
    end
    return println(pp.io)
end
