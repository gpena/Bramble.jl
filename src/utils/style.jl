@inline style_whitespace(name, max_length) = repeat(" ", max(max_length - length(name), 0))

function style_field(name::String, value; max_length::Int)
	whitespace = style_whitespace(name, max_length)
	prefix = styled("{yellow,bold:$whitespace$(name)}: ")
	suffix = value isa Base.AnnotatedString ? value : string(value)
	return prefix * suffix
end

@inline style_join(fields...) = join(fields, "\n")

@inline max_length_fields(labels) = max(length.(labels)...)

function style_color_sets()
	return (:red, :green, :blue)
end

function style_color_markers()
	return (:cyan, :green, :magenta, :blue, :red)
end

function color_markers(labels)
	colors = style_color_markers()
	num_colors = length(colors)

	styled_labels = [let color_sym = colors[mod1(i, num_colors)]
						 styled"{$color_sym:$(label)}"
					 end
					 for (i, label) in enumerate(labels)]

	labels_styled_combined = join(styled_labels, ", ")

	return labels_styled_combined
end

@inline function style_mesh_title(name; max_length = 0)
	whitespace = style_whitespace(name, max_length)
	return styled("$whitespace{red,bold,underline:$(name)}")
end

@inline style_submesh_title(name) = styled("{bold,underline:$(name)}")
