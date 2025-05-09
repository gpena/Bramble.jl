@inline style_whitespace(name, max_length) = repeat(" ", max(max_length - length(name), 0))

function style_field(name::String, value; max_length::Int)
	whitespace = style_whitespace(name, max_length)
	prefix = styled("{yellow,bold:$whitespace$(name)}: ")
	suffix = value isa Base.AnnotatedString ? value : string(value)
	return prefix * suffix
end

@inline style_real_space(::Val{1}) = "ℝ"
@inline style_real_space(::Val{2}) = "ℝ²"
@inline style_real_space(::Val{3}) = "ℝ³"

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

@inline function style_title(name; max_length = 0)
	whitespace = style_whitespace(name, max_length)
	return styled("$whitespace{red,bold,underline:$(name)}")
end

@inline style_submesh_title(name) = styled("{bold,underline:$(name)}")

function format_with_underscores(n::Integer)
	s = string(abs(n))
	len = length(s)
	len <= 3 && return string(n) # Return original string if short

	# Calculate number of digits in the first group (1, 2, or 3)
	first_group_len = mod1(len, 3) # mod1 ensures result is 1, 2, or 3

	# Use an IOBuffer for efficient string building
	buf = IOBuffer()

	# Write the first group
	write(buf, SubString(s, 1, first_group_len))

	# Write subsequent groups prefixed with '_'
	for i in (first_group_len + 1):3:len
		write(buf, '_')
		write(buf, SubString(s, i, i + 2))
	end

	# Add sign if negative
	sign_str = n < 0 ? "-" : ""
	return sign_str * String(take!(buf))
end