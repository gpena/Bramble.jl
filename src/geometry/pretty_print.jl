# Pretty printing utilities for geometry types

"""
	PrettyPrinter

A helper struct for managing pretty printing with colors and indentation.
"""
struct PrettyPrinter
	io::IO
	compact::Bool
	indent_level::Int
end

@inline PrettyPrinter(io::IO) = PrettyPrinter(io, get(io, :compact, false), 0)

"""
	with_indent(pp::PrettyPrinter, levels::Int=1)

Returns a new PrettyPrinter with increased indentation.
"""
@inline with_indent(pp::PrettyPrinter, levels::Int = 1) = PrettyPrinter(pp.io, pp.compact, pp.indent_level + levels)

"""
	print_indent(pp::PrettyPrinter)

Prints the current indentation level.
"""
@inline function print_indent(pp::PrettyPrinter)
	pp.indent_level == 0 && return
	for _ in 1:pp.indent_level
		print(pp.io, "  ")
	end
end

"""
	print_colored(pp::PrettyPrinter, text; color=:default, bold=false)

Prints colored text with the current indentation.
"""
@inline function print_colored(pp::PrettyPrinter, text; color = :default, bold = false)
	if color == :default
		print(pp.io, text)
	else
		printstyled(pp.io, text; color = color, bold = bold)
	end
end

"""
	println_colored(pp::PrettyPrinter, text; color=:default, bold=false)

Prints colored text with newline.
"""
@inline function println_colored(pp::PrettyPrinter, text; color = :default, bold = false)
	print_colored(pp, text; color = color, bold = bold)
	println(pp.io)
end

"""
	print_header(pp::PrettyPrinter, title::String, type_info::String="")

Prints a header with title and optional type information.
"""
function print_header(pp::PrettyPrinter, title::String, type_info::String = "")
	print_indent(pp)
	printstyled(pp.io, title; bold = true, color = :cyan)
	if !isempty(type_info)
		print(pp.io, " ")
		printstyled(pp.io, type_info; color = :yellow)
	end
	println(pp.io)
end

"""
	print_section_header(pp::PrettyPrinter, title::String)

Prints a section header.
"""
function print_section_header(pp::PrettyPrinter, title::String)
	print_indent(pp)
	printstyled(pp.io, title; bold = true, color = :light_blue)
	println(pp.io)
end

"""
	print_subsection_header(pp::PrettyPrinter, title::String, count::Int=0)

Prints a subsection header with optional count.
"""
function print_subsection_header(pp::PrettyPrinter, title::String, count::Int = 0)
	print_indent(pp)
	printstyled(pp.io, title; bold = true, color = :yellow)
	if count > 0
		print(pp.io, " ($count)")
	end
	println(pp.io, ":")
end

"""
	print_key_value(pp::PrettyPrinter, key::String, value::String; 
					key_color=:green, value_color=:blue, separator=" => ")

Prints a key-value pair with colors.
"""
function print_key_value(pp::PrettyPrinter, key::String, value::String;
						 key_color = :green, value_color = :blue, separator = " => ")
	print_indent(pp)
	printstyled(pp.io, key; color = key_color)
	print(pp.io, separator)
	printstyled(pp.io, value; color = value_color)
	println(pp.io)
end

"""
	print_label(pp::PrettyPrinter, label::Symbol)

Prints a colored label (marker or dimension name).
"""
@inline function print_label(pp::PrettyPrinter, label::Symbol)
	printstyled(pp.io, ":$label"; color = :green)
end

"""
	print_value(pp::PrettyPrinter, value; color=:blue)

Prints a colored value.
"""
@inline function print_value(pp::PrettyPrinter, value; color = :blue)
	printstyled(pp.io, "$value"; color = color)
end

"""
	print_interval(pp::PrettyPrinter, min_val, max_val; collapsed=false)

Prints an interval with proper formatting.
"""
function print_interval(pp::PrettyPrinter, min_val, max_val; collapsed = false)
	if collapsed
		printstyled(pp.io, "$min_val"; color = :blue)
		printstyled(pp.io, " (collapsed)"; color = :light_black)
	else
		print(pp.io, "[")
		printstyled(pp.io, "$min_val, $max_val"; color = :blue)
		print(pp.io, "]")
	end
end

"""
	print_dimension_info(pp::PrettyPrinter, label::String, min_val, max_val, collapsed::Bool)

Prints dimension information (coordinate label and range).
"""
function print_dimension_info(pp::PrettyPrinter, label::String, min_val, max_val, collapsed::Bool)
	print_indent(pp)
	printstyled(pp.io, label; color = :green)
	print(pp.io, ": ")
	print_interval(pp, min_val, max_val; collapsed = collapsed)
	println(pp.io)
end

"""
	print_empty_message(pp::PrettyPrinter, message::String="(none)")

Prints an empty/none message in gray.
"""
function print_empty_message(pp::PrettyPrinter, message::String = "(none)")
	print_indent(pp)
	printstyled(pp.io, message; color = :light_black)
	println(pp.io)
end

"""
	print_marker_summary(pp::PrettyPrinter, n_sym::Int, n_tup::Int, n_cond::Int)

Prints a summary of marker counts.
"""
function print_marker_summary(pp::PrettyPrinter, n_sym::Int, n_tup::Int, n_cond::Int)
	total = n_sym + n_tup + n_cond
	print_indent(pp)
	printstyled(pp.io, "$total marker$(total == 1 ? "" : "s")"; color = :yellow)
	print(pp.io, " (")

	# Build parts string without intermediate array allocation
	first = true
	if n_sym > 0
		print(pp.io, "$n_sym symbol$(n_sym == 1 ? "" : "s")")
		first = false
	end
	if n_tup > 0
		first || print(pp.io, ", ")
		print(pp.io, "$n_tup tuple$(n_tup == 1 ? "" : "s")")
		first = false
	end
	if n_cond > 0
		first || print(pp.io, ", ")
		print(pp.io, "$n_cond function$(n_cond == 1 ? "" : "s")")
	end

	println(pp.io, ")")
end

"""
	print_labels_list(pp::PrettyPrinter, labels; prefix="Labels: ")

Prints a comma-separated list of labels.
"""
function print_labels_list(pp::PrettyPrinter, labels; prefix = "Labels: ")
	print_indent(pp)
	printstyled(pp.io, prefix; color = :light_black)

	for (i, lbl) in enumerate(labels)
		printstyled(pp.io, ":$lbl"; color = :green)
		i < length(labels) && print(pp.io, ", ")
	end
	println(pp.io)
end

"""
	remove_trailing_newline(io::IO)

Removes the trailing newline from output.
"""
@inline remove_trailing_newline(io::IO) = print(io, "\b")

# Precomputed dimension labels to avoid repeated allocations
const DIMENSION_LABELS = ("x", "y", "z", "w", "v", "u")

"""
	get_dimension_label(i::Int)

Returns the label for the i-th dimension.
"""
@inline function get_dimension_label(i::Int)
	return i <= length(DIMENSION_LABELS) ? DIMENSION_LABELS[i] : "x$i"
end
