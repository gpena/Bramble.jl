"""
	@forward T.field f, g, h

A macro for automatically forwarding method calls from a type to one of its fields.
This macro is adapted from Lazy.jl and generates delegation methods that forward
function calls to a specific field of a struct.

# Syntax

```julia
@forward TypeName.fieldname (func1, func2, func3, ...)
```

# Arguments

  - `TypeName.fieldname`: The type and field to forward calls to
  - Function list: Single function or tuple of functions to forward

# Behavior

For each function `f` in the list, generates a method:
```julia
f(x::TypeName, args...; kwargs...) = f(x.fieldname, args...; kwargs...)
```

The generated methods are marked with `@inline` for performance.

# Examples

```julia
struct Container
    data::Vector{Float64}
end

# Forward length, size, and eltype to the data field
@forward Container.data (Base.length, Base.size, Base.eltype)

c = Container([1.0, 2.0, 3.0])
length(c)  # Returns 3 (calls length(c.data))
size(c)    # Returns (3,) (calls size(c.data))
```

# Use Cases

  - Delegating collection interface methods (length, iterate, etc.)
  - Forwarding mathematical operations to wrapped types
  - Reducing boilerplate for wrapper types

# Notes

  - This macro is taken from Lazy.jl
  - Generated methods include `@inline` hint for optimization
  - Supports both regular and keyword arguments
"""
macro forward(ex, fs)
	@capture(ex, T_.field_) || error("Syntax: @forward T.x f, g, h")
	T = esc(T)
	fs = isexpr(fs, :tuple) ? map(esc, fs.args) : [esc(fs)]
	:($([:($f(x::$T, args...; kwargs...) = (Base.@_inline_meta; $f(x.$field, args...; kwargs...)))
		 for f in fs]...);
	nothing)
end