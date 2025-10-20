# This macro is taken from Lazy.jl
macro forward(ex, fs)
	@capture(ex, T_.field_) || error("Syntax: @forward T.x f, g, h")
	T = esc(T)
	fs = isexpr(fs, :tuple) ? map(esc, fs.args) : [esc(fs)]
	:($([:($f(x::$T, args...; kwargs...) = (Base.@_inline_meta; $f(x.$field, args...; kwargs...)))
		 for f in fs]...);
	nothing)
end