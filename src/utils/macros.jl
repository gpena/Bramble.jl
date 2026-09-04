"""
    @forward T.field functions
    @forward T.field (f, g, ...)

Generate delegating method definitions forwarding function calls on type `T` to `x.field`.

For each supplied function `f`, generates an inlined method:
```julia
@inline f(x::T, args...; kwargs...) = f(x.field, args...; kwargs...)
```

# Examples
```jldoctest
using Bramble: @forward

struct Container
    data::Vector{Float64}
end

@forward Container.data (Base.length, Base.size)

c = Container([1.0, 2.0, 3.0])
length(c) == 3 && size(c) == (3,)

# output
true
```
"""
macro forward(ex, fs)
    if !(Meta.isexpr(ex, :.) && length(ex.args) == 2 && ex.args[2] isa QuoteNode)
        error("Syntax: @forward T.x f  or  @forward T.x (f, g, h)")
    end
    T = esc(ex.args[1])
    field = ex.args[2].value
    fs = Meta.isexpr(fs, :tuple) ? map(esc, fs.args) : [esc(fs)]
    :($([:(@inline $f(x::$T, args...; kwargs...) = $f(x.$field, args...; kwargs...))
         for f in fs]...);
    nothing)
end
