"""
	$(TYPEDEF)

Internal structure to wrap around functions to make them more type agnostic. It uses `FunctionWrappers.jl` wrap functions.

# Fields

	$(FIELDS)

to provide functions calculated on `ArgsType`. The type arguments are `hastime` to indicate if the function is time-dependent and `CoType`, the time of the codomain of the function. It also stores the type of domain of the space part of the function ([CartesianProduct](@ref))
"""
struct BrambleFunction{ArgsType,hastime,CoType,DType}
	"a `FunctionWrapper` wrapper for the function"
	wrapped::FunctionWrapper{CoType,Tuple{ArgsType}}
	"domain of the function; the standard is a [CartesianProduct](@ref)"
	domain::DType
end

"""
	has_time(f::BrambleFunction)

Checks if a [BrambleFunction](@ref) is time-dependent by inspecting its `hastime` type parameter.
Returns `true` if the function is time-dependent, `false` otherwise.
"""
@inline has_time(::BrambleFunction{ArgsType,hastime}) where {ArgsType,hastime} = hastime
@inline has_time(::Type{<:BrambleFunction{ArgsType,hastime}}) where {ArgsType,hastime} = hastime

"""
	$(SIGNATURES)

Internal helper to determine the expected argument type for a function based on a domain `X`.

  - If the domain is 1D (e.g., a line), the argument type is a scalar `T`.
  - If the domain is D-dimensional, the argument type is an `NTuple{D,T}`.
  - `T` is the element type of the domain `X`.

# Examples

```julia
X_1d = interval(0.0, 1.0)  # D=1, T=Float64
_get_args_type(X_1d)  # Returns Float64

X_2d = interval(0.0, 1.0) × interval(0.0, 1.0)  # D=2, T=Float64  
_get_args_type(X_2d)  # Returns NTuple{2,Float64}
```
"""
@inline function _get_args_type(X)
	D = dim(X)
	T = eltype(X)
	return ifelse(D == 1, T, NTuple{D,T})
end

"""
	$(SIGNATURES)

A low-level constructor for creating a `BrambleFunction`. It wraps a given Julia function `f`
into a `FunctionWrapper` and bundles it with its domain.

# Arguments

  - `f`: The function to wrap.
  - `X`: The spatial domain used to infer the argument type of `f`.
  - `hastime`: A boolean indicating if the function is time-dependent.
  - `CoType`: The codomain (return) type of the function `f`.
  - `domain`: The actual domain object to store in the struct (typically `X`).
"""
function bramble_function_with_domain(f, X, hastime, CoType; domain = X)
	ArgType = _get_args_type(X)
	wrapped_f = FunctionWrapper{CoType,Tuple{ArgType}}(f)
	return BrambleFunction{ArgType,hastime,CoType,typeof(domain)}(wrapped_f, domain)
end

"""
	$(SIGNATURES)

A convenience constructor for creating a time-independent (`hastime=false`) [BrambleFunction](@ref).
It's a simplified wrapper around [`bramble_function_with_domain`](@ref).
"""
@inline _embed_notime(X, f; CoType = eltype(X)) = bramble_function_with_domain(f, X, false, CoType)

"""
	$(SIGNATURES)

Parses a domain specification, which can be:

  - A Symbol representing a spatial domain (e.g., :Ω).
  - An Expr representing the product of space and time domains (e.g., :(Ω × I)).

Returns a tuple `(space_domain_expr, time_domain_expr)`, where `time_domain_expr` is `nothing` if no time domain is specified.
"""
function _get_domains(domain_spec) # Accept Any type for flexibility
	if domain_spec isa Expr
		expr = domain_spec # Use a local variable named expr

		# Check for the specific structure Ω × I -> :(×(Ω, I))
		# Use isequal to safely handle potential `missing` if expr.args[1] is missing
		if expr.head == :call && length(expr.args) == 3 && isequal(expr.args[1], :×)
			# Return the expressions for the space and time domains directly
			space_domain = expr.args[2]
			time_domain = expr.args[3]
			return space_domain, time_domain
			# Check if it's any other function call (assumed to be spatial domain)
		elseif expr.head == :call
			# Assume it's just a space domain expression
			return expr, nothing
		else
			# Handle other Expr types that are not valid domain specifications
			error("Invalid domain format: Unexpected expression structure '$expr'. Expected a symbol, a function call (like `domain()`), or a product (like `Ω × I`).")
		end
	elseif domain_spec isa Symbol
		# Assume it's just a space domain symbol (e.g., Ω)
		return domain_spec, nothing
	else
		# Handle inputs that are neither Expr nor Symbol
		error("Invalid domain format: Input '$domain_spec' (type $(typeof(domain_spec))) is not a Symbol or Expr. Expected a symbol, a function call, or Ω × I.")
	end
end

# Fast path for when the domain is just a symbol like :Ω.
@inline _get_domains(s::Symbol) = s, nothing

# These internal helper functions ensure that the input coordinates `coords` are
# converted to the exact type expected by the `FunctionWrapper` before being called.
# This avoids potential type mismatch errors and improves performance.
@inline _convert_and_wrap(f, coords, ::Val{1}, ::Type{T}) where T = f.wrapped(T(coords))
@inline _convert_and_wrap(f, coords::NTuple{D,T}, ::Val{1}, ::Type{T}) where {D,T} = f.wrapped(T.(coords...))
@inline _convert_and_wrap(f, coords, ::Val{D}, ::Type{T}) where {D,T} = f.wrapped(NTuple{D,T}(coords))

# These methods make `BrambleFunction` instances callable like regular functions (i.e., a functor).
# They handle various input formats (scalars, Tuples, SVectors, Varargs) and dispatch
# to the appropriate `_convert_and_wrap` helper for type conversion before evaluation.
@inline (f::BrambleFunction{AT})(x::Number) where {AT<:Number} = _convert_and_wrap(f, x, Val(1), AT) #f.wrapped(convert(AT, x))
@inline (f::BrambleFunction{T,false})(coords::NTuple{D,T}) where {D,T<:AbstractFloat} = _convert_and_wrap(f, coords, Val(D), T)
@inline (f::BrambleFunction{T,false})(coords::SVector{D,T}) where {D,T<:AbstractFloat} = _convert_and_wrap(f, coords, Val(D), T)
@inline (f::BrambleFunction{T,false})(coords::T) where {T<:AbstractFloat} = _convert_and_wrap(f, coords, Val(1), T)
@inline (f::BrambleFunction{NTuple{D,T},false})(coords::Tuple) where {D,T} = _convert_and_wrap(f, coords, Val(D), T)
@inline (f::BrambleFunction{NTuple{D,T},false})(coords::SVector{D,T}) where {D,T} = _convert_and_wrap(f, Tuple(coords), Val(D), T)
@inline (f::BrambleFunction{NTuple{D,T},false})(coords::Vararg{Number,D}) where {D,T} = _convert_and_wrap(f, coords, Val(D), T)

"""
	embed_function(space_domain, [time_domain], func)

Creates a [BrambleFunction](@ref) for a function `func` defined over `space_domain`. If `time_domain` is provided, it creates a [BrambleFunction](@ref) for a time-dependent function `func(x, t)` defined over `space_domain × time_domain`.
"""
@inline embed_function(space_domain, func) = _embed_notime(space_domain, func)
@inline embed_function(space_domain, func::BrambleFunction) = func
@inline embed_function(space_domain, time_domain::CartesianProduct{1}, func) = _embed_withtime(space_domain, time_domain, func)

"""
	$(SIGNATURES)

Internal implementation for embedding a time-dependent function `f(x, t)`. When this function is called with a specific time `t_val`, it returns another space-only [BrambleFunction](@ref), which represents the spatial function `f(x, t_val)` at that fixed time.

# Arguments

  - `space_domain`: The spatial domain (CartesianProduct)
  - `time_domain`: The time domain (1D CartesianProduct)
  - `f`: A function `f(x, t)` that takes spatial and time coordinates
  - `FinalCoType`: The return type of `f` (optional, defaults to element type of time domain)

# Returns

A `BrambleFunction{ArgType,true,CoType,typeof(time_domain)}` where calling it with time `t` returns a space-only `BrambleFunction`.

# Examples

```julia
Ω = interval(0.0, 1.0)
I = interval(0.0, 10.0)
f(x, t) = sin(x * t)
bf = _embed_withtime(Ω, I, f)
bf_at_t5 = bf(5.0)  # Returns BrambleFunction for f(x, 5.0)
bf_at_t5(0.5)       # Evaluates sin(0.5 * 5.0)
```
"""
function _embed_withtime(space_domain, time_domain::CartesianProduct{1}, f; FinalCoType = eltype(time_domain))
	# Create a function of time `t` that returns a spatial BrambleFunction.
	# Base.Fix2 partially applies the second argument, creating f_t(x) = f(x, t)
	_f(t) = _embed_notime(space_domain, Base.Fix2(f, t), CoType = FinalCoType)

	# To determine the codomain type, we evaluate `_f` at a sample time point (center of time domain).
	CoType = typeof(_f(center(time_domain)))
	# Wrap the time-dependent function `_f`.
	ArgType = _get_args_type(time_domain)
	BFType = BrambleFunction{ArgType,true,CoType,typeof(time_domain)}

	return bramble_function_with_domain(_f, time_domain, true, CoType, domain = time_domain)::BFType
end

"""
	argstype(f::FunctionWrapper{CoType,Tuple{ArgsType}})

Extracts the argument type `ArgsType` from a `FunctionWrapper` instance or type.
"""
argstype(::FunctionWrapper{CoType,Tuple{ArgsType}}) where {CoType,ArgsType} = ArgsType
argstype(::FunctionWrapper{CoType,Tuple{}}) where CoType = Nothing
argstype(::Type{FunctionWrapper{CoType,Tuple{ArgsType}}}) where {CoType,ArgsType} = ArgsType
argstype(::Type{FunctionWrapper{CoType,Tuple{}}}) where CoType = Nothing

"""
	codomaintype(f::FunctionWrapper{CoType})

Extracts the codomain type `CoType` from a `FunctionWrapper` instance or type.
"""
codomaintype(::FunctionWrapper{CoType,Tuple{ArgsType}}) where {CoType,ArgsType} = CoType
codomaintype(::FunctionWrapper{CoType,Tuple{}}) where CoType = CoType  # Not Nothing!
codomaintype(::Type{FunctionWrapper{CoType,Tuple{ArgsType}}}) where {CoType,ArgsType} = CoType
codomaintype(::Type{FunctionWrapper{CoType,Tuple{}}}) where CoType = CoType