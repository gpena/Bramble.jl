"""
	struct BrambleFunction{ArgsType,hastime,CoType}
		wrapped::FunctionWrapper{CoType,Tuple{ArgsType}}
	end

Structure to wrap around functions to make them more type agnostic. It uses `FunctionWrappers` to provide functions calculated on `ArgsType`. The type arguments are `hastime` to indicate if the function is time-dependent and `CoType`, the time of the codomain of the function.
"""
struct BrambleFunction{ArgsType,hastime,CoType}
	wrapped::FunctionWrapper{CoType,Tuple{ArgsType}}
end

function _get_args_type(X)
	D = dim(X)
	T = eltype(X)
	return ifelse(D == 1, T, NTuple{D,T})
end

function _create_bramble_function_with_domain(f, X, hastime, CoType)
	ArgType = _get_args_type(X)
	wrapped_f = FunctionWrapper{CoType,Tuple{ArgType}}(f)
	return BrambleFunction{ArgType,hastime,CoType}(wrapped_f)
end
#=
function _create_bramble_function(f, ArgType, hastime, CoType)
	wrapped_f = FunctionWrapper{CoType,Tuple{ArgType}}(f)
	return BrambleFunction{ArgType,hastime,CoType}(wrapped_f)
end
=#

@inline _embed_notime(X, f; CoType = eltype(X)) = _create_bramble_function_with_domain(f, X, false, CoType)
#=)
	ArgsType = _get_args_type(X)
	return _create_bramble_function(f, ArgsType, false, CoType)
end=#

"""
	@embed(X:CartesianProduct, f)
	@embed(X:Domain, f)
	@embed(X:MeshType, f)
	@embed(X:SpaceType, f)
	@embed(X × I, f)

Returns a new wrapped version of function `f`. If

  - `X` is not a [SpaceType](@ref), the topological dimension of `X` is used to caracterize the types in the returning [BrambleFunction](@ref); in this case, the new function can be applied to `Tuple`s (with point coordinates);

  - `X` is a [SpaceType](@ref), the returning function type, [BrambleFunction](@ref), is caracterized by the dimension of the [element](@ref)s of the space.

If the first argument is of the form `X × I`, where `I` is an interval, then `f` must be a time-dependent function defined as `f(x,t) = ...`.

# Example

```@example
julia> Ω = domain(interval(0, 1) × interval(0, 1));
	   f = @embed Ω x->x[1] * x[2] + 1;  # or f = @embed(Ω, x -> x[1]*x[2]+1)

```
"""

macro embed(domain_expr, func_expr)
	# 1. Parse domain (assuming _get_domains is the robust version from previous step)
	local space_domain, time_domain
	try
		space_domain, time_domain = _get_domains(domain_expr)
	catch e
		# If _get_domains errors, rethrow the error originating from the macro call site
		return :(throw($e))
	end

	# 2. Validate func_expr safely
	local is_valid_func_format = false
	local func_head = nothing # Store head if it's an Expr

	if func_expr isa Symbol
		is_valid_func_format = true
	elseif func_expr isa Expr
		func_head = func_expr.head
		# Use isequal to safely compare, handling potential `missing` in func_head
		if isequal(func_head, :->) || isequal(func_head, :.) || isequal(func_head, :function) # Added :function case
			is_valid_func_format = true
		end
		# Add other valid Expr heads if necessary
	end
	# Otherwise (Number, String, Nothing, Missing, etc.), is_valid_func_format remains false

	# 3. Construct the appropriate call or error
	if is_valid_func_format
		if time_domain isa Nothing
			# Case: function only depends on space variables
			# Ensure Module name is PascalCase: Bramble
			# Ensure function names are snake_case: _embed_notime
			return esc(:(Bramble._embed_notime($space_domain, $func_expr)))
		else
			# Case: function also depends on a time variable
			return esc(:(Bramble._embed_withtime($space_domain, $time_domain, $func_expr)))
		end
	else
		# Invalid func_expr type or structure
		# Use error() for proper error reporting
		# Provide context about the received expression
		func_expr_str = repr(remove_linenums!(func_expr)) # Clean representation for error message
		error("In macro @embed: Invalid function expression format. Received: `$func_expr_str` (type: $(typeof(func_expr))). Expected a function Symbol, an anonymous function (`->`), a regular function definition (`function ...`), or certain dot calls (`.`).")
	end
end

"""
	_get_domains(domain_spec)

Parses a domain specification, which can be:

  - A Symbol representing a spatial domain (e.g., :Ω).
  - An Expr representing a function call for a spatial domain (e.g., :(unit_interval())).
  - An Expr representing the product of space and time domains (e.g., :(Ω × I)).

Returns a tuple `(spatial_domain_expr, time_domain_expr)`, where `time_domain_expr`
is `nothing` if no time domain is specified.
"""
function _get_domains(domain_spec) # Accept Any type for flexibility
	if domain_spec isa Expr
		expr = domain_spec # Use a local variable named expr

		# Check for the specific structure Ω × I -> :(×(Ω, I))
		# Use isequal to safely handle potential `missing` if expr.args[1] is missing
		if expr.head == :call && length(expr.args) == 3 && isequal(expr.args[1], :×)
			# Return the expressions for the space and time domains directly
			spatial_domain = expr.args[2]
			time_domain = expr.args[3]
			return spatial_domain, time_domain
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

@inline _get_domains(s::Symbol) = s, nothing

# function evaluators
@inline _convert_and_wrap(f, coords, ::Val{1}, ::Type{T}) where T = f.wrapped(T(coords))
@inline _convert_and_wrap(f, coords, ::Val{D}, ::Type{T}) where {D,T} = f.wrapped(NTuple{D,T}(coords))

@inline (f::BrambleFunction{AT})(x::Number) where {AT<:Number} = _convert_and_wrap(f, x, Val(1), AT) #f.wrapped(convert(AT, x))
@inline (f::BrambleFunction{NTuple{D,T},false})(coords::Tuple) where {D,T} = _convert_and_wrap(f, coords, Val(D), T)
@inline (f::BrambleFunction{NTuple{D,T},false})(coords::Vararg{Number,D}) where {D,T} = _convert_and_wrap(f, coords, Val(D), T)

#=
@inline (f::BrambleFunction{AT,false})(x) where {AT<:Number} = f.wrapped(convert(ArgsType, x)::ArgsType)
@inline (f::BrambleFunction{AT,true})(t) where AT = f.wrapped(t)
@inline (f::BrambleFunction{AT,true})(t::Number) where AT = f.wrapped(convert(AT, t))

function (f::BrambleFunction{NTuple{D,T},false})(x...) where {D,T}
	@unpack wrapped = f

	if x[1] isa Tuple
		@assert length(x[1]) == D
		y = Tuple(convert.(T, x[1])::NTuple{D,T})

		return wrapped(y)
	end

	@assert length(x) == D
	y = Tuple(convert.(T, x)::NTuple{D,T})
	println(y)

	return wrapped(y)
end=#

"""
	embed_function(space_domain, func)

Creates a BrambleFunction for a function `func` defined over `space_domain`.
Equivalent to `@embed space_domain func`.
"""
@inline embed_function(space_domain, func) = _embed_notime(space_domain, func)

"""
	embed_function(space_domain, time_domain, func)

Creates a BrambleFunction for a time-dependent function `func(x, t)` defined
over `space_domain × time_domain`.
Equivalent to `@embed space_domain × time_domain func`.
"""
function embed_function(space_domain, time_domain::CartesianProduct{1}, func)
	return _embed_withtime(space_domain, time_domain, func)
end

function _embed_withtime(spatial_domain, time_domain::CartesianProduct{1}, f)
	_f(t) = _embed_notime(spatial_domain, Base.Fix2(f, t))

	#TimeArgType = _get_args_type(time_domain)
	CoType = typeof(_f(sum(tails(time_domain)) * 0.5))

	return _create_bramble_function_with_domain(_f, time_domain, true, CoType)
end

argstype(g::FunctionWrapper{CoType,Tuple{ArgsType}}) where {CoType,ArgsType} = ArgsType
argstype(g::FunctionWrapper{CoType,Tuple{}}) where CoType = Nothing
argstype(::Type{FunctionWrapper{CoType,Tuple{ArgsType}}}) where {CoType,ArgsType} = ArgsType
argstype(::Type{FunctionWrapper{CoType,Tuple{}}}) where CoType = Nothing

codomaintype(g::FunctionWrapper{CoType,Tuple{ArgsType}}) where {CoType,ArgsType} = CoType
codomaintype(g::FunctionWrapper{CoType,Tuple{}}) where CoType = Nothing
codomaintype(::Type{FunctionWrapper{CoType,Tuple{ArgsType}}}) where {CoType,ArgsType} = CoType
codomaintype(::Type{FunctionWrapper{CoType,Tuple{}}}) where CoType = CoType