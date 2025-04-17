"""
	struct BrambleFunction{ArgsType,hastime,CoType}
		wrapped::FunctionWrapper{CoType,Tuple{ArgsType}}
	end

Structure to wrap around functions to make them more type agnostic. It uses `FunctionWrappers` to provide functions calculated on `ArgsType`. The type arguments are `hastime` to indicate if the function is time-dependent and `CoType`, the time of the codomain of the function.
"""
struct BrambleFunction{ArgsType,hastime,CoType}
	wrapped::FunctionWrapper{CoType,Tuple{ArgsType}}
end

function _embed_notime(X, f; CoType = eltype(X))
	D = dim(X)
	T = eltype(X)
	ArgsType = D == 1 ? T : NTuple{D,T}

	wrapped_f_tuple = FunctionWrapper{CoType,Tuple{ArgsType}}(f)

	return BrambleFunction{ArgsType,false,CoType}(wrapped_f_tuple)
end

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
	space_domain, time_domain = _get_domains(domain_expr)

	if func_expr isa Symbol || func_expr.head == :-> || func_expr.head == :.
		if time_domain isa Nothing
			# case of function only depending on space variables
			return esc(:(Bramble._embed_notime($space_domain, $func_expr)))
		end

		# case when the function also depends on a time variable
		return esc(:(Bramble._embed_withtime($space_domain, $time_domain, $func_expr)))
	end

	return :(@error "Don't know how to handle this expression")
end

@inline function _get_domains(expr::Expr)
	if expr.args isa Array && length(expr.args) == 3
		args = expr.args
		return args[2], args[3]
	end

	if expr.head == :call
		return :($expr), nothing
	end

	return :(@error "Invalid domain format. Please write the first input as Ω × I, where Ω is the space domain and I is the time domain.")
end

@inline _get_domains(s::Symbol) = :($s), nothing

function (f::BrambleFunction{NTuple{D,T},false})(x...) where {D,T}
	@unpack wrapped = f

	if x[1] isa Tuple
		@assert length(x[1]) == D
		y = Tuple(convert.(T, x[1])::NTuple{D,T})

		return wrapped(y)
	end

	@assert length(x) == D

	y = Tuple(convert.(T, x)::NTuple{D,T})
	return wrapped(y)
end

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
function embed_function(space_domain, time_domain, func)
	if !(time_domain isa CartesianProduct{1})
		error("Time domain must be a 1D CartesianProduct (interval) for this function.")
	end
	return _embed_withtime(space_domain, time_domain, func)
end

(f::BrambleFunction{ArgsType,false})(x) where {ArgsType<:Number} = f.wrapped(convert(ArgsType, x)::ArgsType)
(f::BrambleFunction{ArgsType,true})(t) where ArgsType = f.wrapped(t)

function _embed_withtime(X, I, f)
	@assert I isa CartesianProduct{1}

	_f(t) = _embed_notime(X, Base.Fix2(f, t))

	ArgsType = eltype(X)
	CoType = typeof(_f(sum(tails(I)) * 0.5))

	wrapped_f_tuple = FunctionWrapper{CoType,Tuple{ArgsType}}(_f)

	return BrambleFunction{ArgsType,true,CoType}(wrapped_f_tuple)
end

argstype(g::FunctionWrapper{CoType,Tuple{ArgsType}}) where {CoType,ArgsType} = ArgsType
argstype(g::FunctionWrapper{CoType,Tuple{}}) where CoType = Nothing
argstype(::Type{FunctionWrapper{CoType,Tuple{ArgsType}}}) where {CoType,ArgsType} = ArgsType
argstype(::Type{FunctionWrapper{CoType,Tuple{}}}) where CoType = Nothing

codomaintype(g::FunctionWrapper{CoType,Tuple{ArgsType}}) where {CoType,ArgsType} = CoType
codomaintype(g::FunctionWrapper{CoType,Tuple{}}) where CoType = Nothing
codomaintype(::Type{FunctionWrapper{CoType,Tuple{ArgsType}}}) where {CoType,ArgsType} = CoType
codomaintype(::Type{FunctionWrapper{CoType,Tuple{}}}) where CoType = CoType