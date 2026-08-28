"""
	$(TYPEDEF)

Internal structure wrapping user-defined functions for type-agnostic, zero-overhead dispatch.
Uses `FunctionWrappers.FunctionWrapper` to provide stable, compiled calling signatures.

# Fields
$(FIELDS)
"""
struct BrambleFunction{ArgsType,hastime,CoType,DType}
	"A `FunctionWrapper` wrapper for the function"
	wrapped::FunctionWrapper{CoType,Tuple{ArgsType}}
	"Domain of the function (typically a [`CartesianProduct`](@ref) or [`Domain`](@ref))"
	domain::DType
end

"""
	$(SIGNATURES)

Checks if a [`BrambleFunction`](@ref) is time-dependent by inspecting its `hastime` type parameter.
Returns `true` if time-dependent, `false` otherwise.
"""
@inline has_time(::BrambleFunction{ArgsType,hastime}) where {ArgsType,hastime} = hastime
@inline has_time(::Type{<:BrambleFunction{ArgsType,hastime}}) where {ArgsType,hastime} = hastime

"""
	$(SIGNATURES)

Internal helper to determine the expected argument type for a function based on domain `X`.

- If the domain is 1D: returns scalar element type `T`.
- If the domain is ``D``-dimensional: returns `NTuple{D, T}`.
"""
@inline _get_args_type(::CartesianProduct{1,T}) where T = T
@inline _get_args_type(::CartesianProduct{D,T}) where {D,T} = NTuple{D,T}
@inline _get_args_type(::Type{<:CartesianProduct{1,T}}) where T = T
@inline _get_args_type(::Type{<:CartesianProduct{D,T}}) where {D,T} = NTuple{D,T}
@inline _get_args_type(X::Domain) = _get_args_type(set(X))
@inline _get_args_type(::Type{<:Domain{S}}) where S = _get_args_type(S)
@inline _get_args_type(X) = _get_args_type_d(Val(dim(X)), eltype(X))
@inline _get_args_type_d(::Val{1}, ::Type{T}) where T = T
@inline _get_args_type_d(::Val{D}, ::Type{T}) where {D,T} = NTuple{D,T}

"""
	$(SIGNATURES)

Constructs a [`BrambleFunction`](@ref) bundling function `f` with its domain `X`.

# Arguments
- `f`: Function to wrap
- `X`: Spatial domain used to infer the argument type of `f`
- `hastime`: Boolean flag indicating time dependence
- `CoType`: Return type of function `f`
- `domain`: Domain object to store in the struct (defaults to `X`)
"""
function bramble_function_with_domain(f, X, hastime::Bool, CoType::Type; domain = X)
	ArgType = _get_args_type(X)
	wrapped_f = FunctionWrapper{CoType,Tuple{ArgType}}(f)
	return BrambleFunction{ArgType,hastime,CoType,typeof(domain)}(wrapped_f, domain)
end

"""
	$(SIGNATURES)

Creates a time-independent (`hastime=false`) [`BrambleFunction`](@ref).
"""
@inline _embed_notime(X, f; CoType = eltype(X)) = bramble_function_with_domain(f, X, false, CoType)

"""
	$(SIGNATURES)

Parses a domain specification into `(space_domain, time_domain)`.
"""
function _get_domains(domain_spec)
	if domain_spec isa Expr
		expr = domain_spec
		if expr.head == :call && length(expr.args) == 3 && isequal(expr.args[1], :×)
			return expr.args[2], expr.args[3]
		elseif expr.head == :call
			return expr, nothing
		else
			error("Invalid domain format: Unexpected expression '$expr'. Expected a symbol, function call, or product (like `Ω × I`).")
		end
	else
		error("Invalid domain format: Input '$domain_spec' (type $(typeof(domain_spec))) is not a Symbol or Expr.")
	end
end

@inline _get_domains(s::Symbol) = s, nothing

# --- Functor Call Dispatches for 1D Functions (ArgsType <: Number) ---
@inline (f::BrambleFunction{AT,false})(x::Number) where {AT<:Number} = f.wrapped(convert(AT, x))
@inline (f::BrambleFunction{AT,false})(coords::Tuple{Number}) where {AT<:Number} = f.wrapped(convert(AT, coords[1]))
@inline (f::BrambleFunction{AT,false})(coords::SVector{1}) where {AT<:Number} = f.wrapped(convert(AT, coords[1]))
@inline (f::BrambleFunction{AT,false})(coords::AbstractVector) where {AT<:Number} = f.wrapped(convert(AT, coords[1]))

# --- Functor Call Dispatches for Multi-D Functions (ArgsType <: Tuple) ---
@inline (f::BrambleFunction{AT,false})(coords::AT) where {AT<:Tuple} = f.wrapped(coords)
@inline (f::BrambleFunction{AT,false})(coords::Tuple) where {AT<:Tuple} = f.wrapped(convert(AT, coords))
@inline (f::BrambleFunction{AT,false})(coords::SVector) where {AT<:Tuple} = f.wrapped(convert(AT, Tuple(coords)))
@inline (f::BrambleFunction{AT,false})(coords::AbstractVector) where {AT<:Tuple} = f.wrapped(convert(AT, Tuple(coords)))
@inline (f::BrambleFunction{AT,false})(coords::Number...) where {AT<:Tuple} = f.wrapped(convert(AT, coords))

# --- Functor Call Dispatches for Time-Dependent Functions (hastime=true) ---
@inline (f::BrambleFunction{ArgsType,true})(t::Number) where {ArgsType} = f.wrapped(convert(ArgsType, t))
@inline (f::BrambleFunction{ArgsType,true})(x, t::Number) where {ArgsType} = (f(t))(x)

"""
	$(SIGNATURES)

Embeds a Julia function into a [`BrambleFunction`](@ref) defined over `space_domain` (and optional `time_domain`).
"""
@inline embed_function(space_domain, func) = _embed_notime(space_domain, func)
@inline embed_function(space_domain, func::BrambleFunction) = func
@inline embed_function(space_domain, time_domain::CartesianProduct{1}, func) = _embed_withtime(space_domain, time_domain, func)
@inline embed_function(space_domain, ::CartesianProduct{1}, func::BrambleFunction) = func

"""
	$(SIGNATURES)

Embeds a time-dependent function `f(x, t)` over spatial domain `space_domain` and 1D time domain `time_domain`.
"""
function _embed_withtime(space_domain, time_domain::CartesianProduct{1}, f; FinalCoType = eltype(time_domain))
	_f(t) = _embed_notime(space_domain, Base.Fix2(f, t); CoType = FinalCoType)

	# Determine codomain type from sample evaluation
	sample_center = center(time_domain)[1]
	CoType = typeof(_f(sample_center))
	ArgType = _get_args_type(time_domain)
	BFType = BrambleFunction{ArgType,true,CoType,typeof(time_domain)}

	return bramble_function_with_domain(_f, time_domain, true, CoType; domain = time_domain)::BFType
end

"""
	$(SIGNATURES)

Extracts the argument type `ArgsType` from a `FunctionWrapper`.
"""
@inline argstype(::FunctionWrapper{CoType,Tuple{ArgsType}}) where {CoType,ArgsType} = ArgsType
@inline argstype(::FunctionWrapper{CoType,Tuple{}}) where CoType = Nothing
@inline argstype(::Type{FunctionWrapper{CoType,Tuple{ArgsType}}}) where {CoType,ArgsType} = ArgsType
@inline argstype(::Type{FunctionWrapper{CoType,Tuple{}}}) where CoType = Nothing

"""
	$(SIGNATURES)

Extracts the codomain type `CoType` from a `FunctionWrapper`.
"""
@inline codomaintype(::FunctionWrapper{CoType,Tuple{ArgsType}}) where {CoType,ArgsType} = CoType
@inline codomaintype(::FunctionWrapper{CoType,Tuple{}}) where CoType = CoType
@inline codomaintype(::Type{FunctionWrapper{CoType,Tuple{ArgsType}}}) where {CoType,ArgsType} = CoType
@inline codomaintype(::Type{FunctionWrapper{CoType,Tuple{}}}) where CoType = CoType

function Base.show(io::IO, bf::BrambleFunction{ArgsType,hastime,CoType}) where {ArgsType,hastime,CoType}
	kind = hastime ? "time-dependent" : "spatial"
	if get(io, :compact, false)
		print(io, "BrambleFunction($kind, $ArgsType -> $CoType)")
	else
		print(io, "BrambleFunction($kind):\n  signature: $ArgsType -> $CoType\n  domain: $(bf.domain)")
	end
end