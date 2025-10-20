"""
	struct BrambleGridSpaceFunction{S,T} 
		f_tuple::FunctionWrapper{T, Tuple{VectorElement{S,T}}}
	end

Structure to wrap around functions defined on gridspaces to make them more type agnostic. It uses `FunctionWrappers` to provide functions calculated on [VectorElement](@ref).
"""
#=struct BrambleGridSpaceFunction{ElemType}
	f_vec::FunctionWrapper{ElemType,Tuple{ElemType}}
end

function _embed_notime(Wₕ::SpaceType, f)
	T = eltype(Wₕ)
	ArgsType = VectorElement{typeof(Wₕ),T}
	CoType = ArgsType

	wrapped_f_tuple = FunctionWrapper{CoType,Tuple{ArgsType}}(f)

	return BrambleFunction{ArgsType,false,CoType}(wrapped_f_tuple)
end
=#
#(f::BrambleFunction{VectorElement{SType,T}})(u::VectorElement{SType,T}) where {SType,T} = f.wrapped(u)