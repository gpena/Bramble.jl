struct StaticFunction{D,T} 
	n_static_args::Int
	f_tuple::FunctionWrapper{T, Tuple{NTuple{D,T}}}
	f_cartesian::FunctionWrapper{T, Tuple{CartesianIndex{D}}}
end


function embed(f::FType, M::MType) where {D,FType, MType<:MeshType{D}}
	T = eltype(M)
	wrapped_f_tuple = FunctionWrapper{T, Tuple{NTuple{D,T}}}(f)
	pts = points(M)
	g = Base.Fix1(_index2point, pts)
	fog(idx) = f(g(idx))
	wrapped_f_cartesian = FunctionWrapper{T, Tuple{CartesianIndex{D}}}(fog)

	return StaticFunction{D,T}(D, wrapped_f_tuple, wrapped_f_cartesian)
end

@inline (f::StaticFunction{D,T})(x::NTuple{D,T}) where {D,T} = f.f_tuple(x)
@inline (f::StaticFunction{1,T})(x::T) where T = f.f_tuple((x,))

@inline (f::StaticFunction{D,T})(x::CartesianIndex{D}) where {D,T} = f.f_cartesian(x)