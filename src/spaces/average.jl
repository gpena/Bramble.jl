###############################################################
#                                                             #
# Implementation of the average operators for vector elements #
#                                                             #
###############################################################

#############
#           #
#    Mₕₓ    #
#           #
#############

@inline function Mₕₓ(u::VectorElement) 
	v = similar(u)
	dims = npoints(mesh(space(u)), Tuple)
	backward_averagex!(v.values, u.values, dims)
	return v
end

@inline function backward_averagex!(out, in, dims::NTuple{1,Int})
	@assert length(out) == length(in)
	s = convert(eltype(out), 0.5)
	out[1] = in[1] * s

	@simd for i in 2:dims[1]
		out[i] = (in[i] + in[i - 1]) * s
	end
end

function backward_averagex!(out, in, dims::NTuple{D,Int}) where D
	first_dim = prod(dims[1:(D - 1)])
	last_dim = dims[D]

	first_dims = ntuple(i -> dims[i], D - 1)

	@inbounds for m in 1:last_dim
		idx = ((m - 1) * (first_dim) + 1):(m * (first_dim))
		@views backward_averagex!(out[idx], in[idx], first_dims)
	end
end


#############
#           #
#    Mₕᵧ    #
#           #
#############
@inline function Mₕᵧ(u::VectorElement) 
	if dim(mesh(space(u))) == 1
		@error "This operator is not implemented in 1D"
	end

	v = similar(u)
	dims = npoints(mesh(space(u)), Tuple)
	backward_averagey!(v.values, u.values, dims)
	return v
end

@inline function __aux_back_avg!(out, in, c::Int, N::Int)
	@simd for i in ((c - 1) * N + 1):(c * N)
		out[i] = (in[i] + in[i - N]) * convert(eltype(out), 0.5)
	end
end

function backward_averagey!(out, in, dims::NTuple{2,Int})
    N, M = dims

    @views out[1:N] .= in[1:N] .* convert(eltype(out), 0.5)

    for c = 2:M
		__aux_back_avg!(out, in, c, N)
        #idx_prev = ((c-2)*N + 1):((c-1)*N)
        #idx_next = ((c-1)*N + 1):(c*N)
        #@views out[idx_next] .= (in[idx_next] .+ in[idx_prev])*convert(eltype(out), 0.5)
    end
end

@inline function backward_averagey!(out, in, dims::NTuple{3,Int})
	N, M = dims[1:2]
	O = dims[3]

	@inbounds for lev in 1:O
		idx = ((lev - 1) * N * M + 1):(lev * N * M)
		@views backward_averagey!(out[idx], in[idx], (N, M))
	end
end



#############
#           #
#    Mₕ₂    #
#           #
#############
@inline function Mₕ₂(u::VectorElement) 
	if dim(mesh(space(u))) <= 2
		@error "This operator is not implemented in 1D/2D."
	end

	v = similar(u)
	backward_averagez!(v.values, u.values, npoints(mesh(space(u)), Tuple))

	return v
end

function backward_averagez!(out, in, dims::NTuple{D,Int}) where D
	N = prod(dims[1:(D - 1)])
	O = dims[D]
	@views out[1:N] .= in[1:N] .* convert(eltype(out), 0.5)

	@inbounds for c in 2:O
		__aux_back_avg!(out, in, c, N)
	end
end

Mₕ(u::VectorElement, ::Val{1}) = Mₕₓ(u)
Mₕ(u::VectorElement, ::Val{2}) = Mₕᵧ(u)
Mₕ(u::VectorElement, ::Val{3}) = Mₕ₂(u)
Mₕ(u::VectorElement) = ntuple(i-> Mₕ(u, Val(i)), dim(mesh(space(u))))

Mₕₓ(u::MatrixElement) = elements(space(u), Mₕₓ(mesh(space(u))) * u.values)
Mₕᵧ(u::MatrixElement) = elements(space(u), Mₕᵧ(mesh(space(u))) * u.values)
Mₕ₂(u::MatrixElement) = elements(space(u), Mₕ₂(mesh(space(u))) * u.values)
Mₕ(u::MatrixElement) = (@assert dim(mesh(space(u))) == 1; return Mₕₓ(u))

Mₕₓ(M::MeshType) = (shiftₓ(M, Val(dim(M)), Val(0)) + shiftₓ(M, Val(dim(M)), Val(-1)))*convert(eltype(M), 0.5)
Mₕᵧ(M::MeshType) = (shiftᵧ(M, Val(dim(M)), Val(0)) + shiftᵧ(M, Val(dim(M)), Val(-1)))*convert(eltype(M), 0.5)
Mₕ₂(M::MeshType) = (shift₂(M, Val(dim(M)), Val(0)) + shift₂(M, Val(dim(M)), Val(-1)))*convert(eltype(M), 0.5)

Mₕₓ(S::SpaceType) = Mₕₓ(mesh(S))
Mₕᵧ(S::SpaceType) = Mₕᵧ(mesh(S))
Mₕ₂(S::SpaceType) = Mₕ₂(mesh(S))

Mₕ(M::MeshType) = (@assert dim(M) == 1; return Mₕₓ(M))
Mₕ(S::SpaceType) = (@assert dim(S) == 1; return Mₕₓ(S))
