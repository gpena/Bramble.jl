# Helper functions to calculate difference operators
@inline ⊗(A, B) = kron(A, B)

@inline _Eye(::Type{T}, npts::Int, ::Val{i}) where {T,i} = spdiagm(i => FillArrays.Ones(T, npts - abs(i)))
@inline _Eye(::Type{T}, npts::Int, ::Val{0}) where T = FillArrays.Eye(npts)

# shift functions
#@inline shift(Ωₕ::MeshType, ::Val{i}) where i = _Eye(eltype(Ωₕ), npoints(Ωₕ, Tuple)[1], Val(i))
#@inline shift(Ωₕ::MeshType, ::Val{0}) = FillArrays.Eye(npoints(Ωₕ, Tuple)[1])

@inline shiftₓ(Ωₕ::MeshType, ::Val{1}, ::Val{0}) = _Eye(eltype(Ωₕ), npoints(Ωₕ, Tuple)[1], Val(0))
@inline shiftₓ(Ωₕ::MeshType, ::Val{D}, ::Val{0}) where D = Eye(npoints(Ωₕ))
@inline shiftₓ(Ωₕ::MeshType, ::Val{1}, ::Val{i}) where i = _Eye(eltype(Ωₕ), npoints(Ωₕ, Tuple)[1], Val(i))#shift(M, Val(i))
@inline shiftₓ(Ωₕ::MeshType, ::Val{D}, ::Val{i}) where {D,i} = Eye(npoints(Ωₕ(D))) ⊗ shiftₓ(Ωₕ, Val(D - 1), Val(i))

@inline shiftᵧ(Ωₕ::MeshType, ::Val{2}, ::Val{0}) = Eye(npoints(Ωₕ))
@inline shiftᵧ(Ωₕ::MeshType, ::Val{3}, ::Val{0}) = Eye(npoints(Ωₕ))
@inline shiftᵧ(Ωₕ::MeshType, ::Val{1}, ::Val{i}) where i = _Eye(eltype(Ωₕ), npoints(Ωₕ(2)), Val(i))#shift(M(2), Val(i))
@inline shiftᵧ(Ωₕ::MeshType, ::Val{2}, ::Val{i}) where i = shiftᵧ(Ωₕ, Val(1), Val(i)) ⊗ Eye(npoints(Ωₕ(1)))
@inline shiftᵧ(Ωₕ::MeshType, ::Val{3}, ::Val{i}) where i = Eye(npoints(Ωₕ(3))) ⊗ shiftᵧ(Ωₕ, Val(2), Val(i))

@inline shift₂(Ωₕ::MeshType, ::Val{3}, ::Val{0}) = Eye(npoints(Ωₕ))
@inline shift₂(Ωₕ::MeshType, ::Val{3}, ::Val{i}) where i = _Eye(eltype(Ωₕ), npoints(Ωₕ(3)), Val(i)) ⊗ Eye(prod(npoints(Ωₕ, Tuple)[1:2])) #=shift(M(3), Val(i))=#

# Helper functions to calculate backward difference operators
"""
	_backward_differencex!(out, in, dims::NTuple{1,Int})

In-place computation of the backward difference in the `x` direction of vector `in` for a `1`-dimensional element. The result is stored  in `out`. Allows for `in` and `out` to be the same vector.
"""
@inline function _backward_differencex!(out, in, dims::NTuple{1,Int})
	N = dims[1]
	@assert length(out) == length(in) == N

	aux = in[1]
	out[1] = in[1]
	aux2 = zero(eltype(in))

	@simd for i in 2:N
		aux2 = in[i]
		out[i] = in[i] - aux
		aux = aux2
	end
end

"""
	_backward_finite_differencex!(out, in, hx, dims::NTuple{1,Int})

In-place computation of the backward finite difference in the `x` direction of vector `in` for a `1`-dimensional element. The spacings are encoded in `hx`. The result is stored  in `out`. Allows for `in` and `out` to be the same vector.
"""
@inline function _backward_finite_differencex!(out, in, hx, dims::NTuple{1,Int})
	N = dims[1]
	@assert length(out) == length(in) == N
	aux = in[1]
	out[1] = in[1] / hx(2)
	aux2 = zero(eltype(in))

	@simd for i in 2:N
		aux2 = in[i]
		out[i] = (aux2 - aux) / hx(i)
		aux = aux2
	end
end

"""
	_backward_differencex!(out, in, dims::NTuple{D,Int}, ::Val{D})

In-place computation of the backward difference in the `x` direction of vector `in` for a `D`-dimensional (``D>1``) element. The result is stored  in `out`. Allows for `in` and `out` to be the same vector.
"""
function _backward_differencex!(out, in, dims::NTuple{D,Int}) where D
	@assert length(out) == length(in) == prod(dims)
	N, _ = dims
	last_dofs = @views prod(dims[2:D])
	for i in 1:last_dofs
		idx = ((i - 1) * N + 1):(i * N)
		@views _backward_differencex!(out[idx], in[idx], (N,))
	end
end

"""
	_backward_finite_differencex!(out, in, hx, dims::NTuple{D,Int})

In-place computation of the backward finite difference in the `x` direction of vector `in` for a `D`-dimensional element. The spacings are encoded in `hx`. The result is stored  in `out`. Allows for `in` and `out` to be the same vector.
"""
@inline function _backward_finite_differencex!(out, in, hx, dims::NTuple{D,Int}) where D
	@assert length(out) == length(in) == prod(dims)
	first_dim = prod(dims[1:(D - 1)])
	last_dim = dims[D]

	first_dims = ntuple(i -> dims[i], D - 1)

	@inbounds for m in 1:last_dim
		idx = ((m - 1) * (first_dim) + 1):(m * (first_dim))
		@views _backward_finite_differencex!(out[idx], in[idx], hx, first_dims)
	end
end

"""
	_backward_differencey!(out, in, dims::NTuple{2,Int})

In-place computation of the difference in the `y` direction of vector `in` for a `2`-dimensional element. The result is stored  in `out`. Allows for `in` and `out` to be the same vector.
"""
@inline function _backward_differencey!(out, in, dims::NTuple{2,Int})
	@assert length(out) == length(in) == prod(dims)
	N, M = dims
	@inbounds for i in 1:N
		idx = i:N:(N * (M - 1) + 1 + i)
		@views _backward_differencex!(out[idx], in[idx], (M,))
	end
end

"""
	_backward_finite_differencey!(out, in, hy, dims::NTuple{2,Int})

In-place computation of the backward finite difference in the `y` direction of vector `in` for a `2`-dimensional element. The spacings are encoded in `hy`. The result is stored  in `out`. Allows for `in` and `out` to be the same vector.
"""
@inline function _backward_finite_differencey!(out, in, hy, dims::NTuple{2,Int})
	@assert length(out) == length(in) == prod(dims)
	N, M = dims
	@inbounds for i in 1:N
		idx = i:N:(N * (M - 1) + 1 + i)
		@views _backward_finite_differencex!(out[idx], in[idx], hy, (M,))
	end
end

"""
	_backward_differencey!(out, in, dims::NTuple{3,Int})

In-place computation of the backward difference in the `y` direction of vector `in` for a `3`-dimensional element. The result is stored  in `out`. Allows for `in` and `out` to be the same vector.
"""
@inline function _backward_differencey!(out, in, dims::NTuple{3,Int})
	@assert length(out) == length(in) == prod(dims)
	N, M = dims[1:2]
	O = dims[3]

	@inbounds for lev in 1:O
		idx = ((lev - 1) * N * M + 1):(lev * N * M)
		@views _backward_differencey!(out[idx], in[idx], (N, M))
	end
end

"""
	_backward_finite_differencey!(out, in, hy, dims::NTuple{3,Int})

In-place computation of the backward finite difference in the `y` direction of vector `in` for a `3`-dimensional element. The spacings are encoded in `hy`. The result is stored  in `out`. Allows for `in` and `out` to be the same vector.
"""
@inline function _backward_finite_differencey!(out, in, hy, dims::NTuple{3,Int})
	@assert length(out) == length(in) == prod(dims)
	N, M, O = dims
	first_dims = N * M
	@inbounds for i in 1:O
		idx = ((i - 1) * first_dims + 1):(i * first_dims)
		@views _backward_finite_differencey!(out[idx], in[idx], hy, (N, M))
	end
end

"""
	_backward_differencez!(out, in, dims::NTuple{3,Int})

In-place computation of the backward difference in the `z` direction of vector `in` for a `3`-dimensional element. The result is stored  in `out`. Allows for `in` and `out` to be the same vector.
"""
@inline function _backward_differencez!(out, in, dims::NTuple{3,Int})
	@assert length(out) == length(in) == prod(dims)
	N, M, O = dims
	first_dofs = N * M
	for i in 1:first_dofs
		idx = i:first_dofs:(i + (O - 1) * first_dofs)
		@views _backward_differencex!(out[idx], in[idx], (O,))
	end
end

"""
	_backward_finite_differencez!(out, in, hz, dims::NTuple{3,Int})

In-place computation of the backward finite difference in the `z` direction of vector `in` for a `3`-dimensional element. The spacings are encoded in `hz`. The result is stored  in `out`. Allows for `in` and `out` to be the same vector.
"""
@inline function _backward_finite_differencez!(out, in, hz, dims::NTuple{3,Int})
	@assert length(out) == length(in) == prod(dims)
	N, M, O = dims
	first_dofs = N * M

	for i in 1:first_dofs
		idx = i:first_dofs:(i + (O - 1) * first_dofs)
		@views _backward_finite_differencex!(out[idx], in[idx], hz, (O,))
	end
end

# helper functions for defining the backward finite difference operators
function invert_spacing!(v, Ωₕ::MeshType, component::Int = 1)
	for (i, h) in zip(eachindex(v), spacing(Ωₕ(component), Iterator))
		v[i] = inv(h)
	end
end

@inline function repeat_across_dims1!(v, ncopies, dims)
	#v_reshaped = reshape(v, dims, ncopies)
	#@views v_reshaped[:, 2:end] .= v_reshaped[:, 1]
	@simd for j in 2:ncopies
		for i in 1:dims
			v[i + (j - 1) * dims] = v[i]
		end
	end
end

# do with loop
@inline repeat_across_dims2!(v, ncopies) = @views v[:, 2:ncopies] .= v[:, 1]

"""
	weights_D₋ₓ!(v, Ωₕ::MeshType, ::Val{1})

Sets `v` to the inverse of the [spacing](@ref spacing(Ωₕ::Mesh1D, i)) of `Ωₕ` on the `x` component, ``h_{i}`` .
"""
@inline weights_D₋ₓ!(v, Ωₕ::MeshType, ::Val{1}) = invert_spacing!(v, Ωₕ)

"""
	weights_D₋ₓ!(v, Ωₕ::MeshType, ::Val{D})

Sets `v` to the inverse of the [spacing](@ref spacing(Ωₕ::MeshnD, i)) of `Ωₕ` on the `x` component, ``h_{x,i}`` .
"""
@inline function weights_D₋ₓ!(v, Ωₕ::MeshType, ::Val{D}) where D
	first_dims = prod(npoints(Ωₕ, Tuple)[1:(D - 1)])
	weights_D₋ₓ!(view(v, 1:first_dims), Ωₕ, Val(D - 1))

	repeat_across_dims1!(v, npoints(Ωₕ(D)), first_dims)
end

"""
	weights_D₋ᵧ!(v, Ωₕ::MeshType, ::Val{2})

Sets `v` to the inverse of the [spacing](@ref spacing(Ωₕ::MeshnD, i)) of `Ωₕ` on the `y` component, ``h_{y,j}`` .
"""
@inline function weights_D₋ᵧ!(v, Ωₕ::MeshType, ::Val{2})
	dims = npoints(Ωₕ, Tuple)
	first_dims = (dims[1], dims[2])

	aux2 = transpose(Base.ReshapedArray(v, first_dims, ()))
	t = view(aux2, 1:dims[2], 1)

	invert_spacing!(t, Ωₕ, 2)

	repeat_across_dims2!(aux2, dims[1])
end

"""
	weights_D₋ᵧ!(v, Ωₕ::MeshType, ::Val{3})

Sets `v` to the inverse of the [spacing](@ref spacing(Ωₕ::MeshnD, i)) of `Ωₕ` on the `y` component, ``h_{y,j}`` .
"""
@inline function weights_D₋ᵧ!(v, Ωₕ::MeshType, ::Val{3})
	dims = npoints(Ωₕ, Tuple)
	first_dims = dims[1] * dims[2]

	weights_D₋ᵧ!(view(v, 1:first_dims), Ωₕ, Val(2))

	repeat_across_dims1!(v, dims[3], first_dims)
end

"""
	weights_D₋₂!(v, Ωₕ::MeshType, ::Val{3})

Sets `v` to the inverse of the [spacing](@ref spacing(Ωₕ::MeshnD, i)) of `Ωₕ` on the `z` component, ``h_{z,l}`` .
"""
@inline function weights_D₋₂!(v, Ωₕ::MeshType, ::Val{3})
	dims = npoints(Ωₕ, Tuple)
	first_dims = dims[1] * dims[2]

	xx = Base.ReshapedArray(v, (first_dims, dims[3]), ())

	t = view(xx, 1, 1:dims[3])
	invert_spacing!(t, Ωₕ, 3)

	yy = transpose(xx)
	repeat_across_dims2!(yy, first_dims)
end

function _create_D₋ₓ(Ωₕ::MeshType; vector = _create_vector(Ωₕ))
	weights_D₋ₓ!(vector, Ωₕ, Val(dim(Ωₕ)))
	return Diagonal(vector) * diff₋ₓ(Ωₕ)
end

function _create_D₋ᵧ(Ωₕ::MeshType; vector = _create_vector(Ωₕ))
	weights_D₋ᵧ!(vector, Ωₕ, Val(dim(Ωₕ)))
	return Diagonal(vector) * diff₋ᵧ(Ωₕ)
end

function _create_D₋₂(Ωₕ::MeshType; vector = _create_vector(Ωₕ))
	weights_D₋₂!(vector, Ωₕ, Val(dim(Ωₕ)))
	return Diagonal(vector) * diff₋₂(Ωₕ)
end