###############################################################
#                                                             #
# Implementation of the average operators for vector elements #
#                                                             #
###############################################################

# Average operator as a matrix
"""
	Mₕₓ(Wₕ::SpaceType)

Returns a [MatrixElement](@ref) implementing the average matrix for the mesh grid of `Wₕ`, in the `x` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [`Mₕₓ(uₕ::VectorElement)`](@ref).
"""
@inline Mₕₓ(Wₕ::SpaceType) = elements(Wₕ, Mₕₓ(mesh(Wₕ)))
@inline Mₕₓ(Ωₕ::MeshType) = (shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(0)) + shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(-1))) * convert(eltype(Ωₕ), 0.5)::eltype(Ωₕ)

"""
	Mₕᵧ(Wₕ::SpaceType)

Returns a [MatrixElement](@ref) implementing the average matrix for the mesh grid of `Wₕ`, in the `y` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [`Mₕᵧ(uₕ::VectorElement)`](@ref).
"""
@inline Mₕᵧ(Wₕ::SpaceType) = elements(Wₕ, Mₕᵧ(mesh(Wₕ)))
@inline Mₕᵧ(Ωₕ::MeshType) = (shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(0)) + shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(-1))) * convert(eltype(Ωₕ), 0.5)::eltype(Ωₕ)

"""
	Mₕ₂(Wₕ::SpaceType)

Returns a [MatrixElement](@ref) implementing the average matrix for the mesh grid of `Wₕ`, in the `z` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [`Mₕ₂(uₕ::VectorElement)`](@ref).
"""
@inline Mₕ₂(Wₕ::SpaceType) = elements(Wₕ, Mₕ₂(mesh(Wₕ)))
@inline Mₕ₂(Ωₕ::MeshType) = (shift₂(Ωₕ, Val(dim(Ωₕ)), Val(0)) + shift₂(Ωₕ, Val(dim(Ωₕ)), Val(-1))) * convert(eltype(Ωₕ), 0.5)::eltype(Ωₕ)

@inline Mₕᵢ(Wₕ::SpaceType, ::Val{1}) = Mₕₓ(Wₕ)
@inline Mₕᵢ(Wₕ::SpaceType, ::Val{2}) = Mₕᵧ(Wₕ)
@inline Mₕᵢ(Wₕ::SpaceType, ::Val{3}) = Mₕ₂(Wₕ)

"""
	Mₕ(Wₕ::SpaceType)

Returns a tuple of [MatrixElement](@ref)s implementing the average operators in the `x`, `y`, and `z` directions.
"""
@inline Mₕ(Wₕ::SpaceType) = ntuple(i -> Mₕᵢ(Wₕ, Val(i)), dim(mesh(Wₕ)))

# Average operator applied to vectors
"""
	Mₕₓ(uₕ::VectorElement)

Returns the average, in the `x` direction, of the element `uₕ`.

  - 1D case

```math
\\textrm{M}_{hx} \\textrm{u}_h(x_i) = \\frac{\\textrm{u}_h(x_i) + \\textrm{u}_h(x_{i-1})}{2}
```

  - 2D and 3D case

```math
\\textrm{M}_{hx} \\textrm{u}_h(x_i, \\dots) = \\frac{\\textrm{u}_h(x_i, \\dots)+\\textrm{u}_h(x_{i-1}, \\dots)}{2}
```
"""
@inline Mₕₓ(uₕ::VectorElement) = element(space(uₕ), Mₕₓ(space(uₕ)).values * uₕ.values)

"""
	Mₕᵧ(uₕ::VectorElement)

Returns the Mₕ, in the `y` direction, of the element `uₕ`.

```math
\\textrm{Mₕ}_{hy} \\textrm{u}_h(x_i, y_j,\\dots) = \\textrm{u}_h(x_i, y_j\\dots)-\\textrm{u}_h(x_i, y_{j+1}, \\dots)
```
"""
@inline Mₕᵧ(uₕ::VectorElement) = element(space(uₕ), Mₕᵧ(space(uₕ)).values * uₕ.values)

"""
	Mₕ₂(uₕ::VectorElement)

Returns the average, in the `z` direction, of the element `uₕ`.

```math
\\textrm{Mₕ}_{hz} \\textrm{u}_h(x_i, y_j,z_l) = \\frac{\\textrm{u}_h(x_i, y_j, z_l)+\\textrm{u}_h(x_i, y_j, z_{l-1})}{2}
```
"""
@inline Mₕ₂(uₕ::VectorElement) = element(space(uₕ), Mₕ₂(space(uₕ)).values * uₕ.values)

@inline Mₕᵢ(uₕ::VectorElement, ::Val{1}) = Mₕₓ(uₕ)
@inline Mₕᵢ(uₕ::VectorElement, ::Val{2}) = Mₕᵧ(uₕ)
@inline Mₕᵢ(uₕ::VectorElement, ::Val{3}) = Mₕ₂(uₕ)

"""
	Mₕ(uₕ::VectorElement)

Returns a tuple of [VectorElement](@ref)s implementing the average operators in the `x`, `y`, and `z` directions applied to `uₕ`.
"""
@inline Mₕ(uₕ::VectorElement) = ntuple(i -> element(space(uₕ), Mₕᵢ(space(uₕ), Val(i)).values * uₕ.values), dim(mesh(space(uₕ))))

# Average operator applied to matrices
"""
	Mₕₓ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the Mₕ matrix [Mₕₓ(Wₕ::SpaceType)](@ref) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline Mₕₓ(Uₕ::MatrixElement) = elements(space(Uₕ), Mₕₓ(space(Uₕ)).values * Uₕ.values)

"""
	Mₕᵧ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the Mₕ matrix [Mₕᵧ(Wₕ::SpaceType)](@ref) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline Mₕᵧ(Uₕ::MatrixElement) = elements(space(Uₕ), Mₕᵧ(space(Uₕ)).values * Uₕ.values)

"""
	Mₕ₂(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the Mₕ matrix [Mₕ₂(Wₕ::SpaceType)](@ref) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline Mₕ₂(Uₕ::MatrixElement) = elements(space(Uₕ), Mₕ₂(space(Uₕ)).values * Uₕ.values)

@inline Mₕᵢ(Uₕ::MatrixElement, ::Val{1}) = Mₕₓ(Uₕ)
@inline Mₕᵢ(Uₕ::MatrixElement, ::Val{2}) = Mₕᵧ(Uₕ)
@inline Mₕᵢ(Uₕ::MatrixElement, ::Val{3}) = Mₕ₂(Uₕ)

"""
	Mₕ(Uₕ::MatrixElement)

Returns a tuple of [MatrixElement](@ref)s implementing the Mₕ operators in the `x`, `y`, and `z` directions applied to `Uₕ`. In the 1D case, it returns just the average.
"""
@inline Mₕ(Uₕ::MatrixElement) = Mₕ(Uₕ, Val(dim(mesh(space(Uₕ)))))
@inline Mₕ(Uₕ::MatrixElement, ::Val{1}) = Mₕₓ(Uₕ)
@inline Mₕ(Uₕ::MatrixElement, ::Val{D}) where D = ntuple(i -> Mₕᵢ(Uₕ, Val(i)) * Uₕ, dim(mesh(space(Uₕ))))

#=
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
=#