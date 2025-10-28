"""
	$(TYPEDEF)

Alias for storage of Dirichlet constraints.
"""
const DirichletConstraint{FType} = DomainMarkers{FType}

"""
	dirichlet_constraints(X::CartesianProduct, [I::CartesianProduct{1}], pairs...)

Creates Dirichlet boundary constraints where functions return values of type `eltype(X)`.

Each pair is of the form `:label => func`, where `:label` is a symbol identifying the region where the Dirichlet conditions are applied and `func` is a function or a [BrambleFunction](@ref) defining the values of the Dirichlet conditions. If `I` is provided, then `func` should be a time-dependent function of the form `func(x, t)`, where `x` is a point in the spatial domain and `t` is a point in the time domain.

The provided `:label` should match one of the labels used when creating the [AbstractMeshType](@ref) object where the Dirichlet conditions will be applied.
"""
#@inline dirichlet_constraints(X::CartesianProduct{D,T}, pairs::Pair...) where {D,T} = _create_generic_markers(T, X, pairs...)
#@inline dirichlet_constraints(X::CartesianProduct{D,T}, I::CartesianProduct{1}, pairs::Pair...) where {D,T} = _create_generic_markers(T, X, I, pairs...)
#@inline dirichlet_constraints(Wₕ::AbstractSpaceType, pairs::Pair...) = _create_generic_markers(eltype(Wₕ), set(mesh(Wₕ)), pairs...)
#@inline dirichlet_constraints(Wₕ::AbstractSpaceType, I::CartesianProduct{1}, pairs::Pair...) = _create_generic_markers(eltype(Wₕ), set(mesh(Wₕ)), I, pairs...)
"""
	dirichlet_constraints(cartesian_product, [I::CartesianProduct{1}], pairs...)

Creates Dirichlet boundary constraints.

Each `pair` is of the form `:label => func`, where `:label` identifies the boundary region and `func` defines the Dirichlet values. If the optional time domain `I` is provided, `func` should be a time-dependent function `func(x, t)`.

The `cartesian_product` can be a `CartesianProduct` mesh domain or an `ScalarGridSpace` from which the mesh can be extracted. The `:label` must match a label in the mesh definition.
"""
function dirichlet_constraints(cartesian_product, pairs::Pair...)
	T, domain = _get_eltype_and_domain(cartesian_product)
	_create_generic_markers(T, domain, pairs...)
end

function dirichlet_constraints(cartesian_product, I::CartesianProduct{1}, pairs::Pair...)
	T, domain = _get_eltype_and_domain(cartesian_product)
	_create_generic_markers(T, domain, I, pairs...)
end

# Helper to extract element type and domain from different sources
_get_eltype_and_domain(X::CartesianProduct{D,T}) where {D,T} = (T, X)
_get_eltype_and_domain(Wₕ::ScalarGridSpace) = (eltype(Wₕ), set(mesh(Wₕ)))

"""
	dirichlet_constraints(X::CartesianProduct, f::Function)

	Creates a single Dirichlet boundary constraint with function `f` with the label `:dirichlet`.
"""
@inline dirichlet_constraints(X::CartesianProduct, f::F) where F<:Function = dirichlet_constraints(X, :boundary => f)

#==============================================================================
						APPLYING DIRICHLET BOUNDARY CONDITIONS
==============================================================================#

"""
	dirichlet_bc!(A::AbstractMatrix, Ωₕ::AbstractMeshType, labels::Symbol...)

Applies Dirichlet boundary conditions to matrix `A` based on marked regions in the mesh `Ωₕ`.

For each index `i` associated with the given Dirichlet `labels`, this function:

 1. Sets all elements in the `i`-th row of `A` to zero.
 2. Sets the diagonal element `A[i, i]` to one.
"""
function dirichlet_bc!(A::AbstractMatrix, Ωₕ::AbstractMeshType, labels::Symbol...)
	for p in labels
		vec_bool = index_in_marker(Ωₕ, p)
		_dirichlet_bc_indices!(A, vec_bool)
	end
end

"""
	dirichlet_bc!(v::AbstractVector, Ωₕ::AbstractMeshType, bcs::DirichletConstraint, labels::Symbol...)

Apply Dirichlet boundary conditions to vector `v` using the [DirichletConstraint](@ref) object `bcs` and the mesh `Ωₕ`.
"""
function dirichlet_bc!(v::AbstractVector, Ωₕ::AbstractMeshType, bcs::DirichletConstraint, labels::Symbol...)
	isempty(labels) && return

	for marker in conditions(bcs)
		current_label = label(marker)
		if current_label in labels
			func = identifier(marker)
			marker_indices = index_in_marker(Ωₕ, current_label)
			_dirichlet_bc_indices!(v, Ωₕ, marker_indices, func)
		end
	end

	return
end

"""
	_dirichlet_bc_indices!(A, marker_indices)

Internal helper to apply Dirichlet boundary conditions to matrix `A` for a given set of indices.
"""
function _dirichlet_bc_indices!(A::AbstractMatrix, index_in_marker::BitVector)
	T = eltype(A)

	chunks = index_in_marker.chunks
	@inbounds for (chunk_idx, chunk) in enumerate(chunks)
		chunk == zero(UInt64) && continue # Skip chunks with no Dirichlet nodes

		offset = (chunk_idx - 1) * 64
		temp_chunk = chunk
		while temp_chunk != zero(UInt64)
			bit_pos = trailing_zeros(temp_chunk)
			i = offset + bit_pos + 1

			# Zero out the i-th row and set diagonal to one
			@views A[i, :] .= zero(T)
			A[i, i] = one(T)

			temp_chunk &= temp_chunk - 1 # Clear the processed bit
		end
	end
end

"""
_dirichlet_bc_indices!(A::SparseMatrixCSC, index_in_marker::BitVector)

Applies Dirichlet boundary conditions to a sparse matrix `A` by directly manipulating
its CSC data structure for high performance.
"""
function _dirichlet_bc_indices!(A::SparseMatrixCSC, index_in_marker::BitVector)
	T = eltype(A)
	rows = rowvals(A)
	vals = nonzeros(A)

	# 1. Zero out non-zero values in Dirichlet rows
	@inbounds for j in axes(A, 2)
		@simd for i in nzrange(A, j)
			if index_in_marker[rows[i]]
				vals[i] = zero(T)
			end
		end
	end

	# 2. Set diagonal elements to one for all Dirichlet rows
	chunks = index_in_marker.chunks
	@inbounds for (chunk_idx, chunk) in enumerate(chunks)
		chunk == zero(UInt64) && continue

		offset = (chunk_idx - 1) * 64
		temp_chunk = chunk
		while temp_chunk != zero(UInt64)
			bit_pos = trailing_zeros(temp_chunk)
			i = offset + bit_pos + 1
			A[i, i] = one(T)
			temp_chunk &= temp_chunk - 1
		end
	end
end

_function_in_linear_indices(func, Ωₕ, i) = func(point(Ωₕ, indices(Ωₕ)[i]))

function _dirichlet_bc_indices!(v::AbstractVector, Ωₕ::AbstractMeshType, index_in_marker::BitVector, func::BrambleFunction)
	g = PointwiseEvaluator(func, Ωₕ)
	cart_indices = indices(Ωₕ)

	chunks = index_in_marker.chunks
	@inbounds for (chunk_idx, chunk) in enumerate(chunks)
		chunk == zero(UInt64) && continue

		offset = (chunk_idx - 1) * 64
		temp_chunk = chunk
		while temp_chunk != zero(UInt64)
			bit_pos = trailing_zeros(temp_chunk)
			idx = offset + bit_pos + 1
			v[idx] = g(cart_indices[idx])
			temp_chunk &= temp_chunk - 1
		end
	end

	return
end

#==============================================================================
					SYMMETRIZATION OF THE LINEAR SYSTEM
==============================================================================#

function dirichlet_bc_symmetrize!(A::AbstractMatrix, F::AbstractVector, Ωₕ::AbstractMeshType, labels::Symbol...; dropzeros = false)
	dirichlet_bc!(A, Ωₕ, labels...)
	symmetrize!(A, F, Ωₕ, labels...)

	if dropzeros && A isa SparseMatrixCSC
		dropzeros!(A)
	end
end

"""
	symmetrize!(A, F, Ωₕ, labels)

Modifies the linear system `Ax = F` to make `A` symmetric after applying Dirichlet
conditions. It updates the vector `F` and zeros out the columns of `A` corresponding
to Dirichlet nodes.

The algorithm goes as follows: for any given row `i` where Dirichlet boundary conditions have been applied

	- calculate `dᵢ = cᵢ .* F`, where `cᵢ` is the `i`-th column of `A`;
	- replace `F` by substracting `dᵢ` to `F` (except for the `i`-th component)
	- replace all elements in the `i`-th column of `A` (except the `i`-th by zero).
"""
function symmetrize!(A::AbstractMatrix, F::AbstractVector, Ωₕ::AbstractMeshType, labels::Symbol...)
	for p in labels
		marker_indices = index_in_marker(Ωₕ, p)
		symmetrize!(A, F, marker_indices)
	end
end

# Generic implementation for dense matrices
function symmetrize!(A::AbstractMatrix, F::AbstractVector, index_in_marker::BitVector)
	dirichlet_indices = findall(index_in_marker)
	T = eltype(A)

	for i in dirichlet_indices
		dirichlet_val = F[i]
		for k in axes(A, 1)
			if i != k
				F[k] -= A[k, i] * dirichlet_val
				A[k, i] = zero(T)
			end
		end
	end

	return
end

# Implementation for sparse matrices
function symmetrize!(A::SparseMatrixCSC, F::AbstractVector, index_in_marker::BitVector)
	T = eltype(A)
	rows = rowvals(A)
	vals = nonzeros(A)

	chunks = index_in_marker.chunks
	@inbounds for (chunk_idx, chunk) in enumerate(chunks)
		chunk == zero(UInt64) && continue

		offset = (chunk_idx - 1) * 64
		temp_chunk = chunk
		while temp_chunk != zero(UInt64)
			bit_pos = trailing_zeros(temp_chunk)
			i = offset + bit_pos + 1

			dirichlet_val = F[i]

			# Update F and zero out column `i` using sparse structure
			@simd for k_ptr in nzrange(A, i)
				row_k = rows[k_ptr]
				F[row_k] -= vals[k_ptr] * dirichlet_val
				vals[k_ptr] = zero(T)
			end

			# Restore diagonal and RHS vector value
			A[i, i] = one(T)
			F[i] = dirichlet_val

			temp_chunk &= temp_chunk - 1
		end
	end

	return
end