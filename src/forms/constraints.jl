"""
	ConstraintsType

Abstract type for boundary conditions contraints.
"""
abstract type ConstraintsType <: BrambleType end

"""
	struct Constraints{D,FType} <: ConstraintsType
		markers::NTuple{D,Marker{FType}}
		constraint_type::Symbol
	end

Structure to store boundary constraints. The `constraint_type` should be a symbol identifying the type of conditions (currently, only `:dirichlet` is supported). The tuple `markers` stores all the [Marker](@ref)s informations related with the constraints.
"""
struct Constraints{D,FType} <: ConstraintsType
	markers::NTuple{D,Marker{FType}}
	constraint_type::Symbol
end

"""
	constraints(pairs::NTuple{D,MarkerType}, type::Symbol = :dirichlet)

Returns a [Constraints](@ref) object from a tuple of [Marker](@ref)s and a symbol
defining the type of boundary condition. Currently, the only supported type is for Dirichlet boundary conditions. The default type is `:dirichlet`.
"""
function constraints(pairs::Vararg{MarkerType{BrambleFunction{A,B,C}},N}; type::Symbol = :dirichlet) where {A,B,C,N}
	@assert bool && type == :dirichlet
	mrks = create_markers(ntuple(i -> pairs[i], N)...)
	return Constraints{N,BrambleFunction{A,B,C}}(mrks, type)
end

"""
	constraints(f::BrambleFunction; type::Symbol = :dirichlet)

Returns a [Constraints](@ref) object from a single [BrambleFunction](@ref) and a symbol
defining the type of boundary condition.
"""
function constraints(f::BrambleFunction{A,B,C}; type::Symbol = :dirichlet) where {A,B,C}
	@assert type == :dirichlet
	mrks = create_markers(:dirichlet => f)
	return Constraints{1,BrambleFunction{A,B,C}}(mrks, type)
end

"""
	markers(bcs::Constraints)

Returns the [Marker](@ref)s stored in the [Constraints](@ref) object `bcs`.
"""
@inline markers(bcs::Constraints) = bcs.markers

"""
	constraint_type(bcs::Constraints)

Returns the symbol defining the type of boundary condition stored in the [Constraints](@ref) object `bcs`.
"""
@inline constraint_type(bcs::Constraints) = bcs.constraint_type

"""
	symbols(bcs::Constraints)

Returns an iterator over the symbols of the [Marker](@ref)s stored in the [Constraints](@ref) object `bcs`.
"""
@inline symbols(bcs::Constraints) = (symbol(bc) for bc in bcs)

"""
	labels(bcs::Constraints)

Returns an iterator over the labels of the [Marker](@ref)s stored in the [Constraints](@ref) object `bcs`.
"""
@inline labels(bcs::Constraints) = (label(bc) for bc in bcs)

"""
	apply_dirichlet_bc!(A, bcs::Constraints, M::MeshType)

Apply Dirichlet boundary conditions to matrix `A` using the [Constraints](@ref) object `bcs`
and the mesh `M`. For each index `i` associated with a Dirichlet boundary condition, we set the `i`-th row of matrix `A` to zero and change the diagonal element `A[i,i]` to 1.
"""
function apply_dirichlet_bc!(A::AbstractMatrix, bcs::Constraints{N,BFType}, Ωₕ::MeshType) where {N,BFType}
	@assert constraint_type(bcs) == :dirichlet
	npts = npoints(Ωₕ, Tuple)

	for p in markers(bcs)
		idxs = marker(Ωₕ, label(p))
		_apply_dirichlet_bc!(A, npts, idxs)
	end
end

"""
	__set_diag_one(A::AbstractMatrix, npts, rows)

Set the diagonal elements of matrix `A` to one for the given rows.
"""
function __set_diag_one(A::AbstractMatrix, npts, indices)
	T = eltype(A)
	for idx in indices
		i = sub2ind(npts, idx)
		A[i, i] = one(T)
	end
end

"""
	__set_rows_zero(A::AbstractMatrix, npts, rows)

Set the elements of matrix `A` to zero for the given rows.
"""
function __set_rows_zero(A::AbstractMatrix, npts, indices)
	T = eltype(A)
	for idx in indices
		i = sub2ind(npts, idx)
		@views A[i, :] .= zero(T)
	end
end

"""
	symmetrize!(A, F, bcs::Constraints, Ωₕ::MeshType)

After Dirichlet boundary conditions are applied to matrix `A` and vector `F` using the [Constraints](@ref) object `bcs`, this function allows to make `A` symmetric, if the original matrix (before applying boundary conditions was symmetric). The algorithm goes as follows: for any given row `i` where Dirichlet boundary conditions have been applied

	- calculate `dᵢ = cᵢ .* F`, where `cᵢ` is the `i`-th column of `A`;
	- replace `F` by substracting `dᵢ` to `F` (except for the `i`-th component)
	- replace all elements in the `i`-th column of `A` (except the `i`-th by zero).
"""
function symmetrize!(A::AbstractMatrix, F::AbstractVector, bcs::Constraints, Ωₕ::MeshType)
	npts = npoints(Ωₕ, Tuple)
	T = eltype(A)

	for p in markers(bcs)
		indices = marker(Ωₕ, label(p))
		for idx in indices
			i = sub2ind(npts, idx)
			aux = F[i]
			v = @views A[:, i]
			F .-= v .* aux
			v .= zero(T)
			A[i, i] = one(T)
			F[i] = aux
		end
	end
end

"""
	_apply_dirichlet_bc!(A::AbstractMatrix, npts, rows)

Apply Dirichlet boundary conditions to matrix `A` for the given `rows``.
"""
function _apply_dirichlet_bc!(A::AbstractMatrix, npts, rows)
	__set_rows_zero(A, npts, rows)
	__set_diag_one(A, npts, rows)
end

@generated function sub2ind(dims::NTuple{D,Int}, I::CartesianIndex{D}) where D
	ex = :(I[$D] - 1)
	for i ∈ (D - 1):-1:1
		ex = :(I[$i] - 1 + dims[$i] * $ex)
	end
	return :($ex + 1)
end

"""
	apply_dirichlet_bc!(v::AbstractVector, bcs::Constraints, Ωₕ::MeshType)

Apply Dirichlet boundary conditions to vector `v` using the [Constraints](@ref) object `bcs`
and the mesh `Ωₕ`.
"""
function apply_dirichlet_bc!(v::AbstractVector, bcs::Constraints{N,BFType}, Ωₕ) where {N,BFType}
	@assert constraint_type(bcs) == :dirichlet
	npts = npoints(Ωₕ, Tuple)
	pts = points(Ωₕ)
	f = Base.Fix1(_i2p, pts)
	
	for _marker in markers(bcs), idx in marker(Ωₕ, label(_marker))
		i = sub2ind(npts, idx)
		v[i] = func(_marker)(f(idx))
	end
end