@inline dirichletbcs(p::Pair...) = dirichletbcs(Val(length(p)), p...)

for D in 1:3
	@eval dirichletbcs(::Val{$D}, p::Pair...) = $(Symbol("DirichletBCs" * string(D)))(Tuple(create_markers(p...)))
end

@inline dirichletbcs(f::F) where F = dirichletbcs( :Dirichlet => f)

@inline markers(S::DirichletBCs) = S.markers
@inline symbols(S::DirichletBCs) = (p.symbol for p in S)
@inline labels(S::DirichletBCs) = (p.label for p in S)

function apply_dirichlet_bc!(A::AbstractMatrix, bcs::DirichletBCs, M::MeshType)
	npts = npoints(M, Tuple)

	for p in markers(bcs)
		_apply_dirichlet_bc!(A, npts, M.markers[p.label])
	end
end

function __set_diag_one(A::AbstractMatrix, npts, rows)
	for idx in rows # can be made parallel
		i = sub2ind(npts, idx)
		A[i, i] = one(eltype(A))
	end
end

function __set_rows_zero(A::SparseMatrixCSC{T,Int}, npts, rows) where T
	for idx in rows
		i = sub2ind(npts, idx)
		@views A[i, :] .= zero(T)
	end
	#=
	for row in rows
	for col in 1:size(A, 2)
		start = colptr[col]
		stop = colptr[col + 1] - 1
		for i in start:stop
			if rowval[i] == row
				A.nzval[i] = zero(T)
			end
		end
	end
	end
	=#
end

function symmetrize!(A, F, bcs::DirichletBCs, M::MeshType)
	npts = npoints(M, Tuple)

	for p in markers(bcs)
		rows = M.markers[p.label]
		#@show rows
		for idx in rows
			i = sub2ind(npts, idx)
			aux = F[i]
			v = @views A[:,i]
			F .-= v .* aux
			v .= 0.0
			A[i,i] = 1.0
			F[i] = aux
		end
	end
end


function _apply_dirichlet_bc!(A::AbstractMatrix, npts, rows)
	__set_rows_zero(A, npts, rows)
	__set_diag_one(A, npts, rows)
end

function sub2ind(sz, args...)
	linidx = Base.LinearIndices(sz)
	getindex.([linidx], args...)
end

# apply bc to RHS
function apply_dirichlet_bc!(v::AbstractVector, bcs::DirichletBCs, M::MeshType)
	pts = points(M)
	npts = npoints(M, Tuple)

	for p in markers(bcs) # can be made parallel
		indices = M.markers[p.label]
		for idx in indices
			v[sub2ind(npts, idx)] = p.f(_i2p(pts, idx))
		end
	end
end
