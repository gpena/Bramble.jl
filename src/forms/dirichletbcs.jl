@inline dirichletbcs(p::Pair...) = dirichletbcs(Val(length(p)), p...)

for D in 1:3
    @eval dirichletbcs(::Val{$D}, p::Pair...) = $(Symbol("DirichletBCs"*string(D)))( Tuple(markers(p...)) )
end

@inline dirichletbcs(f::F) where F<:Function = dirichletbcs("Dirichlet" => f)

@inline markers(S::DirichletBCs) = S.markers
@inline symbols(S::DirichletBCs) = (p.symbol for p in S)
@inline labels(S::DirichletBCs) = (p.label for p in S)

function apply_dirichlet_bc!(A::AbstractMatrix, bcs::DirichletBCs, M::MeshType)
    npts = npoints(M)

    for p in markers(bcs)
        _apply_dirichlet_bc!(A, npts, M.markers[p.label])
    end
end

function __set_diag_one(A::AbstractMatrix, npts, rows)
    for idx in rows # can be made parallel
        i = sub2ind(npts, idx)
        A[i,i] = one(eltype(A))
    end
end

function __set_rows_zero(A::SparseMatrixCSC{T,Int}, npts, rows) where T
    Threads.@threads for i in eachindex(A.rowval) # can be made parallel
        idx = CartesianIndex(Base._ind2sub(npts, A.rowval[i]))
       
        if idx in rows
            A.nzval[i] = zero(eltype(A)) 
        end
    end
end

function _apply_dirichlet_bc!(A::AbstractMatrix, npts, rows)
    __set_rows_zero(A, npts, rows)
    __set_diag_one(A, npts, rows)
end

# apply bc to RHS
function apply_dirichlet_bc!(v::AbstractVector, bcs::DirichletBCs, M::MeshType)
    pts = points(M)
    npts = npoints(M)

    for p in markers(bcs) # can be made parallel
        indices = M.markers[p.label]
        for idx in indices
            v[sub2ind(npts, idx)] = p.f(_index2point(pts, idx))
        end
    end
end
