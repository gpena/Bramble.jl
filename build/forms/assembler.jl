assemble(b::BilinearForm) = b.form_expr(elements(test_space(b)), elements(trial_space(b)))#::SparseMatrixCSC{eltype(test_space(b)),Int}

assemble!(A::AbstractMatrix, b::BilinearForm) = ( copyto!(A, assemble(b)) )

function assemble(b::BilinearForm, bcs::DirichletBCs)
    A = assemble(b)

    apply_dirichlet_bc!(A, bcs, mesh(test_space(b)))
    return A
end

assemble!(A::AbstractMatrix, b::BilinearForm, bcs::DirichletBCs) = ( copyto!(A, assemble(b, bcs)) )

function assemble(l::LinearForm{S,F}) where {S,F} 
    z = elements(test_space(l))

    return l.form_expr(z)
end






function assemble!(x::AbstractVector, l::LinearForm{S, F}) where {S, F}
    x .= l.form_expr(elements(test_space(l)))
end














function assemble(l::LinearForm, bcs::DirichletBCs)
    vec = assemble(l)
    apply_dirichlet_bc!(vec, bcs, mesh(test_space(l)))

    return vec
end

function assemble!(vec::AbstractVector, l::LinearForm, bcs::DirichletBCs) 
    assemble!(vec, l)
    apply_dirichlet_bc!(vec, bcs, mesh(test_space(l)))
end



function spmatmul_rvec!(M::AbstractSparseMatrix, v::AbstractVector) 
    @assert length(v) == size(M)[1] == size(M)[2]

    m, _ = size(M)

    for j = 1:m
        vj = v[j]
        @simd for i in nzrange(M, j)
             M.nzval[i] *= vj
        end
    end
end


function spmatmul_lvec!(M::AbstractSparseMatrix, v::AbstractVector) 
    @assert length(v) == size(M)[1] == size(M)[2]
    rows = rowvals(M)
    m, _ = size(M)

    for j = 1:m
        vj = v[rows[j]]
        @simd for i in nzrange(M, j)
             M.nzval[i] *= vj
        end
    end
end