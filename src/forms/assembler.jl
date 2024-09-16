assemble(b::BilinearForm) = b.form_expr(Elements(test_space(b)), Elements(trial_space(b)))#::SparseMatrixCSC{eltype(test_space(b)),Int}

assemble!(A::AbstractMatrix, b::BilinearForm) = ( copyto!(A, assemble(b)) )

function assemble(b::BilinearForm, bcs::DirichletBCs)
    A = assemble(b)

    apply_dirichlet_bc!(A, bcs, mesh(test_space(b)))
    return A
end

assemble!(A::AbstractMatrix, b::BilinearForm, bcs::DirichletBCs) = ( copyto!(A, assemble(b, bcs)) )

function assemble(l::LinearForm{S,F}) where {S,F} 
    z = Elements(trial_space(l))

    return l.form_expr(z)
end






function assemble!(x::AbstractVector, l::LinearForm{S, F}) where {S, F}
    #=vec = 0.0*Element(l.space)
    for i in eachindex(x)
        vec[i] = 1.0
        x[i] = l.form_expr(vec)
        vec[i] = 0.0
    end
        
    =#x .= l.form_expr(Elements(trial_space(l)))
end














function assemble(l::LinearForm, bcs::DirichletBCs)
    vec = assemble(l)
    apply_dirichlet_bc!(vec, bcs, mesh(trial_space(l)))

    return vec
end

function assemble!(vec::AbstractVector, l::LinearForm, bcs::DirichletBCs) 
    assemble!(vec, l)
    apply_dirichlet_bc!(vec, bcs, mesh(trial_space(l)))
end