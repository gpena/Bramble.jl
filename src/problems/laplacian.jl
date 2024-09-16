__stiffness_form(U,V) = inner₊(∇ₕ(U),∇ₕ(V))

function solveLaplacian(problem::Laplacian{DType, Fx, Gx}, nPoints::NTuple{D, Int}, unif::NTuple{D,Bool}) where {D, DType, Fx, Gx}
    Mh = Mesh(problem.X, nPoints, unif)
    Wh = GridSpace(Mh)
    bc = dirichletbcs(problem.u)

    bform = Diff(Wh, Wh)#BilinearForm(__stiffness_form, Wh, Wh)#Diff(Wh, Wh)#
    A = assemble(bform, bc)

    uh = Element(Wh)
    avgₕ!(uh, problem.f)

    lform = Mass(Wh)
    update!(lform, uh)
    F = assemble(lform, bc)
    
    fact = ilu(A, τ = 0.0001)

    solve!(uh, A, F, KrylovJL_GMRES(), prec = fact)
    F .= uh.values
    Rₕ!(uh, problem.u)
    uh.values .-= F

    return hₘₐₓ(mesh(Wh)), norm₁ₕ(uh)
end