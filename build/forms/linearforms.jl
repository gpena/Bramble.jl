abstract type LinearFormType <: BrambleType end

struct LinearForm{S,F} <: LinearFormType
    form_expr::F
    space::S
end

# define functions for creating and updating linear forms
test_space(l::LinearForm) = l.space


struct LinearMass{S,T} <: LinearFormType
    space::S
    vec::Vector{T}
end






























mass(u::VectorElement, V::MatrixElement) = mass(u, V, Val(dim(u)))

function mass(u::VectorElement, _, ::Val{1})
    #x = similar(u.values)
    #for (i, hi, ui) in zip(eachindex(x), hmeanit(mesh(u)), u.values) 
    #    x[i] = hi * ui
    #end
    h = hmeanit(mesh(u))

    return @.. h * u.values
end

stiffness(u::VectorElement, V::MatrixElement) = stiffness(u, V, Val(dim(u)))

function stiffness(u::VectorElement, _, ::Val{1})
    #x .= D₋ₓ(u).values
    #x .= 0.0
    x = similar(u.values)
    N = length(x)
    
    h = Base.Fix1(Bramble.hspace, mesh(u))
    backward_differencex!(x, u.values, h, npoints(mesh(u)))
    
    x[1] = x[2]

    if N > 2
        x[2] = x[2] - x[3]

        for i in 3:N-1
            x[i] = x[i] - x[i+1]
        end
    end

    x[1] *= -1.0
#println(x)
    return x
    #@views x = D₋ₓ(u).values[2:end]

    #@views result = [-x[1]; x[1:end-1] .- x[2:end]; x[end]]
    #return result
end


## implementation of mass operator with scaling: LinearForm(U -> innerₕ(uh, U), Wh)
Mass(s::SpaceType) = LinearMass{typeof(s), eltype(s)}(s, vec(Ones(eltype(s), ndofs(s))))

update!(m::LinearMass, v::AbstractFloat) = (fill!(m.vec, v))
update!(m::LinearMass, v::VectorElement) = (m.vec .= v)

assemble(m::LinearMass) = (m.vec .* m.space.innerh_weights)::Vector{eltype(m.vec)}

function assemble(m::LinearMass, bcs::DirichletBCs)
    u = assemble(m)
    apply_dirichlet_bc!(u, bcs, mesh(m.space))

    return u
end

function assemble!(u::AbstractVector, m::LinearMass) 
    #@tullio u[i] = m.space.innerh_weights.diag[i] * m.vec[i]
    #for i in eachindex(u)
    y = m.space.innerh_weights.diag
    x = m.vec
    @.. u = y * x
    #end
end

function assemble!(u::AbstractVector, m::LinearMass, bcs::DirichletBCs)
    assemble!(u, m)
    apply_dirichlet_bc!(u, bcs, mesh(m.space))
end