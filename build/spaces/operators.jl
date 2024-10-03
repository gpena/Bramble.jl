for ___dim in var2symbol
    s = string("shift") * ___dim
    
    eval(quote
        #M₋ₕₓ(S::SType, ::Val{D}) where {D,SType<:SpaceType} = (shiftₓ(mesh(S), Val(D)) .+ shiftₓ(mesh(S), Val(D), -1)) ./ convert(eltype(S), 2.0)
        function $(Symbol(string("M₋ₕ")  * ___dim))(S::SpaceType)
            ex1 = $(Symbol(s))(mesh(S), Val(dim(S)), Val(0))
            ex2 = $(Symbol(s))(mesh(S), Val(dim(S)), Val(-1))
            return (ex1+ex2).*convert(eltype(S), 0.5)
        end

        #diffₓ(S::SType) where {SType<:SpaceType} = shiftₓ(mesh(S), Val(dim(SType))) .- shiftₓ(mesh(S), Val(dim(SType)), -1)
        #=function $(Symbol(string("diff")  * ___dim))(S::SpaceType)
            ex1 = $(Symbol(s))(mesh(S), Val(dim(S)), Val(0))
            ex2 = $(Symbol(s))(mesh(S), Val(dim(S)), Val(-1))

            return ex1-ex2
        end
        =#

        #diffₓ(S::SType) where {SType<:SpaceType} = shiftₓ(mesh(S), Val(dim(SType))) .- shiftₓ(mesh(S), Val(dim(SType)), 1)
        #=function $(Symbol(string("diff")  * ___dim))(S::SpaceType)
            ex1 = $(Symbol(s))(mesh(S), Val(dim(S)), Val(0))
            ex2 = $(Symbol(s))(mesh(S), Val(dim(S)), Val(1))

            return ex1-ex2
        end=#
    end)
end

#=
diffₕ(S::SType, ::Val{1}) where {SType<:SpaceType} = diffₓ(S)
diffₕ(S::SType, ::Val{D}) where {D,SType<:SpaceType} = nothing
diffₕ(S::SType) where {SType<:SpaceType} = diffₕ(S, Val(dim(SType)))

diffₕ(S::SType, ::Val{1}) where {SType<:SpaceType} = diffₓ(S)
diffₕ(S::SType, ::Val{D}) where {D,SType<:SpaceType} = nothing
diffₕ(S::SType) where {SType<:SpaceType} = diffₕ(S, Val(dim(SType)))

M₋ₕ(S::SType, ::Val{1}) where {SType<:SpaceType} = M₋ₕₓ(S)
M₋ₕ(S::SType, ::Val{D}) where {D,SType<:SpaceType} = nothing
M₋ₕ(S::SType) where {SType<:SpaceType} = M₋ₕ(S, Val(dim(SType)))
=#

for op in ("diff","M₋ₕ", "diff")
    eval(quote
        ## diff_back operators
        #$(Symbol(string(op)))(U::SpaceType, ::Val{1}) = $(Symbol(string(op) * string(var2symbol[1])))(U)
        #$(Symbol(string(op)))(U::SpaceType, ::Val{D}) where D = @error "Not implemented!!!"
        #$(Symbol(string(op)))(S::SpaceType) = Val(dim(S)) isa Val{1} ? $(Symbol(string(op) * string(var2symbol[1])))(S) : @error "Not implemented!!!"
    end)
end




#=
# implementation of D₋ₓ and family
for (i,v) in enumerate(var2symbol)
    diffop = "D₋" * v
    #csymb = "Mat_" * diffop

    eval(quote
        function $(Symbol(diffop))(S::SpaceType) 
            #getcache(S, Symbol($csymb))
            diagonal = _create_diagonal(S)
            $(Symbol("weights_D₋" * v * "!"))(diagonal.diag, mesh(S), Val(dim(S)))
            return diagonal * $(Symbol("diff" * v))(S)
        end
    end)
end
=#


#for op in ("M₋ₕ", "diff"), D in 1:3
#    @eval $(Symbol(op  * string("2dim")))(u::VecOrMatElem, ::Val{$D}) = $(Symbol((op) * string(var2symbol[D])))(u)
#send

#∇₋ₕ(u::VecOrMatElem) = ntuple(i -> D₋2dim(u, Val(i))::VecOrMatElem, Val(dim(u)))::NTuple{dim(u), VecOrMatElem}

#M₋ₕ(u::VecOrMatElem) = dim(u) == 1 ? typeof(u)(space(u), M₋ₕ(space(u))*u.values) : ntuple(i->M₋ₕ2dim(u, Val(i))::VecOrMatElem, Val(dim(u)))
#M₋ₕ(u::NTuple{D,VecOrMatElem}) where D = ntuple(i->M₋ₕ2dim(u[i], Val(i))::VecOrMatElem, D)

#diffₕ(u::VecOrMatElem) = dim(u) == 1 ? typeof(u)(space(u), diffₕ(space(u))*u.values) : ntuple(i->diff2dim(u, Val(i))::VecOrMatElem, Val(dim(u)))
#diffₕ(u::NTuple{D,VecOrMatElem}) where D = ntuple(i->diff2dim(u[i], Val(i))::VecOrMatElem, D)

#diffₕ(u::VecOrMatElem) = dim(u) == 1 ? typeof(u)(space(u), diffₕ(space(u))*u.values) : ntuple(i->diff2dim(u, Val(i))::VecOrMatElem, Val(dim(u)))
#diffₕ(u::NTuple{D,VecOrMatElem}) where D = ntuple(i->diff2dim(u[i], Val(i))::VecOrMatElem, D)


for op in (#:D₋ₓ, :D₋ᵧ, :D₋₂,
           #:diffₓ, :diffᵧ, :diff₂, #:diff, # 2D and 3D (untested)
           #:diffₓ, :diffᵧ, :diff₂,
            #D꜀, D꜀ₓ, D꜀ᵧ, D꜀₂,
           #:M₋ₕ, 
           #:M₋ₕₓ, :M₋ₕᵧ, :M₋ₕ₂,
           #:Dc, :Dstar_x, :Dh
           )

    @eval ($op)(u::VecOrMatElem) = typeof(u)(space(u), $op(space(u)) * u.values)

    @eval ($op)(u::AbstractVector) = $op(space(u)) * u
end