for ___dim in var2symbol
    s = string("shift") * ___dim
    
    eval(quote
        #Mₕₓ(S::SType, ::Val{D}) where {D,SType<:SpaceType} = (shiftₓ(mesh(S), Val(D)) .+ shiftₓ(mesh(S), Val(D), -1)) ./ convert(eltype(S), 2.0)
        function $(Symbol(string("Mₕ")  * ___dim))(S::SpaceType)
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

        #jumpₓ(S::SType) where {SType<:SpaceType} = shiftₓ(mesh(S), Val(dim(SType))) .- shiftₓ(mesh(S), Val(dim(SType)), 1)
        #=function $(Symbol(string("jump")  * ___dim))(S::SpaceType)
            ex1 = $(Symbol(s))(mesh(S), Val(dim(S)), Val(0))
            ex2 = $(Symbol(s))(mesh(S), Val(dim(S)), Val(1))

            return ex1-ex2
        end=#
    end)
end

#=
diff(S::SType, ::Val{1}) where {SType<:SpaceType} = diffₓ(S)
diff(S::SType, ::Val{D}) where {D,SType<:SpaceType} = nothing
diff(S::SType) where {SType<:SpaceType} = diff(S, Val(dim(SType)))

jump(S::SType, ::Val{1}) where {SType<:SpaceType} = jumpₓ(S)
jump(S::SType, ::Val{D}) where {D,SType<:SpaceType} = nothing
jump(S::SType) where {SType<:SpaceType} = jump(S, Val(dim(SType)))

Mₕ(S::SType, ::Val{1}) where {SType<:SpaceType} = Mₕₓ(S)
Mₕ(S::SType, ::Val{D}) where {D,SType<:SpaceType} = nothing
Mₕ(S::SType) where {SType<:SpaceType} = Mₕ(S, Val(dim(SType)))
=#

for op in ("diff","Mₕ", "jump")
    eval(quote
        ## jump_back operators
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


#for op in ("Mₕ", "jump"), D in 1:3
#    @eval $(Symbol(op  * string("2dim")))(u::VecOrMatElem, ::Val{$D}) = $(Symbol((op) * string(var2symbol[D])))(u)
#send

#∇ₕ(u::VecOrMatElem) = ntuple(i -> D₋2dim(u, Val(i))::VecOrMatElem, Val(dim(u)))::NTuple{dim(u), VecOrMatElem}

#Mₕ(u::VecOrMatElem) = dim(u) == 1 ? typeof(u)(space(u), Mₕ(space(u))*u.values) : ntuple(i->Mₕ2dim(u, Val(i))::VecOrMatElem, Val(dim(u)))
#Mₕ(u::NTuple{D,VecOrMatElem}) where D = ntuple(i->Mₕ2dim(u[i], Val(i))::VecOrMatElem, D)

#jump(u::VecOrMatElem) = dim(u) == 1 ? typeof(u)(space(u), jump(space(u))*u.values) : ntuple(i->jump2dim(u, Val(i))::VecOrMatElem, Val(dim(u)))
#jump(u::NTuple{D,VecOrMatElem}) where D = ntuple(i->jump2dim(u[i], Val(i))::VecOrMatElem, D)

#diff(u::VecOrMatElem) = dim(u) == 1 ? typeof(u)(space(u), diff(space(u))*u.values) : ntuple(i->diff2dim(u, Val(i))::VecOrMatElem, Val(dim(u)))
#diff(u::NTuple{D,VecOrMatElem}) where D = ntuple(i->diff2dim(u[i], Val(i))::VecOrMatElem, D)


for op in (#:D₋ₓ, :D₋ᵧ, :D₋₂,
           #:jumpₓ, :jumpᵧ, :jump₂, #:jump, # 2D and 3D (untested)
           #:diffₓ, :diffᵧ, :diff₂,
            #D꜀, D꜀ₓ, D꜀ᵧ, D꜀₂,
           #:Mₕ, 
           #:Mₕₓ, :Mₕᵧ, :Mₕ₂,
           #:Dc, :Dstar_x, :Dh
           )

    @eval ($op)(u::VecOrMatElem) = typeof(u)(space(u), $op(space(u)) * u.values)

    @eval ($op)(u::AbstractVector) = $op(space(u)) * u
end