struct SparsityPattern
    I::Vector{Int64}
    J::Vector{Int64}
end

function SparsityPattern(A::MType) where MType <: AbstractArray
    I,J,_ = findnz(A)
    return SparsityPattern(I,J)
end

function SparsityPattern!(I::VType, J::VType, A::MType) where {VType <: AbstractVector, MType <: AbstractArray}
    IA,JA,_ = findnz(A)
    I .= IA
    J .= JA
end

function __update!(I, J, bs, be, k)
    _min, _max = minmax(0,k)
    L = length(bs:be)

    @inbounds for j in 1:L
        J[bs+j-1] = j + _max
        I[bs+j-1] = j - _min
    end
end

function __update!(IDX, bs, be, k)
    _min, _max = minmax(0,k)
    L = be-bs+1

    for j in 1:L
        IDX[bs+j-1, 1] = j - _min
        IDX[bs+j-1, 2] = j + _max
    end
end

function SparsityPattern(bform::BType) where {BType}
    u = Element(bform.space1, 0.0)
    M = mesh(bform.space1)
    N = npoints(M)

    nsamples = 3
    samples = Vector{Int64}(undef, nsamples)
    
    rand!(samples,1:npoints(M))
    
    V = Elements(bform.space2)
    b = copy(u.values)

    diags = Set{Int64}()

    #collect diagonals
    for i in eachindex(samples)
        u.values[samples[i]] = 1.0
    
        b .= bform.form_expr(u,V)

        for j in findall(b .!= 0.0)
            push!(diags, j-samples[i])
        end

        u.values[samples[i]] = 0.0
    end

    nonzeros = sum(N-abs(k) for k in diags) 

    IDX = Array{Int64, 2}(undef, nonzeros, 2)

    block_start = 1
    block_end = 1
    for k in diags
        block_end = block_start+N-abs(k)-1
        __update!(IDX, block_start, block_end, k)
        block_start = block_end+1
    end
    println("diagonals $diags")
 #=  
    I = Vector{Int64}(undef, nonzeros)
    J = Vector{Int64}(undef, nonzeros)

    counter = 1
    for k in diags
        _min, _max = minmax(0,k)
         for j in 1:N-abs(k)
            J[counter] = j + _max
            I[counter] = j - _min
            counter += 1
        end
    end
=#
    # sort vectors: first order J and then order I
    #println(IDX)
    #I = IDX[:,1]
    #J = IDX[:,2]

    IDX .= sortslices(IDX, dims=1, lt=(x,y)-> x[2]==y[2] ? isless(x[1],y[1]) : isless(x[2],y[2]))

    return SparsityPattern(IDX[:,1],IDX[:,2])
end


function SparsityPattern(form_expr::F, space1::S1, space2::S2) where {F<:Function, S1<:SpaceType, S2<:SpaceType}
    u = Element(space1, 0.0)
    M = mesh(space1)
    N = npoints(M)
    #println("calculate sparsity pattern")
    nsamples = 3
    samples = Vector{Int64}(undef, nsamples)
    
    rand!(samples,1:npoints(M))
    
    V = Elements(space2)
    b = copy(u.values)

    diags = Set{Int64}(0)

    #collect diagonals
    for i in eachindex(samples)
        u.values[samples[i]] = 1.0
    
        b .= form_expr(u,V)

        for j in findall(b .!= 0.0)
            push!(diags, j-samples[i])
        end

        u.values[samples[i]] = 0.0
    end

    nonzeros = sum(N-abs(k) for k in diags) 

    IDX = Array{Int64, 2}(undef, nonzeros, 2)

    block_start = 1
    block_end = 1
    for k in diags
        block_end = block_start+N-abs(k)-1
        __update!(IDX, block_start, block_end, k)
        block_start = block_end+1
    end
    #println(diags)
 #=  
    I = Vector{Int64}(undef, nonzeros)
    J = Vector{Int64}(undef, nonzeros)

    counter = 1
    for k in diags
        _min, _max = minmax(0,k)
         for j in 1:N-abs(k)
            J[counter] = j + _max
            I[counter] = j - _min
            counter += 1
        end
    end
=#
    # sort vectors: first order J and then order I
    #println(IDX)
    #I = IDX[:,1]
    #J = IDX[:,2]

    IDX .= sortslices(IDX, dims=1, lt=(x,y)-> x[2]==y[2] ? isless(x[1],y[1]) : isless(x[2],y[2]))
    #println(IDX)
    return SparsityPattern(IDX[:,1],IDX[:,2])
end