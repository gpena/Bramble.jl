module testcase

using Bramble.Tools: dot_3vec

using Bramble.Geometry
using Bramble.Meshes
import Bramble.Meshes: hmean_extended, hmean, hspace, xmean, _createpoints!, createpoints
import Bramble.Meshes: hmean_extended!, hmean!, hspace!, xmean!, meas_cell, points, npoints
using Bramble.Spaces
using Bramble.Spaces:
    Elements,
    build_D_x,
    weights_D_x,
    invert_hspace,
    innerh_diagonal!,
    _inner_product,
    shift,
    shiftx
using Bramble.Forms
import Bramble.Forms: __apply_dirichlet_bc!

using HCubature
using Bramble.Tools: leastsquares

using SparseArrays
using LinearAlgebra: dot, mul!, Diagonal
using BenchmarkTools

#using LinearMaps
using LoopVectorization
using Tullio

a = 0.0
b = 1.0

I = Interval(a, b)
I2 = I #× I
#markers = add_subdomains((x, y) -> x - a)
markers = add_subdomains(x -> x-a);
Ω = Domain(I2, markers)
bc = add_dirichlet_bc(x -> 0)

N = 50_000
v = Vector{Float64}(undef, N);
#M = Mesh(Ω; nPoints = (N, N + 1), uniform = (true, true))
M = Mesh(Ω; nPoints = N, uniform = false);
S = GridSpace(M)


form = (U,V) -> innerplus(D_x(U),D_x(V));

function dot_3vecsparse1(x::V2, y::V, z::V2, buffer::V) where {V, V2}
    # check bounds on x,y,z
    @assert length(x) == length(y) == length(z)

    s = zero(eltype(x))

    for i in eachindex(x, y, z)
        s += x[i] * y[i] * z[i]
    end

    return s
end

function dot_3vecsparse2(x::V2, y::V, z::V2, buffer::V) where {V, V2} # FASTEST in ASSEMBLY
    # check bounds on x,y,z
    @assert length(x) == length(y) == length(z)

    s = zero(eltype(x))
    
    for (i,val) in enumerate(x.nzind)
       @inbounds s += x.nzval[i] * y[val] * z[val]
    end

    return s
end

function dot_3vecsparse3(x::V2, y::V, z::V2, buffer::V) where {V, V2}
    # check bounds on x,y,z
    @assert length(x) == length(y) == length(z)

    s = zero(eltype(x))

    for i in eachindex(x.nzind)
       @inbounds  s += x.nzval[i] * y[x.nzind[i]] * z[x.nzind[i]]
    end

    return s
end


function dot_3vecsparse4(x::V2, y::V, z::V2, buffer::V) where {V, V2}
    # check bounds on x,y,z
    @assert length(x) == length(y) == length(z)

    buffer .= x .* y .* z
    s = zero(eltype(x))
    @turbo for i in eachindex(y)
        s += buffer[i]
    end

    return s
end

function dot_3vecsparse5(x::V2, y::V, z::V2, buffer::V) where {V, V2}
    # check bounds on x,y,z
    @assert length(x) == length(y) == length(z)
    s = zero(eltype(x))
#=
    buffer .= 0.0
    zidxs = z.nzind
    @turbo for i in eachindex(zidxs)
        buffer[zidxs[i]] = y[zidxs[i]] * z.nzval[i]
    end

    idxs = x.nzind
    
    @turbo for i in eachindex(idxs)
        s += x.nzval[i] * buffer[idxs[i]]
    end
=#
    @turbo for i in eachindex(z.nzind)
        @inbounds w = z.nzval[i] * y[z.nzind[i]]
        for j in eachindex(x.nzind)
            @inbounds s += (z.nzind[i]==x.nzind[j]) ? x.nzval[j] * w :  zero(eltype(x))
        end
    end

    return s
end

function dot_3vecsparse6(x::V2, y::V, z::V2, buffer::V) where {V, V2}
    # check bounds on x,y,z
    @assert length(x) == length(y) == length(z)
    a = x.nzval
    b = x.nzind

    @tullio s :=  a[i] * y[b[i]] * z[b[i]]

    return s
end

function dot_3vecsparse7(x::V2, y::V, z::V2, buffer::V) where {V, V2}
    # check bounds on x,y,z
    @assert length(x) == length(y) == length(z)
    
    @tullio s :=  x[i] * z[i] * y[i] 

    return s
end


function dot_3vecsparse8(x::VSparse, y::V, z::VSparse) where {V, VSparse}
    # check bounds on x,y,z
    @assert length(x) == length(y) == length(z)

    #s = zero(eltype(x))

    #@turbo for i in eachindex(z.nzind)
        #f = Iterators.filter(j -> z.nzind[i]==x.nzind[j], eachindex(x.nzind))
        
        #for j in f
    #      @inbounds  s += x[z.nzind[i]] * z.nzval[i] * y[z.nzind[i]]
            #println(x[i], " ", z.nzval[i])
        #end
    #end
        #@views s =  sum(x[z.nzind] .* z.nzval .* y[z.nzind])
        
    return dot(x, Diagonal(y), z)
end


x1 = sparsevec([1; 4; 6; 200],[1.0; -1.0; 4.0; -5.0],N);
y1 = sparsevec([2; 7; 10; 100],[1.0; 9.0; -1.0; 3.0 ],N);
x = copy(x1);
y = copy(y1);

hspace!(v,M)
buffer = similar(v)
dot_3vecsparse1(x, v, y, buffer)

x = copy(x1);
y = copy(y1);
dot_3vecsparse2(x, v, y, buffer)

x = copy(x1);
y = copy(y1);
dot_3vecsparse3(x, v, y, buffer)

x = copy(x1);
y = copy(y1);
dot_3vecsparse4(x, v, y, buffer)

x = copy(x1);
y = copy(y1);
dot_3vecsparse5(x, v, y, buffer)

x = copy(x1);
y = copy(y1);
dot_3vecsparse6(x, v, y, buffer)

x = copy(x1);
y = copy(y1);
dot_3vecsparse7(x, v, y, buffer)

x = copy(x1);
y = copy(y1);
dot_3vecsparse8(x, v, y)

#@benchmark dot_3vecsparse1($x, $v, $y, $buffer) # 130us
@benchmark dot_3vecsparse2($x, $v, $y, $buffer) # 20ns
@benchmark dot_3vecsparse3($x, $v, $y, $buffer) # 23ns
#@benchmark dot_3vecsparse4($x, $v, $y, $buffer) # 122us
@benchmark dot_3vecsparse5($x, $v, $y, $buffer) # 18ns
#@benchmark dot_3vecsparse6($x, $v, $y, $buffer) # 67ns
#@benchmark dot_3vecsparse7($x, $v, $y, $buffer) # 122us
@benchmark dot_3vecsparse8($x, $v, $y) # 17ms









function assembly1(S::Space) where Space
    U = Elements(S);
#    V = Elements(S);
    return  innerplus(D_x(U),D_x(U))
end


function assembly21(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    N = npoints(mesh(S))
    x1 = sparsevec([1.0],[1.0],N)
    x2 = sparsevec([1.0],[1.0],N)
    ei = sparsevec([1],[1.0],N)
    ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    for idx in eachindex(Knew)
        ei.nzind[1] = I[idx]
        #changevector!(ei,I[idx])
        ej.nzind[1] = J[idx]
        #x1 .= mat*ei;
        x1 .= mat[:,I[idx]]
        x2 .= mat[:,J[idx]]
        #mul!(x2, mat, ej)
        #mul!(x1, mat, ei)
       # x2 .= mat*ej;
        Knew[idx] = dot_3vecsparse1(x1, h, x2, buffer)
    end
    return sparse(I,J,Knew)
end

function assembly22(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    #N = npoints(mesh(S))
    #x1 = sparsevec([1.0],[1.0],N)
    #x2 = sparsevec([1.0],[1.0],N)
    #ei = sparsevec([1],[1.0],N)
    #ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    for idx in eachindex(Knew)
        #ei.nzind[1] = I[idx]
        #changevector!(ei,I[idx])
        #ej.nzind[1] = J[idx]
        #x1 .= mat*ei;
       # x1 .= 
       # x2 .=
        #mul!(x2, mat, ej)
        #mul!(x1, mat, ei)
       # x2 .= mat*ej;
       @views Knew[idx] = dot_3vecsparse2(mat[:,I[idx]], h,  mat[:,J[idx]], buffer)
    end
    return sparse(I,J,Knew)
end

function assembly23(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    N = npoints(mesh(S))
    x1 = sparsevec([1.0],[1.0],N)
    x2 = sparsevec([1.0],[1.0],N)
    ei = sparsevec([1],[1.0],N)
    ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    for idx in eachindex(Knew)
        ei.nzind[1] = I[idx]
        #changevector!(ei,I[idx])
        ej.nzind[1] = J[idx]
        #x1 .= mat*ei;
        x1 .= mat[:,I[idx]]
        x2 .= mat[:,J[idx]]
        #mul!(x2, mat, ej)
        #mul!(x1, mat, ei)
       # x2 .= mat*ej;
       @views  Knew[idx] = dot_3vecsparse3(x1, h, x2, buffer)
    end
    return sparse(I,J,Knew)
end

function assembly24(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    N = npoints(mesh(S))
    x1 = sparsevec([1.0],[1.0],N)
    x2 = sparsevec([1.0],[1.0],N)
    ei = sparsevec([1],[1.0],N)
    ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    for idx in eachindex(Knew)
        ei.nzind[1] = I[idx]
        #changevector!(ei,I[idx])
        ej.nzind[1] = J[idx]
        #x1 .= mat*ei;
        x1 .= mat[:,I[idx]]
        x2 .= mat[:,J[idx]]
        #mul!(x2, mat, ej)
        #mul!(x1, mat, ei)
       # x2 .= mat*ej;
       @views Knew[idx] = dot_3vecsparse4(x1, h, x2, buffer)
    end
    return sparse(I,J,Knew)
end

function assembly25(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    N = npoints(mesh(S))
    x1 = sparsevec([1.0],[1.0],N)
    x2 = sparsevec([1.0],[1.0],N)
    ei = sparsevec([1],[1.0],N)
    ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    for idx in eachindex(Knew)
        ei.nzind[1] = I[idx]
        #changevector!(ei,I[idx])
        ej.nzind[1] = J[idx]
        #x1 .= mat*ei;
        x1 .= mat[:,I[idx]]
        x2 .= mat[:,J[idx]]
        #mul!(x2, mat, ej)
        #mul!(x1, mat, ei)
       # x2 .= mat*ej;
        Knew[idx] = dot_3vecsparse5(x1, h, x2, buffer)
    end
    return sparse(I,J,Knew)
end


function assembly26(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    N = npoints(mesh(S))
    x1 = sparsevec([1.0],[1.0],N)
    x2 = sparsevec([1.0],[1.0],N)
    ei = sparsevec([1],[1.0],N)
    ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    for idx in eachindex(Knew)
        ei.nzind[1] = I[idx]
        #changevector!(ei,I[idx])
        ej.nzind[1] = J[idx]
        #x1 .= mat*ei;
        x1 .= mat[:,I[idx]]
        x2 .= mat[:,J[idx]]
        #mul!(x2, mat, ej)
        #mul!(x1, mat, ei)
       # x2 .= mat*ej;
        Knew[idx] = dot_3vecsparse6(x1, h, x2, buffer)
    end
    return sparse(I,J,Knew)
end


function assembly27(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    for idx in eachindex(Knew)
        
       @views Knew[idx] = dot_3vecsparse7(mat[:,I[idx]], h, mat[:,J[idx]], buffer)
    end
    return sparse(I,J,Knew)
end

function assembly28(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    for idx in eachindex(Knew)
        
       @views Knew[idx] = dot_3vecsparse8(mat[:,I[idx]], h, mat[:,J[idx]])
    end
    return sparse(I,J,Knew)
end

function assembly31(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    #ei = sparsevec([1],[1.0],N)
    #ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    @tullio Knew[idx] = dot_3vecsparse1(mat[:,I[idx]], h, mat[:,J[idx]], buffer)

    
    return sparse(I,J,Knew)
end


function assembly32(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    #ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    @tullio Knew[idx] = dot_3vecsparse2(mat[:,I[idx]], h, mat[:,J[idx]], buffer)

    
    return sparse(I,J,Knew)
end

function assembly33(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    #ei = sparsevec([1],[1.0],N)
    #ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    @tullio Knew[idx] = dot_3vecsparse3(mat[:,I[idx]], h, mat[:,J[idx]], buffer)

    
    return sparse(I,J,Knew)
end

function assembly34(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    #ei = sparsevec([1],[1.0],N)
    #ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    @tullio Knew[idx] = dot_3vecsparse4(mat[:,I[idx]], h, mat[:,J[idx]], buffer)

    
    return sparse(I,J,Knew)
end

function assembly35(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    #ei = sparsevec([1],[1.0],N)
    #ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    @tullio Knew[idx] = dot_3vecsparse5(mat[:,I[idx]], h, mat[:,J[idx]], buffer)

    
    return sparse(I,J,Knew)
end

function assembly36(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    #ei = sparsevec([1],[1.0],N)
    #ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    @tullio Knew[idx] = dot_3vecsparse6(mat[:,I[idx]], h, mat[:,J[idx]], buffer)

    
    return sparse(I,J,Knew)
end

function assembly37(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    #ei = sparsevec([1],[1.0],N)
    #ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    @tullio Knew[idx] = dot_3vecsparse7(mat[:,I[idx]], h, mat[:,J[idx]], buffer)

    
    return sparse(I,J,Knew)
end


function assembly38(A::MType,S::Space, mat::MType) where {Space,MType<:AbstractSparseMatrix}
    I,J,Knew = findnz(A)
    #Knew = similar(K)
    #ei = sparsevec([1],[1.0],N)
    #ej = sparsevec([1],[1.0],N)
    h = hspace(mesh(S));
    #mat=D_x(Elements(S)).values;
    buffer = similar(h)
    @tullio Knew[idx] = dot_3vecsparse8(mat[:,I[idx]], h, mat[:,J[idx]])

    
    return sparse(I,J,Knew)
end


mat=D_x(Elements(S)).values;
A = transpose(mat)*mat
A = sparse(assembly1(S));
#=
B = assembly21(A,S,mat);
isapprox(A,B)

B = assembly22(A,S,mat);
isapprox(A,B)

B = assembly23(A,S,mat);
isapprox(A,B)

B = assembly24(A,S,mat);
isapprox(A,B)

B = assembly25(A,S,mat);
isapprox(A,B)

B = assembly26(A,S,mat);
isapprox(A,B)

B = assembly27(A,S,mat);
isapprox(A,B)

B = assembly28(A,S,mat);
isapprox(A,B)

B = assembly31(A,S,mat);
isapprox(A,B)
=#
B = assembly32(A,S,mat);
isapprox(A,B)

B = assembly33(A,S,mat);
isapprox(A,B)

#B = assembly34(A,S,mat);
#isapprox(A,B)

B = assembly35(A,S,mat);
isapprox(A,B)
#=
B = assembly36(A,S,mat);
isapprox(A,B)

B = assembly37(A,S,mat);
isapprox(A,B)

B = assembly38(A,S,mat);
isapprox(A,B)
=#
#@benchmark assembly1($S) # 890ms, 9.32 Gb
#@benchmark assembly21($A,$S,$mat) # 8s, 34Mb
#@benchmark assembly22($A,$S,$mat) # 21ms, 34Mb
#@benchmark assembly23($A,$S,$mat) # 25ms, 34Mb
#@benchmark assembly24($A,$S,$mat) # 9.9s, 34Mb
#@benchmark assembly25($A,$S,$mat) # 25ms, 34Mb
#@benchmark assembly26($A,$S,$mat) # 28ms, 37Mb
#@benchmark assembly27($A,$S,$mat) # 2.5s, 27Mb
#@benchmark assembly28($A,$S,$mat) # 7s, 31M

#@benchmark assembly31($A,$S,$mat) # 2.5s, 27Mb
@benchmark assembly32($A,$S,$mat) # 6.5ms, 27Mb
@benchmark assembly33($A,$S,$mat) # 6.3ms, 27Mb
#@benchmark assembly34($A,$S,$mat) # 
@benchmark assembly35($A,$S,$mat) # 6.2ms, 27Mb
#@benchmark assembly36($A,$S,$mat) # 7.4ms, 31Mb
#@benchmark assembly37($A,$S,$mat) # 2.3s, 33Mb
#@benchmark assembly38($A,$S,$mat) # 






#=
x=shiftx(M, Val(2), 1);
@benchmark shiftx($M, Val(2), $1)



#d1 = shiftx_alt(M, Val(1))
d2 = shiftx_alt(M, Val(2), 1);
@benchmark shiftx_alt($M, Val(2), $1)
#d3 = shiftx_alt(M, Val(3), 1)
d0 = shiftx(M, Val(2), 1);
d0 == d2
#d2 == shiftx(M, Val(2), 1)
@benchmark shiftx($M, Val(2), $1)

hspace!(v, M);
@benchmark hspace!($v, $M)



S = GridSpace(M);
#f(x) = x[1]^2 + x[2]^2;
#f(x,y) = x^2 + y^2;
f(x) = sin(x[1]);
#@benchmark Element($S, $1.0)
U = Elements(S);
V = D_x(Elements(S));
u = Element(S);
v = Element(S);
h = hspace(M);
@benchmark D_x(S)
@benchmark D_x(U)

U.values === Elements(S).values
=#

end
