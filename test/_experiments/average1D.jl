module testcase


using Bramble.Geometry
using Bramble.Meshes
import Bramble.Meshes: hmean_extended, hmean, hspace, xmean, _createpoints!, createpoints, npoints_tuple, points_tuple, dim
import Bramble.Meshes: hmean_extended!, hmean!, hspace!, xmean!, meas_cell, points, npoints, indices, _index2point, pts_mean
using Bramble.Spaces
using Bramble.Spaces: Elements, build_D_x, weights_D_x, invert_hspace, innerh_diagonal!, _inner_product, shift, shiftx, VectorElement

using SparseArrays
using LinearAlgebra: mul!
using BenchmarkTools

using HCubature
using LoopVectorization
using StaticArrays
using Tullio

a = 0.0
b = 1.0

I = Interval(a, b)
I2 = I #× I
#markers = add_subdomains((x, y) -> x - a)
markers = add_subdomains(x -> x-a);
Ω = Domain(I2, markers)

N = 1_000_000
#M = Mesh(Ω; nPoints = (N, N + 1), uniform = (true, true))
M = Mesh(Ω; nPoints = N, uniform = false);
S = GridSpace(M);
u1 = Element(S);
u2 = Element(S);

f(x) = sin(x[1])+cos(x[1]*exp(2*x[1]));

function average1!(u::VecOrMatElem, f::F) where {F<:Function,VecOrMatElem<:VectorElement}
    M = mesh(u)

    pts_avg = pts_mean(M)
    meas = meas_cell(M)

    g(idx::CartIndex) where {CartIndex<:CartesianIndex} =
        hcubature(f, _index2point(pts_avg, idx), _index2point(pts_avg, idx, 1))[1] /
        meas(idx)

    npts = npoints_tuple(M)
    v = reshape(u.values, npts)
    idxs = indices(M)

    map!(g, v, idxs)
end


function average2!(u::VecOrMatElem, f::F) where {F<:Function,VecOrMatElem<:VectorElement}
    M = mesh(u)

    pts = xmean(M)
    h = hmean_extended(M)

    g(i) = hquadrature(f, pts[i], pts[i+1])[1] / h[i]

    npts = npoints(M)

    for i in 1:npts
        u.values[i] = g(i)
    end
end

function average3!(u::VecOrMatElem, f::F) where {F<:Function,VecOrMatElem<:VectorElement}
    M = mesh(u)

    pts = xmean(M)
    h = hmean_extended(M)
    #x= [0.0;0.0]

    function g(i) 
        x = hquadrature(f, pts[i], pts[i+1])
        #println(typeof(x))
        return x[1] / h[i]
    end
    npts = npoints(M)

    for i in 1:npts
        u.values[i] = g(i)
    end
end


function average4!(u::VecOrMatElem, f::F) where {F<:Function,VecOrMatElem<:VectorElement}
    M = mesh(u)

    pts_avg = pts_mean(M)
    meas = meas_cell(M)

    g(idx::CartIndex) where {CartIndex<:CartesianIndex} =
        hcubature(f, _index2point(pts_avg, idx), _index2point(pts_avg, idx, 1))[1] /
        meas(idx)

    npts = npoints_tuple(M)
    v = reshape(u.values, npts)
    idxs = indices(M)

    vmap!(g, v, idxs)
end


function average5!(u::VecOrMatElem, f::F) where {F<:Function,VecOrMatElem<:VectorElement}
    M = mesh(u)

    pts = xmean(M)
    h = hmean_extended(M)
    g0 = (x,y) -> hquadrature(f,x,y)[1]
    @assert length(pts)-1 == length(h) == length(u.values)
    #@tullio v[i] := g0.(pts[i], pts[i+1])/h[i]  
    
    #u.values .= v
    
    @tullio u.values[i] = @inbounds(begin  # sum over k
        g0.(pts[i], pts[i+1])/h[i]
    end) #(i in 1:N) 
end



function average6!(u::VecOrMatElem, f::F) where {F<:Function,VecOrMatElem<:VectorElement}
    M = mesh(u)
    
    pts = xmean(M)
    h = hmean_extended(M)
    g0 = (x,y) -> HCubature.QuadGK.quadgk(f,x,y)[1]
    @assert length(pts)-1 == length(h) == length(u.values)
    #@tullio v[i] := g0.(pts[i], pts[i+1])/h[i]  
    
    #u.values .= v
    
    @tullio u.values[i] = @inbounds(begin  # sum over k
        g0.(pts[i], pts[i+1])/h[i]
    end) #(i in 1:N) 
end


average1!(u1,f);
average2!(u2,f);
isapprox(u1.values, u2.values)

u2.values .= 0.0
average3!(u2,f);
isapprox(u1.values, u2.values)

u2.values .= 0.0
average4!(u2,f);
isapprox(u1.values, u2.values)

u2.values .= 0.0
average5!(u2,f);
isapprox(u1.values, u2.values)

u2.values .= 0.0
average6!(u2,f);
isapprox(u1.values, u2.values)

@benchmark average1!($u1,$f) # 417ms, 389Mb
@benchmark average2!($u2,$f) # 493ms, 434Mb
@benchmark average3!($u2,$f) # 485ms, 434Mb
@benchmark average4!($u2,$f) # 414ms, 389Mb
@benchmark average5!($u2,$f) # 414ms, 389Mb
@benchmark average6!($u2,$f) # 414ms, 389Mb

end