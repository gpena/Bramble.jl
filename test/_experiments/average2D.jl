module testcase


using Bramble.Geometry
using Bramble.Meshes
import Bramble.Meshes: hmean_extended, disp_mean_extended, hmean, hspace, xmean, _createpoints!, createpoints, npoints_tuple, points_tuple, dim
import Bramble.Meshes: hmean_extended!, hmean!, hspace!, xmean!, meas_cell, points, npoints, indices, _index2point
using Bramble.Spaces
using Bramble.Spaces: Elements, build_D_x, weights_D_x, invert_hspace, innerh_diagonal!, _inner_product, shift, shiftx, VectorElement

using SparseArrays
using LinearAlgebra: mul!
using BenchmarkTools

#using LinearMaps
using LoopVectorization
using StaticArrays
using Tullio

a = 0.0
b = 1.0

I = Interval(a, b)
I2 = I × I
markers = add_subdomains((x, y) -> x - a)
Ω = Domain(I2, markers)

N = 1_000
M = Mesh(Ω; nPoints = (N, N + 7), uniform = (false, true))
S = GridSpace(M);
u1 = Element(S);
u2 = Element(S);

f1(x) = sin(x[1])+cos(x[2]*exp(2*x[1]*x[2]))+1.0;

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
    npts = npoints_tuple(M)
    pts = pts_mean(M)
    meas = meas_cell(M)
    v = reshape(u.values, npts)

    g(i,j) = hcubature(f, 
                        (pts[1][i], pts[2][j]), 
                        (pts[1][i+1], pts[2][j+1]))[1] / meas(CartesianIndex(i,j))


    for j in 1:npts[2]
        for i in 1:npts[1]
            #println(i, " ", j, " ", g(i,j))
            v[i,j] = g(i,j)
        end
    end
end



function average3!(u::VecOrMatElem, f::F) where {F<:Function,VecOrMatElem<:VectorElement}
    M = mesh(u)
    npts = npoints_tuple(M)
    pts = pts_mean(M)
    v = reshape(u.values, npts)

    disp_pts = disp_mean_extended(M)
    measure(i,j) = disp_pts[1][i]*disp_pts[2][j]

    g(i,j) = hcubature(f, 
                        (pts[1][i], pts[2][j]), 
                        (pts[1][i+1], pts[2][j+1]))[1] / measure(i,j)


    for j in 1:npts[2]
        for i in 1:npts[1]
            #println(i, " ", j, " ", g(i,j))
            v[i,j] = g(i,j)
        end
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
    npts = npoints_tuple(M)
    pts = pts_mean(M)
    v = reshape(u.values, npts)

    disp_pts = disp_mean_extended(M)
    #measure(i,j) = 

    g = (x,y,i,j) -> hcubature(f, x,y)[1] / (disp_pts[1][i]*disp_pts[2][j])
    x = pts[1]
    y = pts[2]

    @tullio v[i,j] = @inbounds(begin
            #println(i, " ", j, " ", g(i,j))
            g((x[i], y[j]), (x[i+1], y[j+1]),i,j)
    end)
    





    #g0 = (x,y) -> hquadrature(f,x,y)[1]
    #@assert length(pts)-1 == length(h) == length(u.values)
    #@tullio v[i] := g0.(pts[i], pts[i+1])/h[i]  
    
    #u.values .= v
    
    #@tullio u.values[i] = @inbounds(begin  # sum over k
    #    g0.(pts[i], pts[i+1])/h[i]
    #end) #(i in 1:N) 
end
#=


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
=#

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
#=
u2.values .= 0.0
average6!(u2,f);
isapprox(u1.values, u2.values)
=#
@benchmark average1!($u1,$f) # 535ms, 537Mb 
@benchmark average2!($u2,$f) # 638ms, 599Mb
@benchmark average3!($u2,$f) # 638ms, 599Mb
@benchmark average4!($u2,$f) # 532ms, 537Mb
@benchmark average5!($u2,$f) # 529ms, 537Mb
#@benchmark average6!($u2,$f) # 414ms, 389Mb

end