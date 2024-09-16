module testcase


using Bramble.Geometry
using Bramble.Meshes
import Bramble.Meshes: hmean_extended, hmean, hspace, xmean, _createpoints!, createpoints, npoints_tuple, points_tuple, dim
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

N = 10_000
M = Mesh(Ω; nPoints = (N, N + 7), uniform = (false, true))
S = GridSpace(M);
u1 = Element(S);
u2 = Element(S);

f1(x) = sin(x[1])+cos(x[2]*exp(2*x[1]*x[2]))+1.0;
f2(x,y) = sin(x)+cos(y*exp(2*x*y))+1.0;

function restriction1!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    npts = npoints_tuple(mesh(u))
    v = reshape(u.values, npts)

    # coordinates of mesh points lie in the submeshes
    pts = points_tuple(mesh(u))
    idxs = Meshes.indices(mesh(u))

    broadcast!(i -> f(_index2point(pts, i)), v, idxs)
end


function restriction2!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    npts = npoints_tuple(mesh(u))
    v = reshape(u.values, npts)
    # coordinates of mesh points lie in the submeshes
    pts = points_tuple(mesh(u))

    for j in 1:npts[2]
        for i in 1:npts[1]
            x = (pts[1][i], pts[2][j])
            
            v[i,j] = f(x)
        end
    end
end

function restriction3!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    npts = npoints_tuple(mesh(u))
    v = reshape(u.values, npts)
    # coordinates of mesh points lie in the submeshes
    pts = points_tuple(mesh(u))

    for j in 1:npts[2]
        for i in 1:npts[1]
            x = (pts[1][i], pts[2][j])
            
            v[i,j] = f(x)
        end
    end
end

@inline index2point(pts, i, j) = @SVector [pts[1][i],pts[2][j]]

function restriction4!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    npts = npoints_tuple(mesh(u))
    v = reshape(u.values, npts)
    # coordinates of mesh points lie in the submeshes
    pts = points_tuple(mesh(u))

    g(i,j) = f(index2point(pts,i,j))

    @tullio v[i,j] = g(i,j)
end

function restriction5!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    npts = npoints_tuple(mesh(u))
    v = reshape(u.values, npts)
    # coordinates of mesh points lie in the submeshes
    pts = points_tuple(mesh(u))

    g(i,j) = f(@SVector [pts[1][i],pts[2][j]] )
    M = npts[2]
    N = npts[1]

    @turbo for j in Base.OneTo(M)
         for i in Base.OneTo(N)
             v[i,j] = g(i,j)
        end
    end
end


restriction1!(u1,f1);
restriction2!(u2,f1);
isapprox(u1.values, u2.values)

#u2.values .= 0.0
#restriction22!(u2,f2);
#isapprox(u1.values, u2.values)

u2.values .= 0.0
restriction3!(u2,f1);
isapprox(u1.values, u2.values)

#u2.values .= 0.0
#restriction32!(u2,f2);
#isapprox(u1.values, u2.values)

u2.values .= 0.0;
restriction4!(u2,f1);
isapprox(u1.values, u2.values)

#u2.values .= 0.0
#restriction42!(u2,f2);
#isapprox(u1.values, u2.values)

u2.values .= 0.0
restriction5!(u2,f1);
isapprox(u1.values, u2.values)


@benchmark restriction1!($u1,$f1)  #  1.179s, 416 bytes, 11
#@benchmark restriction2!($u2,$f1)  # 12s, 13Gb, ...allocs
#@benchmark restriction22!($u2,$f2) # 461 ms, 1.19 Kb, 42
#@benchmark restriction3!($u2,$f1) # muito
#@benchmark restriction32!($u2,$f2) # 458ms, 1.19Kb, 42
@benchmark restriction4!($u2,$f1) #  722ms, 512b, 15
#@benchmark restriction42!($u2,$f2) # 458ms, 1.27Kb, 43
@benchmark restriction5!($u2,$f1) # 709ms, 2.67Kb, 42
end