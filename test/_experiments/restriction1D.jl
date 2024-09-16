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

function restriction1!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    npts = npoints_tuple(mesh(u))
    v = reshape(u.values, npts)

    # coordinates of mesh points lie in the submeshes
    pts = points_tuple(mesh(u))
    idxs = Meshes.indices(mesh(u))

    broadcast!(i -> f(_index2point(pts, i)), v, idxs)
end


function restriction2!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    #npts = npoints(mesh(u))
    #v = reshape(u.values, npts)

    # coordinates of mesh points lie in the submeshes
    pts = points(mesh(u))
    idxs = Meshes.indices(mesh(u))

    broadcast!(i -> f(_index2point(pts, i)), u.values, idxs)
end


function restriction3!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    npts = npoints(mesh(u))
    #v = reshape(u.values, npts)

    # coordinates of mesh points lie in the submeshes
    pts = points(mesh(u))
    #idxs = Meshes.indices(mesh(u))
    
    for i in 1:npts
        u.values[i] = f(pts[i])
    end
end

function restriction4!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    pts = points(mesh(u))
    
    @. u.values .= f.(pts)
end

function restriction5!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    #npts = npoints(mesh(u))
    #v = reshape(u.values, npts)

    # coordinates of mesh points lie in the submeshes
    pts = points(mesh(u))
    N = npoints(mesh(u))

    @turbo for i in Base.OneTo(N)
        u.values[i] = f(pts[i])
    end
end

function restriction6!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    #npts = npoints_tuple(mesh(u))
    #v = reshape(u.values, npts)
    # coordinates of mesh points lie in the submeshes
    ptsX = points(submesh(mesh(u),1))

    N = npoints(submesh(mesh(u),1))

    @turbo for i in 1:N
        u.values[i] = f(ptsX[i])
    end
end

function restriction7!(u::Elem, f::F) where {Elem <: VectorElement, F<:Function}
    #npts = npoints_tuple(mesh(u))
    #v = reshape(u.values, npts)
    # coordinates of mesh points lie in the submeshes
    pts = points(mesh(u))

    @tullio u.values[i] = f(pts[i])
end

restriction1!(u1,f);
restriction2!(u2,f);
isapprox(u1.values, u2.values)

u2.values .= 0.0
restriction3!(u2,f);
isapprox(u1.values, u2.values)

u2.values .= 0.0
restriction4!(u2,f);
isapprox(u1.values, u2.values)

u2.values .= 0.0
restriction5!(u2,f);
isapprox(u1.values, u2.values)

u2.values .= 0.0
restriction6!(u2,f);
isapprox(u1.values, u2.values)

u2.values .= 0.0
restriction7!(u2,f);
isapprox(u1.values, u2.values)

@benchmark restriction1!($u1,$f) # 11ms, 48bytes, 3
@benchmark restriction2!($u2,$f) # 11ms, 16 bytes, 1
@benchmark restriction3!($u2,$f) # 58ms, 61 Mb, 212313213
@benchmark restriction4!($u2,$f) # 11 ms, 32 bytes, 2
@benchmark restriction5!($u2,$f) # 4.6ms, 448 bytes, 19
@benchmark restriction6!($u2,$f) # 4.6ms, 448 bytes, 19
@benchmark restriction7!($u2,$f) # 5ms, 64 bytes, 4
end