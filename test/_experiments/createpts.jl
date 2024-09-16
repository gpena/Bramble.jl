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
using LinearAlgebra: mul!
using BenchmarkTools

#using LinearMaps
using LoopVectorization
using Tullio

a = 0.0
b = 1.0

I = Interval(a, b)
markers = add_subdomains(x -> x-a);
Ω = Domain(I, markers)
bc = add_dirichlet_bc(x -> 0)

N = 1_000_000
v = Vector{Float64}(undef, N);
M = Mesh(Ω; nPoints = N, uniform = true);
_createpoints!(v, I, N, Val{false}());

@benchmark _createpoints!($v, $I, $N, Val{false}())
end
