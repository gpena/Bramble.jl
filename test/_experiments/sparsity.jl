module testcase

using Bramble.Tools: dot_3vec

using Bramble.Geometry
using Bramble.Meshes
import Bramble.Meshes: hmean_extended, hmean, hspace, xmean, _createpoints!, createpoints, npoints_tuple, points_tuple, dim
import Bramble.Meshes: hmean_extended!, hmean!, hspace!, xmean!, meas_cell, points, npoints, indices, _index2point, pts_mean
using Bramble.Spaces
using Bramble.Spaces: Elements, build_D_x, weights_D_x, invert_hspace, innerh_diagonal!, _inner_product, shift, shiftx, VectorElement, invert_hspace
using Bramble.Spaces: ElementSparse
using Bramble.Forms
using SparseArrays

using BenchmarkTools

using LoopVectorization
using Tullio
#using Random


a = 0.0
b = 1.0

I = Interval(a, b);
I2 = I;
markers = add_subdomains(x -> x-a);
Ω = Domain(I2, markers)

N = 5
M = Mesh(Ω; nPoints = N, uniform = true);
S = GridSpace(M);


forma_bilinear(U,V) = innerplus(Mh(U),D_x(V))#+innerplus(Mh(U),D_x(V)) +innerplus(D_x(U),Mh(V))# + innerplus(jump(D_x(U)),jump(D_x(V)));
bform = BilinearForm(forma_bilinear, S, S);


I0 = bform.pattern.I;
J0 = bform.pattern.J;
#sparse(I0,J0,1.0*I0)
#sparse(I0,J0,1.0*J0)
#@benchmark SparsityPattern($bform)


mat = assemble(bform);
#mat2 = assemble(bform);
isapprox(mat, mat2)
@benchmark assemble($bform)
@benchmark assemble($bform, $sparsity)

end