module testcase

using Bramble.Geometry
using Bramble.Meshes
import Bramble.Meshes: hmean, hspace, xmean, _createpoints!, createpoints, npoints_tuple, points_tuple, dim
import Bramble.Meshes: hmean!, hspace!, xmean!, meas_cell, points, npoints, indices, _index2point, __index2point, meas_cell, meas_cell2
using Bramble.Spaces
using Bramble.Spaces: Elements, invert_hspace, innerh_diagonal!, shift, VectorElement, invert_hspace, shift
using Bramble.Forms
using SparseArrays
using StaticArrays
#using LinearAlgebra: mul!
using BenchmarkTools

a = 0.0
b = 1.0

I = Interval(a, b);
I2 = I;
markers = add_subdomains(x -> x[1]-a);
Ω = Domain(I2 × I, markers)

N = 100
M = Mesh(Ω; nPoints = (N,N), uniform = (false,false));
S = GridSpace(M);

pts = points(M)
c = CartesianIndex{2}((1,2))



h=hspace(M)

u = Element(S);

forma_bilinear(U,V) = innerₕ(Mₕ(U),D₋ₓ(V))
forma_linear(U) = innerₕ(Mₕ(u),D₋ₓ(U))
U = Elements(S);
#F=assemble(forma_linear);
#@benchmark forma_linear($u)








#=
V1 = Element(S);
V1.values .= 0.0
V1.values[idx1] .= val1
V1.values
z1 = jump(V1).values
@benchmark jump($V1)
=#
u .= 1.0


forma_bilinear(U,V) = inner₊(u*D₋ₓ(U),D₋ₓ(V));
#forma_bilinear2(U,V) = stiffness(U,V)

bform = BilinearForm(forma_bilinear, S, S);

mat0 = assemble(bform);
@benchmark assemble($bform)


end