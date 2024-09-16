module testcase

using Bramble.Tools: dot_3vec

using Bramble.Geometry
using Bramble.Meshes
import Bramble.Meshes: hmean_extended, hmean, hspace, xmean, _createpoints!, createpoints, npoints_tuple, points_tuple, dim
import Bramble.Meshes: hmean_extended!, hmean!, hspace!, xmean!, meas_cell, points, npoints, indices, _index2point, pts_mean
using Bramble.Spaces
using Bramble.Spaces: Elements, build_D_x, weights_D_x, invert_hspace, innerh_diagonal!, _inner_product, shift, shiftx, VectorElement, invert_hspace

using Bramble.Forms
using SparseArrays
using LinearAlgebra: mul!
using BenchmarkTools

using LoopVectorization
using Tullio


a = 0.0
b = 1.0

I = Interval(a, b)
I2 = I #× I
#markers = add_subdomains((x, y) -> x - a)
markers = add_subdomains(x -> x-a);
Ω = Domain(I2, markers)

N = 5
#M = Mesh(Ω; nPoints = (N, N + 1), uniform = (true, true))
M = Mesh(Ω; nPoints = N, uniform = false);
S = GridSpace(M);

u=Element(S,1.0);
f(x) = sin(x[1])
Rh!(u,f)
form(U,V) = stiffness(U,V; scaling = u);
bform = BilinearForm(form, S, S);
mat = sparse(assemble(bform));
mat

I,J,_ = findnz(mat);

function componentwiseassembly(N::Int, S::MType, I::VType, J::VType) where {MType, VType}
    U = Elements(S)
    M = mesh(S)
    h = hspace(M);
    D = D_x(U).values
   # ei = sparsevec([1],[1.0],N);
   # ej = sparsevec([1],[1.0],N);
    #f(du::VType,ux::VType,p,t) where VType <:AbstractVector = leftdiff!(du, ux)
    #D = MatrixOperator(D)
    
    K = Vector{Float64}(undef, length(I));
    h = hspace(M)

    #=
    x = sparsevec([1], [1.0], N)

    @tullio K[i] = @inbounds(begin 
        #x .= D[j,:]
        D[j,J[i]] * h[j] * D[j,I[i]]
    end)
=#
    
    for i in eachindex(K)
        K[i] = dot_3vec(D[:,J[i]], h, D[:,I[i]])
    end
   
    
    return sparse(I,J,K)
    
end

S = GridSpace(M);
U=Elements(S);
D_x(U);
Z = componentwiseassembly(N, S, I, J);
@benchmark componentwiseassembly($N, $S, $I, $J)
#innerplus(left, right) where {E1<:ElementTypes{1},E2<:ElementTypes{1}} =
#    _inner_product(u, hspace(mesh(u)), v)
isapprox(mat, Z)

end