module Advection2D

using Test
using Bramble.Geometry
using Bramble.Meshes
using Bramble.Meshes: hspace, points, submesh, npoints_tuple, npoints
using Bramble.Spaces
using Bramble.Forms

using Bramble.Tools: leastsquares
using Makie, GLMakie, WriteVTK

using LinearAlgebra: \

markers = add_subdomains(x -> x[1] - 0.0, marker = :left)
add_subdomains!(markers, x -> x[1] - 1.0, marker = :right)
add_subdomains!(markers, x -> x[2] - 0.0, marker = :bottom)
add_subdomains!(markers, x -> x[2] - 1.0, marker = :top)
X = Domain(Interval(0.0, 1.0) × Interval(0.0, 1.0), markers)

bc = add_dirichlet_bc(x->0.0, marker = :right)
add_dirichlet_bc!(bc, x->1.0, marker = :left)

heaviside(x) = (x < 0.5 ? 1.0 : 0.0)
add_dirichlet_bc!(bc, x->heaviside(x[1]), marker = :top)
add_dirichlet_bc!(bc, x->heaviside(x[1]), marker = :bottom)

const ϵ = 1e-3
const b = 1.0


#for i = 1:nTests
N = 10

M = Meshes.mesh(X; nPoints = (N,N), uniform = (true,true))

Wh = GridSpace(M)
u = Element(Wh)

gh = Element(Wh, 0.0)

a(U, V) = ϵ*inner₊(∇₋ₕ(U), ∇₋ₕ(V)) - inner₊ᵧ(M₋ₕᵧ(U), D₋ᵧ(V))
bform = BilinearForm(a, Wh, Wh);
mat = assemble(bform, bc)
l(V) = innerₕ(gh, V)

lform = LinearForm(l, Wh);
F = assemble(lform, bc)

solve!(u, mat, F)

xs = Matrix(reshape(repeat(points(submesh(M,1)), npoints(submesh(M,2))), npoints_tuple(M)))
ys = transpose(Matrix(reshape(repeat(points(submesh(M,2)), npoints(submesh(M,1))), npoints_tuple(M))))
v = Matrix(reshape(u.values, npoints_tuple(M)))




Ni, Nj, Nk = 6, 8, 11
x = [i / Ni * cospi(3/2 * (j - 1) / (Nj - 1)) for i = 1:Ni, j = 1:Nj]
y = [i / Ni * sinpi(3/2 * (j - 1) / (Nj - 1)) for i = 1:Ni, j = 1:Nj]
z = x.^2 + y.^2

vtk_grid("teste", xs, ys) do vtk
    vtk["pressure"] = v
end

end
