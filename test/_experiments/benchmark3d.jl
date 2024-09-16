using Test
using Bramble
using Bramble: apply_dirichlet_bc!, cart2linear, sub2ind
using BenchmarkTools

using LinearAlgebra: I, norm

I1 = Interval(0.0, 1.0)
markers = add_subdomains(x -> maximum(abs.(x.-0.5)) - 0.5)
Ω = Domain(I1×I1×I1, markers)

sol(x) = exp(x[1]+x[2]+x[3])
g(x) = -3.0*sol(x)

unif = false;
N = 130;

M = Mesh(Ω; nPoints = (N,N+1,N+1), uniform = (unif,unif,unif))
Wh = GridSpace(M);

u = Element(Wh)
V = Bramble.Elements(Wh)
#hₘₐₓ(M)
#@benchmark hₘₐₓ($M)


#c = CartesianIndex{3}(1,2,3)
#Bramble.meas_cell(M, c)
#@benchmark Bramble.meas_cell($M, $c)
v = Vector{Float64}(undef, Bramble.npoints(M))
Bramble.weights_D₋ᵧ!(v, M, Val(3))

@benchmark Bramble.weights_D₋ᵧ!($v, $M, $Val(3))





bc = add_dirichlet_bc(sol);
a = BilinearForm( (u,v) -> inner₊(∇ₕ(u),∇ₕ(v)), Wh, Wh)
p = BilinearForm( (u,v) -> innerₕ(u,v), Wh, Wh)
Ap = assemble(p, bc);



#innerplus_weights(M, Val(2));
#@benchmark innerplus_weights($M, Val(2))


#a = BilinearForm( (u,v) -> innerₕ(u,v), Wh, Wh);
A = assemble(a);
A = assemble(a, bc);


#diff₂(Wh);
#@benchmark diff₂($Wh)

rhs = avgₕ(Wh, g);
l = LinearForm(v -> innerₕ(rhs, v), Wh);
F = assemble(l, bc);

u = Element(Wh);
solve_gmres(A, F);
@benchmark solve_gmres($A, $F)
