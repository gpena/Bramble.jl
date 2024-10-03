using Bramble
import LinearSolve: LinearProblem, solve, KrylovJL_GMRES, init, solve!, LUFactorization
import ILUZero: ilu0, ilu0!

# Tests for -∇ ⋅ ( A(u) ∇u ) = g, with homogeneous Dirichlet bc in [0,1]^d
# with A(u) = 3 + 1 / (1 + u^2) and g calculated accordingly such that 
# u(x) = exp(sum(x)) is the exact solution

struct PoissonNLProblem{DomainType,SolType,RhsType,CoeffType}
	dom::DomainType
	sol::SolType
	rhs::RhsType
	coeff::CoeffType
end

function poisson_nl(d::Int)
	I = interval(0, 1)
	Ω = domain(reduce(×, ntuple(i -> I, d)))

	sol = @embed(Ω, x -> exp(sum(x)))
	A = @embed(I, u -> 3 + 1 / (1 + u[1]^2))
	Ap = @embed(I, u -> -2 * u[1] / (1 + u[1]^2)^2)
	rhs = @embed(Ω, x -> -d * Ap(sol(x)) * sol(x)^2 - d * A(sol(x)) * sol(x))

	return PoissonNLProblem(Ω, sol, rhs, A)
end

function fixed_point!(mat, F, bform, bc, uold, u, A)
	prec = ilu0(mat)
	prob = LinearProblem(mat, F, KrylovJL_GMRES(), Pl = prec)
	linsolve = init(prob)

	for i in 1:2000
		uold.values .= A(u)
		assemble!(mat, bform, bc)

		uold .= u
		if i % 10 == 0
			ilu0!(prec, mat)
		end

		linsolve.A = mat
		sol = solve!(linsolve)

		u.values .= sol.u
		uold.values .-= u.values

		if norm₁ₕ(uold) < 1e-10
			break
		end
	end
end

function solve_poisson_nl(poisson_nl::PoissonNLProblem, nPoints::NTuple{D,Int}, unif::NTuple{D,Bool}) where D
	Mh = mesh(poisson_nl.dom, nPoints, unif)
	sol = @embed(Mh, poisson_nl.sol)
	rhs = @embed(Mh, poisson_nl.rhs)
	A = poisson_nl.coeff
	bc = constraints(sol)

	Wh = gridspace(Mh)
	u = element(Wh, 0)

	uold = similar(u)
	avgₕ!(uold, rhs)

	l(V) = innerₕ(uold, V)
	lform = LinearForm(l, Wh)
	F = assemble(lform, bc)

	getcoeff = @embed(Wh, u -> D == 1 ? A.(M₋ₕ(u)) : sum(ntuple(i -> A.(M₋ₕ(u)[i]), D)) ./ D)
	a(U, V) = inner₊(getcoeff(u) * ∇₋ₕ(U), ∇₋ₕ(V))
	bform = BilinearForm(a, Wh, Wh)
	mat = assemble(bform, bc)

	uold .= getcoeff(u)
	fixed_point!(mat, F, bform, bc, uold, u, getcoeff)

	Rₕ!(uold, sol)
	u .-= uold
	
	return hₘₐₓ(Mh), norm₁ₕ(u)
end

function test_poisson_nl(poisson_nl_problem::PoissonNLProblem, nTests, npoints_generator, unif::NTuple{D,Bool}) where D
	error = zeros(nTests)
	hmax = zeros(nTests)

	for i in 1:nTests
		nPoints = ntuple(j -> npoints_generator[j](i), D)
		hmax[i], error[i] = solve_poisson_nl(poisson_nl_problem, nPoints, unif)
	end

	threshold = unif[1] ? 0.3 : 0.4
	err2 = error[3:end]
	hmax2 = hmax[3:end]

	# some least squares fitting
	order, _ = leastsquares(log.(hmax2), log.(err2))
	@test(abs(order - 2.0) < threshold||order > 2.0)
end

test_poisson_nl(poisson_nl(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1))
test_poisson_nl(poisson_nl(1), 10, (i -> 2^i + 1,), ntuple(i -> false, 1))

test_poisson_nl(poisson_nl(2), 5, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2))
#test_poisson_nl(poisson_nl(2), 60, (i -> 2*i+1, i -> 3*i), (true, false))

test_poisson_nl(poisson_nl(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 1), ntuple(i -> true, 3))
#test_poisson_nl(poisson_nl(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 1), ntuple(i -> false, 3))