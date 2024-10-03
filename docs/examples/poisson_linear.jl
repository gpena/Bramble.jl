using Bramble
import LinearSolve: LinearProblem, solve, KrylovJL_GMRES
import IncompleteLU: ilu

# Tests for -Δu = g, with homogeneous Dirichlet bc in [0,1]^d
# with u(x) = exp(sum(x)) and g(x) = -d * sol(x)

struct PoissonProblem{DomainType,SolType,RhsType}
	dom::DomainType
	sol::SolType
	rhs::RhsType
end

function solve_poisson(poisson::PoissonProblem, nPoints::NTuple{D,Int}, unif::NTuple{D,Bool}) where D
	Mh = mesh(poisson.dom, nPoints, unif)
	sol = @embed(Mh, poisson.sol)
	rhs = @embed(Mh, poisson.rhs)

	Wh = gridspace(Mh)
	bc = dirichletbcs(sol)

	bform = BilinearForm((U, V) -> inner₊(∇₋ₕ(U), ∇₋ₕ(V)), Wh, Wh)
	A = assemble(bform, bc)

	uh = element(Wh)
	avgₕ!(uh, rhs)

	lform = LinearForm(U -> innerₕ(uh, U), Wh)
	F = assemble(lform, bc)

	prob = LinearProblem(A, F)
	solh = solve(prob, KrylovJL_GMRES(), Pl = ilu(A, τ = 0.0001))

	uh .= solh.u
	F .= uh
	Rₕ!(uh, sol)
	uh .-= F

	return hₘₐₓ(Mh), norm₁ₕ(uh)
end

function poisson(d::Int)
	I = interval(0, 1)
	Ω = Bramble.domain(reduce(×, ntuple(i -> I, d)))
	sol = @embed(Ω, x -> exp(sum(x)))
	rhs = @embed(Ω, x -> -d * sol(x))
	return PoissonProblem(Ω, sol, rhs)
end

function test_poisson(poisson_problem::PoissonProblem, nTests, npoints_generator, unif::NTuple{D,Bool}) where D
	error = zeros(nTests)
	hmax = zeros(nTests)

	for i in 1:nTests
		nPoints = ntuple(j -> npoints_generator[j](i), D)
		hmax[i], error[i] = solve_poisson(poisson_problem, nPoints, unif)
	end
	threshold = unif[1] ? 0.3 : 0.4
	mask = (!isnan).(error)
	err2 = error[mask]
	hmax2 = hmax[mask]

	# some least squares fitting
	order, _ = leastsquares(log.(hmax2), log.(err2))
	@test(abs(order - 2.0) < threshold||order > 2.0)
end

test_poisson(poisson(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1))
test_poisson(poisson(1), 100, (i -> 20 * i,), (false,))

test_poisson(poisson(2), 4, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2))
test_poisson(poisson(2), 7, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> false, 2))

test_poisson(poisson(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 3), ntuple(i -> true, 3))
#test_poisson(poisson(3), 6, (i->2^i+1, i->2^i+2, i->2^i+1), ntuple(i->false, 3)) # the linear solver takes a while to solve
