using Bramble
import LinearSolve: LinearProblem, solve, KrylovJL_GMRES
import IncompleteLU: ilu

# Tests for -Δu = g, with homogeneous Dirichlet bc in [0,1]^d
# with u(x) = exp(sum(x)) and g(x) = -d * sol(x)

struct SimpleScalarPDEProblem{DomainType,SolType,RhsType}
	dom::DomainType
	sol::SolType
	rhs::RhsType
end

function solve_poisson(poisson::SimpleScalarPDEProblem, nPoints::NTuple{D,Int}, unif::NTuple{D,Bool}, strategy) where D
	Mh = mesh(poisson.dom, nPoints, unif)
	sol = @embed(Mh, poisson.sol)
	rhs = @embed(Mh, poisson.rhs)

	Wh = gridspace(Mh)
	bc = constraints(sol)

	bform = form(Wh, Wh, (U, V) -> inner₊(∇₋ₕ(U), ∇₋ₕ(V)))
	A = assemble(bform, bc)

	uh = element(Wh)
	avgₕ!(uh, rhs)

	lform = form(Wh, v -> innerₕ(uh, v), strategy = strategy, verbose = true)
	F = assemble(lform, bc)

	prob = LinearProblem(A, F)
	solh = solve(prob, KrylovJL_GMRES(), Pl = ilu(A, τ = 0.001))

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
	return SimpleScalarPDEProblem(Ω, sol, rhs)
end

function test_poisson(poisson_problem::SimpleScalarPDEProblem, nTests, npoints_generator, unif::NTuple{D,Bool}, strategy) where D
	error = zeros(nTests)
	hmax = zeros(nTests)

	for i in 1:nTests
		nPoints = ntuple(j -> npoints_generator[j](i), D)
		hmax[i], error[i] = solve_poisson(poisson_problem, nPoints, unif, strategy)
	end
	threshold = unif[1] ? 0.3 : 0.4
	mask = (!isnan).(error)
	err2 = error[mask]
	hmax2 = hmax[mask]

	# some least squares fitting
	order, _ = leastsquares(log.(hmax2), log.(err2))
	@test(abs(order - 2.0) < threshold||order > 2.0)
end

