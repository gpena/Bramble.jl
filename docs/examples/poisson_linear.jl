using Bramble
import Bramble: domain
import Bramble: embed_function
import LinearSolve: LinearProblem, solve, KrylovJL_GMRES
import IncompleteLU: ilu

# Tests for -Δu = g, with homogeneous Dirichlet bc in [0,1]^d
# with u(x) = exp(sum(x)) and g(x) = -d * sol(x)

struct SimpleScalarPDEProblem{DomainType,SolType,RhsType}
	dom::DomainType
	sol::SolType
	rhs::RhsType
end

domain(p::SimpleScalarPDEProblem) = p.dom
solution(p::SimpleScalarPDEProblem) = p.sol
right_hand_side(p::SimpleScalarPDEProblem) = p.rhs

function solve_poisson(poisson::SimpleScalarPDEProblem, nPoints::NTuple, unif::NTuple)
	Ω = domain(poisson)
	Ωₕ = mesh(Ω, nPoints, unif)

	sol = solution(poisson)
	rhs = right_hand_side(poisson)

	bcs = dirichlet_constraints(set(Ω), :boundary => x -> sol(x))

	Wₕ = gridspace(Ωₕ)

	aₕ = form(Wₕ, Wₕ, (Uₕ, Vₕ) -> inner₊(∇₋ₕ(Uₕ), ∇₋ₕ(Vₕ)))
	A = assemble(aₕ, dirichlet_labels = :boundary)

	uₕ = element(Wₕ)
	avgₕ!(uₕ, rhs)

	lₕ = form(Wₕ, vₕ -> innerₕ(uₕ, vₕ))
	F = assemble(lₕ, bcs, dirichlet_labels = :boundary)

	prec = ilu(A, τ = 0.001)
	prob = LinearProblem(A, F)
	solₕ = solve(prob, KrylovJL_GMRES(), Pl = prec, verbose = false)

	uₕ .= solₕ.u
	F .= uₕ
	Rₕ!(uₕ, sol)
	uₕ .-= F

	return hₘₐₓ(Ωₕ), norm₁ₕ(uₕ)
end

function poisson(d::Int)
	I = interval(0, 1)
	X = reduce(×, ntuple(i -> I, d))
	Ω = domain(X)
	sol = x -> exp(sum(x))
	rhs = x -> -d * sol(x)
	return SimpleScalarPDEProblem(Ω, sol, rhs)
end

function test_poisson(poisson_problem::SimpleScalarPDEProblem, nTests, npoints_generator, unif::NTuple{D,Bool}) where D
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
	order, _ = least_squares_fit(hmax2, err2)
	@test(abs(order - 2.0) < threshold||order > 2.0)
end

