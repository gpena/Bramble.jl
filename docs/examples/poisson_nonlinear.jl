using Bramble
import Bramble: domain
import LinearSolve: LinearProblem, solve, KrylovJL_GMRES, init, solve!
import ILUZero: ilu0, ilu0!

# Tests for - ∇ ⋅ ( α(u) ∇u ) = g, with homogeneous Dirichlet bc in [0,1]^d
# with α(u) = 3 + 1 / (1 + u^2) and g calculated accordingly such that 
# u(x) = exp(sum(x)) is the exact solution

struct PoissonNLProblem{DomainType,SolType,RhsType,CoeffType}
	dom::DomainType
	sol::SolType
	rhs::RhsType
	α::CoeffType
end

@inline domain(prob::PoissonNLProblem) = prob.dom
@inline solution(prob::PoissonNLProblem) = prob.sol
@inline right_hand_side(prob::PoissonNLProblem) = prob.rhs
@inline coefficient(prob::PoissonNLProblem) = prob.α

function poisson_nl(d::Int)
	I = interval(0, 1)
	Ω = domain(reduce(×, ntuple(i -> I, d)))

	sol = x -> exp(sum(x))
	α = u -> 3 + 1 / (1 + u[1]^2)
	dαdu = u -> -2 * u[1] / (1 + u[1]^2)^2
	rhs = x -> -d * dαdu(sol(x)) * sol(x)^2 - d * α(sol(x)) * sol(x)

	return PoissonNLProblem(Ω, sol, rhs, α)
end

function fixed_point!(matrix, rhs, aₕ, uₚᵣₑᵥ, uₙₑₓₜ, coeff)
	prec = ilu0(matrix)
	prob = LinearProblem(matrix, rhs, KrylovJL_GMRES(), Pl = prec)
	linsolve = init(prob)

	for i in 1:2000
		uₚᵣₑᵥ .= coeff(uₙₑₓₜ)
		assemble!(matrix, aₕ, dirichlet_labels = :boundary)

		uₚᵣₑᵥ .= uₙₑₓₜ
		if i % 10 == 0
			ilu0!(prec, matrix)
		end

		linsolve.A = matrix
		sol = solve!(linsolve)

		uₙₑₓₜ .= sol
		uₚᵣₑᵥ .-= uₙₑₓₜ

		if norm₁ₕ(uₚᵣₑᵥ) < 1e-3
			break
		end
	end
end

function solve_poisson_nl(poisson_nl::PoissonNLProblem, nPoints::NTuple{D,Int}, unif::NTuple{D,Bool}) where D
	Ω = domain(poisson_nl)
	X = set(Ω)
	Ωₕ = mesh(Ω, nPoints, unif)
	sol = solution(poisson_nl)
	rhs = right_hand_side(poisson_nl)
	α = coefficient(poisson_nl)

	bcs = dirichlet_constraints(X, :boundary => x -> sol(x))

	Wₕ = gridspace(Ωₕ)
	uₙ = element(Wₕ, 0)

	u₀ = similar(uₙ)
	avgₕ!(u₀, rhs)

	lₕ = form(Wₕ, vₕ -> innerₕ(u₀, vₕ))
	F = assemble(lₕ, bcs, dirichlet_labels = :boundary)

	αₕ = u -> D == 1 ? α.(M₋ₕ(u)) : sum(ntuple(i -> α.(M₋ₕ(u)[i]), D)) ./ D
	aₕ = form(Wₕ, Wₕ, (U, V) -> inner₊(αₕ(uₙ) * ∇₋ₕ(U), ∇₋ₕ(V)))
	A = assemble(aₕ, dirichlet_labels = :boundary)

	Rₕ!(uₙ, sol)
	fixed_point!(A, F, aₕ, u₀, uₙ, αₕ)

	Rₕ!(u₀, sol)
	uₙ .-= u₀

	return hₘₐₓ(Ωₕ), norm₁ₕ(uₙ)
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
	order, _ = least_squares_fit(hmax2, err2)
	@test(abs(order - 2.0) < threshold || order > 2.0)
end