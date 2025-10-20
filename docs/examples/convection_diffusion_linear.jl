using Bramble
import Bramble: domain
import LinearSolve: LinearProblem, solve, KrylovJL_GMRES
import IncompleteLU: ilu

# Tests for -∇ ⋅ (ϵ∇u + b [1 … 1]^t u) = g, with homogeneous Dirichlet bc in [0,1]^d
# with u(x) = exp(sum(x)) and g(x) = -d * sol(x) (ϵ + b)

struct LinearConvectionDiffusionProblem{DomainType,SolType,RhsType,T}
	dom::DomainType
	sol::SolType
	rhs::RhsType
	ϵ::T
	b::T
end

domain(p::LinearConvectionDiffusionProblem) = p.dom
solution(p::LinearConvectionDiffusionProblem) = p.sol
right_hand_side(p::LinearConvectionDiffusionProblem) = p.rhs
diffusion_coefficient(p::LinearConvectionDiffusionProblem) = p.ϵ
convection_coefficient(p::LinearConvectionDiffusionProblem) = p.b

function solve_convection_diffusion(problem::LinearConvectionDiffusionProblem, nPoints::NTuple{D,Int}, unif::NTuple{D,Bool}) where D
	Ω = domain(problem)
	Ωₕ = mesh(Ω, nPoints, unif)

	sol = solution(problem)
	rhs = right_hand_side(problem)
	b = convection_coefficient(problem)
	ϵ = diffusion_coefficient(problem)

	Wₕ = gridspace(Ωₕ)
	bcs = dirichlet_constraints(set(Ω), :boundary => x -> sol(x))

	aₕ = form(Wₕ, Wₕ, (U, V) -> ϵ * inner₊(∇₋ₕ(U), ∇₋ₕ(V)) + b * inner₊(M₋ₕ(U), ∇₋ₕ(V)))
	A = assemble(aₕ, dirichlet_labels = :boundary)

	uₕ = element(Wₕ)
	avgₕ!(uₕ, rhs)

	lₕ = form(Wₕ, v -> innerₕ(uₕ, v))
	F = assemble(lₕ, bcs, dirichlet_labels = :boundary)

	prob = LinearProblem(A, F)
	solh = solve(prob, KrylovJL_GMRES(), Pl = ilu(A, τ = 0.0001))

	uₕ .= solh.u
	F .= uₕ
	Rₕ!(uₕ, sol)
	uₕ .-= F

	return hₘₐₓ(Ωₕ), norm₁ₕ(uₕ)
end

function convection_diffusion(d::Int)
	I = interval(0, 1)
	Ω = domain(reduce(×, ntuple(i -> I, d)))
	b = 0.1
	ϵ = 1.0
	sol = x -> exp(sum(x))
	rhs = x -> -d * sol(x) * (b + ϵ)

	return LinearConvectionDiffusionProblem(Ω, sol, rhs, ϵ, b)
end

function test_conv_diff(problem::LinearConvectionDiffusionProblem, nTests, npoints_generator, unif::NTuple{D,Bool}) where D
	error = zeros(nTests)
	hmax = zeros(nTests)

	for i in 1:nTests
		nPoints = ntuple(j -> npoints_generator[j](i), D)
		hmax[i], error[i] = solve_convection_diffusion(problem, nPoints, unif)
	end

	threshold = unif[1] ? 0.3 : 0.4
	mask = (!isnan).(error)
	err2 = error[mask]
	hmax2 = hmax[mask]

	# some least squares fitting
	order, _ = least_squares_fit(hmax2, err2)
	@test(abs(order - 2.0) < threshold || order > 2.0)
end
