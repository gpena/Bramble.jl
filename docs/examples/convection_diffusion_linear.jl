using Bramble
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

function solve_convection_diffusion(convdiff::LinearConvectionDiffusionProblem, nPoints::NTuple{D,Int}, unif::NTuple{D,Bool}, strategy) where D
	Mh = mesh(convdiff.dom, nPoints, unif)
	sol = @embed(Mh, convdiff.sol)
	rhs = @embed(Mh, convdiff.rhs)
	b = convdiff.b
	ϵ = convdiff.ϵ
	Wh = gridspace(Mh)
	bc = constraints(sol)

	bform = form(Wh, Wh, (U, V) -> ϵ * inner₊(∇₋ₕ(U), ∇₋ₕ(V)) + b * inner₊(M₋ₕ(U), ∇₋ₕ(V)))
	A = assemble(bform, bc)

	uh = element(Wh)
	avgₕ!(uh, rhs)

	lform = form(Wh, v -> innerₕ(uh, v), strategy = strategy, verbose = false)
	F = assemble(lform, bc)

	prob = LinearProblem(A, F)
	solh = solve(prob, KrylovJL_GMRES(), Pl = ilu(A, τ = 0.0001))

	uh .= solh.u
	F .= uh
	Rₕ!(uh, sol)
	uh .-= F

	return hₘₐₓ(Mh), norm₁ₕ(uh)
end

function convection_diffusion(d::Int)
	I = interval(0, 1)
	Ω = Bramble.domain(reduce(×, ntuple(i -> I, d)))
	b = 0.1
	ϵ = 1.0
	sol = @embed(Ω, x->exp(sum(x)))
	rhs = @embed(Ω, x->-d * sol(x) * (b + ϵ))

	return LinearConvectionDiffusionProblem(Ω, sol, rhs, ϵ, b)
end

function test_conv_diff(convec_diff_problem::LinearConvectionDiffusionProblem, nTests, npoints_generator, unif::NTuple{D,Bool}, strategy) where D
	error = zeros(nTests)
	hmax = zeros(nTests)

	for i in 1:nTests
		nPoints = ntuple(j -> npoints_generator[j](i), D)
		hmax[i], error[i] = solve_convection_diffusion(convec_diff_problem, nPoints, unif, strategy)
	end
	threshold = unif[1] ? 0.3 : 0.4
	mask = (!isnan).(error)
	err2 = error[mask]
	hmax2 = hmax[mask]

	# some least squares fitting
	order, _ = leastsquares(log.(hmax2), log.(err2))
	@test(abs(order - 2.0) < threshold||order > 2.0)
end

for strat in (DefaultAssembly(), AutoDetect())
	test_conv_diff(convection_diffusion(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1), strat)
	test_conv_diff(convection_diffusion(1), 100, (i -> 20 * i,), (false,), strat)

	test_conv_diff(convection_diffusion(2), 4, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2), strat)
	test_conv_diff(convection_diffusion(2), 7, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> false, 2), strat)

	test_conv_diff(convection_diffusion(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 3), ntuple(i -> true, 3), strat)
	#test_conv_diff(convection_diffusion(3), 6, (i->2^i+1, i->2^i+2, i->2^i+1), ntuple(i->false, 3)) # the linear solver takes a while to solve
end
