
function test_operators(::Val{D}) where D
	I = interval(0, 1)
	X = domain(reduce(×, ntuple(j -> I, D)))
	M = mesh(X, ntuple(i -> 4, D), ntuple(i -> false, D))
	W = gridspace(M)
	un = element(W)
	#u0 = element(W, 2);
	Rₕ!(un, x -> exp(sum(x)) + sum(x))

	x0 = Bramble.IdentityOperator(W)

	x1 = 2 * x0
	Bramble.scalar(x1) == 2

	x2 = un * x0
	@test(all(Bramble.scalar(x2) .== un))

	x3 = 4 * x1
	@test(all(Bramble.scalar(x3) .== 8))

	x4 = 4 * x2
	@test(all(Bramble.scalar(x4) .== (4 .* un)))

	x5 = un * x2
	@test(all(Bramble.scalar(x5) .== un .* un))

	x6 = Bramble.GradientOperator(W)

	x7 = 2 * x6
	@test(Bramble.scalar(x7) == 2)

	x8 = un * x6
	@test(all(Bramble.scalar(x8) .== un))

	x9 = 4 * x7
	@test(all(Bramble.scalar(x9) .== 8))

	x10 = 4 * x8
	@test(all(Bramble.scalar(x10) .== 4 .* un))

	x11 = un * x8
	@test(all(Bramble.scalar(x11) .== un .* un))
end

for i in 1:3
	test_operators(Val(i))
end
