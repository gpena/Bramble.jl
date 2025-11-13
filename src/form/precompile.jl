
@setup_workload begin
	function poisson(d::Int)
		I = interval(0, 1)
		cart = reduce(×, ntuple(i -> I, d))
		Ω = domain(cart)
		sol(x) = exp(sum(x))

		nPoints = ntuple(i -> 4, d)
		unif = ntuple(i -> true, d)
		Ωₕ = mesh(Ω, nPoints, unif)
		Wₕ = gridspace(Ωₕ)

		return Ω, sol, Wₕ
	end

	@compile_workload begin
		# ESSENTIAL: 1D only
		max_dim = BRAMBLE_EXTENDED_PRECOMPILE ? 3 : 1
		for dim in 1:max_dim
			Ω, sol, Wₕ = poisson(dim)
			Ωₕ = mesh(Wₕ)

			bform = form(Wₕ, Wₕ, (U, V) -> inner₊(∇₋ₕ(U), ∇₋ₕ(V)))
			trial_space(bform)
			test_space(bform)

			uₕ = element(Wₕ)
			Uₕ = Bramble.elements(Wₕ)
			bform(uₕ, uₕ)
			bform(Uₕ, uₕ)
			bform(uₕ, Uₕ)
			bform(Uₕ, Uₕ)

			dirichlet_conditions = dirichlet_constraints(set(Ω), :boundary => x -> sol(x))
			dirichlet_conditions_time = dirichlet_constraints(set(Ω), interval(0, 1), :boundary => x -> sol(x))

			A = assemble(bform)
			assemble(bform, dirichlet_labels = :boundary)
			assemble!(A, bform)
			assemble!(A, bform, dirichlet_labels = :boundary)

			uₕ = element(Wₕ)

			lform = form(Wₕ, v -> innerₕ(uₕ, v))
			test_space(lform)

			F = assemble(lform)
			assemble(lform, dirichlet_conditions, dirichlet_labels = :boundary)
			assemble!(F, lform)
			assemble!(F, lform, dirichlet_conditions = dirichlet_conditions)

			dirichlet_bc_symmetrize!(A, F, Ωₕ, :boundary)
			dirichlet_bc_symmetrize!(A, F, Ωₕ, :boundary, dropzeros = true)
		end
	end
end
