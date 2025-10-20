import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@info "Start precompilation..."
include("utils/precompile.jl")
include("geometry/precompile.jl")
include("mesh/precompile.jl")
include("space/precompile.jl")

@info "Precompilation finished."

## Space compilation
#=

# add precompilation for linear forms
@compile_workload begin
	@info "Precompiling linear forms..."

	for i in 1:3
		X = domain(reduce(×, ntuple(j -> I0, i)))
		M = mesh(X, npts[i], ntuple(j -> false, i))
		sol = @embed(M, x->sum(x))
		Wh = gridspace(M)

		bc = constraints(sol)

		u = element(Wh, 0.0)
		F = 0.0 * u.values

		list_constraints = (constraints(sol),
							constraints(:dirichlet => sol),
							constraints(:dirichlet => sol, :dirichlet => sol),
							constraints(:dirichlet => sol, :dirichlet => sol, :dirichlet => sol),
							constraints(:dirichlet => sol, :dirichlet => sol, :dirichlet => sol, :dirichlet => sol))

		for strat in (DefaultAssembly(), OperatorsAssembly(), AutoDetect())
			lform = form(Wh, v -> innerₕ(u, v), strategy = strat)
			testspace(lform)

			assemble(lform)
			assemble!(F, lform)

			for bcs in list_constraints
				assemble(lform, bcs)
				assemble!(F, lform, bcs)
			end
		end

		lform_inplace = form(Wh, (res, v) -> innerₕ!(res, u, v), strategy = InPlaceAssembly())
		for bcs in list_constraints
			assemble!(F, lform_inplace, bcs)
		end
	end
end
=#
