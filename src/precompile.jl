import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@setup_workload begin
	pts = [-1, -1.0, 0, 0.0, 1 // 2, π]
	combinations = ((pts[i], pts[j]) for i in eachindex(pts) for j in eachindex(pts) if pts[i] <= pts[j])

	npts = ((2,), (2, 2), (2, 2, 2))

	I0 = interval(-3.0, 10.0)
	Ω0 = cartesianproduct(I0)

	Ω1 = cartesianproduct(I0) × cartesianproduct(I0)
	Ω2 = Ω1 × cartesianproduct(I0)
	Ω3 = cartesianproduct(I0) × Ω1
	omegas = (Ω0, Ω1, Ω2, Ω3)

	@compile_workload begin
		for p in combinations
			interval(p...)
		end

		interval(-3.0, 10.0)
		_Ω0 = cartesianproduct(I0)
		cartesianproduct(Ω0)
		_Ω0(1)

		_Ω1 = cartesianproduct(I0) × cartesianproduct(I0)
		_Ω2 = Ω1 × cartesianproduct(I0)
		_Ω3 = cartesianproduct(I0) × Ω1

		for o in omegas
			eltype(o), dim(o)
			eltype(typeof(o)), dim(typeof(o))
		end

		for i in 1:3
			Ii = ntuple(j -> I0, i)
			Ω = reduce(×, Ii)

			f1 = @embed(Ω, x->x[i])

			projection(Ω, 1)
			tails(Ω)
			tails(Ω, 1)
			Ω

			X = domain(Ω)
			X

			set(X), projection(X, 1)
			dim(X), eltype(X)
			dim(typeof(X)), eltype(typeof(X))
			markers(X)
			labels(X)
			marker_funcs(X)

			X = domain(reduce(×, ntuple(j -> I0, i)))
			f2 = @embed(X, x->x[i])

			m1 = create_markers(:symbol1 => f2)
			m2 = create_markers(:symbol1 => f2, :symbol2 => f2)
			m3 = create_markers(:symbol1 => f2, :symbol2 => f2, :symbol3 => f2)
			domain(Ω, m1)
			domain(Ω, m2)
			domain(Ω, m3)

			M = mesh(X, npts[i], ntuple(j -> false, i))

			M
			M(1)
			eltype(M), eltype(typeof(M)), dim(M), dim(typeof(M))

			indices(M), npoints(M), M(1), npoints(M, Tuple), hₘₐₓ(M)
			points(M)

			f3 = @embed(M, x->x[i])

			if i == 1
				f1(0)
				f2(0)
				f3(0)
				for p in (1, CartesianIndex(1))
					points(M, p)
					spacing(M, p)
					half_spacing(M, p)
					half_points(M, p)
					cell_measure(M, p)
				end

				idxs = generate_indices(1)
				boundary_indices(M)
				interior_indices(M)
			else
				f2(ntuple(j -> j, i))
				f3(ntuple(j -> j, i))

				for p in (npts[i], CartesianIndex(npts[i]))
					points(M, p)
					spacing(M, p)
					half_spacing(M, p)
					half_points(M, p)
					cell_measure(M, p)
				end
			end

			c = CartesianIndex(npts[i]...)
			cell_measure(M, Tuple(c))

			idxs = generate_indices(1)
			boundary_indices(M)
			interior_indices(M)

			points(M, Iterator), half_points(M, Iterator)
			spacing(M, Iterator), half_spacing(M, Iterator)
			cell_measure(M, Iterator)
		end
	end
end

@recompile_invalidations begin
	for o in (Ω0, Ω1, Ω2)
		Base.eltype(o)
		Base.eltype(typeof(o))
	end

	for i in 1:3
		Ω = reduce(×, ntuple(j -> I0, i))
		X = domain(Ω)

		Base.eltype(X), Base.eltype(typeof(X))
	end
end

## Space compilation
@setup_workload begin
	list_scalars = (1, 1.0, π, 1 // 2)

	ops(::Val{1}) = (diff₋ₓ, diffₓ, jumpₓ, M₋ₕₓ, D₋ₓ)

	function ops(::Val{2})
		ops2 = (D₋ᵧ, diffᵧ, jumpᵧ, diff₋ᵧ, M₋ₕᵧ)
		return (ops2..., ops(Val(1))...)
	end

	function ops(::Val{3})
		ops2 = (D₋₂, diff₂, jump₂, diff₋₂, M₋ₕ₂)
		return (ops2..., ops(Val(2))...)
	end

	tuple_ops() = (∇₋ₕ, diffₕ, jumpₕ, M₋ₕ, diff₋ₕ)

	@compile_workload begin
		for i in 1:3
			X = domain(reduce(×, ntuple(j -> I0, i)))
			M = mesh(X, npts[i], ntuple(j -> false, i))
			f = @embed(X, x->sum(x))
			f2 = @embed(M, x->sum(x))
			Wh = gridspace(M)
			Wh

			eltype(Wh), eltype(typeof(Wh))
			mesh(Wh), ndofs(Wh)

			uh = element(Wh, 1.0)
			wh = element(Wh)
			element(Wh, 1)
			element(Wh, uh.values)
			eltype(uh), eltype(typeof(uh))
			length(uh)
			space(uh)
			size(uh)
			similar(uh)

			copyto!(uh, wh)
			copyto!(uh, wh.values)
			isequal(uh, wh)

			for s in list_scalars
				copyto!(uh, s)
				isequal(uh, s)
				element(Wh, s)
			end

			for op in (-, *, /, +)
				op(uh, wh)

				for p in list_scalars
					op(uh, p)
					op(p, uh)
				end
			end

			for p in list_scalars
				uh .= p
				uh .= p .* wh .+ wh .- .+wh ./ p
			end

			Rₕ!(uh, f2), avgₕ!(uh, f2)
			Rₕ(Wh, f2), avgₕ(Wh, f2)

			Uh = elements(Wh)
			Vh = elements(Wh, Uh.values)
			Uh

			eltype(Uh), eltype(typeof(Uh))
			length(Uh), size(Uh)
			space(Uh)
			similar(Uh)
			copyto!(Uh, Vh)
			Uh[1, 1]

			for p in list_scalars
				Uh.values[1, 1] = p
				Uh .= p
				Uh .= p .* Vh .+ Vh .- .+Vh ./ p
			end

			firstindex(Uh), lastindex(Uh), axes(Uh)
			uh * Uh
			Uh * uh

			for op in (+, -, *)
				op(Uh, Vh)
			end

			for op in (+, -, *, /, ^), p in list_scalars
				op(p, Uh)
				op(Uh, p)
			end

			gen_ops = ops(Val(i))

			for op in gen_ops
				op(Wh), op(uh), op(Uh)
			end

			for tup_ops in tuple_ops()
				tup_ops(Wh), tup_ops(uh), tup_ops(Uh)
			end

			z = ∇₋ₕ(uh)
			normₕ(uh), snorm₁ₕ(uh), norm₁ₕ(uh), norm₊(z)
		end
	end
end

@recompile_invalidations begin
	for i in 1:3
		X = domain(reduce(×, ntuple(j -> I0, i)))
		M = mesh(X, npts[i], ntuple(j -> false, i))
		Base.eltype(M), Base.eltype(typeof(M))

		W = gridspace(M)
		eltype(W), eltype(typeof(W))

		uh = element(W, 1)
		vh = element(W, 1.0)
		Uh = elements(W)
		Vh = elements(W)

		Base.eltype(uh), Base.eltype(typeof(uh))
		Base.similar(uh), Base.copyto!(uh, vh)
		Base.similar(Uh), Base.copyto!(Uh, Vh)
		Base.isequal(uh, vh.values), Base.isequal(uh, vh)

		for p in list_scalars
			Base.copyto!(uh, p)
			Base.isequal(uh, p)

			for op in (Base.:-, Base.:*, Base.:/, Base.:+, Base.:^)
				op(uh, p), op(p, uh), op(uh, vh)
				op(Uh, p), op(p, Uh)
			end
		end

		(Base.:*)(Uh, Vh)
		(Base.:*)(uh, Vh)
		(Base.:*)(Uh, vh)
		Base.size(Uh), Base.axes(Uh), Base.length(Uh)
	end
end

## Space compilation
@compile_workload begin
	for i in 1:3
		X = domain(reduce(×, ntuple(j -> I0, i)))
		M = mesh(X, npts[i], ntuple(j -> false, i))
		sol = @embed(M, x->sum(x))
		Wh = gridspace(M)
		bc = constraints(sol)

		list_constraints = (constraints(sol),
							constraints(:dirichlet => sol),
							constraints(:dirichlet => sol, :dirichlet => sol),
							constraints(:dirichlet => sol, :dirichlet => sol, :dirichlet => sol),
							constraints(:dirichlet => sol, :dirichlet => sol, :dirichlet => sol, :dirichlet => sol))

		bform = bilinearform((u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)), Wh, Wh)
		trialspace(bform)
		testspace(bform)

		A = assemble(bform)
		assemble!(A, bform)

		u = element(Wh, 0.0)
		lform = LinearForm(v -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)), Wh)
		testspace(lform)
		F = assemble(lform)

		for bcs in list_constraints
			markers(bcs)
			constraint_type(bcs)
			symbols(bcs)
			labels(bcs)

			assemble(bform, bcs)
			assemble!(A, bform, bcs)
			assemble(lform, bcs)
			assemble!(F, lform, bcs)

			symmetrize!(A, F, bcs, M)
		end
	end
end
