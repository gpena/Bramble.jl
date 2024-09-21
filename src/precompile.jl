import PrecompileTools: @compile_workload, @setup_workload

## Domain precompilation
@setup_workload begin
	pts = [-1, -1.0, 0, 0.0, 1 // 2, π]
	combinations = ((pts[i], pts[j]) for i in eachindex(pts) for j in eachindex(pts) if pts[i] <= pts[j])

	npts = ((2,), (2, 2), (2, 2, 2))

	@compile_workload begin
		for p in combinations
			interval(p...)
		end

		I0 = interval(-3.0, 10.0)
		Ω0 = cartesianproduct(I0)
		Ω0 = cartesianproduct(I0)
		cartesianproduct(Ω0)
		Ω0(1)

		Ω1 = cartesianproduct(I0) × cartesianproduct(I0)
		Ω2 = Ω1 × cartesianproduct(I0)
		Ω3 = cartesianproduct(I0) × Ω1

		omegas = (Ω0, Ω1, Ω2, Ω3)
		for o in omegas
			eltype(o), eltype(o), dim(o)
			eltype(typeof(o)), eltype(typeof(o)), dim(typeof(o))
		end

		for i in 1:3
			Ii = ntuple(j -> I0, i)
			Ω = reduce(×, Ii)
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
		end
	end
end

## Mesh precompilation
@setup_workload begin
	npts = ((2,), (2, 2), (2, 2, 2))

	@compile_workload begin
		I0 = interval(-3.0, 10.0)

		for i in 1:3
			X = domain(reduce(×, ntuple(j -> I0, i)))
			M = mesh(X, npts[i], ntuple(j -> false, i))
			M
			M(1)
			eltype(M), eltype(typeof(M)), dim(M), dim(typeof(M))

			indices(M), npoints(M), M(1), npoints(M, Tuple), hₘₐₓ(M)
			points(M)

			if i == 1
				for p in (1, CartesianIndex(1))
					point(M, p)
					spacing(M, p)
					half_spacing(M, p)
					half_points(M, p)
					cell_measure(M, p)
				end

				idxs = generate_indices(1)
				boundary_indices(M)
				interior_indices(M)
			else
				for p in (npts[i], CartesianIndex(npts[i]))
					point(M, p)
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

## Space compilation
@setup_workload begin
	npts = ((2,), (2, 2), (2, 2, 2))
	list_scalars = (1, 1.0, π, 1 // 2)
	f(x) = sum(x)

	@compile_workload begin
		I0 = interval(-3.0, 10.0)

		for i in 1:3
			X = domain(reduce(×, ntuple(j -> I0, i)))
			M = mesh(X, npts[i], ntuple(j -> false, i))

			Wh = gridspace(M)
			Wh
			getcache(Wh, get_symbol_diff_matrix(Val(1)))
			iscached(Wh, get_symbol_diff_matrix(Val(1)))

			eltype(Wh), eltype(typeof(Wh))
			mesh(Wh), ndofs(Wh)

			uh = element(Wh, 1.0)
			wh = element(Wh)
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

			Rₕ!(uh, f)
			Rₕ(Wh, f)
			avgₕ!(uh, f)
			avgₕ(Wh, f)

			__shift_index1(CartesianIndex(ntuple(i -> 1, i)))

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

			for op in (+, -, *, /), p in list_scalars
				op(p, Uh)
				op(Uh, p)
			end

			#operators = ((D₋ₓ, jumpₓ, Mₕₓ, diff₋ₓ, Mₕ),)
			#for op in operators[i]
			#		wh .= op(uh)[1]
			#		wh .= op(Wh)*uh.values
			#			op(Uh)
			#			end

			#=
						operators = ((D₋ₓ, jumpₓ, Mₕₓ, diff₋ₓ, Mₕ),
									(D₋ₓ, D₋ᵧ, jumpₓ, jumpᵧ, diff₋ₓ, diff₋ᵧ, Mₕₓ, Mₕᵧ),
									(D₋ₓ, D₋ᵧ, D₋₂, jumpₓ, jumpᵧ, jump₂, diff₋ₓ, diff₋ᵧ, diff₋₂, Mₕₓ, Mₕᵧ, Mₕ₂))

						for op in operators[i]
							u1h .= op(uh)[1]
							u2h .= op(Wh)*uh.values
							op(Uh)
						end

						z = ∇ₕ(uh)
						normₕ(uh), snorm₁ₕ(uh), norm₁ₕ(uh), norm₊(z)
			=#
			#jump(z), diff(z), Mₕ(z)
		end
	end
end
