function gridspaces_tests()
	I = interval(-1.0, 4.0)

	m = create_markers("Dirichlet" => x -> x[1] - 1.0)
	Ω = domain(I, m)
	Ωₕ = mesh(Ω, 4, false)

	Wₕ = gridspace(Ωₕ)
	vv = Bramble.elements(Wₕ)
	@test validate_equal(length(vv.values), 4 * 4)
	@test validate_equal(length(vv), 4 * 4)
end

gridspaces_tests()
