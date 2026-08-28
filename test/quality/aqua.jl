using Aqua

@testset "Aqua analysis" begin
	Aqua.test_all(Bramble;
				  piracies = true,
				  ambiguities = false,
				  unbound_args = true,
				  undefined_exports = true,
				  project_extras = false,
				  stale_deps = false,
				  deps_compat = false,
				  persistent_tasks = false)
	Aqua.test_ambiguities(Bramble; recursive = false)
end