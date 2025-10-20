using Aqua

@testset "Aqua analysis" begin
	Aqua.test_all(Bramble;
				  piracies = true,
				  ambiguities = true,
				  unbound_args = false,
				  undefined_exports = true,
				  project_extras = false,
				  stale_deps = false,
				  deps_compat = false,
				  persistent_tasks = true)
end