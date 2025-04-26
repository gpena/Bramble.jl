using JET

@testset "JET Static Analysis" begin
	jet_report = JET.report_package(Bramble, ignored_modules = (Base,), analyze_from_definitions = true, ipo_constant_propagation = true, ignore_missing_comparison = false, unoptimize_throw_blocks = true, aggressive_constant_propagation = true,
									stacktrace_types_limit = nothing,
									print_inference_success = false,
									print_toplevel_success = false, fullpath = false)
	reports = JET.get_reports(jet_report)
	@test length(reports) == 0
end