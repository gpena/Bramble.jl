using JET
using Bramble

@testset "JET static analysis" begin
	jet_report = JET.report_package(Bramble, ignored_modules = (Base,), analyze_from_definitions = true, toplevel_logger = nothing)

	reports = JET.get_reports(jet_report)
	#println(reports)
	@test length(reports) == 0
end