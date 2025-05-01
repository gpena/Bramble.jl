using JET
using StyledStrings
using OhMyREPL

@testset "JET Static Analysis" begin
	jet_report = JET.report_package(Bramble, ignored_modules = (Base, StyledStrings, OhMyREPL), analyze_from_definitions = true, toplevel_logger = nothing)

	reports = JET.get_reports(jet_report)
	@test length(reports) == 0
end
