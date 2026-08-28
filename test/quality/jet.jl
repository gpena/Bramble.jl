using JET
using Bramble

@testset "JET static analysis" begin
	if VERSION >= v"1.12"
		jet_report = JET.report_package(Bramble; target_modules = (Bramble,), toplevel_logger = nothing)
		reports = JET.get_reports(jet_report)
		@test length(reports) == 0
	else
		@test_skip "JET full-package static analysis target is Julia 1.12+"
	end
end