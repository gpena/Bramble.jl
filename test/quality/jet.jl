using JET
using Bramble

@testset "JET static analysis" begin
	if VERSION >= v"1.12" && isempty(VERSION.prerelease)
		try
			jet_report = JET.report_package(Bramble; target_modules = (Bramble,), toplevel_logger = nothing)
			reports = JET.get_reports(jet_report)
			@test length(reports) == 0
		catch e
			@test_skip "JET not functional on this Julia build: $e"
		end
	else
		@test_skip "JET full-package static analysis target is Julia 1.12+ stable"
	end
end