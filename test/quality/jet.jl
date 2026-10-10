module QualityJetTests

using Test
using JET
using Bramble

@testset "JET" begin
    if isempty(VERSION.prerelease)
        try
            jet_report = JET.report_package(
                Bramble; target_modules = (Bramble,), toplevel_logger = nothing
            )
            reports = JET.get_reports(jet_report)
            @test length(reports) == 0
        catch e
            # A JET that cannot analyse the package is a failed gate, not a skip.
            @error "JET could not analyse Bramble" exception = e
            @test false
        end
    else
        @test_skip "JET full-package static analysis is skipped on prerelease Julia"
    end
end

end # module QualityJetTests
