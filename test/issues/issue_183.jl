module IssuesIssue183Tests

using Test
using Bramble

# gpena/Bramble.jl#183: Dₕ (cross_weighted_difference) used to truncate both endpoints to
# zero along every direction. It now falls back, at each end, to the one-sided difference
# the near side still defines. This is the minimal 1D reproducer; the full directional and
# matrix-agreement coverage lives in test/space/cross_weighted_difference.jl.
@testset "Dₕ endpoints are one-sided, not truncated (#183)" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
    Wₕ = gridspace(Ωₕ)
    n = npoints(Ωₕ)
    uₕ = Rₕ(Wₕ, x -> exp(x))
    u = parent(uₕ)
    h = [spacing(Ωₕ, i) for i in 1:n]

    d = parent(Dₕₓ(uₕ))
    @test d[1] ≈ (u[2] - u[1]) / h[1]
    @test d[n] ≈ (u[n] - u[n - 1]) / h[n]
    @test !iszero(d[1])
    @test !iszero(d[n])
end

end # module IssuesIssue183Tests
