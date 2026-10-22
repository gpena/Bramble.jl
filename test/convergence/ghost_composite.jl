module ConvergenceGhostCompositeTests

# Second-order convergence of the coupled two-component problem on a composite of ghost
# spaces (GhostCoupled.jl, #781) on random and graded meshes. Moved here from
# test/ghost/composite.jl, which checks the same problem against a block reference.

using Test
using Bramble
import Bramble
using Bramble: lift

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(@__DIR__, "..", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND
isdefined(Main, :GhostCoupled) ||
    Base.include(Main, joinpath(@__DIR__, "..", "ghost", "GhostCoupled.jl"))
using Main.GhostCoupled: coupled_solve, u1, u2

# The discrete H¹ error `(‖e‖_{H}² + Σ_d ‖D₋_d e‖_+²)^{1/2}` at the physical nodes.
function h1_error(Ωₕ, e)
    xs = O.axis_points(Ωₕ)
    ĥ(x) = [(i > 1 ? x[i] - x[i - 1] : 0.0) / 2 + (i < length(x) ? x[i + 1] - x[i] : 0.0) / 2
            for i in eachindex(x)]
    E = reshape(e, map(length, xs))
    w = ĥ(xs[1]) .* ĥ(xs[2])'
    total = sum(w .* abs2.(E))
    total += sum((1 ./ diff(xs[1])) .* ĥ(xs[2])' .* abs2.(diff(E; dims = 1)))
    total += sum(ĥ(xs[1]) .* (1 ./ diff(xs[2]))' .* abs2.(diff(E; dims = 2)))
    return sqrt(total)
end

# The physical values of each component of a lifted solution.
function physical_values(V, U)
    out, start = Vector{Float64}[], 0
    for C in V.spaces
        G = mesh(C)
        n = prod(Bramble.npoints(G, Tuple))
        plin = LinearIndices(Bramble.npoints(G, Tuple))
        push!(out, parent(U)[start .+ vec(plin[Bramble.physical_indices(G)])])
        start += n
    end
    return out
end

function lsq_order(hs, errs)
    X, Y = log.(hs), log.(errs)
    return sum((X .- sum(X) / length(X)) .* (Y .- sum(Y) / length(Y))) /
           sum(abs2, X .- sum(X) / length(X))
end

@testset "Composite ghost order, $kind meshes" for kind in (:random, :graded)
    hs, errs = Float64[], NTuple{2, Float64}[]
    for ℓ in 0:2
        Ωₕ = O.make_mesh(kind, ℓ; dim = 2, n0 = 9)
        V, u = coupled_solve(Ωₕ)
        xs = O.axis_points(Ωₕ)
        nodes = vec([(x, y) for x in xs[1], y in xs[2]])
        v1, v2 = physical_values(V, lift(V, u))
        push!(hs, maximum(x -> maximum(diff(x)), xs))
        push!(errs, (h1_error(Ωₕ, v1 - u1.(nodes)), h1_error(Ωₕ, v2 - u2.(nodes))))
    end
    @test maximum(hs) / minimum(hs) > 3
    for c in 1:2
        @test lsq_order(hs, getindex.(errs, c)) > 1.8
    end
end

end # module ConvergenceGhostCompositeTests
