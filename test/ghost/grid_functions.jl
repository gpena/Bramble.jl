module GhostGridFunctionsTests

# `normₕ` and `interpolate_at` of an element of a ghost space read the physical nodes, the
# first `npoints` degrees of freedom, and never the ghosts (here the sentinel 1e300). The
# oracle is the same call on the plain space `gridspace(Ωₕ)`, which shares no ghost code,
# and a hand sum or an affine function. A constrained space whose degrees of freedom omit
# fixed nodes or carry data has no grid function in a DOF vector: both refuse naming `lift`.

using Test
using Random
using Bramble
import Bramble
using Bramble: change_points!
using ..TestUtils: WITH_3D_TESTS

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracle1D.jl"))
isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O1 = Main.GhostOracle1D
const ON = Main.GhostOracleND

const SENTINEL = 1.0e300

# An element of `Wg` whose physical values are `vals` and whose ghosts hold the sentinel,
# and the same physical values on the plain space.
function sentinel_pair(Wg, W, vals)
    np = length(vals)
    ug = element(Wg)
    ug .= SENTINEL
    ug[1:np] .= vals
    up = element(W)
    up .= vals
    return ug, up
end

function message(f)
    e = try
        f()
        nothing
    catch err
        err
    end
    return e isa ArgumentError ? sprint(showerror, e) : nothing
end

# `f` throws an `ArgumentError` whose message names `lift`.
refuses_naming_lift(f) = (msg = message(f); msg !== nothing && occursin("lift", msg))

# Seeded points of the unit cube strictly inside, and one cell's width from each facet.
function query_points(rng, D, m)
    return [ntuple(_ -> 0.02 + 0.96 * rand(rng), D) for _ in 1:m]
end

as_arg(x::NTuple{1}) = x[1]
as_arg(x) = x

function reads_physical_only(Ωₕ, ghosts; seed)
    rng = MersenneTwister(seed)
    D = Bramble.dim(Ωₕ)
    W = gridspace(Ωₕ)
    Wg = gridspace(Ωₕ; ghosts = ghosts)
    np = npoints(Ωₕ)
    ug, up = sentinel_pair(Wg, W, rand(rng, np))
    ok = length(ug) == ndofs(Wg) && normₕ(ug) ≈ normₕ(up) && isfinite(normₕ(ug))
    for x in query_points(rng, D, 12)
        ok &= interpolate_at(ug, as_arg(x)) ≈ interpolate_at(up, as_arg(x))
    end
    return ok
end

@testset "Ghost grid functions" begin
    @testset "1D selection space, non-uniform mesh" begin
        Ωₕ = O1.make_mesh(:random, 0; n0 = 9, seed = 5)
        @test !Bramble.is_uniform(Ωₕ)
        x = collect(Float64, points(Ωₕ))
        n = length(x)
        for ghosts in ((:xmin,), (:xmax,), (:xmin, :xmax), :boundary)
            @test reads_physical_only(Ωₕ, ghosts; seed = 1)
        end
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        rng = MersenneTwister(2)
        v = rand(rng, n)
        ug, _ = sentinel_pair(Wg, gridspace(Ωₕ), v)
        # trapezoid weights by hand
        w = [i == 1 ? (x[2] - x[1]) / 2 : i == n ? (x[n] - x[n - 1]) / 2 :
                                          (x[i + 1] - x[i - 1]) / 2 for i in 1:n]
        @test normₕ(ug) ≈ sqrt(sum(w .* v .^ 2))
        # an affine field is interpolated exactly, and at a node it is the node's value
        a(t) = 3.0 - 2.0 * t
        ua = element(Wg)
        ua .= SENTINEL
        ua[1:n] .= a.(x)
        for t in (0.0, 0.123, 0.5, 0.9999, 1.0, x[3], x[n - 1])
            @test interpolate_at(ua, t) ≈ a(t)
        end
        # the domain is the physical one: a ghost coordinate is outside it
        h = x[2] - x[1]
        @test_throws ArgumentError interpolate_at(ua, -0.5 * h)
        @test_throws ArgumentError interpolate_at(ua, 1.0 + 0.5 * h)
        @test interpolate_at(ua, -0.5 * h; outside = :clamp) ≈ a(0.0)
        @test interpolate_at(ua, -0.5 * h; outside = :extrapolate) ≈ a(-0.5 * h)
        @test interpolate_at(ua, 1.5; outside = 7.0) == 7.0
        @test_throws ArgumentError interpolate_at(ua, 0.5; outside = :nowhere)
    end

    @testset "2D selection spaces, non-uniform mesh" begin
        Ωₕ = ON.make_mesh(:random, 0; dim = 2, n0 = 7, seed = 9)
        @test !Bramble.is_uniform(Ωₕ)
        for ghosts in ((:xmin,), (:ymax,), (:xmin, :xmax, :ymin), :boundary)
            @test reads_physical_only(Ωₕ, ghosts; seed = 3)
        end
        # an affine field, interpolated exactly on the physical cells
        W = gridspace(Ωₕ)
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        f(p) = 1.0 + 2.0 * p[1] - 3.0 * p[2]
        ua = element(Wg)
        ua .= SENTINEL
        ua[1:npoints(Ωₕ)] .= [f(Bramble.point(Ωₕ, I)) for I in vec(collect(Bramble.indices(Ωₕ)))]
        for p in query_points(MersenneTwister(4), 2, 10)
            @test interpolate_at(ua, p) ≈ f(p)
        end
        @test_throws ArgumentError interpolate_at(ua, (-0.01, 0.5))
        @test interpolate_at(ua, (-0.01, 0.5); outside = :clamp) ≈ f((0.0, 0.5))
        @test interpolate_at(ua, (0.5, 3.0); outside = -1.0) == -1.0
    end

    WITH_3D_TESTS && @testset "3D selection spaces, non-uniform mesh" begin
        Ωₕ = ON.make_mesh(:random, 0; dim = 3, n0 = 4, seed = 11)
        @test !Bramble.is_uniform(Ωₕ)
        for ghosts in ((:zmin,), (:xmax, :ymin), :boundary)
            @test reads_physical_only(Ωₕ, ghosts; seed = 5)
        end
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        f(p) = 1.0 + p[1] - 2.0 * p[2] + 0.5 * p[3]
        ua = element(Wg)
        ua .= SENTINEL
        ua[1:npoints(Ωₕ)] .= [f(Bramble.point(Ωₕ, I)) for I in vec(collect(Bramble.indices(Ωₕ)))]
        for p in query_points(MersenneTwister(6), 3, 6)
            @test interpolate_at(ua, p) ≈ f(p)
        end
    end

    @testset "constrained DOFs that are the nodes" begin
        # Neumann relations with no data and no fixed node eliminate every ghost, so the DOFs
        # are the physical nodes in order, and the DOF vector is a grid function.
        Ωₕ = ON.make_mesh(:random, 0; dim = 2, n0 = 7, seed = 13)
        W = gridspace(Ωₕ)
        C = constrained(gridspace(Ωₕ; ghosts = (:xmin, :ymax)),
            u -> (:xmin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0))
        @test ndofs(C) == npoints(Ωₕ)
        rng = MersenneTwister(7)
        uc = element(C)
        uc .= rand(rng, ndofs(C))
        up = element(W)
        up .= parent(uc)
        @test normₕ(uc) ≈ normₕ(up)
        # the lifted grid function, whose ghosts hold the relations' values, weighs the same
        @test normₕ(uc) ≈ normₕ(Bramble.lift(C, parent(uc)))
        for p in query_points(rng, 2, 8)
            @test interpolate_at(uc, p) ≈ interpolate_at(up, p)
        end
    end

    @testset "DOFs that are no grid function refuse" begin
        Ωₕ = ON.make_mesh(:random, 0; dim = 2, n0 = 7, seed = 17)
        x = (0.37, 0.61)
        # data alone: inhomogeneous Neumann, no fixed node
        Cd = constrained(gridspace(Ωₕ; ghosts = (:xmin,)), u -> (:xmin => ∂ₙ(u) => 1.0,))
        # a fixed node alone: Dirichlet data 0, so `d = 0`
        Cf = constrained(gridspace(Ωₕ; ghosts = (:xmin, :ymin)),
            u -> (:xmin => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0, :xmax => u => 0.0))
        @test all(iszero, Bramble.closure(Cf)[2])
        @test ndofs(Cf) < npoints(Ωₕ) + 2 * 7
        # both
        Cb = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
            u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u => 2.0))
        # data depending on time, zero at t = 0
        Ct = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
            u -> (:xmin => ∂ₙ(u) => (x, t) -> sin(t),))
        @test all(iszero, Bramble.closure(Ct)[2])
        for (name, C) in (("data", Cd), ("fixed nodes", Cf), ("both", Cb), ("time", Ct))
            u = element(C)
            u .= 1.0
            @test refuses_naming_lift(() -> normₕ(u))
            @test refuses_naming_lift(() -> interpolate_at(u, x))
            @test_throws ArgumentError normₕ(u)
            @test_throws ArgumentError interpolate_at(u, x)
        end
        # the fix named works: the lifted grid function lives on the padded space
        Cl = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
            u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => u => 0.0))
        ul = element(Cl)
        ul .= 1.0
        @test refuses_naming_lift(() -> normₕ(ul))
        lifted = Bramble.lift(Cl, parent(ul))
        @test isfinite(normₕ(lifted))
        # the restriction and the cell average keep refusing
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        @test_throws ArgumentError Rₕ(Wg, p -> p[1])
        @test_throws ArgumentError avgₕ(Wg, p -> p[1])
    end

    @testset "two points and other element types" begin
        # a one-cell mesh, two points: the whole physical mesh is one cell
        Ωₕ = mesh(domain(interval(0.0, 2.0)), 2, false)
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        ug = element(Wg)
        ug .= SENTINEL
        ug[1:2] .= (1.0, 3.0)
        @test length(ug) == 4
        @test normₕ(ug) ≈ sqrt(1.0 * 1.0^2 + 1.0 * 3.0^2)
        @test interpolate_at(ug, 0.5) ≈ 1.5
        # a Float32 mesh keeps its element type
        Ω32 = mesh(domain(interval(0.0f0, 1.0f0)), 6, true)
        W32 = gridspace(Ω32; ghosts = :boundary)
        u32 = element(W32)
        u32 .= 1.0f30      # its square overflows a Float32
        u32[1:6] .= 1.0f0
        @test normₕ(u32) isa Float32
        @test normₕ(u32) ≈ 1.0f0
        @test interpolate_at(u32, 0.3f0) ≈ 1.0f0
        # complex data: the physical nodes only, conjugated
        Ωc = mesh(domain(interval(0.0, 1.0)), 5, true)
        Wc = gridspace(Ωc; ghosts = (:xmin,))
        uc = element(Wc, 1.0e300 + 0im)
        uc[1:5] .= 1.0 + 1.0im
        @test normₕ(uc) ≈ sqrt(2.0)
        @test interpolate_at(uc, 0.4) ≈ 1.0 + 1.0im
    end

    @testset "a moved mesh refuses, no allocation" begin
        Ωₕ = ON.make_mesh(:random, 0; dim = 2, n0 = 6, seed = 21)
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        ug = element(Wg)
        ug .= 1.0
        normₕ(ug)
        interpolate_at(ug, (0.4, 0.4))
        @test (@allocated normₕ(ug)) == 0
        @test (@allocated interpolate_at(ug, (0.4, 0.4))) == 0
        Ω1 = O1.make_mesh(:random, 0; n0 = 6, seed = 3)
        W1 = gridspace(Ω1; ghosts = :boundary)
        u1 = element(W1)
        u1 .= 1.0
        interpolate_at(u1, 0.4)
        @test (@allocated interpolate_at(u1, 0.4)) == 0
        axes = ntuple(d -> collect(points(Ωₕ(d))) .^ 2, 2)
        change_points!(Ωₕ, axes)
        @test_throws ArgumentError normₕ(ug)
        @test_throws ArgumentError interpolate_at(ug, (0.4, 0.4))
    end
end

end # module GhostGridFunctionsTests
