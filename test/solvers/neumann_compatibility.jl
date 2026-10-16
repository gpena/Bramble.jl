module NeumannCompatibilityTests

using Test
using Bramble
using Logging
using LinearAlgebra: I
using ..TestUtils: WITH_SLOW_TESTS

const Ωd = domain(interval(0.0, 1.0) × interval(0.0, 1.0))

# `(A, F, w)` for `-Δu = f` with `∂ₙu = ∇g ⋅ n` on every facet of the unit square, on a
# constrained ghost space; `w` is the assembled `innerₕ(1, v)`.
function _neumann_problem(Ωₕ, ∇g, f)
    C = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u) => p -> -∇g(p)[1], :xmax => ∂ₙ(u) => p -> ∇g(p)[1],
            :ymin => ∂ₙ(u) => p -> -∇g(p)[2], :ymax => ∂ₙ(u) => p -> ∇g(p)[2]))
    A, F = assemble(form(C, C, (u, v) -> innerₕ(-Δₕ(u), v)), form(C, v -> innerₕ(f, v)))
    w = assemble(form(gridspace(Ωₕ), v -> innerₕ(x -> 1.0, v)))
    return A, F, w
end

# Compatible: `e^{5x} cos 5y` is harmonic, so its fluxes integrate to zero, with large
# derivatives, whose quadrature error is large on a coarse mesh.
_harmonic(Ωₕ) = _neumann_problem(Ωₕ,
    p -> (5exp(5p[1]) * cos(5p[2]), -5exp(5p[1]) * sin(5p[2])), x -> 0.0)

# The form tutorial's `cos πx cos πy + xy/2` with `s` added to the source: compatible for
# `s = 0`, off by `s` in the mean otherwise.
_tutorial(s) = Ωₕ -> _neumann_problem(Ωₕ,
    p -> (-π * sin(π * p[1]) * cos(π * p[2]) + p[2] / 2,
        -π * cos(π * p[1]) * sin(π * p[2]) + p[1] / 2),
    p -> 2π^2 * cos(π * p[1]) * cos(π * p[2]) + s)

# `:random` is seeded by `n`; `:graded` puts the points at `(i/(n-1))²`; `:semi` is uniform
# along x and random (seeded by `n`) along y.
function _mesh(n, kind)
    if kind === :graded
        Ωₕ = mesh(Ωd, (n, n), (true, true))
        q = collect(range(0.0, 1.0; length = n)) .^ 2
        Bramble.change_points!(Ωₕ, markers(Ωd), (q, q))
        return Ωₕ
    end
    Bramble._seed_mesh1d_rng!(n)
    Ωₕ = mesh(Ωd, (n, n), kind === :random ? (false, false) : (true, false))
    Bramble._unseed_mesh1d_rng!()
    return Ωₕ
end

# The result and the warnings `neumann_compatibility` logs.
function _check(problem, Ωₕ)
    logger = Test.TestLogger()
    r = with_logger(() -> Bramble.neumann_compatibility(problem, Ωₕ), logger)
    return r, filter(l -> l.level == Logging.Warn, logger.logs)
end

const _NS = WITH_SLOW_TESTS ? (9, 17, 33, 65, 129) : (9, 17, 33)

@testset "Compatible data never warn" begin
    for kind in (:random, :graded, :semi), n in _NS
        r, logs = _check(_harmonic, _mesh(n, kind))
        @test isempty(logs)
        @test r.compatible
        @test r.ratio >= 2 || r.refined_defect <= 5e-3
        r, logs = _check(_tutorial(0.0), _mesh(n, kind))
        @test isempty(logs) && r.compatible
    end
end

@testset "An O(1) incompatibility warns" begin
    for kind in (:random, :graded, :semi), n in _NS
        r, logs = _check(_tutorial(0.2), _mesh(n, kind))
        @test !r.compatible
        @test length(logs) == 1
        msg = string(only(logs).message)
        @test occursin("defect", msg) && occursin("ratio", msg) && occursin("λ", msg)
        # The multiplier tends to the shift, and the defect does not fall.
        @test r.refined_λ ≈ 0.2 rtol = 0.02
        @test 0.8 < r.ratio < 1.2
    end
end

# The multiplier against `ΣF / Σw`, which holds because the columns of these `A` sum to
# zero, and the defect against its definition.
@testset "λ and the defect" begin
    Ωₕ = _mesh(17, :graded)
    r, _ = _check(_tutorial(0.2), Ωₕ)
    A, F, w = _tutorial(0.2)(Ωₕ)
    @test r.λ ≈ sum(F) / sum(w) rtol = 1e-8
    @test r.defect ≈ abs(sum(F)) / sum(abs, F) rtol = 1e-8
    R = copy(Ωₕ)
    iterative_refinement!(R)
    _, Ff, wf = _tutorial(0.2)(R)
    @test r.refined_λ ≈ sum(Ff) / sum(wf) rtol = 1e-8
    @test r.ratio ≈ r.defect / r.refined_defect
end

@testset "The mesh is not refined" begin
    Ωₕ = _mesh(9, :random)
    pts = map(collect, points(Ωₕ))
    Bramble.neumann_compatibility(_harmonic, Ωₕ)
    @test npoints(Ωₕ, Tuple) == (9, 9)
    @test map(collect, points(Ωₕ)) == pts
end

@testset "A non-singular A is refused" begin
    Ωₕ = _mesh(9, :random)
    regular(Ωₕ) = ((A, F, w) = _harmonic(Ωₕ); (A + I, F, w))
    @test_throws ArgumentError Bramble.neumann_compatibility(regular, Ωₕ)
    err = try
        Bramble.neumann_compatibility(regular, Ωₕ)
    catch e
        e
    end
    @test occursin("constants", sprint(showerror, err))
end

# The docstring's example: zero fluxes, a source with mean 0.2.
@testset "Docstring example" begin
    Ωₕ = mesh(Ωd, (17, 17), (false, false))
    function problem(Ωₕ)
        Wₕ = constrained(gridspace(Ωₕ; ghosts = :boundary),
            u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0,
                :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0))
        f = x -> 2π^2 * cos(π * x[1]) * cos(π * x[2]) + 0.2
        A, F = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(-Δₕ(u), v)),
            form(Wₕ, v -> innerₕ(f, v)))
        w = assemble(form(gridspace(Ωₕ), v -> innerₕ(x -> 1.0, v)))
        return A, F, w
    end
    r, logs = _check(problem, Ωₕ)
    @test !r.compatible && length(logs) == 1
    @test r.λ ≈ 0.2 rtol = 0.05
    @test r.refined_λ ≈ 0.2 rtol = 0.05
end

end
