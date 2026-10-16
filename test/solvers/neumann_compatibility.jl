module NeumannCompatibilityTests

using Test
using Bramble
using Logging
using LinearAlgebra: I
using ..TestUtils: WITH_SLOW_TESTS

const Ωd = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
const Ωr = domain(interval(0.0, 1.0) × interval(0.0, 2.0))

# `(A, F, w)` for `-Δu = f` with `∂ₙu = ∇g ⋅ n` on every facet of a rectangle, on a
# constrained ghost space; `w` is the assembled `innerₕ(1, v)`.
function _neumann_problem(Ωₕ, ∇g, f)
    C = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u) => p -> -∇g(p)[1], :xmax => ∂ₙ(u) => p -> ∇g(p)[1],
            :ymin => ∂ₙ(u) => p -> -∇g(p)[2], :ymax => ∂ₙ(u) => p -> ∇g(p)[2]))
    A, F = assemble(form(C, C, (u, v) -> innerₕ(-Δₕ(u), v)), form(C, v -> innerₕ(f, v)))
    w = assemble(form(gridspace(Ωₕ), v -> innerₕ(x -> 1.0, v)))
    return A, F, w
end

# Compatible for `s = 0`: `e^{kx} cos ky` is harmonic, so its fluxes integrate to zero,
# with large derivatives, whose quadrature error is large (and pre-asymptotic) on a coarse
# mesh. A source `s ≠ 0` adds an incompatibility that error can hide on the coarse mesh.
function _harmonic(k, s = 0.0)
    Ωₕ -> _neumann_problem(Ωₕ,
        p -> (k * exp(k * p[1]) * cos(k * p[2]), -k * exp(k * p[1]) * sin(k * p[2])), x -> s)
end

# The form tutorial's `cos πx cos πy + xy/2` with `s` added to the source: compatible for
# `s = 0`, off by `s` in the mean otherwise.
function _tutorial(s)
    Ωₕ -> _neumann_problem(Ωₕ,
        p -> (-π * sin(π * p[1]) * cos(π * p[2]) + p[2] / 2,
            -π * cos(π * p[1]) * sin(π * p[2]) + p[1] / 2),
        p -> 2π^2 * cos(π * p[1]) * cos(π * p[2]) + s)
end

# `:random` is seeded by `seed`; `:graded` puts the points at `(i/(n-1))²`; `:semi` is
# uniform along x and random (seeded by `seed`) along y.
function _mesh(n, kind; Ω = Ωd, seed = n)
    if kind === :graded
        Ωₕ = mesh(Ω, (n, n), (true, true))
        q = collect(range(0.0, 1.0; length = n)) .^ 2
        Bramble.change_points!(Ωₕ, markers(Ω), (q, Ω === Ωr ? 2q : q))
        return Ωₕ
    end
    Bramble._seed_mesh1d_rng!(seed)
    Ωₕ = mesh(Ω, (n, n), kind === :random ? (false, false) : (true, false))
    Bramble._unseed_mesh1d_rng!()
    return Ωₕ
end

# The result and the warnings `neumann_compatibility` logs.
function _check(problem, Ωₕ, args...)
    logger = Test.TestLogger()
    r = with_logger(() -> Bramble.neumann_compatibility(problem, Ωₕ, args...), logger)
    return r, filter(l -> l.level == Logging.Warn, logger.logs)
end

const _NS = WITH_SLOW_TESTS ? (9, 17, 33, 65, 129) : (9, 17, 33)

@testset "Compatible data never warn" begin
    for kind in (:random, :graded, :semi), n in _NS

        r, logs = _check(_harmonic(5), _mesh(n, kind))
        @test isempty(logs)
        @test r.compatible
        r, logs = _check(_tutorial(0.0), _mesh(n, kind))
        @test isempty(logs) && r.compatible
    end
end

# On [0,1]×[0,2] with k = 8 the first refinement is pre-asymptotic on some coarse meshes:
# seed 9 at n = 9 has λ = 1082 then -1500 (defects 0.05 and 0.07), and seed 19 on the
# one-axis-uniform mesh a ratio of 1.63 at a refined defect of 0.036.
@testset "Pre-asymptotic compatible data" begin
    r, logs = _check(_harmonic(8), _mesh(9, :random; Ω = Ωr, seed = 9))
    @test isempty(logs) && r.compatible
    @test r.refined_defect > 5e-3
    r, logs = _check(_harmonic(8), _mesh(9, :semi; Ω = Ωr, seed = 19))
    @test isempty(logs) && r.compatible
    @test r.refined_defect > 5e-3 && 0.5 <= r.ratio <= 2
    @test r.second_ratio > 2.5
    for kind in (:random, :semi), n in (WITH_SLOW_TESTS ? (9, 17, 33) : (9,)), seed in 1:20
        r, logs = _check(_harmonic(8), _mesh(n, kind; Ω = Ωr, seed))
        @test isempty(logs) && r.compatible
    end
end

@testset "An O(1) incompatibility warns" begin
    # The rectangle stops at n = 33: the second refinement of n = 129 has 513² points.
    for Ω in (Ωd, Ωr), kind in (:random, :graded, :semi), n in (Ω === Ωd ? _NS : _NS[1:3])
        r, logs = _check(_tutorial(0.2), _mesh(n, kind; Ω))
        @test !r.compatible
        @test length(logs) == 1
        msg = string(only(logs).message)
        @test occursin("defect", msg) && occursin("ratio", msg) && occursin("λ", msg)
        # The multiplier tends to the shift and keeps its sign at both refinements.
        @test r.refined_λ ≈ 0.2 rtol = 0.02
        @test 0.7 < r.ratio < 1.3
        @test 0.7 < r.second_ratio < 1.3
    end
end

# A shifted source whose coarse multiplier the quadrature error of the fluxes swamps:
# λ = 0.88, -6.9, -2.7 on the coarse meshes against about -15 to -17 on their
# refinements, and a coarse ratio of -0.54 or 2.55. The refined pair decides.
@testset "Incompatibility under a large error" begin
    for (s, n, seed) in ((-20.0, 17, 1), (-20.0, 17, 2), (-20.0, 17, 3), (20.0, 9, 2),
        (5.0, 17, 2))
        r, logs = _check(_harmonic(5, s), _mesh(n, :random; seed))
        @test !r.compatible && length(logs) == 1
        @test 0.4 <= r.second_ratio <= 2.5
    end
end

# The rule reads `λ Σ|w|`, so a weight of another scale or shape decides alike: with
# `w = 1` the multiplier is about 81 times smaller on a 9×9 mesh and 289 on its refinement.
@testset "Unit weights" begin
    ones_w(problem) = Ωₕ -> ((A, F, w) = problem(Ωₕ); (A, F, ones(length(w))))
    Ωₕ = _mesh(9, :random)
    r, logs = _check(ones_w(_tutorial(0.2)), Ωₕ)
    @test !r.compatible && length(logs) == 1
    @test 0.7 < r.ratio < 1.3 && 0.7 < r.second_ratio < 1.3
    q, _ = _check(_tutorial(0.2), Ωₕ)
    @test r.defect ≈ q.defect rtol = 1e-8
    @test r.refined_defect ≈ q.refined_defect rtol = 1e-8
    r, logs = _check(ones_w(_tutorial(0.0)), Ωₕ)
    @test isempty(logs) && r.compatible
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
    @test r.ratio ≈ r.λ / r.refined_λ
    iterative_refinement!(R)
    _, Fff, wff = _tutorial(0.2)(R)
    @test r.second_ratio ≈ r.refined_λ / (sum(Fff) / sum(wff)) rtol = 1e-8
    # Asymptotic compatible data need no second refinement.
    r, _ = _check(_harmonic(5), _mesh(33, :graded))
    @test r.second_ratio === nothing
end

@testset "Zero data are compatible" begin
    zero_data(Ωₕ) = _neumann_problem(Ωₕ, p -> (0.0, 0.0), x -> 0.0)
    r, logs = _check(zero_data, _mesh(9, :random))
    @test isempty(logs) && r.compatible
    @test r.defect == 0 && r.refined_defect == 0 && r.λ == 0 && r.refined_λ == 0
    @test !isnan(r.ratio)
end

@testset "The mesh is not refined" begin
    Ωₕ = _mesh(9, :random)
    pts = map(collect, points(Ωₕ))
    _check(_tutorial(0.2), Ωₕ)
    @test npoints(Ωₕ, Tuple) == (9, 9)
    @test map(collect, points(Ωₕ)) == pts
end

# A domain with a user marker: the copies are refined with the domain's markers, which
# the one-argument refinement cannot re-evaluate.
@testset "Domain markers" begin
    X = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωm = domain(X, markers(X, :Wall => :left))
    Ωₕ = mesh(Ωm, (9, 9), (false, false))
    @test_throws ArgumentError Bramble.neumann_compatibility(_tutorial(0.2), Ωₕ)
    seen = Int[]
    walled(s) = Ωₕ -> (push!(seen, count(markers(Ωₕ)[:Wall])); _tutorial(s)(Ωₕ))
    r, logs = _check(walled(0.2), Ωₕ, markers(Ωm))
    @test !r.compatible && length(logs) == 1
    @test seen == [9, 17, 33]   # the marker is re-evaluated on every refinement
    r, logs = _check(_tutorial(0.0), Ωₕ, markers(Ωm))
    @test isempty(logs) && r.compatible
    @test npoints(Ωₕ, Tuple) == (9, 9)
end

@testset "Float32" begin
    X = interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)
    Ωₕ = mesh(domain(X), (9, 9), (false, false))
    function problem(s)
        return function (Ωₕ)
            C = constrained(gridspace(Ωₕ; ghosts = :boundary),
                u -> (:xmin => ∂ₙ(u) => 0.0f0, :xmax => ∂ₙ(u) => 0.0f0,
                    :ymin => ∂ₙ(u) => 0.0f0, :ymax => ∂ₙ(u) => 0.0f0))
            π₃₂ = Float32(π)
            f = x -> 2π₃₂^2 * cos(π₃₂ * x[1]) * cos(π₃₂ * x[2]) + s
            A, F = assemble(form(C, C, (u, v) -> innerₕ(-Δₕ(u), v)),
                form(C, v -> innerₕ(f, v)))
            return A, F, assemble(form(gridspace(Ωₕ), v -> innerₕ(x -> 1.0f0, v)))
        end
    end
    @test eltype(first(problem(0.0f0)(Ωₕ))) === Float32
    r, logs = _check(problem(0.0f0), Ωₕ)
    @test isempty(logs) && r.compatible
    r, logs = _check(problem(0.2f0), Ωₕ)
    @test !r.compatible && length(logs) == 1
    @test r.refined_λ ≈ 0.2f0 rtol = 0.02
end

@testset "A non-singular A is refused" begin
    Ωₕ = _mesh(9, :random)
    regular(Ωₕ) = ((A, F, w) = _harmonic(5)(Ωₕ); (A + I, F, w))
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
