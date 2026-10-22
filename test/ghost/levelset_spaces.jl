module GhostLevelsetSpacesTests

# `levelset_spaces(Ωₕ, Ω, :D, g)` (#834): the trial and test spaces of the Shortley-Weller
# scheme on a level set. The trial closure keeps the interior nodes, fixes the boundary
# nodes to `g` and leaves the exterior ones at 0; an exterior node read by an interior
# 5-point neighbour takes, in that neighbour's equation, the value of the quadratic through
# the node behind it, the neighbour and the crossing. Solutions are exact for quadratics on
# any grid, for a 3-point second difference is exact on them, and a zero, linear or
# full-arm ghost breaks that. The rows themselves are checked against the hand-built
# oracle in levelset_oracle.jl.

using Test
using Random
using LinearAlgebra: dot, mul!
using Bramble
import Bramble
using Bramble: lift, levelset, levelset_spaces, polygon, Dcₓ, Dcᵧ, D₋ₓ, D₋ᵧ, inner₊ₓ,
               inner₊ᵧ

const BOX = interval(-1.25, 1.25) × interval(-1.25, 1.25)
disk(p) = hypot(p[1], p[2]) - 1
star(p) = hypot(p[1], p[2]) - (0.6 + 0.2 * sin(5 * atan(p[2], p[1])))

# −Δu = −4.
uq(x) = x[1]^2 + x[2]^2 + x[1] + 2x[2]
laplacian(w, v) = innerₕ(-1.0 * Δₕ(w), v)
load(v) = innerₕ(x -> -4.0, v)

labels(Ωₕ) = (Bramble.index_in_marker(Ωₕ, :D), Bramble.index_in_marker(Ωₕ, :D_boundary))

# Whether `f()` throws an `ArgumentError` whose message holds `needle`.
function throws_with(f, needle)
    try
        f()
    catch e
        return e isa ArgumentError && occursin(needle, sprint(showerror, e))
    end
    return false
end

@testset "Level-set spaces (#834)" begin
    @testset "$name: exact for quadratics" for (name, φ, n, seed) in (
        ("disk", disk, (30, 27), 3), ("star", star, (64, 71), 4))
        Random.seed!(seed)
        Ω = domain(BOX, :D => levelset(φ))
        Ωₕ = mesh(Ω, n, (false, false))
        @test Bramble.npoints(Ωₕ, Tuple) == n
        U, W = levelset_spaces(Ωₕ, Ω, :D, uq)
        I, Γ = labels(Ωₕ)
        A, F = assemble(form(U, W, laplacian), form(W, load))
        @test size(A) == (count(I), count(I))
        uh = lift(U, A \ F).data
        uex = Rₕ(gridspace(Ωₕ), uq).data
        @test maximum(abs, uh[I] .- uex[I]) < 1e-10
        @test uh[Γ] == uex[Γ]
        @test all(iszero, uh[.!(I .| Γ)])
        # Exterior nodes read by one interior node along x and by another along y.
        arms = U.rows.arms
        car = CartesianIndices(n)
        readers(p) = arms.reader[arms.ptr[p]:(arms.ptr[p + 1] - 1)]
        along_x(p, r) = car[r][2] == car[p][2]
        @test any(
            p -> any(r -> along_x(p, r), readers(p)) &&
                 any(r -> !along_x(p, r), readers(p)), findall(U.rows.conflict))
        @test count(U.rows.conflict) == count(!=(0), diff(arms.ptr))
        # The test side must be the interior selection.
        @test_throws ArgumentError form(U, U, laplacian)
        @test_throws ArgumentError form(U, load)
    end

    @testset "flux form, grid-function coefficients" begin
        # a = 2, c = 3 and r = 1.5, as grid functions: −2u_xx − 3u_yy + 1.5u = −10 + 1.5u.
        Random.seed!(7)
        Ω = domain(BOX, :D => levelset(star))
        Ωₕ = mesh(Ω, (52, 47), (false, false))
        U, W = levelset_spaces(Ωₕ, Ω, :D, uq)
        S = Bramble.padded_space(U)
        aₕ, cₕ, rₕ = Rₕ(S, x -> 2.0), Rₕ(S, x -> 3.0), Rₕ(S, x -> 1.5)
        f_a = (w, v) -> inner₊ₓ(aₕ * D₋ₓ(w), D₋ₓ(v)) + inner₊ᵧ(cₕ * D₋ᵧ(w), D₋ᵧ(v)) +
                        innerₕ(rₕ * w, v)
        A, F = assemble(form(U, W, f_a), form(W, v -> innerₕ(x -> -10 + 1.5 * uq(x), v)))
        I, _ = labels(Ωₕ)
        @test maximum(abs, lift(U, A \ F).data[I] .- Rₕ(gridspace(Ωₕ), uq).data[I]) < 1e-10
    end

    @testset "Float32" begin
        Random.seed!(5)
        T = Float32
        Ω = domain(interval(T(-1.25), T(1.25)) × interval(T(-1.25), T(1.25)),
            :D => levelset(p -> hypot(p[1], p[2]) - one(T)))
        Ωₕ = mesh(Ω, (25, 25), (false, false))
        U, W = levelset_spaces(Ωₕ, Ω, :D, x -> x[1]^2 + x[2]^2 + x[1] + 2x[2])
        @test eltype(U) == eltype(W) == T
        A, F = assemble(form(U, W, (w, v) -> innerₕ(-one(T) * Δₕ(w), v)),
            form(W, v -> innerₕ(x -> T(-4), v)))
        @test eltype(A) == eltype(F) == T
        I, _ = labels(Ωₕ)
        uh = lift(U, A \ F).data
        @test eltype(uh) == T
        @test maximum(abs, uh[I] .- T.(Rₕ(gridspace(Ωₕ), uq).data[I])) < 1e-4
    end

    @testset "refusals" begin
        Random.seed!(2)
        Ω = domain(BOX, :D => levelset(disk))
        Ωₕ = mesh(Ω, (24, 26), (false, false))
        U, W = levelset_spaces(Ωₕ, Ω, :D, uq)
        l = form(W, load)
        # A diagonal read and a read two nodes away are beyond the 5-point stencil.
        @test_throws ArgumentError form(U, W, (w, v) -> innerₕ(Dcₓ(Dcᵧ(w)), v))
        @test throws_with(() -> form(U, W, (w, v) -> innerₕ(Dcₓ(Dcᵧ(w)), v)),
            "Shortley-Weller stencil")
        @test throws_with(() -> form(U, W, (w, v) -> innerₕ(D₋ₓ(D₋ₓ(w)), v)),
            "Shortley-Weller stencil")
        @test_throws ArgumentError assemble(form(U, W, laplacian), l;
            dirichlet = (:D_boundary => uq,))
        @test_throws ArgumentError form(U, W,
            (w, v) -> innerₕ(w, v) + 2.0 * innerₕ(x -> 1.0, w) * innerₕ(x -> 1.0, v))
        @test_throws ArgumentError form(U × U, W × W,
            (w, v) -> laplacian(w[1], v[1]) + innerₕ(w[2], v[2]))
        @test_throws ArgumentError form(W × W, v -> innerₕ(x -> 1.0, v[1]))
        # The test space as trial side has no rule for the exterior nodes Δₕ reads.
        @test throws_with(() -> form(W, W, laplacian), "Shortley-Weller stencil")
    end

    @testset "matrix-free, contract, semidiscretize" begin
        rng = Xoshiro(13)
        Random.seed!(2)
        Ω = domain(BOX, :D => levelset(star))
        Ωₕ = mesh(Ω, (30, 30), (false, false))
        U, W = levelset_spaces(Ωₕ, Ω, :D, uq)
        # Δₕ reads exterior nodes, whose values only the assembled reduction gives.
        a = form(U, W, laplacian)
        x, y = rand(rng, Bramble.ndofs(U)), rand(rng, Bramble.ndofs(W))
        @test_throws ArgumentError Bramble.matrix_free_operator(a)
        @test_throws ArgumentError a(x, y)
        @test_throws ArgumentError semidiscretize(a, form(W, load))
        # A reaction term reads no exterior node: both apply.
        ar = form(U, W, (w, v) -> innerₕ(w, v))
        Ar = assemble(ar)
        z = zeros(Bramble.ndofs(W))
        mul!(z, Bramble.matrix_free_operator(ar), x)
        @test z ≈ Ar * x
        @test ar(x, y) ≈ dot(y, Ar * x)
    end

    @testset "constructor checks" begin
        Random.seed!(3)
        Ω = domain(BOX, :D => levelset(disk))
        Ωₕ = mesh(Ω, (30, 30), (false, false))
        # A mesh built from another level set: its labels differ.
        Ω9 = domain(BOX, :D => levelset(p -> hypot(p[1], p[2]) - 0.9))
        @test throws_with(() -> levelset_spaces(Ωₕ, Ω9, :D, uq), "not built from")
        @test throws_with(() -> levelset_spaces(Ωₕ, Ω, :Q, uq), "needs a level-set marker")
        ΩP = domain(BOX, :D => polygon([(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]))
        @test throws_with(() -> levelset_spaces(Ωₕ, ΩP, :D, uq), "needs a level-set marker")
        # A saddle of φ in the middle cell: four crossings on its edges.
        Ωx = domain(interval(-1.0, 1.0) × interval(-1.0, 1.0),
            :X => levelset(p -> p[1] * p[2]))
        Mx = mesh(Ωx, (4, 4), (true, true))
        @test_throws ArgumentError levelset_spaces(Mx, Ωx, :X, uq)
        @test throws_with(() -> levelset_spaces(Mx, Ωx, :X, uq), "cell at node (2, 2)")
        # The domain reaches the grid's boundary, or holds no node.
        Ω2 = domain(BOX, :D => levelset(p -> hypot(p[1], p[2]) - 2))
        @test throws_with(() -> levelset_spaces(mesh(Ω2, (9, 9), (true, true)), Ω2, :D, uq),
            "boundary of the grid")
        Ω0 = domain(BOX, :D => levelset(p -> hypot(p[1] - 0.15, p[2] - 0.15) - 0.05))
        @test throws_with(() -> levelset_spaces(mesh(Ω0, (9, 9), (true, true)), Ω0, :D, uq),
            "no interior node")
    end
end

end # module
