module FormLaplacianFormTests

# `Δₕ(u)` in a form is one three-tap `SecondDifference` node per axis, entry for entry the
# spelled-out `Σ_d D̃_d(D₋_d(u))` (gpena/Bramble.jl#774). Checked on random non-uniform
# meshes in 1D and 2D (3D behind `TestUtils.WITH_3D_TESTS`): on the trial and the test side,
# under a grid-function coefficient, inside `jumpₓ`, `D₋ₓ`, `Dcₓ` and itself, and in linear
# forms; against the runtime `Δₕ` of a grid function, an independent engine; and on the
# padded mesh of a `GhostMesh`, where the first physical point reads its ghost.

using Test
using Bramble
import Bramble
using LinearAlgebra: Diagonal, norm, ×
using ..TestUtils: WITH_3D_TESTS

const tol = 1e-12

relerr(A, B) = norm(A - B) / norm(B)

function random_mesh(ns; seed = 7)
    D = length(ns)
    Ω = domain(D == 1 ? interval(0.0, 1.0) :
               foldl(×, ntuple(d -> interval(0.0, Float64(d)), D)))
    Bramble._seed_mesh1d_rng!(seed)
    try
        return D == 1 ? mesh(Ω, ns[1], false) : mesh(Ω, ns, ntuple(_ -> false, D))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

# The spelled-out sum the node replaces.
spelled(u::Bramble.LazyOp{D}) where {D} = foldl(+, ntuple(d -> Bramble.D̃(Bramble.D₋(u, Val(d)), Val(d)), D))

# Each pair: the form with `Δₕ`, the same form with `spelled`.
function bilinear_pairs(W)
    c = Rₕ(W, x -> 1.0 + sum(x) * x[1])
    S = spelled
    J = Bramble.jumpₓ
    return (
        ("trial", (u, v) -> innerₕ(Δₕ(u), v), (u, v) -> innerₕ(S(u), v)),
        ("test", (u, v) -> innerₕ(u, Δₕ(v)), (u, v) -> innerₕ(u, S(v))),
        ("c Δₕ Δₕ", (u, v) -> innerₕ(c * Δₕ(u), Δₕ(v)), (u, v) -> innerₕ(c * S(u), S(v))),
        ("jumpₓ∘Δₕ", (u, v) -> innerₕ(J(Δₕ(u)), J(Δₕ(v))),
            (u, v) -> innerₕ(J(S(u)), J(S(v)))),
        ("D₋ₓ∘Δₕ", (u, v) -> innerₕ(Bramble.D₋ₓ(Δₕ(u)), v),
            (u, v) -> innerₕ(Bramble.D₋ₓ(S(u)), v)),
        ("Dcₓ∘Δₕ", (u, v) -> innerₕ(Bramble.Dcₓ(Δₕ(u)), v),
            (u, v) -> innerₕ(Bramble.Dcₓ(S(u)), v)),
        ("Δₕ∘Δₕ", (u, v) -> innerₕ(Δₕ(Δₕ(u)), v), (u, v) -> innerₕ(S(S(u)), v)),
        ("Δₕ + u", (u, v) -> innerₕ(Δₕ(u) + 2.0 * u, v), (u, v) -> innerₕ(S(u) + 2.0 * u, v))
    )
end

function check_bilinear(W)
    for (name, f, g) in bilinear_pairs(W)
        A = assemble(form(W, W, f))
        B = assemble(form(W, W, g))
        @test norm(B) > 0
        ok = relerr(A, B) <= tol
        ok || @info "Δₕ differs from Σ D̃(D₋)" name relerr(A, B)
        @test ok
    end
end

function check_linear(W)
    D = Bramble.dim(mesh(W))
    for (f, g) in (
        (v -> innerₕ(x -> x[1]^2 + x[D], Δₕ(v)), v -> innerₕ(x -> x[1]^2 + x[D], spelled(v))),
        (v -> innerₕ(x -> exp(x[1]), Bramble.D₋ₓ(Δₕ(v))),
        v -> innerₕ(x -> exp(x[1]), Bramble.D₋ₓ(spelled(v)))))
        l1 = assemble(form(W, f))
        l2 = assemble(form(W, g))
        @test norm(l2) > 0
        @test relerr(l1, l2) <= tol
    end
end

# The runtime `Δₕ` of a grid function is a separate engine (`_accumulate_laplacian!`); with
# the diagonal `innerₕ` mass `M`, `A x = M Δₕ(x)` for any `x`.
function check_runtime(W)
    A = assemble(form(W, W, (u, v) -> innerₕ(Δₕ(u), v)))
    M = assemble(form(W, W, (u, v) -> innerₕ(u, v)))
    x = [sin(3.0i) + 0.1i for i in 1:ndofs(W)]
    Δx = parent(Δₕ(element(W, x)))
    @test norm(Δx) > 0
    @test relerr(A * x, M * Δx) <= tol
end

@testset "Δₕ in forms (#774)" begin
    @testset "One node per axis" begin
        for ns in ((6,), (5, 6))
            W = gridspace(random_mesh(ns))
            u = Bramble.trial_function(W)
            L = Δₕ(u)
            names = string(typeof(L))
            @test occursin("SecondDifference", names)
            @test !occursin("StarDifference", names)
            @test !occursin("BackwardDifference", names)
            if length(ns) == 1
                @test L isa Bramble.SecondDifference{1, 1}
            else
                @test L isa Bramble.OperatorAdd
                @test L.left_op isa Bramble.SecondDifference{2, 1}
                @test L.right_op isa Bramble.SecondDifference{2, 2}
            end
            @test Δₕ(Bramble.test_function(W)) isa Bramble.LazyOp
        end
    end

    @testset "Bilinear forms equal Σ D̃(D₋)" begin
        check_bilinear(gridspace(random_mesh((9,))))
        check_bilinear(gridspace(random_mesh((7, 8))))
        WITH_3D_TESTS && check_bilinear(gridspace(random_mesh((4, 5, 4))))
    end

    @testset "Linear forms equal Σ D̃(D₋)" begin
        check_linear(gridspace(random_mesh((9,))))
        check_linear(gridspace(random_mesh((7, 8))))
        WITH_3D_TESTS && check_linear(gridspace(random_mesh((4, 5, 4))))
    end

    @testset "Matches the runtime Δₕ" begin
        check_runtime(gridspace(random_mesh((9,))))
        check_runtime(gridspace(random_mesh((7, 8))))
        WITH_3D_TESTS && check_runtime(gridspace(random_mesh((4, 5, 4))))
    end

    @testset "Padded mesh reads its ghosts" begin
        G = Bramble.GhostMesh(random_mesh((8,)); ghosts = :boundary)
        P = Bramble._padded(G)
        W = gridspace(P)
        A = assemble(form(W, W, (u, v) -> innerₕ(Δₕ(u), v)))
        M = assemble(form(W, W, (u, v) -> innerₕ(u, v)))
        # row 2 is the first physical point; its backward tap is the ghost at column 1
        h1, h2 = Bramble.spacing(P, 2), Bramble.forward_spacing(P, 2)
        @test A[2, 1] / M[2, 2] ≈ 2 / ((h1 + h2) * h1) rtol = tol
        @test A[2, 3] / M[2, 2] ≈ 2 / ((h1 + h2) * h2) rtol = tol
        @test A[2, 2] / M[2, 2] ≈ -2 / (h1 * h2) rtol = tol
        check_runtime(W)
        check_bilinear(W)
    end
end

end # module FormLaplacianFormTests
