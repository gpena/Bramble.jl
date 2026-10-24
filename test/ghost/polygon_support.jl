module GhostPolygonSupportTests

# The polygon-space operations of #882 that work and had no test, on a jittered convex
# octagon (four oblique edges of slope ±1, the convex case the other polygon meshes lack)
# and on the jittered Figure 1 mesh: an upwind form, matrix-free against `A * x`, ILU- and
# AMG-preconditioned GMRES against `A \ F`, `interpolate_at` and `export_vtk` of a lifted
# solution, `semidiscretize` with the test space on both sides, and a Float32 solve. Then
# #901's level-set "test space as trial side" row, over levelset_spaces.jl's meshes.

using Test
using Random
using SparseArrays
using LinearAlgebra: I as Id, diag, isdiag, ldiv!, mul!, norm
using Bramble
import Bramble
using Bramble: lift, polygon, polygon_spaces, D₋ₓ, D₋ᵧ, Dcₓ, Dcᵧ
using AlgebraicMultigrid: aspreconditioner
using ILUZero: ILUZero
using LinearSolve: KrylovJL_GMRES, LinearProblem, solve
using SciMLBase: ReturnCode
using WriteVTK: WriteVTK

const OCTAGON = [(1.0, 0.0), (3.0, 0.0), (4.0, 1.0), (4.0, 3.0), (3.0, 4.0), (1.0, 4.0),
    (0.0, 3.0), (0.0, 1.0)]
const FIG1 = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0),
    (4.0, 5.0), (2.0, 5.0), (0.0, 3.0)]

function polygon_mesh(V, box, n, seed)
    Random.seed!(seed)
    return mesh(domain(box, :P => polygon(V)), n, (false, false))
end
octagon_mesh(T = Float64) = polygon_mesh([T.(v) for v in OCTAGON],
    interval(T(-1), T(5)) × interval(T(-1), T(5)), (19, 19), 2)
fig1_mesh() = polygon_mesh(FIG1, interval(0.0, 6.0) × interval(0.0, 5.0), (10, 9), 2)

# (name, mesh), both jittered.
const MESHES = (("octagon", octagon_mesh()), ("Figure 1", fig1_mesh()))

# Separable-additive (u_xy = 0), so (3.2) is exact, with −Δu = −4.
uq(x) = x[1]^2 + x[2]^2 + x[1] + 2x[2]
mixed(w) = Dcₓ(Dcᵧ(w)) + Dcᵧ(Dcₓ(w))
upwind(w, v) = innerₕ(-1.0 * Δₕ(w) + D₋ₓ(w) + D₋ᵧ(w), v)

labels(Ωₕ) = (Bramble.index_in_marker(Ωₕ, :P), Bramble.index_in_marker(Ωₕ, :P_boundary))

relerr(x, y) = norm(x - y) / norm(y)

# The reduced system from the plain space's matrix and load, sharing no closure code: rows
# and columns of the interior nodes, the boundary columns times `g` to the load, and each
# exterior column `p = i + (sx, sy)` of row `i` as `+1` on `i + (sx, 0)` and `i + (0, sy)`
# and `−1` on `i`; any other exterior read is an error.
function oracle(Ωₕ, f_a, f_l, g)
    Wp = gridspace(Ωₕ)
    Ap = Matrix(assemble(form(Wp, Wp, f_a)))
    Fp = assemble(form(Wp, f_l))
    I, Γ = labels(Ωₕ)
    car = CartesianIndices(Bramble.npoints(Ωₕ, Tuple))
    lin = LinearIndices(car)
    gv = [Γ[k] ? g(Bramble.point(Ωₕ, car[k])) : 0.0 for k in eachindex(I)]
    B = copy(Ap)
    for i in findall(I), p in findall(.!(I .| Γ))

        a = Ap[i, p]
        iszero(a) && continue
        sx, sy = Tuple(car[p] - car[i])
        @assert abs(sx) == 1 && abs(sy) == 1
        B[i, p] = 0
        B[i, lin[car[i] + CartesianIndex(sx, 0)]] += a
        B[i, lin[car[i] + CartesianIndex(0, sy)]] += a
        B[i, i] -= a
    end
    return B[I, I], Fp[I] .- B[I, Γ] * gv[Γ]
end

# The bilinear blend of `u` (one value per node) on the cell of `Ωₕ` holding `p`, by hand.
function bilinear(Ωₕ, u, p)
    n = Bramble.npoints(Ωₕ, Tuple)
    xs = [Bramble.point(Ωₕ, (i, 1))[1] for i in 1:n[1]]
    ys = [Bramble.point(Ωₕ, (1, j))[2] for j in 1:n[2]]
    i, j = searchsortedlast(xs, p[1]), searchsortedlast(ys, p[2])
    s, t = (p[1] - xs[i]) / (xs[i + 1] - xs[i]), (p[2] - ys[j]) / (ys[j + 1] - ys[j])
    U = reshape(u, n)
    return (1 - s) * (1 - t) * U[i, j] + s * (1 - t) * U[i + 1, j] +
           (1 - s) * t * U[i, j + 1] + s * t * U[i + 1, j + 1]
end

apply(P, x) = ldiv!(similar(x), P, x)

@testset "Polygon support (#882)" begin
    @testset "$name: oblique edges extend nodes" for (name, Ωₕ) in MESHES
        U, _ = polygon_spaces(Ωₕ, :P, uq)
        @test count(U.rows.extended) > 0
        @test !any(U.rows.conflict)
    end

    @testset "$name: upwind D₋ₓ(w) + D₋ᵧ(w)" for (name, Ωₕ) in MESHES
        g(x) = 1 + x[1] - 2x[2] + x[1] * x[2]
        f_l = v -> innerₕ(x -> sin(x[1]) + x[2], v)
        U, W = polygon_spaces(Ωₕ, :P, g)
        a = form(U, W, upwind)
        A, F = assemble(a, form(W, f_l))
        Aoracle, Foracle = oracle(Ωₕ, upwind, f_l, g)
        @test maximum(abs, Matrix(A) .- Aoracle) < 1e-12 * maximum(abs, Aoracle)
        @test maximum(abs, F .- Foracle) < 1e-12 * maximum(abs, Foracle)
        # The scheme is exact for a linear u, for which D₋ₓ u + D₋ᵧ u = 1 − 2.
        ul(x) = 1 + x[1] - 2x[2]
        UL, WL = polygon_spaces(Ωₕ, :P, ul)
        AL, FL = assemble(form(UL, WL, upwind), form(WL, v -> innerₕ(x -> -1.0, v)))
        I, _ = labels(Ωₕ)
        @test maximum(abs, AL \ FL .- Rₕ(gridspace(Ωₕ), ul).data[I]) < 1e-10
        x = rand(Xoshiro(1), size(A, 2))
        z = zeros(size(A, 1))
        mul!(z, Bramble.matrix_free_operator(a), x)
        @test z ≈ A * x
    end

    @testset "$name: ILU and AMG preconditioned GMRES" for (name, Ωₕ) in MESHES
        U, W = polygon_spaces(Ωₕ, :P, uq)
        a = form(U, W, (w, v) -> upwind(w, v) - 0.4 * innerₕ(mixed(w), v))
        A, F = assemble(a, form(W, v -> innerₕ(x -> 1.0 + x[1], v)))
        u = A \ F
        x = rand(Xoshiro(2), size(A, 1))
        Pi = ilu_preconditioner(a)
        @test Pi isa ILUZero.ILU0Precon
        @test apply(Pi, x) == apply(ilu_preconditioner(A), x)
        Pa = aspreconditioner(amg_preconditioner(a))
        @test apply(Pa, x) == apply(aspreconditioner(amg_preconditioner(A)), x)
        for P in (Pi, Pa)
            s = solve(LinearProblem(A, F), KrylovJL_GMRES(precs = (_, _) -> (P, Id));
                reltol = 1e-12, abstol = 0.0)
            @test s.retcode == ReturnCode.Success
            @test relerr(s.u, u) <= 1e-8
        end
    end

    @testset "$name: interpolate_at of lift(U, u)" for (name, Ωₕ) in MESHES
        U, W = polygon_spaces(Ωₕ, :P, uq)
        A, F = assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.4 * mixed(w), v)),
            form(W, v -> innerₕ(x -> -4.0, v)))
        uₕ = lift(U, A \ F)
        p = (2.03, 1.71)
        I, _ = labels(Ωₕ)
        @test interpolate_at(uₕ, p) ≈ bilinear(Ωₕ, uₕ.data, p) rtol = 1e-12
        # The nodal values are the quadratic's, so the blend is the blend of its values.
        uex = Rₕ(gridspace(Ωₕ), uq).data
        @test abs(interpolate_at(uₕ, p) - bilinear(Ωₕ, uex, p)) < 1e-9
        k = findfirst(I)
        @test interpolate_at(uₕ, Bramble.point(Ωₕ, CartesianIndices(
            Bramble.npoints(Ωₕ, Tuple))[k])) ≈ uₕ.data[k]
    end

    @testset "$name: export_vtk of lift(U, u)" for (name, Ωₕ) in MESHES
        U, W = polygon_spaces(Ωₕ, :P, uq)
        A, F = assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w), v)),
            form(W, v -> innerₕ(x -> -4.0, v)))
        uₕ = lift(U, A \ F)
        n = Bramble.npoints(Ωₕ, Tuple)
        mktempdir() do dir
            f = only(export_vtk(joinpath(dir, "u"), uₕ))
            @test endswith(f, ".vtr") && isfile(f)
            # The rectilinear grid of the whole box.
            @test occursin("WholeExtent=\"0 $(n[1] - 1) 0 $(n[2] - 1) 0 0\"", read(f, String))
        end
        # The values handed to WriteVTK, the exterior extension included.
        @test Bramble._vtk_data(uₕ) == reshape(uₕ.data, n)
        @test count(!iszero, Bramble._vtk_data(uₕ)[U.rows.extended]) > 0
    end

    @testset "$name: semidiscretize on (W, W)" for (name, Ωₕ) in MESHES
        _, W = polygon_spaces(Ωₕ, :P, x -> 0.0)
        f_a = (w, v) -> innerₕ(Δₕ(w), v)
        f_l = v -> innerₕ(x -> 1.0 + x[2], v)
        sd = semidiscretize(form(W, W, f_a), form(W, f_l))
        @test sd isa Bramble.Semidiscretization
        @test sd.operator_matrix == assemble(form(W, W, f_a))
        # The default mass is innerₕ(u, v): the dual-cell areas on the diagonal.
        M = sd.mass_matrix
        @test isdiag(M)
        @test diag(M) ≈ assemble(form(W, v -> innerₕ(x -> 1.0, v))) rtol = 1e-14
    end

    @testset "Float32 keeps its eltype" begin
        Ωₕ = octagon_mesh(Float32)
        U, W = polygon_spaces(Ωₕ, :P, uq)
        A, F = assemble(form(U, W, (w, v) -> innerₕ(-1.0f0 * Δₕ(w) - 0.4f0 * mixed(w), v)),
            form(W, v -> innerₕ(x -> -4.0f0, v)))
        @test eltype(A) == Float32 && eltype(F) == Float32
        u = A \ F
        @test eltype(u) == Float32
        uₕ = lift(U, u)
        @test eltype(uₕ.data) == Float32
        I, _ = labels(Ωₕ)
        @test maximum(abs, u .- Rₕ(gridspace(Ωₕ), uq).data[I]) <= 1e-3
    end
end

# #901: levelset_spaces.jl's meshes, with its builder.
const BOX_DISK = interval(-1.25, 1.25) × interval(-1.25, 1.25)
const BOX_EGG = interval(-1.35, 1.35) × interval(-1.35, 1.35)

disk(p) = hypot(p[1], p[2]) - 1

# r = 1 + 0.12 cos θ + 0.05 cos 2θ − 0.06 exp(−(Δθ/0.25)²), Δθ the angle to the
# diagonal direction 3π/4 (wrapped), the whole curve rotated by 10°.
function dented_egg(p)
    θ = atan(p[2], p[1]) - deg2rad(10)
    Δ = mod(θ - 3π / 4 + π, 2π) - π
    return hypot(p[1], p[2]) - (1 + 0.12 * cos(θ) + 0.05 * cos(2θ) - 0.06 * exp(-(Δ / 0.25)^2))
end

function levelset_mesh(φ, box, n, seed)
    Random.seed!(seed)
    return mesh(domain(box, :D => levelset(φ)), (n, n), (false, false))
end

const LEVELSET_MESHES = (("disk", levelset_mesh(disk, BOX_DISK, 20, 2)),
    ("dented egg", levelset_mesh(dented_egg, BOX_EGG, 40, 3)))

@testset "Level-set support (#901)" begin
    @testset "$name: test space as trial side" for (name, Ωₕ) in LEVELSET_MESHES
        # It gives every exterior node 0, so a form reading one throws; a form reading the
        # boundary nodes reads them as 0, the trial space with g = 0.
        U0, W = polygon_spaces(Ωₕ, :D, x -> 0.0)
        @test count(U0.rows.extended) > 0
        @test_throws ArgumentError form(W, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - mixed(w), v))
        f_a = (w, v) -> innerₕ(-1.0 * Δₕ(w), v)
        @test assemble(form(W, W, f_a)) == assemble(form(U0, W, f_a))
    end
end

end # module
