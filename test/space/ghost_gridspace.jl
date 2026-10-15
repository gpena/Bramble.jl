module SpaceGhostGridSpaceTests

using Test
using LinearAlgebra: norm, I
using SparseArrays: nnz, nonzeros
using Bramble
using Bramble: GhostGridSpace, padded_space, closure, physical_indices, is_absent, nghosts,
               change_points!, weights, Innerh, Innerplus, element, D₋ₓ, inner₊ₓ
using ..TestUtils: WITH_3D_TESTS

# A ghost space weighs only the physical nodes, cells and facets. The oracle of every check
# is the space without ghosts, `gridspace(Ωₕ)`, which shares no ghost code: the padded
# assembly reduced to the degrees of freedom, `Eᵀ A E`, must equal its matrix on the
# physical block and vanish on every ghost row and column.

# `Eᵀ A E` of a bilinear form `f` on the padded space of `Wg`.
function reduced(Wg, f)
    E, _ = closure(Wg)
    P = padded_space(Wg)
    return E' * assemble(form(P, P, f)) * E
end

# The bilinear form `f` on `Wg` equals that on `W` at the physical nodes, 0 at the ghosts.
function weighs_physical_only(W, Wg, f)
    K = assemble(form(W, W, f))
    Kr = reduced(Wg, f)
    n = ndofs(W)
    return norm(Kr[1:n, 1:n] - K) <= 1e-13 * norm(K) && iszero(Kr[(n + 1):end, :]) &&
           iszero(Kr[:, (n + 1):end])
end

# The same for the linear form `g`.
function loads_physical_only(W, Wg, g)
    E, _ = closure(Wg)
    b = assemble(form(W, g))
    br = E' * assemble(form(padded_space(Wg), g))
    n = ndofs(W)
    return norm(br[1:n] - b) <= 1e-13 * norm(b) && iszero(br[(n + 1):end])
end

# A weight family of the padded space is the physical one at the physical nodes and 0
# everywhere else, ghosts and absent points included.
function padded_weights_match(w_pad, w_phys, Wg)
    G = mesh(Wg)
    phys = physical_indices(G)
    full = reshape(Array(w_pad), npoints(G, Tuple))
    off = trues(size(full))
    off[phys] .= false
    return full[phys] == reshape(Array(w_phys), size(phys)) && all(iszero, full[off])
end

stiffness(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))
mass(u, v) = innerₕ(u, v)

@testset "Ghost grid space" begin
    @testset "1D, both ends and one end" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
        W = gridspace(Ωₕ)
        for (ghosts, ng) in ((:boundary, 2), ((:xmin,), 1), ((:xmax,), 1))
            Wg = gridspace(Ωₕ; ghosts = ghosts)
            @test Wg isa GhostGridSpace{1}
            @test ndofs(Wg) == 9 + ng
            @test weighs_physical_only(W, Wg, stiffness)
            @test weighs_physical_only(W, Wg, mass)
            @test loads_physical_only(W, Wg, v -> inner_Γ(x -> 2.0 + x[1], v;
                markers = (:boundary,)))
        end
    end

    @testset "2D weights, non-uniform mesh" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(-1.0, 2.0)), (6, 8), (false, false))
        x = collect(range(0.0, 1.0; length = 6)) .^ 2
        y = -1.0 .+ 3.0 .* collect(range(0.0, 1.0; length = 8)) .^ 3
        change_points!(Ωₕ, (x, y))
        @test !(diff(points(Ωₕ(1))) ≈ fill(0.2, 5)) && !(diff(points(Ωₕ(2))) ≈ fill(3 / 7, 7))
        W = gridspace(Ωₕ)
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        P = padded_space(Wg)
        @test padded_weights_match(weights(P, Innerh()), weights(W, Innerh()), Wg)
        for d in 1:2
            @test padded_weights_match(weights(P, Innerplus(), d), weights(W, Innerplus(), d), Wg)
        end
        @test padded_weights_match(weights(P, Val((1, 2))), weights(W, Val((1, 2))), Wg)
        @test weighs_physical_only(W, Wg, stiffness)
        @test weighs_physical_only(W, Wg, mass)
        @test weighs_physical_only(W, Wg, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
    end

    @testset "Facet weights through inner_Γ" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
        W = gridspace(Ωₕ)
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        for m in ((:xmin,), (:ymax,), (:xmax, :ymin), (:boundary,))
            @test loads_physical_only(W, Wg, v -> inner_Γ(x -> 1.0 + x[1] * x[2], v;
                markers = m))
        end
    end

    @testset "Named facets only" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
        W = gridspace(Ωₕ)
        Wg = gridspace(Ωₕ; ghosts = (:xmin, :ymax))
        @test ndofs(Wg) == 35 + 7 + 5
        @test npoints(mesh(Wg), Tuple) == (6, 8)
        @test weighs_physical_only(W, Wg, stiffness)
        @test weighs_physical_only(W, Wg, mass)
        @test loads_physical_only(W, Wg, v -> inner_Γ(x -> 1.0 + x[2], v;
            markers = (:xmax,)))
    end

    @testset "Closure selects each DOF once" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        G = mesh(Wg)
        E, d = closure(Wg)
        @test size(E) == (63, 59)
        @test d == zeros(63)
        @test nnz(E) == 59 && all(isone, nonzeros(E))
        @test E' * E == I
        rows = [findfirst(!iszero, E[:, k]) for k in 1:59]
        cis = CartesianIndices(npoints(G, Tuple))
        # physical nodes first, in the mesh's linear order
        @test cis[rows[1:35]] == vec(collect(physical_indices(G)))
        # then the ghosts: neither physical nor absent, xmin's first
        @test all(J -> !(J in physical_indices(G)) && !is_absent(G, J), cis[rows[36:end]])
        @test all(J -> J[1] == 1, cis[rows[36:42]])
        @test all(J -> J[1] == 7, cis[rows[43:49]])
    end

    @testset "Staleness after change_points!" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, true))
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        P = padded_space(Wg)
        x = collect(points(Ωₕ(1)))
        y = collect(points(Ωₕ(2)))
        change_points!(Ωₕ, (x .^ 2, y))
        @test_throws ArgumentError weights(P)
        msg = sprint(showerror, try
            weights(P)
        catch e
            e
        end)
        @test occursin("ghosts = ...", msg)
        xg = collect(points(mesh(Wg)(1)))
        @test xg[2:(end - 1)] == x .^ 2
        @test xg[1] ≈ x[1]^2 - (x[2]^2 - x[1]^2)
        W = gridspace(Ωₕ)
        Wn = gridspace(Ωₕ; ghosts = :boundary)
        @test padded_weights_match(weights(padded_space(Wn), Innerh()), weights(W, Innerh()), Wn)
        @test weighs_physical_only(W, Wn, stiffness)
    end

    @testset "Inference and today's space" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
        @test (@inferred gridspace(Ωₕ; ghosts = :boundary)) isa GhostGridSpace{2}
        @test (@inferred gridspace(Ωₕ; ghosts = (:xmin,))) isa GhostGridSpace{2}
        @test (@inferred gridspace(Ωₕ; ghosts = ())) isa Bramble.ScalarGridSpace
        @test typeof(gridspace(Ωₕ; ghosts = ())) == typeof(gridspace(Ωₕ))
    end

    @testset "Refusals" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
        @test_throws ArgumentError gridspace(Ωₕ; ghosts = "xmin")
        @test_throws ArgumentError gridspace(Ωₕ; ghosts = (:zmin,))
        @test_throws ArgumentError gridspace(Ωₕ; ghosts = (:nowhere,))
        G = mesh(gridspace(Ωₕ; ghosts = :boundary))
        @test_throws ArgumentError gridspace(G)
        @test_throws ArgumentError gridspace(G; ghosts = :boundary)
    end

    @testset "Restriction and cell average refuse" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        uₕ = element(Wg)
        f = x -> x[1]
        for (op, call) in (
            (:Rₕ, () -> Rₕ(Wg, f)),
            (:Rₕ, () -> Rₕ(Wg, f; markers = (:boundary,))),
            (:Rₕ, () -> Rₕ(Wg, (f,))),
            (:Rₕ!, () -> Rₕ!(uₕ, f)),
            (:Rₕ!, () -> Rₕ!(uₕ, f; markers = (:boundary,))),
            (:avgₕ, () -> avgₕ(Wg, f)),
            (:avgₕ, () -> avgₕ(Wg, f; quad_points = 3)),
            (:avgₕ!, () -> avgₕ!(uₕ, f)),
            (:avgₕ!, () -> avgₕ!(uₕ, f, Val(3))),
            (:avgₕ!, () -> avgₕ!(uₕ, f; markers = (:boundary,)))
        )
            e = try
                call()
                nothing
            catch err
                err
            end
            @test e isa ArgumentError
            msg = e isa ArgumentError ? sprint(showerror, e) : ""
            @test occursin(string(op), msg)
            @test occursin("gridspace(Ωₕ)", msg)
            @test occursin("Bramble.lift", msg)
        end
        # The plain space is untouched.
        @test Rₕ(gridspace(Ωₕ), f) isa Bramble.VectorElement
        @test avgₕ(gridspace(Ωₕ), f) isa Bramble.VectorElement
    end

    @testset "element and its method guard" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        @test length(element(Wg)) == ndofs(Wg)
        @test !any(m -> occursin("GhostGridSpace", string(m.sig)), methods(element))
    end

    WITH_3D_TESTS && @testset "3D weights" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 2.0)),
            (4, 5, 6), (false, false, false))
        W = gridspace(Ωₕ)
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        @test ndofs(Wg) == 120 + 2 * (30 + 24 + 20)
        @test weighs_physical_only(W, Wg, stiffness)
        @test weighs_physical_only(W, Wg, mass)
        @test loads_physical_only(W, Wg, v -> inner_Γ(x -> 1.0 + x[3], v;
            markers = (:zmax, :xmin)))
        Wz = gridspace(Ωₕ; ghosts = (:zmin,))
        @test ndofs(Wz) == 120 + 20
        @test weighs_physical_only(W, Wz, stiffness)
    end
end

end # module SpaceGhostGridSpaceTests
