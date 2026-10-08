module TestFormKroneckerProjection

using Test
using Bramble
using Bramble: D₊ₓ, D₋ₓ, D₋ᵧ, D₊ᵧ, Dc, Mₓ, M₊ᵧ, Mcₓ, jumpₓ, jumpᵧ, S₊ₓ, S₋ᵧ, inner₊ₓ,
               inner₊ᵧ, restrict_to
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, spzeros

# `_kron_project` (gpena/Bramble.jl#427): one addend of a resolved form, projected onto
# each axis of a tensor mesh, gives 1D factors whose summed `kron`s are what `assemble`
# builds for that addend. Every check compares against `assemble(a)` on a graded mesh,
# where no two axes share their nodes, so a factor put on the wrong axis or built with the
# wrong spacing shows.

function _proj_graded_space(n::NTuple{D, Int}) where {D}
    Ω = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
    Ωₕ = mesh(Ω, n, ntuple(_ -> false, D))
    pts = ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D)
    Bramble.change_points!(Ωₕ, pts)
    return gridspace(Ωₕ)
end

_proj_leaves(a) = Bramble._kron_leaves(Bramble.resolve_form_ast(a), ())

# The coefficient-weighted sum of every leaf's Kronecker products, or `nothing` when some
# leaf is refused.
function _proj_sum(a)
    Ωₕ = mesh(Bramble.trial_space(a))
    n = Bramble.ndofs(Bramble.trial_space(a))
    B = spzeros(n, n)
    for (scales, term) in _proj_leaves(a)
        P = Bramble._kron_project(term, Ωₕ)
        P === nothing && return nothing
        for (live, factors) in P
            @assert all(F -> F isa SparseMatrixCSC, factors)
            B += Bramble._kron_coeff((scales..., live...)) * foldl(kron, reverse(factors))
        end
    end
    return B
end

function _proj_matches(a)
    A = assemble(a)
    B = _proj_sum(a)
    B === nothing && return false
    return maximum(abs, B - A; init = 0.0) <= 1e-13 * max(maximum(abs, nonzeros(A)), 1.0)
end

function _proj_refused(a)
    Ωₕ = mesh(Bramble.trial_space(a))
    return any(l -> Bramble._kron_project(l[2], Ωₕ) === nothing, _proj_leaves(a))
end

@testset "Kronecker projection (#427)" begin
    @testset "families match assemble" begin
        # Any: each space is its own type
        for W in Any[_proj_graded_space((9, 7)), _proj_graded_space((6, 5, 7))]
            forms = Any[  # Any: each closure is its own type
                (u, v) -> innerₕ(D₊ₓ(u), D₊ₓ(v)),
                (u, v) -> innerₕ(Dc(u, Val(1)), Dc(v, Val(1))),
                (u, v) -> innerₕ(Mₓ(u), Mₓ(v)),
                (u, v) -> innerₕ(M₊ᵧ(u), Mcₓ(v)),
                (u, v) -> innerₕ(jumpₓ(u), jumpᵧ(v)),
                (u, v) -> innerₕ(S₊ₓ(u), S₋ᵧ(v)),
                (u, v) -> innerₕ(D₋ₓ(Mₓ(u)), D₋ₓ(Mₓ(v))),
                (u, v) -> innerₕ(D₋ₓ(D₋ᵧ(u)), v),
                (u, v) -> innerₕ(D₋ₓ(u), D₊ᵧ(v)),
                (u, v) -> innerₕ(D₋ₓ(u), v) + innerₕ(u, v),
                (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v)),
                (u, v) -> inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(D₋ₓ(v))),
                (u, v) -> innerₕ(restrict_to(:interior, u), v),
                (u, v) -> innerₕ(restrict_to(:interior, D₋ₓ(u)), D₋ᵧ(v))
            ]
            for f in forms
                @test _proj_matches(form(W, W, f))
            end
        end
    end

    @testset "a sum inside a side distributes" begin
        W = _proj_graded_space((9, 7))
        a = form(W, W, (u, v) -> innerₕ(D₋ₓ(u) + D₋ᵧ(u), v + jumpₓ(v)))
        terms = [Bramble._kron_project(t, mesh(W)) for (_, t) in _proj_leaves(a)]
        @test sum(length, terms) == 4
        @test _proj_matches(a)
    end

    @testset "untensored terms are refused" begin
        W = _proj_graded_space((9, 7))
        g = Rₕ(W, x -> x[1] + x[2])
        @test _proj_refused(form(W, W, (u, v) -> innerₕ(g * u, v)))
        @test _proj_refused(form(W, W, (u, v) -> innerₕ(restrict_to(:boundary, u), v)))
        @test !_proj_refused(form(W, W, (u, v) -> innerₕ(u, v)))

        # A 1D mesh is not a `MeshnD`: nothing to factor.
        W1 = gridspace(mesh(domain(interval(0.0, 1.0)), 9, false))
        a1 = form(W1, W1, (u, v) -> innerₕ(u, v))
        @test _proj_refused(a1)
    end

    # A node, shift or `inner₊` weight along an axis the mesh does not have would meet no
    # axis and project to the identity; `assemble` throws on it, and the projection refuses.
    @testset "axes the mesh lacks are refused" begin
        W = _proj_graded_space((7, 6))
        Ωₕ = mesh(W)
        u, v = Bramble.TrialFunction{2, 1}(), Bramble.TestFunction{2, 1}()
        for node in (Bramble.BackwardDifference{2, 3, typeof(u)}(u),
            Bramble.CenteredAverage{2, 3, typeof(u)}(u),
            Bramble.ShiftNode{2, 3, typeof(u)}(1, u))
            @test Bramble._kron_split(node) === nothing
            @test Bramble._kron_project(innerₕ(node, v), Ωₕ) === nothing
        end
        @test Bramble._kron_inners(Bramble.InnerPlus{3}, Ωₕ) === nothing
        @test Bramble._kron_inners(Bramble.InnerPlus{2}, Ωₕ) !== nothing
        @test _proj_refused(form(W, W, (u, v) -> innerₕ(Bramble.D₋₂(u), v)))
        @test_throws BoundsError assemble(form(W, W, (u, v) -> innerₕ(Bramble.D₋₂(u), v)))
    end

    # A plain number scaling a node inside a side goes into the axis-1 chain; a `Ref` there
    # is returned beside the factors, since a factor would read it once.
    @testset "a number inside a side" begin
        W = _proj_graded_space((9, 7))
        a = form(W, W, (u, v) -> innerₕ(D₋ₓ(u), v) + 0.3 * innerₕ(u, v))
        @test any(l -> occursin("OperatorScale", string(typeof(l[2]))), _proj_leaves(a))
        @test _proj_matches(a)
        c = Ref(0.3)
        a = form(W, W, (u, v) -> innerₕ(D₋ₓ(u), v) + c * innerₕ(u, v))
        @test _proj_matches(a)
        P = Bramble._kron_project(only(_proj_leaves(a))[2], mesh(W))
        @test map(first, P) == ((), (c,))
    end

    @testset "inner_Γ is a sum over its faces" begin
        W2 = _proj_graded_space((9, 7))
        W3 = _proj_graded_space((6, 5, 7))
        cases = Any[  # Any: each space and marker set is its own type
            (W2, :xmin, 1), (W2, :ymax, 1), (W2, (:xmin, :ymin), 2), (W2, :boundary, 4),
            (W3, :zmax, 1), (W3, (:xmax, :ymin), 2), (W3, (:xmin, :ymax, :zmin), 3),
            (W3, :boundary, 6)]
        for (W, mk, nfaces) in cases
            a = form(W, W, (u, v) -> inner_Γ(u, v; markers = mk))
            P = Bramble._kron_project(only(_proj_leaves(a))[2], mesh(W))
            @test length(P) == nfaces
            @test _proj_matches(a)
            @test _proj_matches(form(W, W, (u, v) -> inner_Γ(D₋ₓ(u), D₊ᵧ(v); markers = mk)))
            @test _proj_matches(form(W, W,
                (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + 1.5 * inner_Γ(u, v; markers = mk)))
        end
        # The face mask is geometric: a domain-redefined `:xmin` does not move it.
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωc = mesh(domain(I2, :xmin => x -> x[2] < 0.5), (6, 5), (false, true))
        @test count(markers(Ωc)[:xmin]) == 12  # every point with y < 0.5, not the face
        Wc = gridspace(Ωc)
        @test _proj_matches(form(Wc, Wc, (u, v) -> inner_Γ(u, v; markers = :xmin)))
        # On a one-point axis both faces are one point, weighed once; two terms double it.
        W1 = gridspace(mesh(domain(I2), (1, 5), (true, true)))
        @test _proj_matches(form(W1, W1, (u, v) -> inner_Γ(u, v; markers = :xmin)))
        @test _proj_refused(form(W1, W1, (u, v) -> inner_Γ(u, v; markers = :boundary)))
    end

    @testset "single-axis coefficients factor" begin
        # Any: each space is its own type
        for W in Any[_proj_graded_space((9, 7)), _proj_graded_space((6, 5, 7))]
            fx = Rₕ(W, x -> 1 + x[1])
            fy = Rₕ(W, x -> 2 + x[2]^2)
            fl = Rₕ(W, x -> 1 + x[end]^3)  # the last axis: z in 3D
            c = Rₕ(W, x -> 3.0)            # varies along no axis
            forms = Any[  # Any: each closure is its own type
                (u, v) -> innerₕ(fx * (fy * u), v) + inner₊ₓ(fy * D₋ₓ(u), D₋ₓ(v)),
                (u, v) -> innerₕ(D₋ᵧ(fy * u), fx * D₊ₓ(v)),
                (u, v) -> innerₕ(D₋ₓ(fx * (fx * u)), Mₓ(fl * v)),
                (u, v) -> inner₊ᵧ(fl * D₋ᵧ(u), D₋ᵧ(fx * v)),
                (u, v) -> innerₕ(c * (D₋ₓ(u) + fl * D₋ᵧ(u)), v),
                (u, v) -> innerₕ(restrict_to(:interior, fy * u), fx * v),
                (u, v) -> innerₕ(u, v) + inner_Γ(fy * u, fx * v; markers = :boundary),
                (u, v) -> innerₕ(Vector(fl) * u, v),
                (u, v) -> innerₕ((() -> 2.5) * u, v)
            ]
            for f in forms
                @test _proj_matches(form(W, W, f))
            end
        end
        W = _proj_graded_space((9, 7))
        fy = Rₕ(W, x -> 2 + x[2]^2)
        # The coefficient sits where the D-dimensional one sits: `D₋ᵧ(fy * u)` on axis 2.
        a = form(W, W, (u, v) -> innerₕ(D₋ᵧ(fy * u), v))
        P = Bramble._kron_project(only(_proj_leaves(a))[2], mesh(W))
        Wy = gridspace(mesh(W)(2))
        gy = Rₕ(Wy, x -> 2 + x[1]^2)
        @test only(P)[2][2] == assemble(form(Wy, Wy, (u, v) -> innerₕ(D₋ₓ(gy * u), v)))
    end

    @testset "rank-2 or foreign coefficient" begin
        W = _proj_graded_space((9, 7))
        # One ulp off at one point is no longer a single-axis coefficient.
        g = Rₕ(W, x -> 1 + x[1])
        parent(g)[12] = nextfloat(parent(g)[12])
        @test _proj_refused(form(W, W, (u, v) -> innerₕ(g * u, v)))
        @test _proj_refused(form(W, W, (u, v) -> innerₕ(u, D₋ₓ(g * v))))
        # A coefficient on another mesh of the same size.
        Wo = _proj_graded_space((9, 7))
        go = Rₕ(Wo, x -> 1 + x[1])
        @test _proj_refused(form(W, W, (u, v) -> innerₕ(go * u, v)))
        # A plain vector of the wrong length.
        @test _proj_refused(form(W, W, (u, v) -> innerₕ(ones(5) * u, v)))
    end

    @testset "custom :interior marker refused" begin
        # A domain-redefined `:interior` marker is no product of the axes' interiors, and
        # `restrict_to(:interior, …)` reads the mesh's own marker.
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        for (mk, unif) in ((x -> x[1] < 0.5, (false, false)), (:left, (false, true)))
            Ωc = @test_logs (:warn,) match_mode=:any mesh(
                domain(I2, :interior => mk), (6, 5), unif)
            Wc = gridspace(Ωc)
            r = (u, v) -> innerₕ(restrict_to(:interior, u), v)
            rd = (u, v) -> innerₕ(D₋ₓ(restrict_to(:interior, u)), v)
            @test _proj_refused(form(Wc, Wc, r))
            @test _proj_refused(form(Wc, Wc, rd))
            # No restriction in the term: the marker does not matter.
            @test _proj_matches(form(Wc, Wc, (u, v) -> innerₕ(D₋ₓ(u), v)))
        end
        # The default marker still projects.
        Wd = gridspace(mesh(domain(I2), (6, 5), (false, true)))
        @test _proj_matches(form(Wd, Wd, (u, v) -> innerₕ(restrict_to(:interior, u), v)))
    end
end

end # module
