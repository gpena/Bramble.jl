using Test
using Bramble
using Bramble: values, components, ndofs, _grid_dims, _op_mesh

# Operators on composite grid functions.
#
# A composite grid function is a stack of scalar ones sharing a mesh, so every operator
# must give the same answer as applying it to each component on its own. It did not: the
# applicators took their grid shape from `ndofs(space, Tuple)`, which is the grid shape
# for a scalar space but the per-component dof counts for a composite one. A 3-component
# 4x6 space therefore addressed prod((24, 24, 24)) = 13824 slots into a vector holding
# 72, which the engines write with @inbounds. Under `--check-bounds=yes`, which is how the
# suite runs, that is a BoundsError; without it, it segfaults.
#
# These tests pin the invariant rather than the symptom, so they hold whatever the
# internals do later.

@testset "Operators on composite grid functions" begin
    scalar_ops = (
        ("diff₋", diff₋ₓ, diff₋ᵧ, diff₋₂),
        ("diff₊", diff₊ₓ, diff₊ᵧ, diff₊₂),
        ("D₋", D₋ₓ, D₋ᵧ, D₋₂),
        ("D₊", D₊ₓ, D₊ᵧ, D₊₂),
        ("jump", jumpₓ, jumpᵧ, jump₂),
        ("M₋", M₋ₓ, M₋ᵧ, M₋₂),
        ("M₊", M₊ₓ, M₊ᵧ, M₊₂)
    )

    meshes = (
        ("1D", mesh(domain(interval(0.0, 1.0)), 7, false), 1),
        ("2D",
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 6), (true, false)),
            2),
        ("3D",
            mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 4, 5),
                (true, false, true)),
            3)
    )

    # A distinct, non-symmetric function per component, so a component being written into
    # the wrong slot cannot pass by coincidence.
    component_fn(k, D) = D == 1 ? (x -> sin(k * x) + k) : (x -> sin(k * x[1]) + k * x[end])

    @testset "the shape handed to the engines is the grid, not the dof count" begin
        # The direct regression guard on the cause.
        for (lbl, Ωₕ, D) in meshes
            Wₕ = gridspace(Ωₕ)
            for NC in (1, 2, 3)
                Vₕ = gridspace(Ωₕ, Val(NC))
                uₕ = Rₕ(Vₕ, ntuple(k -> component_fn(k, D), NC))
                cs = components(uₕ)
                @test _grid_dims(cs[1]) == npoints(Ωₕ, Tuple)
                # every component holds exactly one value per grid point
                @test all(length(values(c)) == prod(npoints(Ωₕ, Tuple)) for c in cs)
                @test length(values(uₕ)) == NC * prod(npoints(Ωₕ, Tuple))
            end
        end
    end

    @testset "componentwise equality" begin
        for (lbl, Ωₕ, D) in meshes
            @testset "$lbl" begin
                Wₕ = gridspace(Ωₕ)
                for NC in (2, 3)
                    Vₕ = gridspace(Ωₕ, Val(NC))
                    fs = ntuple(k -> component_fn(k, D), NC)
                    uₕ = Rₕ(Vₕ, fs)

                    # the scalar grid functions the components should behave like
                    scalars = ntuple(k -> Rₕ(Wₕ, fs[k]), NC)
                    @test all(values(components(uₕ)[k]) == values(scalars[k]) for k in 1:NC)

                    for (name, ops...) in scalar_ops, d in 1:D

                        op = ops[d]
                        rₕ = op(uₕ)
                        @test length(values(rₕ)) == length(values(uₕ))
                        for k in 1:NC
                            @test values(components(rₕ)[k]) == values(op(scalars[k]))
                        end
                    end
                end
            end
        end
    end

    @testset "vectorial forms" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 5), (true, false))
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(2))
        fs = (x -> sin(x[1]) + x[2], x -> 2x[1] * x[2])
        uₕ = Rₕ(Vₕ, fs)
        scalars = (Rₕ(Wₕ, fs[1]), Rₕ(Wₕ, fs[2]))

        for (vec_op, scalar_ops_pair) in ((∇₋ₕ, (D₋ₓ, D₋ᵧ)), (∇₊ₕ, (D₊ₓ, D₊ᵧ)),
            (diff₋ₕ, (diff₋ₓ, diff₋ᵧ)), (jumpₕ, (jumpₓ, jumpᵧ)),
            (M₋ₕ, (M₋ₓ, M₋ᵧ)))
            g = vec_op(uₕ)
            @test length(g) == 2
            for d in 1:2, k in 1:2

                @test values(components(g[d])[k]) ==
                      values(scalar_ops_pair[d](scalars[k]))
            end
        end
    end

    @testset "a one-component composite matches the scalar space" begin
        # NC == 1 is the boundary between the two dispatches and must not be special.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
        Wₕ = gridspace(Ωₕ)
        V₁ = gridspace(Ωₕ, Val(1))
        f = x -> exp(-x) * sin(3x)
        uₕ = Rₕ(V₁, (f,))
        sₕ = Rₕ(Wₕ, f)
        for op in (diff₋ₓ, diff₊ₓ, D₋ₓ, D₊ₓ, jumpₓ, M₋ₓ, M₊ₓ)
            @test values(op(uₕ)) == values(op(sₕ))
        end
    end

    @testset "the result is a new element, not a view of the input" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
        Vₕ = gridspace(Ωₕ, Val(2))
        uₕ = Rₕ(Vₕ, (x -> x, x -> x^2))
        before = copy(values(uₕ))
        rₕ = D₋ₓ(uₕ)
        rₕ[1] = -1234.0
        @test values(uₕ) == before
    end
end
