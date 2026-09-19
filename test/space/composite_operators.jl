module SpaceCompositeOperatorsTests

using Test
using Bramble
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
import Bramble: diff₋ₓ, diff₋ᵧ, diff₋₂, diff₋ₕ, diff₊ₓ, diff₊ᵧ, diff₊₂, D₊ₓ, D₊ᵧ, D₊₂, ∇₊ₕ, M₊ₓ, M₊ᵧ, M₊₂
using Bramble: components, ndofs, _grid_dims, _op_mesh
using Bramble: diff₋ₓ, diff₋ᵧ, diff₋₂, diff₊ₓ, diff₊ᵧ, diff₊₂, diff₋ₕ
# εₕ/εₕ! (gpena/Bramble.jl#234, S6.7): new names, not yet exported -- the integrator adds
# `export εₕ, εₕ!` to src/Bramble.jl alongside divₕ/curlₕ/Δₕ.
import Bramble: εₕ, εₕ!
using ..TestUtils: alloc_test, @test_allocs

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

@testset "Composite operators" begin
    scalar_ops = (
        ("diff₋", diff₋ₓ, diff₋ᵧ, diff₋₂),
        ("diff₊", diff₊ₓ, diff₊ᵧ, diff₊₂),
        ("D₋", D₋ₓ, D₋ᵧ, D₋₂),
        ("D₊", D₊ₓ, D₊ᵧ, D₊₂),
        ("jump", jumpₓ, jumpᵧ, jump₂),
        ("M", Mₓ, Mᵧ, M₂),
        ("M₊", M₊ₓ, M₊ᵧ, M₊₂)
    )

    meshes = (
        ("1D", mesh(domain(interval(0.0, 1.0)), 7, false), 1),
        (
            "2D",
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 6), (true, false)),
            2
        ),
        (
            "3D",
            mesh(
                domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
                (3, 4, 5),
                (true, false, true)
            ),
            3
        )
    )

    # A distinct, non-symmetric function per component, so a component being written into
    # the wrong slot cannot pass by coincidence.
    component_fn(k, D) = D == 1 ? (x -> sin(k * x) + k) : (x -> sin(k * x[1]) + k * x[end])

    @testset "Grid shape vs dof count" begin
        # The direct regression guard on the cause.
        for (lbl, Ωₕ, D) in meshes
            Wₕ = gridspace(Ωₕ)
            for NC in (1, 2, 3)
                Vₕ = gridspace(Ωₕ, Val(NC))
                uₕ = Rₕ(Vₕ, ntuple(k -> component_fn(k, D), NC))
                cs = components(uₕ)
                @test _grid_dims(cs[1]) == npoints(Ωₕ, Tuple)
                # every component holds exactly one value per grid point
                @test all(length(parent(c)) == prod(npoints(Ωₕ, Tuple)) for c in cs)
                @test length(parent(uₕ)) == NC * prod(npoints(Ωₕ, Tuple))
            end
        end
    end

    @testset "Componentwise equality" begin
        for (lbl, Ωₕ, D) in meshes
            @testset "$lbl" begin
                Wₕ = gridspace(Ωₕ)
                for NC in (2, 3)
                    Vₕ = gridspace(Ωₕ, Val(NC))
                    fs = ntuple(k -> component_fn(k, D), NC)
                    uₕ = Rₕ(Vₕ, fs)

                    # the scalar grid functions the components should behave like
                    scalars = ntuple(k -> Rₕ(Wₕ, fs[k]), NC)
                    @test all(parent(components(uₕ)[k]) == parent(scalars[k]) for k in 1:NC)

                    for (name, ops...) in scalar_ops, d in 1:D

                        op = ops[d]
                        rₕ = op(uₕ)
                        @test length(parent(rₕ)) == length(parent(uₕ))
                        for k in 1:NC
                            @test parent(components(rₕ)[k]) == parent(op(scalars[k]))
                        end
                    end
                end
            end
        end
    end

    @testset "Vectorial forms" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 5), (true, false))
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(2))
        fs = (x -> sin(x[1]) + x[2], x -> 2x[1] * x[2])
        uₕ = Rₕ(Vₕ, fs)
        scalars = (Rₕ(Wₕ, fs[1]), Rₕ(Wₕ, fs[2]))

        for (vec_op, scalar_ops_pair) in (
            (∇ₕ, (D₋ₓ, D₋ᵧ)),
            (∇₊ₕ, (D₊ₓ, D₊ᵧ)),
            (diff₋ₕ, (diff₋ₓ, diff₋ᵧ)),
            (jumpₕ, (jumpₓ, jumpᵧ)),
            (Mₕ, (Mₓ, Mᵧ))
        )
            g = vec_op(uₕ)
            @test length(g) == 2
            for d in 1:2, k in 1:2

                @test parent(components(g[d])[k]) == parent(scalar_ops_pair[d](scalars[k]))
            end
        end
    end

    @testset "One-component composite" begin
        # NC == 1 is the boundary between the two dispatches and must not be special.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
        Wₕ = gridspace(Ωₕ)
        V₁ = gridspace(Ωₕ, Val(1))
        f = x -> exp(-x) * sin(3x)
        uₕ = Rₕ(V₁, (f,))
        sₕ = Rₕ(Wₕ, f)
        for op in (diff₋ₓ, diff₊ₓ, D₋ₓ, D₊ₓ, jumpₓ, Mₓ, M₊ₓ)
            @test parent(op(uₕ)) == parent(op(sₕ))
        end
    end

    @testset "New element allocation" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
        Vₕ = gridspace(Ωₕ, Val(2))
        uₕ = Rₕ(Vₕ, (x -> x, x -> x^2))
        before = copy(parent(uₕ))
        rₕ = D₋ₓ(uₕ)
        rₕ[1] = -1234.0
        @test parent(uₕ) == before
    end

    # gpena/Bramble.jl#234 (v3.3.0 plan S6.7): ∇ₕ, εₕ and divₕ over a *vector field* -- a
    # `D`-leaf composite VectorElement on a `D`-dimensional mesh -- rather than the
    # arbitrary-leaf-count multi-field composites the testsets above exercise.
    @testset "Vector calculus over composites (gpena/Bramble.jl#234 S6.7)" begin
        Dm = (D₋ₓ, D₋ᵧ, D₋₂)
        Mm = (Mₓ, Mᵧ, M₂)

        vc_cases = (
            (
                "2D",
                mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (true, true)),
                2,
                (x -> sin(x[1]) + x[2]^2, x -> cos(x[2]) + x[1]^2)
            ),
            (
                "3D",
                mesh(
                    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 4, 6),
                    (true, true, true)
                ),
                3,
                (
                    x -> sin(x[1]) + x[2] * x[3],
                    x -> cos(x[2]) + x[1] * x[3],
                    x -> sin(x[3]) + x[1] * x[2]
                )
            )
        )

        for (lbl, Ωₕ, D, fs) in vc_cases
            @testset "$lbl" begin
                Wₕ = gridspace(Ωₕ)
                Vₕ = gridspace(Ωₕ, Val(D))
                uₕ = Rₕ(Vₕ, fs)
                # The independent oracle: each component built as its own scalar grid
                # function and differenced/averaged directly, never through `∇ₕ`/`εₕ`
                # applied to the composite.
                scalars = ntuple(k -> Rₕ(Wₕ, fs[k]), D)

                @testset "∇ₕ gradient tensor against a scalar D₋ oracle" begin
                    g = ∇ₕ(uₕ)
                    @test length(g) == D
                    for i in 1:D, j in 1:D
                        @test parent(components(g[i])[j]) == parent(Dm[i](scalars[j]))
                    end
                end

                @testset "εₕ against a hand-built symmetrised oracle" begin
                    ε = εₕ(uₕ)
                    for i in 1:D
                        @test parent(ε[i][i]) == parent(Dm[i](scalars[i]))
                    end
                    for i in 1:D, j in 1:D
                        i == j && continue
                        oracle = 0.5 .* (parent(Mm[i](Dm[j](scalars[i]))) .+
                                  parent(Mm[j](Dm[i](scalars[j]))))
                        @test parent(ε[i][j]) ≈ oracle
                        @test parent(ε[i][j]) == parent(ε[j][i]) # symmetry
                    end
                end

                @testset "εₕ! agrees with εₕ and allocates nothing" begin
                    dest = ntuple(_ -> ntuple(_ -> similar(first(scalars)), Val(D)), Val(D))
                    εₕ!(dest, uₕ)
                    ε = εₕ(uₕ)
                    for i in 1:D, j in 1:D
                        @test parent(dest[i][j]) == parent(ε[i][j])
                    end
                    @test_allocs εₕ!(dest, uₕ)
                end

                @testset "divₕ already covers the composite vector field (#158)" begin
                    # Independent oracle: the plain, unstaggered sum of D₋ᵢ over each
                    # scalar leaf -- exactly what divₕ (#158) already computes, and all
                    # this subplan's goal asks of it. No extension was needed.
                    oracle = mapreduce(k -> parent(Dm[k](scalars[k])), +, 1:D)
                    @test parent(divₕ(uₕ)) ≈ oracle
                end
            end
        end

        @testset "Rejected inputs" begin
            Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
            W2 = gridspace(Ω2)
            V3 = gridspace(Ω2, Val(3))

            # a non-composite (scalar) element on a 2D mesh: arity 1 != 2
            scalar_uₕ = Rₕ(W2, x -> x[1] + x[2])
            @test_throws DimensionMismatch εₕ(scalar_uₕ)

            # a composite whose leaf count (3) differs from the mesh dimension (2)
            wrong_uₕ = Rₕ(V3, (x -> x[1], x -> x[2], x -> x[1] + x[2]))
            @test_throws DimensionMismatch εₕ(wrong_uₕ)
        end
    end
end

end # module SpaceCompositeOperatorsTests
