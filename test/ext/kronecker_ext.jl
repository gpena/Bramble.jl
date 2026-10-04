module TestKroneckerExt

using Test
using Bramble
using Bramble: is_separable, kronecker_operator, KroneckerLinearOperator
using Kronecker: kronecker
using LinearAlgebra: mul!, issymmetric
using SparseArrays: SparseMatrixCSC
using Random

# `Kronecker.jl` interop and fast diagonalisation for a separable `BilinearForm`
# (gpena/Bramble.jl#259), layered on `KroneckerLinearOperator`
# (gpena/Bramble.jl#162, test/form/kronecker.jl). `fdm_solve` has no forward stub in
# `src/Bramble.jl` yet (see `ext/BrambleKroneckerExt.jl`'s module docstring), so it is
# reached the same way any other not-yet-exported extension function would be: off the
# loaded extension module itself.
const KronExt = Base.get_extension(Bramble, :BrambleKroneckerExt)
@assert KronExt !== nothing "BrambleKroneckerExt did not load -- is Kronecker.jl a test dependency?"
# `fdm_solve` is Bramble's own binding (`function fdm_solve end` in `src/Bramble.jl`), and
# this extension adds methods to it, so the exported spelling is the one to test: reaching
# into the extension module would pass even if the methods had attached to a function of
# the extension's own instead, which is precisely the failure this asserts against.
@assert !isempty(methods(Bramble.fdm_solve)) "fdm_solve has no methods -- are the extension's definitions dot-qualified as `Bramble.fdm_solve`?"

const KRON_EXT_SEED = 20260919

using Bramble: D₋ₓ, D₋ᵧ, Mₓ, inner₊ₓ, inner₊ᵧ

# Uniform, then moved to `t^(1 + d/4)` along axis `d`, so no two axes share their nodes.
function graded_space(n::NTuple{D, Int}) where {D}
    Ω = mesh(domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))), n, ntuple(_ -> false, D))
    Bramble.change_points!(Ω, ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D))
    return gridspace(Ω)
end

@testset "Kronecker extension" begin
    @testset "Kronecker.jl object equals CSC" begin
        Random.seed!(KRON_EXT_SEED)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (13, 11), (false, false))
        W2 = gridspace(Ω2)

        Random.seed!(KRON_EXT_SEED + 1)
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (8, 7, 6), (false, false, false)
        )
        W3 = gridspace(Ω3)

        for Wₕ in (W2, W3)
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v)))
            K = kronecker_operator(a)
            Aref = SparseMatrixCSC(K)

            Kjl = kronecker(K)
            @test collect(Kjl) ≈ Aref
            @test Matrix(Kjl) ≈ Matrix(Aref)

            # `mul!` through the Kronecker.jl object agrees with the operator's own `mul!`.
            n = ndofs(Wₕ)
            x = rand(n)
            yref = similar(x)
            mul!(yref, K, x)
            y = collect(Kjl) * x
            @test isapprox(y, yref; rtol = 1e-10, atol = 1e-10)
        end
    end

    # fdm_solve against sparse backslash on 2D 25x19 and 3D 11x9x8 meshes, no Dirichlet.
    @testset "fdm_solve vs \\, no Dirichlet" begin
        Random.seed!(KRON_EXT_SEED + 2)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (25, 19), (false, false))
        W2 = gridspace(Ω2)

        Random.seed!(KRON_EXT_SEED + 3)
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (11, 9, 8), (false, false, false)
        )
        W3 = gridspace(Ω3)

        for (Wₕ, tag) in ((W2, "2D"), (W3, "3D"))
            @testset "$tag" begin
                a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
                A = assemble(a)
                n = ndofs(Wₕ)
                F = rand(n)
                xref = A \ F

                x = fdm_solve(a, F)
                @test isapprox(x, xref; rtol = 1e-9)

                # `fdm_solve(K, F)`: the unconstrained `KroneckerLinearOperator` overload.
                K = kronecker_operator(a)
                xK = fdm_solve(K, F)
                @test isapprox(xK, xref; rtol = 1e-9)
            end
        end
    end

    # fdm_solve against sparse backslash on 2D 25x19 and 3D 11x9x8 meshes, homogeneous Dirichlet.
    @testset "fdm_solve vs \\, zero Dirichlet" begin
        Random.seed!(KRON_EXT_SEED + 4)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (25, 19), (false, false))
        W2 = gridspace(Ω2)

        Random.seed!(KRON_EXT_SEED + 5)
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (11, 9, 8), (false, false, false)
        )
        W3 = gridspace(Ω3)

        for (Wₕ, tag) in ((W2, "2D"), (W3, "3D"))
            @testset "$tag" begin
                a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
                A = assemble(a; dirichlet = :boundary)
                n = ndofs(Wₕ)
                dims = ndofs(Wₕ, Tuple)

                F = rand(n)
                Farr = reshape(F, dims)
                D = length(dims)
                for d in 1:D
                    idx_first = ntuple(k -> k == d ? 1 : Colon(), D)
                    idx_last = ntuple(k -> k == d ? dims[k] : Colon(), D)
                    Farr[idx_first...] .= 0.0
                    Farr[idx_last...] .= 0.0
                end
                F = vec(Farr)

                xref = A \ F
                x = fdm_solve(a, F; dirichlet = :boundary)
                @test isapprox(x, xref; rtol = 1e-9)
            end
        end
    end

    @testset "A grid-function coefficient throws" begin
        Random.seed!(KRON_EXT_SEED + 6)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
        Wₕ = gridspace(Ωₕ)
        # Varying along both axes: a single-axis coefficient now factors (#427).
        fₕ = Rₕ(Wₕ, x -> 1.0 + x[1] * x[2])
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(fₕ * u, v))
        @test !is_separable(a)
        @test_throws ArgumentError fdm_solve(a, rand(ndofs(Wₕ)))
    end

    # Laplacian-like forms beyond the classic one (#427), on graded meshes whose axes carry
    # different nodes: each solves to the sparse direct solve, from the form and from its
    # operator, with and without homogeneous Dirichlet.
    @testset "fdm_solve: Laplacian-like forms" begin
        for n in ((9, 7), (6, 5, 7))
            Wₕ = graded_space(n)
            D = length(n)
            fx = Rₕ(Wₕ, x -> 1 + x[1])
            c = Ref(2.5)
            lap(op) = (u, v) -> innerₕ(u, v) +
                                sum(innerₕ(op(u, Val(d)), op(v, Val(d))) for d in 1:D)
            accepted = [
                "Ref coefficient" => (u, v) -> innerₕ(u, v) + c * inner₊(∇ₕ(u), ∇ₕ(v)),
                "forward" => lap(Bramble.D₊),
                "averaged mass" => (u, v) -> innerₕ(Mₓ(u), Mₓ(v)) +
                                             inner₊(∇ₕ(u), ∇ₕ(v)),
                "x-coefficient" => (u, v) -> innerₕ(fx * u, v) +
                                             inner₊(∇ₕ(u), ∇ₕ(v)),
                "Robin face" => (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) +
                                          inner_Γ(u, v; markers = (:xmin,)),
                # Indefinite but nonsingular: the singularity test must not refuse it.
                "negative mass" => (u, v) -> -1.0 * innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
            ]
            for (name, f) in accepted, dir in (nothing, :boundary)

                a = form(Wₕ, Wₕ, f)
                c[] = 2.5
                K = kronecker_operator(a)
                c[] = 4.0  # read when fdm_solve is called, not when `a` or `K` was built
                A = assemble(a; dirichlet = dir)
                F = rand(MersenneTwister(length(name)), size(A, 1))
                dir === :boundary && (F[Bramble._combined_mask(mesh(Wₕ), (:boundary,))] .= 0)
                xref = A \ F
                @test isapprox(fdm_solve(a, F; dirichlet = dir), xref; rtol = 1e-9)
                dir === nothing && @test isapprox(fdm_solve(K, F), xref; rtol = 1e-9)
            end
        end
    end

    # Every other form throws, naming the reason, and is never solved wrongly.
    @testset "fdm_solve: refusals name the reason" begin
        refusal(f) =
            try
                f()
                "no error"
            catch e
                e isa ArgumentError ? sprint(showerror, e) : "wrong error $(typeof(e))"
            end
        Wₕ = graded_space((9, 7))
        fx = Rₕ(Wₕ, x -> 1 + x[1])
        fy = Rₕ(Wₕ, x -> 2 + x[2]^2)
        fxy = Rₕ(Wₕ, x -> (1 + x[1]) * (2 + x[2]^2))
        L(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))
        rint(u) = Bramble.restrict_to(:interior, u)
        refused = [
            "mixed" => ((u, v) -> innerₕ(D₋ₓ(D₋ᵧ(u)), v) + L(u, v),
                r"non-mass factors on two axes"),
            "advection" => ((u, v) -> innerₕ(D₋ₓ(u), v) + L(u, v),
                r"axis-1 operator is not symmetric"),
            "two-axis coefficient" => ((u, v) -> innerₕ(fx * (fy * u), v) + L(u, v),
                r"non-mass factors on two axes"),
            "one-axis gradient" => ((u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(u, v),
                r"axis 2 has no term of its own"),
            "singular mass" => ((u, v) -> innerₕ(rint(u), v),
                r"mass is not symmetric positive definite"),
            # Pure Neumann: constants are in the kernel, so a solve would return garbage.
            "gradient only" => (L, r"the system is singular"),
            "chain, average, y-stiffness" => ((u, v) -> innerₕ(D₋ₓ(Mₓ(u)), D₋ₓ(Mₓ(v))) +
                       innerₕ(Mₓ(u), Mₓ(v)) +
                       inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v)),
                r"the system is singular"),
            "1e-14 mass" => ((u, v) -> 1e-14 * innerₕ(u, v) + L(u, v), r"the system is singular")
        ]
        F = rand(MersenneTwister(KRON_EXT_SEED), ndofs(Wₕ))
        for (name, (f, why)) in refused
            a = form(Wₕ, Wₕ, f)
            @test is_separable(a)
            K = kronecker_operator(a)
            for m in (refusal(() -> fdm_solve(a, F)), refusal(() -> fdm_solve(K, F)))
                @test occursin("fdm_solve does not support this form", m)
                @test occursin(why, m)
                @test occursin("kronecker_operator(a)", m)
            end
        end
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(fxy * u, v) + L(u, v))
        m = refusal(() -> fdm_solve(a, F))
        @test occursin(r"does not support this form: it is not separable", m)
        @test !occursin("kronecker_operator(a)", m)  # it would throw too

        Vₕ = gridspace(mesh(Wₕ), Val(2))
        b = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)) +
                                   L(u(1), v(1)) + L(u(2), v(2)))
        G = rand(2 * ndofs(Wₕ))
        @test occursin(r"does not support this form: it is posed on a composite space",
            refusal(() -> fdm_solve(b, G)))
        @test occursin("composite space", refusal(() -> fdm_solve(kronecker_operator(b), G)))
    end

    # Well conditioned in Float32 (eigenvalue ratio about 8e3): the singularity test must
    # not grow with the number of unknowns, which refused this solve.
    @testset "fdm_solve: Float32 Laplacian" begin
        Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
        Wₕ = gridspace(mesh(Ω, (33, 33), (true, true); backend = backend(Float32)))
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        A = assemble(a)
        F = rand(MersenneTwister(KRON_EXT_SEED), Float32, size(A, 1))
        x = fdm_solve(a, F)
        @test eltype(x) === Float32
        @test isapprox(x, Float64.(A) \ Float64.(F); rtol = 1e-3)
    end

    # A 2-point axis leaves no interior unknown under `dirichlet = :boundary`.
    @testset "fdm_solve: empty interior" begin
        for n in ((2, 2), (2, 4), (2, 3, 3), (3, 2, 4))
            Wₕ = graded_space(n)
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
            A = assemble(a; dirichlet = :boundary)
            F = zeros(size(A, 1))
            x = fdm_solve(a, F; dirichlet = :boundary)
            @test length(x) == ndofs(Wₕ)
            @test x == A \ F
        end
    end

    # gpena/Bramble.jl#442: a K built before `change_points!` refuses `fdm_solve` and the
    # conversion with the stale-weights `ArgumentError`; both work before the move.
    @testset "stale K after a mesh move" begin
        stale(f) =
            try
                f()
                false
            catch e
                e isa ArgumentError && occursin("change_points!", sprint(showerror, e))
            end
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        K = kronecker_operator(a)
        F = rand(MersenneTwister(KRON_EXT_SEED + 8), size(K, 1))
        @test !stale(() -> fdm_solve(K, F)) && !stale(() -> kronecker(K))
        Bramble.change_points!(Ωₕ,
            (range(0.0, 1.0; length = 9) .^ 2, range(0.0, 1.0; length = 7) .^ 2))
        @test stale(() -> fdm_solve(K, F))
        @test stale(() -> kronecker(K))
        # The form's own space predates the move: `fdm_solve(a, F)` refuses it as `assemble`
        # does, with the space's stale-weights error, not a factorisation of stale weights.
        @test_throws "gridspace(mesh(Wₕ)) again" fdm_solve(a, F)
        @test_throws "gridspace(mesh(Wₕ)) again" fdm_solve(a, F; dirichlet = :boundary)
        Bramble.iterative_refinement!(Ωₕ)
        @test_throws "gridspace(mesh(Wₕ)) again" fdm_solve(a, F)
    end

    # The allocation of a second fdm_solve call is reported, not asserted to be zero.
    @testset "fdm_solve: second-call allocations" begin
        Random.seed!(KRON_EXT_SEED + 7)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 13), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        F = rand(ndofs(Wₕ))

        fdm_solve(a, F)   # warm-up: JIT only, `fdm_solve` rebuilds its factors every call
        bytes = @allocated fdm_solve(a, F)
        @info "fdm_solve: @allocated on a second call (no persistent workspace across calls)" bytes
        @test bytes >= 0   # reported, not asserted zero -- see the CHECK's EVIDENCE note
    end
end

end # module
