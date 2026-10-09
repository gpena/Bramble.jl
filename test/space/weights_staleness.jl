module SpaceWeightsStalenessTests

using Test
using ForwardDiff
using Bramble
using Bramble: norm₊
using Bramble: change_points!, set_points!, weights
using ..TestUtils: alloc_test, @test_allocs

# `ScalarGridSpace` precomputes its inner-product weights once from the mesh at
# `gridspace(Ωₕ)` time. The mesh is mutable (`set_points!`, `change_points!`, `iterative_refinement!` all
# rewrite it in place), so a space built before such a mutation must not silently compute
# against weights for a mesh that no longer exists. Every check here is against an independent reference -- a
# freshly built `gridspace(Ωₕ)` after the mutation, or a hand-computed exact integral --
# never against another call to the code under test.

# The meshes the two moved-source-mesh testsets build, and the in-place move they apply:
# `m1` is 1D, `m2` is 2D with one non-uniform axis, `stretch!` squares the points.
m1(n) = mesh(domain(interval(0.0, 1.0)), n, true)
m2(n) = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n + 1),
    (true, false))
function stretch!(M)
    dim(M) == 1 &&
        return change_points!(M, collect(range(0.0, 1.0; length = npoints(M))) .^ 2)
    n1, n2 = npoints(M, Tuple)
    return change_points!(M, (collect(range(0.0, 1.0; length = n1)) .^ 2,
        collect(range(0.0, 1.0; length = n2))))
end

@testset "Grid space weights staleness (#221)" begin
    # change_points! gives exact known numbers.
    @testset "Reproducer: issue's exact numbers" begin
        Ω = domain(interval(0.0, 1.0))
        Ωₕ = mesh(Ω, 5, true)
        Wₕ = gridspace(Ωₕ)

        change_points!(Ωₕ, markers(Ω), [0.0, 0.1, 0.2, 0.3, 1.0])

        # that this throws at all is "All three mutators trigger staleness"'s; what is
        # reproduced here is the numbers a fresh space gives instead.
        Wₕ_fresh = gridspace(Ωₕ)
        @test weights(Wₕ_fresh).innerh ≈ [0.05, 0.1, 0.1, 0.4, 0.35]

        u = Rₕ(Wₕ_fresh, x -> x[1]^2)
        v = Rₕ(Wₕ_fresh, x -> 1.0)
        # Hand-computed against the weights [0.05, 0.1, 0.1, 0.4, 0.35] and point values
        # [0, 0.01, 0.04, 0.09, 1.0] checked above -- not a quadrature-accuracy claim (5
        # points on a grid this skewed, [0.3, 1.0] is one cell, is coarse by design, to
        # keep the reproducer's own grid), only that innerₕ on the rebuilt space computes
        # the discrete sum it is defined to, independent of `innerₕ`/`weights` themselves.
        @test innerₕ(u, v) ≈
              0.05 * 0.0 + 0.1 * 0.01 + 0.1 * 0.04 + 0.4 * 0.09 + 0.35 * 1.0
    end

    @testset "All three mutators trigger staleness" begin
        Ω = domain(interval(0.0, 1.0))

        @testset "change_points!" begin
            Ωₕ = mesh(Ω, 11, true)
            Wₕ = gridspace(Ωₕ)
            change_points!(Ωₕ, markers(Ω), collect(range(0.0, 1.0; length = 11)))
            @test_throws ArgumentError weights(Wₕ)
        end

        @testset "set_points!" begin
            Ωₕ = mesh(Ω, 11, true)
            Wₕ = gridspace(Ωₕ)
            set_points!(Ωₕ, collect(range(0.0, 1.0; length = 11)))
            @test_throws ArgumentError weights(Wₕ)
        end

        @testset "iterative_refinement!" begin
            Ωₕ = mesh(Ω, 5, true)
            Wₕ = gridspace(Ωₕ)
            iterative_refinement!(Ωₕ)
            @test_throws ArgumentError weights(Wₕ)
        end
    end

    # Not only weights() itself.
    @testset "Every guarded entry point throws" begin
        Ω = domain(interval(0.0, 1.0))
        Ωₕ = mesh(Ω, 11, true)
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> x[1])
        v = Rₕ(Wₕ, x -> 1.0)

        change_points!(Ωₕ, markers(Ω), collect(range(0.0, 1.0; length = 11)))

        @test_throws ArgumentError innerₕ(u, v)
        @test_throws ArgumentError inner₊(u, v)
        @test_throws ArgumentError normₕ(u)
        @test_throws ArgumentError norm₊(u)
        @test_throws ArgumentError norm₁ₕ(u)
        @test_throws ArgumentError snorm₁ₕ(u)
    end

    # 2D: mutating one submesh stales the whole space.
    @testset "MeshnD: one submesh stales the space" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωₕ = mesh(domain(S), (7, 7), (true, true))
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> 1.0)

        # Mutating only the x-axis submesh, directly -- not through the parent's own
        # change_points! -- still invalidates the composite mesh version, since it is
        # derived from the submeshes' own (`_mesh_version(Ωₕ::MeshnD) = sum(...)`), not
        # tracked separately.
        change_points!(Ωₕ(1), markers(interval(0.0, 1.0)), collect(range(0.0, 1.0; length = 7)))

        @test_throws ArgumentError weights(Wₕ)
        @test_throws ArgumentError innerₕ(u, u)

        Wₕ_fresh = gridspace(Ωₕ)
        @test innerₕ(Rₕ(Wₕ_fresh, x -> 1.0), Rₕ(Wₕ_fresh, x -> 1.0)) ≈ 1.0
    end

    @testset "3D (MeshnD)" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωₕ = mesh(domain(S), (5, 5, 5), (true, true, true))
        Wₕ = gridspace(Ωₕ)

        change_points!(Ωₕ(3), markers(interval(0.0, 1.0)), collect(range(0.0, 1.0; length = 5)))

        @test_throws ArgumentError weights(Wₕ)
    end

    # CompositeGridSpace: staleness in one leaf is caught, never silently ignored.
    @testset "Composite: stale leaf is caught" begin
        Ω = domain(interval(0.0, 1.0))
        Ωₕ = mesh(Ω, 9, true)
        Wₕ = gridspace(Ωₕ)
        W_comp = Wₕ × Wₕ

        u = Rₕ(W_comp, (x -> 1.0, x -> 1.0))

        change_points!(Ωₕ, markers(Ω), collect(range(0.0, 1.0; length = 9)))

        # Both leaves share `Ωₕ`, so both are stale; innerₕ recurses leaf by leaf and the
        # first one it reaches already throws.
        @test_throws ArgumentError innerₕ(u, u)
    end

    # No throw.
    @testset "Rebuilt space works after mutation" begin
        Ω = domain(interval(0.0, 1.0))
        Ωₕ = mesh(Ω, 21, true)
        Wₕ = gridspace(Ωₕ)
        n_before = ndofs(Wₕ)   # captured before mutating: mesh(Wₕ) is Ωₕ itself, not a copy
        iterative_refinement!(Ωₕ)

        Wₕ_fresh = gridspace(Ωₕ)
        u = Rₕ(Wₕ_fresh, x -> x[1])
        v = Rₕ(Wₕ_fresh, x -> 1.0)
        @test innerₕ(u, v) ≈ 0.5 atol = 1.0e-3
        @test ndofs(Wₕ_fresh) == 2 * n_before - 1
    end

    @testset "Fresh success path: zero allocation" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 1001, true)
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> x[1]^2)
        v = Rₕ(Wₕ, x -> 1.0)

        @test_allocs innerₕ(u, v)
        @test_allocs normₕ(u)
        @test_allocs weights(Wₕ)
    end

    # An interpolation's source leaf is read through `weights` when the form binds it, so a
    # host refill whose source mesh has moved throws instead of returning numbers for the
    # old mesh. The moved mesh is only ever an interpolation source:
    # no term walks it natively, since a walked stale leaf already throws on its own.
    @testset "Interpolation source mesh moved" begin
        cases = (
            ("trial πₕ scalar", (Ws, Wt) -> (Ws, Wt), (u, v) -> innerₕ(πₕ(u), v)),
            ("test πₕ scalar", (Ws, Wt) -> (Wt, Ws), (u, v) -> innerₕ(u, πₕ(v))),
            ("trial πₕ composite", (Ws, Wt) -> (Ws × Wt, Wt × Wt),
                (U, V) -> innerₕ(πₕ(U(1)), V(1)) + innerₕ(U(2), V(2))),
            ("test πₕ composite", (Ws, Wt) -> (Wt × Wt, Ws × Wt),
                (U, V) -> innerₕ(U(1), πₕ(V(1))) + innerₕ(U(2), V(2)))
        )
        for (mk, ns, nt) in ((m1, 7, 11), (m2, 4, 6)), (name, spaces, f) in cases

            @testset "$name, $(dim(mk(3)))D" begin
                Ms, Mt = mk(ns), mk(nt)
                F = form(spaces(gridspace(Ms), gridspace(Mt))..., f)
                A = assemble(F)
                @test alloc_test(assemble!, A, F) == 0
                stretch!(Ms)
                @test_throws ArgumentError assemble!(A, F)
            end
        end
    end

    # A linear form's `πₕ(uₕ)` source is read at every fill, not sampled once when the form
    # is built: a refill after new values in `uₕ` matches a fresh
    # form, and one after `uₕ`'s mesh has moved throws instead of interpolating on it, as do
    # `interpolate_at` and `πₕ!` themselves. The source mesh is only ever read through
    # `interpolate_at`; the walked mesh is never moved.
    @testset "Interpolated source mesh moved" begin
        g(x) = sum(abs2, x) + 1
        h(x) = 100 * first(x) + 3
        for (mk, x) in ((m1, 0.3), (m2, (0.3, 0.4))), kind in (:scalar, :component)

            @testset "$kind, $(dim(mk(3)))D" begin
                Ms, Mt = mk(5), mk(8)
                Ws, Wt = gridspace(Ms), gridspace(Mt)
                U = kind === :scalar ? Rₕ(Ws, g) : Rₕ(Ws × Ws, (h, g))
                uₕ = kind === :scalar ? U : components(U)[2]
                L = form(Wt, v -> innerₕ(πₕ(uₕ), v))
                b = assemble(L)
                @test alloc_test(assemble!, b, L) == 0

                parent(U) .*= 2
                assemble!(b, L)
                @test b ≈ assemble(form(Wt, v -> innerₕ(πₕ(uₕ), v)))
                @test b ≈ assemble(form(Wt, v -> innerₕ(πₕ(Wt, uₕ), v)))

                stretch!(Ms)
                @test_throws ArgumentError assemble!(b, L)
                @test_throws ArgumentError assemble(L)
                @test_throws ArgumentError interpolate_at(uₕ, x)
                @test_throws ArgumentError πₕ!(element(Wt), uₕ)
                @test_throws ArgumentError πₕ(Wt, uₕ)
            end
        end
    end

    # Unsampled, the interpolant's element type is read off `uₕ`, the walked mesh and the
    # fill, never probed at a point: the walked mesh's middle point lies outside `uₕ`'s mesh
    # here, where a probe would answer with the fill's type.
    @testset "Interpolated source element type" begin
        Ms = mesh(domain(interval(0.0, 0.4)), 11, true)
        Ws = gridspace(Ms)
        Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 21, true))
        g = s -> begin
            uₛ = element(Ws, typeof(s))
            parent(uₛ) .= [s * x^2 for x in points(Ms)]
            assemble(form(Wₕ, v -> innerₕ(πₕ(uₛ; outside = 0.0), v)))
        end
        @test !iszero(g(1.0))
        @test ForwardDiff.derivative(g, 2.0) ≈ g(1.0) rtol = 1.0e-12

        W32 = gridspace(mesh(domain(interval(0.0f0, 1.0f0)), 21, true))
        u64 = Rₕ(Ws, x -> 1 / 3 + x[1])
        @test eltype(assemble(form(W32, v -> innerₕ(πₕ(u64; outside = 0.0f0), v)))) ===
              Float64
    end

    # A fill of another type than the values (`outside = 0` against `Float64`) is converted
    # to the blend's type, so the refill evaluating it at every point stays at 0 bytes.
    @testset "Interpolated source with an integer fill" begin
        for (Ma, Mb, f) in (
            (mesh(domain(interval(0.0, 1.0)), 9, true),
            mesh(domain(interval(0.0, 1.0)), 65, true), x -> sin(x[1])),
            (mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, true)),
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (33, 33),
                (true, true)), x -> x[1] * x[2]))
            Wa, Wb = gridspace(Ma), gridspace(Mb)
            ua = Rₕ(Wa, f)
            L = form(Wb, v -> innerₕ(πₕ(ua; outside = 0), v))
            b = assemble(L)
            @test eltype(b) === Float64
            @test b ≈ assemble(form(Wb, v -> innerₕ(πₕ(ua; outside = 0.0), v)))
            @test alloc_test(assemble!, b, L) == 0
        end
    end

    # The walk checks a leaf's weights once, where it starts (`_bind_walk`), and the stencil
    # then reads them unchecked at every point (gpena/Bramble.jl#437). Every entry a user
    # reaches must still throw the #221 error once the walked mesh has moved, not assemble
    # silently against the old weights: a scalar form, and a composite whose second leaf
    # alone moved, so the check runs per leaf and not only on the first. The point count is
    # unchanged by the move, so nothing else in the walk notices it. `CpuPolyester` needs
    # `BramblePolyesterExt`, and a later file (test/space/inner_product.jl) asserts Polyester
    # is not loaded in this process, so its cases run in a child process that loads it.
    #
    # The cross-mesh cases put the trial leaf on the moved mesh `Ωb` and the test leaf on a
    # third mesh `Ωc` of the same shape. The walk visits the test leaf only, so the trial
    # leaf needs its own check (gpena/Bramble.jl#466). They are bilinear only. Each mesh has
    # its own local name, since assigning an outer name inside the closure would rebind it
    # and both leaves would then go stale together.
    @testset "Assembly entry points throw when stale" begin
        probe = """
        using Bramble, Random
        using Bramble: backend, change_points!, matrix_free_operator, D₋ₓ, D₋ᵧ, inner₊ₓ, inner₊ᵧ
        using LinearAlgebra: mul!
        function stale_probe(policy)
            out = Pair{String, String}[]
            function run!(name, f)
                r = try
                    f()
                    "OK"
                catch e
                    msg = e isa ArgumentError ? e.msg : ""
                    occursin("weights were computed from its mesh before an in-place", msg) ?
                    "STALE" : "OTHER " * first(sprint(showerror, e), 200)
                end
                push!(out, name => replace(r, '\\n' => ' '))
            end
            Random.seed!(437)
            sq = interval(0.0, 1.0) × interval(0.0, 1.0)
            Ωa = mesh(domain(sq), (9, 8), (false, false); backend = backend(policy = policy))
            Ωb = mesh(domain(sq), (7, 10), (false, false); backend = backend(policy = policy))
            Ωc = mesh(domain(sq), (7, 10), (false, false); backend = backend(policy = policy))
            Ωb !== Ωc || error("Ωb and Ωc alias one mesh")
            Wa, Wb, Wc = gridspace(Ωa), gridspace(Ωb), gridspace(Ωc)
            f = x -> x[1] + 2x[2]
            # (name, trial, test, bilinear, linear or `nothing`)
            cases = (
                ("scalar", Wb, Wb, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(u, v),
                    v -> innerₕ(f, v)),
                ("composite", Wa × Wb, Wa × Wb,
                    (U, V) -> innerₕ(U(1), V(1)) + inner₊ᵧ(D₋ᵧ(U(2)), D₋ᵧ(V(2))),
                    V -> innerₕ(f, V(1)) + innerₕ(f, V(2))),
                ("cross-mesh innerₕ", Wb, Wc, (u, v) -> innerₕ(u, v), nothing),
                ("cross-mesh inner₊ₓ", Wb, Wc, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)), nothing),
                ("cross-mesh composite", Wc × Wb, Wc × Wc,
                    (U, V) -> innerₕ(U(1), V(1)) + innerₕ(U(2), V(1)), nothing))
            built = map(cases) do (name, Wu, Wv, a, l)
                F = form(Wu, Wv, a)
                A = assemble(F)
                op = matrix_free_operator(F)
                x, y = ones(ndofs(Wu)), zeros(ndofs(Wv))
                run!("\$name fresh assemble!", () -> assemble!(A, F))
                run!("\$name fresh mul!", () -> mul!(y, op, x))
                L = l === nothing ? nothing : form(Wv, l)
                b = L === nothing ? nothing : assemble(L)
                L === nothing || run!("\$name fresh linear assemble!", () -> assemble!(b, L))
                (name, F, L, A, b, op, x, y)
            end
            pts(n) = vcat(0.0, sort(rand(n - 2)), 1.0)
            change_points!(Ωb, (pts(7), pts(10)))
            for (name, F, L, A, b, op, x, y) in built
                run!("\$name assemble", () -> assemble(F))
                run!("\$name assemble! refill", () -> assemble!(A, F))
                run!("\$name matrix-free mul!", () -> mul!(y, op, x))
                L === nothing && continue
                run!("\$name linear assemble", () -> assemble(L))
                run!("\$name linear assemble!", () -> assemble!(b, L))
            end
            return out
        end
        """
        function check(out)
            @test length(out) == 31
            for (name, r) in out
                @testset "$name" begin
                    @test r == (occursin("fresh", name) ? "OK" : "STALE")
                end
            end
        end

        @testset "CpuSerial" begin
            mod = Module()
            include_string(mod, probe)
            check(Base.invokelatest(mod.stale_probe, Bramble.CpuSerial()))
        end

        @testset "CpuPolyester (child)" begin
            code = "using Polyester\n" * probe * """
            println("POLYESTER_LOADED\t", Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing)
            for (name, r) in stale_probe(Bramble.CpuPolyester())
                println(name, "\t", r)
            end
            """
            proj = Base.active_project()
            cmd = `$(Base.julia_cmd()) --project=$proj --startup-file=no --threads=2 -e $code`
            err = tempname()
            lines = split(readchomp(pipeline(cmd; stderr = err)), '\n')
            isfile(err) && (msg = read(err, String); isempty(msg) || @info msg; rm(err))
            pairs = [String(p[1]) => String(p[2]) for p in split.(lines, '\t'; limit = 2)]
            @test first(pairs) == ("POLYESTER_LOADED" => "true")
            check(pairs[2:end])
        end
    end
end

end # module
