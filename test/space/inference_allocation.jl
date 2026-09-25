module SpaceInferenceAllocationTests

using Test
using Bramble
using Bramble: D₋ᵧ, D₋₂, D₋ₓ, Mᵧ, M₂, Mₓ, VectorElement, inner₊ᵧ, inner₊ₓ, jumpᵧ, jump₂
using Bramble: jumpₓ, norminf_h
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
import Bramble: diff₋ₓ, diff₋ᵧ, diff₋₂, diff₋ₕ, diff₊ₓ, diff₊ᵧ, diff₊₂, diff₊ₕ, D₊ₓ, D₊ᵧ, D₊₂, ∇₊ₕ, M₊ₓ, M₊ᵧ, M₊₂, M₊ₕ
import Bramble: div₊ₕ!
using JET
using Bramble:
               components,
               _difference_engine!,
               _average_engine!,
               backward_spacings_for_derivative,
               Backward,
               Forward,
               diff₋ₓ,
               diff₊ₓ,
               diff₋ᵧ,
               diff₊ᵧ,
               diff₋₂,
               diff₊₂,
               diff₋ₕ,
               diff₊ₕ
using ..TestUtils: alloc_test, @test_allocs

# Type stability and allocation across grid spaces, operators and inner products.
#
# Every property here was established by measurement while optimising this subsystem and
# then left unguarded. Each one has already regressed at least once during that work:
#
#   - the difference engine boxed its spacing callable and allocated 64 bytes per grid
#     point, until `h` was given a type parameter;
#   - the seminorm rebuilt a closure and a weight vector per point;
#   - summing the seminorm directions through `ntuple` reintroduced an allocation;
#   - the operators wrote out of bounds on composite grid functions.
#
# None of it was visible to the suite, because line coverage does not see any of it.

@testset "Inference and allocations" begin
    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 64, false)
    Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 9), (true, false))
    Ωₕ3 = mesh(
        domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 6), (true, false, true)
    )
    Wₕ1, Wₕ2, Wₕ3 = gridspace(Ωₕ1), gridspace(Ωₕ2), gridspace(Ωₕ3)
    Vₕ2 = gridspace(Ωₕ2, Val(2))

    uₕ1 = Rₕ(Wₕ1, sin)
    uₕ2 = Rₕ(Wₕ2, x -> sin(x[1]) * x[2])
    uₕ3 = Rₕ(Wₕ3, x -> sin(x[1]) + x[3])
    cₕ2 = Rₕ(Vₕ2, (x -> x[1], x -> x[2]))

    @testset "Type stability (setup)" begin
        @test @inferred(gridspace(Ωₕ2)) isa Bramble.ScalarGridSpace
        @test @inferred(element(Wₕ2)) isa VectorElement
        @test @inferred(ndofs(Wₕ2)) isa Int
        @test @inferred(Rₕ(Wₕ2, x -> x[1])) isa VectorElement
        @test @inferred(avgₕ(Wₕ2, x -> x[1])) isa VectorElement
        @test @inferred(Rₕ(Vₕ2, (x -> x[1], x -> x[2]))) isa VectorElement
        @test @inferred(parent(uₕ2)) isa AbstractVector
    end

    @testset "Type stability (operators)" begin
        # scalar operators, per direction, in each dimension
        for (lbl, uₕ, ops) in (
            ("1D", uₕ1, (diff₋ₓ, diff₊ₓ, D₋ₓ, D₊ₓ, jumpₓ, Mₓ, M₊ₓ)),
            ("2D", uₕ2, (diff₋ᵧ, diff₊ᵧ, D₋ᵧ, D₊ᵧ, jumpᵧ, Mᵧ, M₊ᵧ)),
            ("3D", uₕ3, (diff₋₂, diff₊₂, D₋₂, D₊₂, jump₂, M₂, M₊₂))
        )
            @testset "$lbl" begin
                for op in ops
                    @test @inferred(op(uₕ)) isa VectorElement
                end
            end
        end

        # the tuple-valued aliases: a bare element in 1D, an NTuple above it
        @test @inferred(∇ₕ(uₕ1)) isa VectorElement
        for op in (∇ₕ, ∇₊ₕ, diff₋ₕ, diff₊ₕ, jumpₕ, Mₕ, M₊ₕ)
            @test @inferred(op(uₕ2)) isa NTuple{2, VectorElement}
            @test @inferred(op(uₕ3)) isa NTuple{3, VectorElement}
        end

        # composite grid functions go through a separate dispatch
        @test @inferred(D₋ₓ(cₕ2)) isa VectorElement
        @test @inferred(∇ₕ(cₕ2)) isa NTuple{2, VectorElement}
    end

    @testset "Type stability (inner products)" begin
        for (lbl, uₕ) in (("1D", uₕ1), ("2D", uₕ2), ("3D", uₕ3))
            @testset "$lbl" begin
                @test @inferred(innerₕ(uₕ, uₕ)) isa Float64
                @test @inferred(inner₊(uₕ, uₕ)) isa Float64
                @test @inferred(normₕ(uₕ)) isa Float64
                @test @inferred(snorm₁ₕ(uₕ)) isa Float64
                @test @inferred(norm₁ₕ(uₕ)) isa Float64
                @test @inferred(norminf_h(uₕ)) isa Float64
                g = ∇ₕ(uₕ)
                @test @inferred(norm₊(g)) isa Float64
                @test @inferred(inner₊(g, g)) isa Float64
                @test @inferred(norminf_h(g)) isa Float64
            end
        end
        @test @inferred(inner₊ₓ(uₕ2, uₕ2)) isa Float64
        @test @inferred(inner₊ᵧ(uₕ2, uₕ2)) isa Float64
        # the surface weight is computed per point rather than read from a stored vector,
        # so it is worth pinning that it still infers and still allocates nothing (#157)
        @test @inferred(inner_Γ(uₕ2, uₕ2, :ymin)) isa Float64
        @test @inferred(inner_Γ(uₕ3, uₕ3, :boundary)) isa Float64
    end

    @testset "Type stability (vector calculus)" begin
        # gpena/Bramble.jl#158: each of these recurses over directions on `Val(d)` rather
        # than looping, for the same boxing reason the vectorial aliases do (#146).
        for (lbl, uₕ) in (("1D", uₕ1), ("2D", uₕ2), ("3D", uₕ3))
            @testset "$lbl" begin
                @test @inferred(Δₕ(uₕ)) isa VectorElement
                @test @inferred(divₕ(∇ₕ(uₕ))) isa VectorElement
            end
        end
        @test @inferred(curlₕ((uₕ2, uₕ2))) isa VectorElement
        @test @inferred(curlₕ((uₕ3, uₕ3, uₕ3))) isa NTuple{3, VectorElement}
    end

    @testset "Zero allocations (inner products)" begin
        # A time-stepping loop evaluates these every step, so any allocation here is
        # per-step garbage.
        for (lbl, uₕ) in (("1D", uₕ1), ("2D", uₕ2), ("3D", uₕ3))
            @testset "$lbl" begin
                @test_allocs innerₕ(uₕ, uₕ)
                @test_allocs inner₊(uₕ, uₕ)
                @test_allocs normₕ(uₕ)
                @test_allocs snorm₁ₕ(uₕ)
                @test_allocs norm₁ₕ(uₕ)
                @test_allocs norminf_h(uₕ)
            end
        end
        # a component of a composite grid function is a scalar grid function, and the
        # contiguous view it holds must not cost anything either
        c = components(cₕ2)[1]
        @test_allocs innerₕ(c, c)
        @test_allocs normₕ(c)
        @test_allocs snorm₁ₕ(c)
        @test_allocs norminf_h(c)
        @test_allocs norminf_h(cₕ2)
        @test_allocs inner_Γ(uₕ2, uₕ2, :ymin)
        @test_allocs inner_Γ(uₕ3, uₕ3, :boundary)

        # the mutating vector-calculus forms accumulate into their destination in one
        # traversal per direction, so none of them needs a scratch grid function (#158)
        let v1 = similar(uₕ1), v2 = similar(uₕ2), v3 = similar(uₕ3)
            @test_allocs Δₕ!(v1, uₕ1)
            @test_allocs Δₕ!(v2, uₕ2)
            @test_allocs Δₕ!(v3, uₕ3)
            @test_allocs divₕ!(v2, (uₕ2, uₕ2))
            @test_allocs div₊ₕ!(v2, (uₕ2, uₕ2))
            @test_allocs curlₕ!(v2, (uₕ2, uₕ2))
        end
    end

    @testset "Zero dynamic dispatch (vectorial aliases)" begin
        # gpena/Bramble.jl#146: `∇ₕ`/`∇₊ₕ`/`diff₋ₕ`/`diff₊ₕ`/`Mₕ`/`M₊ₕ`/`D̃ₕ`/`Dcₕ`/`D̽ₕ`
        # used to generate their 2D/3D methods from `ntuple(i -> base_op(arg, Val(i)),
        # Val(D))`, which boxes `i` as a runtime Int inside the closure: `Val(i)` can
        # never constant-fold, so every coordinate paid for dynamic dispatch all the way
        # down the difference-engine call stack (2-8 dispatches per call, per JET).
        # `_vectorial_expr` now writes the 2D/3D methods out with literal `Val(1)`,
        # `Val(2)`, `Val(3)` calls instead, so this must report zero.
        for op in (∇ₕ, ∇₊ₕ, diff₋ₕ, diff₊ₕ, jumpₕ, Mₕ, M₊ₕ, D̃ₕ, Dcₕ, D̽ₕ)
            rep2 = JET.report_call(op, (typeof(uₕ2),))
            @test isempty(JET.get_reports(rep2))
            rep3 = JET.report_call(op, (typeof(uₕ3),))
            @test isempty(JET.get_reports(rep3))
        end
    end

    @testset "Zero allocations (stencils)" begin
        # The engines are the inner loop of every operator. `h` is passed both as
        # `nothing` and as the mesh's cached spacing vector, because those take different
        # dispatches and only the second ever boxed.
        vₕ = similar(uₕ1)
        h = backward_spacings_for_derivative(Ωₕ1)
        dims = (npoints(Ωₕ1),)
        diff!(o, i, hh) = _difference_engine!(o, i, hh, dims, Backward(), Val(1))
        avg!(o, i) = _average_engine!(o, i, dims, Backward(), Val(1))

        @test_allocs diff!(parent(vₕ), parent(uₕ1), h)
        @test_allocs diff!(parent(vₕ), parent(uₕ1), nothing)
        @test_allocs avg!(parent(vₕ), parent(uₕ1))
    end

    @testset "Callable spacing dispatch" begin
        # `h` may be a callable as well as a vector, and `_difference_engine!` names its
        # type so that Julia specialises on it. Julia does not specialise on an argument
        # of function type when the body only forwards it, which is what this does, so
        # without the type parameter the callable is boxed and every grid point pays a
        # dynamic dispatch.
        #
        # A cached vector is what the operators actually pass, and it specialises anyway,
        # so only a callable exercises this. The property is that the cost does not grow
        # with the grid: measured 32 B at both sizes below with the type parameter, and
        # 57,328 then 516,080 without it.
        function callable_bytes(n)
            Ωₙ = mesh(domain(interval(0.0, 1.0)), n, true)
            uₙ = Rₕ(gridspace(Ωₙ), sin)
            vₙ = similar(uₙ)
            hf = Base.Fix1(Bramble.spacing_for_derivative, Ωₙ)
            run!(o, i) = _difference_engine!(o, i, hf, (n,), Backward(), Val(1))
            alloc_test(run!, parent(vₙ), parent(uₙ))
        end
        @test callable_bytes(1024) == callable_bytes(8192)
    end

    @testset "Operator output allocation" begin
        # The exact property, not a bound: applying an operator costs one `similar`.
        # It is what fails first when a closure starts boxing or a temporary creeps in.
        for (lbl, uₕ, ops) in (
            ("1D", uₕ1, (diff₋ₓ, D₋ₓ, Mₓ, jumpₓ)),
            ("2D", uₕ2, (diff₋ᵧ, D₋ᵧ, Mᵧ, jumpᵧ)),
            ("3D", uₕ3, (diff₋₂, D₋₂, M₂, jump₂))
        )
            @testset "$lbl" begin
                baseline = alloc_test(similar, uₕ)
                for op in ops
                    @test alloc_test(op, uₕ) == baseline
                end
            end
        end
    end

    @testset "In-place restriction scaling" begin
        # Rₕ! and avgₕ! allocate a small constant. The property that matters is that it
        # is constant: anything proportional to the grid would be per-step garbage.
        function inplace_bytes(be, n)
            W = gridspace(mesh(domain(interval(0.0, 1.0)), n, true; backend = be))
            u = element(W)
            (alloc_test(Rₕ!, u, sin), alloc_test(avgₕ!, u, sin))
        end

        # A `Serial()` backend runs every grid size through the same plain loop,
        # so both sizes give exactly 0 bytes.
        be_serial = backend(policy = Serial())
        @test inplace_bytes(be_serial, 16) == (0, 0)
        @test inplace_bytes(be_serial, 2048) == (0, 0)   # 128x the degrees of freedom

        # `Parallel()` costs a small, size-independent constant (task spawn overhead).
        # A single measurement may drift slightly due to thread-spawn machinery
        # scheduling noise, independent of the grid size. Taking the minimum over repeats
        # with a small tolerance verifies that allocation does not scale with problem size.
        be_parallel = backend(policy = Parallel())
        function min_inplace_bytes(be, n)
            trials = ntuple(_ -> inplace_bytes(be, n), 5)
            (minimum(t[1] for t in trials), minimum(t[2] for t in trials))
        end
        small = min_inplace_bytes(be_parallel, 16)
        large = min_inplace_bytes(be_parallel, 2048)
        @test all(abs(s - l) <= 256 for (s, l) in zip(small, large))

        # Zero-allocation guarantees for masked, composite tuple, and Val quadrature paths
        d = domain(box((0.0, 0.0), (1.0, 1.0)), :left => :left)
        Ω = mesh(d, (8, 8))
        W = gridspace(Ω)
        V = gridspace(Ω, Val(2))
        u = element(W)
        v = element(V)
        f(x) = sin(x[1]) * cos(x[2])
        f_tup = (f, f)

        @test alloc_test(Rₕ!, u, f; markers = (:left,)) == 0
        @test alloc_test(avgₕ!, u, f; markers = (:left,)) == 0
        @test alloc_test(Rₕ!, v, f_tup) == 0
        @test alloc_test(avgₕ!, u, f, Val(3)) == 0
        @test alloc_test(avgₕ!, u, f; quad_points = Val(3)) == 0
        @test alloc_test(Rₕ!, v, f_tup; markers = (:left,)) == 0
        @test alloc_test(avgₕ!, v, f_tup; markers = (:left,)) == 0

        # gpena/Bramble.jl#182: the single-vector-function form scatters one evaluation of
        # `f` across every leaf of a shared-mesh composite space (`_rule_scatter_kernel`),
        # rather than one rule per leaf -- verify that path is zero-alloc too, not just the
        # tuple-of-functions form above.
        f_vec(x) = (sin(x[1]), cos(x[2]))
        @test alloc_test(Rₕ!, v, f_vec) == 0
        @test alloc_test(avgₕ!, v, f_vec) == 0
        @test alloc_test(Rₕ!, v, f_vec; markers = (:left,)) == 0
        @test alloc_test(avgₕ!, v, f_vec; markers = (:left,)) == 0

        # gpena/Bramble.jl#64: `_avgₕ!`/`_avg_masked!`'s composite Tuple methods used to
        # route the per-leaf application through `map(f, t1, t2)` with a closure that
        # reconstructs `Val(D)` inside it. Measured 192 B where a plain `ntuple` over the
        # same body gives 0 -- the Core.Box trap this suite otherwise only documents on the
        # difference engine, here on the fix for nested composite spaces.
        Vn = (W × W) × W
        un = element(Vn)
        f_tup3 = (f, f, f)
        @test alloc_test(avgₕ!, un, f_tup3) == 0
        @test alloc_test(avgₕ!, un, f_tup3; markers = (:left,)) == 0
        @test alloc_test(Rₕ!, un, f_tup3; markers = (:left,)) == 0

        # Same single-vector-function scatter path (gpena/Bramble.jl#182), three levels deep.
        f_vec3(x) = (sin(x[1]), cos(x[2]), x[1] + x[2])
        @test alloc_test(Rₕ!, un, f_vec3) == 0
        @test alloc_test(avgₕ!, un, f_vec3) == 0

        # Zero allocations for copyto! (values!'s replacement, gpena/Bramble.jl#73) and πₕ!
        @test alloc_test(copyto!, u, 1.0) == 0
        @test alloc_test(copyto!, u, parent(u)) == 0

        # Zero allocations for the specialised in-place broadcast copyto!
        # (gpena/Bramble.jl#181): unwrapping every VectorElement leaf down to its own
        # `parent` before delegating must not itself allocate, on a compound expression
        # (multiple operators fused) as well as a single scaling.
        _compound!(w, u, v, α) = (w .= u .+ v .* α)
        _scaled!(w, v, β) = (w .= β .* v)
        w = element(W)
        v2 = element(W, 2.0)
        @test alloc_test(_compound!, w, u, v2, 3.0) == 0
        @test alloc_test(_scaled!, w, v2, 2.0) == 0

        W_target = gridspace(mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (8, 8)))
        u_target = element(W_target)
        @test alloc_test(πₕ!, u_target, u) == 0
    end
end

end # module SpaceInferenceAllocationTests
