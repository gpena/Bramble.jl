using Test
using Bramble
using Bramble: values, components, _difference_engine!, _average_engine!,
               backward_spacings_for_derivative, Backward, Forward

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

# Standalone runner fallback
if !@isdefined(alloc_test)
    @inline function alloc_test(f::F, args...; kwargs...) where {F}
        f(args...; kwargs...)
        return @allocated(f(args...; kwargs...))
    end
end

if !@isdefined(var"@test_allocs")
    macro test_allocs(call_expr)
        if Meta.isexpr(call_expr, :call)
            fn = call_expr.args[1]
            args = call_expr.args[2:end]
            quote
                @test alloc_test($(esc(fn)), $(map(esc, args)...)) == 0
            end
        else
            quote
                let
                    $(esc(call_expr))
                    @test (@allocated $(esc(call_expr))) == 0
                end
            end
        end
    end
end

@testset "Inference and allocations" begin
    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 64, false)
    Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 9), (true, false))
    Ωₕ3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 6),
        (true, false, true))
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
        @test @inferred(values(uₕ2)) isa AbstractVector
    end

    @testset "Type stability (operators)" begin
        # scalar operators, per direction, in each dimension
        for (lbl, uₕ, ops) in (
            ("1D", uₕ1, (diff₋ₓ, diff₊ₓ, D₋ₓ, D₊ₓ, jumpₓ, M₋ₓ, M₊ₓ)),
            ("2D", uₕ2, (diff₋ᵧ, diff₊ᵧ, D₋ᵧ, D₊ᵧ, jumpᵧ, M₋ᵧ, M₊ᵧ)),
            ("3D", uₕ3, (diff₋₂, diff₊₂, D₋₂, D₊₂, jump₂, M₋₂, M₊₂)))
            @testset "$lbl" begin
                for op in ops
                    @test @inferred(op(uₕ)) isa VectorElement
                end
            end
        end

        # the tuple-valued aliases: a bare element in 1D, an NTuple above it
        @test @inferred(∇₋ₕ(uₕ1)) isa VectorElement
        for op in (∇₋ₕ, ∇₊ₕ, diff₋ₕ, diff₊ₕ, jumpₕ, M₋ₕ, M₊ₕ)
            @test @inferred(op(uₕ2)) isa NTuple{2, VectorElement}
            @test @inferred(op(uₕ3)) isa NTuple{3, VectorElement}
        end

        # composite grid functions go through a separate dispatch
        @test @inferred(D₋ₓ(cₕ2)) isa VectorElement
        @test @inferred(∇₋ₕ(cₕ2)) isa NTuple{2, VectorElement}
    end

    @testset "Type stability (inner products)" begin
        for (lbl, uₕ) in (("1D", uₕ1), ("2D", uₕ2), ("3D", uₕ3))
            @testset "$lbl" begin
                @test @inferred(innerₕ(uₕ, uₕ)) isa Float64
                @test @inferred(inner₊(uₕ, uₕ)) isa Float64
                @test @inferred(normₕ(uₕ)) isa Float64
                @test @inferred(snorm₁ₕ(uₕ)) isa Float64
                @test @inferred(norm₁ₕ(uₕ)) isa Float64
                g = ∇₋ₕ(uₕ)
                @test @inferred(norm₊(g)) isa Float64
                @test @inferred(inner₊(g, g)) isa Float64
            end
        end
        @test @inferred(inner₊ₓ(uₕ2, uₕ2)) isa Float64
        @test @inferred(inner₊ᵧ(uₕ2, uₕ2)) isa Float64
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
            end
        end
        # a component of a composite grid function is a scalar grid function, and the
        # contiguous view it holds must not cost anything either
        c = components(cₕ2)[1]
        @test_allocs innerₕ(c, c)
        @test_allocs normₕ(c)
        @test_allocs snorm₁ₕ(c)
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

        @test_allocs diff!(values(vₕ), values(uₕ1), h)
        @test_allocs diff!(values(vₕ), values(uₕ1), nothing)
        @test_allocs avg!(values(vₕ), values(uₕ1))
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
            alloc_test(run!, values(vₙ), values(uₙ))
        end
        @test callable_bytes(1024) == callable_bytes(8192)
    end

    @testset "Operator output allocation" begin
        # The exact property, not a bound: applying an operator costs one `similar`.
        # It is what fails first when a closure starts boxing or a temporary creeps in.
        for (lbl, uₕ, ops) in (
            ("1D", uₕ1, (diff₋ₓ, D₋ₓ, M₋ₓ, jumpₓ)),
            ("2D", uₕ2, (diff₋ᵧ, D₋ᵧ, M₋ᵧ, jumpᵧ)),
            ("3D", uₕ3, (diff₋₂, D₋₂, M₋₂, jump₂)))
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

        # Zero allocations for values! and πₕ!
        @test alloc_test(values!, u, 1.0) == 0
        @test alloc_test(values!, u, values(u)) == 0

        W_target = gridspace(mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (8, 8)))
        u_target = element(W_target)
        @test alloc_test(πₕ!, u_target, u) == 0
    end
end
