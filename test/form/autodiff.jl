using Test
using Bramble
using ForwardDiff
using ReverseDiff
using DifferentiationInterface
using SparseArrays
using LinearAlgebra: issymmetric, norm

# Differentiating through the constrained linear system.
#
# There are two independent things one might differentiate here, and they were not in the
# same state.
#
#   - the *system*: the matrix and right-hand side carry `ForwardDiff.Dual` coefficients,
#     and the constraints are applied to them. This already worked — `dirichlet_bc!` and
#     `symmetrize!` read `zero(T)`/`one(T)` off `eltype(A)` and are otherwise generic — but
#     nothing checked, and a `\\` through both is the shape an adjoint actually has.
#
#   - the *boundary data*: the parameter being differentiated sits inside the condition
#     itself, `x -> a * x[1]`. This did not work, originally: the conditions were stored in
#     a `Set{Marker{BrambleFunction{…, CoType, …}}}` whose concrete element type is what
#     kept applying them allocation free, and `CoType` was the *domain's* element type —
#     so a Float64 domain could carry only Float64 boundary values, and a Dual-returning
#     condition met `MethodError: no method matching Float64(::Dual)` inside the wrapper.
#     Point 48 (2026-09-04) removed the wrapper entirely — `conditions` is a `Tuple` of raw
#     closures now, one `Marker{F}` per condition's own type, nothing erased — so there is
#     no `CoType` left to settle at all: `v[idx] = func(point(...))` converts to `v`'s own
#     eltype the same way any assignment does, exactly the rule `Rₕ`/`avgₕ` already used.
#
# As in test/space/autodiff.jl, each case is checked against a central difference of the
# same functional in plain Float64 — so it tests that the derivative is right, not merely
# that it ran.

@testset "Constraint differentiation" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
        (5, 5), (true, true))
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(3))
    n = ndofs(Wₕ)

    @testset "Dual system" begin
        @test _matches_fd(a -> begin
            A = a .* _tri(n)
            dirichlet_bc!(A, Ωₕ, :bottom)
            sum(A)
        end)

        @test _matches_fd(a -> begin
            A, F = a .* _tri(n), a .* ones(n)
            dirichlet_bc!(A, Ωₕ, :bottom)
            symmetrize!(A, F, Ωₕ, :bottom)
            sum(A) + sum(F)
        end)

        # the shape that matters: a solve with both applied
        @test _matches_fd(a -> begin
            A, F = a .* _tri(n), a .* collect(1.0:n)
            dirichlet_bc!(A, Ωₕ, :bottom)
            symmetrize!(A, F, Ωₕ, :bottom)
            sum(A \ F)
        end)

        # and the same through a composite space
        @test _matches_fd(a -> begin
            A = a .* blockdiag(_tri(n), _tri(n), _tri(n))
            F = a .* collect(1.0:(3n))
            dirichlet_bc!(A, Vₕ, :bottom)
            symmetrize!(A, F, Vₕ, :bottom)
            sum(A \ F)
        end)
    end

    @testset "Dual boundary data" begin
        @test _matches_fd(a -> begin
            bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> a * x[1] + a^2))
            v = zeros(typeof(a), n)
            dirichlet_bc!(v, Ωₕ, bcs, :bottom)
            sum(v)
        end)

        # through a composite space, where the value lands in every leaf
        @test _matches_fd(a -> begin
            bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> a * sin(x[1])))
            v = zeros(typeof(a), 3n)
            dirichlet_bc!(v, Vₕ, bcs, :bottom)
            sum(v)
        end)

        # and end to end: the boundary data feeds a solve
        @test _matches_fd(a -> begin
            A = _tri(n)
            F = zeros(typeof(a), n)
            F .= collect(1.0:n)
            bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> a * x[1] + 1))
            dirichlet_bc!(A, Ωₕ, :bottom)
            dirichlet_bc!(F, Ωₕ, bcs, :bottom)
            sum(Matrix(A) \ F)
        end)
    end

    @testset "Vanishing boundary variation" begin
        # `symmetrize!` skips the elimination when the boundary value is zero, which is
        # only sound under AD because `iszero` on a Dual tests the partials as well as the
        # value. A Dual that is 0.0 here with derivative 1.0 must NOT take the short path —
        # narrowing that test to `iszero(value(x))` would silently drop this derivative.
        #
        # F is built so that at a = 1.3 every entry is exactly zero with unit sensitivity.
        @test _matches_fd(a -> begin
            A = _tri(n)
            F = fill(a - 1.3, n)
            dirichlet_bc!(A, Ωₕ, :bottom)
            symmetrize!(A, F, Ωₕ, :bottom)
            sum(F)
        end)

        # the value really is zero-with-derivative at the point being differentiated
        d = ForwardDiff.Dual{ForwardDiff.Tag{typeof(identity), Float64}}(0.0, 1.0)
        @test ForwardDiff.value(d) == 0.0
        @test !iszero(d)
    end

    @testset "Multi-variable gradient" begin
        function J(p)
            bcs = dirichlet_constraints(set(Ωₕ),
                :bottom => (x -> p[1] * x[1] + p[2] * x[1]^2))
            v = zeros(eltype(p), n)
            dirichlet_bc!(v, Ωₕ, bcs, :bottom)
            return sum(abs2, v)
        end
        p0 = [1.3, 0.7]
        g = ForwardDiff.gradient(J, p0)
        @test length(g) == 2
        for k in 1:2
            e = zeros(2)
            e[k] = 1e-6
            @test isapprox(g[k], (J(p0 .+ e) - J(p0 .- e)) / 2e-6; rtol = 1e-5)
        end
    end

    @testset "Element type inference" begin
        # No CoType to settle at all now (see the header note): a condition's return type
        # only ever has to match whatever `v` was already allocated with. Checked directly
        # by applying, not by inspecting an internal wrapper type that no longer exists.
        apply(bcs) = (v = zeros(n); dirichlet_bc!(v, Ωₕ, bcs, :bottom); v)

        # a plain condition applies as-is
        @test eltype(apply(dirichlet_constraints(set(Ωₕ), :bottom => (x -> 7.0)))) ===
              Float64

        # an integer-valued condition still gives Float64 on a Float64 destination —
        # ordinary conversion on assignment, not a promotion decided ahead of time
        @test eltype(apply(dirichlet_constraints(set(Ωₕ), :bottom => (x -> 1)))) === Float64

        # a condition need not be defined everywhere it is merely *registered* — building
        # the constraints no longer evaluates the condition at all (there is no probe point
        # to need one), so a condition undefined off its own boundary — the trap `Rₕ` hit
        # with `x -> sqrt(x - 0.5)` — cannot fail at construction time the way a CoType
        # probe once could. (Applying it is a separate question: `sqrt(x[1] - 0.5)` is
        # genuinely undefined across all of `:bottom` here, since that label spans the
        # whole `x[1] ∈ [0,1]` edge — so only construction is checked, not application.)
        @test_nowarn dirichlet_constraints(set(Ωₕ), :bottom => (x -> sqrt(x[1] - 0.5)))

        # a Dual-returning condition, applied into a Dual-eltype destination, carries a Dual
        a = ForwardDiff.Dual{ForwardDiff.Tag{typeof(identity), Float64}}(1.3, 1.0)
        bcs_dual = dirichlet_constraints(set(Ωₕ), :bottom => (x -> a * x[1]))
        v_dual = zeros(typeof(a), n)
        dirichlet_bc!(v_dual, Ωₕ, bcs_dual, :bottom)
        @test eltype(v_dual) <: ForwardDiff.Dual
        @test !all(iszero, v_dual)

        # the mesh underneath stays undifferentiated
        @test eltype(Ωₕ) === Float64
    end

    @testset "Time-dependent constraints" begin
        @test _matches_fd(a -> begin
            bcs = dirichlet_constraints(set(Ωₕ), interval(0.0, 1.0),
                :bottom => ((x, t) -> a * t * x[1] + a))
            v = zeros(typeof(a), n)
            dirichlet_bc!(v, Ωₕ, bcs(0.5), :bottom)
            sum(v)
        end)
    end

    @testset "Zero allocations (constraints)" begin
        # The inference happens once, when the constraints are built. The hot path is
        # untouched, and has to stay so.
        function bytes()
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
                (24, 24), (true, true))
            W = gridspace(Ω)
            m = ndofs(W)
            bcs = dirichlet_constraints(set(Ω), :bottom => (x -> 7.0))
            v = zeros(m)
            dirichlet_bc!(v, Ω, bcs, :bottom)
            return @allocated dirichlet_bc!(v, Ω, bcs, :bottom)
        end
        @test bytes() == 0
    end
end

@testset "Reverse-mode residual" begin
    # The backend survey in test/space/autodiff_backends.jl establishes that ReverseDiff can
    # differentiate the *space* layer — `Rₕ` and an operator. It says nothing about the form
    # layer, and assembly is a different proposition: `assemble` allocates its vector from
    # `_assembled_eltype` and then a stencil engine writes into it point by point, so the
    # tracked type has to survive both the allocation and the scatter.
    #
    # Written through DifferentiationInterface for the same reason the survey is: it is the
    # layer a caller reaches for, and adding a backend is one more entry rather than another
    # block speaking another API.
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 8, true)
    Wₕ = gridspace(Ωₕ)

    resid = p -> begin
        cₕ = Rₕ(Wₕ, x -> p[1] * sin(x) + p[2] * x^2)
        return sum(assemble(form(Wₕ, v -> innerₕ(cₕ, v))))
    end
    p0 = [1.3, 0.7]

    # a central difference, so a wrong gradient fails rather than merely a thrown one
    h = 1e-6
    fd = [(resid(p0 .+ h .* (1:2 .== i)) - resid(p0 .- h .* (1:2 .== i))) / 2h for i in 1:2]

    gf = DifferentiationInterface.gradient(resid, AutoForwardDiff(), p0)
    gr = DifferentiationInterface.gradient(resid, AutoReverseDiff(), p0)

    @test isapprox(gf, fd; rtol = 1e-5)
    @test isapprox(gr, fd; rtol = 1e-5)
    @test isapprox(gf, gr; rtol = 1e-10)
end

@testset "Coupled routing Jacobian" begin
    # Each block's entries come from its own coefficients, so the Jacobian of the assembled
    # vector is block diagonal. A routing error does not give a wrong number — it puts mass
    # in a block that should be empty, which is invisible to any test that only checks
    # values.
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 8, true)
    Wₕ = gridspace(Ωₕ)
    Vₕ = Wₕ^Val(2)
    n = ndofs(Wₕ)
    w0 = collect(range(0.5, 1.5, length = ndofs(Vₕ)))

    routed = w -> begin
        c = element(Vₕ, eltype(w))
        values(c) .= w
        return assemble(form(Vₕ, v -> innerₕ(c(1), v(1)) + inner₊ₓ(D₋ₓ(c(2)), D₋ₓ(v(2)))))
    end

    Jf = DifferentiationInterface.jacobian(routed, AutoForwardDiff(), w0)
    Jr = DifferentiationInterface.jacobian(routed, AutoReverseDiff(), w0)

    @test size(Jf) == (ndofs(Vₕ), ndofs(Vₕ))
    @test isapprox(Jf, Jr; rtol = 1e-10)

    # the off-diagonal blocks are empty, exactly
    @test norm(Jf[1:n, (n + 1):(2n)]) == 0
    @test norm(Jf[(n + 1):(2n), 1:n]) == 0
    @test norm(Jr[1:n, (n + 1):(2n)]) == 0
    @test norm(Jr[(n + 1):(2n), 1:n]) == 0

    # and the assertion above bites: a form that deliberately crosses the blocks puts its
    # mass off the diagonal and none on it. Without this the four tests above would pass on
    # a Jacobian that was empty for some other reason.
    crossed = w -> begin
        c = element(Vₕ, eltype(w))
        values(c) .= w
        return assemble(form(Vₕ, v -> innerₕ(c(2), v(1))))
    end
    Jx = DifferentiationInterface.jacobian(crossed, AutoForwardDiff(), w0)

    @test norm(Jx[1:n, (n + 1):(2n)]) > 0
    @test norm(Jx[1:n, 1:n]) == 0
end
