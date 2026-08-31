using Test
using Bramble
using ForwardDiff
using LinearAlgebra: Diagonal, diag
using Bramble: LinearForm, form, assemble, assemble!, assemble_parallel!, test_space,
               element, resolve_form_ast, apply_dirichlet_conditions!, LinearProduct,
               values, ParallelWorkspace, TestFunction, TrialFunction,
               IndexedTestFunction, IndexedTrialFunction, test_component_or_nothing,
               routes_by_component

# Assembling the right-hand side of a system.
#
# `form(Wₕ, v -> …)` closes over a test function and records the AST; `assemble` walks the
# grid, evaluates the stencil at each point and scatters the weights into a vector. This is
# the file a time loop calls every step, so the allocation counts below are part of its
# contract rather than a nicety.
#
# The values are checked against integrals that can be worked out by hand: `innerₕ(uₕ, v)`
# assembles the vector whose entries are ``|□ᵢ| uᵢ``, so summing it gives ``∫ u`` over the
# domain to the accuracy of the quadrature — exactly, for the cases below.

@testset "Linear forms" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
        (8, 8), (true, true))
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(2))
    uₕ = Rₕ(Wₕ, x -> x[1] + x[2])
    n = ndofs(Wₕ)

    @testset "what it builds" begin
        lf = form(Wₕ, v -> innerₕ(uₕ, v))
        @test lf isa LinearForm
        @test test_space(lf) === Wₕ
        @test lf.workspace isa ParallelWorkspace{2}
        @test resolve_form_ast(lf) isa LinearProduct
    end

    @testset "what it assembles" begin
        # ∫(x + y) over the unit square is 1, and the assembled vector's entries are the
        # cell measures times the coefficients, so it sums to exactly that.
        b = assemble(form(Wₕ, v -> innerₕ(uₕ, v)))
        @test length(b) == n
        @test sum(b) ≈ 1.0

        # a constant source integrates to the area
        @test sum(assemble(form(Wₕ, v -> innerₕ(Rₕ(Wₕ, x -> 3.0), v)))) ≈ 3.0

        # and a number or a function on the left works the same way
        @test sum(assemble(form(Wₕ, v -> innerₕ(2.0, v)))) ≈ 2.0
        @test sum(assemble(form(Wₕ, v -> innerₕ(x -> x[1], v)))) ≈ 0.5

        # in place, into a vector the caller owns
        b2 = zeros(n)
        returned = assemble!(b2, form(Wₕ, v -> innerₕ(uₕ, v)))
        @test returned === b2
        @test b2 ≈ b

        # assembling twice into the same vector does not accumulate
        assemble!(b2, form(Wₕ, v -> innerₕ(uₕ, v)))
        @test b2 ≈ b
    end

    @testset "over a composite space" begin
        # The composite core walks the components and assembles the *same* AST into each
        # block at its offset. So a form written against one source puts that source in
        # every component: ∫x over the unit square is 0.5, and the assembled vector sums to
        # 1.0 across the two blocks. Giving each component a different integrand needs the
        # indexed trial and test functions, which is the coupled path in bilinear.jl.
        uv = Rₕ(Vₕ, (x -> x[1], x -> x[2]))
        lf = form(Vₕ, v -> innerₕ(uv(1), v))
        b = assemble(lf)
        @test length(b) == ndofs(Vₕ)
        @test sum(b) ≈ 2 * 0.5

        # and each block holds the same thing, which is what "the same AST" means
        m = ndofs(Wₕ)
        @test b[1:m] ≈ b[(m + 1):(2m)]
    end

    @testset "a nested composite space covers every block" begin
        # The cores used to walk `space.spaces`, the top-level components, and reserve
        # `ndofs(sp)` for each. For a component that is itself composite that reserved the
        # whole nested block while `indices(mesh(sp))` covered one leaf's grid, so the
        # assembly wrote into a fraction of the range. Walking the leaves fixes it, and the
        # check is that a nested space and a flat one of the same size now agree.
        inner = gridspace(Ωₕ, Val(2))
        nested = Bramble.CompositeGridSpace((Wₕ, inner, Wₕ))
        flat = gridspace(Ωₕ, Val(4))
        @test ndofs(nested) == ndofs(flat)

        bn = assemble(form(nested, v -> innerₕ(uₕ, v)))
        bf = assemble(form(flat, v -> innerₕ(uₕ, v)))
        @test bn ≈ bf
        @test sum(bn) ≈ 4 * 1.0            # four blocks, each integrating to 1

        # and the parallel core walks the leaves the same way
        bp = similar(bn)
        assemble_parallel!(bp, form(nested, v -> innerₕ(uₕ, v)))
        @test bp ≈ bn
    end

    @testset "a coupled right-hand side, one term per component" begin
        # `v(i)` gives the i-th component of the symbolic test function, so a coupled form
        # reads the way an indexed grid function does:
        #
        #     form(Vₕ, v -> innerₕ(uₕ(1), v(1)) + innerₕ(uₕ(2), v(2)))
        #
        # Three things had to exist for that. `TestFunction` was not callable at all.
        # `block_extract.jl` routed a `BilinearProduct` to its block but not a
        # `LinearProduct`, so a coupled *right-hand side* could not be routed. And the
        # composite core assembled one AST into every block, which is the opposite of what a
        # per-component form means.
        m = ndofs(Wₕ)
        uv = Rₕ(Vₕ, (x -> x[1], x -> 10 * x[1]))
        nblocks = ndofs(Vₕ) ÷ m
        blocks(b) = [sum(b[(k * m + 1):((k + 1) * m)]) for k in 0:(nblocks - 1)]

        @testset "v(i) is the indexed test function" begin
            v = TestFunction{2}()
            @test v(2) === IndexedTestFunction{2}(2)
            @test TrialFunction{2}()(1) === IndexedTrialFunction{2}(1)
        end

        @testset "each term lands in its own block" begin
            # ∫x = 0.5 into the first block, ∫10x = 5.0 into the second
            b = assemble(form(Vₕ, v -> innerₕ(uv(1), v(1)) + innerₕ(uv(2), v(2))))
            @test blocks(b) ≈ [0.5, 5.0]

            # one term alone reaches only its own block
            @test blocks(assemble(form(Vₕ, v -> innerₕ(uv(2), v(2))))) ≈ [0.0, 5.0]
        end

        @testset "a form naming no component still reaches every block" begin
            # The prior behaviour, which has to keep working: the same integrand in each.
            @test blocks(assemble(form(Vₕ, v -> innerₕ(uv(1), v)))) ≈ [0.5, 0.5]

            # and the two spellings mix, because routing is decided per term
            mixed = assemble(form(Vₕ, v -> innerₕ(uv(1), v) + innerₕ(uv(2), v(2))))
            @test blocks(mixed) ≈ [0.5, 0.5 + 5.0]
        end

        @testset "the routing query" begin
            v = TestFunction{2}()
            @test test_component_or_nothing(innerₕ(uv(1), v(2))) == 2
            @test test_component_or_nothing(innerₕ(uv(1), v)) === nothing
            @test test_component_or_nothing(innerₕ(uv(1), D₋ₓ(v(2)))) == 2

            # asked once per assembly, to keep an un-indexed form off the routing branch
            @test routes_by_component(innerₕ(uv(1), v(1)) + innerₕ(uv(2), v(2)))
            @test !routes_by_component(innerₕ(uv(1), v))
            @test routes_by_component(innerₕ(uv(1), v) + innerₕ(uv(2), v(2)))
        end

        @testset "and it costs nothing" begin
            # The routing recurses the AST rather than calling `flatten_sum`, which answers
            # with a Vector{Any}: that allocated 544 B per assembly and made every term a
            # dynamic read.
            function bytes(mk)
                Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 8),
                    (true, true))
                V = gridspace(Ω, Val(3))
                u = Rₕ(V, ntuple(_ -> (x -> x[1] + x[2]), 3))
                lf = form(V, mk(u))
                ast = resolve_form_ast(lf)
                b = zeros(ndofs(V))
                assemble!(b, lf; ast = ast)
                return @allocated assemble!(b, lf; ast = ast)
            end
            @test bytes(u -> (v -> innerₕ(u(1), v))) == 0
            @test bytes(u -> (v -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))) == 0
        end
    end

    @testset "the parallel path agrees with the serial one" begin
        # Not the default, and the docstring says why — slower at every size measured. It
        # still has to give the same answer.
        #
        # Note what this does and does not establish. `Pkg.test()` runs on one thread unless
        # the environment says otherwise, and on one thread the threaded core is a degenerate
        # case: the per-thread buffers and the cross-thread reduction never actually race. CI
        # sets JULIA_NUM_THREADS=auto, so it is exercised there. The concurrent assertions
        # below are skipped rather than passed when only one thread is available, so a
        # single-threaded run cannot claim to have tested concurrency.
        @info "parallel assembly tested on $(Threads.nthreads()) thread(s)"

        for (nm, sp, u) in (("scalar", Wₕ, uₕ),
            ("composite", Vₕ, Rₕ(Vₕ, (x -> sin(x[1]), x -> cos(x[2])))(1)))
            lf = form(sp, v -> innerₕ(u, v))
            bs = assemble(lf)
            bp = similar(bs)
            @test assemble_parallel!(bp, lf) === bp
            @test bp ≈ bs
        end

        if Threads.nthreads() > 1
            # More work than threads, so every thread gets a chunk and the reduction has
            # something to combine. A race in the scatter, or a buffer left unzeroed, shows
            # here and nowhere else.
            Ωb = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (40, 40),
                (true, true))
            Wb = gridspace(Ωb)
            ub = Rₕ(Wb, x -> x[1] * x[2] + 1)
            lfb = form(Wb, v -> innerₕ(ub, v))
            bs = assemble(lfb)
            bp = similar(bs)
            assemble_parallel!(bp, lfb)
            @test bp ≈ bs

            # repeated runs agree with each other, which a race would break
            bp2 = similar(bs)
            assemble_parallel!(bp2, lfb)
            @test bp2 == bp

            # and the coupled path under threads
            Vb = gridspace(Ωb, Val(2))
            uv2 = Rₕ(Vb, (x -> x[1], x -> 10 * x[1]))
            lfc = form(Vb, v -> innerₕ(uv2(1), v(1)) + innerₕ(uv2(2), v(2)))
            bc = assemble(lfc)
            bcp = similar(bc)
            assemble_parallel!(bcp, lfc)
            @test bcp ≈ bc
        else
            @test_skip "concurrency not exercised: only one thread available"
            @test_skip "repeatability under threads not exercised"
            @test_skip "coupled parallel path under threads not exercised"
        end
    end

    @testset "Dirichlet conditions" begin
        bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> 5.0))
        marked = index_in_marker(Ωₕ, :bottom)
        @test any(marked)

        b = assemble(form(Wₕ, v -> innerₕ(uₕ, v));
            dirichlet_conditions = bcs, dirichlet_labels = :bottom)
        @test all(b[marked] .≈ 5.0)

        # a tuple of labels, and an empty tuple which asks for nothing
        b2 = assemble(form(Wₕ, v -> innerₕ(uₕ, v));
            dirichlet_conditions = bcs, dirichlet_labels = (:bottom,))
        @test b2 ≈ b
        plain = assemble(form(Wₕ, v -> innerₕ(uₕ, v)))
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, v)); dirichlet_conditions = bcs,
            dirichlet_labels = ()) ≈ plain

        # naming labels without conditions is a usage error rather than a silent no-op.
        # The conditions default to `nothing` now: they used to default to an empty
        # constraint set, built on every call and then discarded whenever no labels were
        # named — 2,080 B per assembly for an argument nothing read.
        @test_throws ArgumentError assemble(form(Wₕ, v -> innerₕ(uₕ, v));
            dirichlet_labels = :bottom)
        msg = try
            assemble(form(Wₕ, v -> innerₕ(uₕ, v)); dirichlet_labels = :bottom)
        catch e
            sprint(showerror, e)
        end
        @test occursin("dirichlet_conditions", msg)

        # and an invalid label type is rejected before anything is assembled
        @test_throws ErrorException assemble(form(Wₕ, v -> innerₕ(uₕ, v));
            dirichlet_conditions = bcs, dirichlet_labels = 3)
    end

    @testset "allocations, which are part of the contract" begin
        # This is what a time loop calls every step. The assembly kernel itself is
        # allocation free; what used to cost was the two default arguments — an empty
        # constraint set at 2,080 B and the AST resolution at 160 B, both recomputed per
        # call and both invariant for a given form.
        #
        # Measured inside a function on concrete locals: read from a non-const global, the
        # arguments box at the call boundary and the reading is of the box.
        function counts(N)
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (N, N), (true, true))
            W = gridspace(Ω)
            u = Rₕ(W, x -> x[1] + x[2])
            lf = form(W, v -> innerₕ(u, v))
            ast = resolve_form_ast(lf)
            b = zeros(ndofs(W))

            assemble!(b, lf)                      # warm both paths
            assemble!(b, lf; ast = ast)

            return (with_ast = @allocated(assemble!(b, lf; ast = ast)),
                without_ast = @allocated(assemble!(b, lf)))
        end

        for N in (8, 24)          # 9x the degrees of freedom apart
            c = counts(N)
            # handed a resolved AST, assembly allocates nothing at all
            @test c.with_ast == 0
            # and resolving costs a fixed amount that does not grow with the grid
            @test c.without_ast < 512
        end

        # the cost of resolving does not scale, which is what makes the keyword worth having
        @test counts(8).without_ast == counts(24).without_ast

        # the composite path too. It used to allocate a Vector of dof offsets on every
        # call — 128 B, constant in the grid but paid every step of a time loop.
        function composite_bytes(N)
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (N, N), (true, true))
            V = gridspace(Ω, Val(3))
            uv = Rₕ(V, ntuple(_ -> (x -> x[1] + x[2]), 3))
            lf = form(V, v -> innerₕ(uv(1), v))
            ast = resolve_form_ast(lf)
            b = zeros(ndofs(V))
            assemble!(b, lf; ast = ast)
            return @allocated assemble!(b, lf; ast = ast)
        end
        @test composite_bytes(8) == 0
        @test composite_bytes(16) == 0
    end

    @testset "differentiating an assembled residual" begin
        # The shape a nonlinear solve has: build the residual of a form, and let the solver
        # differentiate it with respect to the coefficient vector to get a Jacobian.
        #
        # This did not work. `assemble` allocated its output with `eltype(test_space(form))`
        # — the *space's* element type — so a Float64 space could only assemble a Float64
        # right-hand side, and writing a Dual weight into it met
        # `MethodError: no method matching Float64(::Dual)`. The type now comes from the
        # form's own weights, promoted against the space's, which is the rule `Rₕ` already
        # used and the same defect `dirichlet_constraints` had.
        u0 = values(uₕ)

        @testset "with respect to a parameter in the source" begin
            J(a) = sum(assemble(form(Wₕ, v -> innerₕ(Rₕ(Wₕ, x -> a * (x[1] + x[2])), v))))
            h = 1e-6
            @test isapprox(ForwardDiff.derivative(J, 1.3), (J(1.3 + h) - J(1.3 - h)) / 2h;
                rtol = 1e-5)
        end

        @testset "the Jacobian of a nonlinear residual" begin
            # innerₕ(u², v) assembles the vector with entries |□ᵢ| uᵢ², so its Jacobian is
            # diagonal with 2 |□ᵢ| uᵢ. Checked against that closed form rather than against
            # a finite difference, which is a stronger statement.
            residual(u) = assemble(form(Wₕ, v -> innerₕ(element(Wₕ, u .* u), v)))
            Jm = ForwardDiff.jacobian(residual, u0)

            @test size(Jm) == (n, n)
            @test Jm ≈ Diagonal(diag(Jm))

            measures = assemble(form(Wₕ, v -> innerₕ(element(Wₕ, ones(n)), v)))
            @test diag(Jm) ≈ 2 .* measures .* u0

            # and the undifferentiated path is untouched
            @test eltype(residual(u0)) === Float64
        end

        @testset "and with the constraints applied, as a solve would" begin
            bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> 0.0))
            res(u) = assemble(form(Wₕ, v -> innerₕ(element(Wₕ, u .* u), v));
                dirichlet_conditions = bcs, dirichlet_labels = :bottom)
            Jb = ForwardDiff.jacobian(res, u0)
            @test size(Jb) == (n, n)

            # a pinned value does not depend on the coefficients, so its row is zero —
            # which is what a solver needs in order not to move it
            marked = index_in_marker(Ωₕ, :bottom)
            @test any(marked)
            @test all(iszero, Jb[marked, :])
        end
    end

    @testset "the functor contracts against a vector" begin
        lf = form(Wₕ, v -> innerₕ(uₕ, v))
        b = assemble(lf)
        w = values(Rₕ(Wₕ, x -> 1.0))
        @test lf(w) ≈ sum(b)              # against the all-ones vector, the sum
        @test lf(values(uₕ)) ≈ sum(b .* values(uₕ))
    end
end
