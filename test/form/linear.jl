using Test
using Bramble
using Bramble: LinearForm, form, assemble, assemble!, assemble_parallel!, test_space,
               resolve_form_ast, apply_dirichlet_conditions!, LinearProduct, values,
               ParallelWorkspace

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

    @testset "the parallel path agrees with the serial one" begin
        # It is not the default, and the docstring says why — it was slower at every size
        # measured. It still has to give the same answer.
        for (nm, sp, u) in (("scalar", Wₕ, uₕ), (
            "composite", Vₕ, Rₕ(Vₕ, (x -> sin(x[1]), x -> cos(x[2])))(1)))
            lf = form(sp, v -> innerₕ(u, v))
            bs = assemble(lf)
            bp = similar(bs)
            @test assemble_parallel!(bp, lf) === bp
            @test bp ≈ bs
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

    @testset "the functor contracts against a vector" begin
        lf = form(Wₕ, v -> innerₕ(uₕ, v))
        b = assemble(lf)
        w = values(Rₕ(Wₕ, x -> 1.0))
        @test lf(w) ≈ sum(b)              # against the all-ones vector, the sum
        @test lf(values(uₕ)) ≈ sum(b .* values(uₕ))
    end
end
