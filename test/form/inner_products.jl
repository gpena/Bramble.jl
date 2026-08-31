using Test
using Bramble
using Bramble: IdentityOperator, TrialFunction, TestFunction, IndexedTrialFunction,
               IndexedTestFunction, LazyOp, BilinearProduct, LinearProduct,
               InnerH, InnerPlus, SourceFunction, SourceVector,
               local_stencil, resolve_ast, is_symbolic, source_number,
               inner_plus, compute_weight, weights, Innerh, Innerplus, values

# The inner products, from construction through to the stencil they evaluate to.
#
# `test/form/operators.jl` covers which weight each product *carries*. This covers the rest
# of the file: the overloads for a number, a function or a grid function on the left, the
# tuple forms, and — the part that matters most — the stencil evaluators, which are the
# path assembly will take and which nothing had run.
#
# Why the coverage figure said nothing useful here. Julia marks a line of a method it never
# compiled the same way it marks a comment, so `inner.jl` reported 100% of 57 tracked lines
# out of 390 while two thirds of its method surface had never been called. The number to
# watch is the tracked count; the way to move it is to call things.

@testset "Inner products" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
        (5, 6), (true, false))
    Wₕ = gridspace(Ωₕ)
    id = IdentityOperator(Wₕ)
    u, v = TrialFunction{2}(), TestFunction{2}()
    uₕ = Rₕ(Wₕ, x -> x[1] + 2x[2])
    I = CartesianIndex(3, 3)
    lin = LinearIndices(Bramble.indices(Ωₕ))[I]

    @testset "the stencil a product evaluates to" begin
        # A bilinear product multiplies the trial stencil by the test stencil and weights
        # the result by the cell measure: every offset pair, with the two coefficients and
        # the weight multiplied together. This is what assembly consumes, and it had never
        # been evaluated.
        b = innerₕ(D₋ₓ(id), D₋ₓ(id))
        st = local_stencil(b, Wₕ, I, nothing, lin)

        left = local_stencil(D₋ₓ(id), Wₕ, I, nothing, lin)
        w = weights(Wₕ, Innerh())[lin]
        @test length(st) == length(left)^2          # the outer product of the two
        @test all(e -> length(e) == 3, st)          # (row offset, column offset, value)

        # the weight really is the one the space holds, and it scales the product
        expected = sort([lc * rc * w for (_, lc) in left for (_, rc) in left])
        @test sort([e[3] for e in st]) ≈ expected

        # a linear product keeps only the right offset, having contracted the left away
        l = innerₕ(uₕ, D₋ₓ(id))
        lst = local_stencil(l, Wₕ, I, nothing, lin)
        @test all(e -> length(e) == 2, lst)
        @test length(lst) == length(left)
    end

    @testset "the weight each kind of product looks up" begin
        # InnerH reads the cell measure; InnerPlus reads that direction's staggered weight.
        @test compute_weight(InnerH(), Wₕ, I, lin) == weights(Wₕ, Innerh())[lin]
        for dim in 1:2
            @test compute_weight(InnerPlus{dim}(), Wₕ, I, lin) ==
                  weights(Wₕ, Innerplus(), dim)[lin]
        end

        # and the product's own stencil picks up whichever of them its type names
        for (node, wt) in ((innerₕ(id, id), weights(Wₕ, Innerh())[lin]),
            (inner₊ₓ(id, id), weights(Wₕ, Innerplus(), 1)[lin]),
            (inner₊ᵧ(id, id), weights(Wₕ, Innerplus(), 2)[lin]))
            st = local_stencil(node, Wₕ, I, nothing, lin)
            @test only(st)[3] ≈ wt
        end
    end

    @testset "a number, a function or a grid function on the left" begin
        # Each builds a LinearProduct wrapping the left operand in the right source node,
        # which is what lets a right-hand side be assembled.
        for (mk, T) in (((x -> x[1] + 1), SourceFunction), (3.5, SourceFunction),
            (uₕ, SourceVector))
            for f in (innerₕ, inner₊, inner₊ₓ, inner₊ᵧ, inner₊₂)
                p = f(mk, v)
                @test p isa LinearProduct
                @test p.left_op isa T
            end
        end

        # a number becomes a constant function, not a stored vector
        sf = source_number(7.25, Val(2))
        @test sf isa SourceFunction{2}
        @test only(local_stencil(sf, Wₕ, I, nothing, lin))[2] == 7.25

        # and the grid function's coefficients are carried by reference, read at the point
        p = innerₕ(uₕ, v)
        @test p.left_op.vec === values(uₕ)
        @test only(local_stencil(p.left_op, Wₕ, I, nothing, lin))[2] == values(uₕ)[lin]
    end

    @testset "the directional spellings, for every kind of left operand" begin
        # inner₊₂ in particular had no test at all, for any left operand.
        for f in (inner₊ₓ, inner₊ᵧ, inner₊₂)
            @test f(id, id) isa BilinearProduct
            @test f((x -> 1.0), v) isa LinearProduct
            @test f(2.0, v) isa LinearProduct
            @test f(uₕ, v) isa LinearProduct
        end

        # each names its own direction
        @test typeof(inner₊ₓ(id, id)).parameters[2] === InnerPlus{1}
        @test typeof(inner₊ᵧ(id, id)).parameters[2] === InnerPlus{2}
        @test typeof(inner₊₂(id, id)).parameters[2] === InnerPlus{3}
    end

    @testset "the tuple forms" begin
        # a gradient tuple against a gradient tuple: one product per direction, summed
        g = inner₊(∇₋ₕ(u), ∇₋ₕ(v))
        @test g === inner_plus(∇₋ₕ(u), ∇₋ₕ(v))
        @test g isa Bramble.OperatorAdd

        # a tuple of scalars, functions or grid functions on the left, against a gradient
        for l in ((2.0, 3.0), ((x -> x[1]), (x -> x[2])), (uₕ, uₕ))
            p = inner₊(l, ∇₋ₕ(v))
            @test p isa Bramble.OperatorAdd
            @test is_symbolic(p)
        end

        # tuples of gradient tuples — a velocity field against a velocity field
        vec_trial = (IndexedTrialFunction{2}(1), IndexedTrialFunction{2}(2))
        vec_test = (IndexedTestFunction{2}(1), IndexedTestFunction{2}(2))
        p = inner₊(map(∇₋ₕ, vec_trial), map(∇₋ₕ, vec_test))
        @test p isa Bramble.OperatorAdd
        @test is_symbolic(p)

        # every pair of empty tuples ties without the disambiguator, so it is an error
        # rather than an ambiguity
        @test_throws ArgumentError inner₊((), ())
    end

    @testset "a tuple against non-symbolic operators is refused" begin
        # This branch used to read `first(l).values`, where a VectorElement stores `data`,
        # and call `inner₊!`, which no revision of the package defines — two names that
        # could never resolve, in a branch nothing reached. It is entered when the right
        # side carries no trial or test function, so there is nothing for the product to be
        # a form in, and it now says that.
        concrete = ∇₋ₕ(id)
        @test !is_symbolic(concrete)
        @test_throws ArgumentError inner₊((uₕ, uₕ), concrete)

        msg = try
            inner₊((uₕ, uₕ), concrete)
        catch e
            sprint(showerror, e)
        end
        @test occursin("no trial or test function", msg)
        @test occursin("∇₋ₕ(u)", msg)
    end

    @testset "resolving a product resolves both sides" begin
        # The products were the only nodes whose resolve_ast had never run.
        b = innerₕ(D₋ₓ(u), D₋ₓ(v))
        rb = resolve_ast(b)
        @test rb isa BilinearProduct
        @test typeof(rb).parameters[2] === InnerH

        l = innerₕ(uₕ, D₋ₓ(v))
        rl = resolve_ast(l)
        @test rl isa LinearProduct
        @test typeof(rl).parameters[2] === InnerH

        # a thunk on the left is called once while resolving, as it is elsewhere
        vals = collect(1.0:Float64(ndofs(Wₕ)))
        thunked = innerₕ(D₋ₓ((() -> vals) * id), D₋ₓ(v))
        @test resolve_ast(thunked) isa BilinearProduct

        # and resolving is idempotent
        @test resolve_ast(rb) isa BilinearProduct
    end

    @testset "products are symbolic, and carry that upwards" begin
        @test is_symbolic(innerₕ(u, v))
        @test is_symbolic(innerₕ(uₕ, v))
        @test is_symbolic(inner₊ₓ(2.0, v))
        @test is_symbolic(innerₕ(u, v) + innerₕ(D₋ₓ(u), D₋ₓ(v)))
    end
end
