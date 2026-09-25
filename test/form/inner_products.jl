module FormInnerProductsTests

using Test
using Bramble
using LinearAlgebra: Diagonal, diag, dot
using ..TestUtils: @test_allocs
using Bramble:
               IdentityOperator,
               TrialFunction,
               TestFunction,
               IndexedTrialFunction,
               IndexedTestFunction,
               LazyOp,
               BilinearProduct,
               LinearProduct,
               InnerH,
               InnerPlus,
               InnerPlusSet,
               SourceFunction,
               SourceVector,
               SourceConstant,
               local_stencil,
               resolve_ast,
               is_symbolic,
               source_number,
               inner_plus,
               compute_weight,
               weights,
               Innerh,
               Innerplus,
               SeparableWeights,
               spacing,
               half_spacing,
               values,
               D₋ₓ,
               inner₊ᵧ,
               inner₊₂,
               inner₊ₓ

# The inner products, from construction through to the stencil they evaluate to.
#
# `test/form/operators.jl` covers which weight each product *carries*. This covers the rest
# of the file: the overloads for a number, a function or a grid function on the left, the
# tuple forms, and (the part that matters most) the stencil evaluators, which are the
# path assembly will take and which nothing had run.
#
# Why the coverage figure said nothing useful here. Julia marks a line of a method it never
# compiled the same way it marks a comment, so `inner.jl` reported 100% of 57 tracked lines
# out of 390 while two thirds of its method surface had never been called. The number to
# watch is the tracked count; the way to move it is to call things.

@testset "Inner products" begin
    Ωₕ = mesh(
        domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
        (5, 6),
        (true, false)
    )
    Wₕ = gridspace(Ωₕ)
    id = IdentityOperator(Wₕ)
    u, v = TrialFunction{2}(), TestFunction{2}()
    uₕ = Rₕ(Wₕ, x -> x[1] + 2x[2])
    I = CartesianIndex(3, 3)
    lin = LinearIndices(Bramble.indices(Ωₕ))[I]

    @testset "Product stencils" begin
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

    @testset "Weight lookups" begin
        # InnerH reads the cell measure; InnerPlus reads that direction's staggered weight.
        @test compute_weight(InnerH(), Wₕ, I, lin) == weights(Wₕ, Innerh())[lin]
        for dim in 1:2
            @test compute_weight(InnerPlus{dim}(), Wₕ, I, lin) ==
                  weights(Wₕ, Innerplus(), dim)[lin]
        end

        # and the product's own stencil picks up whichever of them its type names
        for (node, wt) in (
            (innerₕ(id, id), weights(Wₕ, Innerh())[lin]),
            (inner₊ₓ(id, id), weights(Wₕ, Innerplus(), 1)[lin]),
            (inner₊ᵧ(id, id), weights(Wₕ, Innerplus(), 2)[lin])
        )
            st = local_stencil(node, Wₕ, I, nothing, lin)
            @test only(st)[3] ≈ wt
        end
    end

    @testset "Left operands" begin
        # Each builds a LinearProduct wrapping the left operand in the right source node,
        # which is what lets a right-hand side be assembled.
        for (mk, T) in (((x -> x[1] + 1), SourceFunction), (3.5, SourceConstant), (uₕ, SourceVector))
            for f in (innerₕ, inner₊, inner₊ₓ, inner₊ᵧ, inner₊₂)
                p = f(mk, v)
                @test p isa LinearProduct
                @test p.left_op isa T
            end
        end

        # A number becomes a SourceConstant, not a function wrapper or a stored vector:
        # skips point(m, I) entirely rather than computing and discarding it.
        sf = source_number(7.25, Val(2))
        @test sf isa SourceConstant{2}
        @test only(local_stencil(sf, Wₕ, I, nothing, lin))[2] == 7.25

        # and the grid function's coefficients are carried by reference, read at the point
        p = innerₕ(uₕ, v)
        @test p.left_op.vec === parent(uₕ)
        @test only(local_stencil(p.left_op, Wₕ, I, nothing, lin))[2] == parent(uₕ)[lin]
    end

    @testset "Directional spellings" begin
        # inner₊₂ in particular had no test at all, for any left operand.
        for f in (inner₊ₓ, inner₊ᵧ, inner₊₂)
            @test f(id, id) isa BilinearProduct
        end

        # each names its own direction
        @test typeof(inner₊ₓ(id, id)).parameters[2] === InnerPlus{1}
        @test typeof(inner₊ᵧ(id, id)).parameters[2] === InnerPlus{2}
        @test typeof(inner₊₂(id, id)).parameters[2] === InnerPlus{3}
    end

    @testset "Tuple forms" begin
        # a tuple of scalars, functions or grid functions on the left, against a gradient
        for l in ((2.0, 3.0), ((x -> x[1]), (x -> x[2])), (uₕ, uₕ))
            p = inner₊(l, ∇ₕ(v))
            @test p isa Bramble.OperatorAdd
            @test is_symbolic(p)
        end

        # tuples of gradient tuples: a velocity field against a velocity field
        vec_trial = (IndexedTrialFunction{2}(1), IndexedTrialFunction{2}(2))
        vec_test = (IndexedTestFunction{2}(1), IndexedTestFunction{2}(2))
        p = inner₊(map(∇ₕ, vec_trial), map(∇ₕ, vec_test))
        @test p isa Bramble.OperatorAdd
        @test is_symbolic(p)

        # every pair of empty tuples ties without the disambiguator, so it is an error
        # rather than an ambiguity
        @test_throws ArgumentError inner₊((), ())
    end

    @testset "Symbolic and numeric families stay apart" begin
        # `inner₊` names two different things. Given grid functions it computes a number
        # (`src/space/inner_product.jl`); given operators it builds an AST node
        # (`src/form/operators/inner.jl`). Neither file says so from where a reader of it
        # is standing, and what keeps the two families from colliding is the
        # `NTuple{N,<:Tuple}` restriction on the symbolic tuple overload — recorded until
        # now only in a comment inside one of the two files (gpena/Bramble.jl#60).
        #
        # These assert the *resolution* rather than the result, so widening either
        # signature fails here instead of silently returning the wrong kind of thing.
        # `which` is checked by file rather than by line, which moves.
        numeric_file(T) = basename(String(which(inner₊, T).file))

        @test numeric_file(Tuple{typeof(uₕ), typeof(uₕ)}) == "inner_product.jl"
        @test numeric_file(Tuple{typeof((uₕ, uₕ)), typeof((uₕ, uₕ))}) == "inner_product.jl"

        grads = map(∇ₕ, (IndexedTrialFunction{2}(1), IndexedTrialFunction{2}(2)))
        @test numeric_file(Tuple{typeof(∇ₕ(u)), typeof(∇ₕ(v))}) == "inner.jl"
        @test numeric_file(Tuple{typeof(grads), typeof(grads)}) == "inner.jl"

        # The restriction itself: a tuple of grid functions is not a tuple of tuples, and
        # that is the only reason the symbolic tuple overload does not swallow the numeric
        # one. Widen it to `NTuple{N,Any}` and both of these flip.
        @test !(typeof((uₕ, uₕ)) <: NTuple{2, <:Tuple})
        @test typeof(grads) <: NTuple{2, <:Tuple}

        # And the two families really do return different kinds of thing, which is what
        # makes a mis-resolution worth catching.
        @test inner₊(uₕ, uₕ) isa Real
        @test inner₊(∇ₕ(u), ∇ₕ(v)) isa LazyOp
    end

    @testset "Non-symbolic tuple refusal" begin
        # This branch used to read `first(l).values`, where a VectorElement stores `data`,
        # and call `inner₊!`, which no revision of the package defines (two names that
        # could never resolve, in a branch nothing reached). It is entered when the right
        # side carries no trial or test function, so there is nothing for the product to be
        # a form in, and it now says that.
        concrete = ∇ₕ(id)
        @test !is_symbolic(concrete)
        @test_throws ArgumentError inner₊((uₕ, uₕ), concrete)

        msg = try
            inner₊((uₕ, uₕ), concrete)
        catch e
            sprint(showerror, e)
        end
        @test occursin("no trial or test function", msg)
        @test occursin("∇ₕ(u)", msg)
    end

    @testset "Bilateral resolution" begin
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

    @testset "Symbolic property propagation" begin
        @test is_symbolic(innerₕ(uₕ, v))
        @test is_symbolic(inner₊ₓ(2.0, v))
        @test is_symbolic(innerₕ(u, v) + innerₕ(D₋ₓ(u), D₋ₓ(v)))
    end
end

# The general staggered-set entry point, inner₊(u, v, Val(S)) (gpena/Bramble.jl#115, #234).
#
# `S = ()` and a singleton `S` are aliases for innerₕ/inner₊ₓ,ᵧ,₂'s own nodes -- checked by
# comparing the assembled matrix and the AST's own InnerType parameter, not just the value,
# since the whole point of routing them there is that they are the *same* node, which is
# what lets a term written either way fold the same way in the simplifier and resolve to the
# same `which(inner₊, ...).file`. `|S| >= 2` is genuinely new: `InnerPlusSet`, read at
# assembly through the lazy `SeparableWeights` `weights(Wₕ, Val(S))` returns for those sets
# (never a full-grid vector), which this checks is really what gets exercised, not a silent
# fallback to something dense.
@testset "inner₊(u, v, Val(S)), the general staggered-set entry point" begin
    # Independent of `weights`: the mesh's own spacing/half_spacing, hand-multiplied per
    # axis, exactly as test/space/gridspaces.jl's own `weights(Wₕ, Val(S))` testset checks
    # `SpaceWeights` itself.
    aligned(m, d, i) = i == 1 ? 0.0 : spacing(m(d), i)
    cellfac(m, d, i) = half_spacing(m(d), i)
    function hand_built_weight(m, S, dims)
        w = Vector{Float64}(undef, prod(dims))
        li = LinearIndices(dims)
        for I in CartesianIndices(dims)
            w[li[I]] = prod(d -> (d in S ? aligned(m, d, I[d]) : cellfac(m, d, I[d])), 1:length(dims))
        end
        return w
    end

    @testset "2D" begin
        Ω2 = domain(interval(0.0, 1.0) × interval(0.0, 1.3))
        Ωₕ2 = mesh(Ω2, (7, 6), (false, false))
        Wₕ2 = gridspace(Ωₕ2)
        u2, v2 = TrialFunction{2}(), TestFunction{2}()
        dims2 = npoints(Ωₕ2, Tuple)

        @testset "S = () is innerₕ's own node" begin
            p = inner₊(u2, v2, Val(()))
            @test typeof(p).parameters[2] === InnerH
            @test assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊(u, v, Val(())))) ==
                  assemble(form(Wₕ2, Wₕ2, (u, v) -> innerₕ(u, v)))
        end

        @testset "a singleton S is inner₊ₓ/inner₊ᵧ's own node" begin
            px = inner₊(u2, v2, Val((1,)))
            @test typeof(px).parameters[2] === InnerPlus{1}
            @test assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊(u, v, Val((1,))))) ==
                  assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊ₓ(u, v)))

            py = inner₊(u2, v2, Val((2,)))
            @test typeof(py).parameters[2] === InnerPlus{2}
            @test assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊(u, v, Val((2,))))) ==
                  assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊ᵧ(u, v)))
        end

        @testset "S = (1, 2) is a new InnerPlusSet node, read through SeparableWeights" begin
            S = (1, 2)
            @test weights(Wₕ2, Val(S)) isa SeparableWeights

            p = inner₊(D₋ₓ(u2), D₋ₓ(v2), Val(S))
            @test typeof(p).parameters[2] === InnerPlusSet{S}

            A = assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v), Val(S))))
            Dx = D₋ₓ(Ωₕ2)
            wS = hand_built_weight(Ωₕ2, S, dims2)
            @test A ≈ Dx' * Diagonal(wS) * Dx
        end

        @testset "S order does not matter: Val((1,2)) and Val((2,1)) build the same node" begin
            p12 = inner₊(u2, v2, Val((1, 2)))
            p21 = inner₊(u2, v2, Val((2, 1)))
            @test typeof(p12) === typeof(p21)
            @test typeof(p12).parameters[2] === InnerPlusSet{(1, 2)}
        end

        @testset "an invalid staggered set throws" begin
            @test_throws ArgumentError inner₊(u2, v2, Val((1, 3)))   # 3 is out of range for D=2
            @test_throws ArgumentError inner₊(u2, v2, Val((1, 1)))   # repeated axis
        end
    end

    @testset "3D" begin
        Ω3 = domain(box((0.0, 0.0, 0.0), (0.5, 0.6, 0.7)))
        Ωₕ3 = mesh(Ω3, (4, 5, 3), (false, false, false))
        Wₕ3 = gridspace(Ωₕ3)
        u3, v3 = TrialFunction{3}(), TestFunction{3}()
        dims3 = npoints(Ωₕ3, Tuple)

        @testset "S = (1, 2, 3), the full set, is InnerPlusSet through SeparableWeights" begin
            S = (1, 2, 3)
            @test weights(Wₕ3, Val(S)) isa SeparableWeights

            p = inner₊(u3, v3, Val(S))
            @test typeof(p).parameters[2] === InnerPlusSet{S}

            A = assemble(form(Wₕ3, Wₕ3, (u, v) -> inner₊(u, v, Val(S))))
            wS = hand_built_weight(Ωₕ3, S, dims3)
            @test A ≈ Diagonal(wS)
        end

        @testset "a genuine pair S = (1, 2) also goes through SeparableWeights" begin
            S = (1, 2)
            @test weights(Wₕ3, Val(S)) isa SeparableWeights

            A = assemble(form(Wₕ3, Wₕ3, (u, v) -> inner₊(u, v, Val(S))))
            wS = hand_built_weight(Ωₕ3, S, dims3)
            @test A ≈ Diagonal(wS)
        end
    end

    @testset "which(...).file: the new arity stays in this file, not inner_product.jl" begin
        Wₕ2 = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true)))
        u2, v2 = TrialFunction{2}(), TestFunction{2}()
        method_file(T) = basename(String(which(inner₊, T).file))
        @test method_file(Tuple{typeof(u2), typeof(v2), Val{(1, 2)}}) == "inner.jl"
        @test method_file(Tuple{typeof(u2), typeof(v2), Val{()}}) == "inner.jl"
        @test method_file(Tuple{typeof(u2), typeof(v2), Val{(1,)}}) == "inner.jl"
    end
end

# The symbolic surface integral (gpena/Bramble.jl#157). Every assertion ties the assembled
# matrix or vector to the *numeric* `inner_Γ`, which `test/space/inner_product.jl` in turn
# pins to closed forms -- so the chain ends at an edge length, not at a second copy of this
# code.
@testset "inner_Γ, the symbolic surface integral" begin
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))

    @testset "The bilinear term is the surface mass matrix" begin
        Wₕ = gridspace(mesh(Ω, (9, 8), (true, true)))
        A = assemble(form(Wₕ, Wₕ, (u, v) -> inner_Γ(u, v; markers = (:ymin,))))

        uₕ = Rₕ(Wₕ, x -> sin(x[1]) + x[2])
        vₕ = Rₕ(Wₕ, x -> cos(x[2]) * x[1])
        @test dot(parent(vₕ), A * parent(uₕ)) ≈ inner_Γ(uₕ, vₕ, :ymin)

        # it is diagonal, and carries a nonzero only on the face
        @test A ≈ Diagonal(diag(A))
        one_h = Rₕ(Wₕ, x -> 1.0)
        @test sum(diag(A)) ≈ inner_Γ(one_h, one_h, :ymin)

        # corner sharing survives assembly: the two faces' diagonals add to the union's
        Ax = assemble(form(Wₕ, Wₕ, (u, v) -> inner_Γ(u, v; markers = (:xmin,))))
        Au = assemble(form(Wₕ, Wₕ, (u, v) -> inner_Γ(u, v; markers = (:ymin, :xmin))))
        @test diag(A) + diag(Ax) ≈ diag(Au)

        # a single symbol spells the same thing as a one-tuple
        @test assemble(form(Wₕ, Wₕ, (u, v) -> inner_Γ(u, v; markers = :ymin))) ≈ A
    end

    @testset "A coefficient scales it, as anywhere else" begin
        Wₕ = gridspace(mesh(Ω, (7, 7), (true, true)))
        β = 1.7
        A = assemble(form(Wₕ, Wₕ, (u, v) -> inner_Γ(β * u, v; markers = (:ymax,))))
        B = assemble(form(Wₕ, Wₕ, (u, v) -> inner_Γ(u, v; markers = (:ymax,))))
        @test A ≈ β * B
    end

    @testset "The linear term is the Neumann flux vector" begin
        Wₕ = gridspace(mesh(Ω, (9, 6), (true, true)))
        g(x) = 2.0 + x[1]
        F = assemble(form(Wₕ, v -> inner_Γ(g, v; markers = (:ymin,))))
        vₕ = Rₕ(Wₕ, x -> cos(x[2]) * x[1] + 1.0)
        @test dot(F, parent(vₕ)) ≈ inner_Γ(Rₕ(Wₕ, g), vₕ, :ymin)

        # a number and a grid function on the left work the same way
        @test assemble(form(Wₕ, v -> inner_Γ(2.0, v; markers = (:ymin,)))) ≈
              assemble(form(Wₕ, v -> inner_Γ(x -> 2.0, v; markers = (:ymin,))))
        @test assemble(form(Wₕ, v -> inner_Γ(Rₕ(Wₕ, g), v; markers = (:ymin,)))) ≈ F
    end

    @testset "In 1D it is the endpoint pairing" begin
        Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11, true))
        A = assemble(form(Wₕ, Wₕ, (u, v) -> inner_Γ(u, v; markers = (:boundary,))))
        # ω ≡ 1 at the two endpoints and nowhere else: the matrix is diag(1, 0, …, 0, 1)
        @test diag(A) ≈ [1.0; zeros(ndofs(Wₕ) - 2); 1.0]
    end

    @testset "Refilling in place changes nothing, and allocates nothing" begin
        Wₕ = gridspace(mesh(Ω, (9, 9), (true, true)))
        a = form(Wₕ, Wₕ,
            (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + inner_Γ(u, v; markers = (:ymin, :ymax)))
        A = assemble(a)
        B = copy(A)
        assemble!(B, a)
        @test A ≈ B
        @test_allocs assemble!(B, a)
    end

    @testset "Construction refusals" begin
        u, v = TrialFunction{2}(), TestFunction{2}()
        @test_throws ArgumentError inner_Γ(u, v; markers = (:inlet,))
        @test_throws ArgumentError inner_Γ(u, v)
    end
end

end # module FormInnerProductsTests
