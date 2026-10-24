module FormInnerProductsTests

using Test
using Bramble
using LinearAlgebra: Diagonal, diag, dot
using ..TestUtils: @test_allocs, WITH_3D_TESTS
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
               D₋ᵧ,
               inner₊ᵧ,
               inner₊₂,
               inner₊ₓ,
               restrict_to,
               resolve_form_ast,
               πₕ

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
            for f in (innerₕ, (l, r) -> inner₊(l, D₋ᵧ(r)), inner₊ₓ, inner₊ᵧ)
                p = f(mk, v)
                @test p isa LinearProduct
                @test p.left_op isa T
            end
            # z is not a direction of this 2D test function (#692)
            @test_throws ArgumentError inner₊₂(mk, v)

            # a bare source is the source node it stands for: the direction comes from
            # the difference, markers included, and without one 2D inner₊ has none
            p = inner₊(mk, D₋ᵧ(v))
            @test typeof(p).parameters[2] === InnerPlus{2}
            @test typeof(inner₊(mk, D₋ᵧ(v); markers = (:bottom,))) ===
                  typeof(inner₊(p.left_op, D₋ᵧ(v); markers = (:bottom,)))
            @test_throws ArgumentError inner₊(mk, v)
        end

        # A number becomes a SourceConstant, not a function wrapper or a stored vector:
        # skips point(m, I) entirely rather than computing and discarding it.
        sf = source_number(7.25, Val(2))
        @test sf isa SourceConstant{2}
        @test only(local_stencil(sf, Wₕ, I, nothing, lin))[2] == 7.25

        # and the grid function itself is carried by reference, its space kept so `form`
        # can check the source's mesh, and read at the point
        p = innerₕ(uₕ, v)
        @test p.left_op.vec === uₕ
        @test only(local_stencil(p.left_op, Wₕ, I, nothing, lin))[2] == parent(uₕ)[lin]
    end

    @testset "Directional spellings" begin
        # inner₊₂ in particular had no test at all, for any left operand.
        for f in (inner₊ₓ, inner₊ᵧ)
            @test f(id, id) isa BilinearProduct
        end
        # Building a 3D node needs no 3D mesh, so this runs in the unit group.
        u3, v3 = TrialFunction{3}(), TestFunction{3}()
        p3 = inner₊₂(u3, v3)
        @test p3 isa BilinearProduct
        @test typeof(p3).parameters[2] === InnerPlus{3}
        @test typeof(inner₊₂(x -> x[3], v3)).parameters[2] === InnerPlus{3}
        @test typeof(inner₊(u3, v3, 3)).parameters[2] === InnerPlus{3}
    end

    @testset "Direction beyond D refused (#692)" begin
        # These used to build and then throw a BoundsError from the weight lookup at
        # assembly. They now refuse when the node is built, with the numeric twin's message.
        u1, v1 = TrialFunction{1}(), TestFunction{1}()
        Wₕ1 = gridspace(mesh(domain(interval(0.0, 1.0)), 6, false))
        refusals = (
            () -> inner₊ᵧ(u1, v1),
            () -> inner₊(u1, v1, 2),
            () -> inner₊(u1, v1, :y),
            () -> inner₊ᵧ(x -> x[1], v1),
            () -> inner₊ᵧ(2.0, v1),
            () -> inner₊₂(u, v),
            () -> inner₊(u, v, 3),
            () -> inner₊₂(uₕ, v),
            () -> form(Wₕ1, Wₕ1, (a, b) -> inner₊ᵧ(a, b))
        )
        for f in refusals
            @test_throws ArgumentError f()
        end
        msg = try
            inner₊ᵧ(u1, v1)
        catch e
            sprint(showerror, e)
        end
        @test occursin("S = (2,)", msg)
        @test occursin("not a subset of 1:1", msg)

        # the directions that exist still build, and assemble
        @test typeof(inner₊ₓ(u1, v1)).parameters[2] === InnerPlus{1}
        @test typeof(inner₊ᵧ(u, v)).parameters[2] === InnerPlus{2}
        @test typeof(inner₊(u, v, 2)).parameters[2] === InnerPlus{2}
        @test assemble(form(Wₕ1, Wₕ1, (a, b) -> inner₊(a, b, 1))) ==
              assemble(form(Wₕ1, Wₕ1, (a, b) -> inner₊ₓ(a, b)))
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

    @testset "Scaled gradient infers (#708)" begin
        # `κ * ∇ₕ(u)` is a tuple of differences in different directions, so its entries
        # differ in type; the sum must still infer, with weight `InnerPlus{d}` on term `d`,
        # and assemble to the directional products added by hand. Non-uniform mesh.
        Ωn = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (false, false))
        Bramble.change_points!(Ωn, ([0.0, 0.1, 0.35, 0.5, 0.8, 1.0], [0.0, 0.2, 0.25, 0.5, 0.6, 0.9, 1.0]))
        Wn = gridspace(Ωn)
        κ = Rₕ(Wn, x -> 1 + x[1] + x[2]^2)
        p = @inferred inner₊(κ * ∇ₕ(u), ∇ₕ(v))
        @test typeof(p.left_op).parameters[2] === InnerPlus{1}
        @test typeof(p.right_op).parameters[2] === InnerPlus{2}
        @inferred inner₊(∇ₕ(u), ∇ₕ(v))
        A = assemble(form(Wn, Wn, (a, b) -> inner₊(κ * ∇ₕ(a), ∇ₕ(b))))
        B = assemble(form(Wn, Wn, (a, b) -> inner₊ₓ(κ * D₋ₓ(a), D₋ₓ(b)) + inner₊ᵧ(κ * D₋ᵧ(a), D₋ᵧ(b))))
        @test A ≈ B && !iszero(A)
        if WITH_3D_TESTS
            u3, v3 = TrialFunction{3}(), TestFunction{3}()
            I1 = interval(0.0, 1.0)
            W3 = gridspace(mesh(domain(I1 × I1 × I1), (4, 4, 4), (false, false, false)))
            @inferred inner₊(Rₕ(W3, x -> 1 + x[1]) * ∇ₕ(u3), ∇ₕ(v3))
        end
    end

    @testset "Symbolic and numeric families stay apart" begin
        # `inner₊` names two different things. Given grid functions it computes a number
        # (`src/space/inner_product.jl`), and given operators it builds an AST node
        # (`src/operators/inner.jl`). What keeps the two families from colliding is the
        # `NTuple{N,<:Tuple}` restriction on the symbolic tuple overload.
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
# The general staggered-set entry point.
@testset "inner₊(u, v, Val(S))" begin
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

        # A singleton S is inner₊ₓ/inner₊ᵧ's own node.
        @testset "Singleton S: inner₊ₓ/inner₊ᵧ's node" begin
            px = inner₊(u2, v2, Val((1,)))
            @test typeof(px).parameters[2] === InnerPlus{1}
            @test assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊(u, v, Val((1,))))) ==
                  assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊ₓ(u, v)))

            py = inner₊(u2, v2, Val((2,)))
            @test typeof(py).parameters[2] === InnerPlus{2}
            @test assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊(u, v, Val((2,))))) ==
                  assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊ᵧ(u, v)))
        end

        # A new node, read through SeparableWeights.
        @testset "S = (1, 2): InnerPlusSet node" begin
            S = (1, 2)
            @test weights(Wₕ2, Val(S)) isa SeparableWeights

            p = inner₊(D₋ₓ(u2), D₋ₓ(v2), Val(S))
            @test typeof(p).parameters[2] === InnerPlusSet{S}

            A = assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v), Val(S))))
            Dx = D₋ₓ(Ωₕ2)
            wS = hand_built_weight(Ωₕ2, S, dims2)
            @test A ≈ Dx' * Diagonal(wS) * Dx
        end

        # Val((1,2)) and Val((2,1)) build the same node.
        @testset "S order does not matter" begin
            p12 = inner₊(u2, v2, Val((1, 2)))
            p21 = inner₊(u2, v2, Val((2, 1)))
            @test typeof(p12) === typeof(p21)
            @test typeof(p12).parameters[2] === InnerPlusSet{(1, 2)}
        end

        @testset "an invalid staggered set throws" begin
            @test_throws ArgumentError inner₊(u2, v2, Val((1, 3)))   # 3 is out of range for D=2
            @test_throws ArgumentError inner₊(u2, v2, Val((1, 1)))   # repeated axis
        end

        # `true == 1`, so a Bool would pass the range check as the first axis (#861).
        @testset "a Bool direction throws" begin
            @test_throws ArgumentError inner₊(u2, v2, true)
            @test_throws ArgumentError inner₊(u2, v2, false)
            @test_throws ArgumentError inner₊(u2, v2, Val((true,)))
            @test_throws ArgumentError inner₊(u2, v2, Val((true, 2)))
            @test_throws ArgumentError inner₊(u2, v2, Val((2, true)))
        end

        # Any other integer type builds the node the `Int` spelling builds (#861).
        @testset "a non-Int integer S builds the Int node" begin
            @test typeof(inner₊(u2, v2, Int32(2))).parameters[2] === InnerPlus{2}
            @test typeof(inner₊(u2, v2, Val((Int32(2),)))).parameters[2] === InnerPlus{2}
            @test typeof(inner₊(u2, v2, Val((UInt8(2), Int32(1))))).parameters[2] ===
                  InnerPlusSet{(1, 2)}
            @test assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊(u, v, Val((Int32(1), Int32(2)))))) ==
                  assemble(form(Wₕ2, Wₕ2, (u, v) -> inner₊(u, v, Val((1, 2)))))
        end
    end

    # 3D only: runs nightly (`slow`).
    WITH_3D_TESTS && @testset "3D" begin
        Ω3 = domain(box((0.0, 0.0, 0.0), (0.5, 0.6, 0.7)))
        Ωₕ3 = mesh(Ω3, (4, 5, 3), (false, false, false))
        Wₕ3 = gridspace(Ωₕ3)
        u3, v3 = TrialFunction{3}(), TestFunction{3}()
        dims3 = npoints(Ωₕ3, Tuple)

        # Read through SeparableWeights.
        @testset "S = (1, 2, 3): full set, InnerPlusSet" begin
            S = (1, 2, 3)
            @test weights(Wₕ3, Val(S)) isa SeparableWeights

            p = inner₊(u3, v3, Val(S))
            @test typeof(p).parameters[2] === InnerPlusSet{S}

            A = assemble(form(Wₕ3, Wₕ3, (u, v) -> inner₊(u, v, Val(S))))
            wS = hand_built_weight(Ωₕ3, S, dims3)
            @test A ≈ Diagonal(wS)
        end

        # A genuine pair S = (1, 2) also goes through SeparableWeights.
        @testset "Pair S = (1, 2) uses SeparableWeights" begin
            S = (1, 2)
            @test weights(Wₕ3, Val(S)) isa SeparableWeights

            A = assemble(form(Wₕ3, Wₕ3, (u, v) -> inner₊(u, v, Val(S))))
            wS = hand_built_weight(Ωₕ3, S, dims3)
            @test A ≈ Diagonal(wS)
        end
    end

    # which(...).file is this file, not inner_product.jl.
    @testset "New arity stays in this file" begin
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

    @testset "Bilinear term: surface mass matrix" begin
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

    # As anywhere else.
    @testset "A coefficient scales it" begin
        Wₕ = gridspace(mesh(Ω, (7, 7), (true, true)))
        β = 1.7
        A = assemble(form(Wₕ, Wₕ, (u, v) -> inner_Γ(β * u, v; markers = (:ymax,))))
        B = assemble(form(Wₕ, Wₕ, (u, v) -> inner_Γ(u, v; markers = (:ymax,))))
        @test A ≈ β * B
    end

    @testset "Linear term: Neumann flux vector" begin
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

    # Refilling in place changes nothing, and allocates nothing.
    @testset "Refill in place: same, no allocation" begin
        Wₕ = gridspace(mesh(Ω, (9, 9), (true, true)))
        a = form(Wₕ, Wₕ,
            (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + inner_Γ(u, v; markers = (:ymin, :ymax)))
        A = assemble(a)
        B = copy(A)
        assemble!(B, a)
        @test A ≈ B
        @test_allocs assemble!(B, a)
    end

    # A label is read from its definition at `form`, as the numeric `inner_Γ` reads it
    # (#660). On a non-uniform mesh the lumped weight integrates g = 1 + x + 2y exactly on
    # each edge: 2 on x = 0, 3 on x = 1, 1.5 on y = 0, 3.5 on y = 1.
    @testset "Labels read as defined (#660)" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        mk(lab...) = gridspace(mesh(domain(S, lab..., :boundary => boundary_symbols(S)),
            (11, 9), (false, false)))
        g(x) = 1 + x[1] + 2x[2]
        lin(W, m) = assemble(form(W, v -> inner_Γ(g, v; markers = m)))
        bil(W, m) = assemble(form(W, W, (u, v) -> inner_Γ(u, v; markers = m)))
        ig(W, l...) = inner_Γ(Rₕ(W, g), element(W, 1.0), l...)

        W = mk(:left => :ymin)
        @test lin(W, :left) == lin(W, :ymin)
        @test sum(lin(W, :left)) ≈ ig(W, :left) ≈ 1.5
        @test sum(lin(W, :xmin)) ≈ 2.0
        A = bil(W, :left)
        @test A == bil(W, :ymin)
        uₕ = Rₕ(W, x -> sin(x[1]) + x[2])
        vₕ = Rₕ(W, x -> cos(x[2]) * x[1] + 1.0)
        @test dot(parent(vₕ), A * parent(uₕ)) ≈ inner_Γ(uₕ, vₕ, :left)
        # a user-named whole face, and with a face name beside it
        Wi = mk(:inflow => :xmin)
        @test lin(Wi, :inflow) == lin(Wi, :xmin)
        @test sum(lin(Wi, :inflow)) ≈ ig(Wi, :inflow) ≈ 2.0
        @test sum(lin(Wi, (:inflow, :ymax))) ≈ ig(Wi, :inflow, :ymax) ≈ 2.0 + 3.5
        # two half-face markers select the face together
        Wa = mk(:a => x -> x[1] == 0 && x[2] <= 0.5, :b => x -> x[1] == 0 && x[2] >= 0.5)
        @test lin(Wa, (:a, :b)) == lin(Wa, :xmin)
        @test sum(lin(Wa, (:a, :b))) ≈ ig(Wa, :a, :b) ≈ 2.0
        # an open face, its two corners left out (D-E6)
        Wo = mk(:right => x -> x[1] == 1 && 0 < x[2] < 1)
        @test lin(Wo, :right) == lin(Wo, :xmax)
        @test sum(lin(Wo, :right)) ≈ ig(Wo, :right) ≈ 3.0
        # a name-agreeing marker changes nothing
        Wn = mk(:left => :xmin)
        @test lin(Wn, :left) == lin(Wn, :xmin)
        @test bil(Wn, :left) == bil(Wn, :xmin)
        # a face plus half of another is refused at `form`, by the numeric path's error
        Wc = mk(:c => x -> x[1] == 0 || (x[2] == 0 && x[1] <= 0.5))
        @test_throws ArgumentError lin(Wc, :c)
        @test_throws "whole boundary faces" lin(Wc, :c)
        @test_throws "whole boundary faces" bil(Wc, :c)

        # two spellings of one face assemble as the same face
        a2 = form(W, W, (u, v) -> inner_Γ(u, v; markers = :left) +
                                  inner_Γ(u, v; markers = :ymin))
        @test assemble(a2) ≈ 2A
        @test !Bramble._has_surface_labels(resolve_form_ast(a2))
    end

    @testset "Labels under a wrapper (#660)" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        W = gridspace(mesh(domain(S, :left => :ymin, :boundary => boundary_symbols(S)),
            (11, 9), (false, false)))
        Ay = assemble(form(W, W, (u, v) -> inner_Γ(u, v; markers = :ymin)))
        @test assemble(form(W, W,
            (u, v) -> restrict_to(:boundary, inner_Γ(u, v; markers = :left)))) == Ay
        w = fill(2.0, ndofs(W))
        @test assemble(form(W, W, (u, v) -> w * inner_Γ(u, v; markers = :left))) ≈ 2Ay
        @test assemble(form(W, W, (u, v) -> -inner_Γ(u, v; markers = :left))) ≈ -Ay
        g(x) = 1 + x[1] + 2x[2]
        Fy = assemble(form(W, v -> inner_Γ(g, v; markers = :ymin)))
        Fr = assemble(form(W, v -> restrict_to(:boundary, inner_Γ(g, v; markers = :left))))
        @test Fr == Fy
        @test assemble(form(W, v -> 3.0 * inner_Γ(g, v; markers = :left))) ≈ 3Fy
    end

    # The mesh is the test leaf's: only `W1` reads `:left` as y = 0.
    @testset "Labels on a composite (#660)" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        g(x) = 1 + x[1] + 2x[2]
        W1 = gridspace(mesh(domain(S, :left => :ymin, :boundary => boundary_symbols(S)),
            (7, 6), (false, false)))
        W2 = gridspace(mesh(domain(S, :boundary => boundary_symbols(S)),
            (6, 8), (false, false)))
        n = ndofs(W2)
        F2 = assemble(form(W2 × W1, v -> inner_Γ(g, v(2); markers = :left)))
        @test sum(F2[(n + 1):end]) ≈ 1.5
        @test iszero(F2[1:n])
        F1 = assemble(form(W2 × W1, v -> inner_Γ(g, v(1); markers = :xmin)))
        @test sum(F1[1:n]) ≈ 2.0
        @test iszero(F1[(n + 1):end])
        # a term naming no component reaches both leaves, which read `:left` apart
        @test_throws ArgumentError form(W2 × W1, v -> inner_Γ(g, v; markers = :left))
        @test_throws "different faces" form(W2 × W1, v -> inner_Γ(g, v; markers = :left))
        # where they agree it reaches both
        Fb = assemble(form(W2 × W1, v -> inner_Γ(g, v; markers = :ymin)))
        @test sum(Fb[1:n]) ≈ 1.5
        @test sum(Fb[(n + 1):end]) ≈ 1.5
    end

    # Under a test-side `πₕ` the sweep walks the trial leaf, so a label is read there: on a
    # trial mesh holding it, against a test mesh that does not, it is its marked face.
    @testset "Labels on the walked mesh (#660)" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        mk(n, lab...) = gridspace(mesh(domain(S, lab..., :boundary => boundary_symbols(S)),
            n, (false, false)))
        tp(m) = (u, v) -> inner_Γ(u, πₕ(v); markers = m)
        Vt = mk((7, 6))
        Wi = mk((11, 9), :inflow => :xmin)
        @test assemble(form(Wi, Vt, tp(:inflow))) == assemble(form(Wi, Vt, tp(:xmin)))
        Wt = mk((11, 9), :left => :ymin)
        Al = assemble(form(Wt, Vt, tp(:left)))
        @test Al == assemble(form(Wt, Vt, tp(:ymin)))
        @test Al != assemble(form(Wt, Vt, tp(:xmin)))
        # on a composite, the walked leaf is the trial component the term names
        W2 = mk((9, 10))
        tc(k, m) = (u, v) -> inner_Γ(u(k), πₕ(v(k)); markers = m)
        A2 = assemble(form(W2 × Wt, Vt × Vt, tc(2, :left)))
        @test A2 == assemble(form(W2 × Wt, Vt × Vt, tc(2, :ymin)))
        A1 = assemble(form(W2 × Wt, Vt × Vt, tc(1, :left)))
        @test A1 == assemble(form(W2 × Wt, Vt × Vt, tc(1, :xmin)))
        @test_throws "different faces" form(W2 × Wt, Vt × Vt, tp(:left))
        # a term naming one side's component alone meets the block routing's own refusal
        X = Vt × Wt
        @test_throws "names its trial component but not its test one" assemble(
            form(X, X, (u, v) -> inner_Γ(u(1), v; markers = :left)))
        @test_throws "names its test component but not its trial one" assemble(
            form(X, X, (u, v) -> inner_Γ(u, πₕ(v(2)); markers = :left)))
    end

    @testset "Labels refill in place (#660)" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        W = gridspace(mesh(domain(S, :left => :ymin, :boundary => boundary_symbols(S)),
            (11, 9), (false, false)))
        g(x) = 1 + x[1] + 2x[2]
        l = form(W, v -> inner_Γ(g, v; markers = :left))
        b = @inferred assemble(l)
        assemble!(b, l)
        @test_allocs assemble!(b, l)
        a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + inner_Γ(u, v; markers = :left))
        B = @inferred assemble(a)
        assemble!(B, a)
        @test_allocs assemble!(B, a)
    end

    # An empty label set is refused when the node is built; a label the mesh neither holds
    # nor reads as a face name is refused at `form`, where the mesh is known.
    @testset "Construction refusals" begin
        u, v = TrialFunction{2}(), TestFunction{2}()
        @test_throws ArgumentError inner_Γ(u, v)
        Wₕ = gridspace(mesh(Ω, (5, 5), (true, true)))
        @test_throws ArgumentError form(Wₕ, Wₕ,
            (u, v) -> inner_Γ(u, v; markers = (:inlet,)))
        @test_throws ArgumentError form(Wₕ, v -> inner_Γ(1.0, v; markers = :inlet))
    end

    # On an axis with one point a marker's points cover both faces of that axis. The numeric
    # `inner_Γ` refuses that face set, since every grid point lies on it; the symbolic one
    # assembles every face selected, as `:boundary` does, and only the face name names one.
    @testset "A marker on a one-point axis" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        W = gridspace(mesh(domain(S, :l => :ymin), (5, 1), (false, false)))
        oₕ = Rₕ(W, x -> 1.0)
        @test_throws "not (D-1)-dimensional" inner_Γ(oₕ, oₕ, :l)
        @test_throws "not (D-1)-dimensional" inner_Γ(oₕ, oₕ, :boundary)
        @test inner_Γ(oₕ, oₕ, :ymin) ≈ 1.0

        lin(m) = assemble(form(W, v -> inner_Γ(x -> 1.0, v; markers = m)))
        bil(m) = assemble(form(W, W, (u, v) -> inner_Γ(u, v; markers = m)))
        Fy = lin(:ymin)
        @test sum(Fy) ≈ inner_Γ(oₕ, oₕ, :ymin)
        @test lin(:l) == lin(:boundary)
        @test lin(:l) ≈ Fy + lin((:xmin, :xmax))
        @test findall(!iszero, lin(:l)) == 1:5
        @test bil(:l) == bil(:boundary)
    end
end

@testset "One-sided block term (#356)" begin
    # A term naming the trial component alone builds; assembling it is what refuses.
    Ωc = mesh(domain(interval(0.0, 1.0)), 9, false)
    V = gridspace(Ωc, Val(2))
    a = form(V, V, (u, v) -> innerₕ(u[1], v))
    @test_throws ArgumentError assemble(a)
end

end # module FormInnerProductsTests
