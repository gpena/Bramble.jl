using Test
using Bramble
using Bramble: TrialFunction, TestFunction, IndexedTrialFunction, IndexedTestFunction,
               SourceFunction, SourceVector, IdentityOperator, ZeroOperator, LazyOp,
               OperatorAdd, OperatorScale, GridFunctionScale,
               trial_function, test_function, source_function,
               local_stencil, resolve_ast, is_symbolic,
               zero_offset, shift_offset, get_spacing, get_forward_spacing,
               get_half_spacing, shift_stencil, concatenate_stencils, scale_stencil,
               multiply_stencils_bilinear, multiply_stencils_linear,
               restrict_to, shift_op, values

# The AST leaves, the stencil algebra under them, and the two traits every node answers.
#
# A stencil is a tuple of `(offset, coefficient)` pairs, offsets relative to the point being
# evaluated. Everything in this file either produces one, combines two, or walks a tree of
# nodes that do. The `@generated` combinators are the pieces assembly is built from —
# `multiply_stencils_bilinear` is how a trial stencil and a test stencil become matrix
# entries — and none of them had ever run.

@testset "AST nodes and the stencil algebra" begin
    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 9, false)
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
    Wₕ1, Wₕ = gridspace(Ωₕ1), gridspace(Ωₕ)
    id = IdentityOperator(Wₕ)
    I = CartesianIndex(3, 3)
    lin = LinearIndices(Bramble.indices(Ωₕ))[I]
    O = (0, 0)

    @testset "offsets" begin
        @test zero_offset(Val(1)) == (0,)
        @test zero_offset(Val(2)) == (0, 0)
        @test zero_offset(Val(3)) == (0, 0, 0)

        @test shift_offset((0, 0), 1, 1) == (1, 0)
        @test shift_offset((0, 0), 2, -1) == (0, -1)
        @test shift_offset((2, -3, 4), 3, 5) == (2, -3, 9)
        @test shift_offset((1, 1), 1, 0) == (1, 1)          # a zero shift is identity
    end

    @testset "spacings, per direction" begin
        # In one dimension the mesh answers with a number and in more with a tuple, so the
        # accessors have to pick a component out of either.
        i1 = CartesianIndex(4)
        @test get_spacing(Ωₕ1, i1, 1) == spacing(Ωₕ1, i1)
        @test get_forward_spacing(Ωₕ1, i1, 1) == forward_spacing(Ωₕ1, i1)
        @test get_half_spacing(Ωₕ1, i1, 1) == half_spacing(Ωₕ1, i1)

        for d in 1:2
            @test get_spacing(Ωₕ, I, d) == spacing(Ωₕ, I)[d]
            @test get_forward_spacing(Ωₕ, I, d) == forward_spacing(Ωₕ, I)[d]
            @test get_half_spacing(Ωₕ, I, d) == half_spacing(Ωₕ, I)[d]
        end

        # the two directions of a non-uniform mesh really do differ, so the test is not
        # comparing a number with itself
        @test get_spacing(Ωₕ, I, 1) != get_spacing(Ωₕ, I, 2)
    end

    @testset "combining stencils" begin
        s1 = ((O, 2.0), ((1, 0), 3.0))
        s2 = (((0, 1), 5.0),)

        @test scale_stencil(s1, 10) == ((O, 20.0), ((1, 0), 30.0))
        @test scale_stencil(s1, 0) == ((O, 0.0), ((1, 0), 0.0))
        @test scale_stencil((), 3) == ()

        @test concatenate_stencils(s1, s2) == ((O, 2.0), ((1, 0), 3.0), ((0, 1), 5.0))
        @test concatenate_stencils(s1, ()) == s1
        @test concatenate_stencils((), s2) == s2
        @test concatenate_stencils((), ()) == ()

        # the offsets move, the coefficients do not; the shift is available with the step
        # known to the compiler or not, and the two must agree
        @test shift_stencil(s1, Val(2), Val(1)) == (((0, 1), 2.0), ((1, 1), 3.0))
        @test shift_stencil(s1, Val(2), 1) == shift_stencil(s1, Val(2), Val(1))
        @test shift_stencil(s1, Val(1), Val(-2)) == (((-2, 0), 2.0), ((-1, 0), 3.0))
        @test shift_stencil(s1, Val(1), Val(0)) == s1

        # the outer product a bilinear form assembles from: every trial offset against
        # every test offset, coefficients multiplied and weighted by the cell volume
        b = multiply_stencils_bilinear(s1, s2, 2.0)
        @test length(b) == length(s1) * length(s2)
        @test b == ((O, (0, 1), 20.0), ((1, 0), (0, 1), 30.0))

        # the linear form keeps only the right offset — the left has been contracted away
        l = multiply_stencils_linear(s1, s2, 2.0)
        @test length(l) == length(s1) * length(s2)
        @test l == (((0, 1), 20.0), ((0, 1), 30.0))
        @test all(e -> length(e) == 2, l)
        @test all(e -> length(e) == 3, b)
    end

    @testset "the leaves each evaluate to a unit stencil at the point" begin
        for op in (TrialFunction{2}(), TestFunction{2}(),
            IndexedTrialFunction{2}(1), IndexedTestFunction{2}(2), id)
            @test local_stencil(op, Wₕ, I, nothing, lin) == ((O, 1.0),)
        end
        @test local_stencil(ZeroOperator(Wₕ), Wₕ, I, nothing, lin) == ((O, 0.0),)

        # and in one dimension the offset is a 1-tuple
        @test local_stencil(TrialFunction{1}(), Wₕ1, CartesianIndex(4), nothing, 4) ==
              (((0,), 1.0),)
    end

    @testset "the source nodes carry a value rather than a coefficient" begin
        # a function of position, evaluated at the point
        f = x -> x[1] + 10x[2]
        sf = source_function(f, Val(2))
        @test sf isa SourceFunction{2}
        st = local_stencil(sf, Wₕ, I, nothing, lin)
        @test first(first(st)) == O
        @test last(first(st)) ≈ f(Bramble.point(Ωₕ, I))

        # a vector of values, read at the linear index
        vec = collect(1.0:Float64(ndofs(Wₕ)))
        sv = SourceVector{2, Vector{Float64}}(vec)
        @test local_stencil(sv, Wₕ, I, nothing, lin) == ((O, vec[lin]),)
    end

    @testset "the constructors" begin
        @test trial_function(Val(2)) === TrialFunction{2}()
        @test test_function(Val(3)) === TestFunction{3}()
        @test source_function(sin, Val(1)) isa SourceFunction{1}
        @test IndexedTrialFunction{2}(7).component_idx == 7
        @test IndexedTestFunction{2}(4).component_idx == 4
    end

    @testset "the combining nodes" begin
        a = local_stencil(id + id, Wₕ, I, nothing, lin)
        @test a == ((O, 1.0), (O, 1.0))          # concatenated, not summed: assembly adds

        @test local_stencil(3 * id, Wₕ, I, nothing, lin) == ((O, 3.0),)
        @test local_stencil(id / 4, Wₕ, I, nothing, lin) == ((O, 0.25),)
        @test local_stencil(id - id, Wₕ, I, nothing, lin) == ((O, 1.0), (O, -1.0))

        # a grid function scales pointwise, read at the linear index
        uₕ = Rₕ(Wₕ, x -> x[1] + 1)
        @test local_stencil(uₕ * id, Wₕ, I, nothing, lin) == ((O, values(uₕ)[lin]),)
    end

    @testset "a Function in a GridFunctionScale is a thunk, not a field" begin
        # The distinction the SourceVector docstring spells out: `SourceFunction` holds a
        # function of position; a `Function` here is a zero-argument thunk returning the
        # values to scale by, so that building them can wait until the form is resolved.
        @test local_stencil((() -> 3.0) * id, Wₕ, I, nothing, lin) == ((O, 3.0),)

        vals = collect(1.0:Float64(ndofs(Wₕ)))
        @test local_stencil((() -> vals) * id, Wₕ, I, nothing, lin) == ((O, vals[lin]),)

        # resolving calls the thunk once and keeps what it returned
        r = resolve_ast((() -> vals) * id)
        @test r isa GridFunctionScale
        @test r.grid_function == vals
        @test !(r.grid_function isa Function)

        # so a function of position does not belong here, and says so rather than
        # silently scaling by something unintended
        @test_throws MethodError local_stencil((x -> x[1]) * id, Wₕ, I, nothing, lin)
    end

    @testset "resolve_ast" begin
        # the leaves are already resolved and come back identically
        for op in (TrialFunction{2}(), TestFunction{2}(), IndexedTrialFunction{2}(1),
            IndexedTestFunction{2}(1), source_function(sin, Val(2)),
            SourceVector{2, Vector{Float64}}([1.0]), id, ZeroOperator(Wₕ))
            @test resolve_ast(op) === op
        end

        # the combining nodes rebuild around their resolved children, keeping their kind
        @test resolve_ast(id + id) isa OperatorAdd
        @test resolve_ast(3 * id) isa OperatorScale
        @test resolve_ast(3 * id).scalar == 3

        uₕ = Rₕ(Wₕ, x -> x[1])
        @test resolve_ast(uₕ * id) isa GridFunctionScale

        # tuples resolve elementwise, and anything else is returned untouched
        @test resolve_ast((id, 3 * id)) isa NTuple{2, LazyOp}
        @test resolve_ast(42) === 42
        @test resolve_ast("not an ast") == "not an ast"

        # resolving is idempotent on an already-resolved tree
        t = resolve_ast(3 * (id + id))
        @test resolve_ast(t) isa OperatorScale
    end

    @testset "is_symbolic" begin
        u, v = TrialFunction{2}(), TestFunction{2}()

        # the symbolic leaves
        for op in (u, v, IndexedTrialFunction{2}(1), IndexedTestFunction{2}(1),
            source_function(sin, Val(2)), SourceVector{2, Vector{Float64}}([1.0]))
            @test is_symbolic(op)
        end

        # a concrete operator over a space is not symbolic
        @test !is_symbolic(id)
        @test !is_symbolic(ZeroOperator(Wₕ))

        # the products always are: they hold a trial and a test slot
        @test is_symbolic(innerₕ(u, v))

        # and the wrappers inherit it from what they wrap, either way
        for wrap in (D₋ₓ, D₊ₓ, M₋ₓ, M₊ₓ,
            op -> shift_op(op, 1, 1), op -> restrict_to(:interior, op),
            op -> 3 * op)
            @test is_symbolic(wrap(u))
            @test !is_symbolic(wrap(id))
        end

        # a sum is symbolic if either side is
        @test !is_symbolic(id + id)
        @test is_symbolic(D₋ₓ(u) + D₋ₓ(id))
        @test is_symbolic((D₋ₓ(id), D₋ₓ(u)))     # and so is a tuple
        @test !is_symbolic((D₋ₓ(id), M₊ᵧ(id)))
    end

    @testset "the parallel workspace both assembly files share" begin
        # A colouring of the grid into independent groups: no two indices within a group
        # write to the same matrix entry, so a group can be walked without synchronisation.
        #
        # It lives in form/parallel_workspace.jl rather than in bilinear.jl, where it began.
        # `linear.jl` names it as the type of a `LinearForm` field, and a struct definition
        # resolves its field types when it is defined rather than when it is called — so
        # while the type sat in bilinear.jl, linear.jl could not be unlocked on its own.
        groups = [[CartesianIndex(1, 1), CartesianIndex(1, 3)], [CartesianIndex(2, 2)]]

        w = Bramble.ParallelWorkspace{2}(groups)
        @test w.color_groups === groups
        @test isempty(w.thread_buffers)          # the one-argument form allocates none

        buffers = [zeros(4), zeros(4)]
        wb = Bramble.ParallelWorkspace{2}(groups, buffers)
        @test wb.color_groups === groups
        @test wb.thread_buffers === buffers

        # the dimension is a type parameter, so a mismatched index set does not compile
        @test Bramble.ParallelWorkspace{2} !== Bramble.ParallelWorkspace{3}
        @test_throws MethodError Bramble.ParallelWorkspace{3}(groups)
    end
end
