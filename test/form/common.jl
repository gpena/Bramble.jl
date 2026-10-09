module FormCommonTests

using Test
using Bramble
# Internal names: defined and documented, not exported.
import Bramble: D₊ₓ, M₊ₓ, M₊ᵧ
using LinearAlgebra: issymmetric, Diagonal, I as Id
using Bramble:
               TrialFunction,
               TestFunction,
               IndexedTrialFunction,
               IndexedTestFunction,
               SourceFunction,
               SourceVector,
               IdentityOperator,
               ZeroOperator,
               LazyOp,
               OperatorAdd,
               OperatorScale,
               GridFunctionScale,
               trial_function,
               test_function,
               source_function,
               local_stencil,
               resolve_ast,
               is_symbolic,
               zero_offset,
               shift_offset,
               shift_stencil,
               concatenate_stencils,
               scale_stencil,
               entry_offsets,
               entry_weights,
               _offsets_seen_before,
               multiply_stencils_bilinear,
               multiply_stencils_linear,
               restrict_to,
               shift_op,
               form,
               assemble,
               D₋ₓ,
               Mₓ,
               inner₊ₓ,
               jumpₓ

# The AST leaves, the stencil algebra under them, and the two traits every node answers.
#
# A stencil is a tuple of `(offset, coefficient)` pairs, offsets relative to the point being
# evaluated. Everything in this file either produces one, combines two, or walks a tree of
# nodes that do. The combinators are the pieces assembly is built from:
# `multiply_stencils_bilinear` is how a trial stencil and a test stencil become matrix
# entries, and none of them had ever run.

@testset "AST and stencil algebra" begin
    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 9, false)
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
    Wₕ1, Wₕ = gridspace(Ωₕ1), gridspace(Ωₕ)
    id = IdentityOperator(Wₕ)
    I = CartesianIndex(3, 3)
    lin = LinearIndices(Bramble.indices(Ωₕ))[I]
    O = (0, 0)

    @testset "Offsets" begin
        @test zero_offset(Val(1)) == (0,)
        @test zero_offset(Val(2)) == (0, 0)
        @test zero_offset(Val(3)) == (0, 0, 0)

        @test shift_offset((0, 0), 1, 1) == (1, 0)
        @test shift_offset((0, 0), 2, -1) == (0, -1)
        @test shift_offset((2, -3, 4), 3, 5) == (2, -3, 9)
        @test shift_offset((1, 1), 1, 0) == (1, 1)          # a zero shift is identity
    end

    @testset "Directional spacings" begin
        # In one dimension the mesh answers with a number and in more with a tuple, so the
        # 3-arg accessors must agree with picking a component out of either: on `Mesh1D`
        # `dim` is a passthrough (`spacing(Ωₕ1, i1, 1) == spacing(Ωₕ1, i1)`), while on
        # `MeshnD` it queries submesh `dim` directly instead of building the full tuple.
        i1 = CartesianIndex(4)
        @test spacing(Ωₕ1, i1, 1) == spacing(Ωₕ1, i1)
        @test forward_spacing(Ωₕ1, i1, 1) == forward_spacing(Ωₕ1, i1)

        for d in 1:2
            @test spacing(Ωₕ, I, d) == spacing(Ωₕ, I)[d]
            @test forward_spacing(Ωₕ, I, d) == forward_spacing(Ωₕ, I)[d]
        end

        # the two directions of a non-uniform mesh really do differ, so the test is not
        # comparing a number with itself
        @test spacing(Ωₕ, I, 1) != spacing(Ωₕ, I, 2)
    end

    @testset "Stencil combination" begin
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

        # the linear form keeps only the right offset (the left has been contracted away)
        l = multiply_stencils_linear(s1, s2, 2.0)
        @test length(l) == length(s1) * length(s2)
        @test l == (((0, 1), 20.0), ((0, 1), 30.0))
        @test all(e -> length(e) == 2, l)
        @test all(e -> length(e) == 3, b)
    end

    @testset "Offsets and weights split apart (#249)" begin
        # `_visit_entries` (form/bilinear_traversal.jl) reads a stencil entry's offsets and
        # its weight from these two containers rather than from the entry's own mixed
        # `Int`/`Float64` tuple, which is what lets Enzyme differentiate assembly with
        # respect to an operator's coefficient. `test/ext/chainrules_enzyme_ext.jl` checks
        # the gradients themselves; what matters here is the property the traversal relies
        # on -- that the two containers stay aligned with the stencil, entry for entry, for
        # a bilinear entry and a linear one alike.
        s1 = ((O, 2.0), ((1, 0), 3.0))
        s2 = (((0, 1), 5.0),)
        b = multiply_stencils_bilinear(s1, s2, 2.0)

        # bilinear: two offsets per entry, the weight last
        @test entry_offsets(b) == ((O, (0, 1)), ((1, 0), (0, 1)))
        @test entry_weights(b) == (20.0, 30.0)

        # linear: one offset per entry, kept as a 1-tuple so indexing reads the same way
        l = multiply_stencils_linear(s1, s2, 2.0)
        @test entry_offsets(l) == (((0, 1),), ((0, 1),))
        @test entry_weights(l) == (20.0, 30.0)

        # and the pair reconstructs the stencil it came from, which is the alignment the
        # traversal depends on
        @test map((o, w) -> (o..., w), entry_offsets(b), entry_weights(b)) == b
        @test map((o, w) -> (o..., w), entry_offsets(l), entry_weights(l)) == l

        # an empty stencil (a `RegionRestriction` outside its region) stays empty
        @test entry_offsets(()) == ()
        @test entry_weights(()) == ()

        # homogeneous containers, not a tuple of mixed tuples: the whole point, since a
        # container Enzyme has to type per element is what it could not handle
        @test eltype(entry_weights(b)) == Float64
        @test isconcretetype(typeof(entry_offsets(b)))

        # neither allocates, which is what keeps `assemble!`'s replay at 0 bytes
        function _split_bytes(st)
            entry_offsets(st)                             # warm both, then measure
            entry_weights(st)
            return @allocated (entry_offsets(st), entry_weights(st))
        end
        @test _split_bytes(b) == 0
        @test _split_bytes(l) == 0

        # `_offsets_seen_before` is called with both shapes: `entry_offsets(stencil)` from
        # the traversal, and the raw stencil from the pattern builders in
        # form/jacobian_pattern.jl, which never read a weight. Both have to answer the same.
        dup = ((O, (0, 1), 1.0), ((1, 0), (0, 1), 2.0), (O, (0, 1), 4.0))
        for k in eachindex(dup)
            @test _offsets_seen_before(entry_offsets(dup), k, dup[k][1], dup[k][2]) ==
                  _offsets_seen_before(dup, k, dup[k][1], dup[k][2])
        end
        @test !_offsets_seen_before(entry_offsets(dup), 1, dup[1][1], dup[1][2])
        @test !_offsets_seen_before(entry_offsets(dup), 2, dup[2][1], dup[2][2])
        @test _offsets_seen_before(entry_offsets(dup), 3, dup[3][1], dup[3][2])
    end

    @testset "Unit leaf stencils" begin
        for op in (
            TrialFunction{2}(),
            TestFunction{2}(),
            IndexedTrialFunction{2}(1),
            IndexedTestFunction{2}(2),
            id
        )
            @test local_stencil(op, Wₕ, I, nothing, lin) == ((O, 1.0),)
        end
        @test local_stencil(ZeroOperator(Wₕ), Wₕ, I, nothing, lin) == ()

        # and in one dimension the offset is a 1-tuple
        @test local_stencil(TrialFunction{1}(), Wₕ1, CartesianIndex(4), nothing, 4) ==
              (((0,), 1.0),)
    end

    @testset "Source node values" begin
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

    @testset "Constructors" begin
        @test trial_function(Val(2)) === TrialFunction{2}()
        @test test_function(Val(3)) === TestFunction{3}()
        @test source_function(sin, Val(1)) isa SourceFunction{1}
        @test IndexedTrialFunction{2}(7).component_idx == 7
        @test IndexedTestFunction{2}(4).component_idx == 4

        # a point source: the vector spelling of one point normalises to the same tuple of
        # `Float64` coordinates as the tuple spelling, integer coordinates included
        d = Bramble.dirac([0.3, 0.6], 2.5)
        @test d isa Bramble.DiracSource{2}
        @test d.points === (0.3, 0.6)
        @test d.strengths === 2.5
        @test Bramble.dirac([0, 1]).points === (0.0, 1.0)
        @test Bramble.DiracSource{1}((0.5,), 2.0) === Bramble.DiracSource{1, Tuple{Float64}, Float64}((0.5,), 2.0)

        # several points, each normalised by its own spelling: a bare number is a 1D
        # point, a vector a D-dimensional one
        many1 = Bramble.dirac(Union{Float64, Tuple{Float64}}[0.2, (0.7,)], [1.0, 2.0])
        @test many1 isa Bramble.DiracSource{1}
        @test many1.points == [(0.2,), (0.7,)]
        @test many1.strengths == [1.0, 2.0]
        many2 = Bramble.dirac([[0.1, 2], [3, 0.4]])
        @test many2 isa Bramble.DiracSource{2}
        @test many2.points == [(0.1, 2.0), (3.0, 0.4)]
        @test many2.strengths == [1.0, 1.0]
    end

    @testset "Shifted inner stencils, non-uniform mesh" begin
        # `shifted_inner_stencil` is how a tapped node (here a difference) evaluates its
        # operand at a neighbour. Each operand below takes a different branch of it: a
        # grid-function scale (re-reads the coefficient at the tap), a sum (recurses into
        # each summand), a nested point-dependent operand (re-evaluated and relabelled), a
        # scalar times a bare leaf (inlined re-evaluation) and a source (re-evaluated, not
        # relabelled). The oracle is the product of the operator matrices, built by
        # `src/operators/stencil_matrix.jl` without any stencil shift, row by row at every
        # point of the non-uniform mesh, the two boundary points included.
        n = npoints(Ωₕ1)
        u = TrialFunction{1}()
        c = Rₕ(Wₕ1, x -> 1 + x^2)
        C = Diagonal(parent(c))
        M₋, M₊ = Matrix(D₋ₓ(Ωₕ1)), Matrix(D₊ₓ(Ωₕ1))

        # sums the stencil's weights per column; a tap off the grid must carry no weight
        function _row(st, i)
            r = zeros(n)
            for (o, w) in st
                j = i + o[1]
                if 1 <= j <= n
                    r[j] += w
                else
                    @test iszero(w)
                end
            end
            return r
        end

        for (nm, op, reference) in (
            ("D₋ₓ(c u)", D₋ₓ(c * u), M₋ * C),
            ("D₋ₓ(c u + u)", D₋ₓ(c * u + u), M₋ * (C + Id)),
            ("D₊ₓ(D₋ₓ(c u))", D₊ₓ(D₋ₓ(c * u)), M₊ * M₋ * C),
            ("D₋ₓ(2 u)", D₋ₓ(2.0 * u), 2 * M₋)
        )
            @testset "$nm" begin
                for i in 1:n
                    st = local_stencil(op, Wₕ1, CartesianIndex(i), nothing, i)
                    @test _row(st, i) ≈ reference[i, :]
                end
            end
        end

        # a source carries a value, not a column: every entry stays at offset zero, and
        # the weights add up to the difference of the sampled values
        vals = collect(1.0:Float64(n)) .^ 2
        sv = SourceVector{1, Vector{Float64}}(vals)
        for i in 1:n
            st = local_stencil(D₋ₓ(sv), Wₕ1, CartesianIndex(i), nothing, i)
            @test all(e -> first(e) == (0,), st)
            @test sum(last, st) ≈ (M₋ * vals)[i]
        end
    end

    @testset "Source lowering" begin
        # a function of position is sampled once, at every mesh point, into a vector
        # source, through whatever scaling wraps it; a tree with nothing to sample comes
        # back as it was
        f = x -> 1 + x^2
        sampled = [f(Bramble.point(Ωₕ1, CartesianIndex(i))) for i in 1:npoints(Ωₕ1)]
        sf = source_function(f, Val(1))
        cₕ = Rₕ(Wₕ1, x -> 3 - x)

        lowered = Bramble._lower_sources(sf, Wₕ1)
        @test lowered isa SourceVector{1}
        @test lowered.vec ≈ sampled

        scaled = Bramble._lower_sources(3 * sf, Wₕ1)
        @test scaled isa OperatorScale
        @test scaled.scalar == 3
        @test scaled.inner_op isa SourceVector{1}
        @test scaled.inner_op.vec ≈ sampled

        weighted = Bramble._lower_sources(cₕ * sf, Wₕ1)
        @test weighted isa GridFunctionScale
        @test weighted.grid_function === cₕ
        @test weighted.inner_op.vec ≈ sampled

        u1 = TrialFunction{1}()
        @test Bramble._lower_sources(3 * u1, Wₕ1) === 3 * u1
        @test Bramble._lower_sources(cₕ * u1, Wₕ1) === cₕ * u1
    end

    @testset "Combining nodes" begin
        a = local_stencil(id + id, Wₕ, I, nothing, lin)
        @test a == ((O, 1.0), (O, 1.0))          # concatenated, not summed: assembly adds

        @test local_stencil(3 * id, Wₕ, I, nothing, lin) == ((O, 3.0),)
        @test local_stencil(id / 4, Wₕ, I, nothing, lin) == ((O, 0.25),)
        @test local_stencil(id - id, Wₕ, I, nothing, lin) == ((O, 1.0), (O, -1.0))

        # a grid function scales pointwise, read at the linear index
        uₕ = Rₕ(Wₕ, x -> x[1] + 1)
        @test local_stencil(uₕ * id, Wₕ, I, nothing, lin) == ((O, parent(uₕ)[lin]),)
    end

    @testset "GridFunctionScale thunk" begin
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
        for op in (
            TrialFunction{2}(),
            TestFunction{2}(),
            IndexedTrialFunction{2}(1),
            IndexedTestFunction{2}(1),
            source_function(sin, Val(2)),
            SourceVector{2, Vector{Float64}}([1.0]),
            id,
            ZeroOperator(Wₕ)
        )
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

    @testset "Integer divisor keeps eltype (#633)" begin
        # a divisor that fits in an `Int` scales by the exact rational, wider ones and floats
        # by the `Float64` reciprocal as before
        @test (@inferred id / 2).scalar === 1 // 2
        @test (id / -3).scalar === -1 // 3
        @test (id / Int8(-128)).scalar === -1 // 128
        @test (id / UInt32(3)).scalar === 1 // 3
        @test (id / true).scalar === 1 // 1
        @test (id / 0).scalar === 1 // 0
        @test Float64((id / typemin(Int)).scalar) === 1 / typemin(Int)
        @test (id / UInt64(2)).scalar === 0.5
        @test (id / Int128(2)).scalar === 0.5
        @test (id / big(2)).scalar isa BigFloat
        @test (id / 0.5).scalar === 2.0

        # the simplifier neither folds nor combines a rational scale: checked arithmetic
        # would overflow where the floats it stands for do not
        t = Bramble.simplify_ast((id / 2^40) / 2^40)
        @test t isa OperatorScale && t.inner_op isa OperatorScale
        @test Bramble.simplify_ast(id / 2 + id / 2) isa OperatorAdd
        @test Bramble.simplify_ast(2 * (3 * id)).scalar === 6

        # a Float32 space on a non-uniform mesh assembles Float32 forms through `/ 2`
        W32 = gridspace(mesh(domain(interval(0.0f0, 1.0f0)), 11, false;
                             backend = backend(Float32)))
        A32 = assemble(form(W32, W32, (u, v) -> innerₕ(u, v) / 2))
        @test eltype(A32) === Float32
        @test A32 == assemble(form(W32, W32, (u, v) -> 0.5f0 * innerₕ(u, v)))
        b32 = assemble(form(W32, v -> innerₕ(x -> 1.0f0, v) / 2))
        @test eltype(b32) === Float32
        @test b32 == assemble(form(W32, v -> 0.5f0 * innerₕ(x -> 1.0f0, v)))
        N32 = form(W32, W32, (u, v) -> -(innerₕ(u, v) / 2) / 3 + innerₕ(u, v) / UInt8(4))
        @test eltype(assemble(N32)) === Float32

        # on Float64 a lone divisor gives the `1 / c` matrix bit for bit, for a plain and a
        # `Ref`-scaled form
        A = assemble(form(Wₕ1, Wₕ1, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)) / 3))
        @test A == assemble(form(Wₕ1, Wₕ1, (u, v) -> (1 / 3) * innerₕ(D₋ₓ(u), D₋ₓ(v))))
        β = Ref(3.0)
        Aβ = assemble(form(Wₕ1, Wₕ1, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)) / 2))
        @test Aβ == assemble(form(Wₕ1, Wₕ1, (u, v) -> innerₕ(1.5 * D₋ₓ(u), D₋ₓ(v))))

        # divisors whose rational products, sums or negations overflow assemble as the
        # floats did: each scale converts on its own
        A1 = assemble(form(Wₕ1, Wₕ1, (u, v) -> innerₕ(u, v)))
        F(f) = assemble(form(Wₕ1, Wₕ1, f))
        @test F((u, v) -> (innerₕ(u, v) / 2^40) / 2^40) == A1 .* (1 / 2^40)^2
        @test F((u, v) -> innerₕ(u, v) / 2^62 + innerₕ(u, v) / (2^62 - 1)) ≈
              A1 .* (1 / 2^62 + 1 / (2^62 - 1))
        @test F((u, v) -> -(innerₕ(u, v) / Int8(-128))) == A1 ./ 128
        @test F((u, v) -> innerₕ(u, v) / typemin(Int)) == A1 .* (1 / typemin(Int))
        @test F((u, v) -> (innerₕ(u, v) / UInt8(200)) / UInt8(200)) ≈ A1 ./ 40000
        Z = F((u, v) -> innerₕ(u, v) / 0 - innerₕ(u, v) / 0)
        @test all(isnan, Z.nzval)
        @test all(==(Inf), F((u, v) -> innerₕ(u, v) / 0 + innerₕ(u, v) / 0).nzval)

        # the documented limit: a transposed pair multiplies its nested rational scales
        # exactly, and 2^80 does not fit in an `Int`
        @test_throws OverflowError F((u, v) -> (innerₕ(D₋ₓ(u), v) / 2^40) / 2^40 +
                                               innerₕ(u, D₋ₓ(v)))
    end

    @testset "is_symbolic" begin
        u, v = TrialFunction{2}(), TestFunction{2}()

        # the symbolic leaves
        for op in (
            u,
            v,
            IndexedTrialFunction{2}(1),
            IndexedTestFunction{2}(1),
            source_function(sin, Val(2)),
            SourceVector{2, Vector{Float64}}([1.0])
        )
            @test is_symbolic(op)
        end

        # a concrete operator over a space is not symbolic
        @test !is_symbolic(id)
        @test !is_symbolic(ZeroOperator(Wₕ))

        # the products always are: they hold a trial and a test slot
        @test is_symbolic(innerₕ(u, v))

        # and the wrappers inherit it from what they wrap, either way
        for wrap in (
            D₋ₓ,
            D₊ₓ,
            Mₓ,
            M₊ₓ,
            op -> shift_op(op, 1, 1),
            op -> restrict_to(:interior, op),
            op -> 3 * op
        )
            @test is_symbolic(wrap(u))
            @test !is_symbolic(wrap(id))
        end

        # a sum is symbolic if either side is
        @test !is_symbolic(id + id)
        @test is_symbolic(D₋ₓ(u) + D₋ₓ(id))
        @test is_symbolic((D₋ₓ(id), D₋ₓ(u)))     # and so is a tuple
        @test !is_symbolic((D₋ₓ(id), M₊ᵧ(id)))
    end
end

@testset "Element type preservation" begin
    # A `Float32` space assembled a `Float64` vector and a `Float64` matrix, silently,
    # because the stencil leaves returned a literal `1.0` and the boundary masks a literal
    # `0.0`/`1.0`. `_assembled_eltype` promotes the space's type against the weight it finds,
    # and `promote_type(Float32, Float64)` is `Float64`, so single precision was widened at
    # every point of every form, including on the Metal backend, whose arrays are Float32.
    #
    # The weights are now typed: a unit weight is the integer `1`, which carries no precision
    # claim and promotes to whatever it multiplies, and the averages take ½ from
    # `eltype(space)` rather than from a literal. Both branches of each mask keep one type, so
    # the weight does not infer as a union.
    #
    # Every stencil shape is checked, because each builds its weight differently: the
    # identity from a leaf, the difference by dividing a mask by a spacing, the average from
    # a halved mask, the jump from a reach plus a `-1`.
    for T in (Float32, Float64)
        Ωₕ = mesh(domain(interval(T(0), T(1)) × interval(T(0), T(1))), (6, 6), (true, true))
        Wₕ = gridspace(Ωₕ)
        @test eltype(Ωₕ) === T
        fₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])

        for e in (
            v -> innerₕ(fₕ, v),
            v -> innerₕ(fₕ, D₋ₓ(v)),
            v -> innerₕ(fₕ, Mₓ(v)),
            v -> innerₕ(fₕ, jumpₓ(v)),
            v -> inner₊ₓ(fₕ, D₋ₓ(v))
        )
            @test eltype(assemble(form(Wₕ, e))) === T
        end

        @test eltype(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))) === T
        @test eltype(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v))))) === T
    end

    @testset "Through the Dirichlet-constrained path" begin
        # Every check above stops at assemble(form(...)), never reaching
        # apply_dirichlet_labels!/dirichlet_bc!, which write their own mask/diagonal
        # literals (0/1) into the matrix and vector -- exactly the kind of literal that
        # widened a Float32 space to Float64 before _assembled_eltype existed. A Dirichlet
        # constraint on a Float32 problem must not reintroduce that promotion one call later.
        for T in (Float32, Float64)
            S = interval(T(0), T(1)) × interval(T(0), T(1))
            Ωₕ = mesh(domain(S, :walls => boundary_symbols(S)), (6, 6), (true, true))
            Wₕ = gridspace(Ωₕ)
            @test eltype(Ωₕ) === T

            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
            A = assemble(a; dirichlet = :walls)
            @test eltype(A) === T

            fₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])
            l = form(Wₕ, v -> innerₕ(fₕ, v))
            bcs = dirichlet_constraints(Ωₕ, :walls => (x -> zero(T)))
            b = assemble(l; dirichlet = bcs)
            @test eltype(b) === T
        end
    end
end

# Invariants tested: `show` of both form types prints the spaces a caller wants to read back,
# not the whole resolved AST type with every operator node and its parameters.
# The integrand is deliberately not rendered: reconstructing it would need a name for
# every node type, kept in step with each one added, to restate an expression the caller
# just wrote.
@testset "Form display" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 3), (true, true))
    Wₕ = gridspace(Ωₕ)
    W4 = gridspace(
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
    )
    uₕ = Rₕ(Wₕ, x -> x[1])

    sym = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(u, v))
    asym = form(Wₕ, Wₕ, (u, v) -> inner₊(u, D₋ₓ(v)))
    l = form(Wₕ, v -> innerₕ(uₕ, v))

    @testset "LinearForm" begin
        compact = sprint(show, l)
        @test compact == "LinearForm{2D, 9}"
        @test !occursin('\n', compact)

        detailed = sprint(show, MIME"text/plain"(), l)
        @test occursin("LinearForm", detailed)
        @test occursin("Test space", detailed)
        @test occursin("ScalarGridSpace{2D, Float64, 9 dofs}", detailed)
        @test occursin("Vector", detailed)
        @test !endswith(detailed, '\n')
        # The AST type is what the default `show` used to print; it must not appear.
        @test !occursin("LinearProduct", detailed)
        @test !occursin("InnerH", detailed)
    end

    @testset "BilinearForm" begin
        compact = sprint(show, sym)
        @test compact == "BilinearForm{2D, 9×9}"
        @test !occursin('\n', compact)

        detailed = sprint(show, MIME"text/plain"(), sym)
        @test occursin("BilinearForm", detailed)
        @test occursin("Trial space", detailed)
        @test occursin("Test space", detailed)
        @test occursin("9 × 9", detailed)
        @test !endswith(detailed, '\n')
        @test !occursin("BilinearProduct", detailed)
        @test !occursin("OperatorAdd", detailed)
    end

    # Symmetry is reported from the existing structural check.
    @testset "Symmetry from the structural check" begin
        @test occursin("Symmetric: yes", sprint(show, MIME"text/plain"(), sym))
        @test occursin("Symmetric: no", sprint(show, MIME"text/plain"(), asym))
        # Whatever the display says must be what `issymmetric` says.
        for a in (sym, asym)
            expected = issymmetric(a) ? "Symmetric: yes" : "Symmetric: no"
            @test occursin(expected, sprint(show, MIME"text/plain"(), a))
        end
    end

    # The deprecated `ast` keyword assembles the AST it is handed; handing it the form's
    # own gives what the form assembles without it, and the deprecation is announced
    # (when `--depwarn` is on; `@test_deprecated` passes the value through otherwise).
    @testset "Deprecated ast keyword" begin
        b = @test_deprecated r"`ast` keyword" assemble(l; ast = l.ast)
        @test b == assemble(l)
    end

    # Distinct trial and test spaces.
    @testset "Trial and test spaces named separately" begin
        mixed = form(Wₕ, W4, (u, v) -> innerₕ(u, v))
        detailed = sprint(show, MIME"text/plain"(), mixed)
        @test occursin("9 dofs", detailed)
        @test occursin("16 dofs", detailed)
        # `(same as trial)` is only for a form whose two spaces really are one object.
        @test !occursin("same as trial", detailed)
        @test occursin("same as trial", sprint(show, MIME"text/plain"(), sym))

        # Rows are indexed by the test function, columns by the trial one, so the matrix
        # shape reads test × trial.
        @test sprint(show, mixed) == "BilinearForm{2D, 16×9}"
    end
end

end # module FormCommonTests
