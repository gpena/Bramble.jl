module FormLinearRestrictionTests

using Test
using Bramble
using SparseArrays: nnz
using Bramble:
               LinearForm,
               form,
               assemble,
               assemble!,
               assemble_parallel!,
               element,
               LinearProduct,
               ndofs,
               restrict_to,
               D̽ₓ,
               D̽ᵧ,
               D̽₂,
               D₊ₓ,
               D₊ᵧ,
               Dcₓ,
               Mcₓ,
               S₊ₓ,
               source_function,
               D₋ᵧ,
               D₋ₓ,
               Mₓ,
               jumpₓ
using ..TestUtils: alloc_test, WITH_3D_TESTS, WITH_SLOW_TESTS, TEST_DIMS, _Const,
    _one_plus_x1, _x1_lt_half

# The linear-form tests of restricted sums and scales: the refill allocation contract and the
# scaled and nested cases. Split from linear.jl, whose single module took 0.95 GB of the
# process peak to lower under `--code-coverage` before its first testset ran, enough to
# tip the nightly's ubuntu leg over its memory budget.

@testset "Linear forms: restrictions" begin
    @testset "Restricted refill allocation contract" begin
        # A `RegionRestriction`'s stencil is `()` or a full tuple depending on the point's
        # marker, and the scalar path evaluates the whole sum's stencil at every point. The
        # `map` that used to collect the summands' stencils built a tuple with a
        # non-concrete element there, boxed per point: 8736 B on the 2D grid, where the
        # unrestricted form allocated nothing. The refill must also match the two terms
        # assembled apart.
        function restricted_bytes(b, lf)
            assemble!(b, lf)
            assemble!(b, lf)
            return @allocated assemble!(b, lf)
        end
        Ω1d = mesh(domain(interval(0.0, 1.0)), 11, true)
        Ω2d = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9), (false, true))
        for Ω in (Ω1d, Ω2d)
            W = gridspace(Ω)
            lf = form(W, v -> innerₕ(1.0, v) + Ref(2.0) * innerₕ(1.0, restrict_to(:boundary, v)))
            b = zeros(ndofs(W))
            @test restricted_bytes(b, lf) == 0
            parts = assemble(form(W, v -> innerₕ(1.0, v))) +
                    2 * assemble(form(W, v -> innerₕ(1.0, restrict_to(:boundary, v))))
            @test b ≈ parts
        end

        # Past two restricted summands the `Union` is not split and every point boxed
        # (16400, 20256 and 38800 B for three, four and five summands on this 7x9 grid):
        # `form` now splits such a product into products of at most two, and a form with
        # more than two is walked one summand at a time (gpena/Bramble.jl#727). The vector
        # must be the terms' assembled apart.
        R = restrict_to
        Ωm = domain(interval(0.0, 1.0) × interval(0.0, 1.0), :l => :left, :r => :right,
            :b => :bottom, :t => :top, :c => _x1_lt_half)
        W = gridspace(mesh(Ωm, (7, 9), (false, false)))
        labels = (:l, :r, :b, :t, :c)
        g = element(W)
        g .= 1.0 .+ (1:length(g)) ./ 7
        function _check_split(terms; bytes = 0)
            lf = form(W, v -> sum(t(v) for t in terms))
            b = assemble(lf)
            bytes === nothing || @test restricted_bytes(b, lf) == bytes
            parts = sum(assemble(form(W, t)) for t in terms)
            @test b ≈ parts
            vₕ = Rₕ(W, x -> 1 + x[1] * x[2])
            @test lf(vₕ) ≈ sum(parts .* parent(vₕ))
            b2 = zero(b)
            assemble_parallel!(b2, lf)
            @test b2 == b
            return b, parts
        end
        # Tuples of four and five summands compile their own forms: they run in `slow`.
        for k in 3:(WITH_SLOW_TESTS ? 5 : 3)
            on(s) = v -> innerₕ(_one_plus_x1, R(s, v))
            b, parts = _check_split(Tuple(on(s) for s in labels[1:k]))
            @test b == parts
            _check_split((v -> innerₕ(_one_plus_x1, v), Tuple(on(s) for s in labels[1:k])...))
            # Each its own source, so `simplify_ast` folds nothing: k products, one walk each.
            _check_split(Tuple(v -> innerₕ(i * g, R(labels[i], v)) for i in 1:k))
            _check_split((v -> innerₕ(1.0, v),
                Tuple(v -> 2.0 * innerₕ(g, D₋ₓ(R(s, v))) for s in labels[1:k])...))
        end
        # A restriction over a sum is distributed over it; constant scales of several types
        # are widened to the type of their product with the element weight, never narrowed
        # (12816 B for `2.0R + 3R` before).
        _check_split((v -> innerₕ(g, R(:c, v + R(:l, v) + R(:r, v))), v -> innerₕ(g, R(:b, v))))
        _check_split((v -> innerₕ(_Const(1.0), 2.0 * R(:l, v)), v -> innerₕ(_Const(1.0), 3 * R(:r, v))))
        _check_split((v -> innerₕ(g, v), v -> innerₕ(g, 2.0 * R(:l, v)),
            v -> innerₕ(g, 3 * R(:r, v)), v -> innerₕ(g, (1 // 2) * R(:b, v))))
        # A complex scale on a Float64 mesh widens the form to ComplexF64, which refills
        # without allocating, and must still agree with its parts.
        _check_split(
            (v -> innerₕ(g, (1.0 + 2.0im) * R(:l, v)), v -> innerₕ(g, 3 * R(:r, v)),
            v -> innerₕ(g, 2.0 * R(:b, v))))
        # Summands of one type are paired whatever order they are written in.
        _check_split((v -> innerₕ(g, R(:l, v)), v -> innerₕ(g, 2.0 * R(:r, v)),
            v -> innerₕ(g, R(:b, v))))

        # A scale over a sum of three restricted products is walked a summand at a time
        # under the scale: walked whole it boxed per point (48816 B in 2D, 4624 B in 1D).
        Ω1 = domain(interval(0.0, 1.0), :a => x -> x[1] < 0.3, :b => x -> x[1] > 0.6,
            :c => x -> 0.2 < x[1] < 0.5)
        for Ws in (gridspace(mesh(Ω1, 13, false)), W)
            ls = Tuple(v -> innerₕ(1.0, R(s, v)) for s in (Ws === W ? (:l, :b, :c) : (:a, :b, :c)))
            parts = sum(assemble(form(Ws, l)) for l in ls)
            for (c, wrap) in ((2.0, x -> 2.0 * x), (-1, x -> -x))
                lf = form(Ws, v -> wrap(ls[1](v) + ls[2](v) + ls[3](v)))
                b = assemble(lf)
                @test restricted_bytes(b, lf) == 0
                @test b ≈ c * parts
                b2 = zero(b)
                assemble_parallel!(b2, lf)
                @test b2 == b
                vₕ = Rₕ(Ws, _one_plus_x1)
                @test lf(vₕ) ≈ c * sum(parts .* parent(vₕ))
            end
        end

        # A form holding no restriction comes back as `simplify_ast` built it when its test
        # side's scales agree in type. When they differ they are only ever widened, to the
        # type of their products with a weight of the mesh's (`π` beside an Int on a
        # Float32 mesh stays Float32, a Float64 literal stays Float64): each form keeps the
        # unwidened one's element type and values, and refills with 0 B (14112 B on the
        # Float32 mesh when left mixed, 27216 B for the sum of two products).
        _ast(f, S = W) = Bramble.simplify_ast(Bramble._resolve_surfaces_for_space(
            Bramble.resolve_ast(f(Bramble.test_function(S))), S))
        _scale_types(op::Bramble.OperatorAdd) = (
            _scale_types(op.left_op)..., _scale_types(op.right_op)...)
        _scale_types(op::Bramble.OperatorScale) = (typeof(op.scalar),)
        _scale_types(op) = ()
        I32 = interval(0.0f0, 1.0f0)
        W32 = gridspace(mesh(domain(I32 × I32), (7, 9), (false, false); backend = backend(Float32)))
        f32 = _one_plus_x1
        for (S, h) in ((W, v -> innerₕ(g, 2.0 * v + 3.0 * D₋ₓ(v))),
            (W32, v -> innerₕ(f32, 2 * v + 3 * D₋ₓ(v))),
            (W32, v -> innerₕ(f32, 0.1 * v + 0.3 * D₋ₓ(v)) + innerₕ(1.0, v)))
            ast = _ast(h, S)
            @test Bramble._split_restricted_products(ast, eltype(S)) === ast
        end
        for (S, h, T, exact) in (
            (W, v -> innerₕ(g, 0.1 * v + 3 * D₋ₓ(v) + (1 // 3) * D₋ᵧ(v)), Float64, true),
            (W32, v -> innerₕ(f32, 0.1 * v + 3 * D₋ₓ(v)), Float64, false),
            (W32, v -> innerₕ(f32, π * v + 3 * D₋ₓ(v)), Float32, false),
            (W32, v -> innerₕ(f32, 0.1 * v + 3 * D₋ₓ(v)) + innerₕ(1.0, v), Float64, false),
            (W32, v -> 2.0 * innerₕ(f32, Ref(0.1) * v + 3 * D₋ₓ(v)), Float64, false))
            ast = _ast(h, S)
            split = Bramble._split_restricted_products(ast, eltype(S))
            if ast isa Bramble.LinearProduct
                @test all(==(T), _scale_types(split.right_op))
                @test all(t -> promote_type(t, T) === T, _scale_types(ast.right_op))
            end
            lf = form(S, h)
            b = assemble(lf)
            lowered = Bramble._lower_sources_for_space(ast, S)
            U = assemble(Bramble.LinearForm{2, typeof(S), typeof(lowered)}(S, lowered))
            @test eltype(b) === eltype(U) === T
            @test restricted_bytes(b, lf) == 0
            @test exact ? b == U : isapprox(b, U; rtol = 1e-6)
        end
        # Scales joining to a type that is not bits are left mixed: every product with a
        # BigFloat allocates, and converting the Int to one only adds allocations.
        h = v -> innerₕ(g, big(0.1) * v + 3 * D₋ₓ(v))
        ast = _ast(h)
        @test Bramble._split_restricted_products(ast, Float64) === ast
        lf = form(W, h)
        b = assemble(lf)
        @test eltype(b) === BigFloat
        @test restricted_bytes(b, lf) <= 65760

        # The zero stencil a typed restriction stores past 32 entries: `map` boxed it.
        s40 = ntuple(k -> (k, 2, isodd(k) ? 1.5k : k), 40)
        @test Bramble._zero_entries(s40) == map(t -> (Base.front(t)..., zero(t[end])), s40)
        @test alloc_test(Bramble._zero_entries, s40) == 0

        # The split of a restricted form infers its concrete result.
        ast = _ast(v -> innerₕ(g, g * R(:l, v) + g * R(:r, v) + g * R(:b, v) + g * R(:t, v)))
        rt = Base.return_types(Bramble._split_restricted_products, (typeof(ast), Type{Float64}))
        @test length(rt) == 1 && isconcretetype(only(rt))
        @test Bramble._split_restricted_products(ast, Float64) isa Bramble.OperatorAdd
    end

    @testset "Cross-weighted scaled restriction" begin
        # `D̽` taps its operand at three points. A constant scale over a restriction was
        # re-evaluated at each neighbour, `()` or a full tuple by its marker, and three such
        # taps made more stencil shapes than inference splits: every refill boxed (14112 B
        # bilinear on this 7x9 grid, D̽ₓ 14544 B) where `D̽ᵧ(R)`, `D̽ᵧ(2v)` and `D₋ₓ(2R)`
        # allocated nothing. The forms below are bilinear and linear both, on graded grids.
        R = restrict_to
        bytes(A, a) = (assemble!(A, a); assemble!(A, a); @allocated assemble!(A, a))
        I1 = interval(0.0, 1.0)
        cases = ((I1, (11,), (D̽ₓ,)),
            # the 2D case runs in `slow`, and the 3D one too (gpena/Bramble.jl#650)
            (WITH_SLOW_TESTS ? ((I1 × I1, (7, 9), (D̽ₓ, D̽ᵧ)),) : ())...,
            (WITH_3D_TESTS ? ((I1 × I1 × I1, (4, 5, 6), (D̽ₓ, D̽ᵧ, D̽₂)),) : ())...)
        for (S, np, Ds) in cases
            Ωg = mesh(domain(S, :l => :left), np, ntuple(_ -> false, length(np)))
            pts = ntuple(d -> range(0.0, 1.0; length = np[d]) .^ (1 + 0.25d), length(np))
            Bramble.change_points!(Ωg, length(np) == 1 ? only(pts) : pts)
            W = gridspace(Ωg)
            g = element(W)
            g .= 1.0 .+ (1:length(g)) ./ 7
            # `wrap(x) == c * unit(x)`: the scaled form must be `c` times the unit one.
            for D in Ds,
                (label, c, wrap, unit) in (("2.0", 2.0, x -> 2.0 * x, identity),
                    ("-1", -1, x -> -x, identity), ("2.0 * g", 2.0, x -> 2.0 * (g * x), x -> g * x))

                @testset "$(length(np))D $D($label R)" begin
                    a = form(W, W, (u, v) -> innerₕ(u, D(wrap(R(:l, v)))))
                    A = assemble(a)
                    A1 = assemble(form(W, W, (u, v) -> innerₕ(u, D(unit(R(:l, v))))))
                    # The restriction is live: it drops some of the unrestricted entries.
                    full = assemble(form(W, W, (u, v) -> innerₕ(u, D(unit(v)))))
                    @test 0 < count(!iszero, A1) < count(!iszero, full)
                    @test A ≈ c * A1
                    @test bytes(A, a) == 0
                    lf = form(W, v -> innerₕ(g, D(wrap(R(:l, v)))))
                    y = assemble(lf)
                    @test y ≈ c * assemble(form(W, v -> innerₕ(g, D(unit(R(:l, v))))))
                    @test bytes(y, lf) == 0
                end
            end
        end

        # A coefficient infinite outside the region it multiplies is zeroed with the region,
        # never `Inf * 0 = NaN`, in every node that taps a neighbour: the vector is the one a
        # finite coefficient gives, for a finite source and one infinite at both ends.
        ns = 9
        Ws = gridspace(mesh(domain(I1), ns, true))
        u_end = [1 < i < ns ? 2.0 : Inf for i in 1:ns]
        u_fin = fill(2.0, ns)
        sources = (source_function(x -> x + 1, Val(1)),
            source_function(x -> 1 / (x * (1 - x)), Val(1)))
        for op in (D₊ₓ, D₋ₓ, D̽ₓ, Dcₓ, Mcₓ, jumpₓ, S₊ₓ), sf in sources

            rhs(u) = assemble(form(Ws, v -> innerₕ(op(2.0 * (u * R(:interior, sf))), v)))
            @test isequal(rhs(u_end), rhs(u_fin))
            # the vector the walk gave before the shift followed the chain (2afe89a4)
            op === D₊ₓ && sf === first(sources) &&
                @test isequal(rhs(u_end), [2.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -7.5, 0.0])
        end

        # A scale over a restriction of a scaled restriction shifts without boxing (6912 B
        # in 2D, 960 B in 1D, when the shift recursed through one scale per restriction).
        # The interior points are random, so the mesh is seeded: an unseeded draw can leave
        # both restrictions' overlap empty, and `want` is all zero (CI, gpena/Bramble.jl#768).
        for (S, np) in ((I1, (11,)), (I1 × I1, (7, 9)))
            Bramble._seed_mesh1d_rng!(2)
            W = try
                gridspace(mesh(domain(S, :c => x -> x[1] < 0.6, :d => x -> x[1] > 0.2), np,
                    ntuple(_ -> false, length(np))))
            finally
                Bramble._unseed_mesh1d_rng!()
            end
            nested(v) = S₊ₓ(2.0 * R(:c, 3 * R(:d, v)))
            want = 6 * assemble(form(W, W, (u, v) -> innerₕ(u, S₊ₓ(R(:c, R(:d, v))))))
            @test count(!iszero, want) > 0
            for (side, fn) in (("test", (u, v) -> innerₕ(u, nested(v))),
                ("trial", (u, v) -> innerₕ(nested(u), v)))
                a = form(W, W, fn)
                A = assemble(a)
                @test A ≈ (side == "test" ? want : transpose(want))
                @test bytes(A, a) == 0
            end
            lf = form(W, v -> innerₕ(1.0, nested(v)))
            y = assemble(lf)
            @test y ≈ 6 * assemble(form(W, v -> innerₕ(1.0, S₊ₓ(R(:c, R(:d, v))))))
            @test bytes(y, lf) == 0
        end
    end

    @testset "Nested difference of restriction" begin
        # A tap over a difference of a restriction re-evaluated it at the neighbour, `()` or
        # a full tuple by marker, and three taps boxed: `D̽ₓ(D̽ᵧ(R(:l, v)))` 83232 B a refill
        # on this grid, `D̽ₓ(D₊ᵧ(R))` 49248 B. The oracle is the restriction as a mask
        # coefficient, which walks no restriction; the stored entries are pinned, since a
        # tap padded with zeros in place stored more (131 -> 176 for `D₋ₓ(D₊ᵧ(R))`).
        R = restrict_to
        bytes(A, a) = (assemble!(A, a); assemble!(A, a); @allocated assemble!(A, a))
        I1 = interval(0.0, 1.0)
        Ωg = mesh(domain(I1 × I1, :l => :left, :r => :right, :b => :bottom), (7, 9),
            (false, false))
        Bramble.change_points!(Ωg,
            (range(0.0, 1.0; length = 7) .^ 1.25, range(0.0, 1.0; length = 9) .^ 1.5))
        W = gridspace(Ωg)
        g = element(W)
        g .= 1.0 .+ (1:length(g)) ./ 7
        ml = Rₕ(W, x -> x[1] == 0 ? 1.0 : 0.0)
        mb = Rₕ(W, x -> x[2] == 0 ? 1.0 : 0.0)
        wraps = (("l", X -> R(:l, X), X -> ml * X),
            ("2b", X -> 2.0 * R(:b, X), X -> 2.0 * (mb * X)))
        # Stored entries (test side, trial side, both sides), as many as at 6eeabebe.
        nests = (("D̽ₓD̽ᵧ", X -> D̽ₓ(D̽ᵧ(X)), (322, 322, 747), (323, 323, 725)),
            ("D̽ₓD₊ᵧ", X -> D̽ₓ(D₊ᵧ(X)), (170, 170, 281), (171, 171, 319)),
            ("D₋ₓD₊ᵧD₋ₓ", X -> D₋ₓ(D₊ᵧ(D₋ₓ(X))), (204, 204, 400), (194, 194, 425)),
            ("McₓD̽ᵧ", X -> Mcₓ(D̽ᵧ(X)), (322, 322, 747), (323, 323, 725)),
            ("D₊ₓD₋ₓ", X -> D₊ₓ(D₋ₓ(X)), (117, 117, 171), (123, 123, 181)),
            # a sum of nested nodes: both sides refilled 463040 B with its tap 0 a union
            ("D̽ₓ(D̽ᵧ+D₋ₓ)", X -> D̽ₓ(D̽ᵧ(X) + D₋ₓ(X)), (466, 466, 1259), (472, 472, 1267)))
        for (n, N, nl, nc) in nests, ((w, wrap, masked), pins) in zip(wraps, (nl, nc))
            # The restriction-by-scale wrap compiles its own forms: it runs in `slow`.
            WITH_SLOW_TESTS || w == "l" || continue

            @testset "$n($w)" begin
                sides = ((u, v) -> innerₕ(u, N(wrap(v))), (u, v) -> innerₕ(N(wrap(u)), v),
                    (u, v) -> innerₕ(N(wrap(u)), N(wrap(v))))
                masks = ((u, v) -> innerₕ(u, N(masked(v))), (u, v) -> innerₕ(N(masked(u)), v),
                    (u, v) -> innerₕ(N(masked(u)), N(masked(v))))
                for (f, fm, k) in zip(sides, masks, pins)
                    a = form(W, W, f)
                    A = assemble(a)
                    @test count(!iszero, A) > 0
                    @test A ≈ assemble(form(W, W, fm))
                    @test nnz(A) == k
                    @test bytes(A, a) == 0
                end
                lf = form(W, v -> innerₕ(g, N(wrap(v))))
                y = assemble(lf)
                @test y ≈ assemble(form(W, v -> innerₕ(g, N(masked(v)))))
                @test bytes(y, lf) == 0
            end
        end

        # A probed restriction's chain is not walked outside its region: a source under a
        # second, disjoint restriction is never called (it throws there), and the vector is
        # zero at 0 B (1440 B when the shifted chain read the inner region outside `:l`).
        sf = source_function(x -> x[1] > 0.5 ? error("read outside :l") : 1.0, Val(2))
        for op in (D̽ₓ, Mcₓ)
            lf = form(W, v -> innerₕ(op(2.0 * R(:l, 3 * R(:r, sf))), v))
            y = assemble(lf)
            @test all(iszero, y)
            @test bytes(y, lf) == 0
        end

        # Restricted sums and scales between the nodes, both sides (each refilled more than
        # before with the outer tap 0 left a union, and `Mₓ(D₊ᵧ(R(:c, R(:l) + R(:r))))`
        # stored 270 entries, not 189, with zeros left at the point).
        Wu = gridspace(mesh(
            domain(I1 × I1, :l => :left, :r => :right, :b => :bottom,
                :t => :top, :c => _x1_lt_half),
            (7, 9), (true, true)))
        m(f) = Rₕ(Wu, x -> f(x) ? 1.0 : 0.0)
        ml, mr = m(x -> x[1] == 0), m(x -> x[1] == 1)
        mb, mt, mc = m(x -> x[2] == 0), m(x -> x[2] == 1), m(_x1_lt_half)
        LR(X) = R(:l, X) + R(:r, X)
        BT(X) = R(:b, X) + R(:t, X)
        for (n, f, fm, k) in (
            ("MD(Rc)", (u, v) -> innerₕ(Mₓ(D₊ᵧ(R(:c, LR(u)))), Mₓ(D₊ᵧ(BT(v)))),
            (u, v) -> innerₕ(Mₓ(D₊ᵧ(mc * (ml * u + mr * u))), Mₓ(D₊ᵧ(mb * v + mt * v))),
            189),
            ("DRD", (u, v) -> innerₕ(D₋ₓ(R(:c, D₊ᵧ(LR(u)))), D₋ₓ(R(:c, D₊ᵧ(BT(v))))),
            (u, v) -> innerₕ(D₋ₓ(mc * D₊ᵧ(ml * u + mr * u)), D₋ₓ(mc * D₊ᵧ(mb * v + mt * v))),
            108),
            ("D̽(2D̽)", (u, v) -> innerₕ(D̽ₓ(2.0 * D̽ᵧ(LR(u))), D̽ₓ(2.0 * D̽ᵧ(LR(v)))),
            (u, v) -> innerₕ(D̽ₓ(2.0 * D̽ᵧ(ml * u + mr * u)), D̽ₓ(2.0 * D̽ᵧ(ml * v + mr * v))),
            827))
            # The two longer chains compile their own forms: they run in `slow`.
            WITH_SLOW_TESTS || n == "MD(Rc)" || continue
            @testset "$n" begin
                a = form(Wu, Wu, f)
                A = assemble(a)
                @test count(!iszero, A) > 0
                @test A ≈ assemble(form(Wu, Wu, fm))
                @test nnz(A) == k
                @test bytes(A, a) == 0
            end
        end
    end

    @testset "Weights of several types refill with 0 B" begin
        # A test side whose summands' weights differ in type boxed at every point: an
        # unscaled or grid-function-scaled summand beside a Float64 literal on a Float32
        # mesh (14112 B on a 7x9 grid), a `Ref` scale, a sum under a scale, a restriction
        # or a difference, and two products of different weight types walked as one
        # (27216 B). A constant scale between a difference and a restricted sum boxed on
        # a Float64 mesh too. Each form must refill a NaN-filled vector with what its terms
        # give assembled apart, keep the element type they promote to, and read a `Ref` and
        # a grid function live; the rewrite is read off the types, so it infers. On a
        # Float64 mesh the terms apart agree to rounding, and a scale converted through
        # Float32 would not; on a Float32 mesh a term once computed in Float32 is now
        # computed in Float64.
        R = restrict_to
        bytes(y, l) = (assemble!(y, l); assemble!(y, l); @allocated assemble!(y, l))
        function refilled(l)
            y = assemble(l)
            fill!(y, NaN)
            assemble!(y, l)
            return y
        end
        f = _one_plus_x1
        for T in (Float32, Float64), D in TEST_DIMS
            # Each element type and dimension compiles its own forms: unit keeps the 1D
            # Float64 case, and the others run in `slow`.
            WITH_SLOW_TESTS || (T === Float64 && D == 1) || continue

            I = interval(zero(T), one(T))
            Bramble._seed_mesh1d_rng!(20261017)
            W = try
                Ω = domain(foldl(×, ntuple(_ -> I, D)), :l => :left, :r => :right,
                    :c => _x1_lt_half)
                gridspace(D == 1 ? mesh(Ω, 11, false; backend = backend(T)) :
                          mesh(Ω, (7, 9, 5)[1:D], ntuple(_ -> false, D);
                    backend = backend(T)))
            finally
                Bramble._unseed_mesh1d_rng!()
            end
            g = element(W)
            g .= 1 .+ (1:length(g)) ./ 7
            r = Ref(T(0.1))
            RR(v) = R(:l, v) + R(:r, v)
            tol = T === Float32 ? 8 * eps(Float32) : 1.0e-12
            forms = (
                ("(1//3)v + 0.5D₋ₓ(v)", v -> innerₕ(f, (1 // 3) * v + 0.5 * D₋ₓ(v)),
                    (v -> innerₕ(f, (1 // 3) * v), v -> innerₕ(f, 0.5 * D₋ₓ(v)))),
                ("πv + 0.5D₋ₓ(v)", v -> innerₕ(f, π * v + 0.5 * D₋ₓ(v)),
                    (v -> innerₕ(f, π * v), v -> innerₕ(f, 0.5 * D₋ₓ(v)))),
                ("(1//3)R(:l, v) + 2.0R(:r, v) + 0.5v",
                    v -> innerₕ(f, (1 // 3) * R(:l, v) + 2.0 * R(:r, v) + 0.5 * v),
                    (v -> innerₕ(f, (1 // 3) * R(:l, v)), v -> innerₕ(f, 2.0 * R(:r, v)),
                        v -> innerₕ(f, 0.5 * v))),
                ("g*v + 0.5D₋ₓ(v)", v -> innerₕ(g, g * v + 0.5 * D₋ₓ(v)),
                    (v -> innerₕ(g, g * v), v -> innerₕ(g, 0.5 * D₋ₓ(v)))),
                ("(f, 2v + 3D₋ₓ(v)) + (1.0, v)",
                    v -> innerₕ(f, 2 * v + 3 * D₋ₓ(v)) + innerₕ(1.0, v),
                    (v -> innerₕ(f, 2 * v), v -> innerₕ(f, 3 * D₋ₓ(v)),
                        v -> innerₕ(1.0, v))),
                ("(g, v) + (1.0, D₋ₓ(v))", v -> innerₕ(g, v) + innerₕ(1.0, D₋ₓ(v)),
                    (v -> innerₕ(g, v), v -> innerₕ(1.0, D₋ₓ(v)))),
                ("v + 0.1D₋ₓ(v)", v -> innerₕ(f, v + 0.1 * D₋ₓ(v)),
                    (v -> innerₕ(f, v), v -> innerₕ(f, 0.1 * D₋ₓ(v)))),
                ("g*(0.1v + 3D₋ₓ(v))", v -> innerₕ(f, g * (0.1 * v + 3 * D₋ₓ(v))),
                    (v -> innerₕ(f, g * (0.1 * v)), v -> innerₕ(f, g * (3 * D₋ₓ(v))))),
                ("r*v + 0.5D₋ₓ(v)", v -> innerₕ(f, r * v + 0.5 * D₋ₓ(v)),
                    (v -> innerₕ(f, r[] * v), v -> innerₕ(f, 0.5 * D₋ₓ(v)))),
                ("R(:c, 0.1v + 3D₋ₓ(v))", v -> innerₕ(f, R(:c, 0.1 * v + 3 * D₋ₓ(v))),
                    (v -> innerₕ(f, R(:c, 0.1 * v)), v -> innerₕ(f, R(:c, 3 * D₋ₓ(v))))),
                ("D₋ₓ(0.1v + 3D₊ₓ(v))", v -> innerₕ(f, D₋ₓ(0.1 * v + 3 * D₊ₓ(v))),
                    (v -> innerₕ(f, D₋ₓ(0.1 * v)), v -> innerₕ(f, D₋ₓ(3 * D₊ₓ(v))))),
                ("D₋ₓ(2.0RR(v))", v -> innerₕ(f, D₋ₓ(2.0 * RR(v))),
                    (v -> innerₕ(f, D₋ₓ(2.0 * R(:l, v))),
                        v -> innerₕ(f, D₋ₓ(2.0 * R(:r, v))))))
            for (n, whole, parts) in forms
                @testset "$(T) $(D)D $n" begin
                    ast = Bramble.simplify_ast(Bramble._resolve_surfaces_for_space(
                        Bramble.resolve_ast(whole(Bramble.test_function(W))), W))
                    @inferred Bramble._split_restricted_products(ast, T)
                    l = form(W, whole)
                    want = sum(assemble(form(W, p)) for p in parts)
                    y = refilled(l)
                    @test eltype(y) === eltype(want)
                    @test count(!iszero, want) > 0
                    @test isapprox(y, want; rtol = tol)
                    @test bytes(y, l) == 0
                end
            end
            # The `Ref` and the grid function are read at each refill, not at `form`.
            l = form(W, v -> innerₕ(f, r * v + 0.5 * D₋ₓ(v)))
            r[] = T(0.7)
            want = assemble(form(W, v -> innerₕ(f, T(0.7) * v))) +
                   assemble(form(W, v -> innerₕ(f, 0.5 * D₋ₓ(v))))
            @test isapprox(refilled(l), want; rtol = tol)
            l = form(W, v -> innerₕ(g, g * v + 0.5 * D₋ₓ(v)))
            g .*= 2
            want = assemble(form(W, v -> innerₕ(g, g * v))) +
                   assemble(form(W, v -> innerₕ(g, 0.5 * D₋ₓ(v))))
            @test isapprox(refilled(l), want; rtol = tol)
            # Under a restriction a scale over a restricted sum stays whole: distributed,
            # every piece is a restriction over a difference over a restriction, which
            # boxes (61920 B against 16992 B whole on 7x9). A restriction over a difference
            # over a restricted sum is kept as written as one group (61920 B regrouped)
            # and distributed over three (92016 B kept whole).
            if T === Float64 && D == 2
                Bramble._seed_mesh1d_rng!(20261017)
                Wb = try
                    gridspace(mesh(
                        domain(I × I, :l => :left, :r => :right, :b => :bottom,
                            :c => _x1_lt_half), (7, 9), (false, false)))
                finally
                    Bramble._unseed_mesh1d_rng!()
                end
                for (whole, parts, cap) in (
                    (v -> R(:c, D₋ₓ(2.0 * RR(v))),
                    (v -> R(:c, D₋ₓ(2.0 * R(:l, v))), v -> R(:c, D₋ₓ(2.0 * R(:r, v)))),
                    16992),
                    (v -> R(:c, D₋ₓ(RR(v))),
                    (v -> R(:c, D₋ₓ(R(:l, v))), v -> R(:c, D₋ₓ(R(:r, v)))), 54432),
                    (v -> R(:c, D₋ₓ(RR(v) + R(:b, v))),
                    (v -> R(:c, D₋ₓ(R(:l, v))), v -> R(:c, D₋ₓ(R(:r, v))),
                        v -> R(:c, D₋ₓ(R(:b, v)))), 61920))
                    l = form(Wb, v -> innerₕ(f, whole(v)))
                    want = sum(p -> assemble(form(Wb, v -> innerₕ(f, p(v)))), parts)
                    @test count(!iszero, want) > 0
                    @test isapprox(refilled(l), want; rtol = tol)
                    @test bytes(assemble(l), l) <= cap
                end
            end
        end
    end
end

end # module FormLinearRestrictionTests
