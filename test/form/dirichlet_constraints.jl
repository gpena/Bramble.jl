import Bramble: CartesianProduct, DirichletConstraint, label_conditions, embed_function,
                symbols, labels, DomainMarkers, tuples, conditions, identifier,
                EvaluatedDomainMarkers, BrambleFunction, label, markers, point,
                index_in_marker
using Supposition

@testset "Dirichlet constraints" begin
    # --- Setup ---
    I = interval(0.0, 1.0)
    Ω = I × I

    @testset "Boundary conditions" begin
        # Define some functions to use as boundary conditions
        f1 = x -> x[1]^2 + x[2]
        f2 = x -> 2 * x[2]

        # Define a time-dependent function: f(x, t)
        f_t = (x, t) -> x[1] * t

        # Wrap them using the embed_function(macro
        bf1 = embed_function(Ω, f1)
        bf2 = embed_function(Ω, f2)
        bf_t = embed_function(Ω, I, f_t)

        # --- Tests ---

        @testset "Constructor" begin
            bcs = dirichlet_constraints(Ω, :gamma_1 => x->bf1(x), :gamma_2 => x->bf2(x))

            @test bcs isa DirichletConstraint
            @test length(label_conditions(bcs)) == 2
        end

        @testset "Time-dependent functor" begin
            # Create a time-dependent constraint. Point 48 (2026-09-04): the raw two-argument
            # closure is stored directly now, in `DomainMarkers.conditions` -- a `Tuple`, one
            # `Marker{F}` per condition's own type -- rather than a `BrambleFunction`.
            bcs_t = dirichlet_constraints(Ω, I, :time_dep_bc => (x, t) -> f_t(x, t))

            function_markers = bcs_t.conditions
            function_snapshot = first(function_markers)
            @test length(function_markers) == 1
            @test label(function_snapshot) == :time_dep_bc

            # called directly as f(x, t) -- there is no wrapper-provided f(t)(x) currying
            x_point = (10.0, 5.0)
            t_point = 0.5
            @test identifier(function_snapshot)(x_point, t_point) == f_t(x_point, t_point)
        end
    end

    @testset "Lazy time evaluation" begin
        original_markers = markers(
            Ω, I, :moving_front => (x, t) -> x[1] > t, :moving_back => (x, t) -> x[1] < t)
        lazy_markers_at_t = EvaluatedDomainMarkers(original_markers, 0.75)

        @test lazy_markers_at_t isa EvaluatedDomainMarkers
        @test lazy_markers_at_t.evaluation_time == 0.75

        @test symbols(lazy_markers_at_t) === symbols(original_markers)
        evaluated_conditions = collect(conditions(lazy_markers_at_t))

        @test length(evaluated_conditions) == 2
        for marker in evaluated_conditions
            # a plain one-argument closure now, x -> f(x, 0.75) via Base.Fix2 -- not a
            # BrambleFunction, and no longer callable as new_bf(x, t)
            new_bf = identifier(marker)

            if label(marker) == :moving_front
                # equivalent to `x -> x[1] > 0.75`
                @test new_bf(0.8) == true
                @test new_bf(0.7) == false
            end
        end
    end
end

using SparseArrays
using LinearAlgebra: I as LinearAlgebraI

# Imposing the constraints, on scalar and on composite spaces.
#
# The composite case is the one worth pinning: it flattens a possibly nested space into
# leaves with dof offsets, and every property below is that flattening being right. The
# governing equivalence is that a composite space behaves exactly like the scalar space
# repeated once per component, block by block.
@testset "Applying conditions" begin
    Ωₕ = mesh(
        domain(interval(0.0, 1.0) × interval(0.0, 1.0),
            :bottom => :bottom, :top => :top),
        (5, 6), (true, true))
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(3))
    nW, nV = ndofs(Wₕ), ndofs(Vₕ)
    marked = index_in_marker(Ωₕ, :bottom)

    _eye(n) = sparse(one(Float64) * LinearAlgebraI, n, n)
    _full(n) = Matrix(_eye(n))

    @testset "Matrix rows (scalar)" begin
        A = _eye(nW)
        A[1, 2] = 5.0                      # an off-diagonal that must be cleared
        @test dirichlet_bc!(A, Wₕ, :bottom) === A
        for i in 1:nW
            if marked[i]
                @test A[i, i] == 1.0
                @test all(A[i, j] == 0.0 for j in 1:nW if j != i)
            end
        end
        @test any(marked)                  # the marker selects something
    end

    @testset "Dense & sparse agreement" begin
        As, Ad = _eye(nW), _full(nW)
        As[2, 3] = 4.0
        Ad[2, 3] = 4.0
        @test dirichlet_bc!(As, Wₕ, :bottom) === As
        @test dirichlet_bc!(Ad, Wₕ, :bottom) === Ad
        @test Matrix(As) == Ad
    end

    @testset "Matrix rows (composite)" begin
        A = _eye(nV)
        @test dirichlet_bc!(A, Vₕ, :bottom) === A
        # a composite space is the scalar one repeated per component: the marked rows are
        # the marked scalar rows shifted by each component's offset
        for c in 0:2, i in 1:nW

            row = c * nW + i
            if marked[i]
                @test A[row, row] == 1.0
                @test count(!=(0.0), A[row, :]) == 1
            end
        end
        @test nV == 3nW
    end

    @testset "Vector values" begin
        bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> 7.0))

        v = fill(-1.0, nW)
        @test dirichlet_bc!(v, Wₕ, bcs, :bottom) === v
        @test all(v[i] == 7.0 for i in 1:nW if marked[i])
        @test all(v[i] == -1.0 for i in 1:nW if !marked[i])   # untouched elsewhere

        w = fill(-1.0, nV)
        @test dirichlet_bc!(w, Vₕ, bcs, :bottom) === w
        for c in 0:2
            block = view(w, (c * nW + 1):((c + 1) * nW))
            @test block == v          # every component gets the scalar answer
        end
    end

    @testset "Component restriction" begin
        # The Stokes-style case this exists for: constrain one field, leave another free.
        @testset "Matrix (single leaf)" begin
            A = _eye(nV)
            @test dirichlet_bc!(A, Vₕ, :bottom; components = 1) === A
            for c in 0:2, i in 1:nW

                row = c * nW + i
                if c == 0 && marked[i]
                    @test A[row, row] == 1.0
                    @test count(!=(0.0), A[row, :]) == 1
                else
                    # leaves 2 and 3 (c = 1, 2) are untouched: still the identity
                    @test A[row, row] == 1.0
                    @test count(!=(0.0), A[row, :]) == 1
                    @test A[row, :] == _eye(nV)[row, :]
                end
            end
        end

        @testset "Matrix (multiple leaves)" begin
            A = _eye(nV)
            @test dirichlet_bc!(A, Vₕ, :bottom; components = (1, 3)) === A
            for c in 0:2, i in 1:nW

                row = c * nW + i
                if c in (0, 2) && marked[i]
                    @test A[row, row] == 1.0
                    @test count(!=(0.0), A[row, :]) == 1
                end
            end
            # leaf 2 (c = 1) never touched, whatever :bottom marks
            @test A[(nW + 1):(2nW), :] == _eye(nV)[(nW + 1):(2nW), :]
        end

        @testset "Vector (single leaf)" begin
            bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> 7.0))
            w = fill(-1.0, nV)
            @test dirichlet_bc!(w, Vₕ, bcs, :bottom; components = 2) === w
            for c in 0:2
                block = view(w, (c * nW + 1):((c + 1) * nW))
                if c == 1
                    @test all(block[i] == 7.0 for i in 1:nW if marked[i])
                    @test all(block[i] == -1.0 for i in 1:nW if !marked[i])
                else
                    @test all(==(-1.0), block)   # untouched leaves
                end
            end
        end

        @testset "Unrestricted default" begin
            A1, A2 = _eye(nV), _eye(nV)
            dirichlet_bc!(A1, Vₕ, :bottom)
            dirichlet_bc!(A2, Vₕ, :bottom; components = nothing)
            @test A1 == A2
        end

        @testset "symmetrize! keyword" begin
            A = Matrix(_eye(nV))
            F = fill(2.0, nV)
            A0 = copy(A)
            symmetrize!(A, F, Vₕ, :bottom; components = 1)
            # only leaf 1's marked rows/columns could have changed anything
            for c in 1:2, i in 1:nW

                row = c * nW + i
                @test A[:, row] == A0[:, row]
            end
        end

        @testset "Out-of-range component error" begin
            A = _eye(nV)
            @test_throws ArgumentError dirichlet_bc!(A, Vₕ, :bottom; components = 4)
            @test_throws ArgumentError dirichlet_bc!(A, Vₕ, :bottom; components = 0)
            @test_throws ArgumentError dirichlet_bc!(A, Vₕ, :bottom; components = (1, 5))
        end

        @testset "Scalar single leaf" begin
            A = _eye(nW)
            @test dirichlet_bc!(A, Wₕ, :bottom; components = 1) === A   # a no-op-equivalent ok
            @test_throws ArgumentError dirichlet_bc!(_eye(nW), Wₕ, :bottom; components = 2)
        end

        @testset "Component argument type" begin
            @test_throws ErrorException dirichlet_bc!(_eye(nV), Vₕ, :bottom; components = :left)
        end
    end

    @testset "Evaluation points" begin
        bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> x[1] + 10x[2]))
        v = zeros(nW)
        @test dirichlet_bc!(v, Wₕ, bcs, :bottom) === v
        pts = [point(Ωₕ, idx) for idx in indices(Ωₕ)]
        for i in 1:nW
            marked[i] && @test v[i] ≈ pts[i][1] + 10pts[i][2]
        end
    end

    @testset "Multiple labels" begin
        A = _eye(nW)
        @test dirichlet_bc!(A, Wₕ, :bottom, :top) === A
        both = index_in_marker(Ωₕ, :bottom) .| index_in_marker(Ωₕ, :top)
        for i in 1:nW
            both[i] && @test A[i, i] == 1.0
        end
        @test count(both) > count(marked)     # :top really adds rows
    end

    @testset "Empty & missing labels" begin
        A0 = _eye(nW)
        A1 = copy(A0)
        @test dirichlet_bc!(A1, Wₕ) === A1                 # no labels at all
        @test A1 == A0
        v0 = fill(3.0, nW)
        bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> 7.0))
        @test dirichlet_bc!(v0, Wₕ, bcs) === v0            # no labels
        @test all(==(3.0), v0)
    end

    @testset "Set/mesh/space construction" begin
        # `dirichlet_constraints` takes whichever of the three the caller has to hand, and
        # digs out the underlying `CartesianProduct` itself. For a composite space that
        # means its first leaf — the constraint is over the domain, and every leaf of a
        # composite space shares it.
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
            (6, 6), (true, true))
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(3))
        g = x -> 7.0

        from_set = dirichlet_constraints(set(Ωₕ), :bottom => g)
        for src in (Ωₕ, Wₕ, Vₕ)
            c = dirichlet_constraints(src, :bottom => g)
            @test c isa DirichletConstraint
            @test symbols(c) == symbols(from_set)

            v = fill(3.0, ndofs(Wₕ))
            w = fill(3.0, ndofs(Wₕ))
            dirichlet_bc!(v, Ωₕ, c, :bottom)
            dirichlet_bc!(w, Ωₕ, from_set, :bottom)
            @test v == w
        end

        # and the same three, with a time interval, for a time-dependent condition
        Iₜ = interval(0.0, 1.0)
        for src in (set(Ωₕ), Ωₕ, Wₕ, Vₕ)
            @test dirichlet_constraints(src, Iₜ, :bottom => ((x, t) -> t * x[1])) isa
                  DirichletConstraint
        end
    end

    @testset "Nested composite spaces" begin
        # A `CompositeGridSpace` may hold composite spaces, so the leaves form a tree
        # rather than a list. The traversal flattens it depth first, and the offsets have
        # to keep running across the nesting rather than restarting inside each branch.
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
            (5, 5), (true, true))
        Wₕ = gridspace(Ωₕ)
        n = ndofs(Wₕ)
        inner = gridspace(Ωₕ, Val(2))
        nested = Bramble.CompositeGridSpace((Wₕ, inner, Wₕ))

        # a scalar space is its own only leaf, at offset zero
        @test Bramble.first_space(Wₕ) === Wₕ
        @test Bramble.leaf_spaces_offsets(Wₕ) == ((Wₕ, 0),)
        @test Bramble.n_leaf_spaces(Wₕ) == 1

        leaves = Bramble.leaf_spaces_offsets(nested)
        @test length(leaves) == 4
        @test Bramble.n_leaf_spaces(nested) == 4
        @test map(last, leaves) == (0, n, 2n, 3n)
        @test ndofs(nested) == 4n

        # the flat four-component space marks exactly the same rows
        flat = gridspace(Ωₕ, Val(4))
        An = _eye(4n)
        Af = _eye(4n)
        dirichlet_bc!(An, nested, :bottom)
        dirichlet_bc!(Af, flat, :bottom)
        @test An == Af

        bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> 7.0))
        vn, vf = fill(3.0, 4n), fill(3.0, 4n)
        dirichlet_bc!(vn, nested, bcs, :bottom)
        dirichlet_bc!(vf, flat, bcs, :bottom)
        @test vn == vf
    end

    @testset "Zero allocations" begin
        # This runs once per step of a time loop, so it has to cost nothing beyond the
        # work itself. Two things had to go for that: the leaves used to come back in a
        # `Vector{Tuple{Any, Int}}`, which made every read through a leaf dynamic and
        # boxed a Bool per degree of freedom — 809 KB for one call on a 60x60 grid with
        # three components — and the composite matrix path used to build a BitVector over
        # the whole system to hold the marked rows.
        #
        # Both are gone: `leaf_spaces_offsets` answers with a tuple, and each leaf's mask
        # is read at an offset rather than copied. Measured inside a function, on concrete
        # locals, so nothing boxes at the call boundary and the reading is the real one.
        function counts(n)
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
                (n, n), (true, true))
            W, V = gridspace(Ω), gridspace(Ω, Val(3))
            bcs = dirichlet_constraints(set(Ω), :bottom => (x -> 7.0))

            Aw, Av = _eye(ndofs(W)), _eye(ndofs(V))
            vw, vv = zeros(ndofs(W)), zeros(ndofs(V))
            # warm up every path before measuring it
            dirichlet_bc!(Aw, W, :bottom)
            dirichlet_bc!(Av, V, :bottom)
            dirichlet_bc!(vw, W, bcs, :bottom)
            dirichlet_bc!(vv, V, bcs, :bottom)

            dirichlet_bc!(Av, V, :bottom; components = 1)
            dirichlet_bc!(vv, V, bcs, :bottom; components = 1)

            return (matrix_scalar = @allocated(dirichlet_bc!(Aw, W, :bottom)),
                matrix_composite = @allocated(dirichlet_bc!(Av, V, :bottom)),
                vector_scalar = @allocated(dirichlet_bc!(vw, W, bcs, :bottom)),
                vector_composite = @allocated(dirichlet_bc!(vv, V, bcs, :bottom)),
                # `components` restricts the same tuple walk, not a fresh Vector — this must
                # cost the same nothing as the unrestricted call above.
                matrix_one_component = @allocated(dirichlet_bc!(
                    Av, V, :bottom; components = 1)),
                vector_one_component = @allocated(dirichlet_bc!(
                    vv, V, bcs, :bottom; components = 1)))
        end

        for n in (10, 40)          # 16x the degrees of freedom apart
            c = counts(n)
            @test c.matrix_scalar == 0
            @test c.matrix_composite == 0
            @test c.vector_scalar == 0
            @test c.vector_composite == 0
            @test c.matrix_one_component == 0
            @test c.vector_one_component == 0
        end

        # The traversal the composite paths walk is itself free, and type stable. Measured
        # inside a function: read from a non-const global instead, the space boxes at the
        # call boundary and the reading is of that box, not of the traversal.
        function traversal_bytes(V)
            Bramble.leaf_spaces_offsets(V)
            return @allocated Bramble.leaf_spaces_offsets(V)
        end

        Ωt = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
            (8, 8), (true, true))
        Vt = gridspace(Ωt, Val(3))
        @test @inferred(Bramble.leaf_spaces_offsets(Vt)) isa Tuple
        @test isconcretetype(typeof(Bramble.leaf_spaces_offsets(Vt)))
        @test traversal_bytes(Vt) == 0
    end

    @testset "Arbitrary fields (Supposition)" begin
        field_val = Data.Floats{Float64}(; minimum = -100.0, maximum = 100.0,
            nans = false, infs = false)

        @check function check_dirichlet_invariance_2d(
                nx = Data.Integers(4, 10),
                ny = Data.Integers(4, 10),
                v_raw = Data.Vectors(field_val; min_size = 100, max_size = 100)
        )
            Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0),
                :bottom => :bottom, :top => :top)
            Ωₕ = mesh(Ω, (nx, ny), (false, false))
            Wₕ = gridspace(Ωₕ)
            n = ndofs(Wₕ)

            v = copy(v_raw[1:n])
            v_orig = copy(v)

            bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> 2.5 * x[1] + 1.0))
            dirichlet_bc!(v, Wₕ, bcs, :bottom)

            marked = index_in_marker(Ωₕ, :bottom)
            pts = [point(Ωₕ, idx) for idx in indices(Ωₕ)]

            # 1. Marked boundary nodes match prescribed values
            ok_marked = all(
                isapprox(v[i], 2.5 * pts[i][1] + 1.0; atol = 1e-12)
            for i in 1:n if marked[i])

            # 2. Unmarked nodes remain strictly bitwise unchanged
            ok_unmarked = all(v[i] == v_orig[i] for i in 1:n if !marked[i])

            # 3. Idempotence: applying again produces identical result
            v_after = copy(v)
            dirichlet_bc!(v_after, Wₕ, bcs, :bottom)
            ok_idem = (v_after == v)

            ok_marked && ok_unmarked && ok_idem
        end
    end
end
