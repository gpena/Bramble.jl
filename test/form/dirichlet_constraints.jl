module FormDirichletConstraintsTests

using Test
using Bramble
import Bramble:
                CartesianProduct,
                DirichletConstraint,
                label_conditions,
                symbols,
                labels,
                DomainMarkers,
                tuples,
                conditions,
                identifier,
                EvaluatedDomainMarkers,
                label,
                markers,
                point,
                index_in_marker,
                CompositeGridSpace,
                indices
using Bramble: set
using Supposition
using ..TestUtils: WITH_SLOW_TESTS, alloc_test
using ..TestUtils: _tri

@testset "Dirichlet constraints" begin
    # --- Setup ---
    I = interval(0.0, 1.0)
    Ω = I × I

    @testset "Boundary conditions" begin
        # Boundary condition functions are stored directly without BrambleFunction-wrapping.
        f1 = x -> x[1]^2 + x[2]
        f2 = x -> 2 * x[2]

        # Define a time-dependent function: f(x, t)
        f_t = (x, t) -> x[1] * t

        # A `Domain` over the same geometry as `Ω`, registering every label this
        # testset's `dirichlet_constraints` calls use -- the construction-time label
        # check needs something real to validate against, which the bare `Ω` (used
        # as-is by `markers(Ω, ...)` below, which needs a `CartesianProduct`) does not
        # carry.
        Ωd = domain(Ω, :gamma_1 => :left, :gamma_2 => :right, :time_dep_bc => :left)

        # --- Tests ---

        @testset "Constructor" begin
            bcs = dirichlet_constraints(Ωd, :gamma_1 => f1, :gamma_2 => f2)

            @test bcs isa DirichletConstraint
            @test length(label_conditions(bcs)) == 2
        end

        @testset "Rejects an input with no domain" begin
            # A bad `input` is refused by a check that names the accepted types, not by a
            # MethodError from `set`, an internal accessor the caller never named.
            @test_throws "must be a CartesianProduct" dirichlet_constraints(
                "not a domain", :gamma_1 => f1
            )
        end

        @testset "Time-dependent functor" begin
            # Create a time-dependent constraint. The raw two-argument closure is stored
            # directly in `DomainMarkers.conditions`: a `Tuple`, one `Marker{F}` per
            # condition's own type, rather than a `BrambleFunction`.
            bcs_t = dirichlet_constraints(Ωd, I, :time_dep_bc => (x, t) -> f_t(x, t))

            function_markers = bcs_t.conditions
            function_snapshot = first(function_markers)
            @test length(function_markers) == 1
            @test label(function_snapshot) == :time_dep_bc

            # called directly as f(x, t) -- there is no wrapper-provided f(t)(x) currying
            x_point = (10.0, 5.0)
            t_point = 0.5
            @test identifier(function_snapshot)(x_point, t_point) == f_t(x_point, t_point)
        end

        @testset "Time domain rejects space-only BCs" begin
            # A time domain alongside a `func(x)`-only condition has no effect and would
            # break only once `bcs(t)` is called during assembly. It is caught by arity, at
            # construction.
            @test_throws "must accept (x, t)" dirichlet_constraints(Ωd, I, :gamma_1 => f1)
        end

        # For every accepted `input` type.
        @testset "Label validation, every `input` type" begin
            # A mistyped or nonexistent label used to pass `dirichlet_constraints`
            # silently and only fail (or, on a composite space with `dirichlet_components`,
            # silently do nothing) once `assemble`/`dirichlet_bc!` reached it, far from the
            # mistake. `_dirichlet_known_labels` has one method per accepted `input` type;
            # each needs its own check that an unregistered label is actually rejected, and
            # that a registered one still goes through.
            Ωdm = mesh(Ωd, (5, 5), (true, true))
            Wdm = gridspace(Ωdm)
            Vdm = Wdm × Wdm

            for input in (Ωd, Ωdm, Wdm, Vdm)
                @test_throws "is not registered" dirichlet_constraints(
                    input, :not_a_real_label => f1
                )
                bcs = dirichlet_constraints(input, :gamma_1 => f1)
                @test bcs isa DirichletConstraint
            end

            # A bare `CartesianProduct` (never wrapped by `domain(...)`) has no custom
            # labels at all -- only the generic per-axis names and their aliases.
            @test_throws "is not registered" dirichlet_constraints(Ω, :gamma_1 => f1)
            known = first(boundary_symbols(Ω))
            @test dirichlet_constraints(Ω, known => f1) isa DirichletConstraint
            # Coordinate-aligned symbol and legacy viewpoint alias both succeed on CartesianProduct
            @test dirichlet_constraints(Ω, :xmin => f1) isa DirichletConstraint
            @test dirichlet_constraints(Ω, :left => f1) isa DirichletConstraint

            # Alias resolution on Domain: domain has :gamma_1 but also accepts alias when symbol registered
            Ω_alias = domain(Ω, :left => :left)
            @test dirichlet_constraints(Ω_alias, :xmin => f1) isa DirichletConstraint

            # And the time-dependent constructor (`input, I::CartesianProduct{1}, pairs...`)
            # validates the same way, before arity is even checked.
            @test_throws "is not registered" dirichlet_constraints(
                Ωd, I, :not_a_real_label => ((x, t) -> f_t(x, t))
            )
        end
    end

    @testset "Lazy time evaluation" begin
        original_markers = markers(
            Ω, I, :moving_front => (x, t) -> x[1] > t, :moving_back => (x, t) -> x[1] < t
        )
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
using Random
using LinearAlgebra: I as LinearAlgebraI

# The fields `_dirichlet_bc_device!` reads from a device CSR matrix (`rowPtr`, `colVal`,
# `nzVal`, and a host `mirror` with `rowptr`/`colval`/`nzval`), on host arrays.
struct _MockDeviceCSR{M} <: AbstractMatrix{Float64}
    rowPtr::Vector{Int}
    colVal::Vector{Int}
    nzVal::Vector{Float64}
    mirror::M
end
Base.size(A::_MockDeviceCSR) = (length(A.rowPtr) - 1, length(A.rowPtr) - 1)

# Imposing the constraints, on scalar and on composite spaces.
#
# The composite case is the one worth pinning: it flattens a possibly nested space into
# leaves with dof offsets, and every property below is that flattening being right. The
# governing equivalence is that a composite space behaves exactly like the scalar space
# repeated once per component, block by block.
@testset "Applying conditions" begin
    Ωₕ = mesh(
        domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom, :top => :top),
        (5, 6),
        (true, true)
    )
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

        # Alias verification. :ymin on Wₕ (registered as :bottom) yields an identical constrained matrix
        A_alias = _eye(nW)
        A_alias[1, 2] = 5.0
        dirichlet_bc!(A_alias, Wₕ, :ymin)
        @test A_alias == A
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
        A = _tri(nV)
        @test dirichlet_bc!(A, Vₕ, :bottom) === A
        # a composite space is the scalar one repeated per component: the marked rows are
        # the marked scalar rows shifted by each component's offset. A row of `_tri` has
        # off-diagonals, so a pinned row is told from an untouched one.
        for c in 0:2, i in 1:nW

            row = c * nW + i
            if marked[i]
                @test A[row, row] == 1.0
                @test count(!=(0.0), A[row, :]) == 1
            else
                @test A[row, :] == _tri(nV)[row, :]
            end
        end
        @test nV == 3nW
    end

    # A scalar test space against a composite trial space gives more columns than rows,
    # while the mask has one entry per row: the sparse sweep must not read it at a column
    # index past the last row.
    @testset "Scalar rows, wider composite trial" begin
        Random.seed!(20260927)
        W1 = gridspace(mesh(domain(interval(0.0, 1.0)), 9, false))
        bf = form(W1 × W1, W1, (u, v) -> innerₕ(u(1), v))
        A0 = assemble(bf)
        A = assemble(bf; dirichlet = :boundary)
        @test size(A) == (9, 18)
        boundary = index_in_marker(mesh(W1), :boundary)
        @test count(boundary) == 2
        for i in 1:9
            if boundary[i]
                @test A[i, i] == 1.0
                @test count(!=(0.0), A[i, :]) == 1
            else
                @test A[i, :] == A0[i, :]
            end
        end
    end

    # The reverse shape: fewer columns than rows. A marked row past the last column has no
    # diagonal, so it is zeroed and gets no identity entry; dense and sparse must agree.
    @testset "Rows past the last column" begin
        Random.seed!(20260928)
        W1 = gridspace(mesh(domain(interval(0.0, 1.0)), 9, false))
        W2 = W1 × W1
        boundary = index_in_marker(mesh(W1), :boundary)
        for (space, marked, nc) in ((W1, boundary, 5), (W2, vcat(boundary, boundary), 10))
            nr = length(marked)
            A0 = rand(nr, nc) .+ 1.0
            Ad, As = copy(A0), sparse(A0)
            @test dirichlet_bc!(Ad, space, :boundary) === Ad
            @test dirichlet_bc!(As, space, :boundary) === As
            @test Matrix(As) == Ad
            for r in 1:nr
                if !marked[r]
                    @test Ad[r, :] == A0[r, :]
                elseif r <= nc
                    @test Ad[r, r] == 1.0
                    @test count(!=(0.0), Ad[r, :]) == 1
                else
                    @test iszero(Ad[r, :])
                end
            end
            @test any(r -> marked[r] && r > nc, 1:nr)
        end
    end

    @testset "Vector values" begin
        bcs = dirichlet_constraints(Ωₕ, :bottom => (x -> 7.0))

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

    # The single-label leaf entries the composite semidiscretisation walks
    # (`_each_dirichlet_row`, problems/semidiscrete_constraints.jl): one entry per leaf,
    # each carrying that leaf's own mask, its global offset, its size and whether
    # `components` selects it.
    @testset "Leaf entries, one label" begin
        leaves = Bramble.leaf_spaces_offsets(Vₕ)
        for (components, active) in ((nothing, (true, true, true)), (2, (false, true, false)), (
            (1, 3), (true, false, true)))
            entries = Bramble._leaf_entries(leaves, :bottom, components)
            @test length(entries) == 3
            for c in 1:3
                mask, offset, n, act = entries[c]
                @test mask == marked
                @test offset == (c - 1) * nW
                @test n == nW
                @test act == active[c]
            end
            for row in 1:nV
                c, i = divrem(row - 1, nW) .+ (1, 1)
                @test Bramble._row_marked(entries, row) == (active[c] && marked[i])
            end
        end
        @test !Bramble._row_marked((), 1)
    end

    # A device CSR matrix cannot grow a missing diagonal: a constrained row without one
    # throws before anything is written. The duck-typed fields stand in for a device type
    # (no device backend is loaded in the tests); the throw comes before the kernel launch.
    @testset "Device CSR: missing diagonal refused" begin
        mirror = (rowptr = [1, 2, 3], colval = [2, 2], nzval = [5.0, 6.0])
        A = _MockDeviceCSR(mirror.rowptr, mirror.colval, copy(mirror.nzval), mirror)
        entries = ((BitVector([true, false]), 0, 2, true),)
        @test_throws "constrained row 1 of a device sparse matrix has no stored diagonal" Bramble._dirichlet_bc_device!(
            A, entries
        )
        @test mirror.nzval == [5.0, 6.0]
        @test A.nzVal == [5.0, 6.0]
    end

    @testset "Component restriction" begin
        # The Stokes-style case this exists for: constrain one field, leave another free.
        @testset "Matrix (single leaf)" begin
            A = _tri(nV)
            @test dirichlet_bc!(A, Vₕ, :bottom; components = 1) === A
            for c in 0:2, i in 1:nW

                row = c * nW + i
                if c == 0 && marked[i]
                    @test A[row, row] == 1.0
                    @test count(!=(0.0), A[row, :]) == 1
                else
                    # every other row, whether in leaf 1 or in leaves 2 and 3 (c = 1, 2),
                    # keeps its off-diagonals
                    @test A[row, :] == _tri(nV)[row, :]
                end
            end
        end

        @testset "Matrix (multiple leaves)" begin
            A = _tri(nV)
            @test dirichlet_bc!(A, Vₕ, :bottom; components = (1, 3)) === A
            for c in 0:2, i in 1:nW

                row = c * nW + i
                if c in (0, 2) && marked[i]
                    @test A[row, row] == 1.0
                    @test count(!=(0.0), A[row, :]) == 1
                end
            end
            # leaf 2 (c = 1) never touched, whatever :bottom marks
            @test A[(nW + 1):(2nW), :] == _tri(nV)[(nW + 1):(2nW), :]
        end

        @testset "Vector (single leaf)" begin
            bcs = dirichlet_constraints(Ωₕ, :bottom => (x -> 7.0))
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
            A1, A2 = _tri(nV), _tri(nV)
            dirichlet_bc!(A1, Vₕ, :bottom)
            dirichlet_bc!(A2, Vₕ, :bottom; components = nothing)
            @test A1 == A2
            @test A1 != _tri(nV)           # the call changed something to compare
        end

        @testset "symmetrize! keyword" begin
            # leaves coupled by nothing, so leaf 1's elimination stays inside leaf 1
            A = Matrix(blockdiag(_tri(nW), _tri(nW), _tri(nW)))
            F = fill(2.0, nV)
            A0 = copy(A)
            symmetrize!(A, F, Vₕ, :bottom; components = 1)
            # only leaf 1's marked columns could have changed anything
            for c in 1:2, i in 1:nW

                row = c * nW + i
                @test A[:, row] == A0[:, row]
            end
            @test F[(nW + 1):nV] == fill(2.0, 2nW)
            # and leaf 1 did change: its block is what the scalar space gives
            As, Fs = Matrix(_tri(nW)), fill(2.0, nW)
            symmetrize!(As, Fs, Wₕ, :bottom)
            @test As != _tri(nW)
            @test A[1:nW, 1:nW] == As
            @test F[1:nW] == Fs
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
            @test_throws ErrorException dirichlet_bc!(
                _eye(nV), Vₕ, :bottom; components = :left
            )
        end

        @testset "Nested composite (#64)" begin
            # `dirichlet_components` is leaf-indexed (via `leaf_spaces_offsets`); `uₕ(i)`
            # used to be indexed by *immediate* child instead. On a flat space like `Vₕ`
            # above the two coincide, which is why nothing here caught it: `Vn` below
            # has 2 immediate children but 3 leaves (`×` alone would flatten it), and
            # before the fix `components = 3` validated fine while `u(3)` raised a
            # `BoundsError` on the same space.
            Vn = CompositeGridSpace((Wₕ × Wₕ, Wₕ))
            nVn = ndofs(Vn)
            @test nVn == 3nW

            A = _tri(nVn)
            @test dirichlet_bc!(A, Vn, :bottom; components = 3) === A
            for i in 1:nW
                row = 2nW + i          # leaf 3's offset, per leaf_spaces_offsets
                if marked[i]
                    @test A[row, row] == 1.0
                    @test count(!=(0.0), A[row, :]) == 1
                end
            end
            @test A[1:(2nW), :] == _tri(nVn)[1:(2nW), :]   # leaves 1, 2 untouched

            un = element(Vn, 0.0)
            @test length(parent(un(3))) == nW    # was a BoundsError before the fix
            @test_throws BoundsError un(4)
        end
    end

    @testset "Evaluation points" begin
        bcs = dirichlet_constraints(Ωₕ, :bottom => (x -> x[1] + 10x[2]))
        v = zeros(nW)
        @test dirichlet_bc!(v, Wₕ, bcs, :bottom) === v
        pts = [point(Ωₕ, idx) for idx in indices(Ωₕ)]
        for i in 1:nW
            marked[i] && @test v[i] ≈ pts[i][1] + 10pts[i][2]
        end
    end

    @testset "Multiple labels" begin
        A = _tri(nW)
        @test dirichlet_bc!(A, Wₕ, :bottom, :top) === A
        both = index_in_marker(Ωₕ, :bottom) .| index_in_marker(Ωₕ, :top)
        for i in 1:nW
            if both[i]
                @test A[i, i] == 1.0
                @test count(!=(0.0), A[i, :]) == 1
            else
                @test A[i, :] == _tri(nW)[i, :]
            end
        end
        @test count(both) > count(marked)     # :top really adds rows
    end

    @testset "Overlapping label precedence" begin
        # Two Dirichlet labels overlapping at a shared corner node (the (0,0) corner is
        # both :bottom and :left), each carrying a different value function.
        # `_apply_conditions!` (form/dirichlet_constraints.jl) walks `conditions(bcs)` in
        # the order the labels were given to `dirichlet_constraints` and unconditionally
        # overwrites, so the LAST-declared label wins on a shared node. Pinned in both
        # construction orders so a reordering of that walk is caught, not just its current
        # direction.
        Ωo = mesh(
            domain(
                interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom, :left => :left
            ),
            (5, 5),
            (true, true)
        )
        Wo = gridspace(Ωo)
        no = ndofs(Wo)
        bottom_mask = index_in_marker(Ωo, :bottom)
        left_mask = index_in_marker(Ωo, :left)
        shared = bottom_mask .& left_mask
        @test any(shared)   # the (0,0) corner is marked by both

        bcs_left_last = dirichlet_constraints(
            set(Ωo), :bottom => (x -> 1.0), :left => (x -> 2.0)
        )
        v = fill(-1.0, no)
        dirichlet_bc!(v, Wo, bcs_left_last, :bottom, :left)
        @test all(v[i] == 2.0 for i in 1:no if shared[i])
        @test all(v[i] == 1.0 for i in 1:no if bottom_mask[i] && !shared[i])
        @test all(v[i] == 2.0 for i in 1:no if left_mask[i] && !shared[i])

        # Same labels passed to `dirichlet_bc!` in the same order, but declared to
        # `dirichlet_constraints` in the opposite order: the winner flips, so precedence
        # tracks construction order, not the order given to `dirichlet_bc!`.
        bcs_bottom_last = dirichlet_constraints(
            set(Ωo), :left => (x -> 2.0), :bottom => (x -> 1.0)
        )
        w = fill(-1.0, no)
        dirichlet_bc!(w, Wo, bcs_bottom_last, :bottom, :left)
        @test all(w[i] == 1.0 for i in 1:no if shared[i])
    end

    @testset "Empty & missing labels" begin
        A0 = _tri(nW)
        A1 = copy(A0)
        @test dirichlet_bc!(A1, Wₕ) === A1                 # no labels at all
        @test A1 == A0
        v0 = fill(3.0, nW)
        bcs = dirichlet_constraints(Ωₕ, :bottom => (x -> 7.0))
        @test dirichlet_bc!(v0, Wₕ, bcs) === v0            # no labels
        @test all(==(3.0), v0)
    end

    @testset "Set/mesh/space construction" begin
        # `dirichlet_constraints` takes whichever of the three the caller has to hand, and
        # digs out the underlying `CartesianProduct` itself. For a composite space that
        # means its first leaf: the constraint is over the domain, and every leaf of a
        # composite space shares it.
        Ωₕ = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
            (6, 6),
            (true, true)
        )
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(3))
        g = x -> 7.0

        from_set = dirichlet_constraints(Ωₕ, :bottom => g)
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
        Ωₕ = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
            (5, 5),
            (true, true)
        )
        Wₕ = gridspace(Ωₕ)
        n = ndofs(Wₕ)
        inner = gridspace(Ωₕ, Val(2))
        nested = Bramble.CompositeGridSpace((Wₕ, inner, Wₕ))

        # a scalar space is its own only leaf, at offset zero
        @test Bramble.first_space(Wₕ) === Wₕ
        @test Bramble.leaf_spaces_offsets(Wₕ) == ((Wₕ, 0),)

        leaves = Bramble.leaf_spaces_offsets(nested)
        @test length(leaves) == 4
        @test map(last, leaves) == (0, n, 2n, 3n)
        @test ndofs(nested) == 4n

        # the flat four-component space gets the same values (the matrix rows are pinned
        # against it in test/form/symmetrize.jl)
        flat = gridspace(Ωₕ, Val(4))
        bcs = dirichlet_constraints(Ωₕ, :bottom => (x -> 7.0))
        vn, vf = fill(3.0, 4n), fill(3.0, 4n)
        dirichlet_bc!(vn, nested, bcs, :bottom)
        dirichlet_bc!(vf, flat, bcs, :bottom)
        @test vn == vf
    end

    @testset "Zero allocations" begin
        # This runs once per step of a time loop, so it has to cost nothing beyond the
        # work itself. Two things had to go for that: the leaves used to come back in a
        # `Vector{Tuple{Any, Int}}`, which made every read through a leaf dynamic and
        # boxed a Bool per degree of freedom (809 KB for one call on a 60x60 grid with
        # three components), and the composite matrix path used to build a BitVector over
        # the whole system to hold the marked rows.
        #
        # Both are gone: `leaf_spaces_offsets` answers with a tuple, and each leaf's mask
        # is read at an offset rather than copied. Measured inside a function, on concrete
        # locals, so nothing boxes at the call boundary and the reading is the real one.
        function counts(n)
            Ω = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
                (n, n),
                (true, true)
            )
            W, V = gridspace(Ω), gridspace(Ω, Val(3))
            bcs = dirichlet_constraints(Ω, :bottom => (x -> 7.0))

            Aw, Av = _eye(ndofs(W)), _eye(ndofs(V))
            vw, vv = zeros(ndofs(W)), zeros(ndofs(V))
            # warm up every path before measuring it
            dirichlet_bc!(Aw, W, :bottom)
            dirichlet_bc!(Av, V, :bottom)
            dirichlet_bc!(vw, W, bcs, :bottom)
            dirichlet_bc!(vv, V, bcs, :bottom)

            dirichlet_bc!(Av, V, :bottom; components = 1)
            dirichlet_bc!(vv, V, bcs, :bottom; components = 1)

            return (
                matrix_scalar = @allocated(dirichlet_bc!(Aw, W, :bottom)),
                matrix_composite = @allocated(dirichlet_bc!(Av, V, :bottom)),
                vector_scalar = @allocated(dirichlet_bc!(vw, W, bcs, :bottom)),
                vector_composite = @allocated(dirichlet_bc!(vv, V, bcs, :bottom)),
                # `components` restricts the same tuple walk, not a fresh Vector: this must
                # cost the same zero bytes as the unrestricted call above.
                matrix_one_component = @allocated(dirichlet_bc!(Av, V, :bottom; components = 1)),
                vector_one_component = @allocated(dirichlet_bc!(vv, V, bcs, :bottom; components = 1))
            )
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
        Ωt = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
            (8, 8),
            (true, true)
        )
        Vt = gridspace(Ωt, Val(3))
        @test @inferred(Bramble.leaf_spaces_offsets(Vt)) isa Tuple
        @test isconcretetype(typeof(Bramble.leaf_spaces_offsets(Vt)))
        @test alloc_test(Bramble.leaf_spaces_offsets, Vt) == 0
    end

    @testset "Marker reads allocate nothing (#99)" begin
        # `DomainMarkers.symbols`/`.tuples` moved from Set to Tuple, so every marker read
        # reached from a Dirichlet constraint -- including `_normalize_dirichlet`, which
        # every `dirichlet =` keyword on `form`/`assemble!` passes through -- is an
        # unrolled sweep with nothing to allocate: `_normalize_dirichlet` and iterating
        # `label_identifiers` both measure 0 B.
        function marker_read_bytes(bcs)
            Bramble.symbols(bcs)
            Bramble.tuples(bcs)
            Bramble.labels(bcs)
            Bramble._normalize_dirichlet(bcs)
            return (
                symbols = @allocated(Bramble.symbols(bcs)),
                tuples = @allocated(Bramble.tuples(bcs)),
                labels = @allocated(Bramble.labels(bcs)),
                normalize = @allocated(Bramble._normalize_dirichlet(bcs))
            )
        end

        Ωm = domain(
            interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom, :top => :top
        )
        bcs_single = dirichlet_constraints(Ωm, :bottom => (x -> 7.0))
        bcs_multi = dirichlet_constraints(Ωm, :bottom => (x -> 7.0), :top => (x -> x[1]))

        for bcs in (bcs_single, bcs_multi)
            c = marker_read_bytes(bcs)
            @test c.symbols == 0
            @test c.tuples == 0
            @test c.labels == 0
            @test c.normalize == 0
        end
    end

    WITH_SLOW_TESTS && @testset "Arbitrary fields (Supposition)" begin
        field_val = Data.Floats{Float64}(;
            minimum = -100.0, maximum = 100.0, nans = false, infs = false
        )

        @check function check_dirichlet_invariance_2d(
                nx = Data.Integers(4, 10),
                ny = Data.Integers(4, 10),
                v_raw = Data.Vectors(field_val; min_size = 100, max_size = 100)
        )
            Ω = domain(
                interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom, :top => :top
            )
            Ωₕ = mesh(Ω, (nx, ny), (false, false))
            Wₕ = gridspace(Ωₕ)
            n = ndofs(Wₕ)

            v = copy(v_raw[1:n])
            v_orig = copy(v)

            bcs = dirichlet_constraints(Ωₕ, :bottom => (x -> 2.5 * x[1] + 1.0))
            dirichlet_bc!(v, Wₕ, bcs, :bottom)

            marked = index_in_marker(Ωₕ, :bottom)
            pts = [point(Ωₕ, idx) for idx in indices(Ωₕ)]

            # 1. Marked boundary nodes match prescribed values
            ok_marked = all(
                isapprox(v[i], 2.5 * pts[i][1] + 1.0; atol = 1e-12) for i in 1:n if marked[i]
            )

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

end # module FormDirichletConstraintsTests
