import Bramble: VectorElement, spacing, points, half_points, space, values, ndofs, values!,
                half_spacings_iterator, half_points_iterator, indices, point
using LinearAlgebra: norm
using SparseArrays

@inline function _func2array!(u::AbstractArray, g, mesh_indices)
    @inbounds for idx in mesh_indices
        u[idx] = g(idx)
    end
    return u
end

function valid_interior_range(i::Int, dims::NTuple{D}) where {D}
    ntuple(k -> k == i ? (2:dims[k]) : (1:dims[k]), Val(D))
end

"""
Calculates the exact cell average of `x -> exp(-x)` over an interval.
"""
@inline function cell_avg_exp(v::NTuple{3, T}) where {T}
    h, x0, x1 = v
    return (exp(-x0) - exp(-x1)) / h
end

"""
Populates array `w` with cell-averaged values of the separable function
`f(x) = exp(-sum(x))` on the given `mesh`.
"""
function compute_exp_cell_averages!(w::Array{T, D}, mesh) where {T, D}
    # Create an iterator for each dimension that yields `(hᵢ, xᵢ, xᵢ₊₁)` for cell interfaces
    cell_data_iterators = ntuple(Val(D)) do i
        mesh_dim = mesh(i)
        zip(half_spacings_iterator(mesh_dim),
            half_points_iterator(mesh_dim),
            Iterators.drop(half_points_iterator(mesh_dim), 1))
    end

    # Iterate over the Cartesian product of the dimensional iterators
    @inbounds for (i, v_tuple) in enumerate(Iterators.product(cell_data_iterators...))
        w[i] = prod(cell_avg_exp, v_tuple)
    end
end

"""
Sets up the test grid, space, and a sample element for a given dimension `D`.
"""
function setup_test_grid(::Val{D}) where {D}
    # Using tuples indexed by D is a clean way to handle dimension-specific settings
    npts_options = ((4,), (4, 4), (4, 4, 4))
    unif_options = ((false,), (false, false), (false, false, false))

    dims = npts_options[D]
    unif = unif_options[D]

    intervals = ntuple(_ -> interval(-1.0, 4.0), Val(D))
    Ω = domain(reduce(×, intervals))

    Ωₕ = mesh(Ω, dims, unif)
    Wₕ = gridspace(Ωₕ)
    uₕ = element(Wₕ, 1)

    return dims, Wₕ, uₕ
end

@testset "Vector elements" begin
    # Setup a mock space
    W = gridspace(mesh(domain(box(0, 1)), 4, true))

    @testset "Constructors" begin
        u1 = element(W)
        @test u1 isa VectorElement
        @test space(u1) === W
        @test values(u1) isa Vector
        @test length(values(u1)) == ndofs(W)
        @test eltype(values(u1)) == Float64

        u2 = element(W, 5.0)
        @test u2 isa VectorElement
        @test space(u2) === W
        @test all(==(5.0), values(u2))
        @test length(u2) == 4

        v_init = collect(1.0:4.0)
        u3 = element(W, v_init)
        @test u3 isa VectorElement
        @test space(u3) === W
        @test values(u3) == v_init
        @test_throws DimensionMismatch element(W, collect(1.0:5.0))

        u4 = element(W, 3) # Test with Int
        @test u4 isa VectorElement
        @test space(u4) === W
        @test all(==(3.0), values(u4))
        @test eltype(u4) == Float64
    end

    @testset "Getters & setters" begin
        u = element(W, 1.0:4.0)
        @test space(u) === W
        @test values!(u, fill(2.0, 4)) === u
        @test values(u) == fill(2.0, 4)

        # Test copyto! alias
        @test copyto!(u, fill(3.0, 4)) === u
        @test values(u) == fill(3.0, 4)
    end

    @testset "Forwarded methods" begin
        u = element(W, 1.0:4.0)
        @test size(u) == (4,)
        @test length(u) == 4
        @test firstindex(u) == 1
        @test lastindex(u) == 4
        @test eltype(u) == Float64
        @test collect(u) == collect(1.0:4.0)
    end

    @testset "ndims" begin
        @test ndims(VectorElement) == 1
        u = element(W)
        @test ndims(u) == 1 # ndims often works on instances too
    end

    @testset "Indexing" begin
        u = element(W, 1.0:4.0)
        @test u[1] == 1.0
        @test u[4] == 4.0

        u[3] = 99.0
        @test u[3] == 99.0
        @test values(u)[3] == 99.0
    end

    @testset "similar" begin
        u = element(W, 1.0:4.0)
        s = similar(u)
        @test s isa VectorElement
        @test space(s) === space(u)
        @test length(s) == length(u)
        @test eltype(s) == eltype(u)
        # Values are uninitialized, so don't test their content directly
        s[1] = 1.0
        @test s[1] == 1.0
    end

    @testset "copyto!" begin
        u = element(W, 1.0:4.0)
        v = element(W, 11.0:14.0)
        z = element(W) # Uninitialized

        # VectorElement to VectorElement
        copyto!(z, u)
        @test values(z) == values(u)
        @test !(values(z) === values(u)) # Ensure it's a copy

        # AbstractVector to VectorElement
        vec_data = fill(5.5, 4)
        copyto!(z, vec_data)
        @test values(z) == vec_data
    end

    @testset "Broadcasting" begin
        u = element(W, 1.0:4.0)
        v = element(W, fill(2.0, 4))
        w = element(W)
        α = 3.0
        β = 2.0

        # Test similar for broadcast result
        bc = Base.broadcasted(+, u, v)
        s = similar(bc)
        @test s isa VectorElement
        @test space(s) === space(u)
        @test length(s) == length(u)

        # Test copyto! broadcast (u .= v)
        copyto!(u, Base.broadcasted(identity, v))
        @test values(u) == values(v)

        # Test materialize! / fused (w .= u .+ v .* α)
        w .= u .+ v .* α # Uses materialize! implicitly
        expected_w = values(u) .+ values(v) .* α
        @test values(w) ≈ expected_w

        # Test copyto! variant (w .= β .* v)
        w .= β .* v
        expected_w2 = β .* values(v)
        @test values(w) ≈ expected_w2

        # Test scalar assignment via broadcast
        w .= 5.0
        @test all(==(5.0), values(w))
    end

    @testset "Arithmetic" begin
        u_data = collect(1.0:4.0)
        v_data = fill(2.0, 4)
        u = element(W, u_data)
        v = element(W, v_data)
        α = 3.0
        β = 2.0

        # VectorElement + VectorElement
        r3 = u + v
        @test r3 isa VectorElement
        @test space(r3) === space(u)
        @test values(r3) ≈ u_data .+ v_data

        # Scalar * VectorElement
        r4 = α * u
        @test values(r4) ≈ α .* u_data

        # VectorElement * Scalar
        r5 = u * α
        @test values(r5) ≈ u_data .* α

        # VectorElement * VectorElement
        r6 = u .* v
        @test values(r6) ≈ u_data .* v_data

        # Subtraction
        r7 = u - v
        @test values(r7) ≈ u_data .- v_data
        r8 = u .- α
        @test values(r8) ≈ u_data .- α
        r9 = α .- u
        @test values(r9) ≈ α .- u_data

        # Power
        r13 = u .^ β
        @test values(r13) ≈ u_data .^ β

        r15 = u .^ v # Elementwise
        @test values(r15) ≈ u_data .^ v_data
    end
end

@testset "PDE operators" begin
    for D in 1:3
        @testset "$D-Dimensional Tests" begin
            dims, Wₕ, uₕ = setup_test_grid(Val(D))
            @test length(uₕ) == prod(dims)

            @testset "Rₕ!" begin
                test_function(x) = exp(-sum(x))
                Rₕ!(uₕ, test_function)

                # Reference calculation
                w = Array{Float64, D}(undef, dims)
                test_function_idx(idx) = test_function(point(mesh(Wₕ), idx))
                _func2array!(w, test_function_idx, indices(mesh(Wₕ)))

                w_flat = reshape(w, prod(dims))
                @test norm(values(uₕ) - w_flat) < 1e-15
            end

            @testset "avgₕ!" begin
                avgₕ!(uₕ, x -> exp(-sum(x)))

                w = Array{Float64, D}(undef, dims)
                compute_exp_cell_averages!(w, mesh(Wₕ))

                u_reshaped = reshape(values(uₕ), dims)
                interior = valid_interior_range(D, dims)
                @test @views norm(u_reshaped[interior...] - w[interior...]) < 1e-4
            end

            # Defer ∇₋ₕ tests until space/operators/difference.jl is enabled
        end
    end

    @testset "Component indexing" begin
        m = mesh(domain(box((0, 0), (1, 1))), (5, 6), (true, true))
        W = gridspace(m)
        V = W^2

        u_vec = element(V)
        @test length(u_vec) == 2 * ndofs(W)
        @test ncomponents(space(u_vec)) == 2

        # Component extraction via functor call u(i) and component(u, i)
        u1 = u_vec(1)
        u2 = u_vec(2)
        @test u1 isa VectorElement
        @test u2 isa VectorElement
        @test space(u1) === W
        @test space(u2) === W
        @test length(u1) == ndofs(W)
        @test length(u2) == ndofs(W)
        @test component(u_vec, 1) === u1 || values(component(u_vec, 1)) == values(u1)

        # Component ranges
        @test component_range(V, 1) == 1:ndofs(W)
        @test component_range(V, 2) == (ndofs(W) + 1):(2 * ndofs(W))
        @test component_ranges(V) == (1:ndofs(W), (ndofs(W) + 1):(2 * ndofs(W)))

        # components() tuple
        comps = components(u_vec)
        @test length(comps) == 2
        @test comps[1] isa VectorElement
        @test comps[2] isa VectorElement

        # Scalar space component indexing
        u_scal = element(W, 3.0)
        @test u_scal(1) === u_scal
        @test component(u_scal, 1) === u_scal
        @test components(u_scal) === (u_scal,)
        @test_throws BoundsError u_scal(2)
        @test_throws BoundsError u_vec(0)
        @test_throws BoundsError u_vec(3)

        # In-place mutation through component views
        u1 .= 10.0
        u2 .= 25.0
        @test all(==(10.0), values(u_vec)[1:ndofs(W)])
        @test all(==(25.0), values(u_vec)[(ndofs(W) + 1):(2 * ndofs(W))])

        # to_matrix on multi-component elements
        mats = to_matrix(u_vec)
        @test mats isa Tuple
        @test length(mats) == 2
        @test size(mats[1]) == (5, 6)
        @test size(mats[2]) == (5, 6)
        @test all(==(10.0), mats[1])
        @test all(==(25.0), mats[2])

        # Multi-component Rₕ
        Rₕ!(u_vec, (x -> x[1], x -> x[2]))
        @test mats[1][1, 1] ≈ m[1, 1][1]
        @test mats[2][1, 1] ≈ m[1, 1][2]

        # Multi-component avgₕ
        u_avg = avgₕ(V, (x -> 2.0, x -> 5.0))
        mats_avg = to_matrix(u_avg)
        @test mats_avg[1][2, 2] ≈ 2.0
        @test mats_avg[2][2, 2] ≈ 5.0
    end

    @testset "avgₕ quadrature" begin
        import Bramble: _gauss_rule, AVG_QUAD_POINTS, values

        Ωₕ = mesh(domain(interval(0.0, 1.0)), 40, false)   # non-uniform
        W = gridspace(Ωₕ)
        u = element(W)
        f(x) = exp(-sum(x))

        # exact cell averages of exp(-x) over [xᵢ₋₁ᐟ₂, xᵢ₊₁ᐟ₂]
        xh = half_points(Ωₕ)
        exact = [(exp(-xh[i]) - exp(-xh[i + 1])) / (xh[i + 1] - xh[i])
                 for i in 1:npoints(Ωₕ)]

        @testset "Convergence" begin
            errs = map(1:4) do nq
                avgₕ!(u, f; quad_points = nq)
                maximum(abs, values(u) .- exact)
            end
            # The mesh is randomly non-uniform, so assert the trend and generous
            # bounds rather than tight magic constants.
            @test errs[1] > errs[2] > errs[3]
            @test errs[1] < 1e-3             # 1 point is the midpoint rule
            @test errs[2] < 1e-7
            @test errs[4] < 1e-10

            # the shipped default must reach machine precision on this integrand
            avgₕ!(u, f)
            @test maximum(abs, values(u) .- exact) < 1e-11

            @test_throws ArgumentError avgₕ!(u, f; quad_points = 0)
        end

        @testset "Exact degree" begin
            # with N points the rule must integrate x^(2N-1) exactly
            for nq in 1:4
                deg = 2nq - 1
                g(x) = sum(x)^deg
                avgₕ!(u, g; quad_points = nq)
                ex = [(xh[i + 1]^(deg + 1) - xh[i]^(deg + 1)) /
                      ((deg + 1) * (xh[i + 1] - xh[i]))
                      for i in 1:npoints(Ωₕ)]
                @test maximum(abs, values(u) .- ex) < 1e-12
            end
        end

        @testset "Rule construction" begin
            for T in (Float64, Float32)
                nodes, wts = _gauss_rule(Val(3), T)
                @test nodes isa NTuple{3, T}
                @test wts isa NTuple{3, T}
                @test sum(wts) ≈ one(T)
                # folded to a constant at compile time, so obtaining it allocates nothing
                get_rule() = _gauss_rule(Val(3), T)
                get_rule()
                @test (@allocated get_rule()) == 0
            end

            # BigFloat keeps the run-time path: its precision is a run-time setting.
            nb, wb = _gauss_rule(Val(3), BigFloat)
            @test eltype(nb) === BigFloat
            @test abs(sum(wb) - one(BigFloat)) < 1e-50
        end

        @testset "Allocation scaling" begin
            # A direct call, one function-call frame between the test and avgₕ! itself,
            # matching how /tmp/verify_consolidation.jl checked this earlier -- and where a
            # Serial() backend really does measure exactly 0, at every grid size.
            function avg_bytes_direct(be, n)
                Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n); backend = be)
                W2 = gridspace(Ω2)
                u2 = element(W2)
                avgₕ!(u2, f)
                avgₕ!(u2, f)
                return @allocated avgₕ!(u2, f)
            end

            # A second, more indirect wrapper -- an extra `run!` closure between the test
            # and avgₕ!, one function-call frame deeper than avg_bytes_direct. Measured
            # behind a function barrier so that, at global scope, @allocated does not also
            # count the boxing of the non-const globals it touches.
            #
            # This extra frame alone (unrelated to which policy is chosen -- confirmed by
            # checking out the commit right before this one, still costs the same 48 B) is
            # enough to lose the zero-allocation guarantee `avg_bytes_direct` gets: real,
            # deterministic, and not itself point 22's doing, but worth recording here since
            # it is exactly the "inlining budget nothing can query" class of risk this file's
            # own comments already warn about. A real caller wrapping avgₕ! in one more
            # closure than this file's own internal helpers do could hit the same thing.
            function avg_bytes_wrapped(be, n)
                Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n); backend = be)
                W2 = gridspace(Ω2)
                u2 = element(W2)
                run!(uu, g) = avgₕ!(uu, g)
                run!(u2, f)
                run!(u2, f)
                return @allocated run!(u2, f)
            end

            # No threading threshold left to straddle (point 22): a `Serial()` backend runs
            # every grid size through the same plain loop, so the direct-call cost is
            # exactly 0 bytes regardless of how many points there are.
            be_serial = backend(policy = Serial())
            @test avg_bytes_direct(be_serial, 32) == 0        # 1_024 degrees of freedom
            @test avg_bytes_direct(be_serial, 1024) == 0      # 1_048_576 degrees of freedom
            @test avg_bytes_direct(be_serial, 8) == avg_bytes_direct(be_serial, 16) == 0

            # The wrapped path costs a small, size-independent constant instead of exactly
            # zero -- still O(1) in the grid, which is the property under test here.
            wrapped_small = avg_bytes_wrapped(be_serial, 32)
            wrapped_large = avg_bytes_wrapped(be_serial, 1024)
            @test wrapped_small == wrapped_large
            @test wrapped_large < 1000

            # `Parallel()` still costs the same regardless of grid size -- task spawn
            # overhead, not anything proportional to the number of points. This is the
            # test that caught point 51 (docs/form-unlock-plan.md): a five-capture
            # anonymous closure occasionally (1 to 3 in 20 independent compiles) took a
            # miscompiled path costing 176 B *per grid point*, 80 MiB on the large case
            # here. Fixed by replacing the closure with a named, concretely-typed kernel
            # struct (`_AvgKernel1`/`_AvgKernelD`) -- not a guarantee the class of bug can
            # never recur, so this stays a real regression guard, not just documentation.
            be_parallel = backend(policy = Parallel())
            small = avg_bytes_direct(be_parallel, 32)
            large = avg_bytes_direct(be_parallel, 1024)
            let per_point = (large - small) / (1_048_576 - 1_024)
                @info "avgₕ! Parallel() allocation diagnostic: " *
                      "small=$small large=$large per_point=$per_point " *
                      "nthreads=$(Threads.nthreads())"
            end
            @test large < 4 * small + 1     # +1 guards small == 0
            @test large < 100_000           # proportional (176 B/point) would be ~184 MB
        end
    end
end

@testset "Composite evaluation" begin
    import Bramble: component_range, component_ranges, components, values, ndofs, spaces

    W5 = gridspace(mesh(domain(interval(0.0, 1.0)), 5, true))
    W9 = gridspace(mesh(domain(interval(0.0, 1.0)), 9, true))

    @testset "Cumulative indexing" begin
        # Subspaces of the same *type* can hold different numbers of degrees of
        # freedom, so component ranges must be summed, never inferred from types.
        V = W5 × W9
        @test ndofs(V) == 14
        @test component_range(V, 1) == 1:5
        @test component_range(V, 2) == 6:14
        @test component_ranges(V) == (1:5, 6:14)

        u = element(V, 0.0)
        cs = components(u)
        @test length.(cs) == (5, 9)

        # Writing through one component must not touch the other.
        cs[1] .= 1.0
        @test all(==(1.0), values(u)[1:5])
        @test all(==(0.0), values(u)[6:14])

        @test_throws BoundsError component_range(V, 3)
    end

    @testset "Vector vs component functions" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 8))
        W = gridspace(Ωₕ)
        for NC in (2, 3, 4)
            V = W^Val(NC)
            fvec = x -> ntuple(k -> sin(k * x[1]) + cos(k * x[2]), Val(NC))
            ftup = ntuple(k -> (x -> sin(k * x[1]) + cos(k * x[2])), Val(NC))

            a = element(V)
            b = element(V)
            Rₕ!(a, fvec)
            Rₕ!(b, ftup)
            @test values(a) == values(b)

            c = element(V)
            d = element(V)
            avgₕ!(c, fvec)
            avgₕ!(d, ftup)
            @test values(c) == values(d)
        end
    end

    @testset "One-tuple functions" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 8, true)
        W = gridspace(Ωₕ)
        f = x -> 2.0

        u1 = element(W)
        u2 = element(W)
        Rₕ!(u1, f)
        Rₕ!(u2, (f,))
        @test values(u1) == values(u2)

        v1 = element(W)
        v2 = element(W)
        avgₕ!(v1, f)
        avgₕ!(v2, (f,))
        @test values(v1) == values(v2)
    end

    @testset "In-place return" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 6))
        W = gridspace(Ωₕ)
        V = W^Val(2)
        u = element(W)
        uv = element(V)
        f = x -> 1.0
        ft = (x -> 1.0, x -> 2.0)
        fv = x -> (1.0, 2.0)

        @test Rₕ!(u, f) === u
        @test Rₕ!(uv, ft) === uv
        @test Rₕ!(uv, fv) === uv
        @test avgₕ!(u, f) === u
        @test avgₕ!(uv, ft) === uv
        @test avgₕ!(uv, fv) === uv

        # the allocating forms still hand back the element
        @test Rₕ(W, f) isa VectorElement
        @test avgₕ(W, f) isa VectorElement
    end
end

@testset "Rₕ & avgₕ interface" begin
    import Bramble: values, index_in_marker

    Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
    Ωₕ = mesh(Ω, 6, true)
    W = gridspace(Ωₕ)

    @testset "Marker restriction" begin
        # index_in_marker returns a BitVector mask over the linear indices, not a
        # list of indices; iterating it would feed `true`/`false` to the kernel.
        @test index_in_marker(Ωₕ, :left) == Bool[1, 0, 0, 0, 0, 0]

        u = element(W)
        Rₕ!(u, x -> 1.0; markers = (:left,))
        @test values(u) == [1.0, 0, 0, 0, 0, 0]

        # several markers act as a union
        Rₕ!(u, x -> 1.0; markers = (:left, :right))
        @test values(u) == [1.0, 0, 0, 0, 0, 1.0]

        # avgₕ takes the same keyword
        v = element(W)
        avgₕ!(v, x -> 1.0; markers = (:right,))
        @test values(v)[1:5] == zeros(5)
        @test values(v)[6] ≈ 1.0

        w = avgₕ(W, x -> 1.0; markers = (:left,))
        @test w[1] ≈ 1.0
        @test values(w)[2:6] == zeros(5)
    end

    @testset "Argument types" begin
        seen = Ref{Any}(nothing)
        u1 = element(W)
        Rₕ!(u1, x -> (seen[] = x; 0.0))
        @test seen[] isa Float64          # documented: never an SVector

        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4))
        W2 = gridspace(Ω2)
        u2 = element(W2)
        seen2 = Ref{Any}(nothing)
        Rₕ!(u2, x -> (seen2[] = x; 0.0))
        @test seen2[] isa Tuple{Float64, Float64}

        seen3 = Ref{Any}(nothing)
        avgₕ!(u2, x -> (seen3[] = x; 0.0))
        @test seen3[] isa Tuple{Float64, Float64}
    end

    @testset "Keyword sets" begin
        kw(f) = Set(vcat([collect(Base.kwarg_decl(m)) for m in methods(f)]...))
        # markers is shared; quad_points belongs only to the quadrature-based operator
        @test :markers in kw(Rₕ) && :markers in kw(Rₕ!)
        @test :markers in kw(avgₕ) && :markers in kw(avgₕ!)
        @test :quad_points in kw(avgₕ) && :quad_points in kw(avgₕ!)
        @test !(:quad_points in kw(Rₕ)) && !(:quad_points in kw(Rₕ!))
    end
end

@testset "Threaded scatter" begin
    import Bramble: values, components

    # Dispatched on the backend's policy now (point 22), not gated by size -- a `Parallel()`
    # backend exercises the threaded scatter path at any grid size, deterministically,
    # rather than needing a grid large enough to cross a since-removed threshold.
    n = 8
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n); backend = backend(
        policy = Parallel()))
    W = gridspace(Ωₕ)

    for NC in (2, 3)
        V = W^Val(NC)
        fvec = x -> ntuple(k -> sin(k * x[1]) + cos(k * x[2]), Val(NC))
        ftup = ntuple(k -> (x -> sin(k * x[1]) + cos(k * x[2])), Val(NC))

        # threaded scatter must agree with the per-component route exactly
        a = element(V)
        b = element(V)
        Rₕ!(a, fvec)
        Rₕ!(b, ftup)
        @test values(a) == values(b)

        c = element(V)
        d = element(V)
        avgₕ!(c, fvec)
        avgₕ!(d, ftup)
        @test values(c) == values(d)

        # every component was written; nothing left at the uninitialised value
        @test all(isfinite, values(a))
        @test length(components(a)) == NC
    end
end

@testset "Quadrature fallback" begin
    import Bramble: _gauss_rule

    # An isbits float that QuadGK cannot build a rule for must fall back to the
    # run-time path at generation time rather than breaking precompilation.
    primitive type OddFloat <: AbstractFloat 64 end
    @test isbitstype(OddFloat)
    @test_throws Exception _gauss_rule(Val(3), OddFloat)

    # the supported types are unaffected
    for T in (Float32, Float64)
        nodes, wts = _gauss_rule(Val(3), T)
        @test sum(wts) ≈ one(T)
    end
end

@testset "Tuple arithmetic" begin
    import Bramble: values, space_type, _find_vec_in_broadcast,
                    _cell_average_kernel, _gauss_rule, VectorElement

    Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
    W = gridspace(Ωₕ)

    @testset "Tuple scaling" begin
        u = element(W, 3.0)
        v = (element(W, 2.0), element(W, 5.0))

        # uₕ * (v₁, v₂) multiplies componentwise
        z = u * v
        @test z isa NTuple{2, VectorElement}
        @test values(z[1]) == fill(6.0, 6)
        @test values(z[2]) == fill(15.0, 6)

        # a * (v₁, v₂) and the two reversed forms
        z2 = 2.0 * v
        @test values(z2[1]) == fill(4.0, 6)
        @test values(z2[2]) == fill(10.0, 6)
        @test values((v * 2.0)[1]) == values(z2[1])
        @test values((v * u)[2]) == values(z[2])

        # the originals are untouched
        @test values(v[1]) == fill(2.0, 6)
        @test values(u) == fill(3.0, 6)

        # the result takes the type of the product, not of the tuple. Both of these
        # allocated their output with `similar(vₕ[i])`, which copies the element's type and
        # drops the other operand's, so a wider scalar was truncated into a narrower vector
        # — and a Dual scalar threw outright, which is what found this. They delegate to
        # broadcasting now, where `similar(::Broadcasted, ElType)` promotes.
        Ω32 = mesh(domain(interval(0.0f0, 1.0f0)), 6, true)
        W32 = gridspace(Ω32)
        v32 = (element(W32, 2.0f0), element(W32, 5.0f0))
        @test eltype(values(v32[1])) === Float32
        @test eltype(values((2.0 * v32)[1])) === Float64        # Float64 scalar widens it
        @test eltype(values((2.0f0 * v32)[1])) === Float32       # Float32 leaves it alone
        @test values((2.0 * v32)[2]) ≈ fill(10.0, 6)

        # and the space comes from the tuple's elements, as it did before
        @test all(space((u * v)[i]) === space(v[i]) for i in 1:2)
        @test all(space((2.0 * v)[i]) === space(v[i]) for i in 1:2)
    end

    @testset "Inhomogeneous restriction" begin
        # `_scalar_value_type` read the element type of a tuple return with `eltype`, and
        # `eltype(Tuple{Float64, Int})` is `Real` — abstract, so the element was allocated as
        # a `Vector{Real}` of boxed pointers with no contiguity and no SIMD. An integer
        # literal among the components was enough: `x -> (1.0, 2)` measured `eltype = Real`.
        #
        # It promotes the field types now, which is what the arithmetic would have given.
        Ω = mesh(domain(interval(0.0, 1.0)), 6, true)
        V = gridspace(Ω)^Val(2)

        @test eltype(values(Rₕ(V, x -> (1.0, 2)))) === Float64
        @test eltype(values(Rₕ(V, x -> (1.0, 2.0)))) === Float64
        @test eltype(values(Rₕ(V, x -> (1, 2)))) === Float64      # promoted against the space
        @test values(Rₕ(V, x -> (1.0, 2))) == values(Rₕ(V, x -> (1.0, 2.0)))
    end

    @testset "space_type" begin
        u = element(W)
        @test space_type(typeof(u)) === typeof(W)
    end

    @testset "Empty broadcast" begin
        @test _find_vec_in_broadcast(()) === nothing
        @test _find_vec_in_broadcast((1, 2.0, :a)) === nothing
        u = element(W)
        @test _find_vec_in_broadcast((1, u, 2)) === u
    end

    @testset "Composite marker restriction" begin
        Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
        Ω2 = mesh(Ω, 6, true)
        W2 = gridspace(Ω2)
        V = W2^Val(2)

        uv = element(V)
        Rₕ!(uv, x -> (1.0, 2.0); markers = (:left,))
        c = components(uv)
        @test values(c[1]) == [1.0, 0, 0, 0, 0, 0]
        @test values(c[2]) == [2.0, 0, 0, 0, 0, 0]

        # a tuple of functions takes the per-component route with the same result
        wv = element(V)
        Rₕ!(wv, (x -> 1.0, x -> 2.0); markers = (:left,))
        @test values(wv) == values(uv)
    end

    @testset "nD marker averaging" begin
        Ω2 = domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom)
        Ωh = mesh(Ω2, (5, 5), (true, true))
        Wh = gridspace(Ωh)

        u = element(Wh)
        avgₕ!(u, x -> 1.0; markers = (:bottom,))
        marked = index_in_marker(Ωh, :bottom)
        @test all(values(u)[i] ≈ 1.0 for i in eachindex(values(u)) if marked[i])
        @test all(values(u)[i] == 0.0 for i in eachindex(values(u)) if !marked[i])
        @test any(marked)

        # the nD kernel form is what the masked path uses. It takes the quadrature order
        # and element type rather than a built rule, and builds it once.
        k = _cell_average_kernel(x -> 1.0, half_points(Ωh), Val(3), Float64, Val(2))
        @test k(first(indices(Ωh))) ≈ 1.0

        # It carries the rule, deliberately, so it is the same size as a closure written to
        # capture it by hand. It used to fetch the rule inside instead, which made it 96
        # bytes smaller — `nodes` and `wts` are isbits and stored inline, 48 bytes each,
        # and `Threads.@threads` copies the closure once per thread — by relying on
        # `_gauss_rule` being `@generated` and folding to a tuple literal.
        #
        # That fold is not guaranteed. On Julia 1.13 / x86_64 it did not happen, the rule
        # was rebuilt at every grid point, and `avgₕ!` measured 83,887,904 B on a 1024x1024
        # mesh against a 100,000 bound. 96 bytes per thread per call is the price of not
        # depending on an inlining budget nothing can query.
        nodes, wts = _gauss_rule(Val(3), Float64)
        carrying = let f = (x -> 1.0), x = half_points(Ωh), nodes = nodes, wts = wts
            idx -> Bramble._cell_average(f, x, idx, nodes, wts)
        end
        @test sizeof(k) == sizeof(carrying)
        @test k(first(indices(Ωh))) ≈ carrying(first(indices(Ωh)))
    end
end

@testset "Composite vector avgₕ" begin
    # A composite grid function can be averaged either from a tuple of functions, one per
    # component, or from a single function returning all components. The two must agree.
    #
    # In one dimension the second form raised a MethodError: a 1D mesh answers
    # `half_points` with a plain vector rather than a one-tuple of vectors, and the
    # composite kernel only had a method for the D-dimensional shape. The tuple form was
    # unaffected, because it dispatches to the scalar path once per component, which is
    # why nothing caught it.
    for (lbl, Ωₕ, fs, f_all) in (
        ("1D", mesh(domain(interval(0.0, 1.0)), 17, true),
        (sin, cos), x -> (sin(x), cos(x))),
        (
        "2D", mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 8), (true, false)),
        (x -> sin(x[1]), x -> cos(x[2])), x -> (sin(x[1]), cos(x[2]))))
        @testset "$lbl" begin
            Vₕ = gridspace(Ωₕ, Val(2))

            from_tuple = element(Vₕ)
            avgₕ!(from_tuple, fs)
            from_single = element(Vₕ)
            avgₕ!(from_single, f_all)
            @test values(from_single) ≈ values(from_tuple)

            # and the out-of-place form agrees with both
            @test values(avgₕ(Vₕ, f_all)) ≈ values(from_tuple)
            @test values(avgₕ(Vₕ, fs)) ≈ values(from_tuple)

            # the quadrature order still reaches the composite path
            @test values(avgₕ(Vₕ, f_all; quad_points = 3)) ≈ values(from_tuple) rtol=1e-8
        end
    end
end

@testset "Quadrature reuse" begin
    # `avgₕ!` fetches the Gauss rule inside its kernel where `_gauss_rule` folds to a
    # compile-time constant, and hoists it out of the loop where it does not. Getting that
    # predicate wrong is silent: the answers stay correct and the cost moves from once per
    # call to once per grid point.
    #
    # `isbitstype(T)` is not the predicate. `Double64` is isbits, `_gauss_rule` takes its
    # folding branch for it, and the branch does not fold — building the rule costs
    # thousands of bytes per call. Fetching inside then charged that per grid point.
    #
    # BigFloat is used here because it cannot fold either, needs no extra dependency, and
    # makes the two costs easy to separate: the rule is a large constant and the
    # arithmetic is a much smaller per-point cost.
    rule_cost = (_gauss_rule(Val(6), BigFloat); @allocated _gauss_rule(Val(6), BigFloat))
    @test rule_cost > 10_000     # it really is expensive, so the test is not vacuous

    function avg_bytes(n)
        be = backend(vector_type = Vector{BigFloat},
            matrix_type = SparseArrays.SparseMatrixCSC{BigFloat, Int})
        Ωₕ = mesh(domain(interval(BigFloat(0), BigFloat(1))), n, true; backend = be)
        uₕ = element(gridspace(Ωₕ))
        avgₕ!(uₕ, sin)
        return @allocated avgₕ!(uₕ, sin)
    end

    a64, a128 = avg_bytes(64), avg_bytes(128)
    marginal = (a128 - a64) / 64          # bytes per additional grid point

    # If the rule were rebuilt per point the marginal cost would be at least one rule
    # build; it is the BigFloat arithmetic instead, which is far cheaper.
    @test marginal < rule_cost / 10

    # and a folding type pays no per-point cost at all
    W64(n) = gridspace(mesh(domain(interval(0.0, 1.0)), n, true))
    b64(n) = (u = element(W64(n)); avgₕ!(u, sin); @allocated avgₕ!(u, sin))
    @test b64(64) == b64(1024)
end

@testset "Selective evaluation" begin
    # `Rₕ` and `avgₕ` learn the coefficient type by evaluating `f` once. That probe has to
    # land on a point the caller selected: `f` need not be defined anywhere else.
    #
    # Probing the first grid point regardless turned working calls into errors, and made
    # the out-of-place forms disagree with the in-place ones, which never probe.
    Ωₕ = mesh(domain(interval(0.0, 1.0), :right => :right), 11, true)
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(2))
    right_half(x) = sqrt(x - 0.5)          # a DomainError anywhere left of 0.5

    uₕ = Rₕ(Wₕ, right_half; markers = (:right,))
    ref = element(Wₕ)
    Rₕ!(ref, right_half; markers = (:right,))
    @test values(uₕ) == values(ref)

    @test values(avgₕ(Wₕ, right_half; markers = (:right,))) isa AbstractVector

    # composite, both shapes of f
    @test values(Rₕ(Vₕ, (right_half, right_half); markers = (:right,))) isa AbstractVector

    # and the unmarked path is unchanged
    @test values(Rₕ(Wₕ, sin)) ≈ [sin(x) for x in points(Ωₕ)]
end

@testset "Composite avgₕ! restriction" begin
    # The marked branch used to hand `to_matrix(uₕ)` to `_masked_for!` unconditionally.
    # For a composite grid function that is an NTuple of matrices rather than one array,
    # and the scalar kernel was built where the composite one is needed, so the call
    # raised a MethodError. Both shapes of `f` are covered, and they must agree.
    Ωₕ = mesh(domain(interval(0.0, 1.0), :right => :right), 11, true)
    Vₕ = gridspace(Ωₕ, Val(2))

    from_tuple = element(Vₕ)
    avgₕ!(from_tuple, (sin, cos); markers = (:right,))
    from_single = element(Vₕ)
    avgₕ!(from_single, x -> (sin(x), cos(x)); markers = (:right,))
    @test values(from_single) ≈ values(from_tuple)
    @test values(avgₕ(Vₕ, x -> (sin(x), cos(x)); markers = (:right,))) ≈ values(from_tuple)

    # the mask is respected: marked entries written, the rest left at zero
    mask = index_in_marker(Ωₕ, :right)
    for k in 1:2
        vals = values(components(from_tuple)[k])
        @test any(vals[i] != 0 for i in eachindex(vals) if mask[i])
        @test all(vals[i] == 0 for i in eachindex(vals) if !mask[i])
    end

    # 2D as well, where to_matrix really is multidimensional
    Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
        (5, 5), (true, true))
    V2 = gridspace(Ω2, Val(2))
    c2 = element(V2)
    avgₕ!(c2, x -> (sin(x[1]), cos(x[2])); markers = (:bottom,))
    t2 = element(V2)
    avgₕ!(t2, (x -> sin(x[1]), x -> cos(x[2])); markers = (:bottom,))
    @test values(c2) ≈ values(t2)
end
