module TestFormKroneckerEdge

using Test
using Bramble
using Bramble: is_separable, kronecker_operator, KroneckerLinearOperator
using Bramble: KroneckerBlockOperator, CpuSerial, CpuThreaded, CpuPolyester
using Bramble: D₊ₓ, D₋ₓ, D₋ᵧ, D₊ᵧ, Mₓ, Mᵧ, S₊ₓ, inner₊ₓ, restrict_to
using LinearAlgebra: Diagonal, diag, issymmetric, kron, mul!, norm
using SparseArrays: SparseMatrixCSC, sparse, spdiagm, spzeros, nnz, nonzeros, dropzeros!
using Random
using Polyester
using Kronecker
using ForwardDiff

# Regression tests for the edge cases the critics' probes found while the general Kronecker
# operators (gpena/Bramble.jl#427, #439, #442) were built. Each testset is one area; each
# check compares with `assemble` or an explicit `kron`, as the probe did, on small graded
# meshes whose axes carry different nodes.

const KE_SEED = 20261004

# A mesh on the host policy `P`: uniform flags, then (unless `graded` is false, or an axis has
# one point) moved by `change_points!` to `t^(1 + d/4)` along axis `d`, so no two axes share
# their nodes.
function _ke_mesh(n::NTuple{D, Int}, P = CpuSerial(); graded = true) where {D}
    Ωₕ = mesh(domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))), n,
        ntuple(_ -> true, D); backend = backend(; policy = P))
    graded && all(>(1), n) && Bramble.change_points!(Ωₕ,
        ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D))
    return Ωₕ
end
_ke_space(n, P = CpuSerial(); kw...) = gridspace(_ke_mesh(n, P; kw...))

# A hand-built operator from `(scales, factors)` specs, one term each, on policy `P`.
function _ke_op(T, dims, specs, P = CpuSerial())
    terms = map(sp -> Bramble._kron_term(sp[1], sp[2]), specs)
    return KroneckerLinearOperator{T, length(dims), typeof(terms), typeof(P)}(
        terms, dims, prod(dims), P)
end

_ke_coeff(s) = prod(c -> c isa Ref ? c[] : c, s; init = 1.0)
function _ke_oracle(specs)
    return sum(_ke_coeff(s) * kron(map(f -> sparse(Matrix(f)), reverse(f))...)
    for (s, f) in specs)
end

# Equal where both are finite (to rounding), NaN and Inf in the same places.
function _ke_same(A, B; rtol = 1e-13)
    size(A) == size(B) || return false
    fin = filter(isfinite, A)
    scale = max(maximum(abs, fin; init = 0.0), 1.0)
    return all(eachindex(A)) do i
        isequal(A[i], B[i]) ||
            (isfinite(A[i]) && isfinite(B[i]) && abs(A[i] - B[i]) <= rtol * scale)
    end
end

# The coefficient-weighted sum of every leaf's Kronecker products, or `nothing` when some
# leaf is refused.
function _ke_project_sum(a)
    Ωₕ = mesh(Bramble.trial_space(a))
    n = Bramble.ndofs(Bramble.trial_space(a))
    B = spzeros(eltype(assemble(a)), n, n)
    for (scales, term) in Bramble._kron_leaves(Bramble.resolve_form_ast(a), ())
        P = Bramble._kron_project(term, Ωₕ)
        P === nothing && return nothing
        for factors in P
            B += Bramble._kron_coeff(scales) * foldl(kron, reverse(map(sparse, factors)))
        end
    end
    return B
end

_ke_refused(a) = _ke_project_sum(a) === nothing
function _ke_projects(a; rtol = 1e-13)
    A, B = assemble(a), _ke_project_sum(a)
    return B !== nothing && _ke_same(Matrix(A), Matrix(B); rtol)
end

# Runs `f()` with the log silenced: the one-time warning for a grid-function coefficient, or
# for a domain that redefines a geometric marker, is not what the check is about.
_ke_quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# The operator built from the form, as a dense matrix, against `assemble(a)`.
function _ke_kron_matches(a; rtol = 1e-13)
    K = _ke_quiet(() -> kronecker_operator(a))
    return _ke_same(Matrix(assemble(a)), Matrix(SparseMatrixCSC(K)); rtol)
end

# A vector indexed from 0, standing in for an `OffsetVector` (not a test dependency).
struct _KEZeroBased <: AbstractVector{Float64}
    data::Vector{Float64}
end
Base.size(v::_KEZeroBased) = size(v.data)
Base.axes(v::_KEZeroBased) = (0:(length(v.data) - 1),)
Base.getindex(v::_KEZeroBased, i::Int) = v.data[i + 1]
Base.setindex!(v::_KEZeroBased, a, i::Int) = (v.data[i + 1] = a)

@testset "Kronecker edge cases" begin
    # `CpuThreaded` (#439) threads the lines of the product. Fewer lines than threads and
    # calls nested in tasks or in threaded loops all give the serial result bit for bit.
    @testset "edge: threaded line counts" begin
        f = (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v))
        for n in ((3, 2), (2, 2), (40, 2), (2, 40), (2, 2, 2), (3, 2, 2), (5, 4, 3), (2, 2, 40))
            Ws, Wt = _ke_space(n, CpuSerial()), _ke_space(n, CpuThreaded())
            a = form(Wt, Wt, f)
            Ks = kronecker_operator(form(Ws, Ws, f))
            Kt = kronecker_operator(a)
            N = size(Ks, 1)
            x = randn(MersenneTwister(KE_SEED), N)
            y0 = randn(MersenneTwister(KE_SEED + 1), N)
            ys = mul!(zeros(N), Ks, x)
            @test mul!(fill(NaN, N), Kt, x) == ys
            @test isapprox(ys, assemble(a) * x; rtol = 1e-12, atol = 1e-12)
            @test Kt * x == Ks * x
            for (α, β) in ((0.5, -3.0), (0.0, 1.0), (1, 0), (2, 1), (true, false), (0.0, 2.0))
                @test mul!(copy(y0), Kt, x, α, β) == mul!(copy(y0), Ks, x, α, β)
            end
            # `β = 0` overwrites whatever `y` held, NaN included, even for `α = 0`.
            @test mul!(fill(NaN, N), Kt, x, 0.5, 0.0) == mul!(zeros(N), Ks, x, 0.5, 0.0)
            @test all(iszero, mul!(fill(NaN, N), Kt, x, 0.0, 0.0))
            # A vector type the kernels were not written for.
            xd = ForwardDiff.Dual.(x, 1.0)
            @test mul!(similar(xd), Kt, xd) == mul!(similar(xd), Ks, xd)
            # Views in and out.
            xv, yv = view(vcat(x, x), 1:N), view(zeros(2N), 2:(N + 1))
            @test mul!(yv, Kt, xv) == ys
            @test_throws DimensionMismatch mul!(zeros(N + 1), Kt, x)
            @test_throws DimensionMismatch mul!(zeros(N), Kt, zeros(N - 1))
        end

        # A product started inside a task, or inside a user's threaded loop, runs its lines
        # serially there and still gives the serial result.
        for n in ((3, 2), (2, 2, 40))
            Ws, Wt = _ke_space(n, CpuSerial()), _ke_space(n, CpuThreaded())
            Ks, Kt = kronecker_operator(form(Ws, Ws, f)), kronecker_operator(form(Wt, Wt, f))
            N = size(Ks, 1)
            x = randn(MersenneTwister(KE_SEED), N)
            ys = mul!(zeros(N), Ks, x)
            tasks = [Threads.@spawn mul!(zeros(N), Kt, x) for _ in 1:16]
            @test all(t -> fetch(t) == ys, tasks)
            tasks = [Threads.@spawn begin
                         y = zeros(N)
                         for _ in 1:20
                             mul!(y, Kt, x)
                         end
                         y
                     end for _ in 1:8]
            @test all(t -> fetch(t) == ys, tasks)
            for sched in (:dynamic, :greedy)
                ys_nested = [zeros(N) for _ in 1:8]
                if sched === :dynamic
                    Threads.@threads :dynamic for k in 1:8
                        mul!(ys_nested[k], Kt, x)
                    end
                else
                    Threads.@threads :greedy for k in 1:8
                        mul!(ys_nested[k], Kt, x)
                    end
                end
                @test all(==(ys), ys_nested)
            end
        end
    end

    # (#427): a term may carry several non-diagonal, non-symmetric factors of any
    # element type. Oracle: an explicit `kron` of the same factors, last axis leftmost.
    @testset "edge: general factor element types" begin
        rng = MersenneTwister(KE_SEED)
        band(m, lo, hi; T = Float64) = spdiagm(
            (k => randn(rng, T, m - abs(k)) for k in (-lo):hi if abs(k) < m)...)
        dg(m) = Diagonal(rand(rng, m) .+ 0.5)
        function check(T, dims, specs; rtol = 1e-12)
            N = prod(dims)
            A = _ke_oracle(specs)
            Ks, Kt = _ke_op(T, dims, specs), _ke_op(T, dims, specs, CpuThreaded())
            x, y0 = randn(rng, T, N), randn(rng, T, N)
            y = mul!(fill(T(NaN), N), Ks, x)
            @test isapprox(y, A * x; rtol, atol = rtol)
            y5 = mul!(copy(y0), Ks, x, T(0.5), T(-3))
            @test isapprox(y5, T(0.5) * (A * x) - 3 * y0; rtol, atol = rtol)
            @test mul!(fill(T(NaN), N), Kt, x) == y
            @test mul!(copy(y0), Kt, x, T(0.5), T(-3)) == y5
            sym = all(sp -> all(f -> f isa Diagonal || issymmetric(f), sp[2]), specs)
            @test issymmetric(Ks) == sym
        end

        # Axes of one to three points in 2D and 3D, mixing diagonal, banded and non-symmetric
        # factors.
        for dims in ((2, 2), (2, 1, 3))
            D = length(dims)
            check(Float64,
                dims,
                (
                    ((), ntuple(d -> band(dims[d], 1, 1), D)),
                    ((2.0,), ntuple(d -> d == 1 ? Diagonal(rand(rng, dims[d])) :
                                         band(dims[d], 0, 1), D))))
            sym = ntuple(d -> (b = band(dims[d], 1, 0); b + b'), D)
            check(Float64, dims, (((), sym),))
        end
        # Four axes, dense factors and a range-backed diagonal.
        check(Float64, (3, 4, 2, 3), (((), (band(3, 1, 1), band(4, 1, 1), band(2, 1, 1),
            band(3, 1, 1))),))
        check(Float64, (4, 3, 2), (((), (randn(rng, 4, 4), randn(rng, 3, 3),
            randn(rng, 2, 2))),))
        check(Float64, (4, 3), (((), (Diagonal(1.0:4.0), band(3, 1, 1))),))
        # A symmetric pentadiagonal axis 1 is not tridiagonal, and an asymmetric tridiagonal
        # one is not symmetric: both take the row gather.
        P5 = band(6, 2, 0)
        check(Float64, (6, 4), (((), (P5 + P5', band(4, 1, 1))),))
        check(Float64, (6, 4), (((), (band(6, 1, 1), Diagonal(rand(rng, 4)))),))
        # Five terms in 3D, none alike.
        check(Float64,
            (5, 4, 6),
            (((Ref(0.3),), (band(5, 1, 2), band(4, 2, 1),
                    band(6, 1, 1))),
                ((), (dg(5), dg(4), dg(6))), ((), (dg(5), band(4, 1, 0), dg(6))),
                ((2.0, 3.0), (band(5, 0, 1), dg(4), band(6, 2, 2))),
                ((), (dg(5), band(4, 1, 1), band(6, 1, 0)))))

        check(Float32, (5, 4, 3),
            (((), (band(5, 1, 1; T = Float32),
                band(4, 1, 1; T = Float32), band(3, 1, 1; T = Float32))),); rtol = 1e-4)
        check(ComplexF64, (5, 4), (((), (band(5, 1, 1; T = ComplexF64),
            band(4, 1, 1; T = ComplexF64))),))
        check(BigFloat, (4, 3), ((
            (), (sparse(BigFloat.(Matrix(band(4, 1, 1)))),
                sparse(BigFloat.(Matrix(band(3, 1, 1)))))),))
        # A complex symmetric factor is symmetric; a Hermitian one is not.
        cs = band(5, 1, 0; T = ComplexF64)
        cs = cs + transpose(cs)
        ch = band(5, 1, 0; T = ComplexF64)
        ch = ch + ch'
        c4 = band(4, 1, 1; T = ComplexF64)
        check(ComplexF64, (5, 4), (((), (cs, c4)),))
        check(ComplexF64, (5, 4), (((), (ch, c4)), ((), (ch, Diagonal(rand(rng, 4))))))
        @test issymmetric(_ke_op(ComplexF64, (5, 4), (((), (cs, c4 + transpose(c4))),)))
        @test !issymmetric(_ke_op(ComplexF64, (5, 4), (((), (ch, c4 + transpose(c4))),)))

        # Dual numbers in the vector, in the factors, and through the threaded policy.
        D1 = ForwardDiff.Dual{Nothing, Float64, 1}
        dims, specs = (5, 4), (((), (band(5, 1, 1), band(4, 1, 2))),)
        A = _ke_oracle(specs)
        x = [ForwardDiff.Dual{Nothing}(randn(rng), randn(rng)) for _ in 1:20]
        for P in (CpuSerial(), CpuThreaded())
            y = mul!(zeros(D1, 20), _ke_op(D1, dims, specs, P), x)
            @test isapprox(ForwardDiff.value.(y), A * ForwardDiff.value.(x))
            @test isapprox(ForwardDiff.partials.(y, 1), A * ForwardDiff.partials.(x, 1))
        end
        dual(F) = sparse(map(v -> ForwardDiff.Dual{Nothing}(v, 1.0), Matrix(F)))
        specsD = (((), (dual(band(5, 1, 1)), band(4, 1, 1))),)
        AD = _ke_oracle(specsD)
        y = mul!(zeros(D1, 20), _ke_op(D1, dims, specsD), x)
        @test isapprox(ForwardDiff.value.(y), ForwardDiff.value.(AD * x))
        @test isapprox(ForwardDiff.partials.(y, 1), ForwardDiff.partials.(AD * x, 1))
    end

    # (#427): factors with rows or columns that hold nothing, entries stored as an
    # explicit zero, factors that are all zero, and empty axes.
    @testset "edge: zero rows and stored zeros" begin
        rng = MersenneTwister(KE_SEED + 3)
        band(m, lo, hi) = spdiagm((k => randn(rng, m - abs(k)) for k in (-lo):hi)...)
        function check(dims, specs)
            N = prod(dims)
            A = _ke_oracle(specs)
            x, y0 = randn(rng, N), randn(rng, N)
            for P in (CpuSerial(), CpuThreaded())
                K = _ke_op(Float64, dims, specs, P)
                @test isapprox(mul!(fill(NaN, N), K, x), A * x; rtol = 1e-12, atol = 1e-12)
                @test isapprox(mul!(copy(y0), K, x, 0.5, -3.0), 0.5 * (A * x) - 3.0 * y0;
                    rtol = 1e-12, atol = 1e-12)
            end
        end
        # A zero row, a zero column, in either axis position.
        Z = band(5, 1, 1)
        Z[3, :] .= 0
        dropzeros!(Z)
        Zc = band(4, 1, 1)
        Zc[:, 2] .= 0
        dropzeros!(Zc)
        check((5, 4), (((), (Z, Zc)),))
        check((4, 5), (((), (Zc, Z)),))
        # Entries stored as zero: they cost a multiply, and change nothing.
        S = band(5, 1, 1)
        S.nzval[2] = 0.0
        S.nzval[5] = 0.0
        @test count(iszero, nonzeros(S)) == 2
        check((5, 5, 3), (((), (S, S, band(3, 1, 1))),))
        St = spdiagm(0 => ones(5), 1 => ones(4), -1 => ones(4))
        St.nzval[2] = 0.0
        check((5, 4), (((), (St, band(4, 1, 1))),))
        # A factor with no entry at all, alone or beside a full one.
        check((4, 3), (((), (spzeros(4, 4), band(3, 1, 1))),
            ((), (band(4, 1, 1), spzeros(3, 3)))))
        @test iszero(mul!(fill(NaN, 12), _ke_op(Float64, (4, 3),
                (((), (spzeros(4, 4), band(3, 1, 1))),)), ones(12)))

        # A rectangular factor has no square product: refused where the term is built.
        @test_throws ArgumentError Bramble._kron_term((),
            (sparse(randn(rng, 3, 3)), sparse(randn(rng, 3, 4))))
        @test_throws ArgumentError Bramble._kron_term((),
            (sparse(randn(rng, 3, 4)), Diagonal(ones(2))))

        # Empty axes anywhere give an empty product under every policy, and write nothing
        # to the padding around an empty view of `y`.
        mkf(n) = n == 0 ? spzeros(0, 0) : sparse(randn(rng, n, n))
        mksym(n) = n == 0 ? spzeros(0, 0) :
                   (B = spdiagm(0 => randn(rng, n), -1 => randn(rng, max(n - 1, 0)));
            B + transpose(B) - Diagonal(diag(B)))
        for dims in ((0, 3), (3, 0), (2, 0, 3), (1, 0))
            D = length(dims)
            # Any: each spec set and each policy is its own type
            specsets = Any[
                ntuple(d -> mkf(dims[d]), D),
                ntuple(d -> d == 1 ? mksym(dims[d]) : Diagonal(rand(rng, dims[d])), D),
                ntuple(d -> d == 1 ? mkf(dims[d]) : Diagonal(rand(rng, dims[d])), D)]
            for fs in specsets, P in Any[CpuSerial(), CpuThreaded(), CpuPolyester()]

                K = _ke_op(Float64, dims, (((), fs),), P)
                yb = fill(7.0, 3)
                @test K * Float64[] == Float64[]
                @test isempty(mul!(view(yb, 2:1), K, Float64[], 2.0, 0.0))
                @test yb == fill(7.0, 3)
            end
        end
    end

    # (#427): `restrict_to(:interior, ...)` projects onto the axes only when the mesh's
    # `:interior` marker is the product of the axes' interiors. A marker the domain
    # redefines, `set_markers!` replaces or an in-place edit changes is refused.
    @testset "edge: custom interior markers" begin
        box(D) = reduce(×, ntuple(_ -> interval(0.0, 1.0), D))
        forms = (
            (u, v) -> innerₕ(restrict_to(:interior, u), v),
            (u, v) -> innerₕ(S₊ₓ(restrict_to(:interior, Mᵧ(u))), restrict_to(:interior, v)))
        function outcomes(Ωₕ)
            W = gridspace(Ωₕ)
            return map(forms) do f
                a = form(W, W, f)
                (refused = _ke_refused(a), projects = !_ke_refused(a) && _ke_projects(a),
                    separable = is_separable(a))
            end
        end
        _ke_quiet() do
            for (D, n) in ((2, (2, 3)), (2, (6, 5)), (3, (4, 2, 3)), (3, (3, 4, 5))),
                unif in (true, false)

                flags = ntuple(_ -> unif, D)
                # The geometric interior, and a custom marker that equals it, project.
                same = domain(box(D), :interior => x -> all(xi -> 1e-12 < xi < 1 - 1e-12, x))
                for Ωₕ in (mesh(domain(box(D)), n, flags), mesh(same, n, flags),
                    mesh(domain(box(D), :boundary => :left), n, flags))
                    @test all(o -> o.projects && o.separable, outcomes(Ωₕ))
                end
                # A marker that differs is refused, however it came about.
                diff = domain(box(D), :interior => x -> x[1] < 0.5)
                M2 = mesh(domain(box(D)), n, flags)
                Bramble.set_markers!(M2,
                    Bramble.markers(domain(box(D), :interior => x -> x[1] > 0.3));
                    warn_marker_mismatch = false)
                M3 = mesh(domain(box(D)), n, flags)
                Bramble.markers(M3)[:interior][1] = true
                for Ωₕ in (mesh(diff, n, flags), M2, M3)
                    @test all(o -> o.refused && !o.separable, outcomes(Ωₕ))
                end
            end
            # The refusal reaches the caller: no operator, though `assemble` still works.
            W = gridspace(mesh(domain(box(2), :interior => x -> x[1] < 0.5), (6, 5),
                (false, false)))
            a = form(W, W, forms[1])
            @test_throws ArgumentError kronecker_operator(a)
            @test nnz(assemble(a)) > 0
            Wd = gridspace(mesh(domain(box(2), :interior => :left), (6, 5), (false, true)))
            @test all(f -> _ke_refused(form(Wd, Wd, f)), forms)
        end
    end

    # (#427): `inner_Γ` is the sum over its named faces of one Kronecker term each.
    # Every face alias and combination matches `assemble` on axes of 1 to 3 points; the one
    # case with no product form, both faces of a one-point axis, is refused.
    @testset "edge: inner_Γ masks and aliases" begin
        # The face each alias names as (axis, side): in 2D `:left`/`:right` are the x faces and
        # `:bottom`/`:top` the y faces; in 3D `:back`/`:front` are x, `:left`/`:right` are y and
        # `:bottom`/`:top` are z.
        faces = Dict(:xmin => (1, -1), :xmax => (1, 1), :ymin => (2, -1), :ymax => (2, 1),
            :zmin => (3, -1), :zmax => (3, 1))
        aliases2 = Dict(:left => :xmin, :right => :xmax, :bottom => :ymin, :top => :ymax)
        aliases3 = Dict(:back => :xmin, :front => :xmax, :left => :ymin, :right => :ymax,
            :bottom => :zmin, :top => :zmax)
        function named(mk, D)
            mk isa Symbol || return reduce(union, named.(mk, D); init = Set{Tuple{Int, Int}}())
            mk === :boundary && return Set((d, s) for d in 1:D for s in (-1, 1))
            return Set([faces[get(D == 2 ? aliases2 : aliases3, mk, mk)]])
        end
        # Both faces of an axis that holds one point would weigh that point twice.
        both_faces(n, mk) = any(
            d -> n[d] == 1 && (d, -1) in named(mk, length(n)) &&
                 (d, 1) in named(mk, length(n)), eachindex(n))
        # Any: each mask, and each dimension's sizes and masks, is its own type
        masks2 = Any[(:left, :bottom), (:right, :top), (:xmin, :xmax), (:ymin, :ymax),
            (:xmin, :left), :boundary, (:boundary, :xmin)]
        masks3 = Any[(:left, :top, :back), (:right, :bottom, :front), (:back, :front),
            :boundary]
        sizes2 = ((9, 7), (1, 5), (5, 1), (2, 5), (3, 3), (2, 2), (3, 1), (1, 1))
        sizes3 = ((6, 5, 7), (1, 4, 3), (4, 1, 3), (3, 4, 1), (2, 3, 2), (1, 1, 4), (1, 1, 1))
        for (sizes, masks) in Any[(sizes2, masks2), (sizes3, masks3)], n in sizes

            for graded in (true, false)
                (graded || n == first(sizes)) || continue
                W = _ke_space(n; graded)
                for mk in masks
                    a = form(W, W, (u, v) -> inner_Γ(u, v; markers = mk))
                    if both_faces(n, mk)
                        @test _ke_refused(a)
                        @test !is_separable(a)
                    else
                        @test _ke_projects(a)
                        @test is_separable(a)
                    end
                end
            end
        end

        # Chains, sums, restrictions, coefficients and several `inner_Γ` in one form, with
        # the masks each `inner_Γ` names.
        # Any: each size, form and mask list is its own type
        for n in Any[(9, 7), (1, 5), (5, 1), (6, 5, 7), (1, 4, 3)]
            W = _ke_space(n)
            fx = Rₕ(W, x -> 1 + x[1])
            forms = Any[
                ((u, v) -> inner_Γ(D₋ₓ(D₋ᵧ(u)), D₊ₓ(Mₓ(v)); markers = :boundary),
                    (:boundary,)),
                ((u, v) -> inner_Γ(fx * u, restrict_to(:interior, D₋ₓ(v)); markers = :ymin),
                    (:ymin,)),
                (
                    (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                              inner_Γ(u, v; markers = (:xmax, :xmin)) +
                              inner_Γ(D₋ₓ(u), v; markers = :xmax),
                    ((:xmax, :xmin), :xmax))]
            for (f, masks) in forms
                a = form(W, W, f)
                if any(mk -> both_faces(n, mk), masks)
                    @test _ke_refused(a)
                else
                    @test _ke_projects(a)
                end
            end
        end
    end

    # (#427): an axis of one point has no neighbour, so a difference there is NaN
    # in `assemble`. The Kronecker sum puts the NaN in the same places, and a coefficient, a
    # face or a scale on such an axis changes nothing about that.
    @testset "edge: one-point axes" begin
        _ke_quiet() do
            # Any: each size and form is its own type
            for n in Any[(1, 4), (4, 1), (2, 1, 3)]
                W = _ke_space(n)
                fx = Rₕ(W, x -> 1 + x[1])
                fy = Rₕ(W, x -> 2 + x[2]^2)
                forms = Any[
                    (u, v) -> innerₕ(D₋ₓ(Mₓ(u)), D₋ₓ(Mₓ(v))),
                    (u, v) -> innerₕ(restrict_to(:interior, u), v),
                    (u, v) -> inner_Γ(D₋ₓ(u), v; markers = :xmin),
                    # A coefficient along the one-point axis or the other.
                    (u, v) -> innerₕ(fx * D₋ₓ(u), D₋ₓ(v)),
                    (u, v) -> innerₕ(D₋ᵧ(fy * u), v)]
                for f in forms
                    a = form(W, W, f)
                    @test is_separable(a)
                    @test _ke_kron_matches(a)
                end
                # Both faces of the one-point axis have no product form.
                if n[1] == 1
                    a = form(W, W, (u, v) -> inner_Γ(u, v; markers = (:boundary,)) +
                                             innerₕ(u, v))
                    @test !is_separable(a)
                    @test _ke_refused(a)
                end
            end
        end
    end

    # (#427): a grid-function coefficient factors when its values vary along one axis
    # only, tested bit for bit, and its values are copied when the operator is built.
    @testset "edge: coefficient exactness" begin
        W = _ke_space((9, 7))
        Ωₕ = mesh(W)
        fy = Rₕ(W, x -> 2 + x[2]^2)
        coefficient_form(g) = form(W, W, (u, v) -> innerₕ(g * D₋ₓ(u), v))
        leaf(a) = only(Bramble._kron_leaves(Bramble.resolve_form_ast(a), ()))[2]

        # A coefficient stored in another element type factors and matches `assemble`.
        @test _ke_projects(coefficient_form(Float32.(Vector(fy))); rtol = 1e-6)

        # Constancy along the other axes is bitwise: a single -0.0 among 0.0 breaks it, a
        # signed zero that every point of a slice shares does not, and neither does NaN.
        g = zeros(length(fy))
        g[3] = -0.0
        @test _ke_refused(coefficient_form(g))
        g = zeros(length(fy))
        g[1:9] .= -0.0
        @test _ke_projects(coefficient_form(g))
        g = Vector(fy)
        g[10:18] .= NaN
        @test _ke_projects(coefficient_form(g))
        g[11] = 1.0
        @test _ke_refused(coefficient_form(g))

        # Dual numbers: values and derivatives along one axis factor; derivatives that vary
        # along a second axis, under values that do not, are refused.
        p = ForwardDiff.Dual{:t}(0.5, 1.0)
        pts = Iterators.product(Bramble.points(Ωₕ)...)
        gy = [1.0 + x[2] + p * x[2]^2 for x in pts][:]
        ay = coefficient_form(gy)
        A, B = Matrix(assemble(ay)), Matrix(_ke_project_sum(ay))
        @test ForwardDiff.value.(A) ≈ ForwardDiff.value.(B)
        @test first.(ForwardDiff.partials.(A)) ≈ first.(ForwardDiff.partials.(B))
        @test any(!iszero, first.(ForwardDiff.partials.(A)))
        pz = ForwardDiff.Dual{:t}(0.0, 1.0)
        gxy = [1.0 + x[2] + pz * x[1] * x[2] for x in pts][:]
        @test _ke_refused(form(W, W, (u, v) -> innerₕ(gxy * u, v)))

        # The values are a snapshot: a later change reaches `assemble`, not the factors.
        g = Vector(fy)
        a = coefficient_form(g)
        P0 = deepcopy(Bramble._kron_project(leaf(a), Ωₕ))
        K = _ke_quiet(() -> kronecker_operator(a))
        x = rand(MersenneTwister(KE_SEED), size(K, 2))
        y0 = K * x
        g .*= 3
        @test Bramble._kron_project(leaf(a), Ωₕ) != P0
        @test K * x == y0
        @test !isapprox(K * x, assemble(a) * x)
        fe = Rₕ(W, x -> 2 + x[2]^2)
        a = coefficient_form(fe)
        P0 = deepcopy(Bramble._kron_project(leaf(a), Ωₕ))
        K = _ke_quiet(() -> kronecker_operator(a))
        y0 = K * x
        parent(fe) .*= 3
        @test K * x == y0
        @test !isapprox(K * x, assemble(a) * x)
        @test P0 != Bramble._kron_project(leaf(a), Ωₕ)
    end

    # (#427, #442): `kronecker_operator` reads a `Ref` at every product, accepts a
    # number scale inside a side, refuses a node outside the mesh's axes, and builds the
    # same factors under every host policy.
    @testset "edge: cache and Ref liveness" begin
        # A Ref on the whole term or on a whole side follows later changes; one inside a
        # sum or a chain cannot be pulled out of the factors and is refused.
        W = _ke_space((7, 6))
        c = Ref(2.0)
        x = rand(MersenneTwister(KE_SEED), ndofs(W))
        live = (
            (u, v) -> c * innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)),
            (u, v) -> innerₕ(c * D₋ᵧ(u), c * D₋ₓ(v)) + innerₕ(u, v))
        for f in live
            a = form(W, W, f)
            @test is_separable(a)
            K = kronecker_operator(a)
            for value in (-0.7, 5.0, 0.0)
                c[] = value
                @test isapprox(K * x, assemble(a) * x; rtol = 1e-12, atol = 1e-13)
            end
            c[] = 2.0
        end
        for f in ((u, v) -> innerₕ(D₋ₓ(c * u), v) + innerₕ(u, v),
            (u, v) -> innerₕ(D₋ₓ(u) + c * u, v), (u, v) -> innerₕ(u + c * D₋ₓ(u), v))
            a = form(W, W, f)
            @test !is_separable(a)
            @test_throws ArgumentError kronecker_operator(a)
        end

        # A number scale inside a side, merged from like terms or written there: Int, Complex,
        # BigFloat, nested, zero, beside a coefficient, a face or a restriction. On a mesh
        # with a one-point axis the NaNs of the differences agree too.
        # Any: each size and form is its own type
        for n in Any[(7, 6), (1, 5), (4, 3, 5)]
            Wn = _ke_space(n)
            forms = Any[
                (u, v) -> innerₕ(2.0 * D₋ₓ(3.0 * u), v),
                (u, v) -> innerₕ(2.0 * D₋ᵧ(u) + u, 3.0 * v + D₋ₓ(v)),
                (u, v) -> innerₕ(D₋ₓ(u) + 2 * u, v),
                (u, v) -> innerₕ(D₋ₓ(u) + (1.0 + 2.0im) * u, v),
                (u, v) -> innerₕ(D₋ₓ(u) + big"1.25" * u, v),
                (u, v) -> Ref(2.0) * innerₕ(restrict_to(:interior, D₋ₓ(u) + 1.3 * u), v)]
            for f in (length(n) == 2 ? forms : forms[1:1])
                a = form(Wn, Wn, f)
                @test is_separable(a)
                K = _ke_quiet(() -> kronecker_operator(a))
                @test _ke_same(Matrix(assemble(a)), Matrix(SparseMatrixCSC(K)))
                xn = rand(MersenneTwister(KE_SEED), ndofs(Wn))
                @test isapprox(K * xn, assemble(a) * xn; rtol = 1e-12, nans = true)
            end
        end

        # A node whose axis the mesh does not have is not the identity: refused, where
        # `assemble` fails with a bounds error.
        Wp = _ke_space((5, 4))
        for Dim in (0, 3)
            a = form(Wp, Wp,
                (u, v) -> innerₕ(Bramble.BackwardDifference{2, Dim, typeof(u)}(u), v))
            @test !is_separable(a)
            @test_throws ArgumentError kronecker_operator(a)
            @test_throws BoundsError assemble(a)
        end
        for Dim in (1, 2)
            a = form(Wp, Wp,
                (u, v) -> innerₕ(Bramble.BackwardDifference{2, Dim, typeof(u)}(u), v))
            @test is_separable(a) && _ke_kron_matches(a)
        end

        # The 1D assemblies are shared between terms that use the same factor.
        K = kronecker_operator(form(W, W, (u, v) -> Ref(1.5) * innerₕ(u, v) +
                                                    inner₊(∇ₕ(u), ∇ₕ(v))))
        @test length(K.terms) == 3
        @test K.terms[1].factors[2] === K.terms[2].factors[2]
        @test K.terms[1].factors[1] === K.terms[3].factors[1]

        # The factors do not depend on the mesh's policy: same matrices, same product.
        f = (u, v) -> innerₕ(D₋ₓ(Mₓ(u)), D₋ₓ(Mₓ(v))) + 0.3 * innerₕ(u, v)
        Ks = map((CpuSerial(), CpuThreaded(), CpuPolyester())) do P
            Wp = _ke_space((7, 6), P)
            kronecker_operator(form(Wp, Wp, f))
        end
        xs = rand(MersenneTwister(KE_SEED), size(Ks[1], 2))
        for K in Ks[2:3]
            @test SparseMatrixCSC(K) == SparseMatrixCSC(Ks[1])
            for (t, ts) in zip(K.terms, Ks[1].terms), d in 1:2

                @test t.factors[d] == ts.factors[d]
            end
            @test K * xs == Ks[1] * xs
        end
    end

    # (#427): a composite form on one mesh is a grid of Kronecker blocks. Rows no block
    # reaches, blocks off the diagonal only, one leaf, scales on blocks, and the vector
    # conventions of `mul!`.
    @testset "edge: block operators" begin
        Ωₕ = _ke_mesh((9, 7))
        W, V, V3 = gridspace(Ωₕ), gridspace(Ωₕ, Val(2)), gridspace(Ωₕ, Val(3))
        c = Ref(2.0)
        # `K` against `assemble(a)`: entries, product, 5-argument product, and a `β = 0`
        # product over NaN, which must overwrite it, idle rows included.
        function check(a)
            @test is_separable(a)
            K = _ke_quiet(() -> kronecker_operator(a))
            A = assemble(a)
            @test K isa KroneckerBlockOperator
            @test size(K) == size(A)
            @test _ke_same(Matrix(A), Matrix(SparseMatrixCSC(K)); rtol = 1e-12)
            x = rand(MersenneTwister(KE_SEED), size(A, 2))
            y0 = rand(MersenneTwister(KE_SEED + 1), size(A, 1))
            @test isapprox(K * x, A * x; rtol = 1e-12, atol = 1e-12)
            @test isapprox(mul!(copy(y0), K, x, 0.5, -3.0), 0.5 * (A * x) - 3.0 * y0;
                rtol = 1e-12, atol = 1e-12)
            y = mul!(fill(NaN, size(A, 1)), K, x, 1.0, 0.0)
            @test !any(isnan, y)
            @test isapprox(y, A * x; rtol = 1e-12, atol = 1e-12)
            @test all(i -> isapprox(K[i, i], A[i, i]; atol = 1e-12), 1:5:minimum(size(A)))
            # Symmetric is decided block by block: a symmetric `K` is never a false claim,
            # though a zero or cancelled block with no mirror makes it conservative.
            @test !issymmetric(K) || issymmetric(A)
            return K
        end
        K = check(form(V, V, (u, v) -> innerₕ(u(1), v(1)) +
                                       inner₊ₓ(D₋ₓ(u(2)), D₋ₓ(v(2))) + innerₕ(u(2), v(2)) +
                                       innerₕ(D₋ₓ(u(1)), v(2))))
        @test length(K.blocks) == 3
        # Off-diagonal blocks only: no diagonal block exists.
        K = check(form(V, V, (u, v) -> innerₕ(u(2), v(1)) + innerₕ(D₋ₓ(u(1)), v(2))))
        @test length(K.blocks) == 2
        # Two of three test leaves idle: their rows are zero, however `y` started.
        K = check(form(V3, V3, (u, v) -> innerₕ(D₋ᵧ(u(3)), v(1)) + innerₕ(u(1), v(1))))
        n = ndofs(W)
        y = mul!(fill(NaN, 3n), K, ones(3n), 1.0, 0.0)
        @test all(iszero, y[(n + 1):end]) && !any(isnan, y)
        # A cancelled term leaves a block that holds nothing.
        check(form(V, V, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(1)) -
                                   innerₕ(u(2), v(1))))
        # A `Ref` on a block stays live; a face, and a scalar trial space beside a composite
        # test space, are blocks too.
        a = form(V, V, (u, v) -> c * innerₕ(u(1), v(1)) + innerₕ(c * u(2), v(2)))
        K = check(a)
        c[] = -4.0
        x = rand(MersenneTwister(KE_SEED), size(K, 2))
        @test isapprox(K * x, assemble(a) * x; rtol = 1e-12)
        c[] = 2.0
        check(form(W, V, (u, v) -> innerₕ(u, v(1)) + innerₕ(D₋ₓ(u), v(2))))

        # One leaf is the scalar operator, entry for entry.
        V1 = Bramble.CompositeGridSpace((W,))
        K1 = check(form(V1, V1, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(1)), v(1))))
        Ks = kronecker_operator(form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(D₋ₓ(u), v)))
        @test SparseMatrixCSC(K1) == SparseMatrixCSC(Ks)
        x = rand(MersenneTwister(KE_SEED), size(Ks, 2))
        @test K1 * x == Ks * x

        # A vector indexed from 0 is refused, not read a place off.
        zero_based = _KEZeroBased(x)
        @test_throws ArgumentError mul!(zeros(length(x)), K1, zero_based)
        @test_throws ArgumentError mul!(zeros(length(x)), K1, zero_based, 1.0, 0.0)
        @test_throws ArgumentError mul!(_KEZeroBased(zeros(length(x))), K1, x, 1.0, 0.0)
        @test_throws ArgumentError mul!(zeros(length(x)), Ks, zero_based, 1.0, 0.0)
    end

    # (#427): `fdm_solve` refuses a singular system instead of returning garbage,
    # solves the empty interior, a Float32 Laplacian at sizes a threshold growing with the
    # unknown count refused, negative mass, and reads a `Ref` when it is called.
    @testset "edge: fdm_solve scope" begin
        grad(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))
        refusal(f) =
            try
                f()
                "no error"
            catch e
                e isa ArgumentError ? sprint(showerror, e) : "wrong error $(typeof(e))"
            end
        W = _ke_space((9, 7))
        F = rand(MersenneTwister(KE_SEED), ndofs(W))

        # Pure Neumann: constants are in the kernel. Refused from the form and from the
        # operator, whatever the route to the zero eigenvalue; fine with a Dirichlet face.
        W3 = _ke_space((6, 5, 7))
        # Any: each space and form is its own type
        singular = Any[(W, grad), (W3, grad),
            (W, (u, v) -> 1e-14 * innerₕ(u, v) + grad(u, v))]
        for (Ws, f) in singular
            a = form(Ws, Ws, f)
            Fs = rand(MersenneTwister(KE_SEED), ndofs(Ws))
            K = kronecker_operator(a)
            for m in (refusal(() -> fdm_solve(a, Fs)), refusal(() -> fdm_solve(K, Fs)))
                @test occursin("the system is singular", m)
            end
        end
        a = form(W, W, grad)
        A = assemble(a; dirichlet = :boundary)
        Fb = copy(F)
        Fb[Bramble._combined_mask(mesh(W), (:boundary,))] .= 0
        @test isapprox(fdm_solve(a, Fb; dirichlet = :boundary), A \ Fb; rtol = 1e-9)

        # A 2-point axis leaves no unknown under `:boundary`: the answer is zeros.
        # Any: each size is its own type
        for n in Any[(2, 2), (2, 4), (2, 3, 3), (3, 2, 4)]
            Wn = _ke_space(n)
            a = form(Wn, Wn, (u, v) -> innerₕ(u, v) + grad(u, v) +
                                       inner_Γ(u, v; markers = (:xmin,)))
            Fn = rand(MersenneTwister(KE_SEED), ndofs(Wn))
            @test fdm_solve(a, Fn; dirichlet = :boundary) == zeros(ndofs(Wn))
        end

        # Float32 Laplacians: eigenvalue ratios in the thousands, thousands of unknowns.
        box(D, T) = reduce(×, ntuple(_ -> interval(zero(T), one(T)), D))
        for (n, dir) in (((129, 129), :boundary), ((65, 65), nothing))
            Wf = gridspace(mesh(domain(box(length(n), Float32)), n,
                ntuple(_ -> true, length(n)); backend = backend(Float32)))
            a = form(Wf, Wf, (u, v) -> innerₕ(u, v) + grad(u, v))
            A = assemble(a; dirichlet = dir)
            Ff = rand(MersenneTwister(KE_SEED), Float32, size(A, 1))
            dir === :boundary && (Ff[Bramble._combined_mask(mesh(Wf), (:boundary,))] .= 0)
            x = fdm_solve(a, Ff; dirichlet = dir)
            @test eltype(x) === Float32
            # Against the Float64 direct solve of the same matrix: Float32 keeps about three
            # digits at these condition numbers, a wrong solve none.
            x64 = Float64.(A) \ Float64.(Ff)
            @test norm(Float64.(x) - x64) / norm(x64) < 1e-2
        end

        # Negative mass: indefinite but not singular, so it solves.
        for s in (-1.0, -5.0, -20.0)
            a = form(W, W, (u, v) -> s * innerₕ(u, v) + grad(u, v))
            @test isapprox(fdm_solve(a, F), assemble(a) \ F; rtol = 1e-9)
        end

        # A `Ref` is read when `fdm_solve` is called, not when the form or `K` was built:
        # the mass term that was zero then, and a coefficient that is zero now, included.
        c = Ref(2.5)
        a = form(W, W, (u, v) -> innerₕ(u, v) + c * grad(u, v))
        K = kronecker_operator(a)
        c[] = 0.7
        @test isapprox(fdm_solve(K, F), assemble(a) \ F; rtol = 1e-9)
        c[] = 3.1
        @test isapprox(fdm_solve(a, F), assemble(a) \ F; rtol = 1e-9)
        m = Ref(0.0)
        a = form(W, W, (u, v) -> m * innerₕ(u, v) + grad(u, v))
        K = kronecker_operator(a)
        m[] = 1.0
        @test isapprox(fdm_solve(K, F), assemble(a) \ F; rtol = 1e-9)
        d = Ref(1.0)
        a = form(W, W, (u, v) -> innerₕ(u, v) + d * grad(u, v))
        K = kronecker_operator(a)
        d[] = 0.0
        # No term of its own left along an axis: refused with the reason, never wrong.
        @test occursin("has no term of its own", refusal(() -> fdm_solve(K, F)))
    end

    # (#442): an operator built before the mesh changed refuses to run. Every kind of
    # change, from every entry point, under every policy; the stale error comes before the
    # size errors; and a form on stale spaces is refused.
    @testset "edge: stale operators" begin
        stale(f, needle = "KroneckerLinearOperator's factors") =
            try
                f()
                false
            catch e
                e isa ArgumentError && occursin("change_points!", sprint(showerror, e)) &&
                    occursin(needle, sprint(showerror, e))
            end
        lap(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        squares = range(0, 1; length = 7) .^ 2
        # Any: each mutation and each policy is its own type
        mutations = Any[
            :set_points_on_one_axis => Ω -> Bramble.set_points!(Ω(1), collect(squares)),
            :change_points_on_one_axis => Ω -> Bramble.change_points!(Ω(2),
                collect(range(0, 1; length = 6) .^ 2)),
            :set_the_same_points => Ω -> Bramble.set_points!(Ω(1), copy(Bramble.points(Ω(1)))),
            :refine_the_mesh => Bramble.iterative_refinement!,
            :refine_one_axis => Ω -> Bramble.iterative_refinement!(Ω(1)),
            :mutate_and_back => Ω -> (p0 = copy(Bramble.points(Ω(1)));
                Bramble.set_points!(Ω(1), p0 .^ 2); Bramble.set_points!(Ω(1), p0))]
        for (kind, mutate!) in mutations, P in Any[CpuSerial(), CpuPolyester()]

            Ωₕ = _ke_mesh((7, 6), P; graded = false)
            W, V = gridspace(Ωₕ), gridspace(Ωₕ, Val(2))
            K = kronecker_operator(lap(W))
            KB = kronecker_operator(form(V, V, (u, v) -> innerₕ(u(1), v(1)) +
                                                         inner₊(∇ₕ(u(2)), ∇ₕ(v(2)))))
            x, xb = rand(MersenneTwister(KE_SEED), size(K, 2)), rand(size(KB, 2))
            y, yb = fill(NaN, size(K, 1)), fill(NaN, size(KB, 1))
            @test !stale(() -> K * x) && !stale(() -> KB * xb)
            mutate!(Ωₕ)
            for (A, u, v) in ((K, x, y), (KB, xb, yb))
                @test stale(() -> mul!(v, A, u))
                @test stale(() -> mul!(v, A, u, 2.0, 0.0))
                @test all(isnan, v)
                @test stale(() -> A * u)
                @test stale(() -> A[1, 1])
                @test stale(() -> SparseMatrixCSC(A))
                @test stale(() -> Matrix(A))
            end
            @test stale(() -> Kronecker.kronecker(K))
            @test stale(() -> Bramble.fdm_solve(K, x))
            # Rebuilt on the changed mesh, it works (refining one axis alone leaves the
            # mesh inconsistent, and `assemble` with it).
            kind === :refine_one_axis && continue
            Wn = gridspace(Ωₕ)
            Kn = kronecker_operator(lap(Wn))
            xn = rand(MersenneTwister(KE_SEED), size(Kn, 2))
            @test isapprox(Kn * xn, assemble(lap(Wn)) * xn; rtol = 1e-12)
        end

        # Two operators on one threaded mesh both go stale; the one rebuilt does not.
        Ωₕ = _ke_mesh((7, 6, 5), CpuThreaded(); graded = false)
        W = gridspace(Ωₕ)
        K1, K2 = kronecker_operator(lap(W)),
        kronecker_operator(form(W, W, (u, v) -> innerₕ(u, v)))
        x = rand(MersenneTwister(KE_SEED), size(K1, 1))
        y = fill(NaN, size(K1, 1))
        Bramble.change_points!(Ωₕ(3), collect(range(0, 1; length = 5) .^ 2))
        @test stale(() -> mul!(y, K1, x)) && stale(() -> mul!(y, K2, x, 1.0, 0.0))
        @test all(isnan, y)

        # After a refinement the vectors that fit the new mesh meet the stale error first,
        # not a size error; `K * x` alone reports the size, which is accepted. Showing the
        # operator says it is stale.
        Ωₕ = _ke_mesh((7, 6); graded = false)
        W, V = gridspace(Ωₕ), gridspace(Ωₕ, Val(2))
        K = kronecker_operator(lap(W))
        KB = kronecker_operator(form(V, V, (u, v) -> innerₕ(u(1), v(1)) +
                                                     inner₊(∇ₕ(u(2)), ∇ₕ(v(2)))))
        txt(A) = sprint(show, MIME"text/plain"(), A)
        Bramble.iterative_refinement!(Ωₕ)
        n, nb = ndofs(gridspace(Ωₕ)), ndofs(gridspace(Ωₕ, Val(2)))
        for (A, m) in ((K, n), (KB, nb))
            u, v = rand(m), fill(NaN, m)
            @test size(A, 1) < m
            @test stale(() -> mul!(v, A, u))
            @test stale(() -> mul!(v, A, u, 0.5, 1.0))
            @test stale(() -> A[m, m])
            @test_throws DimensionMismatch A * u
            @test all(isnan, v)
            @test startswith(txt(A), summary(A)) && occursin("stale", txt(A))
        end

        # A form built on spaces from before the change is refused by every entry point:
        # the space's error, naming the remedy for the space.
        space_error = "gridspace(mesh(Wₕ)) again"
        for mutate! in (Ω -> Bramble.change_points!(Ω, (range(0, 1; length = 7) .^ 2,
            range(0, 1; length = 6) .^ 2)),
            Bramble.iterative_refinement!)
            Ωₕ = _ke_mesh((7, 6); graded = false)
            W, V = gridspace(Ωₕ), gridspace(Ωₕ, Val(2))
            a = lap(W)
            b = form(V, V, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(1)), v(2)))
            mutate!(Ωₕ)
            F = rand(MersenneTwister(KE_SEED), 42)
            @test stale(() -> assemble(a), space_error)
            @test stale(() -> kronecker_operator(a), space_error)
            @test stale(() -> is_separable(a), space_error)
            @test stale(() -> kronecker_operator(b), space_error)
            @test stale(() -> Bramble.fdm_solve(a, F), space_error)
        end
    end
end # testset

end # module
