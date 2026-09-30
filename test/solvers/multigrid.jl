module SolversMultigridTests

using Test
using Bramble
using Bramble: GeometricMeshHierarchy, set_markers!, spacings, interpolation_matrix, CpuPolyester
using Bramble: AbstractSmoother, JacobiSmoother, ChebyshevSmoother, RedBlackGaussSeidel, max_eigenvalue_estimate,
               trial_space, D₋ₓ, D₋ᵧ, M₊ᵧ
using Bramble: GMGPreconditioner, AbstractMatrixFreePreconditioner, VectorElement, v_cycle!, w_cycle!, fmg!,
               change_points!
using LinearAlgebra: LinearAlgebra, dot, norm, diag, LowerTriangular, UpperTriangular, ldiv!, Symmetric, eigmin
using LinearSolve: LinearProblem, KrylovJL_CG, solve
using ForwardDiff: ForwardDiff
using Random

# Geometric multigrid (gpena/Bramble.jl#329). Meshes are non-uniform throughout: on a uniform
# mesh rebuilding each level from the domain would nest too, and hide a hierarchy that does
# not take every other point.

const MG_SEED = 3291

# The markers a mesh gets from `Ω` at its own points, for comparison with the carried ones.
function _mg_reevaluated_markers(Ωₕ, Ω)
    Ωr = copy(Ωₕ)
    set_markers!(Ωr, markers(Ω); warn_marker_mismatch = false)
    return markers(Ωr)
end

_mg_axes(p::AbstractVector{<:Number}) = (p,)
_mg_axes(p::Tuple) = p

# A vector indexed from 0, to check that the transfers refuse offset axes.
struct _MgZeroBased <: AbstractVector{Float64}
    p::Vector{Float64}
end
Base.size(v::_MgZeroBased) = size(v.p)
Base.axes(v::_MgZeroBased) = (Base.IdentityUnitRange(0:(length(v.p) - 1)),)
Base.getindex(v::_MgZeroBased, i::Int) = v.p[i + 1]
Base.setindex!(v::_MgZeroBased, x, i::Int) = (v.p[i + 1] = x)

# Allocation of a warmed transfer, behind a function barrier.
_mg_palloc(xf, H, l, xc) = (prolongate!(xf, H, l, xc); @allocated prolongate!(xf, H, l, xc))
_mg_calloc(xc, H, l, xf) = (coarsen!(xc, H, l, xf); @allocated coarsen!(xc, H, l, xf))

# The oracle `P`.
_mg_oracle(Ωf, Ωc) = interpolation_matrix(gridspace(Ωf), gridspace(Ωc))

_mg_agree(a, b) = isapprox(a, b; rtol = 1e-13, atol = 1e-13)

# Non-uniform meshes in 1D, 2D and 3D, and two with a collapsed axis (which also cover
# `interpolation_matrix` on collapsed axes, gpena/Bramble.jl#396), with a level count each.
function _mg_transfer_meshes(bk = backend())
    Random.seed!(MG_SEED)
    unit(a = 0.0, b = 1.0) = interval(a, b)
    return (
        (mesh(domain(unit()), 33, false; backend = bk), 4),
        (mesh(domain(unit() × unit(0.0, 2.0)), (17, 9), false; backend = bk), 3),
        (mesh(domain(unit() × unit(-1.0, 1.0) × unit(0.0, 2.0)), (9, 5, 9), false; backend = bk), 3),
        (mesh(domain(unit() × unit(0.5, 0.5)), (17, 4), false; backend = bk), 3),
        (mesh(domain(unit() × unit(0.5, 0.5) × unit()), (9, 4, 5), false; backend = bk), 2)
    )
end

@testset "gmg: nested hierarchy" begin
    Random.seed!(MG_SEED)
    cases = (
        (domain(interval(0.0, 1.0), :inlet => :left, :near => x -> x[1] < 0.3), 33, 4),
        (
            domain(interval(0.0, 1.0) × interval(0.0, 2.0), :walls => (:top, :bottom),
                :blob => x -> (x[1] - 0.5)^2 + (x[2] - 1.0)^2 < 0.2),
            (17, 33),
            3),
        (
            domain(interval(0.0, 1.0) × interval(-1.0, 1.0) × interval(0.0, 2.0),
                :inlet => :left, :half => x -> x[3] < 1.0),
            (9, 5, 17),
            3)
    )
    for (Ω, n, L) in cases
        Ωf = mesh(Ω, n, false)
        H = GeometricMeshHierarchy(Ωf, L)
        @test length(H) == L && lastindex(H) == L && firstindex(H) == 1
        @test H[end] === Ωf
        @test all(Ωₗ -> Ωₗ isa typeof(Ωf), H)
        @test collect(H) == [H[l] for l in 1:L]
        for l in 1:(L - 1)
            pc, pf = _mg_axes(points(H[l])), _mg_axes(points(H[l + 1]))
            # Exact equality: the coarse points are copies of the fine ones, not recomputed.
            @test all(d -> pc[d] == pf[d][1:2:end], eachindex(pc))
            @test npoints(H[l], Tuple) == map(k -> (k - 1) ÷ 2 + 1, npoints(H[l + 1], Tuple))
            # Cached metrics follow the new points, not the uniform ones built first.
            @test all(d -> _mg_axes(spacings(H[l]))[d] ≈ [pc[d][2] - pc[d][1]; diff(pc[d])], eachindex(pc))
        end
        @test npoints(H[1], Tuple) == map(k -> (k - 1) ÷ 2^(L - 1) + 1, npoints(Ωf, Tuple))
        # Custom markers carry over, and agree with re-evaluating the domain at each level.
        for l in 1:L
            @test markers(H[l]) == _mg_reevaluated_markers(H[l], Ω)
        end
        @test keys(markers(H[1])) == keys(markers(Ωf))
        # A space on the coarsest level sees its own points.
        @test ndofs(gridspace(H[1])) == npoints(H[1])
    end

    # The element type and backend of the finest mesh are kept on every level.
    Ω32 = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (9, 17), false)
    H32 = GeometricMeshHierarchy(Ω32, 4)
    @test all(Ωₗ -> eltype(Ωₗ) == Float32 && backend(Ωₗ) == backend(Ω32), H32)
    @test npoints(H32[1], Tuple) == (2, 3)

    # One level is the mesh alone.
    Ω1 = mesh(domain(interval(0.0, 1.0)), 10, false)
    @test only(collect(GeometricMeshHierarchy(Ω1, 1))) === Ω1

    # A collapsed axis stays one point.
    Ωc = mesh(domain(interval(0.0, 1.0) × interval(0.5, 0.5)), (9, 4), false)
    @test npoints(GeometricMeshHierarchy(Ωc, 3)[1], Tuple) == (3, 1)

    # (n - 1) not divisible by 2^(levels - 1) on some axis, and levels < 1.
    @test_throws ArgumentError GeometricMeshHierarchy(mesh(domain(interval(0.0, 1.0)), 18, false), 2)
    @test_throws ArgumentError GeometricMeshHierarchy(
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 11), false), 3)
    @test_throws ArgumentError GeometricMeshHierarchy(
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9, 7), false), 3)
    @test_throws ArgumentError GeometricMeshHierarchy(Ω1, 0)
    Ω33 = mesh(domain(interval(0.0, 1.0)), 33, false)
    @test_throws ArgumentError GeometricMeshHierarchy(Ω33, 65)
    @test eachindex(GeometricMeshHierarchy(Ω33, 2)) == 1:2

    @test sprint(show, GeometricMeshHierarchy(Ω32, 3)) ==
          "GeometricMeshHierarchy{2D, 3 levels, (3, 5) to (9, 17) pts}"
end

# `P` is `interpolation_matrix` of the fine space from the coarse one, never built by the
# transfers; `coarsen!` is `Pᵀ` without scaling.
@testset "gmg: prolongation and transpose" begin
    for (Ωf, L) in _mg_transfer_meshes()
        H = GeometricMeshHierarchy(Ωf, L)
        for l in 2:L
            Wf, Wc = gridspace(H[l]), gridspace(H[l - 1])
            P = _mg_oracle(H[l], H[l - 1])
            xc, yf = randn(ndofs(Wc)), randn(ndofs(Wf))
            xf, yc = fill(NaN, ndofs(Wf)), fill(NaN, ndofs(Wc))
            @test prolongate!(xf, H, l, xc) === xf
            @test _mg_agree(xf, P * xc)
            y0 = copy(yf)
            @test coarsen!(yc, H, l, yf) === yc
            @test _mg_agree(yc, P' * yf)
            # ⟨P xc, yf⟩ = ⟨xc, Pᵀ yf⟩, and the fine input is left as it was.
            @test isapprox(dot(xf, yf), dot(xc, yc); rtol = 1e-13)
            @test yf == y0
            @test _mg_palloc(xf, H, l, xc) == 0
            @test _mg_calloc(yc, H, l, yf) == 0
            # Grid functions give the same as their raw storage.
            uc, uf = element(Wc), element(Wf)
            parent(uc) .= xc
            @test prolongate!(uf, H, l, uc) === uf && parent(uf) == xf
            parent(uf) .= yf
            @test parent(coarsen!(uc, H, l, uf)) == yc
            # A view as the destination, as a column of a matrix.
            M = zeros(ndofs(Wf), 2)
            prolongate!(view(M, :, 2), H, l, xc)
            @test M[:, 2] == xf && iszero(M[:, 1])
        end
        # Two levels composed are the product of the two matrices.
        if L >= 3
            P32, P21 = _mg_oracle(H[3], H[2]), _mg_oracle(H[2], H[1])
            x1 = randn(npoints(H[1]))
            x2, x3 = zeros(npoints(H[2])), zeros(npoints(H[3]))
            prolongate!(x3, H, 3, prolongate!(x2, H, 2, x1))
            @test _mg_agree(x3, P32 * (P21 * x1))
        end
    end

    # Interpolation reproduces the multilinear function x₁ + 2 x₂ x₃ - x₂ exactly.
    Ωf, L = _mg_transfer_meshes()[3]
    H = GeometricMeshHierarchy(Ωf, L)
    f(p) = p[1] + 2 * p[2] * p[3] - p[2]
    grid(Ω) = vec([f((a, b, c)) for a in points(Ω(1)), b in points(Ω(2)), c in points(Ω(3))])
    xf = zeros(npoints(H[3]))
    @test _mg_agree(prolongate!(xf, H, 3, grid(H[2])), grid(H[3]))

    # The unscaled transpose is Galerkin for the 1D stiffness on a non-uniform mesh: the
    # forms carry the discrete measure, so Pᵀ A_f P is the coarse form's matrix.
    Ω1, _ = _mg_transfer_meshes()[1]
    H = GeometricMeshHierarchy(Ω1, 3)
    stiff(Ω) = (W = gridspace(Ω); assemble(form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))))
    for l in 2:3
        n = npoints(H[l - 1])
        PA = zeros(npoints(H[l]), n)
        for j in 1:n
            e = zeros(n)
            e[j] = 1
            prolongate!(view(PA, :, j), H, l, e)
        end
        Ac = stiff(H[l - 1])
        @test norm(PA' * stiff(H[l]) * PA - Ac, Inf) < 1e-12 * norm(Ac, Inf)
    end

    # Dual numbers flow through both transfers: the Jacobians are P and Pᵀ.
    Ωf, L = _mg_transfer_meshes()[2]
    H = GeometricMeshHierarchy(Ωf, L)
    P = _mg_oracle(H[3], H[2])
    Jp = ForwardDiff.jacobian(x -> prolongate!(similar(x, npoints(H[3])), H, 3, x), randn(npoints(H[2])))
    Jc = ForwardDiff.jacobian(y -> coarsen!(similar(y, npoints(H[2])), H, 3, y), randn(npoints(H[3])))
    @test _mg_agree(Jp, Matrix(P)) && _mg_agree(Jc, Matrix(P'))

    # Float32 levels transfer in Float32.
    Ω32 = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (9, 5), false)
    H32 = GeometricMeshHierarchy(Ω32, 2)
    P32 = interpolation_matrix(gridspace(H32[2]), gridspace(H32[1]))
    x32 = rand(Float32, npoints(H32[1]))
    @test isapprox(prolongate!(zeros(Float32, npoints(H32[2])), H32, 2, x32), P32 * x32; rtol = 1.0f-5)

    # Wrong lengths, a level with no coarser one or none at all, and offset axes.
    Ωf, L = _mg_transfer_meshes()[2]
    H = GeometricMeshHierarchy(Ωf, L)
    nf, nc = npoints(H[2]), npoints(H[1])
    for (a, b) in ((zeros(nf + 1), zeros(nc)), (zeros(nf), zeros(nc - 1)), (zeros(nc), zeros(nf)))
        @test_throws DimensionMismatch prolongate!(a, H, 2, b)
        @test_throws DimensionMismatch coarsen!(b, H, 2, a)
    end
    for l in (0, 1, L + 1)
        @test_throws ArgumentError prolongate!(zeros(nf), H, l, zeros(nc))
        @test_throws ArgumentError coarsen!(zeros(nc), H, l, zeros(nf))
    end
    @test_throws ArgumentError prolongate!(_MgZeroBased(zeros(nf)), H, 2, zeros(nc))
    @test_throws ArgumentError prolongate!(zeros(nf), H, 2, _MgZeroBased(zeros(nc)))
    @test_throws ArgumentError coarsen!(_MgZeroBased(zeros(nc)), H, 2, zeros(nf))
    @test_throws ArgumentError coarsen!(zeros(nc), H, 2, _MgZeroBased(zeros(nf)))
    buf = zeros(nf + nc)
    @test_throws ArgumentError prolongate!(view(buf, 1:nf), H, 2, view(buf, 4:(3 + nc)))
    @test_throws ArgumentError coarsen!(view(buf, 4:(3 + nc)), H, 2, view(buf, 1:nf))
    @test_throws ArgumentError prolongate!(zeros(1), GeometricMeshHierarchy(Ω1, 1), 1, zeros(1))
end

# The threaded and Polyester sweeps write every point once, so they equal the serial ones
# bitwise, on every repeat. Run at `--threads=4` for a race to have a chance. `CpuPolyester`
# needs Polyester, which the `unit` group does not load.
@testset "gmg: threaded transfers agree" begin
    policies = Any[Parallel()]
    Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing && push!(policies, CpuPolyester())
    for policy in policies
        for ((Ωs, L), (Ωt, _)) in zip(_mg_transfer_meshes(), _mg_transfer_meshes(backend(; policy)))
            @test Bramble.execution_policy(Ωt) == policy
            @test _mg_axes(points(Ωt)) == _mg_axes(points(Ωs))
            Hs, Ht = GeometricMeshHierarchy(Ωs, L), GeometricMeshHierarchy(Ωt, L)
            for l in 2:L
                xc, yf = randn(npoints(Hs[l - 1])), randn(npoints(Hs[l]))
                xf, yc = prolongate!(zeros(length(yf)), Hs, l, xc), coarsen!(zeros(length(xc)), Hs, l, yf)
                xt, yt = similar(xf), similar(yc)
                @test all(1:20) do _
                    prolongate!(xt, Ht, l, xc)
                    coarsen!(yt, Ht, l, yf)
                    return xt == xf && yt == yc
                end
            end
        end
    end
    # What a threaded transfer allocates is its task spawns, whatever the grid size.
    bytes = map((17, 257)) do n
        Random.seed!(MG_SEED)
        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n), false;
            backend = backend(policy = Parallel()))
        H = GeometricMeshHierarchy(Ω, 2)
        xc, xf = randn(npoints(H[1])), zeros(npoints(H[2]))
        return (_mg_palloc(xf, H, 2, xc), _mg_calloc(xc, H, 2, xf))
    end
    @test bytes[1] == bytes[2]
end

# Smoothers (#329): mass plus variable diffusion on non-uniform meshes, with and without
# Dirichlet rows, against dense references built from `assemble`.
function _mg_smoother_form(D, n; T = Float64)
    Random.seed!(MG_SEED)
    I = interval(T(0), T(1))
    Ω = D == 1 ? I : D == 2 ? I × interval(T(0), T(2)) : I × I × interval(T(-1), T(1))
    W = gridspace(mesh(domain(Ω), n, ntuple(_ -> false, D)))
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    return form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v)))
end

const _MG_SMOOTHER_SHAPES = ((1, 33), (2, (17, 13)), (3, (9, 7, 9)))

# The colour of each unknown: the parity of its zero-based index sum.
_mg_colours(dims) = [mod(sum(Tuple(I)) - length(dims), 2) for I in CartesianIndices(dims)][:]

# `sweeps` of forward Gauss-Seidel in the red-then-black ordering: the lower triangle of the
# permuted matrix, solved densely.
function _mg_rbgs_dense(A, x, b, dims, sweeps)
    c = _mg_colours(dims)
    p = [findall(==(0), c); findall(==(1), c)]
    Ap = A[p, p]
    xp, bp = x[p], b[p]
    for _ in 1:sweeps
        xp += LowerTriangular(Ap) \ (bp - Ap * xp)
    end
    y = similar(x)
    y[p] = xp
    return y
end

# The Chebyshev smoother from its error polynomial: x - x⋆ ↦ T_k((θ - D⁻¹A)/δ) / T_k(θ/δ) (x - x⋆).
function _mg_cheb_dense(A, x, b, λmin, λmax, k)
    θ, δ = (λmax + λmin) / 2, (λmax - λmin) / 2
    Id = Matrix{Float64}(LinearAlgebra.I, size(A))
    t = (θ * Id - A ./ diag(A)) / δ
    T0, T1 = Id, t
    for _ in 2:k
        T0, T1 = T1, 2 * t * T1 - T0
    end
    xs = A \ b
    return xs + (T1 / cosh(k * acosh(θ / δ))) * (x - xs)
end

# The highest-frequency grid function, (-1)^(sum of indices).
_mg_checkerboard(dims) = [(-1.0)^sum(Tuple(I)) for I in CartesianIndices(dims)][:]

_mg_salloc(s, x, b) = (smooth!(s, x, b); @allocated smooth!(s, x, b))

@testset "gmg: smoothers damp high frequencies" begin
    for (D, n) in _MG_SMOOTHER_SHAPES, dl in (nothing, :boundary)

        a = _mg_smoother_form(D, n)
        W = trial_space(a)
        dims = npoints(mesh(W), Tuple)
        kw = dl === nothing ? (;) : (; dirichlet = dl)
        op = matrix_free_operator(a; kw...)
        A = Matrix(assemble(a; kw...))
        Random.seed!(MG_SEED + D)
        x0, b = randn(ndofs(W)), randn(ndofs(W))

        x = copy(x0)
        @test smooth!(jacobi_smoother(op; ω = 0.7, sweeps = 2), x, b) === x
        y = x0 + 0.7 * (b - A * x0) ./ diag(A)
        @test _mg_agree(x, y + 0.7 * (b - A * y) ./ diag(A))
        @test _mg_agree(smooth!(jacobi_smoother(a; kw..., ω = 0.7, sweeps = 2), copy(x0), b), x)

        s = chebyshev_smoother(op; degree = 3)
        @test s.λmax ≈ max_eigenvalue_estimate(op; preconditioner = jacobi_preconditioner(op))
        @test s.λmin ≈ s.λmax / 4
        @test _mg_agree(smooth!(s, copy(x0), b), _mg_cheb_dense(A, x0, b, s.λmin, s.λmax, 3))

        for sweeps in (1, 2)
            x = smooth!(red_black_gauss_seidel(a; kw..., sweeps), copy(x0), b)
            @test _mg_agree(x, _mg_rbgs_dense(A, x0, b, dims, sweeps))
            # A Dirichlet row is the identity row, so Gauss-Seidel solves it exactly.
            if dl !== nothing
                bnd = findall(i -> A[i, i] == 1 && count(!iszero, A[i, :]) == 1, 1:ndofs(W))
                @test !isempty(bnd) && _mg_agree(x[bnd], b[bnd])
            end
        end
        # A black half-sweep solves the black equations: their residual vanishes.
        x = smooth!(red_black_gauss_seidel(op), copy(x0), b)
        @test norm((b - A * x)[_mg_colours(dims) .== 1], Inf) < 1e-10 * norm(b, Inf)

        # The checkerboard is near the top of the spectrum of D⁻¹A, λ ≈ 2: a Jacobi sweep
        # multiplies it by about |1 - 2ω|, 1/3 at ω = 2/3 and (2D - 1)/(2D + 1) at the
        # default; Chebyshev and red-black remove more than half of its residual.
        hf = _mg_checkerboard(dims)
        ratio(sm) = norm(A * smooth!(sm, copy(hf), zeros(ndofs(W)))) / norm(A * hf)
        @test ratio(jacobi_smoother(op; ω = 2 / 3)) < 0.4
        @test ratio(jacobi_smoother(op)) < (2D - 1) / (2D + 1) + 0.05
        @test ratio(chebyshev_smoother(op)) < 0.5
        @test ratio(red_black_gauss_seidel(op)) < 0.5
        for sm in (jacobi_smoother(op), chebyshev_smoother(op), red_black_gauss_seidel(op))
            @test _mg_salloc(sm, copy(x0), b) == 0
            # A `VectorElement` iterate and right-hand side act on their storage.
            uₕ, bₕ = element(W, copy(x0)), element(W, b)
            @test smooth!(sm, uₕ, bₕ) === uₕ
            @test parent(uₕ) == smooth!(sm, copy(x0), b)
        end
    end

    # Transposed difference pairs and region restrictions couple neighbours of opposite
    # colour only, so red-black takes them.
    W = trial_space(_mg_smoother_form(2, (9, 7)))
    for a in (form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(D₋ₓ(u), v) + 2.0 * innerₕ(u, D₋ₓ(v))),
        form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, Bramble.restrict_to(:boundary, v))))
        x0, b = randn(ndofs(W)), randn(ndofs(W))
        @test _mg_agree(smooth!(red_black_gauss_seidel(a), copy(x0), b),
            _mg_rbgs_dense(Matrix(assemble(a)), x0, b, npoints(mesh(W), Tuple), 1))
    end

    # Float32 stays Float32 and allocation-free.
    a32 = _mg_smoother_form(2, (17, 13); T = Float32)
    n32 = ndofs(trial_space(a32))
    x32, b32 = randn(Float32, n32), zeros(Float32, n32)
    hf32 = Float32.(_mg_checkerboard((17, 13)))
    A32 = assemble(a32)
    for sm in (jacobi_smoother(a32; ω = 2 / 3), chebyshev_smoother(a32), red_black_gauss_seidel(a32))
        @test eltype(sm) === Float32
        @test sm.inv_diagonal isa Vector{Float32}
        @test eltype(smooth!(sm, copy(hf32), b32)) === Float32
        @test norm(A32 * smooth!(sm, copy(hf32), b32)) < 0.5f0 * norm(A32 * hf32)
        @test _mg_salloc(sm, x32, b32) == 0
    end

    # A mixed difference couples diagonal neighbours, which share a colour.
    W = trial_space(_mg_smoother_form(2, (9, 7)))
    box = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ᵧ(D₋ₓ(u)), D₋ᵧ(D₋ₓ(v))))
    @test_throws ArgumentError red_black_gauss_seidel(box)
    @test_throws ArgumentError red_black_gauss_seidel(matrix_free_operator(box))
    @test_throws ArgumentError red_black_gauss_seidel(form(W, W, (u, v) -> innerₕ(D₋ₓ(D₋ₓ(u)), v)))
    # An average across y of a difference along x couples diagonal neighbours too, but on a
    # collapsed y axis they do not exist, so the same form is taken there. (A difference
    # along a collapsed axis divides by its zero spacing, hence the average.)
    avg(W) = form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(M₊ᵧ(D₋ₓ(u)), M₊ᵧ(D₋ₓ(v))))
    @test_throws ArgumentError red_black_gauss_seidel(avg(W))
    Random.seed!(MG_SEED)
    Wc = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.5, 0.5)), (17, 4), false))
    @test npoints(mesh(Wc), Tuple) == (17, 1)
    boxc = avg(Wc)
    x0, b = randn(17), randn(17)
    @test _mg_agree(smooth!(red_black_gauss_seidel(boxc), copy(x0), b),
        _mg_rbgs_dense(Matrix(assemble(boxc)), x0, b, (17, 1), 1))
    # A composite space is refused; Jacobi and Chebyshev take it.
    V = W × W
    c = form(V, V, (u, v) -> innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) + innerₕ(u(2), v(2)))
    @test_throws ArgumentError red_black_gauss_seidel(c)
    Ac = Matrix(assemble(c))
    x0, b = randn(ndofs(V)), randn(ndofs(V))
    @test _mg_agree(smooth!(jacobi_smoother(c), copy(x0), b), x0 + 0.8 * (b - Ac * x0) ./ diag(Ac))

    a = _mg_smoother_form(2, (9, 7))
    n = ndofs(trial_space(a))
    for sm in (jacobi_smoother(a), chebyshev_smoother(a; λmax = 2.5), red_black_gauss_seidel(a))
        @test sm isa AbstractSmoother{Float64}
        @test size(sm) == (n, n)
        @test_throws DimensionMismatch smooth!(sm, zeros(n + 1), zeros(n))
        @test_throws DimensionMismatch smooth!(sm, zeros(n), zeros(n - 1))
        x = zeros(n)
        @test_throws ArgumentError smooth!(sm, x, x)
        buf = zeros(2n)
        @test_throws ArgumentError smooth!(sm, view(buf, 1:n), view(buf, 2:(n + 1)))
        @test_throws ArgumentError smooth!(sm, _MgZeroBased(zeros(n)), zeros(n))
    end
    @test jacobi_smoother(a) isa JacobiSmoother
    @test chebyshev_smoother(a) isa ChebyshevSmoother
    @test red_black_gauss_seidel(a) isa RedBlackGaussSeidel
    @test_throws ArgumentError jacobi_smoother(a; sweeps = 0)
    @test_throws ArgumentError jacobi_smoother(a; ω = 0)
    @test_throws ArgumentError jacobi_smoother(a; ω = Inf)
    @test_throws ArgumentError chebyshev_smoother(a; degree = 0)
    @test_throws ArgumentError chebyshev_smoother(a; λmax = -1.0)
    @test_throws ArgumentError red_black_gauss_seidel(a; sweeps = 0)
    Wr = gridspace(mesh(domain(interval(0.0, 1.0)), 9, false))
    rect = form(Wr, trial_space(_mg_smoother_form(1, 5)), (u, v) -> innerₕ(u, v))
    @test_throws DimensionMismatch jacobi_smoother(rect)
end

# Cycles (#329): mass plus variable diffusion with natural boundary conditions, SPD, on
# non-uniform meshes, against dense references built from `assemble` and
# `interpolation_matrix`.
function _mg_box(D)
    D == 2 ? interval(0.0, 1.0) × interval(0.0, 1.0) :
    interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)
end

_mg_spd(W) = (κ = Rₕ(W, x -> 1 + sum(abs2, x)); form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))))

# Uniform points jittered by up to ±0.3h along each axis: non-uniform everywhere, with
# bounded cell aspect ratio.
function _mg_jitter_mesh(D, n; bk = backend(), seed = MG_SEED)
    rng = Random.Xoshiro(seed)
    Ω = mesh(domain(_mg_box(D)), ntuple(_ -> n, D), ntuple(_ -> true, D); backend = bk)
    h = 1 / (n - 1)
    function pts()
        x = collect(range(0.0, 1.0; length = n)) .+ 0.3h .* (2 .* rand(rng, n) .- 1)
        x[1], x[end] = 0.0, 1.0
        return sort!(x)
    end
    change_points!(Ω, ntuple(_ -> pts(), D))
    return Ω
end

function _mg_cg(op, b, P)
    sol = solve(LinearProblem(op, b), KrylovJL_CG(); Pl = P, reltol = 1e-8, abstol = 0.0, maxiters = 2000)
    return sol.iters, norm(op * sol.u - b) / norm(b)
end

# One cycle with damped Jacobi (`ν₁`, `ν₂` sweeps of weight `ω`) and `γ` coarse cycles per
# level, from the iterate `x`, on dense level matrices `As` and prolongations `Ps`.
function _mg_dense_cycle(As, Ps, ω, ν₁, ν₂, γ, l, x, b)
    l == 1 && return As[1] \ b
    A = As[l]
    d = diag(A)
    for _ in 1:ν₁
        x = x + ω * (b - A * x) ./ d
    end
    bc = Ps[l]' * (b - A * x)
    xc = zeros(length(bc))
    for _ in 1:γ
        xc = _mg_dense_cycle(As, Ps, ω, ν₁, ν₂, γ, l - 1, xc, bc)
    end
    x = x + Ps[l] * xc
    for _ in 1:ν₂
        x = x + ω * (b - A * x) ./ d
    end
    return x
end

function _mg_dense_fmg(As, Ps, ω, ν₁, ν₂, b)
    L = length(As)
    bs = Vector{Vector{Float64}}(undef, L)
    bs[L] = b
    for l in L:-1:2
        bs[l - 1] = Ps[l]' * bs[l]
    end
    x = As[1] \ bs[1]
    for l in 2:L
        x = _mg_dense_cycle(As, Ps, ω, ν₁, ν₂, 1, l, Ps[l] * x, bs[l])
    end
    return x
end

# `sweeps` of backward Gauss-Seidel in the red-then-black ordering.
function _mg_rbgs_dense_reverse(A, x, b, dims, sweeps)
    c = _mg_colours(dims)
    p = [findall(==(0), c); findall(==(1), c)]
    Ap = A[p, p]
    xp, bp = x[p], b[p]
    for _ in 1:sweeps
        xp += UpperTriangular(Ap) \ (bp - Ap * xp)
    end
    y = similar(x)
    y[p] = xp
    return y
end

# The cycles compose many products on random meshes (level matrices of condition ~300 and
# more), so they agree with the dense references to about 1e-13 relative, not to the last bit.
_mg_close(a, b) = isapprox(a, b; rtol = 1e-11)

_mg_ldalloc(y, P, x) = (ldiv!(y, P, x); @allocated ldiv!(y, P, x))
_mg_cycalloc(f, x, P, b) = (f(x, P, b); @allocated f(x, P, b))

# The matrix of `x ↦ P \ x`.
_mg_dense_inverse(P) = (n = first(size(P)); reduce(hcat, [P \ [Float64(i == j) for i in 1:n] for j in 1:n]))

@testset "gmg: mesh-independent CG" begin
    # The cycles against dense references, on a random non-uniform mesh.
    Random.seed!(MG_SEED)
    Ω = mesh(domain(_mg_box(2)), (17, 9), (false, false))
    As = Any[]
    Ps = Any[nothing]
    H = GeometricMeshHierarchy(Ω, 3)
    for l in 1:3
        push!(As, Matrix(assemble(_mg_spd(gridspace(H[l])))))
        l > 1 && push!(Ps, interpolation_matrix(gridspace(H[l]), gridspace(H[l - 1])))
    end
    n = npoints(Ω)
    x0, b = randn(n), randn(n)
    for (ν₁, ν₂) in ((1, 1), (2, 1), (0, 2))
        P = gmg_preconditioner(_mg_spd, Ω; ν₁, ν₂, smoother = op -> jacobi_smoother(op; ω = 0.7))
        @test length(P.hierarchy) == 3 && npoints(P.hierarchy[1], Tuple) == (5, 3)
        @test _mg_close(v_cycle!(copy(x0), P, b), _mg_dense_cycle(As, Ps, 0.7, ν₁, ν₂, 1, 3, x0, b))
        @test _mg_close(w_cycle!(copy(x0), P, b), _mg_dense_cycle(As, Ps, 0.7, ν₁, ν₂, 2, 3, x0, b))
        @test _mg_close(fmg!(copy(x0), P, b), _mg_dense_fmg(As, Ps, 0.7, ν₁, ν₂, b))
        @test _mg_close(P \ b, v_cycle!(zeros(n), P, b))
    end
    for (cyc, γ) in ((:V, 1), (:W, 2))
        P = gmg_preconditioner(_mg_spd, Ω; cycle = cyc, smoother = op -> jacobi_smoother(op; ω = 0.7))
        @test _mg_close(P \ b, _mg_dense_cycle(As, Ps, 0.7, 2, 2, γ, 3, zeros(n), b))
    end
    P = gmg_preconditioner(_mg_spd, Ω; cycle = :FMG, smoother = op -> jacobi_smoother(op; ω = 0.7))
    @test _mg_close(P \ b, _mg_dense_fmg(As, Ps, 0.7, 2, 2, b))

    # Reversed red-black is backward Gauss-Seidel; the default order is unchanged.
    rb = red_black_gauss_seidel(_mg_spd(gridspace(Ω)); sweeps = 2)
    @test _mg_agree(smooth!(rb, copy(x0), b; reverse = true), _mg_rbgs_dense_reverse(As[3], x0, b, (17, 9), 2))
    @test _mg_agree(smooth!(rb, copy(x0), b; reverse = false), _mg_rbgs_dense(As[3], x0, b, (17, 9), 2))
    @test _mg_salloc(rb, copy(x0), b) == 0

    # Post-smoothing is the adjoint of pre-smoothing, so the V- and W-cycles are SPD for
    # every smoother, red-black included.
    for sm in (op -> chebyshev_smoother(op), op -> jacobi_smoother(op), op -> red_black_gauss_seidel(op)),
        cyc in (:V, :W)

        B = _mg_dense_inverse(gmg_preconditioner(_mg_spd, Ω; cycle = cyc, smoother = sm))
        @test norm(B - B') <= 1e-12 * norm(B)
        @test eigmin(Symmetric(B)) > 0
    end

    # CG iterations do not grow with the mesh on meshes of bounded cell aspect ratio.
    for (D, ns) in ((2, (17, 33, 65)), (3, (9, 17)))
        its = map(ns) do n
            Ωf = _mg_jitter_mesh(D, n)
            P = gmg_preconditioner(_mg_spd, Ωf)
            @test P isa GMGPreconditioner{Float64} && P isa AbstractMatrixFreePreconditioner{Float64}
            op = matrix_free_operator(_mg_spd(gridspace(Ωf)))
            i, r = _mg_cg(op, randn(size(op, 1)), P)
            @test r < 1e-7
            return i
        end
        @test maximum(its) <= 12 && maximum(its) - minimum(its) <= 3
    end
    for cyc in (:W, :FMG)
        Ωf = _mg_jitter_mesh(2, 33)
        op = matrix_free_operator(_mg_spd(gridspace(Ωf)))
        i, r = _mg_cg(op, randn(size(op, 1)), gmg_preconditioner(_mg_spd, Ωf; cycle = cyc))
        @test r < 1e-7 && i <= 12
    end
    # A random base mesh refined by `iterative_refinement!` keeps its stretched cells, where
    # point smoothers are slower; CG still converges.
    for (D, n₀, k) in ((2, 5, 3), (3, 5, 2))
        Random.seed!(MG_SEED + D)
        Ωf = mesh(domain(_mg_box(D)), ntuple(_ -> n₀, D), ntuple(_ -> false, D))
        foreach(_ -> iterative_refinement!(Ωf), 1:k)
        @test npoints(Ωf, Tuple) == ntuple(_ -> (n₀ - 1) * 2^k + 1, D)
        op = matrix_free_operator(_mg_spd(gridspace(Ωf)))
        for sm in (op -> chebyshev_smoother(op), op -> red_black_gauss_seidel(op))
            P = gmg_preconditioner(_mg_spd, Ωf; levels = k + 1, smoother = sm)
            _, r = _mg_cg(op, randn(size(op, 1)), P)
            @test r < 1e-7
        end
    end

    # Cycles allocate nothing on a serial policy; `VectorElement`s act on their storage.
    Ωf = _mg_jitter_mesh(2, 33)
    W = gridspace(Ωf)
    n = ndofs(W)
    x0, b = randn(n), randn(n)
    for cyc in (:V, :W, :FMG), sm in (op -> chebyshev_smoother(op), op -> red_black_gauss_seidel(op))

        P = gmg_preconditioner(_mg_spd, Ωf; cycle = cyc, smoother = sm)
        @test _mg_ldalloc(zeros(n), P, b) == 0
        y = P \ b
        @test ldiv!(P, copy(b)) == y
        xₕ, bₕ = element(W, 0.0), element(W, b)
        @test ldiv!(xₕ, P, bₕ) === xₕ && parent(xₕ) == y
        for f in (v_cycle!, w_cycle!, fmg!)
            @test _mg_cycalloc(f, copy(x0), P, b) == 0
            uₕ = element(W, copy(x0))
            @test f(uₕ, P, bₕ) === uₕ && parent(uₕ) == f(copy(x0), P, b)
        end
    end
    P = gmg_preconditioner(_mg_spd, Ωf)
    @test sprint(show, P) == "GMGPreconditioner{V(2,2), 5 levels, (3, 3) to (33, 33) pts}"
    @test size(P) == (n, n) && eltype(P) === Float64
    @test_throws DimensionMismatch ldiv!(zeros(n + 1), P, zeros(n))
    @test_throws DimensionMismatch ldiv!(zeros(n), P, zeros(n - 1))
    @test_throws ArgumentError ldiv!(_MgZeroBased(zeros(n)), P, zeros(n))
    for f in (v_cycle!, w_cycle!, fmg!)
        @test_throws DimensionMismatch f(zeros(n + 1), P, zeros(n))
        x = zeros(n)
        @test_throws ArgumentError f(x, P, x)
        @test_throws ArgumentError f(_MgZeroBased(zeros(n)), P, zeros(n))
    end

    # Default levels coarsen until an axis would drop below three points.
    for (np, L, nc) in (((33, 33), 5, (3, 3)), ((97, 17), 4, (13, 3)), ((17, 5), 2, (9, 3)))
        Ωd = mesh(domain(_mg_box(2)), np, (true, true))
        Pd = gmg_preconditioner(_mg_spd, Ωd)
        @test length(Pd.hierarchy) == L && npoints(Pd.hierarchy[1], Tuple) == nc
    end
    Pd = gmg_preconditioner(_mg_spd, mesh(domain(_mg_box(2)), (97, 97), (true, true)))
    @test length(Pd.hierarchy) == 6 && npoints(Pd.hierarchy[1], Tuple) == (4, 4)

    # Float32 stays Float32 and allocation-free.
    Ω32 = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (17, 17), (false, false))
    P32 = gmg_preconditioner(_mg_spd, Ω32)
    b32 = randn(Float32, npoints(Ω32))
    @test eltype(P32) === Float32 && eltype(P32 \ b32) === Float32
    @test _mg_ldalloc(zeros(Float32, npoints(Ω32)), P32, b32) == 0
    # The default tolerance √eps(Float32) sits above Float32's rounding floor; one below it
    # is refused with a message naming the tolerance and the element type.
    u32 = gmg_solve(_mg_spd, Ω32, b32)
    @test eltype(parent(u32)) === Float32
    A32 = assemble(_mg_spd(gridspace(Ω32)))
    @test norm(A32 * parent(u32) - b32) <= sqrt(eps(Float32)) * norm(b32)
    err = try
        gmg_solve(_mg_spd, Ω32, b32; tol = 1e-9, maxiters = 30)
    catch e
        e
    end
    @test err isa ErrorException && occursin("Float32", err.msg) && occursin("tol = 1.0e-9", err.msg)

    # `gmg_solve` reaches its tolerance and agrees with a direct solve.
    A = assemble(_mg_spd(W))
    xs = Matrix(A) \ b
    for cyc in (:V, :W, :FMG)
        uₕ = gmg_solve(_mg_spd, Ωf, b; cycle = cyc, tol = 1e-10)
        @test uₕ isa VectorElement && length(parent(uₕ)) == n
        @test norm(A * parent(uₕ) - b) <= 1e-10 * norm(b)
        @test isapprox(parent(uₕ), xs; rtol = 1e-8)
    end
    @test parent(gmg_solve(_mg_spd, Ωf, element(W, b); tol = 1e-10)) ≈ parent(gmg_solve(_mg_spd, Ωf, b; tol = 1e-10))
    @test iszero(parent(gmg_solve(_mg_spd, Ωf, zeros(n))))
    @test_throws ErrorException gmg_solve(_mg_spd, Ωf, b; tol = 1e-15, maxiters = 1)
    @test_throws ArgumentError gmg_solve(_mg_spd, Ωf, b; tol = 0)
    @test_throws ArgumentError gmg_solve(_mg_spd, Ωf, b; maxiters = 0)
    @test_throws DimensionMismatch gmg_solve(_mg_spd, Ωf, zeros(n + 1))

    # Construction refuses what the cycles cannot run.
    @test_throws ArgumentError gmg_preconditioner(_mg_spd, Ωf; cycle = :F)
    @test_throws ArgumentError gmg_preconditioner(_mg_spd, Ωf; ν₁ = -1)
    @test_throws ArgumentError gmg_preconditioner(_mg_spd, Ωf; ν₁ = 0, ν₂ = 0)
    @test_throws ArgumentError gmg_preconditioner(_mg_spd, Ωf; levels = 2.5)
    @test_throws ArgumentError gmg_preconditioner(_mg_spd, Ωf; smoother = op -> jacobi_preconditioner(op))
    @test_throws ArgumentError gmg_preconditioner(_mg_spd, Ωf; levels = 7)
    @test_throws ArgumentError gmg_preconditioner(W -> 1.0, Ωf)
    @test_throws ArgumentError gmg_preconditioner(_ -> _mg_spd(W), Ωf)
    @test_throws ArgumentError gmg_preconditioner(
        V -> (
            C = V × V; form(C, C, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))), Ωf)
    @test_throws ArgumentError gmg_preconditioner(_mg_spd, mesh(domain(_mg_box(2)), (65, 66), (true, true)))
    # A 3-point axis allows no coarser level, so the message does not suggest more levels.
    err = try
        gmg_preconditioner(_mg_spd, mesh(domain(_mg_box(2)), (2049, 3), (true, true)))
    catch e
        e
    end
    @test err isa ArgumentError && occursin("if the hierarchy allows them", sprint(showerror, err))
end

@testset "gmg: threaded cycles agree" begin
    policies = Any[Parallel()]
    Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing && push!(policies, CpuPolyester())
    for policy in policies, (D, n) in ((2, 33), (3, 9))

        Ωs = _mg_jitter_mesh(D, n)
        Ωt = _mg_jitter_mesh(D, n; bk = backend(; policy))
        @test points(Ωt) == points(Ωs)
        b = randn(npoints(Ωs))
        for cyc in (:V, :W, :FMG)
            Ps = gmg_preconditioner(_mg_spd, Ωs; cycle = cyc)
            Pt = gmg_preconditioner(_mg_spd, Ωt; cycle = cyc)
            @test all(op -> op.policy == policy, Pt.ops)
            ys = Ps \ b
            yt = similar(ys)
            @test all(1:20) do _
                ldiv!(yt, Pt, b)
                return isapprox(yt, ys; rtol = 1e-12, atol = 1e-14)
            end
        end
        @test isapprox(parent(gmg_solve(_mg_spd, Ωt, b)), parent(gmg_solve(_mg_spd, Ωs, b)); rtol = 1e-10)
    end
end

end # module SolversMultigridTests
