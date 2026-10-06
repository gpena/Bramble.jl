#===========================================================================#
# Schur vs eigenvector route across a Peclet sweep: gpena/Bramble.jl#443, S4 of
# .claude/plans/v3-26-0-separable-solvers.md.
#
# Usage:
#     julia --threads=2 --project=benchmark benchmark/schur_peclet.jl
#     SCHUR_PECLET_QUICK=1 julia --threads=2 --project=benchmark benchmark/schur_peclet.jl
#
# `fdm_solve` solves a non-symmetric separable form (an advection term) through the
# complex generalised Schur form `schur(complex(A_d), complex(M_d))` of each axis pair
# (`_SchurFactorization`, `ext/BrambleKroneckerExt.jl`): unitary transforms and triangular
# factors with no 2x2 blocks, then a Bartels-Stewart back substitution.
# #443's premise is that fast diagonalisation through the non-symmetric eigendecomposition
# `M_d \ A_d = V_d Λ_d V_d⁻¹` would formally apply but loses accuracy as advection
# dominates, because the eigenvectors `V_d` grow ill-conditioned. The library never ships
# that route; it is implemented here only, to put numbers on the premise.
#
# The form is `ε inner₊(∇ₕ u, ∇ₕ v) + innerₕ(D₋ₓ u, v) + innerₕ(u, v)` with
# homogeneous Dirichlet (`dirichlet = :boundary`), on the unit square/cube whose axis `d`
# points are moved to `t^(1 + d/4)` (graded, no two axes share their nodes). The Peclet
# number is `Pe = 1/ε`: advection speed 1 over a unit length. For each `Pe` the script prints
#
#     PECLET dim=<D> n=<n> Pe=<Pe> schur=<relerr> eigvec=<relerr>
#
# where both relative errors (2-norm) are against the sparse direct solve `assemble(a) \ F`
# (`n` is the number of unknowns solved). Accuracy, not timing, so no power/load preflight.
# The default run also prints the same rows as a markdown table for the issue comment;
# `SCHUR_PECLET_QUICK=1` runs small meshes and the table is skipped.
#
# Eigenvector route: with `M_d \ A_d = V_d Λ_d V_d⁻¹` (complex in general) the system
# `K = Σ_d M_1 ⊗ … ⊗ A_d ⊗ … ⊗ M_D + c_m M_1 ⊗ … ⊗ M_D` factors as
# `K = (⊗_d M_d V_d) (Σ_d Λ_d + c_m) (⊗_d V_d⁻¹)`, so `K⁻¹ F` is one mode product per
# axis with `(M_d V_d)⁻¹`, a pointwise division by `Σ_d Λ_d + c_m`, then one per axis
# with `V_d`.
#===========================================================================#

using Bramble

using Bramble: D₋ₓ
using Kronecker: Kronecker  # loads BrambleKroneckerExt, which owns fdm_solve
using LinearAlgebra: eigen, norm, inv

const KronExt = Base.get_extension(Bramble, :BrambleKroneckerExt)
const QUICK = get(ENV, "SCHUR_PECLET_QUICK", "") == "1"

# Mesh sizes per dimension, and the Peclet numbers swept (the largest is the last).
const SIZES = QUICK ? Dict(2 => (24, 20), 3 => (10, 9, 8)) :
              Dict(2 => (300, 250), 3 => (40, 36, 32))
const PECLETS = QUICK ? [1.0e0, 1.0e2, 1.0e4] :
                [1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6]

# Uniform, then moved to `t^(1 + d/4)` along axis `d`.
function graded_space(n::NTuple{D, Int}) where {D}
    Ω = mesh(domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))), n,
        ntuple(_ -> false, D))
    Bramble.change_points!(Ω,
        ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D))
    return gridspace(Ω)
end

# `Y = X ×_d R`: apply the dense `R` along axis `d` of the array `X`.
function mode_product(X::AbstractArray{<:Any, D}, R::AbstractMatrix, d::Int) where {D}
    dims = size(X)
    pre, post = prod(dims[1:(d - 1)]; init = 1), prod(dims[(d + 1):end]; init = 1)
    X3 = reshape(X, pre, dims[d], post)
    Y = Array{promote_type(eltype(X), eltype(R)), 3}(undef, pre, size(R, 1), post)
    for k in 1:post
        Y[:, :, k] = X3[:, :, k] * transpose(R)
    end
    return reshape(Y, ntuple(j -> j == d ? size(R, 1) : dims[j], D))
end

# `K \ F` for the interior unknowns through the non-symmetric eigendecomposition of every
# axis pair `(A_d, M_d)`, as the header derives. Returns the full-length real vector.
function eigvec_solve(K, F::AbstractVector, dims_full::NTuple{D, Int}) where {D}
    rng = ntuple(d -> 2:(dims_full[d] - 1), Val(D))
    dims = map(length, rng)
    interior = vec(LinearIndices(dims_full)[rng...])
    M, A, c_m, _ = KronExt._fdm_axis_data(K, rng)
    dec = ntuple(Val(D)) do d
        Md = Matrix(M[d])
        e = eigen(Md \ Matrix(A[d]))
        (V = e.vectors, W = inv(Md * e.vectors), λ = e.values)
    end
    Λ = fill(complex(c_m), dims)
    for d in 1:D
        Λ .+= reshape(dec[d].λ, ntuple(k -> k == d ? dims[d] : 1, Val(D)))
    end
    Y = reshape(F[interior], dims)
    for d in 1:D
        Y = mode_product(Y, dec[d].W, d)
    end
    Y ./= Λ
    for d in 1:D
        Y = mode_product(Y, dec[d].V, d)
    end
    x = zeros(length(F))
    x[interior] = real.(vec(Y))
    return x
end

relerr(x, xref) = norm(x - xref) / norm(xref)

# One row: both routes against the sparse direct solve.
function measure(n::NTuple{D, Int}, Pe::Float64) where {D}
    Wₕ = graded_space(n)
    ε = 1 / Pe
    a = form(Wₕ, Wₕ,
        (u, v) -> ε * inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(u), v) +
                  innerₕ(u, v))
    A = assemble(a; dirichlet = :boundary)
    # Deterministic, and no Random dependency in the benchmark environment.
    F = [sin(0.37i) + 0.5cos(1.3i) for i in 1:size(A, 1)]
    F[Bramble._combined_mask(mesh(Wₕ), (:boundary,))] .= 0
    xref = A \ F
    schur = relerr(fdm_solve(a, F; dirichlet = :boundary), xref)
    eig = relerr(eigvec_solve(kronecker_operator(a), F, n), xref)
    return (dim = D, n = prod(n .- 2), Pe = Pe, schur = schur, eigvec = eig)
end

function main()
    println("Schur vs eigenvector route, Peclet sweep -- gpena/Bramble.jl#443 (S4)")
    println("Threads       : ", Threads.nthreads(), " (", Sys.CPU_THREADS, " CPU cores)")
    println("Mode          : ", QUICK ? "quick (SCHUR_PECLET_QUICK=1)" : "default")
    println()
    rows = NamedTuple[]
    for D in (2, 3), Pe in PECLETS
        r = measure(SIZES[D], Pe)
        push!(rows, r)
        println("PECLET dim=$(r.dim) n=$(r.n) Pe=$(r.Pe) schur=$(r.schur) ",
            "eigvec=$(r.eigvec)")
    end
    if !QUICK
        println()
        println("| dim | unknowns | Pe | Schur | eigenvector | eigenvector / Schur |")
        println("|---|---|---|---|---|---|")
        # Three significant digits as `m.mmeE`, without a Printf dependency.
        function fmt(x)
            (iszero(x) || !isfinite(x)) && return string(x)
            e = floor(Int, log10(abs(x)))
            m = round(x / 10.0^e; digits = 2)
            m >= 10 && ((m, e) = (m / 10, e + 1))
            return string(m, "e", e)
        end
        for r in rows
            println("| $(r.dim)D | $(r.n) | $(fmt(r.Pe)) | $(fmt(r.schur)) | ",
                "$(fmt(r.eigvec)) | $(fmt(r.eigvec / max(r.schur, 1.0e-16))) |")
        end
    end
    return rows
end

main()
