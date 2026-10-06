# ext/BrambleKroneckerExt.jl: `Kronecker.jl` interop and fast diagonalisation for a
# separable `BilinearForm`, layered on top of the dependency-free `KroneckerLinearOperator` in
# `src/assembly/kronecker.jl`.
#
# Two independent pieces, both read `KroneckerLinearOperator`'s own `terms` (each a
# `KroneckerTerm{D}` of `scales` and `factors`, `src/assembly/kronecker.jl`) rather than
# re-walking the form's AST:
#
#   1. `Kronecker.kronecker(K)`: the same object as `SparseMatrixCSC(K)`
#      (`src/assembly/kronecker.jl`), built from `Kronecker.jl`'s own `⊗` instead of `kron` --
#      for a single term this stays the lazy `KroneckerProduct` `Kronecker.jl` itself
#      returns from `⊗`; summing more than one term falls back to `Kronecker.jl`'s own
#      `AbstractMatrix` `+`, which materialises (verified directly: `A ⊗ B` agrees with
#      `kron(A, B)`, not `kron(B, A)`, so no axis-order flip is needed against
#      `SparseMatrixCSC(K)`'s own `foldl(kron, reverse(factors))` convention).
#   2. `fdm_solve`: a direct solve for a Laplacian-like system `assemble(a) \ F`
#      (optionally with homogeneous Dirichlet on the whole mesh boundary) by fast
#      diagonalisation, refusing every other form -- see the derivation comment below.
#
# `Kronecker.kronecker` is `Kronecker.jl`'s own generic function, extended here like any
# other package extension method. `fdm_solve` is Bramble's: `src/Bramble.jl` declares
# `function fdm_solve end` and exports it, the same shape `_sparspak_factorize` has for the
# Sparspak extension, and the methods below attach to that binding. They must therefore be
# written `function Bramble.fdm_solve(...)`, dot-qualified: an unqualified
# `function fdm_solve(...)` alongside `using Bramble: Bramble` defines a *different*
# function local to this module, which then answers nobody's call to `Bramble.fdm_solve`.
# That is what happened here first, and it is silent -- the extension loads, the tests that
# reach it through `Base.get_extension` pass, and only the exported spelling stays empty.
module BrambleKroneckerExt

using Bramble:
               Bramble,
               BilinearForm,
               is_separable,
               kronecker_operator,
               KroneckerLinearOperator,
               domain,
               interval,
               ×,
               mesh,
               gridspace,
               boundary_symbols,
               form,
               assemble,
               Rₕ,
               inner₊,
               ∇ₕ,
               innerₕ
using Kronecker: Kronecker, ⊗
using LinearAlgebra: LinearAlgebra, Diagonal, Symmetric, eigen, isposdef, issymmetric, ldiv!,
                     mul!
using SparseArrays: SparseMatrixCSC, dropzeros!, nonzeros, nnz, sparse
using PrecompileTools: @setup_workload, @compile_workload

# --- 1. Conversion to a Kronecker.jl object ------------------------------------------ #

# One term's coefficient times its Kronecker product, last axis leftmost -- the same
# convention `SparseMatrixCSC(K)` uses (`src/assembly/kronecker.jl`), with `⊗` standing in for
# `kron` (they agree, module docstring above).
@inline function _kron_jl_term(term)
    c = Bramble._kron_coeff(term.scales)
    return c * foldl(⊗, reverse(term.factors))
end

"""
    Kronecker.kronecker(K::KroneckerLinearOperator) -> AbstractMatrix

Convert `K` (see [`kronecker_operator`](@ref)) into the equivalent `Kronecker.jl` object: the
sum, over `K`'s terms, of `coefficient * (F_D ⊗ ... ⊗ F_1)` built with `Kronecker.jl`'s own
`⊗`. A single-term `K` (for instance a bare `innerₕ(u, v)`) stays the lazy
`Kronecker.KroneckerProduct` `⊗` itself returns; summing two or more terms falls back to
`Kronecker.jl`'s own `AbstractMatrix` addition, which materialises a plain `Matrix` -- the
same thing calling `+` on two `Kronecker.jl` objects of unrelated shape does anywhere else,
not a limitation specific to this method.

`collect(Kronecker.kronecker(K))` and `Matrix(Kronecker.kronecker(K))` agree with
`SparseMatrixCSC(K)` (`src/assembly/kronecker.jl`) -- both are the same sum of Kronecker
products, read off the same `K.terms`.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
Wₕ = gridspace(Ωₕ)
K = kronecker_operator(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))
collect(Kronecker.kronecker(K)) ≈ SparseMatrixCSC(K)
```

See also: [`kronecker_operator`](@ref), [`KroneckerLinearOperator`](@ref).
"""
function Kronecker.kronecker(K::KroneckerLinearOperator)
    Bramble._kron_check_fresh(K)
    terms = K.terms
    result = _kron_jl_term(terms[1])
    for term in Base.tail(terms)
        result = result + _kron_jl_term(term)
    end
    return result
end

# --- 2. Fast diagonalisation ---------------------------------------------------------- #
#
# Derivation.
#
# Fast diagonalisation solves a Laplacian-like system: one whose matrix is
#
#     Σ_d M_D ⊗ ... ⊗ A_d ⊗ ... ⊗ M_1   +   c_m * (M_D ⊗ ... ⊗ M_1)
#
# with one symmetric positive definite mass `M_d` and one symmetric `A_d` per axis. How
# `A_d` was built does not matter, so the solve classifies `K`'s terms (each `c * (F_D ⊗ ...
# ⊗ F_1)`, coefficient read now so a `Ref` stays live) by factor equality, not by type
# (`_fdm_axis_data`): it picks a mass per axis among the factors the terms carry on that
# axis such that every term equals the masses on all axes but at most one. A term equal to
# them everywhere adds its coefficient to `c_m`; a term differing on axis `d` adds
# `c * F_d` to `A_d` -- a directional stiffness, an averaged or jump term, an `inner_Γ` face
# (a Robin term), or a mass term whose coefficient varies along `d` alone. A non-diagonal
# mass is fine (an averaged `innerₕ(Mₓ(u), Mₓ(v))` can be the mass). For each axis, solve
# the generalised eigenproblem
#
#     A_d Q_d = M_d Q_d Λ_d,     Q_d' M_d Q_d = I     (Λ_d diagonal)
#
# `LinearAlgebra.eigen(Symmetric(A_d), Symmetric(M_d))` normalises exactly this way (LAPACK's
# symmetric-definite driver, `sygvd`), which is what makes the next step work without an
# extra `M_d` anywhere. Substitute `x = (Q_D ⊗ ... ⊗ Q_1) y` into the axis-`d` addend and
# left-multiply by `(Q_D ⊗ ... ⊗ Q_1)'`:
#
#     (Q_D ⊗ ... ⊗ Q_1)' (M_D ⊗ ... ⊗ A_d ⊗ ... ⊗ M_1) (Q_D ⊗ ... ⊗ Q_1)
#         = (Q_D' M_D Q_D) ⊗ ... ⊗ (Q_d' A_d Q_d) ⊗ ... ⊗ (Q_1' M_1 Q_1)
#         = I ⊗ ... ⊗ Λ_d ⊗ ... ⊗ I
#
# using `Q_c' M_c Q_c = I` on every axis `c ≠ d`, and the mass addend becomes `I ⊗ ... ⊗ I`
# outright by the same identity on every axis. So in the `y`-basis the whole operator is
# diagonal, entry `j = (j_1, ..., j_D)` reading `Λ_total[j] = c_m + Σ_d Λ_d[j_d]`, and
#
#     y = ((Q_D ⊗ ... ⊗ Q_1)' F) ./ Λ_total,     x = (Q_D ⊗ ... ⊗ Q_1) y
#
# applied axis by axis (sum factorisation) rather than by ever forming `Q_D ⊗ ... ⊗ Q_1`.
# Every other form is refused, never solved: a term differing from every choice of masses
# on two axes (a mixed derivative, a coefficient varying along two axes), a mass that is not
# symmetric positive definite, an axis no term differs on, a non-symmetric `A_d` (advection,
# whose eigenvectors grow ill-conditioned, gpena/Bramble.jl#443), a composite space, or a
# singular system (an entry of `Λ_total` that is zero to rounding, as for a pure-Neumann
# operator with no mass term).
#
# Homogeneous Dirichlet (`dirichlet = :boundary`). `:boundary` marks every point with at
# least one coordinate on an axis's first or last index, so its complement -- the interior --
# is exactly the tensor product, over every axis, of that axis's own interior points
# (`2:end-1`): a point is interior in the domain iff it is interior along *every* axis. For a
# zero boundary value, the interior rows/columns of the unconstrained system are exactly the
# reduced operator (a boundary column's contribution is its entry times the -- zero --
# boundary unknown, which drops out), and the interior block of a Kronecker product is the
# product of the factors' interior blocks. So every factor is restricted to `2:end-1` before
# the classification above, and the full solution is the reduced one embedded back with zero
# at every boundary dof. A factor that vanishes there drops its term (an `inner_Γ` face does),
# and a mass with a zero boundary weight (an `:interior` restriction) can become definite.
# A 2-point axis leaves no interior at all; the solution is then zero everywhere.

const _FDM_KRYLOV = "`kronecker_operator(a)` and a Krylov solver (for instance " *
                    "`KrylovJL_GMRES` through LinearSolve.jl), or with "

# Every refusal says `fdm_solve` does not support the form, names `reason`, and points to the
# Krylov route that solves any separable form (`krylov = false` for a form that is not).
@noinline function _throw_fdm_unsupported(reason::AbstractString; krylov::Bool = true)
    throw(
        ArgumentError(
        "fdm_solve does not support this form: $reason. It solves only Laplacian-like " *
        "forms (see its docstring); solve this one with $(krylov ? _FDM_KRYLOV : "")" *
        "`assemble(a)` and a sparse direct solve.",
    ),
    )
end

function _throw_fdm_not_separable(a)
    _throw_fdm_unsupported(
        "it is not separable (see `is_separable`): a grid-function coefficient varying along " *
        "two or more axes (spell a product of one-axis coefficients `fx * (fy * u)`), a " *
        "region restriction, an interpolation, or a 1D mesh has no Kronecker factors";
        krylov = false)
end

function _throw_fdm_composite()
    _throw_fdm_unsupported(
        "it is posed on a composite space, whose `kronecker_operator` is a block operator " *
        "(`KroneckerBlockOperator`); fast diagonalisation needs one scalar Kronecker sum")
end

@noinline function _throw_fdm_bad_dirichlet(dirichlet)
    throw(
        ArgumentError(
        "fdm_solve only supports dirichlet = nothing (unconstrained) or " *
        "dirichlet = :boundary (homogeneous Dirichlet on the whole mesh boundary); got " *
        "$(repr(dirichlet)).",
    ),
    )
end

@noinline function _throw_fdm_length_mismatch(n::Int, m::Int)
    throw(DimensionMismatch("fdm_solve: F has length $m, the operator needs $n"))
end

# The reason the furthest-reaching choice of masses failed at, by stage (see
# `_fdm_axis_data`): 1 no choice fits, 2 a mass is not SPD, 3 an axis has no term, 4 an
# axis operator is not symmetric; 5 (from `_fdm_solve_core`) the system is singular.
function _throw_fdm_stage(stage::Int, d::Int)
    stage == 1 && _throw_fdm_unsupported(
        "a term has non-mass factors on two axes (a mixed derivative, or a coefficient " *
        "varying along two axes), so no choice of one mass per axis leaves every term " *
        "differing from the masses on at most one axis")
    stage == 2 && _throw_fdm_unsupported(
        "the axis-$d mass is not symmetric positive definite (a zero or negative weight, " *
        "for instance from an :interior restriction without dirichlet = :boundary)")
    stage == 3 && _throw_fdm_unsupported(
        "axis $d has no term of its own (every term equals the mass along it), so it has no " *
        "1D operator to diagonalise")
    stage == 5 && _throw_fdm_unsupported(
        "the system is singular: its fast-diagonalisation eigenvalues include zero, for " *
        "instance a pure-Neumann operator with no mass term; add a mass term or use " *
        "dirichlet = :boundary")
    _throw_fdm_unsupported(
        "the axis-$d operator is not symmetric (an advection term such as " *
        "innerₕ(D₋ₓ(u), v)); a non-symmetric Kronecker sum needs a Schur-form solve")
end

# A device-backed `K` holds `_KronDeviceDiagonal`/`_KronDeviceSparse`
# factors (`src/assembly/kronecker.jl`); `_fdm_host_factor` brings each back to the host as a
# `Diagonal`/`SparseMatrixCSC` first. They are the 1D factors, O(n_d) per axis, and the
# eigendecomposition below is a host LAPACK call anyway, so this copy is negligible.
_fdm_host_factor(F) = F
_fdm_host_factor(F::Bramble._KronDeviceDiagonal) = Diagonal(Array(F.diag))
function _fdm_host_factor(F::Bramble._KronDeviceSparse)
    m = length(F.colptr) - 1
    return SparseMatrixCSC(m, m, Vector{Int}(Array(F.colptr)), Vector{Int}(Array(F.rowval)),
        Array(F.nzval))
end

# One factor as the classification compares it: on the host, sparse, restricted to `r`
# (`2:end-1` under `dirichlet = :boundary`), stored zeros dropped.
_fdm_factor(F, r) = dropzeros!(sparse(_fdm_host_factor(F))[r, r])

# Factor equality up to rounding: the same factor is usually the same cached object.
function _fdm_same(A::SparseMatrixCSC, B::SparseMatrixCSC)
    A === B && return true
    scale = max(maximum(abs, nonzeros(A); init = 0.0), maximum(abs, nonzeros(B); init = 0.0))
    return maximum(abs, nonzeros(A - B); init = 0.0) <= 8 * eps(Float64) * scale
end

_fdm_spd(M::SparseMatrixCSC) = issymmetric(M) && isposdef(Symmetric(Matrix(M)))

# Classifies `K`'s terms on the factors restricted to `rng` (see the derivation): returns the
# per-axis masses `M`, the per-axis operators `A` (coefficients folded in) and the summed
# coefficient `c_m` of the terms equal to the masses everywhere, or throws the reason no
# choice works. The candidate masses on axis `d` are the distinct axis-`d` factors of the
# terms, tried most frequent first; every choice that fits is exact, so the first one that
# also passes the SPD, every-axis and symmetry checks is used.
function _fdm_axis_data(K::KroneckerLinearOperator{T, D}, rng::NTuple{D}) where {T, D}
    cs = T[]
    fs = NTuple{D, SparseMatrixCSC{T, Int}}[]
    for term in K.terms
        c = T(Bramble._kron_coeff(term.scales))  # `T`: a Float32 device keeps Float32
        f = ntuple(d -> SparseMatrixCSC{T, Int}(_fdm_factor(term.factors[d], rng[d])), Val(D))
        # A zero term (a zero `Ref`, or a factor that vanishes on the interior) adds nothing.
        (iszero(c) || any(F -> nnz(F) == 0, f)) && continue
        push!(cs, c)
        push!(fs, f)
    end
    # `reps[d]` the distinct axis-`d` factors, `cls[i][d]` which one term `i` carries.
    reps = ntuple(_ -> SparseMatrixCSC{T, Int}[], Val(D))
    cls = [ntuple(Val(D)) do d
               k = findfirst(R -> _fdm_same(R, f[d]), reps[d])
               k === nothing ? (push!(reps[d], f[d]); length(reps[d])) : k
           end
           for f in fs]
    order = ntuple(d -> sort(eachindex(reps[d]); by = k -> -count(c -> c[d] == k, cls)), Val(D))
    stage, axis = 1, 0
    for choice in Iterators.product(order...)
        on = zeros(Int, length(fs))  # the one axis term `i` differs on, 0 if none
        fits = true
        for (i, c) in enumerate(cls)
            diff = findall(d -> c[d] != choice[d], 1:D)
            length(diff) > 1 && (fits = false; break)
            on[i] = isempty(diff) ? 0 : diff[1]
        end
        fits || continue
        M = ntuple(d -> reps[d][choice[d]], Val(D))
        bad = findfirst(d -> !_fdm_spd(M[d]), 1:D)
        bad === nothing || ((stage, axis) = max((stage, axis), (2, bad)); continue)
        bad = findfirst(d -> !any(==(d), on), 1:D)
        bad === nothing || ((stage, axis) = max((stage, axis), (3, bad)); continue)
        A = ntuple(d -> sum(cs[i] * fs[i][d] for i in eachindex(fs) if on[i] == d), Val(D))
        bad = findfirst(d -> !issymmetric(A[d]), 1:D)
        bad === nothing || ((stage, axis) = max((stage, axis), (4, bad)); continue)
        c_m = sum((cs[i] for i in eachindex(fs) if on[i] == 0); init = zero(T))
        return M, A, c_m
    end
    return _throw_fdm_stage(stage, axis)
end

# The generalised eigendecomposition per axis and the combined eigenvalue grid
# `Λ_total[j] = c_m + Σ_d Λ_d[j_d]` the derivation above needs (coefficients are already in
# `A_d`).
function _fdm_eigendecompose(M, A, c_m::T, dims::NTuple{D, Int}) where {T, D}
    decomps = ntuple(Val(D)) do d
        eigen(Symmetric(Matrix(A[d])), Symmetric(Matrix(M[d])))
    end
    Q = ntuple(d -> decomps[d].vectors, Val(D))
    Λ = fill(c_m, dims)
    for d in 1:D
        shape = ntuple(k -> k == d ? dims[d] : 1, Val(D))
        Λ .+= reshape(decomps[d].values, shape)
    end
    return Q, Λ
end

# The factorisation `fdm_solve` applies: everything before the apply, done once
# (classification, restriction, per-axis `eigen`, `Λ_total`, singularity refusal), plus the
# workspace that lets `ldiv!` on host vectors allocate nothing. `u`/`w` are the two
# ping-pong buffers of the solved (interior, under `:boundary`) unknowns; `u3[d]`/`w3[d]`
# are the same memory viewed as `(pre, n_d, post)` for the axis-`d` mode product, built
# here because a `reshape` per application would allocate. `interior` holds the linear
# indices of the solved unknowns in the full vector (empty for `dirichlet = nothing`).
struct _FDMFactorization{T, D}
    n::Int
    boundary::Bool
    interior::Vector{Int}
    Q::NTuple{D, Matrix{T}}
    Λ::Array{T, D}
    λ::Vector{T}
    u::Vector{T}
    w::Vector{T}
    u3::NTuple{D, Array{T, 3}}
    w3::NTuple{D, Array{T, 3}}
end

_fdm_slabs(v::Vector, dims::NTuple{D, Int}) where {D} =
    ntuple(d -> reshape(v, prod(dims[1:(d - 1)]; init = 1), dims[d],
            prod(dims[(d + 1):end]; init = 1)), Val(D))

function _fdm_factorize(K::KroneckerLinearOperator{T, D}, dirichlet) where {T, D}
    (dirichlet === nothing || dirichlet === :boundary) || _throw_fdm_bad_dirichlet(dirichlet)
    dims_full = K.dims
    boundary = dirichlet === :boundary
    rng = ntuple(d -> boundary ? (2:(dims_full[d] - 1)) : (1:dims_full[d]), Val(D))
    dims = map(length, rng)
    interior = boundary ? vec(LinearIndices(dims_full)[rng...]) : Int[]
    if prod(dims) == 0
        # A 2-point axis leaves no interior: every unknown is a Dirichlet one, and zero.
        Q = ntuple(d -> zeros(T, dims[d], dims[d]), Val(D))
        Λ = zeros(T, dims)
    else
        M, A, c_m = _fdm_axis_data(K, rng)
        Q, Λ = _fdm_eigendecompose(M, A, c_m, dims)
        # A zero (to rounding) of `Λ_total` is a zero eigenvalue of the system: dividing by
        # it would return a huge `x` that does not solve it. `Λ_total` sums `D` per-axis
        # eigenvalues, each off by about `eps(T) * maximum(abs, Λ)`, whatever the grid size.
        minimum(abs, Λ) <= 16 * D * eps(T) * maximum(abs, Λ) && _throw_fdm_stage(5, 0)
    end
    u = Vector{T}(undef, prod(dims))
    w = similar(u)
    return _FDMFactorization{T, D}(K.n, boundary, interior, Q, Λ, vec(Λ), u, w,
        _fdm_slabs(u, dims), _fdm_slabs(w, dims))
end

# One mode product `Y = R' *_d X` between the ping-pong buffers, `X` in `u` when `inu`:
# `Y[i, :, k] = X[i, :, k] * R`, a `mul!` per `pre x n_d` slab (one `mul!` on the
# `n_d x post` matricisation when `pre == 1`). Returns where the result now lives.
function _fdm_mode!(f::_FDMFactorization, inu::Bool, d::Int, R::AbstractMatrix)
    X, Y = inu ? (f.u3[d], f.w3[d]) : (f.w3[d], f.u3[d])
    if size(X, 1) == 1
        mul!(view(Y, 1, :, :), transpose(R), view(X, 1, :, :))
    else
        for k in axes(X, 3)
            mul!(view(Y, :, :, k), view(X, :, :, k), R)
        end
    end
    return !inu
end

# `x = K \ F` on full-length host vectors: gather the solved unknowns, apply every `Q_d'`,
# divide by `Λ_total`, apply every `Q_d`, scatter back (zero on the boundary). `2D` mode
# products, an even number, so the result always ends in `u`. `x` may alias `F`.
function LinearAlgebra.ldiv!(x::AbstractVector, f::_FDMFactorization{T, D},
        F::AbstractVector) where {T, D}
    # `interior` holds 1-based positions, used under `@inbounds` below.
    Base.require_one_based_indexing(x, F)
    length(F) == f.n || _throw_fdm_length_mismatch(f.n, length(F))
    length(x) == f.n || _throw_fdm_length_mismatch(f.n, length(x))
    u = f.u
    if f.boundary
        @inbounds for (i, j) in enumerate(f.interior)
            u[i] = F[j]
        end
    else
        copyto!(u, F)
    end
    if !isempty(u)
        inu = true
        for d in 1:D
            inu = _fdm_mode!(f, inu, d, f.Q[d])
        end
        (inu ? f.u : f.w) ./= f.λ
        for d in 1:D
            inu = _fdm_mode!(f, inu, d, transpose(f.Q[d]))
        end
    end
    if f.boundary
        fill!(x, zero(eltype(x)))
        @inbounds for (i, j) in enumerate(f.interior)
            x[j] = u[i]
        end
    else
        copyto!(x, u)
    end
    return x
end

# Device path. `X` and `M` are both device arrays, so the mode
# product is one dense device matmul with no host round trip. Axis `d` is brought to the
# front with `permutedims` (a device kernel), `M * X2` runs on the `(m, pre * post)`
# matricisation, and `permutedims` puts the axis back. For `d == 1` no permutation is
# needed at all.
function _fdm_apply_mode(X::AbstractArray{T}, M::AbstractMatrix, d::Int) where {T}
    dims = size(X)
    pre = prod(dims[1:(d - 1)]; init = 1)
    m = dims[d]
    post = prod(dims[(d + 1):end]; init = 1)
    mo = size(M, 1)
    newdims = ntuple(i -> i == d ? mo : dims[i], length(dims))
    if pre == 1
        Y2 = similar(X, T, mo, post)
        mul!(Y2, M, reshape(X, m, post))
        return reshape(Y2, newdims)
    end
    Xp = permutedims(reshape(X, pre, m, post), (2, 1, 3))
    Y2 = similar(X, T, mo, pre * post)
    mul!(Y2, M, reshape(Xp, m, pre * post))
    return reshape(permutedims(reshape(Y2, mo, pre, post), (2, 1, 3)), newdims)
end

# Copies a host matrix/array to storage like the device vector `F` (`similar` + `copyto!`,
# so no GPU package is named here); the identity for a host `F`.
_fdm_to_storage(::Array, A::AbstractArray) = A
function _fdm_to_storage(F::AbstractArray, A::AbstractArray{T}) where {T}
    B = similar(F, T, size(A))
    copyto!(B, Array(A))
    return B
end

# Sum factorisation: `F` into the eigenbasis axis by axis (`Q_d'`), divide by the combined
# eigenvalue grid, transform back (`Q_d`) -- the three steps the derivation comment ends on.
function _fdm_apply(Q::NTuple{D}, Λ::AbstractArray{T, D}, F::AbstractVector, dims::NTuple{D, Int}) where {T, D}
    # Host `F`: `Q`, `Λ` are used as they are. Device `F`: `Q_d`, `Q_d'` (materialised, so
    # the device matmul never sees a lazy `Transpose`) and `Λ` are copied to the device once
    # per call, and every step below stays device-resident.
    Xv = similar(F, T, length(F))
    Xv .= F
    X = reshape(Xv, dims)
    for d in 1:D
        Qt = Xv isa Array ? transpose(Q[d]) : _fdm_to_storage(Xv, Matrix(transpose(Q[d])))
        X = _fdm_apply_mode(X, Qt, d)
    end
    Y = X ./ _fdm_to_storage(Xv, Λ)
    for d in 1:D
        Y = _fdm_apply_mode(Y, _fdm_to_storage(Xv, Q[d]), d)
    end
    return vec(Y)
end

function _fdm_solve_core(K::KroneckerLinearOperator{T, D}, F::AbstractVector, dirichlet) where {T, D}
    length(F) == K.n || _throw_fdm_length_mismatch(K.n, length(F))
    f = _fdm_factorize(K, dirichlet)
    F isa Array && return ldiv!(Vector{T}(undef, K.n), f, F)

    # Device `F`: the same factorisation, applied with device mode products.
    dims_full = K.dims
    dims_solve = size(f.Λ)
    prod(dims_solve) == 0 && return fill!(similar(F, T, K.n), zero(T))
    if dirichlet === :boundary
        rng = ntuple(d -> 2:(dims_full[d] - 1), Val(D))
        # A view plus broadcast, not `getindex` with ranges: no scalar indexing on a
        # device `F`.
        Fint = similar(F, T, prod(dims_solve))
        reshape(Fint, dims_solve) .= view(reshape(F, dims_full), rng...)
    else
        Fint = F
    end
    xint = _fdm_apply(f.Q, f.Λ, Fint, dims_solve)

    dirichlet === nothing && return xint

    x = fill!(similar(xint, T, K.n), zero(T))
    view(reshape(x, dims_full), rng...) .= reshape(xint, dims_solve)
    return x
end

"""
    fdm_solve(a::BilinearForm, F::AbstractVector; dirichlet = nothing) -> Vector

Solve `assemble(a) \\ F` (or, with `dirichlet = :boundary`, `assemble(a; dirichlet =
:boundary) \\ F` for an `F` that is already zero on the boundary) by fast diagonalisation
instead of a general sparse factorisation -- see the derivation comment above `fdm_solve`
in `ext/BrambleKroneckerExt.jl`.

`a` must be Laplacian-like: [`is_separable`](@ref) on a scalar space, with its
[`kronecker_operator`](@ref) a sum `Σ_d M_D ⊗ ... ⊗ A_d ⊗ ... ⊗ M_1 + c (M_D ⊗ ... ⊗ M_1)`
of one symmetric positive definite mass `M_d` and one symmetric `A_d` per axis, every axis
carrying a term of its own. Every term may differ from the masses on one axis at most. So
`innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))` and its forward, centered, averaged, jump and chained
variants are accepted, as are `inner_Γ` boundary terms (Robin conditions) and grid-function
coefficients that vary along one axis and keep one mass per axis. A coefficient is read
when `fdm_solve` is called, a `Ref` one included.

`dirichlet`:

  - `nothing` (the default): the unconstrained system.
  - `:boundary`: homogeneous Dirichlet on the whole mesh boundary. `F` must already carry
    zero at every boundary dof (matching what `assemble(a; dirichlet = :boundary) \\ F`
    would require of its own right-hand side); the interior solve is embedded back with
    zero on the boundary. A mesh with a 2-point axis has no interior, so the solution is
    zero.

# Throws

  - `ArgumentError` saying `fdm_solve` does not support the form, with the reason, for
    every form that is not Laplacian-like: `a` is not separable, it is posed on a composite
    space, a term has non-mass factors on two axes (a mixed derivative, or a coefficient
    varying along two axes), a mass is not symmetric positive definite, an axis has no term
    of its own, a term is not symmetric (advection), or the system is singular (a
    pure-Neumann operator with no mass term). Solve a separable one with
    `kronecker_operator(a)` and a Krylov solver.
  - `ArgumentError`: `dirichlet` is neither `nothing` nor `:boundary`.
  - `DimensionMismatch`: `F`'s length does not match `a`'s number of degrees of freedom.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (25, 19), (false, false))
Wₕ = gridspace(Ωₕ)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
F = rand(ndofs(Wₕ))
fdm_solve(a, F) ≈ assemble(a) \\ F
```

See also: [`is_separable`](@ref), [`kronecker_operator`](@ref).
"""
function Bramble.fdm_solve(a::BilinearForm, F::AbstractVector; dirichlet = nothing)
    (Bramble.trial_space(a) isa Bramble.CompositeGridSpace ||
     Bramble.test_space(a) isa Bramble.CompositeGridSpace) && _throw_fdm_composite()
    is_separable(a) || _throw_fdm_not_separable(a)
    K = kronecker_operator(a)
    return _fdm_solve_core(K, F, dirichlet)
end

"""
    fdm_solve(K::KroneckerLinearOperator, F::AbstractVector) -> Vector

Unconstrained fast-diagonalisation solve directly from an already-built
[`KroneckerLinearOperator`](@ref) (see [`kronecker_operator`](@ref)), reusing its own
factors rather than rebuilding them from a `BilinearForm`. `K` carries no boundary
constraint of its own (see its docstring), so only the unconstrained case is available
here; call the `BilinearForm` method with `dirichlet = :boundary` for homogeneous
Dirichlet. `K` must be Laplacian-like, as that method describes.

# Throws

  - `ArgumentError` saying `fdm_solve` does not support the operator, with the reason,
    when `K` is not Laplacian-like, or is a composite space's block operator.
  - `DimensionMismatch`: `F`'s length does not match `K`'s size.
  - `ArgumentError` naming `change_points!`: `K`'s mesh was mutated in place after `K` was
    built (see [`KroneckerLinearOperator`](@ref)); build the operator again.

See also: [`fdm_solve(::BilinearForm, ::AbstractVector)`](@ref).
"""
function Bramble.fdm_solve(K::KroneckerLinearOperator, F::AbstractVector)
    Bramble._kron_check_fresh(K)
    return _fdm_solve_core(K, F, nothing)
end

# A composite space's operator is a block of Kronecker sums, not one: refused by name.
Bramble.fdm_solve(::Bramble.KroneckerBlockOperator, ::AbstractVector) = _throw_fdm_composite()

# Warms this extension's entry points -- `Kronecker.kronecker` and both `fdm_solve` calls
# (unconstrained and `dirichlet = :boundary`) -- on a 2D separable, constant-coefficient
# form, only reachable once `Kronecker` is loaded so only this extension's own precompile
# pass reaches them.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0), :boundary =>
            boundary_symbols(interval(0.0, 1.0) × interval(0.0, 1.0)))
        Ωₕ = mesh(Ω, (8, 8), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        fₕ = Rₕ(Wₕ, x -> 1.0)
        l = form(Wₕ, v -> innerₕ(fₕ, v))
        F = assemble(l)

        @compile_workload begin
            K = kronecker_operator(a)
            Kronecker.kronecker(K)
            Bramble.fdm_solve(a, F)
            Bramble.fdm_solve(a, F; dirichlet = :boundary)
        end
    end
end

end # module BrambleKroneckerExt
