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
#      diagonalisation, or by a generalised Schur factorisation when an axis operator is
#      not symmetric, refusing every other form -- see the derivation comments below.
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
using LinearAlgebra: LinearAlgebra, BlasInt, Diagonal, Symmetric, isposdef, issymmetric,
                     ldiv!, mul!
using LinearAlgebra.BLAS: @blasfunc, libblastrampoline
using SparseArrays: SparseMatrixCSC, dropzeros!, nonzeros, nnz, nzrange, rowvals, sparse
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
# (`_fdm_axis_data`), and failing that by equality up to a scalar multiple: a factor
# `r * R` is then read as `R` with `r` moved into the term's coefficient
# (`kronecker_operator` puts the literal of `0.5 * innerₕ(D₋ᵧ(u), v)` into that term's
# axis-1 mass factor). It picks a mass per axis
# among the factors the terms carry on that axis such that every term equals the masses on
# all axes but at most one. A term equal to
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
# A non-symmetric `A_d` (advection) takes the generalised Schur route below instead: its
# eigenvectors grow ill-conditioned as advection dominates (gpena/Bramble.jl#443).
# Every other form is refused, never solved: a term differing from every choice of masses
# on two axes (a mixed derivative, a coefficient varying along two axes), a mass that is not
# symmetric positive definite, an axis no term differs on, a composite space, or a
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
# With `precond`, it is `fdm_preconditioner` that refuses, and it points to the other
# preconditioners instead.
@noinline function _throw_fdm_unsupported(reason::AbstractString; krylov::Bool = true,
        precond::Bool = false)
    precond && throw(
        ArgumentError(
        "fdm_preconditioner does not support this form: $reason. It needs a form with a " *
        "Laplacian-like part (see its docstring); precondition this one with " *
        "`jacobi_preconditioner(a)` or `ilu_preconditioner(assemble(a))` instead.",
    ),
    )
    throw(
        ArgumentError(
        "fdm_solve does not support this form: $reason. It solves only Laplacian-like " *
        "forms (see its docstring); solve this one with $(krylov ? _FDM_KRYLOV : "")" *
        "`assemble(a)` and a sparse direct solve.",
    ),
    )
end

function _throw_fdm_not_separable(a; precond::Bool = false)
    _throw_fdm_unsupported(
        "it is not separable (see `is_separable`): a grid-function coefficient varying along " *
        "two or more axes (spell a product of one-axis coefficients `fx * (fy * u)`), a " *
        "region restriction, an interpolation, or a 1D mesh has no Kronecker factors";
        krylov = false, precond = precond)
end

function _throw_fdm_composite(; precond::Bool = false)
    _throw_fdm_unsupported(
        "it is posed on a composite space, whose `kronecker_operator` is a block operator " *
        "(`KroneckerBlockOperator`); fast diagonalisation needs one scalar Kronecker sum";
        precond = precond)
end

@noinline function _throw_fdm_bad_dirichlet(dirichlet; precond::Bool = false)
    throw(
        ArgumentError(
        "$(precond ? "fdm_preconditioner" : "fdm_solve") only supports dirichlet = nothing (unconstrained) or " *
        "dirichlet = :boundary (homogeneous Dirichlet on the whole mesh boundary); got " *
        "$(repr(dirichlet)).",
    ),
    )
end

# `name` is the argument whose length is wrong, as the caller spells it.
@noinline function _throw_fdm_length_mismatch(n::Int, m::Int, name::String = "F";
        caller::String = "fdm_solve")
    throw(DimensionMismatch("$caller: $name has length $m, the operator needs $n"))
end

# A refill that `fdm_factorize!` refused after it started writing left `f` holding no system.
@noinline function _throw_fdm_invalid()
    throw(ArgumentError("fdm_solve: the last `fdm_factorize!` of this factorisation was " *
                        "refused (a singular system, or a mass that is not symmetric " *
                        "positive definite), so it holds no system to solve; refill it " *
                        "with `fdm_factorize!` first"))
end

# The reason the furthest-reaching choice of masses failed at, by stage (see
# `_fdm_axis_data`): 1 no choice fits, 2 a mass is not SPD, 3 an axis has no term; 5 (from
# `_fdm_factorize`) the system is singular; 6 the same, judged in an eltype narrower than
# Float64, where an ill-conditioned system can look singular. With `precond` (stage 1 cannot
# occur: the preconditioner leaves two-axis terms out), the system is the Laplacian-like part.
function _throw_fdm_stage(stage::Int, d::Int; precond::Bool = false)
    stage == 1 && _throw_fdm_unsupported(
        "a term has non-mass factors on two axes (a mixed derivative, or a coefficient " *
        "varying along two axes), so no choice of one mass per axis leaves every term " *
        "differing from the masses on at most one axis"; precond = precond)
    stage == 2 && _throw_fdm_unsupported(
        "the axis-$d mass is not symmetric positive definite (a zero or negative weight, " *
        "for instance from an :interior restriction without dirichlet = :boundary)";
        precond = precond)
    stage == 3 && precond &&
        _throw_fdm_unsupported(
            "axis $d has no term of its own (every term equals the mass along it, or differs " *
            "from the masses on two or more axes and is left out), so the form has no " *
            "Laplacian-like part"; precond = true)
    stage == 3 && _throw_fdm_unsupported(
        "axis $d has no term of its own (every term equals the mass along it), so it has no " *
        "1D operator to diagonalise")
    system = precond ? "the Laplacian-like part" : "the system"
    stage == 6 && _throw_fdm_unsupported(
        "$system is singular to the rounding of an eltype narrower than Float64, so an " *
        "ill-conditioned system may only look singular: assemble in Float64, where it may " *
        "be solvable, or add a mass term"; precond = precond)
    _throw_fdm_unsupported(
        "$system is singular: its generalised eigenvalues sum to zero somewhere, for " *
        "instance a pure-Neumann operator with no mass term; add a mass term or use " *
        "dirichlet = :boundary"; precond = precond)
end

# The global singularity test on `Λ_total` (or its Schur analogue), in `T`'s arithmetic.
function _fdm_check_global(Λ, ::Type{T}, D::Int; precond::Bool = false) where {T}
    minimum(abs, Λ) <= D * eps(T) * maximum(abs, Λ) || return nothing
    return _throw_fdm_stage(eps(T) > eps(Float64) ? 6 : 5, 0; precond = precond)
end

# The structural singularity test, in Float64 on the factors (gpena/Bramble.jl#443). The
# global test alone misses the zero eigenvalue of a non-normal pencil (Neumann plus strong
# advection), which Float32 arithmetic returns far from zero. Difference operators annihilate
# constants, so axis `d` has a constant kernel (in its right or left null space) when
# `‖A_d 1‖∞ <= n_d ϵ ‖A_d‖∞` or `‖A_d' 1‖∞ <= n_d ϵ ‖A_d‖∞`, `ϵ` the rounding of the data;
# a Dirichlet restriction leaves boundary-adjacent rows whose sums are of order `‖A_d‖∞`.
# With no mass term (`c_m == 0`) and a constant kernel on every axis, the constant (or its
# left analogue) is in the kernel of the whole Kronecker sum: the system is singular.
# `ϵ` is the eps of the least precise factor: a Float64 literal coefficient makes `K` Float64,
# yet the stiffness of a Float32 mesh keeps its Float32 rounding. Mapped over the tuples of
# terms and factors, so a refill computes it without allocating.
function _fdm_data_eps(K::KroneckerLinearOperator{T}) where {T}
    e(F) = eps(real(_fdm_factor_eltype(F)))
    return foldl(max, map(t -> foldl(max, map(e, t.factors)), K.terms); init = eps(real(T)))
end
_fdm_factor_eltype(F) = eltype(F)
_fdm_factor_eltype(F::Bramble._KronDeviceDiagonal) = eltype(F.diag)
_fdm_factor_eltype(F::Bramble._KronDeviceSparse) = eltype(F.nzval)

# On the dense restricted `A_d` a factorisation keeps, in Float64 loops, so a build and a
# refill run the same test and a refill allocates nothing.
function _fdm_constant_kernel(A::Matrix, ϵ::Real)
    n = size(A, 1)
    rows, cols, norm∞ = 0.0, 0.0, 0.0
    for i in 1:n
        s, sa = 0.0, 0.0
        for j in 1:n
            a = Float64(A[i, j])
            s += a
            sa += abs(a)
        end
        rows, norm∞ = max(rows, abs(s)), max(norm∞, sa)
    end
    for j in 1:n
        s = 0.0
        for i in 1:n
            s += Float64(A[i, j])
        end
        cols = max(cols, abs(s))
    end
    bound = n * ϵ * norm∞
    return rows <= bound || cols <= bound
end

# The structural refusal above: no mass term and a constant kernel on every axis.
function _fdm_check_kernel(A::NTuple{D, Matrix}, c_m, ϵ::Real, precond::Bool) where {D}
    iszero(c_m) && all(d -> _fdm_constant_kernel(A[d], ϵ), 1:D) &&
        _throw_fdm_stage(ϵ > eps(Float64) ? 6 : 5, 0; precond = precond)
    return nothing
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

# The `r` with `A == r * R` up to rounding (`one(T)` when `A` equals `R`), or `nothing`.
function _fdm_ratio(R::SparseMatrixCSC{T}, A::SparseMatrixCSC{T}) where {T}
    _fdm_same(R, A) && return one(T)
    (R.colptr == A.colptr && R.rowval == A.rowval) || return nothing
    a, b = nonzeros(A), nonzeros(R)
    k = argmax(abs.(b))
    r = a[k] / b[k]
    err = maximum(abs(a[i] - r * b[i]) for i in eachindex(a, b))
    return err <= 8 * eps(T) * maximum(abs, a) ? r : nothing
end

_fdm_spd(M::SparseMatrixCSC) = issymmetric(M) && isposdef(Symmetric(Matrix(M)))

# Classifies `K`'s terms on the factors restricted to `rng` (see the derivation): returns the
# per-axis masses `M`, the per-axis operators `A` (coefficients folded in), the summed
# coefficient `c_m` of the terms equal to the masses everywhere and whether every `A_d` is
# symmetric, or throws the reason no choice works. A first pass matches factors exactly;
# only when no choice fits there does a second pass match them up to a scalar multiple, so
# a form the exact pass solves keeps its results bit for bit.
#
# With `precond`, when both fail, the same two passes run again splitting `K = K_L + K_R`.
# A term differing from the masses on two or more axes goes to `K_R` and is left out, so
# `M`, `A`, `c_m` describe `K_L` alone. Both split passes run, and the one leaving out fewer
# terms wins (the exact one on a tie), since the exact pass reads a one-axis term with a
# literal folded into its mass factor, `50 * innerₕ(D₋ᵧ(u), v)`, as a two-axis term. So `K_L`
# is the form without its two-axis terms, factorised as `fdm_solve` factorises that form.
# The fifth value is the recipe `fdm_factorize!` replays (`_FDMRecipe`), empty with `precond`.
function _fdm_axis_data(K::KroneckerLinearOperator{T, D}, rng::NTuple{D};
        precond::Bool = false) where {T, D}
    cs = T[]
    fs = NTuple{D, SparseMatrixCSC{T, Int}}[]
    kept = Int[]  # the index in `K.terms` of each term kept
    for (k, term) in enumerate(K.terms)
        c = T(Bramble._kron_coeff(term.scales))  # `T`: a Float32 device keeps Float32
        f = ntuple(d -> SparseMatrixCSC{T, Int}(_fdm_factor(term.factors[d], rng[d])), Val(D))
        # A zero term (a zero `Ref`, or a factor that vanishes on the interior) adds nothing.
        (iszero(c) || any(F -> nnz(F) == 0, f)) && continue
        push!(cs, c)
        push!(fs, f)
        push!(kept, k)
    end
    nterms = length(K.terms)
    exact = _fdm_classify(cs, fs, Val(D), false, false)
    exact isa Tuple{Int, Int} || return _fdm_recorded(exact, fs, kept, nterms, false)
    scaled = _fdm_classify(cs, fs, Val(D), true, false)
    scaled isa Tuple{Int, Int} || return _fdm_recorded(scaled, fs, kept, nterms, true)
    precond || return _throw_fdm_stage(max(exact, scaled)...)
    es = _fdm_classify(cs, fs, Val(D), false, true)
    ss = _fdm_classify(cs, fs, Val(D), true, true)
    es isa Tuple{Int, Int} && ss isa Tuple{Int, Int} &&
        return _throw_fdm_stage(max(es, ss)...; precond = true)
    pick = es isa Tuple{Int, Int} ? ss : ss isa Tuple{Int, Int} ? es : es[5] <= ss[5] ? es : ss
    return (pick[1:4]..., _fdm_no_recipe(T, Val(D)))
end

# One classification pass: `(M, A, c_m, symmetric, on)` (with `split`, the number of terms
# left out in place of `on`), or the furthest `(stage, axis)` reached. `on[i]` is the one
# axis term `i` differs on, 0 for a term equal to the masses everywhere.
# The candidate masses on axis `d` are the distinct axis-`d` factors, tried most frequent
# first. With `proportional`, factors are distinct only up to a scalar multiple: a class is
# represented with a nonnegative trace, so a negative multiple of a mass seen first does not
# make the mass candidate indefinite, and the ratio of a term's factor to the chosen mass
# moves into its coefficient (`kronecker_operator` puts the literal of
# `0.5 * innerₕ(D₋ᵧ(u), v)` into that term's axis-1 mass factor). On a 1-point axis every
# factor is a multiple of the mass, so there, and only in this pass, the axis may carry no
# term of its own (`A_d = 0`). Every choice that fits is exact; the first that passes the
# SPD, every-axis and symmetry checks is used, else the first that passes the first two
# (the Schur route). With `split`, every choice fits, since a term differing from the
# masses on a second axis is marked `-1` and left out of `A` and `c_m` (it belongs to
# `K_R`). The choice leaving out the fewest terms is used, a symmetric one first among
# those. Preferring symmetry alone would take a Neumann stiffness (positive definite to
# rounding) for the mass of an advection form, leaving out the advection and the true mass.
function _fdm_classify(cs::Vector{T}, fs::Vector{NTuple{D, SparseMatrixCSC{T, Int}}},
        ::Val{D}, proportional::Bool, split::Bool) where {T, D}
    same(R, F) = proportional ? _fdm_ratio(R, F) !== nothing : _fdm_same(R, F)
    reps = ntuple(_ -> SparseMatrixCSC{T, Int}[], Val(D))
    cls = [ntuple(Val(D)) do d
               k = findfirst(R -> same(R, f[d]), reps[d])
               k === nothing || return k
               neg = proportional && sum(f[d][j, j] for j in axes(f[d], 1)) < 0
               push!(reps[d], neg ? -f[d] : f[d])
               return length(reps[d])
           end
           for f in fs]
    order = ntuple(d -> sort(eachindex(reps[d]); by = k -> -count(c -> c[d] == k, cls)), Val(D))
    stage, axis = 1, 0
    fallback = nothing
    best, bestkey = nothing, (typemax(Int), true)
    for choice in Iterators.product(order...)
        M = ntuple(d -> reps[d][choice[d]], Val(D))
        on = zeros(Int, length(fs))  # the one axis term `i` differs on, 0 if none, -1 if more
        coef = copy(cs)  # `cs[i]` times the ratios of term `i`'s mass factors to `M`
        fits = true
        for i in eachindex(fs)
            for d in 1:D
                r = proportional ? _fdm_ratio(M[d], fs[i][d]) :
                    (cls[i][d] == choice[d] ? one(T) : nothing)
                if r === nothing
                    on[i] == 0 || (split ? (on[i] = -1) : (fits = false); break)
                    on[i] = d
                else
                    coef[i] *= r
                end
            end
            fits || break
        end
        fits || continue
        bad = findfirst(d -> !_fdm_spd(M[d]), 1:D)
        bad === nothing || ((stage, axis) = max((stage, axis), (2, bad)); continue)
        bad = findfirst(d -> !any(==(d), on) && !(proportional && size(M[d], 1) == 1), 1:D)
        bad === nothing || ((stage, axis) = max((stage, axis), (3, bad)); continue)
        A = ntuple(d -> sum((coef[i] * fs[i][d] for i in eachindex(fs) if on[i] == d);
                init = zero(M[d])), Val(D))
        c_m = sum((coef[i] for i in eachindex(fs) if on[i] == 0); init = zero(T))
        symmetric = all(issymmetric, A)
        if split
            key = (count(==(-1), on), !symmetric)
            key < bestkey && ((best, bestkey) = ((M, A, c_m, symmetric, key[1]), key))
            continue
        end
        symmetric && return M, A, c_m, true, on
        fallback === nothing && (fallback = (M, A, c_m, false, on))
    end
    best === nothing || return best
    fallback === nothing || return fallback
    return (stage, axis)
end

# What the classification found at build, which `fdm_factorize!(f, K)` replays instead of
# searching again. Per term of `K`, `on[i]` is -1 (dropped: a zero coefficient, or a factor
# vanishing on the interior), 0 (equal to the masses everywhere, a `c_m` term) or the one
# axis it differs on. `proportional`: the factors matched the masses up to a scalar
# multiple, not exactly. `mass[d]` is the term whose axis-`d` factor is the mass, negated
# when `neg[d]` (the proportional pass represents a class with a nonnegative trace).
# `coef` is the refill's scratch for each term's coefficient. Empty (`on == Int[]`) for an
# empty interior or a preconditioner, which are never refilled from `K`.
struct _FDMRecipe{T, D}
    on::Vector{Int}
    proportional::Bool
    mass::NTuple{D, Int}
    neg::NTuple{D, Bool}
    coef::Vector{T}
end

function _fdm_no_recipe(::Type{T}, ::Val{D}) where {T, D}
    return _FDMRecipe{T, D}(Int[], false, ntuple(_ -> 0, Val(D)), ntuple(_ -> false, Val(D)),
        T[])
end

# A pass's `(M, A, c_m, symmetric, on)` over the kept terms `fs` (`kept[i]` the index of
# `fs[i]` in `K.terms`, of which there are `nterms`), with its recipe in place of `on`. The
# mass of axis `d` is the factor of the term that opened its class, the very object, or
# that factor negated: then it is the first term not differing on `d`.
function _fdm_recorded(res, fs::Vector{NTuple{D, SparseMatrixCSC{T, Int}}},
        kept::Vector{Int}, nterms::Int, proportional::Bool) where {T, D}
    M, A, c_m, symmetric, on = res
    on_all = fill(-1, nterms)
    on_all[kept] = on
    rep(d) = something(findfirst(i -> fs[i][d] === M[d], eachindex(fs)),
        findfirst(!=(d), on))
    mass = ntuple(d -> kept[rep(d)], Val(D))
    neg = ntuple(d -> fs[rep(d)][d] !== M[d], Val(D))
    return M, A, c_m, symmetric,
    _FDMRecipe{T, D}(on_all, proportional, mass, neg, zeros(T, nterms))
end

# The LAPACK workspace of one axis's symmetric-definite eigenproblem (`xsygvd`), sized by a
# query at build so that `_fdm_decompose!` allocates nothing: `B` takes a copy of `M_d` (and
# then its Cholesky factor), `w` the eigenvalues, `info` LAPACK's status.
struct _SygvdWork{T}
    B::Matrix{T}
    w::Vector{T}
    work::Vector{T}
    iwork::Vector{BlasInt}
    info::Vector{BlasInt}
end

# `A Q = B Q Λ`, `Q' B Q = I` in place: `A` becomes `Q`, `g.w` the eigenvalues. This is the
# call `eigen(Symmetric(A), Symmetric(B))` makes (`LAPACK.sygvd!(1, 'V', 'U', A, B)`), with
# `g`'s stored buffers in place of the ones `sygvd!` allocates. Returns LAPACK's `info`.
for (sygvd, T) in ((:dsygvd_, :Float64), (:ssygvd_, :Float32))
    @eval function _sygvd!(A::Matrix{$T}, g::_SygvdWork{$T}, lwork::BlasInt, liwork::BlasInt)
        n = size(A, 1)
        ld = max(1, n)
        ccall((@blasfunc($sygvd), libblastrampoline), Cvoid,
            (Ref{BlasInt}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt},
                Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
            1, 'V', 'U', n,
            A, ld, g.B, ld,
            g.w, g.work, lwork, g.iwork,
            liwork, g.info, 1, 1)
        return g.info[1]
    end
end

# The workspace for an `n x n` axis, `A` its (unread) input. Only Float32 and Float64 reach
# the query: no LAPACK takes another eltype, which `fdm_solve` cannot factorise.
function _SygvdWork(A::Matrix{T}, query::Bool) where {T}
    n = size(A, 1)
    g = _SygvdWork{T}(Matrix{T}(undef, n, n), Vector{T}(undef, n), Vector{T}(undef, 1),
        Vector{BlasInt}(undef, 1), zeros(BlasInt, 1))
    query || return g
    # `lwork = liwork = -1` returns the optimal sizes in `work[1]` and `iwork[1]`.
    LinearAlgebra.LAPACK.chkargsok(_sygvd!(A, g, BlasInt(-1), BlasInt(-1)))
    resize!(g.work, BlasInt(g.work[1]))
    resize!(g.iwork, g.iwork[1])
    return g
end

# The factorisation `fdm_solve` applies: everything before the apply, done once
# (classification, restriction, per-axis generalised eigenproblem, `Λ_total`, singularity
# refusal), plus the workspace that lets `ldiv!` on host vectors allocate nothing. `u`/`w`
# are the two ping-pong buffers of the solved (interior, under `:boundary`) unknowns;
# `u3[d]`/`w3[d]` are the same memory viewed as `(pre, n_d, post)` for the axis-`d` mode
# product, built here because a `reshape` per application would allocate. `interior` holds
# the linear indices of the solved unknowns in the full vector (empty for
# `dirichlet = nothing`). `M[d]`, `A[d]` are the dense restricted mass and operator of axis
# `d` and `c_m` the mass coefficient, from which `_fdm_decompose!` recomputes `Q`, `Λ` (and
# `λ`, `vec(Λ)`) through the LAPACK workspaces `lapack`; `precond` words its refusal.
# `fdm_factorize!` refills `M`, `A` and `c_m` by replaying `recipe`; `valid` is false from
# the moment it starts writing them until it succeeds, and `ldiv!` refuses meanwhile.
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
    M::NTuple{D, Matrix{T}}
    A::NTuple{D, Matrix{T}}
    c_m::Base.RefValue{T}
    precond::Bool
    lapack::NTuple{D, _SygvdWork{T}}
    recipe::_FDMRecipe{T, D}
    valid::Base.RefValue{Bool}
end

function _fdm_slabs(v::Vector, dims::NTuple{D, Int}) where {D}
    ntuple(d -> reshape(v, prod(dims[1:(d - 1)]; init = 1), dims[d],
            prod(dims[(d + 1):end]; init = 1)), Val(D))
end

# `precond`: factorise the Laplacian-like part `K_L` of `K` (see `_fdm_axis_data`), and word
# every refusal for `fdm_preconditioner`.
function _fdm_factorize(K::KroneckerLinearOperator{T, D}, dirichlet;
        precond::Bool = false) where {T, D}
    (dirichlet === nothing || dirichlet === :boundary) ||
        _throw_fdm_bad_dirichlet(dirichlet; precond = precond)
    dims_full = K.dims
    boundary = dirichlet === :boundary
    rng = ntuple(d -> boundary ? (2:(dims_full[d] - 1)) : (1:dims_full[d]), Val(D))
    dims = map(length, rng)
    interior = boundary ? vec(LinearIndices(dims_full)[rng...]) : Int[]
    if prod(dims) == 0
        # A 2-point axis leaves no interior: every unknown is a Dirichlet one, and zero.
        Z = ntuple(d -> zeros(T, dims[d], dims[d]), Val(D))
        return _fdm_symmetric(K.n, boundary, interior, Z, map(copy, Z), zero(T), dims,
            precond, _fdm_no_recipe(T, Val(D)))
    end
    M, A, c_m, symmetric, recipe = _fdm_axis_data(K, rng; precond = precond)
    Md = ntuple(d -> Matrix(M[d]), Val(D))
    Ad = ntuple(d -> Matrix(A[d]), Val(D))
    # In a narrower precision, an SPD axis whose boundary rows are tiny against its largest
    # (a mesh fine in the middle, coarse at both ends) also passes for one with a constant
    # kernel: the refusal then names Float64 too.
    _fdm_check_kernel(Ad, c_m, _fdm_data_eps(K), precond)
    symmetric ||
        return _schur_factorize(K.n, boundary, interior, Md, Ad, c_m, dims, recipe;
            precond = precond)
    return _fdm_decompose!(_fdm_symmetric(K.n, boundary, interior, Md, Ad, c_m, dims,
        precond, recipe))
end

# An `_FDMFactorization` holding `M`, `A`, `c_m` and every buffer, its LAPACK workspaces
# sized (no query on an empty interior), `Q` and `Λ` zero until `_fdm_decompose!` fills them.
function _fdm_symmetric(n::Int, boundary::Bool, interior::Vector{Int},
        M::NTuple{D, Matrix{T}}, A::NTuple{D, Matrix{T}}, c_m::T, dims::NTuple{D, Int},
        precond::Bool, recipe::_FDMRecipe{T, D}) where {T, D}
    Q = map(copy, A)
    lapack = ntuple(d -> _SygvdWork(Q[d], prod(dims) > 0), Val(D))
    Λ = zeros(T, dims)
    u = Vector{T}(undef, prod(dims))
    w = similar(u)
    return _FDMFactorization{T, D}(n, boundary, interior, Q, Λ, vec(Λ), u, w,
        _fdm_slabs(u, dims), _fdm_slabs(w, dims), M, A, Ref(c_m), precond, lapack, recipe, Ref(true))
end

# Recomputes `f`'s per-axis decompositions from `f.M`, `f.A` and `f.c_m` alone, allocating
# nothing: per axis, `A_d Q_d = M_d Q_d Λ_d` with `Q_d' M_d Q_d = I` (see the derivation),
# then `Λ_total[j] = c_m + Σ_d Λ_d[j_d]`, summed in the order `fill(c_m) .+= Λ_d` takes,
# then the singularity refusal. An empty interior has nothing to decompose.
function _fdm_decompose!(f::_FDMFactorization{T, D}) where {T, D}
    isempty(f.Λ) && return f
    for d in 1:D
        g = f.lapack[d]
        copyto!(f.Q[d], f.A[d])
        copyto!(g.B, f.M[d])
        info = _sygvd!(f.Q[d], g, BlasInt(length(g.work)), BlasInt(length(g.iwork)))
        # An invalid argument; `M_d` not definite (`info > n_d`, possible only on a refill:
        # a build checks the mass first), refused as a build refuses it; else no convergence.
        LinearAlgebra.LAPACK.chkargsok(info)
        info > size(f.M[d], 1) && _throw_fdm_stage(2, d; precond = f.precond)
        info > 0 && throw(LinearAlgebra.LAPACKException(info))
    end
    Λ, c_m = f.Λ, f.c_m[]
    @inbounds for j in CartesianIndices(Λ)
        s = c_m
        for d in 1:D
            s += f.lapack[d].w[j[d]]
        end
        Λ[j] = s
    end
    # A zero (to rounding) of `Λ_total` is a zero eigenvalue of the system: dividing by it
    # would return a huge `x` that does not solve it. `Λ_total` sums `D` per-axis
    # eigenvalues, each off by about `eps(T) * maximum(abs, Λ)`, whatever the grid size.
    # Measured on graded 2D meshes, a singular system's smallest entry is below
    # `0.07 eps(T) * maximum(abs, Λ)`, and a nonsingular Float32 one with condition 2e4 sits
    # at 15 eps, so the bound is `D eps`, not more. A cancellation test per entry misses the
    # zero of a pure-Neumann operator, which is one per-axis eigenvalue.
    _fdm_check_global(Λ, T, D; precond = f.precond)
    return f
end

# One mode product `Y = R' *_d X` between the ping-pong buffers, `X` in `u` when `inu`:
# `Y[i, :, k] = X[i, :, k] * R`, a `mul!` per `pre x n_d` slab (one `mul!` on the
# `n_d x post` matricisation when `pre == 1`). Returns where the result now lives.
function _fdm_mode!(f, inu::Bool, d::Int, R::AbstractMatrix)
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
# divide by `Λ_total`, apply every `Q_d`, scatter back (zero on the boundary, or with `keep`
# `F`'s own boundary values, the identity rows `fdm_preconditioner` applies there). `2D`
# mode products, an even number, so the result always ends in `u`. `x` may alias `F`, as the
# gather reads `F` before `x` is written.
LinearAlgebra.ldiv!(x::AbstractVector, f::_FDMFactorization, F::AbstractVector) = _fdm_ldiv!(x, f, F, false)

function _fdm_ldiv!(x::AbstractVector, f::_FDMFactorization{T, D}, F::AbstractVector,
        keep::Bool) where {T, D}
    # `interior` holds 1-based positions, used under `@inbounds` below.
    Base.require_one_based_indexing(x, F)
    # With `keep`, the preconditioner's `ldiv!(y, P, x)`: `x` is its input, `y` its output.
    caller = keep ? "fdm_preconditioner" : "fdm_solve"
    length(F) == f.n ||
        _throw_fdm_length_mismatch(f.n, length(F), keep ? "x" : "F"; caller = caller)
    length(x) == f.n ||
        _throw_fdm_length_mismatch(f.n, length(x), keep ? "y" : "x"; caller = caller)
    f.valid[] || _throw_fdm_invalid()
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
        keep ? (x === F || copyto!(x, F)) : fill!(x, zero(eltype(x)))
        @inbounds for (i, j) in enumerate(f.interior)
            x[j] = u[i]
        end
    else
        copyto!(x, u)
    end
    return x
end

# --- 3. Generalised Schur factorisation --------------------------------------------- #
#
# Derivation. When the classification above finds the masses `M_d` and operators `A_d` but
# some `A_d` is not symmetric (advection), each axis takes the complex generalised Schur
# form `schur(complex(A_d), complex(M_d))`: unitary `Q_d`, `Z_d` with
#
#     Q_d' A_d Z_d = S_d,     Q_d' M_d Z_d = T_d     (S_d, T_d upper triangular)
#
# and, with `Q = Q_D ⊗ ... ⊗ Q_1` and `Z` likewise, the mixed-product rule gives
#
#     Q' K Z = Σ_d T_D ⊗ ... ⊗ S_d ⊗ ... ⊗ T_1 + c_m (T_D ⊗ ... ⊗ T_1),
#
# upper triangular in lexicographic order. Solve `(Q' K Z) y = Q' F`, then `x = Z y`.
# The complex form is triangular where the real QZ form is only quasi-triangular (2x2
# blocks for complex pairs), and every transform is unitary, so none of the eigenvector
# ill-conditioning of a non-symmetric eigendecomposition enters.
#
# The triangular solve is Bartels-Stewart back substitution along the last axis. The
# level-`k` operator is `α Σ_{d≤k} T_k ⊗ ... ⊗ S_d ⊗ ... ⊗ T_1 + β P_k`, with
# `P_k = T_k ⊗ ... ⊗ T_1` (`α = 1`, `β = c_m` at level `D`). Splitting off axis `k`,
#
#     Op_k = T_k ⊗ L + α S_k ⊗ P_{k-1},     L = α Σ_{d<k} (...) + β P_{k-1},
#
# so slab `i` (unknowns with axis-`k` index `i`, contiguous in column-major order) solves
# `(T_k[i,i] L + α S_k[i,i] P_{k-1}) y_i = r_i`: a level-`k-1` operator with
# `α' = α T_k[i,i]`, `β' = β T_k[i,i] + α S_k[i,i]`, by recursion down to a division at level
# 0. Slabs go `i = n_k` down to `1`; each solved one updates the earlier ones,
# `r_j -= T_k[j,i] L y_i + α S_k[j,i] P_{k-1} y_i`, where `P_{k-1} y_i` is a product with
# triangular factors and `L y_i = (r_i - α S_k[i,i] P_{k-1} y_i) / T_k[i,i]` (`T_k[i,i] ≠ 0`
# because `M_k` is definite) reuses the slab's right-hand side. That is O(N Σ_d n_d) per
# level, after O(n_d³) setup. The diagonal entry at `j` is `Π_d T_d[j_d, j_d]` times
# `c_m + Σ_d S_d[j_d, j_d] / T_d[j_d, j_d]` (the `Λ_total` analogue), and the latter
# zero to rounding is refused as singular, as above. `F` and `K` are real, so `x` is the
# real part of `Z y`.

# The LAPACK workspace of one axis's complex generalised Schur form (`xgges3`), sized by a
# query at build so that `_fdm_decompose!` allocates nothing: `vsl`, `vsr` receive `Q_d`,
# `Z_d`; `sdim` and `info` are LAPACK's one-element outputs.
struct _Gges3Work{R, C}
    alpha::Vector{C}
    beta::Vector{C}
    vsl::Matrix{C}
    vsr::Matrix{C}
    work::Vector{C}
    rwork::Vector{R}
    sdim::Vector{BlasInt}
    info::Vector{BlasInt}
end

# The factorisation the Schur route applies, the `_FDMFactorization` contract (`n`,
# `boundary`, `interior`, `u`/`w` and their slab views `u3`/`w3`, all complex here). `Qc[d]`
# is `conj(Q_d)` and `Zt[d]` is `transpose(Z_d)`, the right factors of the mode products
# applying `Q_d'` and `Z_d`. `stride[k]` is `n_1 ... n_{k-1}` (`stride[D + 1]` the total),
# the length of a level-`k` slab; `r[k]`, `p[k]` are that level's buffers for a slab's
# right-hand side (then `L y_i`) and `P_{k-1} y_i`. `M[d]`, `A[d]`: axis `d`'s dense
# restricted mass and operator. `_fdm_decompose!` recomputes every factor from them alone
# and `c_m`, through the LAPACK workspaces `lapack`; `Λ` serves only the singularity
# refusal. `fdm_factorize!` refills `M`, `A` and `c_m` by replaying `recipe`, `valid` as in
# `_FDMFactorization`.
struct _SchurFactorization{T, D, C <: Complex{T}}
    n::Int
    boundary::Bool
    interior::Vector{Int}
    Qc::NTuple{D, Matrix{C}}
    Zt::NTuple{D, Matrix{C}}
    S::NTuple{D, Matrix{C}}
    Tr::NTuple{D, Matrix{C}}
    c_m::Base.RefValue{C}
    dims::NTuple{D, Int}
    stride::Vector{Int}
    r::Vector{Vector{C}}
    p::Vector{Vector{C}}
    u::Vector{C}
    w::Vector{C}
    u3::NTuple{D, Array{C, 3}}
    w3::NTuple{D, Array{C, 3}}
    M::NTuple{D, Matrix{T}}
    A::NTuple{D, Matrix{T}}
    Λ::Array{C, D}
    precond::Bool
    lapack::NTuple{D, _Gges3Work{T, C}}
    recipe::_FDMRecipe{T, D}
    valid::Base.RefValue{Bool}
end

# The complex QZ comes from OpenBLAS directly, never through libblastrampoline's forwarding:
# loading AppleAccelerate forwards LAPACK to Accelerate, whose `zgges3` fails to converge on
# strongly non-normal advection pencils (LAPACKException 63) and whose `dgges3` can crash
# (gpena/Bramble.jl#443). OpenBLAS stays loaded after the forward is replaced, so its
# `zgges3`/`cgges3` are found in it by name, in `__init__` (a pointer cached at precompile
# time would be invalid). `(C_NULL, C_NULL)` means no OpenBLAS: the active LAPACK's
# `xgges3` is called instead, through libblastrampoline.
const _OPENBLAS_GGES3 = Ref((C_NULL, C_NULL))

function __init__()
    _OPENBLAS_GGES3[] = _openblas_gges3()
    return nothing
end

# `(zgges3, cgges3)` from the OpenBLAS libblastrampoline loaded at start-up, else from the
# OpenBLAS_jll LinearAlgebra loads it with (an Accelerate forward may drop it from the list).
function _openblas_gges3()
    interface = BlasInt === Int64 ? :ilp64 : :lp64
    openblas(lib) = occursin("openblas", lowercase(basename(lib.libname))) &&
                    lib.interface === interface
    libs = [(lib.libname, lib.suffix)
            for lib in filter(openblas, LinearAlgebra.BLAS.get_config().loaded_libs)]
    jll = isdefined(LinearAlgebra, :OpenBLAS_jll) ?
          getfield(LinearAlgebra, :OpenBLAS_jll) : nothing
    jll !== nothing && isdefined(jll, :libopenblas_path) &&
        push!(libs, (jll.libopenblas_path, interface === :ilp64 ? "64_" : ""))
    for (path, suffix) in libs
        h = Libc.Libdl.dlopen(path; throw_error = false)
        h === nothing && continue
        z = Libc.Libdl.dlsym(h, "zgges3_" * suffix; throw_error = false)
        c = Libc.Libdl.dlsym(h, "cgges3_" * suffix; throw_error = false)
        z !== nothing && c !== nothing && return (z, c)
    end
    return (C_NULL, C_NULL)
end

# The `xgges3` to call and whether it is OpenBLAS's: the handle `__init__` found, else the
# active LAPACK's through libblastrampoline.
for (gges3, R) in ((:zgges3_, :Float64), (:cgges3_, :Float32))
    @eval function _gges3_ptr(::Type{$R})
        fptr = _OPENBLAS_GGES3[][$(R === :Float64 ? 1 : 2)]
        fptr == C_NULL || return fptr, true
        return cglobal((@blasfunc($gges3), libblastrampoline)), false
    end
end

# `(A, B) = (Q S Z', Q T Z')` in place: `A` becomes `S`, `B` becomes `T`, `g.vsl` `Q` and
# `g.vsr` `Z`. The call `LinearAlgebra.LAPACK.gges3!` makes (so the factors are the ones
# `schur` returns under the same LAPACK), with `g`'s stored buffers. Returns LAPACK's `info`.
function _gges3!(fptr::Ptr{Cvoid}, A::Matrix{C}, B::Matrix{C}, g::_Gges3Work{R, C},
        lwork::BlasInt) where {R <: Union{Float64, Float32}, C <: Complex{R}}
    n = size(A, 1)
    ld = max(1, n)
    ccall(fptr, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
            Ref{BlasInt}, Ptr{C}, Ref{BlasInt}, Ptr{C},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{C}, Ptr{C},
            Ptr{C}, Ref{BlasInt}, Ptr{C}, Ref{BlasInt},
            Ptr{C}, Ref{BlasInt}, Ptr{R}, Ptr{Cvoid},
            Ptr{BlasInt}, Clong, Clong, Clong),
        'V', 'V', 'N', C_NULL,
        n, A, ld, B,
        ld, g.sdim, g.alpha, g.beta,
        g.vsl, ld, g.vsr, ld,
        g.work, lwork, g.rwork, C_NULL,
        g.info, 1, 1, 1)
    return g.info[1]
end

# OpenBLAS's failure is LAPACK's own exception. Any other LAPACK's failure to converge is a
# property of that LAPACK, named as such, not a raw LAPACKException.
function _gges3_check(info::BlasInt, openblas::Bool)
    (openblas || info <= 0) && return LinearAlgebra.LAPACK.chklapackerror(info)
    return _throw_gges3_failed(info)
end

@noinline function _throw_gges3_failed(info)
    # The libraries by name: `string(get_config())` can print only `LBTConfig(...)`.
    libs = join((lib.libname for lib in LinearAlgebra.BLAS.get_config().loaded_libs), ", ")
    throw(ArgumentError("fdm_solve: the generalised Schur factorisation (`xgges3`, info " *
                        "$info) failed in the active LAPACK, loaded from $libs, and no " *
                        "OpenBLAS was found to call instead"))
end

# The workspace for an `n x n` axis, `A`, `B` its (unread) inputs.
function _Gges3Work(A::Matrix{C}, B::Matrix{C}) where {R, C <: Complex{R}}
    n = size(A, 1)
    ld = max(1, n)
    g = _Gges3Work{R, C}(Vector{C}(undef, n), Vector{C}(undef, n), Matrix{C}(undef, ld, n),
        Matrix{C}(undef, ld, n), Vector{C}(undef, 1), Vector{R}(undef, 8n),
        zeros(BlasInt, 1), zeros(BlasInt, 1))
    n == 0 && return g
    # `lwork = -1` returns the optimal size in `work[1]`.
    fptr, openblas = _gges3_ptr(R)
    _gges3_check(_gges3!(fptr, A, B, g, BlasInt(-1)), openblas)
    resize!(g.work, BlasInt(real(g.work[1])))
    return g
end

function _schur_factorize(n::Int, boundary::Bool, interior::Vector{Int},
        M::NTuple{D, Matrix{T}}, A::NTuple{D, Matrix{T}}, c_m::T, dims::NTuple{D, Int},
        recipe::_FDMRecipe{T, D}; precond::Bool = false) where {T, D}
    C = Complex{T}
    S = ntuple(d -> Matrix{C}(undef, dims[d], dims[d]), Val(D))
    Tr = map(similar, S)
    lapack = ntuple(d -> _Gges3Work(S[d], Tr[d]), Val(D))
    stride = [prod(dims[1:(k - 1)]; init = 1) for k in 1:(D + 1)]
    u = Vector{C}(undef, prod(dims))
    w = similar(u)
    f = _SchurFactorization{T, D, C}(n, boundary, interior, map(similar, S),
        map(similar, S), S, Tr, Ref(C(c_m)), dims, stride,
        [Vector{C}(undef, stride[k]) for k in 1:D], [Vector{C}(undef, stride[k]) for k in 1:D],
        u, w, _fdm_slabs(u, dims), _fdm_slabs(w, dims), M, A, Array{C}(undef, dims), precond,
        lapack, recipe, Ref(true))
    return _fdm_decompose!(f)
end

# Recomputes `f`'s per-axis generalised Schur forms from `f.M` and `f.A` alone, allocating
# nothing (see the derivation), then the `Λ_total` analogue
# `c_m + Σ_d S_d[j_d, j_d] / T_d[j_d, j_d]`, summed in the order `fill(c_m) .+= ...` takes:
# zero to rounding is a singular system, the same test (and tolerance) as on the symmetric
# route. An empty interior has nothing to decompose.
function _fdm_decompose!(f::_SchurFactorization{T, D}) where {T, D}
    isempty(f.Λ) && return f
    fptr, openblas = _gges3_ptr(T)
    for d in 1:D
        g, S, Tr = f.lapack[d], f.S[d], f.Tr[d]
        copyto!(S, f.A[d])
        copyto!(Tr, f.M[d])
        _gges3_check(_gges3!(fptr, S, Tr, g, BlasInt(length(g.work))), openblas)
        Qc, Zt, n = f.Qc[d], f.Zt[d], f.dims[d]
        @inbounds for j in 1:n, i in 1:n

            Qc[i, j] = conj(g.vsl[i, j])
            Zt[j, i] = g.vsr[i, j]
        end
    end
    Λ = f.Λ
    @inbounds for j in CartesianIndices(Λ)
        s = f.c_m[]
        for d in 1:D
            s += f.S[d][j[d], j[d]] / f.Tr[d][j[d], j[d]]
        end
        Λ[j] = s
    end
    _fdm_check_global(Λ, T, D; precond = f.precond)
    return f
end

# `p = (T_k ⊗ ... ⊗ T_1) p` in place, `p` the first `stride[k + 1]` entries read as an
# `n_1 x ... x n_k` array. Along each fibre of axis `d`, `p_i = Σ_{j≥i} T_d[i,j] p_j` in
# ascending `i` reads only entries not yet overwritten.
function _schur_triangular!(p::Vector, f::_SchurFactorization, k::Int)
    @inbounds for d in 1:k
        Td = f.Tr[d]
        pre, nd = f.stride[d], f.dims[d]
        for b in 0:(f.stride[k + 1] ÷ (pre * nd) - 1), i in 1:nd, a in 1:pre
            base = a + pre * nd * b
            s = zero(eltype(p))
            for j in i:nd
                s += Td[i, j] * p[base + pre * (j - 1)]
            end
            p[base + pre * (i - 1)] = s
        end
    end
    return p
end

# Solves the level-`k` system `α Σ_{d≤k} (...) + β P_k` (derivation above) in place on
# `y[off + 1 : off + stride[k + 1]]`.
function _schur_solve!(y::Vector, f::_SchurFactorization, k::Int, off::Int, α, β)
    if k == 0
        @inbounds y[off + 1] /= β
        return y
    end
    S, Tk = f.S[k], f.Tr[k]
    m = f.stride[k]
    r, p = f.r[k], f.p[k]
    @inbounds for i in f.dims[k]:-1:1
        o = off + (i - 1) * m
        i > 1 && copyto!(r, 1, y, o + 1, m)
        _schur_solve!(y, f, k - 1, o, α * Tk[i, i], β * Tk[i, i] + α * S[i, i])
        i == 1 && break
        copyto!(p, 1, y, o + 1, m)
        _schur_triangular!(p, f, k - 1)
        a, t = α * S[i, i], inv(Tk[i, i])
        for l in 1:m
            r[l] = (r[l] - a * p[l]) * t  # now `L y_i`
        end
        for j in 1:(i - 1)
            tj, sj, oj = Tk[j, i], α * S[j, i], off + (j - 1) * m
            for l in 1:m
                y[oj + l] -= tj * r[l] + sj * p[l]
            end
        end
    end
    return y
end

Base.size(f::Union{_FDMFactorization, _SchurFactorization}) = (f.n, f.n)
# As `Base` sizes an `AbstractMatrix`: 1 past the second dimension, a `BoundsError` below 1.
Base.size(f::Union{_FDMFactorization, _SchurFactorization}, i::Integer) = i <= 2 ? size(f)[i] : 1

# `x = K \ F` on full-length host vectors, the `_FDMFactorization` method's contract: gather,
# apply every `Q_d'`, back-substitute, apply every `Z_d`, scatter the real part.
LinearAlgebra.ldiv!(x::AbstractVector, f::_SchurFactorization, F::AbstractVector) = _fdm_ldiv!(x, f, F, false)

function _fdm_ldiv!(x::AbstractVector, f::_SchurFactorization{T, D}, F::AbstractVector,
        keep::Bool) where {T, D}
    # `interior` holds 1-based positions, used under `@inbounds` below.
    Base.require_one_based_indexing(x, F)
    # With `keep`, the preconditioner's `ldiv!(y, P, x)`: `x` is its input, `y` its output.
    caller = keep ? "fdm_preconditioner" : "fdm_solve"
    length(F) == f.n ||
        _throw_fdm_length_mismatch(f.n, length(F), keep ? "x" : "F"; caller = caller)
    length(x) == f.n ||
        _throw_fdm_length_mismatch(f.n, length(x), keep ? "y" : "x"; caller = caller)
    f.valid[] || _throw_fdm_invalid()
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
            inu = _fdm_mode!(f, inu, d, f.Qc[d])
        end
        _schur_solve!(inu ? f.u : f.w, f, D, 0, one(eltype(u)), f.c_m[])
        for d in 1:D
            inu = _fdm_mode!(f, inu, d, f.Zt[d])
        end
    end
    if f.boundary
        keep ? (x === F || copyto!(x, F)) : fill!(x, zero(eltype(x)))
        @inbounds for (i, j) in enumerate(f.interior)
            x[j] = real(u[i])
        end
    else
        @inbounds for i in eachindex(x, u)
            x[i] = real(u[i])
        end
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
    # The Schur route is host-only: one copy of `F` to the host and one of `x` back.
    f isa _SchurFactorization &&
        return copyto!(similar(F, T, K.n), ldiv!(Vector{T}(undef, K.n), f, Array(F)))

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
:boundary) \\ F` for an `F` that is already zero on the boundary) from one-dimensional
factorisations instead of a general sparse factorisation -- see the derivation comments
above `fdm_solve` in `ext/BrambleKroneckerExt.jl`.

`a` must be Laplacian-like: [`is_separable`](@ref) on a scalar space, with its
[`kronecker_operator`](@ref) a sum `Σ_d M_D ⊗ ... ⊗ A_d ⊗ ... ⊗ M_1 + c (M_D ⊗ ... ⊗ M_1)`
of one symmetric positive definite mass `M_d` and one operator `A_d` per axis, every axis
carrying a term of its own. Every term may differ from the masses on one axis at most,
and a factor may be a scalar multiple of the mass. So `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))`
and its forward, centered, averaged, jump and chained variants are accepted, as are
advection terms such as `innerₕ(D₋ₓ(u), v) + 0.5 * innerₕ(D₋ᵧ(u), v)`, `inner_Γ` boundary
terms (Robin conditions) and grid-function coefficients that vary along one axis and keep
one mass per axis. A coefficient is read when `fdm_solve` is called, a `Ref` one included.

Symmetric `A_d` on every axis: fast diagonalisation, one generalised eigendecomposition
per axis. A non-symmetric one (advection): a complex generalised Schur factorisation per
axis and a triangular back substitution, so no ill-conditioned eigenvectors enter. Setup
is `O(n_d^3)` per axis and a solve `O(N Σ_d n_d)`, for `N` unknowns and `n_d` points
along axis `d`. The Schur route is host-only. A device `F` is copied to the host once and
the solution back once.

`dirichlet`:

  - `nothing` (the default): the unconstrained system.
  - `:boundary`: homogeneous Dirichlet on the whole mesh boundary. `F` must already carry
    zero at every boundary dof (matching what `assemble(a; dirichlet = :boundary) \\ F`
    would require of its own right-hand side); the interior solve is embedded back with
    zero on the boundary. A mesh with a 2-point axis has no interior, so the solution is
    zero.

A singular system is refused: one with no mass term whose 1D operators all annihilate
constants (a pure-Neumann operator, with or without advection), or one whose eigenvalue
sums vanish to rounding. A nearly singular non-symmetric form can still be solved
inaccurately, as with a sparse LU. In Float32 the refusal may come from Float32 rounding
alone; the message then says so, and the same form assembled in Float64 may be solvable.

# Throws

  - `ArgumentError` saying `fdm_solve` does not support the form, with the reason, for
    every form that is not Laplacian-like: `a` is not separable, it is posed on a composite
    space, a term has non-mass factors on two axes (a mixed derivative, or a coefficient
    varying along two axes), a mass is not symmetric positive definite, an axis has no term
    of its own, or the system is singular (see below). Solve a separable one with
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

Unconstrained solve, by the same route as the `BilinearForm` method, directly from an already-built
[`KroneckerLinearOperator`](@ref) (see [`kronecker_operator`](@ref)), reusing its own
factors rather than rebuilding them from a `BilinearForm`. `K` carries no boundary
constraint of its own (see its docstring), so only the unconstrained case is available
here; call the `BilinearForm` method with `dirichlet = :boundary` for homogeneous
Dirichlet. `K` must be Laplacian-like, as that method describes.

# Throws

  - `ArgumentError` saying `fdm_solve` does not support the operator, with the reason,
    when `K` is not Laplacian-like, is a composite space's block operator, or is singular
    (as the `BilinearForm` method describes, Float32 included).
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

# `fdm_factorize` hands out the factorisation `fdm_solve` builds and applies once;
# `fdm_solve!` is its `ldiv!`, a solve (zero on the boundary under `:boundary`).
function Bramble.fdm_factorize(a::BilinearForm; dirichlet = nothing)
    (Bramble.trial_space(a) isa Bramble.CompositeGridSpace ||
     Bramble.test_space(a) isa Bramble.CompositeGridSpace) && _throw_fdm_composite()
    is_separable(a) || _throw_fdm_not_separable(a)
    return _fdm_factorize(kronecker_operator(a), dirichlet)
end

function Bramble.fdm_factorize(K::KroneckerLinearOperator; dirichlet = nothing)
    Bramble._kron_check_fresh(K)
    return _fdm_factorize(K, dirichlet)
end

Bramble.fdm_factorize(::Bramble.KroneckerBlockOperator; dirichlet = nothing) = _throw_fdm_composite()

# --- 4. Refill ---------------------------------------------------------------------- #
#
# `fdm_factorize!(f, K)` replays the classification `f` was built with (`f.recipe`) on `K`'s
# factors instead of searching again: it reads each term's coefficient now, re-checks every
# kept term's factors against the recorded masses (within `_fdm_same`'s rounding, or up to
# the ratio `_fdm_ratio` returns), and writes `M_d`, `A_d` and `c_m` into `f`'s buffers in
# the order and arithmetic of `_fdm_classify`, so they are bitwise those of a fresh build.
# Every loop runs over a factor's stored entries, restricted to `2:end-1` under
# `:boundary`. `K.terms` is a tuple of differently typed terms, and a term's factors a
# tuple too, so terms are visited by recursion on the tuple and axes as `Val`s: every
# factor is concretely typed and nothing is allocated.

@noinline function _throw_fdm_refill(reason::AbstractString)
    throw(ArgumentError("fdm_factorize! cannot refill this factorisation: $reason; build a " *
                        "new one with `fdm_factorize`"))
end

@noinline _throw_fdm_refill_mass(d::Int) = _throw_fdm_refill(
    "a term's axis-$d factor no longer matches the mass it matched when `f` was built")

_fdm_d(::Val{d}) where {d} = d
_fdm_f(t, ::Val{d}) where {d} = t.factors[d]

# The rows of axis `d` the factorisation solves for: `2:end-1` under `:boundary`.
_fdm_rng(f, d::Int) = (o = Int(f.boundary); (1 + o):(size(f.M[d], 1) + o))

# `fn(Val(d))` for `d = 1, ..., D` in order.
@inline _fdm_foreach_axis(::F, ::Val{0}) where {F} = nothing
@inline function _fdm_foreach_axis(fn::F, ::Val{D}) where {F, D}
    _fdm_foreach_axis(fn, Val(D - 1))
    fn(Val(D))
    return nothing
end

# `fn(t, i)` for the terms `t` of a tuple, `i` counting from `i`.
@inline _fdm_foreach_term(::F, ::Tuple{}, ::Int) where {F} = nothing
@inline function _fdm_foreach_term(fn::F, terms::Tuple, i::Int) where {F}
    fn(first(terms), i)
    return _fdm_foreach_term(fn, Base.tail(terms), i + 1)
end

# `fn(t)` for the `k`-th term `t` of a tuple.
@inline function _fdm_at(fn::F, terms::Tuple, k::Int) where {F}
    k == 1 && return fn(first(terms))
    return _fdm_at(fn, Base.tail(terms), k - 1)
end
_fdm_at(::F, ::Tuple{}, k::Int) where {F} = throw(BoundsError((), k))

# Column `j` of a factor: the positions of its stored entries, and each one's row and value.
_fdm_nzr(F::SparseMatrixCSC, j::Int) = nzrange(F, j)
_fdm_nzr(::Diagonal, j::Int) = j:j
_fdm_nzr(F::AbstractMatrix, ::Int) = axes(F, 1)
_fdm_row(F::SparseMatrixCSC, ::Int, p::Int) = rowvals(F)[p]
_fdm_row(::AbstractMatrix, ::Int, p::Int) = p
_fdm_val(F::SparseMatrixCSC, ::Int, p::Int) = nonzeros(F)[p]
_fdm_val(F::Diagonal, ::Int, p::Int) = F.diag[p]
_fdm_val(F::AbstractMatrix, j::Int, p::Int) = F[p, j]

# The first position from `p` on, up to `stop`, holding a nonzero in a row of `rng`: the
# entries `_fdm_factor`'s `dropzeros!(sparse(F)[rng, rng])` keeps.
@inline function _fdm_next(F, j::Int, p::Int, stop::Int, rng::UnitRange{Int})
    while p <= stop
        (_fdm_row(F, j, p) in rng && !iszero(_fdm_val(F, j, p))) && return p
        p += 1
    end
    return p
end

# `fn(i, j, v)` for every entry `v` of `_fdm_factor(F, rng)`, at local `(i, j)`.
@inline function _fdm_foreach_entry(fn::Fn, F, rng::UnitRange{Int}) where {Fn}
    o = first(rng) - 1
    for j in rng
        r = _fdm_nzr(F, j)
        p = _fdm_next(F, j, first(r), last(r), rng)
        while p <= last(r)
            fn(_fdm_row(F, j, p) - o, j - o, _fdm_val(F, j, p))
            p = _fdm_next(F, j, p + 1, last(r), rng)
        end
    end
    return nothing
end

# `acc = op(acc, x, y, shared)` over the union of the entries of `_fdm_factor(R, rng)` (`x`,
# negated when `neg`) and `_fdm_factor(F, rng)` (`y`), as `T`, in column-major order: a
# missing entry is zero, and `shared` says both are stored.
function _fdm_merge(op::O, acc, R, neg::Bool, F, rng::UnitRange{Int}, ::Type{T}) where {O, T}
    for j in rng
        pr, qr = _fdm_nzr(R, j), _fdm_nzr(F, j)
        p = _fdm_next(R, j, first(pr), last(pr), rng)
        q = _fdm_next(F, j, first(qr), last(qr), rng)
        while p <= last(pr) || q <= last(qr)
            i = p <= last(pr) ? _fdm_row(R, j, p) : typemax(Int)
            k = q <= last(qr) ? _fdm_row(F, j, q) : typemax(Int)
            x = i <= k ? T(_fdm_val(R, j, p)) : zero(T)
            y = k <= i ? T(_fdm_val(F, j, q)) : zero(T)
            acc = op(acc, neg ? -x : x, y, i == k)
            i <= k && (p = _fdm_next(R, j, p + 1, last(pr), rng))
            k <= i && (q = _fdm_next(F, j, q + 1, last(qr), rng))
        end
    end
    return acc
end

# `_fdm_same(R, F)` on the restricted factors (`R` negated when `neg`).
function _fdm_same_entries(R, neg::Bool, F, rng::UnitRange{Int}, ::Type{T}) where {T}
    diff, rmax, fmax = _fdm_merge((0.0, 0.0, 0.0), R, neg, F, rng, T) do acc, x, y, _
        return (max(acc[1], abs(x - y)), max(acc[2], abs(x)), max(acc[3], abs(y)))
    end
    return diff <= 8 * eps(Float64) * max(rmax, fmax)
end

# `_fdm_ratio(R, F)` on the restricted factors, as `(found, r)`: the same entries, the
# ratio at the first largest `|R|`, the same tolerance.
function _fdm_ratio_entries(R, neg::Bool, F, rng::UnitRange{Int}, ::Type{T}) where {T}
    _fdm_same_entries(R, neg, F, rng, T) && return true, one(T)
    init = (true, -one(T), zero(T), zero(T), zero(T))
    same, _, rk, fk, fmax = _fdm_merge(init, R, neg, F, rng, T) do acc, x, y, shared
        s, big, b, a, m = acc
        abs(x) > big && ((big, b, a) = (abs(x), x, y))
        return (s & shared, big, b, a, max(m, abs(y)))
    end
    same || return false, zero(T)
    r = fk / rk
    err = _fdm_merge((e, x, y, _) -> max(e, abs(y - r * x)), zero(T), R, neg, F, rng, T)
    return err <= 8 * eps(T) * fmax, r
end

# The trace of `_fdm_factor(R, rng)`, whose sign `_fdm_classify` represents a class by.
function _fdm_trace(R, rng::UnitRange{Int}, ::Type{T}) where {T}
    s = zero(T)
    for j in rng, p in _fdm_nzr(R, j)

        _fdm_row(R, j, p) == j && (s += T(_fdm_val(R, j, p)))
    end
    return s
end

# Axis `d` of step one: each kept term not differing on `d` matches the mass there, and its
# coefficient takes the ratio, as `_fdm_classify` multiplies them in, axis by axis.
function _fdm_refill_ratios!(f, rc::_FDMRecipe{T}, terms::Tuple, v::Val) where {T}
    d = _fdm_d(v)
    rng = _fdm_rng(f, d)
    neg = rc.neg[d]
    _fdm_at(terms, rc.mass[d]) do tm
        R = _fdm_f(tm, v)
        rc.proportional && (_fdm_trace(R, rng, T) < 0) != neg && _throw_fdm_refill_mass(d)
        _fdm_foreach_term(terms, 1) do t, i
            (rc.on[i] < 0 || rc.on[i] == d) && return nothing
            if rc.proportional
                found, r = _fdm_ratio_entries(R, neg, _fdm_f(t, v), rng, T)
                found || _throw_fdm_refill_mass(d)
                rc.coef[i] *= r
            else
                _fdm_same_entries(R, false, _fdm_f(t, v), rng, T) || _throw_fdm_refill_mass(d)
            end
            return nothing
        end
    end
    return nothing
end

# Axis `d` of step two: `M_d` is the mass factor (negated when recorded so) and
# `A_d = Σ coef_i F_i[d]` over the terms differing on `d`, summed in term order from zero.
function _fdm_refill_axis!(f, rc::_FDMRecipe{T}, terms::Tuple, v::Val) where {T}
    d = _fdm_d(v)
    rng = _fdm_rng(f, d)
    Md, Ad = fill!(f.M[d], zero(T)), fill!(f.A[d], zero(T))
    neg = rc.neg[d]
    _fdm_at(terms, rc.mass[d]) do tm
        _fdm_foreach_entry(_fdm_f(tm, v), rng) do i, j, x
            Md[i, j] = neg ? -T(x) : T(x)
        end
    end
    _fdm_foreach_term(terms, 1) do t, k
        rc.on[k] == d || return nothing
        c = rc.coef[k]
        _fdm_foreach_entry(_fdm_f(t, v), rng) do i, j, x
            Ad[i, j] += c * T(x)
        end
    end
    return nothing
end

# The structural checks, none of which writes to `f`: `K` must have `f`'s dimension, sizes,
# eltype and number of terms, and host factors.
function _fdm_check_shape(f::Union{_FDMFactorization{T, D}, _SchurFactorization{T, D}},
        K::KroneckerLinearOperator{S, E}) where {T, D, S, E}
    E == D || _throw_fdm_refill("`K` has dimension $E, `f` was built for dimension $D")
    sizes = ntuple(d -> length(_fdm_rng(f, d)), Val(D))
    (K.n == f.n && map(n -> max(n - 2 * f.boundary, 0), K.dims) == sizes) ||
        _throw_fdm_refill("`K` has size $(K.dims), `f` was built for size " *
                          "$(f.boundary ? map(n -> n + 2, sizes) : sizes)")
    S === T || _throw_fdm_refill("`K` has eltype $S, `f` was built for eltype $T")
    return nothing
end

_fdm_on_host(F) = true
_fdm_on_host(::Union{Bramble._KronDeviceDiagonal, Bramble._KronDeviceSparse}) = false
_fdm_all_on_host(::Tuple{}) = true
_fdm_all_on_host(terms::Tuple) = all(_fdm_on_host, first(terms).factors) &&
                                 _fdm_all_on_host(Base.tail(terms))

# Whether `_fdm_factor(F, rng)` has an entry, as `_fdm_axis_data` asks of every factor.
function _fdm_has_entry(F, rng::UnitRange{Int})
    for j in rng
        r = _fdm_nzr(F, j)
        _fdm_next(F, j, first(r), last(r), rng) <= last(r) && return true
    end
    return false
end

# A term dropped at build (`on == -1`) must still be dropped: its coefficient zero, or a
# factor vanishing on the rows solved for. Otherwise the replay would leave it out.
function _fdm_check_dropped(f::Union{_FDMFactorization{T, D}, _SchurFactorization{T, D}},
        terms::Tuple) where {T, D}
    rc = f.recipe
    _fdm_foreach_term(terms, 1) do t, i
        rc.on[i] < 0 || return nothing
        iszero(T(Bramble._kron_coeff(t.scales))) && return nothing
        _fdm_all_axes(v -> _fdm_has_entry(_fdm_f(t, v), _fdm_rng(f, _fdm_d(v))), Val(D)) ||
            return nothing
        return _throw_fdm_refill_dropped(i, f isa _FDMFactorization)
    end
    return nothing
end

@noinline _throw_fdm_refill_dropped(i::Int, symmetric::Bool) = _throw_fdm_refill(
    "term $i of `K` was dropped when `f` was built (a zero coefficient, or a factor " *
    "vanishing on the interior) and is present now" *
    (symmetric ? ", so it may need the generalised Schur route, which a symmetric " *
                 "factorisation cannot take" : ""))

# Entry `(i, j)` of `A_d = Σ coef_k F_k[d]` (`v = Val(d)`), the terms `k` differing on `d`
# summed in term order from `acc` as `_fdm_refill_axis!` sums them, a zero entry skipped.
@inline _fdm_sum_entry(acc, ::Tuple{}, rc, ::Val, i::Int, j::Int, k::Int) = acc
@inline function _fdm_sum_entry(acc::T, terms::Tuple{Any, Vararg}, rc::_FDMRecipe{T},
        v::Val, i::Int, j::Int, k::Int) where {T}
    if rc.on[k] == _fdm_d(v)
        x = _fdm_f(first(terms), v)[i, j]
        iszero(x) || (acc += rc.coef[k] * T(x))
    end
    return _fdm_sum_entry(acc, Base.tail(terms), rc, v, i, j, k + 1)
end

# Whether the `A_d` the refill would write is exactly symmetric, as `issymmetric` judges it
# at build, read from the factors so that `f.A` is not written before the route is known.
function _fdm_axis_symmetric(f, rc::_FDMRecipe{T}, terms::Tuple, v::Val) where {T}
    rng = _fdm_rng(f, _fdm_d(v))
    for j in rng, i in first(rng):(j - 1)

        _fdm_sum_entry(zero(T), terms, rc, v, i, j, 1) ==
        _fdm_sum_entry(zero(T), terms, rc, v, j, i, 1) || return false
    end
    return true
end

# `fn(Val(1)) && ... && fn(Val(D))`.
@inline _fdm_all_axes(::F, ::Val{0}) where {F} = true
@inline _fdm_all_axes(fn::F, ::Val{D}) where {F, D} = _fdm_all_axes(fn, Val(D - 1)) &&
                                                      fn(Val(D))

@noinline function _throw_fdm_refill_route(symmetric::Bool)
    _throw_fdm_refill(symmetric ?
                      "an axis operator of `K` is not symmetric, which needs the " *
                      "generalised Schur route, and `f` is a symmetric factorisation" :
                      "every axis operator of `K` is symmetric, which a new factorisation " *
                      "solves by fast diagonalisation, and `f` is a generalised Schur one")
end

# A refill checks each mass as the build does (`_fdm_spd`): symmetric, and definite. On the
# symmetric route `sygvd` reports an indefinite mass but reads only the upper triangle, so
# symmetry is checked here; on the Schur route `_fdm_decompose!` needs neither, so a
# Cholesky factorisation in `Tr[d]`, which the decomposition overwrites next, checks both.
function _fdm_check_mass(f::_FDMFactorization{T, D}) where {T, D}
    for d in 1:D
        issymmetric(f.M[d]) || _throw_fdm_stage(2, d; precond = f.precond)
    end
    return nothing
end
function _fdm_check_mass(f::_SchurFactorization{T, D}) where {T, D}
    for d in 1:D
        B = copyto!(f.Tr[d], f.M[d])
        (issymmetric(f.M[d]) && _fdm_potrf!(B) == 0) ||
            _throw_fdm_stage(2, d; precond = f.precond)
    end
    return nothing
end

# LAPACK's `xpotrf` on `B`'s upper triangle, in place; returns `info` (0: definite).
for (potrf, C) in ((:zpotrf_, :ComplexF64), (:cpotrf_, :ComplexF32))
    @eval function _fdm_potrf!(B::Matrix{$C})
        n = size(B, 1)
        info = Ref{BlasInt}(0)
        ccall((@blasfunc($potrf), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{BlasInt}, Ptr{$C}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
            'U', n, B, max(1, n), info, 1)
        return info[]
    end
end

function _fdm_refill!(f::Union{_FDMFactorization{T, D}, _SchurFactorization{T, D}},
        K::KroneckerLinearOperator) where {T, D}
    rc, terms = f.recipe, K.terms
    length(terms) == length(rc.on) ||
        _throw_fdm_refill("`K` has $(length(terms)) terms, `f` was built from " *
                          "$(length(rc.on))")
    _fdm_all_on_host(terms) ||
        _throw_fdm_refill("`K` is backed by a device, and a refill runs on the host")
    _fdm_check_dropped(f, terms)
    _fdm_foreach_term(terms, 1) do t, i
        rc.on[i] < 0 || (rc.coef[i] = T(Bramble._kron_coeff(t.scales)))
        return nothing
    end
    _fdm_foreach_axis(v -> _fdm_refill_ratios!(f, rc, terms, v), Val(D))
    symmetric = _fdm_all_axes(v -> _fdm_axis_symmetric(f, rc, terms, v), Val(D))
    symmetric == (f isa _FDMFactorization) || _throw_fdm_refill_route(!symmetric)
    # Every structural check passed: from here on `f`'s buffers are overwritten, and a
    # numerical refusal leaves it invalid until a later refill succeeds.
    f.valid[] = false
    _fdm_foreach_axis(v -> _fdm_refill_axis!(f, rc, terms, v), Val(D))
    c_m = zero(T)
    for i in eachindex(rc.on)
        rc.on[i] == 0 && (c_m += rc.coef[i])
    end
    f.c_m[] = c_m
    _fdm_check_mass(f)
    _fdm_check_kernel(f.A, c_m, _fdm_data_eps(K), f.precond)
    _fdm_decompose!(f)
    f.valid[] = true
    return f
end

# `K` of any dimension and eltype, so that a mismatch is refused by name, not by dispatch.
function Bramble.fdm_factorize!(f::Union{_FDMFactorization, _SchurFactorization},
        K::KroneckerLinearOperator)
    Bramble._kron_check_fresh(K)
    _fdm_check_shape(f, K)
    # An empty interior has nothing to refill.
    isempty(f.Λ) || _fdm_refill!(f, K)
    return f
end

function Bramble.fdm_solve!(x::AbstractVector, f::Union{_FDMFactorization, _SchurFactorization},
        F::AbstractVector)
    _fdm_ldiv!(x, f, F, false)
    return x
end

# The preconditioner (`Bramble.FDMPreconditioner`, src/solvers/matrix_free_preconditioners.jl)
# holds the factorisation `_fdm_factorize` returns for `K_L`, applied with `keep`: the
# identity on the boundary rows `assemble(a; dirichlet = :boundary)` makes identity rows.
function Bramble.fdm_preconditioner(a::BilinearForm; dirichlet = nothing)
    (Bramble.trial_space(a) isa Bramble.CompositeGridSpace ||
     Bramble.test_space(a) isa Bramble.CompositeGridSpace) && _throw_fdm_composite(; precond = true)
    is_separable(a) || _throw_fdm_not_separable(a; precond = true)
    K = kronecker_operator(a)
    f = _fdm_factorize(K, dirichlet; precond = true)
    return Bramble.FDMPreconditioner{eltype(K), typeof(f)}(f)
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::Bramble.FDMPreconditioner,
        x::AbstractVector)
    return _fdm_ldiv!(y, P.factorization, x, true)
end

LinearAlgebra.ldiv!(P::Bramble.FDMPreconditioner, x::AbstractVector) = ldiv!(x, P, x)

# Warms this extension's entry points -- `Kronecker.kronecker` and both `fdm_solve` calls
# (unconstrained and `dirichlet = :boundary`) -- on a 2D separable, constant-coefficient
# form, and the Schur route on that form plus an advection term. They are only reachable
# once `Kronecker` is loaded, so only this extension's own precompile pass reaches them.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0), :boundary =>
            boundary_symbols(interval(0.0, 1.0) × interval(0.0, 1.0)))
        Ωₕ = mesh(Ω, (8, 8), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        b = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                                   innerₕ(Bramble.D₋ₓ(u), v))
        fₕ = Rₕ(Wₕ, x -> 1.0)
        l = form(Wₕ, v -> innerₕ(fₕ, v))
        F = assemble(l)

        @compile_workload begin
            K = kronecker_operator(a)
            Kronecker.kronecker(K)
            Bramble.fdm_solve(a, F)
            Bramble.fdm_solve(a, F; dirichlet = :boundary)
            Bramble.fdm_solve(b, F)
            Bramble.fdm_preconditioner(a; dirichlet = :boundary)
            Bramble.fdm_solve!(similar(F), Bramble.fdm_factorize(b), F)
        end
    end
end

end # module BrambleKroneckerExt
