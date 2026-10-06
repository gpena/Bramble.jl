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
using LinearAlgebra: LinearAlgebra, Diagonal, Symmetric, eigen, isposdef, issymmetric, ldiv!,
                     mul!, schur
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
    stage == 3 && precond && _throw_fdm_unsupported(
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
# yet the stiffness of a Float32 mesh keeps its Float32 rounding.
_fdm_data_eps(K::KroneckerLinearOperator{T}) where {T} =
    maximum(f -> eps(real(_fdm_factor_eltype(f))), (f for t in K.terms for f in t.factors);
        init = eps(real(T)))
_fdm_factor_eltype(F) = eltype(F)
_fdm_factor_eltype(F::Bramble._KronDeviceDiagonal) = eltype(F.diag)
_fdm_factor_eltype(F::Bramble._KronDeviceSparse) = eltype(F.nzval)

function _fdm_constant_kernel(A::SparseMatrixCSC, ϵ::Real)
    Aw = SparseMatrixCSC{Float64, Int}(A)
    o = ones(size(Aw, 2))
    bound = size(Aw, 1) * ϵ * maximum(abs, sum(abs, Aw; dims = 2); init = 0.0)
    return maximum(abs, Aw * o; init = 0.0) <= bound ||
           maximum(abs, transpose(Aw) * o; init = 0.0) <= bound
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
function _fdm_axis_data(K::KroneckerLinearOperator{T, D}, rng::NTuple{D};
        precond::Bool = false) where {T, D}
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
    exact = _fdm_classify(cs, fs, Val(D), false, false)
    exact isa Tuple{Int, Int} || return exact
    scaled = _fdm_classify(cs, fs, Val(D), true, false)
    scaled isa Tuple{Int, Int} || return scaled
    precond || return _throw_fdm_stage(max(exact, scaled)...)
    es = _fdm_classify(cs, fs, Val(D), false, true)
    ss = _fdm_classify(cs, fs, Val(D), true, true)
    es isa Tuple{Int, Int} && ss isa Tuple{Int, Int} &&
        return _throw_fdm_stage(max(es, ss)...; precond = true)
    pick = es isa Tuple{Int, Int} ? ss : ss isa Tuple{Int, Int} ? es : es[5] <= ss[5] ? es : ss
    return pick[1:4]
end

# One classification pass: `(M, A, c_m, symmetric)` (with `split`, the number of terms left
# out appended), or the furthest `(stage, axis)` reached.
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
        symmetric && return M, A, c_m, true
        fallback === nothing && (fallback = (M, A, c_m))
    end
    best === nothing || return best
    fallback === nothing || return (fallback..., false)
    return (stage, axis)
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
        Q = ntuple(d -> zeros(T, dims[d], dims[d]), Val(D))
        Λ = zeros(T, dims)
    else
        M, A, c_m, symmetric = _fdm_axis_data(K, rng; precond = precond)
        ϵ = _fdm_data_eps(K)
        # In a narrower precision, an SPD axis whose boundary rows are tiny against its
        # largest (a mesh fine in the middle, coarse at both ends) also passes for one with
        # a constant kernel: the refusal then names Float64 too.
        iszero(c_m) && all(d -> _fdm_constant_kernel(A[d], ϵ), 1:D) &&
            _throw_fdm_stage(ϵ > eps(Float64) ? 6 : 5, 0; precond = precond)
        symmetric ||
            return _schur_factorize(K.n, boundary, interior, M, A, c_m, dims; precond = precond)
        Q, Λ = _fdm_eigendecompose(M, A, c_m, dims)
        # A zero (to rounding) of `Λ_total` is a zero eigenvalue of the system: dividing by
        # it would return a huge `x` that does not solve it. `Λ_total` sums `D` per-axis
        # eigenvalues, each off by about `eps(T) * maximum(abs, Λ)`, whatever the grid size.
        # Measured on graded 2D meshes, a singular system's smallest entry is below
        # `0.07 eps(T) * maximum(abs, Λ)`, and a nonsingular Float32 one with condition 2e4
        # sits at 15 eps, so the bound is `D eps`, not more. A cancellation test per entry
        # misses the zero of a pure-Neumann operator, which is one per-axis eigenvalue.
        _fdm_check_global(Λ, T, D; precond = precond)
    end
    u = Vector{T}(undef, prod(dims))
    w = similar(u)
    return _FDMFactorization{T, D}(K.n, boundary, interior, Q, Λ, vec(Λ), u, w,
        _fdm_slabs(u, dims), _fdm_slabs(w, dims))
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
LinearAlgebra.ldiv!(x::AbstractVector, f::_FDMFactorization, F::AbstractVector) =
    _fdm_ldiv!(x, f, F, false)

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

# The factorisation the Schur route applies, the `_FDMFactorization` contract (`n`,
# `boundary`, `interior`, `u`/`w` and their slab views `u3`/`w3`, all complex here). `Qc[d]`
# is `conj(Q_d)` and `Zt[d]` is `transpose(Z_d)`, the right factors of the mode products
# applying `Q_d'` and `Z_d`. `stride[k]` is `n_1 ... n_{k-1}` (`stride[D + 1]` the total),
# the length of a level-`k` slab; `r[k]`, `p[k]` are that level's buffers for a slab's
# right-hand side (then `L y_i`) and `P_{k-1} y_i`.
struct _SchurFactorization{T, D, C <: Complex{T}}
    n::Int
    boundary::Bool
    interior::Vector{Int}
    Qc::NTuple{D, Matrix{C}}
    Zt::NTuple{D, Matrix{C}}
    S::NTuple{D, Matrix{C}}
    Tr::NTuple{D, Matrix{C}}
    c_m::C
    dims::NTuple{D, Int}
    stride::Vector{Int}
    r::Vector{Vector{C}}
    p::Vector{Vector{C}}
    u::Vector{C}
    w::Vector{C}
    u3::NTuple{D, Array{C, 3}}
    w3::NTuple{D, Array{C, 3}}
end

function _schur_factorize(n::Int, boundary::Bool, interior::Vector{Int}, M, A, c_m::T,
        dims::NTuple{D, Int}; precond::Bool = false) where {T, D}
    C = Complex{T}
    gs = ntuple(d -> schur(Matrix{C}(A[d]), Matrix{C}(M[d])), Val(D))
    S = ntuple(d -> gs[d].S, Val(D))
    Tr = ntuple(d -> gs[d].T, Val(D))
    # `c_m + Σ_d S_d[j_d, j_d] / T_d[j_d, j_d]`: zero to rounding is a singular system, the
    # same test (and tolerance) as `Λ_total` on the symmetric route.
    Λ = fill(C(c_m), dims)
    for d in 1:D
        shape = ntuple(k -> k == d ? dims[d] : 1, Val(D))
        Λ .+= reshape([S[d][j, j] / Tr[d][j, j] for j in 1:dims[d]], shape)
    end
    _fdm_check_global(Λ, T, D; precond = precond)
    stride = [prod(dims[1:(k - 1)]; init = 1) for k in 1:(D + 1)]
    u = Vector{C}(undef, prod(dims))
    w = similar(u)
    return _SchurFactorization{T, D, C}(n, boundary, interior,
        ntuple(d -> conj.(gs[d].Q), Val(D)), ntuple(d -> Matrix(transpose(gs[d].Z)), Val(D)),
        S, Tr, C(c_m), dims, stride, [Vector{C}(undef, stride[k]) for k in 1:D],
        [Vector{C}(undef, stride[k]) for k in 1:D], u, w, _fdm_slabs(u, dims),
        _fdm_slabs(w, dims))
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
LinearAlgebra.ldiv!(x::AbstractVector, f::_SchurFactorization, F::AbstractVector) =
    _fdm_ldiv!(x, f, F, false)

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
        _schur_solve!(inu ? f.u : f.w, f, D, 0, one(eltype(u)), f.c_m)
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
