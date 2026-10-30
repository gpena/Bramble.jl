# pde_solve.jl
#
# `pde_solve`: the solve entry point, a plain `A \ F` unless `solver` or the macOS
# Accelerate route says otherwise, and the one function a reverse-mode adjoint rule attaches to.
#
# No source-level AD tool -- forward or reverse, `ForwardDiff`, `ReverseDiff`, `Mooncake`,
# `Enzyme` -- can differentiate through `\` itself: it dispatches into compiled
# BLAS/SuiteSparse code, opaque to every one of them, the same reason none can differentiate
# through any other library call written in C or Fortran. Everything *around* the solve --
# `assemble`, `dirichlet_bc!` -- is already reverse-mode-differentiable (`docs/notes/internals/
# autodiff.md`), so this one function is the entire gap between "Bramble can be forward-mode
# differentiated" (true today) and "Bramble can be used inside a gradient-based inverse
# problem or a PDE-constrained training loop without paying one factorisation per parameter"
# (what closes gpena/Bramble.jl#228).
#
# `pde_solve` itself needs no weak dependency: without `ChainRulesCore` it is the same solve,
# just not differentiable. `BrambleChainRulesExt` supplies the
# `ChainRulesCore.rrule`, and `BrambleEnzymeExt` a native `EnzymeRules` rule -- Enzyme gets
# its own rather than a bridged one because `Enzyme.@import_rrule`'s bridge corrupts the
# shadow of a sparse `A` whenever the cotangent carries an explicit zero
# (gpena/Bramble.jl#240, and this file's docstring below). `Mooncake` cannot currently be bridged
# this way at all: `Mooncake.@from_rrule`/`build_rrule` both accept the bridge without
# complaint, but the first time the rule actually *runs* it needs a real tangent value for
# the `SparseMatrixCSC` argument, and `Mooncake` has no `increment_and_get_rdata!` for that
# type combination (verified directly, not assumed) -- a real gap in `Mooncake.jl` itself,
# not something this package can route around, so it is left unsupported and documented
# rather than worked around.

# `:default` takes `accelerate_solve` on macOS only for a symmetric `A`: Accelerate loses on
# unsymmetric systems. An explicit `sym` hint is trusted unchecked, and `:unsymmetric` is the
# one that routes away.
function _default_wants_accelerate(A, sym)
    sym === :spd || sym === :definite || sym == 1 ||
        sym === :symmetric || sym == 2 ||
        (sym === :auto && issymmetric(A))
end

# Accelerate factors `Float32`, `Float64` and their complex types only. `:default` sends it
# nothing else: an integer or `BigFloat` system goes to the next route (`A \ F` or `lu`),
# which converts an integer one to `Float64` and refuses a type it cannot factor. `T` is the
# matrix's element type and `S` the solution's.
_accelerate_eltype(::Type{T}) where {T} = T <: Union{Float32, Float64, ComplexF32, ComplexF64}
function _default_accelerate_eltypes(::Type{T}, ::Type{S}) where {T, S}
    return _accelerate_eltype(T) && _accelerate_eltype(S)
end

"""
    pde_solve(A::SparseMatrixCSC, F::AbstractVector; solver = :default, sym = :auto, mean_zero = nothing, gauge = nothing, kwargs...) -> Vector
    pde_solve(fact::Factorization, F::AbstractVector) -> Vector

Solve `A u = F` (or `fact u = F`) and return `u`.

With no keyword arguments this is `A \\ F`, unless `AppleAccelerate.jl` is loaded on macOS
*and* the system is symmetric (see `:default` below), in which case it is
`accelerate_solve(A, F; sym)`. The name exists so that reverse-mode AD tools have one function
to attach an adjoint rule to, which is what makes a whole `θ -> assemble -> pde_solve -> J(u)`
chain differentiable.

# Keywords
- `solver`: `:default` -- `A \\ F` on Linux, Windows, macOS without `AppleAccelerate.jl`
  loaded, and macOS on an unsymmetric `A`; on macOS with `using AppleAccelerate` in effect and
  `A` symmetric, dispatches to `accelerate_solve` instead (same as passing
  `solver = :accelerate` explicitly). Symmetry is `issymmetric(A)` under `sym = :auto` (the
  default), or trusted outright from an explicit `sym = :spd`/`:definite`/`:symmetric` (and
  conversely `:unsymmetric` skips straight to `A \\ F`) -- the caller has already asserted the
  property, so it is not checked again. The narrowing exists because Accelerate is only a win
  on the symmetric factorisations it reaches (SPD/Cholesky, LDLᵀ). On forms assembled on 2D
  grid spaces it won on a symmetric Poisson-plus-mass system and lost on an unsymmetric
  convection-diffusion one. This only ever narrows which solver runs on macOS; Linux and
  Windows are never affected, and a matrix Accelerate cannot factor (e.g. non-square) throws
  from `accelerate_solve` exactly as `solver = :accelerate` would -- `:default` never silently
  falls back to `\\` after picking Accelerate. It picks Accelerate only for `Float32`,
  `Float64` and their complex types (the element type of `A` and of the solution): an integer
  `A` goes to `A \\ F`, or in the `mean_zero`/`gauge` solve to `lu`, which converts it to
  `Float64`, and a type that backend cannot factor (`BigFloat`) throws an `ArgumentError`
  naming it. `:suitesparse` (CHOLMOD/UMFPACK), `:spqr`
  (sparse QR, for a least-squares or rectangular `A`), `:accelerate` (Apple `libSparse`, needs
  `AppleAccelerate.jl`, honoured unconditionally regardless of symmetry), `:mumps` (needs
  `MUMPS.jl`), `:sparspak` (pure Julia, needs `Sparspak.jl`).
- `kwargs`: passed to the chosen backend (for `:accelerate` and `:mumps` the keywords of
  [`accelerate_solve`](@ref) and [`mumps_solve`](@ref)), which throws on one it does not
  take. `:default` takes only `kind` and `:sparspak` takes none. Both throw an
  `ArgumentError` on any other keyword. `kind` and `sym` are read only when the Accelerate
  route runs. On `A \\ F` they are accepted and ignored, so `kind = :bogus` throws only where
  Accelerate is taken.
- `sym`: symmetry hint for `:suitesparse`, `:accelerate` and `:mumps` (and, on macOS, for
  `:default`'s own choice of solver -- see above). `:auto` (default) detects it;
  `:spd`/`:definite`/`1`, `:symmetric`/`2` and `:unsymmetric`/`0` state it.
- `mean_zero`: a weight vector `w` (for instance the assembled `innerₕ(1, v)`), or `nothing`
  (default). With `w`, `pde_solve` solves the bordered system `[A w; wᵀ 0] [u; λ] = [F; 0]`
  and returns `u`, the solution with discrete zero average ``w^T u = 0`` of a pure-Neumann
  system whose `A` annihilates constants, of the promoted float type of `A`, `F` and `w`;
  `A` is not modified. The bordered matrix is indefinite, so `sym` is not used. For a real
  symmetric `A` and real `w`, `:default` factors it by LU (or by Accelerate's pivoting
  ``LDL^T``) and `:accelerate` and `:mumps` by ``LDL^T``; for any other `A` they use LU, as
  do `:suitesparse` (UMFPACK) and `:sparspak`. `:spqr` throws an `ArgumentError`, and a `w`,
  `F` or `A` of mismatched size a `DimensionMismatch`. Without `mean_zero` or `gauge`, a
  square `A` with `‖A 1‖∞ ≤ 1e-10 ‖A‖∞` (constants in its kernel) logs a warning naming
  `mean_zero`, and the solve proceeds.
- `gauge`: a matrix `Z` with one column per kernel vector to fix (for instance
  `Bramble.ghost_kernel(A, Wₕ)`, the ghost-only kernel of a form such as
  `innerₕ(Δₕ(u), Δₕ(v))` on a ghost space), or `nothing` (default). With `Z`, `pde_solve`
  solves `[A Z; Zᵀ 0] [u; μ] = [F; 0]` and returns `u`, the solution with ``Z^T u = 0``,
  which makes its component in the kernel deterministic. With `mean_zero = w` too, the
  border is `[w Z]`. The routing, the element type and the refusals are those of
  `mean_zero`, a `Z` whose row count is not that of `A` being a `DimensionMismatch`, except
  that `:sparspak`, which factors without pivoting, throws an `ArgumentError`. A complex `A`
  or `Z` is accepted; the border is transposed, not conjugated. `Z` is used as given, and
  the values in the kernel are deterministic only when its columns span the whole kernel.
  A `Z` that spans only part of it is not detected: the factorisation usually proceeds (a
  `SingularException` is a matter of rounding), the physical part is still the solution of
  the quotient system, and the kernel components `Z` leaves out come back arbitrary and
  different from one backend to another. The solution is only as accurate as the
  conditioning of `A` away from its kernel allows: on the forms that produce a ghost-only
  kernel, near `1e-8` relative.

# Returns
- `Vector`: the solution, of the promoted element type of `A` and `F`.

A `SparseMatrixCSR` (`SparseMatricesCSR.jl`) `A` is also accepted: converted to
`SparseMatrixCSC` first (see `docs/src/internals/csr_solvers.md`), then solved exactly as
above.

# Reverse-mode differentiation

With `ChainRulesCore.jl` loaded, `BrambleChainRulesExt`'s `rrule` solves the adjoint system
`Aᴴ λ = ∂J/∂u` once and returns `∂J/∂A = -λ u'` (conjugate transpose, the ChainRules
convention for a complex `A`; `-λ uᵀ` for real data) restricted to `A`'s sparsity (never
densified) and `∂J/∂F = λ`. Without keywords the adjoint solve reuses the forward solve's own
factorisation. The rule accepts the keywords of `pde_solve`: it forwards them to the primal
solve and to the adjoint solve of `sparse(A')`, so `solver` and `sym` apply to `Aᴴ` as well,
and a `sym` hint must hold for `Aᴴ` too. A `SparseMatrixCSR` `A` has no rule (ChainRules sees
a method with no `rrule`, and differentiates the conversion and the solver's internals).
`using Enzyme` is enough for the same adjoint through `BrambleEnzymeExt`'s native
`EnzymeRules` rule, which takes no keywords (it cannot run before v4.5.0); do not
call `Enzyme.@import_rrule`, whose bridge drops the cotangent's explicit zeros from `nzval`
and returns a wrong gradient without any error. The Enzyme rule keeps the unconjugated
`-λ uᵀ` and is not validated for complex `A`, whose convention differs. `Mooncake` cannot
represent a `SparseMatrixCSC` cotangent at all and is unsupported.

A closure capturing a grid space or a form needs `Enzyme.Const(f)` and
`Enzyme.set_runtime_activity(Enzyme.Reverse)`, as the automatic
differentiation tutorial describes for every other path. A gradient with
respect to a *runtime* scalar coefficient of the form should be a `Float64` or a `Ref`: an
`Integer` coefficient known only at run time costs `form` its inferred return type, which
Enzyme's type analysis rejects.

# Examples

```julia
A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
u = pde_solve(A, F)

using SuiteSparse
u_spd = pde_solve(A, F; solver = :suitesparse, sym = :spd)

# pure Neumann: fix the constant by a discrete zero average
w = assemble(form(W, v -> innerₕ(x -> 1.0, v)))
u0 = pde_solve(A_neumann, F_neumann; mean_zero = w)

# a ghost-only kernel: fix the ghost values no row of A sees
Wg = gridspace(Ωₕ; ghosts = :boundary)
Ag = assemble(form(Wg, Wg, (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)))
ug = pde_solve(Ag, Fg; gauge = Bramble.ghost_kernel(Ag, Wg))
```

For the `Bramble.RankOneUpdate` that a form with a rank-one term assembles to, see
`pde_solve(K::Bramble.RankOneUpdate, F)` below.

See also [`assemble`](@ref), [`sparse_factorize`](@ref), [`suitesparse_solve`](@ref),
[`suitesparse_qr_solve`](@ref), [`accelerate_solve`](@ref), [`mumps_solve`](@ref),
[`sparspak_solve`](@ref), [`linear_problem`](@ref).
"""
function pde_solve(A::SparseMatrixCSC, F::AbstractVector; solver::Symbol = :default, sym = :auto,
        mean_zero::Union{Nothing, AbstractVector} = nothing,
        gauge::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    (mean_zero === nothing && gauge === nothing) ||
        return _bordered_solve(A, F, nothing, mean_zero, gauge, solver, kwargs)
    _warn_if_annihilates_constants(A)
    if solver === :default
        _reject_unknown_kwargs(:pde_solve, "solver = :default", kwargs, (:kind,))
        if Sys.isapple() && Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing &&
           _default_accelerate_eltypes(eltype(A), float(promote_type(eltype(A), eltype(F)))) &&
           _default_wants_accelerate(A, sym)
            return accelerate_solve(A, F; sym = sym, kwargs...)
        end
        return A \ F
    elseif solver === :suitesparse
        return suitesparse_solve(A, F; sym = sym, kwargs...)
    elseif solver === :spqr
        return suitesparse_qr_solve(A, F; kwargs...)
    elseif solver === :accelerate
        return accelerate_solve(A, F; sym = sym, kwargs...)
    elseif solver === :mumps
        return mumps_solve(A, F; sym = sym, kwargs...)
    elseif solver === :sparspak
        _reject_unknown_kwargs(:pde_solve, "solver = :sparspak", kwargs)
        return sparspak_solve(A, F)
    else
        throw(
            ArgumentError(
            "Unknown solver: $solver. Expected :default, :suitesparse, :spqr, :accelerate, :mumps, or :sparspak.",
        ),
        )
    end
end

pde_solve(fact::Factorization, F::AbstractVector) = fact \ F

# A pure-Neumann matrix sends the constant vector to zero, so `A \ F` is singular: one product
# `A 1` detects it, and the warning names the keyword that fixes the constant. Not an error,
# since a singular consistent system is legitimate for some solvers.
function _warn_if_annihilates_constants(A::SparseMatrixCSC)
    (size(A, 1) == size(A, 2) && !isempty(A)) || return nothing
    if norm(A * ones(eltype(A), size(A, 2)), Inf) <= 1e-10 * norm(A, Inf)
        @warn "pde_solve: A annihilates constants (‖A 1‖∞ ≤ 1e-10 ‖A‖∞), so the system is " *
              "singular, as for pure Neumann conditions; pass `mean_zero = w` to fix the " *
              "constant by wᵀu = 0."
    end
    return nothing
end

# `[A C; Cᵀ 0]` is symmetric only for a real symmetric `A` and a real border `C` (a
# ghost-point Neumann `A` is not symmetric); otherwise a symmetric route would read one
# triangle, or refuse.
function _bordered_sym(A::SparseMatrixCSC, C::AbstractVecOrMat)
    return eltype(A) <: Real && eltype(C) <: Real && issymmetric(A) ? :symmetric : :unsymmetric
end

# The border's columns: `w` first, then those of `Z`; an absent keyword gives none.
_border_eltype(::Nothing) = Union{}
_border_eltype(C::AbstractVecOrMat) = eltype(C)
_border_columns(::Type{T}, n, ::Nothing) where {T} = spzeros(T, n, 0)
_border_columns(::Type{T}, n, w::AbstractVector) where {T} = SparseMatrixCSC{T, Int}(reshape(w, n, 1))
_border_columns(::Type{T}, n, Z::AbstractMatrix) where {T} = SparseMatrixCSC{T, Int}(Z)

@noinline function _throw_bordered_size(A, F, w, Z)
    got = "A of size $(size(A)), F of length $(length(F))" *
          (w === nothing ? "" : ", w of length $(length(w))") *
          (Z === nothing ? "" : ", Z of size $(size(Z))")
    throw(DimensionMismatch("pde_solve(mean_zero = w, gauge = Z) needs a square A, and F, w and " *
                            "the columns of Z of its size; got $got."))
end

@noinline function _throw_bordered_spqr()
    throw(ArgumentError("pde_solve(solver = :spqr) does not accept mean_zero or gauge: the " *
                        "bordered system is square, not a least-squares problem. Use :default, " *
                        ":suitesparse, :accelerate, :mumps or :sparspak."))
end

# Sparspak factors without pivoting, and its ordering eliminates a sparse border row (two
# entries for a corner gauge column) on its zero diagonal: a `SingularException` on the
# probe's mesh. A dense `w` row is eliminated last, after fill has reached its diagonal.
@noinline function _throw_gauge_sparspak()
    throw(ArgumentError("pde_solve(solver = :sparspak) does not accept gauge: Sparspak " *
                        "factors without pivoting, and the bordered system has a zero block. " *
                        "Use :default, :suitesparse, :accelerate or :mumps."))
end

# `mean_zero = w` and `gauge = Z` solve the bordered system `[A C; Cᵀ 0] [u; μ] = [F; 0]`
# with `C = [w Z]` (either part absent when its keyword is `nothing`) and return `u`. The
# border is never added to `A` as a low-rank term, which would be dense. The bordered matrix is
# indefinite: a symmetric route that pivots (Accelerate and MUMPS `LDLᵀ`) for a real
# symmetric `A`, LU otherwise, never Cholesky; `:default` calls `lu` directly, because `\` on
# a symmetric matrix tries Cholesky first. The caller's `sym` describes `A`, not the bordered
# matrix, so it is not used. A rank-one term `r = (wₛ, wₜ, c)`, `c = −1/σ`, adds the column
# `wₛ` and the row `wₜᵀ` in front of `C`, with `c` in the corner (see the `RankOneUpdate`
# method of `pde_solve`); `r === nothing` builds the matrix above.
function _bordered_solve(A::SparseMatrixCSC, F::AbstractVector, r,
        w::Union{Nothing, AbstractVector}, Z::Union{Nothing, AbstractMatrix}, solver::Symbol,
        kwargs)
    n = size(A, 1)
    (size(A, 2) == n && length(F) == n && (w === nothing || length(w) == n) &&
     (Z === nothing || size(Z, 1) == n)) || _throw_bordered_size(A, F, w, Z)
    solver === :spqr && (r === nothing ? _throw_bordered_spqr() : _throw_rank_one_spqr())
    solver === :sparspak && Z !== nothing && _throw_gauge_sparspak()
    T = promote_type(eltype(A), _border_eltype(w), _border_eltype(Z), _rank_one_eltype(r))
    S = float(promote_type(T, eltype(F)))  # an all-`Int` system has a fractional solution
    n == 0 && return S[]
    C = hcat(_border_columns(T, n, w), _border_columns(T, n, Z))
    B, bsym = _bordered_matrix(T, A, C, r)
    k = size(B, 1) - n
    b = zeros(S, n + k)::Vector{S}  # dense: the factorisations take no sparse `F`
    b[1:n] .= F
    if solver === :default
        _reject_unknown_kwargs(:pde_solve, "solver = :default", kwargs, (:kind,))
        x = if bsym === :symmetric && Sys.isapple() &&
               Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing &&
               _default_accelerate_eltypes(T, S)
            accelerate_solve(B, b; sym = bsym, kwargs...)
        else
            lu(B) \ b
        end
    elseif solver === :suitesparse
        x = suitesparse_solve(B, b; sym = :unsymmetric, kwargs...)
    elseif solver === :accelerate
        x = accelerate_solve(B, b; sym = bsym, kwargs...)
    elseif solver === :mumps
        x = mumps_solve(B, b; sym = bsym, kwargs...)
    elseif solver === :sparspak
        _reject_unknown_kwargs(:pde_solve, "solver = :sparspak", kwargs)
        x = sparspak_solve(B, b)
    else
        throw(
            ArgumentError(
            "Unknown solver: $solver. Expected :default, :suitesparse, :accelerate, :mumps, or :sparspak.",
        ),
        )
    end
    # Copied into a `Vector{S}`: the backends return `Float64` for `Float32` data, and the
    # branches return different types, which would leave the result uninferred.
    u = Vector{S}(undef, n)
    copyto!(u, 1, x, 1, n)
    return u
end

# The bordered matrix and its symmetry: `[A C; Cᵀ 0]`, or with a rank-one term
# `[A wₛ C; wₜᵀ c 0; Cᵀ 0 0]`, symmetric when `[A wₛ C]` gives a symmetric `[A C; Cᵀ 0]` and
# `wₛ == wₜ` (`T` is then real, so `c` is too).
_rank_one_eltype(::Nothing) = Union{}
_rank_one_eltype((ws, wt, c)::Tuple) = promote_type(eltype(ws), eltype(wt), typeof(c))

_int_csc(::Type{T}, A::SparseMatrixCSC) where {T} = convert(SparseMatrixCSC{T, Int}, A)

function _bordered_matrix(::Type{T}, A::SparseMatrixCSC, C, ::Nothing) where {T}
    k = size(C, 2)
    # Asserted: `hvcat` of sparse blocks is not inferred, and an uninferred `B` dispatches
    # at run time on every backend call in `_bordered_solve`. `A` is converted first, since
    # `hvcat` keeps an index type wider than `Int` (`UInt64`).
    B = [_int_csc(T, A) C; transpose(C) spzeros(T, k, k)]::SparseMatrixCSC{T, Int}
    return B, _bordered_sym(A, C)
end

function _bordered_matrix(::Type{T}, A::SparseMatrixCSC, C, (ws, wt, c)::Tuple) where {T}
    n, k = size(C)
    Cs = hcat(_border_columns(T, n, ws), C)
    Ct = hcat(_border_columns(T, n, wt), C)
    D = spzeros(T, k + 1, k + 1)
    D[1, 1] = c
    B = [_int_csc(T, A) Cs; transpose(Ct) D]::SparseMatrixCSC{T, Int}
    return B, ws == wt ? _bordered_sym(A, Cs) : :unsymmetric
end

@noinline function _throw_rank_one_spqr()
    throw(ArgumentError("pde_solve(solver = :spqr) does not accept a RankOneUpdate: its " *
                        "bordered system is square, not a least-squares problem. Use " *
                        ":default, :suitesparse, :accelerate, :mumps or :sparspak."))
end

"""
    pde_solve(K::Bramble.RankOneUpdate, F::AbstractVector; solver = :default, sym = :auto, mean_zero = nothing, gauge = nothing, kwargs...) -> Vector

Solve `(A + σ wₛ wₜᵀ) u = F` for the `Bramble.RankOneUpdate` `K` that [`assemble`](@ref)
returns for a form holding a rank-one term `σ * innerₕ(g, u) * innerₕ(h, v)`, and return
`u`. The dense matrix is never formed: `pde_solve` solves the sparse bordered system

```math
\\begin{bmatrix} A & w_s \\\\ w_t^\\top & -1/\\sigma \\end{bmatrix}
\\begin{bmatrix} u \\\\ \\mu \\end{bmatrix} = \\begin{bmatrix} F \\\\ 0 \\end{bmatrix},
```

whose second row gives `μ = σ wₜᵀu`, so its first row is `A u + σ wₛ (wₜᵀu) = F`. This
holds for any `A`, so a pure-Neumann `A`, singular alone, is solved when the penalty fixes
its constant. For data with `1ᵀF = 0` the solution of `A + σ w wᵀ` (`wₛ = wₜ = w`, with
`1ᵀA = 0`) has `wᵀu = 0`, so in exact arithmetic it is `pde_solve(A, F; mean_zero = w)`
for every `σ ≠ 0`. In floating point a small `σ` costs accuracy, since `−1/σ` then dwarfs
`A`'s entries: with `σ = γ / hₘₐₓ` the relative difference from the `mean_zero` solve is
about `1e-14 / γ` on the meshes tested (`1e-15` for `γ ≥ 1`, `1e-8` at `γ = 1e-6`, `1e-2`
at `γ = 1e-12`). Choose `γ ≥ 1e-2` for ten digits; a large `σ` costs nothing.
`K`, `K'`, `transpose(K)` and `transpose(K')` are all accepted.

`σ` is `K.σ`, the value read when `K` was last assembled: after changing a `Ref` factor of
`σ`, call [`assemble!`](@ref)`(K, a)` before solving.

# Keywords
The keywords and backends of `pde_solve(A, F; mean_zero)`, which this solve shares, its
routing and element type included: `solver = :default` factors by LU, or by Accelerate's
``LDL^T`` when the bordered matrix is symmetric (`A` real symmetric and `wₛ == wₜ`);
`:suitesparse` and `:sparspak` by LU, `:accelerate` and `:mumps` by ``LDL^T`` or LU.
`sym` describes `A` and is not used. `:spqr` throws an `ArgumentError`. `mean_zero = w`
and `gauge = Z` add their columns to the border after `wₛ`, as they do for a matrix.

# Edge cases
- With `σ = 0` the term vanishes, and `pde_solve(K.A, F; ...)` runs with every keyword,
  so the result is that of the plain matrix (including its warning for a pure-Neumann `A`).
- A negative or complex `σ` solves the same bordered system, valid whenever it is
  nonsingular. A complex `σ` or vector makes the border complex, transposed and not
  conjugated. For a nonsingular `A` a negative `σ` with `1 + σ wₜᵀA⁻¹wₛ = 0` makes
  `A + σ wₛ wₜᵀ` singular; the factorisation does not notice, and the solution comes back
  huge without an error. A pure-Neumann `A` with `1ᵀwₛ` and `1ᵀwₜ` nonzero is nonsingular
  for every `σ ≠ 0`.
- With `wₛ ≠ wₜ`, a non-symmetric term, the border `[A wₛ; wₜᵀ −1/σ]` is routed as
  unsymmetric.
- A Dirichlet-modified `K` from `assemble(a; dirichlet)`, with `wₛ` zero at the Dirichlet
  rows, keeps those rows as identity rows of the bordered matrix; [`symmetrize!`](@ref)'s
  `K` solves as well.
- A singular `A` the term cannot fix: when `‖A 1‖∞ ≤ 8 ε ‖A‖∞` (constants in its
  kernel, as for pure Neumann conditions) and `|1ᵀwₜ| ≤ 8 ε ‖wₜ‖₁` (the trial
  functional does not see constants), or the same for `1ᵀA` and `wₛ`, `A + σ wₛ wₜᵀ` is
  singular and an `ArgumentError` names the fix (a functional with nonzero sum, or
  `mean_zero`). `ε` is the machine epsilon of `A`'s element type. The check runs only
  without `mean_zero` and `gauge`; any other singular case reaches the factorisation.
- A `K.A` that is a `SparseMatrixCSR`, or the adjoint or transpose of a sparse matrix, is
  converted to `SparseMatrixCSC`; any other non-sparse `K.A` throws an `ArgumentError`.

# Returns
- `Vector`: the solution, of the float type promoted from `K` and `F` (for `σ = 0`, the
  plain matrix's result type). `F` is one vector, as for a matrix.

# Examples
```julia
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + σ * innerₕ(1, u) * innerₕ(1, v))
K = assemble(a)
u = pde_solve(K, F)  # pde_solve(A, F; mean_zero = w) when sum(F) == 0
```
"""
function pde_solve(K::RankOneUpdate, F::AbstractVector; solver::Symbol = :default, sym = :auto,
        mean_zero::Union{Nothing, AbstractVector} = nothing,
        gauge::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    A = _rank_one_sparse(K.A)
    iszero(K.σ) && return pde_solve(A, F; solver = solver, sym = sym, mean_zero = mean_zero,
        gauge = gauge, kwargs...)
    mean_zero === nothing && gauge === nothing && _check_penalty_fixes_constants(A, K.ws, K.wt)
    return _bordered_solve(A, F, (K.ws, K.wt, -inv(K.σ)), mean_zero, gauge, solver, kwargs)
end

# `K'` and `transpose(K)`, LinearAlgebra's lazy wrappers of `K`:
# `(A + σ wₛ wₜᵀ)ᴴ = Aᴴ + σ̄ w̄ₜ w̄ₛᵀ`, and the transpose without conjugates, solved as the
# rank-one update they are, from the parent's fields at solve time. A `RankOneUpdate` built
# by hand around a lazily wrapped `A` is materialised by `_rank_one_sparse`.
function pde_solve(Kt::Union{Adjoint{<:Any, <:RankOneUpdate}, Transpose{<:Any, <:RankOneUpdate}},
        F::AbstractVector; kwargs...)
    K = parent(Kt)
    f = Kt isa Adjoint ? conj : identity
    A0 = _rank_one_sparse(K.A)
    A = Kt isa Adjoint ? adjoint(A0) : transpose(A0)
    return pde_solve(RankOneUpdate(A, f(K.σ), f.(K.wt), f.(K.ws)), F; kwargs...)
end

# `transpose(K')` and `(transpose(K))'`: both are `conj(K)`, entry by entry, with `wₛ` and
# `wₜ` in their places.
function pde_solve(
        Kc::Union{Transpose{<:Any, <:Adjoint{<:Any, <:RankOneUpdate}},
            Adjoint{<:Any, <:Transpose{<:Any, <:RankOneUpdate}}},
        F::AbstractVector; kwargs...)
    K = parent(parent(Kc))
    return pde_solve(
        RankOneUpdate(conj(_rank_one_sparse(K.A)), conj(K.σ), conj.(K.ws), conj.(K.wt)), F;
        kwargs...)
end

const _WrappedSparse = Union{Adjoint{<:Any, <:SparseMatrixCSC},
    Transpose{<:Any, <:SparseMatrixCSC}}

_rank_one_sparse(A::SparseMatrixCSC) = A
_rank_one_sparse(A::_WrappedSparse) = sparse(A)
_rank_one_sparse(A::AbstractMatrix) = _is_csr(A) ? _csr_to_csc(A) : _throw_rank_one_dense(A)

@noinline function _throw_rank_one_dense(@nospecialize(A))
    throw(ArgumentError("pde_solve(K::RankOneUpdate, F) solves through a sparse bordered " *
                        "system and needs K.A sparse; got a $(nameof(typeof(A))). Rebuild K " *
                        "with sparse(K.A): Bramble.RankOneUpdate(sparse(K.A), K.σ, K.ws, K.wt)."))
end

# `A + σ wₛ wₜᵀ` with constants in the right kernel of `A` (pure Neumann) is singular when
# `wₜ` does not see them, and with constants in the left kernel when `wₛ` does not. The
# factorisation would not notice: rounding leaves a pivot near `1e-16`, and the solution is
# of size `1e14`. Two products with `A` decide it, against `8 ε`: `‖A 1‖∞ / (ε ‖A‖∞)` is
# 0.2 to 0.65 for pure-Neumann matrices from 323 to 15600 unknowns in 1D, 2D and 3D, while
# `−Δ` plus a reaction of `1e-10` (nonsingular, if badly conditioned) reads 50.
function _check_penalty_fixes_constants(A::SparseMatrixCSC, ws, wt)
    n = size(A, 1)
    (size(A, 2) == n && n > 0 && length(ws) == n && length(wt) == n) || return nothing
    tol = 8 * eps(real(float(eltype(A))))
    nA = norm(A, Inf)
    o = ones(eltype(A), n)
    _sees_constants(wt, tol) || norm(A * o, Inf) > tol * nA || _throw_penalty_singular("wₜ", "u")
    _sees_constants(ws, tol) || norm(transpose(A) * o, Inf) > tol * nA ||
        _throw_penalty_singular("wₛ", "v")
    return nothing
end

_sees_constants(w, tol) = abs(sum(w)) > tol * norm(w, 1)

@noinline function _throw_penalty_singular(name::String, f::String)
    throw(ArgumentError("pde_solve(K::RankOneUpdate, F): A annihilates constants, as for " *
                        "pure Neumann conditions, and the rank-one term's $name has zero " *
                        "sum (1ᵀ$name ≈ 0), so the term does not fix them and " *
                        "A + σ wₛ wₜᵀ is singular. Penalise a functional of $f that sees " *
                        "constants, such as σ * innerₕ(1, u) * innerₕ(1, v), or solve " *
                        "pde_solve(K, F; mean_zero = w)."))
end

# `SparseMatrixCSR` support, via the same CSC-conversion fallback `sparse_factorize` uses
# (`_is_csr`/`_csr_to_csc`, src/solvers/sparse_solvers.jl -- see that file's comment for why
# `SparseMatrixCSR` cannot be named as a compile-time type here). A non-CSR, non-CSC
# `AbstractMatrix` (e.g. a dense `Matrix`) still throws a plain `MethodError`, the same as
# before this method existed.
function pde_solve(A::AbstractMatrix, F::AbstractVector; kwargs...)
    _is_csr(A) && return pde_solve(_csr_to_csc(A), F; kwargs...)
    throw(MethodError(pde_solve, (A, F)))
end
