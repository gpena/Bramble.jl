#===========================================================================#
# Fast-diagonalisation preconditioner across a refinement ladder: gpena/Bramble.jl#444, S5
# of .claude/plans/v3-26-0-separable-solvers.md.
#
# Usage:
#     julia --threads=2 --project=benchmark benchmark/fdm_preconditioner_ladder.jl
#     FDM_LADDER_QUICK=1 julia --threads=2 --project=benchmark \
#         benchmark/fdm_preconditioner_ladder.jl
#
# `fdm_preconditioner(a; dirichlet)` (`ext/BrambleKroneckerExt.jl`) applies
# the fast-diagonalisation inverse of the form's Laplacian-like part `K_L`, every term that
# differs from the per-axis masses on at most one axis, and leaves the mixed-derivative
# terms `K_R` to the Krylov solver. #444 expects the iteration count to stay bounded under
# refinement when the diffusion dominates the cross term. This script counts GMRES
# iterations (IterativeSolvers' `gmres`, right preconditioner `Pr`, so that GMRES stops on
# the true residual; not CG, the cross term is not symmetric) on
#
#     2D: inner₊ₓ(D₋ₓ u, D₋ₓ v) + κ inner₊ᵧ(D₋ᵧ u, D₋ᵧ v) + β innerₕ(D₋ₓ(D₋ᵧ(u)), v)
#     3D: the same plus κ₂ inner₊₂(D₋₂ u, D₋₂ v) + β innerₕ(D₋ᵧ(D₋₂(u)), v)
#
# on the unit square/cube whose axis `d` points are moved to `t^(1 + d/4)` (graded, no two
# axes share their nodes), for 2^k + 1 points per axis from 17² and 9³ up, twice:
#
#   - `bc=boundary`: that form with homogeneous Dirichlet (`dirichlet = :boundary`).
#     `gmg_preconditioner` takes no Dirichlet rows (`A` is `assemble(build(W))`), so GMG is
#     `NA` there. Built anyway on the form without them, which is singular (constants are
#     in its kernel), it gave a preconditioner of norm ~1e16, a "converged" preconditioned
#     residual after 1 iteration (left preconditioned) and a true relative residual of 1.
#   - `bc=none`: that form plus `innerₕ(u, v)` (non-singular), with `dirichlet = nothing`,
#     where all four preconditioners apply; GMG is the default V-cycle of
#     `gmg_preconditioner(W -> <the same form on W>, Ωₕ)`.
#
# The right-hand side is `f(x) = Π_d sin(π x_d)` at the mesh points (zero on the boundary),
# the same smooth function of position at every level and for every preconditioner. Per
# level it prints
#
#     LADDER bc=<boundary|none> dim=<D> n=<n> none=<k> jacobi=<k|NA> gmg=<k|NA> fdm=<k>
#
# where `n` is the number of points per axis and each `k` the GMRES iterations to a
# relative residual of `RELTOL`, with no preconditioner, `jacobi_preconditioner`,
# `gmg_preconditioner` and `fdm_preconditioner`. `NA` and a `NOTE` line mark a
# preconditioner that does not apply or could not be built, a run that hit `MAXITER`
# without converging, and a solve whose recomputed relative residual `‖A x - F‖ / ‖F‖`
# ends above `TRUE_TOL` (a breakdown: GMRES's own residual is the true one under `Pr`).
# GMRES restarts every `RESTART` iterations. Counts, not timing, so no power/load preflight.
# The default run also prints one markdown table per `bc` for the issue comment;
# `FDM_LADDER_QUICK=1` runs three 2D and two 3D levels and skips the tables.
#===========================================================================#

using Bramble

using Bramble: D₋ₓ, D₋ᵧ, D₋₂, inner₊ₓ, inner₊ᵧ, inner₊₂
using Kronecker: Kronecker  # loads BrambleKroneckerExt, which owns fdm_preconditioner
using IterativeSolvers: gmres
using LinearAlgebra: norm

const QUICK = get(ENV, "FDM_LADDER_QUICK", "") == "1"

# Points per axis for each level.
const LADDER = QUICK ? Dict(2 => (17, 33, 65), 3 => (9, 17)) :
               Dict(2 => (17, 33, 65, 129, 257), 3 => (9, 17, 33, 65))

const κ = 0.5    # axis-2 diffusion
const κ₂ = 0.25  # axis-3 diffusion
const β = 0.25   # cross terms
const RELTOL = 1.0e-8
const RESTART = 300
const MAXITER = 3000
const TRUE_TOL = 1.0e-7  # largest true relative residual a count is accepted at

# Uniform, then moved to `t^(1 + d/4)` along axis `d`.
function graded_mesh(n::NTuple{D, Int}) where {D}
    Ω = mesh(domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))), n,
        ntuple(_ -> false, D))
    Bramble.change_points!(Ω,
        ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D))
    return Ω
end

# The mixed form of the header on `W`, plus `innerₕ(u, v)` when `mass` is true.
ladder_form(W, mass::Bool) = ladder_form(W, Val(Bramble.dim(mesh(W))), Val(mass))
function ladder_form(W, ::Val{2}, ::Val{false})
    return form(W, W,
        (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + κ * inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v)) +
                  β * innerₕ(D₋ₓ(D₋ᵧ(u)), v))
end
function ladder_form(W, ::Val{2}, ::Val{true})
    return form(W, W,
        (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) +
                  κ * inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v)) + β * innerₕ(D₋ₓ(D₋ᵧ(u)), v))
end
function ladder_form(W, ::Val{3}, ::Val{false})
    return form(W, W,
        (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + κ * inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v)) +
                  κ₂ * inner₊₂(D₋₂(u), D₋₂(v)) + β * innerₕ(D₋ₓ(D₋ᵧ(u)), v) +
                  β * innerₕ(D₋ᵧ(D₋₂(u)), v))
end
function ladder_form(W, ::Val{3}, ::Val{true})
    return form(W, W,
        (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) +
                  κ * inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v)) + κ₂ * inner₊₂(D₋₂(u), D₋₂(v)) +
                  β * innerₕ(D₋ₓ(D₋ᵧ(u)), v) + β * innerₕ(D₋ᵧ(D₋₂(u)), v))
end

# GMRES iterations from zero with right preconditioner `Pr` (`nothing` for none), as a
# string, or `NA` with a NOTE when GMRES stops at MAXITER unconverged or the recomputed
# true relative residual is above `TRUE_TOL`.
function count_iterations(A, F, Pr, label, tag)
    kw = (; reltol = RELTOL, restart = RESTART, maxiter = MAXITER, log = true)
    x, hist = Pr === nothing ? gmres(A, F; kw...) : gmres(A, F; Pr = Pr, kw...)
    res = norm(A * x - F) / norm(F)
    if !hist.isconverged
        println("NOTE $tag $label: no convergence in $MAXITER iterations ",
            "(true relative residual $res)")
        return "NA"
    elseif res > TRUE_TOL
        println("NOTE $tag $label: GMRES stopped after ",
            "$(hist.iters) iterations, but the true relative residual is $res")
        return "NA"
    end
    return string(hist.iters)
end

# A preconditioner, or `nothing` with a NOTE when building it throws.
function try_build(f, label, tag)
    try
        return f()
    catch err
        err isa InterruptException && rethrow()
        msg = first(split(sprint(showerror, err), '\n'))
        println("NOTE $tag $label: not built: ", msg)
        return nothing
    end
end

# One row: `bc` is `:boundary` (Dirichlet, no mass) or `:none` (mass, no Dirichlet).
function measure(bc::Symbol, D::Int, k::Int)
    tag = "bc=$bc dim=$D n=$k"
    Ωₕ = graded_mesh(ntuple(_ -> k, D))
    mass = bc === :none
    dirichlet = mass ? nothing : :boundary
    a = ladder_form(gridspace(Ωₕ), mass)
    A = assemble(a; dirichlet = dirichlet)
    # `Π_d sin(π x_d)` at the points, in the column-major order of the unknowns.
    F = ones(size(A, 1))
    Fa = reshape(F, ntuple(_ -> k, D))
    for (d, p) in enumerate(points(Ωₕ))
        Fa .*= reshape(sinpi.(p), ntuple(j -> j == d ? k : 1, D))
    end
    mass || (F[Bramble._combined_mask(Ωₕ, (:boundary,))] .= 0)
    count(label, P) = P === nothing ? "NA" : count_iterations(A, F, P, label, tag)
    none = count_iterations(A, F, nothing, "none", tag)
    jac = count("jacobi",
        try_build(() -> jacobi_preconditioner(a; dirichlet = dirichlet), "jacobi", tag))
    gmg = if mass
        count("gmg",
            try_build(() -> gmg_preconditioner(W -> ladder_form(W, true), Ωₕ), "gmg", tag))
    else
        println("NOTE $tag gmg: not applicable: gmg_preconditioner takes no Dirichlet rows")
        "NA"
    end
    fdm = count("fdm",
        try_build(() -> fdm_preconditioner(a; dirichlet = dirichlet), "fdm", tag))
    return (bc = bc, dim = D, n = k, none = none, jacobi = jac, gmg = gmg, fdm = fdm)
end

function main()
    println("fdm_preconditioner refinement ladder -- gpena/Bramble.jl#444 (S5)")
    println("Threads       : ", Threads.nthreads(), " (", Sys.CPU_THREADS, " CPU cores)")
    println("Mode          : ", QUICK ? "quick (FDM_LADDER_QUICK=1)" : "default")
    println("GMRES         : reltol=$RELTOL restart=$RESTART maxiter=$MAXITER")
    println("Form          : κ=$κ κ₂=$κ₂ β=$β, graded")
    println()
    rows = NamedTuple[]
    for bc in (:boundary, :none), D in (2, 3), k in LADDER[D]
        r = measure(bc, D, k)
        push!(rows, r)
        println("LADDER bc=$(r.bc) dim=$(r.dim) n=$(r.n) none=$(r.none) ",
            "jacobi=$(r.jacobi) gmg=$(r.gmg) fdm=$(r.fdm)")
    end
    if !QUICK
        for bc in (:boundary, :none)
            println()
            println(bc === :boundary ? "bc=boundary: dirichlet = :boundary, no mass" :
                    "bc=none: innerₕ(u, v) added, dirichlet = nothing")
            println()
            println("| dim | points | none | Jacobi | GMG | FDM |")
            println("|---|---|---|---|---|---|")
            for r in rows
                r.bc === bc || continue
                println("| $(r.dim)D | $(r.n)$(r.dim == 2 ? "²" : "³") | $(r.none) | ",
                    "$(r.jacobi) | $(r.gmg) | $(r.fdm) |")
            end
        end
    end
    return rows
end

main()
