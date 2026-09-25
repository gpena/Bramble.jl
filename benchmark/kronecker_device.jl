#===========================================================================#
# Kronecker device-vs-host throughput: gpena/Bramble.jl#323, S4.3 of
# .claude/plans/v3-11-0-kronecker-hardening.md.
#
# Usage:
#     .claude/scripts/check_power_load.sh && \
#         julia --threads=4 --project=benchmark benchmark/kronecker_device.jl
#
# S4.1/S4.2 gave `KroneckerLinearOperator`'s `mul!` and `fdm_solve` a device path
# (`src/assembly/kronecker.jl`, `ext/BrambleKroneckerExt.jl`): a device-backed `K` moves its
# factors to device storage once and runs sum factorisation with no host round-trip between
# axes. This file measures whether that device path actually beats the host Kronecker path
# for the two sizes #323's own hand-rolled matrix-free CG loop measured -- 2D 3000x3000 and
# 3D 200x200x200 (no 1D case: a 1D form has nothing to factor, `is_separable` requires
# `D >= 2`) -- and reports it back-to-back as a Host/Metal ratio, not a bare absolute, per
# `bramble-benchmarks` §1 and the issue's own acceptance criterion.
#
# #323's hand-rolled figures (quoted from the issue, not reproduced here -- that loop calls
# `Δₕ!` directly, bypassing the forms API and this operator entirely):
#
#   | dim | N | Serial (min) | Metal (min) | Serial/Metal |
#   |---|---|---|---|---|
#   | 1D | 10,000,000 | 1302.9 ms | 547.5 ms | 2.38x |
#   | 2D | 3000x3000 | 1347.2 ms | 867.1 ms | 1.55x |
#   | 3D | 200x200x200 | 1325.2 ms | 1144.0 ms | 1.16x |
#
# (30 CG-shaped iterations per trial: one `Δₕ!` operator apply, two `innerₕ` reductions,
# three broadcasts, all in-place, zero host allocations on the CPU paths.)
#
# Three measurements, host and Metal back-to-back in one warmed process, both Float32 on the
# same points (the device mesh built first, the host operator taken from
# `Bramble._host_mirror_mesh(mesh(Wd))` so no independently-drawn mesh can disagree with it):
#   (a) one `mul!(y, K, x; scratch)` apply;
#   (b) 30 CG-shaped iterations -- one 5-arg `mul!(y, K, x, true, false; scratch)`, two `dot`
#       reductions, three in-place broadcasts per iteration, matching #323's own shape;
#   (c) `fdm_solve(a, F)` total, plus the host eigendecomposition's own share isolated
#       separately (see `_eig_ms` below) -- `_fdm_eigendecompose`
#       (`ext/BrambleKroneckerExt.jl`) is a host LAPACK call every `fdm_solve` invocation,
#       device-backed or not, and at 2D's 3000x3000 that O(n_d^3) dense generalised
#       eigenproblem (n_d = 3000 per axis) is not necessarily negligible next to the device
#       mode-contraction it feeds.
#
# `_eig_ms` is not literally instrumented inside `fdm_solve` (its eigendecomposition is
# `ext/BrambleKroneckerExt.jl`'s own internal `_fdm_eigendecompose`, not an exported name);
# it instead redoes the identical computation through the public API
# (`weights(Wd, Bramble.Innerh())`, `assemble(form(Wd, Wd, (u, v) -> inner₊(D₋ₓ(u),
# D₋ₓ(v))))`, `eigen(Symmetric(...), Symmetric(...))`) the same way `kronecker_operator`
# itself builds the per-axis factors -- an equivalent measurement, not a bare estimate.
#
# Uniform meshes throughout (`bramble-verification`: non-uniform is the point elsewhere, but
# this file measures device-vs-host operator throughput, not stencil correctness across mesh
# families, and a uniform mesh keeps a 3000x3000/200^3 device mesh build cheap and
# deterministic).
#===========================================================================#

using Bramble
using Bramble: D₋ₓ, weights
using Metal
using KernelAbstractions
using Kronecker: Kronecker  # loads BrambleKroneckerExt, which owns fdm_solve
using BenchmarkTools
using LinearAlgebra: mul!, dot, eigen, Symmetric, Diagonal

set_zero_subnormals(true)

# --- preflight: power + load, printed in the header (bramble-benchmarks §1) ----------- #
# `check_power_load.sh` (run before this file, per the CHECK) already refuses to proceed on
# battery or under load; this header just labels the run with what it saw, matching every
# other file in this directory.

function _power_source()
    Sys.isapple() || return (on_battery = false, raw = "unknown (not macOS)")
    try
        out = strip(read(`pmset -g batt`, String))
        return (on_battery = occursin("Battery Power", out), raw = out)
    catch
        return (on_battery = false, raw = "unknown")
    end
end

function _print_header(power)
    println("Kronecker device-vs-host throughput -- gpena/Bramble.jl#323 (S4.3)")
    println("Threads       : ", Threads.nthreads(), " (", Sys.CPU_THREADS, " CPU cores)")
    println("1-min load    : ", round(Sys.loadavg()[1]; digits = 2))
    println("Power (pmset) : ", power.raw)
    if power.on_battery
        println(
            "ON BATTERY: CPU/GPU frequency scaling and thermal throttling can skew absolute ",
            "timings. Read the Host/Metal ratio column, not the absolute ms columns."
        )
    end
    println()
end

# --- geometry: unit square/cube, Float32, uniform ------------------------------------- #

_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0f0, 1.0f0), Val(D)))

_poisson_mass(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))

# --- (c)'s host eigendecomposition share, isolated through the public API ------------- #
#
# One term per axis (`_fdm_eigendecompose`'s own derivation, `ext/BrambleKroneckerExt.jl`):
# the generalised eigenproblem `A_d Q_d = H_d Q_d Λ_d` for the assembled 1D stiffness `A_d`
# and diagonal mass `H_d`, both on axis `d`'s own gridspace over the host mirror mesh.
function _eig_ms(Ωh::Bramble.AbstractMeshType, D::Int; samples::Int)
    total_ns = 0.0
    for d in 1:D
        Wd = gridspace(Ωh(d))
        Hd = weights(Wd, Bramble.Innerh())
        Ad = assemble(form(Wd, Wd, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v))))
        Am = Matrix(Ad)
        Hm = Matrix(Diagonal(Hd))
        eigen(Symmetric(Am), Symmetric(Hm)) # warm-up
        trial = @benchmark eigen(Symmetric($Am), Symmetric($Hm)) samples=samples evals=1
        total_ns += minimum(trial.times)
    end
    return total_ns / 1e6
end

# --- (b) CG-shaped loop: matches #323's own shape (issue table above) ----------------- #
# `x0 = 0` so the initial residual is `b` itself with no `mul!` needed to seed it; each of
# the 30 iterations below then does exactly one 5-arg `mul!` with scratch, two `dot`
# reductions, and three in-place broadcasts -- the same shape the issue's hand-rolled loop
# counted for its own `Δₕ!`-based version.
function _cg30!(x, K, p, r, Ap, b, scratch)
    copyto!(r, b)
    copyto!(p, b)
    fill!(x, zero(eltype(x)))
    rsold = dot(r, r)
    for _ in 1:30
        mul!(Ap, K, p, true, false; scratch = scratch)
        α = rsold / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        rsnew = dot(r, r)
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return x
end

struct Row
    label::String
    dim::String
    host_ms::Float64
    metal_ms::Float64
    ratio::Float64
end

function _run(D::Int, n::Int, dimlabel::String, rows::Vector{Row})
    Ω = domain(_unit_cube(Val(D)))
    dims = ntuple(_ -> n, Val(D))
    unif = ntuple(_ -> true, Val(D))

    Ωd = mesh(Ω, dims, unif; backend = metal_backend())
    Wd = gridspace(Ωd)
    Ωh = Bramble._host_mirror_mesh(mesh(Wd))
    Wh = gridspace(Ωh)

    ad = form(Wd, Wd, _poisson_mass)
    ah = form(Wh, Wh, _poisson_mass)
    Kd = kronecker_operator(ad)
    Kh = kronecker_operator(ah)

    N = size(Kd, 1)
    xh = rand(Float32, N)
    xd = MtlArray(xh)
    yh = similar(xh)
    yd = similar(xd)
    scratch_h = (similar(xh), similar(xh))
    scratch_d = (similar(xd), similar(xd))

    # (a) one apply ----------------------------------------------------------------- #
    mul!(yh, Kh, xh; scratch = scratch_h) # warm-up
    mul!(yd, Kd, xd; scratch = scratch_d)
    Metal.synchronize()

    t_apply_h = minimum((@benchmark mul!($yh, $Kh, $xh; scratch = $scratch_h)).times) / 1e6
    t_apply_d = minimum((@benchmark begin
        mul!($yd, $Kd, $xd; scratch = $scratch_d)
        Metal.synchronize()
    end).times) / 1e6
    push!(rows, Row("mul! apply", dimlabel, t_apply_h, t_apply_d, t_apply_h / t_apply_d))

    # (b) 30 CG-shaped iterations ----------------------------------------------------- #
    bh = rand(Float32, N)
    bd = MtlArray(bh)
    ph, rh, Aph = similar(xh), similar(xh), similar(xh)
    pd, rd, Apd = similar(xd), similar(xd), similar(xd)

    _cg30!(xh, Kh, ph, rh, Aph, bh, scratch_h) # warm-up
    _cg30!(xd, Kd, pd, rd, Apd, bd, scratch_d)
    Metal.synchronize()

    t_cg_h = minimum((@benchmark _cg30!($xh, $Kh, $ph, $rh, $Aph, $bh, $scratch_h)).times) / 1e6
    t_cg_d = minimum((@benchmark begin
        _cg30!($xd, $Kd, $pd, $rd, $Apd, $bd, $scratch_d)
        Metal.synchronize()
    end).times) / 1e6
    push!(rows, Row("30 CG-shaped iters", dimlabel, t_cg_h, t_cg_d, t_cg_h / t_cg_d))

    # (c) fdm_solve total, and the host eigendecomposition's own share ---------------- #
    # Expensive at 2D's 3000x3000 (a 3000x3000 dense generalised eigenproblem per axis), so
    # few samples -- `bramble-verification`'s minimum-of-several still holds with samples=3,
    # just fewer of them than the microsecond-scale measurements above need.
    Fh = rand(Float32, N)
    Fd = MtlArray(Fh)
    fdm_solve(ah, Fh) # warm-up
    fdm_solve(ad, Fd)
    Metal.synchronize()

    solve_samples = D == 2 ? 3 : 10
    t_solve_h = minimum((@benchmark(fdm_solve($ah, $Fh); samples = solve_samples, evals = 1)).times) / 1e6
    t_solve_d = minimum((@benchmark begin
        fdm_solve($ad, $Fd)
        Metal.synchronize()
    end samples=solve_samples evals=1).times) / 1e6
    push!(rows, Row("fdm_solve total", dimlabel, t_solve_h, t_solve_d, t_solve_h / t_solve_d))

    eig_ms = _eig_ms(Ωh, D; samples = solve_samples)
    push!(rows, Row("  of which: host eigendecomposition", dimlabel, eig_ms, NaN, NaN))

    return rows
end

function main()
    power = _power_source()
    _print_header(power)

    # Sizes default to #323's own table (3000x3000 / 200^3); overridable only for a smoke
    # run at tiny sizes (`KRON_BENCH_2D_N`/`KRON_BENCH_3D_N` env vars), never committed here.
    n2 = parse(Int, get(ENV, "KRON_BENCH_2D_N", "3000"))
    n3 = parse(Int, get(ENV, "KRON_BENCH_3D_N", "200"))

    rows = Row[]
    t0 = time()
    _run(2, n2, "2D $(n2)x$(n2)", rows)
    _run(3, n3, "3D $(n3)x$(n3)x$(n3)", rows)
    elapsed_min = (time() - t0) / 60

    println(rpad("Measurement", 38), rpad("Size", 16), rpad("Host (ms)", 14), rpad("Metal (ms)", 14), "Host/Metal")
    for r in rows
        metal_str = isnan(r.metal_ms) ? "--" : string(round(r.metal_ms; digits = 4))
        ratio_str = isnan(r.ratio) ? "--" : string(round(r.ratio; digits = 3))
        println(
            rpad(r.label, 38), rpad(r.dim, 16), rpad(string(round(r.host_ms; digits = 4)), 14),
            rpad(metal_str, 14), ratio_str
        )
    end
    println()
    println("Wall-clock time for this run: ", round(elapsed_min; digits = 2), " min.")
    if elapsed_min > 10
        println(
            "NOTE: this run exceeded the ~10-minute budget S4.3 sets; a later run should ",
            "shrink one or both sizes below 3000x3000 / 200x200x200 and say so here."
        )
    end
    println()
    println("#323's own hand-rolled matrix-free CG figures (quoted from the issue, not")
    println("reproduced -- that loop calls Δₕ! directly, bypassing the forms API/this operator):")
    println()
    println(rpad("dim", 6), rpad("N", 16), rpad("Serial (min)", 15), rpad("Metal (min)", 14), "Serial/Metal")
    println(rpad("1D", 6), rpad("10,000,000", 16), rpad("1302.9 ms", 15), rpad("547.5 ms", 14), "2.38x")
    println(rpad("2D", 6), rpad("3000x3000", 16), rpad("1347.2 ms", 15), rpad("867.1 ms", 14), "1.55x")
    println(rpad("3D", 6), rpad("200x200x200", 16), rpad("1325.2 ms", 15), rpad("1144.0 ms", 14), "1.16x")
    println()
    println(
        "(30 CG-shaped iterations per trial: one Δₕ! operator apply, two innerₕ reductions, ",
        "three broadcasts, all in-place, zero host allocations on the CPU paths -- the same ",
        "shape `_cg30!` above reproduces through the forms API's KroneckerLinearOperator ",
        "instead of a hand-written Δₕ! call.)"
    )
    return nothing
end

main()
