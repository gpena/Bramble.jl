#===========================================================================#
# Polygon and level-set assembly against plain assembly, per DOF (gpena/Bramble.jl#876 and
# #884, S2.1 of .claude/plans/v4-1-3-perf.md).
#
# A one-off A/B tool, not a CI row: it registers nothing in `benchmarks.jl`, so no settled
# baseline key moves. Run it on main and on the branch and compare the printed tables:
#
#     julia --project=benchmark benchmark/polygon_vs_plain.jl
#
# It refuses to time below `--optimize=2`: at -O1 the SIMD loops do not vectorise and the
# figures are not comparable. It uses only API present at 9c34251c, so it runs unchanged on
# main.
#
# Cases, all with the form `-Δₕ + 0.5 Dcₓ` (`innerₕ(-1.0 * Δₕ(u), v) + innerₕ(0.5 * Dcₓ(u), v)`)
# on seeded non-uniform meshes (seed 2), per CPU policy (CpuSerial, then CpuPolyester):
#
#   - fig1:      Figure 1 polygon, `(40, 36)` mesh closing to a 129x113 box, `polygon_spaces`;
#   - disk:      disk level set (`hypot(x, y) - 1` on [-1.5, 1.5]^2), `(60, 60)`, `polygon_spaces`;
#   - rect:      plain non-uniform square mesh with about the polygon's DOF count, one per
#                polygon case (`rect(fig1)`, `rect(disk)`);
#   - plain box: `gridspace` on the Figure 1 box, no polygon (the padded/plain denominator);
#   - padded:    the polygon form's padded form over the whole box (`ap.padded`), refilled
#                into a matrix of its own, `assemble(ap.padded)`: on the branch `ap.A_pad`
#                holds only the rows the test closure reads (#884), which the whole-box
#                refill does not fit;
#   - ghost box: `gridspace(Ωₕ; ghosts = :boundary)` on the Figure 1 box.
#
# Per case: DOFs, warm `assemble!` (minimum of 30 calls) and warm matrix-free `mul!`, both in
# ns per DOF, and the allocations of one warm call of each. Then the polygon/rectangle
# per-DOF ratio (fig1 and disk) and the padded/plain ratio of `assemble!`. A `mul!` the
# build does not offer for a case prints `n/a`.
#===========================================================================#
using Bramble
using Bramble: polygon, polygon_spaces, Dcₓ, levelset, CpuSerial, CpuPolyester
using LinearAlgebra: mul!
using Polyester
using Random

Base.JLOptions().opt_level >= 2 ||
    throw(ArgumentError("run without --optimize=1: timings below the default level are not figures"))

const FIG1 = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0),
    (4.0, 5.0), (2.0, 5.0), (0.0, 3.0)]
const SAMPLES = 30

form_f(u, v) = innerₕ(-1.0 * Δₕ(u), v) + innerₕ(0.5 * Dcₓ(u), v)

# Warm once, then the minimum of `n` calls in seconds.
function tmin(g::F, n = SAMPLES) where {F}
    g()
    return minimum(@elapsed(g()) for _ in 1:n)
end

# Allocations of one warm call.
function allocs(g::F) where {F}
    g()
    return @allocated g()
end

struct Row
    name::String
    dofs::Int
    asm_ns::Float64   # assemble!, ns per DOF
    mul_ns::Float64   # mul!, ns per DOF (NaN when not offered)
    asm_alloc::Int
    mul_alloc::Int    # -1 when not offered
end

# `A`, `a`: the assembled matrix and its form; `opf`: builds the matrix-free operator or
# throws when the build has none for this form.
function measure(name, n, A, a, opf)
    asm! = () -> assemble!(A, a)
    tasm, aasm = tmin(asm!), allocs(asm!)
    tmul, amul = try
        op = opf()
        x = rand(n)
        y = similar(x)
        m! = () -> mul!(y, op, x)
        tmin(m!), allocs(m!)
    catch err
        err isa InterruptException && rethrow()
        NaN, -1
    end
    return Row(name, n, 1.0e9 * tasm / n, 1.0e9 * tmul / n, aasm, amul)
end

meshseed(dom, ns, pol) = (Random.seed!(2);
mesh(dom, ns, (false, false); backend = Bramble.backend(policy = pol)))

function polygon_row(name, Ωₕ, label)
    U, V = polygon_spaces(Ωₕ, label, x -> 0.0)
    ap = form(U, V, form_f)
    Ap = assemble(ap)
    prow = measure(name, ndofs(U), Ap, ap, () -> matrix_free_operator(ap))
    pad = ap.padded
    Wpad = Bramble.trial_space(pad)
    padrow = measure(name * " padded", ndofs(Wpad), assemble(pad), pad,
        () -> matrix_free_operator(pad))
    return prow, padrow
end

function plain_row(name, Ωₕ; kw...)
    W = gridspace(Ωₕ; kw...)
    a = form(W, W, form_f)
    A = assemble(a)
    return measure(name, ndofs(W), A, a, () -> matrix_free_operator(a))
end

function run_policy(pol)
    Ωp = meshseed(domain(interval(0.0, 6.0) × interval(0.0, 5.0), :P => polygon(FIG1)),
        (40, 36), pol)
    Ωd = meshseed(
        domain(interval(-1.5, 1.5) × interval(-1.5, 1.5),
            :D => levelset(p -> hypot(p[1], p[2]) - 1)),
        (60, 60), pol)
    fig, figpad = polygon_row("fig1 polygon", Ωp, :P)
    dsk, dskpad = polygon_row("disk level set", Ωd, :D)
    rect(r) = begin
        n = round(Int, sqrt(r.dofs))
        plain_row("rect ~ $(r.name) ($(n)x$(n))",
            meshseed(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n), pol))
    end
    rfig, rdsk = rect(fig), rect(dsk)
    box = plain_row("plain box", Ωp)
    gst = plain_row("ghost box", Ωp; ghosts = :boundary)
    return [fig, figpad, dsk, dskpad, rfig, rdsk, box, gst],
    (fig_rect = fig.asm_ns / rfig.asm_ns, dsk_rect = dsk.asm_ns / rdsk.asm_ns,
        fig_rect_mul = fig.mul_ns / rfig.mul_ns, dsk_rect_mul = dsk.mul_ns / rdsk.mul_ns,
        pad_plain = figpad.asm_ns / box.asm_ns)
end

fmt(x) = isnan(x) ? "n/a" : string(round(x, digits = 1))
fmt(x::Int) = x < 0 ? "n/a" : string(x)

function report(pname, rows, ratios)
    println("\n== ", pname, " (threads ", Threads.nthreads(), ") ==")
    println(rpad("case", 36), lpad("dofs", 8), lpad("asm! ns/dof", 13), lpad("mul! ns/dof", 13),
        lpad("asm! B", 9), lpad("mul! B", 9))
    for r in rows
        println(rpad(r.name, 36), lpad(r.dofs, 8), lpad(fmt(r.asm_ns), 13),
            lpad(fmt(r.mul_ns), 13), lpad(fmt(r.asm_alloc), 9), lpad(fmt(r.mul_alloc), 9))
    end
    println("polygon/rect per DOF, assemble!: fig1 ", fmt(ratios.fig_rect), "  disk ",
        fmt(ratios.dsk_rect))
    println("polygon/rect per DOF, mul!:      fig1 ", fmt(ratios.fig_rect_mul), "  disk ",
        fmt(ratios.dsk_rect_mul))
    println("padded/plain assemble!:          ", fmt(ratios.pad_plain))
end

println("Bramble ", pkgversion(Bramble), ", Julia ", VERSION, ", opt_level ",
    Base.JLOptions().opt_level)
for (pname, pol) in (("CpuSerial", CpuSerial()), ("CpuPolyester", CpuPolyester()))
    rows, ratios = run_policy(pol)
    report(pname, rows, ratios)
end
