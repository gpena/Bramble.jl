# Reproducer for gpena/Bramble.jl#1127: Enzyme on a constrained ghost space.
#
# Two failures, both from the issue.
#
#   * `Δₕ!(out, v)` on a `GhostVector`, reverse mode, ends in `EnzymeNoDerivativeError`
#     ("cannot handle (augmented) unknown intrinsic llvm.smax.i64" in `_ghost_facet_band!`).
#     Forward mode ends the process with a segmentation fault. 17 s each.
#   * `assemble(a, l)` on the ghost space, reverse mode, does not finish compiling in the
#     20 minutes this reduction gave it. It has two causes, each cut away by a fix below.
#     The load half, `assemble!(l.F_pad, l.padded)` into the load form's own buffer, ends in
#     `EnzymeNoDerivativeError` (a `strcmp` with no derivative) in 13 s (`loadbuffer`).
#
# Run from the repository root, with Bramble from the root project:
#
#     julia --project=. --threads=1 --optimize=1 --check-bounds=auto \
#         docs/notes/internals/ad_reductions/1127_ghost.jl [variant]
#
# Enzyme is not a dependency of the root project, so it is loaded from a temporary
# environment, the way the test harness does. The root Manifest.toml must hold Enzyme (it
# is a weak dependency of Bramble).
#
# Exit status: nonzero exactly when Enzyme fails on the variant (an exception, or the
# segmentation fault of a forward `Δₕ!`). The default variant is the quick one, `Δₕ!` in
# reverse, so the default run exits nonzero exactly when that failure reproduces. A variant
# that differentiates prints the gradient and its largest relative difference from a
# central difference.
#
# A variant is `[fix_]case_mode`, mode `rev` or `fwd`; fixes combine with `+`, as in
# `split+loadbuf_assemble_rev`.
#
# Cases, the default first.
#   runtime     `Δₕ!` on a `GhostVector`, 7 by 6 non-uniform mesh, `CpuSerial`.
#   assemble    `assemble(a, l)` of the issue's ghost cell, with Neumann and Robin sides; the
#               parameters scale the coefficients of `a`, which is built inside the function.
#   inner       the same with `inner₊(∇ₕ(U), ∇ₕ(V))` in place of `Δₕ`.
#   padded      the form is built on the ghost space, then `assemble` runs on its padded
#               form, with no reduction and no load. The smallest that keeps the failure.
#   plain       the same padded form built on the padded space directly, a control with no
#               ghost construction.
#   mass        `p[1] * innerₕ(U, V)` on the ghost space, `assemble(a)`, no `Δₕ`.
#   pairmass    `assemble(a, l)` with the mass form, the gradient reaches the lifted load.
#   load        `assemble!` of the load form's padded form into a buffer made here.
#   loadbuffer  the same into the load form's own `F_pad`.
#   formonly    the ghost `form` call with `Δₕ`, only `a.coef` read, no assembly.
#   mapsonly    `_ghost_walk` and `_ghost_maps` on a padded form with `Δₕ`, no assembly.
#   data, robin `constrained` with the Neumann/Robin data `g`, or the Robin coefficient `β`,
#               set to `p[1]`; the loss reads the closure's `d`, or the entries of its `E`.
#
# Fixes, each a prefix that redefines Bramble code or adds an Enzyme rule in this process.
#   unroll      `_ghost_facet_band!` visits its four `_FacetFix` through `ntuple(Val(4))`
#               in place of `for f in fixes`. For `runtime`.
#   inact-X     `EnzymeRules.inactive` on Bramble functions, `X` any of `maps`
#               (`_ghost_maps`), `walk` (`_ghost_walk`), `red` (`_ghost_reduction_map`,
#               `_ghost_lift_map`), joined by `-` as in `inact-maps-walk_padded_rev`.
#               Each is a probe, not a fix; see the note.
#   loadbuf     `_assemble_ghost_load!` assembles the load into a buffer made inside it, not
#               into `l.F_pad`.
#   split       `_ghost_maps` split in two: the matrix `A_pad` is allocated in the
#               differentiated code, and the index structure is built behind an inactive
#               call.
_e = mktempdir()
write(joinpath(_e, "Project.toml"), "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n")
push!(LOAD_PATH, _e)
using Bramble, Enzyme, SparseArrays, Random, Logging
using Bramble: Δₕ!, GhostVector, update_ghosts!

# Enzyme warns once per stored value it cannot find a use for, thousands of lines of IR.
global_logger(NullLogger())

const VARIANT = isempty(ARGS) ? "runtime_rev" : only(ARGS)
const PARTS = split(VARIANT, "_")
const FIX = length(PARTS) == 3 ? PARTS[1] : ""
const CASE = PARTS[end - 1]
const MODE = PARTS[end]
const NP = 3

# The test harness's `ghost_mesh(policy, (7, 6))`, `CpuSerial`.
const Ωₕ = let
    Bramble._seed_mesh1d_rng!(11)
    m = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (7, 6), (false, false))
    Bramble._unseed_mesh1d_rng!()
    m
end
const W = gridspace(Ωₕ; ghosts = (:xmin, :xmax, :ymax))
const C = constrained(W,
    u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => 2.0 * u + ∂ₙ(u) => 1.0,
        :ymin => u => 0.0, :ymax => ∂ₙ(u) => 1.0))
const P = Bramble.padded_space(C)
const B = randn(MersenneTwister(1234), 42, NP)
const L = form(C, v -> innerₕ(x -> 1.0 + x[1], v))

const FIXES = split(FIX, "+")
for fix in FIXES
    startswith(fix, "inact") || fix in ("split", "unroll", "loadbuf", "") ||
        error("unknown fix ", fix)
end
for fix in filter(startswith("inact"), FIXES)
    kinds = split(fix, "-")[2:end]
    if "maps" in kinds
        Enzyme.EnzymeRules.inactive(::typeof(Bramble._ghost_maps), args...) = nothing
    end
    if "walk" in kinds
        Enzyme.EnzymeRules.inactive(::typeof(Bramble._ghost_walk), args...) = nothing
    end
    if "red" in kinds
        Enzyme.EnzymeRules.inactive(::typeof(Bramble._ghost_reduction_map), args...) = nothing
        Enzyme.EnzymeRules.inactive(::typeof(Bramble._ghost_lift_map), args...) = nothing
    end
end
if "split" in FIXES
    @eval Bramble begin
        function _ghost_maps_structure(Wₕ, Vₕ, A_pad)
            moves = _exterior_moves(Wₕ, Vₕ, A_pad)
            src, dst, coef, colptr, rowval = _ghost_reduction_map(
                backend(Vₕ), A_pad, Vₕ.E, Wₕ.E, moves)
            liftptr, liftsrc, liftcol = _ghost_lift_map(backend(Vₕ), A_pad,
                _affine_support(Wₕ), moves)
            return (src, dst, coef, colptr, rowval, liftptr, liftsrc, liftcol,
                !isempty(moves[1]))
        end
        function _ghost_maps(Wₕ::_GhostSide, Vₕ::_GhostSide, a::BilinearForm, walk = nothing)
            _check_ghost_eltype(_matrix_eltype(a, a.ast))
            A_pad = _allocate_walked(a, a.ast, _cut_walk(walk))::SparseMatrixCSC
            return (A_pad, _ghost_maps_structure(Wₕ, Vₕ, A_pad)...)
        end
    end
    Enzyme.EnzymeRules.inactive(::typeof(Bramble._ghost_maps_structure), args...) = nothing
end
if "loadbuf" in FIXES
    # `assemble!(l.F_pad, l.padded)` writes the load form's own buffer, memory the
    # differentiated function does not allocate; a buffer made here is the fix.
    @eval Bramble function _assemble_ghost_load!(F::AbstractVector, a::_AnyGhostBilinearForm,
            l::_AnyGhostLinearForm, d::AbstractVector = a.trial_space.d)
        F_pad = similar(l.F_pad)
        assemble!(F_pad, l.padded)
        isempty(a.liftsrc) && return _closure_transpose_mul!(F, l, F_pad)
        E = l.test_space.E
        z = zero(promote_type(eltype(F_pad), eltype(a.A_pad), eltype(d)))
        _run_bands!(_ghost_policy(l), _lifted_load_band!, F,
            SparseArrays.getcolptr(E), rowvals(E), nonzeros(E), F_pad, nonzeros(a.A_pad),
            a.liftptr, a.liftsrc, a.liftcol, d, z)
        return F
    end
end
if "unroll" in FIXES
    @eval Bramble @noinline function _ghost_facet_band!(out, x, gb, kind::K, n::NTuple{D, Int},
            fixes::NTuple{F, _FacetFix{T}}, nbands::Int, b::Int) where {K, D, F, T}
        li = LinearIndices(n)
        band = _band_range(1:n[D], nbands, b)
        ntuple(Val(F)) do k
            f = fixes[k]
            f.on || return nothing
            d = f.d
            slab = _band_slab(ntuple(e -> e == d ? (f.i:f.i) : (1:n[e]), Val(D)), band)
            lt = LinearIndices(ntuple(e -> e == d ? 1 : n[e], Val(D)))
            inward = (f.min ? 1 : -1) * prod(ntuple(e -> e < d ? n[e] : 1, Val(D)))
            @inbounds for I in CartesianIndices(slab)
                idx = li[I]
                g = gb[f.off + lt[CartesianIndex(ntuple(e -> e == d ? 1 : I[e], Val(D)))]]
                out[idx] = _ghost_slab(kind, f.min, out[idx], x[idx], g, x[idx + inward], f.h)
            end
            return nothing
        end
        return nothing
    end
end

function loss_runtime(p)
    v = update_ghosts!(GhostVector(C, B * p), C)
    out = zeros(eltype(p), 42)
    Δₕ!(out, v)
    return sum(abs2, out)
end

function loss_assemble(p)
    a = form(C, C, (U, V) -> p[1] * innerₕ(-Δₕ(U), V) + p[2]^2 * innerₕ(U, V))
    A, F = assemble(a, L)
    return sum(abs2, nonzeros(A)) + 1e4 * p[3] * sum(F)
end

function loss_inner(p)
    a = form(C, C, (U, V) -> p[1] * inner₊(∇ₕ(U), ∇ₕ(V)) + p[2]^2 * innerₕ(U, V))
    A, F = assemble(a, L)
    return sum(abs2, nonzeros(A)) + 1e4 * p[3] * sum(F)
end

function loss_padded(p)
    a = form(C, C, (U, V) -> p[1] * innerₕ(-Δₕ(U), V) + p[2]^2 * innerₕ(U, V))
    return sum(abs2, nonzeros(assemble(a.padded)))
end

function loss_plain(p)
    a = form(P, P, (U, V) -> p[1] * innerₕ(-Δₕ(U), V) + p[2]^2 * innerₕ(U, V))
    return sum(abs2, nonzeros(assemble(a)))
end

function loss_mass(p)
    a = form(C, C, (U, V) -> p[1] * innerₕ(U, V))
    return sum(abs2, nonzeros(assemble(a)))
end

function loss_pairmass(p)
    a = form(C, C, (U, V) -> p[1] * innerₕ(U, V))
    A, F = assemble(a, L)
    return 1e4 * p[3] * sum(F) + p[1] * sum(abs2, F)
end

function loss_load(p)
    v = zeros(eltype(p), length(L.F_pad))
    assemble!(v, L.padded)
    return p[1] * sum(abs2, v)
end

function loss_loadbuffer(p)
    assemble!(L.F_pad, L.padded)
    return p[1] * sum(abs2, L.F_pad)
end

function loss_formonly(p)
    a = form(C, C, (U, V) -> p[1] * innerₕ(-Δₕ(U), V))
    return p[1] * sum(a.coef)
end

function loss_mapsonly(p)
    a = form(P, P, (U, V) -> p[1] * innerₕ(-Δₕ(U), V))
    maps = Bramble._ghost_maps(C, C, a, Bramble._ghost_walk(C, a))
    return p[1] * sum(maps[4])
end

function constrained_with(p, what)
    β = what === :robin ? p[1] : 2.0
    g = what === :data ? p[1] : 1.0
    return constrained(W,
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => β * u + ∂ₙ(u) => g,
            :ymin => u => 0.0, :ymax => ∂ₙ(u) => 1.0))
end
loss_data(p) = sum(abs2, constrained_with(p, :data).d)
loss_robin(p) = sum(abs2, nonzeros(constrained_with(p, :robin).E))

const LOSSES = Dict("runtime" => loss_runtime, "assemble" => loss_assemble,
    "inner" => loss_inner, "padded" => loss_padded, "plain" => loss_plain,
    "mass" => loss_mass, "pairmass" => loss_pairmass, "formonly" => loss_formonly, "load" => loss_load, "loadbuffer" => loss_loadbuffer,
    "mapsonly" => loss_mapsonly, "data" => loss_data, "robin" => loss_robin)
const LOSS = LOSSES[CASE]

function central_difference(f, p; h = 1e-6)
    return [(f(p .+ h .* (1:length(p) .== k)) - f(p .- h .* (1:length(p) .== k))) / 2h
            for k in eachindex(p)]
end

println("variant ", VARIANT, ", Enzyme ", pkgversion(Enzyme), ", Julia ", VERSION,
    ", check-bounds ", Base.JLOptions().check_bounds, " (0 auto, 1 yes, 2 no)")
flush(stdout)
const p0 = [0.5 + 0.3 * sin(k) for k in 1:NP]
failed = false
t = @elapsed try
    if MODE == "fwd"
        mode = Enzyme.set_runtime_activity(Enzyme.Forward)
        g = [only(Enzyme.autodiff(mode, Enzyme.Const(LOSS), Enzyme.Duplicated,
                 Enzyme.Duplicated(copy(p0), Float64.(1:NP .== k)))) for k in 1:NP]
    else
        g = zeros(NP)
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
        Enzyme.autodiff(mode, Enzyme.Const(LOSS), Enzyme.Active,
            Enzyme.Duplicated(copy(p0), g))
    end
    ref = central_difference(LOSS, p0)
    err = maximum(abs.(g .- ref) ./ max.(abs.(ref), 1e-12))
    println("NO FAILURE: gradient ", g, "\ncentral difference ", ref, ", max relative difference ", err)
catch e
    global failed = true
    msg = sprint(showerror, e)
    println("FAILURE: ", typeof(e))
    println(first(msg, 300))
    keep = filter(l -> occursin(r"^ *(\[\d+\]|@ |Failure within|cannot handle|No create)", l),
        split(msg, '\n'))
    println(join(first(keep, 30), '\n'))
end
println("seconds ", round(t; digits = 1))
exit(failed ? 1 : 0)
