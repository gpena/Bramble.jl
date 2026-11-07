# Reproducer for gpena/Bramble.jl#1124: Enzyme forward mode adds a constant offset to the
# gradient of a bilinear form that sums several terms.
#
# The issue's claim: through `assemble` of the form
#
#     inner₊(2.0 * Mₕ(κ) * ∇ₕ(U), ∇ₕ(V)) + innerₕ(κ * U, 0.5 * D₋ᵧ(V))
#
# `Enzyme.autodiff(set_runtime_activity(Forward), ...)` returns a gradient that is off by
# one constant in every entry, whatever the seed. Reverse mode and ForwardDiff are exact.
# The form is three summands once `inner₊` expands to two products. This file runs the
# forms and call paths that the repository pins as broken (test/ext/
# enzyme_forward_assembly.jl and test/space/ad_coverage.jl), and compares each gradient
# with ForwardDiff's.
#
# Run from the repository root, with Bramble from the root project:
#
#     julia --project=. --threads=1 --optimize=1 --check-bounds=auto \
#         docs/notes/internals/ad_reductions/1124_forward_offset.jl [variant]
#
# Enzyme, ForwardDiff and DifferentiationInterface are not dependencies of the root
# project, so they are loaded from a temporary environment, the way the test harness does.
# The root Manifest.toml must hold them (they are weak dependencies of Bramble). To try
# another Enzyme, run the same file under a project that holds it (a temporary environment
# with Bramble developed from this checkout and that Enzyme added).
#
# Exit status: nonzero exactly when the failure reproduces, that is when the largest
# relative difference from ForwardDiff exceeds 1e-8 (an exact gradient differs by 1e-15; the
# offset the issue reports is 1e-3 to 1e-2) or Enzyme throws. The default variant is the
# issue's call, `diffusion_single`.
#
# A variant is `case_path`.
#
# Cases, the default first. Every function is a closure built inside the builder, as in the
# tests: the miscompile the repository saw followed how the closure compiled.
#   diffusion   the issue's form on a 17 by 17 seeded non-uniform mesh, `CpuSerial`, four
#               parameters (`test/ext/enzyme_forward_assembly.jl`, `_diffusion`).
#   mass2       two summands, `innerₕ(κU, V) + 0.25 innerₕ(κU, V)`, an exact control.
#   react2      two summands, `innerₕ(κU, V) + innerₕ(κU, 0.5 D₋ᵧ(V))`, another.
#   small       the same form on a 5 by 5 uniform mesh, the smallest that keeps the offset.
#   small_diffonly, small_noscale, small_scalefirst
#               `small` without its last summand; with that summand unscaled; with it first.
#               Only a scaled summand last keeps the offset.
#   three_uniform, three_nonuniform
#               `inner₊(κ∇U, ∇V) + innerₕ(κU, V) + innerₕ(κU, 0.5 D₋ᵧ(V))` on a 9 by 9 mesh,
#               three parameters, `κ = 2 + B p` (`test/space/ad_coverage.jl`,
#               `build_assemble_three`).
#
# Paths.
#   single      `Enzyme.autodiff` once per seed, `Duplicated`, as the issue does.
#   batch       one `Enzyme.autodiff` with every seed in a `BatchDuplicated`.
#   batch2      the first two seeds in a `BatchDuplicated` (the `pushforward` of the test).
#   di          `DifferentiationInterface.gradient` with `AutoEnzyme(mode = set_runtime_
#               activity(Forward), function_annotation = Const)`, as the tests do.
#   reverse     the same call in reverse mode, the control the issue says is exact.
#   contract    `single` on the matrix-free value `a(u, v)` for fixed vectors, no assembly.
_e = mktempdir()
write(joinpath(_e, "Project.toml"),
    "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n" *
    "ForwardDiff = \"f6369f11-7733-5829-9624-2563aa707210\"\n" *
    "DifferentiationInterface = \"a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63\"\n")
push!(LOAD_PATH, _e)
using Bramble, Enzyme, ForwardDiff, SparseArrays, Random, Logging
using Bramble: CpuSerial, D₋ᵧ

# Enzyme warns once about the BLAS fallback; the log is not the result.
global_logger(NullLogger())

const VARIANT = isempty(ARGS) ? "diffusion_single" : only(ARGS)
const CASE, PATH = let parts = split(VARIANT, "_")
    join(parts[1:(end - 1)], "_"), parts[end]
end
# Only the DI paths need it, and an environment under test may not hold it.
const DI = PATH in ("di", "reverse") ?
           Base.require(Main, :DifferentiationInterface) : nothing

# The three forms. `u`, `v` are the trial and test operands, `κ` a grid function.
diffusion(κ, u, v) = inner₊(2.0 * Mₕ(κ) * ∇ₕ(u), ∇ₕ(v)) + innerₕ(κ * u, 0.5 * D₋ᵧ(v))
mass2(κ, u, v) = innerₕ(κ * u, v) + 0.25 * innerₕ(κ * u, v)
react2(κ, u, v) = innerₕ(κ * u, v) + innerₕ(κ * u, 0.5 * D₋ᵧ(v))
diffonly(κ, u, v) = inner₊(2.0 * Mₕ(κ) * ∇ₕ(u), ∇ₕ(v))
noscale(κ, u, v) = inner₊(2.0 * Mₕ(κ) * ∇ₕ(u), ∇ₕ(v)) + innerₕ(κ * u, D₋ᵧ(v))
scalefirst(κ, u, v) = innerₕ(κ * u, 0.5 * D₋ᵧ(v)) + inner₊(2.0 * Mₕ(κ) * ∇ₕ(u), ∇ₕ(v))
three(κ, u, v) = inner₊(κ * ∇ₕ(u), ∇ₕ(v)) + innerₕ(κ * u, v) + innerₕ(κ * u, 0.5 * D₋ᵧ(v))

# `p -> sum(abs2, nonzeros(assemble(a(p))))` for the form `terms` on a `dims` mesh seeded
# with `seed`, `uniform` or not, `np` parameters, `κ = shift + B p`; and its start point.
# `typed` sets the backend's vector and matrix types, as `ad_coverage.jl` does. `contract`
# returns the matrix-free `a(fu, fv)` for two fixed vectors in place of the assembled sum.
function functional(terms, dims, seed, uniform, np, bseed, shift; typed = false,
        contract = false)
    Bramble._seed_mesh1d_rng!(seed)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    M = mesh(Ω, dims, (uniform, uniform);
        backend = typed ?
                  backend(; vector_type = Vector{Float64},
            matrix_type = SparseMatrixCSC{Float64, Int}, policy = CpuSerial()) :
                  backend(policy = CpuSerial()))
    Bramble._unseed_mesh1d_rng!()
    W = gridspace(M)
    B = randn(MersenneTwister(bseed), ndofs(W), np)
    fu = randn(MersenneTwister(3), ndofs(W))
    fv = randn(MersenneTwister(4), ndofs(W))
    function f(p::AbstractVector{T}) where {T}
        κ = element(W, T)
        parent(κ) .= shift .+ B * p
        a = form(W, W, (u, v) -> terms(κ, u, v))
        contract && return a(fu, fv)
        return sum(abs2, nonzeros(assemble(a)))
    end
    return f, [0.5 + 0.3 * sin(k) for k in 1:np]
end

function build(case; contract = false)
    big = Dict("diffusion" => diffusion, "mass2" => mass2, "react2" => react2)
    small = Dict("small" => diffusion, "small_diffonly" => diffonly,
        "small_noscale" => noscale, "small_scalefirst" => scalefirst)
    haskey(big, case) &&
        return functional(big[case], (17, 17), 1124, false, 4, 7, 0.0; contract)
    haskey(small, case) &&
        return functional(small[case], (5, 5), 1124, true, 4, 7, 0.0; contract)
    case == "three_uniform" &&
        return functional(three, (9, 9), 1234, true, 3, 1234, 2.0; typed = true,
            contract)
    case == "three_nonuniform" &&
        return functional(three, (9, 9), 1234, false, 3, 1234, 2.0; typed = true,
            contract)
    error("unknown case ", case)
end

const FWD = Enzyme.set_runtime_activity(Enzyme.Forward)
const REV = Enzyme.set_runtime_activity(Enzyme.Reverse)
unit(n, k) = Float64.(1:n .== k)

function gradient(f, x, path)
    n = length(x)
    if path == "single" || path == "contract"
        return [only(Enzyme.autodiff(FWD, Enzyme.Const(f), Enzyme.Duplicated,
                    Enzyme.Duplicated(copy(x), unit(n, k)))) for k in 1:n]
    elseif path == "batch" || path == "batch2"
        ks = path == "batch" ? (1:n) : (1:2)
        r = Enzyme.autodiff(FWD, Enzyme.Const(f), Enzyme.BatchDuplicated,
            Enzyme.BatchDuplicated(copy(x), ntuple(j -> unit(n, ks[j]), length(ks))))
        return collect(Float64, r[1])
    elseif path == "di"
        return DI.gradient(f,
            DI.AutoEnzyme(; mode = FWD, function_annotation = Enzyme.Const), x)
    elseif path == "reverse"
        return DI.gradient(f,
            DI.AutoEnzyme(; mode = REV, function_annotation = Enzyme.Const), x)
    end
    error("unknown path ", path)
end

println("variant ", VARIANT, ", Enzyme ", pkgversion(Enzyme), ", Julia ", VERSION,
    ", check-bounds ", Base.JLOptions().check_bounds, " (0 auto, 1 yes, 2 no)")
flush(stdout)
failed = false
t = @elapsed try
    f, x = build(CASE; contract = PATH == "contract")
    ref = ForwardDiff.gradient(f, x)
    g = gradient(f, x, PATH)
    r = ref[1:length(g)]
    err = maximum(abs.(g .- r)) / maximum(abs, r)
    println("Enzyme       ", g, "\nForwardDiff  ", r, "\ndifference   ", g .- r,
        "\nmax relative difference ", err)
    if err > 1e-8
        global failed = true
        println("FAILURE: offset ", unique(round.(g .- r; sigdigits = 3)))
    else
        println("NO FAILURE")
    end
catch e
    global failed = true
    println("FAILURE: ", typeof(e))
    println(first(sprint(showerror, e), 400))
end
println("seconds ", round(t; digits = 1))
exit(failed ? 1 : 0)
