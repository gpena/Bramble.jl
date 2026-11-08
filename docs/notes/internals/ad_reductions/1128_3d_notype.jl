# Reproducer for gpena/Bramble.jl#1128: Enzyme through the 3D `assemble` of
# `inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))` on a 5 by 5 by 5 mesh raises `EnzymeNoTypeError`, in
# reverse and in forward mode. The type analysis meets a 256-byte `memcpy` of the form's
# expression tree in the `BilinearForm` constructor (src/assembly/bilinear.jl:249) that it
# cannot type, with a `Cannot deduce type of copy` line. `Enzyme.API.maxtypeoffset!`, whose
# default is 512 bytes, removes it (`BRAMBLE_REPRO_TYPEOFFSET` below).
#
# A second, separate failure shows in 3D when the averaging operator `M₂` is called on a
# grid function inside the differentiated function: `Cannot deduce type of memset`, from the
# boundary loop of `_average_band!` (src/operators/average.jl:178). Variants `avgz` and
# `avgops` reproduce it. The form of the issue does not reach it.
#
# Run from the repository root, with Bramble from the root project:
#
#     julia --project=. --threads=1 --optimize=1 --check-bounds=auto \
#         docs/notes/internals/ad_reductions/1128_3d_notype.jl [variant] [mode]
#
# Enzyme is not a dependency of the root project, so it is loaded from a temporary
# environment, the way the test harness does. The root Manifest.toml must hold Enzyme (it
# is a weak dependency of Bramble).
#
# `mode` is `reverse` (the default) or `forward`. The policy is `CpuSerial`, the mesh a
# 5 by 5 by 5 non-uniform one (seeded).
#
# Environment:
#   BRAMBLE_REPRO_VERBOSE=1       sets `Enzyme.Compiler.VERBOSE_ERRORS[]`, so the error names
#                                 the failing instruction and the type analysis state.
#   BRAMBLE_REPRO_TYPEOFFSET=<n>  calls `Enzyme.API.maxtypeoffset!(n)` before the call.
#
# Exit status: nonzero (1) exactly when Enzyme raises `EnzymeNoTypeError`. If it raises
# another exception or returns a gradient, the script prints which and exits 0. A gradient
# is printed with its relative difference from a central difference.
#
# Variants, the first being the default.
#   averaged     the call of the issue, `assemble` of `inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))`.
#   diffusion    `assemble` of `inner₊(κ * ∇ₕ(U), ∇ₕ(V))`, no averaging.
#   formx, formxy, formxyx   forms with one, two and three `Mₓ(κ)` or `Mᵧ(κ)` terms.
#   formthree    three terms with three separate plain coefficients, no averaging.
#   linear       `assemble` of the linear form `innerₕ(κ, v)`.
#   diffops      the difference operators on the grid function (`D₋ₓ`, `D₋ᵧ`, `D₋₂`).
#   avgops       the averaging operators on the grid function (`Mₓ`, `Mᵧ`, `M₂`).
#   avgx, avgy, avgz   one of those three alone.
#   <name>2d     any variant on a 9 by 9 mesh (`averaged2d` is the control that works).
#   fixed_<name> the variant after redefining the boundary value of the forward and backward
#                averages, see the probe below.
_e = mktempdir()
write(joinpath(_e, "Project.toml"), "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n")
push!(LOAD_PATH, _e)
using Bramble, Enzyme, SparseArrays
using Bramble: D₋ₓ, D₋ᵧ, D₋₂, Mₓ, Mᵧ, M₂, CpuSerial
using Random

const VARIANT = isempty(ARGS) ? "averaged" : ARGS[1]
const MODE = length(ARGS) < 2 ? "reverse" : ARGS[2]
const FIXED = startswith(VARIANT, "fixed_")
const DIM = endswith(VARIANT, "2d") ? 2 : 3
const BASE = replace(VARIANT, r"^fixed_" => "", r"2d$" => "")
const NP = 3
get(ENV, "BRAMBLE_REPRO_VERBOSE", "") == "" || (Enzyme.Compiler.VERBOSE_ERRORS[] = true)
# `BRAMBLE_REPRO_TYPEOFFSET=<bytes>` raises Enzyme's `maxtypeoffset`, one of the limits the
# error text names.
get(ENV, "BRAMBLE_REPRO_TYPEOFFSET", "") == "" ||
    Enzyme.API.maxtypeoffset!(parse(Int, ENV["BRAMBLE_REPRO_TYPEOFFSET"]))

if FIXED
    # A probe, not the fix: the boundary value `zero(cur)` is a constant, so LLVM turns the
    # boundary loop of `_average_band!` into a `memset`. `cur - cur` is the same zero for a
    # finite `cur` and no constant, so the loop stays a loop. It gives `NaN` for an infinite
    # or `NaN` grid value, which is why this is not what Bramble should do.
    @eval Bramble @inline Base.@propagate_inbounds _compute_average(
        ::GridDirection, ::Val{true}, cur) = cur - cur
end

const Wₕ = let
    Bramble._seed_mesh1d_rng!(1234)
    try
        set = DIM == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
              domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0))
        gridspace(mesh(set, DIM == 2 ? (9, 9) : (5, 5, 5), ntuple(_ -> false, DIM);
            backend = backend(policy = CpuSerial())))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end
const B = randn(MersenneTwister(1234), ndofs(Wₕ), NP)

# A grid function holding `2 + B p`, with the element type of `p` (a `Dual`-free Enzyme
# shadow or a plain `Float64`).
function coefficient(p::AbstractVector{T}) where {T}
    κ = element(Wₕ, T)
    parent(κ) .= 2.0 .+ B * p
    return κ
end

function loss_averaged(p)
    κ = coefficient(p)
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V)))
    return sum(abs2, nonzeros(assemble(a)))
end

# One, two and three averaged coefficients in a form, each a separate `Mₓ(κ)` or `Mᵧ(κ)`.
# `formx` and `formxy` avoid `M₂`; `formxyx` has three terms and still avoids it.
function loss_formx(p)
    κ = coefficient(p)
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(Mₓ(κ) * D₋ₓ(U), D₋ₓ(V)))
    return sum(abs2, nonzeros(assemble(a)))
end

function loss_formxy(p)
    κ = coefficient(p)
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(Mₓ(κ) * D₋ₓ(U), D₋ₓ(V)) + inner₊(Mᵧ(κ) * D₋ᵧ(U), D₋ᵧ(V)))
    return sum(abs2, nonzeros(assemble(a)))
end

function loss_formxyx(p)
    κ = coefficient(p)
    a = form(Wₕ, Wₕ,
        (U, V) -> inner₊(Mₓ(κ) * D₋ₓ(U), D₋ₓ(V)) + inner₊(Mᵧ(κ) * D₋ᵧ(U), D₋ᵧ(V)) +
                  inner₊(Mₓ(κ) * D₋₂(U), D₋₂(V)))
    return sum(abs2, nonzeros(assemble(a)))
end

# Three terms with three separate plain coefficients, no averaging. In 2D the third term
# differences along `x` again.
function loss_formthree(p)
    κ₁, κ₂, κ₃ = coefficient(p), coefficient(p), coefficient(p)
    a = form(Wₕ, Wₕ,
        (U, V) -> inner₊(κ₁ * D₋ₓ(U), D₋ₓ(V)) + inner₊(κ₂ * D₋ᵧ(U), D₋ᵧ(V)) +
                  (DIM == 3 ? inner₊(κ₃ * D₋₂(U), D₋₂(V)) : inner₊(κ₃ * D₋ₓ(U), D₋ₓ(V))))
    return sum(abs2, nonzeros(assemble(a)))
end

function loss_diffusion(p)
    κ = coefficient(p)
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V)))
    return sum(abs2, nonzeros(assemble(a)))
end

function loss_linear(p)
    κ = coefficient(p)
    return sum(abs2, assemble(form(Wₕ, v -> innerₕ(κ, v))))
end

function loss_diffops(p)
    κ = coefficient(p)
    return sum(abs2, parent(D₋ₓ(κ))) + sum(abs2, parent(D₋ᵧ(κ))) +
           (DIM == 3 ? sum(abs2, parent(D₋₂(κ))) : 0.0)
end

function loss_avgops(p)
    κ = coefficient(p)
    return sum(abs2, parent(Mₓ(κ))) + sum(abs2, parent(Mᵧ(κ))) +
           (DIM == 3 ? sum(abs2, parent(M₂(κ))) : 0.0)
end

loss_avgx(p) = sum(abs2, parent(Mₓ(coefficient(p))))
loss_avgy(p) = sum(abs2, parent(Mᵧ(coefficient(p))))
loss_avgz(p) = sum(abs2, parent(M₂(coefficient(p))))

const LOSS = BASE == "diffusion" ? loss_diffusion :
             BASE == "formx" ? loss_formx :
             BASE == "formthree" ? loss_formthree :
             BASE == "formxy" ? loss_formxy :
             BASE == "formxyx" ? loss_formxyx :
             BASE == "avgx" ? loss_avgx :
             BASE == "avgy" ? loss_avgy :
             BASE == "avgz" ? loss_avgz :
             BASE == "linear" ? loss_linear :
             BASE == "diffops" ? loss_diffops :
             BASE == "avgops" ? loss_avgops : loss_averaged

println("variant ", VARIANT, ", ", MODE, ", ", DIM, "D, Enzyme ", pkgversion(Enzyme),
    ", Julia ", VERSION, ", check-bounds ", Base.JLOptions().check_bounds,
    " (0 auto, 1 yes, 2 no)")
flush(stdout)
p0 = [0.5 + 0.3 * sin(k) for k in 1:NP]
try
    if MODE == "reverse"
        dp = zeros(NP)
        Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(LOSS),
            Enzyme.Active, Enzyme.Duplicated(copy(p0), dp))
        g = dp
    else
        g = [only(Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Forward),
                 Enzyme.Const(LOSS), Enzyme.Duplicated(copy(p0), float.(1:NP .== k))))
             for k in 1:NP]
    end
    # A central difference, to see that a gradient is a gradient (1e-6 relative at most).
    h = 1e-6
    fd = [(LOSS(p0 .+ h .* (1:NP .== k)) - LOSS(p0 .- h .* (1:NP .== k))) / 2h for k in 1:NP]
    println("NO ERROR: gradient ", g, ", relative difference from a central difference ",
        maximum(abs, g .- fd) / maximum(abs, fd))
catch e
    if e isa Enzyme.Compiler.EnzymeNoTypeError
        println("REPRODUCED: ", typeof(e))
        showerror(stdout, e)
        println()
        exit(1)
    end
    println("NO EnzymeNoTypeError: Enzyme raised ", typeof(e))
    showerror(stdout, e)
    println()
end
