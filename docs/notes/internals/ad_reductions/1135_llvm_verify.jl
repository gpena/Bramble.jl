# Reproducer for gpena/Bramble.jl#1135: Enzyme reverse mode through `A \ b` with a
# symmetric positive definite sparse `A` aborts the process with
# `LLVM ERROR: function failed verification (2)`. Bramble is not involved: the call is
# SparseArrays', and the test line it came from (test/ext/chainrules_enzyme_ext.jl,
# `loss_no_rule`) pins `A \ F` on an assembled Bramble system.
#
# Run from the repository root, the way the weekly ad-4 shard runs a test file:
#
#     julia --project=. --threads=3 --optimize=1 --check-bounds=auto \
#         docs/notes/internals/ad_reductions/1135_llvm_verify.jl
#
# Enzyme is not a dependency of the root project, so it is loaded from a temporary
# environment, the way the test harness does. The root Manifest.toml must hold Enzyme
# (it is a weak dependency of Bramble).
#
# Exit status: nonzero (134, SIGABRT) exactly when the abort reproduces. If Enzyme raises
# an exception instead, or returns a gradient, the script prints which and exits 0.
# An optional argument picks another variant: `tridiag` is the 21-point 1D Laplacian the
# test assembles, `triangular` an upper triangular matrix (aborts too), `cholesky` an
# explicit `cholesky(A) \ b` (Enzyme raises `EnzymeNoDerivativeError`, no abort).
_e = mktempdir()
write(joinpath(_e, "Project.toml"), "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n")
push!(LOAD_PATH, _e)
using Enzyme, SparseArrays, LinearAlgebra

const SPD2 = sparse([2.0 -1.0; -1.0 2.0])
const TRIU2 = sparse([2.0 -1.0; 0.0 2.0])
const TRIDIAG = spdiagm(-1 => fill(-1.0, 20), 0 => fill(2.0, 21), 1 => fill(-1.0, 20))

# The gradient is taken with respect to the right-hand side's scale only.
solve_min(θ) = sum(SPD2 \ [θ, 1.0])
solve_tridiag(θ) = sum(abs2, TRIDIAG \ (θ .* ones(21)))
solve_triangular(θ) = sum(TRIU2 \ [θ, 1.0])
solve_cholesky(θ) = sum(cholesky(SPD2) \ [θ, 1.0])

const VARIANTS = Dict(
    "min" => solve_min, "tridiag" => solve_tridiag, "triangular" => solve_triangular,
    "cholesky" => solve_cholesky
)

variant = isempty(ARGS) ? "min" : only(ARGS)
println("variant ", variant, ", Enzyme ", pkgversion(Enzyme), ", Julia ", VERSION,
    ", check-bounds ", Base.JLOptions().check_bounds, " (0 auto, 1 yes, 2 no)")
flush(stdout)
mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
try
    g = Enzyme.gradient(mode, Enzyme.Const(VARIANTS[variant]), 0.7)[1]
    println("NO ABORT: gradient ", g)
catch e
    println("NO ABORT: Enzyme raised ", typeof(e))
end
