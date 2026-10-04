# Measurements moved out of `docs/src/examples/poisson_nonlinear.jl`

## Pattern from the AST: `prepare_jacobian` cost

```
# What changes is what `prepare_jacobian` has to pay for: no tracing pass, only coloring.
# Measured on this mesh, `prepare_jacobian` costs 0.140 ms with the tracer against 0.062 ms
# given the pattern directly, and [`jacobian_pattern`](@ref) itself costs 0.023 ms of that 0.062,
# read straight off `a`'s AST. The gap widens with the mesh:
```

## Picard against NonlinearSolve, measured

The code still runs in the example's tests (its lines are marked `#src`); only the published
page drops it. The prose as it stood:

```julia
#
# ### Picard against NonlinearSolve, measured
#
# A comparison worth showing rather than only claiming: Picard from the top of this page
# against `nonlinear_problem` plus `NewtonRaphson`, each wrapped in its own top-level function
# so the timing is behind a function barrier, never over top-level globals
# (bramble-verification). Both reuse a sparsity pattern already computed once above rather than
# rediscovering it on every call -- `run_picard` refills `A`/`a` from the Picard section at the
# top of this page, `run_nonlinearsolve` reuses `J_native` -- so this measures the cost either
# method pays *given* a known pattern, not first-time pattern discovery for either one. Both
# run from a zero initial guess to their own convergence test:

function run_picard()
    uₙ .= 0.0
    αvals .= α.(Mₕ(uₙ))
    for it in 1:200
        assemble!(A, a; dirichlet = :boundary)
        unew = A \ F
        step = maximum(abs, unew .- parent(uₙ))
        uₙ .= unew
        αvals .= α.(Mₕ(uₙ))
        step < 1e-12 && break
    end
    return parent(uₙ)
end

function run_nonlinearsolve()
    solve(
        nonlinear_problem(residual!, zeros(ndofs(Wₕ)); jac_prototype = J_native),
        NewtonRaphson();
        abstol = 1e-10
    ).u
end

run_picard()
run_nonlinearsolve() # warm both before timing either
ntrials = 7
t_picard = minimum(@elapsed(run_picard()) for _ in 1:ntrials)
t_ns = minimum(@elapsed(run_nonlinearsolve()) for _ in 1:ntrials)
b_picard = @allocated run_picard()
b_ns = @allocated run_nonlinearsolve()
(picard_ms = 1000t_picard, nonlinearsolve_ms = 1000t_ns, picard_bytes = b_picard, nonlinearsolve_bytes = b_ns)

# A single machine, single process, `ntrials`-sample minimum of each, informative as a ratio
# between the two methods measured together, not as an absolute number to compare against a
# different run (bramble-benchmarks is the formal, commit-indexed baseline for that, and this
# machine was on battery power when these numbers were taken, which a same-run ratio cancels
# but an absolute number would not). `nonlinear_problem` came out both faster and lighter than
# the hand-written Picard loop here -- 0.156 ms against 0.237 ms (0.66x), 321,232 B against
# 595,792 B (0.54x) -- `NewtonRaphson`'s quadratic convergence reaching machine precision in
# fewer steps than Picard's linear rate needs, each step no more expensive than Picard's own
# `assemble!`/solve now that the sparse AD Jacobian is exactly as targeted as the hand-built
# one above.
#
# Bounded loosely (an order of magnitude either way), not pinned to today's ratio: the point  #src
# is that neither method is orders of magnitude slower than the other on this problem, not    #src
# today's precise multiplier, which a different machine or Julia version can shift.           #src
@test 0.1 < t_ns / t_picard < 10                                                             #src
@test 0.1 < b_ns / b_picard < 10                                                             #src
```
