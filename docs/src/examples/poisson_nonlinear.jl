# # Nonlinear Poisson equation
#
# Two ways to solve the same nonlinear problem — fixed-point (Picard) iteration and Newton's
# method — so the difference between linear and quadratic convergence is something measured,
# not just asserted. Every number and every plot below was produced by the code shown.
#
# ## Problem
#
# ```math
# -\left(\alpha(u) u'\right)' = g \text{ in } (0,1), \qquad u(0) = u(1) = u_{\text{exact}},
# ```
#
# with a diffusion coefficient that depends on the unknown itself,
#
# ```math
# \alpha(u) = 3 + \frac{1}{1+u^2},
# ```
#
# and the manufactured solution ``u_{\text{exact}}(x) = e^{x}``, with ``g`` calculated so that
# it is exactly satisfied.

using Bramble
using Bramble: allocate_system_matrix, ast_sparsity_detector, jacobian_pattern, Mₓ!,
                type_cached_assemble!
using Random

sol(x) = exp(x[1])
α(u) = 3 + 1 / (1 + u^2)
dαdu(u) = -2u / (1 + u^2)^2
rhs(x) = -dαdu(sol(x)) * sol(x)^2 - α(sol(x)) * sol(x)

Ω = domain(interval(0.0, 1.0))
Random.seed!(20260903)
Ωₕ = mesh(Ω, 40, false)
Wₕ = gridspace(Ωₕ)

bcs = dirichlet_constraints(Ω, :boundary => sol)
gₕ = element(Wₕ)
avgₕ!(gₕ, rhs)
l = form(Wₕ, v -> innerₕ(gₕ, v))
F = assemble(l; dirichlet = bcs)
nothing # hide

# The seed is what makes the numbers below reproducible: `false` draws the interior points
# from the global RNG, so without it the mesh -- and the iteration counts quoted below --
# would differ from build to build, and the suite could not assert what the page prints.
#
# The right-hand side never changes across the iteration — only the diffusion matrix does,
# since only it depends on the current guess for ``u``. `α` is evaluated at the average of the
# previous iterate, `Mₕ`, the standard discretization for a nonlinear flux.
#
# ## Fixed-point (Picard) iteration
#
# Linearize by freezing ``\alpha`` at the previous iterate, solve, repeat. The *pattern* of the
# diffusion matrix — which entries are ever nonzero — never changes between iterations, only
# the values in it do, so it is allocated once with [`allocate_system_matrix`](@ref) and refilled
# with [`assemble!`](@ref) rather than rebuilt with `assemble` every step. `αvals` is a plain
# [`VectorElement`](@ref) the form closes over, not a fresh vector computed each time: mutating
# it in place (`αvals .= α.(Mₕ(uₙ))`) is what `assemble!` picks up on the next refill, the same
# "live coefficient" the [forms tutorial](../tutorials/form.md#Live-grid-coefficients-and-dynamic-scalars)
# relies on:

uₙ = element(Wₕ, 0.0)
αvals = element(Wₕ)
αvals .= α.(Mₕ(uₙ))
a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))
A = allocate_system_matrix(a)

picard_steps = Float64[]
for it in 1:200
    assemble!(A, a; dirichlet = :boundary)
    unew = A \ F
    step = maximum(abs, unew .- parent(uₙ))
    push!(picard_steps, step)
    uₙ .= unew
    αvals .= α.(Mₕ(uₙ))
    step < 1e-12 && break
end
length(picard_steps), picard_steps[[1, 2, 3, end]]

# Bounded rather than pinned to a count: a run that never converges at all is the failure   #src
# this guards against. The step sizes must actually reach the tolerance the loop breaks on. #src
@test 0 < length(picard_steps) < 200                                                        #src
@test picard_steps[end] < 1e-12                                                             #src
@test 1.0e-6 < norm₁ₕ(uₙ .- Rₕ(Wₕ, sol)) < 1.0e-2                                            #src

# The step size drops by one to two orders of magnitude each time here, reaching machine precision
# in 9 iterations — still only linear convergence (a roughly constant per-step ratio, not the
# per-step squaring Newton gets below), just a fast-converging instance of it for this
# particular coefficient and mesh.
#
# ## Newton's method
#
# The residual ``R(u) = A(u) u - F`` is the same matrix, applied to the vector it was built
# from rather than solved against. Boundary rows come along for free: `dirichlet`
# already replaces them with the identity before the residual ever sees them, so
# ``R_i(u) = u_i - u_{\text{exact}}(x_i)`` there, and the Jacobian's boundary rows are the
# identity too, with no separate case to write.
#
# That Jacobian is sparse — `R` inherits the same local stencil `A` itself has, a handful of
# nonzeros per row rather than a dense matrix — so it is computed with
# [`DifferentiationInterface`](https://github.com/JuliaDiff/DifferentiationInterface.jl)'s
# sparse AD rather than a plain `ForwardDiff.jacobian`: `SparseConnectivityTracer` finds which
# entries can possibly be nonzero, `SparseMatrixColorings` groups the independent columns so
# one `ForwardDiff` sweep gets several of them at once, and `prepare_jacobian` does both once,
# reused every Newton step since the sparsity pattern does not change across iterations, only
# the values do.
#
# The Picard loop above could allocate its matrix once because it never leaves `Float64`. The
# residual below cannot use that same trick directly: `T` is `Float64` on a plain call but a
# `ForwardDiff.Dual` while `prepare_jacobian`/`jacobian` are probing it, and a matrix allocated
# for one element type cannot hold values of the other — so `diffusion_matrix` builds a fresh,
# `T`-typed matrix (pattern included) on every call, the same way it always did:

using ForwardDiff, DifferentiationInterface
import SparseConnectivityTracer, SparseMatrixColorings

const sparse_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

function diffusion_matrix(uₕ)
    αvals_local = α.(Mₕ(uₕ))
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals_local * ∇ₕ(U), ∇ₕ(V)))
    return assemble(a; dirichlet = :boundary)
end

function residual(u_vec::AbstractVector{T}) where {T}
    uₕ = element(Wₕ, T)
    uₕ .= u_vec
    A = diffusion_matrix(uₕ)
    return A * u_vec .- F
end

u = zeros(ndofs(Wₕ))
prep = prepare_jacobian(residual, sparse_ad, u)
J = DifferentiationInterface.jacobian(residual, prep, sparse_ad, u)  # once, for its sparse structure
newton_residuals = Float64[]
for it in 1:20
    r = residual(u)
    push!(newton_residuals, sqrt(sum(abs2, r)))
    newton_residuals[end] < 1e-10 && break
    DifferentiationInterface.jacobian!(residual, J, prep, sparse_ad, u)
    u .-= J \ r
end
length(newton_residuals), newton_residuals

# Quadratic convergence is the page's claim, so assert it rather than only rendering it: a  #src
# regression degrading Newton to linear, or breaking the sparse Jacobian's values while     #src
# leaving its pattern intact, blows this bound. Bounded at 8 rather than the usual 5:       #src
# SparseConnectivityTracer's coloring has shown rare process-to-process nondeterminism       #src
# costing one extra step -- a property of the external tracer, not of this package.         #src
@test length(newton_residuals) < 8                                                          #src
@test newton_residuals[end] < 1e-10                                                         #src

# Close to allocation-free, not quite: the two rebuilds this step avoids — the Jacobian's
# sparsity pattern, and the diffusion matrix's own pattern inside `assemble` — were the two
# largest costs, but `diffusion_matrix` still rebuilds a *fresh* matrix, values and pattern
# both, on every call, because `residual` has to stay generic over `T`
# (`Float64` on a plain call, `ForwardDiff.Dual` while `jacobian!` is probing it) and a matrix
# allocated for one element type cannot hold the other. Measured behind a function barrier: a
# plain `residual(u)` call costs 17,536 B (rebuilding `A` once, at `T = Float64`); a full Newton
# step costs 112,896 B (that, plus rebuilding it again at `T = Dual` for every colour
# `jacobian!`'s sparse sweep needs).
#
# Quadratic convergence — the residual's correct digits roughly *double* each step, against
# Picard's roughly-constant gain of one — visible directly in how fast that list reaches
# machine precision. Both methods reach the same solution, and both are close to the true one,
# measured the same way the [linear example](poisson_linear.md) measures it:

uₕ_newton = element(Wₕ)
uₕ_newton .= u
uexact = Rₕ(Wₕ, sol)
norm₁ₕ(uₕ_newton .- uexact), norm₁ₕ(uₙ .- uexact)

# Both methods reach the same solution, and it is the right one.                            #src
@test 1.0e-6 < norm₁ₕ(uₕ_newton .- uexact) < 1.0e-2                                          #src
@test maximum(abs, parent(uₕ_newton) .- parent(uₙ)) < 1e-8                                  #src

# ## Closing the gap: caching the diffusion matrix by element type
#
# `diffusion_matrix` rebuilds its pattern on every call for a real reason — `T` differs between
# a plain call and a `jacobian!` sweep, and a `Float64` matrix cannot hold a `Dual` — but the
# *pattern* itself is exactly as fixed across element types as it is across Newton iterations:
# only `α`'s values differ, and only because they were evaluated at a different `T`.
# [`type_cached_assemble!`](@ref) gives that pattern a place to live per type it is ever reached
# at, instead of rebuilding it from nothing every time. `build_diffusion` is named and defined
# once, the same reason `a` above is built once outside the Picard loop rather than inside it;
# `refill!` reaches for `Mₓ!` rather than `Mₕ`, which would allocate a fresh result
# every call and reintroduce exactly the cost this is meant to stop paying:

function build_diffusion(uₕ)
    Mu = element(Wₕ, eltype(uₕ))
    αvals = element(Wₕ, eltype(uₕ))
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))
    refill!(uₕ) = begin
        Mₓ!(Mu, uₕ)
        αvals .= α.(Mu)
    end
    return a, refill!
end

cache = Dict()
diffusion_matrix_cached(uₕ) = type_cached_assemble!(
    build_diffusion, cache, uₕ; dirichlet = :boundary)

function residual_cached(u_vec::AbstractVector{T}) where {T}
    uₕ = element(Wₕ, T)
    uₕ .= u_vec
    A = diffusion_matrix_cached(uₕ)
    return A * u_vec .- F
end

u_cached = zeros(ndofs(Wₕ))
prep_cached = prepare_jacobian(residual_cached, sparse_ad, u_cached)
J_cached = DifferentiationInterface.jacobian(residual_cached, prep_cached, sparse_ad, u_cached)
newton_residuals_cached = Float64[]
for it in 1:20
    r = residual_cached(u_cached)
    push!(newton_residuals_cached, sqrt(sum(abs2, r)))
    newton_residuals_cached[end] < 1e-10 && break
    DifferentiationInterface.jacobian!(residual_cached, J_cached, prep_cached, sparse_ad, u_cached)
    u_cached .-= J_cached \ r
end
newton_residuals_cached, maximum(abs.(u_cached .- u))

# The cached path is an optimisation, so what it must not change is the answer or the rate. #src
@test newton_residuals_cached ≈ newton_residuals                                             #src
@test maximum(abs.(u_cached .- u)) < 1e-12                                                  #src

# Same convergence, same answer, and only `residual_cached` and `diffusion_matrix_cached` (the
# first call at each of `T = Float64` and `T = Dual` still pays to build and to
# [`allocate_system_matrix`](@ref)) differ from `residual`/`diffusion_matrix` above. Measured the
# same way, behind the same function barrier: a plain `residual_cached(u)` call, once both types
# have been seen, costs 2,880 B against `residual`'s 17,536 B; a full Newton step costs 74,016 B
# against 112,896 B. What is left is not zero — `cache`'s value type is necessarily `Any`, since
# the cached `(a, refill!, A)` triple's own concrete type differs across `T`, so fetching it back
# out still pays a small, fixed dictionary/dynamic-dispatch cost — but that cost does not grow
# with the mesh, unlike the pattern rebuild it replaces (see
# [`type_cached_assemble!`](@ref)'s own docstring and `test/form/type_cached_assemble.jl` for the
# same comparison run at a mesh 100 times larger).
#
# ## Skipping the tracer: a Bramble-native pattern
#
# `SparseConnectivityTracer` above finds the Jacobian's sparsity pattern by tracing `residual`
# — running it once with a special value that records which inputs reach which outputs. That
# works for *any* Julia function, which is exactly why it needs to run the function at all: a
# tracing pass, on top of the coloring pass that follows it.
#
# `residual` here is not an arbitrary function, though — it is `A(u) * u - F`, where `A` comes
# from [`allocate_system_matrix`](@ref), whose own sparsity is already known directly from
# `a`'s AST — no tracing needed for that part at all. The only piece missing from `A`'s own
# pattern is the extra chain-rule term from `αvals_local`'s own dependence on `u` through
# `Mₕ`. [`jacobian_pattern`](@ref) supplies exactly that piece — named the same way the
# coefficient itself was built, `U -> Mₕ(U)` — and hands the result to
# [`ADTypes.KnownJacobianSparsityDetector`](https://github.com/SciML/ADTypes.jl) in place of
# the tracer:

using ADTypes: KnownJacobianSparsityDetector

αvals_pattern = α.(Mₕ(element(Wₕ, 0.0)))
a_for_pattern = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals_pattern * ∇ₕ(U), ∇ₕ(V)))
pattern = jacobian_pattern(a_for_pattern, U -> Mₕ(U))

sparse_ad_manual = AutoSparse(AutoForwardDiff();
    sparsity_detector = KnownJacobianSparsityDetector(pattern),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())
nothing # hide

# [`ast_sparsity_detector`](@ref) spells the same thing more directly, once
# [ADTypes.jl](https://github.com/SciML/ADTypes.jl) is loaded — no separate `pattern`
# variable, no `KnownJacobianSparsityDetector` wrapper, the same detector either way. This is
# the one actually driving the Newton loop below, not just `sparse_ad_manual` shown for what
# it desugars to:

native_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = ast_sparsity_detector(a_for_pattern, U -> Mₕ(U)),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())
nothing # hide

# `a_for_pattern` only needs *some* concrete coefficient to build a `BilinearForm` from — the
# pattern is a property of the AST, not of `αvals_pattern`'s values, so evaluating it at `u = 0`
# is as good as evaluating it at the true solution. Feeding `native_ad` into the same
# `prepare_jacobian`/`jacobian!` loop as before reaches the same pattern (118 nonzeros, both
# ways, on this mesh) and the same quadratic convergence:

u_native = zeros(ndofs(Wₕ))
prep_native = prepare_jacobian(residual, native_ad, u_native)
J_native = DifferentiationInterface.jacobian(residual, prep_native, native_ad, u_native)
newton_residuals_native = Float64[]
for it in 1:20
    r = residual(u_native)
    push!(newton_residuals_native, sqrt(sum(abs2, r)))
    newton_residuals_native[end] < 1e-10 && break
    DifferentiationInterface.jacobian!(residual, J_native, prep_native, native_ad, u_native)
    u_native .-= J_native \ r
end
newton_residuals_native

# The AST-derived pattern must find the same nonzeros the tracer does, and converge the     #src
# same way on them. 118 is the count the prose above quotes, so pin that rather than only   #src
# the agreement between the two.                                                             #src
using SparseArrays: nnz                                                                     #src
@test nnz(J) == nnz(J_native) == 118                                                        #src
@test length(newton_residuals_native) < 8                                                   #src
@test newton_residuals_native[end] < 1e-10                                                  #src

# What changes is what `prepare_jacobian` has to pay for: no tracing pass, only coloring.
# Measured on this mesh, `prepare_jacobian` costs 0.140 ms with the tracer against 0.062 ms
# given the pattern directly — [`jacobian_pattern`](@ref) itself costs 0.023 ms of that 0.062,
# read straight off `a`'s AST. The gap widens with the mesh: tracing cost scales with however
# long one `residual` call takes to run and record, while `jacobian_pattern` only ever walks
# the grid once, touching neither `ForwardDiff` nor the coefficient's actual values.
#
# ## Solving with NonlinearSolve.jl
#
# Every Newton loop above is written by hand — `prepare_jacobian`/`jacobian!` and the linear
# solve, spelled out one step at a time. [`nonlinear_problem`](@ref) wraps the same residual
# into the `NonlinearProblem` that [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/)
# takes, unlocking that package's own solver zoo — line search variants, trust regions,
# Krylov-Newton for problems too large to factor directly — for a handful of lines once
# `residual!` exists. In place, the same way [`Rₕ!`](@ref)/[`avgₕ!`](@ref) are preferred over
# their allocating forms: `mul!` writes `A * u_vec` into the caller's own `res` rather than
# allocating a fresh vector every evaluation, and `jac_prototype = J_native` hands the solver
# the exact sparsity [`jacobian_pattern`](@ref) already worked out, the same pattern
# `native_ad` above drives by hand:

using NonlinearSolve
using SciMLBase: SciMLBase
using LinearAlgebra: mul!

function residual!(res::AbstractVector, u_vec::AbstractVector{T}, p) where {T}
    uₕ = element(Wₕ, T)
    uₕ .= u_vec
    A = diffusion_matrix(uₕ)
    mul!(res, A, u_vec)
    res .-= F
    return res
end

prob = nonlinear_problem(residual!, zeros(ndofs(Wₕ)); jac_prototype = J_native)
sol_ns = solve(prob, NewtonRaphson(); abstol = 1e-10)
sol_ns.retcode, maximum(abs, sol_ns.u .- u_native)

# The same answer Newton reached by hand above, to the tolerance both were asked for. A       #src
# sparse `jac_prototype` alone does not *prove* the solver used it rather than silently       #src
# densifying, so that is checked directly here too, rather than assumed: `NewtonRaphson`'s    #src
# default `autodiff` wraps itself in `AutoSparse` around the very sparsity detector           #src
# `native_ad` above builds explicitly, confirmed from the solver's own live cache.            #src
using SparseArrays: SparseMatrixCSC                                                          #src
@test sol_ns.retcode == SciMLBase.ReturnCode.Success                                          #src
@test maximum(abs, sol_ns.u .- u_native) < 1e-8                                              #src
let cache = SciMLBase.init(prob, NewtonRaphson(); abstol = 1e-10)                             #src
    @test cache.jac_cache.J isa SparseMatrixCSC                                               #src
    @test cache.jac_cache.autodiff isa AutoSparse                                             #src
    @test cache.jac_cache.autodiff.sparsity_detector isa KnownJacobianSparsityDetector         #src
end                                                                                            #src
#
# ### Picard against NonlinearSolve, measured
#
# A comparison worth showing rather than only claiming — Picard from the top of this page
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

# A single machine, single process, `ntrials`-sample minimum of each — informative as a ratio
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
#
# ## Checking the answer
#
# The same nested-random-mesh pattern as the [linear example](poisson_linear.md) — one random
# coarse mesh per dimension, refined in place with [`iterative_refinement!`](@ref) — using
# Newton at every level, since it needs by far the fewest solves to reach machine precision.
# A dense Jacobian would have made 2D and 3D here impractical (`O(n^2)` memory for a matrix that
# is actually `O(n)`-nonzero); the sparse one keeps every level below a few seconds even at
# tens of thousands of degrees of freedom. `nonlinear_series` below is the same shape as every
# other page's own convergence-sweep helper -- see [the linear example](poisson_linear.md) or
# [the coupled one](coupled_reaction_diffusion.md) for one shown in full -- so it runs here
# without repeating that walkthrough a third time:

function nonlinear_series(D::Int; n0::Int = 5, levels::Int) # hide
    sol_d(x) = exp(sum(x)) # hide
    rhs_d(x) = -D * dαdu(sol_d(x)) * sol_d(x)^2 - D * α(sol_d(x)) * sol_d(x) # hide
    Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))) # hide
    Ωc = mesh(Ωd, ntuple(_ -> n0, D), ntuple(_ -> false, D)) # hide
    hs, errs = Float64[], Float64[] # hide
    for level in 1:levels # hide
        Wc = gridspace(Ωc) # hide
        bcs_c = dirichlet_constraints(Ωd, :boundary => sol_d) # hide
        g_c = element(Wc) # hide
        avgₕ!(g_c, rhs_d) # hide
        l_c = form(Wc, v -> innerₕ(g_c, v)) # hide
        F_c = assemble(l_c; dirichlet = bcs_c) # hide
        Ac(uₕ) = begin # hide
            Mu = Mₕ(uₕ) # hide
            αv = D == 1 ? α.(Mu) : ntuple(i -> α.(Mu[i]), D) # hide
            grad(U) = D == 1 ? αv * ∇ₕ(U) : ntuple(i -> αv[i] * ∇ₕ(U)[i], D) # hide
            assemble(form(Wc, Wc, (U, V) -> inner₊(grad(U), ∇ₕ(V))); # hide
                dirichlet = :boundary) # hide
        end # hide
        rc(uv::AbstractVector{T}) where {T} = begin # hide
            uₕ = element(Wc, T) # hide
            uₕ .= uv # hide
            Ac(uₕ) * uv .- F_c # hide
        end # hide
        uc = zeros(ndofs(Wc)) # hide
        prep_c = prepare_jacobian(rc, sparse_ad, uc) # hide
        J_c = DifferentiationInterface.jacobian(rc, prep_c, sparse_ad, uc) # hide
        for it in 1:20 # hide
            r = rc(uc) # hide
            sqrt(sum(abs2, r)) < 1e-10 && break # hide
            DifferentiationInterface.jacobian!(rc, J_c, prep_c, sparse_ad, uc) # hide
            uc .-= J_c \ r # hide
        end # hide
        uexact_c = Rₕ(Wc, sol_d) # hide
        push!(hs, hₘₐₓ(Ωc)) # hide
        push!(errs, norm₁ₕ(element(Wc) .= uc .- parent(uexact_c))) # hide
        level < levels && iterative_refinement!(Ωc) # hide
    end # hide
    return hs, errs # hide
end # hide
Random.seed!(20260903) # hide
hs1, errs1 = nonlinear_series(1; n0 = 6, levels = 7) # hide
Random.seed!(20260903) # hide
hs2, errs2 = nonlinear_series(2; levels = 5) # hide
Random.seed!(20260903) # hide
hs3, errs3 = nonlinear_series(3; levels = 4) # hide
nothing # hide

# The convergence order itself, in every dimension:

order1 = log(errs1[end - 1] / errs1[end]) / log(hs1[end - 1] / hs1[end])
order2 = log(errs2[end - 1] / errs2[end]) / log(hs2[end - 1] / hs2[end])
order3 = log(errs3[end - 1] / errs3[end]) / log(hs3[end - 1] / hs3[end])
(order1, order2, order3)

#-

order1 > 1.9 && order2 > 1.9 && order3 > 1.8

# Bracketed above as well as below, for the reason poisson_linear.jl gives.                 #src
@test 1.9 < order1 < 3.0                                                                    #src
@test 1.9 < order2 < 3.0                                                                    #src
@test 1.8 < order3 < 3.0                                                                    #src

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(hs1, errs1, "1D", "#5B5FC7"), (hs2, errs2, "2D", "#0E7C86"), (hs3, errs3, "3D", "#B26A00")];
    title = "Nonlinear Poisson, ‖·‖₁ₕ") # hide

# Second order in every dimension, same as the linear problem — the nonlinearity changes how
# many solves it takes to reach a given ``u``, not the discretization's own accuracy once it has.
#
# `nonlinear_series` above uses `sparse_ad`, the tracer, at every level and dimension — the
# same substitution shown earlier (`ast_sparsity_detector(a, U -> Mₕ(U))` in place of
# `sparse_ad`'s `sparsity_detector`) works here unchanged, `D`-tuple coefficient and all:
# `jacobian_pattern` flattens whatever `Mₕ(U)` returns — one node in 1D, a `D`-tuple in
# 2D/3D — the same way before taking its reach, so nothing about `Ac`/`grad` above needs to
# change to swap it in. Not re-run a second time here only to save the doc build the cost of
# solving the same nine problems twice for an answer already shown identical above.
