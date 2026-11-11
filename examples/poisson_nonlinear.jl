# # Nonlinear Poisson equation
#
# Fixed-point (Picard) iteration and Newton's method on the same nonlinear problem, compared
# by their measured convergence.
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
#
# ## The mesh and the load

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

# `false` draws the interior points from the global RNG, so the seed fixes the mesh and with
# it the iteration counts below.
#
# Only the diffusion matrix depends on the current iterate. `α` is evaluated at the average of
# the previous iterate, `Mₕ`, the standard discretization for a nonlinear flux.
#
# ## Fixed-point (Picard) iteration
#
# Linearize by freezing ``\alpha`` at the previous iterate, solve, repeat. The matrix pattern
# is fixed, so it is allocated once with [`allocate_system_matrix`](@ref) and refilled with
# [`assemble!`](@ref). `αvals` is a [`VectorElement`](@ref) the form closes over; updating it
# in place (`αvals .= α.(Mₕ(uₙ))`) is what the next `assemble!` reads, the same
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

# The ratio of each step to the one before it is the convergence rate of the iteration:

picard_ratios = picard_steps[2:end] ./ picard_steps[1:(length(picard_steps) - 1)]
extrema(picard_ratios)

# A slower rate, or a run that never converges, is the failure guarded against: the step    #src
# sizes must reach the tolerance the loop breaks on and shrink by more than a factor of     #src
# ten at every step.                                                                        #src
@test 0 < length(picard_steps) <= 10                                                        #src
@test all(<(0.1), picard_ratios)                                                            #src
@test picard_steps[end] < 1e-12                                                             #src
@test 1.0e-6 < norm₁ₕ(uₙ .- Rₕ(Wₕ, sol)) < 1.0e-2                                            #src

# The step shrinks by a factor of 0.010 to 0.027 per iteration, one to two orders of
# magnitude, and falls below 1e-12 after 9 iterations. A roughly constant ratio is linear
# convergence.
#
# ## Newton's method
#
# The residual ``R(u) = A(u) u - F`` is the same matrix multiplied by the vector it was built
# from. `dirichlet` already replaces the boundary rows with the identity, so there
# ``R_i(u) = u_i - u_{\text{exact}}(x_i)`` and the Jacobian rows are the identity.
#
# `R` has the local stencil of `A`, a handful of nonzeros per row, so its Jacobian is sparse
# and is computed with
# [`DifferentiationInterface`](https://github.com/JuliaDiff/DifferentiationInterface.jl)'s
# sparse AD: `SparseConnectivityTracer` finds which entries can be nonzero,
# `SparseMatrixColorings` groups the independent columns so one `ForwardDiff` sweep gets
# several of them at once, and `prepare_jacobian` does both once. The pattern is the same at
# every Newton step, so only the values are recomputed.
#
# The Picard loop above could allocate its matrix once because it never leaves `Float64`. The
# residual below cannot use that same trick directly: `T` is `Float64` on a plain call but a
# `ForwardDiff.Dual` while `prepare_jacobian`/`jacobian` are probing it, and a matrix allocated
# for one element type cannot hold values of the other, so `diffusion_matrix` builds a fresh,
# `T`-typed matrix (pattern included) on every call:

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
newton_steps = Float64[]
for it in 1:20
    r = residual(u)
    push!(newton_residuals, sqrt(sum(abs2, r)))
    newton_residuals[end] < 1e-10 && break
    DifferentiationInterface.jacobian!(residual, J, prep, sparse_ad, u)
    δ = J \ r
    push!(newton_steps, maximum(abs, δ))
    u .-= δ
end
length(newton_residuals), newton_residuals

# A regression degrading Newton to linear, or breaking the sparse Jacobian's values while   #src
# leaving its pattern intact, blows this bound. The bound is 8 where 5 would do:            #src
# SparseConnectivityTracer's coloring has shown rare process-to-process nondeterminism      #src
# that costs one extra step, a property of the external tracer.                             #src
@test length(newton_residuals) < 8                                                          #src
@test newton_residuals[end] < 1e-10                                                         #src
# The plotted histories: Newton's last step is below the 3/2 power of the one before it,    #src
# which a fixed rate cannot do, and Picard's step ratios stay above 1e-3.                    #src
@test length(newton_steps) == length(newton_residuals) - 1                                   #src
@test newton_steps[end] < newton_steps[end - 1]^1.5                                          #src
@test minimum(picard_ratios) > 1e-3                                                          #src

# Both histories on one log axis, as the largest change in the iterate per iteration. Picard
# falls along a line, a fixed one to two digits per step. Newton's correct digits roughly
# double each step, so its curve bends down and it stops sooner:

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
line_plot([(1:length(picard_steps), picard_steps, "Picard"), # hide
        (1:length(newton_steps), newton_steps, "Newton")]; # hide
    logy = true, xlabel = "iteration", ylabel = "largest change in u", # hide
    title = "Picard against Newton") # hide

# Both methods reach the same solution, close to the true one, measured as in the
# [linear example](poisson_linear.md):

uₕ_newton = element(Wₕ)
uₕ_newton .= u
uexact = Rₕ(Wₕ, sol)
norm₁ₕ(uₕ_newton .- uexact), norm₁ₕ(uₙ .- uexact)

# Both methods reach the same solution, and it is the right one.                            #src
@test 1.0e-6 < norm₁ₕ(uₕ_newton .- uexact) < 1.0e-2                                          #src
@test maximum(abs, parent(uₕ_newton) .- parent(uₙ)) < 1e-8                                  #src

# ## The assembled Jacobian
#
# [`Bramble.jacobian_form`](@ref) builds the same Jacobian from the form's AST, as one form
# `a + ∂a` assembled once per state. The state enters through one coefficient, `αx`, declared
# with the operator it is computed from, `U -> Mₕ(U)`, and the scalar function `α`, so that
# `αx == α.(Mₕ(u))` at every state:

αx = element(Wₕ)
a_asm = form(Wₕ, Wₕ, (U, V) -> inner₊(αx * ∇ₕ(U), ∇ₕ(V)))
jf = Bramble.jacobian_form(a_asm, αx => (U -> Mₕ(U), α); dirichlet = :boundary)
J_assembled = Bramble.jacobian_prototype(jf)
Bramble.jacobian!(J_assembled, jf, u)
J_fd = DifferentiationInterface.jacobian(residual, prep, sparse_ad, u)
maximum(abs, J_assembled - J_fd) / maximum(abs, J_fd)

# No `Dual` passes through assembly, so prefer the assembled Jacobian when assembly must stay
# in `Float64`.
#
# Newton with the assembled Jacobian reaches the same solution in the same number of iterations:

u_asm = zeros(ndofs(Wₕ))
J_newton = Bramble.jacobian_prototype(jf)
asm_residuals = Float64[]
for it in 1:20
    r = residual(u_asm)
    push!(asm_residuals, sqrt(sum(abs2, r)))
    asm_residuals[end] < 1e-10 && break
    Bramble.jacobian!(J_newton, jf, u_asm)
    u_asm .-= J_newton \ r
end
length(asm_residuals), maximum(abs, u_asm .- u)

# The assembled Jacobian must agree with the sparse-FD one to rounding, and Newton with it     #src
# must take the same number of iterations to the same solution. The error is relative to the  #src
# largest entry of `J_fd`.                                                                    #src
@test maximum(abs, J_assembled - J_fd) / maximum(abs, J_fd) < 1e-12                          #src
@test length(asm_residuals) == length(newton_residuals)                                       #src
@test maximum(abs, u_asm .- u) < 1e-10                                                        #src

# ## Closing the gap: caching the diffusion matrix by element type
#
# `diffusion_matrix` rebuilds its pattern on every call because `T` differs between a plain
# call and a `jacobian!` sweep. The pattern is as fixed across element types as it is across
# Newton iterations: only `α`'s values differ, because they were evaluated at a different `T`.
# [`type_cached_assemble!`](@ref) keeps one pattern per type it is reached at.
# `build_diffusion` is defined once, as `a` above is built once outside the Picard loop.
# `refill!` calls `Mₓ!`, which writes in place; `Mₕ` would allocate a result on every call:

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

# The cached path is an optimisation: it must change neither the answer nor the rate.       #src
@test newton_residuals_cached ≈ newton_residuals                                             #src
@test maximum(abs.(u_cached .- u)) < 1e-12                                                  #src

# Same convergence, same answer. The first call at each of `T = Float64` and `T = Dual` builds
# the form and calls [`allocate_system_matrix`](@ref); every later call refills that matrix in
# place. The cache's value type is `Any`, since the cached `(a, refill!, A)` triple has a
# different concrete type per `T`, so fetching an entry costs a small fixed dispatch,
# independent of the mesh (see [`type_cached_assemble!`](@ref)).
#
# ## Skipping the tracer
#
# `SparseConnectivityTracer` finds the Jacobian's sparsity pattern by tracing `residual`:
# running it once with a special value that records which inputs reach which outputs. That
# works for any Julia function, at the price of a tracing pass before the coloring pass.
#
# `residual` is `A(u) * u - F`, and [`allocate_system_matrix`](@ref) already knows the
# sparsity of `A` from `a`'s AST. The only piece missing is the chain-rule term from
# `αvals_local`'s dependence on `u` through `Mₕ`. [`jacobian_pattern`](@ref) supplies it,
# named as the coefficient was built, `U -> Mₕ(U)`, and the result goes to
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

# [`ast_sparsity_detector`](@ref) builds the same detector in one call once
# [ADTypes.jl](https://github.com/SciML/ADTypes.jl) is loaded, with no separate `pattern`
# variable or `KnownJacobianSparsityDetector` wrapper. `native_ad` drives the Newton loop
# below; `sparse_ad_manual` shows what it expands to:

native_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = ast_sparsity_detector(a_for_pattern, U -> Mₕ(U)),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())
nothing # hide

# `a_for_pattern` takes any concrete coefficient, because the pattern is a property of the AST,
# so `u = 0` is as good as the true solution. `native_ad` in the same
# `prepare_jacobian`/`jacobian!` loop gives the same pattern (118 nonzeros both ways, on this
# mesh) and the same quadratic convergence:

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
# same way on them. The prose quotes 118, so the count is pinned as well as the             #src
# agreement between the two.                                                                #src
using SparseArrays: nnz                                                                     #src
@test nnz(J) == nnz(J_native) == 118                                                        #src
@test length(newton_residuals_native) < 8                                                   #src
@test newton_residuals_native[end] < 1e-10                                                  #src

# `prepare_jacobian` now does only the coloring, with the pattern read by
# [`jacobian_pattern`](@ref) off `a`'s AST. The tracer remains the option when the residual's
# matrix does not come from a `BilinearForm`.
#
# ## Solving with NonlinearSolve.jl
#
# [`nonlinear_problem`](@ref) wraps the residual into the `NonlinearProblem` that
# [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/) takes, which gives access
# to that package's solvers, such as line search variants, trust regions and Krylov-Newton for
# problems too large to factor directly. `residual!` is the in-place form, as
# [`Rₕ!`](@ref) and [`avgₕ!`](@ref) are of their allocating counterparts, since `mul!` writes
# `A * u_vec` into the caller's `res` instead of allocating it. `jac_prototype = J_native`
# hands the solver the sparsity [`jacobian_pattern`](@ref) worked out, the pattern `native_ad`
# uses above:

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

# The same answer Newton reached by hand above, to the tolerance both were asked for. A     #src
# sparse `jac_prototype` alone does not prove the solver used it and did not silently       #src
# densify, so the solver's own live cache is checked: `NewtonRaphson`'s default `autodiff`  #src
# wraps itself in `AutoSparse` around the very sparsity detector `native_ad` above builds   #src
# explicitly.                                                                               #src
using SparseArrays: SparseMatrixCSC                                                          #src
@test sol_ns.retcode == SciMLBase.ReturnCode.Success                                          #src
@test maximum(abs, sol_ns.u .- u_native) < 1e-8                                              #src
let cache = SciMLBase.init(prob, NewtonRaphson(); abstol = 1e-10)                             #src
    @test cache.jac_cache.J isa SparseMatrixCSC                                               #src
    @test cache.jac_cache.autodiff isa AutoSparse                                             #src
    @test cache.jac_cache.autodiff.sparsity_detector isa KnownJacobianSparsityDetector         #src
end                                                                                            #src

# ## Convergence of the Newton solution
#
# The same nested-random-mesh pattern as the [linear example](poisson_linear.md): one random
# coarse mesh per dimension, refined in place with [`iterative_refinement!`](@ref), using
# Newton at every level, since it needs the fewest solves. A dense Jacobian would need `O(n^2)`
# memory for a matrix with `O(n)` nonzeros, which makes 2D and 3D impractical; the sparse one
# stores only the nonzeros. `nonlinear_series` is the convergence-sweep helper of the
# [linear example](poisson_linear.md), with Newton at each level:

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

# Second order in every dimension, as for the linear problem. The nonlinearity changes the
# number of solves; the discretization stays second order.
#
# `nonlinear_series` uses `sparse_ad`, the tracer, at every level and dimension.
# `ast_sparsity_detector(a, U -> Mₕ(U))` also covers 2D and 3D: `jacobian_pattern` flattens
# the `D`-tuple that `Mₕ(U)` returns.
#
# ## Where to go next
#
#   - [Coupled reaction-diffusion](coupled_reaction_diffusion.md) for the same Newton setup
#     over a composite space, with two species.
#   - [`nonlinear_problem`](@ref) in the [API reference](../api.md) for the
#     NonlinearSolve.jl wrapper used above.
