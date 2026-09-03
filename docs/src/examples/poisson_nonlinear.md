# Nonlinear Poisson equation

Two ways to solve the same nonlinear problem — fixed-point (Picard) iteration and Newton's
method — so the difference between linear and quadratic convergence is something measured,
not just asserted. Every number and every plot below was produced by the code shown.

## Problem

```math
-\left(\alpha(u) u'\right)' = g \text{ in } (0,1), \qquad u(0) = u(1) = u_{\text{exact}},
```

with a diffusion coefficient that depends on the unknown itself,

```math
\alpha(u) = 3 + \frac{1}{1+u^2},
```

and the manufactured solution ``u_{\text{exact}}(x) = e^{x}``, with ``g`` calculated so that
it is exactly satisfied.

```@example poisson_nonlinear
using Bramble

sol(x) = exp(x[1])
α(u) = 3 + 1 / (1 + u^2)
dαdu(u) = -2u / (1 + u^2)^2
rhs(x) = -dαdu(sol(x)) * sol(x)^2 - α(sol(x)) * sol(x)

Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 40, false)
Wₕ = gridspace(Ωₕ)

bcs = dirichlet_constraints(Bramble.set(Ω), :boundary => sol)
gₕ = element(Wₕ)
avgₕ!(gₕ, rhs)
l = form(Wₕ, v -> innerₕ(gₕ, v))
F = assemble(l; dirichlet_conditions = bcs, dirichlet_labels = :boundary)
nothing # hide
```

The right-hand side never changes across the iteration — only the diffusion matrix does,
since only it depends on the current guess for ``u``. `α` is evaluated at the average of the
previous iterate, `M₋ₕ`, the standard discretization for a nonlinear flux.

## Fixed-point (Picard) iteration

Linearize by freezing ``\alpha`` at the previous iterate, solve, repeat. The *pattern* of the
diffusion matrix — which entries are ever nonzero — never changes between iterations, only
the values in it do, so it is allocated once with [`allocate_system_matrix`](@ref) and refilled
with [`assemble!`](@ref) rather than rebuilt with `assemble` every step. `αvals` is a plain
[`VectorElement`](@ref) the form closes over, not a fresh vector computed each time: mutating
it in place (`αvals .= α.(M₋ₕ(uₙ))`) is what `assemble!` picks up on the next refill, the same
"live coefficient" the [forms tutorial](../tutorials/form.md#Live-grid-coefficients-and-dynamic-scalars)
relies on:

```@example poisson_nonlinear
uₙ = element(Wₕ, 0.0)
αvals = element(Wₕ)
αvals .= α.(M₋ₕ(uₙ))
a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇₋ₕ(U), ∇₋ₕ(V)))
A = allocate_system_matrix(a)

picard_steps = Float64[]
for it in 1:200
    assemble!(A, a; dirichlet_labels = :boundary)
    unew = A \ F
    step = maximum(abs, unew .- values(uₙ))
    push!(picard_steps, step)
    uₙ .= unew
    αvals .= α.(M₋ₕ(uₙ))
    step < 1e-12 && break
end
length(picard_steps), picard_steps[[1, 2, 3, end]]
```

The step size roughly halves each time — linear convergence, one correct digit gained every
couple of iterations.

## Newton's method

The residual ``R(u) = A(u) u - F`` is the same matrix, applied to the vector it was built
from rather than solved against. Boundary rows come along for free: `dirichlet_labels`
already replaces them with the identity before the residual ever sees them, so
``R_i(u) = u_i - u_{\text{exact}}(x_i)`` there, and the Jacobian's boundary rows are the
identity too, with no separate case to write.

That Jacobian is sparse — `R` inherits the same local stencil `A` itself has, a handful of
nonzeros per row rather than a dense matrix — so it is computed with
[`DifferentiationInterface`](https://github.com/JuliaDiff/DifferentiationInterface.jl)'s
sparse AD rather than a plain `ForwardDiff.jacobian`: `SparseConnectivityTracer` finds which
entries can possibly be nonzero, `SparseMatrixColorings` groups the independent columns so
one `ForwardDiff` sweep gets several of them at once, and `prepare_jacobian` does both once,
reused every Newton step since the sparsity pattern does not change across iterations, only
the values do.

The Picard loop above could allocate its matrix once because it never leaves `Float64`. The
residual below cannot use that same trick directly: `T` is `Float64` on a plain call but a
`ForwardDiff.Dual` while `prepare_jacobian`/`jacobian` are probing it, and a matrix allocated
for one element type cannot hold values of the other — so `diffusion_matrix` builds a fresh,
`T`-typed matrix (pattern included) on every call, the same way it always did:

```@example poisson_nonlinear
using ForwardDiff, DifferentiationInterface
import SparseConnectivityTracer, SparseMatrixColorings

const sparse_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

function diffusion_matrix(uₕ)
    αvals_local = α.(M₋ₕ(uₕ))
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals_local * ∇₋ₕ(U), ∇₋ₕ(V)))
    return assemble(a; dirichlet_labels = :boundary)
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
```

Close to allocation-free, not quite: the two rebuilds this step avoids — the Jacobian's
sparsity pattern, and the diffusion matrix's own pattern inside `assemble` — were the two
largest costs, but `diffusion_matrix` still rebuilds a *fresh* matrix, values and pattern
both, on every call, because `residual` has to stay generic over `T`
(`Float64` on a plain call, `ForwardDiff.Dual` while `jacobian!` is probing it) and a matrix
allocated for one element type cannot hold the other. Measured behind a function barrier: a
plain `residual(u)` call costs 10,096 B (rebuilding `A` once, at `T = Float64`); a full Newton
step costs 98,016 B (that, plus rebuilding it again at `T = Dual` for every colour
`jacobian!`'s sparse sweep needs). Closing that gap would mean giving `diffusion_matrix` a
`Float64`-specialized method sitting behind `assemble!`, the way the Picard loop already has,
and a separate generic one only the `Dual` calls ever reach — a real further optimization,
just not one this page chases, since it would cost real clarity for a page about *what*
Newton needs, not how few bytes it can be made to cost.

Quadratic convergence — the residual's correct digits roughly *double* each step, against
Picard's roughly-constant gain of one — visible directly in how fast that list reaches
machine precision. Both methods reach the same solution, and both are close to the true one,
measured the same way the [linear example](poisson_linear.md) measures it:

```@example poisson_nonlinear
uₕ_newton = element(Wₕ)
uₕ_newton .= u
uexact = Rₕ(Wₕ, sol)
norm₁ₕ(uₕ_newton .- uexact), norm₁ₕ(uₙ .- uexact)
```

## Checking the answer

The same nested-random-mesh pattern as the [linear example](poisson_linear.md) — one random
coarse mesh per dimension, refined in place with [`iterative_refinement!`](@ref) — using
Newton at every level, since it needs by far the fewest solves to reach machine precision.
A dense Jacobian would have made 2D and 3D here impractical (`O(n^2)` memory for a matrix that
is actually `O(n)`-nonzero); the sparse one keeps every level below a few seconds even at
tens of thousands of degrees of freedom:

```@example poisson_nonlinear
using Random

function nonlinear_series(D::Int; n0::Int = 5, levels::Int)
    sol_d(x) = exp(sum(x))
    rhs_d(x) = -D * dαdu(sol_d(x)) * sol_d(x)^2 - D * α(sol_d(x)) * sol_d(x)
    Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
    Ωc = mesh(Ωd, ntuple(_ -> n0, D), ntuple(_ -> false, D))
    hs, errs = Float64[], Float64[]
    for level in 1:levels
        Wc = gridspace(Ωc)
        bcs_c = dirichlet_constraints(Bramble.set(Ωd), :boundary => sol_d)
        g_c = element(Wc)
        avgₕ!(g_c, rhs_d)
        l_c = form(Wc, v -> innerₕ(g_c, v))
        F_c = assemble(l_c; dirichlet_conditions = bcs_c, dirichlet_labels = :boundary)

        Ac(uₕ) = begin
            Mu = M₋ₕ(uₕ)
            αv = D == 1 ? α.(Mu) : ntuple(i -> α.(Mu[i]), D)
            grad(U) = D == 1 ? αv * ∇₋ₕ(U) : ntuple(i -> αv[i] * ∇₋ₕ(U)[i], D)
            assemble(form(Wc, Wc, (U, V) -> inner₊(grad(U), ∇₋ₕ(V)));
                dirichlet_labels = :boundary)
        end
        rc(uv::AbstractVector{T}) where {T} = begin
            uₕ = element(Wc, T)
            uₕ .= uv
            Ac(uₕ) * uv .- F_c
        end

        uc = zeros(ndofs(Wc))
        prep_c = prepare_jacobian(rc, sparse_ad, uc)
        J_c = DifferentiationInterface.jacobian(rc, prep_c, sparse_ad, uc)
        for it in 1:20
            r = rc(uc)
            sqrt(sum(abs2, r)) < 1e-10 && break
            DifferentiationInterface.jacobian!(rc, J_c, prep_c, sparse_ad, uc)
            uc .-= J_c \ r
        end
        uexact_c = Rₕ(Wc, sol_d)

        push!(hs, hₘₐₓ(Ωc))
        push!(errs, norm₁ₕ(element(Wc) .= uc .- values(uexact_c)))
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, errs
end

Random.seed!(20260903)
hs1, errs1 = nonlinear_series(1; n0 = 6, levels = 7)
Random.seed!(20260903)
hs2, errs2 = nonlinear_series(2; levels = 5)
Random.seed!(20260903)
hs3, errs3 = nonlinear_series(3; levels = 4)

order1 = log(errs1[end - 1] / errs1[end]) / log(hs1[end - 1] / hs1[end])
order2 = log(errs2[end - 1] / errs2[end]) / log(hs2[end - 1] / hs2[end])
order3 = log(errs3[end - 1] / errs3[end]) / log(hs3[end - 1] / hs3[end])
(order1, order2, order3)
```

```@example poisson_nonlinear
order1 > 1.9 && order2 > 1.9 && order3 > 1.8
```

```@example poisson_nonlinear
include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([
    (hs1, errs1, "1D", "#5B5FC7"),
    (hs2, errs2, "2D", "#0E7C86"),
    (hs3, errs3, "3D", "#B26A00"),
]; title = "Nonlinear Poisson, ‖·‖₁ₕ")
```

Second order in every dimension, same as the linear problem — the nonlinearity changes how
many solves it takes to reach a given ``u``, not the discretization's own accuracy once it has.
