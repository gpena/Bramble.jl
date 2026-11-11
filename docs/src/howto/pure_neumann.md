```@meta
CurrentModule = Bramble
```

# [How to solve a pure Neumann problem](@id howto_pure_neumann)

**Goal.** Solve ``-\Delta u = f`` with a Neumann condition on every facet, fix the constant of the solution, remove the kernel that ghost unknowns add, and check that the data are compatible.

**You need.** A constrained ghost space with Neumann relations ([ghost point tutorial](@ref form_ghost)). The [solver page](@ref howto_solvers) lists the other keywords of `pde_solve`.

## Fix the constant with a mean-zero condition

With Neumann conditions on every facet the solution is defined up to a constant. The matrix annihilates constants and is singular. Fix the constant by a discrete mean-zero condition: [`pde_solve`](@ref) takes `mean_zero = w`, a weight vector, and returns the solution with ``w^T u = 0``. Internally it solves the bordered system

```math
\begin{pmatrix} A & w \\ w^T & 0 \end{pmatrix}
\begin{pmatrix} u \\ \lambda \end{pmatrix} =
\begin{pmatrix} F \\ 0 \end{pmatrix}
```

with a symmetric or LU factorization, never Cholesky, since the border makes it indefinite. `A` is not modified and stays sparse. A matrix that annihilates constants, passed without `mean_zero`, produces a warning that names the keyword.

The problem is ``-\Delta u = f`` on a graded square with the normal derivative given on each facet. Each facet carries its own datum, so a corner node takes two values, one per facet:

```@example howto_neumann
using Bramble
import Bramble: change_points!
using LinearAlgebra: norm, dot

Ω_sq = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ_sq = mesh(Ω_sq, (17, 17), (true, true))
grid = collect(range(0.0, 1.0, length = 17) .^ 2)
change_points!(Ωₕ_sq, markers(Ω_sq), (grid, grid))

u_sq(p) = cos(π * p[1]) * cos(π * p[2]) + p[1] * p[2] / 2
grad_sq(p) = (-π * sin(π * p[1]) * cos(π * p[2]) + p[2] / 2,
    -π * cos(π * p[1]) * sin(π * p[2]) + p[1] / 2)
f_sq(p) = 2π^2 * cos(π * p[1]) * cos(π * p[2])

W_sq = gridspace(Ωₕ_sq; ghosts = :boundary)
C_sq = constrained(W_sq, u -> (
    :xmin => ∂ₙ(u) => (p -> -grad_sq(p)[1]), :xmax => ∂ₙ(u) => (p -> grad_sq(p)[1]),
    :ymin => ∂ₙ(u) => (p -> -grad_sq(p)[2]), :ymax => ∂ₙ(u) => (p -> grad_sq(p)[2])))
A_sq, F_sq = assemble(form(C_sq, C_sq, (u, v) -> innerₕ(-Δₕ(u), v)),
    form(C_sq, v -> innerₕ(f_sq, v)))

@assert norm(A_sq * ones(size(A_sq, 1)), Inf) <= 1e-10 * norm(A_sq, Inf)
@assert norm(A_sq' * ones(size(A_sq, 1)), Inf) <= 1e-10 * norm(A_sq, Inf)

ndofs(C_sq), norm(A_sq * ones(size(A_sq, 1)), Inf)
```

The matrix has 289 unknowns, one per node, and ``A \mathbf{1} = 0`` and ``\mathbf{1}^T A = 0`` to rounding. The weight vector is the cell measure, the linear form of the constant function ``1`` on the space without ghosts:

```@example howto_neumann
W_plain_sq = gridspace(Ωₕ_sq)
w_sq = assemble(form(W_plain_sq, v -> innerₕ(x -> 1.0, v)))
u_sol = pde_solve(A_sq, F_sq; mean_zero = w_sq)

# the defining equations of the bordered system
λ = sum(F_sq) / sum(w_sq)
@assert abs(dot(w_sq, u_sol)) <= 1e-12
@assert norm(A_sq * u_sol - (F_sq - λ * w_sq)) <= 1e-9 * norm(F_sq)

# against the exact solution shifted to discrete mean zero
u_ex = parent(Rₕ(W_plain_sq, u_sq))
u_ex .-= dot(w_sq, u_ex) / sum(w_sq)
err_sq = maximum(abs, u_sol - u_ex)
@assert err_sq < 2e-2

abs(dot(w_sq, u_sol)), err_sq
```

The residual of the system is ``F - \lambda w`` with ``\lambda = \sum F / \sum w``, because ``\mathbf{1}^T A = 0``. The compatibility of the data holds only to the quadrature error, so ``\lambda`` is small but nonzero. The error is the discrete difference to the exact solution shifted to the same mean.

## Remove a kernel in the ghosts

`mean_zero` fixes the constant of a pure Neumann problem. A space that keeps its ghosts as unknowns has a second kind of kernel, in the ghosts themselves. The form `innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)` reads the ghost of a corner node only through `Δₕ`, where the two ghosts of a 2D corner enter in one combination. The matrix assembled on `gridspace(Ωₕ; ghosts = :boundary)` is therefore singular: a vector that is zero at every physical node and nonzero on the ghosts of a corner solves ``A z = 0``. The physical solution is unique, and the ghost values are not.

`Bramble.ghost_kernel(A, Wₕ)` returns a sparse basis of that kernel, one column per corner here, found node by node and never as a dense null space. `pde_solve(A, F; gauge = Z)` solves the bordered system ``[A\ Z; Z^T\ 0]`` and returns the solution with ``Z^T u = 0``:

```@example howto_neumann
using SparseArrays

Ω_rect = domain(interval(0.0, 1.0) × interval(0.0, 2.0))
Ωₕ_rect = mesh(Ω_rect, (9, 11), (true, true))
change_points!(Ωₕ_rect, markers(Ω_rect), (collect(range(0.0, 1.0, length = 9) .^ 1.5),
    2 .* collect(range(0.0, 1.0, length = 11) .^ 1.5)))

W_rect = gridspace(Ωₕ_rect; ghosts = :boundary)
A_rect = assemble(form(W_rect, W_rect, (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)))
Z_rect = Bramble.ghost_kernel(A_rect, W_rect)

n_phys = npoints(Ωₕ_rect)
@assert size(Z_rect) == (ndofs(W_rect), 4)
@assert all(>(n_phys), rowvals(Z_rect))
@assert norm(A_rect * Z_rect, Inf) <= 1e-10 * norm(A_rect, Inf)

size(Z_rect), norm(A_rect * Z_rect, Inf)
```

The kernel has 4 columns, each supported on the two ghosts of a corner, and `A * Z` vanishes to rounding. The physical values come first in the vector of unknowns, then the ghosts. To check the solve, choose a physical vector, set the ghosts to zero, and form the load `F = A x`. The gauged solve returns the physical values of `x`, and the load of a vector shifted by any combination of the kernel columns gives the same answer:

```@example howto_neumann
using Random

Random.seed!(7)
x_phys = rand(n_phys)
x_ghost0 = [x_phys; zeros(ndofs(W_rect) - n_phys)]

u_gauged = pde_solve(A_rect, A_rect * x_ghost0; gauge = Z_rect)
u_shifted = pde_solve(A_rect, A_rect * (x_ghost0 + Z_rect * [1.0, -2.0, 3.0, 0.5]);
    gauge = Z_rect)

@assert maximum(abs, u_gauged[1:n_phys] - x_phys) <= 1e-7
@assert norm(Z_rect' * u_gauged, Inf) <= 1e-10 * norm(u_gauged, Inf)
@assert maximum(abs, u_shifted - u_gauged) <= 1e-6
@assert norm(A_rect * u_gauged - A_rect * x_ghost0, Inf) <= 1e-8 * norm(A_rect, Inf)

maximum(abs, u_gauged[1:n_phys] - x_phys), norm(Z_rect' * u_gauged, Inf)
```

The physical error is about ``10^{-10}`` here. `ghost_kernel` finds kernels supported on the ghosts of a single node. A kernel spread over several nodes is left out, and the gauged solve still gives the physical part, to about ``10^{-8}`` relative in general (see [`pde_solve`](@ref)). With `mean_zero = w` as well, the border is `[w Z]`.

## Check that the data are compatible

The Neumann data of ``-\Delta u = f`` must satisfy ``\int f = -\oint \partial_\eta u``. A source off by a constant leaves a system with no solution: the bordered solve returns a ``\lambda`` that measures the part of `F` no `u` reaches, and the answer looks plausible. `Bramble.neumann_compatibility(problem, Ωₕ)` checks the data.

`problem` is a function of a mesh that returns `(A, F, w)`, the matrix, the load and the weights of the discrete average, because the check builds the problem on `Ωₕ` and on copies of it refined once and twice. For compatible data the removed load is a quadrature error and falls by about four per refinement. For an incompatible datum it tends to a constant, the integral of the discrepancy, and the check logs a warning that reports the relative defects, ``\lambda`` and the ratios. Nothing is thrown, and the result is a `NamedTuple` with `compatible`. The problem below is the square from above on a random non-uniform mesh, once as it is and once with the source shifted by ``0.2``:

```@example howto_neumann
using Logging

struct WarningLog <: Logging.AbstractLogger
    messages::Vector{String}
end
Logging.min_enabled_level(::WarningLog) = Logging.Warn
Logging.shouldlog(::WarningLog, args...) = true
function Logging.handle_message(log::WarningLog, level, message, args...; kwargs...)
    push!(log.messages, string(message))
    return nothing
end

function neumann_problem(shift)
    return function (Ωₕ)
        C = constrained(gridspace(Ωₕ; ghosts = :boundary), u -> (
            :xmin => ∂ₙ(u) => (p -> -grad_sq(p)[1]), :xmax => ∂ₙ(u) => (p -> grad_sq(p)[1]),
            :ymin => ∂ₙ(u) => (p -> -grad_sq(p)[2]), :ymax => ∂ₙ(u) => (p -> grad_sq(p)[2])))
        A, F = assemble(form(C, C, (u, v) -> innerₕ(-Δₕ(u), v)),
            form(C, v -> innerₕ(p -> f_sq(p) + shift, v)))
        w = assemble(form(gridspace(Ωₕ), v -> innerₕ(p -> 1.0, v)))
        return A, F, w
    end
end

Random.seed!(20261016)
Ωₕ_rand = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 17), (false, false))

function check_compatibility(shift)
    log = WarningLog(String[])
    result = Logging.with_logger(log) do
        Bramble.neumann_compatibility(neumann_problem(shift), Ωₕ_rand)
    end
    return result, log.messages
end

ok, ok_warnings = check_compatibility(0.0)
bad, bad_warnings = check_compatibility(0.2)

@assert ok.compatible && isempty(ok_warnings)
@assert ok.defect < 1e-4
@assert !bad.compatible && length(bad_warnings) == 1
@assert occursin("incompatible", only(bad_warnings))
@assert isapprox(bad.λ, 0.2; atol = 1e-3) && isapprox(bad.second_ratio, 1.0; atol = 0.1)

(ok.compatible, ok.defect), (bad.compatible, bad.λ, bad.second_ratio)
```

The compatible problem raises no warning and its relative defect is below ``10^{-4}``. The shifted one warns once. Its ``\lambda`` is ``0.2``, the shift, on the mesh and on both refinements, so the ratio of the removed loads is ``1`` where a quadrature error would give about ``4``. The mesh `Ωₕ` is not modified, and a mesh that carries markers of its own needs the domain's `markers(Ω)` as a third argument so that the refinement can evaluate them again.

The check has limits. A relative defect on the refined mesh below `5e-3` is accepted as compatible, so a small incompatibility is not detected. Compatible data with a singular flux at a boundary point, whose quadrature error falls slower than the factor of four, can be reported incompatible. When the defect is above the floor the problem is built on a mesh with 16 times the points of `Ωₕ` in 2D.

## Carry the constraint in one penalised form

The bordered solve keeps `A` and the weights apart. A form can also carry the constraint itself, as a rank-one penalty ``\sigma\,(1, u)_h (1, v)_h``, and then it assembles to a single matrix ``A + \sigma\, w w^T``, where ``w`` is the weight vector from above. The matrix is nonsingular because the penalty fixes the constant. Take the Neumann problem with zero flux on the graded square: `inner₊(∇ₕ(u), ∇ₕ(v))` on the space without ghosts is its stiffness matrix. The penalty is `σ * innerₕ(1, u) * innerₕ(1, v)` with ``\sigma = \gamma / h_{max}``:

```@example howto_neumann
σ_pen = 1.0 / hₘₐₓ(Ωₕ_sq)   # γ = 1
a_pen = form(W_plain_sq, W_plain_sq,
    (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + σ_pen * innerₕ(1, u) * innerₕ(1, v))
K_pen = assemble(a_pen)

# the load with the data made compatible: 1ᵀF = 0
F_raw = assemble(form(W_plain_sq, v -> innerₕ(f_sq, v)))
F_pen = F_raw - (sum(F_raw) / sum(w_sq)) * w_sq

typeof(K_pen).name.name, size(K_pen), Base.summarysize(K_pen) < 10 * Base.summarysize(K_pen.A)
```

`assemble` returns a `Bramble.RankOneUpdate`: the sparse matrix `K_pen.A` of the other terms, `σ` and the two weight vectors, with no dense block. Its products cost one product by `A` and two passes over the vectors. `K_pen'` and `transpose(K_pen)` are lazy wrappers of `K_pen`, and read its parts when they are applied. `pde_solve(K_pen, F_pen)` does not form the dense matrix either. It solves the sparse bordered system

```math
\begin{pmatrix} A & w \\ w^T & -1/\sigma \end{pmatrix}
\begin{pmatrix} u \\ \mu \end{pmatrix} =
\begin{pmatrix} F \\ 0 \end{pmatrix},
```

whose second row gives ``\mu = \sigma w^T u``, so its first row is ``A u + \sigma w (w^T u) = F``. For compatible data ``w^T u = 0``, and ``\mathbf{1}^T A = 0`` holds for this stiffness matrix, so the solution is the `mean_zero` one for every nonzero ``\sigma``. The next block checks that against the `mean_zero` solve and the dense solve, and shows the relative differences:

```@example howto_neumann
A_pen = assemble(form(W_plain_sq, W_plain_sq, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))))

u_pen = pde_solve(K_pen, F_pen)
u_mz = pde_solve(A_pen, F_pen; mean_zero = w_sq)
u_dense = Matrix(K_pen) \ F_pen

rel_mz = norm(u_pen - u_mz) / norm(u_mz)
rel_dense = norm(u_pen - u_dense) / norm(u_dense)

@assert Matrix(K_pen) ≈ Matrix(A_pen) + σ_pen * w_sq * w_sq'
@assert abs(sum(F_pen)) <= 1e-12 * norm(F_pen)
@assert abs(dot(w_sq, u_pen)) <= 1e-10 * norm(w_sq) * norm(u_pen)
@assert rel_mz < 1e-10 && rel_dense < 1e-10
@assert !all(iszero, u_pen)

rel_mz, rel_dense
```

The difference from the `mean_zero` solve is about ``10^{-15}`` for ``\gamma \ge 1`` and grows as ``1/\gamma``: a small ``\sigma`` makes ``-1/\sigma`` large against the entries of `A` and costs digits, so take ``\gamma \ge 10^{-2}``. A large ``\sigma`` costs nothing. The [`pde_solve`](@ref) docstring gives the measured figures.

The `σ` is read when the form is assembled: with a `Ref` factor, call `assemble!(K_pen, a_pen)` after changing it and before the next solve. With `dirichlet`, the Dirichlet rows of `K_pen` are identity rows and the penalty's row vector is zeroed there. A form takes one rank-one term. Two are refused with an `ArgumentError`, so add the scalars when they share the functionals. On a ghost space the rank-one term is covered by the [support table](../internals/space.md).
