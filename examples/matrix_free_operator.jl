# # The matrix-free operator
#
# [`matrix_free_operator`](@ref) applies a bilinear form to a vector without ever storing its
# matrix. This page answers five questions about it: what it is, how a product is computed,
# what it stores and costs, where it stops, and what it looks like on a form the Kronecker
# operator of the [memory scaling](memory_scaling.md) page cannot take. Every claim below
# that can be checked is checked by a line the documentation build hides and the test suite
# runs.

using Bramble
using LinearAlgebra
using LinearSolve: LinearProblem, solve, KrylovJL_CG
using Test #src

# ## Problem
#
# Applying a bilinear form to a vector normally means assembling its `SparseMatrixCSC` first,
# and that matrix stores one value and one index per nonzero. The problem is to compute the
# same product `A * x` without storing `A`, for any form `assemble` accepts, including those
# with a grid-function coefficient, Dirichlet rows or a composite space. The sections below
# build the operator that does it, measure what it costs, and apply it to such a form.
#
# ## What the operator is
#
# For a form `a`, `op = matrix_free_operator(a)` stands for the matrix `assemble(a)`:
# `mul!(y, op, x, α, β)` computes `α * A * x + β * y`, and `op * x` is `A * x`. Assembly walks
# the form's stencil over the mesh and stores each entry it meets as a nonzero; `op` walks the
# same stencil, through the same `visit_bilinear_stencil`, and adds each entry's contribution
# `weight * x[col]` into `y[row]` instead of storing it. That is why the walk accumulates
# into `y`, and why `mul!` with `β = 0` overwrites it rather than adding to what was there.
#
# Nothing in that walk asks for a tensor structure. The Kronecker operator needs the matrix to
# be a sum of Kronecker products of one-dimensional factors, so it refuses a grid-function
# coefficient, a Dirichlet row or a composite space. The matrix-free operator has no such
# factorisation to find: it evaluates the coefficient at the stencil's points, writes the
# identity into a Dirichlet row, and walks the blocks of a composite space one after another.
# It takes every form `assemble` takes because it reuses the assembly's own walk rather than
# a second evaluator of the stencil.
#
# ## How a product is computed
#
# A product walks the form's terms and blocks in the order the serial assembly replays them.
# A scalar form is walked one summand at a time, a composite form block by block. A pair of
# transposed terms, ``\langle A u, B v\rangle + \langle B u, A v\rangle``, is walked once:
# an entry found for the first term also stands for the entry of the second at the mirrored
# position, so the stencil is not evaluated twice.
#
# On a serial policy that is the whole product. On a threaded one the grid is cut into one
# band per thread along its **last axis**, and the product is one parallel region however many
# terms the form has: each thread walks every term over its own band and adds only into the
# rows of that band's points. Because no two threads add into the same entry of `y`, the
# threaded product needs no atomics and its result does not depend on scheduling. A form whose
# leaves walk grids of different sizes, or carry different policies, falls back to sweeping
# one term at a time, each in the colour bands the threaded `assemble!` uses, which is what keeps
# that fallback race-free as well. A term with a test-side
# interpolation runs serially inside the product.
#
# The order in which a thread adds the entries of a row is the serial order, but this page
# does not rely on that: the products below are compared to the assembled one with `≈`, not
# `==`.
#
# ## What it stores and costs
#
# The operator keeps the [`BilinearForm`](@ref) and, through it, the mesh and the coefficient
# data the form holds, plus a mask of the Dirichlet rows and, on a threaded policy, a small
# plan of the bands. It stores no entries. The assembled `SparseMatrixCSC` stores one value
# and one row index per nonzero, and the number of nonzeros grows with the number of unknowns.
# The comparison below is computed on the page, on the form of the worked example. The
# operator gets a form of its own: `assemble` fills the form it is given with the scatter
# tables of the CSR route, which a product never reads, so measuring the operator on an
# assembled form would count memory it does not need.

# The form is mass plus variable diffusion with the coefficient `Rₕ(W, x -> 1 + x[1] * x[2])`,
# on a smoothly graded, non-uniform mesh.

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
graded(n) = [t + 0.1 * sinpi(2t) for t in range(0.0, 1.0; length = n)]
function graded_mesh(n)
    Ωₕ = mesh(Ω, (n, n), (true, true))
    Bramble.change_points!(Ωₕ, (graded(n), graded(n)))
    return Ωₕ
end
spd_form(W) = form(W, W,
    (u, v) -> innerₕ(u, v) + inner₊(Rₕ(W, x -> 1 + x[1] * x[2]) * ∇ₕ(u), ∇ₕ(v)))

n = 65
Ωₕ = graded_mesh(n)
Wₕ = gridspace(Ωₕ)
a = spd_form(Wₕ)
a_mf = spd_form(Wₕ)
op = matrix_free_operator(a_mf; dirichlet = :boundary)
A = assemble(a; dirichlet = :boundary)

bytes_operator = Base.summarysize(op)
bytes_matrix = Base.summarysize(A)
(ndofs = ndofs(Wₕ), operator_bytes = bytes_operator, csr_bytes = bytes_matrix,
    ratio = round(bytes_matrix / bytes_operator; digits = 1))

@test bytes_operator < bytes_matrix #src

# On this 2D mesh the operator holds fewer bytes than the matrix. That is not general: in 1D
# the two take the same memory, as the table linked below shows, with the gap in favour of
# the operator growing from 2D to 3D. How long a product takes is a
# different question, and a single run here would not answer it. The solvers tutorial
# measured it on one machine at nine sizes in 1D, 2D and 3D: see the table
# [Time and memory against a sparse product](../tutorials/solvers.md#Time-and-memory-against-a-sparse-product).
#
# ## Where it stops
#
# - **CPU policies only.** [`CpuSerial`](@ref) walks on the calling task; any other CPU
#   policy, such as [`CpuThreaded`](@ref), threads the walk. A [`GpuPolicy`](@ref) is refused
#   when the operator is built, with an `ArgumentError` that names milestone v4.4.0, where
#   the device product is tracked. The test `GpuPolicy refused (v4.4.0)` in
#   `test/form/matrix_free.jl` pins the message; here the refusal itself:

W₁ = gridspace(mesh(domain(interval(0.0, 1.0)), 11, false))
a₁ = form(W₁, W₁, (u, v) -> innerₕ(u, v))
try
    matrix_free_operator(a₁; policy = Bramble.GpuKernel())
catch err
    err
end

@test_throws ArgumentError matrix_free_operator(a₁; policy = Bramble.GpuKernel()) #src

# - **Only the five-argument `mul!`.** The operator defines `mul!(y, op, x, α, β)`. The
#   three-argument `mul!` and `*` reach it through `LinearAlgebra`'s own methods, with
#   `α = true` and `β = false`, so they work, but the operator defines no method of its own
#   for them. `getindex` is supported for inspection and costs one product per entry.

methods_of_op = filter(
    m -> any(p -> p <: Bramble.MatrixFreeOperator, Base.unwrap_unionall(m.sig).parameters[2:end]),
    collect(methods(mul!)))
@test length(methods_of_op) == 1 #src
@test length(Base.unwrap_unionall(only(methods_of_op).sig).parameters) == 6 #src

# - **Real work per product.** It recomputes each entry from the mesh and the coefficient,
#   so a serial product is slower than a serial sparse product; the tutorial's table puts
#   numbers on that. It also recomputes a coefficient that was updated in place, which is
#   a feature of the design, not a cost.
# - **Forms.** The docstring of [`matrix_free_operator`](@ref) says it takes any form
#   `assemble` accepts, on scalar or composite spaces, and this page knows of no form it
#   refuses. A form with a region restriction (`restrict_to`) allocates inside the product,
#   as it does in `assemble!`; every other form allocates nothing on a serial policy, which
#   the test `matrix-free: allocation-free mul!` in `test/form/matrix_free.jl` pins.
# - **Square and size-checked.** A vector of the wrong length is a `DimensionMismatch`,
#   as the test `entries, Dirichlet rows, live data` in `test/form/matrix_free.jl` checks.
#
# The Kronecker operator, by contrast, refuses the form of the next section.
#
# ## Worked example
#
# The form is the page's own, with the Dirichlet rows of the whole boundary. Its
# coefficient is a grid function, so the Kronecker operator has nothing to factor and
# refuses it:

try
    kronecker_operator(a_mf)
catch err
    typeof(err)
end

@test_throws ArgumentError kronecker_operator(a_mf) #src

# The product on a random vector matches the assembled one, serial and threaded. The
# threaded operator is built with the `policy` keyword, and its sweep sums in an order this
# page does not pin, so the comparison is relative, to `1e-12`:

x = rand(ndofs(Wₕ))
y_serial = op * x
y_threaded = similar(x)
op_threaded = matrix_free_operator(a_mf; dirichlet = :boundary, policy = Bramble.CpuThreaded())
mul!(y_threaded, op_threaded, x, true, false)
y_assembled = A * x
(serial = norm(y_serial - y_assembled) / norm(y_assembled),
    threaded = norm(y_threaded - y_assembled) / norm(y_assembled))

@test norm(y_serial - y_assembled) <= 1e-12 * norm(y_assembled) #src
@test norm(y_threaded - y_assembled) <= 1e-12 * norm(y_assembled) #src
@test y_serial ≈ y_assembled #src
@test y_threaded ≈ y_assembled #src

# Now solve. The mass-plus-diffusion matrix is symmetric positive definite away from the
# Dirichlet rows. An identity row keeps its column entries, so the matrix is symmetric only
# on vectors that vanish on those rows, and CG needs a right-hand side that vanishes there.
# `b = A * x_exact` with an exact vector that is zero on the boundary is one:

x_exact = parent(Rₕ(Wₕ, x -> x[1] * (1 - x[1]) * x[2] * (1 - x[2])))
b = A * x_exact
prob = LinearProblem(op, b)
solve_cg(; kw...) = solve(prob, KrylovJL_CG(); reltol = 1e-8, abstol = 0.0, maxiters = 5000, kw...)

sol_plain = solve_cg()
P = chebyshev_preconditioner(op)
sol_prec = solve_cg(Pl = P)
(plain_iterations = sol_plain.iters, preconditioned_iterations = sol_prec.iters)

# Neither the operator nor the preconditioner assembles `A`; the assembled solve below is
# only the reference. Both solutions agree with it to a relative `1e-6`, and the
# preconditioner cuts the iteration count:

x_direct = A \ b
rel_err(u) = norm(u - x_direct) / norm(x_direct)
(plain_error = rel_err(sol_plain.u), preconditioned_error = rel_err(sol_prec.u))

@test sol_plain.iters < 5000 #src
@test sol_prec.iters < 5000 #src
@test rel_err(sol_prec.u) < 1e-6 #src
@test rel_err(sol_plain.u) < 1e-6 #src
@test sol_prec.iters < sol_plain.iters #src

# The solve is the same one the solvers tutorial runs under
# [Jacobi and Chebyshev preconditioning](../tutorials/solvers.md#Jacobi-and-Chebyshev-preconditioning),
# here with Dirichlet rows. [`gmg_preconditioner`](@ref) is the third matrix-free
# preconditioner; the tutorial's multigrid section shows it and states where its point
# smoothers stop.

# ## Kronecker or matrix-free?
#
# For a separable form [`kronecker_operator`](@ref) is a third route beside the assembled
# matrix and the matrix-free operator above. The script `benchmark/operator_routes.jl`
# times all three on the same form, on graded non-uniform meshes of the square and the
# cube, and saves the tables to `benchmark/results/operator_routes.toml`. The
# [benchmarks page](../benchmarks.md) charts the same file. This page reads it directly, so
# no figure below is copied by hand. The run's description comes first; the solve ratios
# below depend on the machine, so read the load and the thread count before the numbers:

using TOML

routes_file = joinpath(pkgdir(Bramble), "benchmark", "results", "operator_routes.toml")
routes = TOML.parsefile(routes_file)
meta = routes["meta"]
(cpu = meta["cpu"], threads = meta["threads"], power = meta["power"],
    load = meta["load1"], commit = meta["commit"])

# Each table below gives, per dimension and size, the Kronecker route's measurement divided
# by the same measurement of another route. A ratio under `1` means the Kronecker route is
# cheaper, and one over `1` means it costs more:

function route_ratios(table, column; against, of = "kronecker")
    rows = routes["tables"][table]
    value(route, dim, n) = only(r[column] for r in rows
    if r["route"] == route && r["dim"] == dim && r["n"] == n)
    sizes = sort!(unique((r["dim"], r["n"]) for r in rows))
    return [(; dim, n, (Symbol(route) => value(of, dim, n) / value(route, dim, n)
            for route in against)...) for (dim, n) in sizes]
end

# The comparisons below use the ratios as computed, and only the display rounds them:

shown(ratios) = [map(x -> x isa Float64 ? round(x; sigdigits = 3) : x, r) for r in ratios]

# `falls` holds when a route's ratio decreases with `n` inside each dimension:

falls(ratios, route) = all(
    issorted([r[route] for r in ratios if r.dim == d]; rev = true, lt = <=) for d in (2, 3))

below(ratios, route) = all(r[route] < 1 for r in ratios)
above(ratios, route) = all(r[route] > 1 for r in ratios)
not_below(ratios, route) = [(r.dim, r.n) for r in ratios if r[route] >= 1]

mf_routes = ["assembled", "matrix_free_serial", "matrix_free_threaded"]

# **Construction.** The time to build the operator from a fresh form:

construction_ratios = route_ratios("construction", "time_s"; against = mf_routes)
shown(construction_ratios)

# The Kronecker route is built faster than the assembled matrix at every size, and the gap
# widens with the size. It is built more slowly than either matrix-free operator at every
# size: a matrix-free operator only wraps the form, while the Kronecker route builds the
# one-dimensional factor matrices.

@test below(construction_ratios, :assembled) #src
@test falls(construction_ratios, :assembled) #src
@test above(construction_ratios, :matrix_free_serial) #src
@test above(construction_ratios, :matrix_free_threaded) #src

# **Bytes.** The memory the operator holds once built:

bytes_ratios = route_ratios("product", "bytes_held"; against = mf_routes)
shown(bytes_ratios)

# The Kronecker route holds fewer bytes than the assembled matrix at every size, and the
# ratio falls as the mesh refines. The matrix-free operators hold fewer still, at every
# size, so for memory the matrix-free route is the smaller of the two:

@test below(bytes_ratios, :assembled) #src
@test falls(bytes_ratios, :assembled) #src
@test above(bytes_ratios, :matrix_free_serial) #src
@test above(bytes_ratios, :matrix_free_threaded) #src

# **Product.** One five-argument `mul!`:

product_ratios = route_ratios("product", "time_s"; against = mf_routes)
shown(product_ratios)

# The Kronecker product is faster than both matrix-free products at every size, and faster
# than the assembled product at every size but the smallest cube, where the two are within
# a few percent:

@test below(product_ratios, :matrix_free_serial) #src
@test below(product_ratios, :matrix_free_threaded) #src
@test not_below(product_ratios, :assembled) == [(3, 8)] #src
@test only(r.assembled for r in product_ratios if r.dim == 3 && r.n == 8) < 1.1 #src

# **Solve.** Conjugate gradients to the same tolerance with each operator. The table adds
# the two reference rows, [`fdm_solve`](@ref) on the form and the sparse direct solve of
# the assembled matrix. Neither is an operator route, so they are compared with the
# Kronecker CG solve rather than with each other:

solve_ratios = route_ratios("solve", "time_s"; against = [mf_routes; "fdm_solve"; "direct"])
shown(solve_ratios)

# The Kronecker CG solve is faster than both matrix-free CG solves at every size, and
# faster than the assembled CG solve at every size but the smallest cube, again within a
# few percent. Against the references it does not win. `fdm_solve` is faster than any CG
# route at every size, and the direct solve is faster in the square at every size and
# slower in the cube:

@test below(solve_ratios, :matrix_free_serial) #src
@test below(solve_ratios, :matrix_free_threaded) #src
@test not_below(solve_ratios, :assembled) == [(3, 8)] #src
@test above([r for r in solve_ratios if r.dim == 2], :direct) #src
@test below([r for r in solve_ratios if r.dim == 3], :direct) #src
@test above(solve_ratios, :fdm_solve) #src
@test only(r.assembled for r in solve_ratios if r.dim == 3 && r.n == 8) < 1.1 #src
fdm_ratios = route_ratios("solve", "time_s"; of = "fdm_solve", against = mf_routes) #src
@test all(below(fdm_ratios, route) for route in (:assembled, :matrix_free_serial, :matrix_free_threaded)) #src

# So the Kronecker route is preferred over the assembled matrix whenever the form is
# separable: it is cheaper to build and to hold at every size, and its product and solve
# are faster at every size except the smallest cube (3D, n = 8). It is preferred over the matrix-free
# operator for the product and the CG solve, which it runs faster at every size. The
# matrix-free operator keeps two advantages, construction time and bytes held, and it is
# the only route of the three when the form is not separable. When the aim is to solve a
# separable problem, and not to apply the operator inside another iteration, `fdm_solve`
# is the faster call at every size in this file.

# ## Where to go next
#
#   - [Memory scaling](memory_scaling.md) builds the Kronecker operator that this page
#     compares against, and the fast-diagonalisation solve that beats both on a separable form.
#   - [Choosing a solver](@ref tutorial_solvers) shows the preconditioners that work with an
#     operator, and [Solvers by problem](@ref tutorial_solvers_by_problem) says when to use them.
#   - [Backend policies](@ref backend_policies) explains the serial and threaded policies that
#     the product runs under, and the [backend tutorial](@ref tutorial_backend) shows how to
#     choose one.
