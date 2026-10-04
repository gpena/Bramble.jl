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
# `mul!(y, op, x, α, β)` computes `α * A * x + β * y`, and `op * x` is `A * x`. Assembly
# walks the form's stencil over the mesh and stores each entry it meets as a nonzero; `op`
# evaluates the same stencil, through the same `visit_bilinear_stencil` and `local_stencil`,
# and adds each entry's contribution `weight * x[col]` into `y[row]` instead of storing it.
# That is why a product accumulates into `y`, and why `mul!` with `β = 0` overwrites it
# rather than adding to what was there.
#
# Nothing in that evaluation asks for a tensor structure. The Kronecker operator needs the
# matrix to be a sum of Kronecker products of one-dimensional factors, so it refuses a
# grid-function coefficient, a Dirichlet row or a composite space. The matrix-free operator
# has no such factorisation to find: it evaluates the coefficient at the stencil's points,
# writes the identity into a Dirichlet row, and walks the blocks of a composite space one
# after another. It takes every form `assemble` takes because it reuses the assembly's own
# stencil rather than a second evaluator of it.
#
# ## How a product is computed
#
# A product visits the form's terms and blocks in the order the serial assembly replays
# them. A scalar form is taken one summand at a time, a composite form block by block. Each
# unit (a term on a block) has two kinds of point. The interior is the points whose stencil
# lies wholly inside the grid, which is most of them. For most units it is **gathered**.
# Each row `y[row]` sums its own entries, read from the same `local_stencil` as assembly,
# and is written once, in a loop along the first axis. A row's neighbours along that axis
# share stencil points, so the loop evaluates each point once and carries it to the next
# row. The boundary shell is **scattered**: the walk visits each point, through
# `visit_bilinear_stencil`, and adds each entry into its row. (A unit whose per-axis
# spacings the operator caches gathers its shell as well.) A unit the gather cannot take
# scatters whole. That covers a stencil of more than nine entries, a nested stencil, a
# region restriction, a test-side interpolation, a shift, and a grid too small to have an
# interior.
#
# A pair of transposed terms, ``\langle A u, B v\rangle + \langle B u, A v\rangle``, is
# scattered once. An entry found for the first term also stands for the entry of the second
# at the mirrored position, so the stencil is not evaluated twice there. Its interior is
# gathered in two passes, the first term's rows and then the transposed term's, each from
# the same `local_stencil`.
#
# On a serial policy that is the whole product. On a threaded one the grid is cut into one
# band per thread along its **last axis**, and the product is one parallel region however
# many terms the form has: each thread takes every term over its own band and adds only into
# the rows of that band. Because no two threads add into the same entry of `y`, the threaded
# product needs no atomics and its result does not depend on scheduling. A form whose leaves
# walk grids of different sizes, or carry different policies, falls back to sweeping one
# term at a time, each in the colour bands the threaded `assemble!` uses, which is what
# keeps that fallback race-free as well. A term with a test-side interpolation runs serially
# inside the product.
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
    showerror(stdout, err)
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
#   where a sparse product reads stored entries; the section "Kronecker or matrix-free?"
#   compares the routes form class by form class. It also recomputes a coefficient that was
#   updated in place, which is a feature of the design, not a cost.
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
# For a separable form, [`kronecker_operator`](@ref) and [`matrix_free_operator`](@ref) are
# two alternative routes to the same product, and the choice between them is the user's.
# Neither is built on the other: the Kronecker operator applies one-dimensional factor
# matrices it builds once, and the matrix-free operator evaluates the form's stencil at every
# product. Which one is preferred depends on the class of the form. The script
# `benchmark/operator_routes.jl` times both, with the assembled matrix as the reference, on
# four separable form classes over graded non-uniform meshes of the square and the cube,
# and saves the tables to `benchmark/results/operator_routes.toml`. The classes are:
#
#   - `laplace`: `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))`, symmetric positive definite;
#   - `coefficient`: the same with a grid-function coefficient `c` that varies along one
#     axis, `inner₊(c * ∇ₕ(u), ∇ₕ(v))`, symmetric positive definite;
#   - `mixed`: `laplace` plus the mixed-derivative term `innerₕ(c * D₋ₓ(D₋ᵧ(u)), v)`, not
#     symmetric;
#   - `advection`: `laplace` plus the first-order term `innerₕ(D₋ₓ(u), v)`, not symmetric.
#
# The [benchmarks page](../benchmarks.md) charts the same file. This page reads it
# directly, so no figure below is copied by hand. The run's description comes first; the
# ratios depend on the machine, so read the load and the thread count before the numbers:

using TOML

routes_file = joinpath(pkgdir(Bramble), "benchmark", "results", "operator_routes.toml")
routes = TOML.parsefile(routes_file)
meta = routes["meta"]
(cpu = meta["cpu"], threads = meta["threads"], power = meta["power"],
    load = meta["load1"], commit = meta["commit"])

# Every table of the file has one row per form, route, dimension and size, so a lookup
# names the `"form"` as well as the route. `route_ratios` divides one route's measurement by
# another's, for one form class, at every size. The route compared by default is
# `kronecker_threaded`, the Kronecker operator under `CpuThreaded()`, set against the
# threaded matrix-free operator, so the two run on the same number of threads. A ratio
# under `1` means the Kronecker route is cheaper, and one over `1` means it costs more:

function route_ratios(table, column, form; against, of = "kronecker_threaded")
    rows = [r for r in routes["tables"][table] if r["form"] == form]
    value(route, dim, n) = only(r[column] for r in rows
    if r["route"] == route && r["dim"] == dim && r["n"] == n)
    sizes = sort!(unique((r["dim"], r["n"]) for r in rows))
    return [(; dim, n, (Symbol(route) => value(of, dim, n) / value(route, dim, n)
            for route in against)...) for (dim, n) in sizes]
end

# The comparisons below use the ratios as computed, and only the display rounds them.
# `largest` keeps the square of 512² points and the cube of 64³, both of 262144 unknowns,
# the biggest problem of each dimension in the file. `at_largest` gathers those rows over
# the form classes, and `lead` names the route a ratio favours, with a margin of 20% inside
# which the two count as even:

forms = ["laplace", "coefficient", "mixed", "advection"]
spd_forms = ["laplace", "coefficient"]

shown(ratios) = [map(x -> x isa Float64 ? round(x; sigdigits = 3) : x, r) for r in ratios]
largest(ratios) = filter(r -> r.n^r.dim == 262144, ratios)
function at_largest(table, column; forms = forms, kw...)
    return reduce(vcat,
        [[(; form, r...) for r in largest(route_ratios(table, column, form; kw...))]
         for form in forms])
end
lead(ratio) = ratio < 1 / 1.2 ? :kronecker : ratio > 1.2 ? :matrix_free : :even
with_lead(rows, route) = [(; r..., lead = lead(r[route])) for r in rows]
not_faster(form, route) = [(r.dim, r.n)
                           for r in route_ratios("product", "time_s", form; against = [route])
                           if r[Symbol(route)] >= 1]

# **Product.** One five-argument `mul!`, at 262144 unknowns, the threaded Kronecker product
# against the threaded matrix-free one and against the assembled one, with the route each
# ratio favours:

threaded_product = at_largest("product", "time_s";
    against = ["matrix_free_threaded", "assembled"])
shown(with_lead(threaded_product, :matrix_free_threaded))

# The same comparison serially, the Kronecker product on the calling task against the serial
# matrix-free product:

serial_product = at_largest("product", "time_s"; of = "kronecker",
    against = ["matrix_free_serial"])
shown(with_lead(serial_product, :matrix_free_serial))

# The Kronecker product beats the assembled one in every class, and by a wide margin. Against
# the matrix-free operator the classes split. In the Laplacian and in the form with a
# one-axis grid-function coefficient the Kronecker product is the faster one in 2D and 3D, and
# the coefficient form gains the most: the matrix-free operator evaluates the coefficient
# again at every product, while the Kronecker factors already contain it. In the advection
# form the two are even in both dimensions, and in the mixed form they are even in 2D and the
# Kronecker product leads in 3D. Serially the picture is the same, except that the Laplacian
# narrows to a lead of under ten percent, which the margin counts as even. Each of those
# statements is checked against the table on this page:

lead_of(form, dim, column = :matrix_free_threaded, rows = threaded_product) = only(
    lead(r[column]) for r in rows if r.form == form && r.dim == dim)

@test all(r -> r.assembled < 0.5, threaded_product) #src
@test all(lead_of(f, d) == :kronecker for f in spd_forms, d in (2, 3)) #src
@test all(lead_of("advection", d) == :even for d in (2, 3)) #src
@test lead_of("mixed", 2) == :even #src
@test lead_of("mixed", 3) == :kronecker #src
@test all(r -> r.matrix_free_threaded < 0.5, filter(r -> r.form == "coefficient", threaded_product)) #src
@test all(lead_of("coefficient", d, :matrix_free_serial, serial_product) == :kronecker
          for d in (2, 3)) #src
@test lead_of("mixed", 3, :matrix_free_serial, serial_product) == :kronecker #src
@test lead_of("mixed", 2, :matrix_free_serial, serial_product) == :even #src
@test all(lead_of("advection", d, :matrix_free_serial, serial_product) == :even
          for d in (2, 3)) #src
@test all(r -> 1 / 1.2 < r.matrix_free_serial < 1,
    filter(r -> r.form == "laplace", serial_product)) #src

# The preference above is read at the largest size. Below it, the threaded Kronecker product
# is not always the faster one. The sizes at which it is not, per form class:

(; (Symbol(f) => not_faster(f, "matrix_free_threaded") for f in forms)...)

# Where those sizes are small, the products last tens of microseconds and the choice hardly
# matters. They are also the sizes at which the threaded Kronecker product is slower than the
# serial one, because starting the threads costs more than the product does, so the
# crossover is the thread count's doing and not a property of the form. Both statements
# are checked at the smallest size of each dimension:

smallest_kron = reduce(vcat,
    [filter(r -> r.n^r.dim < 2000,
         [(; form, r...)
          for r in route_ratios("product", "time_s", form;
         of = "kronecker_threaded", against = ["kronecker"])])
     for form in forms])
shown(smallest_kron)

@test length(smallest_kron) == 2 * length(forms) #src
@test all(r -> r.kronecker > 1, smallest_kron) #src
@test all((2, 64) in not_faster(f, "matrix_free_threaded") for f in forms) #src
@test all((3, 16) in not_faster(f, "matrix_free_threaded") for f in forms) #src
@test all(r -> (r.dim, r.n) in not_faster(r.form, "matrix_free_threaded"), smallest_kron) #src

# **Memory.** The bytes each operator holds once built, at 262144 unknowns. The Kronecker
# operator stores the one-dimensional factors and some scratch, a small fraction of what the
# assembled matrix stores, and the matrix-free operator stores less still:

memory = at_largest("product", "bytes_held";
    against = ["matrix_free_threaded", "assembled"])
shown(memory)

# Read the ratio of the first column as the factor by which the matrix-free operator holds
# less than the Kronecker one: at least ten at this size, for every class. Against the
# assembled matrix the Kronecker operator holds under one percent. The Kronecker route is
# therefore a memory saving over the assembled matrix, but not over the matrix-free
# operator:

@test all(r -> r.matrix_free_threaded > 10, memory) #src
@test all(r -> r.assembled < 0.01, memory) #src
@test all(r -> r.matrix_free_threaded > 1 && r.assembled < 1,
    reduce(vcat, [route_ratios("product", "bytes_held", f; against = ["matrix_free_threaded", "assembled"])
                  for f in forms])) #src

# **Solve.** Conjugate gradients to the same tolerance with each operator, for the two
# symmetric positive definite classes only, since CG needs symmetry. The mixed and
# advection forms are not symmetric and have no solve rows. The ratios are again at 262144
# unknowns:

solve_ratios = at_largest("solve", "time_s"; forms = spd_forms,
    against = ["matrix_free_threaded", "assembled"])
shown(with_lead(solve_ratios, :matrix_free_threaded))

# The Kronecker CG solve is the faster one in both classes and both dimensions, against the
# matrix-free operator as against the assembled matrix. For the Laplacian the file also
# times [`fdm_solve`](@ref) on the form and a sparse direct solve of the assembled matrix.
# Neither is an operator route, so each is set against the Kronecker CG solve, not against
# the other. `fdm_solve` is faster than the Kronecker CG solve at every size. To solve a
# separable Laplacian, and not to apply the operator inside another iteration, call it. The
# direct solve wins in the square. In the cube, at this size, the Kronecker CG solve is
# the faster one:

references = at_largest("solve", "time_s"; forms = ["laplace"],
    against = ["fdm_solve", "direct"])
shown(references)

@test all(r -> lead(r.matrix_free_threaded) == :kronecker && r.assembled < 1, solve_ratios) #src
@test all(r -> r.fdm_solve > 1, references) #src
@test all(r -> r.fdm_solve > 1,
    route_ratios("solve", "time_s", "laplace"; against = ["fdm_solve"])) #src
@test only(r.direct for r in references if r.dim == 2) > 1 #src
@test only(r.direct for r in references if r.dim == 3) < 1 #src

# ### Which route for which form
#
# The tables settle the preference for each form class at the sizes where the choice
# matters. They come from the file, so a new run revises them by changing the tests above.
#
#   - **Laplacian-like forms** (`laplace`): prefer the Kronecker operator. Its threaded product
#     is faster than the matrix-free one in 2D and 3D, by a smaller margin serially, its CG
#     solve is faster, and `fdm_solve` is faster still when the solve is the goal.
#   - **A one-axis grid-function coefficient** (`coefficient`): prefer the Kronecker
#     operator, by the widest margin of the four classes, because the matrix-free operator
#     evaluates the coefficient again at every product. The warning below says what the
#     Kronecker operator gives up for that.
#   - **A mixed derivative** (`mixed`): prefer the Kronecker operator in 3D, where it leads.
#     In 2D the two are even, so prefer the matrix-free operator, which holds less, is built
#     faster, and sees a coefficient that changes.
#   - **A first-order term** (`advection`): prefer the matrix-free operator. The Kronecker
#     product is no faster, in 2D or 3D, so the Kronecker route is justified by memory
#     alone, and for that the matrix-free operator holds less still. A Kronecker operator
#     for such a form is worth building only to compare it with the assembled matrix.
#   - **A form that is not separable**, such as the worked example above with a grid
#     function varying along both axes: the matrix-free operator is the only one of the two,
#     as `kronecker_operator` refuses it.
#
# Whatever the class, either operator holds a fraction of the assembled matrix and applies
# it faster, so the choice is between the two and never a reason to assemble.
#
# ### A grid-function coefficient is read once
#
# The preference for the `coefficient` and `mixed` classes has a price, which the `!!!
# warning` below states. The demonstration builds both operators on a form with a one-axis
# grid-function coefficient `c`, edits `c` in place, and compares each operator's product
# with the matrix assembled from the edited form. The Kronecker operator warns when it
# reads the coefficient; the page silences that warning, and a hidden test checks that it is
# raised:

using Logging: with_logger, NullLogger

Wₛ = gridspace(graded_mesh(17))
c = Rₕ(Wₛ, x -> 1 + x[1])
snapshot_form() = form(Wₛ, Wₛ, (u, v) -> innerₕ(u, v) + inner₊(c * ∇ₕ(u), ∇ₕ(v)))

K_snapshot = with_logger(NullLogger()) do
    kronecker_operator(snapshot_form())
end
op_live = matrix_free_operator(snapshot_form())
xₛ = rand(ndofs(Wₛ))
Rₕ!(c, x -> 3 + x[1])
y_reference = assemble(snapshot_form()) * xₛ
rel_gap(y) = norm(y - y_reference) / norm(y_reference)
(kronecker_gap = rel_gap(K_snapshot * xₛ), matrix_free_gap = rel_gap(op_live * xₛ))

@test_logs (:warn, r"read once") kronecker_operator(snapshot_form()) #src
@test rel_gap(op_live * xₛ) <= 1e-12 #src
@test rel_gap(K_snapshot * xₛ) > 1e-2 #src

# !!! warning "Snapshot against live coefficient"
#     `kronecker_operator` reads a grid-function coefficient once, when the operator is built,
#     and copies it into its factors, so a later edit to the coefficient is not seen.
#     `matrix_free_operator` reads it live, at every product. The demonstration above is
#     the difference: after `Rₕ!` the matrix-free product matches the assembled one, and
#     the Kronecker product still applies the old coefficient. For a coefficient that
#     changes, use a `Ref` scalar with the Kronecker operator, which stays live, or the
#     matrix-free operator, or rebuild the Kronecker operator after each edit.

# ## Where to go next
#
#   - [Memory scaling](memory_scaling.md) builds the Kronecker operator that this page
#     compares against, and the fast-diagonalisation solve that beats both on a separable form.
#   - [Choosing a solver](@ref tutorial_solvers) shows the preconditioners that work with an
#     operator, and [Solvers by problem](@ref tutorial_solvers_by_problem) says when to use them.
#   - [Backend policies](@ref backend_policies) explains the serial and threaded policies that
#     the product runs under, and the [backend tutorial](@ref tutorial_backend) shows how to
#     choose one.
