module SpaceAdCoverageTests

using Test
using Bramble
using Bramble: D₋ₓ, D₋ᵧ, D₋₂, Mₓ, Mᵧ, M₂, εₕ, Δₕ!, CpuPolyester, CpuSerial, CpuThreaded,
               GhostVector, update_ghosts!, centered_difference!, semidiscretize_rhs,
               ast_sparsity_detector, dirac, polygon, polygon_spaces
using DifferentiationInterface
using ForwardDiff
using LinearAlgebra: mul!
using Polyester
using Random
using SparseArrays: nonzeros
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using ..TestUtils: _fd, _have, ghost_mesh, TEST_FILE, WITH_3D_TESTS

# The AD coverage matrix of docs/notes/internals/ad_coverage.md: each public feature's
# smallest differentiable call, through DifferentiationInterface with ForwardDiff and with
# Enzyme, under each execution policy, on a uniform and on a non-uniform mesh.
#
# ForwardDiff runs here, in this process. Enzyme runs in child Julia processes, because one
# of its aborts (`needsCacheWholeAllocation`) ends the process that met it, and its compile
# memory adds up across features: one child per policy and group of cells, each loading
# Enzyme through a temporary environment. A child is this file again, told what to run by
# the environment variable `BRAMBLE_AD_COVERAGE_CHILD`, and prints one `CELL` line per
# call. Without Enzyme in the environment the child cells are skipped.
#
# What each call is checked against: a cell passes when its gradient matches ForwardDiff on
# the same policy to 1e-8 relative (ForwardDiff itself against a central difference, and
# every component of that reference clearly non-zero, so a backend that drops a parameter
# cannot pass). A cell the page records as failing must throw, never return a wrong value.

const SEED = 1234
const NP = 3
const DIMS = (9, 9)
const POLICIES = (CpuSerial = Serial(), CpuThreaded = Parallel(), CpuPolyester = CpuPolyester())
const MESHES = (:uniform, :nonuniform)
const ENZYME_TOML = "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n"
const CHILD = get(ENV, "BRAMBLE_AD_COVERAGE_CHILD", "")

# --- The calls ---------------------------------------------------------------------- #

_basis(n) = randn(MersenneTwister(SEED), n, NP)
_start() = [0.5 + 0.3 * sin(k) for k in 1:NP]

# A mesh of `dims` nodes on `[0, b] x [0, 1]`, uniform or seeded non-uniform, under `policy`.
function _mesh(policy, kind::Symbol, dims::NTuple{2, Int} = DIMS, b = 1.0)
    Bramble._seed_mesh1d_rng!(SEED)
    try
        return mesh(domain(interval(0.0, b) × interval(0.0, 1.0)), dims,
            kind === :uniform ? (true, true) : (false, false);
            backend = backend(policy = policy))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

function _mesh1d(policy, kind::Symbol, n = 17)
    Bramble._seed_mesh1d_rng!(SEED)
    try
        return mesh(domain(interval(0.0, 1.0)), n, kind === :uniform;
            backend = backend(policy = policy))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

# A 5 x 5 x 5 mesh on the unit cube (#1128), uniform or seeded non-uniform, under `policy`.
function _mesh3d(policy, kind::Symbol)
    Bramble._seed_mesh1d_rng!(SEED)
    try
        return mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (5, 5, 5), kind === :uniform ? (true, true, true) : (false, false, false);
            backend = backend(policy = policy))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

# A grid function holding `shift + B p`.
function _field(W, B, p::AbstractVector{T}; shift = 0.0) where {T}
    u = element(W, T)
    parent(u) .= shift .+ B * p
    return u
end

# A builder is `(policy, kind) -> (f, x)`: a scalar function of the parameters `x`.
function build_mesh(policy, kind)
    f(p) = begin
        W = gridspace(_mesh(policy, kind, DIMS, p[1]))
        sum(abs2, parent(D₋ₓ(Rₕ(W, x -> sin(x[1]) * x[2]^2 + x[1]))))
    end
    return f, [1.3]
end

function build_difference(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    return p -> sum(abs2, parent(D₋ₓ(_field(W, B, p)))), _start()
end

function build_average(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    return p -> (u = _field(W, B, p); sum(abs2, parent(Mₓ(u))) + sum(abs2, parent(Mᵧ(u)))),
    _start()
end

function build_laplacian(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    return p -> sum(abs2, parent(Δₕ(_field(W, B, p)))), _start()
end

function build_inner(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    w = element(W)
    parent(w) .= cos.(1:ndofs(W))
    return p -> (u = _field(W, B, p); innerₕ(u, w) + normₕ(u)^2), _start()
end

function build_composite(policy, kind)
    W = gridspace(_mesh(policy, kind), Val(2))
    function f(p)
        u = Rₕ(W, x -> (p[1] * sin(x[1]) + p[2] * x[2]^2, p[3] * x[1] * x[2] + p[1] * cos(x[2])))
        return sum(c -> sum(abs2, parent(c)), Iterators.flatten(εₕ(u)))
    end
    return f, _start()
end

function build_interpolate(policy, kind)
    W = gridspace(_mesh(policy, kind))
    return p -> sum(abs2, parent(Rₕ(W, x -> p[1] * sin(x[1]) + p[2] * x[2]^2 + p[3] * x[1]))),
    _start()
end

function build_average_cell(policy, kind)
    W = gridspace(_mesh(policy, kind))
    return p -> sum(abs2, parent(avgₕ(W, x -> p[1] * sin(x[1]) + p[2] * x[2]^2 + p[3] * x[1]))),
    _start()
end

function build_assemble_linear(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    return p -> (u = _field(W, B, p); sum(abs2, assemble(form(W, v -> innerₕ(u, v))))), _start()
end

function build_assemble_diffusion(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    function f(p)
        κ = _field(W, B, p; shift = 2.0)
        return sum(abs2, nonzeros(assemble(form(W, W, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V))))))
    end
    return f, _start()
end

function build_assemble_averaged(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    function f(p)
        κ = _field(W, B, p; shift = 2.0)
        return sum(abs2, nonzeros(assemble(form(W, W, (U, V) -> inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))))))
    end
    return f, _start()
end

# Three summands, the third scaled by a constant: the shape of gpena/Bramble.jl#1124.
function build_assemble_three(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    function f(p)
        κ = _field(W, B, p; shift = 2.0)
        a = form(W, W,
            (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V)) + innerₕ(κ * U, V) + innerₕ(κ * U, 0.5 * D₋ᵧ(V)))
        return sum(abs2, nonzeros(assemble(a)))
    end
    return f, _start()
end

# The mesh endpoint through an assembly: the coordinates reach the form through its plan.
function build_assemble_mesh(policy, kind)
    function f(p)
        W = gridspace(_mesh(policy, kind, DIMS, p[1]))
        a = form(W, W, (U, V) -> inner₊(∇ₕ(U), ∇ₕ(V)) + innerₕ(U, V))
        return sum(abs2, nonzeros(assemble(a)))
    end
    return f, [1.3]
end

# `Mₕ(κ)` on a 3D mesh.
function build_assemble_averaged_3d(policy, kind)
    W = gridspace(_mesh3d(policy, kind))
    B = _basis(ndofs(W))
    function f(p)
        κ = _field(W, B, p; shift = 2.0)
        return sum(abs2, nonzeros(assemble(form(W, W, (U, V) -> inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))))))
    end
    return f, _start()
end

# The 3D linear form, `inner₊(κ∇U, ∇V)` and the difference operators on a grid function
# (#1128): the calls that work in 3D once `Mₕ(κ)` is set aside.
function build_assemble_linear_3d(policy, kind)
    W = gridspace(_mesh3d(policy, kind))
    B = _basis(ndofs(W))
    return p -> (u = _field(W, B, p); sum(abs2, assemble(form(W, v -> innerₕ(u, v))))), _start()
end

function build_assemble_diffusion_3d(policy, kind)
    W = gridspace(_mesh3d(policy, kind))
    B = _basis(ndofs(W))
    function f(p)
        κ = _field(W, B, p; shift = 2.0)
        return sum(abs2, nonzeros(assemble(form(W, W, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V))))))
    end
    return f, _start()
end

function build_difference_3d(policy, kind)
    W = gridspace(_mesh3d(policy, kind))
    B = _basis(ndofs(W))
    return p -> (u = _field(W, B, p); sum(abs2, parent(D₋ₓ(u))) + sum(abs2, parent(D₋₂(u)))),
    _start()
end

# The averaging operator `M₂` (the third axis) on a grid function: its reverse mode was
# `EnzymeNoTypeError` (gpena/Bramble.jl#1155), fixed in v4.2.4.
function build_average_z_3d(policy, kind)
    W = gridspace(_mesh3d(policy, kind))
    B = _basis(ndofs(W))
    return p -> sum(abs2, parent(M₂(_field(W, B, p)))), _start()
end

function build_assemble_dirichlet(policy, kind)
    W = gridspace(_mesh1d(policy, kind))
    B = _basis(ndofs(W))
    function f(p)
        κ = _field(W, B, p; shift = 2.0)
        A, F = assemble(form(W, W, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V))),
            form(W, v -> innerₕ(x -> 1.0 + x[1], v)); dirichlet = :boundary => x -> 0.0)
        return sum(abs2, nonzeros(A)) + sum(abs2, F)
    end
    return f, _start()
end

function build_assemble_dirichlet_2d(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    function f(p)
        κ = _field(W, B, p; shift = 2.0)
        A, F = assemble(form(W, W, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V))),
            form(W, v -> innerₕ(x -> 1.0, v)); dirichlet = :boundary => x -> 0.0)
        return sum(abs2, nonzeros(A)) + sum(abs2, F)
    end
    return f, _start()
end

function build_pde_solve(policy, kind)
    W = gridspace(_mesh1d(policy, kind))
    B = _basis(ndofs(W))
    function f(p)
        κ = _field(W, B, p; shift = 2.0)
        A, F = assemble(form(W, W, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V))),
            form(W, v -> innerₕ(x -> 1.0 + x[1], v)); dirichlet = :boundary => x -> 0.0)
        return sum(abs2, pde_solve(A, F))
    end
    return f, _start()
end

# A form whose matrix is not symmetric, so a reverse that applied `A` for `Aᵀ` is caught.
_nonsymmetric(W) = form(W, W, (U, V) -> inner₊(∇ₕ(U), ∇ₕ(V)) + innerₕ(U, D₋ᵧ(V)))

function build_semidiscretize(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    w = element(W)
    parent(w) .= cos.(1:ndofs(W))
    rhs = semidiscretize_rhs(semidiscretize(_nonsymmetric(W), form(W, v -> innerₕ(w, v))))
    function f(p)
        du = zeros(eltype(p), ndofs(W))
        rhs(du, B * p, nothing, 0.0)
        return sum(abs2, du)
    end
    return f, _start()
end

function build_matrix_free(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    op = matrix_free_operator(_nonsymmetric(W))
    function f(p)
        y = zeros(eltype(p), size(op, 1))
        mul!(y, op, B * p)
        return sum(abs2, y)
    end
    return f, _start()
end

function build_kronecker(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    K = kronecker_operator(_nonsymmetric(W))
    function f(p)
        y = zeros(eltype(p), size(K, 1))
        mul!(y, K, B * p)
        return sum(abs2, y)
    end
    return f, _start()
end

# A ghost space: the parameters scale the form's coefficients, the relation data are
# constants. The Robin coefficient and the Neumann data are fixed.
function _ghost_space(policy, kind)
    Ωₕ = ghost_mesh(policy, (7, 6))
    return constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax, :ymax)),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => 2.0 * u + ∂ₙ(u) => 1.0,
            :ymin => u => 0.0, :ymax => ∂ₙ(u) => 1.0))
end

function build_ghost_assemble(policy, kind)
    C = _ghost_space(policy, kind)
    l = form(C, v -> innerₕ(x -> 1.0 + x[1], v))
    function f(p)
        a = form(C, C, (U, V) -> p[1] * innerₕ(-Δₕ(U), V) + p[2]^2 * innerₕ(U, V))
        A, F = assemble(a, l)
        return sum(abs2, nonzeros(A)) + 1e4 * p[3] * sum(F)
    end
    return f, _start()
end

# The same space with a form that reads no ghost: `inner₊` differences stay on the physical
# nodes, so only the reduction onto the ghost space is differentiated.
function build_ghost_inner(policy, kind)
    C = _ghost_space(policy, kind)
    l = form(C, v -> innerₕ(x -> 1.0 + x[1], v))
    function f(p)
        a = form(C, C, (U, V) -> p[1] * inner₊(∇ₕ(U), ∇ₕ(V)) + p[2]^2 * innerₕ(U, V))
        A, F = assemble(a, l)
        return sum(abs2, nonzeros(A)) + 1e4 * p[3] * sum(F)
    end
    return f, _start()
end

# The same space with relation data that depend on the parameter, and with a Robin
# coefficient that does.
function build_ghost_param(policy, kind, what::Symbol)
    W = gridspace(ghost_mesh(policy, (7, 6)); ghosts = (:xmin, :xmax, :ymax))
    function f(p)
        β = what === :robin ? p[1] : 2.0
        g = what === :data ? p[1] : 1.0
        C = constrained(W,
            u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => β * u + ∂ₙ(u) => g,
                :ymin => u => 0.0, :ymax => ∂ₙ(u) => 1.0))
        a = form(C, C, (U, V) -> innerₕ(-Δₕ(U), V) + innerₕ(U, V))
        A, F = assemble(a, form(C, v -> innerₕ(x -> 1.0 + x[1], v)))
        return sum(abs2, nonzeros(A)) + sum(abs2, F)
    end
    return f, [1.3]
end
build_ghost_data(policy, kind) = build_ghost_param(policy, kind, :data)
build_ghost_robin(policy, kind) = build_ghost_param(policy, kind, :robin)

# The load alone: the form is constant and the parameters enter only the linear form, so the
# derivative must reach the load's own buffer (gpena/Bramble.jl#1127, the load half).
function build_ghost_load(policy, kind)
    C = _ghost_space(policy, kind)
    a = form(C, C, (U, V) -> innerₕ(U, V))
    function f(p)
        l = form(C, v -> innerₕ(x -> p[1] * (1.0 + x[1]) + p[2], v))
        _, F = assemble(a, l)
        return p[3] * sum(F) + sum(abs2, F)
    end
    return f, _start()
end

# The runtime operators on a `GhostVector`, in place.
function build_ghost_runtime(policy, kind)
    C = _ghost_space(policy, kind)
    B = _basis(42)
    function f(p)
        v = update_ghosts!(GhostVector(C, B * p), C)
        out = zeros(eltype(p), 42)
        Δₕ!(out, v)
        return sum(abs2, out)
    end
    return f, _start()
end

# The first difference along x reads the ghosts at `:xmin` and `:xmax`.
function build_ghost_stencil(policy, kind)
    C = _ghost_space(policy, kind)
    B = _basis(42)
    function f(p)
        v = update_ghosts!(GhostVector(C, B * p), C)
        out = zeros(eltype(p), 42)
        centered_difference!(out, v, Val(1))
        return sum(abs2, out)
    end
    return f, _start()
end

# The diffusion residual of test/ext/sparse_ad_ext.jl: `A(u) u - F`.
policy_α(u) = 3 + 1 / (1 + u^2)

function sparse_problem(policy, kind)
    W = gridspace(_mesh1d(policy, kind, 41))
    g = element(W)
    avgₕ!(g, x -> exp(x[1]))
    F = assemble(form(W, v -> innerₕ(g, v)))
    diffusion(u) = form(W, W, (U, V) -> inner₊(policy_α.(Mₕ(u)) * ∇ₕ(U), ∇ₕ(V)))
    function residual(u::AbstractVector{T}) where {T}
        uₕ = element(W, T)
        uₕ .= u
        return assemble(diffusion(uₕ); dirichlet = :boundary) * u .- F
    end
    return residual, diffusion(element(W, 0.0)), [0.1 + 0.01 * i for i in 1:ndofs(W)]
end

# --- The geometric inputs (#1171) ------------------------------------------------- #

# A 12 x 12 mesh of `box` carrying the marker `pair`, uniform or seeded non-uniform, under
# `policy`: the grid-fitted mesh of a polygon or a level set.
function _marker_mesh(policy, kind::Symbol, box, pair::Pair)
    Bramble._seed_mesh1d_rng!(SEED)
    try
        return mesh(domain(box, pair), (12, 12), kind === :uniform ? (true, true) : (false, false);
            backend = backend(policy = policy))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

const POLYGON_BOX = interval(0.0, 3.0) × interval(0.0, 3.0)
const SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
_polygon_mesh(policy, kind, V = SQUARE) = _marker_mesh(policy, kind, POLYGON_BOX, :P => polygon(V))

# The Dirichlet datum `p₁ x₁ + p₂ x₂²` of the polygon spaces (#1170), on a mesh built
# outside `f`; it reaches the load through the boundary columns the reduction moves.
function build_polygon_datum(policy, kind)
    Ωₕ = _polygon_mesh(policy, kind)
    function f(p)
        U, W = polygon_spaces(Ωₕ, :P, x -> p[1] * x[1] + p[2] * x[2]^2)
        _, F = assemble(form(U, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))),
            form(W, v -> innerₕ(x -> 1.0, v)))
        return sum(abs2, F)
    end
    return f, [0.5, 0.8]
end

# A coefficient and a load of the form on the polygon spaces, the datum fixed.
function _polygon_coefficient(Ωₕ, p)
    U, W = polygon_spaces(Ωₕ, :P, x -> 1.0 + x[1])
    A, F = assemble(form(U, W, (u, v) -> p[1] * inner₊(∇ₕ(u), ∇ₕ(v)) + p[2]^2 * innerₕ(u, v)),
        form(W, v -> innerₕ(x -> p[3] * (1.0 + x[1]), v)))
    return sum(abs2, nonzeros(A)) + sum(abs2, F)
end

function build_polygon_coefficient(policy, kind)
    Ωₕ = _polygon_mesh(policy, kind)
    return p -> _polygon_coefficient(Ωₕ, p), [0.5, 0.8, 1.1]
end

# The same call with the mesh built inside `f`, the vertices constant.
function build_polygon_mesh_inside(policy, kind)
    return p -> _polygon_coefficient(_polygon_mesh(policy, kind), p), [0.5, 0.8, 1.1]
end

# The mass matrix of a mesh built inside `f`: it reads the node coordinates, which a vertex
# or a level-set parameter moves, and no cell of it is near-singular.
function _mass_of(Ωₕ)
    W = gridspace(Ωₕ)
    return sum(abs2, nonzeros(assemble(form(W, W, (U, V) -> innerₕ(U, V)))))
end

# The vertices of the polygon, a rectangle of the square's corner, as the parameters.
function build_polygon_vertices(policy, kind)
    function f(p)
        V = [(0.0, 0.0), (p[1], 0.0), (p[1], p[2]), (0.0, p[2])]
        return _mass_of(_polygon_mesh(policy, kind, V))
    end
    return f, [2.0, 1.8]
end

# The circle `x₁² + x₂² − r²` at `r = 0.605`, the radius as the parameter.
function build_levelset_parameter(policy, kind)
    function f(p)
        φ = x -> x[1]^2 + x[2]^2 - p[1]^2
        box = interval(-1.0, 1.0) × interval(-1.0, 1.0)
        return _mass_of(_marker_mesh(policy, kind, box, :D => levelset(φ)))
    end
    return f, [0.605]
end

# A point load on the 9 x 9 mesh, read through a fixed weight vector, off every node.
const DIRAC_POINT = (0.347, 0.612)

function build_dirac_location(policy, kind)
    W = gridspace(_mesh(policy, kind))
    c = cos.(1:ndofs(W))
    return p -> sum(c .* assemble(form(W, v -> innerₕ(dirac((p[1], p[2]), 2.0), v)))),
    collect(DIRAC_POINT)
end

function build_dirac_strength(policy, kind)
    W = gridspace(_mesh(policy, kind))
    return p -> sum(abs2, assemble(form(W, v -> innerₕ(dirac(DIRAC_POINT, p[1]), v)))), [2.5]
end

# --- The cells -------------------------------------------------------------------- #

# `feature` is the row of the page and `group` the child process that runs the cell's Enzyme
# half. `fd`, `rev` and `fwd` are what the page records for ForwardDiff, Enzyme reverse and
# Enzyme forward. Where the policies differ, `rev`, `fwd` and the error texts are a
# NamedTuple by policy. `kinds` are the meshes a cell runs on.
#
# `:ok` means the gradient matches ForwardDiff's. `:throws` means an error whose message
# contains `fd_err`, `rev_err` or `fwd_err`. `:aborts` means Enzyme ends the process, whose
# output contains `rev_err` or `fwd_err`. `:wrong` is a gradient that differs from
# ForwardDiff's, a known upstream bug. `:skip` is a call not run. A `slow` cell compiles for
# tens of minutes before Enzyme fails, so it runs only with `BRAMBLE_AD_COVERAGE_SLOW=true`.
function cell(name, feature, group, build; fd = :ok, rev = :ok, fwd = :ok, fd_err = "", rev_err = "",
        fwd_err = rev_err, kinds = MESHES, slow = false)
    (; name, feature, group, build, fd, rev,
        fwd, fd_err, rev_err, fwd_err, kinds, slow)
end

const CELLS = (
    cell(:mesh_endpoint, "mesh", :dense, build_mesh),
    cell(:difference, "operator", :dense, build_difference),
    cell(:average, "operator", :dense, build_average),
    cell(:laplacian, "operator", :dense, build_laplacian),
    cell(:inner_norm, "operator", :dense, build_inner),
    cell(:composite, "operator", :dense, build_composite),
    cell(:interpolate, "interpolation", :dense, build_interpolate),
    cell(:cell_average, "interpolation", :dense, build_average_cell),
    cell(:assemble_linear, "assemble", :assemble, build_assemble_linear),
    cell(:assemble_diffusion, "assemble", :assemble, build_assemble_diffusion),
    cell(:assemble_averaged, "assemble", :assemble, build_assemble_averaged),
    # Forward mode was wrong under all three policies (relative error 0.17 uniform, 0.10
    # non-uniform, measured 2026-11-07), gpena/Bramble.jl#1124, fixed in v4.2.4 by holding
    # `SpaceWeights` by reference. A child of its own, like `boundary`: with the other
    # `assemble` cells, `CpuThreaded` peaks at 10 GB.
    cell(:assemble_three, "assemble", :three, build_assemble_three),
    cell(:assemble_dirichlet, "assemble", :assemble, build_assemble_dirichlet),
    cell(:assemble_mesh, "mesh", :geometry, build_assemble_mesh,
        rev = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip),
        fwd = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip)),
    # 3D (#1128). `Mₕ(κ)` and `M₂` differentiate at Enzyme's default type-analysis bounds
    # under all three policies (#1154, #1155); the other 3D cells ran under `CpuSerial` only.
    cell(:assemble_linear_3d, "assemble", :threed, build_assemble_linear_3d,
        rev = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip),
        fwd = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip)),
    cell(:assemble_diffusion_3d, "assemble", :threed, build_assemble_diffusion_3d,
        rev = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip),
        fwd = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip)),
    cell(:difference_3d, "operator", :threed, build_difference_3d,
        rev = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip),
        fwd = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip)),
    # `M₂` on a grid function: the reverse memset was gpena/Bramble.jl#1155.
    cell(:average_z_3d, "operator", :threed, build_average_z_3d),
    cell(:assemble_averaged_3d, "assemble", :threed, build_assemble_averaged_3d),
    # Reverse was an abort (gpena/Bramble.jl#1126), fixed in v4.2.2; the threaded cells
    # are the same form under the other policies. A child of its own: with the `assemble`
    # cells, `CpuThreaded` peaks at 11 GB (3.2 GB alone, measured 2026-11-07).
    cell(:assemble_dirichlet_2d, "assemble", :boundary, build_assemble_dirichlet_2d,
        rev = (CpuSerial = :ok, CpuThreaded = :ok, CpuPolyester = :ok),
        fwd = (CpuSerial = :ok, CpuThreaded = :ok, CpuPolyester = :ok), kinds = (:nonuniform,)),
    # Forward was `EnzymeNoDerivativeError` (gpena/Bramble.jl#1129), fixed in v4.2.2.
    cell(:pde_solve, "pde_solve", :solve, build_pde_solve),
    cell(:semidiscretize, "semidiscretize", :solve, build_semidiscretize),
    cell(:matrix_free, "matrix_free", :operators, build_matrix_free),
    cell(:kronecker, "kronecker", :operators, build_kronecker),
    # `Δₕ` in the form and the parameters in its coefficients (gpena/Bramble.jl#1152): both
    # modes under all three policies, 115 to 169 s and 2.3 to 3.3 GB per threaded cell
    # (docs/notes/internals/ad_reductions/1152_ghost_matrix.md). `CpuSerial` failed with
    # `EnzymeNoTypeError` between 3aefc79f and 9faf6345.
    cell(:ghost_assemble, "ghost", :ghost_fast, build_ghost_assemble, kinds = (:nonuniform,)),
    # The parameters scale the matrix and there is no `Δₕ`: reverse works under `CpuSerial`
    # (measured 2026-11-07 and again after 9faf6345); forward and the threaded policies were
    # not run.
    cell(:ghost_inner, "ghost", :ghost_fast, build_ghost_inner,
        rev = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip), fwd = :skip,
        kinds = (:nonuniform,)),
    # Relation data that depend on the parameter, and a Robin coefficient that does
    # (gpena/Bramble.jl#1153, docs/notes/internals/ad_reductions/1153_constrained.md). Reverse
    # compiled for over 10 GB until the rule of #1182 over `_assemble_relation`
    # (ext/BrambleEnzymeExt/relations.jl): now both modes match ForwardDiff under `CpuSerial` and
    # `CpuPolyester` (3.3 to 6.1 GB alone). Under `CpuThreaded` both modes fail with
    # `EnzymeNoTypeError` in the relation's forms, open as gpena/Bramble.jl#1179. Each cell has
    # a child of its own: in the `ghost_fast` child the group peaked at 10.2 GB under
    # `CpuPolyester` and 10.6 GB under `CpuThreaded`, where the 10 GB cap killed it.
    cell(:ghost_data, "ghost", :relation_data, build_ghost_data,
        rev = (CpuSerial = :ok, CpuThreaded = :throws, CpuPolyester = :ok),
        fwd = (CpuSerial = :ok, CpuThreaded = :throws, CpuPolyester = :ok),
        rev_err = "EnzymeNoTypeError", kinds = (:nonuniform,)),
    # The Robin coefficient's ForwardDiff half works (gpena/Bramble.jl#1127); the Enzyme half
    # is the same as `ghost_data`.
    cell(:ghost_robin, "ghost", :relation_robin, build_ghost_robin,
        rev = (CpuSerial = :ok, CpuThreaded = :throws, CpuPolyester = :ok),
        fwd = (CpuSerial = :ok, CpuThreaded = :throws, CpuPolyester = :ok),
        rev_err = "EnzymeNoTypeError", kinds = (:nonuniform,)),
    # The load of `assemble(a, l)` alone (gpena/Bramble.jl#1127, the load half). The threaded
    # reverse is #1157 and the forward under Enzyme #1159, both fixed in v4.2.2.
    cell(:ghost_load, "ghost", :ghost_fast, build_ghost_load, kinds = (:nonuniform,)),
    # `Δₕ!` on a `GhostVector`, reverse under `CpuPolyester` fixed in v4.2.2 (#1127).
    cell(:ghost_runtime, "ghost", :ghost_fast, build_ghost_runtime,
        rev = (CpuSerial = :ok, CpuThreaded = :ok, CpuPolyester = :ok),
        fwd = (CpuSerial = :ok, CpuThreaded = :ok, CpuPolyester = :ok),
        kinds = (:nonuniform,)),
    # `centered_difference!` on a `GhostVector` (gpena/Bramble.jl#1158): differentiates in
    # both modes under all three policies, measured 2026-11-08.
    cell(:ghost_stencil, "ghost", :ghost_fast, build_ghost_stencil,
        rev = (CpuSerial = :ok, CpuThreaded = :ok, CpuPolyester = :ok),
        fwd = (CpuSerial = :ok, CpuThreaded = :ok, CpuPolyester = :ok),
        kinds = (:nonuniform,)),
    # The geometric inputs of gpena/Bramble.jl#1171, measured with Enzyme 0.13.219 on
    # 2026-11-11 (docs/notes/internals/ad_reductions/1171_geometric_inputs.md). The two
    # polygon `works` cells compile for about 5 minutes per mode, so they have a child of
    # their own. Forward is a `MethodError`: Enzyme has no forward rule for `deepcopy`.
    # The datum of `polygon_spaces` (#1170): a `Dual` datum fills the padded space.
    cell(:polygon_datum, "polygon", :polygon, build_polygon_datum,
        rev = (CpuSerial = :ok, CpuThreaded = :ok, CpuPolyester = :ok),
        fwd = (CpuSerial = :throws, CpuThreaded = :throws, CpuPolyester = :throws),
        fwd_err = "MethodError"),
    cell(:polygon_coefficient, "polygon", :polygon, build_polygon_coefficient,
        rev = (CpuSerial = :ok, CpuThreaded = :ok, CpuPolyester = :ok),
        fwd = (CpuSerial = :throws, CpuThreaded = :throws, CpuPolyester = :throws),
        fwd_err = "MethodError"),
    # `dirac` converts its point to `Float64`: ForwardDiff has no method for a `Dual` and
    # Enzyme differentiates it in both modes. The reference of the Enzyme half is a central
    # difference. The threaded policies were not run.
    cell(:dirac_location, "dirac", :inputs, build_dirac_location, fd = :throws,
        fd_err = "no method matching Float64",
        rev = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip),
        fwd = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip)),
    cell(:dirac_strength, "dirac", :inputs, build_dirac_strength,
        rev = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip),
        fwd = (CpuSerial = :ok, CpuThreaded = :skip, CpuPolyester = :skip)),
    # `polygon(V)` holds exact `Rational{BigInt}` vertices, which a `Dual` cannot become.
    # Enzyme meets the same illegal type analysis as `polygon_mesh_inside`, in both modes.
    cell(:polygon_vertices, "polygon", :inputs, build_polygon_vertices, fd = :throws,
        fd_err = "no method matching Rational{BigInt}",
        rev = (CpuSerial = :throws, CpuThreaded = :skip, CpuPolyester = :skip),
        fwd = (CpuSerial = :throws, CpuThreaded = :skip, CpuPolyester = :skip),
        rev_err = (CpuSerial = "IllegalTypeAnalysisException", CpuThreaded = "", CpuPolyester = "")),
    # The mesh built inside `f`, vertices constant: ForwardDiff works and Enzyme meets the
    # exact `Rational` and `BigFloat` arithmetic of the grid closure in both modes.
    cell(:polygon_mesh_inside, "polygon", :inputs, build_polygon_mesh_inside,
        rev = (CpuSerial = :throws, CpuThreaded = :skip, CpuPolyester = :skip),
        fwd = (CpuSerial = :throws, CpuThreaded = :skip, CpuPolyester = :skip),
        rev_err = (CpuSerial = "IllegalTypeAnalysisException", CpuThreaded = "", CpuPolyester = "")),
    # A `φ` that depends on the parameter is refused under ForwardDiff (#1172): a `Float64`
    # grid has no derivative to give. Enzyme is `:skip`: the first fresh process returned a
    # gradient of zeros in both modes (a silent zero), and four later ones aborted in LLVM
    # with `Instruction does not dominate all uses!`, with nothing changed in between.
    # A child of its own, so the abort cannot end the cells of another group.
    cell(:levelset_parameter, "levelset", :levelset, build_levelset_parameter, fd = :throws,
        fd_err = "Level-set marker",
        rev = (CpuSerial = :skip, CpuThreaded = :skip, CpuPolyester = :skip),
        fwd = (CpuSerial = :skip, CpuThreaded = :skip, CpuPolyester = :skip))
)

# --- Enzyme, in a child process ------------------------------------------------------ #

# The Enzyme half runs only as the file's own `ENZYME_FILES` run (test/runtests.jl), never
# inside the unit suite, which holds the ForwardDiff half.
const RUN_ENZYME = _have(:Enzyme) && TEST_FILE == "space/ad_coverage.jl"
CHILD == "" || @eval import Enzyme

function enzyme_backend(mode::Symbol; activity::Bool = true, annotation::Symbol = :const)
    m = mode === :reverse ? Enzyme.Reverse : Enzyme.Forward
    activity && (m = Enzyme.set_runtime_activity(m))
    annotation === :none && return AutoEnzyme(; mode = m)
    ann = annotation === :const ? Enzyme.Const : Enzyme.Duplicated
    return AutoEnzyme(; mode = m, function_annotation = ann)
end

_relerr(g, ref) = maximum(abs, g .- ref) / maximum(abs, ref)

# The reference gradient: ForwardDiff on the same policy, or a central difference where
# ForwardDiff cannot run.
function _reference(f, x)
    try
        return DifferentiationInterface.gradient(f, AutoForwardDiff(), x), 1e-8
    catch
        return [_fd(t -> f(setindex!(copy(x), t, k)), x[k]; h = 1e-4) for k in eachindex(x)], 1e-5
    end
end

# `CELL|name|policy|kind|mode|status|relerr|detail`, one line.
function _report(name, policy, kind, mode, status, relerr = NaN, detail = "")
    detail = replace(first(detail, 160), r"[\n|]" => " ")
    println("CELL|$name|$policy|$kind|$mode|$status|$relerr|$detail")
    flush(stdout)
end

function _gradient_cell(name, f, x, policy, kind, mode, ad)
    ref, tol = _reference(f, x)
    minimum(abs, ref) > 1e-6 * maximum(abs, ref) || return _report(name, policy, kind, mode, :weakref)
    g = try
        DifferentiationInterface.gradient(f, ad, x)
    catch err
        err isa InterruptException && rethrow()
        return _report(name, policy, kind, mode, :throws, NaN,
            string(nameof(typeof(err)), ": ", sprint(showerror, err)))
    end
    r = _relerr(g, ref)
    return _report(name, policy, kind, mode, isfinite(r) && r <= tol ? :ok : :wrong, r)
end

# The residual's sparse Jacobian through `AutoSparse`, against ForwardDiff's dense one.
function _sparse_cell(policy, kind, mode, backend)
    residual, a, u = sparse_problem(POLICIES[policy], kind)
    ref = DifferentiationInterface.jacobian(residual, AutoForwardDiff(), u)
    ad = AutoSparse(backend; sparsity_detector = ast_sparsity_detector(a, U -> Mₕ(U)),
        coloring_algorithm = GreedyColoringAlgorithm())
    J = try
        Matrix(DifferentiationInterface.jacobian(residual, ad, u))
    catch err
        err isa InterruptException && rethrow()
        return _report(:sparse_jacobian, policy, kind, mode, :throws, NaN,
            string(nameof(typeof(err)), ": ", sprint(showerror, err)))
    end
    r = _relerr(J, ref)
    return _report(:sparse_jacobian, policy, kind, mode, isfinite(r) && r <= 1e-8 ? :ok : :wrong, r)
end

# A closure that fills a buffer it captured from the parameters.
function build_buffer(policy, kind)
    W = gridspace(_mesh(policy, kind))
    B = _basis(ndofs(W))
    u = element(W)
    function f(p)
        parent(u) .= B * p
        return sum(abs2, parent(D₋ₓ(u)))
    end
    return f, _start()
end

# The DI settings: `(name, build, mode, activity, annotation)`, each run under `CpuSerial`.
const SETTINGS = (
    (:plain_reverse, build_difference, :reverse, false, :none),
    (:plain_forward, build_difference, :forward, false, :none),
    (:activity_reverse, build_difference, :reverse, true, :none),
    (:activity_forward, build_difference, :forward, true, :none),
    (:const_reverse, build_difference, :reverse, true, :const),
    (:const_forward, build_difference, :forward, true, :const),
    (:buffer_const_reverse, build_buffer, :reverse, true, :const),
    (:buffer_const_forward, build_buffer, :forward, true, :const),
    (:buffer_duplicated_reverse, build_buffer, :reverse, true, :duplicated),
    (:buffer_duplicated_forward, build_buffer, :forward, true, :duplicated)
)

# What the page records for each setting: `:wrong` is a gradient that differs from the
# reference, a captured buffer a `Const` function cannot write its derivative into.
const SETTING_EXPECT = (
    plain_reverse = (:throws, "EnzymeMutabilityException"),
    plain_forward = (:throws, "EnzymeMutabilityException"),
    activity_reverse = (:throws, "EnzymeMutabilityException"),
    activity_forward = (:throws, "EnzymeMutabilityException"),
    const_reverse = (:ok, ""),
    const_forward = (:ok, ""),
    buffer_const_reverse = (:silent_zero, ""),
    buffer_const_forward = (:silent_zero, ""),
    buffer_duplicated_reverse = (:ok, ""),
    buffer_duplicated_forward = (:ok, "")
)

# What a child runs: `policy:group`, `policy:group:cell` or `policy:group:cell:kind:mode`.
function child_main(spec::AbstractString)
    pname, group, only, only_kind, only_mode = (split(spec, ':')..., "", "", "")[1:5]
    policy = Symbol(pname)
    pol = POLICIES[policy]
    if group == "settings"
        for (name, build, mode, activity, annotation) in SETTINGS
            f, x = build(pol, :nonuniform)
            ad = enzyme_backend(mode; activity, annotation)
            _gradient_cell(name, f, x, policy, :nonuniform, mode, ad)
        end
    elseif group == "sparse"
        for kind in MESHES, mode in (:forward, :reverse)

            _sparse_cell(policy, kind, mode, enzyme_backend(mode))
        end
    else
        for c in CELLS
            string(c.group) == group || continue
            only == "" || string(c.name) == only || continue
            for kind in c.kinds
                only_kind in ("", string(kind)) || continue
                f, x = c.build(pol, kind)
                c.fd === :throws && only_mode == "" && _forwarddiff_cell(c.name, f, x, policy, kind)
                for (mode, want) in ((:reverse, _want(c.rev, policy)), (:forward, _want(c.fwd, policy)))
                    want === :skip && continue
                    want === :aborts && only_mode == "" && continue
                    only_mode in ("", string(mode)) || continue
                    _gradient_cell(c.name, f, x, policy, kind, mode, enzyme_backend(mode))
                end
            end
        end
    end
    return nothing
end

# ForwardDiff on a cell the page records as failing.
function _forwarddiff_cell(name, f, x, policy, kind)
    try
        DifferentiationInterface.gradient(f, AutoForwardDiff(), x)
    catch err
        err isa InterruptException && rethrow()
        return _report(name, policy, kind, :forwarddiff, :throws, NaN,
            string(nameof(typeof(err)), ": ", sprint(showerror, err)))
    end
    return _report(name, policy, kind, :forwarddiff, :ok)
end

# --- The parent -------------------------------------------------------------------- #

const THREADED = Threads.nthreads() > 1
const SLOW = get(ENV, "BRAMBLE_AD_COVERAGE_SLOW", "false") == "true"

# One child: whether it succeeded, the cells' results by `(cell, kind, mode)` and its output.
function run_child(spec::AbstractString)
    src = """
        _e = mktempdir()
        write(joinpath(_e, "Project.toml"), $(repr(ENZYME_TOML)))
        push!(LOAD_PATH, _e)
        include($(repr(joinpath(@__DIR__, "..", "TestUtils.jl"))))
        Base.include(Main, $(repr(@__FILE__)))
        """
    opt = max(Base.JLOptions().opt_level, 0)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=$(Threads.nthreads()) --optimize=$opt -e $src`
    cmd = addenv(cmd, "BRAMBLE_AD_COVERAGE_CHILD" => spec, "BRAMBLE_TEST_FILE" => nothing)
    out = IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    text = String(take!(out))
    results = Dict{Tuple{Symbol, Symbol, Symbol}, @NamedTuple{status::Symbol, relerr::Float64, detail::String}}()
    for line in split(text, '\n')
        startswith(line, "CELL|") || continue
        _, name, _, kind, mode, status, relerr, detail = split(line, '|'; limit = 8)
        results[(Symbol(name), Symbol(kind), Symbol(mode))] = (status = Symbol(status),
            relerr = parse(Float64, relerr), detail = String(detail))
    end
    return success(p), results, text
end

_want(x::Union{Symbol, String}, ::Symbol) = x
_want(x::NamedTuple, policy::Symbol) = x[policy]

# One reported cell against what the page records.
function check_cell(got, expected::Symbol, err::AbstractString, label)
    if expected in (:skip, :aborts)
        return nothing
    elseif got === nothing
        @test "$label: the child reported nothing" == ""
    elseif expected === :ok
        @test got.status === :ok
    elseif expected === :throws
        @test got.status === :throws && occursin(err, got.detail)
    elseif expected === :silent_zero
        @test got.status === :wrong && got.relerr == 1.0
    elseif expected === :wrong
        # An upstream bug: the gradient is wrong, not an error. When Enzyme fixes it the
        # broken test passes and fails the run, and the page's open-issue paragraph goes.
        @test got.status === :wrong
        @test_broken got.status === :ok
    end
    return nothing
end

if CHILD == ""
    @testset "AD coverage, ForwardDiff" begin
        for c in CELLS, (pname, policy) in pairs(POLICIES), kind in c.kinds
            c.fd === :ok || continue
            (c.group === :threed && !WITH_3D_TESTS) && continue
            pname === :CpuSerial || THREADED || continue
            @testset "$(c.name), $pname, $kind" begin
                f, x = c.build(policy, kind)
                g = DifferentiationInterface.gradient(f, AutoForwardDiff(), x)
                fd = [_fd(t -> f(setindex!(copy(x), t, k)), x[k]; h = 1e-4) for k in eachindex(x)]
                @test minimum(abs, g) > 1e-6 * maximum(abs, g)
                @test isapprox(g, fd; rtol = 1e-5, atol = 1e-8 * maximum(abs, g))
                if pname !== :CpuSerial
                    fs, xs = c.build(POLICIES.CpuSerial, kind)
                    @test _relerr(g, DifferentiationInterface.gradient(fs, AutoForwardDiff(), xs)) <= 1e-8
                end
            end
        end
        # A cell the page records as throwing under ForwardDiff: the call throws an error
        # whose message contains `fd_err`, and never returns.
        for c in CELLS, (pname, policy) in pairs(POLICIES), kind in c.kinds
            c.fd === :throws || continue
            pname === :CpuSerial || THREADED || continue
            @testset "$(c.name), $pname, $kind, throws" begin
                f, x = c.build(policy, kind)
                err = try
                    DifferentiationInterface.gradient(f, AutoForwardDiff(), x)
                    nothing
                catch e
                    e
                end
                @test err isa Exception && occursin(c.fd_err, sprint(showerror, err))
            end
        end
        for (pname, policy) in pairs(POLICIES), kind in MESHES

            pname === :CpuSerial || THREADED || continue
            @testset "sparse Jacobian, $pname, $kind" begin
                residual, a, u = sparse_problem(policy, kind)
                ref = DifferentiationInterface.jacobian(residual, AutoForwardDiff(), u)
                ad = AutoSparse(AutoForwardDiff(); sparsity_detector = ast_sparsity_detector(a, U -> Mₕ(U)),
                    coloring_algorithm = GreedyColoringAlgorithm())
                @test _relerr(Matrix(DifferentiationInterface.jacobian(residual, ad, u)), ref) <= 1e-8
            end
        end
    end

    @testset "AD coverage, Enzyme" begin
        if !RUN_ENZYME
            @test_skip "Enzyme is not in this environment"
        else
            groups = (:dense, :assemble, :three, :boundary, :geometry, :solve, :operators, :ghost_fast,
                :relation_data, :relation_robin, :inputs, :levelset, :polygon)
            WITH_3D_TESTS && (groups = (groups..., :threed))
            SLOW && (groups = (groups..., unique(c.group for c in CELLS if c.slow)...))
            for pname in keys(POLICIES)
                pname === :CpuSerial || THREADED || continue
                @testset "$pname" begin
                    for group in groups
                        ok, res, _ = run_child("$pname:$group")
                        @test ok
                        for c in CELLS
                            c.group === group || continue
                            for kind in c.kinds
                                check_cell(get(res, (c.name, kind, :reverse), nothing), _want(c.rev, pname),
                                    _want(c.rev_err, pname), "$(c.name) $kind reverse")
                                check_cell(get(res, (c.name, kind, :forward), nothing), _want(c.fwd, pname),
                                    _want(c.fwd_err, pname), "$(c.name) $kind forward")
                                c.fd === :throws && check_cell(
                                    get(res, (c.name, kind, :forwarddiff), nothing),
                                    :throws, c.fd_err, "$(c.name) ForwardDiff")
                            end
                        end
                    end
                    # An abort ends the child, so each call has a child of its own.
                    for c in CELLS
                        (c.group === :threed && !WITH_3D_TESTS) && continue
                        for (mode, want, err) in ((:reverse, _want(c.rev, pname), _want(c.rev_err, pname)),
                            (:forward, _want(c.fwd, pname), _want(c.fwd_err, pname)))
                            want === :aborts || continue
                            ok, res, text = run_child("$pname:$(c.group):$(c.name):nonuniform:$mode")
                            @test !ok
                            @test isempty(res)
                            @test occursin(err, text)
                        end
                    end
                    ok, res, _ = run_child("$pname:sparse")
                    @test ok
                    for kind in MESHES, mode in (:forward, :reverse)

                        check_cell(get(res, (:sparse_jacobian, kind, mode), nothing), :ok, "",
                            "sparse $kind $mode")
                    end
                end
            end
            @testset "DI settings" begin
                ok, res, _ = run_child("CpuSerial:settings")
                @test ok
                for (name, _, mode, _, _) in SETTINGS
                    want, err = SETTING_EXPECT[name]
                    check_cell(get(res, (name, :nonuniform, mode), nothing), want, err,
                        "setting $name")
                end
            end
        end
    end
else
    child_main(CHILD)
end

end # module SpaceAdCoverageTests
