# The warm in-place calls that allocate 0 B, one table for both policies (#854): the Serial
# test in test/form/zero_allocations.jl and the CpuPolyester one in test/ext/ run the same
# `CASES`, so the two cannot drift apart. Loaded into `Main` by both, never by the package.
#
# A case is a name, the dimension it needs and `setup(policy)`, which builds the arguments on
# a mesh under `policy` and returns `(f, args...)`: the test warms `f(args...)` once and
# counts the bytes of the second call. A keyword argument goes into a closure that `setup`
# builds, so `f` stays a concrete type and the call itself is the whole measurement.
#
# What `!` promises is in STANDARDS.md ("What `!` promises"): a runtime entry point listed
# here allocates nothing once warm, a setup function does not. `EXCLUDED` names what is left
# out and why. A case that fails is a defect to fix or report, never a line moved into
# `EXCLUDED` to turn the file green.
module ZeroAllocationCases

using Bramble
using Bramble: evaluate!, reaction!, reaction_density!,
               type_cached_assemble!, v_cycle!, w_cycle!
using Kronecker: Kronecker

struct Case
    name::String
    dim::Int
    setup::Function
end
Case(setup::Function, name, dim) = Case(name, dim, setup)

# ---------------------------------------------------------------------------------------
# Meshes and functions

# The unit box of `n` points per axis, moved by `change_points!` to `t^(1 + d/4)` along axis
# `d`, so no two axes share their nodes (as `TestUtils._graded_mesh`, here not periodic and
# on the policy's backend). Seeded: the first axis of a 1D mesh is the only random draw.
function graded_mesh(policy, n::NTuple{D, Int}, dom = nothing) where {D}
    Ωd = dom === nothing ? domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))) : dom
    Ω = mesh(Ωd, n, ntuple(_ -> false, D); backend = backend(policy = policy))
    Bramble.change_points!(Ω, ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D))
    return Ω
end

# Mesh sizes: small enough that 3D stays cheap, past three points along every axis.
const SIZES = Dict(2 => (11, 9), 3 => (7, 6, 5))

space(policy, D) = gridspace(graded_mesh(policy, SIZES[D]))
vspace(policy, D) = gridspace(graded_mesh(policy, SIZES[D]), Val(D))

# Named functions of a point: a closure over a local would be a new type per call.
f_of_x(x) = 1 + sum(abs2, x) + prod(x)

# ---------------------------------------------------------------------------------------
# Operator families (D*!, M*!, S*!, jump*!)

const SUFFIXES = ("ₓ", "ᵧ", "₂")
const FAMILIES = (:D₋, :D₊, :diff₋, :diff₊, :M, :M₊, :Mc, :jump, :Dc, :D̃, :D̽)
const SHIFTS = (:S₊, :S₋)

function stencil_case(fam, d, D)
    name = Symbol(fam, SUFFIXES[d], :!)
    return Case("$name $(D)D", D) do policy
        W = space(policy, D)
        u = Rₕ(W, f_of_x)
        (getproperty(Bramble, name), similar(u), u)
    end
end

# ---------------------------------------------------------------------------------------
# Vector calculus: the destination's shape follows what the operator returns

# A grid function, a `k`-tuple of them, or a `k`-by-`k` tuple.
dest(W, ::Val{:scalar}, D) = element(W)
dest(W, ::Val{:grad}, D) = ntuple(_ -> element(W), D)
dest(W, ::Val{:curl}, D) = D == 2 ? element(W) : ntuple(_ -> element(W), 3)
dest(W, ::Val{:strain}, D) = ntuple(_ -> ntuple(_ -> element(W), D), D)

# What each operator reads: a grid function, or a field of `D` components.
source(policy, ::Val{:scalar}, D) = Rₕ(space(policy, D), f_of_x)
source(policy, ::Val{:field}, D) = Rₕ(vspace(policy, D), ntuple(_ -> f_of_x, D))

# (name, destination kind, source kind)
const CALCULUS = (
    (:divₕ!, :scalar, :field), (:div₊ₕ!, :scalar, :field), (:divcₕ!, :scalar, :field),
    (:diṽₕ!, :scalar, :field), (:div̽ₕ!, :scalar, :field), (:Δₕ!, :scalar, :scalar),
    (:curlₕ!, :curl, :field), (:curl₊ₕ!, :curl, :field), (:curlcₕ!, :curl, :field),
    (:curl̃ₕ!, :curl, :field), (:curl̽ₕ!, :curl, :field),
    (:∇cₕ!, :grad, :scalar), (:∇̃ₕ!, :grad, :scalar), (:∇̽ₕ!, :grad, :scalar),
    (:εₕ!, :strain, :field), (:εcₕ!, :strain, :field), (:ε₊ₕ!, :strain, :field),
    (:ε̽ₕ!, :strain, :field)
)

function calculus_case(name, dk, sk, D)
    return Case("$name $(D)D", D) do policy
        W = space(policy, D)
        (getproperty(Bramble, name), dest(W, Val(dk), D), source(policy, Val(sk), D))
    end
end

# ---------------------------------------------------------------------------------------
# Interpolation, averages and in-place evaluation

function fill_case(name, fill!, D)
    return Case("$name $(D)D", D) do policy
        (fill!, element(space(policy, D)), f_of_x)
    end
end

# `Rₕ!` over a composite space takes one function per component.
function fill_composite_case(D)
    return Case("Rₕ! composite $(D)D", D) do policy
        (Rₕ!, element(vspace(policy, D)), ntuple(_ -> f_of_x, D))
    end
end

# ---------------------------------------------------------------------------------------
# Forms and assembly

mass_stiffness(u, v) = inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)
laplacian_cubed(u, v) = innerₕ(Δₕ(Δₕ(Δₕ(u))), v)
laplacian_squared(u, v) = innerₕ(Δₕ(Δₕ(u)), v)
laplacian_squared_both(u, v) = innerₕ(Δₕ(Δₕ(u)), Δₕ(Δₕ(v)))

# One source per linear form, stored in the form: a grid function of `f_of_x`.
struct Source{S}
    W::S
end
(s::Source)(v) = innerₕ(Rₕ(s.W, f_of_x), v)

# 40 products: past the 32 entries at which a splat or a `map` over a tuple leaves the
# unrolled paths (#853).
struct FortyProducts{F}
    f::F
end
(p::FortyProducts)(v) = foldl(+, ntuple(k -> (k + 0.5) * innerₕ(p.f, v), Val(40)))

# `n` products restricted to `region`, one `restrict_to` each (#853).
struct Restricted{F, R, N}
    f::F
    region::R
    n::Val{N}
end
function (p::Restricted)(v)
    return foldl(+,
        ntuple(k -> (k + 0.5) * innerₕ(p.f, Bramble.restrict_to(p.region, v)), p.n))
end

# `assemble!(A, a; dirichlet = labels)` as one callable of two arguments.
struct WithDirichlet{L}
    labels::L
end
(w::WithDirichlet)(A, a) = assemble!(A, a; dirichlet = w.labels)

# `type_cached_assemble!` takes a named builder: a literal closure re-made on every call
# would allocate its own type. This one builds the form and the refill on the space `W`.
struct DiffusionBuilder{S}
    W::S
end
function (b::DiffusionBuilder)(uₕ)
    αvals = element(b.W, eltype(uₕ))
    a = form(b.W, b.W, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))
    refill!(uₕ) = (αvals .= 3 .+ 1 ./ (1 .+ uₕ .^ 2); nothing)
    return a, refill!
end

struct CachedAssemble{B, C}
    build::B
    cache::C
end
(c::CachedAssemble)(uₕ) = type_cached_assemble!(c.build, c.cache, uₕ; dirichlet = :boundary)

# A matrix case: `call(A, a)` on the form of `build`, assembled once for the pattern. With
# `dirichlet` the call is `assemble!(A, a; dirichlet)`; `marked` (`:aliased` or `:hidden`)
# takes the space of that domain.
function matrix_case(name, D, build, call = assemble!; dirichlet = nothing, marked = nothing,
        sizes = SIZES)
    return Case("$name $(D)D", D) do policy
        W = marked === nothing ? gridspace(graded_mesh(policy, sizes[D])) :
            marked_space(policy, D, marked)
        a = form(W, W, build)
        if dirichlet === nothing
            (call, assemble(a), a)
        else
            (WithDirichlet(dirichlet), assemble(a; dirichlet), a)
        end
    end
end

# A vector case: `call(b, l)` on the linear form `build` returns for the space.
function vector_case(name, D, build, call = assemble!; sizes = SIZES)
    return Case("$name $(D)D", D) do policy
        W = gridspace(graded_mesh(policy, sizes[D]))
        l = form(W, build(W))
        (call, assemble(l), l)
    end
end

# The unit square or box, and the same with two named regions, so that `:l` and `:b` are
# stored markers and `:boundary` is not a label of the mesh.
unit_domain(D) = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
aliased_domain(::Val{2}) = domain(interval(0.0, 1.0) × interval(0.0, 1.0),
    :l => :xmin, :b => :ymin)
aliased_domain(::Val{3}) = domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    :l => :xmin, :b => :ymin)

# `:left => :ymin` leaves `:xmin` as the geometric face name of the mesh; `:xmax` and `:ymax`
# are not labels there. The marker kind picks the domain: `:aliased` has `:l` and `:b` only.
hidden_domain(D) = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)), :left => :ymin)

function marked_space(policy, D, kind)
    Ωd = kind === :hidden ? hidden_domain(D) : aliased_domain(Val(D))
    return gridspace(graded_mesh(policy, SIZES[D], Ωd))
end
aliased_space(policy, D) = marked_space(policy, D, :aliased)

function evaluate_case(D)
    return Case("evaluate! $(D)D", D) do policy
        W = space(policy, D)
        l = form(W, Source(W))
        (evaluate!, zeros(ndofs(W)), l, Rₕ(W, f_of_x))
    end
end

function cached_case(D)
    return Case("type_cached_assemble! $(D)D", D) do policy
        W = space(policy, D)
        uₕ = Rₕ(W, f_of_x)
        c = CachedAssemble(DiffusionBuilder(W), Dict())
        c(uₕ)
        (c, uₕ)
    end
end

# ---------------------------------------------------------------------------------------
# Dirichlet conditions, reactions, symmetrization

function dirichlet_matrix_case(D)
    return Case("dirichlet_bc! matrix $(D)D", D) do policy
        W = space(policy, D)
        a = form(W, W, mass_stiffness)
        (dirichlet_bc!, assemble(a), W, :boundary)
    end
end

# The label `:l` is a stored marker on the aliased domain.
function dirichlet_label_case(D)
    return Case("dirichlet_bc! one alias $(D)D", D) do policy
        W = aliased_space(policy, D)
        a = form(W, W, mass_stiffness)
        (dirichlet_bc!, assemble(a), W, :l)
    end
end

function dirichlet_vector_case(D, label)
    return Case("dirichlet_bc! vector $label $(D)D", D) do policy
        Ωd = label === :boundary ? unit_domain(D) : aliased_domain(Val(D))
        W = gridspace(graded_mesh(policy, SIZES[D], Ωd))
        bcs = dirichlet_constraints(Ωd, label => f_of_x)
        (dirichlet_bc!, zeros(ndofs(W)), W, bcs, label)
    end
end

function symmetrize_case(D)
    return Case("symmetrize! $(D)D", D) do policy
        W = space(policy, D)
        a = form(W, W, mass_stiffness)
        A = assemble(a; dirichlet = :boundary)
        (symmetrize!, A, zeros(ndofs(W)), W, :boundary)
    end
end

# `reaction!(scratch, A, F, uₕ; marker)` as one callable of four arguments.
struct WithMarker{F, M}
    f::F
    marker::M
end
(w::WithMarker)(scratch, A, F, uₕ) = w.f(scratch, A, F, uₕ; marker = w.marker)

function reaction_case(name, f, marker, D, kind = :aliased)
    return Case("$name $marker $(D)D", D) do policy
        W = marked_space(policy, D, kind)
        a = form(W, W, mass_stiffness)
        l = form(W, Source(W))
        (WithMarker(f, marker), zeros(ndofs(W)), assemble(a), assemble(l), Rₕ(W, f_of_x))
    end
end

# ---------------------------------------------------------------------------------------
# Ghost spaces

# Relation data that depend on time, `(x, t) -> …`, on the ghost space of a 2D mesh.
ghost_data(x, t) = sin(t) + x[2] * cos(t)

function ghost_constrained(policy)
    W = gridspace(graded_mesh(policy, (6, 7)); ghosts = (:xmin, :xmax))
    return constrained(W,
        u -> (:xmin => ∂ₙ(u) => ghost_data, :xmax => u + 0.5 * ∂ₙ(u) => ghost_data))
end

lift_case() = Case("lift! 2D", 2) do policy
    C = ghost_constrained(policy)
    u = collect(range(-1.0, 2.0; length = Bramble.ndofs(C)))
    (Bramble.lift!, Bramble.lift(C, u, 0.0), C, u, 0.4)
end

update_ghosts_case() = Case("update_ghosts! 2D", 2) do policy
    C = ghost_constrained(policy)
    (update_ghosts!, GhostVector(C, zeros(Bramble.ndofs(C))), C, 0.4)
end

# ---------------------------------------------------------------------------------------
# Solvers

# Mass plus variable diffusion κ = 1 + |x|²: symmetric positive definite.
struct VariableDiffusion end
function (::VariableDiffusion)(W)
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    return form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v)))
end

function smoother_case(name, make)
    return Case("smooth! $name 2D", 2) do policy
        W = space(policy, 2)
        s = make(VariableDiffusion()(W))
        n = ndofs(W)
        (smooth!, s, ones(n), fill(0.5, n))
    end
end

function cycle_case(name, call, D)
    return Case("$name $(D)D", D) do policy
        Ω = graded_mesh(policy, D == 2 ? (17, 9) : (9, 9, 9))
        P = gmg_preconditioner(VariableDiffusion(), Ω; levels = 3)
        n = npoints(Ω)
        (call, ones(n), P, fill(0.5, n))
    end
end

function fdm_case(D)
    return Case("fdm_solve! $(D)D", D) do policy
        W = space(policy, D)
        a = form(W, W, mass_stiffness)
        n = ndofs(W)
        (fdm_solve!, zeros(n), fdm_factorize(a), fill(0.5, n))
    end
end

# ---------------------------------------------------------------------------------------
# The table

function build_cases()
    cases = Case[]
    for D in (2, 3)
        for d in 1:D, fam in (FAMILIES..., SHIFTS...)

            push!(cases, stencil_case(fam, d, D))
        end
        for (name, dk, sk) in CALCULUS
            D >= 2 && push!(cases, calculus_case(name, dk, sk, D))
        end
        push!(cases, fill_case("Rₕ!", Rₕ!, D), fill_case("avgₕ!", avgₕ!, D), fill_composite_case(D))
        push!(cases, matrix_case("assemble!", D, mass_stiffness),
            matrix_case("assemble! boundary", D, mass_stiffness; dirichlet = :boundary),
            matrix_case("assemble! one label", D, mass_stiffness; dirichlet = :l, marked = :aliased),
            matrix_case("assemble! two labels", D, mass_stiffness;
                dirichlet = (:l, :b), marked = :aliased),
            matrix_case("assemble_add!", D, mass_stiffness, assemble_add!),
            vector_case("assemble! linear", D, Source),
            vector_case("assemble_add! linear", D, Source, assemble_add!),
            evaluate_case(D), cached_case(D))
        push!(cases, dirichlet_matrix_case(D), dirichlet_label_case(D),
            dirichlet_vector_case(D, :boundary), dirichlet_vector_case(D, :l),
            symmetrize_case(D))
        for (nm, f) in (("reaction!", reaction!), ("reaction_density!", reaction_density!)),
            (marker, kind) in ((:boundary, :aliased), (:l, :aliased), (:xmin, :hidden),
                ((:l, :b), :aliased))

            # `:xmin` stays a face name on the 2D domain only.
            kind === :hidden && D != 2 && continue
            push!(cases, reaction_case(nm, f, marker, D, kind))
        end
        push!(cases, fdm_case(D),
            cycle_case("v_cycle!", v_cycle!, D), cycle_case("w_cycle!", w_cycle!, D))
    end
    # The long stencils and products of #853.
    push!(cases, matrix_case("assemble! Δₕ³", 2, laplacian_cubed),
        matrix_case("assemble! Δₕ² both", 2, laplacian_squared_both),
        matrix_case("assemble! Δₕ²", 3, laplacian_squared; sizes = Dict(3 => (7, 6, 5))),
        vector_case("assemble! 40 products", 2, W -> FortyProducts(Rₕ(W, f_of_x))),
        vector_case("assemble! 40 restricted", 2,
            W -> Restricted(Rₕ(W, f_of_x), :boundary, Val(40))),
        vector_case("assemble! 40 two-label", 2,
            W -> Restricted(Rₕ(W, f_of_x), (:boundary, :interior), Val(40))),
        lift_case(), update_ghosts_case(),
        smoother_case("jacobi", jacobi_smoother),
        smoother_case("chebyshev", chebyshev_smoother),
        smoother_case("red-black", red_black_gauss_seidel))
    return cases
end

const CASES = build_cases()

# ---------------------------------------------------------------------------------------
# Known to allocate

# A call that allocates today and has an open issue. The test measures each and expects
# bytes above zero, so a fix turns the entry red and the entry has to go: the list cannot
# outlive its cause. `reason` names the bytes measured on the 11 × 9 mesh, the cause and
# the issue.
struct Known
    case::Case
    reason::String
end

# `innerₕ(f, v) + innerₕ(f, restrict_to(:boundary, D₋ₓ(v))) + innerₕ(f, restrict_to(:boundary, v))`:
# two restricted summands with different stencil types (#1289).
struct MixedRestricted{F}
    f::F
end
function (p::MixedRestricted)(v)
    r = Bramble.restrict_to
    return innerₕ(p.f, v) + innerₕ(p.f, r(:boundary, Bramble.D₋ₓ(v))) +
           innerₕ(p.f, r(:boundary, v))
end

function build_known()
    two_labels = Case("dirichlet_bc! two labels 2D", 2) do policy
        W = aliased_space(policy, 2)
        a = form(W, W, mass_stiffness)
        (dirichlet_bc!, assemble(a), W, :l, :b)
    end
    return Known[
        Known(two_labels,
            "112 B: the union of two labels goes into a fresh mask. `assemble!` keeps it in " *
            "the form's scratch (#994), a standalone call has no cache to hold one."),
        Known(matrix_case("assemble! hidden face", 2, mass_stiffness;
                dirichlet = :xmin, marked = :hidden),
            "112 B: a face name the domain does not store (`:xmin` once `:left => :ymin` " *
            "is remapped) is built by `index_in_marker`'s geometric fallback on every " *
            "call (#994)."),
        Known(matrix_case("assemble! hidden face, 2", 2, mass_stiffness;
                dirichlet = (:xmin, :left), marked = :hidden),
            "224 B: as above, and the union path builds the first label twice (#994)."),
        Known(reaction_case("reaction!", reaction!, (:left, :xmin), 2, :hidden),
            "336 B: several markers, one a hidden face name, go through " *
            "`_combined_marked_indices`, which builds a mask (#1010). One such marker is 0 B."),
        Known(reaction_case("reaction_density!", reaction_density!, (:left, :xmin), 2, :hidden),
            "336 B: as `reaction!` with several markers (#1010)."),
        Known(vector_case("assemble! mixed restricted", 2,
                W -> MixedRestricted(Rₕ(W, f_of_x))),
            "87 744 B on a 9 × 11 mesh, more as the form grows: restricted summands of " *
            "different shapes leave the inferred walk (#1289). Two of the same shape are 0 B.")
    ]
end

const KNOWN = build_known()

# ---------------------------------------------------------------------------------------
# Excluded

# Functions with no case, and why. A setup function or a builder keeps its `!` and allocates
# by design: `!` says the function mutates an argument, and only the runtime entry points in
# `CASES` promise 0 B once warm (STANDARDS.md, "What `!` promises"; #1091).
const SETUP_REASON = "setup: builds or rebuilds the mesh, the hierarchy or the " *
                     "factorisation a step then reuses"

const BUILDER_REASON = "builder: grows a container the caller owns (a sparsity pattern, a " *
                       "stencil or a triplet list), so it allocates by design; a refill " *
                       "that reuses the container is 0 B"

const EXCLUDED = Tuple{String, String}[
    ("fdm_factorize!",
        SETUP_REASON * "; a refill from a form that differs, or that reads " *
        "a grid-function coefficient, projects the axes again"),
    ("coarsen!", SETUP_REASON * "; it makes the next level's mesh"),
    ("set_points!", SETUP_REASON * "; it replaces the points and rebuilds the spacings"),
    ("change_points!", SETUP_REASON * "; it moves the points and rebuilds the spacings"),
    ("iterative_refinement!",
        "changes the point count, so it allocates the new point arrays by design (#1091)"),
    ("_node_kernel!", "builder: the ghost gauge takes one SVD per node, once per space (#1091)"),
    ("contribute!", "inner closure of `_ghost_reduction_map`, a builder that is not banged " *
                    "(#1091)"),
    ("the do-block in _restricted_matvec", "inner closure of a builder that is not banged " *
                                           "(#1091)"),
    map(n -> (n, BUILDER_REASON),
        ["_keep_pattern!", "_closure_insert!", "_kron_cached!", "_collect_scaled_terms!",
            "_coefficient_reach!", "_pattern_term_jacobian!", "_pattern_blocks_jacobian!",
            "_scalar_jacobian_walk!", "_push_point_dependency!", "_interpolation_triplets!",
            "_mf_collect!", "_factored_products!", "_factored_point!",
            "_one_sided_corner!"])...
]

end # module
