# precompile/form_sessions.jl: the symbolic form layer, assembly, Jacobian sparsity and
# per-type assembly caching (src/ast/, src/assembly/).
#
# The symbolic layer is not reachable from the mesh/space/operator sessions: a `LazyOp`
# tree is built from `IdentityOperator` and the trial/test leaves rather than from a grid
# function, so none of its constructors, stencil evaluators or traits are inferred by
# anything else there.

# Every display path a type has: the two-argument `show` (the embeddable one-liner), the
# `MIME"text/plain"` block, and `summary`, which array headers reach for independently of
# either. They are separate methods since gpena/Bramble.jl#45, so warming one says nothing
# about the others.
function _pc_display(x)
    sprint(show, x)
    sprint(show, MIME"text/plain"(), x)
    summary(x)
    return nothing
end

# Interpolation leaves are only defined over a scalar space.
function _pc_form_ast_interp(Wₕ::ScalarGridSpace, u)
    π_src = πₕ(element(Wₕ, 1.0))
    is_symbolic(π_src)
    resolve_ast(π_src)
    π_node = πₕ(u)
    is_symbolic(π_node)
    resolve_ast(π_node)
    return nothing
end
_pc_form_ast_interp(::AbstractSpaceType, u) = nothing

# Every node kind, built over one space, plus the traits each one answers. Returns `id` and
# the trial and test leaves, so the caller can build the products.
function _pc_form_ast(Wₕ, ::Val{D}) where {D}
    id = IdentityOperator(Wₕ)
    z = ZeroOperator(Wₕ)
    u, v = TrialFunction{D}(), TestFunction{D}()
    p, q = IndexedTrialFunction{D}(1), IndexedTestFunction{D}(1)

    for leaf in (id, z, u, v, p, q, source_function(x -> 1.0, Val(D)))
        is_symbolic(leaf)
        resolve_ast(leaf)
    end

    _pc_form_ast_interp(Wₕ, u)

    # the one-sided families, the averages, the shift and the restriction
    for op in (D₋ₓ(id), D₊ₓ(id), Mₓ(id), M₊ₓ(id), jumpₓ(id), Dcₓ(id), D̃ₓ(id), D̽ₓ(id))
        is_symbolic(op)
        resolve_ast(op)
        stencil_offsets(op)
    end
    shift_op(id, 1, 1)
    restrict_to(:interior, D₋ₓ(id))

    scaled = 3 * D₋ₓ(id)
    summed = D₋ₓ(id) + D₊ₓ(id)
    resolve_ast(scaled)
    resolve_ast(summed)
    stencil_offsets(summed)
    ∇ₕ(id)
    ∇₊ₕ(id)
    D̽ₕ(id)
    jumpₕ(id)
    Dcₕ(id)
    D̃ₕ(id)
    Mₕ(id)
    M₊ₕ(id)

    return id, u, v
end

# The products, and the stencils they evaluate to. This is the path assembly will take.
#
# `id` is annotated where the other operands are not: `_pc_form_ast` builds it from an
# untyped space, so it arrives here as `Any`, and `D₋ₓ(id)` then has both the symbolic node
# and the operator matrix as candidate results. The two products below are defined on the
# symbolic node alone, so JET reports the matrix half of that union as a missing method --
# a node this workload never builds. `IdentityOperator` is what every caller passes.
function _pc_form_stencils(
        Ωₕ::AbstractMeshType, Wₕ, id::IdentityOperator, u, v, label::Symbol
)
    idx = indices(Ωₕ)
    lin = LinearIndices(idx)
    I = first(idx)
    mk = _marker_words(Ωₕ)

    # Bilinear and linear products for each weight kind.
    for prod in (innerₕ(D₋ₓ(id), D₋ₓ(id)), inner₊ₓ(Mₓ(id), Mₓ(id)), innerₕ(id, D₋ₓ(id)))
        local_stencil(prod, Wₕ, I, nothing, lin[I])
        local_stencil(prod, Wₕ, I, mk, lin[I])
        resolve_ast(prod)
    end

    # Every node kind evaluated once, with and without a marker table: restriction is
    # the only node that reads it, and `nothing` is a separate method there. With the
    # table, a restriction is evaluated bound to `Ωₕ`'s marker ids, as every walk does.
    for op in (
        id,
        D₋ₓ(id),
        D₊ₓ(id),
        Mₓ(id),
        jumpₓ(id),
        Dcₓ(id),
        D̃ₓ(id),
        D̽ₓ(id),
        shift_op(id, 1, 1),
        3 * D₋ₓ(id),
        D₋ₓ(id) + D₊ₓ(id),
        restrict_to(:interior, id),
        restrict_to(label, id)
    )
        local_stencil(op, Wₕ, I, nothing, lin[I])
        local_stencil(_bind_marker_ids(op, Ωₕ), Wₕ, I, mk, lin[I])
    end

    # Symbolic inner products over trial and test leaves, including tuple forms.
    innerₕ(u, v)
    innerₕ(2.0, v)
    innerₕ(x -> 1.0, v)
    inner₊(u, D₋ₓ(v))
    inner₊(D₋ₓ(u), D₋ₓ(v))
    inner₊(∇ₕ(u), ∇ₕ(v))
    inner₊ₓ(u, v)
    # y is a direction only from 2D up: in 1D inner₊ᵧ refuses at the call (#692)
    dim(Ωₕ) >= 2 && inner₊ᵧ(u, v)
    return nothing
end

# Block extraction for a coupled form: the symbolic arguments, and the split into blocks.
function _pc_form_blocks(Vₕ, ::Val{D}) where {D}
    leaves = leaf_spaces_offsets(Vₕ)
    n = length(leaves)

    u, v = TrialFunction{D}(), TestFunction{D}()
    a = innerₕ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(2))

    trial_component_or_nothing(a.left_op)
    test_component_or_nothing(a.left_op)
    block_of(a.left_op, n, n)
    return nothing
end

# The constraints, and applying them. Every path: matrix and vector, scalar and composite.
function _pc_form_dirichlet(Ωₕ::AbstractMeshType, Wₕ, Vₕ, be, label::Symbol, f, ft, I_time)
    bcs = dirichlet_constraints(Ωₕ, label => f)
    dirichlet_constraints(Wₕ, label => f)
    dirichlet_constraints(Vₕ, label => f)
    tbcs = dirichlet_constraints(Ωₕ, I_time, label => ft)
    ev = tbcs(0.5)
    symbols(bcs)
    conditions(bcs)

    n, nv = ndofs(Wₕ), ndofs(Vₕ)
    A = matrix(be, n, n)
    A .= 0
    for i in 1:n
        A[i, i] = 1.0
    end
    Av = matrix(be, nv, nv)
    Av .= 0
    for i in 1:nv
        Av[i, i] = 1.0
    end
    F, Fv = ones(n), ones(nv)

    dirichlet_bc!(A, Ωₕ, label)
    dirichlet_bc!(A, Wₕ, label)
    dirichlet_bc!(Av, Vₕ, label)
    dirichlet_bc!(F, Ωₕ, bcs, label)
    dirichlet_bc!(F, Ωₕ, ev, label)
    dirichlet_bc!(F, Wₕ, bcs, label)
    dirichlet_bc!(Fv, Vₕ, bcs, label)

    symmetrize!(A, F, Ωₕ, label)
    symmetrize!(A, F, Wₕ, label)
    symmetrize!(Av, Fv, Vₕ, label)
    return nothing
end

# Assembly. `_assemble_linear_core!`, `_scatter_term!`, `_route_terms!` and `_sweep_colour!`
# are each parameterized on the AST type, so a form shape this workload never names compiles
# from scratch on the caller's first `assemble`: nothing above reaches them, because the
# stencil session evaluates `local_stencil` directly and never drives a loop over the grid.
#
# Measured in a fresh session before this was added, the shapes below cost 1,032 ms of
# first-call latency in 1D and 2D together.
#
# 3D is deliberately absent from the shapes below, as it is in the sessions above: the
# mass/mixed/directional/composite/reaction/Jacobian/type-cache shapes each cost their own
# 78-114 ms and would multiply the specializations of every core in this file. The one 3D
# shape that does matter to a caller -- the scalar Laplacian -- is warmed separately in
# src/precompile.jl's own workload (measured there: +2.7 s precompile, +5.3 MiB cache).
#
# One call per shape rather than a loop over a tuple of closures. A loop creates a union
# type at the call site and compiles a generic fallback rather than concrete specialized kernels.
function _pc_assemble_shape(Wₕ, g, b)
    lf = form(Wₕ, g)
    ast = resolve_form_ast(lf)

    test_space(lf)
    stencil_offsets(ast)
    _colour_strides(stencil_offsets(ast))
    _assembled_eltype(ast, Wₕ)

    assemble(lf)
    assemble!(b, lf)

    vₕ = element(Wₕ, 1.0)
    lf(vₕ)
    evaluate!(b, lf, vₕ)
    return nothing
end

# The threaded sweep as well. Worth having for the two shapes that fix its specialisations:
# one reaching only its own point, which colours into a single flat pass, and one reaching a
# neighbour, which takes the strided path instead.
function _pc_assemble_shape_threaded(Wₕ, g, b)
    _pc_assemble_shape(Wₕ, g, b)

    lf = form(Wₕ, g)
    assemble_parallel!(b, lf)
    return nothing
end

# Bilinear assembly shapes: allocation, matrix pattern, in-place, parallel,
# Dirichlet boundary application, contraction, and structural symmetry.
function _pc_assemble_bilinear_shape(Wₕ, g, label::Symbol)
    bf = form(Wₕ, Wₕ, g)
    ast = resolve_form_ast(bf)

    trial_space(bf)
    test_space(bf)
    issymmetric(bf)
    isposdef(bf)

    A = allocate_system_matrix(bf, ast)
    assemble!(A, bf)
    assemble!(A, bf; dirichlet = label)
    assemble(bf)
    assemble(bf; dirichlet = label)

    uₕ = element(Wₕ, 1.0)
    bf(uₕ, uₕ)
    return nothing
end

function _pc_assemble_bilinear_shape_threaded(Wₕ, g, label::Symbol)
    _pc_assemble_bilinear_shape(Wₕ, g, label)

    bf = form(Wₕ, Wₕ, g)
    ast = resolve_form_ast(bf)
    A = allocate_system_matrix(bf, ast)
    assemble_parallel!(A, bf)
    return nothing
end

# A composite bilinear form reaches leaf block extraction, diagonal blocks,
# and off-diagonal coupled sweeps.
function _pc_assemble_bilinear_composite(Vₕ, g)
    bf = form(Vₕ, Vₕ, g)
    ast = resolve_form_ast(bf)

    A = allocate_system_matrix(bf, ast)
    assemble!(A, bf)
    assemble(bf)
    return nothing
end

_pc_assemble_bilinear_directional(Wₕ, label, ::Val{1}) = nothing

function _pc_assemble_bilinear_directional(Wₕ, label, ::Val{2})
    _pc_assemble_bilinear_shape(Wₕ, (u, v) -> inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v)), label)
    return nothing
end

# A composite form reaches two more cores: the per-leaf sweep for terms that mean the same
# thing on every block, and `_route_terms!` for terms that name a component.
function _pc_assemble_composite(Vₕ, g, b)
    lf = form(Vₕ, g)

    assemble(lf)
    assemble!(b, lf)
    return nothing
end

# The transverse directions, which only exist above 1D.
_pc_assemble_directional(Wₕ, uₕ, b, ::Val{1}) = nothing

function _pc_assemble_directional(Wₕ, uₕ, b, ::Val{2})
    _pc_assemble_shape(Wₕ, v -> innerₕ(uₕ, D₋ᵧ(v)), b)
    return nothing
end

function _pc_form_assembly(
        Ωₕ::AbstractMeshType, Wₕ, Vₕ, label::Symbol, f, dim_val::Val{D}
) where {D}
    uₕ = Rₕ(Wₕ, f)
    b = zeros(eltype(Wₕ), ndofs(Wₕ))
    bv = zeros(eltype(Vₕ), ndofs(Vₕ))

    # Display of the space, element and form cluster: `show`, `MIME"text/plain"` and
    # `summary` are separate methods, and the composite space's block takes another branch
    # from the scalar one, so both are named.
    _pc_display(Wₕ)
    _pc_display(Vₕ)
    _pc_display(uₕ)
    _pc_display(form(Wₕ, v -> innerₕ(uₕ, v)))
    _pc_display(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))

    # One color, then two.
    _pc_assemble_shape_threaded(Wₕ, v -> innerₕ(uₕ, v), b)
    _pc_assemble_shape_threaded(Wₕ, v -> innerₕ(uₕ, D₋ₓ(v)), b)

    # The other weight, a sum of two kinds of inner product, and a linear combination in the
    # test argument: the three shapes a form is most likely to be written as.
    _pc_assemble_shape(Wₕ, v -> inner₊ₓ(uₕ, D₋ₓ(v)), b)
    _pc_assemble_shape(Wₕ, v -> innerₕ(uₕ, v) + inner₊ₓ(uₕ, D₋ₓ(v)), b)
    _pc_assemble_shape(Wₕ, v -> innerₕ(uₕ, v + 2 * D₋ₓ(v) - Mₓ(v)), b)
    _pc_assemble_directional(Wₕ, uₕ, b, dim_val)

    # A raw closure source, not a VectorElement: exercises form(Wₕ, f)'s eager lowering
    # (`_lower_sources_for_space`/`_lower_sources`, gpena/Bramble.jl#197) through the real
    # constructor, unlike `_pc_symbolic_operators`' `innerₕ(x -> 1.0, v)` above, which only
    # builds the bare AST node from symbolic `TrialFunction`/`TestFunction` markers and never
    # reaches `form`'s constructor or a real space at all. Every call above this one sources
    # from `uₕ`, a `VectorElement`, so none of them touch this path either.
    _pc_assemble_shape(Wₕ, v -> innerₕ(f, v), b)
    # `f * uₕ` (gpena/Bramble.jl#197): the operator only `form`'s own lowering needed
    # otherwise had no caller anywhere in this workload.
    _pc_assemble_shape(Wₕ, v -> innerₕ(f * uₕ, v), b)

    # Written out per component, the shorthand that sums them, and a routed term carrying
    # operators.
    uv = Rₕ(Vₕ, (f, f))
    c = components(uv)
    _pc_assemble_composite(Vₕ, v -> innerₕ(c[1], v(1)) + innerₕ(c[2], v(2)), bv)
    _pc_assemble_composite(Vₕ, v -> innerₕ(uv, v), bv)
    _pc_assemble_composite(Vₕ, v -> innerₕ(c[1], v(1) + D₋ₓ(v(1))) + innerₕ(c[2], v(2)), bv)

    # A raw closure source naming one leaf's component (gpena/Bramble.jl#197):
    # `_lower_sources_over_leaves`' target-known branch, sampling against that leaf's own
    # space -- a different path from the shared-across-leaves one right below it.
    _pc_assemble_composite(Vₕ, v -> innerₕ(f, v(1)), bv)
    # A raw closure source naming no component: goes to every leaf, and is deliberately left
    # unlowered (`_lower_sources_over_leaves`'s `target === nothing` branch) since leaves may
    # not share a mesh -- checked here so that branch's own code path is precompiled too.
    _pc_assemble_composite(Vₕ, v -> innerₕ(f, v), bv)

    # and the constrained right-hand side, which is a different path from the bare one
    bcs = dirichlet_constraints(Ωₕ, label => f)
    lf = form(Wₕ, v -> innerₕ(uₕ, v))
    assemble(lf; dirichlet = bcs)

    # The joint (A, F) entry point, every `dirichlet` shape it accepts, with and without
    # symmetrize: a `label => f` Pair, a Tuple of one, and pre-built constraints.
    af = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
    assemble(af, lf; dirichlet = label => f)
    assemble(af, lf; dirichlet = (label => f,), symmetrize = true)
    assemble(af, lf; dirichlet = bcs, symmetrize = true)

    _pc_assemble_bilinear_shape_threaded(Wₕ, (u, v) -> innerₕ(u, v), label)
    _pc_assemble_bilinear_shape_threaded(Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)), label)
    _pc_assemble_bilinear_shape(Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)), label)
    _pc_assemble_bilinear_directional(Wₕ, label, dim_val)

    # Composite bilinear forms: diagonal and coupled
    _pc_assemble_bilinear_composite(Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
    _pc_assemble_bilinear_composite(
        Vₕ, (u, v) -> inner₊ₓ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(1))
    )

    return nothing
end

# `assemble_add!`, `reaction` and `reaction_density` are separate entry points over a real
# `BilinearForm`/`LinearForm`/`VectorElement`, not reached by the bare AST/stencil
# construction above or by any `assemble`/`assemble!` shape already warmed --
# `assemble_add!` takes its own cached-record path with an explicit scale
# (`_assemble_bilinear_core_cached!`/`_assemble_linear_core!` in src/assembly/assemble_add.jl),
# and `reaction`/`reaction_density` walk the unconstrained residual through
# `leaf_spaces_offsets` (gpena/Bramble.jl#283).
#
# Builds its own non-uniform mesh with a single `:boundary` marker covering the whole
# boundary, rather than reusing the session's mesh/label: `markers(Ωₕ)` is a `NamedTuple`
# keyed by label, so a caller whose domain names its boundary `:boundary` (one label) rather
# than `:left`/`:right`/`:wall` (this file's other sessions) hits fresh method instances --
# reusing `label`/`Ωₕ` from the caller left this cold even though the AST/stencil shapes
# were already warm. The Laplacian/`innerₕ` shapes here match what a caller reaching for a
# net-flux boundary quantity actually writes.
# Split by dimension (rather than branching on `D` inside one method) so JET resolves each
# call concretely instead of analysing both branches together against a single generic `D`:
# a `Wₕ` inferred over the `Union` of the 1D/2D grid space types lets the union-split analysis
# pair a 1D `TrialFunction` with a 2D `TestFunction` in the body below, which is never a call
# either concrete session makes. `Val(1)`/`Val(2)` each build their own mesh and delegate to
# the shared, dimension-agnostic body once `Wₕ` is concrete.
function _pc_form_reaction(::Val{1})
    S = interval(0.0, 1.0)
    Ω = domain(S, :boundary => boundary_symbols(S))
    Ωₕ = mesh(Ω, 8, false)
    return _pc_form_reaction(gridspace(Ωₕ))
end

function _pc_form_reaction(::Val{2})
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ω = domain(S, :boundary => boundary_symbols(S))
    Ωₕ = mesh(Ω, (8, 8), (false, false))
    return _pc_form_reaction(gridspace(Ωₕ))
end

function _pc_form_reaction(Wₕ::ScalarGridSpace)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    fₕ = Rₕ(Wₕ, x -> 1.0)
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    A = assemble(a)
    F = assemble(l)
    uₕ = element(Wₕ, 1.0)

    assemble_add!(A, a, 0.5)
    assemble_add!(F, l, 0.5)
    reaction(A, F, uₕ; marker = :boundary)
    reaction_density(A, F, uₕ; marker = :boundary)

    # The calls above are static -- the compiler sees the concrete types at the call site
    # and can inline/constprop through them, so they warm the method *bodies* without
    # necessarily caching a standalone method instance for the entry signature itself. A
    # dynamically dispatched caller (a REPL session, or anything calling through a
    # `Function` value) resolves that entry signature fresh and pays for it. `precompile`
    # forces the standalone instance to exist, for the positional methods and for the
    # `marker` keyword's `Core.kwcall` wrapper (`reaction`/`reaction_density` are
    # keyword-only in `marker`).
    nt = (; marker = :boundary)
    precompile(assemble_add!, (typeof(A), typeof(a), typeof(0.5)))
    precompile(assemble_add!, (typeof(F), typeof(l), typeof(0.5)))
    precompile(Core.kwcall, (typeof(nt), typeof(reaction), typeof(A), typeof(F), typeof(uₕ)))
    precompile(
        Core.kwcall, (typeof(nt), typeof(reaction_density), typeof(A), typeof(F), typeof(uₕ))
    )

    # A constant times the Laplacian, an `OperatorScale` AST no other session builds:
    # alone and jointly with a Dirichlet value.
    aκ = form(Wₕ, Wₕ, (u, v) -> 2.0 * inner₊(∇ₕ(u), ∇ₕ(v)))
    assemble(aκ; dirichlet = :boundary)
    assemble(aκ, l; dirichlet = :boundary => x -> 0.0)

    # The mass form added into a matrix, the first term of an implicit time step.
    m = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
    assemble_add!(A, m, 0.5)
    precompile(assemble_add!, (typeof(A), typeof(m), Float64))
    return nothing
end

function _pc_form_session(
        Ωₕ::AbstractMeshType, be, label::Symbol, f, ft, I_time, dim_val::Val{D}
) where {D}
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(2))

    id, u, v = _pc_form_ast(Wₕ, dim_val)
    _pc_form_stencils(Ωₕ, Wₕ, id, u, v, label)
    _pc_form_blocks(Vₕ, dim_val)
    _pc_form_dirichlet(Ωₕ, Wₕ, Vₕ, be, label, f, ft, I_time)
    _pc_form_assembly(Ωₕ, Wₕ, Vₕ, label, f, dim_val)
    _pc_form_reaction(dim_val)
    _pc_form_nested(dim_val)

    # the composite space reaches the same nodes through a different space type
    _pc_form_ast(Vₕ, dim_val)
    return nothing
end

# --- The nested forms the ghost relations and `Δₕ` need (gpena/Bramble.jl#205, #774) ----- #
#
# A nested form is a type of its own: the resolved node types and the space type are the
# cache key, so each spelling of the same form hits the instance warmed here, and a form
# with another nesting, or on another space, compiles from scratch on the caller's first
# `assemble`. These are the forms warmed, each with its dimension and the side it acts on.
#
#   plain space, 1D and 2D
#     - `Δₕ(u)` trial side, `Δₕ(v)` test side in a bilinear form, `Δₕ(v)` in a linear one
#     - `Δₕ(Δₕ(u))` trial side
#     - in 2D the mixed `Dcₓ(Dcᵧ(u))` and `D̽ₓ(D̽ᵧ(u))`, trial side
#   plain space, 3D (src/precompile.jl), the Laplacian `Δₕ(u)`, trial side
#   ghost space (`gridspace(Ωₕ; ghosts = ...)`), 1D and 2D
#     - the relation rows `constrained(W, u -> F => α * u + ∂ₙ(u) => g)`, which `constrained`
#       builds as forms of the facet operator `∂ₙ` on each axis (depth 1, trial side); 2D
#       adds the conormal `∂ₙ(u; K)`, whose rows read every axis, and 1D adds `u => g` and `∂ₙₙ`
#     - on the constrained space, `-Δₕ(u)` trial side, `Δₕ(v)` test side, `∂ₙ(v)` inside
#       `inner_Γ` (test side, with the lifted solution as data), `∂ₙ(u)` and `jumpₙ(Δₕ(u))`
#       inside `inner_Γ` (trial side), and the joint assembly `assemble(a, l)`
#     - in 2D the mixed `Dcₓ(Dcᵧ(u))` and `D̽ₓ(D̽ᵧ(u))` on the constrained space, trial side
#
# Not warmed: `Δₕ(Δₕ(u))` on a ghost space (the second Laplacian reads past the one ghost
# layer, so the form is refused), and 3D ghost spaces. Float64 only. The meshes are
# non-uniform, built here with one `:boundary` marker like `_pc_form_reaction`'s.
function _pc_form_nested(::Val{1})
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
    return _pc_form_nested(Ωₕ, Val(1))
end

function _pc_form_nested(::Val{2})
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, false))
    return _pc_form_nested(Ωₕ, Val(2))
end

function _pc_form_nested(Ωₕ::AbstractMeshType, dim_val::Val)
    _pc_nested_plain(gridspace(Ωₕ), dim_val)
    _pc_nested_relations(gridspace(Ωₕ; ghosts = :boundary), dim_val)
    return nothing
end

function _pc_nested_plain(Wₕ::ScalarGridSpace, dim_val::Val)
    fₕ = Rₕ(Wₕ, x -> 1.0)
    assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(Δₕ(u), v)))
    assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, Δₕ(v))))
    assemble(form(Wₕ, v -> innerₕ(fₕ, Δₕ(v))))
    assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(Δₕ(Δₕ(u)), v)))
    _pc_nested_mixed(Wₕ, dim_val)
    return nothing
end

# The mixed second differences need two directions.
_pc_nested_mixed(Wₕ, ::Val{1}) = nothing

function _pc_nested_mixed(Wₕ, ::Val{2})
    assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(Dcₓ(Dcᵧ(u)), v)))
    assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(D̽ₓ(D̽ᵧ(u)), v)))
    return nothing
end

# Each `constrained` call builds its relation rows, and its space is then assembled on
# through the same body: the space type is part of the key.
function _pc_nested_relations(W::GhostGridSpace, ::Val{1})
    robin = constrained(W, u -> (:xmin => 2.0 * u + ∂ₙ(u) => 0.5, :xmax => ∂ₙ(u) => x -> x[1]))
    _pc_nested_constrained(robin)
    mixed = constrained(
        W, u -> (:xmin => u => 0.0, :xmin => ∂ₙ(u) => 1.0, :xmax => 1.5 * u + ∂ₙₙ(u) => 1.0)
    )
    a = form(mixed, mixed, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
    l = form(mixed, v -> innerₕ(x -> 1.0, v))
    assemble(a, l)
    return nothing
end

function _pc_nested_relations(W::GhostGridSpace, ::Val{2})
    facets = (:xmin, :xmax, :ymin, :ymax)
    robin = constrained(W, u -> map(F -> F => 2.0 * u + ∂ₙ(u) => 0.5, facets))
    _pc_nested_constrained(robin)
    K = [2.0 0.5; 0.5 1.0]
    conormal = constrained(W, u -> map(F -> F => 2.0 * u + ∂ₙ(u; K) => 1.0, facets))
    a = form(conormal, conormal, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
    l = form(conormal, v -> innerₕ(x -> 1.0, v))
    assemble(a, l)
    assemble(form(robin, robin, (u, v) -> innerₕ(Dcₓ(Dcᵧ(u)), v)))
    assemble(form(robin, robin, (u, v) -> innerₕ(D̽ₓ(D̽ᵧ(u)), v)))
    return nothing
end

function _pc_nested_constrained(C)
    a = form(C, C, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
    assemble(a, form(C, v -> innerₕ(x -> 1.0, v)))
    assemble(a, form(C, v -> inner_Γ(Δₕ(lift(C)), ∂ₙ(v))))
    assemble(form(C, C, (u, v) -> innerₕ(u, Δₕ(v))))
    assemble(form(C, C, (u, v) -> inner_Γ(∂ₙ(u), v)))
    assemble(form(C, C, (u, v) -> inner_Γ(jumpₙ(Δₕ(u)), v)))
    return nothing
end

# --- Jacobian sparsity and per-type assembly caching (gpena/Bramble.jl#21/#95/#20) ------- #
#
# Neither is reachable from the sessions above. jacobian_pattern walks a BilinearForm's own
# local_stencil through a fresh set of helpers (_coefficient_offsets, _resolve_dependency_ops,
# _pattern_term_jacobian! for the composite dispatch) that nothing else here calls.
# type_cached_assemble! is its own dispatch on the element type `T`, not
# exercised anywhere assemble/assemble! already are.
_pc_alpha(u) = 3.0 + 1.0 / (1.0 + u^2)

# The direct (uncached) build jacobian_pattern only needs a BilinearForm from -- how it was
# assembled makes no difference to the pattern.
function _pc_diffusion_form(Wₕ, uₕ)
    αvals = element(Wₕ, eltype(uₕ))
    αvals .= _pc_alpha.(Mₓ(uₕ))
    return form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * D₋ₓ(U), D₋ₓ(V)))
end

# `build` for type_cached_assemble!, built once per session (not re-literalized per call,
# the same economy its own docstring asks a caller to observe) and closing over `Wₕ`.
function _pc_build_diffusion(Wₕ)
    return function (uₕ)
        Mu = element(Wₕ, eltype(uₕ))
        αvals = element(Wₕ, eltype(uₕ))
        a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * D₋ₓ(U), D₋ₓ(V)))
        refill!(uₕ) = begin
            Mₓ!(Mu, uₕ)
            αvals .= _pc_alpha.(Mu)
        end
        return a, refill!
    end
end

function _pc_jacobian_pattern_session(Wₕ)
    u0 = element(Wₕ, 0.0)
    jacobian_pattern(_pc_diffusion_form(Wₕ, u0), U -> Mₓ(U))
    return nothing
end

# The composite dispatch (#95): a coefficient naming a different leaf than the block it
# sits in, both directions, the same shape as the coupled reaction-diffusion example.
function _pc_jacobian_pattern_composite_session(Vₕ)
    v0 = element(Vₕ, 0.0)
    c = components(v0)
    ac = form(Vₕ, Vₕ, (p, q) -> innerₕ(c[2] * p(1), q(1)) + innerₕ(c[1] * p(2), q(2)))
    jacobian_pattern(ac, U -> U(2), U -> U(1))
    return nothing
end

# Two calls, so both the cache-miss (build, allocate_system_matrix) and cache-hit (refill!
# only) paths get their own method instances.
function _pc_type_cached_assemble_session(Wₕ)
    build = _pc_build_diffusion(Wₕ)
    cache = Dict()
    u0 = element(Wₕ, 0.0)
    type_cached_assemble!(build, cache, u0)
    type_cached_assemble!(build, cache, u0)
    return nothing
end
