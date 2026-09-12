# precompile/form_sessions.jl: the symbolic form layer, assembly, Jacobian sparsity and
# per-type assembly caching (src/form/).
#
# The symbolic layer is not reachable from the mesh/space/operator sessions: a `LazyOp`
# tree is built from `IdentityOperator` and the trial/test leaves rather than from a grid
# function, so none of its constructors, stencil evaluators or traits are inferred by
# anything else there. Measured in a fresh session before this was added, the form paths
# cost about 580 ms of first-call latency, of which `stencil_offsets` alone was 176 ms.

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
    π_node = πₕ(Wₕ, u)
    is_symbolic(π_node)
    resolve_ast(π_node)
    return nothing
end
_pc_form_ast_interp(::AbstractSpaceType, u) = nothing

# Every node kind, built over one space, plus the traits each one answers. Returns a tree
# with a trial and a test leaf so the caller can reach the products.
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
    for op in (D₋ₓ(id), D₊ₓ(id), M₋ₓ(id), M₊ₓ(id), jumpₓ(id), Dcₓ(id), Dstar₊ₓ(id), Dₕₓ(id))
        is_symbolic(op)
        resolve_ast(op)
        stencil_offsets(op)
    end
    shift_op(id, 1, 1)
    restrict_to(:interior, D₋ₓ(id))

    # scaling, sums and the vector forms
    scaled = 3 * D₋ₓ(id)
    summed = D₋ₓ(id) + D₊ₓ(id)
    resolve_ast(scaled)
    resolve_ast(summed)
    stencil_offsets(summed)
    ∇₋ₕ(id)
    ∇₊ₕ(id)
    ∇ₕ(id)
    jumpₕ(id)
    Dcₕ(id)
    Dstar₊ₕ(id)
    M₋ₕ(id)
    M₊ₕ(id)

    return id, u, v
end

# The products, and the stencils they evaluate to. This is the path assembly will take.
function _pc_form_stencils(Ωₕ::AbstractMeshType, Wₕ, id, u, v, label::Symbol)
    idx = indices(Ωₕ)
    lin = LinearIndices(idx)
    I = first(idx)
    mk = markers(Ωₕ)

    # Bilinear and linear products for each weight kind.
    for prod in (innerₕ(D₋ₓ(id), D₋ₓ(id)), inner₊ₓ(M₋ₓ(id), M₋ₓ(id)), innerₕ(id, D₋ₓ(id)))
        local_stencil(prod, Wₕ, I, nothing, lin[I])
        local_stencil(prod, Wₕ, I, mk, lin[I])
        resolve_ast(prod)
    end

    # Every node kind evaluated once, with and without a marker table: restriction is
    # the only node that reads it, and `nothing` is a separate method there.
    for op in (
        id,
        D₋ₓ(id),
        D₊ₓ(id),
        M₋ₓ(id),
        jumpₓ(id),
        Dcₓ(id),
        Dstar₊ₓ(id),
        Dₕₓ(id),
        shift_op(id, 1, 1),
        3 * D₋ₓ(id),
        D₋ₓ(id) + D₊ₓ(id),
        restrict_to(:interior, id),
        restrict_to(label, id),
    )
        local_stencil(op, Wₕ, I, nothing, lin[I])
        local_stencil(op, Wₕ, I, mk, lin[I])
    end

    # Symbolic inner products over trial and test leaves, including tuple forms.
    innerₕ(u, v)
    innerₕ(2.0, v)
    innerₕ(x -> 1.0, v)
    inner₊(u, D₋ₓ(v))
    inner₊(D₋ₓ(u), D₋ₓ(v))
    inner₊(∇₋ₕ(u), ∇₋ₕ(v))
    inner₊ₓ(u, v)
    inner₊ᵧ(u, v)
    return nothing
end

# Block extraction for a coupled form: the symbolic arguments, and the split into blocks.
function _pc_form_blocks(Vₕ, ::Val{D}) where {D}
    n = n_leaf_spaces(Vₕ)
    leaves = leaf_spaces_offsets(Vₕ)

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
    dirichlet_bc_symmetrize!(A, F, Ωₕ, label)
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
# 3D is deliberately absent, as it is in the sessions above. Its shapes are the most
# expensive to reach (78 to 114 ms apiece) and introduce three more specializations of
# every core, paying build costs whether or not the caller computes in 3D.
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
    assemble!(A, bf; dirichlet=label)
    assemble(bf)
    assemble(bf; dirichlet=label)

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

    # Display for the space/element/form cluster (gpena/Bramble.jl#17, #45). None of it
    # was reachable from the rest of this workload: these types had no `show` of their own
    # until they gained one, and each has two independent methods now (the embeddable
    # one-liner and the `MIME"text/plain"` block) plus a `summary` that array headers
    # reach for. The composite space's detailed block takes a different branch from the
    # scalar one, so both are named.
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
    _pc_assemble_shape(Wₕ, v -> innerₕ(uₕ, v + 2 * D₋ₓ(v) - M₋ₓ(v)), b)
    _pc_assemble_directional(Wₕ, uₕ, b, dim_val)

    # Written out per component, the shorthand that sums them, and a routed term carrying
    # operators.
    uv = Rₕ(Vₕ, (f, f))
    c = components(uv)
    _pc_assemble_composite(Vₕ, v -> innerₕ(c[1], v(1)) + innerₕ(c[2], v(2)), bv)
    _pc_assemble_composite(Vₕ, v -> innerₕ(uv, v), bv)
    _pc_assemble_composite(Vₕ, v -> innerₕ(c[1], v(1) + D₋ₓ(v(1))) + innerₕ(c[2], v(2)), bv)

    # and the constrained right-hand side, which is a different path from the bare one
    bcs = dirichlet_constraints(Ωₕ, label => f)
    lf = form(Wₕ, v -> innerₕ(uₕ, v))
    assemble(lf; dirichlet=bcs)

    # The joint (A, F) entry point, every `dirichlet` shape it accepts, with and without
    # symmetrize: a `label => f` Pair, a Tuple of one, and pre-built constraints.
    af = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
    assemble(af, lf; dirichlet=label => f)
    assemble(af, lf; dirichlet=(label => f,), symmetrize=true)
    assemble(af, lf; dirichlet=bcs, symmetrize=true)

    # Bilinear forms: mass, stiffness, combination, and transverse
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

    # the composite space reaches the same nodes through a different space type
    _pc_form_ast(Vₕ, dim_val)
    return nothing
end

# --- Jacobian sparsity and per-type assembly caching (gpena/Bramble.jl#21/#95/#20) ------- #
#
# Neither is reachable from the sessions above. jacobian_pattern walks a BilinearForm's own
# local_stencil through a fresh set of helpers (_coefficient_offsets, _pattern_term!,
# _resolve_dependency_ops, _pattern_term_jacobian! for the composite dispatch) that nothing
# else here calls. type_cached_assemble! is its own dispatch on the element type `T`, not
# exercised anywhere assemble/assemble! already are.
_pc_alpha(u) = 3.0 + 1.0 / (1.0 + u^2)

# The direct (uncached) build jacobian_pattern only needs a BilinearForm from -- how it was
# assembled makes no difference to the pattern.
function _pc_diffusion_form(Wₕ, uₕ)
    αvals = element(Wₕ, eltype(uₕ))
    αvals .= _pc_alpha.(M₋ₓ(uₕ))
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
            M₋ₓ!(Mu, uₕ)
            αvals .= _pc_alpha.(Mu)
        end
        return a, refill!
    end
end

function _pc_jacobian_pattern_session(Wₕ)
    u0 = element(Wₕ, 0.0)
    jacobian_pattern(_pc_diffusion_form(Wₕ, u0), U -> M₋ₓ(U))
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
