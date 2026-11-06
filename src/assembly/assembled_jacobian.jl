# assembled_jacobian.jl
#
# The exact Jacobian of a Newton residual `R(u) = A(u) u - F`, assembled from the form's AST
# as one form, `a + ∂a` (gpena/Bramble.jl#1141; the design spike is #1118,
# `docs/notes/internals/assembled_jacobian_spike.md`).
#
# The state reaches `A` only through coefficients the caller declares, `cᵢ = fᵢ.(Lᵢ u)`,
# each read by a `GridFunctionScale` node `cᵢ * T(U)`. The product rule there gives
#
#     δ(cᵢ ⊙ T(u)) = cᵢ ⊙ T(δu) + βᵢ ⊙ Lᵢ(δu),     βᵢ = fᵢ′.(Lᵢ u) ⊙ T(u),
#
# and every node above the coefficient is linear in its operand, so it is rebuilt around
# the derivative of its child (`_dterm`, one method per node kind). `∂a` is the sum of
# those rebuilt trees. At each state the coefficients and the β buffers are refilled
# pointwise (`Lᵢ u` and `T(u)` read with `local_stencil`, as assembly reads them), and one
# `assemble!` of `a + ∂a` writes the Jacobian: no `Dual` passes through assembly.
#
# A node on a path from the root to a registered coefficient with no rule here throws
# `MissingJacobianRule`, naming it. A subtree holding no registered coefficient contributes
# nothing; `_holds_coefficient` decides that by walking every field of every node and of
# every container below it, so a coefficient cannot hide in a field a rule does not read.
#
# On a composite space `T` and `L` each name a component (`_jf_leaves`), read their own
# slices of `u`, and `β ⊙ L(δu)` lands in `L`'s column block, routed by `assemble`.

"""
    MissingJacobianRule(node::Symbol, reason::String)

Thrown by [`jacobian_form`](@ref) when a node on a path from the form's root to a declared
coefficient has no derivative rule. `node` names the node's type. The term is refused
rather than dropped, so the Jacobian is never silently missing a part.
"""
struct MissingJacobianRule <: Exception
    node::Symbol
    reason::String
end

function Base.showerror(io::IO, e::MissingJacobianRule)
    return print(io, "MissingJacobianRule: no Jacobian rule for `", e.node,
        "` above a coefficient that reads the state (", e.reason,
        "); refusing rather than dropping its term")
end

"""
    _value_and_derivative(f, x) -> Tuple

`(f(x), f′(x))` for a scalar function `f`, from one evaluation of `f`. Its `src/` method
only throws, naming ForwardDiff; `BrambleForwardDiffExt` defines the one that runs.
"""
@noinline function _value_and_derivative(f, x)
    throw(
        ArgumentError(
        "jacobian! of a jacobian_form differentiates each coefficient's function with " *
        "ForwardDiff.jl. Add `using ForwardDiff` before calling it.",
    ),
    )
end

# One declared coefficient: `values == f.(L(u))`, `dependency(U) = L(U)`.
struct _JFCoefficient{C, L, F}
    values::C
    dependency::L
    f::F
end

@noinline function _throw_jf_declaration(k::Int, x)
    throw(
        ArgumentError(
        "jacobian_form: coefficient $k must be declared as `values => (U -> L(U), f)`, " *
        "got a $(typeof(x)).",
    ),
    )
end

function _jf_coefficient(k::Int, p::Pair)
    spec = p.second
    (spec isa Tuple && length(spec) == 2) || _throw_jf_declaration(k, spec)
    p.first isa AbstractVector || _throw_jf_declaration(k, p.first)
    return _JFCoefficient(p.first, spec[1], spec[2])
end
_jf_coefficient(k::Int, x) = _throw_jf_declaration(k, x)

# What `_dterm` carries down the walk: the declared coefficients, which of them were
# reached, and one refill per `GridFunctionScale` reading one of them.
struct _JFWalk{C, S}
    coefficients::C
    space::S
    reached::Vector{Bool}
    refills::Vector{Any}
end

_registered(x, coefficients) = findfirst(c -> c.values === x, coefficients)

# --- Whether a value holds a declared coefficient -------------------------------------- #

# Generic over fields and containers: a node with no rule must never hide a coefficient in a
# `NamedTuple`, a `Vector` or a struct field. Spaces and meshes hold no coefficient and are
# large, so the walk stops there; numeric arrays are compared, never walked element-wise.
function _holds_coefficient(x, coefficients)
    return _holds_coefficient!(Base.IdSet{Any}(), x, coefficients)
end

function _jf_opaque(x)
    x isa Union{Number, Symbol, AbstractString, Module, Type, Function, Ptr,
        AbstractSpaceType, AbstractMeshType}
end

function _holds_coefficient!(seen, x, coefficients)
    _registered(x, coefficients) === nothing || return true
    # A closure is a struct of its captures: walked, unlike a plain function.
    closure = x isa Function && nfields(x) > 0
    !closure && (_jf_opaque(x) || isbits(x)) && return false
    if ismutable(x)
        x in seen && return false
        push!(seen, x)
    end
    if x isa Array || x isa Memory
        eltype(x) <: Number && return false
        for i in eachindex(x)
            isassigned(x, i) && _holds_coefficient!(seen, x[i], coefficients) && return true
        end
        return false
    end
    for k in 1:nfields(x)
        isdefined(x, k) && _holds_coefficient!(seen, getfield(x, k), coefficients) &&
            return true
    end
    return false
end

# The fields of `op` other than `handled` must hold no coefficient: a rule reads only the
# fields it names.
function _check_other_fields(op, w::_JFWalk, handled::Tuple, reason::String)
    for f in fieldnames(typeof(op))
        f in handled && continue
        _holds_coefficient(getfield(op, f), w.coefficients) &&
            throw(MissingJacobianRule(nameof(typeof(op)), reason))
    end
    return nothing
end

# --- The derivative walk --------------------------------------------------------------- #

# The derivative subtree of `op` through the declared coefficients, `nothing` where none is
# read. The fallback: a node with no rule contributes nothing only if it holds none.
function _dterm(op::LazyOp, w::_JFWalk)
    _holds_coefficient(op, w.coefficients) || return nothing
    throw(MissingJacobianRule(nameof(typeof(op)), "no derivative rule for this node"))
end

_jf_add(::Nothing, ::Nothing) = nothing
_jf_add(a, ::Nothing) = a
_jf_add(::Nothing, b) = b
_jf_add(a, b) = OperatorAdd(a, b)

function _dterm(op::OperatorAdd, w::_JFWalk)
    return _jf_add(_dterm(op.left_op, w), _dterm(op.right_op, w))
end

function _dterm(op::OperatorScale, w::_JFWalk)
    _check_other_fields(op, w, (:inner_op,), "a coefficient inside the scalar")
    d = _dterm(op.inner_op, w)
    return d === nothing ? nothing : OperatorScale(op.scalar, d)
end

# Differences, averages, jumps and shifts are linear: the rule rebuilds the node around the
# derivative of its operand. A shifted coefficient is read at the shifted point and zeroed
# off the grid by the node itself, and β is read there the same way (#352).
function _dterm(op::Union{TappedNode, ShiftNode}, w::_JFWalk)
    _check_other_fields(op, w, (:inner_op,), "a coefficient in the node's own data")
    d = _dterm(op.inner_op, w)
    return d === nothing ? nothing : _rebuilt_around(op, d)
end

# Unprobed (`()` outside the region): the zero stencil `form` may have stored is typed for
# the old operand.
function _dterm(op::RegionRestriction{D, R}, w::_JFWalk) where {D, R}
    _check_other_fields(op, w, (:inner_op, :zero_stencil), "a coefficient in the region")
    d = _dterm(op.inner_op, w)
    d === nothing && return nothing
    return RegionRestriction{D, R, typeof(d)}(op.region, d)
end

# Every weight multiplies the two sides at the same point, `w_I (L u)_I (R v)_I`, so a
# scale at the top of the test side (no operator above it) moves to the trial side:
# `(L u, c ⊙ R v) = (c ⊙ L u, R v)`. A Robin term under `inner_Γ` is one of these
# products, its weight the face's measure, and needs no rule of its own. Below an operator
# on the test side, the derivative would put the trial function under that operator:
# refused.
function _dterm(op::BilinearProduct{D, W}, w::_JFWalk) where {D, W}
    _check_other_fields(op, w, (:left_op, :right_op), "a coefficient in the weight")
    left, right = _jf_to_trial(op.left_op, op.right_op, w)
    d = _dterm(left, w)
    d === nothing && return nothing
    return BilinearProduct{D, W, typeof(d), typeof(right)}(op.inner, d, right)
end

# The pair `(left, right)` with the test side's top scales moved onto the trial side, as
# long as the test side holds a declared coefficient.
function _jf_to_trial(left, right, w::_JFWalk)
    _holds_coefficient(right, w.coefficients) || return left, right
    right isa GridFunctionScale &&
        return _jf_to_trial(GridFunctionScale(right.grid_function, left), right.inner_op, w)
    right isa OperatorScale &&
        return _jf_to_trial(OperatorScale(right.scalar, left), right.inner_op, w)
    throw(MissingJacobianRule(:BilinearProduct,
        "coefficient on the test side, under $(nameof(typeof(right)))"))
end

@noinline function _throw_jf_dependency(k::Int, what::String)
    throw(ArgumentError("jacobian_form: the dependency of coefficient $k $what."))
end

# The product rule: δ(c ⊙ T(u)) = (f′(L u) ⊙ T(u)) ⊙ L(δu) + c ⊙ T(δu).
function _dterm(op::GridFunctionScale{D}, w::_JFWalk) where {D}
    k = _registered(op.grid_function, w.coefficients)
    k === nothing && _holds_coefficient(op.grid_function, w.coefficients) &&
        throw(MissingJacobianRule(:GridFunctionScale,
            "a coefficient inside the scale's grid function, not the grid function itself"))
    inner = _dterm(op.inner_op, w)
    own = inner === nothing ? nothing : GridFunctionScale(op.grid_function, inner)
    k === nothing && return own
    _has_trial_interp(op.inner_op) &&
        throw(MissingJacobianRule(:InterpolationNode, "πₕ below the coefficient"))
    c = w.coefficients[k]
    L = c.dependency(TrialFunction{D}())
    L isa LazyOp{D} ||
        _throw_jf_dependency(k, "must return one operator, got a $(typeof(L))")
    _has_trial_interp(L) && _throw_jf_dependency(k, "reads through πₕ, not supported here")
    # `L` is applied to `u` alone: a declared coefficient inside it (itself included) would
    # be read stale by the refill, and its own derivative lost from the chain rule.
    j = findfirst(x -> _holds_coefficient(L, (x,)), w.coefficients)
    j === nothing || _throw_jf_dependency(k,
        "reads declared coefficient $j; a dependency may read only the state and " *
        "undeclared (constant) grid functions")
    first_read = !w.reached[k]
    w.reached[k] = true
    sT, oT, sL, oL = _jf_leaves(w.space, op.inner_op, L, k)
    β = element(sT, zero(eltype(backend(sT))))
    T, mT = _bind_walk(op.inner_op, sT)
    Lb, mL = _bind_walk(L, sL)
    push!(w.refills, _JFRefill(β, c.values, Lb, mL, T, mT, c.f, first_read, sL, oL, sT, oT))
    return _jf_add(own, β * L)
end

# The leaf each of `T` and `L` reads and its offset into `u`: `(sT, oT, sL, oL)`. A scalar
# space is both, at no offset (`nothing`, so its refill reads `u` itself). On a composite
# space each names its component, as `jacobian_pattern` resolves a dependency
# (`_dependency_component`); β lives on `T`'s leaf, the grid every block it scales walks,
# and the term `β * L` lands in `L`'s column block, `assemble` routing it.
_jf_leaves(W::ScalarGridSpace, T, L, k::Int) = (W, nothing, W, nothing)
function _jf_leaves(W::CompositeGridSpace, T, L, k::Int)
    leaves = leaf_spaces_offsets(W)
    tc = trial_component_or_nothing(T)
    tc === nothing && throw(
        ArgumentError(
        "jacobian_form: coefficient $k scales a trial function naming no component of " *
        "the composite space, so its derivative differs in every diagonal block; scale " *
        "each component, `c * U(i)`.",
    ),
    )
    target = trial_component_or_nothing(L)
    target === nothing && _throw_jf_dependency(k,
        "names no component of the composite space; name one, as in `U -> U(2)`")
    sT, oT = _dependency_component(leaves, tc)
    sL, oL = _dependency_component(leaves, target)
    _same_mesh_or_throw(target, sL, mesh(sT))
    # A composite space's leaves are scalar: a ghost leaf throws in `leaf_spaces_offsets`.
    return sT::ScalarGridSpace, oT, sL::ScalarGridSpace, oL
end

# --- The refill ------------------------------------------------------------------------ #

# One coefficient read: `β = f′(L u) ⊙ T(u)` into `beta`, and, at the coefficient's first
# read, `values = f(L u)`. `L` and `T` are bound to the mesh's marker ids (`_bind_walk`)
# and read `u` on their leaves, `space_L`/`space_T`, at `offset_L`/`offset_T` (`_jf_leaves`).
struct _JFRefill{B, C, L, ML, T, MT, F, SL, OL, ST, OT}
    beta::B
    values::C
    L::L
    markers_L::ML
    T::T
    markers_T::MT
    f::F
    writes_values::Bool
    space_L::SL
    offset_L::OL
    space_T::ST
    offset_T::OT
end

# What `jacobian_form` returns: the form `a + ∂a`, its refills and its Dirichlet labels.
# Internal; read through `jacobian_prototype` and `jacobian!`.
struct JacobianForm{F, R, B}
    form::F
    refills::R
    dirichlet::B
end

"""
    jacobian_form(a::BilinearForm, coefficients::Pair...; dirichlet = nothing)

The exact Jacobian of the Newton residual `R(u) = A(u) u - F` as one form, `a + ∂a`, where
`a` assembles `A(u)` and reads coefficients that depend on the state. Each is declared as
`values => (U -> L(U), f)`: `values` is the grid function `a` reads (the same object, by
identity), `L` the stencil operator it is computed from, written as a function of the
trial placeholder as [`jacobian_pattern`](@ref) takes it, and `f` a scalar function, so
that `values == f.(L(u))`.

```julia
using Bramble: jacobian_form, jacobian_prototype, jacobian!
using ForwardDiff                      # differentiates each `f`
αx, αy = element(Wₕ), element(Wₕ)
a = form(Wₕ, Wₕ, (U, V) -> inner₊((αx * ∇ₕ(U)[1], αy * ∇ₕ(U)[2]), ∇ₕ(V)))
jf = jacobian_form(a, αx => (U -> Mₕ(U)[1], α), αy => (U -> Mₕ(U)[2], α);
    dirichlet = :boundary)
J = jacobian_prototype(jf)             # the pattern of a + ∂a, wider than A's in general
jacobian!(J, jf, u)                    # J = ∂R/∂u at u
```

`jacobian_prototype(jf)` allocates a matrix with the pattern of `a + ∂a`, which can be
wider than `A`'s. `jacobian!(J, jf, u)` writes the Jacobian at `u` into it: it sets every
declared coefficient to `f.(L(u))` and each derivative buffer to `f′.(L(u)) ⊙ T(u)`, where
`T` is the operator the coefficient multiplies, then assembles `a + ∂a` once, with the
Dirichlet labels `dirichlet` (as [`assemble!`](@ref) takes them). The coefficients are
written from `u` itself, so the Jacobian is exact at `u` whatever state they held before.
Each `f′` is one `ForwardDiff.Dual` evaluation of `f`, so ForwardDiff.jl must be loaded.
The refill runs under the mesh's execution policy.

The derivative is built by a walk over `a`'s AST: the product rule where a declared
coefficient scales an operator, and every node above it (scales, sums, differences,
averages, jumps, shifts, region restrictions, the trial side of an inner product) rebuilt
around the derivative of its operand. Every inner product's weight is pointwise, so a
boundary term such as a Robin term `inner_Γ(c * U, V; markers)` needs nothing more, and a
coefficient scaling the whole test side, `inner₊(∇ₕ(U)[1], c * ∇ₕ(V)[1])`, is moved to the
trial side. A declared coefficient under a node with no such rule, under an operator on the
test side, or in any field a rule does not read, throws [`MissingJacobianRule`](@ref)
naming the node. A declared coefficient the form never reads throws an `ArgumentError`, and
so does a dependency `L` that reads a declared coefficient (itself included): `L` may read
only the state and constant grid functions. Composite spaces are routed per block.

The trial and test spaces are both scalar grid spaces or both composite ones. On a
composite space each declared coefficient scales one component, `c * U(i)`, and its
dependency names the component it reads, `U -> L(U(j))`; the derivative term of
`c * U(i)` in the equation of component `k` then lands in block `(k, j)`, so a coefficient
reading another component couples the two. A coefficient scaling a trial function with no
component, or a dependency naming none, throws an `ArgumentError`, and so does a dependency
naming a component on another mesh.
"""
function jacobian_form(
        a::BilinearForm{D}, coefficients::Pair...; dirichlet = nothing) where {D}
    W = a.trial_space
    scalar = W isa ScalarGridSpace && !_is_block_pair(W, a.test_space)
    scalar || (W isa CompositeGridSpace && a.test_space isa CompositeGridSpace) ||
        throw(
            ArgumentError(
            "jacobian_form: the trial and test spaces must both be scalar grid spaces or " *
            "both composite ones, got a $(nameof(typeof(W))) and a " *
            "$(nameof(typeof(a.test_space))).",
        ),
        )
    coeffs = ntuple(k -> _jf_coefficient(k, coefficients[k]), length(coefficients))
    for k in eachindex(coeffs), j in 1:(k - 1)

        coeffs[j].values === coeffs[k].values && throw(
            ArgumentError(
            "jacobian_form: coefficients $j and $k are the same grid function.",
        ),
        )
    end
    w = _JFWalk(coeffs, W, fill(false, length(coeffs)), Any[])
    d = _dterm(a.ast, w)
    for k in eachindex(w.reached)
        w.reached[k] || throw(
            ArgumentError(
            "jacobian_form: coefficient $k is not read by the form's AST, so its " *
            "derivative would be lost.",
        ),
        )
    end
    ast = d === nothing ? a.ast : a.ast + d
    jform = form(W, a.test_space, (U, V) -> ast)
    return JacobianForm(jform, Tuple(w.refills), dirichlet)
end

jacobian_prototype(jf::JacobianForm) = allocate_system_matrix(jf.form)

function jacobian!(J, jf::JacobianForm, u::AbstractVector)
    W = jf.form.trial_space
    length(u) == ndofs(W) || throw(
        DimensionMismatch(
        "jacobian!: u has length $(length(u)), the space has $(ndofs(W)) dofs.",
    ),
    )
    policy = execution_policy(mesh(W))
    # Every coefficient first, then every β: a β reads the coefficients its `T` holds at
    # neighbouring points, so they must all be written before any β is.
    _jf_values!(jf.refills, policy, u)
    _jf_betas!(jf.refills, policy, u)
    assemble!(J, jf.form; dirichlet = jf.dirichlet)
    return J
end

# The slice of `u` a leaf at `offset` holds: `u` itself on a scalar space.
@inline _jf_slice(u, ::Nothing, sp) = u
@inline _jf_slice(u, offset::Int, sp) = view(u, (offset + 1):(offset + ndofs(sp)))

_jf_values!(::Tuple{}, policy, u) = nothing
function _jf_values!(refills::Tuple, policy, u)
    r = first(refills)
    W = r.space_L
    _run_bands!(policy, _jf_value_band!, r.beta, r.values, r.L, r.markers_L, r.f,
        r.writes_values, W, size(indices(mesh(W))), _jf_slice(u, r.offset_L, W))
    return _jf_values!(Base.tail(refills), policy, u)
end

_jf_betas!(::Tuple{}, policy, u) = nothing
function _jf_betas!(refills::Tuple, policy, u)
    r = first(refills)
    W = r.space_T
    _run_bands!(policy, _jf_beta_band!, r.beta, r.T, r.markers_T, W,
        size(indices(mesh(W))), _jf_slice(u, r.offset_T, W))
    return _jf_betas!(Base.tail(refills), policy, u)
end

# Σ weight·u over a point's stencil. An entry off the grid reads 0, as assembly drops it.
@inline _jf_dot(::Tuple{}, I, li, u) = zero(eltype(u))
@inline function _jf_dot(st::Tuple, I, li, u)
    off, wt = first(st)
    J = I + CartesianIndex(off)
    v = checkbounds(Bool, li, J) ? wt * @inbounds(u[li[J]]) : zero(wt) * zero(eltype(u))
    return v + _jf_dot(Base.tail(st), I, li, u)
end

# `f::F`: passed on uncalled, `f` would not be specialised on, and every point would box
# the derivative. The first pass stores `f′(L u)` in `beta`; the second multiplies it by
# `T(u)`.
@inline function _jf_value_band!(
        beta, values, L, mk, f::F, writes::Bool, W, dims::NTuple{D, Int}, u, nbands::Int,
        b::Int
) where {F, D}
    li = LinearIndices(dims)
    slab = _band_slab(axes(li), _band_range(axes(li, D), nbands, b))
    @inbounds for I in CartesianIndices(slab)
        i = li[I]
        x = _jf_dot(local_stencil(L, W, I, mk, i), I, li, u)
        v, dv = _value_and_derivative(f, x)
        writes && (values[i] = v)
        beta[i] = dv
    end
    return nothing
end

@inline function _jf_beta_band!(
        beta, T, mk, W, dims::NTuple{D, Int}, u, nbands::Int, b::Int
) where {D}
    li = LinearIndices(dims)
    slab = _band_slab(axes(li), _band_range(axes(li, D), nbands, b))
    @inbounds for I in CartesianIndices(slab)
        i = li[I]
        beta[i] *= _jf_dot(local_stencil(T, W, I, mk, i), I, li, u)
    end
    return nothing
end
