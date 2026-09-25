# precompile/semidiscretize_sessions.jl: `semidiscretize`, the `Semidiscretization`
# residual `(du, u, p, t)`, `Bramble.jacobian!`, `jacobian_prototype`, and display --
# src/problems/semidiscrete.jl, none of which needs SciMLBase.
#
# Not reachable from the form sessions: `Semidiscretization` is a new struct built on top
# of an already-assembled `BilinearForm`/`LinearForm` pair. Its constraint handling
# dispatches on one of four types (`NoConstraints`, `LabelsOnly`, `StaticConstraints`,
# `TimeDependentConstraints`), but only the first two are covered here -- measured in a
# fresh process, covering the other two bought nothing:
#
#     dirichlet     before (fresh)   after (this session)
#     nothing            ~261 ms            ~0.15 ms   -- covered
#     bare label          ~61 ms            ~1.4 ms    -- covered
#     label => f(x)       ~51 ms            ~55 ms     -- NOT covered (see below)
#     label => f(x, t)    ~56 ms            ~53 ms     -- NOT covered (see below)
#
# `StaticConstraints`/`TimeDependentConstraints` store the caller's own boundary closure
# and call it per boundary point at residual/Jacobian time (`dirichlet_bc!`,
# `apply_dirichlet_conditions!`) -- the same "one scalar point at a time" shape
# `Rₕ!`/`avgₕ!` have (bramble-performance §1), which is exactly the part a package's own
# workload cannot precompile for a *different* closure. `form(Wₕ, Wₕ, f)` does not have this
# problem: it resolves `f` into a structural `LazyOp` AST once, and it is that AST's type --
# shared across call sites with the same shape -- that the compiled `assemble`/stencil
# machinery is parameterised on (bramble-verification's "parameterised on the AST type"
# note), not the raw closure. A Dirichlet function is never resolved into anything, though:
# it stays exactly the closure the caller wrote, so warming it here only ever helps a second
# call to *this file's own* closure, never a user's. `NoConstraints` (no closure at all) and
# `LabelsOnly` (zero-fill, no closure) have no such barrier, which is the whole gap between
# the two "before" columns above holding up after this session and the other two not moving.
#
# Originally proposed as `nothing` and `label => f(x, t)` (gpena/Bramble.jl#141); the second
# of those turned out to be exactly the kind of call this file cannot help, while `bare
# label` -- not in the original proposal -- is a real, common case (a homogeneous boundary
# with no explicit function) that gets the same full win `nothing` does.
#
# Cost of this session: `julia --project=. -e 'using Bramble'` against a wiped compiled
# cache, before/after this file existed, three-run median each (a single launch swings
# 10-30% run to run, bramble-verification's own warning about separate-process comparisons
# -- the first single-run read here was +0.8 s before the repeat) -- 19.69 s -> 19.87 s,
# +0.18 s (<1%) of one-time package build time bought against ~320 ms of first-call latency
# per fresh session that hits either covered case.
#
# gpena/Bramble.jl#283 extended this file with the second-order semidiscretisation
# (`semidiscretize_second_order`, its `(dv, v, u, p, t)` residual and display,
# `src/problems/second_order_semidiscrete.jl`) and the matrix-free explicit RHS
# (`semidiscretize_rhs`, its `(du, u, p, t)` residual, `src/problems/semidiscrete_rhs.jl`).
# Neither is reachable from the first-order session above: a `SecondOrderSemidiscretization`
# and a `SemidiscretizeRHS` are new structs built on top of an already-assembled
# `BilinearForm`/`LinearForm` pair, same reasoning as `Semidiscretization` itself needing its
# own session.
#
# Both are warmed in 1D and 2D against their own tiny non-uniform spaces
# (`_pc_second_order_rhs_session`, one `:boundary` label covering the whole boundary),
# rather than reusing the uniform 1D space `_pc_semidiscretize_session` above is called with
# (from `src/precompile.jl`, owned elsewhere) -- matching the exact call shape (dimension,
# non-uniform mesh, single whole-boundary label) a caller building a non-uniform mesh (the
# common case, bramble-verification's "non-uniform is the point") actually hits.

# One space, one `a`/`l` pair, the two constraint kinds precompiling actually helps, plus
# the residual, `jacobian!`, `jacobian_prototype` and display paths against each.
function _pc_semidiscretize_session(Wₕ::ScalarGridSpace, label::Symbol)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    fₕ = Rₕ(Wₕ, x -> 1.0)
    l = form(Wₕ, v -> innerₕ(fₕ, v))

    n = ndofs(Wₕ)
    du = zeros(n)
    u = zeros(n)

    for dirichlet in (nothing, label)
        sd = semidiscretize(a, l; dirichlet = dirichlet)
        sd(du, u, nothing, 0.0)
        J = jacobian_prototype(sd)
        jacobian!(J, sd, u, nothing, 0.0)
        _pc_display(sd)
    end

    _pc_second_order_rhs_session(Val(1))
    _pc_second_order_rhs_session(Val(2))

    return nothing
end

# `semidiscretize_second_order`'s residual/display, and `semidiscretize_rhs`'s
# construction/residual, in 1D (`Val(1)`) and 2D (`Val(2)`) -- a tiny non-uniform space of
# its own, one `:boundary` label covering the whole boundary, built the same way as
# `_pc_semidiscretize_session`'s own 1D space above without depending on it or its label.
function _pc_second_order_rhs_session(::Val{1})
    S = interval(0.0, 1.0)
    Ω = domain(S, :boundary => boundary_symbols(S))
    Ωₕ = mesh(Ω, 4, false)
    return _pc_second_order_rhs_session(gridspace(Ωₕ))
end

function _pc_second_order_rhs_session(::Val{2})
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ω = domain(S, :boundary => boundary_symbols(S))
    Ωₕ = mesh(Ω, (3, 3), (false, false))
    return _pc_second_order_rhs_session(gridspace(Ωₕ))
end

function _pc_second_order_rhs_session(Wₕ::ScalarGridSpace)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    fₕ = Rₕ(Wₕ, x -> 1.0)
    l = form(Wₕ, v -> innerₕ(fₕ, v))

    n = ndofs(Wₕ)
    dv, v, u = zeros(n), zeros(n), zeros(n)

    sd2 = semidiscretize_second_order(a, l; dirichlet = :boundary)
    sd2(dv, v, u, nothing, 0.0)
    _pc_display(sd2)

    # A caller reaching `semidiscretize_second_order`/`sd2` through a reassigned `local`
    # (built in one closure, called from another) or otherwise through a non-inferred value
    # forces dynamic dispatch: the callee -- the `Core.kwcall` glue for the `dirichlet =`
    # keyword as much as the residual/show methods themselves -- is then resolved by runtime
    # type rather than inferred statically, and the compiled workload's own static calls above
    # do not cache that standalone, dynamically-reached entry point even though they cache the
    # exact same concrete types (confirmed with SnoopCompileCore.@snoop_inference: 0 triggers
    # for the static calls above in isolation, but the same calls reached dynamically still
    # show up as a fresh compile). `invokelatest` forces each of those same dynamically-reached
    # paths here, so they get cached too.
    Base.invokelatest(semidiscretize_second_order, a, l; dirichlet = :boundary)
    Base.invokelatest(sd2, dv, v, u, nothing, 0.0)
    Base.invokelatest(show, IOBuffer(), MIME"text/plain"(), sd2)

    sd = semidiscretize(a, l)
    rhs = semidiscretize_rhs(sd)
    du = zeros(n)
    rhs(du, u, nothing, 0.0)

    return nothing
end
