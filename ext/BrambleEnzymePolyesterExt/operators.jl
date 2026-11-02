# operators.jl: Enzyme rules for the `CpuPolyester` operator products: the fused
# matrix-free sweep (`_batch_mf_bands!`, src/assembly/matrix_free.jl) and the per-unit
# bilinear replays it reaches with an action target, the Kronecker lines
# (`_batch_kron_lines!`, src/assembly/kronecker.jl) and the compressed-row product
# (`_batch_csr_spmv!`, src/problems/semidiscrete_rhs.jl). Each computes a product linear in
# its input `x` and in the operator's values.
#
# Compressed rows, `du -= A u`: `du` only accumulates, so its cotangent passes through. The
# reverse in `u` is `ū -= Aᵀ d̄u`; a row-parallel loop would scatter it, so the augmented
# primal builds the transpose pattern (`A`'s columns, from the row pattern and `perm`) on
# the host, the tape keeps it, and one `@batch` gathers each `ū[j]` over its column. The
# reverse in the values, `nz̄[perm[k]] -= d̄u[i] u[colval[k]]`, writes each value once (`perm`
# is a permutation), one `@batch` over the rows. Forward adds `A u̇ + Ȧ u` in one `@batch`.
#
# Kronecker lines, `y = β y + Σₜ cₜ Kₜ x`. Forward runs the hook itself on the tangents
# (each line writes only its own slice), plus the terms through `ċ` and `β̇ y`. Reverse
# gathers `x̄ += Σₜ conj(cₜ) Kₜᴴ ȳ` through the hook on adjoint terms, built once on the
# host and kept on the tape; a line of the adjoint product is a line again, so the gather
# is the primal's own line loop and races nowhere. `c̄ₜ = ⟨Kₜ x, ȳ⟩` and
# `β̄ = ⟨y_old, ȳ⟩` reduce under `@batch`, then `ȳ` is scaled by `conj(β)` (cleared for
# `β == 0`, which overwrote `y`). The factors are the mesh's, so a factor with a shadow of
# its own is refused: no caller differentiates the operator's geometry.
#
# Matrix-free, `y += α A x` with `A` the form's and its coefficients captured in the form.
# The rules take the closure route of the replay hooks (assembly.jl), one Enzyme derivative
# of a serial band walk per `@batch` task through a thunk built on the host. The sink, and
# the plan with its form and walk, arrive behind `Ref`s (`_mf_run_bands!`), as do an action
# target and its parts at the per-unit replays. Forward runs Enzyme's forward derivative of
# each task's bands. In reverse `y` only accumulates, so `ȳ` is shared and read; each task
# replays its bands into a scratch `y` and pulls `ȳ` back into its own zeroed copy of `x̄`
# and of the parts' shadow (the coefficient elements), which the host adds into the
# caller's after the join, `x̄` under `@batch`. `x̄` needs the copies because a band's rows
# read `x` across the band edge. The form's spaces and AST cross in a `Ref` of their own, so
# a float the form holds by value (a scalar in its AST) adds into the caller's shadow too.
# `x` is kept on the tape for every live call, since it sits behind the sink's `Ref`, whose
# `overwritten` entry says nothing of `x`'s own buffer.
#
# Batch width: every rule takes `Duplicated` or `BatchDuplicated` arguments. The
# compressed-row rules run every lane in each `@batch` iteration, the Kronecker rules one
# hook launch per lane; the matrix-free rules run a batched forward thunk, and one width-1
# reverse thunk per lane in each task of one launch.

const _CsrSpmv = typeof(Bramble._batch_csr_spmv!)
const _KronLines = typeof(Bramble._batch_kron_lines!)
const _MFBands = typeof(Bramble._batch_mf_bands!)

# The derivative lane `w` of a scalar or a tuple of scalars: `nothing` for a `Const`, the
# tangent of a `Duplicated` (forward) otherwise. Never compared with the primal, as
# `_shadow` compares arrays: a tangent may equal its value.
@inline _scalar_lane(::Const, ::Int) = nothing
@inline _scalar_lane(::Active, ::Int) = nothing
@inline _scalar_lane(x::Union{Duplicated, DuplicatedNoNeed}, ::Int) = x.dval
@inline _scalar_lane(x::Union{BatchDuplicated, BatchDuplicatedNoNeed}, w::Int) = x.dval[w]

# One lane's `Active` derivatives returned from a reverse rule: the value at width 1, a
# tuple of lanes otherwise.
@inline _active_return(d::Tuple, ::Val{1}) = only(d)
@inline _active_return(d::Tuple, ::Val) = d

# --- Compressed rows: _batch_csr_spmv! ---------------------------------------------- #

# The columns of the row pattern `(rowptr, colval, perm)`: for column `j`, the entries
# `colptr[j]:(colptr[j + 1] - 1)` of `rowT` (their rows) and `permT` (their positions in
# `nzval`). A counting sort over the stored entries, on the host.
function _csr_columns(rowptr, colval, perm, ncols::Int)
    colptr = zeros(Int, ncols + 1)
    for k in eachindex(colval)
        @inbounds colptr[colval[k] + 1] += 1
    end
    colptr[1] = 1
    for j in 1:ncols
        @inbounds colptr[j + 1] += colptr[j]
    end
    next = colptr[1:ncols]
    rowT = Vector{Int}(undef, length(colval))
    permT = Vector{Int}(undef, length(colval))
    for i in 1:(length(rowptr) - 1)
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            @inbounds j = colval[k]
            @inbounds q = next[j]
            @inbounds rowT[q], permT[q] = i, perm[k]
            @inbounds next[j] = q + 1
        end
    end
    return colptr, rowT, permT
end

# Row `i` of every lane's tangent: `d_du[i] -= Σₖ nzval[perm[k]] d_u[colval[k]] +
# d_nz[perm[k]] u[colval[k]]`, an inactive `d_u` or `d_nz` left out.
@inline function _csr_tangent_lane!(d_du, d_u, d_nz, rowptr, colval, perm, nzval, u, i)
    d_du === nothing && return nothing
    acc = zero(eltype(d_du))
    @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
        d_u === nothing || (acc += nzval[perm[k]] * d_u[colval[k]])
        d_nz === nothing || (acc += d_nz[perm[k]] * u[colval[k]])
    end
    @inbounds d_du[i] -= acc
    return nothing
end

@inline function _csr_tangent_row!(d_dus, d_us, d_nzs, rowptr, colval, perm, nzval, u, i)
    map((a, b, c) -> _csr_tangent_lane!(a, b, c, rowptr, colval, perm, nzval, u, i),
        d_dus, d_us, d_nzs)
    return nothing
end

function _csr_forward!(d_dus, d_us, d_nzs, rowptr, colval, perm, nzval, u)
    @batch for i in 1:(length(rowptr) - 1)
        _csr_tangent_row!(d_dus, d_us, d_nzs, rowptr, colval, perm, nzval, u, i)
    end
    return nothing
end

# Column `j` of every lane's `ū -= Aᴴ d̄u`, gathered over the column's entries.
@inline function _csr_gather_lane!(d_u, d_du, colptr, rowT, permT, nzval, j)
    (d_u === nothing || d_du === nothing) && return nothing
    acc = zero(eltype(d_u))
    @inbounds for q in colptr[j]:(colptr[j + 1] - 1)
        acc += conj(nzval[permT[q]]) * d_du[rowT[q]]
    end
    @inbounds d_u[j] -= acc
    return nothing
end

@inline function _csr_gather_column!(d_us, d_dus, colptr, rowT, permT, nzval, j)
    map((a, b) -> _csr_gather_lane!(a, b, colptr, rowT, permT, nzval, j), d_us, d_dus)
    return nothing
end

# Row `i` of every lane's `nz̄[perm[k]] -= d̄u[i] conj(u[colval[k]])`.
@inline function _csr_values_lane!(d_nz, d_du, rowptr, colval, perm, u, i)
    (d_nz === nothing || d_du === nothing) && return nothing
    @inbounds g = d_du[i]
    @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
        d_nz[perm[k]] -= g * conj(u[colval[k]])
    end
    return nothing
end

@inline function _csr_values_row!(d_nzs, d_dus, rowptr, colval, perm, u, i)
    map((a, b) -> _csr_values_lane!(a, b, rowptr, colval, perm, u, i), d_nzs, d_dus)
    return nothing
end

function _csr_reverse!(d_dus, d_us, d_nzs, rowptr, colval, perm, nzval, u, cols)
    if cols !== nothing
        colptr, rowT, permT = cols
        @batch for j in 1:(length(colptr) - 1)
            _csr_gather_column!(d_us, d_dus, colptr, rowT, permT, nzval, j)
        end
    end
    if !all(isnothing, d_nzs)
        @batch for i in 1:(length(rowptr) - 1)
            _csr_values_row!(d_nzs, d_dus, rowptr, colval, perm, u, i)
        end
    end
    return nothing
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_CsrSpmv}, ::Type{RT},
        du::Annotation{<:AbstractVector}, rowptr::Annotation, colval::Annotation,
        perm::Annotation, nzval::Annotation, u::Annotation) where {RT}
    _entered!(:_batch_csr_spmv!)
    func.val(du.val, rowptr.val, colval.val, perm.val, nzval.val, u.val)
    W = Val(EnzymeRules.width(config))
    d_dus = _lanes(du, W)
    all(isnothing, d_dus) && return nothing
    _csr_forward!(d_dus, _lanes(u, W), _lanes(nzval, W), rowptr.val, colval.val, perm.val,
        nzval.val, u.val)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_CsrSpmv},
        ::Type{RT}, du::Annotation{<:AbstractVector}, rowptr::Annotation,
        colval::Annotation, perm::Annotation, nzval::Annotation,
        u::Annotation) where {RT}
    _entered!(:_batch_csr_spmv!)
    W = Val(EnzymeRules.width(config))
    live = !all(isnothing, _lanes(du, W))
    u_live = live && !all(isnothing, _lanes(u, W))
    nz_live = live && !all(isnothing, _lanes(nzval, W))
    # `overwritten` counts `func` first: `nzval` is entry 6, `u` entry 7.
    ow = EnzymeRules.overwritten(config)
    cols = u_live ? _csr_columns(rowptr.val, colval.val, perm.val, length(u.val)) : nothing
    nz_tape = u_live && ow[6] ? copy(nzval.val) : nothing
    u_tape = nz_live && ow[7] ? copy(u.val) : nothing
    func.val(du.val, rowptr.val, colval.val, perm.val, nzval.val, u.val)
    return AugmentedReturn(nothing, nothing, (cols, nz_tape, u_tape))
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_CsrSpmv}, ::Type{RT}, tape,
        du::Annotation{<:AbstractVector}, rowptr::Annotation, colval::Annotation,
        perm::Annotation, nzval::Annotation, u::Annotation) where {RT}
    W = Val(EnzymeRules.width(config))
    d_dus = _lanes(du, W)
    if !all(isnothing, d_dus)
        cols, nz_tape, u_tape = tape
        _csr_reverse!(d_dus, _lanes(u, W), _lanes(nzval, W), rowptr.val, colval.val,
            perm.val, something(nz_tape, nzval.val), something(u_tape, u.val), cols)
    end
    return ntuple(_ -> nothing, Val(6))
end

# --- Kronecker lines: _batch_kron_lines! -------------------------------------------- #

# A factor's adjoint as a host factor: a `Diagonal`'s conjugate, any other factor's adjoint
# as a CSC.
_kron_adjoint(F::Diagonal) = Diagonal(conj.(Bramble._kron_diag(F)))
_kron_adjoint(F::AbstractMatrix) = sparse(adjoint(F))

# The terms of `Kᴴ`, term by term (`(⊗ Fₐ)ᴴ = ⊗ Fₐᴴ`), built once on the host.
_kron_adjoint_terms(terms::Tuple) = map(
    t -> Bramble._kron_term(t.scales, map(_kron_adjoint, t.factors)), terms)

# Whether a term's factors carry a shadow of their own: the shadow's storage is not the
# primal's (runtime activity's inactive data is the primal itself).
_kron_storage(F::Diagonal) = Bramble._kron_diag(F)
_kron_storage(F::SparseMatrixCSC) = nonzeros(F)
_kron_storage(F) = F
function _kron_factor_shadowed(d::Tuple, t::Tuple)
    return any(map((dt, pt) -> any(map((a, b) -> _kron_storage(a) !== _kron_storage(b),
            dt.factors, pt.factors)), d, t))
end

@noinline function _throw_kron_factor_shadow()
    throw(ArgumentError("the Enzyme rule for `_batch_kron_lines!` differentiates the \
        product in `x` and the coefficients, not in the operator's factors; build the \
        `KroneckerLinearOperator` outside the differentiated function"))
end

function _kron_check_terms(terms, W::Val)
    terms isa Const && return nothing
    t = terms.val[]
    for d in _lanes(terms, W)
        d === nothing || !_kron_factor_shadowed(d[], t) || _throw_kron_factor_shadow()
    end
    return nothing
end

@inline function _kron_axpy_band!(y, a, x, n::Int, b::Int)
    for i in _band_range(Base.OneTo(length(y)), n, b)
        @inbounds y[i] += a * x[i]
    end
    return nothing
end

# `y .+= a .* x`, one band per task.
function _kron_axpy!(y, a, x)
    n = Threads.nthreads()
    @batch for b in 1:n
        _kron_axpy_band!(y, a, x, n, b)
    end
    return nothing
end

@inline function _kron_dot_band(a, b, n::Int, k::Int)
    s = zero(promote_type(eltype(a), eltype(b)))
    for i in _band_range(Base.OneTo(length(a)), n, k)
        @inbounds s += conj(a[i]) * b[i]
    end
    return s
end

# `⟨a, b⟩`, a band's partial sum per task.
function _kron_dot(a, b)
    n = Threads.nthreads()
    parts = zeros(promote_type(eltype(a), eltype(b)), n)
    @batch for k in 1:n
        @inbounds parts[k] = _kron_dot_band(a, b, n, k)
    end
    return sum(parts)
end

# The product's hook on the host terms `t`: `y = β y + Σₜ cs[t] Kₜ x`.
@inline _kron!(y, t, cs, x, β, ss, lines, m) = Bramble._batch_kron_lines!(
    y, t, cs, x, β, ss, lines, m)

function EnzymeRules.forward(config::FwdConfig, func::Const{_KronLines}, ::Type{RT},
        y::Annotation{<:AbstractVector}, terms::Annotation{<:Base.RefValue{<:Tuple}},
        cs::Annotation, x::Annotation, β::Annotation, ss::Annotation, lines::Annotation,
        m::Annotation) where {RT}
    _entered!(:_batch_kron_lines!)
    W = Val(EnzymeRules.width(config))
    _kron_check_terms(terms, W)
    t, sv, lv, mv = terms.val[], ss.val, lines.val, m.val
    d_ys = _lanes(y, W)
    d_βs = ntuple(w -> _scalar_lane(β, w), W)
    y0 = all(isnothing, d_ys) || all(isnothing, d_βs) ? nothing : copy(y.val)
    _kron!(y.val, t, cs.val, x.val, β.val, sv, lv, mv)
    for w in 1:EnzymeRules.width(config)
        dy = d_ys[w]
        dy === nothing && continue
        dx = _shadow(x, w)
        if dx === nothing
            _kron!(dy, (), (), x.val, β.val, sv, lv, mv)
        else
            _kron!(dy, t, cs.val, dx, β.val, sv, lv, mv)
        end
        dc = _scalar_lane(cs, w)
        dc === nothing || _kron!(dy, t, dc, x.val, true, sv, lv, mv)
        dβ = d_βs[w]
        dβ === nothing || _kron_axpy!(dy, dβ, y0)
    end
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_KronLines},
        ::Type{RT}, y::Annotation{<:AbstractVector},
        terms::Annotation{<:Base.RefValue{<:Tuple}}, cs::Annotation, x::Annotation,
        β::Annotation, ss::Annotation, lines::Annotation, m::Annotation) where {RT}
    _entered!(:_batch_kron_lines!)
    W = Val(EnzymeRules.width(config))
    _kron_check_terms(terms, W)
    t = terms.val[]
    live = !all(isnothing, _lanes(y, W))
    # `overwritten` counts `func` first: `x` is entry 5.
    x_tape = live && cs isa Active && EnzymeRules.overwritten(config)[5] ? copy(x.val) :
             nothing
    y_old = live && β isa Active ? copy(y.val) : nothing
    adj = live && !all(isnothing, _lanes(x, W)) ? _kron_adjoint_terms(t) : nothing
    _kron!(y.val, t, cs.val, x.val, β.val, ss.val, lines.val, m.val)
    return AugmentedReturn(nothing, nothing, (adj, x_tape, y_old))
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_KronLines}, ::Type{RT}, tape,
        y::Annotation{<:AbstractVector}, terms::Annotation{<:Base.RefValue{<:Tuple}},
        cs::Annotation, x::Annotation, β::Annotation, ss::Annotation, lines::Annotation,
        m::Annotation) where {RT}
    V = Val(EnzymeRules.width(config))
    adj, x_tape, y_old = tape
    t, sv, lv, mv = terms.val[], ss.val, lines.val, m.val
    xv = something(x_tape, x.val)
    c̄ = map(c -> zero(c), cs.val)
    β̄ = zero(β.val)
    lanes = ntuple(V) do w
        dy = _shadow(y, w)
        dy === nothing && return (c̄, β̄)
        dx = _shadow(x, w)
        dx === nothing || _kron!(dx, adj, map(conj, cs.val), dy, true, sv, lv, mv)
        dc = cs isa Active ? _kron_coefficient_adjoints(t, dy, xv, sv, lv, mv) : c̄
        db = β isa Active ? _kron_dot(y_old, dy) : β̄
        _kron!(dy, (), (), dy, conj(β.val), sv, lv, mv)
        return (convert(typeof(c̄), dc), db)
    end
    d_cs = cs isa Active ? _active_return(map(first, lanes), V) : nothing
    d_β = β isa Active ? _active_return(map(last, lanes), V) : nothing
    return (nothing, nothing, d_cs, nothing, d_β, nothing, nothing, nothing)
end

# `c̄ₜ = ⟨Kₜ x, ȳ⟩` for each term, through one product of the term alone.
function _kron_coefficient_adjoints(t::Tuple, dy, x, ss, lines, m)
    z = similar(dy)
    return map(t) do term
        _kron!(z, (term,), (one(eltype(z)),), x, false, ss, lines, m)
        _kron_dot(z, dy)
    end
end

# --- Matrix-free: _batch_mf_bands! and the per-unit action replays ------------------ #

const _MFSink = Annotation{<:Base.RefValue{<:Bramble.ActionSink}}
const _MFParts = Annotation{<:Base.RefValue{<:Tuple{Bramble._MFFusedPlan, Any}}}
const _MFForm = Annotation{<:Base.RefValue{<:Tuple}}
const _ActTarget = Annotation{<:Base.RefValue{<:Bramble._ActionTarget}}

# A `CpuPolyester` operator's plan is structure: its grid, row reach and policy, fixed when
# the operator is built, with the form it was built for. Its build walks the form with
# untyped collectors that Enzyme's type analysis cannot bound, so the build is inactive. The
# bands read the form through the hook's own `(Wu, Wv, ast)` argument, whose shadow the
# form's carries, and the plan only through its primal (`_mf_bands_info`).
EnzymeRules.inactive(::typeof(Bramble._mf_plan), ::Bramble.CpuPolyester, args...) = nothing

# What a band walk reads of the plan and the walk: the walk, the plan's geometry and row
# reach, and the banded axis's length. Constant, behind a `Ref` for the thunk.
@inline function _mf_bands_info(parts::Base.RefValue)
    plan, walk = parts[]
    return Ref((walk, plan.geom, plan.omin, plan.omax, last(plan.dims)))
end

# Bands `r` of the fused sweep, walked serially as `_mf_band_task!` walks them.
function _mf_bands_slab!(s::Base.RefValue, form::Base.RefValue, info::Base.RefValue,
        nbands::Int, r)
    Wu, Wv, ast = form[]
    walk, geom, omin, omax, len = info[]
    sk = Bramble._mf_with_geom(s[], geom)
    for b in r
        own = _band_range(1:len, nbands, b)
        Bramble._mf_apply_parts!(
            Bramble._MFPass(Bramble._MF_BAND, own, omin, omax, Bramble._MF_NO_COLLECT),
            sk, Wu, Wv, ast, walk)
    end
    return nothing
end

@noinline function _act_forward_run!(take::T, n::Int, h::H) where {T, H}
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _asm_forward_threw(take(h), n, b)
    end
    return failed
end

@noinline function _act_reverse_run!(take::T, n::Int, h::H) where {T, H}
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _asm_reverse_threw(take(h), n, b)
    end
    return failed
end

# The sink's `x` as the tasks read it: the tape's copy, or `x` itself when a task's zeroed
# copy of its shadow has its type (a view's would not).
@inline _act_input(x, ::Nothing) = x isa Array ? x : copy(x)
@inline _act_input(_, x_tape) = x_tape

# Whether some lane of an action target carries a cotangent for `y`.
function _act_live(target, W::Val)
    pe = _polyester_ext()
    y = first(pe._replay_parts(target.val[]))
    return any(d -> d !== nothing && first(pe._replay_parts(d[])) !== y, _lanes(target, W))
end

# The tape of an action target's call: the parts the slab reads, argument `k` of the hook,
# when later code overwrites them (`overwritten` counts `func` first, so the target is
# entry 2), and `x` whenever a lane is live. `x` sits behind the target's `Ref`, whose
# `overwritten` entry says nothing of `x`'s own buffer, so it is always kept.
function _act_tape(config, target, parts, k::Int = 3)
    ow = EnzymeRules.overwritten(config)
    parts_tape = !(parts isa Const) && ow[k] ? _graph_copy(parts.val, false) : nothing
    live = _act_live(target, Val(EnzymeRules.width(config)))
    x_tape = live ? copy(_polyester_ext()._replay_parts(target.val[])[2]) : nothing
    return (parts_tape, x_tape)
end

# The reverse of an action target's hook: `slab` the serial walk of items `r`, `n` tasks
# over `len` items, `parts` the shadowed parts it reads, `rest` its constant arguments.
# Each task's parts shadow adds into the caller's `Ref` itself, so a float the parts hold
# by value (a scalar in the AST) reaches the caller's shadow too.
function _act_reverse!(config, run!::R, slab::F, n::Int, len::Int, target, parts, tape,
        rest::Tuple) where {R, F}
    pe = _polyester_ext()
    W = EnzymeRules.width(config)
    y, x, cold = pe._replay_parts(target.val[])
    live = [w for w in 1:W
            if (dt = _shadow(target, w)) !== nothing && first(pe._replay_parts(dt[])) !== y]
    isempty(live) && return nothing
    parts_tape, x_tape = tape
    xp = _act_input(x, x_tape)
    p0 = parts.val
    pp = something(parts_tape, p0)
    scratch = Ref(pe._replay_join(similar(y), xp, cold))
    d_parts = _lanes(parts, Val(W))
    cs = map(Const, rest)
    function task(w)
        dy, _, dcold = pe._replay_parts(_shadow(target, w)[])
        dt = Ref(pe._replay_join(dy, zero(xp), _graph_copy(dcold, true)))
        return (Duplicated(scratch, dt),
            Duplicated(pp, _graph_copy(something(d_parts[w], pp), true)), cs...)
    end
    args = [task(w) for w in live, _ in 1:n]
    mode = Enzyme.set_runtime_activity(Enzyme.ReverseSplitNoPrimal,
        EnzymeRules.runtime_activity(config))
    fwd, rev = Enzyme.autodiff_thunk(mode, Const{F}, Const, map(typeof, first(args))...,
        Const{UnitRange{Int}})
    p = _AsmReverse(fwd, rev, slab, args, len, Vector{Any}(nothing, n))
    pe._handed(run!, p, pe._take, n) && _rethrow_first(p.errs)
    for (j, w) in enumerate(live)
        dt = _shadow(target, w)
        dy, dx, dcold = pe._replay_parts(dt[])
        sinks = [args[j, b][1].dval[] for b in 1:n]
        dx === x || _act_add_input!(pe, dx, [pe._replay_parts(s)[2] for s in sinks])
        for s in sinks
            c = pe._replay_parts(s)[3]
            _accumulate!(dcold, c, cold)
            dcold = _add(dcold, c)
        end
        dt[] = pe._replay_join(dy, dx, dcold)
        d_parts[w] === nothing ||
            foreach(b -> _accumulate!(d_parts[w], args[j, b][2].dval, p0), 1:n)
    end
    return nothing
end

# Adds the tasks' `x̄` copies into `dx`: under `@batch` for an array (`_bands_reduce!`), one
# by one otherwise.
function _act_add_input!(pe, dx::Array, copies)
    return _bands_reduce!(pe._handed, pe._take, dx, copies, nothing)
end
function _act_add_input!(pe, dx, copies)
    foreach(c -> dx .+= c, copies)
    return nothing
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_MFBands}, ::Type{RT},
        s::_MFSink, parts::_MFParts, form::_MFForm, nbands::Annotation{Int}) where {RT}
    _entered!(:_batch_mf_bands!)
    n = nbands.val
    _asm_forward!(config, _act_forward_run!, _mf_bands_slab!, n, n, s, form,
        Const(_mf_bands_info(parts.val)), nbands)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_MFBands},
        ::Type{RT}, s::_MFSink, parts::_MFParts, form::_MFForm,
        nbands::Annotation{Int}) where {RT}
    _entered!(:_batch_mf_bands!)
    # The form's parts are entry 4: `func`, the sink and the plan come first.
    tape = _act_tape(config, s, form, 4)
    func.val(s.val, parts.val, form.val, nbands.val)
    return AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_MFBands}, ::Type{RT}, tape,
        s::_MFSink, parts::_MFParts, form::_MFForm, nbands::Annotation{Int}) where {RT}
    n = nbands.val
    _act_reverse!(config, _act_reverse_run!, _mf_bands_slab!, n, n, s, form, tape,
        (_mf_bands_info(parts.val), nbands.val))
    return (nothing, nothing, nothing, nothing)
end

# The per-unit replays with an action target: the slabs of assembly.jl, which call
# `_replay_point!` on whatever target the `Ref` holds.

function EnzymeRules.forward(config::FwdConfig, func::Const{_AsmBandReplay}, ::Type{RT},
        target::_ActTarget, parts::_AsmParts, ax::Annotation, bidx::Annotation,
        nbands::Annotation, rest::Annotation, lin_indices::Annotation,
        row_offset::Annotation, col_offset::Annotation) where {RT}
    _entered!(:_batch_bilinear_band_replay!)
    len = length(bidx.val)
    _asm_forward!(config, _act_forward_run!, _asm_replay_bands!, len, len, target, parts,
        ax, bidx, nbands, rest, lin_indices, row_offset, col_offset)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_AsmBandReplay},
        ::Type{RT}, target::_ActTarget, parts::_AsmParts, ax::Annotation, bidx::Annotation,
        nbands::Annotation, rest::Annotation, lin_indices::Annotation,
        row_offset::Annotation, col_offset::Annotation) where {RT}
    _entered!(:_batch_bilinear_band_replay!)
    tape = _act_tape(config, target, parts)
    func.val(target.val, parts.val, ax.val, bidx.val, nbands.val, rest.val, lin_indices.val,
        row_offset.val, col_offset.val)
    return AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_AsmBandReplay}, ::Type{RT},
        tape, target::_ActTarget, parts::_AsmParts, ax::Annotation, bidx::Annotation,
        nbands::Annotation, rest::Annotation, lin_indices::Annotation,
        row_offset::Annotation, col_offset::Annotation) where {RT}
    len = length(bidx.val)
    _act_reverse!(config, _act_reverse_run!, _asm_replay_bands!, len, len, target, parts,
        tape, (ax.val, bidx.val, nbands.val, rest.val, lin_indices.val, row_offset.val,
            col_offset.val))
    return ntuple(_ -> nothing, Val(9))
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_AsmColourReplay}, ::Type{RT},
        target::_ActTarget, parts::_AsmParts, idxs::Annotation, lin_indices::Annotation,
        row_offset::Annotation, col_offset::Annotation) where {RT}
    _entered!(:_batch_bilinear_colour_replay!)
    _asm_forward!(config, _act_forward_run!, _asm_replay_points!, Threads.nthreads(),
        length(idxs.val), target, parts, idxs, lin_indices, row_offset, col_offset)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_AsmColourReplay},
        ::Type{RT}, target::_ActTarget, parts::_AsmParts, idxs::Annotation,
        lin_indices::Annotation, row_offset::Annotation, col_offset::Annotation) where {RT}
    _entered!(:_batch_bilinear_colour_replay!)
    tape = _act_tape(config, target, parts)
    func.val(target.val, parts.val, idxs.val, lin_indices.val, row_offset.val,
        col_offset.val)
    return AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_AsmColourReplay}, ::Type{RT},
        tape, target::_ActTarget, parts::_AsmParts, idxs::Annotation,
        lin_indices::Annotation, row_offset::Annotation, col_offset::Annotation) where {RT}
    _act_reverse!(config, _act_reverse_run!, _asm_replay_points!, Threads.nthreads(),
        length(idxs.val), target, parts, tape,
        (idxs.val, lin_indices.val, row_offset.val, col_offset.val))
    return ntuple(_ -> nothing, Val(6))
end
