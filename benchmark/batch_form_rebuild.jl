#===========================================================================#
# Arrays out, rebuild in: a prototype for the form-carrying `Polyester.@batch` hooks
# (gpena/Bramble.jl#433, subplan S2.3 of the v3.23.0 plan).
#
#     julia --project=benchmark --threads=4 --startup-file=no benchmark/batch_form_rebuild.jl
#
# `@batch` copies what its loop captures into an argument box. Plain arrays become
# `PtrArray`s and isbits values are copied, and a box holding only those stays on the stack;
# one GC reference puts it on the heap (ext/BramblePolyesterExt.jl, "Allocation bound").
# The form-carrying hooks capture forms, spaces, terms and sinks, so their boxes are on the
# heap. S2.1 removed the box from the Kronecker hook with a split written for its three
# factor types (`_kron_host_raw`/`_kron_host_rebuild`, src/assembly/kronecker.jl). This
# script tries the general version of that split on two form-carrying hooks:
#
#   replay   `_batch_bilinear_band_replay!`, a warm bilinear refill (`assemble!`)
#   mf_band  `_run_bands!` with `_mf_band_task!`, the fused matrix-free product (`mul!`)
#
# ## The split
#
# `_split(x)` walks a value and returns `(skeleton, arrays)`. `arrays` is a flat tuple of
# every `Array` with a native element type the walk reaches. `skeleton` is the value with
# each such array replaced by its slot (`_Slot{k}`). `_rebuild(skeleton, arrays)` is its
# inverse. Run inside each task on the `PtrArray`s `@batch` made of `arrays`, it rebuilds
# the same struct types (`T{P...}` with every array parameter replaced by the `PtrArray`
# type, `_ptype`) around them with `Expr(:new)`. Nothing is written per type. A struct is
# split when it is immutable and each of its fields' new types fits the rebuilt type's
# declared field. Anything else is kept by reference in the skeleton (a mutable struct, a
# `Dict`, a struct with a concretely typed array field, a struct whose type parameter ties a
# kept field to a split one). Those are the split's resistors, and the script lists each
# with its reason (`RESIST` lines). One resistor in a skeleton puts the box back on the heap.
#
# `BilinearForm` resists (its `cache::_AssemblyCache{D, AST}` is mutable and shares the
# `AST` parameter with `ast`), so the matrix-free prototype splits the form's three walked
# parts instead, `(trial_space, test_space, ast)`, and walks them with `_mf_apply_parts!`,
# `_mf_apply!`'s body over the parts. The fused plan's `form::RefValue` is not captured: its
# `dims`, `omin`, `omax` are isbits.
#
# ## The measurement
#
# The prototypes are installed as more specific methods (`_batch_bilinear_band_replay!` on a
# `ReplaySink`, `_mf_run_bands!` on `CpuPolyester`), each of which takes today's route or
# the prototype's from the switch `PROTO`, so both run in one process on the same objects.
# The form is the 2D `innerₕ(u, v) + inner₊(κ ∇ₕ u, ∇ₕ v)` with a grid-function `κ`, on
# the zero-bytes check's non-uniform grid (each point jittered by up to ±0.3h), at 65² and
# 1025².
#
#   bitwise   the prototype's whole result (`assemble!` refill, `mul!`) equals the
#             `CpuSerial` form's bit for bit (`isequal` on the stored values)
#   bytes     `@allocated` of one warm call of the hook alone (every hook call of one
#             refill, or the one band region of one product)
#   time      `BenchmarkTools.@belapsed` of the same hook-level call, today's and the
#             prototype's back to back, alternating which goes first, over `ROUNDS` rounds;
#             the row reports the medians, and each round's ratio has a `ROUND` line
#
# Output, per hook and size (the last four fields are context: today's hook against
# `CpuSerial`, the prototype against today's, the median ratio, the round count):
#
#   REBUILD hook=<replay|mf_band> n=<65|1025> bitwise=<b> bytes_ref=<B> bytes_proto=<B> t_ref_us=<t> t_proto_us=<t> bitwise_ref=<b> proto_eq_ref=<b> ratio=<r> rounds=<k>
#
# The script measures; it judges nothing. Check the machine first
# (`.claude/scripts/check_power_load.sh`) and run it alone. It reaches Bramble internals by
# name, so a refactor in `src/` can break it: it is a prototype, not a maintained benchmark.
#===========================================================================#

using Bramble
using Bramble: CpuPolyester, CpuSerial, change_points!, allocate_system_matrix
using Polyester
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Random
using Printf

Threads.nthreads() == 4 || error("run with --threads=4")
const B = Bramble
const ROUNDS = 7
const SECONDS = 0.5
const SIZES = (65, 1025)

# --- The structural split ------------------------------------------------------------- #

# The `PtrArray` type `@batch` makes of an array of type `A` (through the same
# `object_and_preserve` it calls), so a rebuilt type names exactly what each task receives.
_ptrtype(::Type{A}) where {A <: Array} = typeof(first(Polyester.object_and_preserve(
    A(undef, ntuple(_ -> 0, ndims(A))))))

# An array `@batch` turns into a `PtrArray`.
_isleaf(T) = T <: Array && isconcretetype(T) && isbitstype(eltype(T))

# The rebuilt type of a splittable struct type `T`, or `nothing` if `T` is kept whole.
function _rebuilt(T)
    (T isa DataType && isconcretetype(T) && isstructtype(T) && !(T <: Tuple)) || return nothing
    (ismutabletype(T) || isbitstype(T) || fieldcount(T) == 0) && return nothing
    old = fieldtypes(T)
    new = map(_ptype, old)
    new == old && return nothing
    P = map(p -> p isa Type ? _ptype(p) : p, Tuple(T.parameters))
    T′ = try
        T.name.wrapper{P...}
    catch
        return nothing
    end
    all(i -> new[i] <: fieldtype(T′, i), eachindex(new)) || return nothing
    return T′
end

# What a value of type `T` becomes after the round trip: a `PtrArray`, a rebuilt struct or
# tuple, or `T` itself (isbits, or kept by reference).
function _ptype(T)
    _isleaf(T) && return _ptrtype(T)
    if T isa DataType && T <: Tuple && isconcretetype(T)
        return Tuple{map(_ptype, fieldtypes(T))...}
    end
    T′ = _rebuilt(T)
    return T′ === nothing ? T : T′
end

_splits(T) = (T isa DataType && T <: Tuple && isconcretetype(T)) || _rebuilt(T) !== nothing

struct _Slot{K} end
struct _Node{T, F <: Tuple}
    fields::F
end
_Node{T}(fields::F) where {T, F} = _Node{T, F}(fields)

# Each of `_split` and `_rebuild` is one generated function whose body is the whole walk,
# written out from the type: a generated function recursing through itself per node hit
# inference's recursion limit, left the inner calls dynamic and allocated (6384 B per
# replay call at 65²). The leaves are numbered in the same depth-first field order by both
# walks below, so slot `k` is the `k`-th array.
function _leaves!(acc, T, ex)
    if _isleaf(T)
        push!(acc, ex)
    elseif _splits(T)
        foreach(i -> _leaves!(acc, fieldtype(T, i), :(getfield($ex, $i))), 1:fieldcount(T))
    end
    return acc
end

function _skeleton_expr(T, ex, k::Base.RefValue{Int})
    _isleaf(T) && return :(_Slot{$(k[] += 1)}())
    _splits(T) || return ex
    fs = [_skeleton_expr(fieldtype(T, i), :(getfield($ex, $i)), k) for i in 1:fieldcount(T)]
    return :(_Node{$T}(($(fs...),)))
end

function _rebuild_expr(S, ex)
    S <: _Slot && return :(arrays[$(S.parameters[1])])
    S <: _Node || return ex
    T, F = S.parameters
    fs = [_rebuild_expr(fieldtype(F, i), :(getfield($ex.fields, $i))) for i in 1:fieldcount(F)]
    return T <: Tuple ? :(($(fs...),)) : Expr(:new, _rebuilt(T), fs...)
end

_inline(ex) = Expr(:block, Expr(:meta, :inline), ex)

@generated function _split(x::T) where {T}
    return _inline(:(($(_skeleton_expr(T, :x, Ref(0))), ($(_leaves!(Any[], T, :x)...),))))
end

@generated _rebuild(skeleton, arrays) = _inline(_rebuild_expr(skeleton, :skeleton))

# Each kept struct type the split meets in `x`, with why it is kept. A field of a kept type
# is reported only if the kept type is not itself mutable (a mutable struct is one resistor).
function _resistors!(out, x, path)
    T = typeof(x)
    (_isleaf(T) || isbitstype(T)) && return out
    if _splits(T)
        for i in 1:fieldcount(T)
            _resistors!(out, getfield(x, i), string(path, ".", fieldname(T, i)))
        end
        return out
    end
    _nleaves_any(x) == 0 && return out
    name = string(nameof(T))
    why = if x isa AbstractDict
        "Dict (Memory keys/vals, Symbol keys)"
    elseif ismutabletype(T)
        "mutable struct"
    else
        bad = [string(fieldname(T, i)) for i in 1:fieldcount(T)
               if fieldtype(T, i) <: Array]
        isempty(bad) ? "type parameter ties a kept field to a split one" :
        "field(s) declared as Array: " * join(bad, ", ")
    end
    haskey(out, name) || (out[name] = (why, path))
    return out
end
_resistors(x, path) = _resistors!(Dict{String, Tuple{String, String}}(), x, path)

# Whether a kept value holds an array anywhere (only those matter to the box).
_nleaves_any(x) = _nleaves_any(x, IdDict{Any, Nothing}())
function _nleaves_any(x, seen)
    T = typeof(x)
    (x isa Array || x isa BitArray) && return 1
    (isbitstype(T) || x isa Symbol || x isa Module || x isa Type) && return 0
    ismutabletype(T) && (haskey(seen, x) && return 0; seen[x] = nothing)
    x isa AbstractDict && return sum(v -> _nleaves_any(v, seen), values(x); init = 0)
    isstructtype(T) || return 0
    return sum(i -> isdefined(x, i) ? _nleaves_any(getfield(x, i), seen) : 0,
        1:fieldcount(T); init = 0)
end

# --- The prototype hooks --------------------------------------------------------------- #

const PROTO = Ref(false)
const CAPTURE = Ref(false)
const REPLAY_CALLS = Any[]
const MF_CALLS = Any[]

const _EXT_REPLAY_SIG = Tuple{Union{B._ReplayTarget, B._ActionTarget}, ntuple(_ -> Any, 10)...}

function proto_band_replay!(target, sp, term, ax, bidx, nbands, rest, lin_indices,
        mesh_markers, row_offset, col_offset)
    skel, arrays = _split((target, sp, term, mesh_markers))
    @batch for b in bidx
        t, s, tm, mm = _rebuild(skel, arrays)
        for I in CartesianIndices((rest..., B._band_range(ax, nbands, b)))
            B._replay_point!(t, tm, s, I, lin_indices, mm, row_offset, col_offset)
        end
    end
    return nothing
end

@eval Bramble function _batch_bilinear_band_replay!(
        target::ReplaySink, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers,
        row_offset, col_offset)
    args = (target, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers,
        row_offset, col_offset)
    Main.CAPTURE[] && push!(Main.REPLAY_CALLS, args)
    Main.PROTO[] && return Main.proto_band_replay!(args...)
    return invoke(_batch_bilinear_band_replay!, Main._EXT_REPLAY_SIG, args...)
end

# `_mf_apply!`'s body over the form's walked parts: `BilinearForm` itself does not split.
function _mf_apply_parts!(policy, s, Wu, Wv, ast)
    if B._is_block_pair(Wu, Wv)
        B._mf_blocks!(policy, s, ast, B.leaf_spaces_offsets(Wu), B.leaf_spaces_offsets(Wv))
        return nothing
    end
    bound = B._bind_interp_spaces(ast, Wu, Wv)
    B._check_block_meshes(bound, Wu, Wv)
    sp = B.host_weights(B._walked_leaf(bound, Wu, Wv))
    B._mf_summands!(policy, s, bound, sp)
    return nothing
end

function proto_mf_bands!(s, plan::B._MFFusedPlan{D}, nbands::Int) where {D}
    a = plan.form[]
    skel, arrays = _split((s, a.trial_space, a.test_space, a.ast))
    len, omin, omax = plan.dims[D], plan.omin, plan.omax
    @batch for b in 1:nbands
        s′, Wu, Wv, ast = _rebuild(skel, arrays)
        own = B._band_range(1:len, nbands, b)
        pass = B._MFPass(B._MF_BAND, own, omin, omax, B._MF_NO_COLLECT)
        _mf_apply_parts!(pass, s′, Wu, Wv, ast)
    end
    return nothing
end

ref_mf_bands!(s, plan, nbands) = B._run_bands!(
    CpuPolyester(), B._mf_band_task!, s.y, s, nothing, plan, nbands)

@eval Bramble function _mf_run_bands!(::CpuPolyester, s, a, plan, nbands::Int)
    Main.CAPTURE[] && push!(Main.MF_CALLS, (s, plan, nbands))
    Main.PROTO[] && return Main.proto_mf_bands!(s, plan, nbands)
    return Main.ref_mf_bands!(s, plan, nbands)
end

# Hook-level calls: every captured replay call of one refill, or the one band region.
run_replay_ref(calls) = (foreach(c -> invoke(B._batch_bilinear_band_replay!,
    _EXT_REPLAY_SIG, c...), calls); nothing)
run_replay_proto(calls) = (foreach(c -> proto_band_replay!(c...), calls); nothing)
run_mf_ref(c) = ref_mf_bands!(c...)
run_mf_proto(c) = proto_mf_bands!(c...)

# --- Grids and forms ----------------------------------------------------------------- #

function jitter(n, policy; seed = n)
    rng = Xoshiro(seed)
    X = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ω = mesh(domain(X, :dir => boundary_symbols(X)), (n, n), (true, true);
        backend = backend(policy = policy))
    h = 1 / (n - 1)
    function pts()
        x = collect(range(0.0, 1.0; length = n)) .+ 0.3h .* (2 .* rand(rng, n) .- 1)
        x[1], x[end] = 0.0, 1.0
        return sort!(x)
    end
    change_points!(Ω, (pts(), pts()))
    return Ω
end

function poisson_kappa(n, policy)
    W = gridspace(jitter(n, policy))
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    return form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v)))
end

same_matrix(A, Bm) = A.colptr == Bm.colptr && A.rowval == Bm.rowval &&
                     isequal(A.nzval, Bm.nzval)

@noinline function measure_bytes(f::F, arg) where {F}
    f(arg)
    f(arg)
    return @allocated f(arg)
end

# `ROUNDS` rounds of today's call and the prototype's, back to back, the order alternating.
function time_rounds(hook, n, fref::F, fproto::G, arg) where {F, G}
    tr, tp = Float64[], Float64[]
    for r in 1:ROUNDS
        if isodd(r)
            a = @belapsed $fref($arg) seconds = SECONDS
            b = @belapsed $fproto($arg) seconds = SECONDS
        else
            b = @belapsed $fproto($arg) seconds = SECONDS
            a = @belapsed $fref($arg) seconds = SECONDS
        end
        push!(tr, 1e6a)
        push!(tp, 1e6b)
        @printf("ROUND hook=%s n=%d r=%d t_ref_us=%.2f t_proto_us=%.2f ratio=%.4f\n",
            hook, n, r, 1e6a, 1e6b, b / a)
    end
    med(v) = (s = sort(v); s[(length(s) + 1) ÷ 2])
    return med(tr), med(tp), med(tp ./ tr)
end

function print_resistors(hook, x)
    for (name, (why, path)) in sort!(collect(_resistors(x, "args")); by = first)
        println("RESIST hook=$hook type=$name at=$path reason=\"$why\"")
    end
end

function row(hook, n, bitwise, bitwise_ref, eq_ref, bref, bproto, tr, tp, ratio)
    @printf("REBUILD hook=%s n=%d bitwise=%s bytes_ref=%d bytes_proto=%d t_ref_us=%.2f t_proto_us=%.2f bitwise_ref=%s proto_eq_ref=%s ratio=%.4f rounds=%d\n",
        hook, n, bitwise, bref, bproto, tr, tp, bitwise_ref, eq_ref, ratio, ROUNDS)
end

# The hook calls one `f(args...)` makes, as a tuple. Behind `invokelatest`: called inline,
# the compiler returned an empty tuple although the vector held the calls.
@noinline function captured(store, f, args...)
    empty!(store)
    CAPTURE[] = true
    f(args...)
    CAPTURE[] = false
    return Tuple(store)
end

function replay_rows(n)
    a, aₛ = poisson_kappa(n, CpuPolyester()), poisson_kappa(n, CpuSerial())
    A, Aₛ = allocate_system_matrix(a), allocate_system_matrix(aₛ)
    assemble!(Aₛ, aₛ)
    assemble!(Aₛ, aₛ)
    PROTO[] = false
    assemble!(A, a)                     # records
    assemble!(A, a)                     # replays
    bitwise_ref = same_matrix(A, Aₛ)
    Aref = copy(A)
    fill!(nonzeros(A), NaN)
    PROTO[] = true
    assemble!(A, a)
    PROTO[] = false
    bitwise = same_matrix(A, Aₛ)
    eq_ref = same_matrix(A, Aref)
    calls = Base.invokelatest(captured, REPLAY_CALLS, assemble!, A, a)
    isempty(calls) && error("no band replay reached at n = $n")
    n == first(SIZES) && print_resistors("replay", first(calls))
    bref = measure_bytes(run_replay_ref, calls)
    bproto = measure_bytes(run_replay_proto, calls)
    tr, tp, ratio = time_rounds("replay", n, run_replay_ref, run_replay_proto, calls)
    row("replay", n, bitwise, bitwise_ref, eq_ref, bref, bproto, tr, tp, ratio)
    return nothing
end

function mf_rows(n)
    a, aₛ = poisson_kappa(n, CpuPolyester()), poisson_kappa(n, CpuSerial())
    op, opₛ = matrix_free_operator(a), matrix_free_operator(aₛ)
    x = randn(Xoshiro(7), size(op, 2))
    yₛ, yref, y = similar(x), similar(x), similar(x)
    mul!(yₛ, opₛ, x)
    PROTO[] = false
    mul!(yref, op, x)
    mul!(yref, op, x)
    PROTO[] = true
    mul!(y, op, x)
    mul!(y, op, x)
    PROTO[] = false
    bitwise, bitwise_ref, eq_ref = isequal(y, yₛ), isequal(yref, yₛ), isequal(y, yref)
    mfc = Base.invokelatest(captured, MF_CALLS, mul!, y, op, x)
    length(mfc) == 1 || error("expected one band region at n = $n, got $(length(mfc))")
    call = only(mfc)
    f = call[2].form[]
    n == first(SIZES) && print_resistors("mf_band",
        (call[1], f, (f.trial_space, f.test_space, f.ast)))
    bref = measure_bytes(run_mf_ref, call)
    bproto = measure_bytes(run_mf_proto, call)
    tr, tp, ratio = time_rounds("mf_band", n, run_mf_ref, run_mf_proto, call)
    row("mf_band", n, bitwise, bitwise_ref, eq_ref, bref, bproto, tr, tp, ratio)
    return nothing
end

for n in SIZES
    replay_rows(n)
    mf_rows(n)
end
