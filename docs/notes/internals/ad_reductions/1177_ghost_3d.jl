# Reproducer for gpena/Bramble.jl#1177: Enzyme reverse of the ghost `Δₕ` assembly on a 3D
# mesh aborts the process in LLVM verification ("Instruction does not dominate all uses!").
# One step, one cell and one mode per process: an abort ends the process.
#
# Run from the repository root, with one thread, at Julia's default optimisation level (the
# level the abort was found at):
#
#     julia --startup-file=no --project=. --threads=1 \
#         docs/notes/internals/ad_reductions/1177_ghost_3d.jl <step> <cell> <rev|fwd>
#
# Cells, both under `CpuSerial` on the non-uniform meshes of the test harness (seed 11).
#   3d   the issue's cell, a (4, 4, 3) mesh with ghosts `(:xmin, :zmax)`, a Neumann side, a
#        Robin side and a Dirichlet side, as the v4.2.4 review of #1152 ran it.
#   2d   the control, the `ghost_assemble` cell of 1152_ghost_matrix.jl, a (7, 6) mesh with
#        ghosts `(:xmin, :xmax, :ymax)`, which differentiates since 9faf6345.
#
# The form is `p[1] * innerₕ(-Δₕ(U), V) + p[2]^2 * innerₕ(U, V)`, built from the parameter
# inside the differentiated function so its scalars are active. Every step below `cell`
# works on the padded form of the ghost form (`GhostBilinearForm.padded`), on the padded
# space `P`, with the walk `W` (`_WalkRows`, cut: `W.whole` is false on both cells) and a
# copy of the padded matrix's pattern, all built outside at p0. They hold integers only.
#
# Steps, from the whole cell down to one call.
#   cell            the issue's loss, `form` on the constrained space, `assemble(a, l)`,
#                   `sum(abs2, nonzeros(A)) + 1e4 * p[3] * sum(F)`.
#   record          `_assemble_bilinear_core_cached!` on a fresh cache, as
#                   `_assemble_padded!` runs it on a first fill (the pattern pass, then the
#                   replay).
#   pattern         the pattern pass without the replay, `_form_coordinates` (two
#                   `_coord_walk!` passes per term), `_coordinates_to_positions!`,
#                   `_segments_from_positions`.
#   coord           `_form_coordinates` alone.
#   coord_walk      `_coord_walk!` alone, once per summand, on the cut space
#                   (`_RowRestricted`), the method with the `do piece` closures.
#   replay          `_replay_bilinear_core!` with segments recorded outside.
#   replay_segment  `_replay_segment!` alone, once per summand, with its own segment.
#   control_nest    no Bramble, the loop nest of `_each_piece` around a weighted sum, with
#                   the tail of the region in two loops (as in 3D) and the piece's length
#                   in the inner loop's trip count. It differentiates (see the note).
#   sizes           no Enzyme call, the byte layout of the closures and of the walk space.
#   control_slabs   no Bramble, that nest inside a loop over the interior box and the
#                   boundary slabs, as `_visit_stencil!` runs it, each point adding to a
#                   vector. It differentiates too (see the note).
# The pattern steps produce integers; their loss is `sum(abs2, p) * <an integer sum>`, so
# the reference is not zero and Enzyme still differentiates through the walk.
#
# Set `BRAMBLE_PROTO=flat` to replace `_each_piece` and `_visit_pieces!` (src/assembly/
# bilinear_traversal.jl) by copies whose loop over the region's tail is one loop over
# its linear index, not a `CartesianIndices` loop of `D - 1` nested loops. This is the
# prototype of the fix; it changes no file. `BRAMBLE_PROTO=flat_replay` replaces
# `_visit_pieces!` alone, the replay's loop, and leaves the pattern pass's.
#
# The reference is ForwardDiff on the same loss, through DifferentiationInterface. A step is
# `ok` when the relative error (largest absolute difference over largest absolute
# reference) is at most 1e-8, `fails` when Enzyme throws, `wrong` past that bound. An abort
# prints no RESULT line: the verifier writes the bad function's IR to stderr and the
# process ends on signal 6.
#
# Enzyme and ForwardDiff are weak dependencies of the root project: they load from a
# temporary environment, as 1152_ghost_matrix.jl does.
#
# Output: `step <name> t=<seconds>` lines, then
# `RESULT <step> <cell> <mode> <status> <relerr> <seconds> <detail>`. Exit 0 for `ok`.
println("pid ", getpid())
flush(stdout)
_e = mktempdir()
write(joinpath(_e, "Project.toml"),
    "[deps]\n" *
    "DifferentiationInterface = \"a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63\"\n" *
    "Enzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n" *
    "ForwardDiff = \"f6369f11-7733-5829-9624-2563aa707210\"\n")
push!(LOAD_PATH, _e)
using Bramble, DifferentiationInterface, Enzyme, ForwardDiff, SparseArrays
using Logging
const B = Bramble

# Enzyme warns once per stored value it cannot find a use for, thousands of lines of IR.
global_logger(NullLogger())

const T0 = time()
step(name) = (println("step ", name, " t=", round(time() - T0; digits = 1)); flush(stdout))

length(ARGS) == 3 || error("usage: 1177_ghost_3d.jl <step> <3d|2d> <rev|fwd>")
const STEP, CELL, MODE = ARGS
CELL in ("3d", "2d") || error("cell must be 3d or 2d, not ", CELL)
MODE in ("rev", "fwd") || error("mode must be rev or fwd, not ", MODE)
const PROTO = get(ENV, "BRAMBLE_PROTO", "")

PROTO in ("", "flat", "flat_replay") ||
    error("BRAMBLE_PROTO must be flat, flat_replay or unset, not ", PROTO)
# `_each_piece` (the pattern pass's `do piece` closures, `_each_walked`) and
# `_visit_pieces!` (the replay) with the tail of the region walked by one loop.
if PROTO == "flat"
    @eval B @inline function _each_piece(
            f::F, region::CartesianIndices, g::_GridLines) where {F}
        r1 = first(region.indices)
        tail = CartesianIndices(Base.tail(region.indices))
        for k in 1:length(tail)
            J = @inbounds tail[k]
            l = @inbounds g.lines[J]
            for s in @inbounds(g.ptr[l]:(g.ptr[l + 1] - 1))
                r = max(first(r1), @inbounds g.lo[s]):min(last(r1), @inbounds g.hi[s])
                isempty(r) || f(CartesianIndices((r, map(j -> j:j, Tuple(J))...)))
            end
        end
        return nothing
    end
end
if PROTO in ("flat", "flat_replay")
    @eval B @inline function _visit_pieces!(visit::V, sink, term, wsp, mesh_markers,
            lin_indices, region::CartesianIndices, g::_GridLines, row_offset::Int,
            col_offset::Int) where {V}
        r1 = first(region.indices)
        tail = CartesianIndices(Base.tail(region.indices))
        for k in 1:length(tail)
            J = @inbounds tail[k]
            l = @inbounds g.lines[J]
            for s in @inbounds(g.ptr[l]:(g.ptr[l + 1] - 1))
                r = max(first(r1), @inbounds g.lo[s]):min(last(r1), @inbounds g.hi[s])
                isempty(r) || visit(sink, term, wsp, mesh_markers, lin_indices,
                    CartesianIndices((r, map(j -> j:j, Tuple(J))...)), row_offset,
                    col_offset)
            end
        end
        return nothing
    end
end

# The test harness's `ghost_mesh(policy, ns)`, copied from test/TestUtils.jl.
function ghost_mesh(pol, ns::NTuple{D, Int}) where {D}
    B._seed_mesh1d_rng!(11)
    be = backend(vector_type = Vector{Float64}, matrix_type = SparseMatrixCSC{Float64, Int},
        policy = pol)
    Ω = mesh(domain(foldl(×, ntuple(d -> interval(0.0, Float64(d)), D))), ns,
        ntuple(_ -> false, D); backend = be)
    B._unseed_mesh1d_rng!()
    return Ω
end

const C = if CELL == "3d"
    constrained(gridspace(ghost_mesh(B.CpuSerial(), (4, 4, 3)); ghosts = (:xmin, :zmax)),
        u -> (:xmin => ∂ₙ(u) => 1.0, :zmax => 2.0 * u + ∂ₙ(u) => 1.0, :ymin => u => 0.0))
else
    constrained(
        gridspace(ghost_mesh(B.CpuSerial(), (7, 6)); ghosts = (:xmin, :xmax, :ymax)),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => 2.0 * u + ∂ₙ(u) => 1.0,
            :ymin => u => 0.0, :ymax => ∂ₙ(u) => 1.0))
end
const DIM = CELL == "3d" ? 3 : 2
const L = form(C, v -> innerₕ(x -> 1.0 + x[1], v))
bilinear(p) = (U, V) -> p[1] * innerₕ(-Δₕ(U), V) + p[2]^2 * innerₕ(U, V)

const P0 = [0.5 + 0.3 * sin(k) for k in 1:3]

# The padded pieces, at p0: integers and the pattern only.
const GA = form(C, C, bilinear(P0))
const P = GA.padded.trial_space
const W = GA.walk
const A0 = copy(GA.A_pad)
const RSP = B._restrict_walk(B.host_weights(P), W)
GA.padded.trial_space === GA.padded.test_space || error("the padded form has two spaces")
W.whole && error("the walk is whole: no cut walk to reduce")
const SEGS = let a = form(P, P, bilinear(P0)), pass = B._form_coordinates(P, P, a.ast, W)
    typeof(a.ast) == typeof(GA.padded.ast) || error("the padded form's ast differs")
    all(==(-1), pass.halves) || error("a transposed pair: one segment per summand fails")
    A = copy(A0)
    B._coordinates_to_positions!(A, pass, a.ast)
    B._segments_from_positions(Val(DIM), pass)
end

padded_form(p) = form(P, P, bilinear(p))
summands(a) = B._summands(B._bind_interp_spaces(a.ast, P, P))
# The pattern of `A0`, its values zeros of the element type of `p` (ForwardDiff's duals).
zeroed_pattern(p) = SparseMatrixCSC(size(A0)..., copy(SparseArrays.getcolptr(A0)),
    copy(rowvals(A0)), zeros(eltype(p), nnz(A0)))
matrix_loss(A, p) = sum(abs2, nonzeros(A)) + 1e6 * p[3] * sum(nonzeros(A))

function loss_cell(p)
    a = form(C, C, bilinear(p))
    A, F = assemble(a, L)
    return sum(abs2, nonzeros(A)) + 1e4 * p[3] * sum(F)
end
function loss_record(p)
    a = padded_form(p)
    A = zeroed_pattern(p)
    B._assemble_bilinear_core_cached!(B._SerialReplay(), A, P, P, a.ast, a.cache, true, W)
    return matrix_loss(A, p)
end
function loss_pattern(p)
    a = padded_form(p)
    pass = B._form_coordinates(P, P, a.ast, W)
    A = copy(A0)
    B._coordinates_to_positions!(A, pass, a.ast)
    segs = B._segments_from_positions(Val(DIM), pass)
    n = 0
    for s in segs
        n += sum(s.positions)
    end
    return sum(abs2, p) * n
end
function loss_coord(p)
    a = padded_form(p)
    pass = B._form_coordinates(P, P, a.ast, W)
    return sum(abs2, p) * (sum(pass.I) + sum(pass.J))
end
function loss_coord_walk(p)
    a = padded_form(p)
    n = 0
    for t in summands(a)
        ptr = zeros(Int, length(B.indices(B.mesh(P))) + 1)
        sink = B._CoordSink(ptr, Int[], Int[], 0, 0, 0, 0, -1, false, 0)
        B._coord_walk!(sink, t, RSP, 0, 0)
        n += sink.n + sum(ptr)
    end
    return sum(abs2, p) * n
end
function loss_replay(p)
    a = padded_form(p)
    A = zeroed_pattern(p)
    B._replay_bilinear_core!(B._SerialReplay(), A, P, P, a.ast, SEGS, true, W)
    return matrix_loss(A, p)
end
function loss_replay_segment(p)
    a = padded_form(p)
    A = zeroed_pattern(p)
    k = 0
    for t in summands(a)
        k += 1
        B._replay_segment!(A, t, RSP, 0, 0, SEGS[k], true)
    end
    return matrix_loss(A, p)
end

# No Bramble. The loop nest of `_each_piece` on the walk's lines, as plain vectors: for each
# line `J` of the tail (two loops on a 3D tail), each stored range `s` of that line, the
# piece `max(first(r1), lo[s]):min(last(r1), hi[s])` and an inner loop over its points.
const NEST = let g = RSP.walk.points
    (; lo = copy(g.lo), hi = copy(g.hi), ptr = copy(g.ptr), lines = collect(g.lines),
        n = Tuple(size(B.indices(B.mesh(P)))))
end
function loss_control_nest(p)
    lo, hi, ptr, lines, n = NEST.lo, NEST.hi, NEST.ptr, NEST.lines, NEST.n
    acc = 0.0
    for J in CartesianIndices(Base.tail(map(Base.OneTo, n)))
        l = @inbounds lines[J]
        for s in @inbounds(ptr[l]:(ptr[l + 1] - 1))
            r = max(1, @inbounds lo[s]):min(n[1], @inbounds hi[s])
            for i in r
                acc += (p[1] * i + p[2]^2 * sum(Tuple(J))) * (1 + p[3] * length(r))
            end
        end
    end
    return acc^2
end

# No Bramble. The same nest inside the loop over the boundary slabs, as `_visit_stencil!`
# runs it: a tuple of `CartesianIndices` whose ranges change from slab to slab, so the trip
# count of the tail's inner loop is a value of the slab loop. Each point adds to a vector.
const SLABS = let n = NEST.n, full = map(Base.OneTo, n), D = length(n)
    lo = ntuple(d -> CartesianIndices(ntuple(e -> e < d ? (2:(n[e] - 1)) :
        e == d ? (1:1) : (1:n[e]), D)), D)
    hi = ntuple(d -> CartesianIndices(ntuple(e -> e < d ? (2:(n[e] - 1)) :
        e == d ? (n[d]:n[d]) : (1:n[e]), D)), D)
    (CartesianIndices(map(m -> 2:(m - 1), n)), lo..., hi...)
end
@inline function _control_visit!(A, piece, lin, p)
    for I in piece
        @inbounds A[lin[I]] += p[1] * I[1] + p[2]^2 * I[2] * (1 + p[3] * last(Tuple(I)))
    end
    return nothing
end
function loss_control_slabs(p)
    lo, hi, ptr, lines, n = NEST.lo, NEST.hi, NEST.ptr, NEST.lines, NEST.n
    lin = LinearIndices(map(Base.OneTo, n))
    A = zeros(eltype(p), prod(n))
    for slab in SLABS
        r1 = first(slab.indices)
        for J in CartesianIndices(Base.tail(slab.indices))
            l = @inbounds lines[J]
            for s in @inbounds(ptr[l]:(ptr[l + 1] - 1))
                r = max(first(r1), @inbounds lo[s]):min(last(r1), @inbounds hi[s])
                isempty(r) ||
                    _control_visit!(A, CartesianIndices((r, map(j -> j:j, Tuple(J))...)),
                        lin, p)
            end
        end
    end
    return sum(abs2, A)
end

const LOSSES = Dict("cell" => loss_cell, "record" => loss_record, "pattern" => loss_pattern,
    "coord" => loss_coord, "coord_walk" => loss_coord_walk, "replay" => loss_replay,
    "replay_segment" => loss_replay_segment, "control_nest" => loss_control_nest,
    "control_slabs" => loss_control_slabs)

# Step `sizes`, no Enzyme call: the layout of what the walks capture or pass, per summand.
# The `do piece` closures of `_coord_walk!` (bilinear_pattern.jl) hold the fields of its
# closure types, in their order; `wsp` is the walk space `_visit_stencil!` passes on.
function layout(name, nt)
    T = typeof(nt)
    println("  ", name, ": ", sizeof(T), " bytes; ", join(
        ["$(fieldname(T, i)) at $(fieldoffset(T, i)) ($(sizeof(fieldtype(T, i))))"
         for i in 1:fieldcount(T)], ", "))
end
if STEP == "sizes"
    a = padded_form(P0)
    for t in summands(a)
        sp = RSP.space
        bound, mesh_markers = B._may_restrict(typeof(t)) ? B._bind_walk(t, B.mesh(sp)) :
                              (t, nothing)
        inner = B._has_marks(typeof(t)) ? B._interior_term(bound) : bound
        lin_indices = LinearIndices(B.indices(B.mesh(sp)))
        sink = B._CoordSink(Int[], Int[], Int[], 0, 0, 0, 0, -1, false, 0)
        wsp = B._walk_space(sp, typeof(t))
        println("summand ", nameof(typeof(t)), ", cell ", CELL)
        layout("_coord_walk! interior closure",
            (; sink, row_offset = 0, col_offset = 0, inner, lin_indices, mesh_markers, sp))
        layout("_coord_walk! slab closure",
            (; sink, row_offset = 0, col_offset = 0, lin_indices, mesh_markers, bound, sp))
        println("  walk space wsp: ", sizeof(wsp), " bytes")
    end
    exit(0)
end
haskey(LOSSES, STEP) ||
    error("step must be one of ", join(sort!(collect(keys(LOSSES))), ", "))
const loss = LOSSES[STEP]

const AD = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(MODE == "rev" ? Enzyme.Reverse : Enzyme.Forward),
    function_annotation = Enzyme.Const)

println("step ", STEP, " cell ", CELL, " ", MODE, " proto ",
    isempty(PROTO) ? "none" : PROTO,
    ", Enzyme ", pkgversion(Enzyme), ", Julia ", VERSION, ", threads ", Threads.nthreads(),
    ", optimize ", Base.JLOptions().opt_level, ", check-bounds ",
    Base.JLOptions().check_bounds, " (0 default, 1 yes, 2 no)")
flush(stdout)

ref = DifferentiationInterface.gradient(loss, AutoForwardDiff(), P0)
step("reference")
if minimum(abs, ref) <= 1e-6 * maximum(abs, ref)
    println("RESULT ", STEP, " ", CELL, " ", MODE, " weakref NaN ",
        round(time() - T0; digits = 1), " reference ", ref)
    exit(1)
end

# Top-level `try` blocks do not hand their assignments back, so the call runs in a function.
function measure(ref, t_ref)
    status, relerr, detail, g = "ok", NaN, "", Float64[]
    step("enzyme start")
    try
        g = DifferentiationInterface.gradient(loss, AD, P0)
        relerr = maximum(abs, g .- ref) / maximum(abs, ref)
        relerr <= 1e-8 || (status = "wrong")
        step("enzyme done")
    catch err
        err isa InterruptException && rethrow()
        status = "fails"
        msg = sprint(showerror, err)
        detail = string(nameof(typeof(err)), ": ", first(replace(msg, '\n' => ' '), 400))
        println(first(msg, 1500))
        step("enzyme failed")
    end
    return (; status, relerr, detail, g, secs = time() - T0 - t_ref)
end

r = measure(ref, time() - T0)
println("RESULT ", STEP, " ", CELL, " ", MODE, " ", r.status, " ",
    isnan(r.relerr) ? "NaN" : string(r.relerr), " ", round(r.secs; digits = 1), " ", r.detail)
println("enzyme gradient ", isempty(r.g) ? "none" : string(r.g), " reference ", ref)
flush(stdout)
exit(r.status == "ok" ? 0 : 1)
