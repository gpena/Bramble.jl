module UtilsBatchSplitTests

# `_batch_split`/`_batch_rebuild` (src/utils/batch_split.jl, gpena/Bramble.jl#437 item 4): a
# walk argument splits into an isbits skeleton and a flat tuple of plain arrays, and the
# rebuild evaluates bit for bit like the original. Every form is on a jittered non-uniform
# 2D mesh, each unit split as the walk sees it: `(sp, bound, markers)` after `_bind_walk`.
using Test
using Bramble
using Bramble: restrict_to, πₕ, D₋ₓ, change_points!, indices, local_stencil,
               leaf_spaces_offsets, each_routed_leaf, Mesh1DState, MeshnDState,
               _batch_split, _batch_rebuild, _bind_walk, _foreach_unit, _summands,
               _sweep_point!
using SparseArrays
using Random
using ..TestUtils: alloc_test

function jittered(n; seed)
    rng = Xoshiro(seed)
    X = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(X, :dir => boundary_symbols(X), :bottom => :bottom), (n, n),
        (true, true))
    h = 1 / (n - 1)
    function pts()
        x = collect(range(0.0, 1.0; length = n)) .+ 0.3h .* (2 .* rand(rng, n) .- 1)
        x[1], x[end] = 0.0, 1.0
        return sort!(x)
    end
    change_points!(Ωₕ, (pts(), pts()))
    return Ωₕ
end

plain(a) = a isa Array && isbitstype(eltype(a))

# The bilinear units of `a`, each as the walk argument `(sp, bound, markers)` with offsets.
function bilinear_units(a)
    units = Any[]
    _foreach_unit(a.trial_space, a.test_space, a.ast) do t, sp, ro, co, _...
        bound, mm = _bind_walk(t, sp)
        return push!(units, ((sp, bound, mm), ro, co))
    end
    return units
end

# The linear units of `l`: the whole form on a scalar space, each summand's leaves otherwise.
function linear_units(l)
    sp, ast = l.test_space, l.ast
    sp isa Bramble.ScalarGridSpace && return Any[((sp, _bind_walk(ast, sp)...), 0)]
    units = Any[]
    for t in _summands(ast)
        each_routed_leaf(t, leaf_spaces_offsets(sp)) do s, o
            return push!(units, ((s, _bind_walk(t, s)...), o))
        end
    end
    return units
end

# The walk over one unit, as `_sweep_bilinear_serial!` and `_scatter_term!` do below
# `_bind_walk`.
function sweep!(A::AbstractMatrix, (sp, bound, mm), ro, co)
    lin = LinearIndices(indices(mesh(sp)))
    for I in indices(mesh(sp))
        _sweep_point!(A, bound, sp, I, lin, mm, ro, co, true)
    end
    return A
end
function sweep!(b::AbstractVector, (sp, bound, mm), off)
    lin = LinearIndices(indices(mesh(sp)))
    for I in indices(mesh(sp)), (o, w) in local_stencil(bound, sp, I, mm, lin[I])

        Iv = I + CartesianIndex(o)
        checkbounds(Bool, lin, Iv) && (b[lin[Iv] + off] += w)
    end
    return b
end

zeroed(A::SparseMatrixCSC) = (Z = copy(A); fill!(nonzeros(Z), 0); Z)
zeroed(b::AbstractVector) = zero(b)
evaluate(target, units, args) = foldl((r, (x, o...)) -> sweep!(r, args(x), o...), units;
    init = zeroed(target))

# A view of all of `a`: another array type, standing in for the `PtrArray` `@batch` passes.
whole_view(a) = view(a, ntuple(_ -> :, ndims(a))...)

# One form: the walk over its own units matches `assemble` (the control), each unit splits
# into an isbits skeleton and plain arrays, and the walk over the rebuilt units, around
# copies of the arrays, around the split's own (as a one-iteration `@batch` passes them) and
# around views of them, is bitwise the walk over the originals.
function check_form(form_)
    target = assemble(form_)
    units = form_ isa Bramble.BilinearForm ? bilinear_units(form_) : linear_units(form_)
    ref = evaluate(target, units, identity)
    @test ref ≈ target
    splits = [_batch_split(x) for (x, _...) in units]
    for (sk, arrays) in splits
        @test isbits(sk)
        @test !isempty(arrays) && all(plain, arrays)
    end
    k = Ref(0)
    copies = [map(copy, last(s)) for s in splits]
    rebuilt = evaluate(target, units, _ -> _batch_rebuild(first(splits[k[] += 1]), copies[k[]]))
    @test isequal(rebuilt, ref)
    k[] = 0
    own = evaluate(target, units, _ -> _batch_rebuild(splits[k[] += 1]...))
    @test isequal(own, ref)
    k[] = 0
    views = [map(whole_view, last(s)) for s in splits]
    viewed = evaluate(target, units, _ -> _batch_rebuild(first(splits[k[] += 1]), views[k[]]))
    @test isequal(viewed, ref)
    for (s, c) in zip(splits, copies)
        again = last(_batch_split(_batch_rebuild(first(s), c)))
        @test all(i -> again[i] === c[i], eachindex(c))
    end
    return nothing
end

roundtrip(x) = ((sk, arrays) = _batch_split(x); _batch_rebuild(sk, arrays))

struct Holder{V}
    v::V
end
struct Fixed
    v::Vector{Float64}
end
struct Abstract
    v::AbstractVector{Float64}
end
mutable struct Mutable
    v::Vector{Float64}
end
struct WithDict
    d::Dict{Symbol, Vector{Float64}}
end

# The `ArgumentError` message `f()` throws, or "" if it throws none.
message(f) =
    try
        f()
        ""
    catch e
        e isa ArgumentError ? sprint(showerror, e) : rethrow()
    end

const Wₕ = gridspace(jittered(17; seed = 7))
const Sₕ = gridspace(jittered(11; seed = 3))
const κ = Rₕ(Wₕ, x -> 1 + sum(abs2, x))
const fₕ = Rₕ(Wₕ, x -> sin(3sum(x)))
const sₕ = Rₕ(Sₕ, x -> exp(x[1]) * x[2])

@testset "split and rebuild of forms" begin
    @testset "bilinear forms" begin
        r = Ref(2.5)
        check_form(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))
        check_form(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))))
        check_form(form(Wₕ, Wₕ, (u, v) -> r * innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))
        check_form(form(Wₕ, Wₕ,
            (u, v) -> innerₕ(u, restrict_to(:bottom, v)) + inner₊(∇ₕ(u), ∇ₕ(v))))
        check_form(form(Sₕ, Wₕ, (u, v) -> innerₕ(πₕ(u), v)))
        check_form(form(Wₕ × Wₕ, Wₕ × Wₕ,
            (u, v) -> innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) +
                      innerₕ(D₋ₓ(u(1)), v(2))))
    end

    @testset "linear forms" begin
        r = Ref(2.5)
        check_form(form(Wₕ, v -> innerₕ(fₕ, v)))
        check_form(form(Wₕ, v -> innerₕ(fₕ, v) + r * innerₕ(1.0, v)))
        check_form(form(Wₕ, v -> innerₕ(fₕ, restrict_to(:dir, v))))
        check_form(form(Wₕ, v -> innerₕ(πₕ(sₕ), v)))
        check_form(form(Wₕ × Wₕ, v -> innerₕ(fₕ, v(1)) + innerₕ(κ, v(2))))
    end

    @testset "a Ref is read at split time" begin
        r = Ref(2.5)
        l = form(Wₕ, v -> r * innerₕ(fₕ, v))
        (unit, off), = linear_units(l)
        before = sweep!(zero(assemble(l)), unit, off)
        sk, arrays = _batch_split(unit)
        r[] = 7.0
        after = sweep!(zero(before), unit, off)
        @test !isapprox(after, before)
        @test isequal(sweep!(zero(before), _batch_rebuild(sk, arrays), off), before)
        @test isequal(sweep!(zero(before), roundtrip(unit), off), after)
    end

    @testset "meshes become their walk states" begin
        Ω₁ = mesh(domain(interval(0.0, 1.0)), 7, false)
        @test mesh(roundtrip(gridspace(Ω₁))) isa Mesh1DState
        @test mesh(roundtrip(Wₕ)) isa MeshnDState
        @test mesh(roundtrip(Wₕ)).uid == Bramble._walk_mesh(mesh(Wₕ)).uid
    end

    # `Rₕ!`'s kernel holds the coordinate vectors, not the mesh: its skeleton names no mesh
    # type, its arrays are `points(Ωₕ)` themselves, and the rebuild evaluates bit for bit.
    @testset "Rₕ kernels split without the mesh (#503)" begin
        Ω₁ = mesh(domain(interval(0.0, 1.0)), 7, false)
        change_points!(Ω₁, [0.0, 0.05, 0.2, 0.3, 0.55, 0.8, 1.0])
        Ω₃ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (5, 4, 3), (false, false, false))
        rule = Bramble.PointValue(x -> sum(x) + prod(x))
        rules = Bramble.PointValue(x -> (sum(x), prod(x)))
        for W in (gridspace(Ω₁), Wₕ, gridspace(Ω₃))
            for k in (Bramble._rule_kernel(rule, W),
                Bramble._rule_scatter_kernel(rules, W, Val(2)))
                @test Bramble._batch_splittable(typeof(k))
                sk, arrays = _batch_split(k)
                @test isbits(sk)
                @test !occursin(r"Mesh(nD|1D)(State)?\{|Mesh(nD|1D)State",
                    string(typeof(sk)))
                pts = points(mesh(W))
                @test all(map(===, arrays, pts isa Tuple ? pts : (pts,)))
                r = _batch_rebuild(sk, arrays)
                @test all(i -> isequal(r(i), k(i)), eachindex(indices(mesh(W))))
            end
        end
    end

    @testset "split and rebuild allocate nothing" begin
        (unit, _, _), = bilinear_units(form(Wₕ, Wₕ,
            (u, v) -> innerₕ(u, restrict_to(:bottom, v)) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))))
        @test alloc_test(roundtrip, unit) == 0
    end
end

@testset "split and rebuild of other values" begin
    @testset "rebuild takes any array type" begin
        x = (Holder([1.0, 2.0]), 3, Holder((Ref(4), [5 6; 7 8])))
        sk, (a, b) = _batch_split(x)
        @test isbits(sk) && a === x[1].v && b === x[3].v[2]
        y = _batch_rebuild(sk, (view(a, 2:2), b))
        @test y[1] isa Holder{<:SubArray} && y[1].v == [2.0]
        @test y[2] === 3 && y[3].v[1] === 4 && y[3].v[2] === b
    end

    @testset "a misfit field throws" begin
        sk, (a,) = _batch_split(Fixed([1.0]))
        @test _batch_rebuild(sk, (a,)).v === a
        @test occursin("Fixed: field 1", message(() -> _batch_rebuild(sk, (view(a, :),))))
    end

    @testset "a resistor throws naming its type" begin
        @test occursin("Dict{Symbol", message(() -> _batch_split((Wₕ, WithDict(Dict())))))
        @test occursin(r"x\.v::.*Mutable", message(() -> _batch_split((v = Mutable([1.0]),))))
        @test occursin("AbstractVector", message(() -> _batch_split(Abstract([1.0]))))
        @test occursin("RefValue{Vector", message(() -> _batch_split(Ref([1.0]))))
        unbound = form(Wₕ, v -> innerₕ(fₕ, restrict_to(:dir, v))).ast
        @test occursin("Symbol", message(() -> _batch_split((Wₕ, unbound))))
    end
end

# Types no other test splits: the generators meet them for the first time below.
struct FreshInner{V, R}
    a::V
    r::R
end
struct FreshOuter{I, T}
    i::I
    t::T
end

# The generator helpers, which must take every type unspecialised: a new kernel type that
# re-infers them costs a user function its first call (gpena/Bramble.jl#471).
const SPLIT_HELPERS = (Bramble._batch_split_expr!, Bramble._batch_rebuild_expr,
    Bramble._batch_splits!, Bramble._batch_fits, Bramble._batch_ptype,
    Bramble._batch_slot_types!)

function helper_specializations()
    [string(mi.specTypes)
     for m in Iterators.flatten(methods.(SPLIT_HELPERS))
     for mi in Base.specializations(m) if mi !== nothing]
end

# The split generators compile once for every type: splitting and rebuilding a value of a
# new type leaves no helper specialised on it (the round trip itself is the control).
@testset "split generators compile once per type" begin
    x = FreshOuter(FreshInner([1.0, 2.0], Ref(3.0)), ([4, 5], 6))
    @test Bramble._batch_splittable(typeof(x))
    sk, arrays = _batch_split(x)
    y = _batch_rebuild(sk, arrays)
    @test isbits(sk) && length(arrays) == 2
    @test y isa FreshOuter && y.i.a === x.i.a && y.i.r === 3.0 && y.t[1] === x.t[1]
    specs = helper_specializations()
    @test !isempty(specs)
    @test !any(s -> occursin(r"Fresh(Inner|Outer)", s), specs)
end

end # module UtilsBatchSplitTests
