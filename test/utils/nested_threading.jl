module UtilsNestedThreadingTests

# Every `CpuThreaded` (`Parallel()`) sweep runs a `Threads.@threads :static` loop, and Base
# refuses to start one of those inside another threaded region ("`@threads :static` cannot be
# used concurrently or nested"). So a user who wraps Bramble calls in their own
# `Threads.@threads for p in params ... end` -- a parameter sweep, say -- used to get that
# error from `innerₕ`, `assemble`, the stencil engines and the vector-calculus operators.
# `_in_threaded_region()` (src/utils/linear_algebra.jl) is the exact test Base makes, and
# every such sweep now branches on it, running the same per-chunk or per-band body serially
# on the calling task when nested. That counter is process-wide, so two `Threads.@spawn`ed
# tasks can race: one reads `false`, then the other enters its region before Base's own
# check. `_static_or_serial` therefore also catches exactly Base's error (raised before any
# iteration runs) and answers it serially.
#
# Invariants tested:
# 1. `_in_threaded_region()` is `false` at top level and in a spawned task outside a
#    threaded region, `true` inside a `Threads.@threads` loop, and inferred as `Bool`.
# 2. Each `Parallel()` operation called from inside a user's `Threads.@threads` loop returns
#    exactly (`==`) what the same call returns at top level -- reductions included, since
#    the nested path sums the same partial sums in the same order.
# 3. A top-level call still spreads its work over at least two threads (when there are two),
#    while a nested call stays on the calling task's thread.
# 4. `_is_static_nesting_error` recognises the error Base actually raises; `_static_or_serial`
#    answers that error with the serial body and rethrows any other.
# 5. Many concurrent `Threads.@spawn`ed calls (no threaded region of their own) never throw
#    and each returns the top-level result.
#
# The mesh is non-uniform, so a band that picked up the wrong spacing index would not agree.
using Test
using Bramble
using Bramble: CpuThreaded, _dot, _in_threaded_region, _is_static_nesting_error,
               _static_or_serial, _serial_for!, _threaded_for!, assemble_add!,
               assemble_parallel!, D₋ₓ!, Mₓ!, divₕ!, εₕ!, εcₕ!, curlₕ!, ∇̃ₕ!
using SparseArrays: nonzeros

const _D2 = domain(interval(0.0, 1.0) × interval(0.0, 2.0))
_mesh(n, policy) = mesh(_D2, (n, n), (false, false); backend = backend(policy = policy))

# Run `f` once per iteration of a user-level `Threads.@threads` loop and return every result.
function _nested_results(f)
    out = Vector{Any}(undef, 2 * Threads.nthreads())
    Threads.@threads for i in eachindex(out)
        out[i] = f()
    end
    return out
end

# A vector that records which threads read it.
const _SEEN = Threads.Atomic{UInt64}(0)
struct _Spy{T} <: AbstractVector{T}
    x::Vector{T}
end
Base.size(s::_Spy) = size(s.x)
Base.IndexStyle(::Type{<:_Spy}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(s::_Spy, i::Int)
    Threads.atomic_or!(_SEEN, UInt64(1) << (Threads.threadid() - 1))
    return s.x[i]
end

@testset "Nested threading: Parallel() inside a user's threaded region" begin
    @testset "_in_threaded_region" begin
        @test !_in_threaded_region()
        @test (@inferred _in_threaded_region()) isa Bool
        @test !fetch(Threads.@spawn _in_threaded_region())
        @test all(_nested_results(_in_threaded_region))
    end

    @testset "Base's :static error is caught, and only it" begin
        base_err = _nested_results() do
            try
                Threads.@threads :static for _ in 1:2
                end
                return nothing
            catch err
                return err
            end
        end
        @test all(_is_static_nesting_error, base_err)
        @test !_is_static_nesting_error(ErrorException("something else"))
        @test !_is_static_nesting_error(ArgumentError("`@threads :static` cannot be used concurrently or nested"))

        # A `threaded!` that meets Base's error falls back to the serial body; any other error
        # propagates.
        raises_static(v, idxs, f) = error("`@threads :static` cannot be used concurrently or nested")
        raises_other(v, idxs, f) = error("unrelated")
        dest = zeros(Int, 10)
        @test _static_or_serial(raises_static, _serial_for!, dest, 1:10, i -> 2i) === nothing
        @test dest == 2 .* (1:10)
        @test_throws ErrorException("unrelated") _static_or_serial(
            raises_other, _serial_for!, dest, 1:10, i -> i)
    end

    W = gridspace(_mesh(60, Parallel()))
    Ws = gridspace(_mesh(60, Serial()))
    u = Rₕ(W, x -> sin(sum(x)))
    v = Rₕ(W, x -> prod(x))
    comps = (Rₕ(W, x -> x[1]^2), Rₕ(W, x -> x[2]))
    bilinear(p, q) = innerₕ(p, q) + inner₊(∇ₕ(p), ∇ₕ(q))
    a = form(W, W, bilinear)
    as = form(Ws, Ws, bilinear)
    l = form(W, q -> innerₕ(u, q))
    A0 = assemble(a)
    As0 = assemble(as)
    zeroed(A) = (B = copy(A); fill!(nonzeros(B), 0); B)
    tensor() = ntuple(_ -> ntuple(_ -> similar(u), 2), 2)

    cases = [
        ("Rₕ!", () -> parent(Rₕ!(similar(u), x -> sum(x)))),
        ("Rₕ! masked", () -> parent(Rₕ!(copy(u), x -> sum(x); markers = (:boundary,)))),
        ("innerₕ", () -> innerₕ(u, v)),
        ("innerₕ masked", () -> innerₕ(u, v; markers = (:boundary,))),
        ("inner₊", () -> inner₊(u, v, Val(()))),
        ("inner₊ masked", () -> inner₊(u, v, Val(()); markers = (:boundary,))),
        ("inner₊ two markers", () -> inner₊(u, v, Val(()); markers = (:boundary, :interior))),
        ("assemble bilinear", () -> Matrix(assemble(a))),
        ("assemble! bilinear", () -> (A = copy(A0); assemble!(A, a); Matrix(A))),
        ("assemble_add!", () -> (A = zeroed(A0); assemble_add!(A, a); Matrix(A))),
        ("assemble_parallel!", () -> (A = zeroed(A0); assemble_parallel!(A, a); Matrix(A))),
        ("assemble_parallel! from Serial()",
            () -> (A = zeroed(As0); assemble_parallel!(A, as); Matrix(A))),
        ("assemble linear", () -> assemble(l)),
        ("D₋ₓ!", () -> (o = similar(u); D₋ₓ!(o, u); parent(o))),
        ("Mₓ!", () -> (o = similar(u); Mₓ!(o, u); parent(o))),
        ("∇̃ₕ!", () -> (o = (similar(u), similar(u)); ∇̃ₕ!(o, u); parent.(o))),
        ("divₕ!", () -> (o = similar(u); divₕ!(o, comps); parent(o))),
        ("curlₕ!", () -> (o = similar(u); curlₕ!(o, comps); parent(o))),
        ("εₕ!", () -> (o = tensor(); εₕ!(o, comps); map(r -> parent.(r), o))),
        ("εcₕ!", () -> (o = tensor(); εcₕ!(o, comps); map(r -> parent.(r), o)))
    ]

    @testset "$name: nested == top level" for (name, f) in cases
        ref = f()
        @test all(==(ref), _nested_results(f))
        # A spawned task outside any threaded region is not nested: it threads as usual.
        @test fetch(Threads.@spawn f()) == ref
    end

    @testset "Concurrent spawned calls never throw" begin
        # The critic's scenario (a task-parallel sweep, no threaded region anywhere): a small
        # Float32 space so each call is short and the calls overlap as often as possible.
        M = mesh(_D2, (23, 17), (false, false); backend = backend(Float32; policy = Parallel()))
        W32 = gridspace(M)
        u32 = Rₕ(W32, x -> sin(sum(x)))
        l32 = form(W32, q -> innerₕ(u32, q))
        a32 = form(W32, W32, bilinear)
        work() = (copy(assemble(l32)), Matrix(assemble(a32)), inner₊(u32, u32, Val(())))
        ref = work()
        ok = Threads.Atomic{Int}(0)
        threw = Threads.Atomic{Int}(0)
        calls = 0
        deadline = time() + 3.0
        while time() < deadline
            calls += 6
            @sync for _ in 1:6
                Threads.@spawn try
                    work() == ref && Threads.atomic_add!(ok, 1)
                catch
                    Threads.atomic_add!(threw, 1)
                end
            end
        end
        @test threw[] == 0
        @test ok[] == calls
    end

    @testset "Top-level calls still thread, nested ones stay put" begin
        n = 200_000
        x, y, z = rand(n), rand(n), rand(n)
        P = CpuThreaded()

        _SEEN[] = 0
        ref = _dot(P, _Spy(x), y, z)
        if Threads.nthreads() >= 2
            @test count_ones(_SEEN[]) >= 2
        else
            @test_skip "top-level threading needs --threads >= 2"
        end
        @test all(==(ref), _nested_results(() -> _dot(P, _Spy(x), y, z)))

        # `_threaded_for!` at top level spreads over the threads; nested, every index is
        # written by the task that made the call.
        dest = zeros(Int, n)
        _threaded_for!(dest, 1:n, _ -> Threads.threadid())
        if Threads.nthreads() >= 2
            @test length(unique(dest)) >= 2
        else
            @test_skip "top-level threading needs --threads >= 2"
        end
        onecaller = _nested_results() do
            d = zeros(Int, 1000)
            tid = Threads.threadid()
            _threaded_for!(d, 1:1000, _ -> Threads.threadid())
            return all(==(tid), d)
        end
        @test all(onecaller)
    end
end

end # module
