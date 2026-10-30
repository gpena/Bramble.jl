module SpaceShiftTests

using Test
using Bramble
using Bramble: S₊ₓ, S₊ᵧ, S₊₂, S₋ₓ, S₋ᵧ, S₋₂, S₊ₓ!, S₊ᵧ!, S₊₂!, S₋ₓ!, S₋ᵧ!, S₋₂!, S₊, S₋,
               forward_shift, backward_shift, jumpₓ, jump, VectorElement
using Random
using SparseArrays
using ..TestUtils: TEST_DIMS

# The index shifts: `S₊(u)ᵢ = u_{i+1}`, `S₋(u)ᵢ = u_{i-1}`, one grid
# point along a direction, with 0 where the neighbour is off the grid. The reference below
# is that definition written directly on the reshaped grid values, so it depends on none of
# the machinery under test. The meshes are non-uniform: a shift involves no spacing, and a
# non-uniform mesh is what would expose one entering by mistake.

function _domain(D)
    D == 1 ? domain(interval(0.0, 1.0)) :
    D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 2.0)) :
    domain(interval(0.0, 1.0) × interval(0.0, 2.0) × interval(-1.0, 1.0))
end

function _mesh(D; policy = Serial(), seed = 352)
    Random.seed!(seed)
    n = D == 1 ? 7 : ntuple(i -> 4 + i, D)
    unif = D == 1 ? false : ntuple(_ -> false, D)
    return mesh(_domain(D), n, unif; backend = backend(policy = policy))
end

_f(x) = 1 + sum(abs2, x) + prod(x)
_g(x) = sin(3 * x[1]) - x[end]

# The shift of the grid values `U` by `s` points along `d`, 0 off the grid.
function _reference(U::AbstractArray{T, D}, d, s) where {T, D}
    e = CartesianIndex(ntuple(k -> k == d ? s : 0, D))
    return vec([checkbounds(Bool, U, I + e) ? U[I + e] : zero(T) for I in CartesianIndices(U)])
end

_grid(u) = reshape(parent(u), npoints(mesh(space(u)), Tuple))

_in_place_bytes(f!, v, u) = (f!(v, u); @allocated f!(v, u))

@testset "shift: grid function and matrix" begin
    @test Base.isexported(Bramble, :S₊ₕ) && Base.isexported(Bramble, :S₋ₕ)
    @test !Base.isexported(Bramble, :S₊ₓ) && Base.ispublic(Bramble, :S₊ₓ)
    @test !Base.isexported(Bramble, :forward_shift) && Base.ispublic(Bramble, :forward_shift)

    # the vectorial tuples hand back the very coordinate aliases
    @test S₊ₕ[1] === S₊ₓ && S₊ₕ[:y] === S₊ᵧ && S₊ₕ[3] === S₊₂
    sx, sy, sz = S₋ₕ
    @test (sx, sy, sz) === (S₋ₓ, S₋ᵧ, S₋₂)

    bangs = Dict(S₊ₓ => S₊ₓ!, S₊ᵧ => S₊ᵧ!, S₊₂ => S₊₂!, S₋ₓ => S₋ₓ!, S₋ᵧ => S₋ᵧ!, S₋₂ => S₋₂!)
    for D in TEST_DIMS
        Ωₕ = _mesh(D)
        @test !Bramble.is_uniform(Ωₕ)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, _f)
        U = _grid(uₕ)
        for d in 1:D, (op, stem, s) in ((S₊ₕ[d], S₊, 1), (S₋ₕ[d], S₋, -1))

            E = _reference(U, d, s)
            @test parent(op(uₕ)) == E
            @test op(Ωₕ) * parent(uₕ) == E
            @test op(Wₕ) == op(Ωₕ)
            @test op(Ωₕ) == Bramble.shift(Ωₕ, Val(d), Val(s))
            vₕ = similar(uₕ)
            parent(vₕ) .= NaN
            @test bangs[op](vₕ, uₕ) === vₕ
            @test parent(vₕ) == E
            # the dimensional entry point, by Int and by Symbol
            @test parent(stem(uₕ, d)) == E
            @test parent(stem(uₕ, (:x, :y, :z)[d])) == E
        end

        # the vectorial aliases, one entry per direction
        tp = S₊ₕ(uₕ)
        D == 1 ? (@test parent(tp) == _reference(U, 1, 1)) :
        (@test all(parent(tp[d]) == _reference(U, d, 1) for d in 1:D))

        # composite grid functions shift componentwise
        Vₕ = gridspace(Ωₕ, Val(2))
        wₕ = Rₕ(Vₕ, (_f, _g))
        for d in 1:D, op in (S₊ₕ[d], S₋ₕ[d])

            r = op(wₕ)
            @test all(parent(r(k)) == parent(op(wₕ(k))) for k in 1:2)
        end

        # a direction the mesh does not have is refused, not silently zero
        D < 3 && @test_throws ArgumentError S₊₂(uₕ)
        D < 3 && @test_throws ArgumentError S₋₂(Ωₕ)
    end
end

# `spdiagm` returns `Int` indices whatever the backend asked for, so the shift
# matrices on an `Int32` sparse backend came back `Int64`: the backend's index type must
# survive every `kron` factor, and the values match the default backend's.
@testset "shift: Int32 sparse index type kept" begin
    MT = SparseMatrixCSC{Float32, Int32}
    be = backend(vector_type = Vector{Float32}, matrix_type = MT)
    @test Bramble._Eye(be, 5, Val(1)) isa MT
    @test Bramble._Eye(be, 5, Val(-1)) isa MT
    @test Bramble._Eye(be, 5, Val(1)) == spdiagm(5, 5, 1 => ones(Float32, 4))
    for D in TEST_DIMS
        n = D == 1 ? 7 : ntuple(i -> 4 + i, D)
        unif = D == 1 ? false : ntuple(_ -> false, D)
        Random.seed!(352)
        Ωₕ = mesh(_domain(D), n, unif; backend = be)
        Ωₕ⁰ = _mesh(D)
        for d in 1:D, op in (S₊ₕ[d], S₋ₕ[d])

            A = op(Ωₕ)
            @test A isa MT
            @test A == op(Ωₕ⁰)
            @test op(Ωₕ⁰) isa SparseMatrixCSC{Float64, Int}
        end
    end
    # the conversion must not copy a matrix that already has the backend's index type
    be⁰ = backend()
    eye_bytes(n) = minimum(@allocated(Bramble._Eye(be⁰, n, Val(1))) for _ in 1:3)
    diag_bytes(n) = minimum(@allocated(spdiagm(n, n, 1 => ones(n - 1))) for _ in 1:3)
    eye_bytes(1000), diag_bytes(1000)
    @test eye_bytes(1000) <= diag_bytes(1000)
end

@testset "shift: transpose and jump" begin
    for D in TEST_DIMS
        Ωₕ = _mesh(D)
        uₕ = Rₕ(gridspace(Ωₕ), _f)
        for d in 1:D
            # Both truncate the slice with no neighbour to 0, so the superdiagonal ones of
            # S₊ are exactly the subdiagonal ones of S₋: the transpose holds with no
            # boundary correction.
            @test sparse(S₊ₕ[d](Ωₕ))' == sparse(S₋ₕ[d](Ωₕ))

            # `jump` reads the off-grid neighbour as 0 too, so S₊(u) - u is the jump on
            # every point, the last one (-uₙ) included; likewise u - S₋(u) is the unscaled
            # backward difference, u₁ on the first point.
            @test parent(S₊ₕ[d](uₕ)) .- parent(uₕ) == parent(jump(uₕ, Val(d)))
            @test parent(uₕ) .- parent(S₋ₕ[d](uₕ)) ==
                  parent(Bramble.backward_difference(uₕ, Val(d)))
        end
        @test parent(S₊ₓ(uₕ)) .- parent(uₕ) == parent(jumpₓ(uₕ))
    end
end

# Every point is computed by the same loop body under every policy, so the answers must be
# equal, not merely close. `CpuPolyester` needs `BramblePolyesterExt`:
# test/ext/polyester_ext.jl's "Shift engines under CpuPolyester" owns it.
@testset "shift: engines agree" begin
    policies = (Parallel(),)
    for D in TEST_DIMS, policy in policies

        us = Rₕ(gridspace(_mesh(D)), _f)
        up = Rₕ(gridspace(_mesh(D; policy)), _f)
        @test parent(us) == parent(up)
        vs = Rₕ(gridspace(_mesh(D), Val(2)), (_f, _g))
        vp = Rₕ(gridspace(_mesh(D; policy), Val(2)), (_f, _g))
        for d in 1:D, (op, op!) in ((S₊ₕ[d], (S₊ₓ!, S₊ᵧ!, S₊₂!)[d]),
                (S₋ₕ[d], (S₋ₓ!, S₋ᵧ!, S₋₂!)[d]))

            wp = similar(up)
            parent(wp) .= NaN               # every point must be written
            op!(wp, up)
            @test parent(wp) == parent(op(us))
            @test parent(op(up)) == parent(op(us))
            @test parent(op(vp)) == parent(op(vs))
        end
    end
end

@testset "shift: allocation and inference" begin
    for D in TEST_DIMS
        uₕ = Rₕ(gridspace(_mesh(D)), _f)
        vₕ = similar(uₕ)
        for f! in (S₊ₓ!, S₋ₓ!, S₊ᵧ!, S₋ᵧ!, S₊₂!, S₋₂!)[1:(2D)]
            @test _in_place_bytes(f!, vₕ, uₕ) == 0
        end
        @test (@inferred S₊ₓ(uₕ)) isa VectorElement
        @test (@inferred S₋ₓ(uₕ)) isa VectorElement
        @test (@inferred S₊(uₕ, 1)) isa VectorElement
        @test (@inferred S₊ₕ(uₕ)) isa (D == 1 ? VectorElement : NTuple{D, VectorElement})
        @test (@inferred S₋ₓ(mesh(space(uₕ)))) isa AbstractMatrix
    end
end

# `true == 1`, so a range check alone lets `Val(true)` through as the first axis
# (gpena/Bramble.jl#689): every direction check must name the type, with the Bool message.
function _bool_refused(f)
    err = try
        f()
        nothing
    catch e
        e
    end
    return err isa ArgumentError && occursin("not a Bool", sprint(showerror, err))
end

@testset "shift: Bool direction refused" begin
    Ωₕ = _mesh(2)
    uₕ = Rₕ(gridspace(Ωₕ), _f)
    U = Bramble.TrialFunction{2, 1}()
    for f in (() -> S₊(U, Val(true)), () -> S₋(U, Val(true)),
        () -> forward_shift(U, Val(true)), () -> Bramble.shift_op(U, Val(true), 1),
        () -> Bramble.D₋(U, Val(true)),
        () -> S₊(uₕ, Val(true)), () -> S₋(uₕ, Val(true)),
        () -> Bramble.D₋(uₕ, Val(true)), () -> Bramble.forward_average(uₕ, Val(true)),
        () -> Bramble.centered_average(uₕ, Val(true)), () -> Bramble.D₊(Ωₕ, Val(true)),
        () -> Bramble.backward_average(Ωₕ, Val(true)),
        () -> Bramble.centered_average(Ωₕ, Val(true)))
        @test _bool_refused(f)
    end
    # positive control: an Int direction still builds and shifts
    @test parent(S₊(uₕ, Val(1))) == _reference(_grid(uₕ), 1, 1)
    @test S₊(U, Val(2)) isa Bramble.ShiftNode{2, 2}
    @test Bramble.D₋(U, Val(2)) isa Bramble.LazyOp{2}
    @test (@inferred S₊(uₕ, Val(2))) isa VectorElement
    # the index route: a vectorial alias or `η` indexed by `true` is not its `x` entry
    for f in (() -> ∇ₕ[true], () -> S₊ₕ[true], () -> Bramble.Mₕ[true], () -> Bramble.η[true])
        @test _bool_refused(f)
    end
    @test ∇ₕ[1] === Bramble.∇ₕ[Int32(1)] && S₊ₕ[UInt(2)] === S₊ᵧ
    @test Bramble.η[Int32(2)] === Bramble.η[2]
    # any other integer type is still a direction, and gives the `Int` result
    @test parent(S₊(uₕ, Val(Int32(1)))) == parent(S₊(uₕ, Val(1)))
    @test parent(S₋(uₕ, Val(UInt(2)))) == parent(S₋(uₕ, Val(2)))
    @test jump(Ωₕ, Val(Int32(2))) == jump(Ωₕ, Val(2))
    @test Bramble.Mₕ(Ωₕ, Val(Int32(1))) == Bramble.Mₕ(Ωₕ, Val(1))
    @test (@inferred S₊(uₕ, Val(Int32(2)))) isa VectorElement
    # a symbolic node built from an `Int32` direction is the `Int` node, and assembles so
    @test S₊(U, Val(Int32(2))) === S₊(U, Val(2))
    @test Bramble.D₋(U, Val(UInt(1))) === Bramble.D₋(U, Val(1))
    Wₕ = gridspace(Ωₕ)
    _shift_matrix(s) = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(s(u), v)))
    @test _shift_matrix(u -> Bramble.shift_op(u, Val(Int32(2)), 1)) ==
          _shift_matrix(u -> Bramble.shift_op(u, Val(2), 1))
    @test _shift_matrix(u -> S₊(u, Val(Int32(2)))) == _shift_matrix(u -> S₊(u, Val(2)))
    # a non-integer direction is named as such, without the Bool clause
    err = try
        Bramble.shift_op(U, Val(1.0), 1)
    catch e
        e
    end
    @test err isa ArgumentError &&
          sprint(showerror, err) ==
          "ArgumentError: the stencil direction must be an integer, got 1.0 of type Float64"
end

# GPU testing is off, so the refusal is checked in core: through the `GpuKernel()`
# policy the shift engine dispatches on, and through a host space carrying device-backed
# data, which `_MockDevice` fakes by answering `DeviceLocality()`.
struct _MockDevice{T} <: DenseVector{T}
    data::Vector{T}
end
Base.size(A::_MockDevice) = size(A.data)
Base.getindex(A::_MockDevice, i::Int) = A.data[i]
Base.setindex!(A::_MockDevice, v, i::Int) = (A.data[i] = v)
Base.IndexStyle(::Type{<:_MockDevice}) = IndexLinear()
Bramble.locality(::Type{<:_MockDevice}) = Bramble.DeviceLocality()

function _refusal(f)
    err = try
        f()
        nothing
    catch e
        e
    end
    return err isa ArgumentError && occursin("v4.11.0", sprint(showerror, err))
end

@testset "shift: device refused (v4.11.0)" begin
    Wₕ = gridspace(_mesh(2))
    uₕ = Rₕ(Wₕ, _f)
    out = similar(parent(uₕ))
    dims = npoints(mesh(Wₕ), Tuple)
    @test _refusal(() -> Bramble._shift_engine!(
        Bramble.GpuKernel(), out, parent(uₕ), dims, Bramble.Forward(), Val(1)))
    dₕ = VectorElement(_MockDevice(copy(parent(uₕ))), Wₕ)
    @test _refusal(() -> S₊ₓ(dₕ))
    @test _refusal(() -> S₋ᵧ!(similar(uₕ), dₕ))
    # positive control: the same data on the host shifts
    @test parent(S₊ₓ(uₕ)) == _reference(_grid(uₕ), 1, 1)
end

end # module SpaceShiftTests
