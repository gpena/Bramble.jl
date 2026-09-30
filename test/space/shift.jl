module SpaceShiftTests

using Test
using Bramble
using Bramble: S₊ₓ, S₊ᵧ, S₊₂, S₋ₓ, S₋ᵧ, S₋₂, S₊ₓ!, S₊ᵧ!, S₊₂!, S₋ₓ!, S₋ᵧ!, S₋₂!, S₊, S₋,
               forward_shift, backward_shift, jumpₓ, jump, VectorElement
using Random
using SparseArrays

# The index shifts (gpena/Bramble.jl#352): `S₊(u)ᵢ = u_{i+1}`, `S₋(u)ᵢ = u_{i-1}`, one grid
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
    for D in 1:3
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

@testset "shift: transpose and jump" begin
    for D in 1:3
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
    for D in 1:3, policy in policies

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
    for D in 1:3
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

# GPU testing is off (v4.4.0), so the refusal is checked in core: through the `GpuKernel()`
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
    return err isa ArgumentError && occursin("v4.4.0", sprint(showerror, err))
end

@testset "shift: device refused (v4.4.0)" begin
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
