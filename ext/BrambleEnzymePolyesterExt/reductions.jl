# reductions.jl: Enzyme rules for the `CpuPolyester` reductions `_batch_dot` and
# `_batch_dot_masked` (ext/BramblePolyesterExt.jl). Each hook has a plain-weight method and a
# `SeparableWeights` method; each rule takes `Duplicated` or `BatchDuplicated` arguments, the
# width from `EnzymeRules.width(config)`.
#
# The value is z = Σᵢ conj(uᵢ) wᵢ vᵢ over the indices the mask keeps (all of them without a
# mask). Every derivative loop runs in `@batch` bands over the linear indices, and each band
# writes only its own entries. A forward band sums its share of every lane's tangent into its
# own row of `partials`, in one `@batch`. A reverse band adds into ū, v̄ and a plain w̄ at its
# own indices, in one `@batch`. A `SeparableWeights` weight is the product of its factors, so
# its factor cotangents are gathered over the slice of the grid each factor index owns: one
# `@batch` per factor, a band of that factor's index per task. The forward tangent of such a
# weight is the product rule applied at each index, in the same band as the others.
#
# A `SeparableWeights` weight whose factors are active gets factor shadows; an inactive factor
# of an active weight gets a zero tangent, and its gathered cotangent is dropped.
#
# The conjugations follow Enzyme's rules for `conj`: ū += conj(z̄) w v, v̄ += z̄ u conj(w),
# w̄ += z̄ u conj(v). For real data these are the plain products. The harness is real, so
# complex data is not checked by the CHECK.

const _DotFn = typeof(Bramble._batch_dot)
const _MaskedDotFn = typeof(Bramble._batch_dot_masked)

# The weight at linear index `i`: an entry of a vector, or the product of the factors at the
# index's `CartesianIndex` for a `SeparableWeights` (`cart` is `nothing` for a vector).
@inline _weight_at(w::AbstractVector, ::Nothing, i) = @inbounds w[i]
@inline function _weight_at(w::Bramble.SeparableWeights, cart::CartesianIndices, i)
    return _factor_product(w.factors, @inbounds cart[i])
end
@inline function _factor_product(factors::Tuple, I::CartesianIndex{D}) where {D}
    return prod(ntuple(l -> @inbounds(factors[l][I[l]]), Val(D)))
end

# Whether index `i` survives the mask (every index without one).
@inline _kept(::Nothing, i::Int) = true
@inline function _kept(mask, i::Int)
    word = Bramble._mask_word(mask, ((i - 1) >> 6) + 1)
    return ((word >> ((i - 1) & 63)) & 0x1) != 0
end

@inline _geometry(::AbstractVector) = nothing
@inline _geometry(w::Bramble.SeparableWeights) = CartesianIndices(w.dims)

@inline _mask_value(::Nothing) = nothing
@inline _mask_value(mask::Annotation) = mask.val

# The product rule for a `SeparableWeights` tangent at `I`: each factor's tangent times the
# other factors.
@inline function _factor_tangent_at(dfs::Tuple, factors::Tuple, I::CartesianIndex{D}) where {D}
    return sum(ntuple(
        d -> dfs[d][I[d]] * prod(ntuple(
            l -> l == d ? one(eltype(factors[l])) : factors[l][I[l]], Val(D))),
        Val(D)))
end

# The weight's tangent term at index `i` of `s + ui dw vi`: a vector shadow's entry, the
# factor product rule, or nothing when the weight is inactive.
@inline _dw_term(s, ::Nothing, w, cart, ui, vi, i, ::Type{T}) where {T} = s
@inline function _dw_term(s, dw::AbstractVector, w, cart, ui, vi, i, ::Type{T}) where {T}
    return muladd(ui * T(dw[i]), vi, s)
end
@inline function _dw_term(s, dfs::Tuple, w::Bramble.SeparableWeights, cart, ui, vi, i,
        ::Type{T}) where {T}
    return muladd(ui * T(_factor_tangent_at(dfs, w.factors, cart[i])), vi, s)
end

# One band's share of one lane's tangent: `conj(du) w v + conj(u) w dv + conj(u) dw v` over
# the kept indices of band `b`. A `nothing` shadow contributes nothing.
@noinline function _tangent_band(u, w, cart, v, du, dw, dv, mask, n::Int, nb::Int, b::Int,
        ::Type{T}) where {T}
    s = zero(T)
    @inbounds for i in Bramble._band_range(Base.OneTo(n), nb, b)
        _kept(mask, i) || continue
        ui = conj(T(u[i]))
        vi = T(v[i])
        wi = T(_weight_at(w, cart, i))
        du === nothing || (s = muladd(conj(T(du[i])) * wi, vi, s))
        dv === nothing || (s = muladd(ui * wi, T(dv[i]), s))
        s = _dw_term(s, dw, w, cart, ui, vi, i, T)
    end
    return s
end

@inline function _tangent_lanes(u, w, cart, v, dus::Tuple, dws::Tuple, dvs::Tuple, mask,
        n::Int, nb::Int, b::Int, ::Type{T}) where {T}
    return map((du, dw, dv) -> _tangent_band(u, w, cart, v, du, dw, dv, mask, n, nb, b, T),
        dus, dws, dvs)
end

@inline function _store_lanes!(partials::AbstractMatrix, b::Int, sums::Tuple)
    for k in eachindex(sums)
        @inbounds partials[b, k] = sums[k]
    end
    return nothing
end

# Every lane's tangent in one `@batch` launch, each band summing its share of every lane.
function _dot_tangents(uval, wval, vval, mask, dus, dws, dvs, ::Val{W}, ::Type{T}) where {W, T}
    n = length(uval)
    nb = Threads.nthreads()
    cart = _geometry(wval)
    partials = Matrix{T}(undef, nb, W)
    @batch for b in 1:nb
        _store_lanes!(partials, b,
            _tangent_lanes(uval, wval, cart, vval, dus, dws, dvs, mask, n, nb, b, T))
    end
    return ntuple(k -> sum(view(partials, :, k)), Val(W))
end

# One band's adjoint of one lane: ū, v̄ and a plain w̄ at the kept indices of band `b`.
@noinline function _adjoint_band!(u, w, cart, v, du, dw, dv, mask, z, n::Int, nb::Int, b::Int,
        ::Type{T}) where {T}
    zt = T(z)
    @inbounds for i in Bramble._band_range(Base.OneTo(n), nb, b)
        _kept(mask, i) || continue
        ui = T(u[i])
        vi = T(v[i])
        wi = T(_weight_at(w, cart, i))
        du === nothing || (du[i] += conj(zt) * wi * vi)
        dv === nothing || (dv[i] += zt * ui * conj(wi))
        dw isa AbstractVector && (dw[i] += zt * ui * conj(vi))
    end
    return nothing
end

@inline function _adjoint_lanes!(u, w, cart, v, dus::Tuple, dws::Tuple, dvs::Tuple, mask,
        zs::Tuple, n::Int, nb::Int, b::Int, ::Type{T}) where {T}
    foreach((du, dw, dv, z) -> _adjoint_band!(u, w, cart, v, du, dw, dv, mask, z, n, nb, b, T),
        dus, dws, dvs, zs)
    return nothing
end

# The sum over the slice of the grid where the `d`-th index is `j`, of `ui conj(vi)` times
# the other factors conjugated: the gathered cotangent of factor `d` at `j`, per unit seed.
@inline function _factor_slice_sum(u, v, factors::Tuple, d::Int, j::Int, dims::Tuple, lin,
        mask, ::Type{T}) where {T}
    D = length(dims)
    slice = CartesianIndices(ntuple(l -> l == d ? (j:j) : (1:dims[l]), Val(D)))
    s = zero(T)
    for I in slice
        i = lin[I]
        _kept(mask, i) || continue
        p = prod(ntuple(l -> l == d ? one(T) : conj(T(factors[l][I[l]])), Val(D)))
        s = muladd(T(u[i]) * conj(T(v[i])), p, s)
    end
    return s
end

@noinline function _factor_adjoint!(df::AbstractVector, d::Int, u, v, factors::Tuple,
        dims::Tuple, lin, mask, z, nb::Int, b::Int, ::Type{T}) where {T}
    @inbounds for j in Bramble._band_range(Base.OneTo(length(df)), nb, b)
        df[j] += T(z) * _factor_slice_sum(u, v, factors, d, j, dims, lin, mask, T)
    end
    return nothing
end

@inline _factor_adjoint_lane!(::Nothing, args...) = nothing
@inline _factor_adjoint_lane!(dfs::Tuple, d::Int, args...) = _factor_adjoint!(dfs[d], d, args...)

@inline function _factor_adjoint_lanes!(dws::Tuple, d::Int, u, v, factors::Tuple, dims::Tuple,
        lin, mask, zs::Tuple, nb::Int, b::Int, ::Type{T}) where {T}
    foreach((dfs, z) -> _factor_adjoint_lane!(dfs, d, u, v, factors, dims, lin, mask, z, nb,
            b, T), dws, zs)
    return nothing
end

# Every lane's adjoint in `@batch` launches: one for the elementwise cotangents, then one per
# factor of a `SeparableWeights` weight.
function _dot_adjoints!(uval, wval, vval, mask, dus, dws, dvs, zs::Tuple, ::Type{T}) where {T}
    n = length(uval)
    nb = Threads.nthreads()
    cart = _geometry(wval)
    @batch for b in 1:nb
        _adjoint_lanes!(uval, wval, cart, vval, dus, dws, dvs, mask, zs, n, nb, b, T)
    end
    if wval isa Bramble.SeparableWeights
        lin = LinearIndices(wval.dims)
        for d in eachindex(wval.factors)
            @batch for b in 1:nb
                _factor_adjoint_lanes!(dws, d, uval, vval, wval.factors, wval.dims, lin, mask,
                    zs, nb, b, T)
            end
        end
    end
    return nothing
end

# A lane's factor shadows: the shadow factors, with a zero where a factor is inactive (its
# shadow is its primal); `nothing` where the weight is inactive.
@inline _factor_shadows(::Nothing, w::Bramble.SeparableWeights, ::Type{T}) where {T} = nothing
@inline function _factor_shadows(s::Bramble.SeparableWeights, w::Bramble.SeparableWeights,
        ::Type{T}) where {T}
    return map((df, f) -> df === f ? zeros(T, length(f)) : df, s.factors, w.factors)
end

# The weight's shadow in each lane: a vector's, or the factor shadows of a `SeparableWeights`.
@inline _weight_shadows(w::Annotation{<:AbstractVector}, ::Val{W}, ::Type{T}) where {W, T} = _lanes(w, Val(W))
@inline function _weight_shadows(w::Annotation{<:Bramble.SeparableWeights}, ::Val{W},
        ::Type{T}) where {W, T}
    return map(s -> _factor_shadows(s, w.val, T), _lanes(w, Val(W)))
end

# An argument's primal value: the taped copy when the rule was overwritten, else the primal.
_taped(overwritten::Bool, x) = overwritten ? _tape_copy(x) : nothing
_tape_copy(x::AbstractArray) = copy(x)
_tape_copy(w::Bramble.SeparableWeights) = Bramble.SeparableWeights(map(copy, w.factors), w.dims)

_seeds(dret::Active, ::Val{1}) = (dret.val,)
_seeds(dret::Tuple, ::Val{W}) where {W} = ntuple(k -> dret[k].val, Val(W))

function _forward_dot(config, hook::Symbol, func, u, w, v, mask)
    _entered!(hook)
    mv = _mask_value(mask)
    val = mv === nothing ? func.val(u.val, w.val, v.val) : func.val(u.val, w.val, v.val, mv)
    if !EnzymeRules.needs_shadow(config)
        return EnzymeRules.needs_primal(config) ? val : nothing
    end
    W = EnzymeRules.width(config)
    T = promote_type(eltype(u.val), eltype(w.val), eltype(v.val))
    dvals = _dot_tangents(u.val, w.val, v.val, mv, _lanes(u, Val(W)),
        _weight_shadows(w, Val(W), T), _lanes(v, Val(W)), Val(W), T)
    shadow = W == 1 ? dvals[1] : dvals
    EnzymeRules.needs_primal(config) || return shadow
    return W == 1 ? Duplicated(val, shadow) : BatchDuplicated(val, shadow)
end

function _augmented_dot(config, hook::Symbol, func, u, w, v, mask)
    _entered!(hook)
    mv = _mask_value(mask)
    val = mv === nothing ? func.val(u.val, w.val, v.val) : func.val(u.val, w.val, v.val, mv)
    ow = EnzymeRules.overwritten(config)
    tape = (_taped(ow[2], u.val), _taped(ow[3], w.val), _taped(ow[4], v.val))
    return AugmentedReturn(EnzymeRules.needs_primal(config) ? val : nothing, nothing, tape)
end

function _reverse_dot(config, dret, tape, u, w, v, mask)
    W = EnzymeRules.width(config)
    zs = _seeds(dret, Val(W))
    uval = tape[1] === nothing ? u.val : tape[1]
    wval = tape[2] === nothing ? w.val : tape[2]
    vval = tape[3] === nothing ? v.val : tape[3]
    T = promote_type(eltype(uval), eltype(wval), eltype(vval))
    dus = _lanes(u, Val(W))
    dvs = _lanes(v, Val(W))
    dws = _weight_shadows(w, Val(W), T)
    any(!isnothing, dus) || any(!isnothing, dvs) || any(!isnothing, dws) || return nothing
    _dot_adjoints!(uval, wval, vval, _mask_value(mask), dus, dws, dvs, zs, T)
    return nothing
end

# The plain-weight methods of `_batch_dot`.
function EnzymeRules.forward(config::FwdConfig, func::Const{_DotFn}, ::Type,
        u::Annotation{<:AbstractVector}, w::Annotation{<:AbstractVector},
        v::Annotation{<:AbstractVector})
    return _forward_dot(config, :_batch_dot, func, u, w, v, nothing)
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_DotFn}, ::Type,
        u::Annotation{<:AbstractVector}, w::Annotation{<:AbstractVector},
        v::Annotation{<:AbstractVector})
    return _augmented_dot(config, :_batch_dot, func, u, w, v, nothing)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_DotFn}, dret, tape,
        u::Annotation{<:AbstractVector}, w::Annotation{<:AbstractVector},
        v::Annotation{<:AbstractVector})
    _reverse_dot(config, dret, tape, u, w, v, nothing)
    return (nothing, nothing, nothing)
end

# The `SeparableWeights` methods of `_batch_dot`.
function EnzymeRules.forward(config::FwdConfig, func::Const{_DotFn}, ::Type,
        u::Annotation{<:AbstractVector}, w::Annotation{<:Bramble.SeparableWeights},
        v::Annotation{<:AbstractVector})
    return _forward_dot(config, :_batch_dot, func, u, w, v, nothing)
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_DotFn}, ::Type,
        u::Annotation{<:AbstractVector}, w::Annotation{<:Bramble.SeparableWeights},
        v::Annotation{<:AbstractVector})
    return _augmented_dot(config, :_batch_dot, func, u, w, v, nothing)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_DotFn}, dret, tape,
        u::Annotation{<:AbstractVector}, w::Annotation{<:Bramble.SeparableWeights},
        v::Annotation{<:AbstractVector})
    _reverse_dot(config, dret, tape, u, w, v, nothing)
    return (nothing, nothing, nothing)
end

# The plain-weight methods of `_batch_dot_masked`.
function EnzymeRules.forward(config::FwdConfig, func::Const{_MaskedDotFn}, ::Type,
        u::Annotation{<:AbstractVector}, w::Annotation{<:AbstractVector},
        v::Annotation{<:AbstractVector}, mask::Annotation)
    return _forward_dot(config, :_batch_dot_masked, func, u, w, v, mask)
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_MaskedDotFn}, ::Type,
        u::Annotation{<:AbstractVector}, w::Annotation{<:AbstractVector},
        v::Annotation{<:AbstractVector}, mask::Annotation)
    return _augmented_dot(config, :_batch_dot_masked, func, u, w, v, mask)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_MaskedDotFn}, dret, tape,
        u::Annotation{<:AbstractVector}, w::Annotation{<:AbstractVector},
        v::Annotation{<:AbstractVector}, mask::Annotation)
    _reverse_dot(config, dret, tape, u, w, v, mask)
    return (nothing, nothing, nothing, nothing)
end

# The `SeparableWeights` methods of `_batch_dot_masked`.
function EnzymeRules.forward(config::FwdConfig, func::Const{_MaskedDotFn}, ::Type,
        u::Annotation{<:AbstractVector}, w::Annotation{<:Bramble.SeparableWeights},
        v::Annotation{<:AbstractVector}, mask::Annotation)
    return _forward_dot(config, :_batch_dot_masked, func, u, w, v, mask)
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_MaskedDotFn}, ::Type,
        u::Annotation{<:AbstractVector}, w::Annotation{<:Bramble.SeparableWeights},
        v::Annotation{<:AbstractVector}, mask::Annotation)
    return _augmented_dot(config, :_batch_dot_masked, func, u, w, v, mask)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_MaskedDotFn}, dret, tape,
        u::Annotation{<:AbstractVector}, w::Annotation{<:Bramble.SeparableWeights},
        v::Annotation{<:AbstractVector}, mask::Annotation)
    _reverse_dot(config, dret, tape, u, w, v, mask)
    return (nothing, nothing, nothing, nothing)
end
