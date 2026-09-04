"""
    CartesianProduct{D, T}(box::SVector{D, Tuple{T, T}}, collapsed::SVector{D, Bool})

Cartesian product of `D` closed intervals embedded in ``\\mathbb{R}^D`` with scalar coordinate type `T`.

# Fields
- `box`: Statically sized vector of `D` interval endpoint pairs `(min, max)`.
- `collapsed`: Statically sized vector of `D` boolean flags indicating whether each dimension is degenerate (`min ≈ max`).

See also: [`interval`](@ref), [`point`](@ref), [`cartesian_product`](@ref), [`box`](@ref).
"""
struct CartesianProduct{D, T}
    box::SVector{D, Tuple{T, T}}
    collapsed::SVector{D, Bool}
end

@noinline _throw_bounds_error(X::CartesianProduct, i) = throw(BoundsError(X, i))
@noinline _throw_interval_error(x,
    y) = throw(ArgumentError("Invalid interval: expected x <= y, but got x = $x, y = $y"))
@noinline _throw_box_error(box) = throw(ArgumentError("Invalid box: Each tuple must satisfy x[1] <= x[2]. Found box: $box"))

"""
    is_collapsed(a::Number, b::Number) -> Bool
    is_collapsed(X::CartesianProduct) -> Bool
    is_collapsed(X::CartesianProduct, i::Integer) -> Bool

Check whether an interval endpoint pair, a Cartesian set, or a coordinate dimension `i` is degenerate (`min ≈ max`).

For an ``n``-dimensional [`CartesianProduct`](@ref) `X`, `is_collapsed(X)` returns `true` if any
coordinate dimension is collapsed (`any(X.collapsed)`), equivalently when the topological dimension
is strictly less than the spatial embedding dimension `D`.
`is_collapsed(X, i)` queries whether the `i`-th coordinate axis is degenerate.

# Arguments
- `a`, `b`: Scalar interval endpoints.
- `X`: Cartesian product set.
- `i`: Coordinate dimension index (`1 <= i <= D`).

# Throws
- `BoundsError`: If `i < 1` or `i > D`.
"""
@inline is_collapsed(a::T, b::T) where {T <: Number} = isapprox(a, b)
@inline is_collapsed(a::Number, b::Number) = isapprox(promote(a, b)...)
@inline is_collapsed(X::CartesianProduct) = any(X.collapsed)
@inline function is_collapsed(X::CartesianProduct{D}, i::Integer) where {D}
    @boundscheck (1 <= i <= D) || _throw_bounds_error(X, i)
    return @inbounds X.collapsed[i]
end

"""
    set(X::CartesianProduct) -> CartesianProduct

Identity accessor returning the geometric set `X`.
"""
@inline set(X::CartesianProduct) = X

"""
    interval(x::Number, y::Number) -> CartesianProduct{1, T}
    interval(X::CartesianProduct{1}) -> CartesianProduct{1, T}

Construct a 1D [`CartesianProduct`](@ref) representing the closed interval ``[x, y]``.

Inputs are converted to floating-point representation with promoted element type `T`.

# Arguments
- `x`: Lower endpoint.
- `y`: Upper endpoint.

# Throws
- `ArgumentError`: If `x > y`.

# Examples
```jldoctest
using Bramble
I = interval(0.0, 1.0)
first(I) == 0.0 && last(I) == 1.0

# output
true
```
"""
@inline function interval(x::Number, y::Number)
    T = promote_type(typeof(float(x)), typeof(float(y)))
    _x, _y = T(x), T(y)
    _x <= _y || _throw_interval_error(x, y)

    _is_collapsed = is_collapsed(_x, _y)
    box = SVector{1, Tuple{T, T}}(((_x, _y),))
    collapsed = SVector{1, Bool}((_is_collapsed,))
    return CartesianProduct{1, T}(box, collapsed)
end

@inline interval(x::CartesianProduct{1}) = interval(x(1)...)

"""
    point(x::Number) -> CartesianProduct{1, T}

Construct a degenerate 1D [`CartesianProduct`](@ref) representing the point ``[x, x]`` with collapsed status `true`.
"""
@inline function point(x::Number)
    _x = float(x)
    box = SVector{1, Tuple{typeof(_x), typeof(_x)}}(((_x, _x),))
    collapsed = SVector{1, Bool}((true,))
    return CartesianProduct{1, typeof(_x)}(box, collapsed)
end

"""
    cartesian_product(x::Number, y::Number) -> CartesianProduct{1, T}
    cartesian_product(box::NTuple{D, Tuple{Any, Any}}) -> CartesianProduct{D, T}
    cartesian_product(X::CartesianProduct) -> CartesianProduct

Construct a [`CartesianProduct`](@ref) from scalar interval endpoints, a tuple of interval pairs, or an existing set.

# Throws
- `ArgumentError`: If any interval pair satisfies `pair[1] > pair[2]`.

# Examples
```jldoctest
using Bramble
X = cartesian_product(((0.0, 1.0), (0.0, 2.0)))
dim(X) == 2 && eltype(X) === Float64

# output
true
```
"""
@inline cartesian_product(x::Number, y::Number) = interval(x, y)

@inline function cartesian_product(box::NTuple{D, Tuple{Any, Any}}) where {D}
    all(i -> box[i][1] <= box[i][2], 1:D) || _throw_box_error(box)

    _box_f = ntuple(i -> (float(box[i][1]), float(box[i][2])), Val(D))
    FloatT = mapreduce(t -> promote_type(typeof(t[1]), typeof(t[2])), promote_type, _box_f)
    _box = ntuple(i -> (FloatT(_box_f[i][1]), FloatT(_box_f[i][2])), Val(D))
    _collapsed_flags = ntuple(i -> is_collapsed(_box[i]...), Val(D))

    return CartesianProduct{D, FloatT}(
        SVector{D, Tuple{FloatT, FloatT}}(_box), SVector{D, Bool}(_collapsed_flags))
end

@inline cartesian_product(X::CartesianProduct) = X

"""
    box(a::Number, b::Number) -> CartesianProduct{1, T}
    box(a::NTuple{D}, b::NTuple{D}) -> CartesianProduct{D, T}

Construct a [`CartesianProduct`](@ref) from two opposing corner points `a` and `b`.

Interval bounds for each dimension `i` are defined by ``[\\min(a_i, b_i), \\max(a_i, b_i)]``.
"""
@inline box(a::Number, b::Number) = interval(min(a, b), max(a, b))

@inline function box(a::NTuple{D}, b::NTuple{D}) where {D}
    box_coords = ntuple(i -> (float(min(a[i], b[i])), float(max(a[i], b[i]))), Val(D))
    collapsed_flags = ntuple(i -> is_collapsed(box_coords[i]...), Val(D))
    FloatT = typeof(box_coords[1][1])

    return CartesianProduct{D, FloatT}(
        SVector{D, Tuple{FloatT, FloatT}}(box_coords), SVector{D, Bool}(collapsed_flags))
end

"""
    center(X::CartesianProduct{D, T}) -> SVector{D, T}

Compute the geometric center point of [`CartesianProduct`](@ref) `X`.
"""
@inline function center(cp::CartesianProduct{D, T}) where {D, T}
    return SVector{D, T}(ntuple(i -> (cp.box[i][1] + cp.box[i][2]) * T(0.5), Val(D)))
end

"""
    (X::CartesianProduct{D, T})(i::Integer) -> Tuple{T, T}

Return the `i`-th component interval endpoints `(min, max)`.

# Throws
- `BoundsError`: If `i < 1` or `i > D`.
"""
@inline function (X::CartesianProduct{D})(i::Integer) where {D}
    @boundscheck (1 <= i <= D) || _throw_bounds_error(X, i)
    return @inbounds X.box[i]
end

"""
    eltype(X::CartesianProduct{D, T}) -> Type{T}
    eltype(::Type{<:CartesianProduct{D, T}}) -> Type{T}

Return the scalar coordinate type `T` of [`CartesianProduct`](@ref) `X`.
"""
@inline eltype(::CartesianProduct{D, T}) where {D, T} = T
@inline eltype(::Type{<:CartesianProduct{D, T}}) where {D, T} = T

"""
    dim(X::CartesianProduct{D}) -> Int
    dim(::Type{<:CartesianProduct{D}}) -> Int

Return the spatial embedding dimension `D` of [`CartesianProduct`](@ref) `X`.
"""
@inline dim(::CartesianProduct{D}) where {D} = D
@inline dim(::Type{<:CartesianProduct{D}}) where {D} = D

"""
    topo_dim(X::CartesianProduct{D}) -> Int

Return the topological dimension of `X`, defined as the embedding dimension `D` minus the number of collapsed dimensions.
"""
@inline topo_dim(X::CartesianProduct{D}) where {D} = D - sum(X.collapsed)

"""
    tails(X::CartesianProduct, i::Integer) -> Tuple{T, T}
    tails(X::CartesianProduct{1}) -> Tuple{T, T}
    tails(X::CartesianProduct{D}) -> NTuple{D, Tuple{T, T}}

Return component interval endpoint pairs `(min, max)` for index `i` or all dimensions.
"""
@inline tails(X::CartesianProduct, i::Integer) = X(i)
@inline tails(X::CartesianProduct{1}) = X(1)
@inline tails(X::CartesianProduct{D}) where {D} = ntuple(i -> X(i), Val(D))

"""
    first(X::CartesianProduct{1}) -> T
    last(X::CartesianProduct{1}) -> T

Return the lower (`first`) or upper (`last`) bound of a 1D [`CartesianProduct`](@ref).
"""
@inline first(X::CartesianProduct{1}) = first(X(1))
@inline last(X::CartesianProduct{1}) = last(X(1))

"""
    ×(X::CartesianProduct{D1}, Y::CartesianProduct{D2}) -> CartesianProduct{D1 + D2}

Compute the Cartesian tensor product of sets `X` and `Y`.

The resulting set has embedding dimension `D1 + D2` with promoted scalar coordinate type.
"""
@inline function ×(X::CartesianProduct{D1, T1}, Y::CartesianProduct{
        D2, T2}) where {D1, D2, T1, T2}
    D = D1 + D2
    T = promote_type(T1, T2)
    if T === T1 === T2
        new_box = vcat(X.box, Y.box)
    else
        new_box = SVector{D, Tuple{T, T}}(ntuple(
            i -> i <= D1 ? (T(X.box[i][1]), T(X.box[i][2])) :
                 (T(Y.box[i - D1][1]), T(Y.box[i - D1][2])),
            Val(D)))
    end
    new_collapsed = vcat(X.collapsed, Y.collapsed)

    return CartesianProduct{D, T}(new_box, new_collapsed)
end

"""
    projection(X::CartesianProduct, i::Integer) -> CartesianProduct{1}

Extract the `i`-th coordinate dimension of `X` as a 1D [`CartesianProduct`](@ref).
"""
@inline projection(X::CartesianProduct, i::Integer) = interval(X(i)...)

"""
    point_type(X::CartesianProduct{1, T}) -> Type{T}
    point_type(X::CartesianProduct{D, T}) -> Type{NTuple{D, T}}
    point_type(::Type{<:CartesianProduct{1, T}}) -> Type{T}
    point_type(::Type{<:CartesianProduct{D, T}}) -> Type{NTuple{D, T}}

Return the coordinate point representation type for a point in `X`.
"""
@inline point_type(::CartesianProduct{1, T}) where {T} = T
@inline point_type(::CartesianProduct{D, T}) where {D, T} = NTuple{D, T}
@inline point_type(::Type{<:CartesianProduct{1, T}}) where {T} = T
@inline point_type(::Type{<:CartesianProduct{D, T}}) where {D, T} = NTuple{D, T}

"""
    in(x, X::CartesianProduct) -> Bool

Query whether point `x` is contained in the closed set `X`.
"""
@inline Base.in(x::Number, X::CartesianProduct{1}) = (X.box[1][1] <= x <= X.box[1][2])
@inline Base.in(x::Tuple{Vararg{Number, D}}, X::CartesianProduct{D}) where {D} = all(
    i -> (X.box[i][1] <= x[i] <= X.box[i][2]), 1:D)

# Unrolled over `Val(D)` rather than reduced over `1:D` to prevent heap allocations
# when testing containment of an AbstractVector.
@inline function Base.in(x::AbstractVector{<:Number}, X::CartesianProduct{D}) where {D}
    length(x) == D || return false
    return all(ntuple(i -> (X.box[i][1] <= x[i] <= X.box[i][2]), Val(D)))
end
@inline Base.in(x, X::CartesianProduct) = false

function Base.show(io::IO, X::CartesianProduct{D, T}) where {D, T}
    pp = PrettyPrinter(io)

    if pp.compact
        if D == 1
            collapsed = X.collapsed[1]
            if collapsed
                print(io, "Point(", X.box[1][1], ")")
            else
                print(io, "[", X.box[1][1], ", ", X.box[1][2], "]")
            end
        else
            for i in 1:D
                i > 1 && print(io, " × ")
                if X.collapsed[i]
                    print(io, X.box[i][1])
                else
                    print(io, "[", X.box[i][1], ", ", X.box[i][2], "]")
                end
            end
        end
    else
        topodim = topo_dim(X)

        if D == 1
            collapsed = X.collapsed[1]
            print_colored(pp, "CartesianProduct{$D,$T}"; bold = true, color = :cyan)
            print(io, ": ")
            if collapsed
                print_colored(pp, "Point"; color = :yellow)
                print(io, " at ")
                print_value(pp, X.box[1][1])
            else
                print_colored(pp, "Interval"; color = :yellow)
                print(io, " ")
                print_interval(pp, X.box[1][1], X.box[1][2])
            end
        else
            print_colored(pp, "CartesianProduct{$D,$T}"; bold = true, color = :cyan)
            if topodim < D
                print_colored(pp, " (topological dim $topodim)"; color = :yellow)
            end
            println(io, ":")

            pp_indented = with_indent(pp, 1)
            for i in 1:D
                label = get_dimension_label(i)
                print_dimension_info(
                    pp_indented, label, X.box[i][1], X.box[i][2], X.collapsed[i])
            end

            remove_trailing_newline(io)
        end
    end
end
