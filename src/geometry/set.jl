"""
	$(TYPEDEF)

A type representing the Cartesian product of `D` closed intervals in a space with element type `T`.

# Fields

$(FIELDS)
"""
struct CartesianProduct{D, T}
    "a container of `D` pairs, where each pair represents the bounds `(min, max)` of an interval"
    box::SVector{D, Tuple{T, T}}
    "a container of `D` boolean values indicating whether each dimension is collapsed (i.e., min = max)"
    collapsed::SVector{D, Bool}
end

@noinline _throw_bounds_error(X::CartesianProduct, i) = throw(BoundsError(X, i))
@noinline _throw_interval_error(x,
    y) = throw(ArgumentError("Invalid interval: expected x <= y, but got x = $x, y = $y"))
@noinline _throw_box_error(box) = throw(ArgumentError("Invalid box: Each tuple must satisfy x[1] <= x[2]. Found box: $box"))

"""
	$(SIGNATURES)

Checks if a 1D [`CartesianProduct`](@ref) or two numbers are "collapsed" (i.e., degenerate).

- For two numbers `a` and `b`, returns `true` if `isapprox(a, b)`.
- For a 1-dimensional [`CartesianProduct`](@ref), returns `X.collapsed[1]`.
"""
@inline is_collapsed(a::T, b::T) where {T <: Number} = isapprox(a, b)
@inline is_collapsed(a::Number, b::Number) = isapprox(promote(a, b)...)
@inline is_collapsed(X::CartesianProduct{1}) = X.collapsed[1]

"""
	$(SIGNATURES)

Returns the [`CartesianProduct`](@ref) itself (identity operation).
"""
@inline set(X::CartesianProduct) = X

"""
	$(SIGNATURES)

Constructs a 1-dimensional [`CartesianProduct`](@ref) representing the closed interval ``[x, y]``.
Inputs are converted to floating-point numbers.
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
	$(SIGNATURES)

Creates a collapsed 1D [`CartesianProduct`](@ref) representing the point ``[x, x]``.
"""
@inline function point(x::Number)
    _x = float(x)
    box = SVector{1, Tuple{typeof(_x), typeof(_x)}}(((_x, _x),))
    collapsed = SVector{1, Bool}((true,))
    return CartesianProduct{1, typeof(_x)}(box, collapsed)
end

"""
	$(SIGNATURES)

Returns a [`CartesianProduct`](@ref) from coordinates or intervals.

- If two scalar values `x` and `y` are provided, returns a 1D [`CartesianProduct`](@ref) representing ``[x, y]``.
- If an `NTuple{D, Tuple{T,T}}` is provided, returns a `D`-dimensional [`CartesianProduct`](@ref).
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
	$(SIGNATURES)

Creates a [`CartesianProduct`](@ref) from two points `a` and `b`, defining opposite corners of the bounding box.
Intervals are constructed as ``[\\min(a_i, b_i), \\max(a_i, b_i)]``.
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
	$(SIGNATURES)

Returns the center point of a [`CartesianProduct`](@ref) domain as an `SVector{D,T}`.
"""
@inline function center(cp::CartesianProduct{D, T}) where {D, T}
    return SVector{D, T}(ntuple(i -> (cp.box[i][1] + cp.box[i][2]) * T(0.5), Val(D)))
end

"""
	(X::CartesianProduct)(i)

Returns the `i`-th interval (or point bounds) in the [`CartesianProduct`](@ref) as a `Tuple{T,T}`.
"""
@inline function (X::CartesianProduct{D})(i::Integer) where {D}
    @boundscheck (1 <= i <= D) || _throw_bounds_error(X, i)
    return @inbounds X.box[i]
end

"""
	$(SIGNATURES)

Returns the coordinate element type `T` of a [`CartesianProduct`](@ref).
"""
@inline eltype(::CartesianProduct{D, T}) where {D, T} = T
@inline eltype(::Type{<:CartesianProduct{D, T}}) where {D, T} = T

"""
	$(SIGNATURES)

Returns the spatial dimension `D` where the [`CartesianProduct`](@ref) is embedded.
"""
@inline dim(::CartesianProduct{D}) where {D} = D
@inline dim(::Type{<:CartesianProduct{D}}) where {D} = D

"""
	$(SIGNATURES)

Returns the topological dimension of a [`CartesianProduct`](@ref) (embedded dimension minus collapsed dimensions).
"""
@inline topo_dim(X::CartesianProduct{D}) where {D} = D - sum(X.collapsed)

"""
	$(SIGNATURES)

Returns the `i`-th component interval of [`CartesianProduct`](@ref) `X` as a `Tuple{T,T}`. Alias for `X(i)`.
"""
@inline tails(X::CartesianProduct, i::Integer) = X(i)

"""
	$(SIGNATURES)

Returns the component sets of a [`CartesianProduct`](@ref):
- for a 1D product (`D=1`), returns the single interval tuple `(min, max)`.
- for a `D`-dimensional product, returns an `NTuple{D, Tuple{T,T}}` of all component intervals.
"""
@inline tails(X::CartesianProduct{1}) = X(1)
@inline tails(X::CartesianProduct{D}) where {D} = ntuple(i -> X(i), Val(D))

"""
	$(SIGNATURES)

Return the lower (`first`) and upper (`last`) bounds of a one-dimensional [`CartesianProduct`](@ref).
"""
@inline first(X::CartesianProduct{1}) = first(X(1))
@inline last(X::CartesianProduct{1}) = last(X(1))

"""
	×(X::CartesianProduct, Y::CartesianProduct)

Computes the tensor product of two [`CartesianProduct`](@ref)s `X` and `Y`.
The resulting dimension is `dim(X) + dim(Y)` with promoted element type.
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
	$(SIGNATURES)

Returns the `i`-th component interval of `X` as a new 1D [`CartesianProduct`](@ref).
"""
@inline projection(X::CartesianProduct, i::Integer) = interval(X(i)...)

"""
	$(SIGNATURES)

Determines the coordinate point type within a [`CartesianProduct`](@ref) space.
Returns `T` for 1D spaces and `NTuple{D,T}` for `D`-dimensional spaces.
"""
@inline point_type(::CartesianProduct{1, T}) where {T} = T
@inline point_type(::CartesianProduct{D, T}) where {D, T} = NTuple{D, T}
@inline point_type(::Type{<:CartesianProduct{1, T}}) where {T} = T
@inline point_type(::Type{<:CartesianProduct{D, T}}) where {D, T} = NTuple{D, T}

"""
	Base.in(x, X::CartesianProduct)

Returns `true` if point `x` is contained in the closed [`CartesianProduct`](@ref) domain.
"""
@inline Base.in(x::Number, X::CartesianProduct{1}) = (X.box[1][1] <= x <= X.box[1][2])
@inline Base.in(x::Tuple{Vararg{Number, D}}, X::CartesianProduct{D}) where {D} = all(
    i -> (X.box[i][1] <= x[i] <= X.box[i][2]), 1:D)

# Unrolled over `Val(D)` rather than reduced over `1:D`. The tuple method above closes over
# a tuple and costs nothing, but closing over an `AbstractVector` that way allocated 80
# bytes per call, which is a heap allocation on every containment test against a vector.
@inline function Base.in(x::AbstractVector{<:Number}, X::CartesianProduct{D}) where {D}
    length(x) == D || return false
    return all(ntuple(i -> (X.box[i][1] <= x[i] <= X.box[i][2]), Val(D)))
end
@inline Base.in(x, X::CartesianProduct) = false

"""
	Base.show(io::IO, X::CartesianProduct)

Custom display for [`CartesianProduct`](@ref) objects, showing dimension information, bounds, and collapsed status with colors.
"""
function Base.show(io::IO, X::CartesianProduct{D, T}) where {D, T}
    pp = PrettyPrinter(io)

    if pp.compact
        # Compact display for arrays/collections
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
        # Detailed display
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
            # Header with topological dimension info
            print_colored(pp, "CartesianProduct{$D,$T}"; bold = true, color = :cyan)
            if topodim < D
                print_colored(pp, " (topological dim $topodim)"; color = :yellow)
            end
            println(io, ":")

            # Show each dimension
            pp_indented = with_indent(pp, 1)
            for i in 1:D
                label = get_dimension_label(i)
                print_dimension_info(
                    pp_indented, label, X.box[i][1], X.box[i][2], X.collapsed[i])
            end

            # Remove trailing newline
            remove_trailing_newline(io)
        end
    end
end
