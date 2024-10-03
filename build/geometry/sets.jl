"""
	struct CartesianProduct{D,T}
		data::NTuple{D,Tuple{T,T}}
	end

Type for storage of cartesian products of `D` intervals having elements of type `T`.
"""
struct CartesianProduct{D,T} <: BrambleType
	data::NTuple{D,Tuple{T,T}}
end

"""
	struct BrambleBareFunction{D,T,has_cart} 
		f_tuple::FunctionWrapper{T, Tuple{NTuple{D,T}}}
		f_cartesian::FunctionWrapper{T, Tuple{CartesianIndex{D}}}
	end

Structure to wrap around functions to make them more type agnostic. It uses `FunctionWrappers` to provide functions calculated on `Tuple`s and `CartesianIndices`. The type arguments are `D`, the dimension of the underlying domain in which the function is defined, `f_tuple`, a version of the function appliable to `Tuple`s and `f_cartesian`, a version of the function appliable to `CartesianIndices` (useful when dealing with meshes)
"""
struct BrambleBareFunction{D,T,has_cart}
	f_tuple::FunctionWrapper{T,Tuple{NTuple{D,T}}}
	f_cartesian::FunctionWrapper{T,Tuple{CartesianIndex{D}}}
end

"""
	interval(x, y)
	interval(x::CartesianProduct{1})

Returns a `1`-dimensional [CartesianProduct](@ref) representing the interval [`x`,`y`].

# Example

```@example
julia> interval(0, 1)
CartesianProduct{1,Float64}((0.0,1.0))
```
"""
@inline function interval(x, y)
	_x = float(x)
	_y = float(y)
	@assert _x <= _y

	return CartesianProduct{1,typeof(_x)}(((_x, _y),))
end

@inline interval(x::CartesianProduct{1}) = interval(x.data...)

"""
	cartesianproduct(x, y)
	cartesianproduct(data::NTuple)
	cartesianproduct(X::CartesianProduct)

Returns a `1`-dimensional [CartesianProduct](@ref) to represent the interval [`x`,`y`]. Alternatively, if `D` tuples are provided in `data`, it returns a `D`-dimensional [CartesianProduct](@ref).

# Example

```@example
julia> cartesianproduct(0, 1)
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]
```

```@example
julia> cartesianproduct(((0,1), (4,5)))
Type: Int64 
 Dim: 2 
 Set: [0, 1] × [4, 5]
```
"""
@inline cartesianproduct(x, y) = interval(x, y)
@inline cartesianproduct(X::CartesianProduct) = X

@inline function cartesianproduct(data::NTuple{D,Tuple{T,T}}) where {D,T}
	@assert all(x -> x[1] <= x[2], data)
	return CartesianProduct{D,T}(data)
end

"""
	(X::CartesianProduct)(i)

Returns the `i`-th interval in the [CartesianProduct](@ref).
"""
@inline function (X::CartesianProduct)(i)
	@assert i in eachindex(X.data)
	return X.data[i]
end

"""
	eltype(X::CartesianProduct)
	eltype(::Type{<:CartesianProduct})

Returns the element type of a [CartesianProduct](@ref).

# Example

```@example
julia> X = cartesianproduct(0, 1); eltype(X)
Float64
```
"""
@inline eltype(_::CartesianProduct{D,T}) where {D,T} = T
@inline eltype(::Type{<:CartesianProduct{D,T}}) where {D,T} = T

"""
	dim(X::CartesianProduct)
	dim(::Type{<:CartesianProduct})

Returns the topological dimension of a [CartesianProduct](@ref).

# Example

```@example
julia> X = cartesianproduct(0, 1); dim(X)
1
```

```@example
julia> Y = cartesianproduct(((0,1), (4,5))); dim(Y)
2
```
"""
@inline dim(_::CartesianProduct{D}) where D = D
@inline dim(::Type{<:CartesianProduct{D}}) where D = D

"""
	tails(X::CartesianProduct, i)
	tails(X::CartesianProduct{D})

Returns `i`-th interval in [CartesianProduct](@ref) `X` as a Tuple. It can also be called on `X, returning a `D`-tuple with all intervals defining [CartesianProduct](@ref) `X`.

# Example

```@example
julia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5); tails(X,1)
(0.0, 1.0)
```

```@example
julia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5); tails(X)
((0.0, 1.0), (4.0, 5.0))
```
"""
@inline function tails(X::CartesianProduct, i)
	@assert i in eachindex(X.data)
	return X(i)
end

@inline @generated tails(X::CartesianProduct{D}) where D = :(Base.Cartesian.@ntuple $D i->X(i))

@inline tails(X::CartesianProduct{1}) = X(1)

"""
	×(X::CartesianProduct, Y::CartesianProduct)

Returns the cartesian product of two [CartesianProduct](@ref)s `X` and `Y` as a new [CartesianProduct](@ref).

# Example

```@example
julia> X = interval(0, 1); Y = interval(2, 3); X × Y;
Type: Float64 
 Dim: 2 
 Set: [0.0, 1.0] × [2.0, 3.0]
```
"""
@inline function ×(X::CartesianProduct{D1,T}, Y::CartesianProduct{D2,T}) where {D1,D2,T}
	a = tails(X)
	b = tails(Y)
	c = tuple((a...)..., (b...)...)

	return CartesianProduct{D1 + D2,T}(ntuple(i -> (c[2 * i - 1], c[2 * i]), D1 + D2))
end

"""
	projection(X::CartesianProduct, i)

Returns the `i`-th interval in [CartesianProduct](@ref) `X` as a new `1`-dimensional [CartesianProduct](@ref).

# Example

```@example
julia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5); projection(X, 1)
Type: Float64 
 Dim: 1 
 Set: [0.0, 1.0]
```
"""
@inline projection(X::CartesianProduct, i) = interval(X(i)...)

function show(io::IO, X::CartesianProduct{D}) where D
	sets = ["[$(tails(X,i)[1]), $(tails(X,i)[2])]" for i in eachindex(X.data)]
	sets_string = join(sets, " × ")
	print(io, "Type: $(eltype(X)) \n Dim: $D \n Set: $sets_string")
end


"""
	@embed(X:CartesianProduct, f)
	@embed(X:Domain, f)
	@embed(X:MeshType, f)
	@embed(X:SpaceType, f)

Returns a new wrapped version of function `f`. If 
	
 - `X` is not a [SpaceType](@ref), the topological dimension of `X` is used to caracterize the types in the returning [BrambleBareFunction](@ref); in this case, the new function can be applied to tuples (with point coordinates) or cartesian indices (if `X` is a [MeshType](@ref))

 - `X` is a [SpaceType](@ref), the returning function type, [BrambleGridSpaceFunction](@ref), is caracterized by the dimension of the [element](@ref)s of the space.
 
# Example

```@example
julia> Ω = domain(interval(0,1) × interval(0,1)); f = @embed Ω x -> x[1]*x[2]+1;  # or f = @embed(Ω, x -> x[1]*x[2]+1)
```

"""
macro embed(domain_expr, func_expr)
    if func_expr isa Symbol || func_expr.head == :-> || func_expr.head == :.
        return esc(:(Bramble._embed($domain_expr, $func_expr)))
    end
    
    return :(@error "Don't know how to handle this expression")
end

function _embed(X::CartesianProduct{D,T}, f) where {D,T}
	wrapped_f_tuple = FunctionWrapper{T,Tuple{NTuple{D,T}}}(f)
	wrapped_f_cartesian = FunctionWrapper{T,Tuple{CartesianIndex{D}}}(zero)

	return BrambleBareFunction{D,T,false}(wrapped_f_tuple, wrapped_f_cartesian)
end

@inline _embed(X::CartesianProduct, f::BrambleBareFunction) = f

@inline (f::BrambleBareFunction{1,T})(x::Tin) where {T,Tin<:Number} = f.f_tuple((convert(T, x)::T,))
@inline (f::BrambleBareFunction{1,T})(x::Tuple{Tin}) where {T,Tin} = f.f_tuple(Tuple(convert(T, x[1])::T))
@inline (f::BrambleBareFunction{2,T})(x::Tuple{T1,T2}) where {T,T1,T2} = f.f_tuple(ntuple(i -> convert(T, x[i])::T, 2))
@inline (f::BrambleBareFunction{3,T})(x::Tuple{T1,T2,T3}) where {T,T1,T2,T3} = f.f_tuple(ntuple(i -> convert(T, x[i])::T, 3))

@inline (f::BrambleBareFunction{D,T,true})(x::CartesianIndex{D}) where {D,T} = f.f_cartesian(x)
@inline (f::BrambleBareFunction{D,T,false})(x::CartesianIndex{D}) where {D,T} = @error "This function doesn't isn't _embedded in a mesh"