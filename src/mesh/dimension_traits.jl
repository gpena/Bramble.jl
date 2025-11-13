"""
# dimension_traits.jl

This file defines dimension-related types and traits for compile-time dispatch
in mesh operations. This enables efficient specialization for 1D vs. multi-dimensional
cases without runtime overhead.

## Type System

- `Dimension`: Abstract base for dimension traits
- `OneDimensional`, `TwoDimensional`, `ThreeDimensional`: Specific dimension types
- `MultiDimensional`: Generic n-dimensional case

## Usage

These types are used internally by functions like `generate_indices` to select
the most efficient implementation based on dimensionality at compile time.
"""

"""
	Dimension

Abstract base type for dimension traits used in compile-time dispatch.

Concrete subtypes represent specific dimensional cases (1D, 2D, 3D) or
a generic multi-dimensional case.

See also: [`OneDimensional`](@ref), [`TwoDimensional`](@ref),
[`ThreeDimensional`](@ref), [`MultiDimensional`](@ref)
"""
abstract type Dimension end

"""
	OneDimensional <: Dimension

Dimension trait for one-dimensional (1D) operations.

Used for compile-time dispatch to specialized 1D implementations.
"""
struct OneDimensional <: Dimension end

"""
	TwoDimensional <: Dimension

Dimension trait for two-dimensional (2D) operations.

Used for compile-time dispatch to specialized 2D implementations.
"""
struct TwoDimensional <: Dimension end

"""
	ThreeDimensional <: Dimension

Dimension trait for three-dimensional (3D) operations.

Used for compile-time dispatch to specialized 3D implementations.
"""
struct ThreeDimensional <: Dimension end

"""
	MultiDimensional <: Dimension

Dimension trait for generic multi-dimensional operations.

Used when the exact dimensionality doesn't require special handling,
or for dimensions > 3.
"""
struct MultiDimensional <: Dimension end

"""
	dimension_one_or_all(::Type{T}) -> Dimension

Dispatch helper that returns `OneDimensional()` for scalar types (Int)
or `MultiDimensional()` for tuple/vector types.

This is used to efficiently handle both 1D and nD cases in a single function
by dispatching on the type of the input.

# Arguments

  - `::Type{T}`: Type to analyze (typically `Int`, `NTuple`, or `SVector`)

# Returns

  - `OneDimensional()` if `T <: Int`
  - `MultiDimensional()` if `T <: Union{NTuple, SVector}`

# Examples

```julia
dimension_one_or_all(Int)           # OneDimensional()
dimension_one_or_all(NTuple{3,Int}) # MultiDimensional()
```
"""
@inline dimension_one_or_all(::Type{<:Int}) = OneDimensional()
@inline dimension_one_or_all(::Type{<:Union{NTuple,SVector}}) = MultiDimensional()

"""
	dimension(::Type{<:NTuple{D}}) -> Dimension
	dimension(::Type{<:SVector{D}}) -> Dimension

Maps a tuple or SVector type to its corresponding dimension trait for compile-time dispatch.

Used internally for efficient method selection in `generate_indices` and similar functions.
Provides specialized types for 1D, 2D, and 3D cases.

# Arguments

  - `::Type{<:NTuple{D}}`: Type of tuple with D elements
  - `::Type{<:SVector{D}}`: Type of static vector with D elements

# Returns

  - `OneDimensional()` for D=1
  - `TwoDimensional()` for D=2
  - `ThreeDimensional()` for D=3

# Examples

```julia
dimension(NTuple{1,Int})    # OneDimensional()
dimension(SVector{2,Float64}) # TwoDimensional()
dimension(NTuple{3,Int})    # ThreeDimensional()
```
"""
@inline dimension(::Type{<:NTuple{1}}) = OneDimensional()
@inline dimension(::Type{<:NTuple{2}}) = TwoDimensional()
@inline dimension(::Type{<:NTuple{3}}) = ThreeDimensional()
@inline dimension(::Type{<:SVector{1}}) = OneDimensional()
@inline dimension(::Type{<:SVector{2}}) = TwoDimensional()
@inline dimension(::Type{<:SVector{3}}) = ThreeDimensional()
