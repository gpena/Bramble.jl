###############################################################
#                                                             #
#           Implementation of the average operators           #
#                                                             #
###############################################################

#=
# average.jl

This file implements averaging operators for staggered grid computations.

## Mathematical formulation

For a function uₕ on a grid, the average operator ⟨·⟩ computes the mean value between 
adjacent grid points:

**Forward average** (at point xᵢ):
	⟨u⟩ᵢᶠ = (uᵢ + uᵢ₊₁) / 2

**Backward average** (at point xᵢ):
	⟨u⟩ᵢᵇ = (uᵢ + uᵢ₋₁) / 2

At boundary points where no neighbor exists:
	⟨u⟩ᵢ = uᵢ / 2

## Use cases

Typical uses:
1. Staggered grids: transfer variables between cell centers and faces
2. Discontinuous Galerkin: compute interface values
3. Finite difference: approximate derivatives at intermediate points
4. Conservative schemes: maintain flux conservation

## Example

```julia
# Average velocity from cell centers to faces
u_face = ⟨u⟩ᶠ(Vₕ, dim)  # Forward average in dimension dim

# Average pressure from faces to centers  
p_center = ⟨p⟩ᵇ(Pₕ, dim)  # Backward average in dimension dim
```

## Implementation details

- Uses `@simd` for vectorization
- Separate loops for interior (2-point average) and boundary (1-point)
- Direction (Forward/Backward) controlled via trait dispatch
- Boundary handling: halve the value (maintains consistency with operator algebra)

See also: [`_compute_average`](@ref), [`add_half_shift`](@ref), [`Forward`](@ref), [`Backward`](@ref)
=#

# Both directions average a point with its neighbour, so one method covers them. The
# kernels take `(cur, other)` in that order, matching _compute_difference, and the
# engine below hands them the pair without knowing which way the stencil runs.
#
# Dividing by 2 rather than multiplying by 0.5 keeps the element type: a Float32 grid
# would otherwise be promoted through Float64 on every point.
@inline @propagate_inbounds _compute_average(
    ::GridDirection, ::Val{false}, cur, other) = (cur + other) / 2

# The one boundary slice has no neighbour, so the average is truncated to zero there.
# `zero(cur)` rather than a literal keeps the element type of the grid.
@inline @propagate_inbounds _compute_average(::GridDirection, ::Val{true}, cur) = zero(cur)

# The traversal is shared with the difference engine; see _stencil_ranges in
# operators/difference.jl.
function _average_engine!(out, in_ref, dims::NTuple{D, Int}, dir::GridDirection,
        ::Val{DIM}) where {D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, boundary = _stencil_ranges(axes(li), Val(DIM), dir)

    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        out[idx] = _compute_average(
            dir, Val(false), in_ref[idx], in_ref[li[_neighbour(dir, I, step)]])
    end

    @inbounds @simd for I in CartesianIndices(boundary)
        idx = li[I]
        out[idx] = _compute_average(dir, Val(true), in_ref[idx])
    end

    return nothing
end

# Divided by 2 rather than multiplied by 0.5, for the reason given at `_compute_average`:
# the literal is a Float64 and promotes the whole matrix. On a Float32 backend everything
# else in the library stayed Float32 and only the averaging matrices came back Float64.
function add_half_shift(Ωₕ::AbstractMeshType, ::Val{DIFF_DIM}, ::Val{first},
        ::Val{second}) where {DIFF_DIM, first, second}
    return (shift(Ωₕ, Val(DIFF_DIM), Val(first)) + shift(Ωₕ, Val(DIFF_DIM), Val(second))) /
           2
end

function _average_operator(Ωₕ::AbstractMeshType, ::Forward, ::Val{AVG_DIM}) where {AVG_DIM}
    return add_half_shift(Ωₕ, Val(AVG_DIM), Val(0), Val(1))
end

function _average_operator(Ωₕ::AbstractMeshType, ::Backward, ::Val{AVG_DIM}) where {AVG_DIM}
    return add_half_shift(Ωₕ, Val(AVG_DIM), Val(0), Val(-1))
end

function _average_weights!(v::AbstractVector, Ωₕ::AbstractMeshType,
        dir::GridDirection, ::Val{DIFF_DIM}) where {DIFF_DIM}
    dims = npoints(Ωₕ, Tuple)

    1 <= DIFF_DIM <= dim(Ωₕ) || _throw_stencil_dim_error(DIFF_DIM, dim(Ωₕ))

    li = LinearIndices(dims)

    @inbounds @simd for I in CartesianIndices(dims)
        idx = I[DIFF_DIM]
        if dir isa Forward
            v[li[I]] = idx == dims[DIFF_DIM] ? zero(eltype(v)) : one(eltype(v))
        else # Backward
            v[li[I]] = idx == 1 ? zero(eltype(v)) : one(eltype(v))
        end
    end
    return
end

# Configuration array for average operators, expanded with descriptive strings.
const _AVERAGE_OP_CONFIGS = [
    (direction = Forward(),
        average_name = :forward_average,
        average_alias = :M₊,
        vectorial_average_alias = :M₊ₕ,
        dir_string_lowercase = "forward",
        math_op = "\\frac{u_{i} + u_{i+1}}{2}"),
    (direction = Backward(),
        average_name = :backward_average,
        average_alias = :M₋,
        vectorial_average_alias = :M₋ₕ,
        dir_string_lowercase = "backward",
        math_op = "\\frac{u_{i-1} + u_{i}}{2}")
]

# Metaprogramming loop to generate all specified average operators.
for config in _AVERAGE_OP_CONFIGS
    # Extract ALL values from `config` to avoid scope issues with @eval.
    dir_instance = config.direction
    average_name = config.average_name
    average_name! = Symbol(average_name, :!)
    average_alias = config.average_alias
    vectorial_average_alias = config.vectorial_average_alias
    dir_string_lowercase = config.dir_string_lowercase
    math_op = config.math_op

    @eval begin
        # --- In-place applicators ---
        @doc """
          	$($(QuoteNode(Symbol(average_name, :_dim!))))(out, in, dims, average_dim)

          Low-level, in-place function to compute the $($dir_string_lowercase) average of vector `in` along dimension `average_dim`, storing the result in `out`. This function computes ``$($math_op)``.
          """
        function $(Symbol(average_name, :_dim!))(out, in, h, dims::NTuple{D, Int},
                average_dim::Val{DIFF_DIM}) where {D, DIFF_DIM}
            1 <= DIFF_DIM <= D || _throw_stencil_dim_error(DIFF_DIM, D)
            length(out) == length(in) == prod(dims) ||
                _throw_stencil_size_error(length(out), length(in), dims)
            in_ref = (out === in) ? copy(in) : in
            _average_engine!(out, in_ref, dims, $dir_instance, average_dim)
            return
        end

        function $(Symbol(average_name, :_dim!))(out, in, dims::NTuple{D, Int},
                average_dim::Val{DIFF_DIM}) where {D, DIFF_DIM}
            return $(Symbol(average_name, :_dim!))(out, in, nothing, dims, average_dim)
        end

        # --- Matrix operator functions ---
        @doc """
          	$($(QuoteNode(average_name)))(arg, dim_val::Val)

          Constructs or applies the $($dir_string_lowercase) averaging operator, representing the operation ``$($math_op)``.
          """
        @inline function $average_name(Ωₕ::AbstractMeshType, dim_val::Val; vector_cache = __vector(Ωₕ))
            avg_matrix = _average_operator(Ωₕ, $dir_instance, dim_val)
            _average_weights!(vector_cache, Ωₕ, $dir_instance, dim_val)
            return vector_cache .* avg_matrix
        end

        #@inline function $average_name(Ωₕ::AbstractMeshType, dim_val::Val)
        #avg_matrix = _average_operator(Ωₕ, $dir_instance, dim_val)
        #_average_weights!(vector_cache, Ωₕ, spacing_func, dim_val)
        #return Diagonal(vector_cache) * avg_matrix
        #	return _average_operator(Ωₕ, $dir_instance, dim_val)
        #end

        # --- Generic applicators ---
        #
        # As for the differences, the in-place forms hold the work and the allocating ones
        # are one line each on top of them.
        @inline $average_name(Wₕ::AbstractSpaceType, dim_val::Val) = $average_name(
            mesh(Wₕ), dim_val)

        function $average_name!(vₕ::VectorElement{<:ScalarGridSpace},
                uₕ::VectorElement{<:ScalarGridSpace}, dim_val::Val)
            _average_engine!(vₕ.data, uₕ.data, _grid_dims(uₕ), $dir_instance, dim_val)
            return vₕ
        end

        # A composite grid function is averaged one component at a time.
        function $average_name!(vₕ::VectorElement{<:CompositeGridSpace},
                uₕ::VectorElement{<:CompositeGridSpace}, dim_val::Val)
            _apply_componentwise!(
                (v, u) -> _average_engine!(
                    v.data, u.data, _grid_dims(u), $dir_instance, dim_val),
                vₕ, uₕ)
            return vₕ
        end

        @inline $average_name(uₕ::VectorElement, dim_val::Val) = $average_name!(
            similar(uₕ), uₕ, dim_val)
    end

    # --- Aliases for x, y, z directions ---
    # Generate aliases and their docstrings via the shared helper.
    for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
        direction = _BRAMBLE_var2label[i]
        _define_directional_alias(average_name, Symbol(average_alias, suffix),
            dir_string_lowercase, direction, i, "average", math_op)
        _define_directional_alias!(average_name!, Symbol(average_alias, suffix, :!),
            dir_string_lowercase, direction, i, "average", math_op)
    end

    # --- Alias for the vectorial average tuple ---
    _define_vectorial_alias(average_name, vectorial_average_alias, dir_string_lowercase,
        "average")
end
