##############################################################################
#                                                                            #
#             Implementation of (Finite) Difference Operators                #
#                                                                            #
##############################################################################

#=
# difference.jl

This file implements difference and finite difference operators for grid functions.

## Mathematical Formulation

### Simple Difference Operators (no grid spacing)

**Forward difference**:
	Δ₊uᵢ = uᵢ₊₁ - uᵢ

**Backward difference**:
	Δ₋uᵢ = uᵢ - uᵢ₋₁

### Finite Difference Operators (with grid spacing h)

**Forward finite difference** (approximates ∂u/∂x at xᵢ):
	δ₊uᵢ = (uᵢ₊₁ - uᵢ) / hᵢ

**Backward finite difference** (approximates ∂u/∂x at xᵢ):
	δ₋uᵢ = (uᵢ - uᵢ₋₁) / hᵢ

## Boundary Treatment

At domain boundaries where neighbors don't exist:
- **Forward at last point**: Δ₊uₙ = -uₙ (enforces zero beyond boundary)
- **Backward at first point**: Δ₋u₁ = u₁ (enforces zero before boundary)

This convention:
1. Maintains operator size consistency
2. Respects homogeneous Dirichlet-like conditions
3. Ensures matrix operators remain well-defined

## Grid Spacing Support

The operators support:
- **Uniform grids**: `h` is a scalar or nothing
- **Non-uniform grids**: `h` is a vector of local spacings
- **Adaptive spacing**: `h` is a function `h(i)` returning spacing at index i

## Use Cases

**Simple differences**: Measure changes without physical units
```julia
Δu = Δ₊(uₕ, dim)  # Dimensionless change
```

**Finite differences**: Approximate derivatives with physical meaning
```julia
∂u_∂x = δ₊(uₕ, dim, mesh)  # Has units of [u]/[x]
```

## Performance Optimizations

- `@propagate_inbounds`: Eliminates bounds checking in inner loops
- `@simd`: Enables SIMD vectorization
- `@muladd`: Fuses multiply-add operations (a-b)/h → (a-b)*inv_h
- Separate loops for interior (2-point stencil) and boundary (1-point)

## Accuracy

These are first-order accurate methods:
- Truncation error: O(h) for first derivatives
- For higher accuracy, see centered differences or higher-order stencils

See also: [`Δ₊`](@ref), [`Δ₋`](@ref), [`δ₊`](@ref), [`δ₋`](@ref), [`Forward`](@ref), [`Backward`](@ref)
=#

# --- Type System for Dispatch ---
abstract type GridDirection end
struct Forward <: GridDirection end
struct Backward <: GridDirection end

# --- Core Difference Computation ---
# @boundscheck rather than @assert: this runs once per grid point and the engine's
# loops are marked @inbounds, which elides the former and cannot elide the latter.
@inline function _get_h_val(h::AbstractVector, i::Int)
    @boundscheck 1 <= i <= length(h) ||
                 throw(BoundsError(h, i))
    return @inbounds h[i]
end
@inline _get_h_val(h::F, i::Int) where {F <: Function} = h(i)

# The kernels take the point and its neighbour in that order, whichever direction the
# stencil runs, so that the engine can hand them `(cur, other)` without knowing which
# is which. `Val{false}` is an interior point, which has a neighbour; `Val{true}` is
# the one boundary slice that does not, where the stencil is truncated.
#
# Unscaled, h === nothing: the plain difference.
@inline @propagate_inbounds _compute_difference(
    ::Forward, ::Val{false}, cur, other, ::Nothing, i) = other - cur
@inline @propagate_inbounds _compute_difference(
    ::Backward, ::Val{false}, cur, other, ::Nothing, i) = cur - other
@inline @propagate_inbounds _compute_difference(
    ::Forward, ::Val{true}, cur, ::Nothing, i) = -cur
@inline @propagate_inbounds _compute_difference(
    ::Backward, ::Val{true}, cur, ::Nothing, i) = cur

# Scaled by the grid spacing: the finite difference.
@inline @propagate_inbounds _compute_difference(
    ::Forward, ::Val{false}, cur, other, h, i) = (other - cur) / _get_h_val(h, i)
@inline @propagate_inbounds _compute_difference(
    ::Backward, ::Val{false}, cur, other, h, i) = (cur - other) / _get_h_val(h, i)

# The finite difference has no one-sided stencil at the boundary, so it is zero there.
# `zero(cur)` rather than a literal keeps the element type of the grid.
@inline @propagate_inbounds _compute_difference(
    ::Forward, ::Val{true}, cur, h, i) = zero(cur)
@inline @propagate_inbounds _compute_difference(
    ::Backward, ::Val{true}, cur, h, i) = zero(cur)

# --- The starred forward difference ----------------------------------------------- #
#
#   Dstar₊(uₕ)(i) = (u(xᵢ₊₁) - u(xᵢ)) / ((hᵢ + hᵢ₊₁) / 2)
#
# The forward difference divided by the averaged spacing rather than by the forward
# spacing. Away from the boundary that denominator is the width of the cell around xᵢ,
# so this and `D₊` differ only in what they scale by, but the averaged form is what makes
# the discrete integration-by-parts identity close.
#
# The denominator is read lazily off the mesh's cached spacings rather than stored: entry
# i needs `spacings[i]` and `spacings[i+1]`, so a vector of its own would be a third copy
# of the axis to keep in step with refinement, for two loads it already has.

"""
	StarSpacings(h)

Lazy view of the averaged spacings ``(h_i + h_{i+1})/2`` over a mesh's cached backward
spacings `h`, which is what [`Dstar₊ₓ`](@ref) divides by.

Entry `i` reads `h[i]` and `h[i+1]`, so it is defined for `i < length(h)`. That is exactly
the range the forward stencil's interior covers; the last point has no forward neighbour
and the engine truncates it to zero without consulting this.
"""
struct StarSpacings{T, V <: AbstractVector{T}} <: AbstractVector{T}
    h::V
end

@inline Base.size(s::StarSpacings) = (length(s.h) - 1,)
@inline Base.@propagate_inbounds function Base.getindex(s::StarSpacings, i::Int)
    @boundscheck 1 <= i < length(s.h) || throw(BoundsError(s, i))
    return @inbounds (s.h[i] + s.h[i + 1]) / 2
end

"""
	star_spacings(Ωₕ::Mesh1D)

Returns the averaged spacings ``(h_i + h_{i+1})/2`` of `Ωₕ` as a [`StarSpacings`](@ref)
view over its cached backward spacings. Allocates nothing.

Away from the first point this equals [`half_spacing`](@ref)`(Ωₕ, i)`. At `i = 1` it does
not: the cached `h₁` repeats the first interval, so this gives ``x_2 - x_1`` where the cell
width gives half of it, the boundary cell being a half cell.
"""
@inline star_spacings(Ωₕ::Mesh1D) = StarSpacings(spacings(Ωₕ))

# --- Argument validation shared by every operator --------------------------------- #
# Thrown rather than asserted: these check caller arguments, and an @assert reports a
# size mismatch as an AssertionError, which is not what a caller should have to catch.
@noinline _throw_stencil_dim_error(dim::Int,
    D::Int) = throw(ArgumentError("the stencil direction must be between 1 and $D, got $dim"))

@noinline function _throw_stencil_size_error(lout::Int, lin::Int, dims)
    throw(DimensionMismatch("out has $lout entries and in has $lin, but the grid $(dims) has $(prod(dims))"))
end

# --- Argument handling shared by every operator ---------------------------------- #
# The operators accept a mesh, a grid space or a grid function, and the vectorial aliases
# need the spatial dimension of whichever was passed. Going through `space` alone would
# reject a mesh, which the scalar aliases do accept.
@inline _op_mesh(Ωₕ::AbstractMeshType) = Ωₕ
@inline _op_mesh(Wₕ::AbstractSpaceType) = mesh(Wₕ)
@inline _op_mesh(uₕ::VectorElement) = mesh(space(uₕ))

# A composite grid function is a stack of scalar ones, so an operator applies to each
# component in turn. Their `components` are views onto the parent, so writing into the
# components of `similar(uₕ)` fills it.
#
# The grid shape has to come from the mesh. `ndofs(space, Tuple)` gives it for a scalar
# space but gives the per-component dof counts for a composite one, so using it here
# addressed prod(ndofs) slots into a vector holding ndofs of them: a 3-component 4x6
# space addressed 13824 slots into 72, which segfaulted under the engines' @inbounds.
@inline _grid_dims(uₕ::VectorElement) = npoints(_op_mesh(uₕ), Tuple)

@inline function _apply_stencil!(vₕ::VectorElement{<:ScalarGridSpace},
        uₕ::VectorElement{<:ScalarGridSpace}, h, dir::GridDirection, dim_val::Val)
    _difference_engine!(vₕ.data, uₕ.data, h, _grid_dims(uₕ), dir, dim_val)
    return nothing
end

# `f!` is the single-component applicator; it is called once per component.
@inline function _apply_componentwise!(f!, vₕ::VectorElement{<:CompositeGridSpace{NC}},
        uₕ::VectorElement{<:CompositeGridSpace{NC}}) where {NC}
    vc, uc = components(vₕ), components(uₕ)
    ntuple(i -> (f!(vc[i], uc[i]); nothing), Val(NC))
    return nothing
end

# --- Shared stencil traversal --------------------------------------------------- #
# The difference and the average walk the grid identically: one pass over the interior,
# where every point has a neighbour along the stencil direction, and one over the single
# boundary slice, where it does not. Only the per-point kernel differs, so the traversal
# is written once here and both engines below use it.

# The unit step along `DIM`.
@inline _stencil_step(
    ::Val{DIM}, ::Val{D}) where {
    DIM, D} = CartesianIndex(ntuple(
    i -> i == DIM ? 1 : 0, Val(D)))

# The neighbour of `I`: ahead of it for a forward stencil, behind it for a backward one.
@inline _neighbour(::Forward, I, step) = I + step
@inline _neighbour(::Backward, I, step) = I - step

# The interior and boundary index ranges, as tuples of ranges to build
# `CartesianIndices` from. A forward stencil reaches past the last slice along `DIM`, a
# backward one past the first.
@inline function _stencil_ranges(full_axes::NTuple{D, Any}, ::Val{DIM}, ::Forward) where {
        D, DIM}
    interior = ntuple(
        d -> d == DIM ? (first(full_axes[d]):(last(full_axes[d]) - 1)) : full_axes[d],
        Val(D))
    boundary = ntuple(
        d -> d == DIM ? (last(full_axes[d]):last(full_axes[d])) : full_axes[d], Val(D))
    return interior, boundary
end

@inline function _stencil_ranges(full_axes::NTuple{D, Any}, ::Val{DIM}, ::Backward) where {
        D, DIM}
    interior = ntuple(
        d -> d == DIM ? ((first(full_axes[d]) + 1):last(full_axes[d])) : full_axes[d],
        Val(D))
    boundary = ntuple(
        d -> d == DIM ? (first(full_axes[d]):first(full_axes[d])) : full_axes[d], Val(D))
    return interior, boundary
end

# --- Unified Difference Engine ---
# `h` carries a type parameter on purpose. Julia does not specialise on an argument of
# function type unless the body calls it directly, and this body only forwards it to
# _get_h_val. Without `H` the spacing callable stays boxed and each element pays a
# dynamic dispatch: measured 13768 us and 6.4 MB against 29 us and no allocation on a
# 100000-point 1D grid.
function _difference_engine!(out, in_ref, h::H, dims::NTuple{D, Int},
        dir::GridDirection, ::Val{DIM}) where {H, D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, boundary = _stencil_ranges(axes(li), Val(DIM), dir)

    @inbounds @simd for I in CartesianIndices(interior)
        idx, other = li[I], li[_neighbour(dir, I, step)]
        out[idx] = _compute_difference(
            dir, Val(false), in_ref[idx], in_ref[other], h, I[DIM])
    end

    @inbounds @simd for I in CartesianIndices(boundary)
        idx = li[I]
        out[idx] = _compute_difference(dir, Val(true), in_ref[idx], h, I[DIM])
    end

    return nothing
end

function difference_shift(Ωₕ::AbstractMeshType, ::Val{DIFF_DIM}, ::Val{first},
        ::Val{second}) where {DIFF_DIM, first, second}
    return shift(Ωₕ, Val(DIFF_DIM), Val(first)) - shift(Ωₕ, Val(DIFF_DIM), Val(second))
end

function _difference_operator(Ωₕ::AbstractMeshType, ::Forward, ::Val{DIFF_DIM}) where {DIFF_DIM}
    return difference_shift(Ωₕ, Val(DIFF_DIM), Val(1), Val(0))
end

function _difference_operator(Ωₕ::AbstractMeshType, ::Backward, ::Val{DIFF_DIM}) where {DIFF_DIM}
    return difference_shift(Ωₕ, Val(DIFF_DIM), Val(0), Val(-1))
end

# `spacing_func` carries a type parameter so Julia specialises on it. An argument of
# function type is not specialised on unless the body calls it directly, and this one used
# to be wrapped in a `Base.Fix1` instead, so the closure stayed boxed and every grid point
# paid a dynamic dispatch: 12047 us and 6.5 MB on a 450x450 grid against 322 us and no
# allocation once the parameter is named and the submesh is hoisted.
#
# It is the per-index function rather than the mesh's cached spacing vector on purpose.
# The vector's truncated entry is not meaningful, whereas `spacing_for_derivative` returns
# zero there, and this loop covers every index and turns that zero into a zero weight. The
# engines can use the vector because they visit the truncated slice separately.
function _derivative_weights!(v::AbstractVector, Ωₕ::AbstractMeshType,
        spacing_func::F, ::Val{DIFF_DIM}) where {F, DIFF_DIM}
    dims = npoints(Ωₕ, Tuple)

    1 <= DIFF_DIM <= dim(Ωₕ) || _throw_stencil_dim_error(DIFF_DIM, dim(Ωₕ))

    sub = Ωₕ(DIFF_DIM)
    li = LinearIndices(dims)

    @inbounds @simd for I in CartesianIndices(dims)
        x = spacing_func(sub, I[DIFF_DIM])
        v[li[I]] = iszero(x) ? zero(eltype(v)) : inv(x)
    end
    return nothing
end

"""
	_define_directional_alias(base_op_name, alias_name, dir_string, suffix,
	                          direction_index, what, formula)

Defines `alias_name(arg)` as `base_op_name(arg, Val(direction_index))` and attaches a
docstring to it.

`what` names the quantity, such as `"finite difference"`, and `formula` is the LaTeX for
it. Both are needed because the four operator families share this generator: describing
every alias as a "difference" would be wrong for the averages and the jumps, and would
not separate the unscaled difference from the finite difference.
"""

function _define_directional_alias(
        base_op_name, alias_name, dir_string, suffix, direction_index, what, formula)
    # 1. Construct the docstring content.
    doc_string = """
     	$alias_name(arg)

     The `$dir_string` $what along the `$suffix` direction, ``$formula``. The unscaled
     difference is not divided by the grid spacing; the finite difference is.

     Alias for `$base_op_name(arg, Val($direction_index))`. `arg` is a mesh, a grid space
     or a [`VectorElement`](@ref): the first two give the operator as a sparse matrix, the
     third applies it and returns a `VectorElement`.
     """

    # 2. Construct the function definition as an expression.
    func_def_expr = :(@inline $(alias_name)(arg) = $(base_op_name)(arg, Val($(direction_index))))

    # 3. Combine them using the @doc macro syntax into a final expression.
    #    The `__source__` variable is replaced with `nothing`.
    final_expr = Expr(
        :macrocall, GlobalRef(Core, Symbol("@doc")), nothing, doc_string, func_def_expr)

    # 4. Evaluate the final, complete expression in the module's global scope.
    Core.eval(@__MODULE__, final_expr)
end

"""
	_define_vectorial_alias(base_op_name, alias_name, dir_string, what)

Defines the `ₕ` alias that applies `base_op_name` along every coordinate and returns a
tuple, one entry per spatial dimension. On a one-dimensional mesh it returns that single
entry rather than a one-tuple.

The counterpart of `_define_directional_alias` for the tuple-valued aliases
(`∇₋ₕ`, `diff₋ₕ`, `jump₋ₕ`, `M₋ₕ`). The three operator families generated the same three
methods independently before this existed.
"""
function _define_vectorial_alias(base_op_name, alias_name, dir_string, what)
    doc_string = """
     	$alias_name(arg)

     The $dir_string $what of `arg` along every coordinate, as a tuple with one entry per
     spatial dimension. On a one-dimensional mesh it returns that single entry rather than
     a one-tuple.

     For a 2D space, `$alias_name(uₕ)` is
     `($base_op_name(uₕ, Val(1)), $base_op_name(uₕ, Val(2)))`. `arg` is a mesh, a grid
     space or a [`VectorElement`](@ref), as for `$base_op_name`.
     """

    # Built one at a time, as in _define_directional_alias: @doc takes a single
    # definition, and only the entry point carries the docstring.
    entry = :(@inline $(alias_name)(arg) = $(alias_name)(arg, Val(dim(_op_mesh(arg)))))
    one_d = :(@inline $(alias_name)(arg, ::Val{1}) = $(base_op_name)(arg, Val(1)))
    n_d = :(@inline $(alias_name)(arg, ::Val{D}) where {D} = ntuple(
        i -> $(base_op_name)(arg, Val(i)), Val(D)))

    Core.eval(@__MODULE__,
        Expr(:macrocall, GlobalRef(Core, Symbol("@doc")), nothing, doc_string, entry))
    Core.eval(@__MODULE__, one_d)
    Core.eval(@__MODULE__, n_d)
end

# Configuration array to define forward and backward difference operators.
const _DIFFERENCE_OP_CONFIGS = [
    (direction = Forward(),
        diff_name = :forward_difference,
        finite_diff_name = :forward_finite_difference,
        weights_func! = :forward_derivative_weights!,
        spacing_func = :forward_spacing_for_derivative,
        spacings_func = :forward_spacings_for_derivative,
        diff_alias = :diff₊,
        finite_diff_alias = :D₊,
        grad_alias = :diff₊ₕ,
        finite_grad_alias = :∇₊ₕ,
        dir_string = "Forward",
        dir_string_lowercase = "forward",
        math_op = "u_{i+1} - u_i", math_finite_op = "\\frac{u_{i+1} - u_i}{h_i}"),
    (direction = Backward(),
        diff_name = :backward_difference,
        finite_diff_name = :backward_finite_difference,
        weights_func! = :backward_derivative_weights!,
        spacing_func = :spacing_for_derivative,
        spacings_func = :backward_spacings_for_derivative,
        diff_alias = :diff₋,
        finite_diff_alias = :D₋,
        grad_alias = :diff₋ₕ,
        finite_grad_alias = :∇₋ₕ,
        dir_string = "Backward",
        dir_string_lowercase = "backward",
        math_op = "u_{i} - u_{i-1}", math_finite_op = "\\frac{u_{i} - u_{i-1}}{h_i}")
]

# Metaprogramming loop to generate all specified difference operators.
for config in _DIFFERENCE_OP_CONFIGS
    # Extract ALL values from `config` into local variables here.
    dir_instance = config.direction
    diff_name = config.diff_name
    finite_diff_name = config.finite_diff_name
    weights_func! = config.weights_func!
    spacing_func = config.spacing_func
    spacings_func = config.spacings_func
    diff_alias = config.diff_alias
    finite_diff_alias = config.finite_diff_alias
    grad_alias = config.grad_alias
    finite_grad_alias = config.finite_grad_alias
    dir_string = config.dir_string
    dir_string_lowercase = config.dir_string_lowercase
    math_op = config.math_op
    math_finite_op = config.math_finite_op

    # This first @eval block is fine because it doesn't depend on any inner loops.
    @eval begin
        # --- In-place applicators ---
        @doc """
          	$($(QuoteNode(Symbol(diff_name, :_dim!))))(out, in, [h], dims, diff_dim)

          Low-level, in-place function to compute the **unscaled** $($dir_string_lowercase) difference of vector `in` along dimension `diff_dim`, storing the result in `out`. This function computes ``$($math_op)``.
          """
        function $(Symbol(diff_name, :_dim!))(out, in, h, dims::NTuple{D, Int},
                diff_dim::Val{DIFF_DIM}) where {D, DIFF_DIM}
            1 <= DIFF_DIM <= D || _throw_stencil_dim_error(DIFF_DIM, D)
            length(out) == length(in) == prod(dims) ||
                _throw_stencil_size_error(length(out), length(in), dims)
            in_ref = (out === in) ? copy(in) : in
            _difference_engine!(out, in_ref, h, dims, $dir_instance, diff_dim)
            return
        end

        function $(Symbol(diff_name, :_dim!))(
                out, in, dims::NTuple{D, Int}, diff_dim::Val{DIFF_DIM}) where {D, DIFF_DIM}
            return $(Symbol(diff_name, :_dim!))(out, in, nothing, dims, diff_dim)
        end

        # --- Weight calculation function ---
        @doc """
          	$($(QuoteNode(weights_func!)))(v::AbstractVector, Ωₕ::AbstractMeshType, diff_dim::Val)

          Computes the geometric weights for the $($dir_string_lowercase) finite difference operator and stores them in-place in vector `v`.
          """
        @inline function $weights_func!(v::AbstractVector, Ωₕ::AbstractMeshType, diff_dim::Val)
            _derivative_weights!(v, Ωₕ, $spacing_func, diff_dim)
        end

        # --- Matrix operator functions ---
        @doc """
          	$($(QuoteNode(diff_name)))(arg, dim_val::Val)

          Constructs the **unscaled** $($dir_string_lowercase) difference operator, representing the operation ``$($math_op)``.
          """
        @inline $diff_name(
            Ωₕ::AbstractMeshType, dim_val::Val) = _difference_operator(
            Ωₕ, $dir_instance, dim_val)

        @doc """
          	$($(QuoteNode(finite_diff_name)))(arg, dim_val::Val)

          Constructs the $($dir_string_lowercase) **finite difference** operator, which approximates the first derivative using the formula ``$($math_finite_op)``.
          """
        function $finite_diff_name(Ωₕ::AbstractMeshType, dim_val::Val; vector_cache = __vector(Ωₕ))
            diff_matrix = $diff_name(Ωₕ, dim_val)
            $weights_func!(vector_cache, Ωₕ, dim_val)
            return Diagonal(vector_cache) * diff_matrix
        end

        # --- Generic applicators ---
        @inline $diff_name(Wₕ::AbstractSpaceType, dim_val::Val) = $diff_name(
            mesh(Wₕ), dim_val)
        function $diff_name(uₕ::VectorElement{<:ScalarGridSpace}, dim_val::Val)
            vₕ = similar(uₕ)
            _apply_stencil!(vₕ, uₕ, nothing, $dir_instance, dim_val)
            return vₕ
        end

        # A composite grid function is differenced one component at a time.
        function $diff_name(uₕ::VectorElement{<:CompositeGridSpace}, dim_val::Val)
            vₕ = similar(uₕ)
            _apply_componentwise!(
                (v, u) -> _apply_stencil!(v, u, nothing, $dir_instance, dim_val), vₕ, uₕ)
            return vₕ
        end
        @inline $finite_diff_name(Wₕ::AbstractSpaceType, dim_val::Val) = $finite_diff_name(
            mesh(Wₕ), dim_val)
        function $finite_diff_name(
                uₕ::VectorElement{<:ScalarGridSpace}, dim_val::Val{DIM}) where {DIM}
            vₕ = similar(uₕ)
            # The mesh caches its spacings, so hand the engine that vector rather
            # than a callable: indexing it is 3.6x faster than one call per grid
            # point, and it needs no allocation of its own.
            h = $spacings_func(_op_mesh(uₕ)(DIM))
            _apply_stencil!(vₕ, uₕ, h, $dir_instance, dim_val)
            return vₕ
        end

        # Every component shares the mesh, so the spacings are fetched once.
        function $finite_diff_name(
                uₕ::VectorElement{<:CompositeGridSpace}, dim_val::Val{DIM}) where {DIM}
            vₕ = similar(uₕ)
            h = $spacings_func(_op_mesh(uₕ)(DIM))
            _apply_componentwise!(
                (v, u) -> _apply_stencil!(v, u, h, $dir_instance, dim_val), vₕ, uₕ)
            return vₕ
        end
    end

    # --- Aliases for x, y, z directions ---
    # ❗️ FIX: Call the helper function to generate the aliases safely.
    for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
        direction = _BRAMBLE_var2label[i]
        _define_directional_alias(diff_name, Symbol(diff_alias, suffix),
            dir_string_lowercase, direction, i, "unscaled difference", math_op)
        _define_directional_alias(finite_diff_name, Symbol(finite_diff_alias, suffix),
            dir_string_lowercase, direction, i, "finite difference", math_finite_op)
    end

    # --- Aliases for gradient tuples ---
    _define_vectorial_alias(diff_name, grad_alias, dir_string_lowercase,
        "unscaled difference")
    _define_vectorial_alias(finite_diff_name, finite_grad_alias, dir_string_lowercase,
        "finite difference")
end

# --- Dstar₊: the forward difference over the averaged spacing ---------------------- #

"""
	forward_star_difference(uₕ::VectorElement, dim_val::Val)

The forward difference of `uₕ` along `dim_val`, divided by the averaged spacing:

```math
\\textrm{Dstar}_{+}(\\textrm{u}_h)(i) =
    \\frac{\\textrm{u}_h(x_{i+1}) - \\textrm{u}_h(x_i)}{(h_i + h_{i+1})/2}
```

Reached through [`Dstar₊ₓ`](@ref) and its siblings. Unlike the other difference families
this one takes a grid function only, not a mesh or a grid space: there is no matrix form.

The last point has no forward neighbour, so it is truncated to zero, as in
[`D₊ₓ`](@ref).

See also: [`star_spacings`](@ref), [`D₊ₓ`](@ref).
"""
function forward_star_difference(
        uₕ::VectorElement{<:ScalarGridSpace}, dim_val::Val{DIM}) where {DIM}
    vₕ = similar(uₕ)
    h = star_spacings(_op_mesh(uₕ)(DIM))
    _apply_stencil!(vₕ, uₕ, h, Forward(), dim_val)
    return vₕ
end

# A composite grid function is differenced one component at a time; every component
# shares the mesh, so the denominator is built once.
function forward_star_difference(
        uₕ::VectorElement{<:CompositeGridSpace}, dim_val::Val{DIM}) where {DIM}
    vₕ = similar(uₕ)
    h = star_spacings(_op_mesh(uₕ)(DIM))
    _apply_componentwise!(
        (v, u) -> _apply_stencil!(v, u, h, Forward(), dim_val), vₕ, uₕ)
    return vₕ
end

# The directional aliases are written out rather than generated: the shared generator
# documents its aliases as taking a mesh, a grid space or a grid function, and this
# family takes only the last.
for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
    alias = Symbol(:Dstar₊, suffix)
    direction = _BRAMBLE_var2label[i]
    @eval begin
        @doc """
          	$($(QuoteNode(alias)))(uₕ::VectorElement)

          The forward difference of `uₕ` along the `$($direction)` direction over the
          averaged spacing, ``\\\\frac{u_{i+1} - u_i}{(h_i + h_{i+1})/2}``.

          Alias for `forward_star_difference(uₕ, Val($($i)))`. Takes a grid function only;
          there is no matrix form. The last point along `$($direction)` is truncated to
          zero.
          """
        @inline $alias(uₕ::VectorElement) = forward_star_difference(uₕ, Val($i))
    end
end

"""
	Dstar₊ₕ(uₕ::VectorElement)

The starred forward difference of `uₕ` along every coordinate, as a tuple with one grid
function per spatial dimension. On a one-dimensional mesh it returns that single entry
rather than a one-tuple.
"""
@inline Dstar₊ₕ(uₕ::VectorElement) = Dstar₊ₕ(uₕ, Val(dim(_op_mesh(uₕ))))
@inline Dstar₊ₕ(uₕ::VectorElement, ::Val{1}) = forward_star_difference(uₕ, Val(1))
@inline Dstar₊ₕ(
    uₕ::VectorElement, ::Val{D}) where {D} = ntuple(
    i -> forward_star_difference(uₕ, Val(i)), Val(D))
