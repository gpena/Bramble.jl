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
@inline function _get_h_val(h::AbstractVector, i::Int)
	@assert 1 <= i <= length(h) "Index $i out of bounds for spacing vector of length $(length(h))."
	return h[i]
end
@inline _get_h_val(h::F, i::Int) where {F<:Function} = h(i)

# Case 1: Simple difference (h is Nothing)
@inline @propagate_inbounds _compute_difference(::Forward, ::Val{false}, next, cur, ::Nothing, i) = next - cur
@inline @propagate_inbounds _compute_difference(::Forward, ::Val{true}, cur, ::Nothing, i) = -cur

@inline @propagate_inbounds _compute_difference(::Backward, ::Val{false}, cur, prev, ::Nothing, i) = cur - prev
@inline @propagate_inbounds _compute_difference(::Backward, ::Val{true}, cur, ::Nothing, i) = cur

# Case 2: Finite difference (h is provided)
@inline @propagate_inbounds @muladd _compute_difference(::Forward, ::Val{false}, next, cur, h, i) = (next - cur) / _get_h_val(h, i)
@inline @propagate_inbounds _compute_difference(::Forward, ::Val{true}, cur, h, i) = -cur / _get_h_val(h, i)

@inline @propagate_inbounds @muladd _compute_difference(::Backward, ::Val{false}, cur, prev, h, i) = (cur - prev) / _get_h_val(h, i)
@inline @propagate_inbounds _compute_difference(::Backward, ::Val{true}, cur, h, i) = cur / _get_h_val(h, 2) # or is it _get_h_val(h, 1)

# --- Unified Difference Engine ---
@inbounds function _difference_engine!(out, in_ref, h, dims::NTuple{D,Int}, dir::GridDirection, ::Val{DIFF_DIM}) where {D,DIFF_DIM}
	li = LinearIndices(dims)
	step_cartesian = CartesianIndex(ntuple(i -> i == DIFF_DIM ? 1 : 0, D))
	full_axes = axes(li)

	if dir isa Forward
		interior_axes = ntuple(d -> d == DIFF_DIM ? (first(full_axes[d]):(last(full_axes[d]) - 1)) : full_axes[d], D)
		boundary_axes = ntuple(d -> d == DIFF_DIM ? (last(full_axes[d]):last(full_axes[d])) : full_axes[d], D)

		@simd for I in CartesianIndices(interior_axes)
			idx, idx_next = li[I], li[I + step_cartesian]
			out[idx] = _compute_difference(dir, Val(false), in_ref[idx_next], in_ref[idx], h, I[DIFF_DIM])
		end

		@simd for I in CartesianIndices(boundary_axes)
			idx = li[I]
			out[idx] = _compute_difference(dir, Val(true), in_ref[idx], h, I[DIFF_DIM])
		end
	else # Backward
		interior_axes = ntuple(d -> d == DIFF_DIM ? ((first(full_axes[d]) + 1):last(full_axes[d])) : full_axes[d], D)
		boundary_axes = ntuple(d -> d == DIFF_DIM ? (first(full_axes[d]):first(full_axes[d])) : full_axes[d], D)

		@simd for I in CartesianIndices(interior_axes)
			idx, idx_prev = li[I], li[I - step_cartesian]
			out[idx] = _compute_difference(dir, Val(false), in_ref[idx], in_ref[idx_prev], h, I[DIFF_DIM])
		end
		@simd for I in CartesianIndices(boundary_axes)
			idx = li[I]
			out[idx] = _compute_difference(dir, Val(true), in_ref[idx], h, 1)
		end
	end
end

function difference_shift(Ωₕ::AbstractMeshType, ::Val{DIFF_DIM}, ::Val{first}, ::Val{second}) where {DIFF_DIM,first,second}
	return shift(Ωₕ, Val(DIFF_DIM), Val(first)) - shift(Ωₕ, Val(DIFF_DIM), Val(second))
end

function _difference_operator(Ωₕ::AbstractMeshType, ::Forward, ::Val{DIFF_DIM}) where DIFF_DIM
	return difference_shift(Ωₕ, Val(DIFF_DIM), Val(1), Val(0))
end

function _difference_operator(Ωₕ::AbstractMeshType, ::Backward, ::Val{DIFF_DIM}) where DIFF_DIM
	return difference_shift(Ωₕ, Val(DIFF_DIM), Val(0), Val(-1))
end

function _derivative_weights!(v::AbstractVector, Ωₕ::AbstractMeshType, spacing_func, ::Val{DIFF_DIM}) where DIFF_DIM
	dims = npoints(Ωₕ, Tuple)

	@assert 1 <= DIFF_DIM <= dim(Ωₕ) "The differentiation dimension must be between 1 and $(dim(Ωₕ))."

	spacings_1d = Base.Fix1(spacing_func, Ωₕ(DIFF_DIM))
	li = LinearIndices(dims)

	@inbounds @simd for I in CartesianIndices(dims)
		v[li[I]] = inv(spacings_1d(I[DIFF_DIM]))
	end
	return
end

function _define_directional_alias(base_op_name, alias_name, dir_string, suffix, direction_index)
	# 1. Construct the docstring content.
	doc_string = """
		$alias_name(arg)

	Alias for `$base_op_name(arg, Val($direction_index))`. Computes the
	`$dir_string` difference in the `$suffix`-direction.
	"""

	# 2. Construct the function definition as an expression.
	func_def_expr = :(@inline $(alias_name)(arg) = $(base_op_name)(arg, Val($(direction_index))))

	# 3. Combine them using the @doc macro syntax into a final expression.
	#    The `__source__` variable is replaced with `nothing`.
	final_expr = Expr(:macrocall, GlobalRef(Core, Symbol("@doc")), nothing, doc_string, func_def_expr)

	# 4. Evaluate the final, complete expression in the module's global scope.
	Core.eval(@__MODULE__, final_expr)
end

# Configuration array to define forward and backward difference operators.
op_configs = [
	(direction = Forward(),
	 diff_name = :forward_difference,
	 finite_diff_name = :forward_finite_difference,
	 weights_func! = :forward_derivative_weights!,
	 spacing_func = :forward_spacing,
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
	 spacing_func = :spacing,
	 diff_alias = :diff₋,
	 finite_diff_alias = :D₋,
	 grad_alias = :diff₋ₕ,
	 finite_grad_alias = :∇₋ₕ,
	 dir_string = "Backward",
	 dir_string_lowercase = "backward",
	 math_op = "u_{i} - u_{i-1}", math_finite_op = "\\frac{u_{i} - u_{i-1}}{h_i}")
]

# Metaprogramming loop to generate all specified difference operators.
for config in op_configs
	# Extract ALL values from `config` into local variables here.
	dir_instance = config.direction
	diff_name = config.diff_name
	finite_diff_name = config.finite_diff_name
	weights_func! = config.weights_func!
	spacing_func = config.spacing_func
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
		function $(Symbol(diff_name, :_dim!))(out, in, h, dims::NTuple{D,Int}, diff_dim::Val{DIFF_DIM}) where {D,DIFF_DIM}
			@assert 1 <= DIFF_DIM <= D "Differentiation dimension must be between 1 and $D."
			@assert length(out) == length(in) == prod(dims) "Vector and grid dimensions must match."
			in_ref = (out === in) ? copy(in) : in
			_difference_engine!(out, in_ref, h, dims, $dir_instance, diff_dim)
			return
		end

		function $(Symbol(diff_name, :_dim!))(out, in, dims::NTuple{D,Int}, diff_dim::Val{DIFF_DIM}) where {D,DIFF_DIM}
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
		@inline $diff_name(Ωₕ::AbstractMeshType, dim_val::Val) = _difference_operator(Ωₕ, $dir_instance, dim_val)

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
		@inline $diff_name(Wₕ::AbstractSpaceType, dim_val::Val) = elements(Wₕ, $diff_name(mesh(Wₕ), dim_val))
		function $diff_name(uₕ::VectorElement, dim_val::Val)
			vₕ = similar(uₕ)
			dims = ndofs(space(uₕ), Tuple)
			_difference_engine!(vₕ.data, uₕ.data, nothing, dims, $dir_instance, dim_val)
			return vₕ
		end
		@inline $diff_name(Uₕ::MatrixElement, dim_val::Val) = $diff_name(space(Uₕ), dim_val) * Uₕ

		@inline $finite_diff_name(Wₕ::AbstractSpaceType, dim_val::Val) = elements(Wₕ, $finite_diff_name(mesh(Wₕ), dim_val))
		function $finite_diff_name(uₕ::VectorElement, dim_val::Val{DIM}) where DIM
			vₕ = similar(uₕ)
			dims = ndofs(space(uₕ), Tuple)
			spacings = Base.Fix1($spacing_func, mesh(space(uₕ))(DIM))
			_difference_engine!(vₕ.data, uₕ.data, spacings, dims, $dir_instance, dim_val)
			return vₕ
		end
		@inline $finite_diff_name(Uₕ::MatrixElement, dim_val::Val) = $finite_diff_name(space(Uₕ), dim_val) * Uₕ
	end

	# --- Aliases for x, y, z directions ---
	# ❗️ FIX: Call the helper function to generate the aliases safely.
	for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
		direction = _BRAMBLE_var2label[i]
		_define_directional_alias(diff_name, Symbol(diff_alias, suffix), dir_string_lowercase, direction, i)
		_define_directional_alias(finite_diff_name, Symbol(finite_diff_alias, suffix), dir_string_lowercase, direction, i)
	end

	# --- Aliases for gradient tuples ---
	# This loop is fine because the `i` is inside the generated function's body (`ntuple(i->...`)),
	# not being interpolated from the outer scope.
	for (grad_op, base_op) in [(grad_alias, diff_name), (finite_grad_alias, finite_diff_name)]
		@eval begin
			@doc """
				$($(QuoteNode(grad_op)))(arg)

			Computes the $($dir_string_lowercase) gradient of `arg`, returning a tuple of
			operators/elements for each spatial dimension.

			For a 2D space, `$($(QuoteNode(grad_op)))(uₕ)` is equivalent to
			`($($(QuoteNode(base_op)))(uₕ, Val(1)), $($(QuoteNode(base_op)))(uₕ, Val(2)))`.
			"""
			@inline $grad_op(arg) = $grad_op(arg, Val(dim(mesh(space(arg)))))
			@inline $grad_op(arg, ::Val{1}) = $base_op(arg, Val(1))
			@inline $grad_op(arg, ::Val{D}) where D = ntuple(i -> $base_op(arg, Val(i)), Val(D))
		end
	end
end