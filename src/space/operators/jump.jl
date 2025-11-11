##############################################################################
#                                                                            #
#                   Implementation of the jump operators                     #
#                                                                            #
##############################################################################

#=
# jump.jl

This file implements jump operators for discontinuous Galerkin and interface computations.

## Mathematical Formulation

The jump ⟦·⟧ measures the discontinuity of a function across element interfaces:

**Forward jump** (at interface xᵢ₊½):
    ⟦u⟧ᵢ₊½ᶠ = uᵢ₊₁ - uᵢ

**Backward jump** (at interface xᵢ₋½):
    ⟦u⟧ᵢ₋½ᵇ = uᵢ - uᵢ₋₁

## Physical Interpretation

Jumps quantify:
1. **Discontinuities**: Size of jumps in DG methods
2. **Flux differences**: Change in flux across interfaces
3. **Shock strength**: Magnitude of discontinuities in hyperbolic PDEs
4. **Penalty terms**: Used in interior penalty methods

## Relationship to Differences

The jump is equivalent to a first-order difference, but conceptually:
- **Difference**: Approximates derivatives (∂u/∂x ≈ Δu/Δx)
- **Jump**: Measures discontinuities (⟦u⟧ = u⁺ - u⁻)

Mathematically, for uniform grids:
    ⟦u⟧ᵢ = Δuᵢ = uᵢ₊₁ - uᵢ

## Example - DG Method

```julia
# Interior penalty term: σ ∫⟦uₕ⟧² ds
u_jump = jump₊(uₕ, dim)  # Jump at all interfaces
penalty = penalty_parameter * sum(u_jump.^2 .* face_measure)
```

## Implementation Strategy

Uses metaprogramming to generate:
- In-place applicators: `jump_dim!`
- Matrix operators: Returns difference matrices
- Multiple dispatch on direction traits

See also: [`jump₊`](@ref), [`jump₋`](@ref), [`_jump_operator`](@ref), [`difference_shift`](@ref)
=#

# --- Unified Matrix Operator Implementation ---
function _jump_operator(Ωₕ::AbstractMeshType, ::Forward, ::Val{JUMP_DIM}) where JUMP_DIM
	return difference_shift(Ωₕ, Val(JUMP_DIM), Val(1), Val(0))
end

function _jump_operator(Ωₕ::AbstractMeshType, ::Backward, ::Val{JUMP_DIM}) where JUMP_DIM
	return difference_shift(Ωₕ, Val(JUMP_DIM), Val(0), Val(-1))
end
# This code assumes the `_define_directional_alias` function from your previous
# example is defined and available in the current scope.

# Configuration array for jump operators, expanded with descriptive strings.
op_configs = [
	(direction = Forward(),
	 jump_name = :forward_jump,
	 jump_alias = :jump₊,
	 vectorial_jump_alias = :jump₊ₕ,
	 dir_string_lowercase = "forward",
	 math_op = "u_{i+1} - u_i"),
	(direction = Backward(),
	 jump_name = :backward_jump,
	 jump_alias = :jump₋,
	 vectorial_jump_alias = :jump₋ₕ,
	 dir_string_lowercase = "backward",
	 math_op = "u_{i} - u_{i-1}")
]

# Metaprogramming loop to generate all specified jump operators.
for config in op_configs
	# Extract ALL values from `config` to avoid scope issues with @eval.
	dir_instance = config.direction
	jump_name = config.jump_name
	jump_alias = config.jump_alias
	vectorial_jump_alias = config.vectorial_jump_alias
	dir_string_lowercase = config.dir_string_lowercase
	math_op = config.math_op

	@eval begin
		# --- In-place applicators ---
		@doc """
			$($(QuoteNode(Symbol(jump_name, :_dim!))))(out, in, dims, jump_dim)

		Low-level, in-place function to compute the $($dir_string_lowercase) jump of vector `in` along dimension `jump_dim`, storing the result in `out`. This function computes ``$($math_op)``.
		"""
		function $(Symbol(jump_name, :_dim!))(out, in, h, dims::NTuple{D,Int}, jump_dim::Val{DIFF_DIM}) where {D,DIFF_DIM}
			@assert 1 <= DIFF_DIM <= D "Dimension must be between 1 and $D."
			@assert length(out) == length(in) == prod(dims) "Vector and grid dimensions must match."
			in_ref = (out === in) ? copy(in) : in
			_difference_engine!(out, in_ref, nothing, dims, $dir_instance, jump_dim)
			return
		end

		function $(Symbol(jump_name, :_dim!))(out, in, dims::NTuple{D,Int}, jump_dim::Val{DIFF_DIM}) where {D,DIFF_DIM}
			return $(Symbol(jump_name, :_dim!))(out, in, nothing, dims, jump_dim)
		end

		# --- Matrix operator functions ---
		@doc """
			$($(QuoteNode(jump_name)))(arg, dim_val::Val)

		Constructs or applies the $($dir_string_lowercase) jump operator, representing the operation ``$($math_op)``.
		"""
		@inline $jump_name(Ωₕ::AbstractMeshType, dim_val::Val) = _jump_operator(Ωₕ, $dir_instance, dim_val)

		# --- Generic applicators ---
		@inline $jump_name(Wₕ::AbstractSpaceType, dim_val::Val) = elements(Wₕ, $jump_name(mesh(Wₕ), dim_val))
		function $jump_name(uₕ::VectorElement, dim_val::Val)
			vₕ = similar(uₕ)
			dims = ndofs(space(uₕ), Tuple)
			_difference_engine!(vₕ.data, uₕ.data, nothing, dims, $dir_instance, dim_val)
			return vₕ
		end
		@inline $jump_name(Uₕ::MatrixElement, dim_val::Val) = $jump_name(space(Uₕ), dim_val) * Uₕ
	end

	# --- Aliases for x, y, z directions ---
	# Use the robust helper function to generate aliases and their docstrings.
	for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
		direction = _BRAMBLE_var2label[i]
		_define_directional_alias(jump_name, Symbol(jump_alias, suffix), dir_string_lowercase, direction, i)
	end

	# --- Aliases for vectorial jump tuples ---
	for (vectorial_jump_op, base_op) in [(vectorial_jump_alias, jump_name),]
		@eval begin
			@doc """
				$($(QuoteNode(vectorial_jump_op)))(arg)

			Computes the vectorial $($dir_string_lowercase) jump of `arg`, returning a tuple of
			operators/elements for each spatial dimension.

			For a 2D space, `$($(QuoteNode(vectorial_jump_op)))(uₕ)` is equivalent to
			`($($(QuoteNode(base_op)))(uₕ, Val(1)), $($(QuoteNode(base_op)))(uₕ, Val(2)))`.
			"""
			@inline $vectorial_jump_op(arg) = $vectorial_jump_op(arg, Val(dim(mesh(space(arg)))))
			@inline $vectorial_jump_op(arg, ::Val{1}) = $base_op(arg, Val(1))
			@inline $vectorial_jump_op(arg, ::Val{D}) where D = ntuple(i -> $base_op(arg, Val(i)), Val(D))
		end
	end
end
