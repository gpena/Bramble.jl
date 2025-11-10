###############################################################
#                                                             #
#           Implementation of the average operators           #
#                                                             #
###############################################################

@inline @propagate_inbounds _compute_average(::Forward, ::Val{false}, in_next, in_val) = (in_next + in_val) * 0.5
@inline @propagate_inbounds _compute_average(::Forward, ::Val{true}, in_val) = in_val * 0.5

@inline @propagate_inbounds _compute_average(::Backward, ::Val{false}, in_val, in_prev) = (in_val + in_prev) * 0.5
@inline @propagate_inbounds _compute_average(::Backward, ::Val{true}, in_val) = in_val * 0.5

@inbounds function _average_engine!(out, in_ref, dims::NTuple{D,Int}, dir::GridDirection, ::Val{AVG_DIM}) where {D,AVG_DIM}
	li = LinearIndices(dims)
	step_cartesian = CartesianIndex(ntuple(i -> i == AVG_DIM ? 1 : 0, D))
	full_axes = axes(li)

	if dir isa Forward
		interior_axes = ntuple(d -> d == AVG_DIM ? (first(full_axes[d]):(last(full_axes[d]) - 1)) : full_axes[d], D)
		boundary_axes = ntuple(d -> d == AVG_DIM ? (last(full_axes[d]):last(full_axes[d])) : full_axes[d], D)

		@simd for I in CartesianIndices(interior_axes)
			idx, idx_next = li[I], li[I + step_cartesian]
			out[idx] = _compute_average(dir, Val(false), in_ref[idx_next], in_ref[idx])
		end

		@simd for I in CartesianIndices(boundary_axes)
			idx = li[I]
			out[idx] = _compute_average(dir, Val(true), in_ref[idx])
		end
	else # Backward
		interior_axes = ntuple(d -> d == AVG_DIM ? ((first(full_axes[d]) + 1):last(full_axes[d])) : full_axes[d], D)
		boundary_axes = ntuple(d -> d == AVG_DIM ? (first(full_axes[d]):first(full_axes[d])) : full_axes[d], D)

		@simd for I in CartesianIndices(interior_axes)
			idx, idx_prev = li[I], li[I - step_cartesian]
			out[idx] = _compute_average(dir, Val(false), in_ref[idx], in_ref[idx_prev])
		end

		@simd for I in CartesianIndices(boundary_axes)
			idx = li[I]
			out[idx] = _compute_average(dir, Val(true), in_ref[idx])
		end
	end
end

function add_half_shift(Ωₕ::AbstractMeshType, ::Val{DIFF_DIM}, ::Val{first}, ::Val{second}) where {DIFF_DIM,first,second}
	return (shift(Ωₕ, Val(DIFF_DIM), Val(first)) + shift(Ωₕ, Val(DIFF_DIM), Val(second))) * 0.5
end

function _average_operator(Ωₕ::AbstractMeshType, ::Forward, ::Val{AVG_DIM}) where AVG_DIM
	return add_half_shift(Ωₕ, Val(AVG_DIM), Val(0), Val(1))
end

function _average_operator(Ωₕ::AbstractMeshType, ::Backward, ::Val{AVG_DIM}) where AVG_DIM
	return add_half_shift(Ωₕ, Val(AVG_DIM), Val(0), Val(-1))
end
# Configuration array for average operators, expanded with descriptive strings.
op_configs = [
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
for config in op_configs
	# Extract ALL values from `config` to avoid scope issues with @eval.
	dir_instance = config.direction
	average_name = config.average_name
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
		function $(Symbol(average_name, :_dim!))(out, in, h, dims::NTuple{D,Int}, average_dim::Val{DIFF_DIM}) where {D,DIFF_DIM}
			@assert 1 <= DIFF_DIM <= D "Dimension must be between 1 and $D."
			@assert length(out) == length(in) == prod(dims) "Vector and grid dimensions must match."
			in_ref = (out === in) ? copy(in) : in
			_average_engine!(out, in_ref, dims, $dir_instance, average_dim)
			return
		end

		function $(Symbol(average_name, :_dim!))(out, in, dims::NTuple{D,Int}, average_dim::Val{DIFF_DIM}) where {D,DIFF_DIM}
			return $(Symbol(average_name, :_dim!))(out, in, nothing, dims, average_dim)
		end

		# --- Matrix operator functions ---
		@doc """
			$($(QuoteNode(average_name)))(arg, dim_val::Val)

		Constructs or applies the $($dir_string_lowercase) averaging operator, representing the operation ``$($math_op)``.
		"""
		@inline $average_name(Ωₕ::AbstractMeshType, dim_val::Val) = _average_operator(Ωₕ, $dir_instance, dim_val)

		# --- Generic applicators ---
		@inline $average_name(Wₕ::AbstractSpaceType, dim_val::Val) = elements(Wₕ, $average_name(mesh(Wₕ), dim_val))
		function $average_name(uₕ::VectorElement, dim_val::Val)
			vₕ = similar(uₕ)
			dims = ndofs(space(uₕ), Tuple)
			_average_engine!(vₕ.data, uₕ.data, dims, $dir_instance, dim_val)
			return vₕ
		end
		@inline $average_name(Uₕ::MatrixElement, dim_val::Val) = $average_name(space(Uₕ), dim_val) * Uₕ
	end

	# --- Aliases for x, y, z directions ---
	# Use the robust helper function to generate aliases and their docstrings.
	for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
		direction = _BRAMBLE_var2label[i]
		_define_directional_alias(average_name, Symbol(average_alias, suffix), dir_string_lowercase, direction, i)
	end

	# --- Aliases for vectorial average tuples ---
	for (vectorial_average_op, base_op) in [(vectorial_average_alias, average_name),]
		@eval begin
			@doc """
				$($(QuoteNode(vectorial_average_op)))(arg)

			Computes the vectorial $($dir_string_lowercase) average of `arg`, returning a tuple of
			operators/elements for each spatial dimension.

			For a 2D space, `$($(QuoteNode(vectorial_average_op)))(uₕ)` is equivalent to
			`($($(QuoteNode(base_op)))(uₕ, Val(1)), $($(QuoteNode(base_op)))(uₕ, Val(2)))`.
			"""
			@inline $vectorial_average_op(arg) = $vectorial_average_op(arg, Val(dim(mesh(space(arg)))))
			@inline $vectorial_average_op(arg, ::Val{1}) = $base_op(arg, Val(1))
			@inline $vectorial_average_op(arg, ::Val{D}) where D = ntuple(i -> $base_op(arg, Val(i)), Val(D))
		end
	end
end

@inline _create_average_matrix(Ωₕ::AbstractMeshType, i::Int) = backward_average(Ωₕ, Val(i))
