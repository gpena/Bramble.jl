##############################################################################
#                                                                            #
#             Implementation of (Finite) Difference Operators                #
#                                                                            #
##############################################################################

# --- Type System for Dispatch ---
abstract type GridDirection end
struct Forward <: GridDirection end
struct Backward <: GridDirection end

# --- Core Difference Computation ---
@inline _get_h_val(h::AbstractVector, i::Int) = h[i]
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
@inline @propagate_inbounds _compute_difference(::Backward, ::Val{true}, cur, h, i) = cur / _get_h_val(h, 2)

# --- Unified Difference Engine ---
function _difference_engine!(out, in_ref, h, dims::NTuple{D,Int}, dir::GridDirection, ::Val{DIFF_DIM}) where {D,DIFF_DIM}
	li = LinearIndices(dims)
	step_cartesian = CartesianIndex(ntuple(i -> i == DIFF_DIM ? 1 : 0, D))
	full_axes = axes(li)

	if dir isa Forward
		interior_axes = ntuple(d -> d == DIFF_DIM ? (first(full_axes[d]):(last(full_axes[d]) - 1)) : full_axes[d], D)
		boundary_axes = ntuple(d -> d == DIFF_DIM ? (last(full_axes[d]):last(full_axes[d])) : full_axes[d], D)

		@inbounds @simd for I in CartesianIndices(interior_axes)
			idx, idx_next = li[I], li[I + step_cartesian]
			out[idx] = _compute_difference(dir, Val(false), in_ref[idx_next], in_ref[idx], h, I[DIFF_DIM])
		end

		@inbounds @simd for I in CartesianIndices(boundary_axes)
			idx = li[I]
			out[idx] = _compute_difference(dir, Val(true), in_ref[idx], h, I[DIFF_DIM])
		end
	else # Backward
		interior_axes = ntuple(d -> d == DIFF_DIM ? ((first(full_axes[d]) + 1):last(full_axes[d])) : full_axes[d], D)
		boundary_axes = ntuple(d -> d == DIFF_DIM ? (first(full_axes[d]):first(full_axes[d])) : full_axes[d], D)

		@inbounds @simd for I in CartesianIndices(interior_axes)
			idx, idx_prev = li[I], li[I - step_cartesian]
			out[idx] = _compute_difference(dir, Val(false), in_ref[idx], in_ref[idx_prev], h, I[DIFF_DIM])
		end
		@inbounds @simd for I in CartesianIndices(boundary_axes)
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

op_configs = [
	(direction = Forward(),
	 diff_name = :forward_difference,
	 finite_diff_name = :forward_finite_difference,
	 weights_func! = :forward_derivative_weights!,
	 spacing_func = :forward_spacing,
	 diff_alias = :diff₊,
	 finite_diff_alias = :D₊,
	 grad_alias = :diff₊ₕ,
	 finite_grad_alias = :∇₊ₕ),
	(direction = Backward(),
	 diff_name = :backward_difference,
	 finite_diff_name = :backward_finite_difference,
	 weights_func! = :backward_derivative_weights!,
	 spacing_func = :spacing,
	 diff_alias = :diff₋,
	 finite_diff_alias = :D₋,
	 grad_alias = :diff₋ₕ,
	 finite_grad_alias = :∇₋ₕ)
]

for config in op_configs
	dir_instance = config.direction
	diff_name = config.diff_name
	finite_diff_name = config.finite_diff_name
	weights_func! = config.weights_func!
	spacing_func = config.spacing_func

	@eval begin
		# --- In-place applicators ---
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
		@inline function $weights_func!(v::AbstractVector, Ωₕ::AbstractMeshType, diff_dim::Val)
			_derivative_weights!(v, Ωₕ, $spacing_func, diff_dim)
		end

		# --- Matrix operator functions ---
		@inline $diff_name(Ωₕ::AbstractMeshType, dim_val::Val) = _difference_operator(Ωₕ, $dir_instance, dim_val)

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
	for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
		diff_alias_op_name = Symbol(config.diff_alias, suffix)
		finite_diff_alias_op_name = Symbol(config.finite_diff_alias, suffix)

		@eval begin
			@inline $(diff_alias_op_name)(arg) = $diff_name(arg, Val($i))
			@inline $(finite_diff_alias_op_name)(arg) = $finite_diff_name(arg, Val($i))
		end
	end

	# --- Aliases for gradient tuples ---
	for (grad_op, base_op) in [(config.grad_alias, diff_name), (config.finite_grad_alias, finite_diff_name)]
		@eval begin
			@inline $grad_op(arg) = $grad_op(arg, Val(dim(mesh(space(arg)))))
			@inline $grad_op(arg, ::Val{1}) = $base_op(arg, Val(1))
			@inline $grad_op(arg, ::Val{D}) where D = ntuple(i -> $base_op(arg, Val(i)), Val(D))
		end
	end
end