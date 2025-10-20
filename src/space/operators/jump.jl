##############################################################################
#                                                                            #
#                   Implementation of the jump operators                     #
#                                                                            #
##############################################################################

# --- Unified Matrix Operator Implementation ---
function _jump_operator(Ωₕ::AbstractMeshType, ::Forward, ::Val{JUMP_DIM}) where JUMP_DIM
	return difference_shift(Ωₕ, Val(JUMP_DIM), Val(1), Val(0))
end

function _jump_operator(Ωₕ::AbstractMeshType, ::Backward, ::Val{JUMP_DIM}) where JUMP_DIM
	return difference_shift(Ωₕ, Val(JUMP_DIM), Val(0), Val(-1))
end

op_configs = [
	(direction = Forward(),
	 jump_name = :forward_jump,
	 jump_alias = :jump₊,
	 vectorial_jump_alias = :jump₊ₕ),
	(direction = Backward(),
	 jump_name = :backward_jump,
	 jump_alias = :jump₋,
	 vectorial_jump_alias = :jump₋ₕ)
]

for config in op_configs
	dir_instance = config.direction
	jump_name = config.jump_name

	@eval begin
		# --- In-place applicators ---
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
		@inline $jump_name(Ωₕ::AbstractMeshType, dim_val::Val) = _jump_operator(Ωₕ, $dir_instance, dim_val)

		# --- Generic applicators ---
		@inline $jump_name(Wₕ::AbstractSpaceType, dim_val::Val) = elements(Wₕ, $jump_name(mesh(Wₕ), dim_val))
		function $jump_name(uₕ::VectorElement, dim_val::Val{DIM}) where DIM
			vₕ = similar(uₕ)
			dims = ndofs(space(uₕ), Tuple)
			_difference_engine!(vₕ.data, uₕ.data, nothing, dims, $dir_instance, dim_val)
			return vₕ
		end
		@inline $jump_name(Uₕ::MatrixElement, dim_val::Val) = $jump_name(space(Uₕ), dim_val) * Uₕ
	end

	# --- Aliases for x, y, z directions ---
	for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
		jump_alias_op_name = Symbol(config.jump_alias, suffix)

		@eval begin
			@inline $(jump_alias_op_name)(arg) = $jump_name(arg, Val($i))
		end
	end

	# --- Aliases for gradient tuples ---
	for (vectorial_jump_op, base_op) in [(config.vectorial_jump_alias, jump_name),]
		@eval begin
			@inline $vectorial_jump_op(arg) = $vectorial_jump_op(arg, Val(dim(mesh(space(arg)))))
			@inline $vectorial_jump_op(arg, ::Val{1}) = $base_op(arg, Val(1))
			@inline $vectorial_jump_op(arg, ::Val{D}) where D = ntuple(i -> $base_op(arg, Val(i)), Val(D))
		end
	end
end