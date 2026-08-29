##############################################################################
#                                                                            #
#                   Implementation of the jump operators                     #
#                                                                            #
##############################################################################

#=
# jump.jl

The jump across an interface is arithmetically the unscaled difference:

    forward   ⟦u⟧ᵢ₊½ = u_{i+1} - u_i
    backward  ⟦u⟧ᵢ₋½ = u_i - u_{i-1}

which is exactly what `forward_difference` and `backward_difference` compute. This file
therefore defines no arithmetic of its own: every name below forwards to its counterpart
in operators/difference.jl, and so shares the stencil engine and the boundary convention
with it.

The names are kept because they carry intent rather than arithmetic. A penalty term in an
interior penalty method reads as a jump, not as a difference, and writing `jump₋ₓ` says
which of the two is meant. The distinction is one of meaning, so it lives in the names and
the docstrings, and nowhere in the code that runs.

If the two ever need to diverge, for instance if a jump should be evaluated on an
interface mesh rather than on the grid points, this is the file to grow. Until then a
second implementation would be a second thing to keep correct.

See also: [`diff₋ₓ`](@ref), [`diff₊ₓ`](@ref), [`_difference_engine!`](@ref).
=#

const _JUMP_OP_CONFIGS = [
    (jump_name = :forward_jump,
        difference_name = :forward_difference,
        jump_alias = :jump₊,
        vectorial_jump_alias = :jump₊ₕ,
        dir_string_lowercase = "forward",
        math_op = "u_{i+1} - u_i"),
    (jump_name = :backward_jump,
        difference_name = :backward_difference,
        jump_alias = :jump₋,
        vectorial_jump_alias = :jump₋ₕ,
        dir_string_lowercase = "backward",
        math_op = "u_{i} - u_{i-1}")
]

for config in _JUMP_OP_CONFIGS
    jump_name = config.jump_name
    difference_name = config.difference_name
    jump_alias = config.jump_alias
    vectorial_jump_alias = config.vectorial_jump_alias
    dir_string_lowercase = config.dir_string_lowercase
    math_op = config.math_op

    jump_dim! = Symbol(jump_name, :_dim!)
    difference_dim! = Symbol(difference_name, :_dim!)

    @eval begin
        @doc """
          	$($(QuoteNode(jump_dim!)))(out, in, dims, jump_dim::Val)

          In-place $($dir_string_lowercase) jump of `in` along `jump_dim`, written into
          `out`, computing ``$($math_op)``.

          Forwards to [`$($(QuoteNode(difference_dim!)))`](@ref); the jump and the unscaled
          difference are the same arithmetic.
          """
        @inline $jump_dim!(out, in, dims::NTuple{D, Int}, jump_dim::Val) where {D} = $difference_dim!(
            out, in, dims, jump_dim)

        @doc """
          	$($(QuoteNode(jump_name)))(arg, dim_val::Val)

          The $($dir_string_lowercase) jump along the direction `dim_val`, ``$($math_op)``.

          `arg` is a mesh, a grid space or a [`VectorElement`](@ref): the first two give the
          operator as a sparse matrix, the third applies it and returns a `VectorElement`.

          Forwards to [`$($(QuoteNode(difference_name)))`](@ref), since a jump across an
          interface and an unscaled difference are the same quantity. The name records
          which of the two is meant.
          """
        @inline $jump_name(arg, dim_val::Val) = $difference_name(arg, dim_val)
    end

    # --- Aliases for x, y, z directions ---
    for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
        direction = _BRAMBLE_var2label[i]
        _define_directional_alias(jump_name, Symbol(jump_alias, suffix),
            dir_string_lowercase, direction, i, "jump", math_op)
    end

    # --- Alias for the vectorial jump tuple ---
    _define_vectorial_alias(jump_name, vectorial_jump_alias, dir_string_lowercase, "jump")
end
