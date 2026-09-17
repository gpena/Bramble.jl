##############################################################################
#                                                                            #
#                   Implementation of the jump operators                     #
#                                                                            #
##############################################################################

#=
# jump.jl

The jump across an interface:

    ⟦u⟧ᵢ = u_{i+1} - u_i

There is one of these, not a forward and a backward pair. The jump is a property of the
interface between two cells rather than of a direction of travel across it, so ⟦u⟧ at the
interface between xᵢ and xᵢ₊₁ is a single quantity; naming a backward jump as well would
name the same interface twice, once from each side, and give the same numbers shifted by
one index.

Arithmetically it is the unscaled forward difference, which is what `forward_difference`
computes, so this file defines no arithmetic of its own: every name below forwards to its
counterpart in operators/difference.jl, and so shares the stencil engine and the boundary
convention with it. In particular the last point along the jump direction has no forward
neighbour and is treated as though it were zero, giving -uₙ rather than 0, which is what
makes the operator agree with its matrix.

The names are kept because they carry intent rather than arithmetic. A penalty term in an
interior penalty method reads as a jump, not as a difference, and writing `jumpₓ` says
which of the two is meant. The distinction is one of meaning, so it lives in the names and
the docstrings, and nowhere in the code that runs.

If the two ever need to diverge, for instance if a jump should be evaluated on an
interface mesh rather than on the grid points, this is the file to grow. Until then a
second implementation would be a second thing to keep correct.

See also: [`diff₊ₓ`](@ref), [`_difference_engine!`](@ref).
=#

"""
    jump_dim!(out, in, dims, jump_dim::Val)

In-place jump of `in` along `jump_dim`, written into `out`, computing
``u_{i+1} - u_i``.

Forwards to [`forward_difference_dim!`](@ref); the jump and the unscaled forward
difference are the same arithmetic.
"""
@inline jump_dim!(out, in, dims::NTuple{D, Int}, jump_dim::Val) where {D} = forward_difference_dim!(out, in, dims, jump_dim)

"""
    jump(arg, dim_val::Val)

The jump across the interfaces along the direction `dim_val`, ``u_{i+1} - u_i``.

`arg` is a mesh, a grid space or a [`VectorElement`](@ref): the first two give the operator
as a sparse matrix, the third applies it and returns a `VectorElement`.

Forwards to [`forward_difference`](@ref), since a jump across an interface and an unscaled
forward difference are the same quantity. The name records which of the two is meant.
"""
@inline jump(arg, dim_val::Val) = forward_difference(arg, dim_val)

"""
    jump!(vₕ, uₕ, dim_val::Val)

The jump across the interfaces along `dim_val`, written into `vₕ`, which is returned.

Forwards to `forward_difference!`, as the allocating form forwards to
`forward_difference`: a jump across an interface and an unscaled forward difference
are the same quantity.
"""
@inline jump!(vₕ, uₕ, dim_val::Val) = forward_difference!(vₕ, uₕ, dim_val)

# The alias surface comes from `@operator_family` like every other family's
# (gpena/Bramble.jl#258). It was written out here for years because the shared generators
# put a direction word into the docstring ("The `forward` jump along ..."), and a jump has
# no direction to name: it belongs to the interface between two cells, not to a direction of
# travel across it. `opening_sentence` and `bang_opening_sentence` now say what the family
# needs said, so there is nothing left for a second copy of the loop to do.
#
# Folding it in also gives `jumpₕ` the unrolled `Val{2}`/`Val{3}` methods every other
# vectorial alias has. The hand-written version closed over `ntuple(i -> jump(arg, Val(i)),
# Val(D))`, which boxes `i` as a runtime `Int` and so cannot constant-fold into the stencil
# engine -- the shape gpena/Bramble.jl#146 measured on the other families and fixed there.
@operator_family(base=jump,
    stem=jump,
    opening_sentence="The jump across the interfaces along the `{direction}` "*
    "direction, ``\\llbracket u \\rrbracket = u_{i+1} - u_i``.",
    trailing_note="The last point along `{direction}` has no forward neighbour and is "*
    "treated as though it were zero, as in [`diff₊{suffix}`](@ref).",
    bang_opening_sentence="The jump across the interfaces along the `{direction}` "*
                          "direction, ``\\llbracket u \\rrbracket = u_{i+1} - u_i``, "*
                          "written into `vₕ`.",
    vectorial_alias=jumpₕ,
    vectorial_dir_string="",
    vectorial_what="jump")
