##############################################################################
#                                                                            #
#                  Shared stencil traversal and alias framework              #
#                                                                            #
##############################################################################

#=
# stencil.jl

The one-sided grid traversal, argument validation and alias/docstring generators that
`difference.jl`, `average.jl`, `jump.jl` and `inner_product.jl` all build on.

Extracted from `difference.jl` (gpena/Bramble.jl#42): none of this is about differences in
particular, but until now `average.jl` and `jump.jl` had to depend on the largest file in
`src/`, about a different operator family, because there was nowhere better to point.
Included before all four consumers; Julia does not care which file a definition lives in
within one module, so this is a pure move with no runtime change.

What stays behind in `difference.jl` instead: everything actually specific to differencing
-- `CenteredStencil`/`Centered`/`CrossWeighted` (only the centered/cross-weighted families
use them), `_difference_engine!` and `_centered_stencil_ranges`, `_apply_stencil!` and
`_apply_spaced!` (the difference-family applicator, distinct from the generic
`_apply_componentwise!` below), and `_throw_centered_too_few_points` (the three-point
precondition, specific to centered stencils).
=#

# --- Direction traits -------------------------------------------------------------- #
abstract type GridDirection end
struct Forward <: GridDirection end
struct Backward <: GridDirection end

# --- Argument validation shared by every operator ----------------------------------- #
# Thrown rather than asserted: these check caller arguments, and an @assert reports a
# size mismatch as an AssertionError, which is not what a caller should have to catch.
@noinline _throw_stencil_dim_error(dim::Int,
    D::Int) = throw(ArgumentError("the stencil direction must be between 1 and $D, got $dim"))

@noinline function _throw_stencil_size_error(lout::Int, lin::Int, dims)
    throw(DimensionMismatch("out has $lout entries and in has $lin, but the grid $(dims) has $(prod(dims))"))
end

# Every stencil reads a neighbour of the coordinate it writes, and the traversal is a
# single contiguous pass: aliased destination and source would overwrite an entry before
# the interior point that still needs it as a neighbour has been computed, corrupting every
# value downstream of the first write (see `D₋ₓ!` in the docs). Checked with `mightalias`
# rather than `===` so that two distinct `VectorElement`s sharing the same backing array (a
# view, or one built directly on the other's data) are caught too.
@noinline _throw_alias_error() = throw(ArgumentError(
    "destination and source must not alias"))

@inline _check_no_alias(vₕ::VectorElement, uₕ::VectorElement) = Base.mightalias(
    values(vₕ), values(uₕ)) && _throw_alias_error()

# --- Argument handling shared by every operator ------------------------------------- #
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

# `f!` is the single-component applicator; it is called once per *leaf* (`components`
# flattens any nesting), so this needs no component count of its own — `map` over the two
# tuples `components` returns unrolls exactly as the old `ntuple(…, Val(NC))` did, and stays
# correct regardless of how deeply either space nests.
@inline function _apply_componentwise!(f!, vₕ::VectorElement{<:CompositeGridSpace},
        uₕ::VectorElement{<:CompositeGridSpace})
    map(f!, components(vₕ), components(uₕ))
    return nothing
end

# --- Shared one-sided stencil traversal ---------------------------------------------- #
# The difference and the average walk the grid identically: one pass over the interior,
# where every point has a neighbour along the stencil direction, and one over the single
# boundary slice, where it does not. Only the per-point kernel differs, so the traversal
# is written once here and both engines using it (difference.jl, average.jl) share it.

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

# --- Alias and docstring generators -------------------------------------------------- #

"""
    _define_directional_alias!(base_op_name, alias_name, dir_string, suffix,
                               direction_index, what, formula; opening_sentence = "")

Defines `alias_name(vₕ, uₕ)` as `base_op_name(vₕ, uₕ, Val(direction_index))` and attaches a
docstring to it.

The in-place sibling of `_define_directional_alias`. Two generators rather than one
because the shapes differ: the allocating alias takes a single argument that may be a mesh,
a space or a grid function, while this one takes a destination and a source and is only ever
about grid functions.

`opening_sentence`, given non-empty, replaces the generic "The `\$dir_string` `\$what` of
`uₕ` along the `\$suffix` direction, ``\$formula``, written into `vₕ`." the same way it does
for `_define_directional_alias` -- `Dc!`/`Dₕ!` have no backward/forward adjective to put in
`dir_string` either.
"""
function _define_directional_alias!(
        base_op_name, alias_name, dir_string, suffix, direction_index, what, formula;
        opening_sentence::String = "")
    opening = isempty(opening_sentence) ?
              "The `$dir_string` $what of `uₕ` along the `$suffix` direction, " *
              "``$formula``, written into `vₕ`." : opening_sentence
    doc_string = """
        $alias_name(vₕ, uₕ)

    $opening

    The in-place form of [`$(replace(String(alias_name), "!" => ""))`](@ref): it allocates
    nothing, where the allocating form allocates its result. Returns `vₕ`, so it composes:
    `normₕ($alias_name(vₕ, uₕ))`.

    `vₕ` and `uₕ` must be grid functions of the same space, and must not be the same
    object, as every stencil reads neighbours of the target coordinate; aliasing them
    would read values that have already been overwritten.

    Alias for `$base_op_name(vₕ, uₕ, Val($direction_index))`. Accepts a grid function of a
    scalar or of a composite grid space, componentwise on the latter.
    """

    func_def_expr = :(@inline $(alias_name)(vₕ, uₕ) = $(base_op_name)(
        vₕ, uₕ, Val($(direction_index))))
    final_expr = Expr(
        :macrocall, GlobalRef(Core, Symbol("@doc")), nothing, doc_string, func_def_expr)
    Core.eval(@__MODULE__, final_expr)
end

"""
    _define_directional_alias(base_op_name, alias_name, dir_string, suffix,
                              direction_index, what, formula;
                              opening_sentence = "", formula_note = "", alias_note = "",
                              trailing_note = "")

Defines `alias_name(arg)` as `base_op_name(arg, Val(direction_index))` and attaches a
docstring to it.

`what` names the quantity, such as `"finite difference"`, and `formula` is the LaTeX for
it. Both are needed because the four operator families share this generator: describing
every alias as a "difference" would be wrong for the averages, and would
not separate the unscaled difference from the finite difference.

`opening_sentence`, given non-empty, replaces the generic "The `\$dir_string` `\$what`
along the `\$suffix` direction, ``\$formula``." with the caller's own wording: `Dc` and
`Dₕ` have no backward/forward adjective to put in `dir_string` at all.

The three remaining keyword notes are each a sentence the docstring includes only when
given (non-empty), one per insertion point a family may need: `formula_note` follows the
opening sentence (the diff/finite-difference families use this to contrast the two, which
does not apply to an average); `alias_note` follows the `Alias for ...` sentence, before
`arg` is described (`Dₕ` uses this to compare itself with `Dc`); `trailing_note` follows
the description of `arg`, before the closing "Accepts a grid function..." paragraph
(`Dstar₊`, `Dc` and `Dₕ` use this for their truncation/precondition caveats, which differ
in whether a mesh needs at least three points along the direction).
"""
function _define_directional_alias(
        base_op_name, alias_name, dir_string, suffix, direction_index, what, formula;
        opening_sentence::String = "", formula_note::String = "", alias_note::String = "",
        trailing_note::String = "")
    fn = isempty(formula_note) ? "" : " " * formula_note
    an = isempty(alias_note) ? "" : " " * alias_note
    tn = isempty(trailing_note) ? "" : " " * trailing_note
    opening = isempty(opening_sentence) ?
              "The `$dir_string` $what along the `$suffix` direction, ``$formula``.$fn" :
              opening_sentence

    # 1. Construct the docstring content.
    doc_string = """
        $alias_name(arg)

    $opening

    Alias for `$base_op_name(arg, Val($direction_index))`.$an `arg` is a mesh, a grid space
    or a [`VectorElement`](@ref): the first two give the operator as a sparse matrix, the
    third applies it and returns a `VectorElement`.$tn

    Accepts a grid function of a scalar or of a composite grid space. On a composite one
    the operator is applied to each component in turn, and the result is the composite
    grid function whose components are those results.
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
    _define_vectorial_alias(base_op_name, alias_name, dir_string, what; note = "")

Defines the `ₕ` alias that applies `base_op_name` along every coordinate and returns a
tuple, one entry per spatial dimension. On a one-dimensional mesh it returns that single
entry rather than a one-tuple.

The counterpart of `_define_directional_alias` for the tuple-valued aliases
(`∇₋ₕ`, `diff₋ₕ`, `M₋ₕ`). The operator families generated the same three methods
independently before this existed.

`note`, given non-empty, is an extra sentence appended after the worked 2D example --
`∇ₕ` uses this to place itself relative to `∇₋ₕ`/`∇₊ₕ`, a comparison none of the other
vectorial aliases need.
"""
function _define_vectorial_alias(
        base_op_name, alias_name, dir_string, what; note::String = "")
    n = isempty(note) ? "" : " " * note
    doc_string = """
        $alias_name(arg)

    The $dir_string $what of `arg` along every coordinate, as a tuple with one entry per
    spatial dimension. On a one-dimensional mesh it returns that single entry rather than
    a one-tuple.

    For a 2D space, `$alias_name(uₕ)` is
    `($base_op_name(uₕ, Val(1)), $base_op_name(uₕ, Val(2)))`. `arg` is a mesh, a grid
    space or a [`VectorElement`](@ref), as for `$base_op_name`.$n

    Accepts a grid function of a scalar or of a composite grid space, componentwise on
    the latter: each entry of the tuple is then itself a composite grid function.
    """

    # Built one at a time, as in _define_directional_alias: @doc takes a single
    # definition, and only the entry point carries the docstring.
    entry = :(@inline $(alias_name)(arg) = $(alias_name)(arg, Val(dim(_op_mesh(arg)))))
    one_d = :(@inline $(alias_name)(arg, ::Val{1}) = $(base_op_name)(arg, Val(1)))
    n_d = :(@inline $(alias_name)(arg, ::Val{D}) where {D} = ntuple(
        i -> $(base_op_name)(arg, Val(i)), Val(D)))

    final_expr = Expr(
        :macrocall, GlobalRef(Core, Symbol("@doc")), nothing, doc_string, entry)
    Core.eval(@__MODULE__, final_expr)
    Core.eval(@__MODULE__, one_d)
    Core.eval(@__MODULE__, n_d)
end
