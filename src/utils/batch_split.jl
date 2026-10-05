#===========================================================================#
# The generic split of a walk argument (gpena/Bramble.jl#437 item 4).
#
# `Polyester.@batch` copies what its loop captures into an argument box: plain arrays become
# `PtrArray`s and isbits values are copied, and a box holding only those stays on the stack.
# One GC reference puts it on the heap. A form-carrying loop captures spaces, terms and
# sinks, so it passes them as `_batch_split(x)` instead, an isbits skeleton and a flat tuple
# of plain arrays, and each task calls `_batch_rebuild` on what it receives.
#
# Both are one generated function whose body is the whole walk, written out from the type: a
# generated function recursing through itself per node hits inference's recursion limit and
# leaves the inner calls dynamic (benchmark/batch_form_rebuild.jl, the S2.3 prototype). The
# arrays are numbered in the same depth-first field order by both walks.
#===========================================================================#

# Slot `K` of the arrays tuple; `A` is the type of the array split out of it.
struct _BatchSlot{K, A} end

# A split node of type `T`: its fields' skeletons, in field order.
struct _BatchNode{T, F <: Tuple}
    fields::F
end
_BatchNode{T}(fields::F) where {T, F <: Tuple} = _BatchNode{T, F}(fields)

# A plain array whose elements `@batch` can pass by pointer.
_batch_is_slot(T) = T isa DataType && T <: Array && isbitstype(eltype(T))

# A mesh the walk reads through `_walk_mesh`: an outer, mutable one.
_batch_is_mesh(T) = T isa DataType && T <: AbstractMeshType && ismutabletype(T)

# A `Ref` coefficient, read once at split time.
_batch_is_ref(T) = T isa DataType && T <: Base.RefValue && isbitstype(T.parameters[1])

# A value split field by field: a concrete immutable struct or tuple that is not isbits.
function _batch_is_node(T)
    T isa DataType && isconcretetype(T) && isstructtype(T) || return false
    return !ismutabletype(T) && !isbitstype(T)
end

# The type of `_walk_mesh(Ωₕ)` for a mesh `Ωₕ` of type `T`.
_batch_state_type(::Type{T}) where {T <: Mesh1D} = fieldtype(T, :state)
function _batch_state_type(::Type{MeshnD{D, BT, CI, SM, T}}) where {D, BT, CI, SM, T}
    SM′ = Tuple{map(_batch_state_type, fieldtypes(SM))...}
    return MeshnDState{D, BT, CI, SM′, T, fieldtype(MeshnD, :words)}
end

@noinline function _throw_batch_resistor(T, path::String)
    throw(ArgumentError("_batch_split has no rule for $path::$T: only plain arrays of an " *
                        "isbits element type, meshes, isbits `Ref`s, isbits values and " *
                        "immutable structs of these split"))
end

@noinline function _throw_batch_misfit(T, i::Int, F)
    throw(ArgumentError("_batch_rebuild cannot rebuild $T: field $i would hold a $F, " *
                        "which its declared type does not accept"))
end

# Appends to `pre` the statements the split needs before its result (one `_walk_mesh` per
# mesh), to `leaves` the expression of each array, and returns the skeleton's expression for
# the value `ex` of type `T`; `path` names the value in an error.
function _batch_split_expr!(pre, leaves, T, ex, path)
    if _batch_is_slot(T)
        push!(leaves, ex)
        return :(_BatchSlot{$(length(leaves)), $T}())
    elseif _batch_is_mesh(T)
        s = gensym(:state)
        push!(pre, :($s = _walk_mesh($ex)))
        return _batch_split_expr!(pre, leaves, _batch_state_type(T), s, path)
    elseif _batch_is_ref(T)
        return :($ex[])
    elseif isbitstype(T)
        return ex
    elseif _batch_is_node(T)
        fs = map(1:fieldcount(T)) do i
            p = "$path.$(T <: Tuple ? i : fieldname(T, i))"
            return _batch_split_expr!(pre, leaves, fieldtype(T, i), :(getfield($ex, $i)), p)
        end
        return :(_BatchNode{$T}(($(fs...),)))
    end
    return :(_throw_batch_resistor($T, $path))
end

"""
    _batch_split(x) -> (skeleton, arrays::Tuple)

Split `x` into an isbits `skeleton` and the flat tuple `arrays` of the plain arrays it holds,
so a `Polyester.@batch` loop can capture both without a heap-allocated argument box;
[`_batch_rebuild`](@ref) is the inverse. `x` is any walk argument, typically the tuple
`(space, term)` of one walk unit.

The walk goes through every field of every immutable struct and tuple in `x`, and each leaf
takes one rule:

  - an `Array` of an isbits element type becomes a slot of `arrays`;
  - a mutable mesh is replaced by its walk state [`_walk_mesh`](@ref), which is split in
    turn;
  - a `Base.RefValue` of an isbits value is replaced by that value, read once at split time
    on the host: a `Ref` coefficient is read per call at launch, not per point;
  - an isbits value is kept.

# Throws
- `ArgumentError`: naming the type and the path of any other value (a `Dict`, a `Symbol`, a
  mutable struct, a field of abstract type), so a new value that would box fails loudly.
"""
@generated function _batch_split(x)
    pre, leaves = Any[], Any[]
    sk = _batch_split_expr!(pre, leaves, x, :x, "x")
    return Expr(:block, Expr(:meta, :inline), pre..., :(($sk, ($(leaves...),))))
end

# The slots' array types: the type each original array type rebuilds as, for the type
# parameters; a field a misfit leaves is caught by its declared type.
function _batch_slot_types!(sub, S, A)
    if S <: _BatchSlot
        K, old = S.parameters
        sub[old] = fieldtype(A, K)
    elseif S <: _BatchNode
        foreach(F -> _batch_slot_types!(sub, F, A), fieldtypes(S.parameters[2]))
    end
    return sub
end

# What a value of type `T` becomes after the round trip, given the slots' array types `sub`.
function _batch_ptype(T, sub)
    haskey(sub, T) && return sub[T]
    _batch_is_mesh(T) && return _batch_ptype(_batch_state_type(T), sub)
    _batch_is_ref(T) && return T.parameters[1]
    _batch_is_node(T) || return T
    T <: Tuple && return Tuple{(_batch_ptype(F, sub) for F in fieldtypes(T))...}
    isempty(T.parameters) && return T
    P = map(p -> p isa Type ? _batch_ptype(p, sub) : p, Tuple(T.parameters))
    return T.name.wrapper{P...}
end

# The expression rebuilding the skeleton piece `ex` of type `S`, and its type; `A` is the
# arrays tuple's type.
function _batch_rebuild_expr(S, ex, A, sub)
    if S <: _BatchSlot
        K = S.parameters[1]
        return :(arrays[$K]), fieldtype(A, K)
    end
    S <: _BatchNode || return ex, S
    T, F = S.parameters
    rs = [_batch_rebuild_expr(fieldtype(F, i), :(getfield($ex.fields, $i)), A, sub)
          for i in 1:fieldcount(F)]
    fs, types = first.(rs), last.(rs)
    T′ = _batch_ptype(T, sub)
    T <: Tuple && return :(($(fs...),)), T′
    for i in eachindex(types)
        types[i] <: fieldtype(T′, i) || return :(_throw_batch_misfit($T′, $i, $(types[i]))), T′
    end
    return Expr(:new, T′, fs...), T′
end

"""
    _batch_rebuild(skeleton, arrays::Tuple) -> rebuilt

The value [`_batch_split`](@ref)`(x)` split, rebuilt around `arrays`: the same struct types,
with each array type parameter replaced by the type of the array now in its slot, each mesh
by its walk state and each `Ref` by its value. `arrays` may hold any array type, such as the
`PtrArray`s `@batch` passes, or the plain arrays it passes a single-iteration loop, provided
every array split from one original array type arrives as one array type: the type
parameters are substituted per original type, not per slot. `@batch` converts all arrays of
one type alike, so it always meets this.

# Throws
- `ArgumentError`: if a struct's declared field type does not accept the rebuilt field, as
  when two arrays of one original type arrive as different array types.
"""
@generated function _batch_rebuild(skeleton, arrays::Tuple)
    sub = _batch_slot_types!(IdDict{Any, Any}(), skeleton, arrays)
    ex = first(_batch_rebuild_expr(skeleton, :skeleton, arrays, sub))
    return Expr(:block, Expr(:meta, :inline), ex)
end
