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
# leaves the inner calls dynamic (the prototype of commit f0f2d538). The arrays are numbered
# in the same depth-first field order by both walks.
#===========================================================================#

# Slot `K` of the arrays tuple; `A` is the type of the array split out of it.
struct _BatchSlot{K, A} end

# A split node of type `T`: its fields' skeletons, in field order.
struct _BatchNode{T, F <: Tuple}
    fields::F
end
_BatchNode{T}(fields::F) where {T, F <: Tuple} = _BatchNode{T, F}(fields)

# A plain array whose elements `@batch` can pass by pointer.
_batch_is_slot(@nospecialize(T)) = T isa DataType && T <: Array && isbitstype(eltype(T))

# A mesh the walk reads through `_walk_mesh`: an outer, mutable one.
function _batch_is_mesh(@nospecialize(T))
    return T isa DataType && T <: AbstractMeshType && ismutabletype(T)
end

# A `Ref` coefficient, read once at split time.
function _batch_is_ref(@nospecialize(T))
    return T isa DataType && T <: Base.RefValue && isbitstype(T.parameters[1])
end

# A value split field by field: a concrete immutable struct or tuple that is not isbits.
function _batch_is_node(@nospecialize(T))
    T isa DataType && isconcretetype(T) && isstructtype(T) || return false
    return !ismutabletype(T) && !isbitstype(T)
end

# The type of `_walk_mesh(Ωₕ)` for a mesh `Ωₕ` of type `T`. The `Mesh1D` method names the
# mesh's parameters so `Type{Union{}}` cannot reach its `fieldtype`, and a state is its own,
# as `_walk_mesh(s::Mesh1DState)` is: inference reaches both from a mesh's state, where
# `_batch_is_mesh` is false at run time but not to JET.
function _batch_state_type(::Type{Mesh1D{BT, CI, VT, T}}) where {BT, CI, VT, T}
    return fieldtype(Mesh1D{BT, CI, VT, T}, :state)
end
_batch_state_type(::Type{T}) where {T <: Mesh1DState} = T
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
function _batch_split_expr!(pre::Vector{Any}, leaves::Vector{Any}, @nospecialize(T),
        @nospecialize(ex), path::String)
    if _batch_is_slot(T)
        push!(leaves, ex)
        return :(_BatchSlot{$(length(leaves)), $T}())
    elseif _batch_is_mesh(T)
        s = gensym(:state)
        push!(pre, :($s = _walk_mesh($ex)))
        return _batch_split_expr!(pre, leaves, _batch_state_type(T)::Type, s, path)
    elseif _batch_is_ref(T)
        return :($ex[])
    elseif isbitstype(T)
        return ex
    elseif _batch_is_node(T)
        fs = Any[]
        for i in 1:fieldcount(T)
            p = string(path, ".", T <: Tuple ? i : fieldname(T, i))
            F, fex = fieldtype(T, i), :(getfield($ex, $i))
            push!(fs, _batch_split_expr!(pre, leaves, F, fex, p))
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
  mutable struct, a field of abstract type), so a new value that would box fails loudly. A hook
  that must accept such a value asks [`_batch_splittable`](@ref) first and, on `false`,
  captures it whole, which boxes.
"""
@generated function _batch_split(x)
    pre, leaves = Any[], Any[]
    sk = _batch_split_expr!(pre, leaves, x, :x, "x")
    return Expr(:block, Expr(:meta, :inline), pre..., :(($sk, ($(leaves...),))))
end

# The slots' array types: the type each original array type rebuilds as, for the type
# parameters; a field a misfit leaves is caught by its declared type.
function _batch_slot_types!(sub::IdDict{Any, Any}, @nospecialize(S), @nospecialize(A))
    if S <: _BatchSlot
        K, old = S.parameters
        sub[old] = fieldtype(A, K)
    elseif S <: _BatchNode
        for F in fieldtypes(S.parameters[2])
            _batch_slot_types!(sub, F, A)
        end
    end
    return sub
end

# What a value of type `T` becomes after the round trip, given the slots' array types `sub`.
function _batch_ptype(@nospecialize(T), sub::IdDict{Any, Any})
    haskey(sub, T) && return sub[T]
    _batch_is_mesh(T) && return _batch_ptype(_batch_state_type(T), sub)
    _batch_is_ref(T) && return T.parameters[1]
    _batch_is_node(T) || return T
    if T <: Tuple
        Fs = Any[]
        for F in fieldtypes(T)
            push!(Fs, _batch_ptype(F, sub))
        end
        return Tuple{Fs...}
    end
    isempty(T.parameters) && return T
    P = Any[]
    for p in T.parameters
        push!(P, p isa Type ? _batch_ptype(p, sub) : p)
    end
    return T.name.wrapper{P...}
end

# Whether `_batch_split_expr!` takes a value of type `T` apart without reaching
# `_throw_batch_resistor`, by the same leaf rules, recording in `sub` a stand-in for the
# array type each slot rebuilds as.
function _batch_splits!(sub::IdDict{Any, Any}, @nospecialize(T))
    if _batch_is_slot(T)
        sub[T] = _BatchStandIn{eltype(T), ndims(T)}
        return true
    end
    (_batch_is_ref(T) || isbitstype(T)) && return true
    _batch_is_mesh(T) && return _batch_splits!(sub, _batch_state_type(T))
    _batch_is_node(T) || return false
    for i in 1:fieldcount(T)
        _batch_splits!(sub, fieldtype(T, i)) || return false
    end
    return true
end

# An array type no struct field names: a field typed as a concrete `Array` does not take it,
# as it does not take the `PtrArray` a task receives.
struct _BatchStandIn{T, N} <: AbstractArray{T, N} end

# Whether `_batch_rebuild_expr` rebuilds `T` without reaching `_throw_batch_misfit` when its
# arrays come back as the stand-ins `sub`: every rebuilt field fits its declared type.
function _batch_fits(@nospecialize(T), sub::IdDict{Any, Any})
    _batch_is_mesh(T) && return _batch_fits(_batch_state_type(T), sub)
    _batch_is_node(T) || return true
    # A type parameter bound the stand-in breaks (`S{V <: Vector}`) breaks the rebuild too.
    T′ = try
        _batch_ptype(T, sub)
    catch e
        e isa TypeError || rethrow()
        return false
    end
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        T <: Tuple || _batch_ptype(F, sub) <: fieldtype(T′, i) || return false
        _batch_fits(F, sub) || return false
    end
    return true
end

function _batch_splits(@nospecialize(T))
    sub = IdDict{Any, Any}()
    return _batch_splits!(sub, T) && _batch_fits(T, sub)
end

"""
    _batch_splittable(::Type{T}) -> Bool

Whether [`_batch_split`](@ref) takes a value of type `T` apart and [`_batch_rebuild`](@ref)
puts it back together around the arrays a `Polyester.@batch` task receives, rather than
either throwing: `true` when every leaf of `T` takes one of the split's rules and every
struct field's declared type accepts any array in place of the one it held (a field typed
`Vector{T}` does not). Generated, so the answer is a constant of the type and a branch on it
costs the caller nothing. A split hook that meets `false` (an array of `BigFloat`s, a
`BigFloat` coefficient, a struct holding a `Vector{Float64}` field) captures its arguments
whole instead.
"""
@generated _batch_splittable(::Type{T}) where {T} = _batch_splits(T)

# The expression rebuilding the skeleton piece `ex` of type `S`, and its type; `A` is the
# arrays tuple's type.
function _batch_rebuild_expr(@nospecialize(S), @nospecialize(ex), @nospecialize(A),
        sub::IdDict{Any, Any})
    if S <: _BatchSlot
        K = S.parameters[1]
        return :(arrays[$K]), fieldtype(A, K)
    end
    S <: _BatchNode || return ex, S
    T, F = S.parameters
    fs, types = Any[], Any[]
    for i in 1:fieldcount(F)
        f, t = _batch_rebuild_expr(fieldtype(F, i), :(getfield($ex.fields, $i)), A, sub)
        push!(fs, f)
        push!(types, t)
    end
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
