#=
# buffer.jl

A pool of reusable vectors, so that iterative methods can take scratch space
without allocating on every iteration.

```julia
pool = simple_space_buffer(backend(), n; nbuffers = 2)

for iter in 1:1000
    with_buffer(pool) do tmp
        # `tmp` is an n-element vector owned by the pool
    end                       # returned to the pool automatically
end
```

`with_buffer` is the preferred entry point: the buffer is released even if the
body throws. The lower-level pair is also available when the lifetime cannot be
expressed as a block,

```julia
tmp, key = vector_buffer(pool)
# ...
unlock!(pool, key)
```

but then releasing is the caller's responsibility.

A buffer is tracked by its integer key into `pool.buffer`; `pool.free` is a
stack of keys that are *probably* available, used to avoid scanning the pool on
every acquisition. `in_use` on the buffer itself remains the single source of
truth, so a stale entry on the stack is simply skipped.

!!! warning
    A pool is not thread-safe. Give each thread its own pool rather than
    sharing one, or guard access with a lock.

See also: [`VectorBuffer`](@ref), [`GridSpaceBuffer`](@ref), [`with_buffer`](@ref)
=#

"""
	$(TYPEDEF)

A vector together with a flag recording whether it is currently lent out.

This is a plain wrapper, not an `AbstractVector`: the pool hands callers the
underlying vector via [`vector`](@ref), so the buffer itself is never indexed.

# Fields

$(FIELDS)
"""
mutable struct VectorBuffer{T,VT<:AbstractVector{T}}
	"the underlying vector that holds the data."
	vector::VT
	"whether the buffer is currently lent out."
	in_use::Bool
end

"""
	vector_buffer(b::Backend, n::Int)

Creates a single unlocked [`VectorBuffer`](@ref) of length `n` on backend `b`.
"""
@inline vector_buffer(b::Backend, n::Int) = VectorBuffer{eltype(b),vector_type(b)}(vector(b, n), false)

"""
	in_use(buffer::VectorBuffer)

Returns `true` if the [`VectorBuffer`](@ref) is currently lent out.
"""
@inline in_use(buffer::VectorBuffer) = buffer.in_use

"""
	vector(buffer::VectorBuffer)

Returns the vector held by the [`VectorBuffer`](@ref).
"""
@inline vector(buffer::VectorBuffer) = buffer.vector

"""
	lock!(buffer::VectorBuffer)

Marks the [`VectorBuffer`](@ref) as lent out.
"""
@inline lock!(buffer::VectorBuffer) = (buffer.in_use = true; return)

"""
	unlock!(buffer::VectorBuffer)

Marks the [`VectorBuffer`](@ref) as available.
"""
@inline unlock!(buffer::VectorBuffer) = (buffer.in_use = false; return)

"""
	BufferType{T,VectorType}

The storage backing a [`GridSpaceBuffer`](@ref): a `Vector` of
[`VectorBuffer`](@ref)s indexed by their integer key.

# Type parameters
- `T`: element type of the vectors (e.g. `Float64`).
- `VectorType`: the vector container (e.g. `Vector{Float64}`).
"""
const BufferType{T,VectorType} = Vector{VectorBuffer{T,VectorType}}

"""
	$(TYPEDEF)

A pool of reusable [`VectorBuffer`](@ref)s, all of length `npts`, on a common backend.

# Fields

$(FIELDS)
"""
struct GridSpaceBuffer{BT,VT,T}
	"every [`VectorBuffer`](@ref) in the pool, indexed by its key."
	buffer::BufferType{T,VT}
	"a stack of keys that are probably free; `in_use` remains authoritative."
	free::Vector{Int}
	"the backend the buffers are allocated on."
	backend::BT
	"the length of every vector in the pool."
	npts::Int
end

"""
	simple_space_buffer(b::Backend, npts::Int; nbuffers::Int = 1)

Creates a [`GridSpaceBuffer`](@ref) of `npts`-element vectors on backend `b`,
pre-allocating `nbuffers` of them.

The pool grows on demand, so `nbuffers` is only a warm start.
"""
function simple_space_buffer(b::Backend, npts::Int; nbuffers::Int = 1)
	nbuffers >= 0 || throw(ArgumentError("nbuffers must be non-negative, got $nbuffers."))
	npts >= 0 || throw(ArgumentError("npts must be non-negative, got $npts."))

	T, VT, _, BT = backend_types(b)
	space_buffer = GridSpaceBuffer{BT,VT,T}(BufferType{T,VT}(), Int[], b, npts)

	sizehint!(space_buffer.buffer, nbuffers)
	sizehint!(space_buffer.free, nbuffers)
	for _ in 1:nbuffers
		add_buffer!(space_buffer)
	end

	return space_buffer
end

"""
	add_buffer!(space_buffer::GridSpaceBuffer)

Appends one new, unlocked [`VectorBuffer`](@ref) to the pool and returns
`(vector, key)` for it.
"""
function add_buffer!(space_buffer::GridSpaceBuffer)
	(; buffer, free, backend, npts) = space_buffer

	buf = vector_buffer(backend, npts)
	push!(buffer, buf)
	key = length(buffer)
	push!(free, key)

	return vector(buf), key
end

"""
	nbuffers(space_buffer::GridSpaceBuffer)

Returns the number of buffers in the pool, locked and unlocked alike.
"""
@inline nbuffers(space_buffer::GridSpaceBuffer) = length(space_buffer.buffer)

"""
	lock!(space_buffer::GridSpaceBuffer, i)

Locks the `i`-th buffer and returns its vector.

The key is left on the free stack and skipped when it next comes up, so this
stays O(1).
"""
@inline function lock!(space_buffer::GridSpaceBuffer, i)
	b = space_buffer.buffer[i]
	lock!(b)
	return vector(b)
end

"""
	unlock!(space_buffer::GridSpaceBuffer, i)

Releases the `i`-th buffer back to the pool.

Releasing a buffer that is already free is a no-op, so a double release cannot
put the same key on the free stack twice.
"""
@inline function unlock!(space_buffer::GridSpaceBuffer, i)
	b = space_buffer.buffer[i]
	in_use(b) || return          # already free: nothing to do
	unlock!(b)
	push!(space_buffer.free, i)
	return
end

"""
	vector_buffer(space_buffer::GridSpaceBuffer)

Takes a free vector from the pool, growing it if every buffer is in use, and
returns `(vector, key)`. The buffer is locked on return; pass `key` to
[`unlock!`](@ref) when finished, or prefer [`with_buffer`](@ref), which releases
it for you.
"""
function vector_buffer(space_buffer::GridSpaceBuffer)
	(; buffer, free) = space_buffer

	# Pop candidate keys until one is genuinely free; entries can be stale
	# because lock!(pool, i) does not remove i from the stack.
	while !isempty(free)
		key = pop!(free)
		b = buffer[key]
		if !in_use(b)
			lock!(b)
			return vector(b), key
		end
	end

	# Everything is in use, so grow the pool. add_buffer! pushes the new key
	# onto the free stack; take it straight back off.
	_, key = add_buffer!(space_buffer)
	pop!(free)
	lock!(buffer[key])

	return vector(buffer[key]), key
end

"""
	with_buffer(f, space_buffer::GridSpaceBuffer)

Calls `f(v)` with a vector borrowed from the pool and returns its result,
releasing the buffer afterwards even if `f` throws.

# Examples

```julia
total = with_buffer(pool) do tmp
    tmp .= 1.0
    sum(tmp)
end
```
"""
function with_buffer(f, space_buffer::GridSpaceBuffer)
	v, key = vector_buffer(space_buffer)
	try
		return f(v)
	finally
		unlock!(space_buffer, key)
	end
end
