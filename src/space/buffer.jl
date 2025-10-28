"""
	$(TYPEDEF)

A simple, mutable wrapper around a vector, designed to be part of a buffer pool to avoid repeated memory allocations during iterative computations.

The `in_use` flag acts as a simple locking mechanism to track whether the buffer's data is currently being used in a computation.

# Fields

$(FIELDS)
"""
mutable struct VectorBuffer{T,VT<:AbstractVector{T}} <: AbstractVector{T}
	"the underlying vector that holds the data."
	vector::VT
	"a boolean flag indicating if the buffer is currently locked (in use)."
	in_use::Bool
end

#=
The following definitions make a [VectorBuffer](@ref) behave like a standard Julia vector.
- `@forward` delegates common vector methods (like `size`, `length`, `iterate`) directly to the inner `vector` field.
- `getindex` and `setindex!` are implemented to allow direct element access `u[i]`. This improves usability, allowing buffers to be used in place of regular vectors.
=#
@forward VectorBuffer.vector (Base.size, Base.length, Base.firstindex, Base.lastindex, Base.iterate, Base.eltype)

@inline @propagate_inbounds function getindex(u::VectorBuffer, i)
	@unpack vector = u
	@boundscheck checkbounds(vector, i)
	return getindex(vector, i)
end

@inline @propagate_inbounds function setindex!(u::VectorBuffer, val, i)
	@unpack vector = u
	@boundscheck checkbounds(vector, i)
	setindex!(vector, val, i)
	return
end

"""
	vector_buffer(b::Backend, n::Int)

Creates a single [VectorBuffer](@ref) of size `n`, associated with a computational backend `b`. The buffer is initialized as unlocked (`in_use = false`).
"""
@inline vector_buffer(b::Backend, n::Int) = VectorBuffer{eltype(b),vector_type(b)}(vector(b, n), false)

"""
	in_use(buffer::VectorBuffer)

Checks if the [VectorBuffer](@ref) is currently marked as in use (locked).
"""
@inline in_use(buffer::VectorBuffer) = buffer.in_use

"""
	vector(buffer::VectorBuffer)

Returns the underlying vector stored in the [VectorBuffer](@ref).
"""
@inline vector(buffer::VectorBuffer) = buffer.vector

"""
	lock!(buffer::VectorBuffer)

Marks the [VectorBuffer](@ref) as currently in use (locks it).
"""
@inline lock!(buffer::VectorBuffer) = (buffer.in_use = true; return)

"""
	unlock!(buffer::VectorBuffer)

Marks the [VectorBuffer](@ref) as available (unlocks it).
"""
@inline unlock!(buffer::VectorBuffer) = (buffer.in_use = false; return)

# Define a type alias for the ordered dictionary that will hold the buffers.
const BufferType{T,VectorType} = OrderedDict{Int,VectorBuffer{T,VectorType}}

"""
	$(TYPEDEF)

Manages a pool of reusable [VectorBuffer](@ref)s for a specific grid size and backend.

This structure is the core of the buffer management system. It holds an `OrderedDict` of [VectorBuffer](@ref)s, allowing temporary vectors to be efficiently reused, thus minimizing memory allocation during iterative computations.

# Fields

$(FIELDS)
"""
struct GridSpaceBuffer{BT,VT,T}
	"an `OrderedDict` mapping an integer key to each [VectorBuffer](@ref) in the pool."
	buffer::BufferType{T,VT}
	"the computational backend associated with the buffers."
	backend::BT
	"the size (`npts`) of the vectors managed by this buffer pool."
	npts::Int
end

"""
	simple_space_buffer(b::Backend, npts::Int; nbuffers::Int = 0)

Creates a [GridSpaceBuffer](@ref) pool, optionally pre-allocating a number of buffers. "Warming up" the pool by pre-allocating buffers can improve performance on the first few iterations.
"""
function simple_space_buffer(b::Backend, npts::Int; nbuffers::Int = 1)
	# Ensure a non-negative number of buffers is requested.
	@assert nbuffers >= 0 "Number of buffers must be non-negative."

	# Determine the concrete types for the buffer system from the backend.
	T, VT, _, BT = backend_types(b)
	# Create the main buffer pool structure with an empty dictionary.
	space_buffer = GridSpaceBuffer{BT,VT,T}(BufferType{T,VT}(), b, npts)

	# Pre-allocate the requested number of buffers.
	for _ in 1:nbuffers
		add_buffer!(space_buffer)
	end

	return space_buffer
end

"""
	add_buffer!(space_buffer::GridSpaceBuffer)

Dynamically adds one new, available [VectorBuffer](@ref) to the pool. This is called when a request is made for a buffer but all existing ones are in use.
"""
function add_buffer!(space_buffer::GridSpaceBuffer)
	@unpack buffer, backend, npts = space_buffer

	# The key for the new buffer is simply the next integer.
	n = length(buffer) + 1
	buffer[n] = vector_buffer(backend, npts)

	# Return the new vector and its key.
	return vector(buffer[n]), n
end

"""
	nbuffers(space_buffer::GridSpaceBuffer)

Returns the total number of buffers (both locked and unlocked) currently in the pool.
"""
@inline function nbuffers(space_buffer::GridSpaceBuffer)
	@unpack buffer = space_buffer

	return length(buffer)
end

"""
	lock!(space_buffer::GridSpaceBuffer, i)

Locks the `i`-th buffer in the pool and returns the underlying vector for immediate use.
"""
@inline function lock!(space_buffer::GridSpaceBuffer, i)
	@unpack buffer = space_buffer
	b = buffer[i]

	lock!(b)
	# Return the vector for convenient chaining, e.g., `my_vec = lock!(pool, 1)`.
	return vector(b)
end

"""
	unlock!(space_buffer::GridSpaceBuffer, i)

Unlocks the `i`-th buffer in the pool, marking it as available for reuse.
"""
@inline function unlock!(space_buffer::GridSpaceBuffer, i)
	@unpack buffer = space_buffer
	b = buffer[i]

	unlock!(b)
	return
end

"""
	vector_buffer(space_buffer::GridSpaceBuffer)

Retrieves an available vector from the buffer pool.

This is the main function for acquiring a temporary vector. It first searches for any unlocked buffer. If all existing buffers are locked, it transparently allocates a new one and adds it to the pool. The function returns the vector itself and its integer key, which must be used later to `unlock!` it. The buffer is marked as locked upon retrieval.
"""
function vector_buffer(space_buffer::GridSpaceBuffer)
	@unpack buffer = space_buffer

	# Search for the first available (unlocked) buffer.
	key_free_buffer = 0
	for (key, buf) in buffer
		if !in_use(buf)
			key_free_buffer = key
			break # Stop searching once a free one is found.
		end
	end

	# Determine which key to lock.
	key_to_lock = 0
	if key_free_buffer == 0
		# Case 1: No free buffers were found. The pool must grow.
		# Add a new buffer to the pool.
		_, new_key = add_buffer!(space_buffer)
		key_to_lock = new_key
	else
		# Case 2: A free buffer was found.
		key_to_lock = key_free_buffer
	end

	# Lock the chosen buffer and return it along with its key.
	return lock!(space_buffer, key_to_lock), key_to_lock
end