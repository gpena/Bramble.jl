"""
	struct VectorBuffer{T,VT<:AbstractVector{T}} <: AbstractVector{T}
		const vector::VT
		in_use::Bool
	end

A `VBuffer` is a structure that holds a vector of type `VT` and a boolean flag `in_use` indicating whether the buffer is currently being used.
"""
mutable struct VectorBuffer{T,VT<:AbstractVector{T}} <: AbstractVector{T}
	vector::VT
	in_use::Bool
end

@forward VectorBuffer.vector (Base.size, Base.length, Base.firstindex, Base.lastindex, Base.iterate, Base.eltype)

@inline Base.@propagate_inbounds function getindex(u::VectorBuffer, i)
	@unpack vector = u
	@boundscheck checkbounds(vector, i)
	return getindex(vector, i)
end

@inline Base.@propagate_inbounds function setindex!(u::VectorBuffer, val, i)
	@unpack vector = u
	@boundscheck checkbounds(vector, i)
	setindex!(vector, val, i)
	return
end

"""
	vector_buffer(b::Backend, n::Int)

Creates a `VectorBuffer` associated with a backend `b` and a size `n`.
"""
@inline vector_buffer(b::Backend, n::Int) = VectorBuffer{eltype(b),vector_type(b)}(vector(b, n), false)

"""
	is_in_use(buffer::VectorBuffer)

Checks if a `VectorBuffer` is currently in use.
"""
@inline is_in_use(buffer::VectorBuffer) = buffer.in_use

"""
	vector(buffer::VectorBuffer)

Returns the vector stored in a `VectorBuffer`.
"""
@inline vector(buffer::VectorBuffer) = buffer.vector

"""
	lock!(buffer::VectorBuffer)

Marks a `VectorBuffer` as in use.
"""
@inline lock!(buffer::VectorBuffer) = (buffer.in_use = true; return)

"""
	unlock!(buffer::VectorBuffer)

Marks a `VectorBuffer` as not in use.
"""
@inline unlock!(buffer::VectorBuffer) = (buffer.in_use = false; return)

BufferType{T,VectorType} = OrderedDict{Int,VectorBuffer{T,VectorType}}

"""
	struct GridSpaceBuffer{BT,VT}
		buffer::BufferType{VT}
		backend::BT
		npts::Int
	end

A `GridSpaceBuffer` manages a collection of `VectorBuffer`s for a given backend and number of points.
"""
struct GridSpaceBuffer{BT,VT,T} <: BrambleType
	buffer::BufferType{T,VT}
	backend::BT
	npts::Int
end

"""
	create_simple_space_buffer(b::Backend, npts::Int; nbuffers::Int = 0)

Creates a `GridSpaceBuffer` with an initial number of buffers.
"""
function create_simple_space_buffer(b::Backend, npts::Int; nbuffers::Int = 1)
	@assert nbuffers >= 0

	T, VT, _, BT = backend_types(b)
	space_buffer = GridSpaceBuffer{BT,VT,T}(BufferType{T,VT}(), b, npts)

	for _ in 1:nbuffers
		add_buffer!(space_buffer)
	end

	return space_buffer
end

"""
	add_buffer!(space_buffer::GridSpaceBuffer)

Adds a new `VectorBuffer` to a `GridSpaceBuffer`.
"""
function add_buffer!(space_buffer::GridSpaceBuffer)
	@unpack buffer, backend, npts = space_buffer

	n = length(buffer) + 1
	buffer[n] = vector_buffer(backend, npts)

	return vector(buffer[n]), n
end

"""
	nbuffers(space_buffer::GridSpaceBuffer)

Returns the number of buffers in a `GridSpaceBuffer`.
"""
@inline function nbuffers(space_buffer::GridSpaceBuffer)
	@unpack buffer = space_buffer

	return length(buffer)
end

"""
	lock!(space_buffer::GridSpaceBuffer, i)

Locks the `i`-th buffer in a `GridSpaceBuffer` and returns the associated vector.
"""
@inline function lock!(space_buffer::GridSpaceBuffer, i)
	@unpack buffer = space_buffer
	b = buffer[i]

	lock!(b)
	return vector(b)
end

"""
	unlock!(space_buffer::GridSpaceBuffer, i)

Unlocks the `i`-th buffer in a `GridSpaceBuffer`.
"""
@inline function unlock!(space_buffer::GridSpaceBuffer, i)
	@unpack buffer = space_buffer
	b = buffer[i]

	unlock!(b)
	return
end

"""
	get_vector_buffer(space_buffer::GridSpaceBuffer)

Retrieves a free vector buffer from a `GridSpaceBuffer`, locking it and returning the associated vector and key. If no free buffers are available, a new one is added.
"""
function get_vector_buffer(space_buffer::GridSpaceBuffer)
	@unpack buffer = space_buffer

	key_free_buffer = 0
	for (key, buf) in buffer
		if !is_in_use(buf)
			key_free_buffer = key
			break
		end
	end

	key_to_lock = 0
	if key_free_buffer == 0
		new_key = 0
		_, new_key = add_buffer!(space_buffer)
		key_to_lock = new_key
	else
		key_to_lock = key_free_buffer
	end

	return lock!(space_buffer, key_to_lock), key_to_lock
end