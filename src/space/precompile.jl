# NOTE: this file is not currently included by src/Bramble.jl (space/ is commented
# out there). Fold these calls into src/precompile.jl when space/ is re-enabled.

# --- Buffers ------------------------------------------------------------- #
@compile_workload begin
    precompile_backend = backend()
    precompile_vector_len = 10
    precompile_num_buffers = 2

    # --- VectorBuffer Operations ---
    vb = vector_buffer(precompile_backend, precompile_vector_len)
    in_use(vb)
    v = vector(vb)
    lock!(vb)
    unlock!(vb)

    # --- GridSpaceBuffer Operations ---
    gsb = simple_space_buffer(precompile_backend, precompile_vector_len; nbuffers = precompile_num_buffers)
    nbuffers(gsb)

    # Ensure at least one buffer exists for lock/unlock/get
    if nbuffers(gsb) == 0
        add_buffer!(gsb)
    end

    local key1::Int # Ensure type stability
    local vec1 # Type will be inferred from vector_buffer
    vec1, key1 = vector_buffer(gsb) # This locks the buffer

    # Need another buffer to test lock/unlock on a specific key if vector_buffer added one
    if nbuffers(gsb) < 2
        add_buffer!(gsb)
    end
    key_to_test = 1 == key1 ? 2 : 1 # Pick a key different from the one locked by vector_buffer

    # Lock/Unlock a specific buffer (that wasn't just locked by vector_buffer)
    lock!(gsb, key_to_test)
    unlock!(gsb, key_to_test)

    # Unlock the one obtained from vector_buffer
    unlock!(gsb, key1)

    # Add another buffer
    add_buffer!(gsb)
end

# --- Grid spaces, vector elements, Rₕ / avgₕ ----------------------------- #
@setup_workload begin
    _interval = interval(0, 1)

    @compile_workload begin
        for D in 1:3
            S = reduce(×, ntuple(i -> _interval, D))
            X = domain(S)
            dims_tuple = ntuple(i -> 3, D)
            unif_tuple = ntuple(i -> true, D)
            test_mesh = mesh(X, dims_tuple, unif_tuple)

            # --- Grid space basics ---
            space_weights(test_mesh)
            Wh = gridspace(test_mesh)
            weights(Wh)
            mesh(Wh)
            eltype(Wh)
            dim(Wh)
            ndofs(Wh)
            weights(Wh, Innerh(), 1)
            weights(Wh, Innerplus(), 1)

            # --- Vector elements ---
            W = gridspace(test_mesh)
            α = 3.0
            β = 2

            # Constructors
            u1 = element(W)
            u2 = element(W, α)

            v_data = deepcopy(values(u2))
            u3 = element(W, v_data)
            u4 = element(W, β) # Int constructor

            # Getters/Setters
            space(u3)
            vals = values(u3)
            values!(u1, vals)

            # Forwarded methods (exercise them)
            length(u3)
            eltype(u3)
            size(u3)
            firstindex(u3)
            lastindex(u3)
            sum(u3) # Test iteration
            mesh(u3)

            # ndims
            ndims(VectorElement)

            # Indexing
            u3[1]
            u3[2] = 99.0
            u3[3] = 99 # Set Int

            # Similar
            similar(u3)

            # Copyto!
            z = element(W)
            copyto!(z, u3)
            copyto!(z, v_data)

            # Arithmetic Operators
            α .+ u3
            u3 .+ α
            u3 + u2
            α * u3
            u3 * α
            u3 .* u2
            u3 - u2
            u3 .- α
            α .- u3
            u3 ./ β
            β ./ u3
            u3 ./ u2
            u3 .^ β
            u3 .^ u2

            # Broadcasting
            w = element(W)
            copyto!(w, Base.broadcasted(identity, u3))
            # materialize! / fused broadcast
            w .= u3 .+ u2 .* α
            w .= β # Scalar assignment
            w .= u3 ./ β

            # Other common operations (example)
            norm(u3)

            # --- Rₕ and avgₕ ---
            u_example = element(W, 1.0)

            axes(u_example)
            to_matrix(u_example)

            f = x -> x[1]
            Rₕ(W, f)
            Rₕ!(u_example, f)

            avgₕ(W, f)
            avgₕ!(u_example, f)
        end
    end
end

# --- Difference / jump / average operators -------------------------------- #
@setup_workload begin
    I0 = interval(0, 1)
    ops(::Val{1}) = (diff₋ₓ, diff₊ₓ, D₋ₓ, D₊ₓ, jump₋ₓ, jump₊ₓ, M₋ₓ, M₊ₓ)

    function ops(::Val{2})
        ops2 = (diff₋ᵧ, diff₊ᵧ, D₋ᵧ, D₊ᵧ, jump₋ᵧ, jump₊ᵧ, M₋ᵧ, M₊ᵧ)
        return (ops2..., ops(Val(1))...)
    end

    function ops(::Val{3})
        ops2 = (diff₋₂, diff₊₂, D₋₂, D₊₂, jump₋₂, jump₊₂, M₋₂, M₊₂)
        return (ops2..., ops(Val(2))...)
    end

    tuple_ops() = (diff₋ₕ, diff₊ₕ, ∇₋ₕ, ∇₊ₕ, jump₋ₕ, jump₊ₕ, M₋ₕ, M₊ₕ)

    @compile_workload begin
        for i in 1:3
            X = domain(reduce(×, ntuple(j -> I0, i)))
            M = mesh(X, ntuple(j -> 4, i), ntuple(j -> false, i))

            Wh = gridspace(M)
            uh = element(Wh)
            wh = element(Wh)
            Uh = elements(Wh)
            Vh = elements(Wh)

            for op in (.-, .*, ./, .+)
                op(uh, wh)
            end

            for op in (.-, .*, ./, .+)
                op(uh, 1.0)
                op(1.0, uh)
            end

            uh .= 1.0
            uh .= 1.0 .* wh .+ wh .- .+wh ./ 1.0

            Rₕ!(uh, x->x[1]), avgₕ!(uh, x->x[1])
            Rₕ(Wh, x->x[1]), avgₕ(Wh, x->x[1])

            Uh.data[1, 1] = 1.0
            Uh .= 1.0
            Uh .= 1.0 .* Vh .+ Vh .- .+Vh ./ 1.0

            uh * Uh
            Uh * uh

            for op in (+, -, *)
                op(Uh, Vh)
            end

            for op in (.+, .-, .*, ./, .^)
                op(1.0, Uh)
                op(Uh, 1.0)
            end

            gen_ops = ops(Val(i))

            for op in gen_ops
                op(Wh), op(uh), op(Uh)
            end

            for tup_ops in tuple_ops()
                tup_ops(Wh), tup_ops(uh), tup_ops(Uh)
            end

            z = ∇₋ₕ(uh)
            normₕ(uh), snorm₁ₕ(uh), norm₁ₕ(uh), norm₊(z)
        end
    end
end
