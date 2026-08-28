using Bramble
using SparseArrays
using LinearAlgebra

# Resolve domain cross product
import Bramble: ×, normₕ, norm₁ₕ

# ==============================================================================
# Example: 3-Species Coupled Linear Reaction-Diffusion System in 2D
# ==============================================================================
# This example solves a system of 3 coupled PDE variables (u1, u2, u3) defined
# on a hierarchical composite grid space.
#
# The physical equations are:
#   -Δu1 + u1 + u2 = f1   in Ω
#   -Δu2 + u2 + u3 = f2   in Ω
#   -Δu3 + u3 + u1 = f3   in Ω
#
# with homogeneous Dirichlet boundary conditions (u1 = u2 = u3 = 0) on ∂Ω,
# where Ω = [0, 1] × [0, 1].
# ==============================================================================

"""
    custom_symmetrize!(A::SparseMatrixCSC, F::AbstractVector, index_in_marker::BitVector)

Applies Dirichlet boundary conditions to the sparse system matrix `A` and RHS `F` symmetrically.
Prevents adjacent Dirichlet boundary nodes from corrupting each other's RHS values.
"""
function custom_symmetrize!(A::SparseMatrixCSC, F::AbstractVector, index_in_marker::BitVector)
    T = eltype(A)
    rows = rowvals(A)
    vals = nonzeros(A)

    for i in 1:length(index_in_marker)
        if index_in_marker[i]
            dirichlet_val = F[i]
            for k_ptr in nzrange(A, i)
                row_k = rows[k_ptr]
                if !index_in_marker[row_k]
                    F[row_k] -= vals[k_ptr] * dirichlet_val
                end
                vals[k_ptr] = zero(T)
            end
            A[i, i] = one(T)
            F[i] = dirichlet_val
        end
    end
end

"""
    solve_reaction_diffusion(N)

Sets up and solves the 3-species coupled system on an N×N mesh, returning
the L2 relative errors for each component.
"""
function solve_reaction_diffusion(N)
    # 1. Domain and Mesh setup
    I = interval(0.0, 1.0)
    Ω = I × I
    X_domain = domain(Ω, markers(Ω,
        :bottom => x -> x[2] < 1 / N / 2,
        :top => x -> x[2] > 1 - 1 / N / 2,
        :left => x -> x[1] < 1 / N / 2,
        :right => x -> x[1] > 1 - 1 / N / 2))

    Mh = mesh(X_domain, (N, N), (true, true))
    Wh = gridspace(Mh)

    # 2. Setup the Hierarchical Composite Space
    # To use the CoupledBilinearForm framework, the space must be hierarchical.
    # We construct Vh as a 2-component flat space, then cross it with Wh to form
    # a hierarchical space X = (Wh × Wh) × Wh with 3 scalar components.
    Vh = Wh × Wh
    X = Vh × Wh
    n = ndofs(Wh)

    # 3. Manufactured Solutions (chosen to satisfy homogeneous boundary conditions)
    u1_ex(x) = sin(π * x[1]) * sin(π * x[2])
    u2_ex(x) = sin(2π * x[1]) * sin(2π * x[2])
    u3_ex(x) = sin(3π * x[1]) * sin(3π * x[2])

    # Right-hand side functions (calculated by applying the operator to the exact solutions)
    f1_rhs(x) = 2π^2 * sin(π * x[1]) * sin(π * x[2]) + u1_ex(x) + u2_ex(x)
    f2_rhs(x) = 8π^2 * sin(2π * x[1]) * sin(2π * x[2]) + u2_ex(x) + u3_ex(x)
    f3_rhs(x) = 18π^2 * sin(3π * x[1]) * sin(3π * x[2]) + u3_ex(x) + u1_ex(x)

    # 4. Assemble the Coupled Bilinear Form
    # The arguments ((u, p), (v, q)) match the hierarchical block structure:
    #   u = (u[1], u[2]) represents trial functions in Vh,
    #   p represents the trial function in Wh,
    #   v = (v[1], v[2]) represents test functions in Vh,
    #   q represents the test function in Wh.
    a = form(X, X, ((u, p), (v, q)) ->
    # Diffusion terms: -Δu1, -Δu2, -Δu3
        inner₊(∇₋ₕ(u), ∇₋ₕ(v)) + inner₊(∇₋ₕ(p), ∇₋ₕ(q)) +
        # Diagonal linear reaction terms
        innerₕ(u[1], v[1]) + innerₕ(u[2], v[2]) + innerₕ(p, q) +
        # Crossed linear reaction terms
        innerₕ(u[2], v[1]) + innerₕ(p, v[2]) + innerₕ(u[1], q))

    A = assemble(a)

    # 5. Assemble Right Hand Side (F)
    f1_el = element(Wh)
    avgₕ!(f1_el, f1_rhs)
    f2_el = element(Wh)
    avgₕ!(f2_el, f2_rhs)
    f3_el = element(Wh)
    avgₕ!(f3_el, f3_rhs)

    l1 = form(Wh, v -> innerₕ(f1_el, v))
    l2 = form(Wh, v -> innerₕ(f2_el, v))
    l3 = form(Wh, v -> innerₕ(f3_el, v))

    F = [assemble(l1); assemble(l2); assemble(l3)]

    # 6. Apply Boundary Conditions
    boundary_mask = Bramble.index_in_marker(Mh, :left) .| Bramble.index_in_marker(Mh, :right) .| Bramble.index_in_marker(Mh, :bottom) .| Bramble.index_in_marker(Mh, :top)

    global_vec_bool = BitVector(undef, 3n)
    fill!(global_vec_bool, false)
    for i in 1:n
        if boundary_mask[i]
            global_vec_bool[i] = true
            global_vec_bool[n+i] = true
            global_vec_bool[2n+i] = true

            # Dirichlet boundary values are 0.0
            F[i] = 0.0
            F[n+i] = 0.0
            F[2n+i] = 0.0
        end
    end

    Bramble._dirichlet_bc_indices!(A, global_vec_bool)
    custom_symmetrize!(A, F, global_vec_bool)
    dropzeros!(A)

    # 7. Solve
    sol = A \ F

    # 8. Extract solutions and compute errors
    u1_sol = sol[1:n]
    u2_sol = sol[n+1:2n]
    u3_sol = sol[2n+1:end]

    u1_exact = element(Wh)
    Rₕ!(u1_exact, u1_ex)
    u2_exact = element(Wh)
    Rₕ!(u2_exact, u2_ex)
    u3_exact = element(Wh)
    Rₕ!(u3_exact, u3_ex)

    u1_sol_el = element(Wh)
    u1_sol_el.data .= u1_sol
    u2_sol_el = element(Wh)
    u2_sol_el.data .= u2_sol
    u3_sol_el = element(Wh)
    u3_sol_el.data .= u3_sol

    err_u1 = norm₁ₕ(u1_sol_el - u1_exact)
    err_u2 = norm₁ₕ(u2_sol_el - u2_exact)
    err_p = norm₁ₕ(u3_sol_el - u3_exact)

    return hₘₐₓ(Mh), err_u1, err_u2, err_p
end
#=
function main()
    println("=========================================================================")
    println("      COUPLED REACTION-DIFFUSION SYSTEM ON HIERARCHICAL SPACE")
    println("=========================================================================")
    println("| N  | u1 H1 Error | Rate | u2 H1 Error | Rate | u3 H1 Error | Rate |")
    println("|---|---|---|---|---|---|---|")

    grids = [8, 16, 32, 64]
    errs = []
    for N in grids
        push!(errs, solve_reaction_diffusion(N))
    end

    for (i, N) in enumerate(grids)
        hmax, u1_err, u2_err, u3_err = errs[i]
        if i == 1
            println("| $N | $(round(u1_err, sigdigits=4)) |  -   | $(round(u2_err, sigdigits=4)) |  -   | $(round(u3_err, sigdigits=4)) |  -   |")
        else
            u1_rate = round(log2(errs[i-1][2] / u1_err), digits=2)
            u2_rate = round(log2(errs[i-1][3] / u2_err), digits=2)
            u3_rate = round(log2(errs[i-1][4] / u3_err), digits=2)
            println("| $N | $(round(u1_err, sigdigits=4)) | $u1_rate | $(round(u2_err, sigdigits=4)) | $u2_rate | $(round(u3_err, sigdigits=4)) | $u3_rate |")
        end
    end
    println("=========================================================================")
end

main()=#
