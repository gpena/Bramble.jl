module BrambleVTKSciMLExt

# Writes a `SciMLBase` ODE solution as a ParaView time-series collection. Kept as its own
# extension, triggered only once *both* `WriteVTK` and `SciMLBase` are loaded, rather than
# folded into either `BrambleVTKExt` or `BrambleSciMLExt` -- the same two-weakdep wiring
# `BrambleMeshesExt` already uses for `Meshes` + `MakieCore`. `_vtk_axes`/`_vtk_data`
# (src/exporters/vtk_export.jl) touch no WriteVTK type, so this extension shares them with
# `BrambleVTKExt` directly through `Bramble` instead of depending on that extension.

using Bramble:
               Bramble,
               AbstractSpaceType,
               mesh,
               element,
               export_vtk,
               _vtk_axes,
               _vtk_data,
               domain,
               interval,
               gridspace,
               Rₕ
using WriteVTK: WriteVTK, vtk_grid, paraview_collection
using SciMLBase: SciMLBase, AbstractODESolution
using PrecompileTools: @setup_workload, @compile_workload

# Padded step index, so filenames sort the same way their times do -- cosmetic only,
# WriteVTK's own `relpath` bookkeeping in `collection_add_timestep` does not care.
_step_name(filename::AbstractString, i::Integer, n::Integer) = string(filename, "_", lpad(i, ndigits(n), '0'))

# `times === nothing`: exactly `sol`'s own saved steps, non-uniform and all, each already a
# plain coefficient vector. Given `times` instead, each is sampled from `sol`'s own
# continuous interpolation (`sol(t)`, the same call `heat_equation.jl`'s surface plot uses).
_solution_steps(sol::AbstractODESolution, ::Nothing) = collect(zip(sol.t, sol.u))
_solution_steps(sol::AbstractODESolution, times) = collect(zip(times, (sol(t) for t in times)))

function Bramble._export_vtk_solution(
        filename::AbstractString, Wₕ::AbstractSpaceType, sol::AbstractODESolution;
        name::AbstractString = "u", times = nothing
)
    steps = _solution_steps(sol, times)
    Ωₕ = mesh(Wₕ)
    return paraview_collection(filename) do pvd
        for (i, (t, u)) in enumerate(steps)
            uₕ = element(Wₕ, u)
            vtk = vtk_grid(_step_name(filename, i, length(steps)), _vtk_axes(Ωₕ)...)
            vtk[name] = _vtk_data(uₕ)
            pvd[t] = vtk
        end
    end
end

# Warms `_export_vtk_solution` for both branches of `_solution_steps` above. No solver
# package is available at this dependency level to actually integrate anything -- neither
# `OrdinaryDiffEq` nor any other package that defines `SciMLBase.solve` for an `ODEProblem`
# is a Bramble dependency -- so the "solution" is built by hand with
# `SciMLBase.build_solution(prob, alg, t, u)`, the same constructor solver packages
# themselves call; `alg = nothing` and two saved points are enough for a real, callable
# `ODESolution` (verified: `.t`, `.u` and `sol(t)` all work) without solving anything.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
        Wₕ = gridspace(Ωₕ)
        u0 = collect(parent(Rₕ(Wₕ, x -> x[1])))
        u1 = collect(parent(Rₕ(Wₕ, x -> 2 * x[1])))
        prob = SciMLBase.ODEProblem((du, u, p, t) -> nothing, u0, (0.0, 1.0))
        sol = SciMLBase.build_solution(prob, nothing, [0.0, 1.0], [u0, u1])

        @compile_workload begin
            mktempdir() do dir
                export_vtk(joinpath(dir, "sol1"), Wₕ, sol)
                export_vtk(joinpath(dir, "sol2"), Wₕ, sol; times = (0.0, 0.5, 1.0))
            end
        end
    end
end

end
