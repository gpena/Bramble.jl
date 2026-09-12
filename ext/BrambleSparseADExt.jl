module BrambleSparseADExt

using Bramble:
    Bramble,
    BilinearForm,
    jacobian_pattern,
    ast_sparsity_detector,
    domain,
    interval,
    mesh,
    gridspace,
    form,
    innerₕ
using ADTypes: ADTypes
using PrecompileTools: @setup_workload, @compile_workload

# Wraps a `BilinearForm` and the coefficient dependencies `jacobian_pattern` needs, so the
# pattern can be recomputed on demand rather than materialized once and stored: cheap either
# way (see the timings on `docs/src/examples/poisson_nonlinear.md`), and this way a caller
# never has to remember to rebuild the detector if `a`'s coefficients change identity.
struct ASTSparsityDetector{F<:BilinearForm,D<:Tuple} <: ADTypes.AbstractSparsityDetector
    form::F
    coefficient_dependencies::D
end

function Bramble._ast_sparsity_detector(
    a::BilinearForm, coefficient_dependencies::Function...
)
    return ASTSparsityDetector(a, coefficient_dependencies)
end

# `f`/`x` (or `f!`/`y`/`x`) are the residual and the point it would otherwise be traced at --
# unused here, the same way `ADTypes.KnownJacobianSparsityDetector` ignores them, since the
# pattern is a property of `a`'s AST alone.
function ADTypes.jacobian_sparsity(f, x, sd::ASTSparsityDetector)
    return jacobian_pattern(sd.form, sd.coefficient_dependencies...)
end
function ADTypes.jacobian_sparsity(f!, y, x, sd::ASTSparsityDetector)
    return jacobian_pattern(sd.form, sd.coefficient_dependencies...)
end

# Warms `ast_sparsity_detector` (this extension's only public entry point) and the
# `ADTypes.jacobian_sparsity` methods above, for a scalar and a composite form -- both only
# reachable once `ADTypes` is loaded, so only this extension's own precompile pass, not the
# core package's, ever reaches them.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
        Wₕ = gridspace(Ωₕ)
        a_scalar = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        Vₕ = Wₕ^Val(2)
        a_composite = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))

        @compile_workload begin
            sd_scalar = ast_sparsity_detector(a_scalar)
            ADTypes.jacobian_sparsity(nothing, nothing, sd_scalar)
            sd_composite = ast_sparsity_detector(a_composite)
            ADTypes.jacobian_sparsity(nothing, nothing, sd_composite)
        end
    end
end

end
