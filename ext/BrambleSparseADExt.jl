module BrambleSparseADExt

using Bramble: Bramble, BilinearForm, jacobian_pattern
using ADTypes: ADTypes

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

end
