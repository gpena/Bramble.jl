module QualityInvalidationsPolyesterReinferTests

#===========================================================================#
# Run in a fresh process with --threads=2 (never `include`d into a session
# that has already loaded Polyester). After `using Bramble, Polyester`, the
# extension workload's call forms (the matrix-free product and the
# other first-call paths of benchmark/polyester_first_call.jl, but its `_newf` rows)
# must not trigger inference of anything CpuPolyester-typed or defined in
# BramblePolyesterExt: a cached method that Polyester's load invalidated would be
# re-inferred here.
#
# Prints `REINFER_POLYESTER=<n>`, then the inferred MethodInstances.
#===========================================================================#

using Bramble, Polyester, LinearAlgebra
using Bramble: CpuPolyester
using SnoopCompileCore
Threads.nthreads() >= 2 || error("run with --threads >= 2")
X = interval(0.0, 1.0) × interval(0.0, 1.0)
n = (5, 4)
nu = (false, false)
W = gridspace(mesh(domain(X), n, nu))
Wp = gridspace(mesh(domain(X), n, nu; backend = backend(policy = CpuPolyester())))
f(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
x = ones(ndofs(Wp))
y = similar(x)
# The same shapes as the extension workload's functions: static inference of the call sites
# in a function body is what a user's first call pays for (see BramblePolyesterExt).
function first_calls(u, w, g, a, l, x, y)
    avgₕ!(u, g)
    innerₕ(u, w)
    w .= 2.0 .* u .+ w
    A = Bramble.allocate_system_matrix(a)
    assemble!(A, a)
    assemble!(A, a)
    Bramble.semidiscretize_rhs(semidiscretize(a, l))(y, x, nothing, 0.0)
    return nothing
end
kron_call(a, x, y) = (mul!(y, kronecker_operator(a), x); nothing)
mf_call(a, x, y) = (mul!(y, matrix_free_operator(a), x); nothing)
# Bound first: `@snoop_inference` assigns in a branch JET cannot see is always taken.
tinf = nothing
tinf = @snoop_inference begin
    mul!(y, matrix_free_operator(f(W); policy = CpuPolyester()), x)
    mul!(y, matrix_free_operator(f(Wp)), x)
    op = matrix_free_operator(f(Wp); dirichlet = :boundary)
    mul!(y, op, x)
    mul!(y, op, x, 0.5, 2.0)
    # The other paths the workload covers (benchmark/polyester_first_call.jl), called from
    # inside functions as the workload does: user code is inferred statically.
    g(x) = sum(x)
    u = Rₕ(Wp, g)
    w = Rₕ(Wp, x -> x[1])
    a = f(Wp)
    l = form(Wp, v -> innerₕ(u, v))
    first_calls(u, w, g, a, l, x, y)
    kron_call(a, x, y)
    mf_call(a, x, y)
end
using SnoopCompile
_mi(t) = (d = t.ci.def; d isa Core.MethodInstance ? d : d.def)
mis = [m for m in map(_mi, flatten(tinf)) if m isa Core.MethodInstance && m.def isa Method]
# A type defined in Main or in this wrapper module (the user's form closure) is new to
# every session, so its inference is never a cache miss; only instances built entirely from
# package types count.
_from_main(T) = false
_from_main(T::UnionAll) = _from_main(Base.unwrap_unionall(T))
_from_main(T::Union) = _from_main(T.a) || _from_main(T.b)
_from_main(T::DataType) = parentmodule(T) in (Main, @__MODULE__) || any(_from_main, T.parameters)
function hit(mi)
    !_from_main(mi.specTypes) &&
        (occursin("CpuPolyester", string(mi.specTypes)) ||
         startswith(string(parentmodule(mi.def)), "BramblePolyesterExt"))
end
bad = filter(hit, mis)
println("REINFER_TOTAL=", length(mis))
for mi in mis
    println("  ", first(string(mi), 160))
end
println("REINFER_POLYESTER=", length(bad))

end # module QualityInvalidationsPolyesterReinferTests
