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
# A second snoop warms `Rₕ!`, composite `Rₕ!`, masked `Rₕ!` and `avgₕ!` with one function
# each, then calls them with new ones: the split sweeps' `@batch` loop is typed on nothing
# that depends on the user function (BramblePolyesterExt's `_erased_run`), so no instance of
# Polyester or the packages under it may be inferred. It counts by module, not by the
# `_from_main` rule, which would drop every instance whose type holds the new closure.
#
# Prints `REINFER_POLYESTER=<n>`, then the inferred MethodInstances, then
# `REINFER_NEWF_EXT=<n>` (instances of BramblePolyesterExt, the positive control) and
# `REINFER_NEWF_POLYESTER=<n>`.
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

Wm = gridspace(mesh(domain(X, :dir => boundary_symbols(X)), n, nu;
    backend = backend(policy = CpuPolyester())))
v = element(Wm)
vc = element(Wm × Wm)
Rₕ!(v, x -> x[1])
Rₕ!(vc, x -> (x[1], x[2]))
Rₕ!(v, x -> x[1] + 1; markers = (:dir,))
avgₕ!(v, x -> x[2])
h1 = x -> sin(x[1]) + x[2]
h2 = x -> (cos(x[1]), x[1] * x[2])
h3 = x -> x[1] - 2x[2]
h4 = x -> x[1]^2 - x[2]
tinf_newf = nothing
tinf_newf = @snoop_inference begin
    Rₕ!(v, h1)
    Rₕ!(vc, h2)
    Rₕ!(v, h3; markers = (:dir,))
    avgₕ!(v, h4)
end
mods = [string(parentmodule(m.def))
        for m in map(_mi, flatten(tinf_newf))
        if m isa Core.MethodInstance && m.def isa Method]
const POLYESTER_FAMILY = ("Polyester", "PolyesterWeave", "ManualMemory", "StrideArraysCore",
    "ThreadingUtilities")
println("REINFER_NEWF_EXT=", count(==("BramblePolyesterExt"), mods))
println("REINFER_NEWF_POLYESTER=", count(in(POLYESTER_FAMILY), mods))

end # module QualityInvalidationsPolyesterReinferTests
