module FormCompileAssemblyTests

using Test

# #763 step 3. The first `assemble` of 3D elasticity compiled 2611 method instances
# module-wide at a70e3159 (2643 on 8065ffb3), most of them per term type in the block and
# pair walks, the marker scans, the type-level pair plan and the element-type fold through
# every prefix of the sum. The bound is three quarters of 2611. Under the nightly's flags
# (`--optimize=1 --check-bounds=yes --inline=yes`) the count was 1924 at 9c34251c and 1977
# at bda0619d, over the bound: a no-op `_check_product_reads` off a ghost space (29119591),
# a `_walk_space(sp, ::Type{T})` on a plain space (e2e8df7f) and a forwarding
# `_same_stencil_shape` (6db0596d) each specialised once per term, 21 apiece. With the
# three removed it is 1914 (#855). Counting method instances is a deterministic proxy for
# the inference half of compile time, which a shared runner cannot time. It runs in a
# fresh process, so earlier test files cannot have compiled any of them already. The matrix
# is checked against `nnz` and `norm` pinned to what 8bf56e95, before this change,
# assembles, and against `assemble_parallel!`'s matrix, whose threaded replay evaluates
# each point through `_replay_point!`, not the serial walk.
# `matrix_free_operator` would not do: it walks through `visit_bilinear_stencil` too.
const ELASTICITY_COUNT = raw"""
using Bramble, Random, LinearAlgebra, SparseArrays
nspec(f) = sum(m -> count(!isnothing, Base.specializations(m)), methods(f); init = 0)
fns = [getfield(Bramble, n) for n in names(Bramble; all = true)
       if isdefined(Bramble, n) && getfield(Bramble, n) isa Function]
total() = sum(nspec, fns)
Random.seed!(20260903)
Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 5, 5), (false, false, false))
Vₕ = gridspace(Ωₕ)^Val(3)
μ, λ = 1.0, 2.0
a = form(Vₕ, Vₕ, (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v)))
c0 = total()
A = assemble(a)
n = total() - c0
B = copy(A)
fill!(nonzeros(B), 0)
Bramble.assemble_parallel!(B, a)
println(n, " ", nnz(A), " ", norm(A), " ", maximum(abs, B - A) / maximum(abs, A))
"""

@testset "Elasticity first assemble bound (#763)" begin
    project = Base.active_project()
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$project
        -e $ELASTICITY_COUNT`
    out = split(readchomp(cmd))
    n = parse(Int, out[1])
    @test n <= 1958
    @test parse(Int, out[2]) == 19773
    @test isapprox(parse(Float64, out[3]), 92.46818171109827; rtol = 1e-12)
    @test parse(Float64, out[4]) < 1e-12
end

end
