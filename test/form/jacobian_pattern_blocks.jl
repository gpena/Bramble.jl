module FormJacobianPatternBlocksTests

using Test
using Bramble
using SparseArrays: sparse, findnz, dropzeros
using Bramble: D₋ₓ, jacobian_pattern

# `jacobian_pattern` on a pair with a composite space on one side and a scalar space on the
# other (gpena/Bramble.jl#367). `assemble` walks such a pair block by block, the scalar side
# as a one-leaf composite; before this the pattern took the scalar path, whose `_walked_leaf`
# picks one whole space and cannot name the component a term reads on the composite side:
# a `πₕ` term threw a MethodError, and a native-only pair returned a pattern short of
# `assemble`'s (31 of 42 entries in 1D) without complaint.
#
# Kept out of jacobian_pattern.jl, which runs only in the slow and AD groups: this needs no
# weak dependency and costs little, so it runs on every push.

_m1(n) = mesh(domain(interval(0.0, 1.0)), n, true)
_m2(n) = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n + 1), (true, false))

_pattern(A) = Set(zip(findnz(sparse(A))[1:2]...))
_live_pattern(A) = _pattern(dropzeros(sparse(A)))

# `Ws` is a scalar space on a second, different mesh, reached only through `πₕ`; `Wt` is the
# space every native term reads.
const _CASES = (
    ("composite trial, πₕ only", (Ws, Wt) -> (Ws × Ws, Wt),
        (U, v) -> innerₕ(πₕ(U(1)), v) + innerₕ(πₕ(U(2)), v)),
    ("composite trial, native only", (Ws, Wt) -> (Wt × Wt, Wt),
        (U, v) -> innerₕ(U(1), v) + inner₊(D₋ₓ(U(2)), D₋ₓ(v))),
    ("composite trial, native and πₕ", (Ws, Wt) -> (Wt × Ws, Wt),
        (U, v) -> inner₊(D₋ₓ(U(1)), D₋ₓ(v)) + innerₕ(πₕ(U(2)), v)),
    ("composite test, native only", (Ws, Wt) -> (Wt, Wt × Wt),
        (u, V) -> innerₕ(u, V(1)) + inner₊(D₋ₓ(u), D₋ₓ(V(2)))),
    ("composite test, native and πₕ", (Ws, Wt) -> (Wt, Wt × Ws),
        (u, V) -> inner₊(D₋ₓ(u), D₋ₓ(V(1))) + innerₕ(u, πₕ(V(2))))
)

@testset "jacobian_pattern: mixed block pairs" begin
    for (mk, ns, nt) in ((_m1, 7, 11), (_m2, 4, 6)), (name, spaces, f) in _CASES

        @testset "$name, $(dim(mk(3)))D" begin
            Wu, Wv = spaces(gridspace(mk(ns)), gridspace(mk(nt)))
            a = form(Wu, Wv, f)
            A = assemble(a)
            P = jacobian_pattern(a)
            @test size(P) == size(A)
            # A cancellation could zero an entry `assemble` still stores: the pattern must
            # match either the stored entries or the live ones.
            @test _pattern(P) == _pattern(A) || _pattern(P) == _live_pattern(A)
        end
    end
end

end # module
