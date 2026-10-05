module FormMarkerIdsTests

using Test
using Random
using SparseArrays
using LinearAlgebra: Diagonal
using Bramble
using Bramble: restrict_to, index_in_marker, local_stencil, IdentityOperator, D₋ₓ,
               CompositeGridSpace, _bind_walk, _is_marked, _marker_id, _marker_words

# Restricted terms are bound, per walked mesh, to that mesh's marker ids, and the walk reads
# one bit of the mesh's word matrix per point instead of a `Dict` (gpena/Bramble.jl#437).
# Every mesh here is non-uniform and has more than 64 points, so a region crosses a word
# boundary, and every assembled matrix is checked against masks from `index_in_marker`.

S = interval(0.0, 1.0) × interval(0.0, 1.0)
blob = x -> (x[1] - 0.4)^2 + (x[2] - 0.5)^2 < 0.1
build_mesh() = (Random.seed!(437);
    mesh(domain(S, :dir => :left, :top => :top, :blob => blob), (23, 7), (false, false)))
mask(Ωₕ, label) = collect(index_in_marker(Ωₕ, label))
asm(Wₕ, f) = Matrix(assemble(form(Wₕ, Wₕ, f)))
diag_weights(Wₕ) = (M = asm(Wₕ, (u, v) -> innerₕ(u, v)); [M[i, i] for i in 1:ndofs(Wₕ)])

@testset "Marker ids" begin
    Ωₕ = build_mesh()
    Wₕ = gridspace(Ωₕ)
    n = ndofs(Wₕ)
    w = diag_weights(Wₕ)
    @test n > 64

    @testset "Word reads match the stored masks" begin
        words = _marker_words(Ωₕ)
        for (label, bits) in markers(Ωₕ)
            id = _marker_id(Ωₕ, label)
            @test [_is_marked(words, id, i) for i in 1:n] == collect(bits)
        end
        ids = (_marker_id(Ωₕ, :dir), _marker_id(Ωₕ, :blob))
        @test [_is_marked(words, ids, i) for i in 1:n] == (mask(Ωₕ, :dir) .| mask(Ωₕ, :blob))
    end

    @testset "Bound regions are ids" begin
        id = IdentityOperator(Wₕ)
        bound, words = _bind_walk(restrict_to((:dir, :blob), D₋ₓ(id)), Ωₕ)
        @test bound.region === (_marker_id(Ωₕ, :dir), _marker_id(Ωₕ, :blob))
        @test isbits(bound.region)
        @test words === _marker_words(Ωₕ)
        nested, _ = _bind_walk(D₋ₓ(restrict_to(:blob, id)), Ωₕ)
        @test nested.inner_op.region === _marker_id(Ωₕ, :blob)
        plain = D₋ₓ(id)
        @test _bind_walk(plain, Ωₕ) === (plain, nothing)
        @test_throws ArgumentError _bind_walk(restrict_to(:nowhere, id), Ωₕ)
    end

    @testset "Assembly reads the word bits" begin
        mb, md, mt = mask(Ωₕ, :blob), mask(Ωₕ, :dir), mask(Ωₕ, :top)
        @test 0 < count(mb) < n
        @test asm(Wₕ, (u, v) -> innerₕ(u, v; markers = (:blob,))) == Diagonal(mb .* w)
        @test asm(Wₕ, (u, v) -> innerₕ(restrict_to((:top, :dir), u), v)) ==
              Diagonal((mt .| md) .* w)
        fsrc = x -> 1.0 + x[1]
        b0 = assemble(form(Wₕ, v -> innerₕ(fsrc, v)))
        @test assemble(form(Wₕ, v -> innerₕ(fsrc, v; markers = (:blob,)))) == mb .* b0
    end

    @testset "Each leaf binds its own table" begin
        # Labels sorted differently, so `:dir` has a different column on each mesh.
        Ω2 = mesh(domain(S, :aa => (x -> x[1] > 0.5), :ab => (x -> x[2] > 0.5), :dir => :right),
            (9, 8), (false, true))
        @test _marker_id(Ω2, :dir) != _marker_id(Ωₕ, :dir)
        W2 = gridspace(Ω2)
        V = CompositeGridSpace((Wₕ, W2))
        A = Matrix(assemble(form(V, V,
            (u, v) -> innerₕ(u(1), v(1); markers = (:dir,)) +
                      innerₕ(u(2), v(2); markers = (:dir,)))))
        R = blockdiag(sparse(Diagonal(mask(Ωₕ, :dir) .* w)),
            sparse(Diagonal(mask(Ω2, :dir) .* diag_weights(W2))))
        @test A == Matrix(R)
    end

    @testset "Label change after the form is built" begin
        # Decided: the ids are bound at every walk, so a form built before a label change
        # assembles against the new labels, and a label removed since throws.
        Ωc = build_mesh()
        Wc = gridspace(Ωc)
        a = form(Wc, Wc, (u, v) -> innerₕ(u, v; markers = (:blob,)))
        @test Matrix(assemble(a)) == Diagonal(mask(Ωc, :blob) .* w)
        old_id = _marker_id(Ωc, :blob)
        relabelled = copy(markers(Ωc))
        moved = .!relabelled[:blob]
        relabelled[:blob] = moved
        relabelled[:aaa] = relabelled[:top]  # sorts first: every other id moves up one
        Bramble.markers!(Ωc, relabelled)
        new_id = _marker_id(Ωc, :blob)
        @test new_id == old_id + 1
        @test _marker_words(Ωc)[:, old_id] != _marker_words(Ωc)[:, new_id]
        @test Matrix(assemble(a)) == Diagonal(moved .* w)
        delete!(relabelled, :blob)
        Bramble.markers!(Ωc, relabelled)
        @test_throws ArgumentError assemble(a)
    end

    @testset "Label added in place is rejected" begin
        # Unsupported: `markers(Ωₕ)` is a read-only view. The label passes validation (it is
        # in the dictionary) but has no id, and the error names the supported route.
        Ωp = build_mesh()
        Wp = gridspace(Ωp)
        markers(Ωp)[:new] = copy(markers(Ωp)[:top])
        a = form(Wp, Wp, (u, v) -> innerₕ(u, restrict_to(:new, v)))
        err = try
            assemble(a)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("in place", err.msg)
        @test occursin("markers!", err.msg)
    end

    @testset "No table: interior is everything" begin
        id = IdentityOperator(Wₕ)
        I = first(Bramble.indices(Ωₕ))
        @test local_stencil(restrict_to(:interior, id), Wₕ, I, nothing, 1) ==
              local_stencil(id, Wₕ, I, nothing, 1)
        @test local_stencil(restrict_to(:blob, id), Wₕ, I, nothing, 1) == ()
    end
end

end # module
