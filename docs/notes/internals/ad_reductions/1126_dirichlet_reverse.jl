# Reproducer for gpena/Bramble.jl#1126: Enzyme reverse mode through the Dirichlet part of a
# 2D `assemble(l; dirichlet)` aborts the process with
#
#     Assertion failed: (false && "caching potentially capturing/offset of allocation"),
#     function needsCacheWholeAllocation, file .../Enzyme/GradientUtils.cpp, line 10570.
#
# The allocation is the tuple `boundary_indices(Ωₕ)` that `_geometric_face_bits`
# (src/mesh/interface.jl) builds on the stack and then indexes with `2axis - 2 + side`, a
# run-time value.
#
# Run from the repository root, with Bramble from the root project:
#
#     julia --project=. --threads=1 --optimize=1 --check-bounds=auto \
#         docs/notes/internals/ad_reductions/1126_dirichlet_reverse.jl [variant]
#
# Enzyme is not a dependency of the root project, so it is loaded from a temporary
# environment, the way the test harness does. The root Manifest.toml must hold Enzyme (it
# is a weak dependency of Bramble).
#
# Exit status: nonzero (134, SIGABRT) exactly when the abort reproduces. If Enzyme raises an
# exception instead, or returns a gradient, the script prints which and exits 0.
#
# Variants (the first is the default):
#   apply         `apply_dirichlet_conditions!` on a vector, with the linear form built inside
#                 the differentiated function, 2D, serial policy. Aborts.
#   assemble      the call from the issue, `assemble(a, l; dirichlet)` in 2D. Aborts.
#   apply1d       `apply` on a 1D mesh. No abort.
#   apply3d       `apply` on a 3D mesh. No abort.
#   assemble3d    `assemble(a, l; dirichlet)` on a 3D mesh. See the note.
#   fixed         `apply` after redefining `_geometric_face_bits` and `_marks_only_face` to build
#                 the one face they need, with no tuple of faces to index. No abort.
#   fixed_assemble  `assemble` with the same two redefinitions. No abort, and a gradient.
_e = mktempdir()
write(joinpath(_e, "Project.toml"), "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n")
push!(LOAD_PATH, _e)
using Bramble, Enzyme, SparseArrays

const VARIANT = isempty(ARGS) ? "apply" : only(ARGS)
const DIM = endswith(VARIANT, "1d") ? 1 : endswith(VARIANT, "3d") ? 3 : 2
const NP = 3

if startswith(VARIANT, "fixed")
    # What a Bramble fix would be: the face as one `CartesianIndices`, not the entry of an
    # `NTuple{2D, CartesianIndices}` picked by a run-time index. Same faces, same order.
    @eval Bramble begin
        @inline function _face_cart(Ωₕ::AbstractMeshType{D}, axis, side) where {D}
            ci = indices(Ωₕ)
            return CartesianIndices(
                ntuple(
                d -> d == axis ? (side == 1 ? (1:1) : (size(ci, d):size(ci, d))) : (1:size(ci, d)),
                Val(D)
            ),
            )
        end
        @noinline function _geometric_face_bits(Ωₕ::AbstractMeshType{D}, label::Symbol) where {D}
            axis, side = _face_of_symbol(Val(D), label)
            face = _face_cart(Ωₕ, axis, side)
            lin = LinearIndices(indices(Ωₕ))
            bits = falses(length(lin))
            for I in face
                @inbounds bits[lin[I]] = true
            end
            return bits
        end
        @inline function _marks_only_face(
                Ωₕ::AbstractMeshType{D}, bits::BitVector, label::Symbol) where {D}
            axis, side = _face_of_symbol(Val(D), label)
            face = _face_cart(Ωₕ, axis, side)
            lin = LinearIndices(indices(Ωₕ))
            n = 0
            for I in face
                n += @inbounds bits[lin[I]]
            end
            return n > 0 && n == count(bits)
        end
    end
end

const Wₕ = let
    set = DIM == 1 ? domain(interval(0.0, 1.0)) :
          DIM == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
          domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0))
    gridspace(mesh(set, DIM == 1 ? 17 : DIM == 2 ? (9, 9) : (5, 5, 5),
        DIM == 1 ? true : ntuple(_ -> true, DIM)))
end
const BC = :boundary => x -> 0.0
const LABELS, CONDS = Bramble._normalize_dirichlet(BC)

# The coefficient scales the parameter into a grid function, as the test's `_field` does.
function coefficient(p::AbstractVector{T}) where {T}
    κ = element(Wₕ, T)
    parent(κ) .= 2.0 .+ p[1]
    return κ
end

# The form is built inside the function: the same call with a global form does not abort.
function loss_apply(p)
    lf = form(Wₕ, v -> innerₕ(x -> 1.0, v))
    v = zeros(eltype(p), ndofs(Wₕ))
    v .= p[1]
    Bramble.apply_dirichlet_conditions!(v, lf, CONDS, LABELS, nothing)
    return sum(abs2, v)
end

function loss_assemble(p)
    κ = coefficient(p)
    A, F = assemble(form(Wₕ, Wₕ, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V))),
        form(Wₕ, v -> innerₕ(x -> 1.0, v)); dirichlet = BC)
    return sum(abs2, nonzeros(A)) + sum(abs2, F)
end

const LOSS = occursin("assemble", VARIANT) ? loss_assemble : loss_apply
println("variant ", VARIANT, ", ", DIM, "D, Enzyme ", pkgversion(Enzyme), ", Julia ", VERSION,
    ", check-bounds ", Base.JLOptions().check_bounds, " (0 auto, 1 yes, 2 no)")
flush(stdout)
mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
dp = zeros(NP)
try
    Enzyme.autodiff(mode, Enzyme.Const(LOSS), Enzyme.Active, Enzyme.Duplicated([0.5, 0.2, 0.1], dp))
    println("NO ABORT: gradient ", dp)
catch e
    println("NO ABORT: Enzyme raised ", typeof(e))
end
