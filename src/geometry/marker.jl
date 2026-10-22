"""
    Marker(label::Symbol, identifier::F)

Labeled geometric region or boundary marker on a computational domain.

# Arguments
- `label`: Region identifier (e.g. `:inlet`, `:wall`, `:boundary`).
- `identifier`: Location specification, either a predefined boundary `Symbol` (e.g. `:left`),
  a `Set{Symbol}` of boundary names, a boolean spatial predicate function `f(x)`, or a
  `Polygon` (from `polygon`) classifying the nodes of a 2D mesh.
  [`markers`](@ref) builds a `Marker` from a tuple, vector or set of boundary symbols by
  storing it as a `Set{Symbol}`.
"""
struct Marker{F}
    label::Symbol
    identifier::F
end

"""
    MarkerPair{F}

Type alias for `Pair{Symbol, F}` used to specify region markers (e.g. `:boundary => :left`).
"""
const MarkerPair{F} = Pair{Symbol, F}

"""
    label(m::Marker) -> Symbol
    label(m::MarkerPair) -> Symbol

Return the `Symbol` label of marker `m`.
"""
@inline label(m::Marker) = m.label
@inline label(m::MarkerPair) = first(m)

"""
    identifier(m::Marker{F}) -> F
    identifier(m::MarkerPair{F}) -> F

Return the identifying symbol, set of symbols, predicate function or polygon of marker `m`.
"""
@inline identifier(m::Marker) = m.identifier
@inline identifier(m::MarkerPair) = last(m)

"""
    DomainMarkers(symbols::Tuple, tuples::Tuple, conditions::Tuple)
    DomainMarkers(symbols::Tuple, tuples::Tuple, conditions::Tuple, regions::Tuple)

Container categorizing and indexing boundary and interior markers for a computational domain.

Every field is a statically typed tuple, so `symbols`, `tuples`, `conditions` and `regions`
markers are each an unrolled, zero-allocation sweep to iterate. The three-argument form has
no `regions`.

# Fields
- `symbols`: Tuple of markers identified by a single predefined boundary `Symbol` (e.g. `:left`).
- `tuples`: Tuple of markers identified by collections of predefined boundary symbols,
  given as a tuple, vector or set (e.g. `(:top, :right)`, `[:top, :right]`), stored as `Set{Symbol}`.
- `conditions`: Statically typed tuple of predicate function markers `f(x)` or `f(x, t)`.
- `regions`: Tuple of polygon markers `:L => polygon(V)`. On a mesh, each gives the three
  labels `L` (nodes inside the polygon), `L_boundary` (nodes on its boundary) and
  `L_exterior` (nodes outside it).

See also: [`markers`](@ref), [`symbols`](@ref), [`tuples`](@ref), [`conditions`](@ref).
"""
struct DomainMarkers{ST <: Tuple, TT <: Tuple, CT <: Tuple, RT <: Tuple}
    symbols::ST
    tuples::TT
    conditions::CT
    regions::RT
end

function DomainMarkers(symbols::Tuple, tuples::Tuple, conditions::Tuple)
    return DomainMarkers(symbols, tuples, conditions, ())
end

# The polygon markers of `domain_markers`.
@inline regions(domain_markers::DomainMarkers) = domain_markers.regions

# The three mesh labels of the polygon marker labelled `L`: its interior, boundary and
# exterior nodes.
@inline _polygon_labels(L::Symbol) = (L, Symbol(L, :_boundary), Symbol(L, :_exterior))

# A polygon marker `:L => P`. Its three mesh labels are built once, here, so that reading
# them (`labels`, `set_markers!`) allocates nothing.
struct _PolygonMarker{T}
    label::Symbol
    identifier::Polygon{T}
    labels::NTuple{3, Symbol}
end
_PolygonMarker(L::Symbol, P::Polygon) = _PolygonMarker(L, P, _polygon_labels(L))

@inline label(m::_PolygonMarker) = m.label
@inline identifier(m::_PolygonMarker) = m.identifier
@inline _polygon_labels(m::_PolygonMarker) = m.labels

"""
    symbols(domain_markers::DomainMarkers) -> Tuple

Return the tuple of single-symbol markers configured in `domain_markers`.
"""
@inline symbols(domain_markers::DomainMarkers) = domain_markers.symbols

"""
    tuples(domain_markers::DomainMarkers) -> Tuple

Return the tuple of multi-symbol markers configured in `domain_markers`.
"""
@inline tuples(domain_markers::DomainMarkers) = domain_markers.tuples

"""
    conditions(domain_markers::DomainMarkers) -> Tuple

Return the tuple of predicate condition markers configured in `domain_markers`.
"""
@inline conditions(domain_markers::DomainMarkers) = domain_markers.conditions

"""
    markers(space_set::CartesianProduct, pairs::Pair...) -> DomainMarkers
    markers(space_set::CartesianProduct, time_set::CartesianProduct{1}, pairs::Pair...) -> DomainMarkers

Construct a [`DomainMarkers`](@ref) collection from `label => identifier` pairs.

# Arguments
- `space_set`: Geometric spatial set.
- `time_set`: Optional 1D temporal interval for time-dependent boundary conditions. With it,
  every predicate `Function` takes `(x, t)`. `markers(...)(t)` fixes `(x, t)` predicates at
  `t` and keeps spatial `x` predicates, so it also applies to markers built without it.
- `pairs`: Vararg sequence of `label => identifier` pairs where identifier is a `Symbol`,
  a tuple (`NTuple{N, Symbol}`), `AbstractVector{Symbol}` or `AbstractSet{Symbol}` of
  boundary symbols, a predicate `Function` (`x -> Bool`, or `(x, t) -> Bool` with
  `time_set`), or, on a 2D `space_set`, a `Polygon` built by `polygon`. A pair
  `:L => polygon(V)` gives a mesh the labels `L`, `L_boundary` and `L_exterior`: the nodes
  inside the polygon, on its boundary and outside it. A node is on the boundary when its
  distance to the boundary is at most
  ``\\max(\\sqrt{\\varepsilon} h_{\\min}, 10^3 \\varepsilon L) + 4 \\varepsilon M``, with
  ``h_{\\min}`` the smallest mesh spacing, ``L`` the polygon's extent, ``M`` the largest
  coordinate magnitude of the polygon's vertices, and ``\\varepsilon`` the larger
  machine epsilon of the mesh's points and the polygon's vertices.

# Throws
- `ArgumentError`: if an identifier is of any other kind (for example a `String`, a `Dict`
  or a callable object that is not a `Function`), names a symbol that is not a boundary
  of `space_set`, or, with `time_set`, is a predicate that does not accept `(x, t)`.
  Also if a polygon marks a set that is not 2D, has a vertex outside `space_set`, is
  labelled `:boundary` or `:interior`, or one of its three labels is another marker's.

# Examples
```jldoctest
using Bramble
I = interval(0.0, 1.0)
m = markers(I, :left_boundary => :left, :internal => x -> 0.2 < x < 0.8)
length(symbols(m)) == 1 && length(conditions(m)) == 1

# output
true
```
"""
@inline function markers(space_set::CartesianProduct, pairs::Pair...)
    for p in pairs
        _validate_marker_pair(space_set, p)
    end
    _check_polygon_label_clash(pairs)
    return _create_generic_markers(pairs...)
end

@inline function markers(
        space_set::CartesianProduct, time_set::CartesianProduct{1}, pairs::Pair...
)
    for p in pairs
        _validate_marker_pair(space_set, time_set, p)
    end
    _check_polygon_label_clash(pairs)
    return _create_generic_markers(pairs...)
end

# Identifier kinds that name several boundary symbols; stored as `Marker{Set{Symbol}}`.
const _SymbolCollection = Union{NTuple{N, Symbol} where {N}, AbstractVector{Symbol}, AbstractSet{Symbol}}

# The point a predicate is probed at: the centre of the space set, a scalar in 1D.
_marker_probe_point(space_set::CartesianProduct{D}) where {D} = D == 1 ? center(space_set)[1] : center(space_set)

# Call predicate `f` on the probe arguments and require a `Bool` back.
function _check_marker_predicate(lbl::Symbol, D::Integer, f, args...)
    res = try
        f(args...)
    catch err
        _throw_invalid_marker_predicate_call(lbl, D, err)
    end
    res isa Bool || _throw_non_bool_marker_predicate(lbl, res)
    return nothing
end

function _validate_marker_pair(space_set::CartesianProduct{D}, p::Pair) where {D}
    lbl = p.first
    ident = p.second
    if ident isa Symbol
        if !(ident in _all_boundary_symbols(space_set))
            _throw_unknown_boundary_symbol_for_domain(lbl, ident, space_set)
        end
    elseif ident isa _SymbolCollection
        for s in ident
            if !(s in _all_boundary_symbols(space_set))
                _throw_unknown_boundary_symbol_for_domain(lbl, s, space_set)
            end
        end
    elseif ident isa Function
        _check_marker_predicate(lbl, D, ident, _marker_probe_point(space_set))
    elseif ident isa Polygon
        _validate_polygon_marker(space_set, lbl, ident)
    else
        _throw_unsupported_marker_identifier(lbl, ident)
    end
    return nothing
end

# A polygon marks the nodes of a 2D mesh, lies in the set, and never takes a reserved label.
function _validate_polygon_marker(
        space_set::CartesianProduct{D}, lbl::Symbol, P::Polygon
) where {D}
    D == 2 || _throw_polygon_marker_error(lbl, "a polygon marks a 2D set, but this set is $(D)D")
    lbl in (:boundary, :interior) && _throw_polygon_marker_error(lbl,
        ":$lbl is reserved for the mesh's own geometric markers; choose another label")
    (a1, b1), (a2, b2) = extrema(space_set)
    for (k, v) in enumerate(P.vertices)
        (a1 <= v[1] <= b1 && a2 <= v[2] <= b2) || _throw_polygon_marker_error(lbl,
            "vertex $k at $v lies outside the set [$a1, $b1] × [$a2, $b2]")
    end
    return nothing
end

# The three labels of each polygon marker are told apart from every other marker's labels.
function _check_polygon_label_clash(pairs::Tuple)
    for (i, p) in enumerate(pairs)
        p.second isa Polygon || continue
        for (j, q) in enumerate(pairs)
            i == j && continue
            theirs = q.second isa Polygon ? _polygon_labels(q.first) : (q.first,)
            for L in _polygon_labels(p.first)
                L in theirs && _throw_polygon_marker_error(p.first,
                    "its label :$L is also the label of another marker")
            end
        end
    end
    return nothing
end

@noinline function _throw_polygon_marker_error(lbl::Symbol, msg)
    throw(ArgumentError("Polygon marker :$lbl: " * msg * "."))
end

function _validate_marker_pair(
        space_set::CartesianProduct{D}, time_set::CartesianProduct{1}, p::Pair
) where {D}
    lbl = p.first
    ident = p.second
    if ident isa Symbol || ident isa _SymbolCollection || ident isa Polygon
        _validate_marker_pair(space_set, p)
    elseif ident isa Function
        probe_x = _marker_probe_point(space_set)
        probe_t = first(extrema(time_set))
        hasmethod(ident, Tuple{typeof(probe_x), typeof(probe_t)}) ||
            _throw_time_marker_needs_xt(lbl)
        _check_marker_predicate(lbl, D, ident, probe_x, probe_t)
    else
        _throw_unsupported_marker_identifier(lbl, ident)
    end
    return nothing
end

@noinline function _throw_time_marker_needs_xt(lbl::Symbol)
    throw(
        ArgumentError(
        "Predicate for marker :$lbl has no method for (x, t). A time-dependent domain " *
        "needs (x, t) predicates, evaluated at the time given to `Ω(t)`; write " *
        "`(x, t) -> ...`, or drop the time set for a spatial-only predicate.",
    ),
    )
end

@noinline function _throw_unsupported_marker_identifier(lbl::Symbol, ident)
    throw(
        ArgumentError(
        "Unsupported identifier of type $(typeof(ident)) for marker :$lbl. " *
        "Accepted identifiers are a boundary Symbol, a tuple, vector or set of boundary " *
        "Symbols, a predicate Function, or a Polygon.",
    ),
    )
end

@noinline function _throw_non_bool_marker_predicate(lbl::Symbol, res)
    throw(
        ArgumentError(
        "Marker predicate for label :$lbl returned a $(typeof(res)) ($res), " *
        "but expected a Bool. For level-set or geometric expressions, write a boolean condition " *
        "(e.g. `x -> predicate(x) <= 0`).",
    ),
    )
end

@noinline function _throw_invalid_marker_predicate_call(lbl::Symbol, D::Int, err)
    throw(
        ArgumentError(
        "Marker predicate for label :$lbl failed when evaluated on sample domain point: " *
        "expected a function accepting a $(D == 1 ? "scalar coordinate" : "$D-element coordinate tuple"). " *
        "Underlying error: $err",
    ),
    )
end

@noinline function _throw_unknown_boundary_symbol_for_domain(lbl::Symbol, sym::Symbol, space_set)
    known = sort!(collect(_all_boundary_symbols(space_set)))
    avail = join(map(s -> ":$s", known), ", ")
    throw(
        ArgumentError(
        "Unknown boundary symbol :$sym for marker :$lbl. " *
        "Valid boundary symbols for this $(dim(space_set))D domain are: $avail.",
    ),
    )
end

# Parse identifier-based markers (Symbols and tuples, vectors or sets of Symbols) from input pairs.
#
# `filter` on the pair tuple const-folds the type tests, so the marker tuples keep their
# length and element types and `DomainMarkers` stays concrete. Declaration order is kept; a
# repeated pair marks the same bits twice, which is harmless, but `conditions` is
# order-sensitive (the last declaration wins).
@inline function _extract_identifier_markers(pairs::Tuple)
    symbol_pairs = filter(p -> p.second isa Symbol, pairs)
    tuple_pairs = filter(p -> p.second isa _SymbolCollection, pairs)

    return map(p -> Marker(p.first, p.second), symbol_pairs),
    map(p -> Marker(p.first, Set(p.second)), tuple_pairs)
end

# Construct DomainMarkers from label-identifier pairs, extracting symbol and tuple sets
# while preserving concrete closure types in a specialized conditions tuple.
function _create_generic_markers(pairs::Pair...)
    symbols, tuples = _extract_identifier_markers(pairs)
    conditions = _pairs_to_tuple_conditions(pairs)
    regions = map(p -> _PolygonMarker(p.first, p.second), filter(p -> p.second isa Polygon, pairs))

    return DomainMarkers(symbols, tuples, conditions, regions)
end

# Extract function-valued pairs into a tuple of Marker{F} instances, keeping each closure's
# concrete type specialized for zero heap allocations during evaluation.
@inline function _pairs_to_tuple_conditions(pairs::Tuple)
    fn_pairs = filter(p -> p.second isa Function, pairs)
    return map(p -> Marker(p.first, p.second), fn_pairs)
end

"""
    EvaluatedDomainMarkers(original_markers::DomainMarkers, evaluation_time::Number)

[`DomainMarkers`](@ref) collection evaluated at timestamp `t`. The collection may hold
`(x, t)` predicates, spatial `x` predicates, or both;
[`conditions`](@ref conditions(::EvaluatedDomainMarkers)) fixes only the former at `t`.

# Fields
- `original_markers`: Underlying [`DomainMarkers`](@ref) object.
- `evaluation_time`: Evaluation timestamp `t`.
"""
struct EvaluatedDomainMarkers{M <: DomainMarkers, T <: Number}
    original_markers::M
    evaluation_time::T
end

"""
    conditions(edm::EvaluatedDomainMarkers) -> Tuple

Return condition markers evaluated at timestamp `edm.evaluation_time`, each a spatial
predicate of `x`.

A predicate `f` with a one-argument method and no two-argument method is spatial and passes
through unchanged. Any other predicate is fixed at `t` as `Base.Fix2(f, t)`. So
`(x, t = 0.0) -> ...` is fixed at `t`, not left at its default. A predicate typed on its
argument (`(x::NTuple{2, Float64}) -> ...`) has no `Tuple{Any}` method, so it is also
wrapped and must accept `(x, t)`.

A `ComposedFunction` or a `Base.Fix1`/`Base.Fix2` is classified by what it wraps, not by its
own varargs call method: a composition by its innermost function, which receives every
argument, and a `Fix` by the function it fixes, with one more argument since the `Fix`
supplies one itself. So `(>(0.5)) ∘ first` and `Base.Fix1(<, 0.5)` are spatial and pass
through. A lone `Base.Fix2(isless, 0.5)` is not, as `isless` has no generic two-argument
method, so it is wrapped like any typed predicate.
"""
function conditions(edm::EvaluatedDomainMarkers)
    t = edm.evaluation_time
    return map(
        m -> Marker(label(m), _fix_time(identifier(m), t)), conditions(edm.original_markers)
    )
end

# `hasmethod` is `@nospecialize` in `f`: called normally it folds only for a singleton `f`,
# so a capturing closure or a callable struct with fields gives a `Union` per slot and
# allocates. Inlined at the call site, its `Core._hasmethod(Tuple{F, ...})` folds on the
# concrete type `F`, so the tuple stays concrete and `conditions` allocates nothing. A
# method added to `F` later invalidates the folded result, as for any method lookup.
#
# Both evaluation paths share `_is_spatial`. Every helper here stays `@inline`, or the fold
# never reaches the `map` closure in `conditions`.
#
# `_accepts(f, Val(N))` asks whether `f` takes `N` arguments. A `ComposedFunction` and a
# `Fix` are varargs-callable, so they answer for what they wrap: the innermost function of a
# composition receives every argument, and a `Fix{K}` (`Fix1`, `Fix2`, ...) supplies the
# `K`-th itself, so it needs at least `K - 1` arguments before it.
@inline function _accepts(f::F, ::Val{N}) where {F, N}
    return @inline hasmethod(f, NTuple{N, Any})
end
@inline _accepts(c::ComposedFunction, ::Val{N}) where {N} = _accepts(c.inner, Val(N))
@inline function _accepts(f::Base.Fix{K}, ::Val{N}) where {K, N}
    return N >= K - 1 && _accepts(f.f, Val(N + 1))
end

@inline function _is_spatial(f::F) where {F}
    return _accepts(f, Val(1)) && !_accepts(f, Val(2))
end

@inline _fix_time(f::F, t) where {F} = _is_spatial(f) ? f : Base.Fix2(f, t)

# The `(t, p)` twin of `_fix_time`: classifies the raw predicate, not the `Fix` wrappers
# built below. A 3-argument method wins over a 1-argument one, as the validators route such
# an `f` here for its `(x, t, p)` method. The 3-argument test goes through `_accepts` too,
# or a wrapper (varargs, so it has a `Tuple{Any, Any, Any}` method) would always be fixed.
@inline function _fix_time_param(f::F, t, p) where {F}
    is_spatial = _is_spatial(f) && !_accepts(f, Val(3))
    return is_spatial ? f : Base.Fix2(Base.Fix{3}(f, p), t)
end

"""
    (dm::DomainMarkers)(t::Number) -> EvaluatedDomainMarkers

Evaluate the condition markers of `dm` at timestamp `t`. Each `(x, t)` predicate is fixed at
`t` and each spatial `x` predicate is kept as it is; see
[`conditions`](@ref conditions(::EvaluatedDomainMarkers)).
"""
(dm::DomainMarkers)(t::Number) = EvaluatedDomainMarkers(dm, t)

"""
    EvaluatedParametricDomainMarkers(original_markers::DomainMarkers, evaluation_time::Number, p)

Time- and parameter-evaluated wrapper for a [`DomainMarkers`](@ref) collection whose
conditions accept `(x, t, p)` rather than `(x, t)`, evaluated at timestamp `t` and parameter
`p`.

Kept separate from [`EvaluatedDomainMarkers`](@ref) rather than adding a `p` field to it: the
two-argument `Base.Fix2(identifier(m), t)` this file already builds for the `(x, t)` case
would otherwise need a runtime branch on whether `p` is present, and this exact kind of
branch-carrying `Union`/optional field is what made Enzyme's strict-aliasing type analysis
reject `AnySegment{D}` (gpena/Bramble.jl#240) -- a fresh concrete type costs only the two
`conditions` bodies, which differ, and keeps every existing `EvaluatedDomainMarkers` call site
untouched.

# Fields
- `original_markers`: Underlying [`DomainMarkers`](@ref) object.
- `evaluation_time`: Evaluation timestamp `t`.
- `p`: Evaluation parameter, passed through unmodified (e.g. a `Vector` or a scalar).
"""
struct EvaluatedParametricDomainMarkers{M <: DomainMarkers, T <: Number, P}
    original_markers::M
    evaluation_time::T
    p::P
end

"""
    (dm::DomainMarkers)(t::Number, p) -> EvaluatedParametricDomainMarkers

Evaluate the condition markers of `dm` at timestamp `t` and parameter `p`. Each `(x, t, p)`
predicate is fixed at `(t, p)` and each spatial `x` predicate is kept as it is; see
[`conditions`](@ref conditions(::EvaluatedParametricDomainMarkers)).
"""
(dm::DomainMarkers)(t::Number, p) = EvaluatedParametricDomainMarkers(dm, t, p)

"""
    conditions(edm::EvaluatedParametricDomainMarkers) -> Tuple

Return condition markers evaluated at `edm.evaluation_time` and `edm.p`, each a spatial
predicate of `x`.

A predicate `f` with a one-argument method and no two- or three-argument method is spatial
and passes through unchanged, a `ComposedFunction` or `Base.Fix1`/`Base.Fix2` being judged
by what it wraps, as in the method above. Any other predicate is fixed at `(t, p)` as
`x -> f(x, t, p)`, so `(x, t = 0, p = 1) -> ...` is fixed at both. An `(x, t)` predicate is
not fixed at `t` alone, and its wrapper throws a `MethodError` when called;
`dirichlet_constraints` and `semidiscretize` never send one here, since they route only
uniformly `(x, t, p)` conditions
to this path.

`Base.Fix{N}` generalizes `Base.Fix1`/`Base.Fix2` to insert its fixed value at position `N`
of *whatever args a given call supplies*, not at position `N` of `f`'s own argument list
(`Base.Fix`'s own docstring example: `Fix{1}(Fix{2}(f, 4), 4)` fixes the first and second
arg, not the first and fourth). So `Base.Fix{3}(f, p)` first, fixing `f`'s own third
argument to `p` and leaving a 2-argument `(x, t) -> f(x, t, p)`; `Base.Fix2` on *that* then
fixes its second argument to `t`, landing on `x -> f(x, t, p)`. Reversing the order --
`Base.Fix2(Base.Fix2(f, p), t)`, the tempting reading of "fix p, then fix t" -- instead
inserts `p` at position 2 and shoves `t` to position 3, calling `f(x, p, t)` (caught by the
test that checks a `(1.0, 3.0)` vs a swapped-order failure, not by inspection).
"""
function conditions(edm::EvaluatedParametricDomainMarkers)
    t = edm.evaluation_time
    p = edm.p
    return map(
        m -> Marker(label(m), _fix_time_param(identifier(m), t, p)),
        conditions(edm.original_markers)
    )
end

# The accessors the two evaluated types share are written once over their union, and the
# label functions once over that union plus `DomainMarkers`.
const _EvaluatedMarkers = Union{EvaluatedDomainMarkers, EvaluatedParametricDomainMarkers}
const _MarkerCollection = Union{DomainMarkers, _EvaluatedMarkers}

"""
    symbols(edm::Union{EvaluatedDomainMarkers, EvaluatedParametricDomainMarkers}) -> Tuple

Return single-symbol markers from the underlying domain markers.
"""
@inline symbols(edm::_EvaluatedMarkers) = symbols(edm.original_markers)

"""
    tuples(edm::Union{EvaluatedDomainMarkers, EvaluatedParametricDomainMarkers}) -> Tuple

Return multi-symbol markers from the underlying domain markers.
"""
@inline tuples(edm::_EvaluatedMarkers) = tuples(edm.original_markers)
@inline regions(edm::_EvaluatedMarkers) = regions(edm.original_markers)

@inline Base.length(edm::_EvaluatedMarkers) = length(edm.original_markers)
@inline Base.isempty(edm::_EvaluatedMarkers) = isempty(edm.original_markers)

"""
    label_identifiers(markers)
    labels(markers)

Return a tuple of the `Symbol` label of every marker in `markers`, a [`DomainMarkers`](@ref),
an [`EvaluatedDomainMarkers`](@ref) or an [`EvaluatedParametricDomainMarkers`](@ref).

Concatenates the per-category labels of `symbols`, `tuples`, and `conditions`, then the
three labels of each polygon marker (built with the marker), as a single
compile-time-unrolled tuple build, so this allocates 0 bytes. For the labels of one
category alone, [`label_symbols`](@ref), [`label_tuples`](@ref), or
[`label_conditions`](@ref) are equally zero-allocation.
"""
@inline label_identifiers(markers::_MarkerCollection) = (
    map(label, symbols(markers))...,
    map(label, tuples(markers))...,
    map(label, conditions(markers))...,
    _region_mesh_labels(regions(markers))...
)

@inline _region_mesh_labels(::Tuple{}) = ()
@inline function _region_mesh_labels(rs::Tuple)
    return (_polygon_labels(first(rs))..., _region_mesh_labels(Base.tail(rs))...)
end

@inline labels(markers::_MarkerCollection) = label_identifiers(markers)

"""
    label_symbols(markers)

Return an iterator yielding labels of all single-symbol markers.
"""
@inline label_symbols(markers::_MarkerCollection) = (label(m)::Symbol for m in symbols(markers))

"""
    label_tuples(markers)

Return an iterator yielding labels of all multi-symbol markers.
"""
@inline label_tuples(markers::_MarkerCollection) = (label(m)::Symbol for m in tuples(markers))

"""
    label_conditions(markers)

Return an iterator yielding labels of all condition predicate markers.
"""
@inline label_conditions(markers::_MarkerCollection) = (label(m)::Symbol for m in conditions(markers))

function Base.show(io::IO, m::Marker{F}) where {F}
    if F <: Symbol
        print(io, "Marker(:$(m.label) => :$(m.identifier))")
    elseif F <: Set{Symbol}
        syms = join(m.identifier, ", ")
        print(io, "Marker(:$(m.label) => ($syms))")
    else
        print(io, "Marker(:$(m.label) => <function>)")
    end
end

function Base.show(io::IO, m::_PolygonMarker)
    print(io, "Marker(:$(m.label) => <polygon>)")
end

function Base.show(io::IO, dm::DomainMarkers)
    total = length(dm)
    print(io, "DomainMarkers($total total)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", dm::DomainMarkers)
    return show_block(io) do io
        pp = PrettyPrinter(io)

        n_sym = length(dm.symbols)
        n_tup = length(dm.tuples)
        n_cond = length(dm.conditions)
        n_reg = length(dm.regions)
        total = n_sym + n_tup + n_cond + n_reg

        if total == 0
            print_empty_message(pp, "DomainMarkers: (empty)")
            return nothing
        end

        print_header(pp, "DomainMarkers:")
        println(io, " with $total marker$(total == 1 ? "" : "s"):")

        pp_indented = with_indent(pp, 1)

        if n_sym > 0
            print_section_header(pp_indented, "Symbol markers ($n_sym):")
            pp_double_indent = with_indent(pp, 2)
            for m in dm.symbols
                print_key_value(pp_double_indent, ":$(label(m))", ":$(identifier(m))")
            end
        end

        if n_tup > 0
            print_section_header(pp_indented, "Tuple markers ($n_tup):")
            pp_double_indent = with_indent(pp, 2)
            for m in dm.tuples
                print_indent(pp_double_indent)
                printstyled(io, ":$(label(m))"; color = :green)
                print(io, " => (")
                print_joined(pp, sort!(collect(identifier(m)))) do s
                    return printstyled(io, ":$s"; color = :blue)
                end
                println(io, ")")
            end
        end

        if n_cond > 0
            print_section_header(pp_indented, "Function markers ($n_cond):")
            pp_double_indent = with_indent(pp, 2)
            for m in dm.conditions
                print_indent(pp_double_indent)
                printstyled(io, ":$(label(m))"; color = :green)
                print(io, " => ")
                printstyled(io, "<function>"; color = :magenta)
                println(io)
            end
        end

        if n_reg > 0
            print_section_header(pp_indented, "Polygon markers ($n_reg):")
            pp_double_indent = with_indent(pp, 2)
            for m in dm.regions
                print_indent(pp_double_indent)
                printstyled(io, ":$(label(m))"; color = :green)
                print(io, " => ")
                nv = length(identifier(m).vertices)
                printstyled(io, "<polygon, $nv vertices>"; color = :magenta)
                println(io)
            end
        end
    end
end

@inline function Base.length(dm::DomainMarkers)
    length(dm.symbols) + length(dm.tuples) + length(dm.conditions) + length(dm.regions)
end

@inline function Base.isempty(dm::DomainMarkers)
    isempty(dm.symbols) && isempty(dm.tuples) && isempty(dm.conditions) && isempty(dm.regions)
end
