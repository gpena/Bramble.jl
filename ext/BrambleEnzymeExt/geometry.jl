# The rules for `_levelset_grid`, which fits a level-set mesh's grid lines to the curve
# `φ = 0` (#1211). The grid comes from the signs of `φ` by bisection, in `Float64`, so its
# nodes carry no derivative of a parameter in `φ`. Without a rule Enzyme compiles the
# bisection and either returns zeros or aborts in LLVM, differently from one process to the
# next. With it the fit never reaches Enzyme: a `φ` that is data runs the primal and the
# grid gets a zero shadow, and a `φ` with a shadow of its own is refused. The box must not
# depend on the parameter either. Forward mode refuses one that does by the seed lines'
# tangents, reverse mode by the box's bounds (`_levelset_box_bounds`, #1236).

@noinline function _throw_levelset_parameter()
    throw(
        ArgumentError(
        "Enzyme cannot differentiate a level-set mesh with respect to a parameter of its " *
        "level-set function `φ`: the grid is fitted to the signs of `φ` and carries no " *
        "derivative. Hold the level set fixed: build the mesh outside the differentiated " *
        "function, or from a `φ` that does not depend on the parameter. A parameter may " *
        "enter the form instead. In reverse mode a `φ` that captures an array built inside " *
        "the differentiated function is refused even when the array is constant: capture " *
        "its numbers instead (`c₁, c₂ = c` before `φ`). The level-set box must not depend " *
        "on the parameter either.",
    ),
    )
end

@noinline function _throw_levelset_box()
    throw(
        ArgumentError(
        "Enzyme cannot differentiate a level-set mesh whose box depends on the parameter: " *
        "the grid lines are fitted to the curve `φ = 0` and carry no derivative of the " *
        "box. Hold the box and the level set fixed; a parameter may enter the form.",
    ),
    )
end

# Whether `φ` is data. In forward mode it is when no lane of its shadow carries a tangent
# (`_moving_shadow` in solvers.jl): a closure's shadow is read field by field, since `===`
# compares an immutable closure's bits and a seed equal to a captured float would pass as
# data. A closure over the parameter vector, or over a float read from it, is refused. In
# reverse mode `φ` is data if it is a `Const`, or a `Duplicated` whose shadow is the primal
# (runtime activity's inactive data, as `_space_is_data` in relations.jl), a closure over
# an array passed in `Const` among them. A closure over an array built inside the
# differentiated function is refused: in reverse mode its shadow looks like the parameter's.
# A closure over constant floats comes in `Const`.
_levelset_is_data(::Const) = true
_levelset_is_data(φ::Duplicated) = φ.dval === φ.val
_levelset_is_data(::Annotation) = false

# Whether a seed line carries a nonzero tangent in some lane: forward mode's sign that the
# box depends on the parameter. A box built inside the differentiated function from
# constants has a shadow of its own, all zeros.
_has_tangent(::Const) = false
function _has_tangent(x::Union{Duplicated, DuplicatedNoNeed})
    return x.dval !== x.val && any(!iszero, x.dval)
end
function _has_tangent(x::Union{BatchDuplicated, BatchDuplicatedNoNeed})
    return any(d -> d !== x.val && any(!iszero, d), x.dval)
end
_has_tangent(::Annotation) = true

function EnzymeRules.forward(
        config::FwdConfig,
        func::Const{typeof(Bramble._levelset_grid)},
        ::Type{RT},
        φ::Annotation,
        xs::Annotation,
        ys::Annotation
) where {RT}
    _moving_shadow(φ, EnzymeRules.width(config)) && _throw_levelset_parameter()
    (_has_tangent(xs) || _has_tangent(ys)) && _throw_levelset_box()
    r = func.val(φ.val, xs.val, ys.val)
    needs_primal = EnzymeRules.needs_primal(config)
    (RT <: Const || !EnzymeRules.needs_shadow(config)) && return needs_primal ? r : nothing
    W = EnzymeRules.width(config)
    shadow = W == 1 ? Enzyme.make_zero(r) : ntuple(_ -> Enzyme.make_zero(r), Val(W))
    needs_primal || return shadow
    return W == 1 ? Duplicated(r, shadow) : BatchDuplicated(r, shadow)
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(Bramble._levelset_grid)},
        ::Type{RT},
        φ::Annotation,
        xs::Annotation,
        ys::Annotation
) where {RT}
    _levelset_is_data(φ) || _throw_levelset_parameter()
    r = func.val(φ.val, xs.val, ys.val)
    primal = EnzymeRules.needs_primal(config) ? r : nothing
    W = EnzymeRules.width(config)
    shadow = if RT <: Const || !EnzymeRules.needs_shadow(config)
        nothing
    else
        W == 1 ? Enzyme.make_zero(r) : ntuple(_ -> Enzyme.make_zero(r), Val(W))
    end
    ART = EnzymeRules.augmented_rule_return_type(typeof(config), RT, Nothing)
    return ART(primal, shadow, nothing)::ART
end

# The grid's cotangent stops here: nothing reaches `φ` or the seed lines. At
# `augmented_primal` the seed lines' shadows are fresh zeros whether or not the box depends
# on the parameter, so the rules for `_levelset_box_bounds` below refuse such a box instead.
function EnzymeRules.reverse(
        ::RevConfig,
        ::Const{typeof(Bramble._levelset_grid)},
        ::Type,
        tape,
        φ::Annotation,
        xs::Annotation,
        ys::Annotation
)
    return (nothing, nothing, nothing)
end

# The rules for `_levelset_box_bounds`, which `_mesh` calls on the box of a domain with a
# level-set marker. Enzyme decides a float argument's activity at compile time: a bound
# that depends on the parameter comes in `Active`, a constant one `Const`, so reverse mode
# refuses the box here. A bound Enzyme cannot prove constant comes in `Active` too and is
# refused. Forward mode refuses at `_levelset_grid`, by the seed lines' tangents, and
# passes the bounds through.
function EnzymeRules.forward(
        ::FwdConfig,
        ::Const{typeof(Bramble._levelset_box_bounds)},
        ::Type,
        x₀::Annotation,
        x₁::Annotation,
        y₀::Annotation,
        y₁::Annotation
)
    return nothing
end

function EnzymeRules.augmented_primal(
        ::RevConfig,
        ::Const{typeof(Bramble._levelset_box_bounds)},
        ::Type,
        x₀::Annotation,
        x₁::Annotation,
        y₀::Annotation,
        y₁::Annotation
)
    any(b -> b isa Active, (x₀, x₁, y₀, y₁)) && _throw_levelset_box()
    return AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        ::RevConfig,
        ::Const{typeof(Bramble._levelset_box_bounds)},
        ::Type,
        tape,
        x₀::Annotation,
        x₁::Annotation,
        y₀::Annotation,
        y₁::Annotation
)
    _zero(b) = b isa Active ? zero(b.val) : nothing
    return (_zero(x₀), _zero(x₁), _zero(y₀), _zero(y₁))
end
