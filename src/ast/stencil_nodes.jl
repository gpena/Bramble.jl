# The stencil node families. Every stencil node subtypes `StencilNode` (node_family.jl),
# and a method or union meaning all of them names that (gpena/Bramble.jl#930). These two
# Unions name a family only: the nodes whose value scales by the spacing type `E`
# (`_type_value`, linear.jl), and the averages a restricted product keeps whole
# (`_summands_of`, bilinear.jl). `ShiftNode` is in neither; it relabels, it does not weigh.
const _DifferenceNode = Union{BackwardDifference, ForwardDifference, CenteredDifference,
    StarDifference, CrossWeightedDifference, SecondDifference}
const _AverageNode = Union{JumpNode, BackwardAverage, ForwardAverage, CenteredAverage}
