# The stencil nodes: differences, averages, jumps and shifts, each linear in its operand
# entry for entry and reading it at points of its own. One list serves the simplifier's
# distribution over a mixing sum (`_LinearWrapper`, simplifier.jl) and the split of a
# restricted product into its summands (`_summands_of`, bilinear.jl), so a new node is
# added here once (gpena/Bramble.jl#767). It comes before the simplifier for that reason.
const _DifferenceNode = Union{BackwardDifference, ForwardDifference, CenteredDifference,
    StarDifference, CrossWeightedDifference, SecondDifference}
const _AverageNode = Union{JumpNode, BackwardAverage, ForwardAverage, CenteredAverage}
const _StencilNode = Union{_DifferenceNode, _AverageNode, ShiftNode}
