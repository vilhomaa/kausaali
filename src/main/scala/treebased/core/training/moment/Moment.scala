package treebased.core.training.moment

import treebased.core.domain.data.DataPoint

/**
 * The estimating-equation ("moment condition" / psi function) axis from Athey, Tibshirani &
 * Wager (2019), "Generalized Random Forests": computes per-observation pseudo-outcomes that
 * linearly approximate how each point would shift a node's estimate of the target parameter
 * theta (e.g. the ATE for a causal forest, the conditional mean for a regression forest).
 * [[treebased.core.training.split.GradientSplitRule]] reduces the split search to a
 * standard weighted-variance-reduction scan over these pseudo-outcomes instead of exactly
 * refitting theta per candidate child.
 */
trait Moment[O <: DataPoint] {
  /** Returns one pseudo-outcome per input point, aligned by index to `nodeDataPoints`. */
  def pseudoOutcomes(nodeDataPoints: Array[O]): Array[Double]
}
