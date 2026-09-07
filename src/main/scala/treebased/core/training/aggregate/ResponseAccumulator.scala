package treebased.core.training.aggregate

import treebased.core.domain.data.Observation

/** Running sum and count of raw responses, folded to produce a regression forest's honest-leaf
 *  conditional mean ([[treebased.core.prediction.LeafEstimator.mean]]). */
object ResponseAccumulator extends Accumulator[Observation, SumCount] with SumCountGroup {
  def lift(o: Observation): SumCount = SumCount(o.y, 1)
}
