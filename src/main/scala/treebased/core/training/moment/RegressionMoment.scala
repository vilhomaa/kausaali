package treebased.core.training.moment

import treebased.core.domain.data.Observation

/**
 * Moment condition for a plain regression forest: theta is the conditional mean E[Y|X], so the
 * pseudo-outcome is just the node-mean-centered label, i.e. psi = Y - theta. Combined with
 * [[treebased.core.training.split.criterion.GradientVarianceReduction]] this reduces to standard
 * CART variance-reduction splitting on Y.
 */
case class RegressionMoment() extends Moment[Observation] {
  override def pseudoOutcomes(nodeDataPoints: Array[Observation]): Array[Double] =
    val y_p_avg = nodeDataPoints.map(_.y).sum / nodeDataPoints.length
    nodeDataPoints.map(dp => dp.y - y_p_avg)
}
