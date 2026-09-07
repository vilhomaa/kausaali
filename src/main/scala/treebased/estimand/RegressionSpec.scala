package treebased.estimand

import treebased.core.prediction.{ForestAggregation, KernelRow, LeafEstimator, WeightedEstimator}
import treebased.core.training.moment.{Moment, RegressionMoment}
import treebased.core.training.split.guard.{SplitGuard, MinNodeSizeGuard}
import treebased.core.domain.data.Observation
import treebased.core.training.centering.{CenteringTransform, NoCentering}

/** Estimand recipe for a regression forest: leaf prediction is the honest-sample mean of `Y`. */
object RegressionSpec extends ForestSpec[Observation] {
  val moment: Moment[Observation] = RegressionMoment()
  val sizeGuard: SplitGuard[Observation] = MinNodeSizeGuard
  val leaf: LeafEstimator[Observation, ?] = LeafEstimator.mean
  val predictOps: ForestAggregation = ForestAggregation.mean
  val estimator: WeightedEstimator = WeightedEstimator.Mean
  def kernelRow(o: Observation): KernelRow = KernelRow(o.weight, o.y, Double.NaN)
  val centering: CenteringTransform[Observation] = new NoCentering[Observation]
}
