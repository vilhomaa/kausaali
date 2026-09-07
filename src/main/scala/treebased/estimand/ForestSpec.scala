package treebased.estimand

import treebased.core.prediction.{ForestAggregation, KernelRow, LeafEstimator, WeightedEstimator}
import treebased.core.training.moment.Moment
import treebased.core.training.split.guard.SplitGuard
import treebased.core.domain.data.DataPoint
import treebased.core.training.centering.CenteringTransform

/**
 * Bundles the per-estimand knobs for a Generalized Random Forest (Athey, Tibshirani & Wager,
 * 2019). Every GRF-family forest shares the same splitting mechanism
 * ([[treebased.core.training.split.GradientSplitRule]] with
 * [[treebased.core.training.split.criterion.GradientVarianceReduction]], both estimand-agnostic), so
 * only what genuinely varies per estimand is bundled here: the moment condition used to score
 * candidate splits, the node-size guard those candidates must satisfy, the leaf estimator that
 * turns a finished leaf's honest sample into a prediction, how tree predictions combine into a
 * forest prediction (the linear fast path) or, equivalently, how the adaptive-kernel-weighted
 * estimating equation is solved ([[estimator]], the correct path for non-linear estimands), the
 * projection of a training row down to the fields that solver needs ([[kernelRow]]), and the
 * local-centering transform applied to the training data upstream of subsampling. `O` is the
 * observation shape the estimand consumes.
 */
trait ForestSpec[O <: DataPoint] {
  def moment: Moment[O]
  def sizeGuard: SplitGuard[O]
  def leaf: LeafEstimator[O, ?]
  def predictOps: ForestAggregation
  def estimator: WeightedEstimator
  def kernelRow(o: O): KernelRow
  def centering: CenteringTransform[O]

  /**
   * Hook called by the training engine once [[centering]] has been applied, with the
   * (possibly residualized) training data. Lets an estimand calibrate data-dependent
   * regularization it cannot know up front — e.g. a relative floor on the within-node
   * treatment variance for a locally centered causal forest, where centering turns W into a
   * small-variance continuous residual and the discrete "both arms present" guard no longer
   * protects the pseudo-outcome / leaf-slope denominator. Estimands that need no calibration
   * return `this` (the default).
   */
  def calibrate(centeredData: Array[O]): ForestSpec[O] = this
}
