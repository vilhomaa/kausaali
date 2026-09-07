package treebased.core.prediction

import treebased.core.training.aggregate.{Accumulator, TreatmentEffectStats, TreatmentEffectStatsAccumulator, TreatmentSlopeStats, TreatmentSlopeAccumulator, ResponseAccumulator, SumCount}
import treebased.core.domain.data.{CausalObservation, Observation}

/**
 * Turns a finished leaf's honest sample into its scalar prediction: fold the sample into a
 * sufficient statistic via the [[Accumulator]], then read the estimand's point estimate off it
 * (the arm mean difference for a causal leaf, the sample mean for a regression leaf).
 */
trait LeafEstimator[O, S] {
  protected def acc: Accumulator[O, S]
  def predict(stats: S): Double

  final def aggregate(sample: Array[O]): S = acc.foldMap(sample)
  final def prediction(sample: Array[O]): Double = predict(acc.foldMap(sample))
}

object LeafEstimator {
  val ate: LeafEstimator[CausalObservation, TreatmentEffectStats] =
    new LeafEstimator[CausalObservation, TreatmentEffectStats] {
      protected val acc: Accumulator[CausalObservation, TreatmentEffectStats] = TreatmentEffectStatsAccumulator
      def predict(stats: TreatmentEffectStats): Double = stats.ate
    }

  /**
   * Within-leaf OLS slope of Y on W (the partially-linear-model ATE), used by the locally
   * centered causal GRF. `treatmentVarianceFloor` is a lower bound on the leaf's
   * treatment-residual variance below which the slope is treated as unidentified (`NaN`, so the
   * honesty pruner collapses the leaf); it defaults to `0`. Supplied by
   * [[treebased.estimand.CausalSpec.calibrate]] as a small fraction of the global variance.
   */
  def ateSlope(treatmentVarianceFloor: Double = 0.0): LeafEstimator[CausalObservation, TreatmentSlopeStats] =
    new LeafEstimator[CausalObservation, TreatmentSlopeStats] {
      protected val acc: Accumulator[CausalObservation, TreatmentSlopeStats] = TreatmentSlopeAccumulator
      def predict(stats: TreatmentSlopeStats): Double = stats.slope(treatmentVarianceFloor)
    }

  val mean: LeafEstimator[Observation, SumCount] =
    new LeafEstimator[Observation, SumCount] {
      protected val acc: Accumulator[Observation, SumCount] = ResponseAccumulator
      def predict(stats: SumCount): Double = stats.mean
    }
}
