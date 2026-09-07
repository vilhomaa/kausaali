package treebased.estimand

import treebased.core.prediction.{ForestAggregation, KernelRow, LeafEstimator, WeightedEstimator}
import treebased.core.training.TreatmentMoments
import treebased.core.training.moment.{Moment, CausalMoment}
import treebased.core.training.split.guard.{SplitGuard, TreatmentVarianceGuard}
import treebased.core.domain.data.CausalObservation
import treebased.core.training.centering.CenteringTransform

/**
 * The causal-ATE estimand via gradient-based splitting — contrast with
 * [[treebased.api.CausalTree]]/[[treebased.api.CausalForest]], which estimate the same quantity
 * via the original Athey-Wager exact-splitting honest tree instead.
 *
 * Built by [[treebased.api.GeneralizedRandomForest.causal]] from a [[CausalForestConfig]]; the
 * [[centering]] transform is resolved there from the caller's [[Centering]] choice. The split
 * moment ([[CausalMoment]]) and the leaf estimator ([[LeafEstimator.ateSlope]], the OLS slope of
 * Y on W) both work with a continuous residualized W, and [[TreatmentVarianceGuard]] keeps that
 * slope estimable rather than counting treatment arms.
 *
 * Every causal quantity here divides by the within-node treatment variance `Var(W | node)`.
 * Local centering turns W into a small-variance continuous residual, so in
 * near-treatment-homogeneous nodes that denominator collapses and the pseudo-outcome / leaf
 * slope diverge (on a randomized treatment the raw 0/1 assignment is self-flooring; the residual
 * is not). [[calibrate]] is called by the engine once the data is centered and installs a
 * relative floor — [[TreatmentVarianceFloorFraction]] times the global treatment-residual
 * variance — on all three: the moment denominator, the split guard, and the leaf slope, so a node
 * with too little treatment variation to identify the effect degrades to its parent-scale estimate.
 */
final class CausalSpec(
  val centering: CenteringTransform[CausalObservation],
  treatmentVarianceFloor: Double = 0.0
) extends ForestSpec[CausalObservation] {
  val moment: Moment[CausalObservation] = CausalMoment(treatmentVarianceFloor)
  val sizeGuard: SplitGuard[CausalObservation] = TreatmentVarianceGuard(treatmentVarianceFloor)
  val leaf: LeafEstimator[CausalObservation, ?] = LeafEstimator.ateSlope(treatmentVarianceFloor)
  val predictOps: ForestAggregation = ForestAggregation.mean
  val estimator: WeightedEstimator = WeightedEstimator.Slope(treatmentVarianceFloor)
  def kernelRow(o: CausalObservation): KernelRow = KernelRow(o.weight, o.y, o.w)

  override def calibrate(centeredData: Array[CausalObservation]): ForestSpec[CausalObservation] =
    if (centeredData.isEmpty) this
    else new CausalSpec(
      centering,
      CausalSpec.TreatmentVarianceFloorFraction * TreatmentMoments.variance(centeredData)
    )
}

object CausalSpec {
  /**
   * Floor on the within-node treatment-residual variance, as a fraction of the global
   * treatment-residual variance. A node below this carries essentially no identifying
   * information about the treatment effect, so clamping the denominator (moment / leaf) or
   * stopping the split (guard) there costs no real signal while preventing the `1 / Var(W)`
   * blow-up. Set at the 1e-2 scale: below roughly 1e-3, near-degenerate residual-W leaves
   * still feed noise into the (kernel-weighted or averaged) slope.
   */
  val TreatmentVarianceFloorFraction: Double = 1e-2
}
