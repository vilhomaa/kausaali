package treebased.core.training.split.guard

import treebased.config.TreeBaseConfig
import treebased.core.domain.data.CausalObservation
import treebased.core.training.TreatmentMoments

/**
 * Causal-forest node-size guard for a continuous (locally centered) treatment: a node is worth
 * splitting when it holds enough points and W varies enough for the within-leaf slope of Y on W
 * to be estimable. Replaces [[TreatmentBalanceGuard]]'s treated/control counting, which is
 * meaningless once W is a residual rather than a 0/1 assignment.
 *
 * `treatmentVarianceFloor` is a lower bound on the node's treatment-residual variance
 * (`Σ(w - w̄)² / n`). Once centering shrinks W to a small-variance continuous residual, a node
 * can pass every count-based check while its treatment variance has collapsed, which makes the
 * GRF pseudo-outcome (`∝ 1 / Var(W | node)`) diverge. Stopping the split here — so the node
 * becomes a leaf estimated from its parent-scale sample — degrades it to the parent-scale
 * estimate. It defaults to `0` (only reject a node with no W variation at all), and is
 * set by [[treebased.estimand.CausalSpec.calibrate]] to a small fraction of the global
 * treatment-residual variance.
 */
final case class TreatmentVarianceGuard(treatmentVarianceFloor: Double = 0.0)
  extends SplitGuard[CausalObservation] {

  def hasEnoughDataForSplit(data: Array[CausalObservation], config: TreeBaseConfig): Boolean =
    data.nonEmpty &&
    data.length >= config.minNodeSize * 2 &&
    TreatmentMoments.variance(data) > treatmentVarianceFloor

  /** Split-stabilization weights: `(w_i - w̄)²`, so the `alpha` rule keeps treatment-residual
   *  variance — not just row count — on both sides of every split. */
  override def splitBalanceWeights(nodeData: Array[CausalObservation]): Array[Double] = {
    val centered = TreatmentMoments.centeredW(nodeData)
    val out = new Array[Double](centered.length)
    var i = 0
    while (i < centered.length) { out(i) = centered(i) * centered(i); i += 1 }
    out
  }

  def trimToValidWindow(sortedData: Array[CausalObservation], minSize: Int): Array[CausalObservation] =
    sortedData.drop(minSize).dropRight(minSize)
}
