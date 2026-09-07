package treebased.core.training.moment

import treebased.core.domain.data.CausalObservation
import treebased.core.training.TreatmentMoments

/**
 * Moment condition for the ATE: theta is the treatment effect, so the pseudo-outcome
 * orthogonalizes each point's label against both the node's mean treatment and its current ATE
 * estimate. See [[treebased.core.training.moment.RegressionMoment]] for the corresponding
 * (simpler) conditional-mean moment.
 *
 * `beta_p_hat` is the least-squares slope of Y on W in the node (the partially-linear-model ATE);
 * for binary W this equals the difference in arm means, so results are unchanged on raw treatment
 * assignments, but it also stays finite when W is a residual (locally centered) treatment.
 *
 * The pseudo-outcome scales by `1 / a_p`, where `a_p = Σ(w - w̄)² / n` is the within-node
 * variance of the (possibly centered) treatment — the GRF Jacobian `A_P`. With a residualized W
 * this can collapse toward zero in near-treatment-homogeneous nodes and make the pseudo-outcome
 * diverge. `treatmentVarianceFloor` (a small fraction of the global treatment-residual variance,
 * supplied by [[treebased.estimand.CausalSpec.calibrate]]) is a lower bound on `a_p`, so a node
 * with `Var(W | node)` too small to identify the effect degrades to the parent-scale estimate.
 * It defaults to `0`, which reproduces the unregularized behaviour.
 */
case class CausalMoment(treatmentVarianceFloor: Double = 0.0) extends Moment[CausalObservation] {
  override def pseudoOutcomes(nodeDataPoints: Array[CausalObservation]): Array[Double] =
    val n = nodeDataPoints.length
    val y_p_avg = nodeDataPoints.map(_.y).sum / n
    val centeredW = TreatmentMoments.centeredW(nodeDataPoints)
    val treatmentSumSq = centeredW.map(d => d * d).sum
    // Floor the treatment-variance denominator: never below `treatmentVarianceFloor` per point.
    val treatmentSumSqFloored = math.max(treatmentSumSq, treatmentVarianceFloor * n)
    if (treatmentSumSqFloored <= 0.0) Array.fill(n)(0.0) // no treatment variation at all → no split signal
    else {
      val beta_p_hat = centeredW.lazyZip(nodeDataPoints).map((wCentered, row) => wCentered * (row.y - y_p_avg)).sum / treatmentSumSqFloored
      val a_p = treatmentSumSqFloored / n
      centeredW.lazyZip(nodeDataPoints).map((wCentered, row) => (1 / a_p) * wCentered * (row.y - y_p_avg - beta_p_hat * wCentered)).toArray
    }
}
