package treebased.core.training.aggregate

/**
 * Sufficient statistics for the within-leaf OLS slope of Y on W — the partially-linear-model
 * treatment effect used by the locally centered causal forest, where W is a residualized
 * (continuous) treatment. The additive group lives in
 * [[treebased.core.training.aggregate.TreatmentSlopeAccumulator]].
 *
 * `slope` returns `NaN` when W has (near-)no variation in the leaf — an empty sample, every W
 * identical, or a treatment-residual sum of squares at/below `treatmentVarianceFloor` per point
 * (a small fraction of the global treatment-residual variance; see
 * [[treebased.estimand.CausalSpec.calibrate]]). [[treebased.core.training.HonestyPruner]] then
 * collapses the leaf into its sibling, mirroring [[TreatmentEffectStats.ate]]. The floor defaults
 * to `0`, i.e. only an exactly (or negatively, from float cancellation) degenerate leaf.
 */
final case class TreatmentSlopeStats(
  wSum: Double,
  ySum: Double,
  wySum: Double,
  wwSum: Double,
  count: Int
) {
  def slope(treatmentVarianceFloor: Double = 0.0): Double =
    if (count == 0) Double.NaN
    else {
      val denom = wwSum - wSum * wSum / count // = Σ (w - w̄)²  over the leaf
      if (denom <= treatmentVarianceFloor * count) Double.NaN
      else (wySum - wSum * ySum / count) / denom
    }
}

object TreatmentSlopeStats {
  val zero: TreatmentSlopeStats = TreatmentSlopeStats(0.0, 0.0, 0.0, 0.0, 0)
}
