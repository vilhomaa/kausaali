package treebased.core.training.aggregate

/**
 * Treated- and control-group sums and counts for the difference-in-means treatment-effect
 * estimate. A pure record: the additive group that scans it lives in
 * [[treebased.core.training.aggregate.TreatmentEffectStatsAccumulator]].
 *
 * `ate` returns `NaN` whenever either group is empty rather than silently substituting `0.0`
 * for a missing group mean. It cannot rely on `0.0 / 0` alone: the split scan reaches an empty
 * group by subtracting points back out ([[TreatmentEffectStatsAccumulator.remove]]), which
 * leaves a tiny floating-point residual, so `residual / 0` would yield `±Infinity` instead.
 * [[treebased.core.training.HonestyPruner]] collapses any leaf whose prediction comes out `NaN`
 * into its honesty-sample sibling, so a leaf with an empty treated or control group is repaired
 * rather than silently biased.
 */
final case class TreatmentEffectStats(
  treatmentSum: Double,
  controlSum: Double,
  treatmentCount: Int,
  controlCount: Int
) {
  def count: Int = treatmentCount + controlCount
  def ate: Double =
    if (treatmentCount == 0 || controlCount == 0) Double.NaN
    else (treatmentSum / treatmentCount) - (controlSum / controlCount)
}

object TreatmentEffectStats {
  val zero: TreatmentEffectStats = TreatmentEffectStats(0.0, 0.0, 0, 0)
}
