package treebased.core.training.aggregate

import treebased.core.domain.data.CausalObservation

/**
 * Treated- and control-group sums and counts accumulated from causal observations. Backs the exact causal split
 * scan ([[treebased.core.training.split.ExactCausalSplitRule]]) and the ATE honest leaf
 * ([[treebased.core.prediction.LeafEstimator.ate]]).
 */
object TreatmentEffectStatsAccumulator extends Accumulator[CausalObservation, TreatmentEffectStats] {
  val empty: TreatmentEffectStats = TreatmentEffectStats.zero

  def lift(o: CausalObservation): TreatmentEffectStats =
    if (o.w == 1) TreatmentEffectStats(o.y, 0.0, 1, 0) else TreatmentEffectStats(0.0, o.y, 0, 1)

  def combine(a: TreatmentEffectStats, b: TreatmentEffectStats): TreatmentEffectStats =
    TreatmentEffectStats(
      a.treatmentSum + b.treatmentSum,
      a.controlSum + b.controlSum,
      a.treatmentCount + b.treatmentCount,
      a.controlCount + b.controlCount
    )

  def remove(whole: TreatmentEffectStats, part: TreatmentEffectStats): TreatmentEffectStats =
    TreatmentEffectStats(
      whole.treatmentSum - part.treatmentSum,
      whole.controlSum - part.controlSum,
      whole.treatmentCount - part.treatmentCount,
      whole.controlCount - part.controlCount
    )
}
