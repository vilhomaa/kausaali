package treebased.core.training.aggregate

import treebased.core.domain.data.CausalObservation

/** Folds causal observations into [[TreatmentSlopeStats]] for the locally centered causal forest's
 *  honest leaf ([[treebased.core.prediction.LeafEstimator.ateSlope]]). */
object TreatmentSlopeAccumulator extends Accumulator[CausalObservation, TreatmentSlopeStats] {
  val empty: TreatmentSlopeStats = TreatmentSlopeStats.zero

  def lift(o: CausalObservation): TreatmentSlopeStats =
    TreatmentSlopeStats(o.w, o.y, o.w * o.y, o.w * o.w, 1)

  def combine(a: TreatmentSlopeStats, b: TreatmentSlopeStats): TreatmentSlopeStats =
    TreatmentSlopeStats(
      a.wSum + b.wSum,
      a.ySum + b.ySum,
      a.wySum + b.wySum,
      a.wwSum + b.wwSum,
      a.count + b.count
    )

  def remove(whole: TreatmentSlopeStats, part: TreatmentSlopeStats): TreatmentSlopeStats =
    TreatmentSlopeStats(
      whole.wSum - part.wSum,
      whole.ySum - part.ySum,
      whole.wySum - part.wySum,
      whole.wwSum - part.wwSum,
      whole.count - part.count
    )
}
